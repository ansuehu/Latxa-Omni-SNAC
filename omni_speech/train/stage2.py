"""Latxa-Omni-SNAC Stage 2: train the SNAC talker on top of a Stage 1 checkpoint.

Each row needs the Stage 1 fields (``question_audio`` + ``answer``, or ``turns``)
plus ``answer_snac``: an .npy path with the flat SNAC tokens of the last spoken
answer, written by dataset_generation/dump_snac_tokens.py. By default the LLM is
frozen and only the talker trains; with --tune_llm the LLM and projector also
train (at --llm_learning_rate).

With the LLM frozen, checkpoints hold only the talker (talker.safetensors, ~2 GB)
plus a config whose talker_base_model points at the Stage 1 checkpoint; the
DeepSpeed resume state also leaves the frozen weights out. load_pretrained_model
and create_model rebuild the full model from the two.
"""

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import DatasetDict, load_from_disk
from safetensors.torch import save_file
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, Trainer, TrainerCallback, TrainingArguments

from omni_speech.constants import IGNORE_INDEX
from omni_speech.model.builder import QWEN3_5_MODEL_TYPES, TALKER_WEIGHTS_NAME, create_model
from omni_speech.model.language_model.omni_speech2s_llama import OmniSpeech2SConfig, TALKER_CONFIG_KEYS
from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE
from omni_speech.model.speech_generator.talker import last_answer_mask
from omni_speech.train.stage1 import (
    TRAINER_TOKENIZER_ARG, WARMUP_ARG, Stage1Dataset, build_prompt_and_labels, collate_fn as stage1_collate_fn,
    split_dev,
)


class Stage2Dataset(Stage1Dataset):
    def snac_path(self, index):
        path = Path(self.rows[index]["answer_snac"])
        if not path.is_absolute() and self.data_root is not None:
            path = self.data_root / path
        return path

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["tgt_units"] = torch.from_numpy(np.load(self.snac_path(index)).astype(np.int64))
        return item


def collate_fn(batch, pad_token_id):
    out = stage1_collate_fn(batch, pad_token_id)
    out["tgt_units"] = pad_sequence(
        [row["tgt_units"] for row in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    return out


@torch.no_grad()
def synthesize_reference(model, dataset, index, codec, **talker_kwargs):
    """Speak the reference answer of a row (teacher-forced text) -> 24 kHz waveform."""
    device = next(model.parameters()).device
    input_ids, labels, audio_paths = build_prompt_and_labels(
        dataset.turns(index), dataset.tokenizer, dataset.conv_mode
    )
    speech = dataset.speech_inputs(audio_paths)
    _, _, _, _, embeds, labels = model.prepare_inputs_labels_for_speech_and_text(
        input_ids[None].to(device), None, None, None, labels[None].to(device),
        pad_sequence(speech, batch_first=True).to(device),
        torch.tensor([s.shape[0] for s in speech], device=device),
    )
    hidden = model.get_model()(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]
    mask = last_answer_mask(labels[0])
    text_embeds = model.get_model().embed_tokens(labels[0][mask])
    cond = model.speech_generator.fuse(hidden[mask], text_embeds)
    return codec.decode(model.speech_generator.generate_units(cond, **talker_kwargs))


class DevAudioCallback(TrainerCallback):
    """After each evaluation, speak a few dev answers and save/log the audio (rank 0)."""

    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.indices = list(range(min(num_samples, len(dataset))))
        self.codec = None

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        import soundfile as sf
        from omni_speech.model.speech_generator.snac_codec import SnacCodec

        if self.codec is None:
            self.codec = SnacCodec(device=next(model.parameters()).device)
        was_training = model.training
        model.eval()
        out_dir = Path(args.output_dir) / "dev_audio" / f"step-{state.global_step}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logged = {}
        for index in self.indices:
            wav = synthesize_reference(model, self.dataset, index, self.codec, max_tokens=2048)
            name = str(self.dataset.rows[index].get("id", index))
            sf.write(out_dir / f"{name}.wav", wav, SNAC_SAMPLE_RATE)
            logged[name] = wav
        model.train(was_training)
        if "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                wandb.log({f"eval/audio/{k}": wandb.Audio(v, sample_rate=SNAC_SAMPLE_RATE)
                           for k, v in logged.items()} | {"train/global_step": state.global_step})


class Stage2Trainer(Trainer):
    """Separate learning rates for the talker and (with --tune_llm) the LLM side.

    With ``talker_only`` (LLM frozen), checkpoints keep only the talker weights.
    """

    def __init__(self, *args, llm_learning_rate=None, talker_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_learning_rate = llm_learning_rate
        self.talker_only = talker_only

    def _save(self, output_dir=None, state_dict=None):
        if not self.talker_only:
            return super()._save(output_dir, state_dict)
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        talker = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()
                  if k.startswith("speech_generator.")}
        save_file(talker, os.path.join(output_dir, TALKER_WEIGHTS_NAME), metadata={"format": "pt"})
        self.accelerator.unwrap_model(self.model).config.save_pretrained(output_dir)
        tokenizer = getattr(self, TRAINER_TOKENIZER_ARG)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _save_optimizer_and_scheduler(self, output_dir):
        if not (self.talker_only and self.is_deepspeed_enabled):
            return super()._save_optimizer_and_scheduler(output_dir)
        # The frozen LLM is already in talker_base_model; keep it out of the resume state.
        engine = self.model_wrapped
        engine.save_checkpoint = functools.partial(engine.save_checkpoint, exclude_frozen_parameters=True)
        try:
            super()._save_optimizer_and_scheduler(output_dir)
        finally:
            del engine.save_checkpoint

    def train(self, *args, **kwargs):
        if not (self.talker_only and self.is_deepspeed_enabled):
            return super().train(*args, **kwargs)
        # Resume states lack the frozen weights (see above), so load them non-strictly.
        import transformers.trainer as hf_trainer

        original = hf_trainer.deepspeed_load_checkpoint
        hf_trainer.deepspeed_load_checkpoint = (
            lambda engine, path, load_module_strict=True: original(engine, path, load_module_strict=False)
        )
        try:
            return super().train(*args, **kwargs)
        finally:
            hf_trainer.deepspeed_load_checkpoint = original

    def create_optimizer(self):
        if self.optimizer is None:
            talker, llm = [], []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    (talker if name.startswith("speech_generator.") else llm).append(param)
            groups = [{"params": talker, "lr": self.args.learning_rate}]
            if llm:
                groups.append({"params": llm, "lr": self.llm_learning_rate})
            self.optimizer = torch.optim.AdamW(
                groups, betas=(0.9, 0.95), weight_decay=self.args.weight_decay
            )
        return self.optimizer


def train_model(args):
    dataset = load_from_disk(args.train_file)
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        raise ValueError("--train-file must be a saved DatasetDict with a train split")
    if PretrainedConfig.get_config_dict(args.model_path)[0].get("model_type") in QWEN3_5_MODEL_TYPES:
        # transformers >= 5.2 only (the omni-qwen env).
        from omni_speech.model.language_model.omni_speech2s_qwen3_5 import OmniSpeech2SQwen3_5Config as config_class
        args.conv_mode = args.conv_mode or "qwen3_5"
    else:
        config_class = OmniSpeech2SConfig
        args.conv_mode = args.conv_mode or "llama_3"
    model_config = config_class.from_pretrained(args.model_path)
    if getattr(model_config, "speech_generator_type", None) != "snac_ar":
        # Starting from Stage 1: take the talker settings from --talker-config.
        with open(args.talker_config, encoding="utf-8") as stream:
            talker_settings = json.load(stream)
        for key in TALKER_CONFIG_KEYS:
            if key in talker_settings:
                setattr(model_config, key, talker_settings[key])
        model_config.speech_generator_type = "snac_ar"
        model_config.talker_base_model = str(Path(args.model_path).resolve())
        init_talker = args.talker_init_from
    else:
        init_talker = None
    model_config.tune_speech_generator_only = not args.tune_llm
    if (getattr(model_config, "talker_attn_implementation", None) == "flash_attention_2"
            and importlib.util.find_spec("flash_attn") is None):
        # omni-qwen has no flash-attn build (torch 2.8); the talker is a plain Llama, so sdpa works too.
        print("flash-attn is not installed: the talker uses sdpa")
        model_config.talker_attn_implementation = "sdpa"
    tokenizer, model, _ = create_model(args.model_path, None, s2s=True, config=model_config)
    if init_talker:
        model.speech_generator.load_pretrained_decoder(init_talker)
    model.config.tokenizer_model_max_length = args.max_sequence_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for name, param in model.named_parameters():
        if name.startswith("speech_generator."):
            param.requires_grad = True
        else:
            param.requires_grad = args.tune_llm and "speech_encoder" not in name

    def rows(split):
        if args.max_units and "answer_snac_len" in split.column_names:
            split = split.filter(lambda n: n <= args.max_units, input_columns="answer_snac_len")
        return Stage2Dataset(split, tokenizer, model.config, args.input_type,
                             args.mel_size, args.data_root, args.conv_mode)

    train_rows = rows(dataset["train"])
    eval_rows = None
    if args.eval_steps > 0:
        if "dev" in dataset:
            dev_split = dataset["dev"].select(range(min(args.dev_size, len(dataset["dev"]))))
        elif "test" in dataset:
            dev_split, _ = split_dev(dataset["test"], args.dev_size, args.dev_seed)
        else:
            raise ValueError("--eval_steps needs a dev or test split")
        eval_rows = rows(dev_split)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latxa-Omni-SNAC Stage 2: {len(train_rows)} train examples, {trainable:,} trainable parameters")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        **{WARMUP_ARG: 0.02},
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy="steps" if eval_rows is not None else "no",
        eval_steps=args.eval_steps if eval_rows is not None else None,
        per_device_eval_batch_size=args.train_batch_size,
        prediction_loss_only=True,
        seed=3407,
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )
    callbacks = []
    if eval_rows is not None and args.num_audio_samples > 0:
        callbacks.append(DevAudioCallback(eval_rows, args.num_audio_samples))
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        **{TRAINER_TOKENIZER_ARG: tokenizer},
        callbacks=callbacks,
        llm_learning_rate=args.llm_learning_rate,
        talker_only=not args.tune_llm,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Stage 1 (or Stage 2) checkpoint")
    parser.add_argument("--talker-config", default="configs/talker-snac.json")
    parser.add_argument("--talker-init-from", help="pretrained Llama for the talker decoder, e.g. meta-llama/Llama-3.2-1B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--input_type", default="mel", choices=("mel", "raw"))
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=4096)
    parser.add_argument("--max_units", type=int, default=2460, help="skip answers above this many SNAC tokens (~30 s)")
    parser.add_argument("--tune_llm", action="store_true", help="also train the LLM and projector (joint Stage 3)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="talker learning rate")
    parser.add_argument("--llm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--deepspeed", default=str(Path(__file__).with_name("ds_config_stage2.json")))
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--save_steps", type=int, default=0, help="0 saves once per epoch")
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_only_model", action="store_true",
                        help="skip the optimizer/resume state (~15 GB per checkpoint for the 1B talker)")
    parser.add_argument("--eval_steps", type=int, default=500, help="0 disables dev evaluation")
    parser.add_argument("--dev_size", type=int, default=500)
    parser.add_argument("--dev_seed", type=int, default=3407)
    parser.add_argument("--num_audio_samples", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--output_dir", default="saves/stage2")
    parser.add_argument("--run_name")
    train_model(parser.parse_args())
