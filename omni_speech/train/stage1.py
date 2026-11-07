"""Latxa-Omni Stage 1 speech-projector training on HF speech datasets."""

import argparse
import dataclasses
import inspect
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import whisper
from datasets import DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer import TRAINING_ARGS_NAME

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.model.builder import create_model, is_qwen3_5
from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig
from omni_speech.train.dev_metrics import task_metrics


USER_PROMPT = "<speech>\nPlease directly answer the questions in the user's speech."
# The Latxa LLM's attention and MLP projections; anchored so the Whisper encoder's
# q/k/v projections (model.speech_encoder.encoder.layers.*) never match. Qwen3.5's linear-attention
# (Gated DeltaNet) layers have fused input projections instead of q/k/v/o.
LORA_TARGET_MODULES = (
    r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate|up|down)_proj)"
)
# transformers 4.46 renamed Trainer's `tokenizer` argument to `processing_class` (removed in 5.x).
TRAINER_TOKENIZER_ARG = (
    "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
)
# transformers 5 dropped warmup_ratio; a float warmup_steps below 1 is a ratio there.
WARMUP_ARG = (
    "warmup_ratio" if "warmup_ratio" in {f.name for f in dataclasses.fields(TrainingArguments)} else "warmup_steps"
)


def add_lora(model, r, alpha, dropout):
    """Wrap the LLM in LoRA adapters. The speech projector stays fully trainable;
    peft freezes everything else, including the (already frozen) speech encoder."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(model, config)
    for param in model.get_base_model().get_model().speech_projector.parameters():
        param.requires_grad = True
    return model


@torch.no_grad()
def merged_state_dict(peft_model):
    """State dict of the base model with every LoRA delta folded into its weight.

    Keys match a full fine-tune checkpoint, so downstream loaders need no changes.
    The live weights are left untouched (no merge/unmerge round trip in bf16).
    """
    from peft.tuners.lora import LoraLayer

    base = peft_model.get_base_model()
    deltas = {
        f"{name}.weight": module.get_delta_weight(module.active_adapters[0])
        for name, module in base.named_modules() if isinstance(module, LoraLayer)
    }
    state = {}
    for key, value in base.state_dict().items():
        if ".lora_" in key:
            continue
        key = key.replace(".base_layer.", ".")
        if key in deltas:
            value = (value.float() + deltas.pop(key).float()).to(value.dtype)
        state[key] = value
    if deltas:
        raise RuntimeError(f"LoRA deltas without a base weight: {sorted(deltas)}")
    return state


class Stage1Trainer(Trainer):
    """Saves a LoRA model with its adapters merged, as a plain OmniSpeech checkpoint."""

    def _save(self, output_dir=None, state_dict=None):
        from peft import PeftModel

        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, PeftModel):
            return super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model.get_base_model().save_pretrained(
            output_dir, state_dict=merged_state_dict(model),
            # transformers 5 always writes safetensors and dropped the argument.
            safe_serialization=getattr(self.args, "save_safetensors", True),
        )
        tokenizer = getattr(self, TRAINER_TOKENIZER_ARG)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


def user_prompt(turn):
    """Multi-task rows carry an ``instruction`` (empty means bare speech); older rows get USER_PROMPT.

    A user turn without audio is written text (e.g. TTS rows): the instruction, if any, then the text.
    """
    instruction = turn.get("instruction")
    if not turn.get("audio"):
        if not turn.get("text"):
            raise ValueError("a user turn needs audio or text")
        return f"{instruction}\n{turn['text']}" if instruction else turn["text"]
    if instruction is None:
        return USER_PROMPT
    return f"{instruction}\n<speech>" if instruction else "<speech>"


def build_prompt_and_labels(turns, tokenizer, conv_mode):
    """Supervise assistant bodies only, retaining preceding turns as context."""
    conv = conv_templates[conv_mode].copy()
    audio_paths = []
    answer_spans = []
    user_turns = 0
    previous_ids = None
    for index, turn in enumerate(turns):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn["role"] != expected:
            raise ValueError(f"expected {expected} at turn {index}")
        if expected == "user":
            if turn.get("audio"):
                audio_paths.append(turn["audio"])
            user_turns += 1
            conv.append_message(conv.roles[0], user_prompt(turn))
        else:
            conv.append_message(conv.roles[1], None)
            header_ids = tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")
            conv.messages[-1][1] = turn["text"]
            completed_ids = tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")
            if not torch.equal(header_ids, completed_ids[:len(header_ids)]):
                raise ValueError(f"tokenization changed across assistant header at turn {index}")
            answer_spans.append((len(header_ids), len(completed_ids)))
        current_ids = tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")
        if previous_ids is not None and not torch.equal(previous_ids, current_ids[:len(previous_ids)]):
            raise ValueError(f"tokenization changed across turn {index}")
        previous_ids = current_ids
    if not answer_spans or user_turns != len(answer_spans):
        raise ValueError("conversation must contain complete user/assistant pairs")
    input_ids = previous_ids
    if int((input_ids == SPEECH_TOKEN_INDEX).sum()) != len(audio_paths):
        raise ValueError("speech token count differs from user audio count")
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for start, end in answer_spans:
        labels[start:end] = input_ids[start:end]
    if not labels.ne(IGNORE_INDEX).any():
        raise ValueError("conversation has no assistant target tokens")
    return input_ids, labels, audio_paths


class Stage1Dataset(Dataset):
    def __init__(self, rows, tokenizer, model_config, input_type, mel_size, data_root, conv_mode):
        self.rows = rows
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.data_root = Path(data_root) if data_root else None
        self.conv_mode = conv_mode

    def __len__(self):
        return len(self.rows)

    def task(self, index):
        return self.rows[index].get("task") or "chat"

    def turns(self, index):
        row = self.rows[index]
        if "turns" in row:
            return row["turns"]
        return [
            {"role": "user", "audio": row["question_audio"]},
            {"role": "assistant", "text": row["answer"]},
        ]

    def load_speech(self, audio_path):
        path = Path(audio_path)
        if not path.is_absolute() and self.data_root is not None:
            path = self.data_root / path
        waveform = whisper.load_audio(str(path))
        if self.input_type == "raw":
            tensor = torch.from_numpy(waveform)
            if self.model_config.speech_normalize:
                tensor = torch.nn.functional.layer_norm(tensor, tensor.shape)
        elif self.input_type == "mel":
            tensor = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(waveform), n_mels=self.mel_size
            ).permute(1, 0)
        else:
            raise ValueError(f"unsupported input_type: {self.input_type}")
        return tensor.to(torch.bfloat16)

    def speech_inputs(self, audio_paths):
        """Encoder inputs for a row's audio. A row with no audio gets one silent placeholder: the
        model consumes a zero-length slice of it, which keeps every row aligned with its own clips
        and the projector in the graph on every rank (ZeRO-2 hangs on unused parameters)."""
        if audio_paths:
            return [self.load_speech(audio_path) for audio_path in audio_paths]
        if self.input_type == "raw":
            return [torch.zeros(whisper.audio.N_SAMPLES, dtype=torch.bfloat16)]
        return [torch.zeros(whisper.audio.N_FRAMES, self.mel_size, dtype=torch.bfloat16)]

    def __getitem__(self, index):
        input_ids, labels, audio_paths = build_prompt_and_labels(
            self.turns(index), self.tokenizer, self.conv_mode
        )
        speech = self.speech_inputs(audio_paths)
        speech_lengths = [tensor.shape[0] for tensor in speech]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "speech": speech,
            "speech_lengths": speech_lengths,
        }


def collate_fn(batch, pad_token_id):
    token_lengths = torch.tensor([len(row["input_ids"]) for row in batch])
    input_ids = pad_sequence(
        [row["input_ids"] for row in batch], batch_first=True, padding_value=pad_token_id
    )
    labels = pad_sequence(
        [row["labels"] for row in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    speech = [clip for row in batch for clip in row["speech"]]
    lengths = [length for row in batch for length in row["speech_lengths"]]
    # Rows without a <speech> token carry one placeholder clip (Stage1Dataset.speech_inputs).
    expected = sum(max(int((row["input_ids"] == SPEECH_TOKEN_INDEX).sum()), 1) for row in batch)
    if len(speech) != expected:
        raise ValueError(f"expected {expected} audio clips, got {len(speech)}")
    return {
        "input_ids": input_ids,
        "attention_mask": torch.arange(input_ids.shape[1])[None, :] < token_lengths[:, None],
        "labels": labels,
        "speech": pad_sequence(speech, batch_first=True),
        "speech_lengths": torch.tensor(lengths, dtype=torch.long),
    }


def split_dev(held_out, dev_size, seed):
    """Fixed dev subset of the held-out split; the remainder stays as the final test."""
    if not 0 < dev_size < len(held_out):
        raise ValueError(f"dev size {dev_size} must be between 1 and {len(held_out) - 1}")
    shuffled = held_out.shuffle(seed=seed)
    return shuffled.select(range(dev_size)), shuffled.select(range(dev_size, len(shuffled)))


def build_generation_prompt(turns, tokenizer, conv_mode):
    """Prompt with every turn but the last assistant answer, which becomes the reference."""
    if not turns or turns[-1]["role"] != "assistant":
        raise ValueError("conversation must end with an assistant turn")
    conv = conv_templates[conv_mode].copy()
    audio_paths = []
    for turn in turns[:-1]:
        if turn["role"] == "user":
            if turn.get("audio"):
                audio_paths.append(turn["audio"])
            conv.append_message(conv.roles[0], user_prompt(turn))
        else:
            conv.append_message(conv.roles[1], turn["text"])
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")
    return input_ids, audio_paths


@torch.no_grad()
def generate_dev_answers(model, dataset, indices, tokenizer, max_new_tokens, batch_size=1):
    """Greedy answers to the last user turn of each selected dev conversation.

    With batch_size > 1, rows of similar prompt length are generated together, left-padded; every row's
    clips (or its silent placeholder) are concatenated in row order, as the speech splicing expects.
    """
    was_training = model.training
    padding_side = getattr(model.config, "tokenizer_padding_side", "right")
    model.eval()
    model.config.tokenizer_padding_side = "left"
    device = next(model.parameters()).device
    prompts = {}
    for index in indices:
        turns = dataset.turns(index)
        input_ids, audio_paths = build_generation_prompt(turns, tokenizer, dataset.conv_mode)
        prompts[index] = (turns, input_ids, dataset.speech_inputs(audio_paths))
    order = sorted(indices, key=lambda i: prompts[i][1].shape[0]) if batch_size > 1 else list(indices)
    predictions = {}
    for start in tqdm(range(0, len(order), batch_size), desc="generate", disable=batch_size == 1):
        batch = order[start:start + batch_size]
        input_ids = [prompts[i][1] for i in batch]
        width = max(ids.shape[0] for ids in input_ids)
        padded = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=input_ids[0].dtype)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, ids in enumerate(input_ids):
            padded[row, width - ids.shape[0]:] = ids
            attention_mask[row, width - ids.shape[0]:] = 1
        speech = [clip for i in batch for clip in prompts[i][2]]
        output_ids = model.generate(
            padded.to(device),
            attention_mask=attention_mask.to(device),
            speech=pad_sequence(speech, batch_first=True).to(device),
            speech_lengths=torch.tensor([s.shape[0] for s in speech], device=device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        for index, ids in zip(batch, output_ids):
            predictions[index] = tokenizer.decode(ids, skip_special_tokens=True).strip()
    model.config.tokenizer_padding_side = padding_side
    model.train(was_training)
    return [{
        "id": dataset.rows[index]["id"],
        "task": dataset.task(index),
        "question": prompts[index][0][-2].get("text") or "",
        "reference": prompts[index][0][-1]["text"],
        "prediction": predictions[index],
    } for index in indices]


def generation_indices(tasks, num_samples):
    """The first ``num_samples`` dev rows of each task, in dev order."""
    picked, counts = [], Counter()
    for index, task in enumerate(tasks):
        if counts[task] < num_samples:
            picked.append(index)
            counts[task] += 1
    return picked


def summarize_generations(results):
    """Per-task metrics, keyed like ``asr_wer`` or ``st_en_bleu``."""
    by_task = {}
    for row in results:
        by_task.setdefault(row["task"], []).append(row)
    metrics = {}
    for task, rows in sorted(by_task.items()):
        scores = task_metrics(task, [r["reference"] for r in rows], [r["prediction"] for r in rows])
        metrics.update({f"{task}_{name}": round(value, 2) for name, value in scores.items()})
    return by_task, metrics


class DevGenerationCallback(TrainerCallback):
    """After each evaluation, answer a fixed set of dev rows per task, score and log them.

    ASR rows get WER and speech-translation rows BLEU and chrF (see dev_metrics.py).
    """

    def __init__(self, dataset, num_samples, tokenizer, max_new_tokens):
        self.dataset = dataset
        self.indices = generation_indices([dataset.task(i) for i in range(len(dataset))], num_samples)
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        results = generate_dev_answers(
            model, self.dataset, self.indices[rank::world_size],
            self.tokenizer, self.max_new_tokens,
        )
        if distributed:
            gathered = [None] * world_size
            dist.all_gather_object(gathered, results)
            results = [row for part in gathered for row in part]
        if not state.is_world_process_zero:
            return
        by_task, metrics = summarize_generations(results)
        print(f"dev generation step {state.global_step}: "
              f"{ {task: len(rows) for task, rows in by_task.items()} } {metrics}")
        output_dir = Path(args.output_dir) / "dev_generations"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / f"step-{state.global_step}.jsonl", "w", encoding="utf-8") as stream:
            for row in results:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": state.global_step, **metrics}) + "\n")
        if "wandb" in args.report_to:
            import wandb
            if wandb.run is not None:
                columns = ["id", "question", "reference", "prediction"]
                log = {f"eval/{name}": value for name, value in metrics.items()}
                for task, rows in by_task.items():
                    log[f"eval/generations_{task}"] = wandb.Table(
                        columns=["step", *columns],
                        data=[[state.global_step, *(row[c] for c in columns)] for row in rows],
                    )
                wandb.log({**log, "train/global_step": state.global_step})


class SaveOnBestDevCallback(TrainerCallback):
    """Save a checkpoint whenever dev loss reaches a new minimum.

    Best-dev checkpoints hold model weights only; end-of-epoch checkpoints keep
    the optimizer state for resuming. Only the latest best-dev checkpoint is
    kept: the previous one is removed once the new one is saved, unless it
    holds optimizer state (an epoch checkpoint, or one from an older run).
    With a patience, training stops after that many evaluations without a new
    minimum.
    """

    def __init__(self, patience=0):
        self.best = float("inf")
        self.best_step = None
        self.new_best = False
        self.epoch_steps = set()
        self.patience = patience
        self.evals_since_best = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # When resuming, continue from the best dev loss in the trainer state.
        if state.best_metric is not None:
            self.best = state.best_metric
        if state.best_model_checkpoint:
            self.best_step = int(Path(state.best_model_checkpoint).name.split("-")[-1])

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_steps.add(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is None:
            return control
        if loss < self.best:
            self.best = loss
            self.new_best = True
            self.evals_since_best = 0
            control.should_save = True
            args.save_only_model = state.global_step not in self.epoch_steps
        else:
            self.evals_since_best += 1
            if self.patience and self.evals_since_best >= self.patience:
                control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        args.save_only_model = False
        if not self.new_best:
            return
        self.new_best = False
        previous, self.best_step = self.best_step, state.global_step
        if not state.is_world_process_zero or previous in (None, state.global_step):
            return
        previous_dir = Path(args.output_dir) / f"checkpoint-{previous}"
        if not any(previous_dir.glob("global_step*")):
            shutil.rmtree(previous_dir, ignore_errors=True)


def train_model(args):
    dataset = load_from_disk(args.train_file)
    if not isinstance(dataset, DatasetDict) or "train" not in dataset:
        raise ValueError("--train-file must be a saved DatasetDict with a train split")
    if is_qwen3_5(AutoConfig.from_pretrained(args.model_path)):
        from omni_speech.model.language_model.omni_speech_qwen3_5 import OmniSpeechQwen3_5Config as config_class
        args.conv_mode = args.conv_mode or "qwen3_5"
    else:
        config_class = OmniSpeechConfig
        args.conv_mode = args.conv_mode or "llama_3"
    model_config = config_class.from_pretrained(args.model_path)
    with open(args.model_config, encoding="utf-8") as stream:
        speech_config = json.load(stream)
    for key, value in speech_config.items():
        if key.startswith(("speech_", "tokenizer_")) or key == "tune_speech_projector":
            setattr(model_config, key, value)
    if args.speech_encoder:
        model_config.speech_encoder = args.speech_encoder
    if args.speech_encoder_type:
        model_config.speech_encoder_type = args.speech_encoder_type
    if model_config.speech_encoder_type == "whisper" and not Path(model_config.speech_encoder).is_file():
        raise FileNotFoundError(f"Whisper checkpoint missing: {model_config.speech_encoder}")
    if model_config.speech_encoder_type == "whisper_hf" and not model_config.speech_encoder:
        raise ValueError("a Hugging Face Whisper model path or ID is required")
    tokenizer, model, _ = create_model(
        args.model_path, args.model_base, is_lora=args.is_lora,
        s2s=args.s2s, config=model_config,
    )
    model.config.tokenizer_model_max_length = args.max_sequence_length
    if args.conv_mode not in conv_templates:
        raise ValueError(f"unknown conversation mode: {args.conv_mode}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = Stage1Dataset(
        dataset["train"], tokenizer, model.config, args.input_type,
        args.mel_size, args.data_root, args.conv_mode,
    )
    eval_rows = None
    if args.eval_steps > 0:
        if "dev" in dataset:
            # A saved dev split is used as is (--dev_size only trims it, e.g. for smoke tests).
            dev_split = dataset["dev"].select(range(min(args.dev_size, len(dataset["dev"]))))
        elif "test" in dataset:
            dev_split, _ = split_dev(dataset["test"], args.dev_size, args.dev_seed)
        else:
            raise ValueError("--eval_steps needs a dev or test split")
        eval_rows = Stage1Dataset(
            dev_split, tokenizer, model.config, args.input_type,
            args.mel_size, args.data_root, args.conv_mode,
        )
    # Stage 1 trains the projector and the language model (fully, or through LoRA
    # adapters); create_model already freezes the speech encoder.
    if args.lora_r > 0:
        model = add_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latxa-Omni Stage 1: {len(train_rows)} train examples, {trainable:,} trainable parameters")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        adam_beta2=0.95,
        **{WARMUP_ARG: 0.01},
        lr_scheduler_type="cosine",
        report_to=args.report_to,
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_rows is not None else "no",
        eval_steps=args.eval_steps if eval_rows is not None else None,
        per_device_eval_batch_size=args.train_batch_size,
        prediction_loss_only=True,
        # A peft-wrapped model's forward(*args, **kwargs) hides `labels` from the Trainer.
        label_names=["labels"],
        # With a save limit, rotation always keeps the best checkpoint and the latest.
        metric_for_best_model="eval_loss" if eval_rows is not None else None,
        greater_is_better=False,
        seed=3407,
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    callbacks = []
    if eval_rows is not None:
        if args.save_strategy != "no":
            # Dev-loss improvements add checkpoints on top of the epoch saves.
            callbacks.append(SaveOnBestDevCallback(args.early_stopping_patience))
        if args.num_generation_samples > 0:
            callbacks.append(DevGenerationCallback(
                eval_rows, args.num_generation_samples, tokenizer, args.max_new_tokens,
            ))
    trainer = Stage1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_rows,
        eval_dataset=eval_rows,
        data_collator=lambda rows: collate_fn(rows, tokenizer.pad_token_id),
        callbacks=callbacks,
        **{TRAINER_TOKENIZER_ARG: tokenizer},
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base")
    parser.add_argument("--model-config", default="configs/config-Llama-3.1-8B-Instruct.json")
    parser.add_argument("--speech-encoder", help="OpenAI .pt path or Hugging Face Whisper model")
    parser.add_argument("--speech-encoder-type", choices=("whisper", "whisper_hf"))
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--input_type", default="mel", choices=("mel", "raw"))
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=4096)
    parser.add_argument("--s2s", action="store_true")
    parser.add_argument("--is_lora", action="store_true", help="--model-path is a LoRA checkpoint to load")
    parser.add_argument("--lora_r", type=int, default=0, help="train the LLM through LoRA of this rank; 0 = full FT")
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="faster when memory allows (Qwen3.5-4B LoRA: 2.2x per micro-step)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--deepspeed", default=str(Path(__file__).with_name("ds_config.json")))
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--save_strategy", choices=("epoch", "no"), default="epoch")
    parser.add_argument("--save_total_limit", type=int, help="default keeps every checkpoint")
    parser.add_argument("--eval_steps", type=int, default=250, help="0 disables dev evaluation")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="stop after this many evaluations without a new best dev loss; 0 disables")
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--dev_seed", type=int, default=3407)
    parser.add_argument("--num_generation_samples", type=int, default=32, help="dev rows generated per task")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--output_dir", default="saves/stage1")
    parser.add_argument("--run_name")
    train_model(parser.parse_args())
