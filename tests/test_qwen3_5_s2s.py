"""Qwen3.5 thinker + SNAC talker (Stage 2): checkpoint loading/saving, forward and generation.

Needs transformers >= 5.2 (omni-qwen env); skipped elsewhere. The forward tests need a GPU, as
fla's Gated DeltaNet kernels are Triton-only.
"""

import tempfile
import unittest

import torch
from transformers import TrainingArguments

from omni_speech.constants import IGNORE_INDEX
from omni_speech.model.speech_generator.snac_codec import FRAME_TOKENS
from omni_speech.model.speech_generator.talker import SpeechGeneratorSNAC
from test_qwen3_5 import needs_gpu, needs_qwen, tiny_model
from test_snac_talker import TINY, random_units

try:
    from omni_speech.model.language_model.omni_speech2s_qwen3_5 import (
        OmniSpeech2SQwen3_5Config, OmniSpeech2SQwen3_5ForCausalLM,
    )
except ImportError:
    pass

TALKER_SETTINGS = {"speech_generator_type": "snac_ar", "talker_config": TINY, "talker_read": 2,
                   "talker_write": 7, "talker_attn_implementation": "eager"}


def save_stage1(directory):
    """A tiny Qwen3.5 Stage 1 checkpoint (projector included, no speech encoder weights)."""
    model = tiny_model()
    model.config.speech_encoder, model.config.speech_encoder_type = "whisper", "whisper_hf"
    model.get_model().speech_encoder = None
    model.save_pretrained(directory)
    return model


def s2s_config(stage1_dir, **overrides):
    """As stage2.py: the Stage 1 config plus the talker settings."""
    config = OmniSpeech2SQwen3_5Config.from_pretrained(stage1_dir)
    for key, value in dict(TALKER_SETTINGS, **overrides).items():
        setattr(config, key, value)
    config.talker_base_model = stage1_dir
    return config


def tiny_s2s(**overrides):
    with tempfile.TemporaryDirectory() as directory:
        save_stage1(directory)
        model = OmniSpeech2SQwen3_5ForCausalLM.from_pretrained(directory, config=s2s_config(directory, **overrides))
    model.initialize_speech_generator(model.config)  # as create_model does
    model.get_model().speech_encoder = torch.nn.Identity()
    return model


@needs_qwen
class QwenS2SCheckpointTest(unittest.TestCase):
    def test_stage1_weights_load_and_the_talker_is_initialised(self):
        with tempfile.TemporaryDirectory() as directory:
            stage1 = save_stage1(directory)
            # The rope scaling of configs/talker-snac.json (Llama-3.2-1B's).
            rope_scaling = {"rope_type": "llama3", "factor": 32.0, "high_freq_factor": 4.0,
                            "low_freq_factor": 1.0, "original_max_position_embeddings": 8192}
            config = s2s_config(directory, talker_config=dict(TINY, rope_theta=500000.0, rope_scaling=rope_scaling,
                                                              max_position_embeddings=131072))
            model = OmniSpeech2SQwen3_5ForCausalLM.from_pretrained(directory, config=config)
        expected = stage1.state_dict()
        loaded = model.state_dict()
        for name, value in expected.items():
            torch.testing.assert_close(loaded[name], value, msg=name)
        self.assertTrue(any(".layers.0.linear_attn." in name for name in expected))
        talker = model.speech_generator
        self.assertEqual(talker.hidden_proj[0].in_features, model.config.hidden_size)
        self.assertEqual(talker.text_proj.in_features, model.config.hidden_size)
        # Built on the meta device: buffers and norms must still come out right.
        fresh = SpeechGeneratorSNAC(model.config)
        self.assertEqual(talker.decoder.rotary_emb.rope_type, "llama3")
        torch.testing.assert_close(talker.decoder.rotary_emb.inv_freq, fresh.decoder.rotary_emb.inv_freq)
        torch.testing.assert_close(talker.decoder.norm.weight, torch.ones_like(talker.decoder.norm.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in talker.parameters()))

    def test_talker_only_and_full_checkpoints_reload(self):
        from omni_speech.model.builder import load_talker_weights, talker_only_base
        from omni_speech.train.stage2 import Stage2Trainer
        from transformers import AutoConfig

        with tempfile.TemporaryDirectory() as directory:
            save_stage1(f"{directory}/stage1")
            config = s2s_config(f"{directory}/stage1")
            model = OmniSpeech2SQwen3_5ForCausalLM.from_pretrained(f"{directory}/stage1", config=config)
            with torch.no_grad():
                for param in model.speech_generator.parameters():
                    param.normal_(std=0.1)
            expected = model.state_dict()

            # LLM frozen: only talker.safetensors plus a config pointing at Stage 1.
            args = TrainingArguments(output_dir=f"{directory}/stage2", report_to=[])
            Stage2Trainer(model=model, args=args, talker_only=True).save_model(f"{directory}/stage2")
            base = talker_only_base(f"{directory}/stage2")
            self.assertEqual(base, f"{directory}/stage1")
            config = AutoConfig.from_pretrained(f"{directory}/stage2")
            self.assertIsInstance(config, OmniSpeech2SQwen3_5Config)
            reloaded = OmniSpeech2SQwen3_5ForCausalLM.from_pretrained(base, config=config)
            load_talker_weights(reloaded, f"{directory}/stage2")
            for name, value in reloaded.state_dict().items():
                torch.testing.assert_close(value, expected[name], msg=name)

            # --tune_llm: a full checkpoint.
            model.save_pretrained(f"{directory}/full")
            reloaded = OmniSpeech2SQwen3_5ForCausalLM.from_pretrained(f"{directory}/full")
            for name, value in reloaded.state_dict().items():
                torch.testing.assert_close(value, expected[name], msg=name)


@needs_qwen
class QwenS2STrainerTest(unittest.TestCase):
    def test_trainer_averages_the_loss_itself(self):
        """The talker loss is a batch mean: the Trainer must divide it by the accumulation steps."""
        from omni_speech.train.stage2 import Stage2Trainer

        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, report_to=[])
            trainer = Stage2Trainer(model=tiny_s2s(), args=args, talker_only=True)
        self.assertFalse(trainer.model_accepts_loss_kwargs)


@needs_qwen
@needs_gpu
class QwenS2SModelTest(unittest.TestCase):
    def batch(self):
        input_ids = torch.randint(1, 64, (2, 12), device="cuda")
        labels = torch.full((2, 12), IGNORE_INDEX, device="cuda")
        labels[0, 4:10] = input_ids[0, 4:10]
        labels[1, 6:12] = input_ids[1, 6:12]
        units = torch.full((2, 21), IGNORE_INDEX, device="cuda")
        units[0] = random_units(3)[1]
        units[1, :14] = random_units(2)[1]
        return input_ids, labels, units

    def test_talker_loss_with_frozen_and_trained_llm(self):
        for tune_llm in (False, True):
            model = tiny_s2s(tune_speech_generator_only=not tune_llm).cuda().train()
            input_ids, labels, units = self.batch()
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           labels=labels, tgt_units=units)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertIsNotNone(model.speech_generator.gate.weight.grad)
            llm_grad = model.get_model().layers[0].linear_attn.out_proj.weight.grad
            self.assertEqual(llm_grad is not None, tune_llm)

    def test_streaming_matches_generate(self):
        model = tiny_s2s().cuda().eval()
        inputs = torch.tensor([[1, 5, 9, 13]], device="cuda")
        talker_kwargs = {"temperature": 0, "repetition_penalty": 1.0, "max_tokens": 70}
        text_ids, units = model.generate(
            inputs, max_new_tokens=6, do_sample=False, talker_kwargs=talker_kwargs,
            eos_token_id=None, pad_token_id=0,
        )
        self.assertEqual(text_ids.shape, (1, 6))
        text, streamed = [], []
        for kind, ids in model.stream_generate(inputs, max_new_tokens=6, eos_token_id=-1,
                                               talker_kwargs=talker_kwargs):
            (text if kind == "text" else streamed).extend(ids)
        self.assertEqual(text, text_ids[0].tolist())
        whole = len(units) // FRAME_TOKENS * FRAME_TOKENS
        self.assertEqual(streamed, units[:whole].tolist())


if __name__ == "__main__":
    unittest.main()
