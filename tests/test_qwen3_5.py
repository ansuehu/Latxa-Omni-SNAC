"""Qwen3.5 thinker: chat template, LoRA targets, forward/generate and merged LoRA save.

Needs transformers >= 5.2 (omni-qwen env); skipped elsewhere. The forward tests need a GPU, as
fla's Gated DeltaNet kernels are Triton-only.
"""

import tempfile
import unittest

import torch
from torch import nn
from transformers import TrainingArguments

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.conversation import conv_templates
from omni_speech.model.speech_projector.builder import build_speech_projector
from omni_speech.train.stage1 import Stage1Trainer, add_lora, build_generation_prompt, build_prompt_and_labels

try:
    from omni_speech.model.language_model.omni_speech_qwen3_5 import (
        OmniSpeechQwen3_5Config, OmniSpeechQwen3_5ForCausalLM,
    )
except ImportError:
    OmniSpeechQwen3_5ForCausalLM = None

TOKENIZER = "HiTZ/Latxa-Qwen3.5-4B"
needs_qwen = unittest.skipIf(OmniSpeechQwen3_5ForCausalLM is None, "transformers without Qwen3.5")
needs_gpu = unittest.skipUnless(torch.cuda.is_available(), "fla kernels need a GPU")


def tiny_model():
    """A linear-attention layer and a full-attention layer; the speech encoder is a frozen stand-in."""
    torch.manual_seed(0)
    config = OmniSpeechQwen3_5Config(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        tie_word_embeddings=True,
        speech_encoder_hidden_size=16, speech_encoder_ds_rate=5, speech_projector_type="linear",
    )
    model = OmniSpeechQwen3_5ForCausalLM(config)
    model.get_model().speech_projector = build_speech_projector(config)
    encoder = nn.Linear(16, 16)
    encoder.requires_grad_(False)
    model.get_model().speech_encoder = encoder
    return model


class ChatTemplateTest(unittest.TestCase):
    def test_prompt_grows_by_appending(self):
        conv = conv_templates["qwen3_5"].copy()
        conv.append_message(conv.roles[0], "<speech>")
        conv.append_message(conv.roles[1], None)
        header = conv.get_prompt()
        self.assertTrue(header.startswith("<|im_start|>system\n"))
        self.assertTrue(header.endswith("<|im_end|>\n<|im_start|>user\n<speech><|im_end|>\n"
                                        "<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        conv.messages[-1][1] = "Kaixo."
        completed = conv.get_prompt()
        self.assertEqual(completed, header + "Kaixo.<|im_end|>")
        conv.append_message(conv.roles[0], "Eta?")
        self.assertTrue(conv.get_prompt().startswith(completed + "\n<|im_start|>user\n"))


@needs_qwen
class QwenTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        except OSError as error:
            raise unittest.SkipTest(f"tokenizer {TOKENIZER} not cached: {error}")

    def test_labels_cover_answers_up_to_end_of_turn(self):
        turns = [
            {"role": "user", "audio": "q1.wav"},
            {"role": "assistant", "text": "Bilbo da."},
            {"role": "user", "text": "Eta hiriburua?"},
            {"role": "assistant", "text": "Gasteiz."},
        ]
        input_ids, labels, audio_paths = build_prompt_and_labels(turns, self.tokenizer, "qwen3_5")
        self.assertEqual(audio_paths, ["q1.wav"])
        self.assertEqual(int((input_ids == SPEECH_TOKEN_INDEX).sum()), 1)
        self.assertNotEqual(input_ids[0].item(), self.tokenizer.bos_token_id)
        targets = input_ids[labels.ne(IGNORE_INDEX)]
        self.assertEqual(self.tokenizer.decode(targets), "Bilbo da.<|im_end|>Gasteiz.<|im_end|>")
        prompt_ids, _ = build_generation_prompt(turns, self.tokenizer, "qwen3_5")
        self.assertTrue(self.tokenizer.decode(prompt_ids[prompt_ids.ge(0)]).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))


@needs_qwen
class QwenLoraTest(unittest.TestCase):
    def test_every_projection_gets_an_adapter(self):
        model = add_lora(tiny_model(), r=4, alpha=8, dropout=0.05)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        for name in trainable:
            self.assertTrue(".lora_" in name or ".speech_projector." in name, name)
        lora_targets = {name.split(".lora_A")[0].split(".model.layers.")[1]
                        for name in trainable if ".lora_A." in name}
        mlp = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
        expected = {f"0.{proj}" for proj in mlp | {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                                                   "linear_attn.out_proj"}}
        expected |= {f"1.{proj}" for proj in mlp | {f"self_attn.{p}_proj" for p in "qkvo"}}
        self.assertEqual(lora_targets, expected)


@needs_qwen
class QwenCheckpointTest(unittest.TestCase):
    def test_loads_every_weight_of_a_plain_qwen3_5_checkpoint(self):
        """transformers saves Qwen3.5 weights as model.language_model.*; they must load, not stay random."""
        import json

        from safetensors.torch import load_file, save_file
        from transformers import Qwen3_5ForCausalLM

        source = tiny_model()
        plain = Qwen3_5ForCausalLM(source.config)
        plain.load_state_dict(source.state_dict(), strict=False)
        with tempfile.TemporaryDirectory() as directory:
            plain.save_pretrained(directory)
            # Make it look like extract_qwen3_5_text.py's output: a qwen3_5_text config and
            # multimodal weight names.
            with open(f"{directory}/config.json", encoding="utf-8") as stream:
                config = json.load(stream)
            config["model_type"] = "qwen3_5_text"
            with open(f"{directory}/config.json", "w", encoding="utf-8") as stream:
                json.dump(config, stream)
            weights = load_file(f"{directory}/model.safetensors")
            save_file({key.replace("model.", "model.language_model.", 1): value for key, value in weights.items()},
                      f"{directory}/model.safetensors", metadata={"format": "pt"})
            # As in stage1.py: the base config plus the speech fields, so a projector is built.
            config = OmniSpeechQwen3_5Config.from_pretrained(directory)
            config.speech_encoder, config.speech_encoder_type = "whisper", "whisper_hf"
            loaded = OmniSpeechQwen3_5ForCausalLM.from_pretrained(directory, config=config)
            self.assertIsNotNone(loaded.get_model().speech_projector)
            # A Stage 1 checkpoint (projector included) must round-trip too.
            loaded.save_pretrained(f"{directory}/stage1")
            reloaded = OmniSpeechQwen3_5ForCausalLM.from_pretrained(f"{directory}/stage1")
        expected = plain.state_dict()
        for name, value in loaded.state_dict().items():
            if name in expected:
                torch.testing.assert_close(value, expected[name], msg=name)
        self.assertTrue(any(".layers.0.linear_attn." in name for name in expected))
        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[name], msg=name)


@needs_qwen
@needs_gpu
class QwenModelTest(unittest.TestCase):
    def test_speech_features_replace_the_speech_token(self):
        model = tiny_model().cuda().eval()
        features = torch.randn(1, 3, 64, device="cuda")
        input_ids = torch.tensor([[5, SPEECH_TOKEN_INDEX, 6, 7]], device="cuda")
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 6, 7]], device="cuda")
        _, _, _, _, embeds, new_labels = model.prepare_inputs_labels_for_speech_and_text(
            input_ids, None, torch.ones_like(input_ids), None, labels, None,
            torch.tensor([3], device="cuda"), speech_embeds=features,
        )
        self.assertEqual(embeds.shape, (1, 6, 64))
        torch.testing.assert_close(embeds[0, 1:4], features[0])
        output = model(inputs_embeds=embeds, labels=new_labels)
        self.assertTrue(torch.isfinite(output.loss))
        generated = model.generate(
            input_ids, speech_embeds=features, speech_lengths=torch.tensor([3], device="cuda"),
            max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0,
        )
        self.assertEqual(generated.shape, (1, 4))

    def test_merged_save_reloads_with_the_same_logits(self):
        model = add_lora(tiny_model(), r=4, alpha=8, dropout=0.0)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_B." in name:  # lora_B starts at zero, which would make the merge a no-op
                    param.normal_(std=0.1)
        model = model.cuda().eval()
        input_ids = torch.tensor([[1, 5, 9, 3, 7]], device="cuda")
        with torch.no_grad():
            expected = model(input_ids=input_ids).logits
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(output_dir=output_dir, report_to=[])
            Stage1Trainer(model=model, args=args).save_model(output_dir)
            loaded = OmniSpeechQwen3_5ForCausalLM.from_pretrained(output_dir).cuda().eval()
        with torch.no_grad():
            actual = loaded(input_ids=input_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
