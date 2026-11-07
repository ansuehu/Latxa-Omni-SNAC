import tempfile
import unittest

import torch
from torch import nn
from transformers import LlamaForCausalLM, TrainingArguments
from safetensors import safe_open

from omni_speech.model.language_model.omni_speech_llama import OmniSpeechConfig, OmniSpeechLlamaForCausalLM
from omni_speech.model.speech_projector.builder import build_speech_projector
from omni_speech.train.stage1 import Stage1Trainer, add_lora


def tiny_model():
    """A 2-layer OmniSpeech model; the speech encoder is a stand-in with Whisper's module names."""
    torch.manual_seed(0)
    config = OmniSpeechConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        speech_encoder_hidden_size=16, speech_encoder_ds_rate=5, speech_projector_type="linear",
    )
    model = OmniSpeechLlamaForCausalLM(config)
    model.get_model().speech_projector = build_speech_projector(config)
    attention = nn.Module()
    attention.q_proj, attention.k_proj, attention.v_proj = (nn.Linear(16, 16) for _ in range(3))
    layer = nn.Module()
    layer.self_attn = attention
    encoder = nn.Module()
    encoder.encoder = nn.Module()
    encoder.encoder.layers = nn.ModuleList([layer])
    for param in encoder.parameters():
        param.requires_grad = False
    model.get_model().speech_encoder = encoder
    return model


class Stage1LoraTest(unittest.TestCase):
    def test_trainable_parameters(self):
        model = add_lora(tiny_model(), r=4, alpha=8, dropout=0.05)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertTrue(trainable)
        for name in trainable:
            self.assertTrue(".lora_" in name or ".speech_projector." in name, name)
        self.assertFalse(any("speech_encoder" in name for name in trainable))
        # Every attention and MLP projection of every LLM layer gets an adapter.
        lora_targets = {name.split(".lora_A")[0].split(".model.layers.")[1]
                        for name in trainable if ".lora_A." in name}
        expected = {f"{layer}.{proj}" for layer in range(2) for proj in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        self.assertEqual(lora_targets, expected)
        projector = model.get_base_model().get_model().speech_projector
        self.assertTrue(all(param.requires_grad for param in projector.parameters()))

    def test_merged_save_loads_as_plain_checkpoint(self):
        model = add_lora(tiny_model(), r=4, alpha=8, dropout=0.0)
        plain_keys = set(tiny_model().state_dict())
        with torch.no_grad():
            for name, param in model.named_parameters():
                if ".lora_B." in name:  # lora_B starts at zero, which would make the merge a no-op
                    param.normal_(std=0.1)
        model.eval()
        input_ids = torch.tensor([[1, 5, 9, 3, 7]])
        with torch.no_grad():
            expected = model(input_ids=input_ids).logits
        with tempfile.TemporaryDirectory() as output_dir:
            args = TrainingArguments(output_dir=output_dir, report_to=[], use_cpu=True)
            Stage1Trainer(model=model, args=args).save_model(output_dir)
            with safe_open(f"{output_dir}/model.safetensors", "pt") as stream:
                self.assertEqual(set(stream.keys()), plain_keys)
            loaded = LlamaForCausalLM.from_pretrained(output_dir).eval()
            with torch.no_grad():
                actual = loaded(input_ids=input_ids).logits
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        # Saving left the live LoRA weights unmerged.
        self.assertFalse(any(getattr(m, "merged", False) for m in model.modules()))


if __name__ == "__main__":
    unittest.main()
