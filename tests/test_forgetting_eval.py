import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from omni_speech.evaluate.merge_stage1_with_base import interpolate_model, merge, parse_spec
from omni_speech.evaluate.text_forgetting_eval import chat_example, letter_choices, letter_prompt, load_task


def write_checkpoint(directory, shards):
    directory.mkdir()
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, directory / shard)
        weight_map.update({key: shard for key in tensors})
    with open(directory / "model.safetensors.index.json", "w", encoding="utf-8") as stream:
        json.dump({"weight_map": weight_map}, stream)
    (directory / "config.json").write_text("{}")


class MergeTest(unittest.TestCase):
    def test_interpolates_shared_weights_and_copies_speech_weights(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            projector = torch.full((2, 2), 7.0, dtype=torch.bfloat16)
            write_checkpoint(root / "stage1", {
                "a.safetensors": {"lm_head.weight": torch.full((2, 2), 4.0, dtype=torch.bfloat16)},
                "b.safetensors": {
                    "model.norm.weight": torch.full((2,), 2.0, dtype=torch.bfloat16),
                    "model.speech_projector.linear.weight": projector,
                },
            })
            # Base shards are laid out differently from Stage 1's.
            write_checkpoint(root / "base", {"x.safetensors": {
                "lm_head.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                "model.norm.weight": torch.ones(2, dtype=torch.bfloat16),
            }})
            outputs = merge(root / "stage1", root / "base", [0.0, 0.25, 1.0], root / "out")
            merged = load_file(outputs[0.25] / "a.safetensors")
            self.assertTrue(torch.equal(merged["lm_head.weight"], torch.full((2, 2), 1.0, dtype=torch.bfloat16)))
            merged = load_file(outputs[0.25] / "b.safetensors")
            self.assertTrue(torch.equal(merged["model.norm.weight"], torch.full((2,), 1.25, dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(merged["model.speech_projector.linear.weight"], projector))
            self.assertTrue(torch.equal(
                load_file(outputs[0.0] / "a.safetensors")["lm_head.weight"], torch.zeros(2, 2, dtype=torch.bfloat16)))
            self.assertTrue(torch.equal(
                load_file(outputs[1.0] / "a.safetensors")["lm_head.weight"], torch.full((2, 2), 4.0, dtype=torch.bfloat16)))
            self.assertTrue((outputs[0.25] / "config.json").is_file())
            self.assertTrue((outputs[0.25] / "model.safetensors.index.json").is_file())

    def test_rejects_base_weight_missing_from_stage1(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_checkpoint(root / "stage1", {"a.safetensors": {"lm_head.weight": torch.zeros(1)}})
            write_checkpoint(root / "base", {"a.safetensors": {
                "lm_head.weight": torch.zeros(1), "model.norm.weight": torch.zeros(1)}})
            with self.assertRaises(ValueError):
                merge(root / "stage1", root / "base", [0.5], root / "out")


class InterpolateModelTest(unittest.TestCase):
    def test_merges_shared_parameters_in_place(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            write_checkpoint(root / "base", {"x.safetensors": {"lm_head.weight": torch.zeros(2, 2)}})
            model = torch.nn.Module()
            model.lm_head = torch.nn.Linear(2, 2, bias=False)
            model.speech_projector = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.constant_(model.lm_head.weight, 4.0)
            torch.nn.init.constant_(model.speech_projector.weight, 7.0)
            interpolate_model(model, str(root / "base"), 0.25)
            self.assertTrue(torch.equal(model.lm_head.weight, torch.ones(2, 2)))
            self.assertTrue(torch.equal(model.speech_projector.weight, torch.full((2, 2), 7.0)))

    def test_parse_spec(self):
        self.assertEqual(parse_spec("/ckpt/checkpoint-3250@0.5"), ("/ckpt/checkpoint-3250", 0.5))
        self.assertEqual(parse_spec("HiTZ/Latxa"), ("HiTZ/Latxa", None))


class PromptTest(unittest.TestCase):
    def test_letter_prompt(self):
        self.assertEqual(
            letter_prompt("Zer?", ["bat", "bi"], passage="Testua."),
            "Pasartea: Testua.\nGaldera: Zer?\nA: bat\nB: bi\nErantzuna:",
        )
        self.assertEqual(letter_choices(3), [" A", " B", " C"])

    def test_chat_example(self):
        prompt, choices = chat_example("Galdera: Zer?\nA: bat\nB: bi\nErantzuna:", letter_choices(2))
        self.assertTrue(prompt.startswith("<|start_header_id|>system<|end_header_id|>"))
        self.assertIn("<|start_header_id|>user<|end_header_id|>\n\nErantzun galderari", prompt)
        self.assertTrue(prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))
        self.assertEqual(choices, ["A", "B"])

    def test_chat_example_qwen3_5(self):
        prompt, choices = chat_example("Galdera: Zer?\nA: bat\nB: bi\nErantzuna:", letter_choices(2), "qwen3_5")
        self.assertTrue(prompt.startswith("<|im_start|>system\n"))
        self.assertIn("<|im_end|>\n<|im_start|>user\nErantzun galderari", prompt)
        self.assertTrue(prompt.endswith("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(choices, ["A", "B"])


class LoadTaskTest(unittest.TestCase):
    def test_eus_reading_and_eus_exams(self):
        reading = load_task("eus_reading")
        self.assertEqual(len(reading), 352)
        context, choices, gold = reading[0]
        self.assertTrue(context.startswith("Pasartea: "))
        self.assertIn(gold, range(len(choices)))
        exams = load_task("eus_exams")
        self.assertEqual(len(exams), 895 + 2531 + 6005 + 1361)
        self.assertTrue(all(len(choices) == 4 for _, choices, _ in exams))


if __name__ == "__main__":
    unittest.main()
