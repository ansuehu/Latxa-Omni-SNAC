import os
import unittest

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.train.stage1 import build_generation_prompt, build_prompt_and_labels, split_dev

TOKENIZER = os.environ.get(
    "LATXA_TOKENIZER",
    "HiTZ/Latxa-Llama-3.1-8B-Instruct",
)
TURNS = [
    {"role": "user", "text": "Kaixo", "audio": "a.flac"},
    {"role": "assistant", "text": "Kaixo, zer moduz?", "audio": ""},
    {"role": "user", "text": "Ondo", "audio": "b.flac"},
    {"role": "assistant", "text": "Pozten naiz.", "audio": ""},
]


class SplitDevTest(unittest.TestCase):
    def test_split_is_fixed_and_disjoint(self):
        held_out = Dataset.from_dict({"id": [str(i) for i in range(100)]})
        dev, test = split_dev(held_out, 30, seed=3407)
        dev_again, _ = split_dev(held_out, 30, seed=3407)
        self.assertEqual(dev["id"], dev_again["id"])
        self.assertEqual((len(dev), len(test)), (30, 70))
        self.assertFalse(set(dev["id"]) & set(test["id"]))
        self.assertEqual(set(dev["id"]) | set(test["id"]), set(held_out["id"]))

    def test_rejects_dev_as_large_as_split(self):
        held_out = Dataset.from_dict({"id": ["a", "b"]})
        with self.assertRaises(ValueError):
            split_dev(held_out, 2, seed=0)


class GenerationPromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    def test_prompt_is_training_input_up_to_last_answer(self):
        input_ids, labels, _ = build_prompt_and_labels(TURNS, self.tokenizer, "llama_3")
        prompt_ids, audio_paths = build_generation_prompt(TURNS, self.tokenizer, "llama_3")
        last_answer_start = int(labels.ne(IGNORE_INDEX).nonzero()[-1]) + 1
        while labels[last_answer_start - 1] != IGNORE_INDEX:
            last_answer_start -= 1
        self.assertTrue(torch.equal(prompt_ids, input_ids[:last_answer_start]))
        self.assertEqual(audio_paths, ["a.flac", "b.flac"])
        self.assertEqual(int((prompt_ids == SPEECH_TOKEN_INDEX).sum()), 2)


if __name__ == "__main__":
    unittest.main()
