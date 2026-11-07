import os
import random
import unittest

import torch
from transformers import AutoTokenizer

from dataset_generation.add_tts_rows import stage2_tts_row
from dataset_generation.build_multitask import tts_row
from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX
from omni_speech.train.stage1 import Stage1Dataset, build_prompt_and_labels, collate_fn, user_prompt

TOKENIZER = os.environ.get(
    "LATXA_TOKENIZER",
    "HiTZ/Latxa-Llama-3.1-8B-Instruct",
)
SNAC_ROW = {
    "id": "instruct_en_0-1",
    "turns": [
        {"role": "user", "audio": "/x/instruct_en_0-1-user.flac", "speaker": "s", "text": "Kaixo?", "text_orig": ""},
        {"role": "assistant", "audio": "/x/instruct_en_0-1-assistant.flac", "speaker": "maider",
         "text": "Kaixo, zer moduz?", "text_orig": "Kaixo."},
    ],
    "answer_snac": "/x/0.npy",
    "answer_snac_len": 700,
}


class TextTurnPromptTest(unittest.TestCase):
    def test_written_turns(self):
        self.assertEqual(user_prompt({"audio": "", "instruction": "Irakurri ozen.", "text": "Kaixo"}),
                         "Irakurri ozen.\nKaixo")
        self.assertEqual(user_prompt({"audio": "", "instruction": "", "text": "Kaixo"}), "Kaixo")
        with self.assertRaises(ValueError):
            user_prompt({"audio": "", "instruction": "x", "text": ""})


class TtsRowTest(unittest.TestCase):
    def test_copies_the_spoken_answer(self):
        row = tts_row(SNAC_ROW, ["Irakurri ozen."], random.Random(0))
        user, answer = row["turns"]
        self.assertEqual((row["task"], user["audio"], user["text"], answer["text"]),
                         ("tts", "", "Kaixo, zer moduz?", "Kaixo, zer moduz?"))

    def test_stage2_row_keeps_the_snac_tokens(self):
        tts = tts_row(SNAC_ROW, ["Irakurri ozen."], random.Random(0))
        row = stage2_tts_row(SNAC_ROW, tts)
        self.assertEqual(row["answer_snac"], "/x/0.npy")
        self.assertEqual(row["turns"][0]["audio"], "")
        self.assertEqual(row["turns"][1]["audio"], SNAC_ROW["turns"][1]["audio"])
        self.assertEqual(set(row["turns"][0]), set(row["turns"][1]))


@unittest.skipUnless(os.path.isdir(TOKENIZER), f"tokenizer {TOKENIZER} not found")
class TextTurnBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=False)

    def test_written_turn_has_no_speech_token(self):
        turns = [{"role": "user", "audio": "", "instruction": "Irakurri ozen.", "text": "Kaixo"},
                 {"role": "assistant", "text": "Kaixo"}]
        input_ids, labels, audio_paths = build_prompt_and_labels(turns, self.tokenizer, "llama_3")
        self.assertEqual(audio_paths, [])
        self.assertEqual(int((input_ids == SPEECH_TOKEN_INDEX).sum()), 0)
        self.assertEqual(self.tokenizer.decode(labels[labels.ne(IGNORE_INDEX)]).strip()[:5], "Kaixo")

    def test_mixed_batch_gets_one_placeholder_per_text_row(self):
        dataset = Stage1Dataset([], self.tokenizer, None, "mel", 128, None, "llama_3")
        placeholder = dataset.speech_inputs([])
        self.assertEqual([tuple(t.shape) for t in placeholder], [(3000, 128)])
        spoken = torch.tensor([1, SPEECH_TOKEN_INDEX, 2])
        written = torch.tensor([1, 2, 3, 4])
        clip = torch.zeros(3000, 128)
        batch = [
            {"input_ids": written, "labels": written, "speech": placeholder, "speech_lengths": [3000]},
            {"input_ids": spoken, "labels": spoken, "speech": [clip], "speech_lengths": [3000]},
        ]
        self.assertEqual(collate_fn(batch, 0)["speech"].shape[0], 2)
        with self.assertRaises(ValueError):
            collate_fn([{**batch[0], "speech": [], "speech_lengths": []}, batch[1]], 0)


if __name__ == "__main__":
    unittest.main()
