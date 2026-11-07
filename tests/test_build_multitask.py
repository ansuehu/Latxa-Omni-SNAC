import json
import os
import random
import tempfile
import unittest

from dataset_generation.build_multitask import (
    chat_row, chat_text_row, composite_source, latxa_st_row, latxa_st_text_row, load_translations,
    sample_text, st_en_row, st_en_text_row,
)
from omni_speech.train.stage1 import USER_PROMPT, user_prompt

ROW = {"id": "instruct_en_3", "turns": [
    {"role": "user", "audio": "/x/instruct_en_3-1-user.flac", "text": "Nola murriztu airearen kutsadura?"},
    {"role": "assistant", "audio": "", "text": "Garraio publikoa erabiliz."},
    {"role": "user", "audio": "/x/instruct_en_3-2-user.flac", "text": "Eta uraren kutsadura?"},
    {"role": "assistant", "audio": "", "text": "Hondakinak ondo botaz."},
]}
ENGLISH = {"instruct_en_3": ["How can we reduce air pollution?", "What about water pollution?"]}


class CompositeSourceTest(unittest.TestCase):
    def test_sources(self):
        self.assertEqual(composite_source("common_voice_eu_38606144.wav"), "cv")
        self.assertEqual(composite_source("12-079_20220217_01_13965.96_13969.89.wav"), "parl")
        self.assertEqual(composite_source("eu_female_00012.wav"), "oslr")


class ChatRowTest(unittest.TestCase):
    def test_instruction_only_on_first_turn(self):
        row = chat_row(ROW, ["Erantzun."], 1.0, random.Random(0))
        self.assertEqual([t["instruction"] for t in row["turns"]], ["Erantzun.", "", "", ""])
        self.assertEqual(row["task"], "chat")

    def test_probability_zero_leaves_bare_speech(self):
        row = chat_row(ROW, ["Erantzun."], 0.0, random.Random(0))
        self.assertTrue(all(t["instruction"] == "" for t in row["turns"]))


class StEnRowTest(unittest.TestCase):
    def test_pairs_audio_with_english_of_same_turn(self):
        for seed in range(10):
            row = st_en_row(ROW, ENGLISH, ["Itzuli."], random.Random(seed))
            user, answer = row["turns"]
            turn = int(user["audio"].split("-")[-2])
            self.assertEqual(answer["text"], ENGLISH["instruct_en_3"][turn - 1])

    def test_skips_unknown_conversation(self):
        self.assertIsNone(st_en_row({"id": "mt_switch_train_1", "turns": ROW["turns"]}, ENGLISH, ["x"], random.Random(0)))


class TextRowTest(unittest.TestCase):
    def test_st_en_text_pairs_basque_text_with_english_of_same_turn(self):
        basque = {t["text"]: i // 2 for i, t in enumerate(ROW["turns"]) if t["role"] == "user"}
        for seed in range(10):
            user, answer = st_en_text_row(ROW, ENGLISH, ["Itzuli:"], random.Random(seed))["turns"]
            self.assertEqual(user["audio"], "")
            self.assertEqual(user_prompt(user), f"Itzuli:\n{user['text']}")
            self.assertEqual(answer["text"], ENGLISH["instruct_en_3"][basque[user["text"]]])

    def test_spoken_st_en_rows_are_unchanged(self):
        # st_en_row draws from the generator as before the written variant existed.
        rng = random.Random(5)
        expected = rng.choice([(1, "How can we reduce air pollution?"), (2, "What about water pollution?")])
        row = st_en_row(ROW, ENGLISH, ["Itzuli.", "Itzuli!"], random.Random(5))
        self.assertEqual(row["turns"][1]["text"], expected[1])
        self.assertEqual(row["turns"][0]["instruction"], ["Itzuli.", "Itzuli!"][rng.randrange(2)])

    def test_chat_text_writes_every_user_turn(self):
        row = chat_text_row(ROW)
        self.assertEqual(row["task"], "chat_text")
        self.assertEqual([t["audio"] for t in row["turns"]], [""] * 4)
        self.assertEqual([t["text"] for t in row["turns"]], [t["text"] for t in ROW["turns"]])
        self.assertEqual(user_prompt(row["turns"][0]), "Nola murriztu airearen kutsadura?")
        empty = {"id": "x", "turns": [{**ROW["turns"][0], "text": ""}, ROW["turns"][1]]}
        self.assertIsNone(chat_text_row(empty))

    def test_sample_text_skips_unusable_rows(self):
        rows = [ROW, {"id": "mt_switch_train_1", "turns": ROW["turns"]}] * 3
        make = lambda row: st_en_text_row(row, ENGLISH, ["x"], random.Random(0))
        self.assertEqual(len(sample_text(rows, 2, make, random.Random(0))), 2)
        self.assertEqual(len(sample_text(rows, None, make, random.Random(0))), 3)


class LatxaTranslationTest(unittest.TestCase):
    def test_load_filters_and_groups(self):
        sources = [
            {"id": "asr_cv_1", "source": "cv", "split": "train", "eu": "Gure beldurrak ere hor daude.", "audio": "/a/1.flac"},
            {"id": "e_1", "source": "eusparallel", "split": "train", "eu": "Lekeition antzara jokoa egiten da.", "audio": ""},
            {"id": "e_2", "source": "eusparallel", "split": "train", "eu": "Bai.", "audio": ""},
            {"id": "e_3", "source": "eusparallel", "split": "train", "eu": "Ez dago ezer.", "audio": ""},
            {"id": "f_1", "source": "flores", "split": "devtest", "eu": "Kaixo mundua.", "audio": "", "en_ref": "Hello world."},
        ]
        english = [
            {"id": "asr_cv_1", "en": "Our fears are also there.", "finish_reason": "stop"},
            {"id": "e_1", "en": "The goose game is held in Lekeitio.", "finish_reason": "stop"},
            {"id": "e_2", "en": "Here is the translation of the Basque word: yes.", "finish_reason": "stop"},
            {"id": "e_3", "en": "There is nothing", "finish_reason": "length"},
            {"id": "f_1", "en": "Hi world.", "finish_reason": "stop"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name, rows in (("sources.jsonl", sources), ("en.0.jsonl", english)):
                with open(os.path.join(directory, name), "w") as stream:
                    stream.writelines(json.dumps(row) + "\n" for row in rows)
            by_split = load_translations(directory)
        # e_2 is far longer than its Basque and e_3 did not stop: both dropped.
        self.assertEqual({k: [r["id"] for r in v] for k, v in by_split.items()},
                         {("cv", "train"): ["asr_cv_1"], ("eusparallel", "train"): ["e_1"], ("flores", "devtest"): ["f_1"]})
        spoken = latxa_st_row(by_split[("cv", "train")][0], "en", ["Itzuli."], random.Random(0))
        self.assertEqual((spoken["task"], spoken["source"], spoken["turns"][0]["audio"], spoken["turns"][1]["text"]),
                         ("st_en", "cv", "/a/1.flac", "Our fears are also there."))
        flores = latxa_st_text_row(by_split[("flores", "devtest")][0], "en", ["Itzuli:"], random.Random(0))
        self.assertEqual(user_prompt(flores["turns"][0]), "Itzuli:\nKaixo mundua.")
        self.assertEqual(flores["turns"][1]["text"], "Hello world.")  # the human reference, not Latxa's


class UserPromptTest(unittest.TestCase):
    def test_prompts(self):
        spoken = {"role": "user", "audio": "a.flac"}
        self.assertEqual(user_prompt(spoken), USER_PROMPT)
        self.assertEqual(user_prompt({**spoken, "instruction": None}), USER_PROMPT)
        self.assertEqual(user_prompt({**spoken, "instruction": ""}), "<speech>")
        self.assertEqual(user_prompt({**spoken, "instruction": "Transkribatu."}), "Transkribatu.\n<speech>")


if __name__ == "__main__":
    unittest.main()
