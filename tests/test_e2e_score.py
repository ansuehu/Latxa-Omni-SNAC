import json
import tempfile
import unittest
from pathlib import Path

from omni_speech.evaluate import e2e_score


def generation(conv_id, turn, first_audio=0.3, audio_s=2.0):
    return {"id": conv_id, "turn": turn, "last_turn": True, "question": "Zer da?", "instruction": "",
            "reference": "", "history": [{"question": "Kaixo?", "answer": "Kaixo!"}] if turn else [],
            "prediction": "Bai.", "prediction_wav": "x.wav", "audio_s": audio_s, "encode_s": 0.01,
            "first_text_s": 0.1, "first_audio_s": first_audio, "total_s": 1.0}


class E2EScoreTest(unittest.TestCase):
    def test_parse_score(self):
        self.assertEqual(e2e_score.parse_score('{score: 4}'), 4.0)
        self.assertEqual(e2e_score.parse_score('```json\n{"score": 2}\n```'), 2.0)
        self.assertIsNone(e2e_score.parse_score("no idea"))

    def test_is_basque(self):
        self.assertTrue(e2e_score.is_basque("Bai, noski. Euskal Herrian 3 lurralde daude."))
        self.assertFalse(e2e_score.is_basque("Hona hemen: def batu(a, b): return a + b"))
        self.assertFalse(e2e_score.is_basque('"The Daily Show with Jon Stewart" saioa'))
        self.assertFalse(e2e_score.is_basque('Velázquez-en "Las Meninas" eta "La familia de Carlos IV"'))

    def test_windows(self):
        self.assertEqual([len(w) for w in e2e_score.windows(list(range(50)), 1)], [10] * 5)
        self.assertEqual([len(w) for w in e2e_score.windows(list(range(25)), 1, 10, 3)], [10, 10, 5])
        self.assertEqual([len(w) for w in e2e_score.windows(list(range(22)), 1, 10, 3)], [10, 10])
        self.assertEqual([len(w) for w in e2e_score.windows(list(range(8)), 1, 10, 3)], [8])

    def test_judge_prompt_adds_context_only_for_follow_ups(self):
        self.assertNotIn("follow-up", e2e_score.judge_prompt(generation("a", 0), "Bai."))
        prompt = e2e_score.judge_prompt(generation("a", 1), "Bai.")
        self.assertIn("follow-up", prompt)
        self.assertIn("Kaixo!", prompt)

    def test_summary_merges_steps_and_keeps_last_copy(self):
        with tempfile.TemporaryDirectory() as out_dir:
            out = Path(out_dir)
            with open(out / "generations.shard00.jsonl", "w") as stream:
                for row in [generation("a", 0, first_audio=9.0), generation("a", 0), generation("a", 1)]:
                    stream.write(json.dumps(row) + "\n")
            with open(out / "asr.jsonl", "w") as stream:
                for k, errors in [("a-0", 1), ("a-1", 0)]:
                    stream.write(json.dumps({"key": k, "asr": "bai", "ref_words": 10, "word_errors": errors,
                                             "ref_chars": 40, "char_errors": errors}) + "\n")
            with open(out / "quality.jsonl", "w") as stream:
                stream.write(json.dumps({"key": "a-0", "nisqa": 3.0, "secs_intra": 0.8, "secs_turn0": None}) + "\n")
                stream.write(json.dumps({"key": "a-1", "nisqa": 4.0, "secs_intra": None, "secs_turn0": 0.6}) + "\n")
            with open(out / "judge.jsonl", "w") as stream:
                stream.write(json.dumps({"key": "a-0", "judge": 4.0}) + "\n")
                stream.write(json.dumps({"key": "a-1", "judge": 2.0}) + "\n")
            e2e_score.run_summary(type("Args", (), {"out_dir": [out_dir]}))
            summary = json.load(open(out / "summary.json"))
        self.assertEqual(summary["all"]["answers"], 2)
        self.assertEqual(summary["all"]["judge"], 3.0)
        self.assertEqual(summary["all"]["wer"], 5.0)
        self.assertEqual(summary["all"]["latency_first_audio_ms_median"], 300)
        self.assertEqual(summary["first_turn"]["judge"], 4.0)
        self.assertEqual(summary["basque_only"]["answers"], 2)
        self.assertEqual(summary["all"]["nisqa"], 3.5)
        self.assertEqual(summary["all"]["secs_intra"], 0.8)
        self.assertEqual(summary["follow_up"]["secs_turn0"], 0.6)
        self.assertIsNone(summary["all"]["utmos_win"])


if __name__ == "__main__":
    unittest.main()
