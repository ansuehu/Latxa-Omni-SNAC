import unittest

from omni_speech.train.dev_metrics import corpus_bleu, corpus_chrf, corpus_wer, task_metrics
from omni_speech.train.stage1 import generation_indices, summarize_generations


class MetricsTest(unittest.TestCase):
    def test_wer_ignores_case_and_punctuation(self):
        self.assertEqual(corpus_wer(["Kaixo, zer moduz?"], ["kaixo zer moduz"]), 0.0)
        # one substitution and one deletion over 4 reference words
        self.assertEqual(corpus_wer(["bat bi hiru lau"], ["bat bost hiru"]), 50.0)

    def test_bleu_and_chrf_bounds(self):
        reference = ["How can we reduce air pollution in big cities?"]
        self.assertAlmostEqual(corpus_bleu(reference, reference), 100.0)
        self.assertAlmostEqual(corpus_chrf(reference, reference), 100.0)
        self.assertEqual(corpus_bleu(reference, ["Kaixo"]), 0.0)
        self.assertEqual(corpus_bleu(reference, [""]), 0.0)

    def test_task_metrics(self):
        self.assertEqual(set(task_metrics("asr", ["a"], ["a"])), {"wer"})
        self.assertEqual(set(task_metrics("st_en", ["a"], ["a"])), {"bleu", "chrf"})
        self.assertEqual(set(task_metrics("st_en_text", ["a"], ["a"])), {"bleu", "chrf"})
        self.assertEqual(task_metrics("chat", ["a"], ["b"]), {})
        self.assertEqual(task_metrics("chat_text", ["a"], ["b"]), {})


class GenerationSplitTest(unittest.TestCase):
    def test_first_rows_of_each_task(self):
        tasks = ["chat", "asr", "chat", "st_en", "chat", "asr", "asr"]
        self.assertEqual(generation_indices(tasks, 2), [0, 1, 2, 3, 5])

    def test_summary_keys(self):
        results = [
            {"task": "asr", "reference": "bat bi", "prediction": "bat bi"},
            {"task": "st_en", "reference": "one two three four", "prediction": "one two three four"},
            {"task": "chat", "reference": "x", "prediction": "y"},
        ]
        by_task, metrics = summarize_generations(results)
        self.assertEqual(sorted(by_task), ["asr", "chat", "st_en"])
        self.assertEqual(metrics, {"asr_wer": 0.0, "st_en_bleu": 100.0, "st_en_chrf": 100.0})


if __name__ == "__main__":
    unittest.main()
