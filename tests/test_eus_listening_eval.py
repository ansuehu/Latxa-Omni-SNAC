import tempfile
import unittest
import wave
from pathlib import Path

import torch
from transformers import AutoTokenizer

from omni_speech.constants import SPEECH_TOKEN_INDEX
from omni_speech.evaluate.eus_listening_eval import audio_chunks, build_prompt, letter_ids


class EusListeningTest(unittest.TestCase):
    def test_audio_chunks_cover_the_whole_recording(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "a.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0\0" * 16000 * 65)
            chunks = audio_chunks(str(path), 128)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(chunk.shape == (3000, 128) and chunk.dtype == torch.bfloat16 for chunk in chunks))

    def test_prompt_and_letters(self):
        tokenizer = AutoTokenizer.from_pretrained("HiTZ/Latxa-Llama-3.1-8B-Instruct")
        item = {"question": "Zer?", "candidates": ["bat", "bi", "hiru"]}
        input_ids = build_prompt(item, 3, tokenizer, "llama_3")
        self.assertEqual(int((input_ids == SPEECH_TOKEN_INDEX).sum()), 3)
        self.assertEqual(int((build_prompt(item, 0, tokenizer, "llama_3") == SPEECH_TOKEN_INDEX).sum()), 0)
        self.assertTrue(tokenizer.decode(input_ids[-5:]).endswith("assistant<|end_header_id|>\n\n"))
        self.assertEqual(len(set(letter_ids(tokenizer, 4))), 4)


if __name__ == "__main__":
    unittest.main()
