import types
import unittest

import torch

from omni_speech.constants import IGNORE_INDEX
from omni_speech.model.speech_generator.snac_codec import (
    CODEBOOK_SIZE, FRAME_TOKENS, flatten_codes, unflatten_codes,
)
from omni_speech.model.speech_generator.talker import (
    BOS, COND, EOS, SpeechGeneratorSNAC, build_layout, last_answer_mask,
)

TINY = {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8}


def tiny_talker(layout="interleave", read=2, write=7):
    config = types.SimpleNamespace(
        hidden_size=48, talker_config=TINY, talker_layout=layout, talker_read=read,
        talker_write=write, talker_attn_implementation="eager",
    )
    torch.manual_seed(0)
    return SpeechGeneratorSNAC(config)


def random_units(frames):
    codes = [torch.randint(0, CODEBOOK_SIZE, (frames * n,)) for n in (1, 2, 4)]
    return codes, flatten_codes(codes)


class SnacCodesTest(unittest.TestCase):
    def test_round_trip(self):
        codes, flat = random_units(5)
        self.assertEqual(flat.shape[0], 5 * FRAME_TOKENS)
        for original, restored in zip(codes, unflatten_codes(flat)):
            self.assertTrue(torch.equal(original, restored[0]))

    def test_slot_offsets(self):
        _, flat = random_units(3)
        slots = flat.view(3, FRAME_TOKENS) // CODEBOOK_SIZE
        self.assertTrue(torch.equal(slots, torch.arange(FRAME_TOKENS).expand(3, -1)))

    def test_partial_frame_dropped(self):
        _, flat = random_units(2)
        self.assertEqual(unflatten_codes(flat[:-3])[0].shape[1], 1)


class LayoutTest(unittest.TestCase):
    def test_prefix(self):
        audio = torch.arange(4)
        tokens, cond_index, labels = build_layout(3, audio, "prefix", 2, 7)
        self.assertEqual(tokens.tolist(), [COND] * 3 + [BOS, 0, 1, 2, 3, EOS])
        self.assertEqual(cond_index.tolist()[:4], [0, 1, 2, -1])
        # Only the BOS -> audio -> EOS transitions are supervised.
        self.assertEqual(labels.tolist(), [IGNORE_INDEX] * 3 + [0, 1, 2, 3, EOS, IGNORE_INDEX])

    def test_interleave(self):
        audio = torch.arange(10)
        tokens, cond_index, labels = build_layout(5, audio, "interleave", 2, 3)
        self.assertEqual(tokens.tolist(),
                         [COND, COND, 0, 1, 2, COND, COND, 3, 4, 5, COND, 6, 7, 8, 9, EOS])
        self.assertEqual(cond_index[cond_index >= 0].tolist(), [0, 1, 2, 3, 4])
        # The last audio token of a chunk is followed by text: no label there.
        self.assertEqual(labels[4].item(), IGNORE_INDEX)
        self.assertEqual(labels[1].item(), 0)
        self.assertEqual(labels[-2].item(), EOS)

    def test_last_answer_mask(self):
        labels = torch.tensor([IGNORE_INDEX, 5, 6, IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9, IGNORE_INDEX])
        self.assertEqual(last_answer_mask(labels).nonzero().squeeze(-1).tolist(), [5, 6, 7])


class TalkerTest(unittest.TestCase):
    def batch(self):
        hidden = torch.randn(2, 12, 48)
        text_embeds = torch.randn(2, 12, 48)
        labels = torch.full((2, 12), IGNORE_INDEX)
        labels[0, 4:10] = 1
        labels[1, 6:12] = 1
        units = torch.full((2, 21), IGNORE_INDEX)
        units[0] = random_units(3)[1]
        units[1, :14] = random_units(2)[1]
        return hidden, labels, text_embeds, units

    def test_loss_backward(self):
        for layout in ("prefix", "interleave"):
            talker = tiny_talker(layout)
            loss = talker(*self.batch())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(talker.gate.weight.grad)

    def test_overfits_one_example(self):
        talker = tiny_talker("interleave")
        hidden, labels, text_embeds, units = self.batch()
        optimizer = torch.optim.Adam(talker.parameters(), lr=3e-3)
        for _ in range(150):
            optimizer.zero_grad()
            loss = talker(hidden[:1], labels[:1], text_embeds[:1], units[:1])
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), 0.1)
        mask = last_answer_mask(labels[0])
        cond = talker.fuse(hidden[0][mask], text_embeds[0][mask])
        generated = talker.generate_units(cond, temperature=0, repetition_penalty=1.0)
        self.assertEqual(generated.tolist(), units[0].tolist())

    def test_generation_respects_frame_slots(self):
        for layout in ("prefix", "interleave"):
            talker = tiny_talker(layout).eval()
            cond = talker.fuse(torch.randn(5, 48), torch.randn(5, 48))
            units = talker.generate_units(cond, max_tokens=50)
            self.assertNotIn(EOS, units.tolist())
            slots = units // CODEBOOK_SIZE
            self.assertTrue(torch.equal(slots, torch.arange(len(units)) % FRAME_TOKENS))


class StreamGenerateTest(unittest.TestCase):
    def test_streaming_matches_generate(self):
        from omni_speech.model.language_model.omni_speech2s_llama import (
            OmniSpeech2SConfig, OmniSpeech2SLlamaForCausalLM,
        )

        config = OmniSpeech2SConfig(
            vocab_size=100, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, eos_token_id=99,
            speech_generator_type="snac_ar", talker_config=TINY, talker_read=2,
            talker_write=7, talker_attn_implementation="eager", attn_implementation="eager",
        )
        torch.manual_seed(0)
        model = OmniSpeech2SLlamaForCausalLM(config).eval()
        inputs = torch.tensor([[1, 5, 9, 13]])
        talker_kwargs = {"temperature": 0, "repetition_penalty": 1.0, "max_tokens": 70}
        text_ids, units = model.generate(
            inputs, max_new_tokens=6, do_sample=False, talker_kwargs=talker_kwargs, pad_token_id=0,
        )
        text, streamed = [], []
        for kind, ids in model.stream_generate(inputs, max_new_tokens=6, talker_kwargs=talker_kwargs):
            (text if kind == "text" else streamed).extend(ids)
        self.assertEqual(text, text_ids[0].tolist())
        whole = len(units) // FRAME_TOKENS * FRAME_TOKENS
        self.assertEqual(streamed, units[:whole].tolist())


if __name__ == "__main__":
    unittest.main()
