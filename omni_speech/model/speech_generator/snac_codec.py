"""SNAC 24 kHz codec helpers.

SNAC has three codebooks (4096 entries each) at ~12, ~23 and ~47 Hz. One
coarse frame (~85 ms) holds 1 + 2 + 4 = 7 codes, which we flatten in the
Orpheus order:

    c0[i], c1[2i], c2[4i], c2[4i+1], c1[2i+1], c2[4i+2], c2[4i+3]

Code k of a frame is offset by k * 4096, so the talker vocabulary is
7 * 4096 and the codebook of a token is implied by its value. That gives
~82 tokens per second of speech.

Round-trip a file to hear what the codec does to Basque speech:

    python -m omni_speech.model.speech_generator.snac_codec in.wav out.wav
"""

import numpy as np
import torch

SNAC_MODEL = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24000
CODEBOOK_SIZE = 4096
FRAME_TOKENS = 7
# For each of the 7 slots in a frame: (codebook level, index offset inside the frame).
FRAME_LAYOUT = ((0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (2, 2), (2, 3))


def flatten_codes(codes):
    """[c0 (T,), c1 (2T,), c2 (4T,)] -> flat (7T,) LongTensor with slot offsets."""
    c0, c1, c2 = (c.reshape(-1).long() for c in codes)
    frames = c0.shape[0]
    if c1.shape[0] != 2 * frames or c2.shape[0] != 4 * frames:
        raise ValueError(f"bad SNAC code lengths: {c0.shape[0]}, {c1.shape[0]}, {c2.shape[0]}")
    levels = (c0.view(frames, 1), c1.view(frames, 2), c2.view(frames, 4))
    slots = [levels[level][:, offset] + slot * CODEBOOK_SIZE
             for slot, (level, offset) in enumerate(FRAME_LAYOUT)]
    return torch.stack(slots, dim=1).reshape(-1)


def unflatten_codes(flat):
    """Flat talker tokens -> [c0 (1,T), c1 (1,2T), c2 (1,4T)]; a trailing partial frame is dropped."""
    flat = torch.as_tensor(flat).reshape(-1).long()
    frames = flat.shape[0] // FRAME_TOKENS
    grid = flat[:frames * FRAME_TOKENS].view(frames, FRAME_TOKENS)
    grid = grid - torch.arange(FRAME_TOKENS, device=grid.device) * CODEBOOK_SIZE
    if grid.numel() and (grid.min() < 0 or grid.max() >= CODEBOOK_SIZE):
        raise ValueError("token outside its frame slot's codebook range")
    levels = [torch.empty(frames, n, dtype=torch.long, device=grid.device) for n in (1, 2, 4)]
    for slot, (level, offset) in enumerate(FRAME_LAYOUT):
        levels[level][:, offset] = grid[:, slot]
    return [level.reshape(1, -1) for level in levels]


class SnacCodec:
    def __init__(self, model_name=SNAC_MODEL, device=None):
        from snac import SNAC

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SNAC.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def encode(self, waveform, sample_rate):
        """Mono waveform (numpy or tensor, any rate) -> flat int16 numpy array of talker tokens."""
        import torchaudio

        waveform = torch.as_tensor(waveform, dtype=torch.float32).to(self.device)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != SNAC_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, SNAC_SAMPLE_RATE)
        codes = self.model.encode(waveform.view(1, 1, -1))
        return flatten_codes([c[0] for c in codes]).cpu().numpy().astype(np.int16)

    @torch.no_grad()
    def decode(self, flat):
        """Flat talker tokens -> 24 kHz mono float32 numpy waveform."""
        codes = [c.to(self.device) for c in unflatten_codes(flat)]
        if codes[0].shape[1] == 0:
            return np.zeros(0, dtype=np.float32)
        return self.model.decode(codes)[0, 0].float().cpu().numpy()


class SnacStreamDecoder:
    """Turns a growing stream of flat talker tokens into audio chunks.

    SNAC's decoder is non-causal, so a frame's samples only settle once a few
    later frames exist. Each decode covers ``context`` already-emitted frames,
    the new frames and ``lookahead`` frames, and only the new frames' samples
    are returned, so chunks join without seams. The first chunk is smaller
    to cut the time to first audio.
    """

    def __init__(self, codec, chunk_frames=4, first_chunk_frames=2, context=2, lookahead=2):
        self.codec = codec
        self.chunk_frames = chunk_frames
        self.first_chunk_frames = first_chunk_frames
        self.context = context
        self.lookahead = lookahead
        model = codec.model
        self.frame_samples = model.hop_length * model.vq_strides[0]
        self.tokens = []
        self.done = 0

    def push(self, tokens):
        """Add tokens; return the newly final samples (possibly empty)."""
        self.tokens.extend(int(t) for t in tokens)
        frames = len(self.tokens) // FRAME_TOKENS
        ready = frames - self.lookahead
        need = self.chunk_frames if self.done else self.first_chunk_frames
        if ready - self.done < need:
            return np.zeros(0, dtype=np.float32)
        return self._emit(ready, frames)

    def flush(self):
        """Samples for every remaining whole frame."""
        frames = len(self.tokens) // FRAME_TOKENS
        if frames <= self.done:
            return np.zeros(0, dtype=np.float32)
        return self._emit(frames, frames)

    def _emit(self, stop, window_end):
        start = max(0, self.done - self.context)
        audio = self.codec.decode(self.tokens[start * FRAME_TOKENS:window_end * FRAME_TOKENS])
        out = audio[(self.done - start) * self.frame_samples:(stop - start) * self.frame_samples]
        self.done = stop
        return out


if __name__ == "__main__":
    import sys

    import soundfile as sf

    source, target = sys.argv[1:3]
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    codec = SnacCodec()
    tokens = codec.encode(audio.T, rate)
    sf.write(target, codec.decode(tokens), SNAC_SAMPLE_RATE)
    print(f"{len(tokens)} tokens for {audio.shape[0] / rate:.2f}s -> {target}")
