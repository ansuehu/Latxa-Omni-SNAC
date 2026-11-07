"""Autoregressive SNAC talker (replaces the CTC head + HuBERT units + HiFi-GAN).

The LLM (thinker) writes the text answer. For every answer token the talker
gets one condition vector: a gated fusion of the LLM's last hidden state at
that token and the token's own embedding (as in LLaMA-Omni2). A small Llama
decoder then writes flattened SNAC tokens (see snac_codec.py) that the SNAC
decoder turns into 24 kHz audio.

Two sequence layouts:

* ``prefix``: all conditions, BOS, audio tokens, EOS. Simplest, not streamable.
* ``interleave``: R conditions, W audio tokens, R conditions, W audio tokens,
  ..., then the remaining audio and EOS. The talker can start speaking after R
  text tokens, so it streams alongside the LLM. W must be a multiple of 7 (whole
  SNAC frames) and W / R must stay below the real audio-tokens-per-text-token
  ratio (measured 12-15 on the OmniVoice InstructS2S answers) so text
  always stays ahead of audio.
"""

from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, LlamaConfig, LlamaModel

from omni_speech.constants import IGNORE_INDEX
from .snac_codec import CODEBOOK_SIZE, FRAME_TOKENS

AUDIO_VOCAB = FRAME_TOKENS * CODEBOOK_SIZE
BOS = AUDIO_VOCAB
EOS = AUDIO_VOCAB + 1
TALKER_VOCAB = AUDIO_VOCAB + 2
COND = -1  # placeholder id for condition positions in a layout

# Llama-3.2-1B shape, so the decoder can start from its weights (--talker-init-from).
DEFAULT_TALKER_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "max_position_embeddings": 8192,
}


def last_answer_mask(labels):
    """Mask of the last contiguous supervised span (the final assistant turn)."""
    positions = torch.nonzero(labels != IGNORE_INDEX).squeeze(-1)
    if positions.numel() == 0:
        raise ValueError("no supervised answer tokens to condition the talker on")
    breaks = torch.nonzero(positions[1:] - positions[:-1] > 1).squeeze(-1)
    start = positions[breaks[-1] + 1] if breaks.numel() else positions[0]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[start:positions[-1] + 1] = True
    return mask


def build_layout(n_cond, audio, layout, read, write):
    """Token ids (COND at condition slots), condition indices and next-token labels."""
    audio = audio.tolist()
    if layout == "prefix":
        tokens = [COND] * n_cond + [BOS] + audio + [EOS]
    elif layout == "interleave":
        tokens, a = [], 0
        for c in range(0, n_cond, read):
            tokens += [COND] * min(read, n_cond - c)
            tokens += audio[a:a + write]
            a += write
        tokens += audio[a:] + [EOS]
    else:
        raise ValueError(f"unknown talker layout: {layout}")
    tokens = torch.tensor(tokens, dtype=torch.long)
    is_cond = tokens == COND
    cond_index = torch.cumsum(is_cond, 0) - 1
    cond_index[~is_cond] = -1
    following = tokens[1:]
    labels = torch.full_like(tokens, IGNORE_INDEX)
    supervised = (following >= 0) & (following != BOS)
    labels[:-1][supervised] = following[supervised]
    return tokens, cond_index, labels


class SpeechGeneratorSNAC(nn.Module):
    def __init__(self, config):
        super().__init__()
        talker_config = dict(DEFAULT_TALKER_CONFIG, **(getattr(config, "talker_config", None) or {}))
        self.layout = getattr(config, "talker_layout", "interleave")
        self.read = getattr(config, "talker_read", 3)
        self.write = getattr(config, "talker_write", 28)
        if self.write % FRAME_TOKENS:
            raise ValueError(f"talker_write must be a multiple of {FRAME_TOKENS}")
        decoder_config = LlamaConfig(
            vocab_size=TALKER_VOCAB,
            attn_implementation=getattr(config, "talker_attn_implementation", "sdpa"),
            **talker_config,
        )
        self.decoder = LlamaModel(decoder_config)
        dim = decoder_config.hidden_size
        self.hidden_proj = nn.Sequential(
            nn.Linear(config.hidden_size, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.text_proj = nn.Linear(config.hidden_size, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.lm_head = nn.Linear(dim, TALKER_VOCAB, bias=False)

    def load_pretrained_decoder(self, model_name_or_path):
        """Start the decoder from a pretrained Llama (its token embeddings are not reused)."""
        source = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        state = {k: v for k, v in source.state_dict().items() if not k.startswith("embed_tokens.")}
        missing, unexpected = self.decoder.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("embed_tokens.")]
        if missing or unexpected:
            raise ValueError(f"talker init mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def fuse(self, hidden, text_embeds):
        """LLM hidden states and token embeddings of answer tokens -> talker conditions."""
        dtype = self.gate.weight.dtype
        h = self.hidden_proj(hidden.to(dtype))
        e = self.text_proj(text_embeds.to(dtype))
        g = torch.sigmoid(self.gate(torch.cat([h, e], dim=-1)))
        return g * h + (1 - g) * e

    def embed_layout(self, cond, tokens, cond_index):
        embeds = self.decoder.embed_tokens(tokens.clamp(min=0))
        cond_rows = cond[cond_index.clamp(min=0)]
        return torch.where((cond_index >= 0).unsqueeze(-1), cond_rows, embeds)

    def forward(self, hidden, labels, text_embeds, tgt_units):
        """Mean next-token loss over audio tokens and EOS.

        hidden / text_embeds: (B, L, H) LLM last hidden states and token embeddings,
        labels: (B, L) LLM labels after speech expansion, tgt_units: (B, N) flat SNAC
        tokens of the final assistant turn, padded with IGNORE_INDEX.
        """
        device = hidden.device
        seq_embeds, seq_labels = [], []
        for h, lab, emb, units in zip(hidden, labels, text_embeds, tgt_units):
            mask = last_answer_mask(lab)
            cond = self.fuse(h[mask], emb[mask])
            tokens, cond_index, targets = build_layout(
                cond.shape[0], units[units != IGNORE_INDEX], self.layout, self.read, self.write
            )
            seq_embeds.append(self.embed_layout(cond, tokens.to(device), cond_index.to(device)))
            seq_labels.append(targets.to(device))
        lengths = torch.tensor([len(x) for x in seq_labels], device=device)
        inputs_embeds = pad_sequence(seq_embeds, batch_first=True)
        targets = pad_sequence(seq_labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.arange(targets.shape[1], device=device)[None, :] < lengths[:, None]
        out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask.long())
        logits = self.lm_head(out.last_hidden_state).float()
        return F.cross_entropy(logits.view(-1, TALKER_VOCAB), targets.view(-1), ignore_index=IGNORE_INDEX)

    def _sample(self, logits, slot, allow_eos, seen, temperature, top_k, top_p, repetition_penalty):
        """One token restricted to the codebook of its frame slot (plus EOS at frame boundaries).

        ``seen`` is a (TALKER_VOCAB,) bool mask of tokens generated so far. Work is done on
        the 4096 + 1 candidates only, on the logits' device, with one host sync per token.
        """
        start = slot * CODEBOOK_SIZE
        logits = torch.cat([logits[start:start + CODEBOOK_SIZE], logits[EOS:EOS + 1]])
        if repetition_penalty != 1.0:
            penalised = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
            hit = torch.cat([seen[start:start + CODEBOOK_SIZE], seen[EOS:EOS + 1]])
            logits = torch.where(hit, penalised, logits)
        if not allow_eos:
            logits[-1] = float("-inf")
        if temperature <= 0:
            index = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p < 1.0:
                sorted_logits, order = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, -1)
                drop = probs.cumsum(-1) - probs > top_p
                logits[order[drop]] = float("-inf")
            index = int(torch.multinomial(torch.softmax(logits, -1), 1))
        return EOS if index == CODEBOOK_SIZE else start + index

    @torch.no_grad()
    def stream_units(self, conds, max_tokens=4096, temperature=0.7, top_k=50, top_p=0.95,
                     repetition_penalty=1.1):
        """Yield flat SNAC tokens.

        ``conds`` is an iterable of (dim,) condition vectors from ``fuse``. It may be a
        generator fed by the LLM as it decodes, so audio starts after ``read`` tokens.
        """
        from transformers import DynamicCache

        cache = DynamicCache()
        device = self.lm_head.weight.device
        seen = torch.zeros(TALKER_VOCAB, dtype=torch.bool, device=device)
        count = 0

        def step(embeds):
            out = self.decoder(inputs_embeds=embeds.unsqueeze(0), past_key_values=cache, use_cache=True)
            return self.lm_head(out.last_hidden_state[0, -1]).float()

        def embed(token):
            return self.decoder.embed_tokens(torch.tensor([token], device=device))

        def sample(logits, allow_eos):
            nonlocal count
            token = self._sample(logits, count % FRAME_TOKENS, allow_eos, seen,
                                 temperature, top_k, top_p, repetition_penalty)
            if token != EOS:
                seen[token] = True
                count += 1
            return token

        conds = iter(conds)
        if self.layout == "prefix":
            chunk = list(conds)
            if not chunk:
                return
            logits = step(torch.cat([torch.stack(chunk), embed(BOS)]))
        else:
            logits = None
            while chunk := list(islice(conds, self.read)):
                logits = step(torch.stack(chunk))
                for _ in range(self.write):
                    token = sample(logits, allow_eos=False)
                    yield token
                    if count >= max_tokens:
                        return
                    logits = step(embed(token))
            if logits is None:
                return
        while count < max_tokens:
            token = sample(logits, allow_eos=count % FRAME_TOKENS == 0)
            if token == EOS:
                return
            yield token
            logits = step(embed(token))

    def generate_units(self, cond, **kwargs):
        """Non-streaming helper: (T, dim) conditions -> (N,) LongTensor of flat SNAC tokens."""
        return torch.tensor(list(self.stream_units(cond, **kwargs)), dtype=torch.long)
