# Latxa-Omni-SNAC: SNAC speech output + a training plan for a strong Basque SpeechLM

## Context

Latxa-Omni works like LLaMA-Omni. The Whisper-large-v3 encoder (Basque fine-tune) feeds a projector into Latxa-Llama-3.1-8B-Instruct. Stage 2 then adds a non-autoregressive CTC head that predicts 1000 k-means mHuBERT units, and a unit HiFi-GAN turns those into audio. The weak points are known:
- CTC units carry no prosody or speaker information, so the voice sounds flat.
- The output speech is 16 kHz.
- Stage 1 causes catastrophic forgetting (`Latxa-Omni/FORGETTING.md`).

The goal is a new sibling directory that swaps the CTC/HuBERT/HiFi-GAN output for **SNAC 24 kHz** codec tokens, plus a staged training plan grounded in the SpeechLLM Playbook (HF, June 2026).

The original CTC/HiFi-GAN system is in [Latxa-Omni](https://github.com/ansuehu/Latxa-Omni).

---

## Part A — Architecture (Thinker–Talker, LLaMA-Omni2 style)

**Why a separate talker instead of Orpheus-style "LLM emits SNAC tokens":**
- The 8B LLM stays a text model, which avoids worsening forgetting.
- Text quality is decoupled from audio.
- A talker of about 1B parameters is cheap to run autoregressively, at about 82 SNAC tokens per second of speech.
- The existing Stage 2 structure (frozen LLM, train only the generator) is unchanged.

**SNAC encoding** (`omni_speech/model/speech_generator/snac_codec.py`)
- Model: `hubertsiuzdak/snac_24khz`. It has 3 codebooks of 4096 entries each, at about 12, 23 and 47 Hz.
- Each frame is flattened to 7 tokens in Orpheus order, `c0, c1, c2, c2, c1, c2, c2`. Slot k is offset by k·4096, so the talker vocabulary is 28,672 + BOS + EOS.
- `SnacCodec.encode/decode`, plus a CLI round-trip (`python -m …snac_codec in.wav out.wav`).

**Talker** (`omni_speech/model/speech_generator/talker.py`, `SpeechGeneratorSNAC`)
- Condition per answer token: a gated fusion of the LLM's last hidden state and that token's embedding, `g·MLP(h) + (1−g)·Lin(e)`.
- Decoder: a Llama decoder with the Llama-3.2-1B shape, initialised from `meta-llama/Llama-3.2-1B` (`--talker-init-from`). Its own audio embeddings and LM head are new.
- Two layouts:
  - `prefix`: conditions, then BOS, audio, EOS.
  - `interleave` (default, streamable): R=3 conditions, then W=28 audio tokens (4 frames ≈ 0.34 s), repeated, then the remaining audio and EOS. This needs W/R below the real ratio of audio tokens per text token (measured 12–15 on the OmniVoice InstructS2S answers; 28/3 ≈ 9.3), so the text stays ahead of the audio.
- Generation constrains each token to its frame slot's codebook and allows EOS only at a frame boundary. It also applies temperature, top-k/top-p and repetition penalty.
- `stream_units()` accepts a generator of conditions, so it can be fed while the LLM is still decoding.

**Wiring and scripts**
- `model/language_model/omni_speech2s_llama.py`: `GenerationWithCTC` is removed.
  - Forward: `tgt_units` triggers the talker loss on the **last assistant span**, using `last_answer_mask` on the expanded labels.
  - `tune_speech_generator_only` runs the LLM under `no_grad`. Otherwise the loss is LM loss + talker loss.
  - `generate()` produces the text first, re-runs one forward pass over prompt + answer (exactly matching training), then produces the SNAC units.
- `train/stage2.py`: rewritten in the style of `stage1.py`.
  - Reuses `Stage1Dataset`, `build_prompt_and_labels`, `collate_fn` and `split_dev` from `train/stage1.py`.
  - Adds `answer_snac` .npy loading and a `--max_units` filter.
  - Two learning rates: the talker at 2e-4, and the LLM at 1e-5 with `--tune_llm`.
  - `DevAudioCallback` writes and logs dev wavs with wandb.
  - DeepSpeed: `ds_config_stage2.json` uses ZeRO-2 with no optimizer block, so the HF optimizer and its param groups are used.
- `infer/infer.py`: speech in, JSONL with text and a 24 kHz wav out.
- `dataset_generation/dump_snac_tokens.py` + `.slurm`: an 8-shard array writing int16 .npy files plus `answer_snac_len`.
- `configs/talker-snac.json`. The `ctc_*` and `speech_generator_type` keys are removed from the LLM configs, so Stage 1 does not copy them.
- Copied unchanged: the encoder, projector, `stage1.py`, the eval scripts and the tests. 
- `tests/test_snac_talker.py`: CPU tests for the flatten round-trip, both layouts, loss/backward, overfitting one example then decoding it back exactly, and the frame-slot constraint.

---

## Part B — Training plan

### Phase 0 — Codec and measurement gate (1–2 days)
- Round-trip about 200 Basque clips through SNAC: real speech (Common Voice eu, Dialektoak) and the synthetic OmniVoice voices.
- Score the reconstructions with Basque ASR WER (fine-tuned whisper-large-v3, eval WER 12.09), UTMOS and speaker similarity (WavLM SECS), against the originals.
- **Gate:** WER within about 1 point of the originals. If SNAC fails on Basque, fall back to Mimi or XCodec2 (the talker is codec-agnostic apart from `FRAME_TOKENS`/vocabulary).
- Freeze the eval suite now, so every later stage reports the same numbers:
  - text (EusProficiency, EusTrivia, XNLI-eu via `text_forgetting_eval.py`)
  - spoken QA (text accuracy + LLM-judge)
  - S2S ASR-WER/CER of the generated speech
  - UTMOS, SECS
  - latency to first audio

### Phase 1 — Data (the largest lever)
- **Self-distilled targets:** have base Latxa-Instruct answer the transcript of every spoken question with vLLM 0.11. These become the Stage 1/2 text targets. This keeps Latxa's style and is the biggest forgetting fix.
- **Spoken questions:** use real speech wherever possible (CV-eu, Basque Parliament, Dialektoak). For synthetic questions, use many speakers with augmentation (noise, reverb, gain, codec, telephone). The playbook warns that training only on TTS speech generalises badly to accented speech.
- **Spoken answers:** use **one consistent target voice**, plus optionally 2–3 selectable voices with a speaker tag. Synthesise with the best Basque TTS (OmniVoice), then filter by ASR-WER (≤5%), UTMOS and length. Spell out numbers with `basque_numbers.spell_numbers()` before TTS. Write answers in a spoken register: short, no markdown or lists.
- **Talker-only TTS data** (text → SNAC, no LLM answer needed): Basque TTS corpora such as Maider/Antton plus filtered OmniVoice output, about 500–2000 h. Run Latxa over the text to get hidden states. This teaches pronunciation broadly.
- **Text replay:** 20–30% Latxa SFT text-only examples. Needs a small change in `build_prompt_and_labels` so a user turn can be text without audio.
- **Task diversity:** add ASR ("transcribe"), speech translation (eu↔es/en), summarise-what-was-said, and speech + separate text instruction.
- Pre-encode everything once: mel features for questions, SNAC .npy for answers. Store tokens, not wavs.

### Phase 2 — Stage 1: speech understanding (thinker)
- **1a, alignment:** train only the projector on ASR and paraphrase data (the encoder and LLM stay frozen). LR 1e-3, about 1 epoch. This gives a stable starting point; projector-only training plateaued at about 1.1 loss on QA, which is fine as a warm-up only.
- **1b, SFT:** full LLM at LR 1e-5 (lower than the current 2e-5), or LoRA r=64–128 plus the projector.
  - Data: self-distilled spoken QA + 20–30% text replay + varied tasks.
  - Global batch 64–128, early stopping on dev loss (the existing `SaveOnBestDevCallback`).
  - Optionally unfreeze the top layers of the encoder at 1e-6 after about 20% of tokens.
- **Gate:** text benchmarks within about 1–2 points of base Latxa. If they drop further, weight-merge with α≈0.5–0.8 (`merge_stage1_with_base.py`).

### Phase 3 — Stage 2: talker (LLM frozen)
- **2a:** pretrain the talker on talker-only TTS data plus the S2S answers.
  - Talker initialised from Llama-3.2-1B, LR 2e-4 cosine with 2% warmup.
  - Global batch about 128 utterances, up to 30 s of audio (2460 tokens).
  - Layout `interleave` R=3 W=28, 3 epochs on 4×H100 (`run_stage2.slurm`).
- **2b:** fine-tune on the S2S set only, using the hidden states of the Stage 1 thinker, at LR 5e-5.
- Monitor dev loss, dev wavs every 500 steps, and ASR-WER of the generated speech (target under 8% CER).
- Sweep R/W, sampling temperature (0.6–0.8) and repetition penalty (1.05–1.2).

### Phase 4 — Stage 3: joint fine-tuning (optional)
- `JOINT=1`: LLM at 5e-6 and talker at 5e-5, with the talker loss weight tuned at 0.5–1.
- Run it briefly on high-quality spoken data with text replay. Stop if text benchmarks regress.

### Phase 5 — Preference and RL (after a solid SFT)
- **Talker GRPO / DPO:** for each answer, sample 4–8 audio versions and score them with a composite reward: −CER from Basque ASR, + ASR log-likelihood, + UTMOS, + SECS to the target voice, − a duration penalty. The playbook warns that a WER-only reward causes flat prosody and voice drift. CosyVoice2 GRPO: CER 1.41→1.07, MOS 4.42→4.58.
- **Thinker DPO:** run it on spoken QA with judge-ranked pairs. Check ASR and text scores before and after; Voxtral's DPO regressed ASR, so keep the SFT checkpoint if this one regresses.

### Phase 6 — Streaming and serving
- Feed `stream_units()` with conditions as the LLM decodes, with a one-token lag.
- Decode SNAC every 4 frames with overlap.
- Target under 500 ms to first audio.
- Port `serve/` to SNAC (follow-up).

### Later options
- Full-duplex / Moshi-style multi-stream training. Needs two-channel Basque dialogue, which does not exist yet.
- Moving to a Mimi/XCodec2 semantic codec if SNAC limits quality.

---

## Verification
0. Env check: `omni-snac/bin/python -c "import snac, torch, transformers, flash_attn"` runs without errors. `omni/bin/pip list` matches its state from before the clone, i.e. `omni` is unchanged.
1. Unit tests: `tests/test_snac_talker.py` plus the existing Stage 1 tests, all run with the `omni-snac` interpreter.
2. SNAC round-trip on 10 Basque wavs, listen and run ASR (Phase 0 gate).
3. `SMOKE=1 sbatch omni_speech/train/run_stage2.slurm` with `MODEL_PATH` set to an existing Stage 1 checkpoint and a 1k-row SNAC-dumped dataset: 2 steps, eval, one dev wav written.
4. Overfit about 50 examples for a few hundred steps, then check that `infer.py` produces intelligible speech that matches the text.
