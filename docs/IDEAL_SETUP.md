# Ideal setup for Latxa-Omni-SNAC

This is the setup I'd aim for if time and compute were no object. Each item is followed by why it matters and where the current setup differs. The staged schedule is in [TRAINING_PLAN.md](TRAINING_PLAN.md); this file covers the **choices**: which data, which targets, which models, which thresholds.

## 1. The most important decisions

1. **Write the answers with Latxa itself.** Base Latxa-Llama-3.1-8B-Instruct should answer the *text transcript* of every spoken question, and those answers become the training targets for both stages.
   - **Why:** the model only has to learn that this speech means this text. It doesn't also have to learn a new style or new content. This is the largest fix for forgetting (see `../Latxa-Omni/FORGETTING.md`). DeSTA2 and BLSP report almost no forgetting with this approach.
   - **Today:** the targets are the translated InstructS2S answers.
2. **Use different data for Stage 1 and Stage 2.** They teach different things:

   | | Stage 1 (thinker: understanding) | Stage 2 (talker: speaking) |
   |---|---|---|
   | Input speech | As **diverse** as possible: many speakers, real recordings, accents and dialects, noise | Doesn't matter much (the LLM is frozen) |
   | Text target | Latxa's own answers + varied tasks + text replay | Latxa's own answers (what the talker will actually have to say) + plain TTS text |
   | Output audio | Not used | **One consistent, clean voice**, filtered hard for quality |
   | Size | As many conversations as possible | 1–2k h of target-voice speech |

   They can share conversations, but each stage filters and augments them differently.
3. **Re-synthesize the answer audio after changing the answer text.** The talker learns to speak the text it is conditioned on. If the Stage 1/2 answers change to Latxa's own answers, the `aintzane` answer audio, which speaks the translated answers, no longer matches. Regenerate it with the same voice.
4. **Keep Latxa-8B a text model.** Speech goes in through the projector and comes out through the talker. Never put SNAC tokens into Latxa's vocabulary.

## 2. Data

### 2.1 Spoken questions (Stage 1 input)

| Source | Role | Share of Stage 1 speech |
|---|---|---|
| **Real Basque speech with transcripts**: Common Voice eu, Basque Parliament, Dialektoak, EusPodcast, faktoria | Accents and real acoustics; ASR and "answer the transcript" tasks | ≥ 30% |
| **Synthetic questions, multi-speaker**: OmniVoice plus re-synthesized CV voices, 50+ speakers | Volume and conversational content | ≤ 70% |
| **Augmentation** on ~50% of clips: noise (MUSAN), reverb (RIR), gain, telephone band (8 kHz), codec artefacts | Robustness to microphones and rooms | (applied on the fly) |

- **Hold out speakers:** dev and test speakers must not appear in training.
- **Real-speech test set:** build a test set of real recorded spoken questions, about 300 with 20+ speakers including dialects. This is the number that matters; synthetic test sets overestimate quality.
- **Why:** the playbook warns that models trained only on TTS questions fail on real accented users.

### 2.2 Text targets (both stages)

- **Generate the answers:** run Latxa-Instruct (vLLM 0.11) on the question transcript, with a system prompt asking for a *spoken* register: short (2–5 sentences), no markdown, lists, URLs or code, and numbers that can be spoken. Generate with temperature 0.7 and keep one sample.
- **Filter the answers:** drop refusals and empty or overlong answers (more than about 25 s of speech). Drop answers where Latxa switches language. Drop math and code; these were already dropped once (`dropped_math_ids.list`).
- **Multi-turn data:** generate the turns in order, so each answer sees Latxa's own earlier answers as history.
- **Normalise before synthesis:** spell out numbers, dates, units and abbreviations (`basque_numbers.spell_numbers()`) so the text the talker sees matches what is spoken. Keep the unnormalised text for the Stage 1 target.

### 2.3 Stage 1 task mix

| Task | Share | Notes |
|---|---|---|
| Spoken question → Latxa answer (single and multi-turn) | 45% | The main task |
| **Text-only instruct replay** (Latxa SFT mix, Magpie-eu) | 25% | Needs user turns without audio in `build_prompt_and_labels` |
| ASR: "transcribe this" | 10% | On real speech; keeps acoustic grounding |
| Speech + separate text instruction (summarise, translate to es/en, answer in N words, extract) | 15% | Stops the model ignoring written instructions |
| Speech translation eu → es/en | 5% | Cheap to create from CV and Parliament data |

Use **varied user prompts**, not the single fixed `USER_PROMPT`: around 50 paraphrases, plus a "no prompt" case.

### 2.4 Talker data (Stage 2)

- **Voice:** **one** target voice (currently `aintzane`), optionally 2–3 extra voices selected by a speaker-id token added to the conditions.
- **Data:**
  - (a) Answer audio for the Latxa answers (section 2.2), one row per assistant turn.
  - (b) **TTS-only data:** about 500–2000 h of the same voice reading varied Basque text (news, Wikipedia, dialogue), with the hidden states taken from Latxa reading that text as an assistant turn. This teaches pronunciation of rare words, names and numbers.
- **Quality filters** (stricter than the current WER > 0.3 cut):

  | Filter | Threshold |
  |---|---|
  | ASR CER of the synthesized audio (Basque whisper-large-v3 fine-tune) | ≤ 5% |
  | UTMOS | ≥ 3.5 |
  | Speaker similarity to the reference voice (WavLM SECS) | ≥ 0.85 |
  | Length | 1–30 s (≤ 2460 SNAC tokens) |
  | Speaking rate | inside the 5th–95th percentile, which catches stuck or truncated TTS |
  | Leading/trailing silence | trimmed to ≤ 0.3 s |

- **Codec check:** before any training, measure SNAC 24 kHz round-trip WER and UTMOS on Basque (TRAINING_PLAN Phase 0). Switch codecs if it costs more than about 1 WER point.

## 3. Models and freezing

| Component | Choice | Train when |
|---|---|---|
| Speech encoder | Basque fine-tuned whisper-large-v3 (eval WER 12.09) | Frozen; optionally unfreeze the top 8 layers at LR 1e-6 for the last 50% of Stage 1 |
| Projector | 5× frame concat + 2-layer MLP (current) | Stage 1a alone (LR 1e-3), then with the LLM |
| Thinker | Latxa-Llama-3.1-8B-Instruct | Stage 1b: full fine-tuning at LR 1e-5 **or** LoRA r=128 at 1e-4. Frozen in Stage 2 |
| Talker | Llama-3.2-1B shape, initialised from Llama-3.2-1B, gated fusion of hidden state + token embedding | Stage 2 (LR 2e-4 then 5e-5); Stage 3 joint (LR 5e-5, LLM at 5e-6) |
| Codec | SNAC 24 kHz (7 tokens per 85 ms frame) | Frozen |
| Talker layout | `interleave`, R=3 text, W=28 audio | Measured audio/text ratio is 12–15, which keeps text ahead |

## 4. Hyperparameters

| Stage | Trainable | LR | Global batch | Length | Stop |
|---|---|---|---|---|---|
| 1a projector | projector | 1e-3, cosine, 3% warmup | 256 | ≤ 30 s audio | 1 epoch |
| 1b SFT | projector + LLM | 1e-5 (full FT), betas (0.9, 0.95), wd 0.01 | 64–128 conversations | 4096 tokens | Early stop on dev loss **and** a text-benchmark gate |
| 2a talker pretraining | talker | 2e-4, cosine, 2% warmup | ~128 utterances | ≤ 2460 SNAC tokens | ~3 epochs over TTS + S2S |
| 2b talker SFT | talker | 5e-5 | 128 | same | Dev loss and ASR-CER of generated speech |
| 3 joint (optional) | LLM + talker | 5e-6 / 5e-5 | 64 | same | Stop at the first text-benchmark regression |
| 4 RL (optional) | talker (GRPO), thinker (DPO) | 1e-6 – 5e-6 | 4–8 samples per prompt | — | Composite reward; keep the SFT checkpoint if ASR or text regresses |

**Checkpoints:** in Stage 2 save **only the talker**, plus the config pointing at the frozen Stage 1 checkpoint. A full checkpoint is currently 66 GB, of which 47 GB is DeepSpeed training state.

## 5. Evaluation (fixed before training, same numbers every stage)

| What | Metric | Data |
|---|---|---|
| Text ability (forgetting) | EusProficiency, EusTrivia, XNLI-eu accuracy (`text_forgetting_eval.py`) | Standard benchmarks |
| Spoken QA | LLM-judge score (1–10) and accuracy against the text-input answer | **Real recorded** test questions, unseen speakers |
| ASR (sanity) | WER | CV-eu test |
| Speech output | ASR-CER of the generated audio, UTMOS, SECS to the target voice, speaking rate | 300 test answers |
| Latency | Time to first audio, real-time factor | 50 prompts on 1 H100 |
| Human | MOS on 50 dialogues, 5+ native listeners | At milestones only |

**Gates:**
- Stage 1: text benchmarks within 1–2 points of base Latxa.
- Stage 2: CER of the generated speech under 5% and UTMOS at or above that of the TTS used for the data.

## 6. What I'd change first, given what exists today

1. Freeze the evaluation suite, including a small real-speech test set (section 5).
2. Generate Latxa answers for the existing InstructS2S-eu questions (vLLM, spoken style), then re-synthesize the `aintzane` answer audio for them with the strict filters in 2.4.
3. Retrain Stage 1 with those targets + 25% text replay + ASR + varied prompts, at LR 1e-5.
4. Train the talker (Stage 2) on the new answer audio, one row per turn, with Stage 2 saving only the talker.
5. Then add TTS-only talker data, joint fine-tuning, and RL, in that order.

The current full run uses the translated answers and their `aintzane` audio. It's a good way to validate the talker and SNAC end to end, but it isn't the final data.
