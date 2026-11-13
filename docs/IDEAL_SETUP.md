# Ideal setup for Latxa-Omni-SNAC

This is the setup I'd aim for if time and compute were no object. Each item is followed by why it matters and where the current setup differs. The staged schedule is in [TRAINING_PLAN.md](TRAINING_PLAN.md); this file covers the **choices**: which data, which targets, which models, which thresholds.

Section 0 is the current status (updated 2026-11-10). Section 6 is what to do next, and section 7 lists new ideas that are not in the plan yet.

## 0. Where we are (2026-11-10)

**Best system:** Latxa-Qwen3.5-4B thinker, Stage 1 LoRA r=64 on multi-task v2 (run 57497, checkpoint-13000) + SNAC talker trained on its hidden states (run 57634, checkpoint-11859). The v3 thinker (57592, checkpoint-14750) matches it on the v2 tests and adds spoken follow-ups and emotion; the demo pairs it with the same talker.

**End-to-end test** (`e2e_generate.py` + `e2e_score.py`: the 2,720 spoken chats of the multi-task v2 `test` split, 5,485 answers, streaming demo path on 1 H100):

| System | Judge 1–5 (first / follow-up) | WER / CER of the speech | UTMOS (10 s windows) | NISQA | SECS within answer | First audio (median) | RTF |
|---|---|---|---|---|---|---|---|
| Latxa-Omni (IberSPEECH, CTC + HiFi-GAN) | 3.59 (4.14 / 3.05) | 7.72 / 2.58 | 2.64 | 4.37 | 0.86 | not streamed | 0.09 |
| Llama LoRA 57411 + talker 57393 | 3.68 (3.80 / 3.55) | 10.82 / 4.92 | 3.30 | 4.76 | 0.73 | 329 ms | 0.80 |
| Llama full FT 57389 + talker 57393 (300 chats) | — | 5.18 / 1.91 | — | — | — | 328 ms | 0.81 |
| **Qwen LoRA 57497 + talker 57634** | **4.37 (4.43 / 4.30)** | **4.35 / 1.47** | **3.30** | **4.80** | 0.73 | 344 ms | 0.76 |

What this says:
- The Qwen system wins on content (judge +0.8 over Latxa-Omni, +1.25 on follow-ups), intelligibility (CER 1.5%) and audio quality (UTMOS +0.66).
- **Voice drift is the clearest regression:** the first and last 4 s of an answer match at SECS 0.73, against 0.86 for the old system. Across turns the voice holds (0.91).
- **The talker depends on the exact thinker it was trained on.** Talker 57393 was trained on the full-FT Llama thinker (57389); paired with the LoRA thinker (57411) its WER doubles (5.2 → 10.8).
- Latency to first audio is about 330 ms, inside the 500 ms target, but the talker runs at RTF 0.76–0.81: barely faster than real time, with no headroom for a second user or a larger talker.
- Answers average 17–18 s of speech, longer than the 2–5 sentences the spoken register asks for.

**Thinker** (full numbers in `Latxa-Omni_evals_2026-11-07.xlsx`):

| | Llama LoRA 57411 | Qwen LoRA 57497 | Qwen LoRA v3 57592 |
|---|---|---|---|
| Text benchmarks, chat (base: 0.547 Llama, 0.635 Qwen) | 0.402 | 0.494 | 0.487 |
| Text benchmarks, plain (base: 0.500 Llama, 0.565 Qwen) | 0.381 | 0.519 | 0.516 |
| ASR WER (test_asr) | 7.89 | 5.89 | 5.89 |
| Speech → en / es BLEU (real speech) | 38.3 / 39.3 | 48.9 / 53.4 | 46.9 / 50.9 |
| Emotion from the voice (7 classes) | — | 0.12 | 0.71 |

- Forgetting is much smaller with LoRA and with Qwen, but still far from the gate: Qwen chat accuracy is **14 points** under its base (gate: 1–2). Chat prompts lose more than plain ones, so the chat behaviour itself is what gets damaged.
- EusListening has only been run on the old full-FT Llama thinkers, which sit at chance (0.32, the same with and without audio). The cascade Whisper + Latxa-8B gets 0.57 and Qwen3-Omni-30B 0.45.

**Status of each decision in this file:**

| Item | Status |
|---|---|
| 1.1 Latxa writes the answers | **Partly.** The answers come from Latxa-Llama-3.1-70B in a spoken register (`generate_latxa_answers.py`, with `--retry` and `--shorten`), not from the thinker itself. For the Qwen thinker this is still another model's style. |
| 1.2 Different data per stage | **No.** Both stages use the same conversations. |
| 1.3 Re-synthesize the answer audio | **Done**, with OmniVoice `maider` (not `aintzane`), via `run_answers_pipeline.sh`. Filter is still WER ≤ 0.3. |
| 1.4 Thinker stays a text model | Done. |
| 2.1 Real speech ≥ 30%, augmentation, real-speech test set | ASR and speech translation use real speech. Spoken chat is still synthetic only, no augmentation, **no real-speech QA test set**. |
| 2.3 Task mix + varied prompts | **Done** (multi-task v2: chat, ASR, st_en/es, text tasks, TTS, text replay, paraphrased instructions; v3 adds follow-ups and paralinguistics). |
| 2.4 TTS-only talker data, strict filters | Not done. |
| 2.4 Codec check | Not recorded as a gate. Talker CER 1.5% shows SNAC is not the bottleneck for intelligibility. |
| 3 Thinker choice | Extended: Latxa-Qwen3.5-4B is now the main thinker. LoRA r=64 at 2e-4 instead of full FT. |
| 4 Stage 2 saves only the talker | **Done** (`talker.safetensors`, ~2 GB; DeepSpeed state still saved alongside). |
| 4 Stage 3 joint / Stage 4 RL | Not started. |
| 5 Frozen eval suite | **Mostly done**: text benchmarks, per-task tests, EusListening, and now the e2e suite (judge, WER/CER, UTMOS, NISQA, SECS, latency). Missing: real recorded questions, human MOS. |
| 6 Streaming | **Done**: `stream_generate` feeds the talker as the thinker decodes; Gradio demo and demo page. |

## 1. The most important decisions

1. **Write the answers with the thinker itself.** The base thinker (Latxa-Qwen3.5-4B, or Latxa-Llama-3.1-8B-Instruct) should answer the *text transcript* of every spoken question, and those answers become the training targets for both stages.
   - **Why:** the model only has to learn that this speech means this text. It doesn't also have to learn a new style or new content. This is the largest fix for forgetting (see `../Latxa-Omni/FORGETTING.md`). DeSTA2 and BLSP report almost no forgetting with this approach.
   - **Today:** the targets are Latxa-70B answers in a spoken register. That fixed the translated-answer style, but the 4B thinker is still imitating a 70B model. Use 70B only where the thinker's own answer is wrong (judge-filtered), and keep the thinker's own answer otherwise.
2. **Use different data for Stage 1 and Stage 2.** They teach different things:

   | | Stage 1 (thinker: understanding) | Stage 2 (talker: speaking) |
   |---|---|---|
   | Input speech | As **diverse** as possible: many speakers, real recordings, accents and dialects, noise | Doesn't matter much (the LLM is frozen) |
   | Text target | The thinker's own answers + varied tasks + text replay | The same answers (what the talker will actually have to say) + plain TTS text |
   | Output audio | Not used | **One consistent, clean voice**, filtered hard for quality |
   | Size | As many conversations as possible | 1–2k h of target-voice speech |

   They can share conversations, but each stage filters and augments them differently.
3. **Re-synthesize the answer audio after changing the answer text.** The talker learns to speak the text it is conditioned on. Done once for the 70B answers (`maider`); it has to be redone whenever the targets change again.
4. **Keep the thinker a text model.** Speech goes in through the projector and comes out through the talker. Never put SNAC tokens into the thinker's vocabulary.

## 2. Data

### 2.1 Spoken questions (Stage 1 input)

| Source | Role | Share of Stage 1 speech |
|---|---|---|
| **Real Basque speech with transcripts**: Common Voice eu, Basque Parliament, Dialektoak, EusPodcast, faktoria | Accents and real acoustics; ASR and "answer the transcript" tasks | ≥ 30% |
| **Synthetic questions, multi-speaker**: OmniVoice plus re-synthesized CV voices, 50+ speakers | Volume and conversational content | ≤ 70% |
| **Augmentation** on ~50% of clips: noise (MUSAN), reverb (RIR), gain, telephone band (8 kHz), codec artefacts | Robustness to microphones and rooms | (applied on the fly) |

- **Hold out speakers:** dev and test speakers must not appear in training.
- **Real-speech test set:** build a test set of real recorded spoken questions, about 300 with 20+ speakers including dialects. This is the number that matters; synthetic test sets overestimate quality. **Still missing; the e2e suite can score it as soon as it exists.**
- **Why:** the playbook warns that models trained only on TTS questions fail on real accented users.

### 2.2 Text targets (both stages)

- **Generate the answers:** run the base thinker (vLLM) on the question transcript, with a system prompt asking for a *spoken* register: short (2–5 sentences), no markdown, lists, URLs or code, and numbers that can be spoken. Generate with temperature 0.7 and keep one sample. The 70B pipeline already does this; point it at the thinker.
- **Filter the answers:** drop refusals and empty or overlong answers (more than about 25 s of speech). Drop answers where the model switches language. Drop math and code; these were already dropped once (`dropped_math_ids.list`). Today's answers average 17–18 s in the e2e test, so tighten `--short-words`.
- **Multi-turn data:** generate the turns in order, so each answer sees the model's own earlier answers as history. Done in `generate_latxa_answers.py`.
- **Normalise before synthesis:** spell out numbers, dates, units and abbreviations (`basque_numbers.spell_numbers()`) so the text the talker sees matches what is spoken. Keep the unnormalised text for the Stage 1 target.

### 2.3 Stage 1 task mix

| Task | Share | Notes |
|---|---|---|
| Spoken question → answer (single and multi-turn) | 45% | The main task |
| **Text-only instruct replay** (Latxa SFT mix, Magpie-eu) | 25% | Done in v2 |
| ASR: "transcribe this" | 10% | On real speech; keeps acoustic grounding |
| Speech + separate text instruction (summarise, translate to es/en, answer in N words, extract) | 15% | Done in v2 |
| Speech translation eu → es/en | 5% | Done; BLEU 49/53 on real speech |
| *New:* context-dependent follow-ups, paralinguistics | — | v3 (+47k rows); emotion accuracy 0.12 → 0.71 |

Use **varied user prompts**, not a single fixed prompt: done (`multitask_instructions_expanded.json`).

### 2.4 Talker data (Stage 2)

- **Voice:** **one** target voice (now `maider`), optionally 2–3 extra voices selected by a speaker-id token added to the conditions.
- **Data:**
  - (a) Answer audio for the thinker's answers (section 2.2), one row per assistant turn. Done for the 70B answers.
  - (b) **TTS-only data:** about 500–2000 h of the same voice reading varied Basque text (news, Wikipedia, dialogue), with the hidden states taken from the thinker reading that text as an assistant turn. This teaches pronunciation of rare words, names and numbers. Not done.
- **Quality filters** (stricter than the current WER > 0.3 cut):

  | Filter | Threshold |
  |---|---|
  | ASR CER of the synthesized audio (HiTZ conformer, as in `e2e_score.py`) | ≤ 5% |
  | UTMOS | ≥ 3.5 |
  | Speaker similarity to the reference voice (TitaNet or WavLM SECS) | ≥ 0.85 |
  | **Speaker similarity, first vs last 4 s of the clip** | ≥ 0.85 (the e2e test shows the talker copies drift) |
  | Length | 1–30 s (≤ 2460 SNAC tokens) |
  | Speaking rate | inside the 5th–95th percentile, which catches stuck or truncated TTS |
  | Leading/trailing silence | trimmed to ≤ 0.3 s |

  `e2e_score.py` already has every one of these scorers (`asr`, `utmos`, `quality`); run it on the training audio too.

- **Codec check:** measure SNAC 24 kHz round-trip WER and UTMOS on Basque once and record it, so we know the ceiling the talker is chasing.

## 3. Models and freezing

| Component | Choice | Train when |
|---|---|---|
| Speech encoder | Basque fine-tuned whisper-large-v3 (eval WER 12.09) | Frozen; optionally unfreeze the top 8 layers at LR 1e-6 for the last 50% of Stage 1 |
| Projector | 5× frame concat + 2-layer MLP (current) | Stage 1a alone (LR 1e-3), then with the LLM |
| Thinker | **Latxa-Qwen3.5-4B** (stronger base: 0.635 vs 0.547 chat; better on every speech task). Latxa-Llama-3.1-8B kept as the second backbone | Stage 1b: LoRA r=64 at 2e-4 today. Frozen in Stage 2 |
| Talker | Llama-3.2-1B shape, initialised from Llama-3.2-1B, gated fusion of hidden state + token embedding | Stage 2 (LR 2e-4 then 5e-5); Stage 3 joint (LR 5e-5, LLM at 5e-6) |
| Codec | SNAC 24 kHz (7 tokens per 85 ms frame) | Frozen |
| Talker layout | `interleave`, R=3 text, W=28 audio | Measured audio/text ratio is 12–15, which keeps text ahead |

## 4. Hyperparameters

| Stage | Trainable | LR | Global batch | Length | Stop |
|---|---|---|---|---|---|
| 1a projector | projector | 1e-3, cosine, 3% warmup | 256 | ≤ 30 s audio | 1 epoch |
| 1b SFT | projector + LoRA (or full LLM at 1e-5) | 2e-4 LoRA, betas (0.9, 0.95), wd 0.01 | 64–128 conversations | 4096 tokens | Early stop on dev loss **and** a text-benchmark gate (LoRA overfits in epoch 2: 57411) |
| 2a talker pretraining | talker | 2e-4, cosine, 2% warmup | ~128 utterances | ≤ 2460 SNAC tokens | ~3 epochs over TTS + S2S |
| 2b talker SFT | talker | 5e-5 | 128 | same | Dev loss and ASR-CER of generated speech |
| 3 joint (optional) | LLM + talker | 5e-6 / 5e-5 | 64 | same | Stop at the first text-benchmark regression |
| 4 RL (optional) | talker (GRPO), thinker (DPO) | 1e-6 – 5e-6 | 4–8 samples per prompt | — | Composite reward; keep the SFT checkpoint if ASR or text regresses |

**Checkpoints:** Stage 2 now saves only the talker (`talker.safetensors`, ~2 GB). The DeepSpeed state is still saved next to it (a checkpoint is 18 GB); drop it for the last checkpoint of a finished run.

## 5. Evaluation (fixed before training, same numbers every stage)

| What | Metric | Data | Status |
|---|---|---|---|
| Text ability (forgetting) | EusProficiency, EusTrivia, EusReading, EusExams accuracy (`text_forgetting_eval.py`), chat and plain | Standard benchmarks | Done |
| Spoken QA | Latxa-70B judge (1–5), first turn and follow-ups (`e2e_score.py judge`) | v2 `test`, synthetic questions | Done; **real recorded questions missing** |
| Listening comprehension | EusListening accuracy, with and without audio | 882 questions | Run on old thinkers only; run on 57497/57592 |
| ASR (sanity) | WER | `test_asr` | Done |
| Speech output | ASR WER/CER of the generated audio, UTMOS and windowed UTMOS, NISQA, SECS within an answer and across turns | e2e test (5,485 answers) | Done |
| Latency | Time to first text and first audio, RTF | e2e test, 1 H100 | Done |
| Human | MOS on 50 dialogues, 5+ native listeners | At milestones only | Not done; the demo page (`demo_page/`) can host it |

**Gates:**
- Stage 1: text benchmarks within 1–2 points of base. **Currently failed by 14 points** (Qwen, chat).
- Stage 2: CER of the generated speech under 5% (**met**: 1.5%) and UTMOS at or above that of the TTS used for the data (**unknown**: score the `maider` training audio with the same scorer).
- New: SECS within an answer at or above 0.85 (**failed**: 0.73).

## 6. What I'd do next, given what exists today

1. **Close the measurement gaps** (cheap, no training):
   - Record the ~300 real spoken questions and run the e2e suite on them.
   - Score the `maider` training audio with `e2e_score.py utmos/quality`, so the talker has a ceiling to compare against.
   - Run EusListening and the e2e suite on the v3 thinker (57592) with talker 57634.
   - Try WiSE-FT merges of 57497 with the base Qwen (α 0.5–0.8). This recovered most of the text score for the Llama full-FT runs, and has not been tried on Qwen. Retrain or re-score the talker on the merged thinker, since the talker is thinker-specific.
2. ~~Regenerate the Stage 1 chat targets with Latxa-Qwen3.5-4B itself.~~ **Decided against (2026-11-10):** the Stage 1 answers stay the Latxa-70B ones. Forgetting is tackled with merges, the LoRA footprint and KL-to-base (7.1) instead.
3. **Fix voice drift in the talker** (section 7.2): drift filter on the training audio, then speaker conditioning or a SECS reward.
4. **Speed up the talker** (7.3) before making it any bigger.
5. Then add TTS-only talker data, joint fine-tuning, and RL, in that order.

## 7. New ideas

### 7.1 Thinker: less forgetting, better listening

- **Distillation instead of SFT targets (DiVA-style).** Run the frozen base thinker on the text transcript and train the speech model to match its *output distribution* (KL on the logits) for the same continuation, on ASR data alone. This needs no answer generation at all, cannot change the style, and DiVA reports better instruction following than SFT on synthetic chat. A KL term against the base model on the text-replay rows is a cheaper first step, and also targets the chat-template damage.
- **Smaller update footprint.** LoRA r=64 at 2e-4 still loses 14 points. Try lower rank (r=16), LoRA only on attention, or LoRA only on the lower layers (the speech-to-text mapping is mostly early), and compare at equal dev loss. Freezing the thinker completely and training only the projector plus a few prefix tokens is the extreme case; worth one run as a lower bound on forgetting.
- **Per-task adapters.** Keep the speech LoRA separate from the base and load it only when the input has audio; a text-only turn then runs the untouched base. This makes text forgetting zero by construction for text input, at the cost of mixing speech and text turns in one conversation.
- **Long audio for EusListening.** The old thinkers ignored the audio on EusListening (same accuracy without it). Its clips are longer than anything in training. Add long-audio comprehension data: chunk 1–5 min of podcast or Parliament audio, have the 70B model write questions and answers from the transcript, and train "listen, then answer". Check that the encoder chunking handles more than 30 s.
- **Encoder layers for paralinguistics.** The probe in PARALINGUISTICS.md shows middle Whisper layers keep more pitch. Feed a learned mix of a middle and the last layer to the projector; this should help the weak emotion classes (disgust 0.29, fear 0.44, happy 0.44).
- **Spoken instructions.** Kimi-Audio gives the task instruction as TTS audio half of the time. Cheap to add from the existing instruction pool, and it is how real users will talk to the model.

### 7.2 Talker: voice consistency and robustness

- **Make the talker thinker-agnostic.** Today a talker only works with the thinker it was trained on (WER 5.2 → 10.8 when the thinker changes), so every thinker experiment needs a new 24-hour Stage 2 run. Options, cheapest first:
  - Train the talker on hidden states from several thinker checkpoints (the base, LoRA runs, merges) mixed together.
  - Add hidden-state dropout and noise, and lean more on the token embedding (raise the gate's bias towards `Lin(e)`).
  - Condition on a lower thinker layer, which LoRA changes less than the last.
  - Ablation: token embeddings only. If CER stays under 2%, hidden states are only needed for prosody, and the talker becomes a reusable TTS head.
- **Speaker conditioning against drift.** Add a fixed speaker embedding (TitaNet or a learned voice token) to every condition, or a short reference SNAC prompt of the target voice before the first audio window. Both give the talker something to return to over a 20 s answer.
- **Shorter answers drift less.** Drift grows with length; enforcing the 2–5 sentence register (answers now 17–18 s) helps intelligibility, latency and drift at once.
- **Talker RL with a drift term.** In the GRPO reward of TRAINING_PLAN Phase 5, use the *within-answer* SECS (first vs last window), not only SECS to the reference voice.
- **Prosody from the question.** Pass the paralinguistic signal (the v3 emotion task) to the talker as a style token so it answers a sad user in a calmer voice. Needs a TTS that can render styles in the target voice; OmniVoice instruct prompts may be enough for 3–4 styles.

### 7.3 Speed and serving

- **The talker is the bottleneck** (RTF ~0.8 on an H100, with the text first token at ~100 ms). Before anything else: CUDA graphs / `torch.compile` for the talker decode loop and a static KV cache; these usually give 2–3× on a 1B decoder at batch 1.
- **Multi-token prediction.** Predict a whole SNAC frame (7 tokens) per step with 7 heads or a small depth transformer (Moshi/Mimi-style RQ-Transformer, or Orpheus-style delay). This cuts the talker's sequential steps by up to 7×.
- **Smaller talker.** Distil the 1B talker into a 300–500M one once quality is settled; check CER and SECS on the e2e suite.
- **Batching in the server.** With RTF 0.8 one GPU serves one user. Continuous batching of talker requests (or vLLM for the thinker plus a separate talker worker) is needed for a public demo.

### 7.4 Data and evaluation

- **Judge calibration.** The Latxa-70B judge scores the ASR transcript of the spoken answer. Check it against 100 human ratings once, and add a judge run on the *text* answer so content and speech errors can be separated.
- **Thinker-vs-cascade baseline in the e2e table.** Run Whisper-eu → Latxa-Qwen3.5-4B (text) → OmniVoice `maider` through the same e2e scorer. This is the system users will compare against; the end-to-end model should beat it on latency and match it on judge score.
- **Barge-in and turn-taking.** Before full duplex (which needs two-channel Basque dialogue), add a voice-activity-based interruption to the demo: stop the talker when the user speaks, keep the partial answer in the history. Cheap, and it makes the demo feel much more like a conversation.
- **Code-switching.** Basque users mix Spanish words and names. Add a small set of code-switched spoken questions to the real-speech test set, and Spanish named entities to the TTS-only talker data, so pronunciation of Spanish names is measured.
