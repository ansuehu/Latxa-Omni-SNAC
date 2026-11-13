# Latxa-Omni-SNAC evaluation results

All results so far, with what each evaluation measures. Updated 2026-11-10. Earlier numbers come from
`Latxa-Omni_evals_2026-11-07.xlsx`; numbers from 2026-11-10 are new. Items marked *pending* are still running.

## Systems

| Name | What it is |
|---|---|
| **Latxa-Llama-3.1-8B / Latxa-Qwen3.5-4B (base)** | The text LLMs before any speech training. The reference for forgetting. |
| **Old Latxa-Omni** | The predecessor (IberSPEECH): Llama-8B full fine-tune, CTC HuBERT units + HiFi-GAN vocoder. |
| **57389** | Stage 1, Latxa-Llama-8B, full fine-tune, multi-task v2. |
| **57411** | Stage 1, Latxa-Llama-8B, LoRA r=64, multi-task v2 (checkpoint-7000). |
| **57497** | Stage 1, Latxa-Qwen3.5-4B, LoRA r=64, multi-task v2 (checkpoint-13000). Best thinker overall. |
| **57592 (v3)** | 57497's recipe plus spoken follow-ups and paralinguistic (emotion) data (checkpoint-14750). |
| **Talker 57393** | SNAC talker trained on the hidden states of 57389 (Llama full FT). |
| **Talker 57634** | SNAC talker trained on the hidden states of 57497 (Qwen). Also used with 57592, which it was not trained on. |
| **α 0.5 / 0.7 / 0.85** | WiSE-FT merge: the thinker's LLM weights set to α · fine-tuned + (1 − α) · base. Done in memory, no retraining. |

---

## 1. Text benchmarks (forgetting)

**What it measures.** How much Basque text knowledge the thinker keeps after Stage 1. Speech training tends to
overwrite what the LLM knew ("catastrophic forgetting"); a speech assistant that forgets facts gives worse answers.

**How.** Multiple-choice questions with text input only (`text_forgetting_eval.py`). The answer is the letter with
the highest probability. Suite v2 = EusProficiency (Basque language proficiency exam), EusTrivia (general
knowledge), EusReading (reading comprehension), EusExams (public-service exams). "Chat" wraps the question in the
chat template, as the model is used in practice; "plain" is a bare completion prompt. The score is the mean
accuracy of the four. **Gate:** within 1–2 points of the base model.

| Thinker | Chat | Plain |
|---|---|---|
| Latxa-Llama-3.1-8B base | 0.547 | 0.500 |
| 57389 (Llama full FT) | 0.273 | 0.314 |
| 57389 merged α 0.7 | 0.329 | 0.385 |
| 57411 (Llama LoRA) | 0.402 | 0.381 |
| 57411 merged α 0.7 | — | 0.439 |
| Latxa-Qwen3.5-4B base | **0.635** | 0.565 |
| 57497 (Qwen LoRA) | 0.494 | 0.519 |
| **57497 merged α 0.5** (new) | **0.620** | **0.581** |
| 57497 merged α 0.7 (new) | 0.590 | 0.557 |
| 57497 merged α 0.85 (new) | 0.564 | 0.536 |
| 57592 (Qwen LoRA v3) | 0.487 | 0.516 |

**Takeaways.** Full fine-tuning halves the Llama score; LoRA and the Qwen backbone lose much less. Merging 57497
at α 0.5 brings chat accuracy within 1.5 points of base, **the first thinker to pass the gate**.

**Speech-side cost of the merge (new).** The same 57497 merges on the speech tests of section 2, run on 500-item
subsets of each split, so the numbers differ a little from the full-split ones in section 2:

| 57497 | Spoken chat test loss | ASR WER | Speech → English BLEU / chrF | Speech → Spanish BLEU / chrF |
|---|---|---|---|---|
| unmerged | 0.541 | 4.39 | 49.4 / 68.5 | 55.8 / 73.7 |
| α 0.5 | 0.561 | 4.52 | 47.7 / 67.0 | 54.7 / 73.0 |
| α 0.7 | 0.535 | 4.37 | 49.8 / 68.7 | **56.6 / 74.0** |
| α 0.85 | **0.533** | **4.29** | **50.4 / 69.1** | 56.3 / 73.8 |

α 0.7 and 0.85 are free, or slightly better than unmerged, on every speech task. α 0.5 costs about 2 BLEU and
0.13 WER but is the only one that passes the text gate. **α 0.7 is the balanced choice**: speech is unchanged, text
recovers most of the way (chat 0.590 vs 0.494 unmerged and 0.635 base), and EusListening rises 0.559 → 0.629 (section 4).
Through the talker, the α 0.7 merge also holds up end to end (section 5).

## 2. Per-task speech tests (thinker, text output)

**What it measures.** Whether the thinker does each spoken task right, judged on its text output only (no talker).

**How.** The full held-out test splits of multi-task v2, greedy decoding (`stage1_dev_eval.py`):
- **Chat loss:** cross-entropy of the reference answers to spoken questions (lower = closer to the targets).
- **ASR WER:** word error rate when asked to transcribe real Basque speech (lower is better).
- **Speech translation BLEU / chrF:** spoken Basque → English or Spanish text, on real speech and on InstructS2S
  speech; higher is better. "Text" variants are the same task with written Basque input (FLORES devtest).
- **TTS WER:** asked to repeat a written text verbatim (a sanity check; 0 = perfect).

| Split | Metric | 57411 Llama LoRA | 57497 Qwen LoRA | 57592 Qwen v3 |
|---|---|---|---|---|
| Spoken chat | test loss | 0.585 | **0.541** | 0.549 |
| Multi-turn (templated) | test loss | 0.545 | **0.505** | 0.508 |
| ASR | WER | 7.89 | **5.89** | **5.89** |
| Speech → English, real speech | BLEU / chrF | 38.3 / 61.3 | **48.9 / 69.6** | 46.9 / 67.9 |
| Speech → Spanish, real speech | BLEU / chrF | 39.3 / 64.1 | **53.4 / 74.0** | 50.9 / 72.2 |
| Speech → English, InstructS2S | BLEU / chrF | 50.6 / 68.5 | **58.6 / 75.4** | 57.7 / 74.5 |
| Text → English, FLORES | BLEU / chrF | 30.5 / 58.2 | **31.4 / 59.6** | 31.0 / 59.3 |
| Text → Spanish, FLORES | BLEU / chrF | 18.7 / 47.4 | 20.3 / 49.0 | **20.3 / 49.1** |
| Repeat text (TTS) | WER | 0 | 0 | 0 |

**Takeaways.** The Qwen thinkers beat the Llama LoRA on every task. v3 is within noise of 57497 on these splits.

## 3. v3 tests: spoken follow-ups and emotion

**What it measures.** The two abilities v3 added: answering a spoken follow-up that depends on the previous turn,
and recognising emotion from the voice.

**How.** `test_followup`: 496 two-turn spoken chats, loss only. `test_para`: 4,981 emotional speech clips
(CREMA-D, ESD, RAVDESS); the answer counts as correct if it names the same emotion as the reference (7 classes,
chance ≈ 14%).

| Metric | 57497 (never saw this data) | 57592 v3 |
|---|---|---|
| Follow-up test loss | 0.620 | **0.542** |
| Emotion accuracy, all | 0.12 | **0.71** |
| angry / sad / neutral | 0.37 / 0.05 / 0.00 | **0.91 / 0.85 / 0.86** |
| surprise / fear / happy / disgust | 0.00 / 0.00 / 0.21 / 0.00 | **0.67 / 0.44 / 0.44 / 0.29** |

**Takeaways.** v3 learns both. Disgust, fear and happy stay weak.

## 4. EusListening (long-audio comprehension)

**What it measures.** Whether the model understands long recordings: listening-comprehension exam questions about
4–12 minute Basque recordings (HiTZ/EusListening), as multiple choice.

**How.** `eus_listening_eval.py`: the recording is cut into 30 s chunks, all given to the model, then the
question and options; the answer is the most likely letter. Every question is also asked **without the audio**: a
model scoring the same without it is answering from general knowledge, not listening. 882 questions after
de-duplication (the old Llama rows used the earlier 981-question set). Chance = 0.31. Cascades transcribe with
Whisper-eu and give the transcript to a text LLM.

| Model | Accuracy | Without audio | Gain from audio |
|---|---|---|---|
| Cascade: Whisper-eu → Latxa-Llama-70B | **0.781** | 0.557 | +0.22 |
| **57497 merged α 0.7** (new) | **0.629** | 0.492 | +0.14 |
| Cascade: Whisper-eu → Latxa-Llama-8B | 0.568 | 0.456 | +0.11 |
| 57592 v3 merged α 0.7 (new) | 0.563 | 0.503 | +0.06 |
| 57497 (new) | 0.559 | 0.438 | +0.12 |
| 57592 v3 (new) | 0.482 | 0.444 | +0.04 |
| Qwen3-Omni-30B-A3B (no Basque training) | 0.451 | 0.464 | −0.01 |
| Old Llama thinkers 57347 / 57389 (981 q.) | 0.33 / 0.32 | 0.32 / 0.32 | 0 |

**Takeaways.** The Qwen thinkers really listen (the old Llama ones were at chance). Merged 57497 beats the 8B text
cascade. v3 lost much of the long-audio gain (+0.04 vs +0.12). Missing baseline: a cascade with base Latxa-Qwen3.5-4B.

## 5. End-to-end spoken conversations

**What it measures.** The whole system as a user meets it: spoken question in, spoken answer out, through the
streaming demo path (`e2e_generate.py`, scored by `e2e_score.py`). 2,720 spoken test chats, 5,485 answers
(first turns and follow-ups; follow-ups see the model's own earlier answers).

**How each score is computed:**
- **Judge (1–5):** Latxa-Llama-70B rates each answer for helpfulness, relevance, fluency, factual accuracy and
  suitability for speech, on the ASR transcript of the spoken answer, with a strict rubric (IberSPEECH protocol).
- **WER / CER:** the answer audio is transcribed (HiTZ Basque conformer) and compared with the model's own text
  answer. Measures whether the speech says what the text says (intelligibility), not whether the answer is right.
- **UTMOS:** a neural predictor of mean opinion score (naturalness, 1–5) on the whole answer; **UTMOS (10 s)**
  averages it over 10 s windows, which is fairer to long answers.
- **NISQA:** another quality predictor (overall, 1–5), also on 10 s windows.
- **SECS within answer:** speaker similarity (TitaNet) between the first and last 4 s of an answer: does the voice
  stay the same? **SECS across turns:** each follow-up answer vs the chat's first answer.
- **First audio:** time from the end of the user's speech to the first audio chunk, encoder included. **RTF:**
  generation time / audio duration (below 1 = faster than real time).

| System | Judge (first / follow-up) | WER / CER | UTMOS / UTMOS 10 s | NISQA | SECS within / across | First audio | RTF |
|---|---|---|---|---|---|---|---|
| Old Latxa-Omni | 3.59 (4.14 / 3.05) | 7.72 / 2.58 | 2.27 / 2.64 | 4.37 | **0.86 / 0.97** | not streamed | 0.09 |
| Llama LoRA 57411 + talker 57393 | 3.68 (3.80 / 3.55) | 10.82 / 4.92 | 3.08 / 3.30 | 4.76 | 0.73 / 0.90 | 329 ms | 0.80 |
| Llama full FT 57389 + talker 57393 (300 chats) | — | 5.18 / 1.91 | 3.00 / — | — | — | 328 ms | 0.81 |
| **Qwen LoRA 57497 + talker 57634** | 4.37 (4.43 / 4.30) | **4.35 / 1.47** | 3.06 / 3.30 | **4.80** | 0.73 / 0.91 | 344 ms | 0.76 |
| Qwen v3 57592 + talker 57634 (new) | 4.35 (4.43 / 4.27) | **4.30 / 1.45** | 3.07 / 3.30 | **4.80** | 0.73 / 0.91 | not comparable* | not comparable* |
| **Qwen 57497 merged α 0.7 + talker 57634** (new) | **4.40 (4.51 / 4.29)** | 4.83 / 1.92 | 3.01 / 3.29 | **4.80** | 0.73 / 0.92 | not comparable* | not comparable* |

\* The v3 and merged runs shared each GPU between three processes to finish faster, so their timings are inflated.

The α 0.7 merge of 57497 gets the best judge score so far, 4.40 against 4.37 unmerged, all of it on first turns
(4.51 vs 4.43); follow-ups are unchanged. The talker was trained on the unmerged thinker, yet speech quality is the
same; WER rises from 4.35 to 4.83, and the answers are 2.5 s longer on average (20.9 s vs 18.3 s).

v3 ties 57497 on everything the e2e eval measures: judge 4.35 vs 4.37, a gap well within noise. The extra
v3 training on spoken follow-ups does not show up as better follow-up answers here (4.27 vs 4.30).

**Takeaways.** The Qwen system wins on content (+0.8 judge over the old system, +1.25 on follow-ups),
intelligibility and quality, and starts speaking in about a third of a second. The old system keeps the voice more
consistent (see section 6 for why). A talker only works well with the thinker it was trained on: talker 57393 has
twice the WER with the LoRA thinker as with its own.

## 6. Ceiling of the training audio (new)

**What it measures.** How good the talker could possibly be. The talker learns from `maider` TTS recordings,
encoded with the SNAC codec; it cannot beat that data, and the codec loses something on top.

**How.** The 5,406 answer recordings of the talker's test split, scored with exactly the e2e scorers of section 5:
once as recorded, once after a SNAC encode/decode round trip.

| Audio | WER / CER | UTMOS / UTMOS 10 s | NISQA | SECS within answer |
|---|---|---|---|---|
| `maider` TTS, as recorded | 3.38 / 0.72 | 3.25 / 3.45 | 4.83 | 0.80 |
| After a SNAC round trip | 3.74 / 0.81 | 2.78 / 3.06 | 4.78 | 0.71 |
| Talker output (Qwen 57497 + 57634, section 5) | 4.35 / 1.47 | 3.06 / 3.30 | 4.80 | 0.73 |

(The talker row is on different texts, the e2e test answers, so the comparison is approximate.)

**Takeaways.** The codec, not the talker, is the main cost to voice quality and consistency: a SNAC round trip
alone lowers UTMOS 3.25 → 2.78 and speaker consistency 0.80 → 0.71, and the talker is already at or above that
round-trip level. The training audio itself drifts somewhat (0.80), but the old system's 0.86 cannot be reached
with this codec. Intelligibility loss is mostly the talker's (CER 0.8 → 1.5).

## 7. Talker speed (new)

**What it measures.** Whether the experimental `--fast-talker` decoding path (preallocated KV cache + CUDA graph
per token) speeds the talker up, and where its time goes.

**How.** (a) 100 e2e chats per backbone with `--fast-talker`, compared with the earlier runs on the same answers;
(b) a controlled benchmark of the talker alone on one H100, both paths alternating on the same inputs;
(c) per-component timing of one token.

| | Current path | Fast path |
|---|---|---|
| Qwen e2e, 194 answers: WER / CER / UTMOS | 3.57 / 1.08 / 3.035 | 3.69 / 1.12 / 3.038 |
| Llama e2e, 194 answers: WER / CER / UTMOS | 3.95 / 1.17 / 3.010 | 3.88 / 1.22 / 3.027 |
| Talker alone, Qwen env (torch 2.8) | 145 tok/s (RTF 0.57) | 140 tok/s |
| Talker alone, Llama env (torch 2.1) | 136 tok/s (RTF 0.60) | 133 tok/s |
| One decoder step | 6.2 ms | 9.1 ms |
| Sampling, one token | 0.3 ms | 0.3 ms |

**Takeaways.** The fast path keeps quality but gives no speedup, so it stays off. The talker's ~2 GB of weights
can be read in ~0.7 ms on this H100 (measured), yet a step takes 6.2 ms: most of the time is outside the big matrix
products, so a kernel-level profile is the next step. Real-time playback needs ~82 tokens/s.

## 8. Multi-turn probe (original Latxa-Omni, 56132)

**What it measures.** Whether the model handles a second spoken turn correctly: a follow-up question, a topic
switch (in another voice, in demo audio, or in the same voice), or recalling something from the first turn.

| Model | Follow-up | Switch (other voice) | Switch (demo audio) | Switch (same voice) | Recall |
|---|---|---|---|---|---|
| 56132 checkpoint-3250 | 96/100 | 98/100 | 61/100 | 67/67 | 99/100 |
| merged α 0.3 | 78/100 | 55/100 | 14/100 | 36/67 | 0/100 |
| merged α 0.7 | 91/100 | 98/100 | 51/100 | 64/67 | 93/100 |

(The column headings follow the spreadsheet. This probe has not been run on the SNAC-era models.)

## Still pending

- Not yet run: a real recorded test set, human MOS ratings, a cascade baseline with base Latxa-Qwen3.5-4B, a clean
  latency measurement for v3.
