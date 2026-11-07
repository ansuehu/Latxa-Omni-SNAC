# Multi-task training and evaluation for Latxa-Omni-SNAC

This note has three parts:

1. How Kimi-Audio covers many audio tasks with a single model.
2. A proposal for doing the same in Latxa-Omni.
3. A proposal for a Basque SpeechLM benchmark to measure it.

Sources:

- Kimi-Audio paper: https://arxiv.org/abs/2504.18425
- Kimi-Audio code: https://github.com/MoonshotAI/Kimi-Audio, in particular `finetune_codes/`

Items marked *(unverified)* were not checked against the source.

## 1. How Kimi-Audio handles many tasks

### Representation

- **Input.** Each audio position at 12.5 Hz sums two signals:
  - continuous Whisper-large-v3 features, downsampled from 50 Hz by an adaptor;
  - the embedding of a discrete semantic token from a single codebook (GLM-4-Voice tokenizer).
- **LLM.** Initialised from Qwen2.5-7B. The bottom layers are shared and then split into two parallel heads:
  - a text head, initialised from the LLM;
  - an audio head that predicts semantic tokens, initialised at random.
- **Generating text and audio together.** Both streams are padded with blank tokens to the same length. The audio stream starts 6 blank tokens behind the text.
- **Speech output.** Semantic tokens go through flow matching to mel spectrograms, then a BigVGAN vocoder, in streaming chunks.

### Telling the model which task to do

- **No task tokens and no task IDs.** Every task is defined by a natural-language instruction.
- **Paraphrases.** An LLM wrote about 200 paraphrases of the ASR instruction and about 30 for every other task. One is sampled per training example.
- **Text or speech.** The instruction is randomly given as text or as TTS audio. This teaches the model to follow spoken instructions as well as written ones.

### Pre-training mixture

About 13M hours of audio plus text: 585B audio tokens and 585B text tokens, one epoch.

| Task | Weight |
|---|---|
| text only | 7 |
| audio only | 1 |
| audio → text (ASR-like) | 1 |
| text → audio (TTS-like) | 1 |
| audio → semantic interleaving | 1 |
| audio → text interleaving | 1 |
| audio → semantic + text interleaving | 2 |

The Whisper encoder stays frozen for the first ~20% of tokens and is then unfrozen. The learning rate follows a cosine schedule from 2e-5 to 2e-6.

### SFT mixture

About 300K hours. Each source is trained for 2–4 epochs, a number chosen by ablation; small sources get more epochs and ASR gets 2.

- **ASR:** about 55K h in-house, plus Emilia, Libriheavy, MLS, WenetSpeech and GigaSpeech.
- **Audio QA:** ClothoAQA, AVQA, CompA-R, plus 5.2K h in-house.
- **Audio captioning:** WavCaps, AudioCaps, Clotho.
- **Speech emotion:** ESD, IEMOCAP, MELD, RAVDESS. Only about 50 h in total.
- **Sound and scene classification:** VGGSound, FSD50K, CochlScene, TAU/TUT.
- **Speech conversation:**
  - user turns are written by an LLM and synthesised with Kimi-TTS in more than 125K voices;
  - assistant turns are spoken by one voice actor in more than 20 styles.
- **Speech-in, text-out chat:** text SFT sets turned into speech (OpenHermes, Tulu3, Infinity-Instruct, NuminaMath, …).

### Fine-tuning data format (`finetune_codes/`)

```json
{"task_type": "understanding",
 "conversation": [
   {"role": "user", "message_type": "text", "content": "<instruction>"},
   {"role": "user", "message_type": "audio", "content": "/path/a.wav"},
   {"role": "assistant", "message_type": "text", "content": "<transcript / answer>"}]}
```

- **`task_type`** only sets the output mode:
  - `"understanding"` means text output only;
  - any other value means text plus audio output.
- **Loss** is computed on assistant turns only. An assistant turn with audio also trains the audio head.
- **Adding a task.** Write the data in this format. A new message structure also needs a change to `tokenize_message` in `datasets.py`.
- **Steps:** extract the semantic codes, run `finetune_ds.sh` (DeepSpeed), export the checkpoint, then check it with `check_sft_infer`.

## 2. Proposed methodology: multi-task Latxa-Omni

### Where we start

- **Stage 1** trains the projector and the full LLM (Whisper frozen) on audio-only conversations.
  - Every user turn is built from the fixed `USER_PROMPT` plus the audio (`omni_speech/train/stage1.py`).
  - So the model only ever learns one task: answer the spoken question.
- **Stage 2** trains the SNAC talker on top of that, with the LLM frozen.

### Proposal

**1. One conversation format for all tasks.** Let a user turn optionally carry an instruction alongside its audio:

```
[system] + instruction (text, or TTS audio) + <speech> → text answer
```

- This is a small change to `build_prompt_and_labels`: use `turn.get("instruction", USER_PROMPT)` instead of the fixed prompt.
- The rest of the architecture stays the same.

**2. Instruction paraphrases.**

- Have Latxa-70B write 30–50 Basque instruction paraphrases per task, and about 200 for ASR.
- Sample one per example.
- Render 30–50% of them with TTS, in voices *not* used for the user audio, and prepend that audio to the user turn. The model then learns spoken instructions too.

**3. Tasks.** All of these are speech in, text out, so they fit Stage 1 without architecture changes:

| Task | Data | Target |
|---|---|---|
| Spoken QA / chat (current task) | instruct set with Latxa-70B answers | answer |
| ASR | Common Voice eu, Basque Parliament (eu), HiTZ ASR corpora | transcript |
| Speech translation eu→es/en | Basque Parliament (bilingual eu/es); TTS of the translated instruct data (its English originals give eu→en pairs) | translation |
| Spoken summarisation / listening comprehension | TTS of Basque paragraphs + questions | summary / answer |
| Speaker attributes | Common Voice eu gender/age metadata | label as a sentence |
| Emotion *(optional)* | emotion-controlled TTS, or ESD/IEMOCAP with Basque instructions | label |
| Sound classification *(optional)* | VGGSound / FSD50K, labels translated to Basque | label |

**4. Text-only replay.**

- Mix Basque text instruction data into Stage 1, for example 20–50% of the steps. Kimi uses 7:1 text to audio in pre-training.
- This is also the main defence against forgetting (see `../Latxa-Omni/FORGETTING.md`).

**5. Sample by task, not by size.**

- Use a fixed per-task sampling weight, or epochs per task as Kimi does: more for small sets, about 2 for ASR.
- Otherwise ASR, which has the most hours, takes over the mix.

**6. Stage 2 trains speech output on conversational tasks only.**

- ASR, translation and classification keep text-only answers. This is our version of Kimi's `task_type="understanding"`.
- The talker should only learn to speak chat answers, not transcripts or labels.

**7. Ablations, in order:**

1. instruction field + paraphrases with chat data only (should be neutral);
2. + ASR;
3. + translation;
4. + text replay;
5. + the rest.

Score each run on the benchmark in section 3 and on text-only Basque benchmarks (forgetting).

## 3. Proposed methodology: a Basque SpeechLM benchmark

### Existing benchmarks and what to borrow from each

| Benchmark | What to borrow |
|---|---|
| Kimi-Audio-Evalkit | inference and scoring as separate steps (`--skip-eval`, `--reeval`); a fixed WER normaliser; a gpt-4o-mini judge for QA |
| VoiceBench | spoken AlpacaEval / IFEval / MMSU / OpenBookQA / AdvBench; TTS for most of it plus some real recordings; judge for open answers, accuracy for multiple choice, rules for IFEval, refusal rate for AdvBench |
| URO-Bench | full speech-to-speech scoring: transcribe the spoken output with Whisper, then judge or rule-score it; also report UTMOS and the WER between the text output and the speech output (consistency) |
| VoxEval | MMLU-style questions in speech, with varied voices, speed, pitch, noise and room acoustics, to test robustness |
| AIR-Bench, MMAU | multiple-choice audio understanding scored by accuracy; open chat scored by a judge |
| SD-Eval | whether replies take emotion, accent, age or environment into account; an LLM judge agrees with humans better than BLEU/ROUGE |
| Dynamic-SUPERB, AudioBench | broad task coverage, with an LLM that maps free-form answers to labels *(unverified details)* |
| Big Bench Audio | BBH reasoning in TTS audio; 23 voices from 3 providers, filtered by ASR |

### Tasks and data

| Category | Source | Metric |
|---|---|---|
| ASR | Common Voice eu test, Basque Parliament test | WER/CER, Basque normalisation, numbers spelled out |
| Speech translation | Basque Parliament eu→es; TTS of FLORES-200 eu | BLEU / chrF / COMET |
| Knowledge (multiple choice) | spoken EusExams, EusTrivia, EusProficiency, BertaQA | accuracy (regex, then LLM fallback) |
| Listening comprehension | spoken EusReading; XNLI-eu | accuracy |
| Open chat | translated AlpacaEval + ~200 real user questions recorded by people | LLM judge with reference answer, 1–10 rubric |
| Instruction following | translated IFEval | rule-based |
| Safety | translated AdvBench | refusal rate |
| Multi-turn | held-out spoken conversations | LLM judge |
| Paralinguistics | Common Voice gender/age; small recorded emotion set | accuracy |
| Robustness | the same items with noise, speed changes, and Gipuzkoan/Bizkaian speakers | relative drop |

### Audio: TTS or human recordings

- **TTS for scale.** Use several voices and ideally a *different* TTS system from the one used for training (not OmniVoice, not the Maider/Antton voices).
- **A human subset.** Record 200–500 items with several speakers, dialects and real devices.
- **Report both.** TTS-only scores overstate quality, and the gap between TTS and human audio is a result in itself.

### Scoring speech output (Stage 2 models)

1. **Content.** Transcribe the generated speech with a strong Basque ASR (HiTZ whisper-large-v3-eu or the conformer), then score the transcript like a text answer.
2. **Consistency.** WER between the model's text output and the transcript of its speech.
3. **Quality.** UTMOS, which is trained on English, so only compare it between our own models. Also a small human MOS set.
4. **Latency.** Time to first audio.

### Pitfalls

1. **TTS leakage.** Benchmark audio made with the training TTS or voices inflates scores.
2. **ASR coupling.** ASR errors count as model errors. Report the scoring ASR's own WER on the reference audio.
3. **Judge bias.** Judges reward long, English-style answers and may handle Basque poorly.
   - Use reference-guided rubrics and penalise length.
   - Check the judge on about 100 items against native-speaker ratings.
4. **Contamination.** The Stage-1 data comes from translated public sets. Check n-gram and embedding overlap with AlpacaEval, IFEval and the Eus* sets, and keep a private held-out split.
5. **Translationese.** Translated benchmarks test English-language culture. Include native Basque content such as BertaQA's local subset and EusTrivia.
6. **Normalisation.** Spell out numbers and dates the same way in references and outputs (`../Latxa-Omni/dataset_generation/omnivoice/basque_numbers.py`).
7. **Prompt sensitivity.** Fix the instructions, and report the variance over 2–3 paraphrases.

### References

- Latxa and the Eus* benchmarks: https://arxiv.org/abs/2403.20266
- BertaQA: https://arxiv.org/abs/2406.07302
- VoiceBench: https://github.com/MatthewCYM/VoiceBench
- URO-Bench: https://arxiv.org/abs/2502.17810
- VoxEval: https://arxiv.org/abs/2501.04962
- Kimi-Audio-Evalkit: https://github.com/MoonshotAI/Kimi-Audio-Evalkit
- HiTZ Basque ASR: https://huggingface.co/HiTZ/whisper-large-v3-eu
