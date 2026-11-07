# Paralinguistics: hearing how something is said

Goal: Latxa-Omni should notice intonation, pitch, speed and emotion in the user's voice (e.g. that the user is
sad, angry or hesitant) and use it in the answer. Today it does not.

## Why the model cannot do it now

1. **Nothing in training asks for it.** Spoken questions are OmniVoice TTS of InstructS2S text in neutral voices,
   and every target (answer, transcript, translation) depends only on the words. The model never needs the
   voice, so it learns to ignore it.
2. **The encoder may have lost it.** The Whisper encoder (54377) is fine-tuned for Basque ASR, which rewards
   throwing away everything but the words. Its last layer (the only one the projector sees) likely keeps less
   emotion/pitch than the original Whisper; middle layers usually keep more. (The probe below shows this is
   true for pitch but hardly for emotion.)
3. **The transcribe-first demo mode** answers from the audio plus its transcript, so the audio is still there;
   it is not what blocks this.

## Ideas, cheapest first

1. **Emotion tag in the prompt (demo only, no training).** Run a speech emotion classifier (e.g. emotion2vec+,
   fairly language-independent) and add a line like "Ahots-tonua: haserre" next to the transcript. Works today,
   but only a few coarse labels, and the LLM follows the tag rather than listening.
2. **Probe the encoder first (one GPU, ~1 h).** Linear probes on frozen features, on emotional speech where the
   same sentence is read in every emotion (so the words give nothing away). Tells us whether training alone can
   fix it or the encoder must change. Done: see "Probe" below.
3. **Encoder changes (if the probe shows the information is lost).** Either feed the projector a learned weighted
   sum of Whisper layers instead of the last one, or add a small second encoder (emotion2vec / WavLM) next to
   Whisper and concatenate its features before the projector. ASR keeps using Whisper either way.
4. **Paralinguistic training data (the real fix).** New Stage 1 tasks whose answer depends on the voice, not the
   words:
   - emotion recognition: "Nola sentitzen da hiztuna?" with a Basque answer;
   - describing the voice: pitch, speed, energy, gender, hesitation, tiredness;
   - emotion-aware replies: neutral words said with emotion, and a target that reacts to the emotion.

   Key: the same text in several emotions, or the model learns the emotion from the words. Labelled emotional
   speech in Basque is scarce, but prosody transfers fairly well across languages, so English/Spanish corpora
   (CREMA-D, RAVDESS, ESD, MSP-Podcast, IEMOCAP) with Basque answers are a reasonable start. If OmniVoice or
   another TTS can control emotion in Basque, we can synthesise our own.
5. **Expressive output (later).** Once the model recognises emotion, condition the talker on it (e.g. a softer
   voice for a sad user).

Recommendation, after the probe: the encoder keeps emotion, so (4) on its own, with the projector trainable, is
the way forward; add a layer mix (3) only if pitch and voice description matter. (1) is still a same-day demo
addition.

## Probe

- Data: CREMA-D (`mteb/crema-d`): 7,442 clips, 91 actors,
  the same 12 English sentences in 6 emotions (anger, disgust, fear, happy, neutral, sad).
  Speaker-independent split: actors 1073-1091 are the test set.
- Features, mean-pooled over the clip's real frames:
  our Whisper encoder (54377) at layers 8/16/24/32; the original openai/whisper-large-v3 at the same layers;
  the Stage 1 projector outputs that the LLM sees (Qwen 57497-7000, Llama 57411-7000); baselines: log-mel
  mean+std (plain acoustics) and the sentence id (the words only, should be chance).
- Probes: 6-way emotion with logistic regression (accuracy and UAR, chance about 0.17); clip log-F0 with ridge
  regression (R^2).
- Note: the LLM gets all 300 projector tokens of the padded 30 s window, padding included; the probe pools only
  the real frames.

### Results (job 57583, 1,330 test clips from 19 unseen actors)

| Features | Emotion UAR | log-F0 R^2 |
|---|---|---|
| Sentence id (words only) | 0.167 (chance) | -0.24 |
| Log-mel mean+std (plain acoustics) | 0.408 | **0.678** |
| Original Whisper-large-v3, layer 8 / 16 / 24 / 32 | 0.626 / 0.709 / 0.744 / 0.745 | 0.543 / 0.475 / 0.465 / 0.426 |
| Our Basque-ASR Whisper, layer 8 / 16 / 24 / 32 | 0.633 / 0.708 / **0.748** / 0.730 | 0.556 / 0.480 / 0.427 / 0.346 |
| Projector output, Qwen 57497-7000 (what the LLM sees) | 0.608 | 0.188 |
| Projector output, Llama 57411-7000 | 0.635 | 0.167 |

- **Emotion survives the encoder.** Our ASR fine-tuning barely touched it: last layer 0.730 vs 0.745 for the
  original Whisper, and layer 24 is the best of all (0.748). For comparison, strong dedicated CREMA-D models
  are around 0.8.
- **The projector loses part of it, but not most.** What the LLM receives still gives 0.61-0.64, far above
  chance (0.17). The projector was only ever trained for words, so retraining it on voice-dependent tasks should
  recover some of the gap.
- **Pitch is weaker.** Absolute pitch fades with depth (layer 8: 0.56, last: 0.35 ours / 0.43 original) and
  the projector halves it again (0.17-0.19); plain log-mel stats keep it best (0.68). Here clip pitch is mostly
  speaker identity (e.g. gender), so this is about voice description more than intonation.

**Conclusion:** the information is there; the missing piece is training data (idea 4). An encoder change (idea
3) is not needed for emotion. A learned mix of Whisper layers (e.g. adding layer 8 or 16) is the cheap
option if pitch and voice description matter; a second encoder is not justified by these numbers.

Caveats: acted English emotions (CREMA-D), mean-pooled clips and linear probes only (a lower bound on what the
LLM could extract from the token sequence), one regularisation setting (C=0.1).
