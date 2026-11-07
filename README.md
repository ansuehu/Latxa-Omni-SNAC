# Latxa-Omni-SNAC

**A Basque speech-to-speech assistant built on top of a text LLM.** You speak in Basque and Latxa-Omni-SNAC answers with text and with 24 kHz speech that streams while the answer is being written.

This is the successor of [Latxa-Omni](https://github.com/ansuehu/Latxa-Omni). It replaces Latxa-Omni's CTC HuBERT units and HiFi-GAN with an autoregressive [SNAC](https://github.com/hubertsiuzdak/snac) talker, and trains the thinker with LoRA to limit forgetting.

The same code supports two Basque thinkers (the LLM in the middle):

| Thinker | Size | Architecture | Stage 1 update | Environment |
|---|---|---|---|---|
| [HiTZ/Latxa-Llama-3.1-8B-Instruct](https://huggingface.co/HiTZ/Latxa-Llama-3.1-8B-Instruct) | 8B | Llama 3.1 (dense) | LoRA r=64 | `requirements-llama.txt` (transformers 4.45) |
| [HiTZ/Latxa-Qwen3.5-4B](https://huggingface.co/HiTZ/Latxa-Qwen3.5-4B) | 4B | Qwen3.5 (hybrid Gated DeltaNet + attention) | LoRA r=64 | `requirements-qwen.txt` (transformers 5) |

Both thinkers share the speech encoder, the projector, the SNAC talker, the data, the training scripts and the evaluation suite. The backbone is detected from the checkpoint's `config.json`, so you only pick the starting model and the environment.

## Architecture

```
speech ─► Whisper-large-v3 (Basque) ─► projector ─► thinker (Latxa-Llama-8B or Latxa-Qwen3.5-4B) ─► text
                                                          │ hidden states + answer tokens
                                                          ▼
                                       talker (Llama-3.2-1B shape, autoregressive)
                                                          │ flat SNAC tokens (7 per ~85 ms frame)
                                                          ▼
                                                 SNAC decoder ─► 24 kHz speech
```

- **Speech encoder:** Whisper-large-v3 fine-tuned on Basque. It stays frozen.
- **Projector:** downsamples the encoder frames and maps them into the thinker's embedding space.
- **Thinker:** a Basque LLM. It is trained with LoRA, so it keeps more of its text knowledge than with full fine-tuning (see [Text forgetting](#text-forgetting)).
- **Talker:** for each answer token it takes a gated fusion of the thinker's hidden state and the token embedding, in the style of LLaMA-Omni2. It writes [SNAC](https://github.com/hubertsiuzdak/snac) 24 kHz tokens in an interleaved layout: 3 text conditions, then 28 audio tokens (4 frames), repeated. This lets audio start before the text answer is finished.

## Installation

The two thinkers need different `transformers` versions, so use one environment per backbone.

**Latxa-Llama (8B):**

```bash
conda create -n latxa-omni-llama python=3.10 -y
conda activate latxa-omni-llama
pip install -r requirements-llama.txt
pip install flash-attn==2.7.3 --no-build-isolation   # optional, faster attention
```

**Latxa-Qwen3.5 (4B):**

```bash
conda create -n latxa-omni-qwen python=3.10 -y
conda activate latxa-omni-qwen
pip install -r requirements-qwen.txt
pip install causal-conv1d==1.6.2.post1   # optional: fast short-conv kernel (the prebuilt wheel needs glibc >= 2.32)
```

Qwen3.5's linear-attention kernels (`flash-linear-attention`, `tilelang`) only run on a GPU. On Hopper with Triton 3.4, fla's gated delta-rule backward gives wrong gradients ([fla #640](https://github.com/fla-org/flash-linear-attention/issues/640)), so the code uses tilelang for it instead.

In both environments, run everything from the repository root:

```bash
export PYTHONPATH=$(pwd)
mkdir -p .slurm   # SLURM logs go here
```

`ffmpeg` must be on `PATH` for the evaluation and demo scripts.

## Quick start

### 1. Prepare the thinker

The Llama thinker is used straight from the Hub. For Qwen3.5, first save the text-only model. This drops the vision tower and the MTP head, and sets `<|im_end|>` as EOS:

```bash
python omni_speech/model/extract_qwen3_5_text.py HiTZ/Latxa-Qwen3.5-4B models/Latxa-Qwen3.5-4B-text
```

### 2. Stage 1: speech understanding (thinker + projector)

The projector is trained in full and the thinker through LoRA. Checkpoints are saved with the adapters merged, so they load like a plain model.

```bash
# Latxa-Llama-3.1-8B
CONDA_ENV=latxa-omni-llama \
SPEECH_ENCODER_MODEL=<whisper checkpoint> DATASET=<stage 1 dataset> \
sbatch omni_speech/train/run_stage1_lora.slurm

# Latxa-Qwen3.5-4B
CONDA_ENV=latxa-omni-qwen MODEL_PATH=models/Latxa-Qwen3.5-4B-text \
SPEECH_ENCODER_MODEL=<whisper checkpoint> DATASET=<stage 1 dataset> \
STAGE1_ARGS="--no_gradient_checkpointing" \
sbatch omni_speech/train/run_stage1_lora.slurm
```

The defaults are LoRA r=64, alpha 128, dropout 0.05, learning rate 2e-4, 2 epochs and a global batch of 64 on 4 GPUs. Override them with `LORA_R`, `LORA_ALPHA`, `LEARNING_RATE`, `EPOCHS` and `TRAIN_BATCH_SIZE`. `run_stage1_latxa_omni_omnivoice.slurm` is the same run with full fine-tuning. `SMOKE=1` runs 2 steps.

The chat template (`llama_3` or `qwen3_5`, with thinking turned off) and the model class are chosen from the checkpoint. For Qwen3.5, turning off gradient checkpointing makes each step about 2× faster and fits in 80 GB with a micro-batch of 1.

### 3. SNAC targets for the spoken answers

```bash
ROOT=<dataset with answer audio> sbatch dataset_generation/dump_snac_tokens.slurm
```

This adds `answer_snac` (an `.npy` path) and `answer_snac_len` to each row. Merge the shards with `datasets.concatenate_datasets`.

### 4. Stage 2: speech generation (talker)

The thinker is frozen and a talker initialised from Llama-3.2-1B-Instruct is added on top of it (config in `configs/talker-snac.json`).

```bash
CONDA_ENV=<env> MODEL_PATH=<stage 1 run or checkpoint> DATASET=<dataset with SNAC tokens> \
sbatch omni_speech/train/run_stage2.slurm
```

If `MODEL_PATH` is a run folder, its last checkpoint is used. `JOINT=1` also unfreezes the thinker (Stage 3) and expects a Stage 2 checkpoint.

### 5. Inference and demo

```bash
python omni_speech/infer/infer.py --model-path <stage 2 ckpt> \
    --question-file questions.json --answer-file out/answers.jsonl   # add --conv-mode qwen3_5 for Qwen

MODEL_PATH=<stage 2 ckpt> sbatch omni_speech/serve/run_gradio_demo.slurm   # SHARE=1 for a public link
```

The Gradio demo streams the answer:

1. The thinker decodes token by token, and each hidden state goes straight to the talker (`stream_generate`).
2. Every 3 text tokens become 4 SNAC frames.
3. The frames are decoded with 2 frames of context and 2 of lookahead (`SnacStreamDecoder`) and sent to the browser as soon as they are ready.

Two options:
- `--transcribe-first`: the model transcribes each spoken turn before it answers. This avoids the multi-turn repetition described under [Known limitations](#known-limitations).
- `BENCH="a.wav b.wav"`: prints the time to first audio and the real-time factor instead of starting the server.

To hear what the codec alone does to Basque speech:

```bash
python -m omni_speech.model.speech_generator.snac_codec in.wav out.wav
```

## Data

Stage 1 is multi-task. Each row is a conversation whose user turns are audio:

- **Spoken chat:** Basque questions with answers from Latxa-Llama-3.1-70B-Instruct (`generate_latxa_answers.py`), spoken with OmniVoice TTS (`synthesize_answers.py`).
- **ASR:** Basque speech corpora (Common Voice and others) with a "transcribe this" instruction.
- **Speech translation:** eu→en and eu→es, with translations from Latxa-70B (`translate_eu_en.py`).
- **TTS-style rows** and text-only replay, to limit forgetting.
- **v3 additions:** context-dependent spoken follow-ups (`dataset_generation/followups/`) and paralinguistic tasks such as emotion and speaking rate (`dataset_generation/paralinguistics/`).

`dataset_generation/build_multitask.py` and `build_multitask_v3.py` build the final `DatasetDict`. The instruction pools are in `multitask_instructions*.json`. The TTS and ASR-WER helpers come from the [Latxa-Omni data pipeline](https://github.com/ansuehu/Latxa-Omni) (set `LATXA_OMNI_DIR` to its checkout).

The LLM generation scripts (`generate_latxa_answers.py`, `translate_eu_en.py`, `expand_instructions.py`, `followups/generate_followups.py`, `paralinguistics/generate_emotion_replies.py`) run Latxa-70B with [vLLM](https://github.com/vllm-project/vllm) in a separate environment.

See [docs/MULTITASK.md](docs/MULTITASK.md) and [docs/PARALINGUISTICS.md](docs/PARALINGUISTICS.md) for the design.

## Evaluation

| Script | What it measures |
|---|---|
| `evaluate/text_forgetting_eval.py`, `forgetting_sweep.slurm` | Basque text multiple-choice benchmarks (EusProficiency, EusTrivia, EusReading, EusExams) with plain or chat prompts. Optionally merges with the base weights first (`<ckpt>@<alpha>`, WiSE-FT). |
| `evaluate/stage1_dev_eval.py` | Per-task dev/test metrics: ASR WER, translation BLEU/chrF, chat loss |
| `evaluate/eus_listening_eval.py` | EusListening, spoken multiple-choice comprehension |
| `evaluate/eus_listening_cascade.py`, `eus_listening_qwen_omni.py` | EusListening baselines: an ASR→LLM cascade, and Qwen-Omni |
| `evaluate/calculate_utmos.py` | Predicted MOS of the generated speech |

### Text forgetting

How much of the thinker's Basque text knowledge survives Stage 1 (chat prompts, mean of the 4 benchmarks):

| Thinker | Mean |
|---|---|
| Latxa-Llama-3.1-8B-Instruct, base | 0.547 |
| + Stage 1 with full fine-tuning | 0.273–0.284 |
| + Stage 1 with LoRA r=64 | 0.402 |
| Latxa-Qwen3.5-4B, base | 0.635 |

LoRA keeps clearly more than full fine-tuning. Latxa-Qwen3.5-4B starts from a stronger base than the 8B Llama, on every one of the four benchmarks.

## Known limitations

- **Spoken multi-turn context.** With spoken follow-ups, both thinkers tend to repeat their previous answer. This happens because none of the current training follow-ups depend on earlier turns. Use `--transcribe-first` in the demo. The v3 follow-up data is meant to fix this.
- **Qwen3.5 path.** Training and loading are tested end to end. `infer.py` and some evaluation scripts still default to the `llama_3` template, so pass `--conv-mode qwen3_5`.
- **SLURM scripts** are written for a 4×H100 node. Add your account and partition with `sbatch -A <account> -p <partition>`, and set `WORK_DIR` (default `/scratch/$USER`) and `CONDA_ENVS` (default `~/.conda/envs`).

## Repository layout

```
omni_speech/
  model/              encoder, projector, thinker wrappers (Llama, Qwen3.5), SNAC talker
  train/              stage1.py (thinker), stage2.py (talker, --tune_llm for joint), SLURM launchers
  infer/  serve/      offline inference and the streaming Gradio demo
  evaluate/           forgetting, dev/test metrics, EusListening, UTMOS
dataset_generation/   answer generation, TTS, translation, multi-task builds, SNAC dumps
configs/              talker and LLM configs
docs/                 training plan, ideal setup, multi-task and paralinguistics design
tests/                unit tests
```

## Tests

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py'
```

Run the suite in both environments. The Qwen tests are skipped when transformers has no Qwen3.5, and the GPU tests are skipped without a GPU. `test_snac_talker.py` runs on CPU and does not download SNAC.

## Acknowledgements

Latxa-Omni-SNAC builds on [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) and [LLaMA-Omni2](https://github.com/ictnlp/LLaMA-Omni2), the [Latxa](https://huggingface.co/HiTZ) models by HiTZ, [SNAC](https://github.com/hubertsiuzdak/snac) and OmniVoice.

## License

Apache 2.0, the same as LLaMA-Omni. The model weights keep their own licenses (the Llama 3.1 and 3.2 community licenses, and Qwen's license).
