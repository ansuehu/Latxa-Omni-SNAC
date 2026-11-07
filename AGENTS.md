# Repository Guidelines

## Structure

`omni_speech/` contains the following:
- `model/`: the Whisper encoder, projector, the thinker wrappers (`language_model/omni_speech*_llama.py`, `omni_speech*_qwen3_5.py`) and the SNAC talker (`model/speech_generator/talker.py`, `snac_codec.py`).
- `train/`: `stage1.py` (thinker) and `stage2.py` (talker; `--tune_llm` for joint training).
- `infer/`, `serve/` and `evaluate/`.

`dataset_generation/` builds the multi-task data and encodes answer audio into flat SNAC tokens (`dump_snac_tokens.py`). `docs/TRAINING_PLAN.md` holds the design and the staged plan.

## Environment

Two environments, one per thinker: `requirements-llama.txt` (transformers 4.45) and `requirements-qwen.txt` (transformers 5). Changes to shared code must work in both. Run `export PYTHONPATH=$(pwd)` from the repo root. SLURM scripts pick the interpreter from `CONDA_ENV` under `CONDA_ENVS` (default `~/.conda/envs`).

## Key conventions

- Flat SNAC token = `slot * 4096 + code`, with 7 slots per frame in Orpheus order. Talker vocabulary is 28,672 audio tokens plus BOS and EOS.
- The talker conditions only on the **last assistant turn**, found with `last_answer_mask` over the speech-expanded labels.
- With the `interleave` layout, `talker_write` must be a multiple of 7, and `talker_write / talker_read` must stay below the real audio-to-text token ratio.
- Stage 2 detects a Stage 1 checkpoint by `speech_generator_type != "snac_ar"` and adds the talker from `configs/talker-snac.json`.
- The backbone (Llama or Qwen3.5) and its chat template are detected from the checkpoint config. Qwen classes are imported only when transformers supports them.

## Style and testing

Follow the surrounding code: four-space indentation and `snake_case`. Avoid unrelated reformatting. Add focused `tests/test_*.py` files and run them in both environments with `HF_HUB_OFFLINE=1 python -m unittest discover -s tests -p 'test_*.py'`. For model changes, also report a `SMOKE=1` run and the checkpoint you used. Keep checkpoints, audio, logs and W&B artifacts out of commits.
