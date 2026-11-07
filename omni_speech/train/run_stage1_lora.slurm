#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-Stage1-LoRA
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=4-00:00
#SBATCH --output=.slurm/latxa-omni-stage1-%j.out.log
#SBATCH --error=.slurm/latxa-omni-stage1-%j.err.log

# Stage 1 with LoRA on the LLM instead of full fine-tuning, to keep Latxa's text ability.
# Same data mix and schedule as the full-FT run (run_stage1_latxa_omni_omnivoice.slurm, which this
# calls); only the LLM update and the LR change. The projector stays fully trainable, Whisper frozen.
# Checkpoints are saved with the adapters merged, so they load like a full-FT checkpoint.
# Usage: sbatch omni_speech/train/run_stage1_lora.slurm
#   LORA_R=128 LORA_ALPHA=256 LEARNING_RATE=1e-4 sbatch ...;  SMOKE=1 sbatch --time=01:00:00 ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa_omni_multitask_v2_hf}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export STAGE1_ARGS="--lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT ${STAGE1_ARGS:-}"
export RUN_NAME="${RUN_NAME:-latxa_omni_stage1_lora_r${LORA_R}_lr${LEARNING_RATE}}"
exec bash omni_speech/train/run_stage1_latxa_omni_omnivoice.slurm
