#!/usr/bin/env bash
# OmniVoice (maider) for the assistant turns of the Latxa answers; one shard per task.
# Run `synthesize_answers.py --prepare` (latxa-txat python) first. Resumable.
# Full chain (TTS -> ASR WER -> build -> SNAC dump): dataset_generation/run_answers_pipeline.sh
#SBATCH --job-name=tts-answers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=.slurm/tts-answers-%A_%a.out.log
#SBATCH --error=.slurm/tts-answers-%A_%a.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

"${CONDA_ENVS:-$HOME/.conda/envs}/omnivoice/bin/python" dataset_generation/synthesize_answers.py \
  --out-dir "${OUT_DIR:-${WORK_DIR:-/scratch/$USER}/datasets/latxa70b_answers_tts}" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "$@"
