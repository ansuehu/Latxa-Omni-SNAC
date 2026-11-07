#!/usr/bin/env bash
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=generate-followups
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=.slurm/generate-followups.%A_%a.out.log
#SBATCH --error=.slurm/generate-followups.%A_%a.err.log

# Latxa-70B context-dependent follow-ups (generate_followups.py), one TP=2 shard per array task
# (2 tasks = 4 GPUs).
#   sbatch dataset_generation/followups/generate_followups.slurm
#   SPLIT=test LIMIT=50 sbatch --array=0 ...   # pilot
# Resubmitting resumes each shard.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .slurm
ROOT=${WORK_DIR:-/scratch/$USER}/followups_v3
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
SPLIT="${SPLIT:-train}"
EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA=(--limit "$LIMIT")
${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python dataset_generation/followups/generate_followups.py \
  --input "$ROOT/sources.$SPLIT.jsonl" --output "$ROOT/${OUT_PREFIX:-followups}.$SPLIT.$SLURM_ARRAY_TASK_ID.jsonl" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-2}}" "${EXTRA[@]}"
