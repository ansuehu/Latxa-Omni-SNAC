#!/usr/bin/env bash
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=translate-eu-en
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=.slurm/translate-eu-en.%A_%a.out.log
#SBATCH --error=.slurm/translate-eu-en.%A_%a.err.log

# Latxa-70B eu->en translation of prepare_st_en_sources.py output,.
#   sbatch dataset_generation/translate_eu_en.slurm
#   LIMIT=200 sbatch --array=0 dataset_generation/translate_eu_en.slurm   # pilot
#   TARGET=es sbatch dataset_generation/translate_eu_en.slurm             # Spanish (es.*.jsonl)
# The array tasks are the shards, two at a time (%2: 4 GPUs).
# Resubmitting resumes each shard.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .slurm
ROOT=${WORK_DIR:-/scratch/$USER}/st_en_latxa
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA=(--limit "$LIMIT")
${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python dataset_generation/translate_eu_en.py \
  --input "$ROOT/sources.jsonl" --output "$ROOT/${OUT_PREFIX:-${TARGET:-en}}.$SLURM_ARRAY_TASK_ID.jsonl" --target "${TARGET:-en}" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-4}}" "${EXTRA[@]}"
