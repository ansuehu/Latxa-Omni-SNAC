#!/usr/bin/env bash
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=synthesize-followups
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=.slurm/synthesize-followups.%A_%a.out.log
#SBATCH --error=.slurm/synthesize-followups.%A_%a.err.log

# OmniVoice for the follow-up questions (synthesize_followups.py), one GPU per array task (4 tasks = 4 GPUs).
# Resubmitting resumes.
#   sbatch dataset_generation/followups/synthesize_followups.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .slurm
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
${CONDA_ENVS:-$HOME/.conda/envs}/omnivoice/bin/python dataset_generation/followups/synthesize_followups.py \
  --root ${WORK_DIR:-/scratch/$USER}/followups_v3 \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "${SLURM_ARRAY_TASK_COUNT:-4}"
