#!/usr/bin/env bash
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=emotion-replies
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=.slurm/emotion-replies.%j.out.log
#SBATCH --error=.slurm/emotion-replies.%j.err.log

# Latxa-70B emotion-aware replies (generate_emotion_replies.py), TP=2. Resubmitting resumes.
#   sbatch dataset_generation/paralinguistics/generate_emotion_replies.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .slurm
ROOT=${WORK_DIR:-/scratch/$USER}/paralinguistic_v3
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true VLLM_WORKER_MULTIPROC_METHOD=spawn
${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python dataset_generation/paralinguistics/generate_emotion_replies.py \
  --clips "$ROOT/clips.jsonl" --output "$ROOT/replies.jsonl"
