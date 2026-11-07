#!/usr/bin/env bash
# Multi-task Stage 1 dataset (CPU only; reads most of the 87 GB composite corpus for the ASR sample).
# Resumable: ASR audio already written is skipped. Extra arguments go to build_multitask.py.
#   sbatch dataset_generation/build_multitask.slurm
#   CHAT_DATASET=$DS/latxa70b_answers_hf sbatch dataset_generation/build_multitask.slurm --asr-train 30000
# Wait for the shortened answers first:
#   sbatch --dependency=afterok:57288 dataset_generation/build_multitask.slurm
#SBATCH --job-name=build-multitask
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=.slurm/build-multitask-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH=$(pwd)
export HF_HUB_OFFLINE=1
DS=${WORK_DIR:-/scratch/$USER}/datasets
CHAT_DATASET="${CHAT_DATASET:-$DS/latxa70b_answers_short_hf}"
OUT="${OUT:-$DS/latxa_omni_multitask}"
test -f "$CHAT_DATASET/dataset_dict.json"

${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python dataset_generation/build_multitask.py \
  --chat-dataset "$CHAT_DATASET" --audio-dir "$OUT/audio" --output-dataset "${OUT}_hf" "$@"
