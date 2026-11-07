#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-EusListening-Qwen3Omni
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=128GB
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-eus-listening-qwen-%j.out.log
#SBATCH --error=.slurm/latxa-omni-eus-listening-qwen-%j.err.log

# EusListening (deduplicated) with Qwen3-Omni-30B-A3B-Instruct, as a reference for the test.
# Usage: sbatch omni_speech/evaluate/eus_listening_qwen_omni.slurm
#   SMOKE=1 sbatch ...   (8 questions on the longest recordings)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON=${CONDA_ENVS:-$HOME/.conda/envs}/omni-qwen/bin/python

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  EXTRA_ARGS=(--limit 8 --longest-first)
fi
OUTPUT_DIR="saves/eus_listening/qwen3omni_$SLURM_JOB_ID"
"$PYTHON" omni_speech/evaluate/eus_listening_qwen_omni.py --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
