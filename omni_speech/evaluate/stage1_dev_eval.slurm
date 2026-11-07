#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-Stage1-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=06:00:00
#SBATCH --output=.slurm/latxa-omni-stage1-dev-%j.out.log
#SBATCH --error=.slurm/latxa-omni-stage1-dev-%j.err.log

# Usage: sbatch omni_speech/evaluate/stage1_dev_eval.slurm <output_dir> <checkpoint>...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1

OUTPUT_DIR="$1"
shift
"${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python" omni_speech/evaluate/stage1_dev_eval.py \
  --output-dir "$OUTPUT_DIR" "$@"
