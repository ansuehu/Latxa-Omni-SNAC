#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-EusListening-cascade
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-eus-listening-cascade-%j.out.log
#SBATCH --error=.slurm/latxa-omni-eus-listening-cascade-%j.err.log

# Runs EusListening (deduplicated) as ASR (HiTZ/whisper-large-v3-eu) + Latxa 8B and 70B
# on the transcript. Two A100s for the 70B model. Transcripts are cached in saves/eus_listening/.
# Usage: sbatch omni_speech/evaluate/eus_listening_cascade.slurm
#   SMOKE=1 sbatch --gres=gpu:1 ...   (first 8 questions, 8B only)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON=${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python

EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  EXTRA_ARGS=(HiTZ/Latxa-Llama-3.1-8B-Instruct --limit 8)
fi
OUTPUT_DIR="saves/eus_listening/cascade_$SLURM_JOB_ID"
"$PYTHON" omni_speech/evaluate/eus_listening_cascade.py --output-dir "$OUTPUT_DIR" "${EXTRA_ARGS[@]}"
