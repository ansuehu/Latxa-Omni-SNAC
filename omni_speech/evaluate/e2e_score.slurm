#!/usr/bin/env bash
#SBATCH --account=external
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Latxa-Omni-e2e-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-e2e-score-%j.out.log
#SBATCH --error=.slurm/latxa-omni-e2e-score-%j.err.log

# ASR WER/CER (NeMo conformer) + UTMOS of e2e_generate.py output, then the summary.
#   sbatch omni_speech/evaluate/e2e_score.slurm <out_dir>...
set -euo pipefail
cd /shared/home/andoni.sudupe/Latxa-Omni-SNAC
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 LATXA_OMNI_DIR=/shared/home/andoni.sudupe/Latxa-Omni
ENVS=/home/andoni.sudupe/.conda/envs
"$ENVS/nemo/bin/python" omni_speech/evaluate/e2e_score.py asr --out-dir "$@"
"$ENVS/omni-snac/bin/python" omni_speech/evaluate/e2e_score.py utmos --out-dir "$@"
"$ENVS/omni-snac/bin/python" omni_speech/evaluate/e2e_score.py summary --out-dir "$@"
