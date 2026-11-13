#!/usr/bin/env bash
#SBATCH --account=external
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Latxa-Omni-e2e-judge
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-e2e-judge-%j.out.log
#SBATCH --error=.slurm/latxa-omni-e2e-judge-%j.err.log

# Latxa-70B judge (vLLM, fp8, TP 2) on the ASR transcripts from e2e_score.slurm, then the summary.
#   sbatch omni_speech/evaluate/e2e_judge.slurm <out_dir>...
set -euo pipefail
cd /shared/home/andoni.sudupe/Latxa-Omni-SNAC
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
ENVS=/home/andoni.sudupe/.conda/envs
"$ENVS/latxa-txat/bin/python" omni_speech/evaluate/e2e_score.py judge --out-dir "$@" --tp "$SLURM_GPUS_ON_NODE"
"$ENVS/omni-snac/bin/python" omni_speech/evaluate/e2e_score.py summary --out-dir "$@"
