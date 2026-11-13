#!/usr/bin/env bash
#SBATCH --account=external
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Latxa-Omni-e2e-score
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-e2e-quality-%j.out.log
#SBATCH --error=.slurm/latxa-omni-e2e-quality-%j.err.log

# Length-robust speech quality of e2e_generate.py output: UTMOS on 10 s windows (omni-snac env),
# NISQA on 10 s windows and TitaNet speaker consistency (nemo env), then the summary.
#   sbatch omni_speech/evaluate/e2e_quality.slurm <out_dir>...
set -euo pipefail
cd /shared/home/andoni.sudupe/Latxa-Omni-SNAC
export PYTHONPATH="$PWD" LATXA_OMNI_DIR=/shared/home/andoni.sudupe/Latxa-Omni
ENVS=/home/andoni.sudupe/.conda/envs
HF_HUB_OFFLINE=1 "$ENVS/omni-snac/bin/python" omni_speech/evaluate/e2e_score.py utmos_win --out-dir "$@"
"$ENVS/nemo/bin/python" omni_speech/evaluate/e2e_score.py quality --out-dir "$@"
HF_HUB_OFFLINE=1 "$ENVS/omni-snac/bin/python" omni_speech/evaluate/e2e_score.py summary --out-dir "$@"
