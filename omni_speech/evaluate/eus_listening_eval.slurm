#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-EusListening
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=12:00:00
#SBATCH --output=.slurm/latxa-omni-eus-listening-%j.out.log
#SBATCH --error=.slurm/latxa-omni-eus-listening-%j.err.log

# EusListening on Stage 1 checkpoints, raw and merged toward base Latxa at each alpha.
# Usage: CHECKPOINTS="<ckpt> <ckpt>" ALPHAS="0.7" sbatch omni_speech/evaluate/eus_listening_eval.slurm
#   SMOKE=1 sbatch ...   (first 8 questions, raw checkpoints only)
#   Qwen3.5: CONDA_ENV=omni-qwen-cc BASE=<Latxa-Qwen3.5-4B-text dir> CHECKPOINTS=... sbatch ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON=${CONDA_ENVS:-$HOME/.conda/envs}/${CONDA_ENV:-omni-snac}/bin/python

ALPHAS="${ALPHAS:-0.7}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  ALPHAS=""
  EXTRA_ARGS=(--limit 8)
fi
SPECS=()
for checkpoint in $CHECKPOINTS; do
  SPECS+=("$checkpoint")
  for alpha in $ALPHAS; do
    SPECS+=("$checkpoint@$alpha")
  done
done
OUTPUT_DIR="saves/eus_listening/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"
echo "specs=${SPECS[*]}" | tee "$OUTPUT_DIR/run.txt"
"$PYTHON" omni_speech/evaluate/eus_listening_eval.py "${SPECS[@]}" --output-dir "$OUTPUT_DIR" \
  --base "${BASE:-HiTZ/Latxa-Llama-3.1-8B-Instruct}" "${EXTRA_ARGS[@]}"
