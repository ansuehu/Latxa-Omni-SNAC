#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-forgetting
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=1-00:00
#SBATCH --output=.slurm/latxa-omni-forgetting-%j.out.log
#SBATCH --error=.slurm/latxa-omni-forgetting-%j.err.log

# Weight-merge sweep (FORGETTING.md steps 1-2): merge a Stage 1 checkpoint's LLM
# back toward base Latxa at each alpha (in memory, nothing written to disk), then
# measure text-only forgetting and speech dev loss. Alpha 1 is the checkpoint
# itself; base Latxa is text-evaluated as the reference.
# Usage: sbatch omni_speech/evaluate/forgetting_sweep.slurm
#   CHECKPOINT=<weights-only checkpoint> ALPHAS="0 0.5" sbatch ...
#   SMOKE=1 sbatch ...   (alpha 0.5 only, 20 examples per task, 8 dev examples)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON=${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python

BASE=HiTZ/Latxa-Llama-3.1-8B-Instruct
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to a Stage 1 checkpoint}"
DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa_omni_stage1_omnivoice_cv_mt_hf}"
ALPHAS="${ALPHAS:-0 0.3 0.5 0.7 0.85}"
TEXT_ARGS=()
DEV_ARGS=(--split dev --num_generation_samples 16)
if [[ "${SMOKE:-0}" == 1 ]]; then
  ALPHAS=0.5
  TEXT_ARGS=(--limit 20 --max_new_tokens 32)
  DEV_ARGS=(--dev_size 8 --num_generation_samples 2 --max_new_tokens 32)
fi
OUTPUT_DIR="saves/forgetting_sweep/$SLURM_JOB_ID"
mkdir -p "$OUTPUT_DIR"
echo "checkpoint=$CHECKPOINT alphas=$ALPHAS dataset=$DATASET" | tee "$OUTPUT_DIR/run.txt"

MERGED=()
for alpha in $ALPHAS; do
  MERGED+=("$CHECKPOINT@$alpha")
done
"$PYTHON" omni_speech/evaluate/text_forgetting_eval.py "$BASE" "$CHECKPOINT" "${MERGED[@]}" \
  --base "$BASE" --output-dir "$OUTPUT_DIR/text" "${TEXT_ARGS[@]}"
"$PYTHON" omni_speech/evaluate/stage1_dev_eval.py "$CHECKPOINT" "${MERGED[@]}" \
  --base "$BASE" --dataset "$DATASET" --output-dir "$OUTPUT_DIR/speech" "${DEV_ARGS[@]}"
