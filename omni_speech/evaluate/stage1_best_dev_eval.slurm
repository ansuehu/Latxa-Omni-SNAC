#!/usr/bin/env bash
# Score the final (or CHECKPOINT=best: best-dev) checkpoint of a Stage 1 run on its whole multi-task dev split:
# eval_loss per task, and WER / BLEU / chrF on up to N generated rows per task.
#   sbatch --dependency=afterany:<train job> omni_speech/evaluate/stage1_best_dev_eval.slurm <run_dir> [N]
#SBATCH --job-name=Latxa-Omni-Stage1-best-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=06:00:00
#SBATCH --output=.slurm/latxa-omni-stage1-best-dev-%j.out.log
#SBATCH --error=.slurm/latxa-omni-stage1-best-dev-%j.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
PYTHON=${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python

RUN_DIR="$1"
SAMPLES="${2:-300}"
DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa_omni_multitask_hf}"
# The final checkpoint (highest step) by default; CHECKPOINT=best takes the best-dev one,
# named by the newest trainer_state.json.
if [[ "${CHECKPOINT:-final}" == best ]]; then
  STATE="$(ls -1v "$RUN_DIR"/checkpoint-*/trainer_state.json | tail -1)"
  CKPT="$("$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'])" "$STATE")"
  OUT="$RUN_DIR/best_dev_eval"
else
  CKPT="$(ls -1dv "$RUN_DIR"/checkpoint-*/ | tail -1)"
  CKPT="${CKPT%/}"
  OUT="$RUN_DIR/final_dev_eval"
fi
echo "checkpoint: $CKPT"
"$PYTHON" omni_speech/evaluate/stage1_dev_eval.py "$CKPT" --dataset "$DATASET" --split dev \
  --num_generation_samples "$SAMPLES" --output-dir "$OUT"
