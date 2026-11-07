#!/usr/bin/env bash
#SBATCH --job-name=latxa-omni-snac-compare
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/latxa-omni-snac-compare-%j.out.log
#SBATCH --error=.slurm/latxa-omni-snac-compare-%j.err.log

# Side-by-side Gradio demo, one GPU per model; --gres must match the number of models.
#   MODELS="unmerged=<ckpt> merged=<ckpt>" sbatch omni_speech/serve/run_gradio_compare.slurm
# SHARE=1 prints a public gradio.live link; otherwise tunnel to the node:
#   ssh -N -L 7860:<node>:7860 <login node>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export GRADIO_TEMP_DIR="${WORK_DIR:-/scratch/$USER}/gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

DEMO="${DEMO:-${WORK_DIR:-/scratch/$USER}/checkpoints/latxa_omni_snac_demo}"
MODELS="${MODELS:-unmerged=$DEMO/new-thinker-unmerged merged-0.7=$DEMO/new-thinker-merged07}"
PORT="${PORT:-7860}"
# Every question and answer (text + wav) is kept here, indexed in log.jsonl.
LOG_DIR="${LOG_DIR:-$DEMO/compare_logs/$SLURM_JOB_ID}"
ARGS=(--port "$PORT" --log-dir "$LOG_DIR")
for spec in $MODELS; do
    ARGS+=(--model "$spec")
done
echo "Compare demo on http://$(hostname):$PORT, logging to $LOG_DIR"
[[ "${SHARE:-0}" == 1 ]] && ARGS+=(--share)
${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python omni_speech/serve/gradio_compare.py "${ARGS[@]}"
