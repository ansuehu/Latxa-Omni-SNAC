#!/usr/bin/env bash
#SBATCH --job-name=latxa-omni-snac-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=08:00:00
#SBATCH --output=.slurm/latxa-omni-snac-demo-%j.out.log
#SBATCH --error=.slurm/latxa-omni-snac-demo-%j.err.log

# Streaming Gradio demo on one H100.
#   MODEL_PATH=<stage2 ckpt> sbatch omni_speech/serve/run_gradio_demo.slurm [gradio_demo.py flags]
# A Stage 1 checkpoint gives a text-only demo; pre-multi-task models want --legacy-prompt.
# SHARE=1 prints a public gradio.live link; otherwise tunnel to the node:
#   ssh -N -L 7860:<node>:7860 <login node>
# BENCH="q1.wav q2.wav" prints latency / real-time factor instead of serving.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export GRADIO_TEMP_DIR="${WORK_DIR:-/scratch/$USER}/gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a Stage 2 checkpoint}"
PORT="${PORT:-7860}"
ARGS=(--model-path "$MODEL_PATH" --port "$PORT")
if [[ -n "${BENCH:-}" ]]; then
    read -ra FILES <<< "$BENCH"
    ARGS+=(--benchmark "${FILES[@]}")
else
    echo "Demo on http://$(hostname):$PORT"
    [[ "${SHARE:-0}" == 1 ]] && ARGS+=(--share)
fi
${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python omni_speech/serve/gradio_demo.py "${ARGS[@]}" "$@"
