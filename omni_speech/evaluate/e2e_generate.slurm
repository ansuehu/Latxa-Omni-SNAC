#!/usr/bin/env bash
#SBATCH --account=external
#SBATCH --partition=gpu-H100
#SBATCH --job-name=Latxa-Omni-e2e-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=96GB
#SBATCH --time=2-00:00
#SBATCH --array=0-3%4
#SBATCH --output=.slurm/latxa-omni-e2e-gen-%A_%a.out.log
#SBATCH --error=.slurm/latxa-omni-e2e-gen-%A_%a.err.log

# End-to-end test generation, one shard per array task (resumable: resubmit to continue).
#   MODEL_PATH=<stage2 ckpt or run dir> OUT_DIR=<dir> [CONDA_ENV=omni-qwen-cc] sbatch omni_speech/evaluate/e2e_generate.slurm
# PER_GPU=<n> (below). GEN_ARGS="--fast-talker" adds generation flags. A run dir (no config.json) means its latest checkpoint-*, picked when the task starts.
set -euo pipefail
cd /shared/home/andoni.sudupe/Latxa-Omni-SNAC
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
MODEL_PATH="${MODEL_PATH:?}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  MODEL_PATH=$(ls -d "$MODEL_PATH"/checkpoint-* | sort -t- -k2 -n | tail -1)
fi
echo "model: $MODEL_PATH  node: $(hostname)  $(date)"
mkdir -p "$OUT_DIR"
echo "$MODEL_PATH" > "$OUT_DIR/model_path.txt"
# PER_GPU=3 runs three shards side by side on this task's GPU (one uses ~13 GB and under half the GPU), so
# ~2x the throughput, but they slow each other down: latency and RTF of such a run are not comparable.
PER_GPU="${PER_GPU:-1}"
pids=()
for part in $(seq 0 $((PER_GPU - 1))); do
  "/home/andoni.sudupe/.conda/envs/${CONDA_ENV:-omni-snac}/bin/python" omni_speech/evaluate/e2e_generate.py \
    --model-path "$MODEL_PATH" --out-dir "$OUT_DIR" \
    --num-shards $((SLURM_ARRAY_TASK_COUNT * PER_GPU)) --shard $((SLURM_ARRAY_TASK_ID * PER_GPU + part)) \
    ${LIMIT:+--limit $LIMIT} ${GEN_ARGS:-} &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
