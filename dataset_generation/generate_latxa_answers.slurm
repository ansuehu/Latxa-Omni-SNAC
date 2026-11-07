#!/usr/bin/env bash
# One Latxa-70B replica per array task, tensor parallel over its GPUs (4 in bf16,
# or 2 with --gres=gpu:2 --mem=192G ... --quantization fp8); the
# default 2-task array uses 8 (bf16) or 4 (fp8) GPUs. Resumable: resubmit to continue.
#   sbatch dataset_generation/generate_latxa_answers.slurm
# Re-answer cut-off turns ("2-3 sentences" prompt), same array size:
#   sbatch --dependency=afterok:<generate job> dataset_generation/generate_latxa_answers.slurm --retry
# Condense cut-off / markdown / >60-word answers (keeps the same array size):
#   sbatch --dependency=afterok:<retry job> dataset_generation/generate_latxa_answers.slurm --shorten
# Then merge:
#   python dataset_generation/generate_latxa_answers.py --build --dataset $DATASET \
#       --out-dir $OUT_DIR --output-dataset ${OUT_DIR}_hf
#SBATCH --job-name=latxa70b-answers
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=.slurm/latxa70b-answers-%A_%a.out.log
#SBATCH --error=.slurm/latxa70b-answers-%A_%a.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn

DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa_omni_stage1_omnivoice_cv_mt_hf}"
OUT_DIR="${OUT_DIR:-${WORK_DIR:-/scratch/$USER}/datasets/latxa70b_answers}"
MODEL="${MODEL:-${WORK_DIR:-/scratch/$USER}/models/Latxa-Llama-3.1-70B-Instruct}"
TP="${TP:-$SLURM_GPUS_ON_NODE}"

# Allows submitting while the model is still downloading: wait until every
# shard in the index exists and huggingface_hub has no partial files left.
until [[ -f "$MODEL/model.safetensors.index.json" ]] \
  && "${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python" -c "
import json, os, sys
m = sys.argv[1]
shards = set(json.load(open(os.path.join(m, 'model.safetensors.index.json')))['weight_map'].values())
sys.exit(not all(os.path.exists(os.path.join(m, s)) for s in shards))" "$MODEL" \
  && [[ -z "$(find "$MODEL/.cache" -name '*.incomplete' 2>/dev/null)" ]]; do
  echo "$(date) waiting for $MODEL to finish downloading"; sleep 60
done

# vLLM 0.11 lives in latxa-txat (CUDA 12.8 build that matches this cluster's driver).
"${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python" dataset_generation/generate_latxa_answers.py \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "$@"
