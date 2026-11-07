#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-Stage1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=4-00:00
#SBATCH --output=.slurm/latxa-omni-stage1-%j.out.log
#SBATCH --error=.slurm/latxa-omni-stage1-%j.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export WANDB_PROJECT=latxa-omni

MODEL_PATH="${MODEL_PATH:-HiTZ/Latxa-Llama-3.1-8B-Instruct}"
# The Qwen3.5 thinker runs in omni-qwen (transformers 5): CONDA_ENV=omni-qwen
# MODEL_PATH=${WORK_DIR:-/scratch/$USER}/models/Latxa-Qwen3.5-4B-text (see extract_qwen3_5_text.py).
CONDA_ENV="${CONDA_ENV:-omni-snac}"
# Whisper large-v3 fine-tuned on Basque: an openai-whisper .pt file or a Hugging Face model folder.
SPEECH_ENCODER_MODEL="${SPEECH_ENCODER_MODEL:?set SPEECH_ENCODER_MODEL to a Whisper checkpoint}"
if [[ -f "$SPEECH_ENCODER_MODEL" ]]; then
  SPEECH_ENCODER_TYPE=whisper
else
  test -f "$SPEECH_ENCODER_MODEL/model.safetensors"
  SPEECH_ENCODER_TYPE=whisper_hf
  export HF_HUB_OFFLINE=1
fi
DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa_omni_stage1_omnivoice_cv_hf}"
test -f "$DATASET/dataset_dict.json"

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
# 4 epochs overfit (dev loss rises from epoch 2). Early stopping is off by default: dev loss jumps at
# every new epoch and then falls again (57347: 0.620 -> 0.640 -> 0.626 in 4 evals, stopped by patience 4),
# so 2 epochs run to the end of the cosine schedule. The best-dev checkpoint is still saved.
EPOCHS="${EPOCHS:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
GLOBAL_BATCH_SIZE=64
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  GLOBAL_BATCH_SIZE="$GPUS_PER_NODE"
  EXTRA_ARGS=(--max_steps 2 --save_strategy no --report_to none --dataloader_num_workers 0
    --eval_steps 1 --dev_size 8 --num_generation_samples 4 --max_new_tokens 16)
fi
# Extra stage1.py flags, e.g. STAGE1_ARGS="--save_total_limit 3".
read -r -a USER_ARGS <<< "${STAGE1_ARGS:-}"
EXTRA_ARGS+=("${USER_ARGS[@]}")
# Resume: RESUME_FROM=<output_dir>/checkpoint-N keeps writing into that run's folder.
OUTPUT_DIR="${WORK_DIR:-/scratch/$USER}/checkpoints/latxa_omni_instructs2s_stage1/$SLURM_JOB_ID"
if [[ -n "${RESUME_FROM:-}" ]]; then
  test -d "$RESUME_FROM/$(cat "$RESUME_FROM/latest")"
  OUTPUT_DIR="$(dirname "$RESUME_FROM")"
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi
"${CONDA_ENVS:-$HOME/.conda/envs}/$CONDA_ENV/bin/torchrun" \
  --nproc_per_node="$GPUS_PER_NODE" --rdzv_endpoint="localhost:$((29400 + SLURM_JOB_ID % 1000))" \
  omni_speech/train/stage1.py \
  --model-path "$MODEL_PATH" \
  --speech-encoder "$SPEECH_ENCODER_MODEL" \
  --speech-encoder-type "$SPEECH_ENCODER_TYPE" \
  --train-file "$DATASET" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$((GLOBAL_BATCH_SIZE / (TRAIN_BATCH_SIZE * GPUS_PER_NODE)))" \
  --max_sequence_length 4096 \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$EPOCHS" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "${RUN_NAME:-latxa_omni_stage1_omnivoice_${EPOCHS}ep}_${SLURM_JOB_ID}" \
  "${EXTRA_ARGS[@]}"
