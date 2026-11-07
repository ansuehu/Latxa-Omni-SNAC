#!/usr/bin/env bash
#SBATCH --job-name=Latxa-Omni-SNAC-Stage2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=4-00:00
#SBATCH --output=.slurm/latxa-omni-snac-stage2-%j.out.log
#SBATCH --error=.slurm/latxa-omni-snac-stage2-%j.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=1800
export WANDB_PROJECT=latxa-omni-snac
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Stage 1 checkpoint (thinker); the talker is added on top of it.
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a Stage 1 checkpoint}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  # A Stage 1 run directory: take its final checkpoint (the highest step), not the best-dev one:
  # dev loss jumps at each new epoch, but the end of training gives better results.
  MODEL_PATH="$(ls -1dv "$MODEL_PATH"/checkpoint-*/ | tail -1)"
  MODEL_PATH="${MODEL_PATH%/}"
  echo "Stage 1 final checkpoint: $MODEL_PATH"
fi
DATASET="${DATASET:?set DATASET to a dataset saved by dump_snac_tokens.py}"
# A Qwen3.5 Stage 1 checkpoint needs transformers 5: CONDA_ENV=omni-qwen (the chat template and model
# class are picked from its config).
CONDA_ENV="${CONDA_ENV:-omni-snac}"
# The Instruct 1B is the one in the offline HF cache (56979, 57251 and 57265 all used it).
TALKER_INIT="${TALKER_INIT:-meta-llama/Llama-3.2-1B-Instruct}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-3}"
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == 1 ]]; then
  GLOBAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GPUS_PER_NODE))
  EXTRA_ARGS=(--max_steps 2 --report_to none --dataloader_num_workers 0
    --eval_steps 1 --dev_size 8 --num_audio_samples 1 --save_steps 1000000)
fi
if [[ "${JOINT:-0}" == 1 ]]; then
  # Stage 3: unfreeze the LLM too; MODEL_PATH should then be a Stage 2 checkpoint.
  EXTRA_ARGS+=(--tune_llm --learning_rate 5e-5 --llm_learning_rate 5e-6)
fi
# Extra stage2.py flags, e.g. STAGE2_ARGS="--save_steps 1000 --num_audio_samples 2".
read -r -a USER_ARGS <<< "${STAGE2_ARGS:-}"
EXTRA_ARGS+=("${USER_ARGS[@]}")
OUTPUT_DIR="${WORK_DIR:-/scratch/$USER}/checkpoints/latxa_omni_snac_stage2/$SLURM_JOB_ID"
if [[ -n "${RESUME_FROM:-}" ]]; then
  OUTPUT_DIR="$(dirname "$RESUME_FROM")"
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
fi
"${CONDA_ENVS:-$HOME/.conda/envs}/$CONDA_ENV/bin/torchrun" \
  --nproc_per_node="$GPUS_PER_NODE" --rdzv_endpoint="localhost:$((29400 + SLURM_JOB_ID % 1000))" \
  omni_speech/train/stage2.py \
  --model-path "$MODEL_PATH" \
  --talker-init-from "$TALKER_INIT" \
  --train-file "$DATASET" \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$((GLOBAL_BATCH_SIZE / (TRAIN_BATCH_SIZE * GPUS_PER_NODE)))" \
  --num_train_epochs "$EPOCHS" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "latxa_omni_snac_stage2_${SLURM_JOB_ID}" \
  "${EXTRA_ARGS[@]}"
