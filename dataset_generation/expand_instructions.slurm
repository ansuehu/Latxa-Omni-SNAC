#!/usr/bin/env bash
#SBATCH --constraint=a100-sxm4
#SBATCH --job-name=expand-instructions
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=.slurm/expand-instructions.%j.out.log
#SBATCH --error=.slurm/expand-instructions.%j.err.log

# expand_instructions.py with Latxa-70B; 2 GPUs.
#   sbatch dataset_generation/expand_instructions.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p .slurm
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python dataset_generation/expand_instructions.py \
  --output ${WORK_DIR:-/scratch/$USER}/st_en_latxa/instructions_expanded.json "$@"
