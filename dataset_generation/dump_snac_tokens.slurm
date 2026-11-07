#!/usr/bin/env bash
#SBATCH --job-name=dump-snac
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=1-00:00
#SBATCH --array=0-7
#SBATCH --output=.slurm/dump-snac-%A_%a.out.log
#SBATCH --error=.slurm/dump-snac-%A_%a.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="$PWD"

# Either ROOT (dataset in $ROOT/dataset, relative audio paths under $ROOT) or
# DATASET + SNAC_DIR + OUTPUT_PREFIX (absolute audio paths). Extra args are
# forwarded, e.g. --per-turn --exclude-tsv high_wer_turns.tsv.
ROOT="${ROOT:-}"
DATASET="${DATASET:-${ROOT:?set ROOT or DATASET}/dataset}"
"${CONDA_ENVS:-$HOME/.conda/envs}/omni-snac/bin/python" dataset_generation/dump_snac_tokens.py \
  --dataset "$DATASET" \
  --data-root "$ROOT" \
  --out-dir "${SNAC_DIR:-$ROOT/snac}" \
  --output-dataset "${OUTPUT_PREFIX:-$ROOT/dataset_snac_shard}$SLURM_ARRAY_TASK_ID" \
  --num-shards "$SLURM_ARRAY_TASK_COUNT" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "$@"
