#!/usr/bin/env bash
# Basque NeMo ASR + WER of the synthesized answers (Latxa-Omni's asr_wer.py), one
# shard per task, results in $OUT_DIR/asr_wer/. Resumable.
#SBATCH --job-name=wer-answers
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=.slurm/wer-answers-%A_%a.out.log
#SBATCH --error=.slurm/wer-answers-%A_%a.err.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUT_DIR="${OUT_DIR:-${WORK_DIR:-/scratch/$USER}/datasets/latxa70b_answers_tts}"
shard=$(printf 'shard%02d-of-%02d' "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_COUNT")
"${CONDA_ENVS:-$HOME/.conda/envs}/nemo/bin/python" \
  ${LATXA_OMNI_DIR:-../Latxa-Omni}/dataset_generation/omnivoice/asr_wer.py \
  --metadata-file "$OUT_DIR/metadata/$shard.speech.jsonl" \
  --output-dir "$OUT_DIR/asr_wer" \
  "$@"
