#!/usr/bin/env bash
# Queue the whole answer-audio chain with SLURM dependencies:
#   1. OmniVoice TTS of the assistant turns (16 x 1 GPU)
#   2. ASR WER per shard, each task starting when its TTS shard is done
#   3. WER summary (high_wer_turns.tsv) + dataset with the new audio paths (CPU)
#   4. SNAC dump, one row per assistant turn, high-WER turns excluded (8 x 1 GPU)
#   5. merge of the SNAC shards (CPU)
# Step 0 (CPU), skipped when $OUT_DIR/input exists: `synthesize_answers.py --prepare`
# from DATASET, hard-linking the audio of unchanged turns from REUSE_DIR if set.
# WER_THRESHOLD (default 0.3) sets the cut; rerun step 3+ with another value to change it.
# Run it directly (it only calls sbatch), not inside `sbatch --wrap`, which uses sh:
#   DATASET=$DS/latxa70b_answers_short_hf OUT_DIR=$DS/latxa70b_answers_short_tts \
#     REUSE_DIR=$DS/latxa70b_answers_tts bash dataset_generation/run_answers_pipeline.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

OUT_DIR="${OUT_DIR:-${WORK_DIR:-/scratch/$USER}/datasets/latxa70b_answers_tts}"
DATASET="${DATASET:-${WORK_DIR:-/scratch/$USER}/datasets/latxa70b_answers_hf}"
REUSE_DIR="${REUSE_DIR:-}"
WER_THRESHOLD="${WER_THRESHOLD:-0.3}"
TXAT=${CONDA_ENVS:-$HOME/.conda/envs}/latxa-txat/bin/python
NEMO=${CONDA_ENVS:-$HOME/.conda/envs}/nemo/bin/python
SB="sbatch --parsable ${SBATCH_ARGS:-}"
export OUT_DIR

test -f "$DATASET/dataset_dict.json"
prep=""
if [[ ! -d "$OUT_DIR/input" ]]; then
  prep=$($SB --job-name=prepare-answers --cpus-per-task=4 --mem=64G --time=02:00:00 \
    --output=.slurm/prepare-answers-%j.log --wrap "set -e
$TXAT dataset_generation/synthesize_answers.py --prepare --dataset $DATASET --out-dir $OUT_DIR \
  ${REUSE_DIR:+--reuse-dir $REUSE_DIR}")
fi
tts=${TTS_JOB:-$($SB ${prep:+--dependency=afterok:$prep} dataset_generation/synthesize_answers.slurm)}
wer=$($SB --dependency=aftercorr:$tts dataset_generation/answers_wer.slurm)
build=$($SB --dependency=afterok:$wer --job-name=build-answers --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --output=.slurm/build-answers-%j.log --wrap "set -e
$NEMO ${LATXA_OMNI_DIR:-../Latxa-Omni}/dataset_generation/omnivoice/asr_wer.py --summarize \
  --output-dir $OUT_DIR/asr_wer --wer-threshold $WER_THRESHOLD
$TXAT dataset_generation/synthesize_answers.py --build --dataset $DATASET --out-dir $OUT_DIR")
dump=$(DATASET=$OUT_DIR/dataset SNAC_DIR=$OUT_DIR/snac OUTPUT_PREFIX=$OUT_DIR/dataset_snac_shard \
  $SB --dependency=afterok:$build dataset_generation/dump_snac_tokens.slurm \
  --per-turn --exclude-tsv $OUT_DIR/asr_wer/high_wer_turns.tsv)
merge=$($SB --dependency=afterok:$dump --job-name=merge-snac --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --output=.slurm/merge-snac-%j.log --wrap "$TXAT -c \"
from datasets import DatasetDict, concatenate_datasets, load_from_disk
parts = [load_from_disk('$OUT_DIR/dataset_snac_shard%d' % k) for k in range(8)]
ds = DatasetDict({s: concatenate_datasets([p[s] for p in parts]) for s in parts[0]})
print(ds); ds.save_to_disk('$OUT_DIR/dataset_snac')\"")
echo "prepare=${prep:-skipped} tts=$tts wer=$wer build=$build dump=$dump merge=$merge"
