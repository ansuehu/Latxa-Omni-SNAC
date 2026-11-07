"""Encode answer audio with SNAC 24 kHz and store flat talker tokens as int16 .npy files.

Adds ``answer_snac`` (absolute .npy path) and ``answer_snac_len`` (token count) to
every row, drops rows that fail, and saves the dataset. Rows are either flat
(``--audio-column``, default ``answer_audio``) or multi-turn ``turns``; for the
latter the last assistant turn is encoded, taking its ``audio`` or, when empty,
the OmniVoice sibling of the last user clip (``...-N-user.flac`` -> ``...-N-assistant.flac``).
With --per-turn, a conversation becomes one row per assistant turn (earlier turns
kept as context, id suffixed with ``-N``), so every spoken answer is used.
--exclude-tsv drops answers whose audio path is in the ``audio`` column of a TSV,
e.g. the ASR high-WER list of the OmniVoice synthesis.
Shard with --num-shards/--shard-index (e.g. a SLURM array), then merge the
saved shards with datasets.concatenate_datasets.

    python dataset_generation/dump_snac_tokens.py \
        --dataset /scratch/.../VoiceAssistant-400K_eu/dataset \
        --data-root /scratch/.../VoiceAssistant-400K_eu \
        --out-dir /scratch/.../VoiceAssistant-400K_eu/snac \
        --output-dataset /scratch/.../VoiceAssistant-400K_eu/dataset_snac
"""

import argparse
import csv
import logging
import os

import numpy as np
import soundfile as sf
from datasets import DatasetDict, load_from_disk

from omni_speech.model.speech_generator.snac_codec import SnacCodec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    codec = SnacCodec()
    ds = load_from_disk(args.dataset)

    excluded = set()
    if args.exclude_tsv:
        with open(args.exclude_tsv, encoding="utf-8") as stream:
            excluded = {row["audio"] for row in csv.DictReader(stream, delimiter="\t")}
        logger.info(f"excluding {len(excluded)} audio files listed in {args.exclude_tsv}")

    def turn_audio(turns):
        if turns[-1]["audio"]:
            return turns[-1]["audio"]
        user_audio = turns[-2]["audio"]
        if "-user." not in user_audio:
            raise ValueError(f"cannot derive assistant audio from {user_audio!r}")
        return user_audio.replace("-user.", "-assistant.")

    def answer_audio(example):
        if "turns" not in example:
            return example[args.audio_column]
        return turn_audio(example["turns"])

    def keep(turns):
        try:
            return turn_audio(turns) not in excluded
        except ValueError:
            return False

    def expand(batch):
        out = {key: [] for key in batch}
        for i, turns in enumerate(batch["turns"]):
            for j in range(1, len(turns), 2):
                prefix = turns[:j + 1]
                if prefix[-1]["role"] != "assistant" or not keep(prefix):
                    continue
                for key in batch:
                    out[key].append(batch[key][i])
                out["id"][-1] = f"{batch['id'][i]}-{(j + 1) // 2}"
                out["turns"][-1] = prefix
        return out

    def encode(example, idx, name):
        try:
            rel = os.path.join(name, f"{idx // 10000 * 10000:08d}", f"{idx:08d}.npy")
            try:
                # Rerunning an interrupted dump reuses the files already written.
                tokens = np.load(os.path.join(args.out_dir, rel))
            except (OSError, ValueError):
                audio, rate = sf.read(os.path.join(args.data_root, answer_audio(example)),
                                      dtype="float32", always_2d=True)
                tokens = codec.encode(audio.T, rate)
                os.makedirs(os.path.join(args.out_dir, os.path.dirname(rel)), exist_ok=True)
                np.save(os.path.join(args.out_dir, rel), tokens)
            example["answer_snac"] = os.path.join(args.out_dir, rel)
            example["answer_snac_len"] = len(tokens)
        except Exception as e:
            logger.warning(f"Failed to encode row {example.get('id', idx)!r}: {e}")
            example["answer_snac"] = "error"
            example["answer_snac_len"] = -1
        return example

    def process(split, name):
        # Row ids in the unsharded split keep file names unique across shards
        # (same start index as datasets' contiguous sharding).
        if args.max_rows:
            split = split.select(range(min(args.max_rows, len(split))))
        if "turns" in split.column_names:
            n_rows = len(split)
            if args.per_turn:
                split = split.map(expand, batched=True, desc=f"one row per assistant turn ({name})")
            elif excluded:
                split = split.filter(lambda turns: keep(turns), input_columns="turns")
            logger.info(f"{name}: {n_rows} conversations -> {len(split)} rows")
        size, k, n = len(split), args.shard_index, args.num_shards
        offset = size // n * k + min(k, size % n)
        split = split.shard(n, k, contiguous=True)
        split = split.map(lambda ex, i: encode(ex, offset + i, name), with_indices=True,
                          desc=f"SNAC-encoding {name}")
        n_total = len(split)
        split = split.filter(lambda ex: ex["answer_snac"] != "error")
        if len(split) < n_total:
            logger.warning(f"{name}: dropped {n_total - len(split)}/{n_total} rows that failed")
        return split

    if isinstance(ds, DatasetDict):
        ds = DatasetDict({name: process(split, name) for name, split in ds.items()})
    else:
        ds = process(ds, "train")
    ds.save_to_disk(args.output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default="", help="prefix for relative audio paths")
    parser.add_argument("--audio-column", default="answer_audio")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="keep only the first N rows of each split (smoke tests)")
    parser.add_argument("--per-turn", action="store_true", help="one row per assistant turn of multi-turn rows")
    parser.add_argument("--exclude-tsv", help="TSV with an 'audio' column of answer clips to drop")
    main(parser.parse_args())
