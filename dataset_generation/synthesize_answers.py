#!/usr/bin/env python3
"""Synthesize the assistant turns of a Latxa-answers dataset with OmniVoice.

Three steps, each run separately:

``--prepare`` (latxa-txat python, needs ``datasets``): export the HF dataset
written by ``generate_latxa_answers.py --build`` to ``--num-shards`` JSONL shards
under ``<out-dir>/input``. With ``--reuse-dir`` (an earlier out-dir), every
assistant turn whose text is unchanged gets that run's audio hard-linked into
``<out-dir>/wav``, so only changed turns are synthesized again.

default (omnivoice python, one shard per array task): synthesize every assistant
turn of ``input/shardXX.jsonl`` with the ``--answer-speaker`` voice at 24 kHz
(OmniVoice's native rate, which SNAC also uses). Numbers are spelled out first;
turns that still contain digits are skipped. Writes
``wav/<id>/<id>-<n>-assistant.flac`` and ``metadata/shardXX.speech.jsonl`` in
the layout ``Latxa-Omni/.../omnivoice/asr_wer.py`` reads (only assistant turns
carry "speech"). Resumable: existing files are not synthesized again.

``--build`` (latxa-txat python): copy the dataset with each assistant turn's
``audio`` set to its new file. A conversation is cut before its first assistant
turn without audio, so ``dump_snac_tokens.py`` never falls back to the old
OmniVoice clip of the original answer. High-WER turns stay in the dataset; pass
``high_wer_turns.tsv`` to ``dump_snac_tokens.py --exclude-tsv`` to drop them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

OMNIVOICE_DIR = Path(os.environ.get("LATXA_OMNI_DIR", "../Latxa-Omni")) / "dataset_generation/omnivoice"
REFS_TSV = Path("data/InstructS2S-200K/refs_eu/refs.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=Path("data/latxa70b_answers_hf"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/latxa70b_answers_tts"))
    parser.add_argument("--output-dataset", type=Path, help="--build: default <out-dir>/dataset")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reuse-dir", type=Path, help="--prepare: earlier out-dir whose audio of unchanged turns is linked")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refs-tsv", type=Path, default=REFS_TSV)
    parser.add_argument("--answer-speaker", default="maider")
    parser.add_argument("--language", default="eu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--ref-rms", type=float, default=0.1)
    parser.add_argument("--max-records", type=int, help="pilot: only the first N conversations")
    return parser.parse_args()


def shard_name(args) -> str:
    return f"shard{args.shard_index:02d}-of-{args.num_shards:02d}"


def prepare(args) -> None:
    from datasets import load_from_disk

    ds = load_from_disk(str(args.dataset))
    rows = [
        {"split": split, "id": r["id"], "turns": [{"role": t["role"], "text": t["text"]} for t in r["turns"]]}
        for split in ds
        for r in ds[split]
    ]
    out = args.out_dir / "input"
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_shards
    for k in range(n):
        start, end = len(rows) * k // n, len(rows) * (k + 1) // n
        args.shard_index = k
        with (out / f"{shard_name(args)}.jsonl").open("w", encoding="utf-8") as stream:
            for r in rows[start:end]:
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} conversations -> {n} shards in {out}")
    if args.reuse_dir:
        reuse(args, rows)


def reuse(args, rows) -> None:
    """Hard-link the earlier run's audio of every assistant turn whose text did not change."""
    old = {}
    for path in sorted((args.reuse_dir / "metadata").glob("*.speech.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                for index, turn in enumerate(record["turns"]):
                    if turn.get("speech"):
                        old[record["id"], index] = (turn["text"], Path(turn["speech"]))
    linked = 0
    for r in rows:
        for index, turn in enumerate(r["turns"]):
            text, src = old.get((r["id"], index), (None, None))
            if turn["role"] != "assistant" or text != turn["text"] or not src.is_file():
                continue
            dst = args.out_dir / "wav" / src.parent.name / src.name  # same <id>/<id>-<n>-assistant.flac layout
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.hardlink_to(src)
            linked += 1
    print(f"{linked} unchanged assistant turns linked from {args.reuse_dir}")


def synthesize(args) -> None:
    sys.path.insert(0, str(OMNIVOICE_DIR))
    from basque_numbers import spell_numbers
    from synthesize_omnivoice import atomic_write_audio, atomic_write_lines
    from synthesize_omnivoice_test import (
        Turn, audible, load_reference_audio, read_refs_tsv, retry_short_text, safe_id,
    )

    shard = shard_name(args)
    ref = read_refs_tsv(args.refs_tsv)[args.answer_speaker]
    records, pending, failures, done = [], [], [], 0
    with (args.out_dir / "input" / f"{shard}.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if args.max_records is not None and len(records) >= args.max_records:
                break
            record = json.loads(line)
            item_id = safe_id(record["id"])
            for index, turn in enumerate(record["turns"]):
                turn["from"] = "human" if turn["role"] == "user" else "gpt"
                if turn["role"] != "assistant":
                    continue
                tts_text = spell_numbers(turn["text"].strip())
                if any(c.isdigit() for c in tts_text):
                    failures.append((item_id, index, "unspellable digits"))
                    continue
                if tts_text != turn["text"]:
                    turn["tts_text"] = tts_text
                path = args.out_dir / "wav" / item_id / f"{item_id}-{index // 2 + 1}-assistant.flac"
                turn["speaker"], turn["speech"] = args.answer_speaker, str(path)
                if path.is_file():
                    done += 1
                else:
                    pending.append(Turn(item_id, index, "assistant", args.answer_speaker, tts_text, path, None))
            records.append(record)
    pending.sort(key=lambda t: len(t.text))  # length bucketing
    print(f"{shard}: {len(records)} conversations, {done} done, {len(pending)} pending", flush=True)

    if pending:
        import torch
        from omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=torch.float16)
        prompt = model.create_voice_clone_prompt(
            ref_audio=load_reference_audio(ref.wav, args.ref_rms), ref_text=ref.text
        )
        sr = model.sampling_rate
        start, audio_total = time.monotonic(), 0.0
        n_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        for b, offset in enumerate(range(0, len(pending), args.batch_size), 1):
            batch = pending[offset : offset + args.batch_size]
            gen = dict(language=args.language, num_step=args.num_step, guidance_scale=args.guidance_scale)
            try:
                audios = model.generate(text=[t.text for t in batch], voice_clone_prompt=[prompt] * len(batch), **gen)
            except Exception as exc:
                print(f"Batch {b} failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                torch.cuda.empty_cache()
                audios = [None] * len(batch)
                # One by one, so a single bad text does not sink the batch.
                for i, turn in enumerate(batch):
                    try:
                        audios[i] = model.generate(text=turn.text, voice_clone_prompt=prompt, **gen)[0]
                    except Exception as single_exc:
                        failures.append((turn.item_id, turn.turn_index, f"{type(single_exc).__name__}: {single_exc}"))
                        torch.cuda.empty_cache()
            for turn, audio in zip(batch, audios):
                if audio is None:
                    continue
                if not audible(audio):
                    audio = retry_short_text(model, turn, prompt, args)
                    if audio is None:
                        failures.append((turn.item_id, turn.turn_index, "empty/silent audio"))
                        continue
                atomic_write_audio(audio, sr, turn.wav_path, "flac")
                audio_total += len(audio) / sr
            if b % 50 == 0 or b == n_batches:
                elapsed = time.monotonic() - start
                print(
                    f"Batch {b}/{n_batches}  audio={audio_total / 3600:.2f}h  elapsed={elapsed / 3600:.2f}h  "
                    f"RTF={elapsed / max(audio_total, 1e-6):.3f}  failed={len(failures)}",
                    flush=True,
                )

    for record in records:
        for turn in record["turns"]:
            if "speech" in turn and not Path(turn["speech"]).is_file():
                del turn["speech"]
    atomic_write_lines((json.dumps(r, ensure_ascii=False) for r in records),
                       args.out_dir / "metadata" / f"{shard}.speech.jsonl")
    atomic_write_lines((f"{i}\t{t}\t{e}".replace("\n", " ") for i, t, e in failures),
                       args.out_dir / "failures" / f"{shard}.failed.tsv")
    print(f"Failures: {len(failures)}")


def build(args) -> None:
    from datasets import DatasetDict, load_from_disk

    speech = {}
    for path in sorted((args.out_dir / "metadata").glob("*.speech.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                for index, turn in enumerate(record["turns"]):
                    if turn.get("speech"):
                        speech[record["id"], index] = turn["speech"]
    print(f"{len(speech)} synthesized assistant turns")

    def attach(example):
        turns = example["turns"]
        for index, turn in enumerate(turns):
            if turn["role"] != "assistant":
                continue
            path = speech.get((example["id"], index))
            if path is None:
                example["turns"] = turns[:index - 1]  # drop the unanswered user turn too
                break
            turn["audio"] = path
        return example

    ds = load_from_disk(str(args.dataset))
    out = {}
    for split in ds:
        built = ds[split].map(attach).filter(lambda turns: len(turns) >= 2, input_columns="turns")
        print(f"{split}: {len(ds[split])} -> {len(built)} conversations")
        out[split] = built
    target = args.output_dataset or args.out_dir / "dataset"
    DatasetDict(out).save_to_disk(str(target))
    print(f"saved {target}")


def main() -> int:
    args = parse_args()
    if args.prepare:
        prepare(args)
    elif args.build:
        build(args)
    else:
        synthesize(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
