"""Speak generate_followups.py's follow-up questions with OmniVoice, in each conversation's user voice.

The voice is the seed's "speaker" (export_sources.py), cloned from refs/<speaker>.wav and refs/refs.tsv, the same
reference the seed question was synthesised from. Writes wav/<seed id>-<turn>.flac (16 kHz); existing files are
skipped, so it resumes. --shard/--num-shards split the work for a SLURM array.

    python dataset_generation/followups/synthesize_followups.py --root data/followups_v3
"""

import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 16000


def load_reference(path, target_rms=0.1):
    """(waveform, rate), RMS-normalised with the peak kept below 0.99 (as synthesize_omnivoice.py)."""
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    rms, peak = float(np.sqrt(np.mean(audio ** 2))), float(np.abs(audio).max())
    if rms > 0:
        audio = audio * min(target_rms / rms, 0.99 / peak)
    return torch.from_numpy(audio).unsqueeze(0), rate


def audible(audio, rate, words):
    """Not silent and not absurdly short or long for its word count."""
    seconds = len(audio) / rate
    return float(np.sqrt(np.mean(audio ** 2))) > 0.005 and 0.25 * words <= seconds + 0.5 and seconds <= 1.5 * words + 3


def pending_turns(root):
    seeds = {}
    for path in root.glob("sources.*.jsonl"):
        with open(path, encoding="utf-8") as stream:
            seeds.update((row["id"], row) for row in map(json.loads, stream))
    turns = []
    for path in sorted(root.glob("followups.*.jsonl")):
        with open(path, encoding="utf-8") as stream:
            for row in map(json.loads, stream):
                for index, turn in enumerate(row["turns"], start=2):
                    wav = root / "wav" / f"{row['id']}-{index}.flac"
                    if not wav.exists():
                        turns.append((seeds[row["id"]]["speaker"], turn["question"], wav))
    return turns


def main(args):
    from omnivoice import OmniVoice

    root = Path(args.root)
    (root / "wav").mkdir(exist_ok=True)
    with open(root / "refs" / "refs.tsv", encoding="utf-8", newline="") as stream:
        refs = {row["speaker"]: row for row in csv.DictReader(stream, delimiter="\t")}
    # Shard by a stable hash, not by position: shards start at different times and see different pending lists.
    turns = [t for t in pending_turns(root)
             if zlib.crc32(t[2].name.encode()) % args.num_shards == args.shard]
    # Same voice together (one clone prompt in memory at a time), then by length for batching.
    turns.sort(key=lambda t: (t[0], len(t[1])))
    print(f"shard {args.shard}/{args.num_shards}: {len(turns)} questions to synthesise", flush=True)
    model = OmniVoice.from_pretrained(args.model, device_map="cuda:0", dtype=torch.float16)
    prompts, failed = {}, 0

    def prompt(speaker):
        if speaker not in prompts:
            prompts.clear()
            ref = refs[speaker]
            prompts[speaker] = model.create_voice_clone_prompt(
                ref_audio=load_reference(root / "refs" / ref["wav"]), ref_text=ref["text"])
        return prompts[speaker]

    def generate(batch):
        return model.generate(text=[t[1] for t in batch], language="eu",
                              voice_clone_prompt=[prompt(t[0]) for t in batch],
                              num_step=args.num_step, guidance_scale=args.guidance_scale)

    for offset in range(0, len(turns), args.batch_size):
        batch = turns[offset:offset + args.batch_size]
        audios = generate(batch)
        for turn, audio in zip(batch, audios):
            if not audible(audio, model.sampling_rate, len(turn[1].split())):
                audio = generate([turn])[0]  # one retry; sampling differs
                if not audible(audio, model.sampling_rate, len(turn[1].split())):
                    failed += 1
                    continue
            audio = torchaudio.functional.resample(torch.from_numpy(audio), model.sampling_rate, SAMPLE_RATE).numpy()
            tmp = turn[2].with_suffix(".tmp.flac")
            sf.write(tmp, audio, SAMPLE_RATE)
            tmp.rename(turn[2])
        if (offset // args.batch_size) % 50 == 0:
            print(f"{offset + len(batch)}/{len(turns)} failed={failed}", flush=True)
    print(f"done, failed={failed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-step", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    main(parser.parse_args())
