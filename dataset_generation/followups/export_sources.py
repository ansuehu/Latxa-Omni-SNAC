"""Pick first exchanges of spoken chat conversations as seeds for context-dependent follow-ups (data v3).

Stage 1 follow-up turns are self-contained questions ("Nola laguntzen du hazkunde-pentsamoldeak barkamen-prozesuan?")
or recall questions, so the model never learns spoken follow-ups that need the context ("Eta zenbat neurtzen
du?"). generate_followups.py writes such follow-ups for these seeds; the seed question's audio and text are the
voice-clone reference for synthesize_followups.py. Each seed also gets its conversation's user voice ("speaker",
from the OmniVoice source dataset); refs/ holds those voices' reference clips and refs.tsv for synthesis.

    python dataset_generation/followups/export_sources.py --output-dir data/followups_v3
"""

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from datasets import load_from_disk

DATASET = "data/latxa_omni_multitask_v2_hf"
VOICE_SOURCE = "data/latxa_omni_stage1_omnivoice_cv_hf"
REFS_TSVS = ("data/InstructS2S-200K/refs_eu/refs.tsv",
             "data/InstructS2S-200K/refs_eu_cv/refs.tsv")


def seed(row):
    """(question, answer, audio, instruction) of the first exchange, or None."""
    turns = row["turns"]
    if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
        return None
    user, answer = turns[0], turns[1]["text"] or ""
    if not user.get("audio") or not user.get("text") or not 20 <= len(answer) <= 1200:
        return None
    return user["text"], answer, user["audio"], user.get("instruction") or ""


def main(args):
    voices = load_from_disk(VOICE_SOURCE)
    speakers = {id_: turns[0].get("speaker") for split in voices
                for id_, turns in zip(voices[split]["id"], voices[split]["turns"])}
    dataset = load_from_disk(args.dataset)
    rng = random.Random(args.seed)
    used = set()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, source_split, count in (("train", "train", args.train), ("test", "test", args.test)):
        rows = dataset[source_split].filter(
            lambda task, source: task == "chat" and not source.startswith("chat_mt"),
            input_columns=["task", "source"], num_proc=8)
        seeds = []
        for index in rng.sample(range(len(rows)), min(len(rows), count * 2)):
            found = seed(rows[index])
            if found and speakers.get(rows[index]["id"]):
                question, answer, audio, instruction = found
                seeds.append({"id": f"followup_{split}_{rows[index]['id']}", "question": question,
                              "answer": answer, "audio": audio, "instruction": instruction,
                              "speaker": speakers[rows[index]["id"]]})
            if len(seeds) == count:
                break
        with open(output_dir / f"sources.{split}.jsonl", "w", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds)
        print(split, len(seeds), "of", len(rows))
        used.update(s["speaker"] for s in seeds)
    # The used voices' reference clips, copied next to the sources (refs/<speaker>.wav, refs/refs.tsv).
    refs_dir = output_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    with open(refs_dir / "refs.tsv", "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(["speaker", "wav", "text"])
        for tsv in REFS_TSVS:
            with open(tsv, encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    if row["speaker"] in used:
                        shutil.copy(row["wav"], refs_dir / f"{row['speaker']}.wav")
                        writer.writerow([row["speaker"], f"{row['speaker']}.wav", row["text"]])
                        used.discard(row["speaker"])
    if used:
        raise SystemExit(f"no reference for voices: {sorted(used)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train", type=int, default=25000)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=3407)
    main(parser.parse_args())
