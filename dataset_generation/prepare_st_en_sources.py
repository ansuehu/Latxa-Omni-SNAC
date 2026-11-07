"""Basque sources for the eu->en translation tasks, to be translated by translate_eu_en.py.

  asr          the transcripts of the composite ASR clips already in the multi-task dataset
               (train, dev, test_asr), so st_en gets real speech instead of TTS'd InstructS2S
  eusparallel  sentences of HiTZ/EusParallel's human-written Basque documents (news and web
               text), at most --per-doc per document, for the written st_en_text task
  flores       FLORES-200 dev and devtest (human eu->en and eu->es): the st_en_text dev/test sets. They are
               translated too, so Latxa-70B's own BLEU against the human references is known

Writes one JSONL line per text: {"id", "source", "split", "eu", "audio"}.

    python dataset_generation/prepare_st_en_sources.py \\
        --multitask-dataset $DS/latxa_omni_multitask_tts_text_hf \\
        --eusparallel "$DS/st_en_latxa/src/eusparallel-*.parquet" \\
        --flores-dir $DS/st_en_latxa/src/flores200/flores200_dataset --output $DS/st_en_latxa/sources.jsonl
"""

import argparse
import glob
import json
import random
import re

import pyarrow.parquet as pq
from datasets import load_from_disk

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9\"'«(])")
CLEAN = re.compile(r"^[A-ZÁÉÍÓÚÑ\"'«(][^\n|<>{}\[\]@#=_*]*[.!?]$")


def sentences(document):
    for paragraph in document.split("\n"):
        yield from SENTENCE_END.split(paragraph.strip())


def usable(sentence, min_words, max_words):
    words = sentence.split()
    return (min_words <= len(words) <= max_words and CLEAN.match(sentence) is not None
            and "http" not in sentence and "www." not in sentence
            and sum(c.isdigit() for c in sentence) <= 0.1 * len(sentence))


def asr_rows(dataset):
    splits = {"train": dataset["train"], "dev": dataset["dev"], "test": dataset["test_asr"]}
    for split, rows in splits.items():
        for row in rows.filter(lambda task: task == "asr", input_columns="task"):
            user, answer = row["turns"]
            yield {"id": row["id"], "source": row["source"], "split": split, "eu": answer["text"], "audio": user["audio"]}


def eusparallel_rows(files, n, per_doc, min_words, max_words, rng):
    table = pq.read_table(files if isinstance(files, list) else sorted(glob.glob(files)), columns=["eu"])
    documents = table.column("eu").to_pylist()
    rows, seen = [], set()
    for index in rng.sample(range(len(documents)), len(documents)):
        picked = [s for s in sentences(documents[index] or "") if usable(s, min_words, max_words) and s not in seen]
        for number, sentence in enumerate(rng.sample(picked, min(per_doc, len(picked)))):
            seen.add(sentence)
            rows.append({"id": f"eusparallel_{index}_{number}", "source": "eusparallel", "split": "train",
                         "eu": sentence, "audio": ""})
        if len(rows) >= n:
            return rows[:n]
    return rows


def flores_rows(directory):
    for split in ("dev", "devtest"):
        with open(f"{directory}/{split}/eus_Latn.{split}", encoding="utf-8") as basque, \
                open(f"{directory}/{split}/eng_Latn.{split}", encoding="utf-8") as english, \
                open(f"{directory}/{split}/spa_Latn.{split}", encoding="utf-8") as spanish:
            for number, (eu, en, es) in enumerate(zip(basque, english, spanish)):
                yield {"id": f"flores_{split}_{number}", "source": "flores", "split": split,
                       "eu": eu.strip(), "audio": "", "en_ref": en.strip(), "es_ref": es.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multitask-dataset", required=True)
    parser.add_argument("--eusparallel", required=True, help="glob of the EusParallel parquet files")
    parser.add_argument("--flores-dir", required=True, help="flores200_dataset directory (dev/, devtest/)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--eusparallel-sentences", type=int, default=40000)
    parser.add_argument("--per-doc", type=int, default=2)
    parser.add_argument("--min-words", type=int, default=6)
    parser.add_argument("--max-words", type=int, default=35)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rows = list(asr_rows(load_from_disk(args.multitask_dataset)))
    rows += eusparallel_rows(args.eusparallel, args.eusparallel_sentences, args.per_doc,
                             args.min_words, args.max_words, random.Random(args.seed))
    rows += flores_rows(args.flores_dir)
    with open(args.output, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[(row["source"], row["split"])] = counts.get((row["source"], row["split"]), 0) + 1
    print(args.output, len(rows), counts)


if __name__ == "__main__":
    main()
