"""Build the multi-task Stage 1 dataset (MULTITASK.md §2): spoken chat, ASR and eu->en speech translation.

Every row is a conversation whose user turns carry an ``instruction`` next to their audio
(stage1.py prompts it as ``<instruction>\\n<speech>``; an empty instruction is bare ``<speech>``).

  chat   the Latxa-70B answer conversations. With --chat-instruction-prob the first user
         turn gets a paraphrased "answer this" instruction; otherwise, and on later turns, it
         is empty, so the model still answers plain speech.
  asr    composite_corpus_eu_v2.1 (Common Voice, Basque Parliament, OpenSLR) -> transcript.
         Audio is copied out of the parquet files as FLAC under --audio-dir; utterances over
         30 s are skipped (Whisper mel input is cut at 30 s).
  st_en  one user turn per InstructS2S conversation -> the English original of that turn
         (the Basque audio is a TTS of its translation). Splits follow the chat splits.
  tts    a written user turn (no audio) asking to read a text aloud -> that same text. The
         texts are spoken answers of --tts-dataset (dump_snac_tokens.py output), so Stage 2 can
         train the talker on the same rows with their SNAC tokens (add_tts_rows.py).
  chat_text, st_en_text
         chat and st_en with the user turns written (their Basque text) instead of spoken, so
         the thinker does not forget how to follow typed turns.

With --st-en-translations (prepare_st_en_sources.py + translate_eu_en.py), Latxa-70B translations
replace InstructS2S as the translation data. st_en then pairs real composite speech (the ASR clips)
with the English of its transcript, and st_en_text uses human-written EusParallel sentences, with
FLORES-200 (human references) as its dev and test sets.

Output splits: train and dev (all tasks mixed, each row has ``task`` and ``source``),
test and test_mt (chat), test_asr, test_st_en, test_tts, test_st_en_text. The tts and then
the written rows are drawn last, each with their own random generator, so the other rows are
the same with or without them (only the train and dev order changes).

    python dataset_generation/build_multitask.py --chat-dataset <hf> \\
        --audio-dir $DS/latxa_omni_multitask/audio --output-dataset $DS/latxa_omni_multitask_hf
    # small run
    python dataset_generation/build_multitask.py ... --chat-train 200 --asr-train 200 --st-en-train 200
"""

import argparse
import glob
import io
import json
import logging
import os
import random
import re
from collections import Counter

import pyarrow.parquet as pq
import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DS = "data"
COMPOSITE = os.path.expanduser("~/.cache/huggingface/hub/datasets--HiTZ--composite_corpus_eu_v2.1/snapshots/*/data")
MAX_SECONDS = 30.0
USER_TURN = re.compile(r"-(\d+)-user\.\w+$")
TURN = {"role": Value("string"), "audio": Value("string"), "instruction": Value("string"), "text": Value("string")}
FEATURES = Features({
    "id": Value("string"), "task": Value("string"), "source": Value("string"), "turns": [TURN],
})


def composite_source(name):
    if name.startswith("common_voice_eu_"):
        return "cv"
    if re.match(r"\d+-\d+_\d{8}_", name):
        return "parl"
    return "oslr"


def single_turn(row_id, task, source, audio, instruction, target):
    return {"id": row_id, "task": task, "source": source, "turns": [
        {"role": "user", "audio": audio, "instruction": instruction, "text": ""},
        {"role": "assistant", "audio": "", "instruction": "", "text": target},
    ]}


def chat_row(row, instructions, prob, rng):
    turns = []
    for index, turn in enumerate(row["turns"]):
        instruction = ""
        if index == 0 and rng.random() < prob:
            instruction = rng.choice(instructions)
        turns.append({
            "role": turn["role"], "audio": turn["audio"] or "", "instruction": instruction, "text": turn["text"],
        })
    return {"id": row["id"], "task": "chat", "source": "chat_mt" if row["id"].startswith("mt_") else "chat",
            "turns": turns}


def st_en_candidates(row, english):
    """(audio, Basque text, English original) for each user turn of an InstructS2S conversation."""
    human = english.get(row["id"]) or []
    candidates = []
    for turn in row["turns"]:
        match = USER_TURN.search(turn["audio"] or "") if turn["role"] == "user" else None
        if match and int(match.group(1)) <= len(human) and human[int(match.group(1)) - 1].strip():
            candidates.append((turn["audio"], turn["text"].strip(), human[int(match.group(1)) - 1].strip()))
    return candidates


def st_en_row(row, english, instructions, rng):
    """One random user turn of an InstructS2S conversation, paired with its English original."""
    candidates = st_en_candidates(row, english)
    if not candidates:
        return None
    audio, _, target = rng.choice(candidates)
    return single_turn(f"st_en_{row['id']}", "st_en", "instructs2s", audio, rng.choice(instructions), target)


def st_en_text_row(row, english, instructions, rng):
    """st_en_row with the turn written: its Basque text in, the English original out."""
    candidates = [c for c in st_en_candidates(row, english) if c[1]]
    if not candidates:
        return None
    _, text, target = rng.choice(candidates)
    return {"id": f"st_en_text_{row['id']}", "task": "st_en_text", "source": "instructs2s", "turns": [
        {"role": "user", "audio": "", "instruction": rng.choice(instructions), "text": text},
        {"role": "assistant", "audio": "", "instruction": "", "text": target},
    ]}


def chat_text_row(row):
    """A chat conversation with every user turn written instead of spoken, and no instructions."""
    if not all(turn["text"].strip() for turn in row["turns"]):
        return None
    return {"id": f"chat_text_{row['id']}", "task": "chat_text",
            "source": "chat_mt" if row["id"].startswith("mt_") else "chat",
            "turns": [{"role": turn["role"], "audio": "", "instruction": "", "text": turn["text"]}
                      for turn in row["turns"]]}


def sample_text(split, n, make, rng):
    """Up to n rows of make(row) over a random order of split, skipping rows it returns None for."""
    rows = []
    for index in rng.sample(range(len(split)), len(split)):
        row = make(split[index])
        if row:
            rows.append(row)
            if len(rows) == n:
                break
    return rows


def tts_row(row, instructions, rng):
    """Read the last (spoken) answer of a dump_snac_tokens.py row aloud: text in, same text out."""
    text = row["turns"][-1]["text"]
    return {"id": f"tts_{row['id']}", "task": "tts", "source": "latxa70b_answers", "turns": [
        {"role": "user", "audio": "", "instruction": rng.choice(instructions), "text": text},
        {"role": "assistant", "audio": "", "instruction": "", "text": text},
    ]}


def sample_tts(source, sizes, instructions, rng):
    """{split: [(dump_snac_tokens row, tts row)]}. add_tts_rows.py calls this with the same seed,
    so Stage 2 gets exactly the Stage 1 tts rows."""
    picked = {}
    for name, n in sizes:
        split = source[name]
        indices = rng.sample(range(len(split)), min(n, len(split)))
        picked[name] = [(row, tts_row(row, instructions, rng)) for row in split.select(indices)]
    return picked


def load_translations(directory, lang="en", min_ratio=0.4, max_ratio=2.5):
    """prepare_st_en_sources.py rows with translate_eu_en.py's translation (key ``lang``: "en" or "es"),
    keyed by source and split.

    Drops translations that did not stop, are empty, span lines, or are far longer or shorter
    than the Basque (a sign of a note or a cut answer)."""
    translated = {}
    for path in sorted(glob.glob(os.path.join(directory, f"{lang}.*.jsonl"))):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                text = row[lang].strip()
                if row["finish_reason"] == "stop" and text and "\n" not in text:
                    translated[row["id"]] = text
    by_split, dropped = {}, 0
    with open(os.path.join(directory, "sources.jsonl"), encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            text = translated.get(row["id"])
            if not text or not min_ratio <= len(text) / max(len(row["eu"]), 1) <= max_ratio:
                dropped += 1
                continue
            by_split.setdefault((row["source"], row["split"]), []).append({**row, lang: text})
    logger.info("%s translations: %s, dropped %d", lang, {k: len(v) for k, v in sorted(by_split.items())}, dropped)
    return by_split


def latxa_st_row(row, lang, instructions, rng):
    """A composite ASR clip (real speech) -> Latxa-70B's translation of its transcript (task st_<lang>)."""
    task = f"st_{lang}"
    return single_turn(f"{task}_{row['id']}", task, row["source"], row["audio"], rng.choice(instructions), row[lang])


def latxa_st_text_row(row, lang, instructions, rng):
    """Written Basque -> translation (task st_<lang>_text): the human reference for FLORES, Latxa-70B's otherwise."""
    task = f"st_{lang}_text"
    return {"id": f"{task}_{row['id']}", "task": task, "source": row["source"], "turns": [
        {"role": "user", "audio": "", "instruction": rng.choice(instructions), "text": row["eu"]},
        {"role": "assistant", "audio": "", "instruction": "", "text": row.get(f"{lang}_ref") or row[lang]},
    ]}


def magpie_rows(path, sizes, rng, max_user_words=150, max_answer_words=400):
    """{split: rows} of typed Basque instruction -> answer pairs from HiTZ's translated Magpie."""
    with open(path, encoding="utf-8") as stream:
        total = sum(1 for _ in stream)
    wanted = sum(n for _, n in sizes)
    picked = set(rng.sample(range(total), min(total, 2 * wanted)))
    rows = []
    with open(path, encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index not in picked:
                continue
            user, answer = (turn["content"].strip() for turn in json.loads(line)["conversations"][:2])
            if user and answer and len(user.split()) <= max_user_words and len(answer.split()) <= max_answer_words:
                rows.append({"id": f"magpie_{index}", "task": "text_replay", "source": "magpie", "turns": [
                    {"role": "user", "audio": "", "instruction": "", "text": user},
                    {"role": "assistant", "audio": "", "instruction": "", "text": answer},
                ]})
    rng.shuffle(rows)
    split_rows, start = {}, 0
    for name, n in sizes:
        split_rows[name] = rows[start:start + n]
        start += n
    return split_rows


def load_english(path):
    english = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            english[row["id"]] = [turn["text"] for turn in row["turns"] if turn["from"] == "human"]
    return english


def sample_composite(files, n, rng):
    """(file, row) pairs drawn uniformly over all rows of ``files``; n=None takes every row."""
    rows = [(path, index) for path in files for index in range(pq.ParquetFile(path).metadata.num_rows)]
    if n is not None and n < len(rows):
        rows = rng.sample(rows, n)
    return sorted(rows)


def read_rows(path, indices):
    """Rows ``indices`` of a parquet file, reading only the row groups that hold them."""
    parquet = pq.ParquetFile(path)
    starts, offset = [], 0
    for group in range(parquet.num_row_groups):
        starts.append(offset)
        offset += parquet.metadata.row_group(group).num_rows
    by_group = {}
    for index in indices:
        group = max(g for g, start in enumerate(starts) if start <= index)
        by_group.setdefault(group, []).append(index - starts[group])
    for group, local in sorted(by_group.items()):
        yield from parquet.read_row_group(group).take(local).to_pylist()


def asr_rows(files, n, split, audio_dir, instructions, rng):
    picks = sample_composite(files, n, rng)
    out_dir = os.path.join(audio_dir, "asr", split)
    os.makedirs(out_dir, exist_ok=True)
    rows, skipped = [], Counter()
    by_file = {}
    for path, index in picks:
        by_file.setdefault(path, []).append(index)
    for path, indices in by_file.items():
        for record in read_rows(path, indices):
            name = os.path.basename(record["audio"]["path"])
            text = (record["sentence"] or "").strip()
            if not text:
                skipped["empty"] += 1
                continue
            if record["duration"] > MAX_SECONDS:
                skipped["long"] += 1
                continue
            target = os.path.join(out_dir, os.path.splitext(name)[0] + ".flac")
            if not os.path.exists(target):
                waveform, rate = sf.read(io.BytesIO(record["audio"]["bytes"]))
                sf.write(target + ".tmp", waveform, rate, format="FLAC")
                os.replace(target + ".tmp", target)
            source = composite_source(name)
            rows.append(single_turn(f"asr_{source}_{os.path.splitext(name)[0]}", "asr", source, target,
                                    rng.choice(instructions), text))
    logger.info("asr %s: %d rows %s, skipped %s", split, len(rows), Counter(r["source"] for r in rows), dict(skipped))
    return rows


def take(rows, n, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows if n is None else rows[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat-dataset", default=f"{DS}/latxa70b_answers_short_hf")
    parser.add_argument("--tts-dataset", default=f"{DS}/latxa70b_answers_short_tts/dataset_snac")
    parser.add_argument("--english-jsonl", default=f"{DS}/InstructS2S-200K/instruct_en_200k.jsonl")
    parser.add_argument("--composite-dir", default=COMPOSITE)
    parser.add_argument("--instructions", default=os.path.join(os.path.dirname(__file__), "multitask_instructions.json"))
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--chat-train", type=int, default=None, help="default: every chat train conversation")
    parser.add_argument("--chat-instruction-prob", type=float, default=0.5)
    parser.add_argument("--asr-train", type=int, default=60000)
    parser.add_argument("--st-en-train", type=int, default=30000)
    parser.add_argument("--tts-train", type=int, default=20000, help="0 leaves tts out")
    parser.add_argument("--text-chat-train", type=int, default=20000, help="written chat; 0 leaves it out")
    parser.add_argument("--text-st-en-train", type=int, default=20000, help="written st_en; 0 leaves it out")
    parser.add_argument("--st-en-translations",
                        help="prepare_st_en_sources.py + translate_eu_en.py directory: Latxa-70B translations "
                             "replace InstructS2S for st_en (--latxa-st-en-train real-speech clips) and st_en_text "
                             "(--latxa-text-st-en-train EusParallel sentences; FLORES dev/devtest as dev/test)")
    parser.add_argument("--latxa-st-en-train", type=int, default=40000,
                        help="with --st-en-translations: real-speech eu->en rows")
    parser.add_argument("--latxa-text-st-en-train", type=int, default=30000,
                        help="with --st-en-translations: written eu->en rows (EusParallel)")
    parser.add_argument("--st-en-instructs2s-train", type=int, default=5000,
                        help="with --st-en-translations: InstructS2S st_en train rows kept")
    parser.add_argument("--st-es-train", type=int, default=0,
                        help="with --st-en-translations: real-speech eu->es rows (needs translate_eu_en.py --target es)")
    parser.add_argument("--text-st-es-train", type=int, default=0,
                        help="with --st-en-translations: written eu->es rows (EusParallel; FLORES dev/devtest)")
    parser.add_argument("--instructions-expanded",
                        help="reviewed paraphrases (expand_instructions.py output): train/dev instructions are redrawn "
                             "from it; rows and test splits keep the seeds")
    parser.add_argument("--magpie-jsonl", default=f"{DS}/magpie_eu/magpie_eu.jsonl")
    parser.add_argument("--magpie-train", type=int, default=0,
                        help="typed Basque Magpie instruction -> answer rows (task text_replay); 0 leaves them out")
    parser.add_argument("--dev-per-task", type=int, default=300,
                        help="asr, st_en and tts rows in dev (chat keeps its dev); also the size of test_tts")
    parser.add_argument("--test-cv", type=int, default=1000, help="Common Voice rows in test_asr (Parliament/OpenSLR: all)")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.instructions, encoding="utf-8") as stream:
        instructions = json.load(stream)
    chat = load_from_disk(args.chat_dataset)
    english = load_english(args.english_jsonl)
    files = sorted(glob.glob(os.path.join(args.composite_dir, "*.parquet")))
    split_files = lambda prefix: [f for f in files if os.path.basename(f).startswith(prefix)]
    if not split_files("train-"):
        raise FileNotFoundError(f"no composite parquet files in {args.composite_dir}")

    def chat_split(name, n=None):
        return [chat_row(row, instructions["chat"], args.chat_instruction_prob, rng) for row in take(chat[name], n, rng)]

    def st_split(name, n=None):
        rows = (st_en_row(row, english, instructions["st_en"], rng) for row in chat[name])
        return take([row for row in rows if row], n, rng)

    train = (
        chat_split("train", args.chat_train)
        + asr_rows(split_files("train-"), args.asr_train, "train", args.audio_dir, instructions["asr"], rng)
        + st_split("train", args.st_en_train)
    )
    dev = (
        chat_split("dev")
        + asr_rows(split_files("dev_"), args.dev_per_task, "dev", args.audio_dir, instructions["asr"], rng)
        + st_split("dev", args.dev_per_task)
    )
    test_asr = (
        asr_rows(split_files("test_cv"), args.test_cv, "test", args.audio_dir, instructions["asr"], rng)
        + asr_rows(split_files("test_parl") + split_files("test_oslr"), None, "test", args.audio_dir,
                   instructions["asr"], rng)
    )
    splits = {
        # dev is shuffled with its own generator, so --dev_size trims it evenly across tasks.
        "train": take(train, None, rng), "dev": take(dev, None, random.Random(args.seed)),
        "test": chat_split("test"), "test_asr": test_asr,
        "test_st_en": st_split("test"),
    }
    if "test_mt" in chat:
        splits["test_mt"] = chat_split("test_mt")
    if args.tts_train:
        # After everything else, with its own generator: the other rows are unchanged.
        tts_rng = random.Random(args.seed + 1)
        sizes = (("train", args.tts_train), ("dev", args.dev_per_task), ("test", args.dev_per_task))
        tts = {name: [row for _, row in pairs] for name, pairs in
               sample_tts(load_from_disk(args.tts_dataset), sizes, instructions["tts"], tts_rng).items()}
        splits["train"] = take(splits["train"] + tts["train"], None, tts_rng)
        splits["dev"] = take(splits["dev"] + tts["dev"], None, random.Random(args.seed))
        splits["test_tts"] = tts["test"]
    if args.text_chat_train or args.text_st_en_train:
        # Written versions of the spoken tasks, so the thinker keeps reading text. Drawn last with
        # their own generator, like tts, which leaves every other row unchanged.
        text_rng = random.Random(args.seed + 2)
        st_text = lambda row: st_en_text_row(row, english, instructions["st_en_text"], text_rng)
        text = {"train": [], "dev": []}
        if args.text_chat_train:
            text["train"] += sample_text(chat["train"], args.text_chat_train, chat_text_row, text_rng)
            text["dev"] += sample_text(chat["dev"], args.dev_per_task, chat_text_row, text_rng)
        if args.text_st_en_train:
            text["train"] += sample_text(chat["train"], args.text_st_en_train, st_text, text_rng)
            text["dev"] += sample_text(chat["dev"], args.dev_per_task, st_text, text_rng)
            splits["test_st_en_text"] = sample_text(chat["test"], None, st_text, text_rng)
        splits["train"] = take(splits["train"] + text["train"], None, text_rng)
        splits["dev"] = take(splits["dev"] + text["dev"], None, random.Random(args.seed))
    if args.st_en_translations:
        # Latxa-70B translations replace InstructS2S as the translation data, each language with its
        # own generator so every other row is unchanged. InstructS2S keeps --st-en-instructs2s-train
        # rows (conversational questions) and its test sets under new names.
        # Their own size flags: changing --st-en-train/--text-st-en-train would shift the base build's draws.
        sizes = {"en": (args.latxa_st_en_train, args.latxa_text_st_en_train),
                 "es": (args.st_es_train, args.text_st_es_train)}
        for offset, lang in enumerate(("en", "es"), start=3):
            spoken_n, text_n = sizes[lang]
            if not spoken_n and not text_n:
                continue
            st_rng = random.Random(args.seed + offset)
            latxa = load_translations(args.st_en_translations, lang)
            rows = lambda source, split: latxa.get((source, split), [])
            spoken = lambda split: [r for s in ("cv", "parl", "oslr") for r in rows(s, split)]
            task, text_task = f"st_{lang}", f"st_{lang}_text"
            speech_row = lambda r: latxa_st_row(r, lang, instructions[task], st_rng)
            text_row = lambda r: latxa_st_text_row(r, lang, instructions[text_task], st_rng)
            keep = lambda name: [r for r in splits[name] if r["task"] not in (task, text_task)]
            new = {
                "train": ([speech_row(r) for r in take(spoken("train"), spoken_n, st_rng)]
                          + [text_row(r) for r in take(rows("eusparallel", "train"), text_n, st_rng)]),
                "dev": ([speech_row(r) for r in spoken("dev")]
                        + [text_row(r) for r in take(rows("flores", "dev"), args.dev_per_task, st_rng)]),
            }
            if lang == "en":
                new["train"] += take([r for r in splits["train"] if r["task"] == task],
                                     args.st_en_instructs2s_train, st_rng)
                for name in ("test_st_en", "test_st_en_text"):
                    if name in splits:
                        splits[f"{name}_instructs2s"] = splits.pop(name)
            splits["train"] = take(keep("train") + new["train"], None, st_rng)
            splits["dev"] = take(keep("dev") + new["dev"], None, random.Random(args.seed))
            splits[f"test_{task}"] = [speech_row(r) for r in spoken("test")]
            splits[f"test_{text_task}"] = [text_row(r) for r in rows("flores", "devtest")]
    if args.magpie_train:
        # Typed general-purpose instructions (text replay), so the thinker keeps its text skills.
        magpie = magpie_rows(args.magpie_jsonl, (("train", args.magpie_train), ("dev", args.dev_per_task)),
                             random.Random(args.seed + 5))
        splits["train"] = take(splits["train"] + magpie["train"], None, random.Random(args.seed + 5))
        splits["dev"] = take(splits["dev"] + magpie["dev"], None, random.Random(args.seed))
    if args.instructions_expanded:
        # Redraw the non-empty instructions of train and dev from the reviewed paraphrases, with their own
        # generator, so the rows themselves (and every test split) stay as built from the seeds.
        with open(args.instructions_expanded, encoding="utf-8") as stream:
            expanded = json.load(stream)
        redraw_rng = random.Random(args.seed + 6)
        for name in ("train", "dev"):
            for row in splits[name]:
                for turn in row["turns"]:
                    if turn["instruction"] and row["task"] in expanded:
                        turn["instruction"] = redraw_rng.choice(expanded[row["task"]])
    dataset = DatasetDict({name: Dataset.from_list(rows, features=FEATURES) for name, rows in splits.items()})
    for name, split in dataset.items():
        logger.info("%s: %d rows %s", name, len(split), dict(Counter(split["task"])))
    dataset.save_to_disk(args.output_dataset)
    logger.info("saved %s", args.output_dataset)


if __name__ == "__main__":
    main()
