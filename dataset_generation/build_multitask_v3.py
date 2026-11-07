"""Multitask v3 = v2 + context-dependent spoken follow-ups + paralinguistic tasks (see PARALINGUISTICS.md).

Follow-ups (dataset_generation/followups/): the seed exchange, then the generated follow-up exchanges whose audio
exists. Each training conversation gets one of three formats, so the model meets the demo's inputs:
  spoken   every user turn is audio (the end-to-end goal), 50 %
  typed    earlier user turns as text, the last one as audio, 25 %
  hinted   earlier user turns as text, the last one as audio with its transcript as the instruction line
           (the demo's "Transkribatu lehenik" mode), 25 %
Test conversations (test_followup) are all "spoken".

Paralinguistics (dataset_generation/paralinguistics/, clips.jsonl + replies.jsonl): each clip becomes one row of
  emotion        "Nola sentitzen da hiztuna?" -> a templated Basque sentence (40 %)
  voice          "Deskribatu hiztunaren ahotsa." -> gender, pitch, intonation, rate, volume and emotion (25 %)
  emotion_reply  bare audio, or audio with its English transcript as the instruction line -> a Latxa-70B reply
                 that reacts to how it was said (35 %)
plus `--neutral` Basque OmniVoice questions from v2 with the emotion instruction and a neutral answer, so the
model does not hear emotions in every voice. Held-out speakers' clips form test_para (emotion task only).
Templates and instructions are hand-written and need a native speaker's review.

    python dataset_generation/build_multitask_v3.py --output data/latxa_omni_multitask_v3_hf
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

V2 = "data/latxa_omni_multitask_v2_hf"
FOLLOWUPS = "data/followups_v3"
PARALINGUISTIC = "data/paralinguistic_v3"

EMOTION_INSTRUCTIONS = [
    "Nola sentitzen da hiztuna?", "Zer emozio adierazten du ahotsak?", "Entzun audioa eta esan zer sentitzen duen hiztunak.",
    "Zein da hiztunaren aldartea?", "Ahots-tonuari erreparatuta, nola dago hiztuna?",
    "Zer sentimendu antzematen diozu ahots honi?", "Deskribatu hiztunaren egoera emozionala.",
    "Nolako aldartea du grabazioko pertsonak?", "Zer emozio transmititzen du hiztunak?",
    "Esango al didazu nola sentitzen den hitz egiten ari dena?", "Nola dago hiztuna, ahotsaren arabera?",
    "Zer emozio dago esaldi honen atzean, hitzei ez, ahotsari begiratuta?",
]
VOICE_INSTRUCTIONS = [
    "Deskribatu hiztunaren ahotsa.", "Nolakoa da grabazioko ahotsa?", "Esan nolakoa den ahotsa: tonua, abiadura eta bolumena.",
    "Deskribatu, mesedez, ahotsaren ezaugarriak.", "Nola hitz egiten du hiztunak?",
    "Entzun eta deskribatu nola hitz egiten duen pertsona honek.", "Zer esan dezakezu hiztunaren ahotsari buruz?",
    "Aztertu ahotsa: tonua, intonazioa, erritmoa eta emozioa.",
]
EMOTIONS = {  # adjective (after "dago"), noun
    "angry": ("haserre", "haserrea"), "disgusted": ("nazkatuta", "nazka"), "fearful": ("beldurtuta", "beldurra"),
    "happy": ("pozik", "poza"), "sad": ("triste", "tristura"), "calm": ("lasai", "lasaitasuna"),
    "surprised": ("harrituta", "harridura"),
}
INTENSITY_ADVERB = {"low": "pixka bat ", "normal": "", "high": "oso "}
INTENSITY_NOUN = {"low": " apur bat", "normal": "", "high": " handia"}
EMOTION_TEMPLATES = [
    "Hiztuna {adj} dagoela dirudi.", "{Adj} entzuten da hiztuna.", "Ahots-tonuagatik, {adj} dagoela esango nuke.",
    "{Noun} nabari zaio ahotsean.", "{Adj} dago, ahotsaren arabera.",
]
NEUTRAL_TEMPLATES = [
    "Ez du emozio berezirik adierazten; tonu neutroan hitz egiten du.",
    "Lasai eta modu neutroan hitz egiten du; ez da emozio nabarmenik antzematen.",
    "Tonu neutroa du, emozio berezirik gabe.",
]
GENDER = {"male": "Gizonezko baten ahotsa da.", "female": "Emakumezko baten ahotsa da."}
PITCH = ["Tonu baxua", "Tonu ertaina", "Tonu altua"]
INTONATION = ["intonazio nahiko laua", "intonazio arrunta", "intonazio bizia"]
RATE = ["poliki", None, "azkar"]
VOLUME = ["apal", None, "ozen"]


def unit(*parts):
    """A stable number in [0, 1) for these parts (assignments do not change between runs)."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16 ** 12


def pick(options, *parts):
    return options[int(unit(*parts) * len(options))]


def emotion_sentence(emotion, intensity, *parts):
    if emotion == "neutral":
        return pick(NEUTRAL_TEMPLATES, *parts)
    adjective, noun = EMOTIONS[emotion]
    adjective = INTENSITY_ADVERB[intensity] + adjective
    noun = noun + INTENSITY_NOUN[intensity]
    text = pick(EMOTION_TEMPLATES, *parts).format(adj=adjective, Adj=adjective, Noun=noun)
    return text[0].upper() + text[1:]


def tercile(value, edges):
    return int(np.searchsorted(edges, value))


def voice_bins(clips):
    """Tercile edges per feature: pitch within gender (or corpus when unknown), the rest within corpus."""
    groups = defaultdict(lambda: defaultdict(list))
    for clip in clips:
        if clip["f0_median"] is None:
            continue
        groups[("pitch", clip["gender"] or clip["corpus"])]["v"].append(clip["f0_median"])
        groups[("intonation", clip["corpus"])]["v"].append(clip["f0_iqr_semitones"])
        groups[("rate", clip["corpus"])]["v"].append(clip["words_per_second"])
        groups[("volume", clip["corpus"])]["v"].append(clip["rms_db"])
    return {key: np.percentile(values["v"], [100 / 3, 200 / 3]) for key, values in groups.items()}


def voice_description(clip, bins):
    pitch = tercile(clip["f0_median"], bins[("pitch", clip["gender"] or clip["corpus"])])
    intonation = tercile(clip["f0_iqr_semitones"], bins[("intonation", clip["corpus"])])
    rate = RATE[tercile(clip["words_per_second"], bins[("rate", clip["corpus"])])]
    volume = VOLUME[tercile(clip["rms_db"], bins[("volume", clip["corpus"])])]
    if rate and volume:
        manner = f"{rate} eta {volume} hitz egiten du."
    elif rate:
        manner = f"{rate} hitz egiten du, bolumen arruntean."
    elif volume:
        manner = f"erritmo arruntean eta {volume} hitz egiten du."
    else:
        manner = "erritmo eta bolumen arruntean hitz egiten du."
    parts = [GENDER.get(clip["gender"], ""), f"{PITCH[pitch]} eta {INTONATION[intonation]} ditu;",
             manner,
             emotion_sentence(clip["emotion"], clip["intensity"], clip["id"], "voice")]
    return " ".join(p for p in parts if p)


def user(audio="", instruction="", text=""):
    return {"role": "user", "audio": audio, "instruction": instruction, "text": text}


def assistant(text):
    return {"role": "assistant", "audio": "", "instruction": "", "text": text}


def followup_rows(root, split):
    root = Path(root)
    with open(root / f"sources.{split}.jsonl", encoding="utf-8") as stream:
        seeds = {row["id"]: row for row in map(json.loads, stream)}
    rows = []
    for path in sorted(root.glob(f"followups.{split}.*.jsonl")):
        with open(path, encoding="utf-8") as stream:
            for generated in map(json.loads, stream):
                seed = seeds[generated["id"]]
                exchanges = []
                for index, turn in enumerate(generated["turns"], start=2):
                    wav = root / "wav" / f"{generated['id']}-{index}.flac"
                    if not wav.exists():
                        break
                    exchanges.append((str(wav), turn))
                if not exchanges:
                    continue
                draw = 0.0 if split == "test" else unit(generated["id"], "format")
                kind = "spoken" if draw < 0.5 else "typed" if draw < 0.75 else "hinted"
                spoken = kind == "spoken"
                turns = [user(seed["audio"] if spoken else "", seed["instruction"] if spoken else "",
                              seed["question"]), assistant(seed["answer"])]
                for position, (wav, turn) in enumerate(exchanges):
                    last = position == len(exchanges) - 1
                    if spoken or last:
                        turns.append(user(wav, turn["question"] if kind == "hinted" else "", turn["question"]))
                    else:
                        turns.append(user("", "", turn["question"]))
                    turns.append(assistant(turn["answer"]))
                rows.append({"id": generated["id"], "task": "chat", "source": f"followup_{kind}", "turns": turns})
    return rows


def paralinguistic_rows(root):
    root = Path(root)
    with open(root / "clips.jsonl", encoding="utf-8") as stream:
        clips = [clip for clip in map(json.loads, stream) if clip["f0_median"] is not None]
    replies = {}
    with open(root / "replies.jsonl", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            replies[row["key"]] = row["replies"]
    bins = voice_bins(clips)
    rows = {"train": [], "test_para": []}
    for clip in clips:
        emotion_answer = emotion_sentence(clip["emotion"], clip["intensity"], clip["id"], "emotion")
        emotion_row = [user(clip["audio"], pick(EMOTION_INSTRUCTIONS, clip["id"], "instruction")),
                       assistant(emotion_answer)]
        if clip["split"] == "test":
            rows["test_para"].append({"id": f"para_emotion_{clip['id']}", "task": "para_emotion",
                                      "source": clip["corpus"], "turns": emotion_row})
            continue
        draw = unit(clip["id"], "task")
        key = f"{clip['corpus']}|{clip['text']}|{clip['emotion']}|{clip['intensity']}"
        if draw < 0.35 and replies.get(key):
            hinted = unit(clip["id"], "hint") < 0.5
            turns = [user(clip["audio"], clip["text"] if hinted else ""), assistant(pick(replies[key], clip["id"]))]
            task = "para_reply"
        elif draw < 0.6:
            turns = [user(clip["audio"], pick(VOICE_INSTRUCTIONS, clip["id"], "instruction")),
                     assistant(voice_description(clip, bins))]
            task = "para_voice"
        else:
            turns, task = emotion_row, "para_emotion"
        rows["train"].append({"id": f"{task}_{clip['id']}", "task": task, "source": clip["corpus"], "turns": turns})
    return rows


def neutral_rows(v2_train, count, seed):
    """Basque OmniVoice questions (neutral read speech) with the emotion instruction and a neutral answer."""
    rng = random.Random(seed)
    rows = []
    for index in rng.sample(range(len(v2_train)), count * 3):
        first = v2_train[index]["turns"][0]
        if v2_train[index]["task"] != "chat" or not first.get("audio"):
            continue
        id_ = f"para_neutral_{v2_train[index]['id']}"
        rows.append({"id": id_, "task": "para_emotion", "source": "omnivoice_neutral",
                     "turns": [user(first["audio"], pick(EMOTION_INSTRUCTIONS, id_, "instruction")),
                               assistant(pick(NEUTRAL_TEMPLATES, id_))]})
        if len(rows) == count:
            break
    return rows


def main(args):
    v2 = load_from_disk(args.v2)
    features = v2["train"].features
    para = paralinguistic_rows(args.paralinguistic)
    added = {
        "train": followup_rows(args.followups, "train") + para["train"]
        + neutral_rows(v2["train"], args.neutral, args.seed),
        "test_followup": followup_rows(args.followups, "test"),
        "test_para": para["test_para"],
    }
    splits = dict(v2)
    for split, rows in added.items():
        new = Dataset.from_list(rows, features=features)
        splits[split] = concatenate_datasets([v2[split], new]).shuffle(seed=args.seed) if split == "train" else new
        print(split, len(rows), "added;", len(splits[split]), "rows")
    from datasets import DatasetDict
    DatasetDict(splits).save_to_disk(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2", default=V2)
    parser.add_argument("--followups", default=FOLLOWUPS)
    parser.add_argument("--paralinguistic", default=PARALINGUISTIC)
    parser.add_argument("--neutral", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
