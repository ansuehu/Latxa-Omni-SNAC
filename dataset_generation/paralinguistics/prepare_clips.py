"""Emotional speech clips for the paralinguistic Stage 1 tasks (data v3, see PARALINGUISTICS.md).

Reads CREMA-D (mteb/crema-d), RAVDESS (birgermoell/ravdess) and ESD English (AbstractTTS/ESD_english), writes
16 kHz mono wavs and one metadata line per clip:
  {"id", "corpus", "speaker", "gender" (or null), "emotion" (unified, English), "intensity" (low/normal/high),
   "text" (the English sentence), "duration", "f0_median", "f0_iqr_semitones", "rms_db", "words_per_second",
   "split"}
Pitch is pYIN on the trimmed clip; held-out speakers (split "test"): CREMA-D actors >= 1073, RAVDESS actors
23-24, ESD speakers 0019-0020.

    python dataset_generation/paralinguistics/prepare_clips.py --output-dir data/paralinguistic_v3
"""

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

DATASETS = Path("data")
SAMPLE_RATE = 16000
CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.", "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.", "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}
CREMA_EMOTIONS = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMA_INTENSITY = {"LO": "low", "MD": "normal", "HI": "high", "XX": "normal", "X": "normal"}  # one file says X
RAVDESS_EMOTIONS = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
                    "07": "disgusted", "08": "surprised"}
RAVDESS_SENTENCES = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}
ESD_EMOTIONS = {"Neutral": "neutral", "Angry": "angry", "Happy": "happy", "Sad": "sad", "Surprise": "surprised"}


def crema():
    for path in sorted((DATASETS / "crema_d_mteb/data").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            actor, sentence, emotion, level = Path(row["audio"]["path"]).stem.split("_")
            yield {"id": f"crema_{Path(row['audio']['path']).stem}", "corpus": "crema_d", "speaker": actor,
                   "gender": None, "emotion": CREMA_EMOTIONS[emotion], "intensity": CREMA_INTENSITY[level],
                   "text": CREMA_SENTENCES[sentence], "split": "test" if int(actor) >= 1073 else "train"}, \
                row["audio"]["bytes"]


def ravdess():
    for path in sorted((DATASETS / "birgermoell_ravdess").glob("Actor_*/*.wav")):
        _, _, emotion, intensity, statement, _, actor = path.stem.split("-")
        yield {"id": f"ravdess_{path.stem}", "corpus": "ravdess", "speaker": actor,
               "gender": "male" if int(actor) % 2 else "female", "emotion": RAVDESS_EMOTIONS[emotion],
               "intensity": "high" if intensity == "02" else "normal", "text": RAVDESS_SENTENCES[statement],
               "split": "test" if int(actor) >= 23 else "train"}, path.read_bytes()


def esd():
    for path in sorted((DATASETS / "AbstractTTS_ESD_english/data").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            speaker = row["file"].split("_")[0]
            yield {"id": f"esd_{Path(row['file']).stem}", "corpus": "esd_en", "speaker": speaker,
                   "gender": row["gender"], "emotion": ESD_EMOTIONS[row["emotion"]], "intensity": "normal",
                   "text": row["transcription"].strip(), "split": "test" if speaker in ("0019", "0020") else "train"}, \
                row["audio"]["bytes"]


def process(job):
    meta, data, wav_path = job
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, SAMPLE_RATE)
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    f0, voiced, _ = librosa.pyin(trimmed, fmin=65, fmax=500, sr=SAMPLE_RATE, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    seconds = len(trimmed) / SAMPLE_RATE
    return {**meta, "audio": str(wav_path), "duration": round(len(audio) / SAMPLE_RATE, 3),
            # Median and interquartile range in semitones: robust to pYIN's octave errors.
            "f0_median": round(float(np.median(f0)), 1) if len(f0) else None,
            "f0_iqr_semitones": round(float(np.subtract(*np.percentile(12 * np.log2(f0), [75, 25]))), 2)
            if len(f0) else None,
            "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(trimmed ** 2)) + 1e-9)), 2),
            "words_per_second": round(len(meta["text"].split()) / max(seconds, 0.1), 2)}


def main(args):
    output_dir = Path(args.output_dir)
    jobs = [(meta, data, output_dir / "wav" / meta["corpus"] / f"{meta['id']}.wav")
            for corpus in (crema, ravdess, esd) for meta, data in corpus()]
    print(f"{len(jobs)} clips", flush=True)
    with ProcessPoolExecutor(args.workers) as pool, open(output_dir / "clips.jsonl", "w", encoding="utf-8") as stream:
        for count, row in enumerate(pool.map(process, jobs, chunksize=16), 1):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if count % 2000 == 0:
                print(count, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    main(parser.parse_args())
