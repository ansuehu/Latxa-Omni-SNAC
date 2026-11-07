"""EusListening loading and prompt helpers shared by the Qwen3-Omni and cascade evals.

Kept free of omni_speech model imports so it works in any env (omni-qwen, omni-snac).
"""

import json
import subprocess
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

INSTRUCTION = "Entzun audioa eta erantzun galderari aukeraren letrarekin soilik."
SAMPLE_RATE = 16000


def letter_prompt(question, candidates):
    # As text_forgetting_eval.letter_prompt, copied because that module needs the omni-snac model code.
    lines = [f"Galdera: {question}"]
    lines += [f"{letter}: {candidate}" for letter, candidate in zip("ABCD", candidates)]
    lines.append("Erantzuna:")
    return "\n".join(lines)


def load_items(data):
    root = Path(snapshot_download("HiTZ/EusListening", repo_type="dataset"))
    with open(data or root / "data" / "entzumena.jsonl", encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream]
    for item in items:
        item["audio"] = str(root / item["audio"])
    return items


def load_audio(path):
    """Mono 16 kHz float32, decoded with ffmpeg as in whisper.load_audio."""
    command = ["ffmpeg", "-nostdin", "-threads", "0", "-i", path, "-f", "s16le", "-ac", "1",
               "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(command, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
