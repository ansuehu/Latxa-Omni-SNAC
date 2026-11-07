"""EusListening (HiTZ): Basque listening comprehension, scored on Latxa-Omni Stage 1 checkpoints.

Each item is a multiple-choice question about a 4-12 minute recording. The model only
saw clips of up to 30 s in training (whisper pad_or_trim), so a recording is split into
30 s chunks, each given as its own <speech> segment, in order. The answer is the letter
with the highest probability as the first token of the reply. Every question is also
scored with no audio, which shows how much of the score comes from the recording.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import whisper
from huggingface_hub import snapshot_download
from tqdm import tqdm

from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.evaluate.merge_stage1_with_base import interpolate_model, parse_spec
from omni_speech.evaluate.text_forgetting_eval import letter_prompt
from omni_speech.model.builder import create_model

INSTRUCTION = "Entzun audioa eta erantzun galderari aukeraren letrarekin soilik."
SPEECH_TOKENS_PER_CHUNK = 300  # 3000 mel frames, halved by Whisper, then the projector's k=5


def load_items():
    root = Path(snapshot_download("HiTZ/EusListening", repo_type="dataset"))
    with open(root / "data" / "entzumena.jsonl", encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream]
    for item in items:
        item["audio"] = str(root / item["audio"])
    return items


def audio_chunks(path, mel_size):
    """30 s log-mel chunks, the last one padded, as in Stage1Dataset.load_speech."""
    waveform = whisper.load_audio(path)
    step = whisper.audio.N_SAMPLES
    return [
        whisper.log_mel_spectrogram(whisper.pad_or_trim(waveform[start:start + step]), n_mels=mel_size)
        .permute(1, 0).to(torch.bfloat16)
        for start in range(0, len(waveform), step)
    ]


def build_prompt(item, num_chunks, tokenizer, conv_mode):
    conv = conv_templates[conv_mode].copy()
    speech = "<speech>" * num_chunks + "\n" if num_chunks else ""
    conv.append_message(conv.roles[0], f"{INSTRUCTION}\n{speech}{letter_prompt(item['question'], item['candidates'])}")
    conv.append_message(conv.roles[1], None)
    return tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")


def letter_ids(tokenizer, count):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"[:count]]
    if any(len(token) != 1 for token in ids):
        raise ValueError(f"answer letters are not single tokens: {ids}")
    return [token[0] for token in ids]


@torch.no_grad()
def predict(model, input_ids, chunks, choice_ids):
    device = next(model.parameters()).device
    kwargs = {}
    if chunks:
        kwargs = {
            "speech": torch.stack(chunks).to(device),
            "speech_lengths": torch.tensor([chunk.shape[0] for chunk in chunks], device=device),
        }
    logits = model(input_ids=input_ids[None].to(device), **kwargs).logits[0, -1].float()
    return int(logits[choice_ids].argmax())


def main(args):
    items = load_items()
    if args.limit:
        items = items[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.checkpoints:
        checkpoint, alpha = parse_spec(spec)
        checkpoint = Path(checkpoint)
        name = f"{checkpoint.parent.name}_{checkpoint.name}" + ("" if alpha is None else f"_alpha{alpha:g}")
        tokenizer, model, _ = create_model(str(checkpoint), None)
        model = model.cuda().eval()
        # Training cut prompts at 4096 tokens; a long recording (300 tokens per chunk) would lose the question.
        model.config.tokenizer_model_max_length = None
        if alpha is not None:
            interpolate_model(model, args.base, alpha)
        rows = []
        chunks, chunks_path = None, None
        for item in tqdm(items, desc=name):
            if item["audio"] != chunks_path:
                chunks, chunks_path = audio_chunks(item["audio"], args.mel_size), item["audio"]
            choice_ids = letter_ids(tokenizer, len(item["candidates"]))
            input_ids = build_prompt(item, len(chunks), tokenizer, args.conv_mode)
            with_audio = predict(model, input_ids, chunks, choice_ids)
            no_audio = predict(model, build_prompt(item, 0, tokenizer, args.conv_mode), [], choice_ids)
            rows.append({
                "id": item["id"], "source": item["source_dataset"], "answer": item["answer"],
                "prediction": with_audio, "prediction_no_audio": no_audio, "chunks": len(chunks),
                "prompt_tokens": len(input_ids) + len(chunks) * (SPEECH_TOKENS_PER_CHUNK - 1),
            })
        by_source = defaultdict(list)
        for row in rows:
            by_source[row["source"]].append(row["prediction"] == row["answer"])
        metrics = {
            "checkpoint": str(checkpoint), "alpha": alpha,
            "eus_listening": sum(row["prediction"] == row["answer"] for row in rows) / len(rows),
            "eus_listening_no_audio": sum(row["prediction_no_audio"] == row["answer"] for row in rows) / len(rows),
            "chance": sum(1 / len(item["candidates"]) for item in items) / len(items),
            **{f"eus_listening_{source}": sum(hits) / len(hits) for source, hits in sorted(by_source.items())},
        }
        print(json.dumps(metrics), flush=True)
        with open(output_dir / "summary.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        with open(output_dir / f"{name}.predictions.jsonl", "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+",
                        help="Stage 1 checkpoint directories; <checkpoint>@<alpha> merges the LLM toward --base")
    parser.add_argument("--base", default="HiTZ/Latxa-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conv-mode", default="llama_3")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--limit", type=int, help="first N questions, for smoke tests")
    main(parser.parse_args())
