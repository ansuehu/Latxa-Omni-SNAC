"""EusListening with Qwen3-Omni, a strong external SpeechLM, to see how much the test can be solved.

Same prompt and scoring as eus_listening_eval.py: the answer is the letter with the highest
probability as the first token of the reply, with and without the recording. Qwen3-Omni takes
the whole recording (up to ~40 min) as one audio input, so there is no 30 s chunking.
Runs in the omni-qwen env (transformers 5), not omni-snac.
"""

import argparse
import json
import types
from collections import Counter, defaultdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import (AutoTokenizer, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor,
                          WhisperFeatureExtractor)

from omni_speech.evaluate.eus_listening_data import INSTRUCTION, SAMPLE_RATE, letter_prompt, load_audio, load_items


class AudioOnlyProcessor(Qwen3OmniMoeProcessor):
    """The processor without its image and video parts, which need torchvision (not in omni-qwen)."""

    def check_argument_for_proper_class(self, name, arg):
        if not isinstance(arg, types.SimpleNamespace):
            super().check_argument_for_proper_class(name, arg)


def load_processor(model):
    root = Path(snapshot_download(model))
    with open(root / "chat_template.json", encoding="utf-8") as stream:
        chat_template = json.load(stream)["chat_template"]
    stub = types.SimpleNamespace(merge_size=2, model_input_names=[])  # only read for image/video tokens
    return AudioOnlyProcessor(image_processor=stub, video_processor=stub,
                              feature_extractor=WhisperFeatureExtractor.from_pretrained(model),
                              tokenizer=AutoTokenizer.from_pretrained(model), chat_template=chat_template)


def build_inputs(processor, item, audio):
    content = [{"type": "audio", "audio": "placeholder"}] if audio is not None else []
    content.append({"type": "text", "text": f"{INSTRUCTION}\n{letter_prompt(item['question'], item['candidates'])}"})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         add_generation_prompt=True, tokenize=False)
    return processor(text=text, audio=[audio] if audio is not None else None, return_tensors="pt")


@torch.no_grad()
def predict(model, inputs, choice_ids):
    inputs = inputs.to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    logits = model.thinker(**inputs).logits[0, -1].float()
    return int(logits[choice_ids].argmax()), int(logits.argmax())


def main(args):
    items = load_items(args.data)
    if args.longest_first:
        # Stable sort, so a recording's questions stay together; puts the memory peak first.
        items.sort(key=lambda item: -Path(item["audio"]).stat().st_size)
    if args.limit:
        items = items[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(args.model)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", enable_audio_output=False,
        attn_implementation=args.attn).eval()
    tokenizer = processor.tokenizer
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError(f"answer letters are not single tokens: {letters}")
    letters = [ids[0] for ids in letters]

    rows, audio, audio_path = [], None, None
    for item in tqdm(items):
        if item["audio"] != audio_path:
            audio, audio_path = load_audio(item["audio"]), item["audio"]
        choice_ids = letters[:len(item["candidates"])]
        inputs = build_inputs(processor, item, audio)
        with_audio, top_token = predict(model, inputs, choice_ids)
        no_audio, _ = predict(model, build_inputs(processor, item, None), choice_ids)
        rows.append({
            "id": item["id"], "source": item["source_dataset"], "audio": Path(item["audio"]).name,
            "answer": item["answer"], "prediction": with_audio, "prediction_no_audio": no_audio,
            "top_token": tokenizer.decode([top_token]), "seconds": round(len(audio) / SAMPLE_RATE),
            "prompt_tokens": inputs["input_ids"].shape[1],
        })
        with open(output_dir / "predictions.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row["prediction"] == row["answer"])
    metrics = {
        "model": args.model, "data": args.data, "questions": len(rows),
        "eus_listening": sum(row["prediction"] == row["answer"] for row in rows) / len(rows),
        "eus_listening_no_audio": sum(row["prediction_no_audio"] == row["answer"] for row in rows) / len(rows),
        "chance": sum(1 / len(item["candidates"]) for item in items) / len(items),
        **{f"eus_listening_{source}": sum(hits) / len(hits) for source, hits in sorted(by_source.items())},
        "predicted_letters": dict(Counter("ABCD"[row["prediction"]] for row in rows)),
        "top_tokens": dict(Counter(row["top_token"] for row in rows).most_common(5)),
    }
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--data", default="saves/eus_listening/entzumena_dedup.jsonl",
                        help="EusListening jsonl; audio paths are resolved against the HF snapshot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--limit", type=int, help="first N questions, for smoke tests")
    parser.add_argument("--longest-first", action="store_true", help="largest recordings first, for smoke tests")
    main(parser.parse_args())
