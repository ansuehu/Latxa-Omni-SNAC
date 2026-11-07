"""EusListening as a cascade: Basque ASR transcribes each recording, then a text LLM answers from the transcript.

This is the reference for what the test can measure when the recording is understood.
Same letter scoring as eus_listening_eval.py; every question is also scored with no
transcript, which shows how much comes from the question and options alone.
Runs in transformers 4.45 (omni-snac) and 5 (omni-qwen).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from omni_speech.evaluate.eus_listening_data import INSTRUCTION, SAMPLE_RATE, letter_prompt, load_audio, load_items

TRANSCRIPT_INSTRUCTION = "Irakurri grabazioaren transkripzioa eta erantzun galderari aukeraren letrarekin soilik."


def transcribe(items, asr_model, path):
    """Transcripts keyed by recording path; appended as they finish, so a rerun resumes."""
    transcripts = {}
    if path.exists():
        with open(path, encoding="utf-8") as stream:
            transcripts = {row["audio"]: row["text"] for row in map(json.loads, stream)}
    todo = sorted({item["audio"] for item in items} - set(transcripts))
    if not todo:
        return transcripts
    asr = pipeline("automatic-speech-recognition", model=asr_model, torch_dtype=torch.float16, device="cuda:0")
    for audio_path in tqdm(todo, desc="asr"):
        audio = load_audio(audio_path)
        text = asr({"raw": audio, "sampling_rate": SAMPLE_RATE}, chunk_length_s=30, batch_size=16,
                   generate_kwargs={"language": "basque", "task": "transcribe"})["text"].strip()
        transcripts[audio_path] = text
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"audio": audio_path, "seconds": round(len(audio) / SAMPLE_RATE),
                                     "text": text}, ensure_ascii=False) + "\n")
    del asr
    torch.cuda.empty_cache()
    return transcripts


def build_prompt(tokenizer, item, transcript):
    question = letter_prompt(item["question"], item["candidates"])
    if transcript is None:
        content = f"{INSTRUCTION}\n{question}"
    else:
        content = f"{TRANSCRIPT_INSTRUCTION}\nTranskripzioa: {transcript}\n{question}"
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)["input_ids"]


@torch.no_grad()
def predict(model, input_ids, choice_ids):
    logits = model(input_ids=input_ids.to(model.device)).logits[0, -1].float()
    return int(logits[choice_ids].argmax()), int(logits.argmax())


def score(items, transcripts, model_name, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto").eval()
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
    if any(len(ids) != 1 for ids in letters):
        raise ValueError(f"answer letters are not single tokens: {letters}")
    letters = [ids[0] for ids in letters]
    name = model_name.split("/")[-1]
    rows = []
    for item in tqdm(items, desc=name):
        choice_ids = letters[:len(item["candidates"])]
        input_ids = build_prompt(tokenizer, item, transcripts[item["audio"]])
        with_transcript, top_token = predict(model, input_ids, choice_ids)
        no_transcript, _ = predict(model, build_prompt(tokenizer, item, None), choice_ids)
        rows.append({
            "id": item["id"], "source": item["source_dataset"], "audio": Path(item["audio"]).name,
            "answer": item["answer"], "prediction": with_transcript, "prediction_no_transcript": no_transcript,
            "top_token": tokenizer.decode([top_token]), "prompt_tokens": input_ids.shape[1],
        })
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row["prediction"] == row["answer"])
    metrics = {
        "model": model_name, "questions": len(rows),
        "eus_listening_transcript": sum(row["prediction"] == row["answer"] for row in rows) / len(rows),
        "eus_listening_no_transcript": sum(row["prediction_no_transcript"] == row["answer"] for row in rows) / len(rows),
        "chance": sum(1 / len(item["candidates"]) for item in items) / len(items),
        **{f"eus_listening_{source}": sum(hits) / len(hits) for source, hits in sorted(by_source.items())},
        "predicted_letters": dict(Counter("ABCD"[row["prediction"]] for row in rows)),
        "top_tokens": dict(Counter(row["top_token"] for row in rows).most_common(5)),
    }
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    with open(output_dir / "summary.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    with open(output_dir / f"{name}.predictions.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    del model
    torch.cuda.empty_cache()


def main(args):
    items = load_items(args.data)
    if args.limit:
        items = items[:args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transcripts = transcribe(items, args.asr, Path(args.transcripts))
    for model_name in args.models:
        score(items, transcripts, model_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*",
                        default=["HiTZ/Latxa-Llama-3.1-8B-Instruct", "HiTZ/Latxa-Llama-3.1-70B-Instruct"])
    parser.add_argument("--asr", default="HiTZ/whisper-large-v3-eu")
    parser.add_argument("--transcripts", default="saves/eus_listening/transcripts_whisper-large-v3-eu.jsonl",
                        help="cached ASR output, shared between runs")
    parser.add_argument("--data", default="saves/eus_listening/entzumena_dedup.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, help="first N questions, for smoke tests")
    main(parser.parse_args())
