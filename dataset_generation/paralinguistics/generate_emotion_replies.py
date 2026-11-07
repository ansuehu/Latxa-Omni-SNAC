"""Basque replies that react to how a sentence was said, written by Latxa-70B (vLLM), for the emotion-aware task.

One request per (sentence, emotion, intensity) of prepare_clips.py's clips.jsonl; each asks for VARIANTS
different replies, so every clip of that combination can get its own. Output, one line per combination:
{"key", "replies": [...], "finish_reason"} ("replies": [] and "raw" when the reply is not valid JSON).
Resumable.

    python dataset_generation/paralinguistics/generate_emotion_replies.py --clips clips.jsonl --output replies.jsonl
    python dataset_generation/paralinguistics/generate_emotion_replies.py --clips clips.jsonl --dry-run
"""

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = 4
SYSTEM = (
    "You write training data for a Basque voice assistant that can hear HOW the user speaks. You get an English "
    "sentence the user said and the emotion in their voice. Write {variants} different replies the assistant "
    "could give, in standard Basque (batua).\n"
    "Rules:\n"
    "- React both to what was said and to how it was said: comfort a sad user, calm an angry or frightened one, "
    "share a happy one's joy, respond to surprise or disgust.\n"
    "- The words alone do not show the emotion; only the voice does. Vary how you show you noticed it: in some "
    "replies name the feeling (\"Haserre zaudela nabari da\"), in others just respond in the fitting way.\n"
    "- If the voice is neutral or calm, reply normally and do not invent an emotion.\n"
    "- 1-3 short sentences each, natural when read aloud, no lists, markdown or emojis.\n"
    "Reply with a JSON list of {variants} strings only."
)
EXAMPLE = (
    ("It's eleven o'clock.", "sad", "high"),
    ["Bai, hamaikak dira. Oso triste entzuten zaitut; zerbait gertatu al da?",
     "Hamaikak dira dagoeneko. Ahots apala duzu gaur, ondo zaude?",
     "Bai, berandu da. Triste zaudela dirudi; hitz egin nahi baduzu, hemen nago.",
     "Hamaikak, bai. Gaur egun gogorra izan dela ematen du; zertan lagun zaitzaket?"],
)
INTENSITY = {"low": "slightly ", "normal": "", "high": "very "}


def key(clip):
    return f"{clip['corpus']}|{clip['text']}|{clip['emotion']}|{clip['intensity']}"


def user_message(text, emotion, intensity):
    return f"Sentence: {text}\nVoice: {INTENSITY[intensity]}{emotion}"


def messages(text, emotion, intensity):
    (example, replies) = EXAMPLE
    return [
        {"role": "system", "content": SYSTEM.format(variants=VARIANTS)},
        {"role": "user", "content": user_message(*example)},
        {"role": "assistant", "content": json.dumps(replies, ensure_ascii=False)},
        {"role": "user", "content": user_message(text, emotion, intensity)},
    ]


def parse(text):
    start, end = text.find("["), text.rfind("]")
    try:
        replies = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(replies, list) or not all(isinstance(r, str) and r.strip() for r in replies):
        return None
    return [r.strip() for r in replies] or None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default="HiTZ/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    combos = {}
    with open(args.clips, encoding="utf-8") as stream:
        for line in stream:
            clip = json.loads(line)
            combos.setdefault(key(clip), (clip["text"], clip["emotion"], clip["intensity"]))
    if args.dry_run:
        print(len(combos), "combinations")
        print(json.dumps(messages(*next(iter(combos.values()))), ensure_ascii=False, indent=1))
        return
    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as stream:
            done = {json.loads(line)["key"] for line in stream}
    todo = [k for k in combos if k not in done]
    logger.info("%d combinations to generate, %d already done", len(todo), len(done))
    if not todo:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.max_tokens, skip_special_tokens=True, seed=0)
    began = time.time()
    outputs = llm.chat([messages(*combos[k]) for k in todo], sampling_params=params, use_tqdm=False)
    failed = 0
    with open(args.output, "a", encoding="utf-8") as stream:
        for k, output in zip(todo, outputs):
            result = output.outputs[0]
            replies = parse(result.text)
            row = {"key": k, "replies": replies or [], "finish_reason": result.finish_reason}
            if replies is None:
                failed += 1
                row["raw"] = result.text
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("%d generated, %d unparsable, %.0f s", len(todo), failed, time.time() - began)


if __name__ == "__main__":
    main()
