"""Translate prepare_st_en_sources.py's Basque texts into English (or Spanish, --target es) with Latxa-70B (vLLM).

Greedy, one text per request, with two fixed examples (a written sentence and a spoken
fragment). Resumable: ids already in the output are skipped. --shard/--num-shards split
the input for a SLURM array.

    python dataset_generation/translate_eu_en.py --input sources.jsonl --output en.0.jsonl \\
        --shard 0 --num-shards 4 --tensor-parallel-size 2
    python dataset_generation/translate_eu_en.py --input sources.jsonl --dry-run   # prompts only
"""

import argparse
import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LANGUAGES = {"en": "English", "es": "Spanish"}
SYSTEM = (
    "You are a professional Basque-to-{language} translator. The user sends one Basque text: a written "
    "sentence, or the transcript of a spoken utterance, possibly a fragment without capitals or "
    "punctuation. Translate it into natural, fluent {language} that keeps the full meaning and register. "
    "If it is a fragment, translate it as a fragment. Reply with the {language} translation only, with no "
    "notes, quotes or explanations."
)
EXAMPLES = {
    "en": [
        ("Gaur goizean bilera luzea izan dugu udaletxean, eta datorren astean berriro elkartuko gara.",
         "We had a long meeting at the town hall this morning, and we will meet again next week."),
        ("eta horregatik uste dugu proposamen hau ez dela nahikoa gure herriarentzat",
         "and that is why we believe this proposal is not enough for our country"),
    ],
    "es": [
        ("Gaur goizean bilera luzea izan dugu udaletxean, eta datorren astean berriro elkartuko gara.",
         "Esta mañana hemos tenido una reunión larga en el ayuntamiento, y volveremos a reunirnos la semana que viene."),
        ("eta horregatik uste dugu proposamen hau ez dela nahikoa gure herriarentzat",
         "y por eso creemos que esta propuesta no es suficiente para nuestro país"),
    ],
}


def messages(text, target="en"):
    turns = [{"role": "system", "content": SYSTEM.format(language=LANGUAGES[target])}]
    for basque, translation in EXAMPLES[target]:
        turns += [{"role": "user", "content": basque}, {"role": "assistant", "content": translation}]
    return turns + [{"role": "user", "content": text}]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--target", choices=sorted(LANGUAGES), default="en")
    parser.add_argument("--model", default="HiTZ/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=4000, help="texts per vLLM call (and per write)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    rows = rows[args.shard::args.num_shards][:args.limit]
    if args.dry_run:
        print(json.dumps(messages(rows[0]["eu"], args.target), ensure_ascii=False, indent=1))
        return
    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as stream:
            done = {json.loads(line)["id"] for line in stream}
    rows = [row for row in rows if row["id"] not in done]
    logger.info("shard %d/%d: %d to translate, %d already done", args.shard, args.num_shards, len(rows), len(done))
    if not rows:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, enable_prefix_caching=True)
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True)
    with open(args.output, "a", encoding="utf-8") as stream:
        for start in range(0, len(rows), args.chunk):
            chunk, began = rows[start:start + args.chunk], time.time()
            outputs = llm.chat([messages(row["eu"], args.target) for row in chunk], sampling_params=params,
                               use_tqdm=False)
            for row, output in zip(chunk, outputs):
                result = output.outputs[0]
                stream.write(json.dumps({"id": row["id"], args.target: result.text.strip(),
                                         "finish_reason": result.finish_reason}, ensure_ascii=False) + "\n")
            stream.flush()
            logger.info("%d/%d translated (%.1f texts/s)", start + len(chunk), len(rows),
                        len(chunk) / (time.time() - began))


if __name__ == "__main__":
    main()
