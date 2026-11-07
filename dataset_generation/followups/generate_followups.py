"""Two context-dependent spoken follow-ups per seed exchange, written by Latxa-70B (vLLM).

The follow-ups must not make sense on their own (ellipsis, pronouns, "eta ...?"), so the model can only answer
them from the conversation. Each request gets two follow-up kinds drawn from KINDS, for variety. Output, one line
per seed: {"id", "turns": [{"question", "answer", "kind"}, ...], "finish_reason"}; seeds whose reply is not
valid JSON are kept with "turns": [] and "raw". Resumable; --shard/--num-shards for a SLURM array.

    python dataset_generation/followups/generate_followups.py --input sources.train.jsonl --output followups.0.jsonl \\
        --shard 0 --num-shards 2
    python dataset_generation/followups/generate_followups.py --input sources.test.jsonl --dry-run
"""

import argparse
import json
import logging
import os
import random
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KINDS = {
    "why": "asks why or how about something in the last answer (\"Eta zergatik?\", \"Nola, ba?\")",
    "detail": "asks for a detail of something just mentioned, with a pronoun or ellipsis (\"Eta zenbat kostatzen da?\")",
    "example": "asks for an example of what was just said (\"Adibide bat eman diezadakezu?\")",
    "other_entity": "asks the same question about another thing or person (\"Eta Frantzian?\", \"Eta bere anaiak?\")",
    "clarify": "says the answer was not what they meant and narrows the question (\"Ez, nik ... esan nahi nuen\")",
    "simpler": "asks to say it shorter or more simply (\"Laburrago, mesedez\", \"Ez dut ulertu, errazago?\")",
    "confirm": "checks their own understanding with a yes/no question (\"Orduan, egunero egin behar da?\")",
    "opinion": "asks the assistant what it would choose or recommend among the options it gave (\"Zuk zein aukeratuko zenuke?\")",
}
SYSTEM = (
    "You write training data for a Basque voice assistant. You get the first exchange of a spoken conversation "
    "(the user's question and the assistant's answer, both in Basque) and two kinds of follow-up. Write the next "
    "two exchanges: each a short user follow-up question, as people really speak Basque, and the assistant's "
    "answer.\n"
    "Rules:\n"
    "- Each follow-up must NOT make sense on its own: it refers to the conversation with pronouns, ellipsis or "
    "phrases like \"eta...?\", \"hori\", \"bera\". Never repeat the full topic.\n"
    "- The second follow-up continues from the first follow-up's answer.\n"
    "- Questions: 2-15 words, spoken style, no lists.\n"
    "- Answers: correct, consistent with the earlier answer, 1-3 sentences, plain text that sounds natural "
    "read aloud (no lists, markdown or emojis).\n"
    "- Everything in standard Basque (batua).\n"
    "Reply with JSON only: [{\"question\": \"...\", \"answer\": \"...\"}, {\"question\": \"...\", \"answer\": \"...\"}]"
)
EXAMPLE = (
    {"question": "Zein da Euskal Herriko mendirik altuena?",
     "answer": "Euskal Herriko mendirik altuena Hiru Erregeen Mahaia da, Nafarroako Pirinioetan, 2.444 metrorekin.",
     "kinds": ("detail", "other_entity")},
    [{"question": "Eta zenbat denbora behar da gailurrera igotzeko?",
      "answer": "Belaguatik abiatuta, lau edo bost ordu inguru behar dira gailurrera iristeko, eguraldiaren eta "
                "sasoiaren arabera."},
     {"question": "Eta Gipuzkoan?",
      "answer": "Gipuzkoan, Aitzgorri inguruko gailurrak dira altuenak; Aitzuri, 1.551 metrorekin, da garaiena."}],
)


def user_message(seed, kinds):
    return (f"User: {seed['question']}\nAssistant: {seed['answer']}\n\n"
            f"First follow-up kind: {KINDS[kinds[0]]}\nSecond follow-up kind: {KINDS[kinds[1]]}")


def messages(seed, kinds):
    example, reply = EXAMPLE
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_message(example, example["kinds"])},
        {"role": "assistant", "content": json.dumps(reply, ensure_ascii=False)},
        {"role": "user", "content": user_message(seed, kinds)},
    ]


def parse(text):
    """The two exchanges, or None if the reply is not the requested JSON."""
    start, end = text.find("["), text.rfind("]")
    try:
        turns = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(turns, list) or len(turns) != 2:
        return None
    for turn in turns:
        if not isinstance(turn, dict) or not all(isinstance(turn.get(k), str) and turn[k].strip()
                                                 for k in ("question", "answer")):
            return None
    return [{"question": t["question"].strip(), "answer": t["answer"].strip()} for t in turns]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model", default="HiTZ/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chunk", type=int, default=2000, help="seeds per vLLM call (and per write)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as stream:
        seeds = [json.loads(line) for line in stream]
    # Kinds are drawn per seed id, so they do not depend on sharding or resuming.
    kinds = {s["id"]: tuple(random.Random(f"{args.seed}:{s['id']}").sample(sorted(KINDS), 2)) for s in seeds}
    seeds = seeds[args.shard::args.num_shards][:args.limit]
    if args.dry_run:
        print(json.dumps(messages(seeds[0], kinds[seeds[0]["id"]]), ensure_ascii=False, indent=1))
        return
    done = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as stream:
            done = {json.loads(line)["id"] for line in stream}
    seeds = [s for s in seeds if s["id"] not in done]
    logger.info("shard %d/%d: %d to generate, %d already done", args.shard, args.num_shards, len(seeds), len(done))
    if not seeds:
        return

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size,
              max_model_len=args.max_model_len, enable_prefix_caching=True)
    params = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens,
                            skip_special_tokens=True, seed=args.seed)
    failed = 0
    with open(args.output, "a", encoding="utf-8") as stream:
        for start in range(0, len(seeds), args.chunk):
            chunk, began = seeds[start:start + args.chunk], time.time()
            outputs = llm.chat([messages(s, kinds[s["id"]]) for s in chunk], sampling_params=params, use_tqdm=False)
            for seed, output in zip(chunk, outputs):
                result = output.outputs[0]
                turns = parse(result.text)
                row = {"id": seed["id"], "turns": [], "finish_reason": result.finish_reason}
                if turns is None:
                    failed += 1
                    row["raw"] = result.text
                else:
                    row["turns"] = [{**turn, "kind": kind} for turn, kind in zip(turns, kinds[seed["id"]])]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            logger.info("%d/%d generated, %d unparsable (%.1f seeds/s)", start + len(chunk), len(seeds), failed,
                        len(chunk) / (time.time() - began))


if __name__ == "__main__":
    main()
