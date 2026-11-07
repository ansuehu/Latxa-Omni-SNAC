"""Expand the seed instructions of multitask_instructions.json with Latxa-70B (vLLM).

For each task, Latxa-70B is shown the seeds and a description of the task and writes new
Basque paraphrases. It is sampled several times; the lines are cleaned, deduplicated against
the seeds and each other, and up to --per-task (--asr-count for asr) are kept. The output keeps
the seeds first, so it can replace the seed file once a native speaker has reviewed it.

    python dataset_generation/expand_instructions.py --output instructions_expanded.json
"""

import argparse
import json
import os
import random
import re

DESCRIPTIONS = {
    "chat": "the user sends a spoken question as audio and asks the assistant to listen to it and answer it",
    "asr": "the user asks the assistant to transcribe the Basque audio word for word",
    "st_en": "the user asks the assistant to translate the spoken Basque audio into English",
    "st_es": "the user asks the assistant to translate the spoken Basque audio into Spanish",
    "st_en_text": "the user asks the assistant to translate the written Basque text that follows into English",
    "st_es_text": "the user asks the assistant to translate the written Basque text that follows into Spanish",
    "tts": "the user asks the assistant to read the written text that follows aloud, exactly as written",
}
PROMPT = """These are instructions that a user gives a voice assistant, in Basque. In all of them, {description}:

{seeds}

Write {n} new instructions in Basque with the same meaning. Make them varied: short and long, formal and \
informal (but never hika), polite requests, direct orders and questions, with different verbs and word \
order. They must be correct, natural Basque that a native speaker would say. Do not repeat the examples \
and do not add explanations. Write one instruction per line, numbered 1 to {n}."""
NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def candidates(text):
    for line in text.splitlines():
        match = NUMBERED.match(line)
        if match:
            instruction = match.group(1).strip().strip('"«»“”').strip()
            if 2 <= len(instruction.split()) <= 20:
                yield instruction


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default=os.path.join(os.path.dirname(__file__), "multitask_instructions.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="HiTZ/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--per-task", type=int, default=40)
    parser.add_argument("--asr-count", type=int, default=150)
    parser.add_argument("--samples", type=int, default=12, help="generations per task")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    with open(args.seeds, encoding="utf-8") as stream:
        seeds = json.load(stream)
    tasks = [task for task in seeds if task in DESCRIPTIONS]
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tensor_parallel_size, max_model_len=4096)
    params = SamplingParams(n=args.samples, temperature=0.9, top_p=0.95, max_tokens=1500, seed=args.seed)
    prompts = [[{"role": "user", "content": PROMPT.format(description=DESCRIPTIONS[task], n=30,
                                                         seeds="\n".join(f"- {s}" for s in seeds[task]))}]
               for task in tasks]
    outputs = llm.chat(prompts, sampling_params=params, use_tqdm=False)
    rng = random.Random(args.seed)
    expanded = {"_note": seeds.get("_note", "")}
    for task, output in zip(tasks, outputs):
        seen = {s.lower().rstrip(".:?!") for s in seeds[task]}
        new = []
        for sample in output.outputs:
            for instruction in candidates(sample.text):
                key = instruction.lower().rstrip(".:?!")
                if key not in seen:
                    seen.add(key)
                    new.append(instruction)
        target = args.asr_count if task == "asr" else args.per_task
        rng.shuffle(new)
        expanded[task] = seeds[task] + sorted(new[:max(target - len(seeds[task]), 0)])
        print(f"{task}: {len(new)} new candidates, kept {len(expanded[task])}")
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(expanded, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
