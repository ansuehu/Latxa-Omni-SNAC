"""Answer the spoken-question conversations with Latxa itself (self-distilled targets).

The text transcript of every user turn is answered by Latxa-Llama-3.1-70B-Instruct
with a system prompt asking for a spoken register. Conversations are answered turn
by turn, so each answer sees Latxa's *own* earlier answers as history. The new
answers replace the translated InstructS2S answers as Stage 1/2 targets (see
IDEAL_SETUP.md); their audio must then be re-synthesized.

Three modes:

  generate (default): vLLM, one shard of every split per process; resumable JSONL
      {"split", "id", "answers": [...], "finish": [...]} with one entry per answered
      assistant turn. Shard with --num-shards/--shard-index (a SLURM array task each).
  --retry: after generate, re-answer every turn that hit --max-tokens (or came back
      empty) with SYSTEM_PROMPT_SHORT ("2-3 sentences"), keeping the earlier answers
      as history, then continue that conversation's remaining turns with the same
      prompt. Writes answers.retry.shard*.jsonl with the full answer lists.
  --shorten: after generate (and retry), rewrite every answer that is cut off, uses markdown or
      is longer than --short-words with SHORTEN_PROMPT (Latxa ignores length limits when
      answering, but follows them when condensing a given answer). Writes
      answers.short.shard*.jsonl with the full answer lists; turns after a cut-off stay unanswered.
  --build: merge every shard (a shortened conversation replaces its retry, which replaces its first attempt) into a DatasetDict with the same schema as the input.
      Assistant ``text`` becomes Latxa's answer (the original is kept in ``text_orig``)
      and ``audio`` is emptied. A conversation is cut before its first failed or
      rejected answer, and dropped if no complete user/assistant pair is left.

    # prompts only, no GPU
    python dataset_generation/generate_latxa_answers.py --dataset <in> --out-dir <dir> --dry-run --limit 3
    # one shard
    python dataset_generation/generate_latxa_answers.py --dataset <in> --out-dir <dir> \
        --model models/Latxa-Llama-3.1-70B-Instruct --tensor-parallel-size 4
    # cut-off answers again, shorter (same shard split as generate)
    python dataset_generation/generate_latxa_answers.py --dataset <in> --out-dir <dir> --retry
    # condense long / cut-off answers (same shard split as generate)
    python dataset_generation/generate_latxa_answers.py --dataset <in> --out-dir <dir> --shorten
    # merge
    python dataset_generation/generate_latxa_answers.py --dataset <in> --out-dir <dir> --build --output-dataset <out>
"""

import argparse
import glob
import json
import logging
import os
import re
import time

from datasets import Dataset, DatasetDict, load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ahots-laguntzaile adeitsu eta zehatza zara, eta zure erantzunak ozen irakurriko dira. "
    "Erantzun beti euskaraz, hizketa-erregistroan: esaldi oso eta naturalak, gehienez bost "
    "esaldi, zuzenean galderari erantzunez. Ez erabili markdown-ik, zerrendarik, izenbururik, "
    "taularik, kode-blokerik, URLrik edo emojirik. Galdera ulertzen ez baduzu, eskatu "
    "argibideak labur."
)
SYSTEM_PROMPT_SHORT = SYSTEM_PROMPT.replace("gehienez bost esaldi", "bi edo hiru esaldi labur")
assert SYSTEM_PROMPT_SHORT != SYSTEM_PROMPT
RETRY_FINISH = ("length", "empty")
SHORTEN_PROMPT = (
    "Hau ahots-laguntzaile baten erantzuna da, baina luzeegia da ozen irakurtzeko, eta erdian "
    "moztuta egon daiteke. Berridatzi euskaraz, gehienez hiru esaldi eta {words} hitz erabiliz: "
    "erantzun zuzenean galderari eta gorde ideia nagusiak. Ez gehitu informazio berririk eta ez "
    "erabili markdown-ik, zerrendarik edo izenbururik. Idatzi erantzun berria bakarrik.\n\n"
    "Galdera: {question}\n\nErantzuna: {answer}"
)
# Lines that read as written text rather than speech: list items, headings, tables, code, bold.
MARKDOWN = re.compile(r"(^\s*([-*•#>|]|\d+[.)])\s)|```|\*\*|https?://", re.MULTILINE)


def shard(split, num_shards, shard_index):
    return split.shard(num_shards, shard_index, contiguous=True)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by a killed job; that conversation is redone
                done.add((row["split"], row["id"]))
    return done


def messages_for(turns, answers, upto, system=SYSTEM_PROMPT):
    """Chat messages for answering user turn ``upto`` (0-based index into turns)."""
    messages = [{"role": "system", "content": system}]
    for j in range(0, upto, 2):
        messages.append({"role": "user", "content": turns[j]["text"]})
        messages.append({"role": "assistant", "content": answers[j // 2]})
    messages.append({"role": "user", "content": turns[upto]["text"]})
    return messages


def answer_chunk(llm, tokenizer, params, rows, args, prefixes=None, system=SYSTEM_PROMPT):
    """Answer every conversation in ``rows`` round by round (all first turns, then all second turns...).

    ``prefixes`` are (answers, finish) already accepted per row; answering resumes after them.
    """
    answers = [list(p[0]) for p in prefixes] if prefixes else [[] for _ in rows]
    finish = [list(p[1]) for p in prefixes] if prefixes else [[] for _ in rows]
    alive = list(range(len(rows)))
    for upto in range(0, max(len(r["turns"]) for r in rows), 2):
        batch, prompts, waiting = [], [], []
        for i in alive:
            turns = rows[i]["turns"]
            if upto < 2 * len(answers[i]):
                waiting.append(i)  # this turn is in the accepted prefix
                continue
            if upto >= len(turns) or turns[upto]["role"] != "user":
                continue
            messages = messages_for(turns, answers[i], upto, system)
            n_prompt = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            if n_prompt + args.max_tokens > args.max_model_len:
                finish[i].append("prompt_too_long")
                continue
            batch.append(i)
            prompts.append(messages)
        if not batch:
            if not waiting:
                break
            alive = waiting
            continue
        outputs = llm.chat(prompts, sampling_params=params, use_tqdm=False)
        alive = waiting
        for i, out in zip(batch, outputs):
            text = out.outputs[0].text.strip()
            reason = out.outputs[0].finish_reason if text else "empty"
            answers[i].append(text)
            finish[i].append(reason)
            if reason == "stop":
                alive.append(i)
    return answers, finish


def needs_shortening(answer, finish, max_words):
    return bool(answer) and (finish == "length" or (finish == "stop" and (
        MARKDOWN.search(answer) or len(answer.split()) > max_words)))


def shorten_messages(question, answer, words):
    return [{"role": "user", "content": SHORTEN_PROMPT.format(words=words, question=question, answer=answer)}]


def shorten_chunk(llm, params, rows, results, args):
    """Condense every answer of ``rows`` that needs_shortening; returns (answers, finish) per row."""
    out, prompts, slots = [], [], []
    for n, row in enumerate(rows):
        answers, finish = results[n]
        answers, finish = list(answers), list(finish[:len(answers)])  # drop a trailing prompt_too_long
        out.append((answers, finish))
        for k, (ans, fin) in enumerate(zip(answers, finish)):
            if needs_shortening(ans, fin, args.short_words):
                prompts.append(shorten_messages(row["turns"][2 * k]["text"], ans, args.short_words))
                slots.append((n, k))
    if prompts:
        for (n, k), o in zip(slots, llm.chat(prompts, sampling_params=params, use_tqdm=False)):
            text = o.outputs[0].text.strip()
            out[n][0][k] = text
            out[n][1][k] = o.outputs[0].finish_reason if text else "empty"
    return [a for a, _ in out], [f for _, f in out]


def read_results(paths):
    results = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                results[(row["split"], row["id"])] = row
    return results


def generate(args):
    from vllm import LLM, SamplingParams

    ds = load_from_disk(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"shard{args.shard_index:02d}-of-{args.num_shards:02d}.jsonl"
    first_path = os.path.join(args.out_dir, f"answers.{tag}")
    retry_path = os.path.join(args.out_dir, f"answers.retry.{tag}")
    out_path = (os.path.join(args.out_dir, f"answers.short.{tag}") if args.shorten
                else retry_path if args.retry else first_path)
    done = load_done(out_path)
    retry, previous = {}, {}
    if args.shorten:
        for key, row in read_results([first_path, retry_path]).items():
            if key not in done and any(needs_shortening(a, f, args.short_words)
                                       for a, f in zip(row["answers"], row["finish"])):
                previous[key] = (row["answers"], row["finish"])
    if args.retry:
        # Prefix = every answer before the first cut-off/empty one.
        for key, row in read_results([first_path]).items():
            if key in done or not row["finish"] or row["finish"][-1] not in RETRY_FINISH:
                continue
            retry[key] = (row["answers"][:-1], row["finish"][:-1])
    todo = []
    for name in args.splits or list(ds):
        split = shard(ds[name], args.num_shards, args.shard_index)
        if args.limit:
            split = split.select(range(min(args.limit, len(split))))
        if args.retry or args.shorten:
            wanted = {i for (n, i) in (previous if args.shorten else retry) if n == name}
            split = split.filter(lambda i: i in wanted, input_columns="id")
        todo += [(name, row) for row in split if (name, row["id"]) not in done]
    mode = " shorten" if args.shorten else " retry" if args.retry else ""
    logger.info(f"shard {args.shard_index}/{args.num_shards}{mode}: "
                f"{len(done)} done, {len(todo)} conversations to answer")
    if args.dry_run and args.shorten:
        for name, row in todo[:args.limit or 3]:
            answers, finish = previous[(name, row["id"])]
            for k, (ans, fin) in enumerate(zip(answers, finish)):
                if needs_shortening(ans, fin, args.short_words):
                    print(f"--- {name}/{row['id']} turn {k}")
                    print(shorten_messages(row["turns"][2 * k]["text"], ans, args.short_words)[0]["content"])
        return
    if args.dry_run:
        for name, row in todo[:args.limit or 3]:
            print(f"--- {name}/{row['id']}")
            print(json.dumps(messages_for(row["turns"], [t["text"] for t in row["turns"][1::2]], 0),
                     ensure_ascii=False, indent=1))
        return
    if not todo:
        return

    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    logger.info(f"model loaded in {time.time() - t0:.0f}s")

    with open(out_path, "a", encoding="utf-8") as stream:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start:start + args.chunk_size]
            t0 = time.time()
            if args.shorten:
                answers, finish = shorten_chunk(
                    llm, params, [row for _, row in chunk],
                    [previous[(name, row["id"])] for name, row in chunk], args,
                )
            elif args.retry:
                answers, finish = answer_chunk(
                    llm, tokenizer, params, [row for _, row in chunk], args,
                    prefixes=[retry[(name, row["id"])] for name, row in chunk], system=SYSTEM_PROMPT_SHORT,
                )
            else:
                answers, finish = answer_chunk(llm, tokenizer, params, [row for _, row in chunk], args)
            for (name, row), ans, fin in zip(chunk, answers, finish):
                stream.write(json.dumps({"split": name, "id": row["id"], "answers": ans, "finish": fin},
                                        ensure_ascii=False) + "\n")
            stream.flush()
            logger.info(f"{start + len(chunk)}/{len(todo)} conversations "
                        f"({len(chunk) / (time.time() - t0):.1f} conv/s)")


def rejected(answer, finish, max_words):
    return finish != "stop" or MARKDOWN.search(answer) or len(answer.split()) > max_words


def build(args):
    ds = load_from_disk(args.dataset)
    results = read_results(sorted(glob.glob(os.path.join(args.out_dir, "answers.shard*.jsonl"))))
    retried = read_results(sorted(glob.glob(os.path.join(args.out_dir, "answers.retry.shard*.jsonl"))))
    shortened = read_results(sorted(glob.glob(os.path.join(args.out_dir, "answers.short.shard*.jsonl"))))
    logger.info(f"{len(results)} conversations answered, {len(retried)} of them retried, "
                f"{len(shortened)} shortened")
    results.update(retried)
    results.update(shortened)
    out = {}
    for name in ds:
        rows, stats = [], {"missing": 0, "dropped": 0, "cut": 0, "kept": 0}
        for row in ds[name]:
            result = results.get((name, row["id"]))
            if result is None:
                stats["missing"] += 1
                continue
            turns = []
            for k, (ans, fin) in enumerate(zip(result["answers"], result["finish"])):
                if rejected(ans, fin, args.max_words):
                    break
                user, assistant = row["turns"][2 * k], row["turns"][2 * k + 1]
                turns += [user, dict(assistant, text=ans, text_orig=assistant["text"], audio="")]
            if not turns:
                stats["dropped"] += 1
                continue
            stats["cut" if len(turns) < len(row["turns"]) else "kept"] += 1
            rows.append(dict(row, turns=turns))
        logger.info(f"{name}: {stats}")
        if rows:
            out[name] = Dataset.from_list(rows)
        else:
            logger.warning(f"{name}: no answered conversations, split left out")
    DatasetDict(out).save_to_disk(args.output_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="DatasetDict with id + turns (role/text/audio/speaker)")
    parser.add_argument("--out-dir", required=True, help="where the shard JSONL files go")
    parser.add_argument("--splits", nargs="*", help="default: every split")
    parser.add_argument("--model", default="models/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--quantization", choices=("fp8",), help="fp8 halves memory (TP=2 fits a 70B on H100)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="0.92 hit CUDA OOM spikes with fp8 on 2 H100s")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=320, help="longer answers count as failed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--chunk-size", type=int, default=4096, help="conversations per vLLM round trip / write")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="first N conversations of each split (tests)")
    parser.add_argument("--dry-run", action="store_true", help="print prompts, no model")
    parser.add_argument("--retry", action="store_true", help="re-answer cut-off turns with the 2-3 sentence prompt")
    parser.add_argument("--shorten", action="store_true",
                        help="condense cut-off, markdown or longer than --short-words answers")
    parser.add_argument("--short-words", type=int, default=60,
                        help="--shorten: rewrite answers longer than this (~20 s of speech) to this length")
    parser.add_argument("--build", action="store_true", help="merge shard outputs into a dataset")
    parser.add_argument("--output-dataset", help="--build: where to save the new DatasetDict")
    parser.add_argument("--max-words", type=int, default=90, help="--build: reject longer answers (~35 s of speech)")
    args = parser.parse_args()
    if args.build:
        if not args.output_dataset:
            parser.error("--build needs --output-dataset")
        build(args)
    else:
        generate(args)
