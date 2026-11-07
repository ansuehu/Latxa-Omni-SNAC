"""Answer-only dev loss and sample generations for Latxa-Omni Stage 1 checkpoints.

Uses the same dev split, loss and generation prompts as the in-training
evaluation in omni_speech/train/stage1.py, so the numbers are comparable.
Multi-task dev sets also get eval_loss per task, and the generations are scored
per task (ASR WER, speech-translation BLEU/chrF; see train/dev_metrics.py).
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

from omni_speech.constants import IGNORE_INDEX
from omni_speech.evaluate.merge_stage1_with_base import interpolate_model, parse_spec
from omni_speech.model.builder import create_model, is_qwen3_5
from omni_speech.train.stage1 import (
    Stage1Dataset, collate_fn, generate_dev_answers, generation_indices, split_dev, summarize_generations,
)


@torch.no_grad()
def dev_loss(model, dataset, pad_token_id):
    """Token-weighted loss, plus the per-conversation mean the Trainer reports as eval_loss."""
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    per_example = []
    for index in tqdm(range(len(dataset)), desc="dev loss"):
        batch = collate_fn([dataset[index]], pad_token_id)
        batch = {key: value.to(device) for key, value in batch.items()}
        loss = model(**batch).loss.float().item()
        tokens = int(batch["labels"][:, 1:].ne(IGNORE_INDEX).sum())
        total_nll += loss * tokens
        total_tokens += tokens
        per_example.append(loss)
    token_loss = total_nll / total_tokens
    metrics = {
        "dev_loss": token_loss,
        "dev_perplexity": math.exp(token_loss),
        "eval_loss": sum(per_example) / len(per_example),
        "dev_tokens": total_tokens,
    }
    by_task = {}
    for index, loss in enumerate(per_example):
        by_task.setdefault(dataset.task(index), []).append(loss)
    if len(by_task) > 1:
        metrics.update({f"{task}_eval_loss": sum(losses) / len(losses) for task, losses in sorted(by_task.items())})
    return metrics


def main(args):
    dataset = load_from_disk(args.dataset)
    if args.split:
        # A whole held-out split, e.g. test_mt from build_multiturn_augment.py.
        dev_split = dataset[args.split]
    else:
        dev_split, _ = split_dev(dataset["test"], args.dev_size, args.dev_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.checkpoints:
        checkpoint, alpha = parse_spec(spec)
        checkpoint = Path(checkpoint)
        name = f"{checkpoint.parent.name}_{checkpoint.name}" + ("" if alpha is None else f"_alpha{alpha:g}")
        tokenizer, model, _ = create_model(str(checkpoint), None)
        model = model.cuda().eval()
        if alpha is not None:
            interpolate_model(model, args.base, alpha)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dev = Stage1Dataset(
            dev_split, tokenizer, model.config, args.input_type,
            args.mel_size, None, args.conv_mode or ("qwen3_5" if is_qwen3_5(model.config) else "llama_3"),
        )
        metrics = {"checkpoint": str(checkpoint), "alpha": alpha, **dev_loss(model, dev, tokenizer.pad_token_id)}
        generations = generate_dev_answers(
            model, dev, generation_indices([dev.task(i) for i in range(len(dev))], args.num_generation_samples),
            tokenizer, args.max_new_tokens, args.generation_batch_size,
        )
        by_task, scores = summarize_generations(generations)
        metrics.update(scores, generated={task: len(rows) for task, rows in by_task.items()})
        print(json.dumps(metrics), flush=True)
        with open(output_dir / "summary.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        with open(output_dir / f"{name}.generations.jsonl", "w", encoding="utf-8") as stream:
            for row in generations:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+",
                        help="checkpoint directories; <checkpoint>@<alpha> merges the LLM toward --base")
    parser.add_argument("--base", default="HiTZ/Latxa-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="saves/latxa_omni_stage1_omnivoice_hf")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", help="evaluate this whole split instead of the dev subset of test")
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--dev_seed", type=int, default=3407)
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--input_type", default="mel", choices=("mel", "raw"))
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--num_generation_samples", type=int, default=32, help="dev rows generated per task")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_batch_size", type=int, default=1, help="rows generated together")
    main(parser.parse_args())
