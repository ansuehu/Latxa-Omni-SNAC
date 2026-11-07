"""Text-only Basque benchmarks to measure what Stage 1 makes Latxa forget.

Loads the LLM part of a Latxa-Omni checkpoint (or base Latxa) as a plain
LlamaForCausalLM; speech weights are ignored. A Latxa-Qwen3.5 model loads as the
text-only Qwen3_5ForCausalLM (needs transformers >= 5.2, the omni-qwen env); its
chat prompts use the qwen3_5 training template (thinking off).
Multiple-choice tasks are scored
zero-shot by the log-likelihood of each choice, as lm-eval-harness does, so
compare models with each other rather than with published few-shot numbers.
A few chat prompts are also answered greedily to inspect style drift.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM, PretrainedConfig

from omni_speech.conversation import conv_templates
from omni_speech.evaluate.merge_stage1_with_base import interpolate_model, parse_spec

CHAT_INSTRUCTION = "Erantzun galderari aukeraren letrarekin soilik."

CHAT_PROMPTS = [
    "Zein da Euskal Herriko mendirik altuena?",
    "Azaldu labur fotosintesia zer den.",
    "Idatzi lau lerroko olerki bat itsasoari buruz.",
    "Zein dira hiru aholku ondo lo egiteko?",
    "Itzuli gaztelaniara: 'Gaur goizean euria egin du eta etxean geratu naiz.'",
    "Zenbat da 17 bider 23? Azaldu pausoz pauso.",
    "Laburbildu hiru esalditan Bigarren Mundu Gerraren arrazoiak.",
    "Idatzi Python funtzio bat zerrenda bateko zenbaki bikoitiak itzultzen dituena.",
]


def letter_prompt(question, candidates, passage=None):
    lines = [f"Pasartea: {passage}"] if passage else []
    lines.append(f"Galdera: {question}")
    lines += [f"{letter}: {candidate}" for letter, candidate in zip("ABCD", candidates)]
    lines.append("Erantzuna:")
    return "\n".join(lines)


def letter_choices(count):
    return [f" {letter}" for letter in "ABCD"[:count]]


def chat_example(context, choices, conv_mode="llama_3"):
    """The question as a user turn in the training chat template; the letter is the reply's first token.

    choice_logprobs adds BOS itself, so the template's own is dropped.
    """
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], f"{CHAT_INSTRUCTION}\n{context}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt().removeprefix("<|begin_of_text|>")
    return prompt, [choice.strip() for choice in choices]


def model_type(model_path):
    return PretrainedConfig.get_config_dict(model_path)[0]["model_type"]


def load_model(model_path):
    kind = model_type(model_path)
    if kind == "omni_speech_qwen3_5":
        # A Stage 1 checkpoint: only our class maps its weight names back (see omni_speech_qwen3_5.py).
        # Without a speech_encoder the Whisper encoder is not loaded; the projector is unused.
        from omni_speech.model.language_model.omni_speech_qwen3_5 import OmniSpeechQwen3_5ForCausalLM
        return OmniSpeechQwen3_5ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    if kind in ("qwen3_5", "qwen3_5_text"):
        from transformers import Qwen3_5ForCausalLM
        return Qwen3_5ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    return LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)


def load_task(name):
    """Return (context, choices, gold index) triples."""
    if name in ("eus_proficiency", "eus_trivia"):
        repo = "HiTZ/EusProficiency" if name == "eus_proficiency" else "HiTZ/EusTrivia"
        return [
            (letter_prompt(row["question"], row["candidates"]),
             letter_choices(len(row["candidates"])), row["answer"])
            for row in load_dataset(repo, split="test")
        ]
    if name == "eus_reading":
        return [
            (letter_prompt(row["question"], row["candidates"], row["context"]),
             letter_choices(len(row["candidates"])), row["answer"])
            for row in load_dataset("HiTZ/EusReading", split="test")
        ]
    if name == "eus_exams":
        # The four Basque configs of EusExams-v2, pooled into one score.
        return [
            (letter_prompt(row["question"], row["candidates"]),
             letter_choices(len(row["candidates"])), row["answer"])
            for config in ("eu_council", "eu_government", "eu_health", "eu_university")
            for row in load_dataset("HiTZ/EusExams-v2", config, split="test")
        ]
    raise ValueError(f"unknown task: {name}")


@torch.no_grad()
def choice_logprobs(model, tokenizer, context, choices):
    """Sum of log-probabilities of each choice's tokens after the context."""
    # Qwen has no BOS token.
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    context_ids = bos + tokenizer.encode(context, add_special_tokens=False)
    sequences, spans = [], []
    for choice in choices:
        choice_ids = tokenizer.encode(choice, add_special_tokens=False)
        sequences.append(context_ids + choice_ids)
        spans.append((len(context_ids), len(context_ids) + len(choice_ids)))
    width = max(len(ids) for ids in sequences)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids in sequences])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences])
    device = next(model.parameters()).device
    logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
    logprobs = F.log_softmax(logits.float(), dim=-1)
    scores = []
    for row, (start, end) in enumerate(spans):
        targets = input_ids[row, start:end].to(device)
        scores.append(logprobs[row, start - 1:end - 1].gather(1, targets[:, None]).sum().item())
    return scores


def evaluate_task(model, tokenizer, examples, desc):
    correct = 0
    for context, choices, gold in tqdm(examples, desc=desc):
        scores = choice_logprobs(model, tokenizer, context, choices)
        correct += int(max(range(len(scores)), key=scores.__getitem__) == gold)
    return correct / len(examples)


@torch.no_grad()
def chat_answers(model, tokenizer, max_new_tokens):
    device = next(model.parameters()).device
    # Latxa-Qwen3.5's generation config has no EOS; its turns end with the tokenizer's <|im_end|>.
    eos = model.generation_config.eos_token_id
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    eos = eos + [tokenizer.eos_token_id] * (tokenizer.eos_token_id not in eos)
    answers = []
    for prompt in CHAT_PROMPTS:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False,
        )["input_ids"].to(device)
        output = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos, pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
        answers.append({"prompt": prompt, "answer": answer})
    return answers


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {name: load_task(name) for name in args.tasks}
    if args.limit:
        tasks = {name: examples[:args.limit] for name, examples in tasks.items()}
    for spec in args.models:
        model_path, alpha = parse_spec(spec)
        name = "_".join(Path(model_path).parts[-2:]) + ("" if alpha is None else f"_alpha{alpha:g}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = load_model(model_path)
        model = model.cuda().eval()
        if alpha is not None:
            interpolate_model(model, args.base, alpha)
        model_tasks = tasks
        if args.chat:
            conv_mode = "llama_3" if isinstance(model, LlamaForCausalLM) else "qwen3_5"
            model_tasks = {
                task: [(*chat_example(context, choices, conv_mode), gold) for context, choices, gold in examples]
                for task, examples in tasks.items()
            }
        metrics = {"model": model_path, "alpha": alpha, "prompt": "chat" if args.chat else "plain"}
        for task, examples in model_tasks.items():
            metrics[task] = evaluate_task(model, tokenizer, examples, f"{name} {task}")
        metrics["mean"] = sum(metrics[task] for task in tasks) / len(tasks)
        print(json.dumps(metrics), flush=True)
        with open(output_dir / "summary.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        with open(output_dir / f"{name}.chat.jsonl", "w", encoding="utf-8") as stream:
            for row in chat_answers(model, tokenizer, args.max_new_tokens):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+",
                        help="checkpoint directories or Hub IDs; <checkpoint>@<alpha> merges toward --base")
    parser.add_argument("--base", default="HiTZ/Latxa-Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=["eus_proficiency", "eus_trivia", "eus_reading", "eus_exams"])
    parser.add_argument("--limit", type=int, help="first N examples per task, for smoke tests")
    parser.add_argument("--chat", action="store_true",
                        help="ask each question inside the training chat template (Stage 1 models only ever see that)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    main(parser.parse_args())
