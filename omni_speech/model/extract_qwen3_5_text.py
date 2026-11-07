"""Save the text-only part of a Latxa-Qwen3.5 checkpoint as a Stage 1 base model.

HiTZ/Latxa-Qwen3.5-* are vision-language checkpoints (Qwen3_5ForConditionalGeneration). The
thinker needs only the language model: loading it as Qwen3_5ForCausalLM drops the vision tower and
the multi-token-prediction head. The model's generation config has no EOS, so <|im_end|> (end of
turn) and <|endoftext|> are set as EOS. Run in the omni-qwen env:

    python omni_speech/model/extract_qwen3_5_text.py HiTZ/Latxa-Qwen3.5-4B \
        models/Latxa-Qwen3.5-4B-text
"""

import argparse

import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    model = Qwen3_5ForCausalLM.from_pretrained(args.source, dtype=torch.bfloat16)
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    model.config.eos_token_id = eos
    model.generation_config = GenerationConfig(eos_token_id=eos, pad_token_id=tokenizer.pad_token_id)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved {sum(p.numel() for p in model.parameters()):,} parameters to {args.output_dir}; eos {eos}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output_dir")
    main(parser.parse_args())
