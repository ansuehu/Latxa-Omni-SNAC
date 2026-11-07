"""Stage 2 data for the multi-task thinker: chat rows with Stage 1 prompts, plus tts rows.

Takes a dump_snac_tokens.py dataset and
  * gives every user turn an empty ``instruction``, so Stage 2 prompts chat audio as bare
    ``<speech>`` like multi-task Stage 1 (without it, stage1.user_prompt falls back to the old
    USER_PROMPT, which the multi-task thinker never saw);
  * adds the tts rows of build_multitask.py (same seed and sizes, so the same texts and
    instructions): a written "read this aloud" turn, answered by the same text, whose SNAC
    tokens are the ones already dumped for that answer.
train and dev get tts rows (dev is reshuffled so --dev_size keeps both tasks); the test
splits only get the instruction field.

    python dataset_generation/add_tts_rows.py --dataset $DS/latxa70b_answers_short_tts/dataset_snac \\
        --output-dataset $DS/latxa70b_answers_short_tts/dataset_snac_tts
"""

import argparse
import json
import logging
import os
import random
from collections import Counter

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from dataset_generation.build_multitask import DS, sample_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def with_instructions(row):
    return {"turns": [{**turn, "instruction": ""} for turn in row["turns"]]}


def stage2_tts_row(row, tts):
    """The tts row with the source row's spoken answer (audio, speaker, SNAC tokens)."""
    user, answer = tts["turns"]
    spoken = row["turns"][-1]
    return {
        "id": tts["id"],
        "turns": [
            {"audio": "", "role": "user", "speaker": "", "text": user["text"], "text_orig": "",
             "instruction": user["instruction"]},
            {**spoken, "instruction": ""},
        ],
        "answer_snac": row["answer_snac"],
        "answer_snac_len": row["answer_snac_len"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=f"{DS}/latxa70b_answers_short_tts/dataset_snac")
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--instructions", default=os.path.join(os.path.dirname(__file__), "multitask_instructions.json"))
    # Keep these equal to build_multitask.py's --tts-train, --dev-per-task and --seed.
    parser.add_argument("--tts-train", type=int, default=20000)
    parser.add_argument("--dev-per-task", type=int, default=300)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    with open(args.instructions, encoding="utf-8") as stream:
        instructions = json.load(stream)["tts"]
    source = load_from_disk(args.dataset)
    rng = random.Random(args.seed + 1)  # build_multitask.py's tts generator
    sizes = (("train", args.tts_train), ("dev", args.dev_per_task), ("test", args.dev_per_task))
    tts = sample_tts(source, sizes, instructions, rng)

    output = {}
    for name, split in source.items():
        split = split.map(with_instructions, desc=f"{name}: instruction field")
        if name in ("train", "dev"):
            extra = Dataset.from_list([stage2_tts_row(row, t) for row, t in tts[name]], features=split.features)
            split = concatenate_datasets([split, extra]).shuffle(seed=args.seed).flatten_indices()
        output[name] = split
        logger.info("%s: %d rows %s", name, len(split),
                    dict(Counter("tts" if i.startswith("tts_") else "chat" for i in split["id"])))
    DatasetDict(output).save_to_disk(args.output_dataset)
    logger.info("saved %s", args.output_dataset)


if __name__ == "__main__":
    main()
