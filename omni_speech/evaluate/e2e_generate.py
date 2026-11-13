"""End-to-end test generation: spoken question in, text + 24 kHz speech out, with latency.

    python omni_speech/evaluate/e2e_generate.py --model-path <stage2 checkpoint> \
        --out-dir <dir> [--num-shards 4 --shard 0]

Runs every user turn of the multi-task v2 `test` split (2,720 spoken chats) through the
streaming demo path (gradio_demo.Assistant), so latency is what a user would wait. A chat's
later turns see the model's own earlier answers, as in the demo. Writes
<out-dir>/generations.shardXX.jsonl (one line per answer) and <out-dir>/wav/<id>-<turn>.wav.
Resumable: conversations already in the shard file are skipped. Score with e2e_score.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk

from omni_speech.evaluate.merge_stage1_with_base import interpolate_model
from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE
from omni_speech.serve.gradio_demo import Assistant


def timed_answer(assistant, features, answers, user_turns, args):
    """Stream one answer; returns (text, audio, first text s, first audio s, total s)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    first_text = first_audio = None
    pieces, text = [], ""
    for text, audio in assistant.stream(features, answers, user_turns, args.temperature, args.top_p,
                                        args.talker_temperature):
        now = time.perf_counter() - start
        if text and first_text is None:
            first_text = now
        if len(audio):
            if first_audio is None:
                first_audio = now
            pieces.append(audio)
    total = time.perf_counter() - start
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return text, audio, first_text, first_audio, total


def main(args):
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".part{args.part}of{args.parts}{'r' if args.reverse else ''}" if args.parts > 1 or args.reverse else ""
    out_path = out_dir / f"generations.shard{args.shard:02d}{suffix}.jsonl"

    def finished():
        """Chats finished in any shard file, so a rerun with another --num-shards (or a helper) keeps them."""
        done = set()
        for path in out_dir.glob("generations.shard*.jsonl"):
            for line in path.open(encoding="utf-8"):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line of a killed (or still writing) run
                if row["last_turn"]:
                    done.add(row["id"])
        return done

    done = finished()

    rows = load_from_disk(args.dataset)[args.split]
    rows = rows.select(range(args.shard, len(rows), args.num_shards))
    if args.limit:
        rows = rows.select(range(min(args.limit, len(rows))))
    # A helper next to a running shard process: every --parts-th chat of the shard, from the end backwards, re-reading
    # what is finished before each chat, so it stops duplicating work once it meets the forward process.
    order = range(args.part, len(rows), args.parts)
    rows = rows.select(list(reversed(order)) if args.reverse else list(order))
    print(f"shard {args.shard}/{args.num_shards}: {len(rows)} chats, {len(done)} done", flush=True)

    assistant = Assistant(args)
    if args.merge_alpha is not None:  # WiSE-FT: the thinker's LLM toward its base; the talker is left as trained
        interpolate_model(assistant.model, args.merge_base, args.merge_alpha)
    # Warm-up (CUDA kernels, SNAC), so the first measured latency is not an outlier.
    first_user = next(t for t in rows[0]["turns"] if t["role"] == "user")
    timed_answer(assistant, [assistant.encode(first_user["audio"])], [],
                 [{"instruction": first_user["instruction"], "text": ""}], args)

    with out_path.open("a", encoding="utf-8") as out:
        for row in rows:
            if args.reverse:
                done = finished()
            if row["id"] in done:
                continue
            # A chat cut by a killed run is redone; e2e_score.py keeps the last copy of each turn.
            turns = row["turns"]
            user_indices = [i for i, t in enumerate(turns) if t["role"] == "user"]
            features, answers, user_turns, history = [], [], [], []
            for n, index in enumerate(user_indices):
                turn = turns[index]
                reference = turns[index + 1]["text"] if index + 1 < len(turns) else ""
                torch.cuda.synchronize()
                start = time.perf_counter()
                features.append(assistant.encode(turn["audio"]))
                torch.cuda.synchronize()
                encode_s = time.perf_counter() - start
                user_turns.append({"instruction": turn["instruction"], "text": ""})
                text, audio, first_text, first_audio, total = timed_answer(
                    assistant, features, answers, user_turns, args)
                answers.append(text)
                wav_path = wav_dir / f"{row['id']}-{n}.wav"
                sf.write(wav_path, audio, SNAC_SAMPLE_RATE)
                record = {
                    "id": row["id"], "turn": n, "last_turn": n == len(user_indices) - 1,
                    "question": turn["text"], "instruction": turn["instruction"],
                    "reference": reference, "history": history.copy(),
                    "prediction": text, "prediction_wav": str(wav_path),
                    "audio_s": round(len(audio) / SNAC_SAMPLE_RATE, 3),
                    "encode_s": round(encode_s, 4),
                    # Latency after the user stops speaking: encoder + first text / audio chunk.
                    "first_text_s": None if first_text is None else round(encode_s + first_text, 4),
                    "first_audio_s": None if first_audio is None else round(encode_s + first_audio, 4),
                    "total_s": round(encode_s + total, 3),
                }
                history.append({"question": turn["text"], "answer": text})
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            print(f"{row['id']}: {answers[-1][:100]}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", default="/scratch/andoni.sudupe/datasets/latxa_omni_multitask_v2_hf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, help="first N chats of the shard (smoke test)")
    parser.add_argument("--parts", type=int, default=1, help="split the shard into this many parts")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--reverse", action="store_true",
                        help="go through the part backwards, re-checking finished chats before each one (a helper)")
    parser.add_argument("--conv-mode", help="chat template; default from the checkpoint config")
    parser.add_argument("--legacy-prompt", action="store_true")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_units", type=int, default=4096)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--merge-alpha", type=float, help="merge the thinker's LLM weights toward --merge-base")
    parser.add_argument("--merge-base", default="/scratch/andoni.sudupe/models/Latxa-Qwen3.5-4B-text")
    parser.add_argument("--fast-talker", action="store_true", help="talker decoding with a CUDA graph per token")
    parser.add_argument("--first-chunk-frames", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--talker_temperature", type=float, default=0.7)
    main(parser.parse_args())
