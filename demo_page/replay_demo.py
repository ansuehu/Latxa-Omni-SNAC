"""Replay spoken questions through the Gradio demo's Assistant, keeping multi-turn history.

    python demo_page/replay_demo.py --model-path <stage2 checkpoint> \
        --questions demo_page/questions_snac_demo.json --answer-file out/answers.jsonl

Questions that share a "conversation" value are asked in order in one chat, so a
follow-up sees the earlier turns, as in the demo. Writes answers.jsonl in infer.py's
format (wav/<id>.wav next to it), ready for demo_page/build.py.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE
from omni_speech.serve.gradio_demo import Assistant


def main(args):
    assistant = Assistant(args)
    questions = json.load(open(args.questions, encoding="utf-8"))
    answer_file = Path(args.answer_file)
    wav_dir = answer_file.parent / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    with open(answer_file, "w", encoding="utf-8") as out:
        for _, chat in itertools.groupby(questions, key=lambda q: q.get("conversation", q["id"])):
            features, answers, user_turns = [], [], []
            for q in chat:
                features.append(assistant.encode(q["speech"]))
                user_turns.append({"instruction": "", "text": ""})
                pieces, text = [], ""
                for text, audio in assistant.stream(features, answers, user_turns, args.temperature, args.top_p,
                                                    args.talker_temperature):
                    pieces.append(audio)
                answers.append(text)
                wav_path = wav_dir / f"{q['id']}.wav"
                sf.write(wav_path, np.concatenate(pieces), SNAC_SAMPLE_RATE)
                out.write(json.dumps({"question_id": q["id"], "prediction": text,
                                      "prediction_wav": str(wav_path)}, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{q['id']}: {text}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--legacy-prompt", action="store_true")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_units", type=int, default=4096)
    parser.add_argument("--chunk-frames", type=int, default=4)
    parser.add_argument("--first-chunk-frames", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--talker_temperature", type=float, default=0.7)
    main(parser.parse_args())
