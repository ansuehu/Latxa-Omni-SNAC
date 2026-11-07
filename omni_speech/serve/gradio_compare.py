"""Side-by-side Gradio demo: one spoken question, several models answering in parallel.

    python omni_speech/serve/gradio_compare.py --model unmerged=<stage2 ckpt> --model merged=<stage2 ckpt>

Each model sits on its own GPU (cuda:0, cuda:1, ...) and generates in its own
thread, so the answers stream back at the same time, each as text and 24 kHz
speech in its own column. Every model keeps its own conversation history.
"""

import argparse
import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE
from omni_speech.serve.gradio_demo import Assistant, patch_gradio_schema_bug, to_pcm16, wav_stream_header


def run_stream(index, assistant, features, answers, settings, events):
    """Worker thread: push (index, text, audio) updates, then (index, None, error or None) when done."""
    try:
        with torch.cuda.device(assistant.device):
            # The compared models predate the multi-task instructions: every turn is bare speech.
            for text, audio in assistant.stream(features, answers, [{"instruction": "", "text": ""}] * len(features),
                                                *settings):
                events.put((index, text, audio))
        events.put((index, None, None))
    except Exception as error:  # surface the error in that model's column instead of hanging the UI
        events.put((index, None, error))


def build_demo(names, assistants, log_dir=None):
    """log_dir: if set, every turn's question and answers (text + wav) are kept there, indexed in log.jsonl."""
    import gradio as gr

    patch_gradio_schema_bug()
    count = len(assistants)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def respond(audio_path, *state):
        chats, turns, settings = list(state[:count]), list(state[count:2 * count]), state[2 * count:]
        if not audio_path:
            yield (*chats, *[b""] * count, *turns, None)
            return
        features, answers = [], []
        for i, assistant in enumerate(assistants):
            with torch.cuda.device(assistant.device):
                features.append([t["features"] for t in turns[i]] + [assistant.encode(audio_path)])
            answers.append([t["text"] for t in turns[i]])
            chats[i] = chats[i] + [{"role": "user", "content": {"path": audio_path}},
                                   {"role": "assistant", "content": ""}]
        yield (*chats, *[wav_stream_header()] * count, *turns, None)

        events = queue.Queue()
        for i, assistant in enumerate(assistants):
            threading.Thread(target=run_stream, args=(i, assistant, features[i], answers[i], settings, events),
                             daemon=True).start()
        pieces, texts, running = [[] for _ in assistants], [""] * count, count
        while running:
            index, text, audio = events.get()
            chunks = [b""] * count
            if text is None:
                running -= 1
                if audio is not None:
                    chats[index][-1]["content"] = f"{texts[index]}\n\n[error: {audio}]"
            else:
                texts[index] = text
                chats[index][-1]["content"] = text
                pieces[index].append(audio)
                chunks[index] = to_pcm16(audio)
            yield (*chats, *chunks, *turns, None)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        record = {"time": stamp, "turn": len(turns[0]) + 1, "question": audio_path,
                  "settings": dict(zip(["temperature", "top_p", "talker_temperature"], settings)), "answers": {}}
        if log_dir:
            record["question"] = str(log_dir / f"{stamp}_question{Path(audio_path).suffix}")
            shutil.copy(audio_path, record["question"])
        for i in range(count):
            wav = None
            if pieces[i]:
                wav = (str(log_dir / f"{stamp}_{names[i]}.wav") if log_dir
                       else tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
                sf.write(wav, np.concatenate(pieces[i]), SNAC_SAMPLE_RATE)
                chats[i].append({"role": "assistant", "content": {"path": wav}})
            turns[i] = turns[i] + [{"features": features[i][-1], "text": texts[i]}]
            record["answers"][names[i]] = {"text": texts[i], "audio": wav}
        if log_dir:
            with open(log_dir / "log.jsonl", "a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
        yield (*chats, *[b""] * count, *turns, None)

    with gr.Blocks(title="Latxa-Omni-SNAC compare") as demo:
        gr.Markdown("# Latxa-Omni-SNAC: konparaketa\nGrabatu galdera; eredu guztiek batera erantzungo dute. "
                    "/ Record a question; every model answers at the same time.")
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Zure galdera")
        with gr.Row():
            send = gr.Button("Bidali", variant="primary")
            clear = gr.Button("Garbitu")
        chats, audio_outs, turns = [], [], []
        with gr.Row():
            for name in names:
                with gr.Column():
                    chats.append(gr.Chatbot(type="messages", height=480, label=name))
                    audio_outs.append(gr.Audio(streaming=True, autoplay=True, label=f"{name}: erantzuna",
                                               interactive=False))
                    turns.append(gr.State([]))
        with gr.Accordion("Ezarpenak", open=False):
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="LLM temperature (0 = greedy)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="LLM top-p")
            talker_temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Talker temperature")

        inputs = [audio_in, *chats, *turns, temperature, top_p, talker_temperature]
        outputs = [*chats, *audio_outs, *turns, audio_in]
        audio_in.stop_recording(respond, inputs, outputs)
        send.click(respond, inputs, outputs)
        clear.click(lambda: ([[]] * count + [[]] * count + [None] + [None] * count), None,
                    [*chats, *turns, audio_in, *audio_outs])
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                        help="Stage 2 checkpoint to compare; repeat once per model (one GPU each)")
    parser.add_argument("--conv-mode", default="llama_3")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_units", type=int, default=4096, help="talker tokens per answer (~50 s)")
    parser.add_argument("--chunk-frames", type=int, default=4, help="SNAC frames (~85 ms each) per audio chunk")
    parser.add_argument("--first-chunk-frames", type=int, default=2)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--log-dir", help="save every question and answer here (log.jsonl + wavs)")
    args = parser.parse_args()
    args.legacy_prompt = True  # see gradio_demo.py --legacy-prompt

    names, paths = zip(*(spec.split("=", 1) for spec in args.model))
    if len(paths) > torch.cuda.device_count():
        parser.error(f"{len(paths)} models need {len(paths)} GPUs, found {torch.cuda.device_count()}")
    assistants = [Assistant(args, model_path=path, device=f"cuda:{i}") for i, path in enumerate(paths)]
    build_demo(names, assistants, args.log_dir).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
