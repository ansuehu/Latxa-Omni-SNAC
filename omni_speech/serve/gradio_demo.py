"""Streaming Gradio demo for Latxa-Omni-SNAC: speak, and the answer streams back as text and 24 kHz speech.

    python omni_speech/serve/gradio_demo.py --model-path <stage2 checkpoint> [--share]

One process holds the whole model on one GPU. Recording stops -> the answer
starts: the LLM writes text token by token, the talker turns every 3 text
tokens into 4 SNAC frames, and finished frames are decoded and streamed to
the browser as raw PCM, so speech starts after a few text tokens instead of
after the whole answer. Earlier turns stay in the prompt (multi-turn).

Each turn can carry an instruction (e.g. "Transkribatu audioa." or "Itzuli ingelesera."),
prompted as in multi-task Stage 1; leave it empty to just talk. A turn can also be typed
instead of recorded (e.g. "Irakurri ozen testu hau." + a text, for TTS). "Text only" skips the
talker (the talker is trained on chat answers only). A Stage 1 checkpoint has no talker,
so it always answers in text. --legacy-prompt uses the old fixed prompt for empty
instructions, for models trained before the multi-task data.

"Transkribatu lehenik" (--transcribe-first sets its default) first transcribes a spoken turn
with the same model, then answers the audio with its transcript above it, and keeps the turn in
the history as text. Stage 1 models do not follow spoken follow-up questions that depend on
earlier turns (they repeat their previous answer), even with a text history; with the
transcript next to the audio they do.

Measure latency and the real-time factor without the UI:

    python omni_speech/serve/gradio_demo.py --model-path <ckpt> --benchmark q1.wav q2.wav
"""

import argparse
import json
import os
import struct
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper

from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.model.builder import QWEN3_5_MODEL_TYPES, load_pretrained_model
from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE, SnacCodec, SnacStreamDecoder
from omni_speech.train.stage1 import USER_PROMPT, user_prompt

TRANSCRIBE = "Transkribatu audioa."


def wav_stream_header(sample_rate=SNAC_SAMPLE_RATE):
    """16-bit mono WAV header with unknown (0xFFFFFFFF) sizes, for a growing stream."""
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


def to_pcm16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def to_wav(audio, sample_rate=SNAC_SAMPLE_RATE):
    """A complete 16-bit mono WAV file of these samples."""
    data = to_pcm16(audio)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


class Assistant:
    def __init__(self, args, model_path=None, device="cuda"):
        self.args = args
        model_path = model_path or args.model_path
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as stream:
            config = json.load(stream)
        # Stage 1 checkpoints have no talker (see stage2.py).
        self.has_talker = config.get("speech_generator_type") == "snac_ar"
        self.conv_mode = args.conv_mode or ("qwen3_5" if config["model_type"] in QWEN3_5_MODEL_TYPES else "llama_3")
        self.tokenizer, self.model, _ = load_pretrained_model(model_path, None, s2s=self.has_talker,
                                                              device=device)
        self.model.eval()
        self.device = self.model.device
        self.codec = SnacCodec(device=self.device)

    @torch.no_grad()
    def encode(self, path):
        """Audio file -> projected speech features (on CPU, so they can live in session state)."""
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(audio.mean(axis=1))
        if rate != 16000:
            audio = torchaudio.functional.resample(audio, rate, 16000)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=self.args.mel_size).permute(1, 0)
        mel = mel[None].to(self.device, self.model.dtype)
        lengths = torch.tensor([mel.shape[1]], device=self.device)
        return self.model.encode_speech(mel, lengths)[0].cpu()

    def user_prompt(self, turn):
        """turn: {"instruction", "text"}; a non-empty text is a typed turn, otherwise it is spoken."""
        instruction, text = turn["instruction"].strip(), turn.get("text", "").strip()
        if not instruction and not text and self.args.legacy_prompt:
            return USER_PROMPT
        return user_prompt({"instruction": instruction, "text": text, "audio": "" if text else "spoken"})

    def prompt_ids(self, answers, user_turns):
        """Prompt for len(answers) + 1 user turns, ending at the assistant header."""
        conv = conv_templates[self.conv_mode].copy()
        for answer, turn in zip(answers, user_turns):
            conv.append_message(conv.roles[0], self.user_prompt(turn))
            conv.append_message(conv.roles[1], answer)
        conv.append_message(conv.roles[0], self.user_prompt(user_turns[-1]))
        conv.append_message(conv.roles[1], None)
        return tokenizer_speech_token(conv.get_prompt(), self.tokenizer, return_tensors="pt")[None].to(self.device)

    def stream(self, features, answers, user_turns, temperature, top_p, talker_temperature, text_only=False):
        """Yield (answer text so far, new 24 kHz samples) while generating.

        features: one entry per spoken user turn, in order; user_turns: see user_prompt.
        """
        if text_only or not self.has_talker:
            yield from self.stream_text(features, answers, user_turns, temperature, top_p)
            return
        decoder = SnacStreamDecoder(self.codec, chunk_frames=self.args.chunk_frames,
                                    first_chunk_frames=self.args.first_chunk_frames)
        text_ids = []
        events = self.model.stream_generate(
            self.prompt_ids(answers, user_turns),
            speech_embeds=[f.to(self.device) for f in features],
            max_new_tokens=self.args.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            talker_kwargs={"temperature": talker_temperature, "max_tokens": self.args.max_units,
                           "fast": getattr(self.args, "fast_talker", False)},
        )
        for kind, ids in events:
            if kind == "text":
                text_ids += ids
                yield self.decode_text(text_ids), np.zeros(0, dtype=np.float32)
            else:
                yield self.decode_text(text_ids), decoder.push(ids)
        yield self.decode_text(text_ids), decoder.flush()

    @torch.no_grad()
    def stream_text(self, features, answers, user_turns, temperature, top_p):
        """The LLM's answer alone, streamed as text (no talker, no audio)."""
        from transformers import GenerationMixin, TextIteratorStreamer

        input_ids = self.prompt_ids(answers, user_turns)
        _, _, _, _, embeds, _ = self.model.prepare_inputs_labels_for_speech_and_text(
            input_ids, None, None, None, None, None, None,
            speech_embeds=[f.to(self.device) for f in features],
        )
        if embeds is None:
            embeds = self.model.get_model().embed_tokens(input_ids)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        sampling = {"do_sample": True, "temperature": temperature, "top_p": top_p} if temperature > 0 else \
            {"do_sample": False, "temperature": None, "top_p": None}
        # The plain generate (the model's own one expects speech inputs, not embeddings).
        thread = threading.Thread(target=GenerationMixin.generate, args=(self.model,), kwargs=dict(
            inputs_embeds=embeds, max_new_tokens=self.args.max_new_tokens, streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id, **sampling,
        ))
        thread.start()
        text = ""
        for piece in streamer:
            text += piece
            yield text.strip(), np.zeros(0, dtype=np.float32)
        thread.join()

    def transcribe(self, feature):
        text = ""
        for text, _ in self.stream_text([feature], [], [{"instruction": TRANSCRIBE, "text": ""}], 0.0, 1.0):
            pass
        return text

    def decode_text(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


def patch_gradio_schema_bug():
    """gradio_client 1.3 crashes on boolean JSON schemas (newer pydantic), which breaks launch()."""
    from gradio_client import utils

    original = utils._json_schema_to_python_type
    if getattr(original, "patched", False):
        return

    def patched(schema, defs=None):
        return "Any" if isinstance(schema, bool) else original(schema, defs)

    patched.patched = True
    utils._json_schema_to_python_type = patched


def build_demo(assistant):
    import gradio as gr

    patch_gradio_schema_bug()
    # Gradio 4 appends streamed bytes to one growing WAV; Gradio 5 (omni-qwen) re-encodes every chunk on its own as
    # an AAC segment, so each chunk must be a complete file. Short segments play with gaps (and each conversion holds
    # up the text), so under Gradio 5 audio is sent in segments of at least --segment-seconds, the first one shorter
    # (--first-segment-seconds) so the speech starts soon after the text.
    gradio_major = int(gr.__version__.split(".")[0])
    gradio5 = gradio_major >= 5
    segment = int(assistant.args.segment_seconds * SNAC_SAMPLE_RATE)
    first_segment = int(assistant.args.first_segment_seconds * SNAC_SAMPLE_RATE)

    def respond(audio_path, user_text, instruction, text_only, transcribe_first, chat, turns, temperature, top_p,
                talker_temperature):
        user_text = (user_text or "").strip()
        if not audio_path and not user_text:
            yield chat, None, turns, None, user_text
            return
        # A typed text wins over a recording left in the audio box.
        feature = None if user_text else assistant.encode(audio_path)
        transcript = assistant.transcribe(feature) if feature is not None and transcribe_first else None
        features = [t["features"] for t in turns if t["features"] is not None] + ([feature] if feature is not None else [])
        answers = [t["text"] for t in turns]
        # The transcript goes on the instruction line, above the <speech> placeholder.
        prompt_instruction = instruction if transcript is None else f"{instruction.strip()}\n{transcript}".strip()
        user_turns = [t["user"] for t in turns] + [{"instruction": prompt_instruction, "text": user_text}]
        if transcript is not None:
            shown = [instruction.strip(), {"path": audio_path}, f"*{transcript}*"]
        else:
            shown = [instruction.strip(), user_text] if user_text else [instruction.strip(), {"path": audio_path}]
        chat = chat + [{"role": "user", "content": c} for c in shown if c] + [{"role": "assistant", "content": ""}]
        speak = assistant.has_talker and not text_only
        # None, not b"", when silent: Gradio 5 (omni-qwen) runs every streamed chunk through ffmpeg.
        yield chat, wav_stream_header() if speak and not gradio5 else None, turns, None, ""
        pieces, text, pending, sent = [], "", [], False
        for text, audio in assistant.stream(features, answers, user_turns, temperature, top_p,
                                            talker_temperature, text_only=text_only):
            chat[-1]["content"] = text
            pieces.append(audio)
            if not gradio5:
                yield chat, to_pcm16(audio) if speak else None, turns, None, ""
                continue
            pending.append(audio)
            ready = sum(len(a) for a in pending) >= (segment if sent else first_segment)
            yield chat, to_wav(np.concatenate(pending)) if speak and ready else None, turns, None, ""
            if ready:
                pending, sent = [], True
        if gradio5 and speak and sum(len(a) for a in pending):
            yield chat, to_wav(np.concatenate(pending)), turns, None, ""
        if speak:
            wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(wav, np.concatenate(pieces), SNAC_SAMPLE_RATE)
            chat.append({"role": "assistant", "content": {"path": wav}})
        if transcript is None:
            turns = turns + [{"features": feature, "text": text, "user": user_turns[-1]}]
        else:  # later turns see this one as text
            turns = turns + [{"features": None, "text": text, "user": {"instruction": instruction, "text": transcript}}]
        yield chat, b"" if speak and not gradio5 else None, turns, None, ""

    with gr.Blocks(title="Latxa-Omni-SNAC") as demo:
        gr.Markdown("# Latxa-Omni-SNAC\nGrabatu galdera; erantzuna testu eta ahots moduan iritsiko da, "
                    "sortu ahala. / Record a question; the answer streams back as text and speech.")
        turns = gr.State([])
        # Gradio 6 only has the messages format and no longer takes type=.
        chat = gr.Chatbot(height=480, label="Elkarrizketa", **({"type": "messages"} if gradio_major < 6 else {}))
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Zure galdera")
            audio_out = gr.Audio(streaming=True, autoplay=True, label="Erantzuna", interactive=False)
        user_text = gr.Textbox(label="Edo idatzi (aukerakoa)", lines=2,
                               placeholder="Grabatu beharrean idatzi. Adib. TTSrako: agindua 'Irakurri ozen "
                                           "testu hau.' eta hemen irakurtzeko testua.")
        with gr.Row():
            instruction = gr.Textbox(label="Agindua (aukerakoa)", scale=4,
                                     placeholder="Hutsik: elkarrizketa. Adib.: Transkribatu audioa. / "
                                                 "Itzuli ingelesera esaten dena.")
            text_only = gr.Checkbox(label="Testua bakarrik", value=not assistant.has_talker,
                                    interactive=assistant.has_talker, scale=1)
            transcribe_first = gr.Checkbox(label="Transkribatu lehenik", value=assistant.args.transcribe_first,
                                           scale=1)
        with gr.Row():
            send = gr.Button("Bidali", variant="primary")
            clear = gr.Button("Garbitu")
        with gr.Accordion("Ezarpenak", open=False):
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="LLM temperature (0 = greedy)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="LLM top-p")
            talker_temperature = gr.Slider(0.1, 1.2, value=0.7, step=0.05, label="Talker temperature")

        inputs = [audio_in, user_text, instruction, text_only, transcribe_first, chat, turns, temperature, top_p,
                  talker_temperature]
        outputs = [chat, audio_out, turns, audio_in, user_text]
        # Answer as soon as the recording stops; uploads go through the button.
        audio_in.stop_recording(respond, inputs, outputs)
        send.click(respond, inputs, outputs)
        user_text.submit(respond, inputs, outputs)
        clear.click(lambda: ([], [], None, None, ""), None, [chat, turns, audio_in, audio_out, user_text])
    return demo


def benchmark(assistant, paths):
    for path in paths:
        start = time.perf_counter()
        features = [assistant.encode(path)]
        first_text = first_audio = None
        samples = 0
        for text, audio in assistant.stream(features, [], [{"instruction": "", "text": ""}], 0.0, 1.0, 0.7):
            now = time.perf_counter() - start
            if text and first_text is None:
                first_text = now
            if len(audio) and first_audio is None:
                first_audio = now
            samples += len(audio)
        total = time.perf_counter() - start
        seconds = samples / SNAC_SAMPLE_RATE
        print(f"{path}: first text {first_text:.2f}s, first audio {first_audio or float('nan'):.2f}s, "
              f"{seconds:.1f}s of speech in {total:.1f}s (RTF {total / max(seconds, 1e-6):.2f}, "
              f"{seconds * 82 / total:.0f} talker tok/s) | {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True,
                        help="Stage 2 checkpoint (talker-only or full), or a Stage 1 checkpoint (text only)")
    parser.add_argument("--legacy-prompt", action="store_true",
                        help="empty instructions use the old fixed prompt (pre-multi-task models)")
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_units", type=int, default=4096, help="talker tokens per answer (~50 s)")
    parser.add_argument("--chunk-frames", type=int, default=4, help="SNAC frames (~85 ms each) per audio chunk")
    parser.add_argument("--fast-talker", action="store_true", help="talker decoding with a CUDA graph per token")
    parser.add_argument("--first-chunk-frames", type=int, default=2)
    parser.add_argument("--segment-seconds", type=float, default=1.5,
                        help="Gradio 5+ only: shortest audio segment sent to the browser")
    parser.add_argument("--first-segment-seconds", type=float, default=0.5,
                        help="Gradio 5+ only: shortest first audio segment (lower = speech starts sooner)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--transcribe-first", action="store_true",
                        help="default for the checkbox: answer the transcript of each spoken turn as text")
    parser.add_argument("--benchmark", nargs="+", metavar="WAV", help="print latency/RTF for these files and exit")
    args = parser.parse_args()

    assistant = Assistant(args)
    if args.benchmark:
        benchmark(assistant, args.benchmark)
        return
    build_demo(assistant).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
