"""Speech in, text + 24 kHz speech out.

    python omni_speech/infer/infer.py --model-path <stage2 checkpoint> \
        --question-file questions.json --answer-file out/answers.jsonl

questions.json is a list of {"id", "speech", ...} (the LLaMA-Omni format; extra
keys are ignored). Each answer's audio is written next to the answer file as
wav/<id>.wav, and its path is stored in the JSONL.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
import whisper
from tqdm import tqdm

from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.model.builder import QWEN3_5_MODEL_TYPES, load_pretrained_model
from omni_speech.model.speech_generator.snac_codec import SNAC_SAMPLE_RATE, SnacCodec
from omni_speech.train.stage1 import USER_PROMPT


def load_mel(path, mel_size):
    audio = whisper.pad_or_trim(whisper.load_audio(path))
    return whisper.log_mel_spectrogram(audio, n_mels=mel_size).permute(1, 0)


def main(args):
    with open(Path(args.model_path) / "config.json", encoding="utf-8") as stream:
        model_type = json.load(stream)["model_type"]
    conv_mode = args.conv_mode or ("qwen3_5" if model_type in QWEN3_5_MODEL_TYPES else "llama_3")
    tokenizer, model, _ = load_pretrained_model(args.model_path, None, s2s=True)
    model.eval()
    codec = SnacCodec()
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], USER_PROMPT)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_speech_token(conv.get_prompt(), tokenizer, return_tensors="pt")[None].cuda()

    questions = json.load(open(args.question_file, encoding="utf-8"))
    answer_file = Path(args.answer_file)
    wav_dir = answer_file.parent / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    with open(answer_file, "w", encoding="utf-8") as out:
        for item in tqdm(questions):
            speech = load_mel(item["speech"], args.mel_size)[None].to("cuda", torch.float16)
            text_ids, units = model.generate(
                input_ids,
                speech=speech,
                speech_lengths=torch.tensor([speech.shape[1]], device="cuda"),
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                talker_kwargs={"temperature": args.talker_temperature, "max_tokens": args.max_units},
            )
            wav_path = wav_dir / f"{item['id']}.wav"
            sf.write(wav_path, codec.decode(units), SNAC_SAMPLE_RATE)
            out.write(json.dumps({
                "question_id": item["id"],
                "prediction": tokenizer.decode(text_ids[0], skip_special_tokens=True).strip(),
                "prediction_wav": str(wav_path),
                "num_units": len(units),
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--conv-mode", help="chat template; default llama_3, or qwen3_5 for a Qwen3.5 model")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--talker_temperature", type=float, default=0.7)
    parser.add_argument("--max_units", type=int, default=4096)
    main(parser.parse_args())
