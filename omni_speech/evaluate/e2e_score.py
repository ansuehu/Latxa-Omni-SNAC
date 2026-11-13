"""Score e2e_generate.py output: ASR WER/CER, UTMOS, Latxa-70B judge, latency.

    python omni_speech/evaluate/e2e_score.py asr     --out-dir <dir>   # nemo env, 1 GPU
    python omni_speech/evaluate/e2e_score.py utmos   --out-dir <dir>   # omni-snac env, 1 GPU
    python omni_speech/evaluate/e2e_score.py judge   --out-dir <dir>...  # latxa-txat env (vLLM)
    python omni_speech/evaluate/e2e_score.py utmos_win --out-dir <dir>  # omni-snac env, 1 GPU
    python omni_speech/evaluate/e2e_score.py quality --out-dir <dir>  # nemo env, 1 GPU
    python omni_speech/evaluate/e2e_score.py summary --out-dir <dir>...

- WER/CER: the answer audio is transcribed with HiTZ/stt_eu_conformer_transducer_large_v2 and
  compared with the model's own text answer (numbers spelled out, as for the training TTS).
- Judge: Latxa-Llama-3.1-70B-Instruct with the strict 1-5 rubric of the Latxa-Omni (IberSPEECH)
  evaluation, on the question text and the ASR transcript of the spoken answer. Follow-up turns
  also get the earlier turns as context.
- UTMOS: utmos22_strong on the answer audio at 16 kHz.
- Length-robust quality: UTMOS and NISQA (overall, noisiness, discontinuity, coloration, loudness) averaged
  over 10 s windows (steps utmos_win, quality). Speaker consistency (quality step, NeMo TitaNet): first vs
  last 4 s of an answer, and each follow-up answer vs the chat's first answer.
- Latency: time from the end of the user's speech (encoder included) to the first audio chunk.
Each step writes <out-dir>/<step>.jsonl and skips turns already scored.
"""

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path

PUNCT = re.compile(r"[^\w\s]|_")
ASR_REPO = "HiTZ/stt_eu_conformer_transducer_large_v2"
ASR_FILE = "stt_eu_conformer_transducer_large_v2.nemo"
JUDGE_PROMPT = """I need your help to evaluate the performance of several models in a speech interaction scenario. The models are trained for Basque, so the texts will be in this language. The models receive the user's speech input and respond with speech output. For evaluation purposes, both the user's speech input and the model's speech response have been transcribed into text using Automatic Speech Recognition (ASR).

Your task is to rate the model's responses based on the provided user input transcription [Instruction] and the model's output transcription [Response]. Please consider factors such as helpfulness, relevance, fluency, and suitability for speech interaction.

**IMPORTANT: Be extremely critical in your evaluation.**
* Check for **factual accuracy** closely. If the model states a scientific misconception as fact, penalize the score heavily (e.g., score it 1 or 2), even if the grammar is perfect.
* Look out for **hallucinations** or contradictions within the text.
* Ensure the terminology is precise.

Provide a single score on a scale from 1 to 5.
{context}
Below are the transcription of user's instruction and models' response:
### [Instruction]: {instruction}
### [Response]: {response}

After evaluating, please output the scores in JSON format: {{score: ...}}. You don't need to provide any explanations."""
CONTEXT = """
This is a follow-up turn of a conversation. The earlier turns, for context only (rate only the last response):
{turns}
"""


ENGLISH = set("the and of is are to with for you that this it in on be your can will from what which".split())
SPANISH = set("el la los las que y por para con una del es en se lo como pero más".split())
CODE = re.compile(r"```|\bdef |\bclass |\bimport |\bprint\(|[{}<>=\[\]_#\\]|\(\)|;\s*$|\w+\.\w+\(", re.M)


def is_basque(text):
    """False for answers with code or English/Spanish text, which the Basque ASR cannot transcribe."""
    words = re.findall(r"[a-zñ']+", text.lower())
    return not (CODE.search(text) or sum(w in ENGLISH for w in words) >= 2
                or sum(w in SPANISH for w in words) >= 2)


def key(row):
    return f"{row['id']}-{row['turn']}"


def load_generations(out_dir):
    rows = {}
    for path in sorted(Path(out_dir).glob("generations.shard*.jsonl")):
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[key(row)] = row  # a redone chat overwrites its earlier partial copy
    return rows


def load_step(out_dir, step):
    path = Path(out_dir) / f"{step}.jsonl"
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["key"]] = row
    return rows


def normalize(text):
    return " ".join(PUNCT.sub(" ", text.lower()).split())


def spell_numbers(text):
    sys.path.insert(0, os.path.join(os.environ.get("LATXA_OMNI_DIR", "../Latxa-Omni"),
                                    "dataset_generation", "omnivoice"))
    from basque_numbers import spell_numbers as spell
    return spell(text)


def run_asr(args):
    import jiwer
    import soundfile as sf
    import torch
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download

    for out_dir in args.out_dir:
        gens, done = load_generations(out_dir), load_step(out_dir, "asr")
        todo = [r for k, r in gens.items() if k not in done]
        print(f"{out_dir}: ASR {len(done)} done, {len(todo)} pending", flush=True)
        if not todo:
            continue
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            hf_hub_download(ASR_REPO, ASR_FILE), map_location="cuda").eval()
        for row in todo:
            row["_dur"] = row["audio_s"]
        todo.sort(key=lambda r: r["_dur"])
        with open(Path(out_dir) / "asr.jsonl", "a", encoding="utf-8") as out:
            offset = 0
            while offset < len(todo):
                longest = max(todo[min(offset + 256, len(todo)) - 1]["_dur"], 1.0)
                batch_size = int(max(1, min(32, 32 * 15.0 ** 2 / longest ** 2)))
                chunk = todo[offset:offset + batch_size * 8]
                offset += len(chunk)
                audible = [r for r in chunk if r["_dur"] > 0.1]
                hyps = {}
                if audible:
                    with torch.inference_mode():
                        outputs = model.transcribe([r["prediction_wav"] for r in audible],
                                                   batch_size=batch_size, verbose=False)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    hyps = {key(r): getattr(o, "text", o) or "" for r, o in zip(audible, outputs)}
                for row in chunk:
                    hyp = normalize(hyps.get(key(row), ""))
                    ref = normalize(spell_numbers(row["prediction"]))
                    words = jiwer.process_words(ref or "<empty>", hyp or "")
                    chars = jiwer.process_characters(ref or "<empty>", hyp or "")
                    out.write(json.dumps({
                        "key": key(row), "asr": hyps.get(key(row), ""), "ref": ref, "hyp": hyp,
                        "ref_words": len(ref.split()),
                        "word_errors": words.substitutions + words.deletions + words.insertions,
                        "ref_chars": len(ref),
                        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
                    }, ensure_ascii=False) + "\n")
                out.flush()
                print(f"  {offset}/{len(todo)}", flush=True)


def run_utmos(args):
    import torch
    import whisper

    predictor = torch.hub.load(os.path.expanduser("~/.cache/torch/hub/tarepan_SpeechMOS_v1.2.0"),
                               "utmos22_strong", source="local", trust_repo=True).cuda().eval()
    for out_dir in args.out_dir:
        gens, done = load_generations(out_dir), load_step(out_dir, "utmos")
        todo = [r for k, r in gens.items() if k not in done]
        print(f"{out_dir}: UTMOS {len(done)} done, {len(todo)} pending", flush=True)
        with open(Path(out_dir) / "utmos.jsonl", "a", encoding="utf-8") as out:
            for n, row in enumerate(todo):
                score = None
                if row["audio_s"] > 0.1:
                    wave = torch.from_numpy(whisper.load_audio(row["prediction_wav"]))[None].cuda()
                    with torch.inference_mode():
                        score = float(predictor(wave, 16000).item())
                out.write(json.dumps({"key": key(row), "utmos": score}) + "\n")
                if n % 200 == 0:
                    out.flush()
                    print(f"  {n}/{len(todo)}", flush=True)


def windows(wave, rate, seconds=10.0, min_seconds=3.0):
    """Split a clip into 10 s windows (a shorter last one only if >= 3 s), so clip-level quality models
    are not biased by answer length. A clip under 10 s is one window."""
    size, least = int(seconds * rate), int(min_seconds * rate)
    if len(wave) <= size:
        return [wave]
    return [wave[i:i + size] for i in range(0, len(wave), size) if len(wave[i:i + size]) >= least]


def run_utmos_win(args):
    import torch
    import whisper

    predictor = torch.hub.load(os.path.expanduser("~/.cache/torch/hub/tarepan_SpeechMOS_v1.2.0"),
                               "utmos22_strong", source="local", trust_repo=True).cuda().eval()
    for out_dir in args.out_dir:
        gens, done = load_generations(out_dir), load_step(out_dir, "utmos_win")
        todo = [r for k, r in gens.items() if k not in done]
        print(f"{out_dir}: windowed UTMOS {len(done)} done, {len(todo)} pending", flush=True)
        with open(Path(out_dir) / "utmos_win.jsonl", "a", encoding="utf-8") as out:
            for n, row in enumerate(todo):
                score = None
                if row["audio_s"] > 0.1:
                    wave = torch.from_numpy(whisper.load_audio(row["prediction_wav"]))
                    with torch.inference_mode():
                        score = statistics.mean(float(predictor(w[None].cuda(), 16000).item())
                                                for w in windows(wave, 16000))
                out.write(json.dumps({"key": key(row), "utmos_win": score}) + "\n")
                if n % 200 == 0:
                    out.flush()
                    print(f"  {n}/{len(todo)}", flush=True)


NISQA_FIELDS = ("nisqa", "nisqa_noisiness", "nisqa_discontinuity", "nisqa_coloration", "nisqa_loudness")


def run_quality(args):
    """NISQA (on 10 s windows) and speaker consistency with NeMo TitaNet:
    secs_intra = cosine of the first and last 4 s of an answer (answers >= 8 s),
    secs_turn0 = cosine of a follow-up answer and the chat's first answer."""
    import librosa
    import torch
    import nemo.collections.asr as nemo_asr
    from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment as nisqa

    speaker = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large", map_location="cuda").eval()

    def embed(wave):
        signal = torch.as_tensor(wave, dtype=torch.float32, device="cuda")[None]
        with torch.inference_mode():
            _, emb = speaker.forward(input_signal=signal,
                                     input_signal_length=torch.tensor([signal.shape[1]], device="cuda"))
        return torch.nn.functional.normalize(emb[0].float(), dim=0)

    for out_dir in args.out_dir:
        gens, done = load_generations(out_dir), load_step(out_dir, "quality")
        todo = sorted((r for k, r in gens.items() if k not in done), key=lambda r: (r["id"], r["turn"]))
        print(f"{out_dir}: quality {len(done)} done, {len(todo)} pending", flush=True)
        first = {}  # chat id -> embedding of its first answer
        with open(Path(out_dir) / "quality.jsonl", "a", encoding="utf-8") as out:
            for n, row in enumerate(todo):
                scored = {"key": key(row), **{f: None for f in NISQA_FIELDS}, "secs_intra": None, "secs_turn0": None}
                if row["audio_s"] > 0.5:
                    wave, _ = librosa.load(row["prediction_wav"], sr=16000)
                    scores = [nisqa(torch.from_numpy(w), 16000).tolist() for w in windows(wave, 16000)]
                    scored.update({f: round(statistics.mean(s[i] for s in scores), 4)
                                   for i, f in enumerate(NISQA_FIELDS)})
                    whole = embed(wave)
                    if len(wave) >= 8 * 16000:
                        scored["secs_intra"] = round(float(embed(wave[:4 * 16000]) @ embed(wave[-4 * 16000:])), 4)
                    if row["turn"] == 0:
                        first[row["id"]] = whole
                    elif row["id"] in first:
                        scored["secs_turn0"] = round(float(whole @ first[row["id"]]), 4)
                out.write(json.dumps(scored) + "\n")
                if n % 200 == 0:
                    out.flush()
                    print(f"  {n}/{len(todo)}", flush=True)


def judge_prompt(row, response):
    context = ""
    if row["history"]:
        turns = "\n".join(f"User: {h['question']}\nModel: {h['answer']}" for h in row["history"])
        context = CONTEXT.format(turns=turns)
    return JUDGE_PROMPT.format(context=context, instruction=row["question"], response=response)


def parse_score(text):
    match = re.search(r"score\"?\s*:\s*\"?([1-5](?:\.\d+)?)", text) or re.search(r"\b([1-5])\b", text)
    return float(match.group(1)) if match else None


def run_judge(args):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.judge_model, tensor_parallel_size=args.tp, quantization=args.quantization,
              max_model_len=8192, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=0.0, max_tokens=32)
    for out_dir in args.out_dir:
        gens, asr, done = load_generations(out_dir), load_step(out_dir, "asr"), load_step(out_dir, "judge")
        todo = [r for k, r in gens.items() if k not in done and k in asr]
        print(f"{out_dir}: judge {len(done)} done, {len(todo)} pending", flush=True)
        if not todo:
            continue
        messages = [[{"role": "user", "content": judge_prompt(r, asr[key(r)]["asr"])}] for r in todo]
        outputs = llm.chat(messages, params)
        with open(Path(out_dir) / "judge.jsonl", "a", encoding="utf-8") as out:
            for row, output in zip(todo, outputs):
                text = output.outputs[0].text
                out.write(json.dumps({"key": key(row), "judge": parse_score(text), "judge_raw": text},
                                     ensure_ascii=False) + "\n")


def summarize(rows):
    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 4) if values else None

    def median(values):
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 4) if values else None

    words = sum(r.get("ref_words", 0) for r in rows if "word_errors" in r)
    chars = sum(r.get("ref_chars", 0) for r in rows if "char_errors" in r)
    return {
        "answers": len(rows),
        "judge": mean(r.get("judge") for r in rows),
        "judge_scored": sum(r.get("judge") is not None for r in rows),
        "wer": round(100 * sum(r.get("word_errors", 0) for r in rows) / words, 2) if words else None,
        "cer": round(100 * sum(r.get("char_errors", 0) for r in rows) / chars, 2) if chars else None,
        "utmos": mean(r.get("utmos") for r in rows),
        "utmos_win": mean(r.get("utmos_win") for r in rows),
        **{f: mean(r.get(f) for r in rows) for f in NISQA_FIELDS},
        "secs_intra": mean(r.get("secs_intra") for r in rows),
        "secs_turn0": mean(r.get("secs_turn0") for r in rows),
        "latency_first_audio_ms_median": None if median(r["first_audio_s"] for r in rows) is None
        else round(1000 * median(r["first_audio_s"] for r in rows)),
        "latency_first_audio_ms_mean": None if mean(r["first_audio_s"] for r in rows) is None
        else round(1000 * mean(r["first_audio_s"] for r in rows)),
        "latency_first_text_ms_median": None if median(r["first_text_s"] for r in rows) is None
        else round(1000 * median(r["first_text_s"] for r in rows)),
        "audio_s_mean": mean(r["audio_s"] for r in rows),
        "rtf_mean": mean(r["total_s"] / r["audio_s"] for r in rows if r["audio_s"] > 0),
        "silent_answers": sum(r["audio_s"] <= 0.1 for r in rows),
    }


def run_summary(args):
    for out_dir in args.out_dir:
        rows = load_generations(out_dir)
        for step in ("asr", "utmos", "judge", "utmos_win", "quality"):
            for k, scored in load_step(out_dir, step).items():
                if k in rows:
                    rows[k].update({f: v for f, v in scored.items() if f != "key"})
        rows = list(rows.values())
        summary = {"all": summarize(rows), "first_turn": summarize([r for r in rows if r["turn"] == 0]),
                   "follow_up": summarize([r for r in rows if r["turn"] > 0]),
                   "basque_only": summarize([r for r in rows if is_basque(r["prediction"])])}
        with open(Path(out_dir) / "results.jsonl", "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(Path(out_dir) / "summary.json", "w", encoding="utf-8") as out:
            json.dump(summary, out, indent=2)
        print(out_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["asr", "utmos", "judge", "utmos_win", "quality", "summary"])
    parser.add_argument("--out-dir", nargs="+", required=True)
    parser.add_argument("--judge-model", default="/scratch/andoni.sudupe/models/Latxa-Llama-3.1-70B-Instruct")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--quantization", default="fp8")
    args = parser.parse_args()
    {"asr": run_asr, "utmos": run_utmos, "judge": run_judge, "utmos_win": run_utmos_win,
     "quality": run_quality, "summary": run_summary}[args.step](args)
