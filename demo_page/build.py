"""Build the static demo page (ansuehu.github.io/Latxa-Omni) from infer.py output.

    python demo_page/build.py --questions questions.json \
        --answers out/answers.jsonl --out ../Latxa-Omni --model "Latxa-Qwen3.5-4B"

questions.json is the same file passed to infer.py. Besides "id" and "speech",
each item may carry "text" (the transcript), "translation", "answer_translation"
and "category" (a filter tab on the page). The output folder gets index.html,
examples.js and the audio under audio/; push it to the Latxa-Omni repo and
enable GitHub Pages on its main branch.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def copy_audio(src, stem):
    """Copy a wav as mp3 (a tenth of the size) when ffmpeg is available, else as is."""
    if shutil.which("ffmpeg"):
        dst = stem.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-b:a", "64k", dst], check=True)
    else:
        dst = stem.with_suffix(".wav")
        shutil.copyfile(src, dst)
    return dst


def main(args):
    questions = {q["id"]: q for q in json.load(open(args.questions, encoding="utf-8"))}
    answers = [json.loads(line) for line in open(args.answers, encoding="utf-8") if line.strip()]
    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for answer in answers:
        q = questions[answer["question_id"]]
        qid = str(q["id"])
        question_audio = audio_dir / f"{qid}_q{Path(q['speech']).suffix}"
        shutil.copyfile(q["speech"], question_audio)
        answer_audio = copy_audio(answer["prediction_wav"], audio_dir / f"{qid}_a")
        examples.append({
            "id": qid,
            "category": q.get("category"),
            "question_text": q.get("text"),
            "question_translation": q.get("translation"),
            "question_audio": question_audio.relative_to(out).as_posix(),
            "answer_text": answer["prediction"],
            "answer_translation": q.get("answer_translation"),
            "answer_audio": answer_audio.relative_to(out).as_posix(),
        })

    # A script rather than JSON, so the page also opens from file:// without a server.
    payload = json.dumps({"model": args.model, "examples": examples}, ensure_ascii=False, indent=1)
    (out / "examples.js").write_text(f"window.EXAMPLES = {payload};\n", encoding="utf-8")
    shutil.copyfile(HERE / "index.html", out / "index.html")
    (out / ".nojekyll").touch()
    print(f"wrote {len(examples)} examples to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", help="model name shown above the examples")
    main(parser.parse_args())
