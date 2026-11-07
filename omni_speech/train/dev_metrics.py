"""Dependency-free dev metrics for Stage 1: WER for ASR, BLEU and chrF for speech translation.

They are training-time monitors on a few dozen dev rows. BLEU follows sacrebleu's corpus BLEU
(13a-style tokenization, exp smoothing off) and chrF its chrF2 (char 6-grams, beta 2), so the
numbers are close to sacrebleu's but not a replacement for a proper test-set evaluation.
"""

import math
import re
import unicodedata
from collections import Counter


def normalize_for_wer(text):
    """Lowercase, drop punctuation and symbols, collapse whitespace."""
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text.lower())
    return " ".join(text.split())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp)))
        previous = current
    return previous[-1]


def corpus_wer(references, predictions):
    """Word errors over reference words, in percent, after normalize_for_wer."""
    errors = words = 0
    for reference, prediction in zip(references, predictions):
        reference, prediction = normalize_for_wer(reference).split(), normalize_for_wer(prediction).split()
        errors += edit_distance(reference, prediction)
        words += len(reference)
    return 100.0 * errors / max(words, 1)


def tokenize_13a(text):
    text = re.sub(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 ", f" {text} ")
    text = re.sub(r"([^0-9])([\.,])", r"\1 \2 ", text)
    text = re.sub(r"([\.,])([^0-9])", r" \1 \2", text)
    text = re.sub(r"([0-9])(-)", r"\1 \2 ", text)
    return text.split()


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references, predictions, max_order=4):
    matches, totals = [0] * max_order, [0] * max_order
    ref_length = hyp_length = 0
    for reference, prediction in zip(references, predictions):
        reference, prediction = tokenize_13a(reference), tokenize_13a(prediction)
        ref_length += len(reference)
        hyp_length += len(prediction)
        for n in range(1, max_order + 1):
            hyp_ngrams = ngrams(prediction, n)
            matches[n - 1] += sum((hyp_ngrams & ngrams(reference, n)).values())
            totals[n - 1] += max(len(prediction) - n + 1, 0)
    if hyp_length == 0 or min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals)) / max_order
    brevity = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length)
    return 100.0 * brevity * math.exp(log_precision)


def corpus_chrf(references, predictions, order=6, beta=2):
    """chrF2: character n-gram F-score averaged over n=1..order, whitespace removed."""
    matches, hyp_totals, ref_totals = [0] * order, [0] * order, [0] * order
    for reference, prediction in zip(references, predictions):
        reference, prediction = "".join(reference.split()), "".join(prediction.split())
        for n in range(1, order + 1):
            ref_ngrams, hyp_ngrams = ngrams(reference, n), ngrams(prediction, n)
            matches[n - 1] += sum((ref_ngrams & hyp_ngrams).values())
            hyp_totals[n - 1] += sum(hyp_ngrams.values())
            ref_totals[n - 1] += sum(ref_ngrams.values())
    scores = []
    for match, hyp_total, ref_total in zip(matches, hyp_totals, ref_totals):
        if hyp_total == 0 or ref_total == 0:
            continue
        precision, recall = match / hyp_total, match / ref_total
        scores.append(0.0 if precision + recall == 0 else
                      (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall))
    return 100.0 * sum(scores) / order


def task_metrics(task, references, predictions):
    """Metrics logged for each task's dev generations (chat has none).

    For tts the thinker only has to copy the text it was given, so WER scores the copy.
    """
    if task in ("asr", "tts"):
        return {"wer": corpus_wer(references, predictions)}
    if task.startswith("st_"):
        return {"bleu": corpus_bleu(references, predictions), "chrf": corpus_chrf(references, predictions)}
    return {}
