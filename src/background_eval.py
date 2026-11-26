"""Does a background class cut false alarms on sounds the classifier never heard?"""
import joblib

import json

from . import config as C
from . import detect, embeddings, other_class, synth
from .score_events import score

FOLDER = C.ROOT / "data" / "long_heldout"
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def run():
    _, held_out = other_class.split_categories()
    if not (FOLDER / "truth.json").exists():
        synth.build(seed=1, categories=held_out, out=FOLDER)
    truth = json.loads((FOLDER / "truth.json").read_text())
    hours = len(truth) * synth.DURATION / 3600

    model = embeddings.load_yamnet()
    heads = {"10-class": detect.load_head(),
             "11-class (+ background)": joblib.load(C.ROOT / "models" / "yamnet_head_bg.joblib")["head"]}
    embs = {n: embeddings.embed_waveform(model, detect.read_wav(FOLDER / n))[1] for n in truth}

    lines = ["Background sounds in these recordings come only from categories neither head saw "
             "in training: " + ", ".join(held_out), "",
             "| Head | Threshold | Threats detected | Recall | Predicted | Precision | "
             "False alarms (distractor / background) | False alarms per hour |",
             "|---|---|---|---|---|---|---|---|"]
    for label, head in heads.items():
        cached = {n: detect.window_probs(e, head) for n, e in embs.items()}
        for t in THRESHOLDS:
            pred = {n: detect.find_events(c, p, t) for n, (c, p) in cached.items()}
            r = score(truth, pred, hours)
            lines.append(f"| {label} | {t} | {r['detected']} | {r['recall']:.2f} | {r['predicted']} | "
                         f"{r['precision']:.2f} | {r['fa_distractor']} / {r['fa_background']} | "
                         f"{r['fa_per_hour']:.0f} |")
    print("\n".join(lines))
    (C.ROOT / "results" / "background_class.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()