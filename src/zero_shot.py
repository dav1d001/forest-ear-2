"""Zero-shot baseline: use YAMNet's own AudioSet labels, no training at all."""
import json

import numpy as np

from . import config as C
from . import embeddings, evaluate
from .data import split_masks

# Which AudioSet labels count as evidence for each of our 10 classes.
LABEL_MAP = {
    "chainsaw": ["Chainsaw"],
    "clock_tick": ["Clock", "Tick-tock"],
    "crackling_fire": ["Fire", "Crackle"],
    "crying_baby": ["Baby cry, infant cry", "Crying, sobbing"],
    "dog": ["Dog"],
    "helicopter": ["Helicopter"],
    "rain": ["Rain", "Rain on surface"],
    "rooster": ["Chicken, rooster", "Crowing, cock-a-doodle-doo"],
    "sea_waves": ["Ocean", "Waves, surf"],
    "sneezing": ["Sneeze"],
}


def class_scores(scores, names):
    """scores [n, frames, 521] -> [n, 10] probabilities (max over frames, then over mapped labels)."""
    index = {n: i for i, n in enumerate(names)}
    missing = [n for labels in LABEL_MAP.values() for n in labels if n not in index]
    if missing:
        raise SystemExit(f"labels not in YAMNet class map: {missing}")
    peak = scores.max(axis=1)                                   # [n, 521]
    out = np.stack([peak[:, [index[n] for n in LABEL_MAP[c]]].max(axis=1) for c in C.CLASSES], axis=1)
    return out / out.sum(axis=1, keepdims=True)


def run():
    d = embeddings.load()
    m = split_masks(d["fold"])
    proba = class_scores(d["scores"][m["test"]], list(d["class_names"]))
    y = d["y"][m["test"]]
    metrics = evaluate.summarize(y, proba)
    print("ZERO-SHOT, TEST fold:", json.dumps(metrics, indent=2))
    evaluate.print_report(y, proba)
    (C.ROOT / "results").mkdir(exist_ok=True)
    (C.ROOT / "results" / "zero_shot_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()