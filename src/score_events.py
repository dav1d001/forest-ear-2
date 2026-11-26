"""Score threat-event detection on the synthetic recordings, across several thresholds."""
import json

from . import config as C
from . import detect, embeddings
from .synth import DURATION

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_IOU = 0.2             # a prediction matches a true event only if they overlap enough


def overlap(a, b):
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def iou(a, b):
    inter = overlap(a, b)
    return inter / ((a["end"] - a["start"]) + (b["end"] - b["start"]) - inter)


def score(truth, pred, hours):
    tp = fn = ok = fa_dis = fa_bg = 0
    for name, events in truth.items():
        threats = [e for e in events if e["threat"]]
        others = [e for e in events if not e["threat"]]
        for e in threats:
            if any(iou(e, p) >= MIN_IOU for p in pred[name]):
                tp += 1
            else:
                fn += 1
        for p in pred[name]:
            if any(iou(p, e) >= MIN_IOU for e in threats):
                ok += 1
            elif any(overlap(p, e) > 0 for e in others):
                fa_dis += 1
            else:
                fa_bg += 1
    n_pred = ok + fa_dis + fa_bg
    return {"detected": f"{tp}/{tp + fn}", "recall": tp / (tp + fn),
            "predicted": n_pred, "precision": ok / n_pred if n_pred else 0.0,
            "fa_distractor": fa_dis, "fa_background": fa_bg,
            "fa_per_hour": (fa_dis + fa_bg) / hours}


def run():
    folder = C.ROOT / "data" / "long"
    truth = json.loads((folder / "truth.json").read_text())
    model, head = embeddings.load_yamnet(), detect.load_head()
    cached = {}
    for name in truth:                       # YAMNet runs once per recording
        cached[name] = detect.analyse(folder / name, model, head)
    hours = len(truth) * DURATION / 3600

    rows = []
    for t in THRESHOLDS:
        pred = {n: detect.find_events(c, p, t) for n, (c, p) in cached.items()}
        rows.append({"threshold": t, **score(truth, pred, hours)})

    lines = [f"Match rule: a predicted event matches a true threat if IoU >= {MIN_IOU}.", "",
             "| Threshold | Threats detected | Recall | Predicted events | Precision | "
             "False alarms (distractor / background) | False alarms per hour |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['threshold']} | {r['detected']} | {r['recall']:.2f} | {r['predicted']} | "
                     f"{r['precision']:.2f} | {r['fa_distractor']} / {r['fa_background']} | "
                     f"{r['fa_per_hour']:.0f} |")
    print("\n".join(lines))
    (C.ROOT / "results").mkdir(exist_ok=True)
    (C.ROOT / "results" / "event_detection.md").write_text("\n".join(lines) + "\n")
    (C.ROOT / "results" / "event_detection.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    run()