"""YAMNet embeddings -> clip features -> logistic-regression head, tuned on the validation fold."""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import embeddings, evaluate
from .data import split_masks

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def clip_features(emb):
    """[n, frames, 1024] -> [n, 2048]: mean and max over time."""
    return np.concatenate([emb.mean(axis=1), emb.max(axis=1)], axis=1)


def make_head(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def train():
    d = embeddings.load()
    X, y = clip_features(d["emb"]), d["y"]
    m = split_masks(d["fold"])

    best_c, best_val = None, -1.0
    for c in C_GRID:                       # choose C on the validation fold only
        val_acc = make_head(c).fit(X[m["train"]], y[m["train"]]).score(X[m["val"]], y[m["val"]])
        print(f"C={c:<6} validation accuracy = {val_acc:.3f}")
        if val_acc > best_val:
            best_c, best_val = c, val_acc
    print(f"selected C={best_c}")

    head = make_head(best_c).fit(X[m["train"]], y[m["train"]])
    proba = head.predict_proba(X[m["test"]])   # test fold is used once, at the end
    metrics = evaluate.summarize(y[m["test"]], proba)
    print("\nTEST fold:", json.dumps(metrics, indent=2))
    evaluate.print_report(y[m["test"]], proba)

    (C.ROOT / "models").mkdir(exist_ok=True)
    (C.ROOT / "results").mkdir(exist_ok=True)
    joblib.dump({"head": head, "classes": C.CLASSES}, C.ROOT / "models" / "yamnet_head.joblib")
    (C.ROOT / "results" / "head_metrics.json").write_text(json.dumps({"C": best_c, **metrics}, indent=2))
    return head, metrics


if __name__ == "__main__":
    train()