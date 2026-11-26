"""Train an 11-class head: the 10 ESC-10 classes plus 'other' (background)."""
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import embeddings, other_class
from .classifier import clip_features
from .data import split_masks

C_GRID = (0.001, 0.01, 0.1, 1.0)
OTHER = len(C.CLASSES)                      # label 10
CLASSES_BG = C.CLASSES + ["other"]
THREAT_IDX = [C.CLASSES.index(c) for c in sorted(C.THREAT)]


def make_head(c):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))


def train():
    seen, held = other_class.split_categories()
    print("background categories used for training:", ", ".join(seen))
    print("held out (never seen):", ", ".join(held))

    d = embeddings.load()
    o = np.load(other_class.build_seen_cache(), allow_pickle=True)
    m = split_masks(d["fold"])
    X10, y10 = clip_features(d["emb"]), d["y"]
    Xo, fo = clip_features(o["emb"]), o["fold"]

    def stack(mask10, mask_o):
        X = np.concatenate([X10[mask10], Xo[mask_o]])
        y = np.concatenate([y10[mask10], np.full(int(mask_o.sum()), OTHER)])
        return X, y

    Xtr, ytr = stack(m["train"], np.isin(fo, C.TRAIN_FOLDS))
    Xva, yva = stack(m["val"], np.isin(fo, C.VAL_FOLDS))
    print(f"train: {len(ytr)} clips ({int((ytr == OTHER).sum())} 'other') | "
          f"validation: {len(yva)} clips ({int((yva == OTHER).sum())} 'other')")

    best_c, best = None, -1.0
    for c in C_GRID:                        # choose C on the validation folds only
        pred = make_head(c).fit(Xtr, ytr).predict(Xva)
        f1 = f1_score(yva, pred, average="macro")
        print(f"C={c:<6} validation macro-F1 = {f1:.3f}")
        if f1 > best:
            best_c, best = c, f1
    head = make_head(best_c).fit(Xtr, ytr)

    y = y10[m["test"]]                      # ESC-10 test fold, used once
    pred = head.predict(X10[m["test"]])
    is_threat, flagged = np.isin(y, THREAT_IDX), np.isin(pred, THREAT_IDX)
    res = {"C": best_c,
           "esc10_test_accuracy": float(accuracy_score(y, pred)),
           "esc10_test_macro_f1": float(f1_score(y, pred, labels=list(range(OTHER)), average="macro")),
           "clips_predicted_as_other": int((pred == OTHER).sum()),
           "threats_flagged": f"{int(flagged[is_threat].sum())}/{int(is_threat.sum())}",
           "false_alarms": f"{int(flagged[~is_threat].sum())}/{int((~is_threat).sum())}",
           "background_trained_on": seen, "background_held_out": held}
    print("\nESC-10 TEST fold:", json.dumps({k: v for k, v in res.items() if not k.startswith("background")}, indent=2))

    joblib.dump({"head": head, "classes": CLASSES_BG}, C.ROOT / "models" / "yamnet_head_bg.joblib")
    (C.ROOT / "results" / "background_head_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    train()