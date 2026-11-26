"""Metrics shared by every approach, so the comparison table is like-for-like."""
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)

from . import config as C

LABELS = list(range(len(C.CLASSES)))
THREAT_IDX = [C.CLASSES.index(c) for c in sorted(C.THREAT)]


def summarize(y_true, proba):
    """proba: [n, 10] class probabilities, each row summing to 1."""
    pred = proba.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    is_threat = np.isin(y_true, THREAT_IDX)
    flagged = np.isin(pred, THREAT_IDX)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1), labels=LABELS)),
        "roc_auc_macro": float(roc_auc_score(y_true, proba, multi_class="ovr", labels=LABELS)),
        "threats_flagged": f"{int(flagged[is_threat].sum())}/{int(is_threat.sum())}",
        "false_alarms": f"{int(flagged[~is_threat].sum())}/{int((~is_threat).sum())}",
    }


def print_report(y_true, proba):
    pred = proba.argmax(axis=1)
    print(classification_report(y_true, pred, labels=LABELS, target_names=C.CLASSES,
                                digits=2, zero_division=0))
    print("confusion matrix (rows = true, columns = predicted):")
    print(confusion_matrix(y_true, pred, labels=LABELS))