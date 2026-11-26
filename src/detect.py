"""Slide a 5 s window over a long recording and report threat events with timestamps."""
import json
import sys

import joblib
import numpy as np
import soundfile as sf

from . import config as C
from . import embeddings
from .classifier import clip_features

FRAME_S = 0.48            # YAMNet patch hop in seconds
WINDOW_FRAMES = 10        # 10 patches is about one 5 s training clip
HOP_FRAMES = 2            # about 1 s between windows
THREAT_IDX = [C.CLASSES.index(c) for c in sorted(C.THREAT)]


def load_head():
    return joblib.load(C.ROOT / "models" / "yamnet_head.joblib")["head"]


def read_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == C.SR, f"expected {C.SR} Hz, got {sr}"
    return wav.mean(axis=1) if wav.ndim > 1 else wav


def window_probs(emb, head):
    """emb [frames, 1024] -> (window centres in seconds, probabilities [n_windows, 10])."""
    starts = list(range(0, max(1, len(emb) - WINDOW_FRAMES + 1), HOP_FRAMES))
    windows = np.stack([emb[s:s + WINDOW_FRAMES] for s in starts])
    centres = np.array([s * FRAME_S + C.CLIP_SECONDS / 2 for s in starts])
    return centres, head.predict_proba(clip_features(windows))


MERGE_GAP_S = 2.0         # join same-label events separated by this gap or less


def find_events(centres, proba, threshold=0.5):
    """Merge consecutive windows whose summed threat probability is >= threshold."""
    threat = proba[:, THREAT_IDX].sum(axis=1)
    half_hop = HOP_FRAMES * FRAME_S / 2
    events, run = [], []
    for i, flagged in enumerate(threat >= threshold):
        if flagged:
            run.append(i)
        if (not flagged or i == len(threat) - 1) and run:
            idx = np.array(run)
            top = int(proba[idx][:, THREAT_IDX].sum(axis=0).argmax())
            events.append({"start": round(float(centres[idx[0]] - half_hop), 2),
                           "end": round(float(centres[idx[-1]] + half_hop), 2),
                           "label": C.CLASSES[THREAT_IDX[top]],
                           "score": round(float(threat[idx].max()), 3)})
            run = []
    merged = []
    for e in events:                      # fragments of one sound: same label, small gap
        if merged and merged[-1]["label"] == e["label"] and e["start"] - merged[-1]["end"] <= MERGE_GAP_S:
            merged[-1]["end"] = e["end"]
            merged[-1]["score"] = max(merged[-1]["score"], e["score"])
        else:
            merged.append(e)
    return merged


def analyse(path, model=None, head=None):
    model = model or embeddings.load_yamnet()
    head = head or load_head()
    _, emb = embeddings.embed_waveform(model, read_wav(path))
    return window_probs(emb, head)


if __name__ == "__main__":
    wav_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    centres, proba = analyse(wav_path)
    print(json.dumps(find_events(centres, proba, threshold), indent=2))