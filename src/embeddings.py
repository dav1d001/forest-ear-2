"""Run YAMNet once over every clip and cache frame-level embeddings and class scores."""
import csv

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from . import config as C
from . import data

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


def load_yamnet():
    return hub.load(YAMNET_URL)


def class_names(model):
    path = model.class_map_path().numpy().decode()
    with tf.io.gfile.GFile(path) as f:
        return [row["display_name"] for row in csv.DictReader(f)]


def embed_waveform(model, wav):
    """wav: float32 mono at 16 kHz -> (scores [frames, 521], embeddings [frames, 1024])."""
    scores, emb, _ = model(wav)
    return scores.numpy(), emb.numpy()


def build_cache(force: bool = False):
    out = C.CACHE / "yamnet_esc10.npz"
    if out.exists() and not force:
        return out
    d = data.load()
    model = load_yamnet()
    scores, embs = [], []
    for wav in d["X"]:
        s, e = embed_waveform(model, wav)
        scores.append(s)
        embs.append(e)
    np.savez_compressed(out, scores=np.stack(scores), emb=np.stack(embs), y=d["y"],
                        fold=d["fold"], filename=d["filename"],
                        class_names=np.array(class_names(model)))
    return out


def load():
    d = np.load(build_cache(), allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    d = load()
    print("embeddings:", d["emb"].shape, "| scores:", d["scores"].shape)
    names = d["class_names"]
    print("\nYAMNet's top-3 labels per ESC-10 class (mean score):")
    for i, c in enumerate(C.CLASSES):
        mean = d["scores"][d["y"] == i].mean(axis=(0, 1))
        top = np.argsort(-mean)[:3]
        print(f"  {c:15s}", ", ".join(f"{names[j]} ({mean[j]:.2f})" for j in top))
