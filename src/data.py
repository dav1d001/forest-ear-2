"""Extract ESC-10 from the ESC-50 zip, resample to 16 kHz mono, cache as one .npz."""
import io
import zipfile
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C

PREFIX = "ESC-50-master/"


def prepare(force: bool = False):
    out = C.CACHE / "esc10_16k.npz"
    if out.exists() and not force:
        return out
    C.CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(C.ZIP_PATH) as z:
        meta = pd.read_csv(io.BytesIO(z.read(PREFIX + "meta/esc50.csv")))
        meta = meta[meta["esc10"]].reset_index(drop=True)
        assert set(meta["category"]) == set(C.CLASSES), "unexpected ESC-10 classes"
        clips = []
        for name in meta["filename"]:
            wav, sr = sf.read(io.BytesIO(z.read(PREFIX + "audio/" + name)), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            g = gcd(sr, C.SR)
            wav = resample_poly(wav, C.SR // g, sr // g).astype("float32")
            wav = np.pad(wav, (0, max(0, C.CLIP_SAMPLES - len(wav))))[: C.CLIP_SAMPLES]
            clips.append(wav)
    labels = meta["category"].map({c: i for i, c in enumerate(C.CLASSES)}).to_numpy()
    np.savez_compressed(out, X=np.stack(clips), y=labels,
                        fold=meta["fold"].to_numpy(), filename=meta["filename"].to_numpy())
    return out


def load():
    d = np.load(prepare(), allow_pickle=True)
    return {k: d[k] for k in d.files}


def split_masks(fold):
    return {"train": np.isin(fold, C.TRAIN_FOLDS),
            "val": np.isin(fold, C.VAL_FOLDS),
            "test": np.isin(fold, C.TEST_FOLDS)}


if __name__ == "__main__":
    d = load()
    masks = split_masks(d["fold"])
    print("clips:", d["X"].shape, "| classes:", len(C.CLASSES))
    print({k: int(m.sum()) for k, m in masks.items()})
    print("clips per class:", np.bincount(d["y"]).tolist())
