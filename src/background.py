"""Load non-ESC-10 clips (wind, birds, insects, ...) as realistic 'unknown' background."""
import io
import zipfile
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C

PREFIX = "ESC-50-master/"


def load_other(fold: int = 5):
    """Clips from the 40 ESC-50 classes that are not in ESC-10, from one fold."""
    out = C.CACHE / f"other_fold{fold}.npz"
    if out.exists():
        d = np.load(out, allow_pickle=True)
        return d["X"], d["category"]
    C.CACHE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(C.ZIP_PATH) as z:
        meta = pd.read_csv(io.BytesIO(z.read(PREFIX + "meta/esc50.csv")))
        meta = meta[(~meta["esc10"]) & (meta["fold"] == fold)].reset_index(drop=True)
        clips = []
        for name in meta["filename"]:
            wav, sr = sf.read(io.BytesIO(z.read(PREFIX + "audio/" + name)), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            g = gcd(sr, C.SR)
            wav = resample_poly(wav, C.SR // g, sr // g).astype("float32")
            clips.append(np.pad(wav, (0, max(0, C.CLIP_SAMPLES - len(wav))))[: C.CLIP_SAMPLES])
    X = np.stack(clips)
    np.savez_compressed(out, X=X, category=meta["category"].to_numpy())
    return X, meta["category"].to_numpy()