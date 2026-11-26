"""Background ('other') class: YAMNet embeddings of sound categories outside ESC-10."""
import io
import zipfile
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from . import config as C
from . import embeddings

PREFIX = "ESC-50-master/"
N_HELD_OUT = 10


def _meta(z):
    meta = pd.read_csv(io.BytesIO(z.read(PREFIX + "meta/esc50.csv")))
    return meta[~meta["esc10"]].reset_index(drop=True)


def split_categories(seed: int = 0):
    """Fixed random split of the 40 non-ESC-10 categories into (seen, held_out)."""
    with zipfile.ZipFile(C.ZIP_PATH) as z:
        cats = sorted(_meta(z)["category"].unique())
    order = np.random.default_rng(seed).permutation(len(cats))
    held = sorted(cats[i] for i in order[:N_HELD_OUT])
    seen = sorted(set(cats) - set(held))
    return seen, held


def build_seen_cache(force: bool = False):
    """YAMNet embeddings of the 'seen' categories, folds 1-4 (fold 5 stays unused here)."""
    out = C.CACHE / "other_seen_yamnet.npz"
    if out.exists() and not force:
        return out
    seen, _ = split_categories()
    model = embeddings.load_yamnet()
    embs, cats, folds = [], [], []
    with zipfile.ZipFile(C.ZIP_PATH) as z:
        meta = _meta(z)
        meta = meta[meta["category"].isin(seen) & (meta["fold"] <= 4)]
        for row in meta.itertuples():
            wav, sr = sf.read(io.BytesIO(z.read(PREFIX + "audio/" + row.filename)), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            g = gcd(sr, C.SR)
            wav = resample_poly(wav, C.SR // g, sr // g).astype("float32")
            wav = np.pad(wav, (0, max(0, C.CLIP_SAMPLES - len(wav))))[: C.CLIP_SAMPLES]
            _, e = embeddings.embed_waveform(model, wav)
            embs.append(e)
            cats.append(row.category)
            folds.append(row.fold)
            if len(embs) % 100 == 0:
                print(f"  embedded {len(embs)} background clips", flush=True)
    np.savez_compressed(out, emb=np.stack(embs), category=np.array(cats), fold=np.array(folds))
    return out