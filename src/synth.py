"""Build synthetic long 'forest' recordings with known threat-event timestamps."""
import json

import numpy as np
import soundfile as sf

from . import background, data
from . import config as C

DURATION = 60            # seconds per recording (12 background clips of 5 s)
N_RECORDINGS = 12
OUT = C.ROOT / "data" / "long"


def place(rng, count, taken):
    """Random event start times (s), at least 6 s apart from each other."""
    starts = []
    while len(starts) < count:
        s = float(rng.uniform(0, DURATION - C.CLIP_SECONDS))
        if all(abs(s - t) >= 6.0 for t in taken + starts):
            starts.append(s)
    return starts


def build(seed: int = 0):
    rng = np.random.default_rng(seed)
    d = data.load()
    test = data.split_masks(d["fold"])["test"]
    X, y = d["X"][test], d["y"][test]            # held-out clips only
    bg, _ = background.load_other(fold=5)
    threat_idx = [C.CLASSES.index(c) for c in sorted(C.THREAT)]
    n = DURATION * C.SR
    OUT.mkdir(parents=True, exist_ok=True)

    truth = {}
    for r in range(N_RECORDINGS):
        audio = np.zeros(n, dtype="float32")
        for start in range(0, n, C.CLIP_SAMPLES):      # unknown-sound background, low level
            audio[start:start + C.CLIP_SAMPLES] += 0.5 * bg[rng.integers(len(bg))]

        events, taken = [], []
        for is_threat, count in ((True, int(rng.integers(2, 4))), (False, int(rng.integers(1, 3)))):
            pool = np.where(np.isin(y, threat_idx) == is_threat)[0]
            starts = place(rng, count, taken)
            taken += starts
            for s in starts:
                i = int(rng.choice(pool))
                a = int(s * C.SR)
                audio[a:a + C.CLIP_SAMPLES] += X[i]
                events.append({"start": round(s, 2), "end": round(s + C.CLIP_SECONDS, 2),
                               "label": C.CLASSES[int(y[i])], "threat": is_threat})

        peak = float(np.abs(audio).max())
        if peak > 0.95:
            audio *= 0.95 / peak
        name = f"rec_{r:02d}.wav"
        sf.write(OUT / name, audio, C.SR, subtype="PCM_16")
        truth[name] = sorted(events, key=lambda e: e["start"])

    (OUT / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


if __name__ == "__main__":
    t = build()
    events = [e for ev in t.values() for e in ev]
    print(f"{len(t)} recordings | {len(events)} events | {sum(e['threat'] for e in events)} threat events")