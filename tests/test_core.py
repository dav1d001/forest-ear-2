"""Fast tests: pure logic on small synthetic inputs, plus one check that ties the saved
classifier to the numbers reported in results/ (skipped if the embedding cache is missing)."""
import json

import joblib
import numpy as np
import pytest

from src import config as C
from src import detect, evaluate, score_events, synth, zero_shot
from src.classifier import clip_features

CHAINSAW, FIRE, HELI, RAIN = (C.CLASSES.index(c) for c in
                              ("chainsaw", "crackling_fire", "helicopter", "rain"))
HALF_HOP = detect.HOP_FRAMES * detect.FRAME_S / 2


def make_proba(top_classes, p=0.9):
    """One row per window: probability p on the given class, the rest spread evenly."""
    rows = []
    for k in top_classes:
        row = np.full(len(C.CLASSES), (1 - p) / (len(C.CLASSES) - 1))
        row[k] = p
        rows.append(row)
    return np.array(rows)


def centres_for(n):
    return np.array([i * detect.HOP_FRAMES * detect.FRAME_S + C.CLIP_SECONDS / 2 for i in range(n)])


def ev(start, end, threat=True, label="chainsaw"):
    return {"start": start, "end": end, "threat": threat, "label": label}


class FakeHead:
    def predict_proba(self, X):
        return np.full((len(X), len(C.CLASSES)), 1 / len(C.CLASSES))


# ---------- detect.find_events ----------

def test_single_run_becomes_one_event():
    top = [RAIN, RAIN, CHAINSAW, CHAINSAW, RAIN]
    c = centres_for(len(top))
    events = detect.find_events(c, make_proba(top))
    assert len(events) == 1
    assert events[0]["label"] == "chainsaw"
    assert events[0]["start"] == pytest.approx(c[2] - HALF_HOP, abs=0.01)
    assert events[0]["end"] == pytest.approx(c[3] + HALF_HOP, abs=0.01)


def test_no_threat_means_no_events():
    top = [RAIN] * 5
    assert detect.find_events(centres_for(5), make_proba(top)) == []


def test_run_at_the_end_is_closed():
    top = [RAIN, HELI, HELI]
    c = centres_for(3)
    events = detect.find_events(c, make_proba(top))
    assert len(events) == 1
    assert events[0]["label"] == "helicopter"
    assert events[0]["end"] == pytest.approx(c[2] + HALF_HOP, abs=0.01)


def test_same_label_fragments_merge():
    top = [CHAINSAW, RAIN, CHAINSAW]          # gap of about 1 s between fragments
    events = detect.find_events(centres_for(3), make_proba(top))
    assert len(events) == 1


def test_far_apart_fragments_stay_separate():
    top = [CHAINSAW, RAIN, RAIN, RAIN, RAIN, CHAINSAW]   # gap of about 3.8 s
    events = detect.find_events(centres_for(6), make_proba(top))
    assert len(events) == 2


def test_back_to_back_different_threats_form_one_event_with_the_dominant_label():
    # Known limitation: consecutive flagged windows form one run whatever the threat type,
    # and the run takes the label with the most probability mass.
    top = [CHAINSAW, CHAINSAW, FIRE]
    events = detect.find_events(centres_for(3), make_proba(top))
    assert len(events) == 1
    assert events[0]["label"] == "chainsaw"


def test_threshold_is_applied():
    proba = make_proba([CHAINSAW], p=0.6)     # summed threat probability is about 0.69
    assert len(detect.find_events(centres_for(1), proba, threshold=0.5)) == 1
    assert detect.find_events(centres_for(1), proba, threshold=0.8) == []


# ---------- detect.window_probs ----------

def test_window_probs_counts_and_centres():
    emb = np.zeros((20, 1024), dtype="float32")
    centres, proba = detect.window_probs(emb, FakeHead())
    n_windows = len(range(0, 20 - detect.WINDOW_FRAMES + 1, detect.HOP_FRAMES))
    assert len(centres) == proba.shape[0] == n_windows
    assert proba.shape[1] == len(C.CLASSES)
    assert centres[0] == pytest.approx(C.CLIP_SECONDS / 2)
    assert centres[1] - centres[0] == pytest.approx(detect.HOP_FRAMES * detect.FRAME_S)


def test_window_probs_handles_a_very_short_recording():
    centres, proba = detect.window_probs(np.zeros((5, 1024), dtype="float32"), FakeHead())
    assert len(centres) == 1 and proba.shape == (1, len(C.CLASSES))


# ---------- evaluate.summarize ----------

def test_summarize_perfect_predictions():
    y = np.repeat(np.arange(len(C.CLASSES)), 3)        # 30 clips: 9 threat, 21 harmless
    m = evaluate.summarize(y, make_proba(y))
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["threats_flagged"] == "9/9"
    assert m["false_alarms"] == "0/21"


def test_summarize_counts_a_false_alarm():
    y = np.repeat(np.arange(len(C.CLASSES)), 3)
    pred = y.copy()
    pred[np.where(y == RAIN)[0][0]] = CHAINSAW          # one harmless clip called a chainsaw
    m = evaluate.summarize(y, make_proba(pred))
    assert m["false_alarms"] == "1/21"
    assert m["threats_flagged"] == "9/9"
    assert m["accuracy"] == pytest.approx(29 / 30)


# ---------- score_events ----------

def test_barely_touching_prediction_is_not_a_hit():
    # The case that made the first scoring rule too lenient: a long false alarm that only
    # grazes the start of a real event.
    truth = {"a.wav": [ev(29.9, 34.9, label="crackling_fire")]}
    pred = {"a.wav": [ev(20.26, 30.82)]}
    r = score_events.score(truth, pred, hours=1.0)
    assert r["detected"] == "0/1"
    assert r["fa_background"] == 1


def test_well_aligned_prediction_is_a_hit():
    truth = {"a.wav": [ev(29.9, 34.9, label="crackling_fire")]}
    pred = {"a.wav": [ev(31.78, 34.66)]}
    r = score_events.score(truth, pred, hours=1.0)
    assert r["detected"] == "1/1"
    assert r["precision"] == 1.0
    assert r["fa_per_hour"] == 0


def test_false_alarms_are_split_by_cause():
    truth = {"a.wav": [ev(10, 15), ev(30, 35, threat=False, label="dog")]}
    pred = {"a.wav": [ev(10, 15), ev(31, 34)]}          # second one fires on the dog
    r = score_events.score(truth, pred, hours=0.5)
    assert r["fa_distractor"] == 1 and r["fa_background"] == 0
    assert r["precision"] == pytest.approx(0.5)
    assert r["fa_per_hour"] == pytest.approx(2.0)


def test_missed_threat_and_no_predictions():
    r = score_events.score({"a.wav": [ev(10, 15)]}, {"a.wav": []}, hours=1.0)
    assert r["detected"] == "0/1" and r["recall"] == 0 and r["precision"] == 0.0


# ---------- synth.place, clip_features, config ----------

def test_place_respects_spacing_and_bounds():
    starts = synth.place(np.random.default_rng(0), 5, taken=[])
    assert len(starts) == 5
    assert all(0 <= s <= synth.DURATION - C.CLIP_SECONDS for s in starts)
    s = sorted(starts)
    assert all(b - a >= 6.0 for a, b in zip(s, s[1:]))


def test_place_is_deterministic_for_a_seed():
    assert synth.place(np.random.default_rng(1), 3, []) == synth.place(np.random.default_rng(1), 3, [])


def test_clip_features_are_mean_and_max_over_frames():
    emb = np.random.default_rng(0).normal(size=(4, 10, 1024)).astype("float32")
    f = clip_features(emb)
    assert f.shape == (4, 2048)
    assert np.allclose(f[:, :1024], emb.mean(axis=1))
    assert np.allclose(f[:, 1024:], emb.max(axis=1))


def test_config_is_consistent():
    assert len(C.CLASSES) == len(set(C.CLASSES)) == 10
    assert C.THREAT <= set(C.CLASSES)
    train, val, test = set(C.TRAIN_FOLDS), set(C.VAL_FOLDS), set(C.TEST_FOLDS)
    assert not (train & val or train & test or val & test)


# ---------- zero-shot mapping ----------

def test_zero_shot_picks_the_mapped_label():
    names = ["Silence"] + sorted({n for labels in zero_shot.LABEL_MAP.values() for n in labels})
    scores = np.zeros((1, 3, len(names)))
    scores[0, 1, names.index("Helicopter")] = 0.8
    proba = zero_shot.class_scores(scores, names)
    assert proba.shape == (1, 10)
    assert proba[0].argmax() == C.CLASSES.index("helicopter")
    assert proba[0].sum() == pytest.approx(1.0)


def test_zero_shot_rejects_labels_missing_from_yamnet():
    with pytest.raises(SystemExit):
        zero_shot.class_scores(np.zeros((1, 3, 2)), ["Silence", "Speech"])


# ---------- the saved model ----------

def test_saved_head_has_expected_classes_and_valid_probabilities():
    saved = joblib.load(C.ROOT / "models" / "yamnet_head.joblib")
    assert saved["classes"] == C.CLASSES
    proba = saved["head"].predict_proba(np.random.default_rng(0).normal(size=(5, 2048)))
    assert proba.shape == (5, 10)
    assert proba.sum(axis=1) == pytest.approx(np.ones(5))


CACHE = C.CACHE / "yamnet_esc10.npz"


@pytest.mark.skipif(not CACHE.exists(), reason="run `python -m src.embeddings` first")
def test_saved_head_reproduces_the_reported_test_metrics():
    from src import embeddings
    from src.data import split_masks
    d = embeddings.load()
    test = split_masks(d["fold"])["test"]
    head = joblib.load(C.ROOT / "models" / "yamnet_head.joblib")["head"]
    m = evaluate.summarize(d["y"][test], head.predict_proba(clip_features(d["emb"][test])))
    reported = json.loads((C.ROOT / "results" / "head_metrics.json").read_text())
    assert m["accuracy"] == pytest.approx(reported["accuracy"])
    assert m["f1_macro"] == pytest.approx(reported["f1_macro"])
    assert m["threats_flagged"] == reported["threats_flagged"]