"""Paths, constants and the fixed data split (same as ForestEar v1)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
ZIP_PATH = RAW / "esc50.zip"

SR = 16000                      # YAMNet expects 16 kHz mono
CLIP_SECONDS = 5
CLIP_SAMPLES = SR * CLIP_SECONDS

CLASSES = ["chainsaw", "clock_tick", "crackling_fire", "crying_baby", "dog",
           "helicopter", "rain", "rooster", "sea_waves", "sneezing"]
THREAT = {"chainsaw", "crackling_fire", "helicopter"}

# Official ESC folds: no clip from the same recording appears in two splits.
TRAIN_FOLDS = (1, 2, 3)
VAL_FOLDS = (4,)
TEST_FOLDS = (5,)               # held out, never used for any decision
