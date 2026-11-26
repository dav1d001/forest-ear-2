# ForestEar 2.0

Detecting chainsaw, crackling-fire and helicopter sounds in forest audio with pretrained YAMNet embeddings, and testing how well that holds up on long recordings with unknown background sound.

This is a follow-up to [ForestEar](https://github.com/dav1d001/forest-ear), which classified 5-second clips with a CNN trained from scratch. ForestEar 2.0 uses the same data and the same train/validation/test split, so the two are directly comparable.

## Questions and short answers

1. **Does a pretrained audio embedding beat the from-scratch CNN?** Yes, by about 6 points of accuracy on the held-out fold, once a small classifier is trained on top. With no training at all, YAMNet's own labels already match the CNN.
2. **What happens on long recordings with unknown background?** Most threats are found (28 of 31 at threshold 0.5), but between 38% and 48% of the predicted events are false alarms. Clean-clip accuracy overstates what to expect in the field.
3. **Does adding a "background" class fix that?** No. It lowered recall more than it lowered false alarms. This negative result is documented in section 3.

## 1. Clip classification (ESC-10 test fold, 80 clips)

| Metric | ForestEar v1 CNN | YAMNet zero-shot | YAMNet + trained head |
|---|---|---|---|
| Accuracy | 0.900 | 0.900 | **0.9625** |
| Macro F1 | 0.898 | 0.896 | **0.961** |
| ROC-AUC (macro, one-vs-rest) | 0.993 | 0.993 | **0.998** |
| Threat clips flagged (of 24) | 20 | 20 | **23** |
| False threat alarms (of 56 harmless clips) | 1 | 0 | **0** |
| Chainsaw recall | 0.62 | 0.88 | **1.00** |

v1 figures are copied from the ForestEar README. The other columns are in `results/zero_shot_metrics.json` and `results/head_metrics.json`.

- **Zero-shot** uses YAMNet's own 521 AudioSet labels with no training, for example "Chainsaw", "Fire" and "Helicopter", mapped to the 10 classes by taking the strongest frame.
- **Trained head** is a standardised logistic regression on the YAMNet embeddings of each clip. Its regularisation strength was chosen on the validation fold only. The test fold was used once.

Read these numbers with care:
- The test fold has 80 clips, so one clip is 1.25 points of accuracy. The gap to v1 is 5 clips.
- YAMNet was trained on AudioSet, which already contains chainsaws, helicopters and fire, so it brings prior knowledge the CNN did not have.
- The zero-shot label mapping was chosen after looking at YAMNet's top labels over all 400 clips, test fold included, so the zero-shot figure is slightly optimistic.

## 2. Long recordings with known answers

Real forest recordings with labelled events are not available here, so I built test recordings with known timestamps (`src/synth.py`):

- 12 recordings of 60 seconds each (12 minutes in total).
- Background built from clips of the 40 ESC-50 categories that are not in ESC-10 (wind, birds, engines, trains, saws and so on) at half level.
- 2 to 3 threat events and 1 to 2 harmless distractor sounds (dog, rain, rooster and so on) per recording, 31 threat events and 20 distractors in total.
- Event clips come only from the test fold, which the classifier never saw during training.

The detector (`src/detect.py`) runs YAMNet once over the recording, slides a 10-frame window (about 5 s) with a hop of about 1 s, and sums the probabilities of the three threat classes in each window. Consecutive windows at or above a threshold are merged into one event with start and end times. A predicted event counts as a hit if its IoU with a true threat is at least 0.2.

| Threshold | Threats detected | Recall | Predicted events | Precision | False alarms (distractor / background) | False alarms per hour |
|---|---|---|---|---|---|---|
| 0.3 | 27/31 | 0.87 | 47 | 0.53 | 9 / 13 | 110 |
| 0.5 | 28/31 | 0.90 | 50 | 0.52 | 8 / 16 | 120 |
| 0.7 | 22/31 | 0.71 | 34 | 0.62 | 6 / 7 | 65 |
| 0.9 | 7/31 | 0.23 | 12 | 0.50 | 0 / 6 | 30 |

All thresholds from 0.3 to 0.9 are in `results/event_detection.md`.

What this shows:
- Detection is good at moderate thresholds, but precision is low. At 0.5 there were 24 false alarms in 12 minutes of audio.
- Raising the threshold removes false alarms only by also missing real threats.
- The "per hour" column is extrapolated from 12 minutes of audio, so treat it as a relative measure, not a field estimate.
- The background here includes engines, airplanes, trains and hand saws, which sound like chainsaws and helicopters. The false alarm rate is probably higher than in a quiet forest, but I did not measure which background categories cause the false alarms, so that is a guess.
- A false alarm rate this high would need a human to review alerts.

## 3. Negative result: a background class

The v1 classifier must label every sound as one of its 10 classes. To reduce false alarms on unknown sounds, I trained an 11-class head with an extra "other" class, using 30 of the 40 non-ESC-10 categories. The other 10 categories were held out completely and used as the only background in a new set of test recordings (seed 1), so neither head had heard any of them. The held-out categories are breathing, brushing_teeth, can_opening, car_horn, cow, fireworks, hen, insects, mouse_click and toilet_flush.

On the ESC-10 test clips, the 11-class head was worse (`results/background_head_metrics.json`):

| | 10-class head | 11-class head |
|---|---|---|
| Accuracy | 0.9625 | 0.850 |
| Threat clips flagged (of 24) | 23 | 16 |
| Clips labelled "other" | not applicable | 10 of 80 |

On the long recordings with held-out background (`results/background_class.md`):

| Head | Threshold | Threats detected | Recall | Precision | False alarms (distractor / background) |
|---|---|---|---|---|---|
| 10-class | 0.5 | 28/31 | 0.90 | 0.62 | 3 / 12 |
| 10-class | 0.7 | 25/31 | 0.81 | 0.69 | 1 / 10 |
| 11-class | 0.3 | 18/31 | 0.58 | 0.63 | 3 / 7 |
| 11-class | 0.5 | 14/31 | 0.45 | 0.64 | 1 / 7 |

The 11-class head has fewer false alarms, but at similar false alarm rates it finds far fewer threats. For example, the 10-class head at threshold 0.7 has recall 0.81 and precision 0.69, against 0.58 and 0.63 for the 11-class head at 0.3. It became more cautious without becoming more accurate, so the 10-class head stays the default.

Possible reasons, which I have not tested:
- Both heads are trained on clean single-sound clips, while the long recordings contain a threat mixed with background.
- "Other" combines 30 very different sounds into one class, which a linear head separates poorly.

This is one random split of the background categories and 31 threat events, so small differences are not reliable. The drop in clip accuracy and recall is large enough to trust. Note that the 10-class head scores better on this recording set (precision 0.62 at 0.5) than on the one in section 2 (0.52), because the held-out categories are less like chainsaws and helicopters. Background difficulty matters a lot.

## Method

- **Data:** ESC-10, 400 five-second clips in 10 classes, from the ESC-50 corpus (Freesound clips). Official folds: 1 to 3 train, 4 validation, 5 test. All audio is resampled to 16 kHz mono.
- **Embeddings:** YAMNet (TensorFlow Hub) gives 10 frames per 5-second clip, each with 1024 embedding values and 521 AudioSet scores. They are computed once and cached.
- **Clip features:** the mean and the max of the 10 embeddings, giving 2048 values per clip.
- **Classifier:** standardisation plus logistic regression. The regularisation strength C was chosen from {0.001, 0.01, 0.1, 1, 10} on the validation fold.
- **Metrics (`src/evaluate.py`):** accuracy, macro precision/recall/F1, log loss, ROC-AUC, threat clips flagged and false alarms. The same function scores every approach.

## Limitations

- The test fold has only 80 clips.
- The long recordings are synthetic, built from the same dataset. They have not been checked against real forest audio.
- Many ESC clips contain a short sound padded with silence, but each event is labelled as a full 5 seconds. Event boundaries are approximate.
- The classifier is closed-set and trained on clean clips.
- `src/detect.py` accepts 16 kHz WAV files only.

## Reproduce

Python 3.12 is needed, because TensorFlow has no stable release for newer versions.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data/raw
curl -4 -L --fail -o data/raw/esc50.zip https://github.com/karolpiczak/ESC-50/archive/master.zip   # about 600 MB

python -m src.data               # extract ESC-10, resample, fixed folds
python -m src.embeddings         # YAMNet embeddings (downloads YAMNet on first run)
python -m src.classifier         # trained head, results/head_metrics.json
python -m src.zero_shot          # zero-shot baseline
python -m src.synth              # build the long test recordings in data/long/
python -m src.score_events       # event detection scores
python -m src.train_background   # 11-class experiment (embeds about 960 clips, a few minutes)
python -m src.background_eval    # compare both heads on held-out background

python -m src.detect path/to/recording.wav 0.5    # events with timestamps for one file
```

## Repository layout

```
src/
  config.py            paths, constants, fixed folds
  data.py              ESC-10 extraction and splits
  embeddings.py        YAMNet embeddings and cache
  classifier.py        clip features and logistic-regression head
  evaluate.py          shared metrics
  zero_shot.py         zero-shot baseline from AudioSet labels
  background.py        non-ESC-10 clips for test-recording background
  synth.py             synthetic long recordings with known events
  detect.py            sliding-window detector and event merging
  score_events.py      event-level scoring
  other_class.py,