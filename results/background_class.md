Background sounds in these recordings come only from categories neither head saw in training: breathing, brushing_teeth, can_opening, car_horn, cow, fireworks, hen, insects, mouse_click, toilet_flush

| Head | Threshold | Threats detected | Recall | Predicted | Precision | False alarms (distractor / background) | False alarms per hour |
|---|---|---|---|---|---|---|---|
| 10-class | 0.3 | 29/31 | 0.94 | 45 | 0.58 | 9 / 10 | 95 |
| 10-class | 0.5 | 28/31 | 0.90 | 40 | 0.62 | 3 / 12 | 75 |
| 10-class | 0.7 | 25/31 | 0.81 | 35 | 0.69 | 1 / 10 | 55 |
| 10-class | 0.9 | 10/31 | 0.32 | 15 | 0.67 | 1 / 4 | 25 |
| 11-class (+ background) | 0.3 | 18/31 | 0.58 | 27 | 0.63 | 3 / 7 | 50 |
| 11-class (+ background) | 0.5 | 14/31 | 0.45 | 22 | 0.64 | 1 / 7 | 40 |
| 11-class (+ background) | 0.7 | 9/31 | 0.29 | 15 | 0.60 | 0 / 6 | 30 |
| 11-class (+ background) | 0.9 | 7/31 | 0.23 | 10 | 0.70 | 0 / 3 | 15 |
