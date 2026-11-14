# Experiment Results Summary

## Headline Metrics Across Folds (Mean ± Std)

| Run ID | Modality | Folds/Seeds | Frame AUC | Frame AP | FPR@95%TPR | Normal Video FPR | False Alarms/hr | Event Recall | Mean Delay |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_v0` | `ir` | 15 | 0.5963 ± 0.0267 | 0.3119 ± 0.0373 | 0.6791 ± 0.0683 | 0.0000 | 0.0000 | 0.0000 | NaN |
| `baseline_v0` | `rgb` | 15 | 0.6000 ± 0.0207 | 0.3145 ± 0.0314 | 0.6923 ± 0.0408 | 0.0000 | 0.0000 | 0.0000 | NaN |
| `baseline_v0` | `rgb+ir` | 15 | 0.6077 ± 0.0200 | 0.3230 ± 0.0422 | 0.6676 ± 0.0503 | 0.0000 | 0.0000 | 0.0000 | NaN |

## Per-Anomaly-Type Breakdown (Frame AUC)

| Run ID | Modality | Anomaly Type | Frame AUC |
| :--- | :--- | :--- | :--- |
| `baseline_v0` | `ir` | `freeze` | 0.4843 ± 0.0442 |
| `baseline_v0` | `ir` | `jumpcut` | 0.5691 ± 0.0503 |
| `baseline_v0` | `ir` | `reverse` | 0.6023 ± 0.0343 |
| `baseline_v0` | `ir` | `shuffle` | 0.6043 ± 0.0387 |
| `baseline_v0` | `ir` | `speedup_x2` | 0.6508 ± 0.0271 |
| `baseline_v0` | `ir` | `speedup_x4` | 0.6670 ± 0.0324 |
| `baseline_v0` | `rgb` | `freeze` | 0.4792 ± 0.0289 |
| `baseline_v0` | `rgb` | `jumpcut` | 0.5719 ± 0.0467 |
| `baseline_v0` | `rgb` | `reverse` | 0.6085 ± 0.0308 |
| `baseline_v0` | `rgb` | `shuffle` | 0.6096 ± 0.0321 |
| `baseline_v0` | `rgb` | `speedup_x2` | 0.6549 ± 0.0206 |
| `baseline_v0` | `rgb` | `speedup_x4` | 0.6762 ± 0.0258 |
| `baseline_v0` | `rgb+ir` | `freeze` | 0.4888 ± 0.0349 |
| `baseline_v0` | `rgb+ir` | `jumpcut` | 0.5762 ± 0.0426 |
| `baseline_v0` | `rgb+ir` | `reverse` | 0.6153 ± 0.0320 |
| `baseline_v0` | `rgb+ir` | `shuffle` | 0.6162 ± 0.0350 |
| `baseline_v0` | `rgb+ir` | `speedup_x2` | 0.6643 ± 0.0230 |
| `baseline_v0` | `rgb+ir` | `speedup_x4` | 0.6856 ± 0.0292 |
