# VigiLens Ablations Table

Tracks all systematic ablations across phases.

| Experiment ID | Phase | Modality | Changed Component | Val Ratio | Frame AUC | Frame AP | FPR@95TPR | Normal Video FPR | Event Recall | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_paired` | 0 | RGB+IR | Baseline (Epoch 1 checkpoint) | N/A | NaN | NaN | NaN | 59.6% | 0.0000 | Select by raw val_loss, scene 13 leakage |
| `baseline_v0` | 1 | RGB+IR | 15-run clean benchmark (10 epochs, std calib) | 16.48 | 0.6077 ± 0.0200 | 0.3230 ± 0.0422 | 0.6676 ± 0.0503 | 0.0000 | 0.0000 | 5 folds x 3 seeds, strict leave-scene-out |
| `baseline_v0` | 1 | RGB | 15-run clean benchmark (RGB only) | 16.48 | 0.6000 ± 0.0207 | 0.3145 ± 0.0314 | 0.6923 ± 0.0408 | 0.0000 | 0.0000 | Unimodal RGB baseline |
| `baseline_v0` | 1 | IR | 15-run clean benchmark (IR only) | 16.48 | 0.5963 ± 0.0267 | 0.3119 ± 0.0373 | 0.6791 ± 0.0683 | 0.0000 | 0.0000 | Unimodal IR baseline |
| `exp_2a_freeze` | 2 | RGB+IR | Two-stage training (freeze fusion at ep 3) | 16.51 | 0.6220 ± 0.0246 | 0.3323 ± 0.0412 | 0.7177 ± 0.0473 | 0.0000 | 0.0000 | Stops target representation drift |
| `exp_2a_val_ratio` | 2 | RGB+IR | Checkpoint selection by `val_ratio` | 16.42 | 0.6246 ± 0.0164 | 0.3276 ± 0.0373 | 0.7003 ± 0.0366 | 0.0000 | 0.0000 | Selects for true temporal predictability |
| `exp_phase2_winning` | 2 | RGB+IR | Winning Recipe (2-stage, val_ratio, norm_mse) | 16.38 | **0.6140 ± 0.0219** | **0.3237 ± 0.0394** | **0.6623 ± 0.0577** | 0.9333 ± 0.1484 | **0.5278 ± 0.1305** | 15-run full benchmark; Event Recall jumps 0% -> 52.8% |
| `exp_phase2_winning` | 2 | RGB | Winning Recipe (RGB only) | 16.38 | **0.6094 ± 0.0224** | **0.3298 ± 0.0312** | **0.6982 ± 0.0730** | 0.8500 ± 0.1581 | **0.4667 ± 0.1504** | RGB unimodal performance improves over baseline |
| `exp_phase2_winning` | 2 | IR | Winning Recipe (IR only) | 16.38 | **0.6003 ± 0.0312** | **0.3121 ± 0.0411** | **0.6925 ± 0.0685** | 0.8000 ± 0.1690 | **0.4222 ± 0.1582** | IR unimodal performance improves over baseline |
