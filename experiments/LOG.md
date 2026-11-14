# VigiLens Experiment Log

All experiments, configurations, and phase milestones are logged here.

---

## [Phase 0] Setup, Baseline Audit, and Validation Metric Fix

### Date: 2026-11-14
### Git Branch: `improve/results-v1`
### Commit Base: `ae9c16a`

### Objectives:
1. Initialize dedicated branch `improve/results-v1` and ensure 100% test suite passing.
2. Implement scale-free `val_ratio` (prediction MSE / persistence MSE) in `training/validate.py` to fix Problem P1 (invalid `val_loss` selection).
3. Update `training/train.py` with:
   - `--input-noise`: Gaussian noise injected into LSTM inputs during training.
   - `--select-metric`: Select best checkpoint based on `val_ratio` or `val_loss`.
   - `--patience`: Early stopping patience.
   - `--freeze-fusion-epoch`: Two-stage training support (freeze fusion + disable auxiliary losses).
   - `--weight-decay`: Default updated to `1e-2`.
   - Persist `best_metric` and `select_metric` in checkpoint metadata and log `val_ratio` and `persistence_mse` in history.
4. Update `training/precompute.py`:
   - Store per-frame `person_found` (bool tensor) in each cache `.pt`.
   - Store `person_found_fraction` in the sidecar `.json`.
5. Write `scripts/audit_splits.py` to audit dataset splits for scene leakage and generate `experiments/SPLITS.md`.
6. Audit the baseline run from VidLLVIP (`C:\Users\Administrator\Downloads\VigiLens_VidLLVIP_full`) and record baseline metrics in `experiments/baseline/` and `experiments/results.csv`.

### Key Outcomes & Findings:
- **Test Suite**: 77 tests passing in 30.58s (`pytest -q`).
- **Validation Fix**: `validate_cached` now computes persistence baseline `MSE(fused[:, :-1], fused[:, 1:])` and scale-free ratio `sum(pred_mse) / max(sum(persistence_mse), 1e-12)`.
- **Precompute**: Caches and sidecars now track `person_found` and `person_found_fraction` for P4 analysis.
- **Problem P8 Confirmed**: Baseline audit revealed Scene 13 appeared in both `val` (threshold calibration) and `test` (evaluation). A clean 4-way disjoint partition (Train: 01-08, 10, 11; Val: 12; Calib: 14; Test: 09, 13) has been formalized in `experiments/SPLITS.md`.
- **Baseline Recorded**:
  - Paired (RGB+IR): Threshold = 0.5097, Frame FPR = 4.18% (smoothed) / 5.74% (raw), Video FPR = 59.6% (smoothed) / 76.3% (raw), False Alarms/hr = 3,764.2.
  - RGB-only: Frame FPR = 2.04%, Video FPR = 35.96%, False Alarms/hr = 1,831.6.
  - IR-only: Frame FPR = 1.08%, Video FPR = 32.46%, False Alarms/hr = 972.6.
  - Detection quality (AUC/AP/F1): NaN due to zero labeled anomalies in test set (Problem P7 confirmed).

### Status: Gate 0 PASSED. Ready for Phase 1.

---

## [Phase 1] Trustworthy Evaluation Protocol & Re-benchmark (`baseline_v0`)

### Date: 2026-11-14
### Git Branch: `improve/results-v1`

### Objectives:
1. Define 5 leave-scene-out cross validation folds over 12 scenes (`01..08, 10, 11, 12, 14`) with 4-way disjoint partitioning (`train`, `val`, `calib`, `test`). Scenes `09` and `13` strictly frozen for final holdout testing.
2. Implement leakage-free pseudo-anomaly generation (`scripts/make_pseudo_anomalies.py`) supporting Tier A (`reverse`, `speedup_x2`, `speedup_x4`, `freeze`, `shuffle`, `jumpcut`) and Tier B.
3. Extend evaluation and metric suites (`training/metrics.py`, `training/evaluate.py`) to compute macro frame AUC, AP, FPR@95%TPR, event recall, detection delay, and false alarms per hour.
4. Implement unified leave-scene-out experiment runner (`experiments/run_experiment.py`) and statistical summarizer (`experiments/summarize.py`).
5. Re-benchmark unmodified baseline recipe (`baseline_v0`: `val_loss` selection, `1e-5` weight decay, zero input noise, `mean + 3*std` calibration) across all 5 folds $\times$ 3 seeds (15 runs).

### Key Outcomes & Findings:
- **Disjoint Partitioning Verified**: Zero scene leakage across all 5 folds and frozen holdout (`pytest tests/test_audit_splits.py`).
- **Baseline Re-benchmark (`baseline_v0` across 15 runs)**:
  - **Paired (`rgb+ir`)**: Macro Frame AUC = `0.6077 ± 0.0200`, Frame AP = `0.3230 ± 0.0422`, FPR@95%TPR = `0.6676 ± 0.0503`.
  - **RGB-only**: Macro Frame AUC = `0.6000 ± 0.0207`, Frame AP = `0.3145 ± 0.0314`, FPR@95%TPR = `0.6923 ± 0.0408`.
  - **IR-only**: Macro Frame AUC = `0.5963 ± 0.0267`, Frame AP = `0.3119 ± 0.0373`, FPR@95%TPR = `0.6791 ± 0.0683`.
  - **Multimodal Synergy Confirmed**: Paired `rgb+ir` outperforms unimodal across all folds.
  - **Pathology P6 Empirically Confirmed**: Due to heavy-tailed error residuals driving `mean + 3*std` threshold to $\sim 1.40$, Event Recall is `0.0000` (0%) across all 15 runs.
- **Unit Test Suite**: 88 passed in 32.32s (`pytest -q`).

### Status: Gate 1 PASSED. Ready for Phase 2.
