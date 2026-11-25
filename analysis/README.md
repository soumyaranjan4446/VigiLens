# VigiLens: Data, Advancements & Benchmark Results Analysis

**Branch:** `improve/results-v1`  
**Verification Status:** `92/92` automated tests passing (`pytest -q`, `39.88s`)  
**Architectural Invariant Preserved:** Strictly normal-only training, strictly causal streaming (`1`-frame latency, zero lookahead), and unmodified 5-stage neural dimensions (`2048 -> 512 -> 256 -> 512`).

---

## Executive Overview

This folder documents the complete quantitative and engineering trajectory of **VigiLens** across three evaluation stages:

1. **Multimodal Leakage-Free Validation (VidLLVIP, Phases 0–3):** Diagnosed 8 methodology/statistical pathologies (`P1`–`P8`), established a strict 4-way scene-disjoint 5-fold Leave-Scene-Out cross-validation benchmark with frozen holdout (`scenes 09, 13`), and lifted paired `rgb+ir` Event Recall from **`0.0%` to `52.8% ± 13.1%`** in 15-run CV (**`50.0%`** on frozen holdout).
2. **Cross-Dataset Domain Transfer (VidLLVIP -> UCSD Ped2, RGB-Only Zero-Shot):** Transferred the night/urban multimodal model (`checkpoints/best_model.pth`) to an overhead daylight pedestrian walkway with IR disabled. Empirically demonstrated why **in-domain single-camera fine-tuning fails (`65.03% -> 54.08%` AUC)** and solved cross-domain detection without retraining using **three mathematical advancements**:
   - **Unit $L_2$-Hypersphere Feature Normalization** on Market-1501 Re-ID embeddings (**`52.24% -> 63.96%` AUC**, distant cyclist `Test001` reaching **`96.40%` AUC**).
   - **Dual-Backbone Z-Score Ensembling** (`0.3 * Z_ImageNet + 0.7 * Z_Market1501`), lifting global UCSD Ped2 ROC-AUC to **`71.45%`** and AP to **`80.89%`**.
   - **Dual-Threshold Alert Hysteresis (`H = 15` frames)**, lifting precision from **`45.92%` -> `97.8%`**, cutting false positive rate from **`7.22%` -> `0.95%`** (`87%` reduction, with **`0/727` post-warmup false positives** across all remaining normal frames), and extending target tracking recall up to **`6.8x`–`11.2x`**.
3. **Production Streaming & Interactive UI Integration (`app.py` / `pipeline/inference.py`):** Brought full `norm_mse` + `diag_variance` parity into live streaming (`models/anomaly_model.py`) and added interactive sensitivity override controls in the Streamlit sidebar.

---

## Headline Progression Summary

### A. VidLLVIP Multimodal Benchmark (Tier-A Pseudo-Anomalies, `rgb+ir`)

| Milestone / Stage | Eval Protocol | Global ROC-AUC | Global AP | FPR @ 95% TPR | Event Recall | Mean Delay |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1 Baseline (`baseline_v0`)** | 15 runs CV (5 folds × 3 seeds) | `0.6077 ± 0.0200` | `0.3230` | `0.6676 ± 0.0503` | `0.0%` (0/events) | N/A |
| **Phase 2 Winning Recipe (`exp_phase2_winning`)** | 15 runs CV (5 folds × 3 seeds) | **`0.6140 ± 0.0219`** | **`0.3237`** | **`0.6623 ± 0.0577`** | **`52.8% ± 13.1%`** | **`13.8 fr`** (~0.55s) |
| **Phase 3 Frozen Holdout (`final_production_model`)** | Scenes `09, 13` (3,500 frames) | `0.5784` (`ir`: `0.5993`) | `0.3213` (`ir`: `0.3607`) | `0.6929` | **`50.0%` (12/24)** | `18.5 fr` (~0.74s) |

### B. UCSD Ped2 Zero-Shot Cross-Dataset Benchmark (`2,010` Test Frames, RGB-Only)

| Milestone / Stage | ROC-AUC | AP | Precision | Frame Recall | FPR | Event Recall | Mean Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot VidLLVIP Thr (`3.2326`)** | `65.03%` | `75.99%` | `0.00%` | `0.00%` | `0.00%` | `0.0% (0/9)` | N/A |
| **Site-Calibrated ImageNet (`norm_mse`, `p95`)** | `63.39%` | `70.18%` | `45.92%` | `3.53%` | `7.22%` | `77.8% (7/9)` | `1.06s` |
| **In-Domain Fine-Tuned** *(Negative Result)* | `54.08%` | `65.06%` | `58.62%` | `2.66%` | `3.27%` | `77.8% (7/9)` | `6.86s` |
| **Unified Ultra-Precision** (`α=0.7, p=(95,80), H=15`) | **`71.45%`** (`71.4%`) | **`80.89%`** | **`97.8%`** | `23.9%` (`6.8x`) | **`0.95%`** | `77.8% (7/9)` | `1.11s` |
| **Unified High-Catch** (`α=0.6, p=(85,70), H=15`) | **`70.8%`** | **`80.98%`** | **`86.4%`** | **`35.7%`** (`10.1x`) | `9.81%` | **`88.9% (8/9)`** | **`0.76s`** |
| **Unified Full-Recall** (`α=0.6, p=(75,70), H=15`) | **`70.8%`** | **`80.98%`** | **`85.6%`** | **`39.6%`** (`11.2x`) | `11.58%` | **`100.0% (9/9)`** | `1.22s` |

---

## Documents in This Folder

| Document | Contents |
| :--- | :--- |
| [01_DATASETS_AND_EVALUATION.md](01_DATASETS_AND_EVALUATION.md) | Complete dataset inventory (Synthetic Smoke, VidLLVIP splits & Tier-A pseudo-anomalies, UCSD Ped2 train/test composition & anomaly types) and leakage-audit fixes. |
| [02_ENGINEERING_ADVANCEMENTS.md](02_ENGINEERING_ADVANCEMENTS.md) | Technical deep-dive into pathologies `P1`–`P8`, mask-flicker refutation, catastrophic forgetting under fine-tuning, Unit $L_2$ normalization, dual-backbone Z-score ensembling, hysteresis state machine, and Streamlit streaming parity. |
| [03_RESULTS_AND_BENCHMARKS.md](03_RESULTS_AND_BENCHMARKS.md) | Full numerical benchmark tables across VidLLVIP CV, frozen holdout, UCSD Ped2 domain-transfer ablations, alpha/window sensitivity sweeps, threshold grid search, per-video `Test001–Test012` breakdown, and test suite verification. |
