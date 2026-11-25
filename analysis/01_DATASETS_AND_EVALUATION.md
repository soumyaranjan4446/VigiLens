# 01. Datasets, Splitting Protocol & Leakage Fixes

This document details every dataset used across VigiLens development, validation, cross-domain evaluation, and live Streamlit demonstration, along with the data-leakage fixes introduced in Phase 0–1.

---

## 1. Dataset Inventory Overview

| Dataset | Modalities | Domain / Camera View | Purpose in VigiLens | Resolution & FPS | Scale Used |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Smoke** | Paired `RGB` + `IR` | Synthetic moving shapes | CI pipeline & unit/smoke verification | Synthetic clips | `4` train, `2` val, `4` eval (`384` eval frames) |
| **VidLLVIP** | Paired `RGB` + `IR` | Street-level multimodal pedestrian scenes (night/dusk/day) | Unsupervised multi-scene model training, 5-fold CV, & holdout benchmark | `1280 × 1024`, `25 FPS`, `5s` clips (`125` frames/clip) | `14` scenes (`894` paired clips total) |
| **UCSD Ped2** | Grayscale (`RGB` mode, `IR` disabled) | Overhead daylight pedestrian walkway | Unsupervised **zero-shot cross-dataset** evaluation & calibration | `360 × 240` (`Test` encoded at `10 FPS`) | `16` normal train clips (`Train001–016`); `12` test clips (`Test001–012`, `2,010` frames) |
| **Live Demo Clips** | `RGB` + optional `IR` | Selected UCSD Ped2 + Evaluation sequences | Interactive Streamlit dashboard testing (`demo_videos/`) | `10 FPS` | `5` curated demo clips |

---

## 2. VidLLVIP Corpus & Data Leakage Remediation

### 2.1 Why VidLLVIP Needed Structural Fixing (`P7` & `P8`)

Audit script [`scripts/audit_splits.py`](../scripts/audit_splits.py) exposed two methodological flaws in the initial VidLLVIP setup (`baseline_vidllvip_30ep`, 114 test clips / 14,250 frames):

1. **Pathology `P8` — Scene Leakage Across Splits:** Validation previously included scenes `[12, 13, 14]`, while the evaluation split (`held_out_09_13`) consisted of 45 clips from scene `09` and 69 clips from scene `13`. Thus **60.5% of test clips (`69/114`) leaked directly from scene `13` into threshold calibration**.
2. **Pathology `P7` — Unlabeled Normal-Only Test Set:** VidLLVIP contains normal pedestrian traffic with no native anomaly labels. Evaluating on 14,250 normal frames yielded `AUC = NaN` and `AP = NaN`, leaving only false positive rate (`4.18%` smoothed FPR / `59.65%` video FPR / `3,764.21` false alarms/hr on paired `rgb+ir`) without any recall measurement.

### 2.2 Clean 4-Way Scene-Disjoint Partition & 5-Fold Leave-Scene-Out CV

To eliminate all cross-split overlap while keeping final holdout scenes strictly untouched:

* **Clean Production Partition (`experiments/splits/production.json`):**
  * **Train:** Scenes `01, 02, 03, 04, 05, 06, 07, 08, 10, 11` (`10` scenes)
  * **Validation (Early Stopping / Model Selection):** Scene `12` (`1` scene)
  * **Calibration (Threshold & Diagonal Residual Variance):** Scene `14` (`1` scene)
  * **Frozen Holdout Test (Evaluated Once in Phase 3):** Scenes `09, 13` (`2` scenes, `28` clips / `3,500` frames with pseudo-anomalies)
* **5-Fold Leave-Scene-Out Cross-Validation (`experiments/splits/folds.json`):** Defined over the `12` development scenes (`01..08, 10, 11, 12, 14`), rotating `2` test scenes, `1` calibration scene, `1` validation scene, and `8` training scenes per fold across `3` random seeds (`42, 43, 44` = `15` runs per experiment). Holdout scenes `09` and `13` remained quarantined across all folds.

### 2.3 Controlled Tier-A Feature-Space Pseudo-Anomalies (`scripts/make_pseudo_anomalies.py`)

To measure true discriminative sensitivity (`AUC`, `AP`, `Event Recall`, `Mean Delay`) on held-out VidLLVIP scenes without modifying training data, six temporal Tier-A anomaly generators were applied strictly on held-out `test` sequences (`[20, 60]` frame segments after a `>=25` frame warmup):

| Anomaly Type | Mechanism | What It Probes |
| :--- | :--- | :--- |
| **`reverse`** | Reverses temporal frame order within segment `[s, e)` | Backward walking / motion reversal |
| **`speedup_x2`** | Subsamples frames with stride `2` | Rapid / running motion |
| **`speedup_x4`** | Subsamples frames with stride `4` | High-velocity intrusion |
| **`freeze`** | Replicates a single frame across `[s, e)` | Stalled / loitering / frozen feed |
| **`shuffle`** | Randomly permutes frame order inside `[s, e)` | Erratic / chaotic motion |
| **`jumpcut`** | Splices frame features from a disjoint scene into `[s, e)` | Abrupt scene/entity intrusion |

---

## 3. UCSD Ped2 Cross-Dataset Benchmark (`data/ucsd_ped2/manifest.json`)

To evaluate **zero-shot cross-dataset transfer** on genuine real-world surveillance anomalies without fine-tuning on target anomalies:

* **Training Pool (Used ONLY for Unsupervised Calibration / Negative-Result Test):** `Train001`–`Train016` (normal pedestrians walking casually on an elevated campus walkway).
  * Unsupervised Site-Calibration subset: First 4 normal clips (`Train001`–`Train004`, `596` non-zero transition frames at window=1 or full sequences at window=7).
  * In-Domain Fine-Tuning negative-result test: Split into `Train001`–`Train012` (`12` clips, ~1,800 frames train) and `Train013`–`Train016` (`4` clips, ~600 frames validation).
* **Test Evaluation Set (`Test001`–`Test012`):** `12` video sequences totaling **`2,010` frames** (`734` normal frames + `1,276` anomalous frames across `9` anomalous events and `3` completely normal sequences), evaluated in **RGB-only mode** (`ir=None`, testing single-modality resilience of the multimodal pipeline):

| Sequence ID | Ground-Truth Label | Frames (`N`) | Anomaly Segment `[start, end)` | Anomalous Frames | Normal Frames | Target Characteristics |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`Test001`** | Anomaly (Bicyclist) | `180` | `[60, 180)` | `120` | `60` | Distant background bicycle (`20px` scale) |
| **`Test002`** | Anomaly (Bicyclist) | `180` | `[94, 180)` | `86` | `94` | Distant background bicycle (`20px` scale) |
| **`Test003`** | Anomaly (Skateboarder) | `150` | `[0, 146)` | `146` | `4` | Skateboarder present from frame 0 |
| **`Test004`** | Anomaly (Bicyclist) | `180` | `[30, 180)` | `150` | `30` | Normal flow `0..29`, foreground cyclist `30+` |
| **`Test005`** | Anomaly (Bicyclist) | `150` | `[0, 129)` | `129` | `21` | Bicyclist active from frame 0 |
| **`Test006`** | **Normal Only** | `180` | `[]` | `0` | `180` | Crowd pedestrian flow (false-alarm stress test) |
| **`Test007`** | Anomaly (Skateboarder) | `180` | `[45, 180)` | `135` | `45` | Normal `0..44`, midground skateboarder `45+` |
| **`Test008`** | Anomaly (Service Cart) | `180` | `[0, 180)` | `180` | `0` | Full-clip anomalous vehicle/cart |
| **`Test009`** | **Normal Only** | `120` | `[]` | `0` | `120` | Pure normal pedestrian sequence |
| **`Test010`** | Anomaly (Skateboarder) | `150` | `[0, 150)` | `150` | `0` | Full-clip skateboarder intrusion |
| **`Test011`** | Anomaly (Bicyclist) | `180` | `[0, 180)` | `180` | `0` | Full-clip bicyclist intrusion |
| **`Test012`** | **Normal Only** | `180` | `[]` | `0` | `180` | Pure normal pedestrian sequence |
| **Total** | **`9` Anom / `3` Norm** | **`2,010`** | **`9` Events** | **`1,276`** | **`734`** | Overhead daylight campus walkway (`10 FPS`) |
