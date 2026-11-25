# 02. Engineering Diagnostics & System Advancements

Every improvement in VigiLens was engineered under three strict constraints:
1. **Unchanged Neural Architecture:** Preserve `YOLOv8n-seg -> ResNet50 (2048-D) -> ModalityAwareFusion (512-D) -> Causal 2-Layer LSTM (256-D) -> NextFeaturePredictor (512-D)` with zero added layers or parameter bloat.
2. **Strict Temporal Causality:** Zero future-frame lookahead anywhere in LSTM states, feature scoring, moving-average smoothing (`W`), or alert hysteresis (`H`).
3. **Normal-Only Supervision:** Zero anomaly clips or anomaly labels ever touched training or calibration.

---

## Part I — Core Training & Calibration Pathologies Resolved (Phases 0–3, VidLLVIP)

Six systemic bottlenecks (`P1, P2, P4, P5, P6`, alongside split leakage `P8` and unlabeled evaluation `P7`) were diagnosed and resolved across commits `05dc7b0` → `48eccfa`:

### 1. Scale-Free Checkpoint Selection (`P1`) & Two-Stage Training (`P2`)
* **Root Cause:** Selecting checkpoints by raw Smooth L1 `val_loss` favored Epoch 1 weights or arbitrary scale shrinkage in `ModalityAwareFusion` embeddings rather than genuine temporal predictability. Simultaneously updating fusion projections alongside LSTM weights forced LSTM prediction targets `z_{t+1}` to drift continuously across epochs.
* **Fix:**
  1. Added persistence-normalized validation ratio in [`training/validate.py`](../training/validate.py):
     $$\text{val\_ratio} = \frac{\sum \text{MSE}(\hat{z}_{t+1}, z_{t+1})}{\max(\sum \text{MSE}(z_t, z_{t+1}), 10^{-12})}$$
  2. Added two-stage training (`--freeze-fusion-epoch 3`) in [`training/train.py`](../training/train.py), locking the `512-D` fused target manifold after Epoch 3 so the Causal LSTM converges on stationary targets (`weight_decay = 1e-2`).
* **Verified Effect:** Frame ROC-AUC on Fold 0 (`rgb+ir`) increased from `0.6077` (`baseline_v0`) to `0.6220` (`exp_2a_freeze`) and `0.6246` (`exp_2a_val_ratio`).

### 2. Empirical Refutation of Mask Flicker (`P4`) — Negative Result
* **Hypothesis Tested:** When YOLOv8n-seg switches between detecting a person (`person_found=True`) and falling back to whole-frame pooling (`person_found=False`), the sudden ROI mask change might inject false-alarm spikes.
* **Measurement ([`scripts/diagnose_mask_flicker.py`](../scripts/diagnose_mask_flicker.py)):** Analyzed `3,500` frames across `28` scenes. Mean embedding shift `||Δz||` on mask transitions was `2.80` vs `2.93` on stable frames, with point-biserial correlation `r = -0.012` against prediction error spikes.
* **Takeaway:** Mask dropout transitions were **not** the primary cause of false alarms; whole-frame fallback pooling preserves feature scale cleanly.

### 3. Diagonal Mahalanobis Error Weighting (`P5`) & Standardized Calibration (`P6`)
* **Root Cause:** Unweighted 512-D MSE penalized noisy, high-variance dimensions equally with stable kinematic dimensions, while skewed heavy-tailed normal errors drove parametric `mean + 3*std` thresholds above `1.37`, resulting in **`0.0%` Event Recall across all 15 baseline runs**.
* **Fix ([`training/scoring.py`](../training/scoring.py), [`training/calibration.py`](../training/calibration.py)):**
  * Compute per-dimension normal residual variance `σ_i^2` on disjoint normal calibration sequences and score using:
    $$\text{norm\_mse}(\hat{z}, z) = \frac{1}{512} \sum_{i=1}^{512} \frac{(\hat{z}_i - z_i)^2}{\sigma_i^2 + 10^{-4}}$$
  * Support percentile (`p95`), MAD, and split-conformal quantile calibration alongside parametric `std`.
* **Verified Effect:** Lifted 15-run VidLLVIP CV Event Recall from **`0.0000` (0%) → `0.5278 ± 0.1305` (52.8%)** and Frozen Holdout Event Recall (`scenes 09, 13`) to **`50.0%` (12/24 events)**.

---

## Part II — Cross-Domain Findings on UCSD Ped2 (`4bd4d2e` → `4aed624`)

Deploying `checkpoints/best_model.pth` (trained exclusively on VidLLVIP street scenes) directly to **UCSD Ped2** (overhead daylight walkway, RGB-only) revealed three pivotal domain-transfer lessons:

### 4. Uncalibrated Cross-Domain Threshold Collapse
* Applying VidLLVIP's threshold (`3.2326`) directly to UCSD Ped2 produced **`65.03%` ROC-AUC / `75.99%` AP**, yet **`0/9` (0.0%) Event Recall** and **`0.0%` Precision** because prediction error scale differs across camera perspectives and lighting domains (`mean = 18.03` under UCSD-normalized `norm_mse`, or `0.0896` under raw ImageNet MSE).
* Unsupervised **Site-Calibration** using only 4 normal clips (`Train001`–`Train004`) via [`scripts/site_calibrate_ucsd.py`](../scripts/site_calibrate_ucsd.py) immediately restored Event Recall to **`77.78%` (7/9)**, **but** with `ImageNet + norm_mse` it exhibited high false-alarm fatigue (`45.92%` precision, `7.22%` FPR, `3.53%` frame recall).

### 5. Negative Scientific Result: Single-Camera In-Domain Fine-Tuning Causes Catastrophic Forgetting
* **Experiment ([`scripts/train_ucsd_indomain.py`](../scripts/train_ucsd_indomain.py)):** Initialized from `checkpoints/best_model.pth` and fine-tuned for 10 epochs (`lr = 5e-5`, `freeze_fusion_epoch = 3`) strictly on normal UCSD clips `Train001`–`Train012` (validated on `Train013`–`Train016`).
* **Measured Outcome (`Evaluation/results_ucsd_indomain/results.json`):**
  * Global Frame ROC-AUC **collapsed from `65.03%` (uncalibrated) / `63.39%` (site-calibrated) down to `54.08%`** (near random chance).
  * Average Precision fell from **`70.18%` / `75.99%` down to `65.06%`**.
  * Mean Detection Delay degraded **$6.5\times$** from `10.57 frames` (`1.06s`) to **`68.57 frames` (`6.86s`)**; Frame Recall dropped to **`2.66%` (`F1 = 0.0510`)**.
* **Engineering Rule:** **Do not fine-tune deep recurrent anomaly predictors on small single-camera normal datasets.** Single-camera clips lack scene variance, causing the LSTM to memorize fixed walkway textures/speeds and forget generalized motion physics learned from VidLLVIP.

---

## Part III — The 3 Breakthrough Advancements Without Retraining (`2056313` → `c83530b`)

Keeping `checkpoints/best_model.pth` frozen, three complementary mathematical refinements in [`scripts/evaluate_unified_system.py`](../scripts/evaluate_unified_system.py), [`models/feature_extractor.py`](../models/feature_extractor.py), and live inference ([`c83530b`](../app.py)) resolved precision, distance bias, and alert continuity:

### Advancement 1: Unit $L_2$-Hypersphere Feature Normalization (Perspective & Distance Invariance)
* **Problem:** Standard ResNet50 GAP vector norms `||x||_2` correlate with pixel mass. Far-away intruders (`Test001`, `Test002` at ~50m) yielded small unnormalized feature vectors overwhelmed by close pedestrians, dropping raw Market-1501 Re-ID ROC-AUC to **`52.24%`** (MSE) / **`52.44%`** (`norm_mse`).
* **Solution:** Project the 2048-D Re-ID embedding onto the unit hypersphere before fusion:
  $$\bar{x}_t = \frac{x_t}{\|x_t\|_2 + \epsilon} \implies \|\bar{x}_t\|_2 = 1$$
  MSE on unit-length representations measures **angular velocity divergence** rather than object pixel area.
* **Key Ablation Discovery ([`scripts/test_variance_and_window.py`](../scripts/test_variance_and_window.py)):**
  * Single-backbone Re-ID ROC-AUC jumped **+11.72 points** from **`52.24%` (`ReID-raw`) → `63.96%` (`ReID-unit`)** (window=7, MSE), and **`65.14%` AUC / `77.28%` AP** at window=15.
  * Crucially, applying unit normalization to ImageNet *degraded* ImageNet AUC (`66.20% -> 62.10%`), and applying diagonal `norm_mse` to already unit-normalized Re-ID vectors *degraded* Re-ID AUC (`63.96% -> 58.49%` at window=7) by amplifying noisy low-variance coordinates. Hence the optimal pairing combines **`ImageNet (raw) MSE` + `Market-1501 Re-ID (unit L2) MSE`**.
  * Distant background cyclists rose to **`96.40%` AUC** on `Test001` and **`90.57%–91.44%` AUC** on `Test002`.

### Advancement 2: Dual-Backbone Z-Score Ensembling (`30% ImageNet + 70% Market-1501 Re-ID`)
* **Mechanism:** Site-calibrate mean `μ` and std `σ` from normal clips (`Train001`–`Train004`) independently per expert (`ImageNet: μ=0.0896, σ=0.0080`; `ReID-Unit: μ=0.1080, σ=0.0093`), then combine standardized anomaly Z-scores:
  $$Z_{\text{ens}}(t) = (1 - \alpha)\,Z_{\text{ImageNet}}(t) + \alpha\,Z_{\text{ReID-Unit}}(t)$$
* **Rationale:** ImageNet captures broad optical/scene stability while Market-1501 Re-ID captures fine-grained pedestrian gait and posture anomalies.
* **Result:** Global UCSD Ped2 ROC-AUC peaked monotonically at **`α = 0.7` (`71.45%` ROC-AUC, `80.89%` AP)** — outperforming every single-backbone variant (`66.20%` ImageNet-only, `63.96%` ReID-only).

### Advancement 3: Dual-Threshold Alert Hysteresis & Peak-Hold (`H = 15` frames)
* **Mechanism:** Trigger alert when smoothed ensemble score crosses upper threshold `T_high` (`p_high` percentile on normal calibration), and sustain active tracking for up to `H = 15` frames (`1.5s @ 10 FPS`) as long as score remains above lower threshold `T_low` (`p_low`).
* **Result:** Eliminates single-frame flicker during ongoing intrusions while preventing false alarms on normal frames:
  * **Ultra-Precision (`α=0.7, p=(95,80), H=15`):** Precision **`97.8%`**, FPR **`0.95%`** (just `7/734` normal frames — specifically a 1-frame post-warmup transient at `frame 7` across 7 clips, leaving **`0/727` post-warmup false positives**), Frame Recall **`23.9%`** (`6.8x` baseline), `F1 = 0.3841`.
  * **High-Catch (`α=0.6, p=(85,70), H=15`):** Event Recall **`88.9% (8/9)`**, Precision **`86.4%`**, Frame Recall **`35.7%`** (`10.1x` baseline), `F1 = 0.5055`, Mean Delay **`0.76s`**.
  * **Full-Recall (`α=0.6, p=(75,70), H=15`):** Event Recall **`100.0% (9/9)`**, Precision **`85.6%`**, Frame Recall **`39.6%`** (`11.2x` baseline), `F1 = 0.5413`, Mean Delay **`1.22s`**.

---

## Part IV — Live Streaming Inference Parity & Streamlit App Controls (`c83530b`)

Offline cached evaluation used `norm_mse` and `diag_variance`, whereas live streaming previously defaulted strictly to raw `mse`. Commit `c83530b` closed this production gap:

* [`models/anomaly_model.py`](../models/anomaly_model.py): Extended `VigiLensModel.process_frame()` with `score_variant` (`"mse"` / `"norm_mse"`) and `diag_variance` parameters so real-time frame-by-frame scoring computes exact diagonal Mahalanobis errors matching offline evaluation.
* [`pipeline/inference.py`](../pipeline/inference.py): Updated `InferencePipeline` to read `score_variant` and `diag_variance` automatically from `calibration.json` and pass them into `model.process_frame()`.
* [`app.py`](../app.py): Added **"Manual Sensitivity Override"** checkbox and **"Custom Threshold"** slider (`0.01` to `5.0`, default `0.20`, step `0.01`) in the Streamlit sidebar so operators can tune sensitivity live on new cameras without retraining or editing calibration files.
