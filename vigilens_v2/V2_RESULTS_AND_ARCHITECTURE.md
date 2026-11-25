# VigiLens V2 — Architecture Advancements & Verified Results

## 1. Preserving V1 Identity While Fixing Its Core Bottlenecks

**V1 Pipeline (`app.py`, `models/anomaly_model.py`):**
`YOLOv8n-seg` (single union mask) → Masked Global `ResNet-50 (2048-D)` → `ModalityAwareFusion (512-D)` → `Causal 2-Layer LSTM (256-D)` → `NextFeaturePredictor (512-D)` → L2 Prediction Error → Causal Smoothing → Threshold.

**Diagnosed V1 Limitations Solved in V2:**
1. **Ped2 Ground-Truth Bug (`frameXXX.bmp`):** Original `scripts/prepare_ucsd_ped2.py` failed to match `Test006_gt`, `Test009_gt`, and `Test012_gt` mask filenames (`frame001.bmp` instead of `001.bmp`), wrongly treating **372 real anomaly frames** as normal (`0`) and hiding failures on those sequences. Fixed cleanly in `vigilens_v2/training/data_gt.py` (`load_ped2_test_labels`, verified against `UCSDped2.m`: all 12 test clips contain anomalies totaling **1,648 anomalous + 362 normal frames**).
2. **Global Crowd Dilution & Missing Localization:** V1 collapsed all detections into one union mask and performed global average pooling, letting 10+ normal walking pedestrians dilute one fast cyclist/cart and preventing bounding-box localization.
3. **Motion Blindness on Monocular CCTV:** Single-frame RGB `ResNet-50` embeddings cannot reliably separate speed/motion anomalies from appearance.
4. **Frame-0 Warmup Artifact:** V1 included `raw[0] = 0.0` inside the trailing moving average (`causal_moving_average`), creating an artificial trough across frames `1..6` and a sudden false-alarm spike at `frame 7`.
5. **Streamlit Session State Contention:** Caching stateful trackers or optical-flow buffers in `@st.cache_resource` mixed frames across concurrent tabs/reruns. V2 caches **only immutable weights** (`load_immutable_weights`) and isolates every stream cleanly (`build_isolated_pipeline`).

---

## 2. VigiLens V2 Architectural Upgrades

1. **Causal Object-Centric Tracking (`YOLOCausalTracker`):** Associates individual detections across frames via Hungarian IoU + centroid proximity (`CausalIoUTracker`), preserving per-entity identities without future lookahead.
2. **Per-Track Causal Dynamics LSTM (`TrackDynamicsModel`):** Models each tracked pedestrian's perspective-normalized trajectory `[p90_flow, p95_flow, mean_flow, bbox_disp, bh, bw, aspect_ratio, flow_dispersion]` relative to linear `y`-perspective curves fitted strictly on normal `Train001–Train004`. Outputs both per-track anomaly Z-scores (driving **RED vs GREEN bounding boxes**) and dilution-free scene maximums.
3. **Perspective-Normalized Flow Grid & Dual-Stream `VigiLensV2Model`:** Combines `TrackDynamicsModel` (`0.56`), perspective-row-normalized Farneback `12x18` Optical Flow Grid (`0.24`), and V1's pretrained `ModalityAwareFusion + Causal 2-Layer LSTM` Scene Appearance (`0.20`, pairing ImageNet RGB + unit-$L_2$ Market-1501 Re-ID or Thermal IR).
4. **Warmup-Safe Smoothing & Dual-Threshold Hysteresis (`smoothing.py`):** `causal_moving_average_skip0` removes the frame-7 warmup spike, while `CausalHysteresisDetector` (`hold_frames = 15`) prevents flickering mid-event alerts.

---

## 3. Verified Quantitative Results

### 3.1 UCSD Ped2 Benchmark (Official Corrected GT — All 12 Test Clips: `1,648` Anomaly / `362` Normal Frames)

Source: `vigilens_v2/results/ucsd_ped2_v2_results.json` (trained and calibrated **strictly on Normal `Train001–Train004`**, zero test leakage):

| Model / Variant | Global Frame ROC-AUC | Global Average Precision (AP) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **V1 Baseline (`Unified ImageNet + ReID-Unit`)** | `69.11%` | `86.57%` | Fails on dense/distant scenes (`Test003=34.9%`, `Test005=12.3%`, `Test006=17.4%`); no boxes |
| **V2 Scene Appearance (`ImageNet + ReID-Unit`)** | `69.60%` | — | Warmup fix stabilized frame-0 boundary |
| **V2 Flow Grid (`12x18` row-normalized)** | `70.76%` | — | Spatial velocity baseline independent of appearance |
| **V2 `TrackDynamicsModel` (`Per-Track Causal LSTM`)** | `86.65%` | — | Eliminates crowd dilution & enables per-track RED boxes |
| **Unified VigiLens V2 (`Track + Grid + Appearance`)** | **`87.33%`** (`+18.22 pts`) | **`97.23%`** (`+10.66 pts`) | **Best overall discrimination + object localization** |

#### Operating Regimes on `Train001–Train004` Calibrated Percentiles (`H = 15` Hysteresis)

| Operating Regime | Threshold `(high / low)` | Precision | Recall | FPR | F1 Score | Event Recall | Mean Delay |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated (`p95 / p80`)** | `1.2300 / 0.6828` | **`99.23%`** | `70.69%` | **`2.49%`** | `0.8257` | **`12/12 (100.0%)`** | `7.67 frames` |
| **Balanced High-Catch (`p92 / p75`)** | `1.0426 / 0.5614` | `98.21%` | `73.30%` | `6.08%` | `0.8395` | **`12/12 (100.0%)`** | `7.42 frames` |
| **Full-Sensitivity (`p85 / p70`)** | `0.7912 / 0.4795` | `96.61%` | **`77.73%`** | `12.43%` | **`0.8615`** | **`12/12 (100.0%)`** | **`6.00 frames`** |

#### Per-Sequence AUC Progression (`V1 -> V2` on Mixed-Label Test Sequences)

| Sequence | Event Segment | `V1` ROC-AUC | `V2` ROC-AUC | `V2` Track-Only AUC | Delta (`V2 - V1`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Test001` | `[60, 180)` | `84.97%` | **`85.79%`** | `86.43%` | `+0.82 pts` |
| `Test002` | `[94, 180)` | `82.25%` | **`99.94%`** | `99.93%` | **`+17.69 pts`** |
| `Test003` | `[0, 146)` | `34.93%` | **`91.78%`** | `90.24%` | **`+56.85 pts`** |
| `Test004` | `[30, 180)` | `76.24%` | **`96.42%`** | `95.89%` | **`+20.18 pts`** |
| `Test005` | `[0, 129)` | `12.33%` | **`92.99%`** | `89.96%` | **`+80.66 pts`** |
| `Test006` | `[0, 159)` | `17.37%` | **`89.04%`** | `90.84%` | **`+71.67 pts`** |
| `Test007` | `[45, 180)` | `83.65%` | **`92.64%`** | `86.30%` | **`+8.99 pts`** |
| `Test012` | `[87, 180)` | `81.09%` | `69.92%` | `78.21%` | `-11.17 pts` (Track `78.2%`) |

*(Note: `Test008`, `Test009`, `Test010`, and `Test011` are `100%` anomalous across every frame `[0, N)`, so clip-level ROC-AUC is undefined, though all 4 events are captured in global AUC and `12/12` Event Recall.)*

---

### 3.2 VidLLVIP Multimodal Frozen Holdout (`Scenes 09 & 13`, Paired `RGB + IR`, `3,500` Frames)

Source: `vigilens_v2/results/vidllvip_holdout_v2_results.json`

| Scoring Method | Frame ROC-AUC | Average Precision (AP) | Delta |
| :--- | :---: | :---: | :---: |
| **V1 `norm_mse` (`checkpoints/best_model.pth`)** | `58.08%` | `32.19%` | Baseline |
| **V2 (`Pred + Normal Manifold + Transition Delta + Skip0`)** | **`60.62%`** | **`38.94%`** | **`+2.54 pts` AUC / `+6.75 pts` AP** |

---

## 4. Verification Summary

- **Automated Unit Tests (`pytest -q vigilens_v2/tests tests`):** **`102/102` passed in `12.95s`** (92 original V1 tests + 10 V2 tests covering `frameXXX.bmp` GT loading, tracker determinism, streaming vs offline TrackLSTM parity, warmup-safe smoothing, hysteresis state machine, RED/GREEN overlay rendering, and cross-session optical-flow state isolation).
- **Live Browser UI Verification (`vigilens_v2/app.py` at `http://localhost:8501` on `Ped2 Test004`):** Confirmed normal walkers stay GREEN (`-0.40` Z-score, `0` false alarms in frames `0–29`) and the entering utility vehicle is sharply localized inside a RED bounding box (`ANOMALY ID:41 (7.04)`, Fused Z-Score `4.02`, Streaming `4.8 FPS`).
