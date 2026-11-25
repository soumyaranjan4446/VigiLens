# VigiLens V2 — Object-Centric Dual-Stream Anomaly Detection & Localization

Standalone, non-breaking V2 overhaul inside `vigilens_v2/` preserving V1's core architectural backbone (`YOLOv8n-seg` → `ResNet-50 (2048-D)` → `ModalityAwareFusion (512-D)` → `Causal 2-Layer LSTM (256-D)` → `NextFeaturePredictor (512-D)`) while eliminating crowd-dilution and motion-blindness bottlenecks via per-track causal kinematic modeling, optical-flow lifting, warmup-safe causal smoothing, dual-threshold hysteresis, and fixed official UCSD Ped2 ground-truth evaluation.

---

## Quick Commands (from repository root)

### 1. Launch Live Streamlit V2 Dashboard (with RED Per-Track Localization)
```powershell
.\.venv\Scripts\streamlit.exe run vigilens_v2/app.py --server.port 8501
```
- **Quick Presentation Presets**: Switch effortlessly between `Ped2 Test004` (utility cart + cyclist), `Test007` (skateboarder), `Test002` (cyclist in crowd), `Train001` (100% normal baseline), or `Paired RGB + Thermal IR`, OR upload custom `.mp4/.avi` files.
- **Per-Track Localization**: Normal pedestrians remain outlined in subtle **GREEN** (`ID:k`), while anomalous subjects trigger bold **RED** bounding boxes (`ANOMALY ID:k (score)`) alongside an interactive live score curve and threshold selector.

### 2. Run All Automated Tests (`102` tests across V1 + V2)
```powershell
.\.venv\Scripts\pytest.exe -q vigilens_v2/tests tests
```

### 3. Re-Train & Re-Calibrate V2 Strictly on Normal Training Clips
```powershell
.\.venv\Scripts\python.exe -m vigilens_v2.training.train_v2
```
Saves the trained `TrackDynamicsModel` + perspective baselines + normal-only z-score calibration into `vigilens_v2/artifacts/vigilens_v2_ped2.pt` and `vigilens_v2/results/ped2_v2_calibration.json`.

### 4. Re-Run Reproducible Benchmarks (UCSD Ped2 True GT + VidLLVIP Holdout)
```powershell
.\.venv\Scripts\python.exe -m vigilens_v2.training.evaluate_v2
```
Writes complete JSON results to:
- `vigilens_v2/results/ucsd_ped2_v2_results.json`
- `vigilens_v2/results/vidllvip_holdout_v2_results.json`

---

## Package Structure
```text
vigilens_v2/
├── app.py                          # Session-isolated live Streamlit V2 dashboard
├── config.py                       # V2 dataclasses (Tracker, Motion, ModelV2, Decision)
├── models/
│   ├── tracker.py                  # YOLOv8n-seg + Hungarian/IoU CausalIoUTracker
│   ├── motion_encoder.py           # Causal Farneback grid (12x18) + per-track kinematics + 2048-D lifter
│   ├── feature_extractor.py        # DualVisualExtractor (ImageNet RGB + unit-L2 Market-1501 ReID + IR)
│   ├── track_model.py              # Per-track Causal 2-Layer LSTM + perspective normal manifold
│   └── vigilens_v2_model.py        # Unified VigiLensV2Model (V1 backbone + TrackDynamicsModel + Grid)
├── pipeline/
│   ├── smoothing.py                # Warmup-safe causal_moving_average_skip0 + CausalHysteresisDetector
│   └── streaming_v2.py             # Strict 1-frame causal InferencePipelineV2 + RED/GREEN renderer
├── training/
│   ├── data_gt.py                  # Fixed UCSD Ped2 GT loader (supports XXX.bmp + frameXXX.bmp + UCSDped2.m)
│   ├── calibrate_v2.py             # Normal-only z-score standardization & hysteresis calibration
│   ├── train_v2.py                 # Normal-only TrackDynamicsModel trainer & calibrator
│   └── evaluate_v2.py              # Leak-free UCSD Ped2 & VidLLVIP frozen holdout evaluator
├── results/                        # Reproducible verified benchmark JSONs
└── tests/
    └── test_vigilens_v2.py         # Unit tests covering GT fix, tracker, LSTM parity, smoothing & UI safety
```
