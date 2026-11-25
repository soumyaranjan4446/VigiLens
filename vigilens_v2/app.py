"""VigiLens V2 Real-Time Object-Centric Video Anomaly Detection & Localization Dashboard.

Launch from repository root:
    streamlit run vigilens_v2/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = Path(__file__).resolve().parent
# Keep package dir out of sys.path[0] so `from models.*` resolves to VigiLens root cleanly
sys.path = [p for p in sys.path if Path(p or ".").resolve() != PKG_DIR]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vigilens_v2.config import get_v2_config
from vigilens_v2.models.feature_extractor import DualVisualExtractor
from vigilens_v2.models.motion_encoder import CausalMotionFlowEncoder
from vigilens_v2.models.tracker import YOLOCausalTracker
from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model
from vigilens_v2.pipeline.streaming_v2 import InferencePipelineV2


st.set_page_config(
    page_title="VigiLens V2 — Real-Time Anomaly Localization",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_immutable_weights(use_reid: bool, load_ir: bool) -> Dict[str, Any]:
    """Load ONLY immutable/read-only model weights & projection matrices once.

    All stateful trackers, optical-flow frame buffers (_prev_gray), and LSTM
    hidden states are instantiated fresh per streaming run to prevent cross-tab
    or cross-rerun state corruption.
    """
    from ultralytics import YOLO

    cfg = get_v2_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = ROOT / cfg.model.v2_checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    else:
        tmp_model = VigiLensV2Model()
        v1_path = ROOT / cfg.model.v1_checkpoint
        if v1_path.exists():
            tmp_model.load_v1_backbone(str(v1_path))
        ckpt = {
            "model_state_dict": tmp_model.state_dict(),
            "calibration": dict(tmp_model.calibration),
            "config": {},
        }

    yolo_net = YOLO(str(ROOT / cfg.tracker.yolo_weights))
    tmp_motion = CausalMotionFlowEncoder(
        flow_height=cfg.motion.flow_height,
        flow_width=cfg.motion.flow_width,
        grid_rows=cfg.motion.grid_rows,
        grid_cols=cfg.motion.grid_cols,
    )
    reid_path = str(ROOT / cfg.model.reid_weights) if use_reid else None
    visual_ext = DualVisualExtractor(
        device=device,
        input_size=cfg.model.input_size,
        reid_weights_path=reid_path,
        load_ir=load_ir,
    )
    return {
        "device": device,
        "ckpt": ckpt,
        "yolo_net": yolo_net,
        "proj_2048": tmp_motion.proj_2048,
        "visual_ext": visual_ext,
        "infer_lock": threading.Lock(),
    }


def build_isolated_pipeline(
    weights_bundle: Dict[str, Any],
    use_reid: bool,
    th_high: Optional[float] = None,
    th_low: Optional[float] = None,
) -> InferencePipelineV2:
    """Create a fresh stateful InferencePipelineV2 isolated to the current run."""
    cfg = get_v2_config()
    device = weights_bundle["device"]
    ckpt = weights_bundle["ckpt"]
    m_cfg = ckpt.get("config", {})

    model = VigiLensV2Model(
        track_in_dim=m_cfg.get("track_in_dim", cfg.model.track_feature_dim),
        track_hidden_dim=m_cfg.get("track_hidden_dim", cfg.model.track_hidden_dim),
        grid_rows=m_cfg.get("grid_rows", cfg.motion.grid_rows),
        grid_cols=m_cfg.get("grid_cols", cfg.motion.grid_cols),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if "calibration" in ckpt:
        model.calibration = dict(ckpt["calibration"])
    model.eval()

    tracker = YOLOCausalTracker(
        weights_path=str(ROOT / cfg.tracker.yolo_weights),
        conf_thresh=cfg.tracker.conf_thresh,
        imgsz=cfg.tracker.imgsz,
        iou_match_thresh=cfg.tracker.iou_match_thresh,
        max_center_dist_frac=cfg.tracker.max_center_dist_frac,
        max_missed_frames=cfg.tracker.max_missed_frames,
        device=device,
        yolo_model=weights_bundle["yolo_net"],
    )
    motion_enc = CausalMotionFlowEncoder(
        flow_height=cfg.motion.flow_height,
        flow_width=cfg.motion.flow_width,
        grid_rows=cfg.motion.grid_rows,
        grid_cols=cfg.motion.grid_cols,
        proj_2048=weights_bundle["proj_2048"],
    )
    pipe = InferencePipelineV2(
        model=model,
        tracker=tracker,
        motion_encoder=motion_enc,
        visual_extractor=weights_bundle["visual_ext"],
        use_reid=use_reid,
        threshold_high=th_high,
        threshold_low=th_low,
        hold_frames=int(model.calibration.get("hold_frames", cfg.decision.hysteresis_hold_frames)),
    )
    pipe.reset()
    return pipe


def load_v2_resources(use_reid: bool, load_ir: bool) -> Tuple[VigiLensV2Model, YOLOCausalTracker, CausalMotionFlowEncoder, DualVisualExtractor]:
    """Compatibility helper returning an isolated (model, tracker, motion_enc, visual_ext) tuple."""
    bundle = load_immutable_weights(use_reid=use_reid, load_ir=load_ir)
    pipe = build_isolated_pipeline(bundle, use_reid=use_reid)
    return pipe.model, pipe.tracker, pipe.motion_encoder, pipe.visual_extractor


def _save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def main() -> None:
    st.title("🛡️ VigiLens V2 — Dual-Stream & Object-Centric Anomaly Detection")
    st.caption(
        "Causal YOLOv8n-seg Tracking + Perspective Optical-Flow Kinematics + ModalityAwareFusion & Causal LSTM "
        "| Normal subjects highlighted in GREEN, anomalous subjects localized in RED."
    )

    with st.sidebar:
        st.header("1. Video Source")
        demo_presets: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            "Upload Custom Video": (None, None),
            "Demo: Bicycle & Car (Ped2 Test004)": (str(ROOT / "Data/ucsd_ped2/videos/Test004_rgb.mp4"), None),
            "Demo: Skateboarder (Ped2 Test007)": (str(ROOT / "Data/ucsd_ped2/videos/Test007_rgb.mp4"), None),
            "Demo: Cyclist in Crowd (Ped2 Test002)": (str(ROOT / "Data/ucsd_ped2/videos/Test002_rgb.mp4"), None),
            "Demo: Normal Walkers (Ped2 Train001)": (str(ROOT / "Data/ucsd_ped2/train_videos/Train001_rgb.mp4"), None),
            "Demo: Paired RGB + Thermal IR": (
                str(ROOT / "demo_videos/4_Multimodal_RGB_anomaly.mp4"),
                str(ROOT / "demo_videos/4_Multimodal_IR_anomaly.mp4"),
            ),
        }
        selected_preset = st.selectbox("Quick Presentation Preset", list(demo_presets.keys()), index=1)

        vis_upload = st.file_uploader("Upload Visible Video (.mp4, .avi)", type=["mp4", "avi", "mov"])
        ir_upload = st.file_uploader("Upload Thermal IR Video (Optional)", type=["mp4", "avi", "mov"])

        st.header("2. Operating Regime & Calibration")
        regime = st.radio(
            "Threshold Regime (from Normal Train)",
            [
                "Calibrated (p95 / p80 — Low FPR 2.5%)",
                "Balanced High-Catch (p92 / p75)",
                "High-Sensitivity (p85 / p70)",
                "Custom Manual Threshold",
            ],
            index=0,
        )
        custom_thr = None
        if regime.startswith("Custom"):
            custom_thr = st.slider("Manual Alert Threshold (Z-score)", 0.25, 4.0, 1.23, 0.05)

        st.header("3. Runtime Options")
        use_reid = st.checkbox("Enable Market-1501 Re-ID Stream", value=True)
        playback_fps_cap = st.slider("Display FPS Cap", 5, 30, 15, 1)

    preset_vis, preset_ir = demo_presets[selected_preset]
    temp_files: List[str] = []

    if vis_upload is not None:
        vis_path = _save_upload_to_temp(vis_upload)
        temp_files.append(vis_path)
    else:
        vis_path = preset_vis

    if ir_upload is not None:
        ir_path = _save_upload_to_temp(ir_upload)
        temp_files.append(ir_path)
    else:
        ir_path = preset_ir

    has_ir = bool(ir_path and Path(ir_path).exists())
    bundle = load_immutable_weights(use_reid=use_reid, load_ir=has_ir)
    cal = bundle["ckpt"].get("calibration", {})

    if regime.startswith("Calibrated"):
        th_high, th_low = float(cal.get("threshold_high", 1.23)), float(cal.get("threshold_low", 0.68))
    elif regime.startswith("Balanced"):
        th_high, th_low = 1.0426, 0.5614
    elif regime.startswith("High-Sensitivity"):
        th_high, th_low = 0.7912, 0.4795
    else:
        th_high = float(custom_thr or 1.23)
        th_low = th_high * 0.60

    col_video, col_metrics = st.columns([1.45, 1.0], gap="large")
    with col_video:
        video_box = st.empty()
    with col_metrics:
        status_banner = st.empty()
        m_row1 = st.columns(3)
        metric_score = m_row1[0].empty()
        metric_track = m_row1[1].empty()
        metric_fps = m_row1[2].empty()
        m_row2 = st.columns(3)
        metric_grid = m_row2[0].empty()
        metric_app = m_row2[1].empty()
        metric_alerts = m_row2[2].empty()
        chart_box = st.empty()
        track_detail_box = st.empty()

    start_clicked = st.button("▶ Run Live Analysis", type="primary", width="stretch")

    if start_clicked:
        if not vis_path or not Path(vis_path).exists():
            st.error("No valid video file found. Please select a preset or upload a video.")
            return

        pipeline = build_isolated_pipeline(bundle, use_reid=use_reid, th_high=th_high, th_low=th_low)
        infer_lock: threading.Lock = bundle["infer_lock"]

        cap_vis = cv2.VideoCapture(vis_path)
        cap_ir = cv2.VideoCapture(ir_path) if has_ir else None

        scores_hist: List[float] = []
        thr_hist: List[float] = []
        track_hist: List[float] = []
        alert_count = 0
        min_frame_dt = 1.0 / max(1, playback_fps_cap)

        try:
            while cap_vis.isOpened():
                t_loop = time.perf_counter()
                ok, frame_bgr = cap_vis.read()
                if not ok:
                    break

                ir_frame = None
                if cap_ir is not None and cap_ir.isOpened():
                    ok_ir, ir_raw = cap_ir.read()
                    if ok_ir:
                        ir_frame = ir_raw

                with infer_lock:
                    res = pipeline.process_frame(frame_bgr, ir_frame=ir_frame, draw_overlay=True)
                if res.is_anomalous:
                    alert_count += 1

                scores_hist.append(res.score)
                thr_hist.append(res.threshold_high)
                track_hist.append(res.score_track)

                if res.annotated_bgr is not None:
                    rgb_disp = cv2.cvtColor(res.annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_box.image(
                        rgb_disp,
                        caption=f"Frame {res.frame_index} | Active Tracks: {len(res.tracks)}",
                        width="stretch",
                    )

                if res.is_anomalous:
                    anom_str = ", ".join(f"ID:{tid}" for tid in res.anomalous_track_ids) or "Scene-level"
                    status_banner.error(f"🚨 ANOMALY ACTIVE | Localized: {anom_str}")
                else:
                    status_banner.success("✅ NORMAL SURVEILLANCE STATE")

                metric_score.metric("V2 Fused Z-Score", f"{res.score:.2f}", delta=f"Thr {res.threshold_high:.2f}")
                metric_track.metric("Track Dynamics Z", f"{res.score_track:.2f}")
                metric_fps.metric("Streaming FPS", f"{res.fps:.1f}")
                metric_grid.metric("Flow Grid Z", f"{res.score_grid:.2f}")
                metric_app.metric("Appearance Z", f"{res.score_app:.2f}")
                metric_alerts.metric("Anomaly Frames", f"{alert_count}")

                df_chart = pd.DataFrame(
                    {
                        "V2 Fused Score": scores_hist[-150:],
                        "Alert Threshold": thr_hist[-150:],
                        "Track LSTM Score": track_hist[-150:],
                    }
                )
                chart_box.line_chart(df_chart, height=230)

                if res.track_scores:
                    top_trks = sorted(res.track_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    track_rows = [
                        {
                            "Track ID": f"ID:{tid}",
                            "Z-Score": round(val, 2),
                            "Status": "RED (Anomalous)" if tid in res.anomalous_track_ids else "GREEN (Normal)",
                        }
                        for tid, val in top_trks
                    ]
                    track_detail_box.dataframe(pd.DataFrame(track_rows), width="stretch", hide_index=True)

                elapsed = time.perf_counter() - t_loop
                if elapsed < min_frame_dt:
                    time.sleep(min_frame_dt - elapsed)
        finally:
            cap_vis.release()
            if cap_ir is not None:
                cap_ir.release()
            for p in temp_files:
                if os.path.exists(p):
                    os.unlink(p)

        st.success(f"Completed stream ({len(scores_hist)} frames, {alert_count} anomaly frames flagged).")


if __name__ == "__main__":
    main()
