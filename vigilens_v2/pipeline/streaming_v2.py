"""Strict causal streaming inference + per-track RED/GREEN localization renderer for VigiLens V2."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vigilens_v2.models.feature_extractor import DualVisualExtractor
from vigilens_v2.models.motion_encoder import CausalMotionFlowEncoder
from vigilens_v2.models.tracker import TrackDetection, YOLOCausalTracker
from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model
from vigilens_v2.pipeline.smoothing import CausalHysteresisDetector


@dataclass
class FrameInferenceResult:
    frame_index: int
    score: float
    is_anomalous: bool
    threshold_high: float
    threshold_low: float
    score_track: float
    score_grid: float
    score_app: float
    anomalous_track_ids: List[int]
    track_scores: Dict[int, float]
    tracks: List[TrackDetection]
    fps: float
    annotated_bgr: Optional[np.ndarray] = field(default=None, repr=False)


class InferencePipelineV2:
    """Real-time object-centric VigiLens V2 streaming pipeline."""

    def __init__(
        self,
        model: VigiLensV2Model,
        tracker: YOLOCausalTracker,
        motion_encoder: CausalMotionFlowEncoder,
        visual_extractor: DualVisualExtractor,
        use_reid: bool = True,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
        hold_frames: Optional[int] = None,
        track_alert_threshold: Optional[float] = None,
    ):
        self.model = model
        self.tracker = tracker
        self.motion_encoder = motion_encoder
        self.visual_extractor = visual_extractor
        self.use_reid = bool(use_reid and visual_extractor.reid_extractor is not None)

        cal = model.calibration
        th_h = float(threshold_high if threshold_high is not None else cal.get("threshold_high", 1.5))
        th_l = float(threshold_low if threshold_low is not None else cal.get("threshold_low", 0.8))
        hold = int(hold_frames if hold_frames is not None else cal.get("hold_frames", 15))
        self.track_alert_threshold = float(
            track_alert_threshold
            if track_alert_threshold is not None
            else cal.get("track_alert_threshold", th_h * 0.85)
        )
        self.hysteresis = CausalHysteresisDetector(th_h, th_l, hold_frames=hold)
        self._frame_idx = 0

    def reset(self) -> None:
        self._frame_idx = 0
        self.tracker.reset()
        self.motion_encoder.reset()
        self.model.reset_stream_state()
        self.hysteresis.reset()

    def set_thresholds(self, threshold_high: float, threshold_low: Optional[float] = None) -> None:
        self.hysteresis.threshold_high = float(threshold_high)
        if threshold_low is not None:
            self.hysteresis.threshold_low = float(min(threshold_low, threshold_high))
        else:
            self.hysteresis.threshold_low = float(threshold_high * 0.65)
        self.track_alert_threshold = float(threshold_high * 0.85)

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        ir_frame: Optional[np.ndarray] = None,
        draw_overlay: bool = True,
    ) -> FrameInferenceResult:
        """Process one BGR (and optional IR) frame strictly causally and annotate tracks."""
        t0 = time.perf_counter()
        idx = self._frame_idx
        self._frame_idx += 1

        tracks, fg_mask = self.tracker.process_frame(frame_bgr)
        grid_flow, meta_frame = self.motion_encoder.compute_frame_flow_and_track_stats(
            frame_bgr, tracks, frame_index=idx
        )

        rgb_feat = self.visual_extractor.extract_rgb(frame_bgr, mask=fg_mask)
        reid_feat = (
            self.visual_extractor.extract_reid(frame_bgr, mask=fg_mask)
            if (self.use_reid and ir_frame is None)
            else None
        )
        ir_feat = self.visual_extractor.extract_ir(ir_frame, mask=fg_mask) if ir_frame is not None else None

        step_out = self.model.step_frame(
            frame_index=idx,
            grid_flow_frame=grid_flow,
            meta_frame=meta_frame,
            rgb_feat=rgb_feat,
            reid_feat=reid_feat,
            ir_feat=ir_feat,
        )

        score = float(step_out["score"])
        is_anom, _ = self.hysteresis.step(score)
        track_scores: Dict[int, float] = step_out["track_scores"]

        # Highlight anomalous track(s): any track exceeding track threshold OR top track during scene alarm
        anom_ids = [
            tid for tid, t_sc in track_scores.items() if t_sc >= self.track_alert_threshold
        ]
        if is_anom and not anom_ids and track_scores:
            top_tid = max(track_scores, key=lambda k: track_scores[k])
            if track_scores[top_tid] > 0.25:
                anom_ids = [top_tid]

        dt = max(1e-4, time.perf_counter() - t0)
        fps = 1.0 / dt

        annotated = (
            self.render_overlay(
                frame_bgr=frame_bgr,
                tracks=tracks,
                track_scores=track_scores,
                anom_ids=anom_ids,
                frame_score=score,
                is_anomalous=is_anom,
                threshold_high=self.hysteresis.threshold_high,
                fps=fps,
                frame_idx=idx,
                has_ir=(ir_frame is not None),
            )
            if draw_overlay
            else None
        )

        return FrameInferenceResult(
            frame_index=idx,
            score=score,
            is_anomalous=is_anom,
            threshold_high=self.hysteresis.threshold_high,
            threshold_low=self.hysteresis.threshold_low,
            score_track=step_out["score_track"],
            score_grid=step_out["score_grid"],
            score_app=step_out["score_app"],
            anomalous_track_ids=anom_ids,
            track_scores=track_scores,
            tracks=tracks,
            fps=fps,
            annotated_bgr=annotated,
        )

    @staticmethod
    def render_overlay(
        frame_bgr: np.ndarray,
        tracks: List[TrackDetection],
        track_scores: Dict[int, float],
        anom_ids: List[int],
        frame_score: float,
        is_anomalous: bool,
        threshold_high: float,
        fps: float,
        frame_idx: int,
        has_ir: bool = False,
    ) -> np.ndarray:
        """Render clean per-track GREEN (normal) and bold RED (anomalous) boxes + HUD bar."""
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]
        if w < 640:
            canvas = cv2.resize(canvas, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            h, w = canvas.shape[:2]
        anom_set = set(anom_ids)

        # Draw normal tracks first, then anomalous tracks on top for crystal visibility
        ordered = sorted(tracks, key=lambda t: (t.track_id in anom_set, t.track_id))
        for trk in ordered:
            x1, y1, x2, y2 = trk.box_norm
            px1, py1 = max(0, int(round(x1 * w))), max(0, int(round(y1 * h)))
            px2, py2 = min(w - 1, int(round(x2 * w))), min(h - 1, int(round(y2 * h)))
            t_sc = track_scores.get(trk.track_id, 0.0)
            is_trk_anom = trk.track_id in anom_set

            if is_trk_anom:
                color = (48, 48, 255)  # vivid RED in BGR
                thick = 3
                label = f"ANOMALY ID:{trk.track_id} ({t_sc:.2f})"
                font_scale = 0.50
            else:
                color = (80, 210, 90)  # subtle crisp GREEN in BGR
                thick = 1
                label = f"ID:{trk.track_id}"
                font_scale = 0.36

            cv2.rectangle(canvas, (px1, py1), (px2, py2), color, thick, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ly1 = max(0, py1 - th - 6)
            cv2.rectangle(canvas, (px1, ly1), (min(w - 1, px1 + tw + 6), py1), color, -1)
            txt_color = (255, 255, 255) if is_trk_anom else (15, 25, 15)
            cv2.putText(
                canvas,
                label,
                (px1 + 3, max(th + 2, py1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                txt_color,
                1,
                cv2.LINE_AA,
            )

        # Top translucent HUD bar
        banner_h = max(32, int(round(h * 0.08)))
        overlay = canvas.copy()
        bar_color = (22, 22, 130) if is_anomalous else (24, 34, 42)
        cv2.rectangle(overlay, (0, 0), (w, banner_h), bar_color, -1)
        canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0)

        status_txt = "ANOMALY DETECTED" if is_anomalous else "NORMAL"
        mode_tag = "RGB+IR" if has_ir else "RGB+Motion"
        hud = f"VigiLens v2 [{mode_tag}] | {status_txt} | Score: {frame_score:.2f}/{threshold_high:.2f} | Trks: {len(tracks)} | F:{frame_idx}"
        font_hud = max(0.42, min(0.54, w / 1280.0))
        cv2.putText(
            canvas,
            hud,
            (8, int(round(banner_h * 0.68))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_hud,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas
