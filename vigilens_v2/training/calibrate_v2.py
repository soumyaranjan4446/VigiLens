"""Strict normal-only calibration utilities for VigiLens V2."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model


def calibrate_v2_from_normal_clips(
    model: VigiLensV2Model,
    normal_clips: List[Dict[str, torch.Tensor]],
    percentile_high: float = 95.0,
    percentile_low: float = 80.0,
    hold_frames: int = 15,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compute normal stream z-score normalization statistics and hysteresis thresholds.

    Each entry in `normal_clips` contains:
      - "grid_flow": np.ndarray (N, R, C)
      - "meta": np.ndarray (M, 13)
      - "rgb_features": torch.Tensor (N, 2048)
      - optional "reid_features": torch.Tensor (N, 2048)
      - optional "ir_features": torch.Tensor (N, 2048)
    """
    w_cfg = weights or {"track": 0.56, "grid": 0.24, "app": 0.20}
    model.eval()

    trk_vals, grid_vals, app_vals = [], [], []
    clip_triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for clip in normal_clips:
        gf = np.asarray(clip["grid_flow"], dtype=np.float32)
        meta = np.asarray(clip["meta"], dtype=np.float32)
        n = len(gf)
        sc_trk, _ = model.track_model.score_meta_offline(meta, num_frames=n)
        sc_grid = model.score_grid_excess(gf, smoothing_window=5)
        sc_app = model.score_scene_appearance(
            clip["rgb_features"],
            reid_feats=clip.get("reid_features"),
            ir_feats=clip.get("ir_features"),
            smoothing_window=7,
        )
        if n > 1:
            trk_vals.append(sc_trk[1:])
            grid_vals.append(sc_grid[1:])
            app_vals.append(sc_app[1:])
            clip_triples.append((sc_trk[1:], sc_grid[1:], sc_app[1:]))

    all_t = np.concatenate(trk_vals) if trk_vals else np.zeros(1, dtype=np.float32)
    all_g = np.concatenate(grid_vals) if grid_vals else np.zeros(1, dtype=np.float32)
    all_a = np.concatenate(app_vals) if app_vals else np.zeros(1, dtype=np.float32)

    t_med, t_std = float(np.median(all_t)), float(np.std(all_t) + 1e-4)
    g_med, g_std = float(np.median(all_g)), float(np.std(all_g) + 1e-4)
    a_med, a_std = float(np.median(all_a)), float(np.std(all_a) + 1e-4)

    fused_normal = []
    for t_arr, g_arr, a_arr in clip_triples:
        f = (
            float(w_cfg["track"]) * ((t_arr - t_med) / t_std)
            + float(w_cfg["grid"]) * ((g_arr - g_med) / g_std)
            + float(w_cfg["app"]) * ((a_arr - a_med) / a_std)
        )
        fused_normal.append(f)

    all_f = np.concatenate(fused_normal) if fused_normal else np.zeros(1, dtype=np.float32)
    th_high = float(np.percentile(all_f, percentile_high))
    th_low = float(np.percentile(all_f, percentile_low))

    # Calibrate per-track RED box threshold on normal standardized track scores
    trk_z_all = (all_t - t_med) / t_std
    trk_alert_thr = float(max(0.75, np.percentile(trk_z_all, min(97.0, percentile_high + 1.5))))

    calib = {
        "track_med": t_med,
        "track_std": t_std,
        "grid_med": g_med,
        "grid_std": g_std,
        "app_med": a_med,
        "app_std": a_std,
        "threshold_high": th_high,
        "threshold_low": th_low,
        "hold_frames": int(hold_frames),
        "percentile_high": float(percentile_high),
        "percentile_low": float(percentile_low),
        "weights": dict(w_cfg),
        "track_alert_threshold": trk_alert_thr,
        "normal_frames_calibrated": int(len(all_f)),
    }
    model.calibration = calib
    return calib
