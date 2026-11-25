"""Unified VigiLens V2 Anomaly Detection Model.

Preserves V1's core ModalityAwareFusion (2048 -> 512) + Causal 2-Layer LSTM (256)
+ NextFeaturePredictor (512) architecture, augmented with:
  - TrackDynamicsModel (per-track causal LSTM + perspective kinematics/geometry)
  - Perspective row-normalized Optical Flow Grid encoder
  - Normal-Manifold Mahalanobis Residual + Multi-stream score calibration
  - Seamless support for BOTH Paired RGB+IR (VidLLVIP) and Single-Cam RGB (UCSD Ped2 / Live CCTV)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import ModalityAwareFusion
from models.temporal_lstm import TemporalModel
from vigilens_v2.models.track_model import TrackDynamicsModel
from vigilens_v2.pipeline.smoothing import causal_moving_average_skip0


class VigiLensV2Model(nn.Module):
    """Unified Object-Centric Dual-Stream VigiLens V2 Neural Architecture."""

    def __init__(
        self,
        fusion_hidden_dim: int = 256,
        fusion_output_dim: int = 512,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        track_in_dim: int = 8,
        track_hidden_dim: int = 64,
        grid_rows: int = 12,
        grid_cols: int = 18,
    ):
        super().__init__()
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)

        # Core VigiLens Fusion + Causal Temporal LSTM (binary-compatible with V1 checkpoint!)
        self.fusion = ModalityAwareFusion(
            input_dim=2048,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
        )
        self.temporal = TemporalModel(
            input_size=fusion_output_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # V2 Track-Level Causal Dynamics Model
        self.track_model = TrackDynamicsModel(in_dim=track_in_dim, hidden_dim=track_hidden_dim)

        # Calibration buffers populated strictly from normal training clips
        self.register_buffer("row_flow_p95", torch.ones(1, grid_rows, 1, dtype=torch.float32))
        self.register_buffer("scene_z_mu", torch.zeros(fusion_output_dim, dtype=torch.float32))
        self.register_buffer("scene_z_var", torch.ones(fusion_output_dim, dtype=torch.float32))

        # Per-component normal median/std stats for balanced z-score fusion
        self.calibration: Dict[str, Any] = {
            "track_med": 0.0,
            "track_std": 1.0,
            "grid_med": 0.0,
            "grid_std": 1.0,
            "app_med": 0.0,
            "app_std": 1.0,
            "threshold_high": 1.5,
            "threshold_low": 0.8,
            "hold_frames": 15,
            "weights": {"track": 0.56, "grid": 0.24, "app": 0.20},
            "track_alert_threshold": 0.65,
        }

        # Streaming state for VigiLens scene backbone (ImageNet RGB, optional ReID, optional IR)
        self._scene_hidden_a: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._scene_pred_a: Optional[torch.Tensor] = None
        self._scene_hidden_b: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._scene_pred_b: Optional[torch.Tensor] = None
        self._hist_grid: List[float] = []
        self._hist_app: List[float] = []

    def reset_stream_state(self) -> None:
        self.track_model.reset_stream_state()
        self._scene_hidden_a = None
        self._scene_pred_a = None
        self._scene_hidden_b = None
        self._scene_pred_b = None
        self._hist_grid.clear()
        self._hist_app.clear()

    def load_v1_backbone(self, v1_checkpoint_path: str) -> None:
        """Load pretrained V1 ModalityAwareFusion + CausalLSTM weights cleanly."""
        ckpt = torch.load(v1_checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        fusion_sd = {k[len("fusion."):]: v for k, v in state.items() if k.startswith("fusion.")}
        temporal_sd = {k[len("temporal."):]: v for k, v in state.items() if k.startswith("temporal.")}
        if fusion_sd:
            self.fusion.load_state_dict(fusion_sd, strict=True)
        if temporal_sd:
            self.temporal.load_state_dict(temporal_sd, strict=True)

    def fit_grid_baseline(self, train_grid_flows: List[np.ndarray]) -> None:
        """Fit perspective row-wise 95th-percentile baseline flow strictly on normal train clips."""
        valid = [g[1:] for g in train_grid_flows if len(g) > 1]
        if not valid:
            return
        all_g = np.concatenate(valid, axis=0)  # (T_tot, R, C)
        row_vals = all_g.transpose(1, 0, 2).reshape(self.grid_rows, -1)
        p95 = np.percentile(row_vals, 95, axis=1).astype(np.float32).reshape(1, self.grid_rows, 1)
        self.row_flow_p95.copy_(torch.from_numpy(p95))

    def score_grid_excess(self, grid_flow: np.ndarray, smoothing_window: int = 5) -> np.ndarray:
        """Compute causal perspective-normalized top-2 optical flow cell excess over frame sequence."""
        n = len(grid_flow)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        base = self.row_flow_p95.detach().cpu().numpy()
        rz = np.maximum(0.0, (grid_flow - base) / np.maximum(base, 0.40))
        top2 = np.sort(rz.reshape(n, -1), axis=1)[:, -2:].mean(axis=1).astype(np.float32)
        top2[0] = 0.0
        return causal_moving_average_skip0(top2, window=smoothing_window)

    @torch.inference_mode()
    def score_scene_appearance(
        self,
        rgb_feats: torch.Tensor,
        reid_feats: Optional[torch.Tensor] = None,
        ir_feats: Optional[torch.Tensor] = None,
        smoothing_window: int = 7,
    ) -> np.ndarray:
        """Compute causal scene appearance prediction error (+ optional ReID / paired IR)."""
        device = next(self.parameters()).device
        rgb = rgb_feats.float().to(device)
        n = len(rgb)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        if ir_feats is not None:
            ir = ir_feats.float().to(device)
            z = self.fusion(rgb, ir, rgb_available=True, ir_available=True)
            pred, _ = self.temporal.forward_sequence(z.unsqueeze(0))
            err = torch.zeros(n, device=device)
            err[1:] = (pred[0, :-1] - z[1:]).pow(2).mean(dim=-1)
            return causal_moving_average_skip0(err.cpu().numpy(), window=smoothing_window)

        z_rgb = self.fusion(rgb, None, rgb_available=True, ir_available=False)
        pred_rgb, _ = self.temporal.forward_sequence(z_rgb.unsqueeze(0))
        err_rgb = torch.zeros(n, device=device)
        err_rgb[1:] = (pred_rgb[0, :-1] - z_rgb[1:]).pow(2).mean(dim=-1)
        sc_rgb = causal_moving_average_skip0(err_rgb.cpu().numpy(), window=smoothing_window)

        if reid_feats is None:
            return sc_rgb

        reid = reid_feats.float().to(device)
        # Apply pure hypersphere unit-L2 normalization (matches ReID-Unit ablation)
        reid = F.normalize(reid, dim=-1)
        z_reid = self.fusion(reid, None, rgb_available=True, ir_available=False)
        pred_reid, _ = self.temporal.forward_sequence(z_reid.unsqueeze(0))
        err_reid = torch.zeros(n, device=device)
        err_reid[1:] = (pred_reid[0, :-1] - z_reid[1:]).pow(2).mean(dim=-1)
        sc_reid = causal_moving_average_skip0(err_reid.cpu().numpy(), window=smoothing_window)
        return (0.5 * sc_rgb + 0.5 * sc_reid).astype(np.float32)

    def combine_z_scores(
        self,
        sc_track: np.ndarray,
        sc_grid: np.ndarray,
        sc_app: np.ndarray,
    ) -> np.ndarray:
        """Fuse normalized Track + Grid + Scene Appearance anomaly curves with frame-0 reset."""
        c = self.calibration
        zt = (sc_track - float(c["track_med"])) / max(1e-4, float(c["track_std"]))
        zg = (sc_grid - float(c["grid_med"])) / max(1e-4, float(c["grid_std"]))
        za = (sc_app - float(c["app_med"])) / max(1e-4, float(c["app_std"]))
        w = c.get("weights", {"track": 0.56, "grid": 0.24, "app": 0.20})
        fused = (
            float(w["track"]) * zt + float(w["grid"]) * zg + float(w["app"]) * za
        ).astype(np.float32)
        if len(fused) > 0:
            fused[0] = 0.0
        return fused

    @torch.inference_mode()
    def step_frame(
        self,
        frame_index: int,
        grid_flow_frame: np.ndarray,
        meta_frame: np.ndarray,
        rgb_feat: torch.Tensor,
        reid_feat: Optional[torch.Tensor] = None,
        ir_feat: Optional[torch.Tensor] = None,
        grid_window: int = 5,
        app_window: int = 7,
    ) -> Dict[str, Any]:
        """Strict 1-frame causal forward pass returning fused score + per-track localization scores."""
        device = next(self.parameters()).device

        # 1. Per-track causal dynamics score + individual track localization
        raw_trk_max, track_scores = self.track_model.step_frame_tracks(meta_frame)

        # 2. Perspective-normalized grid excess score
        if frame_index == 0:
            raw_grid = 0.0
            self._hist_grid.clear()
        else:
            base = self.row_flow_p95.detach().cpu().numpy().squeeze(0)  # (R, 1)
            rz = np.maximum(0.0, (grid_flow_frame - base) / np.maximum(base, 0.40))
            flat = np.sort(rz.reshape(-1))
            raw_grid = float(flat[-2:].mean()) if len(flat) >= 2 else float(flat.mean())
            self._hist_grid.append(raw_grid)
            if len(self._hist_grid) > grid_window:
                self._hist_grid.pop(0)
        sc_grid = float(sum(self._hist_grid) / len(self._hist_grid)) if self._hist_grid else 0.0

        # 3. Causal Scene Appearance (ModalityAwareFusion -> Causal LSTM)
        r_t = rgb_feat.float().to(device).view(1, 2048)
        if ir_feat is not None:
            i_t = ir_feat.float().to(device).view(1, 2048)
            z_a = self.fusion(r_t, i_t, rgb_available=True, ir_available=True)
        else:
            z_a = self.fusion(r_t, None, rgb_available=True, ir_available=False)

        err_a = 0.0
        if self._scene_pred_a is not None and frame_index > 0:
            err_a = float((self._scene_pred_a - z_a.squeeze(0)).pow(2).mean().item())
        pred_a, self._scene_hidden_a = self.temporal(z_a.unsqueeze(1), self._scene_hidden_a)
        self._scene_pred_a = pred_a.squeeze(0).squeeze(0).detach()

        raw_app = err_a
        if reid_feat is not None and ir_feat is None:
            re_t = F.normalize(reid_feat.float().to(device).view(1, 2048), dim=-1)
            z_b = self.fusion(re_t, None, rgb_available=True, ir_available=False)
            err_b = 0.0
            if self._scene_pred_b is not None and frame_index > 0:
                err_b = float((self._scene_pred_b - z_b.squeeze(0)).pow(2).mean().item())
            pred_b, self._scene_hidden_b = self.temporal(z_b.unsqueeze(1), self._scene_hidden_b)
            self._scene_pred_b = pred_b.squeeze(0).squeeze(0).detach()
            raw_app = 0.5 * err_a + 0.5 * err_b

        if frame_index == 0:
            self._hist_app.clear()
            sc_app = 0.0
        else:
            self._hist_app.append(raw_app)
            if len(self._hist_app) > app_window:
                self._hist_app.pop(0)
            sc_app = float(sum(self._hist_app) / len(self._hist_app))

        c = self.calibration
        zt = (raw_trk_max - float(c["track_med"])) / max(1e-4, float(c["track_std"]))
        zg = (sc_grid - float(c["grid_med"])) / max(1e-4, float(c["grid_std"]))
        za = (sc_app - float(c["app_med"])) / max(1e-4, float(c["app_std"]))
        w = c.get("weights", {"track": 0.56, "grid": 0.24, "app": 0.20})

        score = (
            0.0
            if frame_index == 0
            else float(w["track"]) * zt + float(w["grid"]) * zg + float(w["app"]) * za
        )
        # Convert per-track scores into standardized track z-scores so RED threshold is uniform!
        trk_med = float(c["track_med"])
        trk_std = max(1e-4, float(c["track_std"]))
        track_z = {tid: (val - trk_med) / trk_std for tid, val in track_scores.items()}

        return {
            "score": score,
            "score_track": float(zt if frame_index > 0 else 0.0),
            "score_grid": float(zg if frame_index > 0 else 0.0),
            "score_app": float(za if frame_index > 0 else 0.0),
            "track_scores": track_z,
        }

    def save_v2_checkpoint(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "calibration": self.calibration,
                "config": {
                    "grid_rows": self.grid_rows,
                    "grid_cols": self.grid_cols,
                    "track_in_dim": self.track_model.in_dim,
                    "track_hidden_dim": self.track_model.hidden_dim,
                },
            },
            str(p),
        )

    @classmethod
    def load_v2_checkpoint(cls, path: str, device: str = "cpu") -> "VigiLensV2Model":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = cls(
            track_in_dim=cfg.get("track_in_dim", 8),
            track_hidden_dim=cfg.get("track_hidden_dim", 64),
            grid_rows=cfg.get("grid_rows", 12),
            grid_cols=cfg.get("grid_cols", 18),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "calibration" in ckpt:
            model.calibration = dict(ckpt["calibration"])
        model.eval()
        return model
