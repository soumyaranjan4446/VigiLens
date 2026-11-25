"""Causal Dense Optical Flow + Perspective-Normalized Motion-Flow 2048-D Encoder."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from vigilens_v2.models.tracker import TrackDetection


class CausalMotionFlowEncoder:
    """Computes causal Farneback optical flow grids, per-track flow stats, and 2048-D embeddings."""

    def __init__(
        self,
        flow_height: int = 240,
        flow_width: int = 360,
        grid_rows: int = 12,
        grid_cols: int = 18,
        seed: int = 2026,
        proj_2048: Optional[torch.Tensor] = None,
    ):
        self.flow_height = int(flow_height)
        self.flow_width = int(flow_width)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self._prev_gray: Optional[np.ndarray] = None

        if proj_2048 is not None:
            self.proj_2048 = proj_2048
        else:
            raw_dim = self.grid_rows * self.grid_cols + 8
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            q, _ = torch.linalg.qr(torch.randn(2048, raw_dim, generator=gen))
            # Fixed semi-orthogonal lifter into 2048-D so Motion plugs natively into ModalityAwareFusion
            self.proj_2048 = q.T.contiguous()  # (raw_dim, 2048)

    def reset(self) -> None:
        self._prev_gray = None

    def compute_frame_flow_and_track_stats(
        self,
        frame_bgr: np.ndarray,
        tracks: List[TrackDetection],
        frame_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (grid_flow [R, C], track_meta [N_trk, 13]).

        track_meta columns:
          0: frame_idx, 1: track_id, 2..5: (x1,y1,x2,y2), 6: yc, 7: bh, 8: bw,
          9: flow_p90, 10: flow_p95, 11: mean_flow, 12: box_disp_norm
        """
        gray = cv2.resize(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
            (self.flow_width, self.flow_height),
            interpolation=cv2.INTER_AREA,
        )
        r_n, c_n = self.grid_rows, self.grid_cols

        if self._prev_gray is None or frame_index == 0:
            self._prev_gray = gray
            grid_flow = np.zeros((r_n, c_n), dtype=np.float32)
            mag = np.zeros((self.flow_height, self.flow_width), dtype=np.float32)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            self._prev_gray = gray
            mag = np.hypot(flow[..., 0], flow[..., 1]).astype(np.float32)
            # Average pooling over (R, C) cells
            cell_h = self.flow_height // r_n
            cell_w = self.flow_width // c_n
            grid_flow = mag[: cell_h * r_n, : cell_w * c_n].reshape(r_n, cell_h, c_n, cell_w).mean(axis=(1, 3)).astype(np.float32)

        meta_rows = []
        fh, fw = self.flow_height, self.flow_width
        for trk in tracks:
            x1, y1, x2, y2 = trk.box_norm
            yc = 0.5 * (y1 + y2)
            bh = max(0.0, y2 - y1)
            bw = max(0.0, x2 - x1)
            ix1 = max(0, min(fw - 1, int(round(x1 * fw))))
            iy1 = max(0, min(fh - 1, int(round(y1 * fh))))
            ix2 = max(ix1 + 1, min(fw, int(round(x2 * fw))))
            iy2 = max(iy1 + 1, min(fh, int(round(y2 * fh))))
            patch = mag[iy1:iy2, ix1:ix2]
            if patch.size > 0 and frame_index > 0:
                p90 = float(np.percentile(patch, 90))
                p95 = float(np.percentile(patch, 95))
                mf = float(patch.mean())
            else:
                p90 = p95 = mf = 0.0
            meta_rows.append(
                [
                    float(frame_index),
                    float(trk.track_id),
                    x1,
                    y1,
                    x2,
                    y2,
                    yc,
                    bh,
                    bw,
                    p90,
                    p95,
                    mf,
                    float(trk.displacement_norm),
                ]
            )

        meta = (
            np.asarray(meta_rows, dtype=np.float32)
            if meta_rows
            else np.zeros((0, 13), dtype=np.float32)
        )
        return grid_flow, meta

    def encode_motion_2048(
        self,
        grid_norm_excess: np.ndarray,
        max_pos_kinematics: np.ndarray,
    ) -> torch.Tensor:
        """Project normalized grid flow (R*C) + max track kinematics (8) into 2048-D."""
        g_flat = np.asarray(grid_norm_excess, dtype=np.float32).reshape(-1, self.grid_rows * self.grid_cols)
        k_flat = np.asarray(max_pos_kinematics, dtype=np.float32).reshape(-1, 8)
        vec = np.concatenate([g_flat * 0.25, k_flat], axis=1).astype(np.float32)
        return torch.from_numpy(vec) @ self.proj_2048
