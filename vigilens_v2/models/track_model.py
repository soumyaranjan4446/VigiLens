"""Per-Track Causal 2-Layer LSTM + Perspective-Normalized Normal Manifold Model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vigilens_v2.pipeline.smoothing import causal_moving_average_skip0


@dataclass
class TrackStepState:
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    prev_pred: Optional[torch.Tensor] = None
    step_count: int = 0
    raw_history: List[float] = None  # trailing window of track scores

    def __post_init__(self):
        if self.raw_history is None:
            self.raw_history = []


class TrackDynamicsModel(nn.Module):
    """Models normal pedestrian spatio-temporal dynamics on individual tracks.

    Preserves VigiLens's causal 2-layer LSTM + next-step predictor pattern at
    object-track granularity to eliminate crowd-dilution false negatives and
    localize anomalous entities with track-specific bounding boxes.
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, smoothing_window: int = 7, min_track_len: int = 3):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.smoothing_window = int(smoothing_window)
        self.min_track_len = int(min_track_len)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.predictor = nn.Linear(hidden_dim, in_dim)

        # Perspective linear fit coefficients [slope, intercept] vs image y-center
        self.register_buffer("coef_p90", torch.tensor([0.5, 0.5], dtype=torch.float32))
        self.register_buffer("coef_p95", torch.tensor([0.5, 0.6], dtype=torch.float32))
        self.register_buffer("coef_mf", torch.tensor([0.3, 0.3], dtype=torch.float32))
        self.register_buffer("coef_bsp", torch.tensor([0.4, 0.4], dtype=torch.float32))
        self.register_buffer("coef_bh", torch.tensor([0.2, 0.08], dtype=torch.float32))
        self.register_buffer("coef_bw", torch.tensor([0.08, 0.03], dtype=torch.float32))

        # Calibrated normal feature dispersion & asymmetric channel weights
        # Channels: [p90, p95, mf, bsp, bh, bw, bw/bh, (p95-mf)/mf]
        self.register_buffer("pos_scale", torch.ones(in_dim, dtype=torch.float32))
        self.register_buffer(
            "dim_weights",
            torch.tensor([0.38, 0.06, 0.32, 0.16, 0.0, 0.09, 0.09, 0.0], dtype=torch.float32),
        )
        self.register_buffer("lstm_err_weight", torch.tensor(0.05, dtype=torch.float32))

        self._stream_states: Dict[int, TrackStepState] = {}

    def reset_stream_state(self) -> None:
        self._stream_states.clear()

    def fit_perspective_baselines(self, train_meta: np.ndarray) -> None:
        """Fit linear perspective curves from normal training track metadata (yc -> expected magnitude)."""
        valid = train_meta[train_meta[:, 0] > 0]
        if len(valid) < 10:
            return
        yc = valid[:, 6]
        pairs = [
            ("coef_p90", valid[:, 9]),
            ("coef_p95", valid[:, 10]),
            ("coef_mf", valid[:, 11]),
            ("coef_bsp", valid[:, 12]),
            ("coef_bh", valid[:, 7]),
            ("coef_bw", valid[:, 8]),
        ]
        for buf_name, target in pairs:
            coef = np.polyfit(yc, target, 1).astype(np.float32)
            getattr(self, buf_name).copy_(torch.from_numpy(coef))

    def build_kinematics_vector(self, meta: np.ndarray) -> np.ndarray:
        """Transform raw 13-col track metadata into 8-D perspective-normalized residuals."""
        if len(meta) == 0:
            return np.zeros((0, self.in_dim), dtype=np.float32)

        yc = meta[:, 6]
        bh = meta[:, 7]
        bw = meta[:, 8]
        p90 = meta[:, 9]
        p95 = meta[:, 10]
        mf = meta[:, 11]
        bsp = meta[:, 12]

        def _eval(coef_buf: torch.Tensor, floor: float) -> np.ndarray:
            c = coef_buf.detach().cpu().numpy()
            return np.maximum(c[0] * yc + c[1], floor)

        ep90 = _eval(self.coef_p90, 0.40)
        ep95 = _eval(self.coef_p95, 0.45)
        emf = _eval(self.coef_mf, 0.15)
        ebsp = _eval(self.coef_bsp, 0.35)
        ebh = _eval(self.coef_bh, 0.08)
        ebw = _eval(self.coef_bw, 0.03)

        vec = np.stack(
            [
                p90 / ep90 - 1.0,
                p95 / ep95 - 1.0,
                mf / emf - 1.0,
                bsp / ebsp - 1.0,
                bh / ebh - 1.0,
                bw / ebw - 1.0,
                bw / np.maximum(bh, 0.02) - 0.40,
                (p95 - mf) / emf - 2.0,
            ],
            axis=1,
        ).astype(np.float32)
        return np.clip(vec, -3.0, 12.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward sequence x (B, T, 8) -> (next_pred, hidden_state, hidden_seq)."""
        h_in = self.encoder(x)
        h_seq, new_hidden = self.lstm(h_in, hidden)
        pred = self.predictor(h_seq)
        return pred, new_hidden, h_seq

    def train_on_normal_meta(
        self,
        train_meta_list: List[np.ndarray],
        epochs: int = 25,
        lr: float = 1e-3,
        seq_len: int = 10,
        seed: int = 42,
    ) -> float:
        """Fit perspective curves & train causal 2-layer LSTM strictly on normal track trajectories."""
        all_meta = np.concatenate([m for m in train_meta_list if len(m) > 0], axis=0)
        self.fit_perspective_baselines(all_meta)

        seqs = []
        for meta in train_meta_list:
            if len(meta) == 0:
                continue
            kv = self.build_kinematics_vector(meta)
            by_t: Dict[int, List[Tuple[int, np.ndarray]]] = {}
            for i in range(len(meta)):
                fr = int(meta[i, 0])
                if fr <= 0:
                    continue
                by_t.setdefault(int(meta[i, 1]), []).append((fr, kv[i]))
            for items in by_t.values():
                items.sort(key=lambda it: it[0])
                if len(items) >= seq_len:
                    arr = np.stack([it[1] for it in items])
                    for s in range(0, len(arr) - seq_len + 1, 2):
                        seqs.append(arr[s : s + seq_len])

        if not seqs:
            return 0.0

        seq_t = torch.tensor(np.stack(seqs), dtype=torch.float32)
        pos_flat = seq_t.reshape(-1, self.in_dim).clamp(min=0.0)
        self.pos_scale.copy_(pos_flat.std(dim=0) + 0.08)

        torch.manual_seed(seed)
        opt = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-2)
        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            perm = torch.randperm(len(seq_t))
            ep_loss = 0.0
            n_b = 0
            for s in range(0, len(seq_t), 64):
                b = seq_t[perm[s : s + 64]]
                pred, _, _ = self.forward(b[:, :-1])
                loss = F.smooth_l1_loss(pred, b[:, 1:])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                n_b += 1
            final_loss = ep_loss / max(1, n_b)
        self.eval()
        return final_loss

    @torch.inference_mode()
    def score_meta_offline(self, meta: np.ndarray, num_frames: int) -> Tuple[np.ndarray, Dict[int, Dict[int, float]]]:
        """Compute frame-level max track score and per-(frame, track_id) localized scores."""
        frame_scores = np.zeros(num_frames, dtype=np.float32)
        per_track_frame_scores: Dict[int, Dict[int, float]] = {}
        if len(meta) == 0:
            return frame_scores, per_track_frame_scores

        kv = self.build_kinematics_vector(meta)
        by_t: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for i in range(len(meta)):
            fr = int(meta[i, 0])
            if fr <= 0 or fr >= num_frames:
                continue
            by_t.setdefault(int(meta[i, 1]), []).append((fr, kv[i]))

        device = next(self.parameters()).device
        w_dim = self.dim_weights.to(device)
        scale = self.pos_scale.to(device)
        err_w = float(self.lstm_err_weight.item())

        for tid, items in by_t.items():
            items.sort(key=lambda it: it[0])
            if len(items) < self.min_track_len:
                continue
            frs = [it[0] for it in items]
            kt = torch.tensor(np.stack([it[1] for it in items]), dtype=torch.float32, device=device).unsqueeze(0)
            pred, _, _ = self.forward(kt)
            err = torch.zeros(len(frs), device=device)
            err[1:] = (pred[0, :-1] - kt[0, 1:]).pow(2).mean(dim=-1)
            pos_ex = ((kt[0].clamp(min=0.0) / scale) * w_dim).sum(dim=-1)
            raw = (pos_ex + err_w * err).cpu().numpy()
            # Causal trailing smoothing over the track trajectory itself
            sm = causal_moving_average_skip0(np.concatenate(([0.0], raw)), window=self.smoothing_window)[1:]
            for j, fr in enumerate(frs):
                if j + 1 >= self.min_track_len:
                    val = float(sm[j])
                    per_track_frame_scores.setdefault(fr, {})[tid] = val
                    if val > frame_scores[fr]:
                        frame_scores[fr] = val

        return frame_scores, per_track_frame_scores

    @torch.inference_mode()
    def step_frame_tracks(self, meta_frame: np.ndarray) -> Tuple[float, Dict[int, float]]:
        """Strict causal 1-frame update for active tracks in the current frame."""
        if len(meta_frame) == 0:
            return 0.0, {}

        kv = self.build_kinematics_vector(meta_frame)
        device = next(self.parameters()).device
        w_dim = self.dim_weights.to(device)
        scale = self.pos_scale.to(device)
        err_w = float(self.lstm_err_weight.item())

        active_ids = set()
        track_scores: Dict[int, float] = {}
        frame_max = 0.0

        for i in range(len(meta_frame)):
            tid = int(meta_frame[i, 1])
            active_ids.add(tid)
            st = self._stream_states.setdefault(tid, TrackStepState())
            xt = torch.tensor(kv[i], dtype=torch.float32, device=device).view(1, 1, self.in_dim)
            err = 0.0
            if st.prev_pred is not None:
                err = float((st.prev_pred - xt.squeeze(0).squeeze(0)).pow(2).mean().item())
            pred, new_h, _ = self.forward(xt, st.hidden)
            st.hidden = tuple(h.detach() for h in new_h)
            st.prev_pred = pred.squeeze(0).squeeze(0).detach()
            st.step_count += 1

            pos_ex = float(((xt.squeeze(0).squeeze(0).clamp(min=0.0) / scale) * w_dim).sum().item())
            raw_val = pos_ex + err_w * err
            st.raw_history.append(raw_val)
            if len(st.raw_history) > self.smoothing_window:
                st.raw_history.pop(0)

            if st.step_count >= self.min_track_len:
                sm_val = float(sum(st.raw_history) / len(st.raw_history))
                track_scores[tid] = sm_val
                if sm_val > frame_max:
                    frame_max = sm_val

        # Prune vanished tracks softly if state grows large
        if len(self._stream_states) > 256:
            stale = [tid for tid in self._stream_states if tid not in active_ids]
            for tid in stale[:128]:
                del self._stream_states[tid]

        return frame_max, track_scores
