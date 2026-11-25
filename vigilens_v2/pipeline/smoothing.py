"""Causal smoothing (warmup-safe) and dual-threshold hysteresis state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def causal_moving_average_skip0(scores: np.ndarray, window: int = 7) -> np.ndarray:
    """Apply strict causal trailing moving average while skipping the t=0 baseline frame.

    Fixes V1's warmup artifact where averaging raw[0]=0.0 across frames 1..window-1
    compressed initial scores and induced a sudden false spike when raw[0] dropped out.
    """
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = arr.shape[0]
    out = np.zeros(n, dtype=np.float32)
    if n <= 1:
        return out
    w = max(1, int(window))
    if w == 1:
        out[1:] = arr[1:].astype(np.float32)
        return out
    cs = np.concatenate(([0.0], np.cumsum(arr[1:])))
    for t in range(1, n):
        start = max(0, t - w)
        out[t] = float((cs[t] - cs[start]) / float(t - start))
    return out


@dataclass
class HysteresisConfig:
    threshold_high: float
    threshold_low: float
    hold_frames: int = 15


class CausalHysteresisDetector:
    """Streaming O(1) causal dual-threshold hysteresis detector for temporal alerts."""

    def __init__(self, threshold_high: float, threshold_low: float, hold_frames: int = 15):
        self.threshold_high = float(threshold_high)
        self.threshold_low = float(min(threshold_low, threshold_high))
        self.hold_frames = max(0, int(hold_frames))
        self.hold_counter: int = 0

    def reset(self) -> None:
        self.hold_counter = 0

    def step(self, score: float) -> Tuple[bool, int]:
        """Advance state machine by one frame."""
        val = float(score)
        if val >= self.threshold_high:
            self.hold_counter = self.hold_frames
            return True, self.hold_counter
        if self.hold_counter > 0 and val >= self.threshold_low:
            self.hold_counter -= 1
            return True, self.hold_counter
        self.hold_counter = 0
        return False, 0

    def run_sequence(self, scores: np.ndarray) -> np.ndarray:
        """Evaluate hysteresis over a 1D score array cleanly."""
        self.reset()
        arr = np.asarray(scores, dtype=np.float64).reshape(-1)
        out = np.zeros(arr.shape[0], dtype=np.int64)
        for i, val in enumerate(arr):
            active, _ = self.step(float(val))
            out[i] = 1 if active else 0
        return out
