"""Visual feature extractors for VigiLens V2 (ImageNet RGB, Market-1501 Re-ID, Thermal IR)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import IRFeatureExtractor, RGBFeatureExtractor


class DualVisualExtractor:
    """Extracts 2048-D visual representations with optional unit-L2 Re-ID and IR support."""

    def __init__(
        self,
        device: str = "cpu",
        input_size: Tuple[int, int] = (224, 224),
        reid_weights_path: Optional[str] = None,
        load_ir: bool = False,
        reid_scale: float = 1.0,
    ):
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.reid_scale = float(reid_scale)
        self.rgb_extractor = RGBFeatureExtractor(device=self.device, input_size=input_size)
        self.ir_extractor = IRFeatureExtractor(device=self.device, input_size=input_size) if load_ir else None

        self.reid_extractor: Optional[RGBFeatureExtractor] = None
        if reid_weights_path and Path(reid_weights_path).exists():
            ext = RGBFeatureExtractor(device=self.device, input_size=input_size, weights_path=reid_weights_path)
            self.reid_extractor = ext

    def _to_1d_tensor(self, feat) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            return feat.detach().reshape(-1).to(dtype=torch.float32, device="cpu")
        return torch.as_tensor(np.asarray(feat), dtype=torch.float32).reshape(-1).cpu()

    def extract_rgb(self, frame_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """Extract 2048-D ImageNet ResNet-50 feature vector on host device."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            feat = self.rgb_extractor.extract(rgb, mask=mask)
        return self._to_1d_tensor(feat)

    def extract_reid(self, frame_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[torch.Tensor]:
        """Extract unit-L2 scaled 2048-D Market-1501 Re-ID feature vector."""
        if self.reid_extractor is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            feat = self._to_1d_tensor(self.reid_extractor.extract(rgb, mask=mask))
            return F.normalize(feat, dim=-1) * self.reid_scale

    def extract_ir(self, ir_frame: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[torch.Tensor]:
        """Extract 2048-D ResNet-50 Thermal IR feature vector."""
        if self.ir_extractor is None:
            return None
        with torch.inference_mode():
            feat = self.ir_extractor.extract(ir_frame, mask=mask)
        return self._to_1d_tensor(feat)
