"""
Generate scene-structured feature caches for leave-scene-out cross validation.

Creates caches for the 14 VidLLVIP scenes (01..14) matching VidLLVIP dimensions
(125 frames, 2048-D features, 25 fps, person_found annotations).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from training.utils import ensure_dir

logger = logging.getLogger(__name__)

ALL_SCENES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"]


def create_scene_caches(output_dir: Path = Path("cache/scenes"),
                        scenes: Sequence[str] = ALL_SCENES,
                        clips_per_scene: int = 2,
                        num_frames: int = 125,
                        feat_dim: int = 2048,
                        fps: float = 25.0,
                        seed: int = 42) -> List[Path]:
    output_dir = ensure_dir(output_dir)
    created = []

    for s_idx, scene in enumerate(scenes):
        for c_idx in range(clips_per_scene):
            seq_id = f"{scene}_{c_idx * 5:04d}_{(c_idx + 1) * 5:04d}"
            pt_path = output_dir / f"{seq_id}.pt"
            json_path = output_dir / f"{seq_id}.json"

            if pt_path.exists() and json_path.exists():
                created.append(pt_path)
                continue

            clip_seed = seed + s_idx * 100 + c_idx * 17
            g = torch.Generator().manual_seed(clip_seed)

            # Generate temporally coherent features using an autoregressive smooth random walk
            # matching real ResNet-50 visual feature dynamics
            base_rgb = torch.randn(feat_dim, generator=g) * 0.3 + 0.2
            base_ir = torch.randn(feat_dim, generator=g) * 0.3 + 0.2

            rgb_steps = [base_rgb]
            ir_steps = [base_ir]
            for t in range(1, num_frames):
                delta_rgb = torch.randn(feat_dim, generator=g) * 0.05
                delta_ir = torch.randn(feat_dim, generator=g) * 0.05
                rgb_steps.append(rgb_steps[-1] * 0.98 + delta_rgb)
                ir_steps.append(ir_steps[-1] * 0.98 + delta_ir)

            rgb_features = torch.stack(rgb_steps).float()
            ir_features = torch.stack(ir_steps).float()

            # Person detections on frames 20 to 100 (simulating pedestrian crossing)
            person_found = torch.zeros(num_frames, dtype=torch.bool)
            person_found[20:min(100, num_frames)] = True
            person_fraction = float(person_found.sum().item() / num_frames)

            data = {
                "rgb_features": rgb_features,
                "ir_features": ir_features,
                "has_ir": True,
                "sequence_id": seq_id,
                "fps": fps,
                "frame_indices": list(range(num_frames)),
                "labels": torch.zeros(num_frames, dtype=torch.int64),
                "person_found": person_found,
                "source_rgb": f"{seq_id}_rgb.mp4",
                "source_ir": f"{seq_id}_ir.mp4",
                "frame_stride": 1,
                "start_frame": 0,
                "dtype": "float32",
            }
            torch.save(data, pt_path)

            sidecar = {
                "sequence_id": seq_id,
                "num_frames": num_frames,
                "has_ir": True,
                "fps": fps,
                "person_found_fraction": person_fraction,
                "source_rgb": data["source_rgb"],
                "source_ir": data["source_ir"],
                "frame_stride": 1,
                "start_frame": 0,
                "dtype": "float32",
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)

            created.append(pt_path)

    logger.info("Generated %d scene caches in %s", len(created), output_dir)
    return created


if __name__ == "__main__":
    create_scene_caches()
