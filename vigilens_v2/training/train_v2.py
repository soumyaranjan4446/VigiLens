"""Train & calibrate VigiLens V2 strictly on normal training videos (zero test leakage).

Usage:
    python -m vigilens_v2.training.train_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vigilens_v2.config import get_v2_config
from vigilens_v2.models.feature_extractor import DualVisualExtractor
from vigilens_v2.models.motion_encoder import CausalMotionFlowEncoder
from vigilens_v2.models.tracker import YOLOCausalTracker
from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model
from vigilens_v2.training.calibrate_v2 import calibrate_v2_from_normal_clips


def _extract_motion_from_video(
    video_path: Path,
    tracker: YOLOCausalTracker,
    motion_enc: CausalMotionFlowEncoder,
) -> Dict[str, np.ndarray]:
    """Extract tracked boxes and Farneback optical flow grid strictly causally from MP4."""
    tracker.reset()
    motion_enc.reset()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open training video: {video_path}")
    grids = []
    metas = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tracks, _ = tracker.process_frame(frame)
        g, m = motion_enc.compute_frame_flow_and_track_stats(frame, tracks, frame_index=idx)
        grids.append(g)
        if len(m) > 0:
            metas.append(m)
        idx += 1
    cap.release()
    return {
        "N": idx,
        "grid_flow": np.stack(grids, axis=0) if grids else np.zeros((0, 12, 18), dtype=np.float32),
        "meta": np.concatenate(metas, axis=0) if metas else np.zeros((0, 13), dtype=np.float32),
    }


def load_or_build_ped2_train_clips(
    train_ids: List[str],
    probe_cache_dir: Path,
    img_cache_dir: Path,
    reid_cache_dir: Path,
    video_dir: Path,
) -> List[Dict[str, object]]:
    """Load cached normal training features/tracks or extract automatically if cache missing."""
    cfg = get_v2_config()
    tracker = None
    motion_enc = None
    vis_ext = None

    clips = []
    for cid in train_ids:
        probe_p = probe_cache_dir / f"{cid}.pt"
        img_p = img_cache_dir / f"{cid}.pt"
        reid_p = reid_cache_dir / f"{cid}.pt"
        vid_p = video_dir / f"{cid}_rgb.mp4"

        if probe_p.exists():
            d_probe = torch.load(probe_p, map_location="cpu", weights_only=False)
        else:
            if tracker is None:
                tracker = YOLOCausalTracker(str(ROOT / cfg.tracker.yolo_weights))
                motion_enc = CausalMotionFlowEncoder()
            d_probe = _extract_motion_from_video(vid_p, tracker, motion_enc)

        if img_p.exists():
            rgb_feats = torch.load(img_p, map_location="cpu", weights_only=False)["rgb_features"].float()
        else:
            if vis_ext is None:
                vis_ext = DualVisualExtractor(device="cpu", reid_weights_path=str(ROOT / cfg.model.reid_weights))
            cap = cv2.VideoCapture(str(vid_p))
            feats = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                feats.append(vis_ext.extract_rgb(fr))
            cap.release()
            rgb_feats = torch.stack(feats, dim=0)

        reid_feats = (
            torch.load(reid_p, map_location="cpu", weights_only=False)["rgb_features"].float()
            if reid_p.exists()
            else None
        )

        clips.append(
            {
                "id": cid,
                "grid_flow": d_probe["grid_flow"],
                "meta": d_probe["meta"],
                "rgb_features": rgb_feats,
                "reid_features": reid_feats,
            }
        )
    return clips


def train_and_calibrate_ped2(
    probe_cache_dir: Path,
    output_ckpt: Path,
    calib_json: Path,
    epochs: int = 25,
    percentile_high: float = 95.0,
    percentile_low: float = 80.0,
) -> Dict[str, object]:
    cfg = get_v2_config()
    train_ids = ["Train001", "Train002", "Train003", "Train004"]

    t0 = time.perf_counter()
    clips = load_or_build_ped2_train_clips(
        train_ids=train_ids,
        probe_cache_dir=probe_cache_dir,
        img_cache_dir=ROOT / "cache" / "ucsd_ped2_train",
        reid_cache_dir=ROOT / "cache" / "ucsd_ped2_train_reid",
        video_dir=ROOT / "Data" / "ucsd_ped2" / "train_videos",
    )

    model = VigiLensV2Model(
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        fusion_output_dim=cfg.model.fusion_output_dim,
        lstm_hidden_size=cfg.model.lstm_hidden_size,
        lstm_layers=cfg.model.lstm_layers,
        lstm_dropout=cfg.model.lstm_dropout,
        track_in_dim=cfg.model.track_feature_dim,
        track_hidden_dim=cfg.model.track_hidden_dim,
    )
    v1_path = ROOT / cfg.model.v1_checkpoint
    if v1_path.exists():
        model.load_v1_backbone(str(v1_path))

    model.fit_grid_baseline([c["grid_flow"] for c in clips])
    track_loss = model.track_model.train_on_normal_meta(
        [c["meta"] for c in clips],
        epochs=epochs,
        lr=1e-3,
        seq_len=10,
        seed=42,
    )

    calib = calibrate_v2_from_normal_clips(
        model,
        clips,
        percentile_high=percentile_high,
        percentile_low=percentile_low,
        hold_frames=cfg.decision.hysteresis_hold_frames,
    )
    calib["track_train_loss"] = float(track_loss)
    calib["train_seconds"] = float(time.perf_counter() - t0)
    calib["train_ids"] = train_ids

    model.save_v2_checkpoint(str(output_ckpt))
    calib_json.parent.mkdir(parents=True, exist_ok=True)
    calib_json.write_text(json.dumps(calib, indent=2), encoding="utf-8")
    return calib


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & calibrate VigiLens V2 on UCSD Ped2 Normal Train.")
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=Path(r"C:\Users\Administrator\.gemini\antigravity\brain\bc3d31f5-6c92-49fd-81c3-a336251907a8\scratch\v2_probe_cache"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "vigilens_v2" / "artifacts" / "vigilens_v2_ped2.pt",
    )
    parser.add_argument(
        "--calib-json",
        type=Path,
        default=ROOT / "vigilens_v2" / "results" / "ped2_v2_calibration.json",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--p-high", type=float, default=95.0)
    parser.add_argument("--p-low", type=float, default=80.0)
    args = parser.parse_args()

    calib = train_and_calibrate_ped2(
        probe_cache_dir=args.probe_cache,
        output_ckpt=args.output,
        calib_json=args.calib_json,
        epochs=args.epochs,
        percentile_high=args.p_high,
        percentile_low=args.p_low,
    )
    print(f"Saved V2 checkpoint -> {args.output}")
    print(f"Calibration summary -> {json.dumps(calib, indent=2)}")


if __name__ == "__main__":
    main()
