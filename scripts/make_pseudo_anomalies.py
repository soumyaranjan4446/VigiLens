"""
Controlled Pseudo-Anomaly Generator for VigiLens Evaluation.

VidLLVIP contains no labeled anomalies. To evaluate detection quality and benchmark
scoring and calibration without data leakage, this script generates controlled proxy
anomalies on held-out normal scenes only.

Tier A (Feature-space temporal anomalies — fast, exact, zero re-extraction):
  - reverse: reverse temporal frame order inside a segment [s, e).
  - speedup_x2 / speedup_x4: fast-forward time with stride 2 or 4.
  - freeze: replicate a single frame's features across [s, e).
  - shuffle: random permutation of frames within [s, e).
  - jumpcut: splice features from a disjoint scene into [s, e).

Tier B (Video-level appearance anomalies):
  - blur, gaussian_noise, brightness_shift, occlusion_patch, modality_mismatch.

All generated sequences have frame-level ground-truth labels and standard manifest
segments [[s, e]] consumed directly by evaluate.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from training.utils import ensure_dir, safe_stem

logger = logging.getLogger(__name__)

TIER_A_TYPES = ("reverse", "speedup_x2", "speedup_x4", "freeze", "shuffle", "jumpcut")
TIER_B_TYPES = ("blur", "gaussian_noise", "brightness_shift", "occlusion_patch", "modality_mismatch")


def _sample_segment(num_frames: int,
                    min_len: int = 20,
                    max_len: int = 60,
                    min_warmup: int = 25,
                    rng: Optional[random.Random] = None) -> Tuple[int, int]:
    """Sample an anomaly segment [start, end) satisfying warm-up and length bounds."""
    rng = rng or random.Random()
    if num_frames < 10:
        return 0, num_frames

    warmup = min(min_warmup, max(1, num_frames // 4))
    remaining = num_frames - warmup
    if remaining < 4:
        return warmup, num_frames

    max_possible_len = min(max_len, remaining)
    min_possible_len = max(2, min(min_len, max_possible_len))

    seg_len = rng.randint(min_possible_len, max_possible_len)
    max_start = num_frames - seg_len
    start = rng.randint(warmup, max(warmup, max_start))
    end = min(num_frames, start + seg_len)
    return start, end


def apply_tier_a_anomaly(data: Dict[str, Any],
                         anomaly_type: str,
                         donor_data: Optional[Dict[str, Any]] = None,
                         min_len: int = 20,
                         max_len: int = 60,
                         min_warmup: int = 25,
                         seed: int = 42) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    """Inject a Tier A temporal anomaly into a feature cache dictionary.

    Returns:
        (perturbed_data, segments) where segments is [[start, end], ...]
    """
    rng = random.Random(seed)
    torch_rng = torch.Generator().manual_seed(seed)

    out = copy.deepcopy(data)
    rgb = out["rgb_features"].clone()
    ir = out["ir_features"].clone() if out.get("ir_features") is not None else None
    num_frames = rgb.shape[0]

    start, end = _sample_segment(num_frames, min_len=min_len, max_len=max_len,
                                 min_warmup=min_warmup, rng=rng)
    seg_len = end - start

    if anomaly_type == "reverse":
        idx = torch.arange(end - 1, start - 1, -1, dtype=torch.long)
        rgb[start:end] = rgb[idx]
        if ir is not None:
            ir[start:end] = ir[idx]

    elif anomaly_type in ("speedup_x2", "speedup_x4"):
        stride = 2 if anomaly_type == "speedup_x2" else 4
        # Stride through from start into future frames
        available = num_frames - start
        needed_src = seg_len * stride
        if available >= needed_src:
            src_indices = torch.arange(start, start + needed_src, stride, dtype=torch.long)[:seg_len]
        else:
            # Wrap or repeat if not enough future frames
            cand = torch.arange(start, num_frames, stride, dtype=torch.long)
            if cand.numel() == 0:
                cand = torch.tensor([start], dtype=torch.long)
            repeat_count = int(np.ceil(seg_len / cand.numel()))
            src_indices = cand.repeat(repeat_count)[:seg_len]

        rgb[start:end] = rgb[src_indices]
        if ir is not None:
            ir[start:end] = ir[src_indices]

    elif anomaly_type == "freeze":
        # Repeat the single frame feature across the segment
        rgb[start:end] = rgb[start:start + 1].expand(seg_len, -1)
        if ir is not None:
            ir[start:end] = ir[start:start + 1].expand(seg_len, -1)

    elif anomaly_type == "shuffle":
        perm = torch.randperm(seg_len, generator=torch_rng)
        rgb[start:end] = rgb[start:end][perm]
        if ir is not None:
            ir[start:end] = ir[start:end][perm]

    elif anomaly_type == "jumpcut":
        if donor_data is not None and "rgb_features" in donor_data:
            donor_rgb = donor_data["rgb_features"]
            donor_ir = donor_data.get("ir_features")
            donor_len = donor_rgb.shape[0]
            d_start = rng.randint(0, max(0, donor_len - seg_len)) if donor_len > seg_len else 0
            d_indices = torch.arange(d_start, min(donor_len, d_start + seg_len), dtype=torch.long)
            if d_indices.numel() < seg_len:
                d_indices = d_indices.repeat(int(np.ceil(seg_len / max(1, d_indices.numel()))))[:seg_len]
            rgb[start:end] = donor_rgb[d_indices].clone()
            if ir is not None and donor_ir is not None:
                ir[start:end] = donor_ir[d_indices].clone()
        else:
            # Fallback if no donor provided: permuted reverse
            idx = torch.arange(end - 1, start - 1, -1, dtype=torch.long)
            rgb[start:end] = rgb[idx]
            if ir is not None:
                ir[start:end] = ir[idx]
    else:
        raise ValueError(f"Unknown Tier A anomaly type: {anomaly_type}")

    labels = torch.zeros(num_frames, dtype=torch.int64)
    labels[start:end] = 1

    out["rgb_features"] = rgb
    out["ir_features"] = ir
    out["labels"] = labels
    out["anomaly_type"] = anomaly_type
    out["pseudo_anomaly"] = True
    out["segments"] = [[int(start), int(end)]]

    return out, [[int(start), int(end)]]


def generate_tier_a_dataset(cache_paths: Sequence[Union[str, Path]],
                            output_dir: Union[str, Path],
                            types: Sequence[str] = TIER_A_TYPES,
                            copies_per_clip: int = 1,
                            seed: int = 42) -> Dict[str, Any]:
    """Generate a suite of Tier A pseudo-anomalies from normal cache paths.

    Creates per-type subdirectories and a master manifest.
    """
    output_dir = ensure_dir(output_dir)
    cache_paths = [Path(p) for p in cache_paths if Path(p).is_file() and Path(p).suffix == ".pt"]
    if not cache_paths:
        raise ValueError("No valid .pt cache paths provided")

    all_caches = [torch.load(p, map_location="cpu", weights_only=False) for p in cache_paths]
    manifest_videos: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {t: 0 for t in types}

    rng = random.Random(seed)

    for anomaly_type in types:
        type_dir = ensure_dir(output_dir / anomaly_type)
        for clip_idx, (orig_path, orig_data) in enumerate(zip(cache_paths, all_caches)):
            for copy_i in range(copies_per_clip):
                # Pick a donor from a different scene for jumpcut
                donor_data = None
                if anomaly_type == "jumpcut" and len(all_caches) > 1:
                    orig_scene = orig_data.get("sequence_id", orig_path.stem).split("_")[0]
                    candidates = [
                        c for c in all_caches
                        if c.get("sequence_id", "").split("_")[0] != orig_scene
                    ]
                    donor_data = rng.choice(candidates) if candidates else rng.choice(all_caches)

                cur_seed = seed + clip_idx * 100 + copy_i * 13
                perturbed, segments = apply_tier_a_anomaly(
                    orig_data,
                    anomaly_type=anomaly_type,
                    donor_data=donor_data,
                    seed=cur_seed,
                )

                seq_id = f"{orig_path.stem}_{anomaly_type}_{copy_i}"
                perturbed["sequence_id"] = seq_id
                save_path = type_dir / f"{seq_id}.pt"
                sidecar_path = save_path.with_suffix(".json")

                torch.save(perturbed, save_path)

                sidecar = {
                    "sequence_id": seq_id,
                    "num_frames": int(perturbed["rgb_features"].shape[0]),
                    "has_ir": bool(perturbed.get("has_ir", perturbed.get("ir_features") is not None)),
                    "fps": float(perturbed.get("fps", 20.0)),
                    "anomaly_type": anomaly_type,
                    "segments": segments,
                    "is_pseudo_anomaly": True,
                }
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)

                manifest_videos.append({
                    "sequence_id": seq_id,
                    "cache": str(save_path.resolve()),
                    "label": "anomaly",
                    "anomaly_type": anomaly_type,
                    "segments": segments,
                })
                type_counts[anomaly_type] += 1

    manifest = {
        "name": "vidllvip_pseudo_anomalies_tier_a",
        "description": "Controlled feature-space temporal pseudo-anomalies (Tier A proxy benchmark)",
        "is_proxy_benchmark": True,
        "type_counts": type_counts,
        "videos": manifest_videos,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generated %d pseudo-anomalies across types %s -> %s", len(manifest_videos), types, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled pseudo-anomalies for VigiLens evaluation")
    parser.add_argument("--cache-dir", required=True, help="Directory containing source normal .pt caches")
    parser.add_argument("--output-dir", required=True, help="Destination directory for pseudo-anomalies")
    parser.add_argument("--tier", choices=["A", "B"], default="A", help="Tier A (feature-space) or Tier B (video)")
    parser.add_argument("--types", default=None, help="Comma-separated anomaly types to generate (default: all)")
    parser.add_argument("--copies-per-clip", type=int, default=1, help="Number of anomaly variants per clip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    selected_types = [t.strip() for t in args.types.split(",")] if args.types else TIER_A_TYPES
    cache_paths = sorted(Path(args.cache_dir).rglob("*.pt"))
    if not cache_paths:
        raise SystemExit(f"No .pt files found in {args.cache_dir}")

    manifest = generate_tier_a_dataset(
        cache_paths=cache_paths,
        output_dir=args.output_dir,
        types=selected_types,
        copies_per_clip=args.copies_per_clip,
        seed=args.seed,
    )
    print(f"\n[Generated {len(manifest['videos'])} pseudo-anomalies] Manifest: {args.output_dir}/manifest.json")
    for t, cnt in manifest["type_counts"].items():
        print(f"  {t:15s}: {cnt} sequences")


if __name__ == "__main__":
    main()
