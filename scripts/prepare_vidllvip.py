"""
Prepare the VidLLVIP (Visible-Infrared Paired Video) dataset for VigiLens.

VidLLVIP (https://github.com/jianfeng0369/VidLLVIP) provides temporally and
spatially aligned 5-second paired visible (vi) and infrared (ir) video clips:

    dataset/
      vi/01_0000_0005.mp4, 01_0005_0010.mp4, ...
      ir/01_0000_0005.mp4, 01_0005_0010.mp4, ...

This script:
1. Discovers aligned VI/IR pairs matching by filename.
2. Groups clips by scene ID (e.g. "01", "02", ..., "14") to prevent data
   leakage across frames of the same scene.
3. Splits into train and val (by scene ID or ratio).
4. Creates paired videos under:
     Data/train/rgb/<seq_id>_rgb.mp4
     Data/train/ir/<seq_id>_ir.mp4
     Data/val/rgb/<seq_id>_rgb.mp4
     Data/val/ir/<seq_id>_ir.mp4
   using symlinks (or copies on Windows / when --copy is passed).
5. Optionally generates an evaluation manifest (Evaluation/manifest_vidllvip.json)
   for the held-out validation clips so evaluate.py runs seamlessly.

Example usage:
    # Prepare unzipped dataset directory:
    python scripts/prepare_vidllvip.py --input /content/VidLLVIP --output Data

    # Or point directly to dataset.zip:
    python scripts/prepare_vidllvip.py --input /content/dataset.zip --output Data

    # Fast preview with max 50 clips:
    python scripts/prepare_vidllvip.py --input /content/VidLLVIP --output Data --max-clips 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.utils import (
    DEFAULT_VIDEO_EXTENSIONS,
    ensure_dir,
    natural_key,
    safe_stem,
)

logger = logging.getLogger(__name__)


def extract_scene_id(filename: str) -> str:
    """
    Extract scene identifier from VidLLVIP filename (e.g. '01_0000_0005.mp4' -> '01').
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    return parts[0] if parts else stem


def discover_vidllvip_dirs(root: Path) -> Tuple[Path, Path]:
    """
    Locate the visible ('vi' or 'rgb') and infrared ('ir') directories.
    Accepts root as the dataset root, or a parent directory containing 'dataset/'.
    """
    candidates_vi = [
        root / "dataset" / "vi",
        root / "vi",
        root / "dataset" / "visible",
        root / "visible",
        root / "dataset" / "rgb",
        root / "rgb",
    ]
    candidates_ir = [
        root / "dataset" / "ir",
        root / "ir",
        root / "dataset" / "infrared",
        root / "infrared",
        root / "dataset" / "thermal",
        root / "thermal",
    ]

    vi_dir = next((d for d in candidates_vi if d.is_dir()), None)
    ir_dir = next((d for d in candidates_ir if d.is_dir()), None)

    if vi_dir is None or ir_dir is None:
        # Fallback search across directory tree
        for d in root.rglob("*"):
            if d.is_dir():
                name = d.name.lower()
                if vi_dir is None and name in {"vi", "visible"}:
                    vi_dir = d
                elif ir_dir is None and name in {"ir", "infrared", "thermal"}:
                    ir_dir = d

    if vi_dir is None or ir_dir is None:
        raise FileNotFoundError(
            f"Could not locate both visible and infrared directories under {root}. "
            f"Expected 'vi/' (or 'dataset/vi/') and 'ir/' (or 'dataset/ir/')."
        )

    return vi_dir, ir_dir


def discover_vidllvip_pairs(vi_dir: Path, ir_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """
    Discover matching visible and infrared video pairs by filename.

    Returns:
        List of (vi_path, ir_path, sequence_id, scene_id)
    """
    allowed_exts = set(DEFAULT_VIDEO_EXTENSIONS)
    vi_files = {
        f.name: f
        for f in vi_dir.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_exts
    }
    ir_files = {
        f.name: f
        for f in ir_dir.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_exts
    }

    common_names = sorted(set(vi_files) & set(ir_files), key=natural_key)
    pairs = []
    for name in common_names:
        vi_path = vi_files[name]
        ir_path = ir_files[name]
        seq_id = safe_stem(vi_path.stem)
        scene_id = extract_scene_id(name)
        pairs.append((vi_path, ir_path, seq_id, scene_id))

    return pairs


def split_by_scenes(pairs: List[Tuple[Path, Path, str, str]],
                    val_scenes: Optional[Sequence[str]] = None,
                    val_ratio: float = 0.2,
                    seed: int = 42) -> Tuple[List[Tuple[Path, Path, str, str]], List[Tuple[Path, Path, str, str]]]:
    """
    Split pairs into train and val keeping scenes disjoint to avoid frame leakage.
    """
    # Group pairs by scene
    by_scene: Dict[str, List[Tuple[Path, Path, str, str]]] = {}
    for item in pairs:
        by_scene.setdefault(item[3], []).append(item)

    unique_scenes = sorted(by_scene.keys(), key=natural_key)
    if not unique_scenes:
        return [], []

    if val_scenes:
        val_set = {s.strip().zfill(2) if s.strip().isdigit() else s.strip() for s in val_scenes}
        # Also check without leading zero
        val_set |= {s.lstrip("0") for s in val_scenes if s.strip()}
    else:
        rng = random.Random(seed)
        shuffled = list(unique_scenes)
        rng.shuffle(shuffled)
        num_val_scenes = max(1, int(round(len(unique_scenes) * val_ratio)))
        val_set = set(shuffled[:num_val_scenes])

    train_pairs: List[Tuple[Path, Path, str, str]] = []
    val_pairs: List[Tuple[Path, Path, str, str]] = []

    for scene_id, scene_pairs in by_scene.items():
        if scene_id in val_set or scene_id.lstrip("0") in val_set or scene_id.zfill(2) in val_set:
            val_pairs.extend(scene_pairs)
        else:
            train_pairs.extend(scene_pairs)

    train_pairs.sort(key=lambda item: natural_key(item[2]))
    val_pairs.sort(key=lambda item: natural_key(item[2]))
    return train_pairs, val_pairs


def _link_or_copy(src: Path, dst: Path, copy_only: bool = False) -> None:
    """Create a symlink, falling back to copy if unsupported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_only:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        # Fallback to copy on Windows without symlink privileges
        shutil.copy2(src, dst)


def prepare_vidllvip(input_path: Path,
                     output_dir: Path,
                     scenes: Optional[Sequence[str]] = None,
                     val_scenes: Optional[Sequence[str]] = None,
                     val_ratio: float = 0.2,
                     all_train: bool = False,
                     max_clips: Optional[int] = None,
                     copy_files: bool = False,
                     eval_manifest: Optional[Path] = None,
                     seed: int = 42) -> Dict[str, int]:
    """
    Main preparation workflow for VidLLVIP.
    """
    # Check if input is a zip file
    temp_extract_dir: Optional[Path] = None
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        logger.info(f"Extracting zip archive: {input_path}")
        temp_extract_dir = input_path.parent / (input_path.stem + "_extracted")
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(temp_extract_dir)
        root = temp_extract_dir
    else:
        root = input_path

    vi_dir, ir_dir = discover_vidllvip_dirs(root)
    pairs = discover_vidllvip_pairs(vi_dir, ir_dir)

    if not pairs:
        raise ValueError(f"No matching VI/IR video pairs found in {vi_dir} and {ir_dir}")

    logger.info(f"Discovered {len(pairs)} matching VidLLVIP pairs across scenes")

    # Filter by specific scenes if requested
    if scenes:
        allowed = {s.strip().zfill(2) if s.strip().isdigit() else s.strip() for s in scenes}
        allowed |= {s.lstrip("0") for s in scenes if s.strip()}
        pairs = [p for p in pairs if p[3] in allowed or p[3].zfill(2) in allowed or p[3].lstrip("0") in allowed]
        if not pairs:
            raise ValueError(f"No pairs matched --scenes {scenes}")

    # Split train / val
    if all_train:
        train_pairs, val_pairs = pairs, []
    else:
        train_pairs, val_pairs = split_by_scenes(
            pairs, val_scenes=val_scenes, val_ratio=val_ratio, seed=seed
        )

    # Apply max_clips if specified (stratified across splits)
    if max_clips is not None and max_clips > 0:
        total = len(train_pairs) + len(val_pairs)
        if total > max_clips:
            if val_pairs:
                train_quota = max(1, int(round(max_clips * (len(train_pairs) / total))))
                val_quota = max(1, max_clips - train_quota)
                train_pairs = train_pairs[:train_quota]
                val_pairs = val_pairs[:val_quota]
            else:
                train_pairs = train_pairs[:max_clips]

    output_dir = Path(output_dir)
    splits = [("train", train_pairs), ("val", val_pairs)]
    counts = {"train": len(train_pairs), "val": len(val_pairs)}

    manifest_entries = []

    for split_name, split_pairs in splits:
        rgb_dest_dir = output_dir / split_name / "rgb"
        ir_dest_dir = output_dir / split_name / "ir"
        rgb_dest_dir.mkdir(parents=True, exist_ok=True)
        ir_dest_dir.mkdir(parents=True, exist_ok=True)

        for vi_path, ir_path, seq_id, _ in split_pairs:
            rgb_dest = rgb_dest_dir / f"{seq_id}_rgb.mp4"
            ir_dest = ir_dest_dir / f"{seq_id}_ir.mp4"

            _link_or_copy(vi_path, rgb_dest, copy_only=copy_files)
            _link_or_copy(ir_path, ir_dest, copy_only=copy_files)

            if split_name == "val" and eval_manifest is not None:
                manifest_dir = eval_manifest.parent
                manifest_entries.append({
                    "sequence_id": seq_id,
                    "rgb": os.path.relpath(rgb_dest, manifest_dir).replace("\\", "/"),
                    "ir": os.path.relpath(ir_dest, manifest_dir).replace("\\", "/"),
                    "label": "normal",
                })

    if eval_manifest is not None and manifest_entries:
        eval_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {
            "name": "vidllvip_val",
            "description": "Held-out normal validation clips from VidLLVIP",
            "videos": manifest_entries,
        }
        with open(eval_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest_data, handle, indent=2)
        logger.info(f"Wrote evaluation manifest with {len(manifest_entries)} clips to {eval_manifest}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare VidLLVIP visible + infrared video dataset for VigiLens"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to VidLLVIP dataset root (or dataset.zip)",
    )
    parser.add_argument(
        "--output",
        default="Data",
        help="Destination directory for Data/train and Data/val (default: Data)",
    )
    parser.add_argument(
        "--scenes",
        default=None,
        help="Comma-separated scene IDs to include (e.g. 01,02,03). Default: all scenes.",
    )
    parser.add_argument(
        "--val-scenes",
        default=None,
        help="Comma-separated scene IDs to reserve for validation (e.g. 12,13,14).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of scenes held out for validation when --val-scenes is not given (default: 0.2)",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Place all discovered clips into Data/train (no validation split).",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit total number of clips (useful for fast smoke tests or previews).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy video files instead of creating symlinks.",
    )
    parser.add_argument(
        "--eval-manifest",
        default="Evaluation/manifest_vidllvip.json",
        help="Path to generate evaluation manifest for validation clips (default: Evaluation/manifest_vidllvip.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic scene splitting (default: 42)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    val_scenes = [s.strip() for s in args.val_scenes.split(",")] if args.val_scenes else None
    eval_manifest = Path(args.eval_manifest) if args.eval_manifest else None

    counts = prepare_vidllvip(
        input_path=input_path,
        output_dir=Path(args.output),
        scenes=scenes,
        val_scenes=val_scenes,
        val_ratio=args.val_ratio,
        all_train=args.all_train,
        max_clips=args.max_clips,
        copy_files=args.copy,
        eval_manifest=eval_manifest,
        seed=args.seed,
    )

    print(
        f"\n[VidLLVIP Preparation Complete]\n"
        f"  Output directory: {args.output}\n"
        f"  Train pairs:     {counts['train']}\n"
        f"  Val pairs:       {counts['val']}\n"
        f"  Eval manifest:   {eval_manifest if eval_manifest else 'None'}\n"
    )


if __name__ == "__main__":
    main()
