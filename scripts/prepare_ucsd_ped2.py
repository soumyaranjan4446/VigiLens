"""
Prepare UCSD Ped2 dataset for VigiLens evaluation.

Extracts UCSDped2 from the downloaded archive, encodes image sequences to .mp4 videos,
extracts frame-level ground-truth labels from the _gt folders, and generates an
evaluation manifest for `python -m training.precompute` and `python -m training.evaluate`.
"""

import os
import re
import json
import tarfile
from pathlib import Path
import cv2
import numpy as np


def extract_ucsdped2(tar_path: Path, extract_to: Path) -> Path:
    """Extract only UCSDped2 files from UCSD_Anomaly_Dataset.tar.gz."""
    print(f"Extracting UCSDped2 from {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "UCSDped2" in m.name]
        tar.extractall(path=extract_to, members=members)
    print(f"Extracted {len(members)} files.")

    # Locate UCSDped2 root
    candidate = extract_to / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
    if candidate.exists():
        return candidate
    for p in extract_to.rglob("UCSDped2"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not find UCSDped2 directory after extraction.")


def convert_clips_to_mp4(ped2_root: Path, output_dir: Path) -> Path:
    """Convert TestXXX image folders to mp4 videos and labels."""
    test_dir = ped2_root / "Test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found in {ped2_root}")

    videos_dir = output_dir / "videos"
    labels_dir = output_dir / "labels"
    videos_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    clip_folders = sorted([d for d in test_dir.iterdir() if d.is_dir() and not d.name.endswith("_gt") and d.name.startswith("Test")])
    manifest_entries = []

    print(f"Found {len(clip_folders)} test clips to convert...")
    for folder in clip_folders:
        clip_name = folder.name
        img_files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        if not img_files:
            continue

        first_img = cv2.imread(str(img_files[0]))
        h, w = first_img.shape[:2]

        video_path = videos_dir / f"{clip_name}_rgb.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))

        gt_dir = test_dir / f"{clip_name}_gt"
        labels = []

        for idx, img_p in enumerate(img_files):
            img = cv2.imread(str(img_p))
            writer.write(img)

            # Check ground truth
            # Frame is anomalous if pixel ground truth has positive values
            is_anomaly = 0
            if gt_dir.exists():
                gt_file = gt_dir / f"{img_p.stem}.bmp"
                if not gt_file.exists():
                    gt_file = gt_dir / f"{int(img_p.stem):03d}.bmp"
                if gt_file.exists():
                    gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None and (gt_mask > 0).any():
                        is_anomaly = 1
            labels.append(is_anomaly)

        writer.release()

        # Save label file
        labels_path = labels_dir / f"{clip_name}_labels.txt"
        with open(labels_path, "w", encoding="utf-8") as f:
            for l in labels:
                f.write(f"{l}\n")

        # Calculate segments of anomalous frames
        segments = []
        in_seg = False
        start = 0
        for i, l in enumerate(labels):
            if l == 1 and not in_seg:
                in_seg = True
                start = i
            elif l == 0 and in_seg:
                in_seg = False
                segments.append([start, i])
        if in_seg:
            segments.append([start, len(labels)])

        has_anomaly = any(l == 1 for l in labels)
        manifest_entries.append({
            "sequence_id": clip_name,
            "rgb": str(video_path.resolve()),
            "frame_labels": str(labels_path.resolve()),
            "segments": segments,
            "label": "anomaly" if has_anomaly else "normal",
            "num_frames": len(img_files),
            "num_anomaly_frames": sum(labels),
        })
        print(f"  Processed {clip_name}: {len(img_files)} frames, {sum(labels)} anomaly frames -> {video_path.name}")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"name": "ucsd_ped2_test", "videos": manifest_entries}, f, indent=2)

    print(f"Manifest written to {manifest_path} with {len(manifest_entries)} clips.")
    return manifest_path


def main():
    base_dir = Path("data/ucsd_ped2")
    tar_path = Path("data/UCSD_Anomaly_Dataset.tar.gz")
    if not tar_path.exists():
        raise FileNotFoundError(f"Archive not found: {tar_path}")

    ped2_root = extract_ucsdped2(tar_path, base_dir)
    manifest = convert_clips_to_mp4(ped2_root, base_dir)
    print("UCSD Ped2 preparation complete!")


if __name__ == "__main__":
    main()
