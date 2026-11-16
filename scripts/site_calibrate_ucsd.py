"""
Site-Calibration for UCSD Ped2.

Performs unsupervised site-calibration using the official normal training sequences
(Train001-Train004, representing ~2 minutes of ambient sidewalk traffic).
Generates `checkpoints/calibration_ucsd.json` with:
- Site-specific diagonal residual variance (512-D)
- Site-specific threshold (mean + 3*std, p95, and conformal)
- Score variant: norm_mse

Then runs evaluation on the 12 test sequences and prints the comprehensive report.
"""

import sys
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import torch
import numpy as np
from tqdm import tqdm

from models.anomaly_model import VigiLensModel
from training.calibration import AnomalyCalibrator
from training.evaluate import evaluate_manifest
from training.metrics import compute_event_metrics, compute_false_alarm_rate


def convert_train_clips(train_dir: Path, output_videos_dir: Path, clip_names=None):
    """Convert specified Train image folders to MP4 videos."""
    output_videos_dir.mkdir(parents=True, exist_ok=True)
    all_folders = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("Train")])
    if clip_names is not None:
        all_folders = [d for d in all_folders if d.name in clip_names]

    video_paths = []
    print(f"Converting {len(all_folders)} normal training clips to MP4...")
    for folder in all_folders:
        img_files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        if not img_files:
            continue
        first_img = cv2.imread(str(img_files[0]))
        h, w = first_img.shape[:2]
        video_p = output_videos_dir / f"{folder.name}_rgb.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_p), fourcc, 10.0, (w, h))
        for img_p in img_files:
            img = cv2.imread(str(img_p))
            writer.write(img)
        writer.release()
        video_paths.append(video_p)
        print(f"  Converted {folder.name} ({len(img_files)} frames) -> {video_p.name}")
    return video_paths


def precompute_train_clips(video_paths, output_cache_dir: Path, device="cpu"):
    """Precompute features for normal training clips."""
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    from training.precompute import precompute_pair, build_extractors
    from models.segmentation import YOLOSegmentation

    seg = YOLOSegmentation(device=device)
    rgb_ext, ir_ext = build_extractors(device=device)

    cache_paths = []
    print(f"Precomputing features for {len(video_paths)} calibration clips...")
    for v_p in tqdm(video_paths, desc="Precomputing normal clips"):
        c_p = precompute_pair(
            rgb_path=v_p,
            ir_path=None,
            output_dir=output_cache_dir,
            segmentation=seg,
            rgb_extractor=rgb_ext,
            ir_extractor=ir_ext,
            batch_size=16,
            sequence_id=v_p.stem.replace("_rgb", ""),
        )
        if c_p is not None:
            cache_paths.append(c_p)
    return cache_paths


def calibrate_and_save(cache_paths, checkpoint_path: Path, output_json: Path, device="cpu"):
    """Run AnomalyCalibrator on normal caches with norm_mse and save calibration JSON."""
    model = VigiLensModel(device=device, load_extractors=False)
    model.load_checkpoint(str(checkpoint_path))
    model.to(device)
    model.eval()

    calibrator = AnomalyCalibrator(model, k_std=3.0, percentile=95.0)

    # 1. Compute site-specific diagonal residual variance
    print("Computing site-specific diagonal residual variance across normal frames...")
    diag_var = calibrator.compute_residual_variance(cache_paths, device=device)

    # 2. Collect normal prediction errors using norm_mse
    print("Collecting normal prediction errors with Mahalanobis weighting...")
    errors = calibrator.collect_errors_from_caches(
        cache_paths,
        device=device,
        score_variant="norm_mse",
        diag_variance=diag_var,
    )

    # 3. Compute calibration statistics
    info = calibrator.compute_threshold(errors, method="percentile")  # p95 calibrated
    info["score_variant"] = "norm_mse"
    info["diag_variance"] = diag_var.tolist()
    info["site_name"] = "ucsd_ped2"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Site calibration saved to {output_json}:")
    print(f"  Mean Error: {info['mean_error']:.4f}")
    print(f"  Std Error:  {info['std_error']:.4f}")
    print(f"  Threshold (p95): {info['threshold']:.4f}")
    print(f"  Threshold (3-std): {info['threshold_std']:.4f}")
    return info


def main():
    device = "cpu"
    ped2_train_dir = ROOT / "data" / "ucsd_ped2" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Train"
    train_videos_dir = ROOT / "data" / "ucsd_ped2" / "train_videos"
    train_cache_dir = ROOT / "cache" / "ucsd_ped2_train"
    checkpoint = ROOT / "checkpoints" / "best_model.pth"
    calibration_output = ROOT / "checkpoints" / "calibration_ucsd.json"

    # Step 1: Convert first 4 normal training clips (Train001 - Train004, ~500 frames, ~50s)
    calib_clips = ["Train001", "Train002", "Train003", "Train004"]
    v_paths = convert_train_clips(ped2_train_dir, train_videos_dir, calib_clips)

    # Step 2: Precompute normal features
    c_paths = precompute_train_clips(v_paths, train_cache_dir, device=device)

    # Step 3: Run calibration and save profile
    calib_info = calibrate_and_save(c_paths, checkpoint, calibration_output, device=device)

    # Step 4: Evaluate test set with new site-calibration profile
    print("\n" + "=" * 80)
    print("Evaluating Test Set with Site Calibration Profile...")
    print("=" * 80)

    model = VigiLensModel(device=device, load_extractors=False)
    model.load_checkpoint(str(checkpoint))
    model.to(device)

    results = evaluate_manifest(
        model=model,
        manifest_path=ROOT / "data" / "ucsd_ped2" / "manifest.json",
        cache_dir=ROOT / "cache" / "ucsd_ped2",
        output_dir=ROOT / "Evaluation" / "results_ucsd_site_calibrated",
        calibration_path=calibration_output,
        smoothing_window=5,
        device=device,
        modality="rgb",
        score_variant="norm_mse",
    )

    m = results["metrics"]
    ev = results["event_metrics"]
    fa = results["false_alarm_rate"]

    print("\n" + "#" * 80)
    print("FINAL RESULTS WITH SITE-CALIBRATION PROFILE (OPTION 1)")
    print("#" * 80)
    print(f"ROC-AUC:        {m['auc']:.4f} ({m['auc']*100:.2f}%)")
    print(f"Avg Precision:  {m['ap']:.4f} ({m['ap']*100:.2f}%)")
    print(f"Threshold:      {results['threshold']:.4f}")
    print(f"Precision:      {m['precision']:.4f} ({m['precision']*100:.2f}%)")
    print(f"Frame Recall:   {m['recall']:.4f} ({m['recall']*100:.2f}%)")
    print(f"F1 Score:       {m['f1']:.4f}")
    print(f"Event Recall:   {ev['num_detected']}/{ev['num_events']} events ({ev['event_recall']*100:.2f}%)")
    print(f"Mean Delay:     {ev['mean_delay_frames']:.1f} frames ({ev['mean_delay_frames']/10.0:.2f}s @ 10fps)")
    print(f"Frame FPR:      {fa['frame_fpr']:.4f} ({fa['frame_fpr']*100:.2f}%)")
    print(f"False Alarms:   {fa['false_alarm_frames']} normal frames flagged")
    print("#" * 80)


if __name__ == "__main__":
    main()
