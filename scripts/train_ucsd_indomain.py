"""
In-Domain Training of VigiLens on UCSD Ped2 Normal Walkway Clips.

Adapts the temporal predictor (Causal LSTM + NextFeaturePredictor) to the specific
camera perspective, crowd density, and walking speeds of the UCSD Ped2 walkway.

Strictly preserves the normal-only training invariant:
- Uses ONLY the 16 normal training clips (Train001-Train016).
- Zero anomaly frames or test clips used for training or validation.
- Evaluates on the 12 test sequences (Test001-Test012).
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import torch
import numpy as np
from tqdm import tqdm

from config.config import get_config
from models.anomaly_model import VigiLensModel
from training.dataset import CachedFeatureDataset
from training.train import train_cached
from training.calibration import AnomalyCalibrator
from training.evaluate import evaluate_manifest
from training.precompute import precompute_pair, build_extractors
from models.segmentation import YOLOSegmentation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert_all_train_clips(train_dir: Path, output_videos_dir: Path):
    """Convert any un-converted Train image folders to MP4 videos."""
    output_videos_dir.mkdir(parents=True, exist_ok=True)
    all_folders = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("Train")])

    video_paths = []
    print(f"Checking {len(all_folders)} normal training clips...")
    for folder in all_folders:
        video_p = output_videos_dir / f"{folder.name}_rgb.mp4"
        if video_p.exists() and video_p.stat().st_size > 1000:
            video_paths.append(video_p)
            continue

        img_files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        if not img_files:
            continue
        first_img = cv2.imread(str(img_files[0]))
        h, w = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_p), fourcc, 10.0, (w, h))
        for img_p in img_files:
            img = cv2.imread(str(img_p))
            writer.write(img)
        writer.release()
        video_paths.append(video_p)
        print(f"  Converted {folder.name} ({len(img_files)} frames) -> {video_p.name}")
    return video_paths


def precompute_all_train_clips(video_paths, output_cache_dir: Path, device="cpu"):
    """Precompute features for all 16 training clips."""
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    seg = YOLOSegmentation(device=device)
    rgb_ext, ir_ext = build_extractors(device=device)

    cache_paths = []
    for v_p in tqdm(video_paths, desc="Precomputing normal clips"):
        seq_id = v_p.stem.replace("_rgb", "")
        pt_path = output_cache_dir / f"{seq_id}.pt"
        if pt_path.exists() and pt_path.stat().st_size > 1000:
            cache_paths.append(pt_path)
            continue

        c_p = precompute_pair(
            rgb_path=v_p,
            ir_path=None,
            output_dir=output_cache_dir,
            segmentation=seg,
            rgb_extractor=rgb_ext,
            ir_extractor=ir_ext,
            batch_size=16,
            sequence_id=seq_id,
        )
        if c_p is not None:
            cache_paths.append(c_p)
    return cache_paths


def main():
    device = "cpu"
    ped2_train_dir = ROOT / "data" / "ucsd_ped2" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Train"
    train_videos_dir = ROOT / "data" / "ucsd_ped2" / "train_videos"
    train_cache_dir = ROOT / "cache" / "ucsd_ped2_train"
    ckpt_dir = ROOT / "checkpoints" / "indomain_ucsd"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ensure all 16 clips are converted to MP4
    v_paths = convert_all_train_clips(ped2_train_dir, train_videos_dir)

    # Step 2: Ensure all 16 clips are precomputed
    c_paths = precompute_all_train_clips(v_paths, train_cache_dir, device=device)
    print(f"Total cached training clips: {len(c_paths)}")

    # Step 3: Split into train (Train001-Train012) and val (Train013-Train016)
    train_caches = [p for p in c_paths if int(p.stem.replace("Train", "")) <= 12]
    val_caches = [p for p in c_paths if int(p.stem.replace("Train", "")) > 12]
    print(f"Train split: {len(train_caches)} clips (~1,800 frames)")
    print(f"Val split:   {len(val_caches)} clips (~600 frames)")

    train_dataset = CachedFeatureDataset(train_caches, sequence_length=16, sequence_stride=1, rgb_dropout_prob=0.0)
    val_dataset = CachedFeatureDataset(val_caches, sequence_length=16, sequence_stride=1, rgb_dropout_prob=0.0)

    # Step 4: Initialize model from production checkpoint
    model = VigiLensModel(device=device, load_extractors=False)
    prod_ckpt = ROOT / "checkpoints" / "best_model.pth"
    if prod_ckpt.exists():
        model.load_checkpoint(str(prod_ckpt))
        print(f"Initialized from production weights: {prod_ckpt}")
    model.to(device)

    # Step 5: Train temporal model on normal sidewalk dynamics
    print("\nStarting In-Domain Training (10 Epochs on Normal Walkway Feeds)...")
    config = get_config()
    train_res = train_cached(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=10,
        batch_size=8,
        learning_rate=5e-5,
        weight_decay=1e-4,
        grad_clip=1.0,
        checkpoint_dir=str(ckpt_dir),
        config_dict=config.to_dict(),
        freeze_fusion_epoch=3,
        select_metric="val_loss",
    )
    print(f"Training complete! Best checkpoint: {train_res.get('best_checkpoint')}")

    # Load best checkpoint
    best_ckpt = train_res.get("best_checkpoint")
    if best_ckpt and Path(best_ckpt).exists():
        model.load_checkpoint(best_ckpt)
        model.to(device)

    # Step 6: Site Calibration with Diagonal Mahalanobis Variance
    print("\nCalibrating In-Domain Model on Held-out Normal Validation Clips...")
    calibrator = AnomalyCalibrator(model, k_std=3.0, percentile=95.0)
    diag_var = calibrator.compute_residual_variance(val_caches, device=device)
    errors = calibrator.collect_errors_from_caches(
        val_caches,
        device=device,
        score_variant="norm_mse",
        diag_variance=diag_var,
    )
    info = calibrator.compute_threshold(errors, method="percentile")
    info["score_variant"] = "norm_mse"
    info["diag_variance"] = diag_var.tolist()
    info["site_name"] = "ucsd_ped2_indomain"

    calib_json = ckpt_dir / "calibration.json"
    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Calibration saved to {calib_json} (p95 threshold = {info['threshold']:.4f})")

    # Step 7: Evaluate on the 12 Test Sequences
    print("\n" + "=" * 80)
    print("Evaluating In-Domain Model on UCSD Ped2 Test Sequences...")
    print("=" * 80)

    eval_out = ROOT / "Evaluation" / "results_ucsd_indomain"
    results = evaluate_manifest(
        model=model,
        manifest_path=ROOT / "data" / "ucsd_ped2" / "manifest.json",
        cache_dir=ROOT / "cache" / "ucsd_ped2",
        output_dir=eval_out,
        calibration_path=calib_json,
        smoothing_window=5,
        device=device,
        modality="rgb",
        score_variant="norm_mse",
    )

    m = results["metrics"]
    ev = results["event_metrics"]
    fa = results["false_alarm_rate"]

    print("\n" + "#" * 80)
    print("IN-DOMAIN TRAINED MODEL FINAL RESULTS (TIER 1 - IMPROVEMENT #1)")
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
