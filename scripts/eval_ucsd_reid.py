"""
Evaluate VigiLens with Pedestrian Re-ID (Market-1501) Backbone Weights.

Extracts specialized pedestrian posture/gait features (2048-D) across the UCSD Ped2
test sequences and normal calibration sequences without altering the network architecture.
Computes site calibration and evaluates test set performance.
"""

import sys
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from tqdm import tqdm

from models.anomaly_model import VigiLensModel
from training.precompute import precompute_manifest, precompute_pair, build_extractors
from models.segmentation import YOLOSegmentation
from training.calibration import AnomalyCalibrator
from training.evaluate import evaluate_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    device = "cpu"
    reid_weights = ROOT / "checkpoints" / "resnet50_market1501.pt"
    test_manifest = ROOT / "data" / "ucsd_ped2" / "manifest.json"
    test_cache_dir = ROOT / "cache" / "ucsd_ped2_reid"
    train_videos_dir = ROOT / "data" / "ucsd_ped2" / "train_videos"
    train_cache_dir = ROOT / "cache" / "ucsd_ped2_train_reid"
    calib_json = ROOT / "checkpoints" / "calibration_ucsd_reid.json"
    results_dir = ROOT / "Evaluation" / "results_ucsd_reid"

    test_cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: Extracting Pedestrian Re-ID Features on UCSD Ped2 Test Sequences...")
    print("=" * 80)

    seg = YOLOSegmentation(device=device)
    rgb_ext, ir_ext = build_extractors(device=device, weights_path=str(reid_weights))

    # Precompute test clips
    test_caches = precompute_manifest(
        manifest_path=test_manifest,
        output_dir=test_cache_dir,
        segmentation=seg,
        rgb_extractor=rgb_ext,
        ir_extractor=ir_ext,
        batch_size=16,
        overwrite=False,
    )
    print(f"Test feature precomputation complete: {len(test_caches)} clips.")

    print("\n" + "=" * 80)
    print("STEP 2: Extracting Pedestrian Re-ID Features on Normal Calibration Clips...")
    print("=" * 80)

    calib_clips = ["Train001", "Train002", "Train003", "Train004"]
    calib_caches = []
    for c_name in calib_clips:
        v_path = train_videos_dir / f"{c_name}_rgb.mp4"
        pt_path = train_cache_dir / f"{c_name}.pt"
        if pt_path.exists() and pt_path.stat().st_size > 1000:
            calib_caches.append(pt_path)
            continue
        c_p = precompute_pair(
            rgb_path=v_path,
            ir_path=None,
            output_dir=train_cache_dir,
            segmentation=seg,
            rgb_extractor=rgb_ext,
            ir_extractor=ir_ext,
            batch_size=16,
            sequence_id=c_name,
        )
        if c_p is not None:
            calib_caches.append(c_p)
    print(f"Calibration feature precomputation complete: {len(calib_caches)} clips.")

    print("\n" + "=" * 80)
    print("STEP 3: Computing Site Calibration with Re-ID Residual Variance...")
    print("=" * 80)

    model = VigiLensModel(device=device, load_extractors=False)
    prod_ckpt = ROOT / "checkpoints" / "best_model.pth"
    model.load_checkpoint(str(prod_ckpt))
    model.to(device)

    calibrator = AnomalyCalibrator(model, k_std=3.0, percentile=95.0)
    diag_var = calibrator.compute_residual_variance(calib_caches, device=device)
    errors = calibrator.collect_errors_from_caches(
        calib_caches,
        device=device,
        score_variant="norm_mse",
        diag_variance=diag_var,
    )
    info = calibrator.compute_threshold(errors, method="percentile")
    info["score_variant"] = "norm_mse"
    info["diag_variance"] = diag_var.tolist()
    info["site_name"] = "ucsd_ped2_reid"

    with open(calib_json, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"Re-ID site calibration saved to {calib_json} (p95 threshold = {info['threshold']:.4f})")

    print("\n" + "=" * 80)
    print("STEP 4: Evaluating Model on Test Sequences with Pedestrian Re-ID Features...")
    print("=" * 80)

    results = evaluate_manifest(
        model=model,
        manifest_path=test_manifest,
        cache_dir=test_cache_dir,
        output_dir=results_dir,
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
    print("PEDESTRIAN RE-ID BACKBONE (STRATEGY 2) FINAL RESULTS")
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
