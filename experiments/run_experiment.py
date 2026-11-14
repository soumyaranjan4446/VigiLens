"""
Experiment Runner for VigiLens Leave-Scene-Out Evaluation.

Executes reproducible training, calibration, and pseudo-anomaly evaluation across
scene-disjoint folds and multiple seeds, logging every run to experiments/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from models.anomaly_model import VigiLensModel
from scripts.audit_splits import audit_splits
from scripts.make_pseudo_anomalies import TIER_A_TYPES, generate_tier_a_dataset
from training.calibration import AnomalyCalibrator
from training.dataset import CachedFeatureDataset
from training.evaluate import evaluate_manifest
from training.train import train_cached
from training.utils import ensure_dir

logger = logging.getLogger(__name__)

RESULTS_COLUMNS = [
    "run_id",
    "group",
    "git_commit",
    "fold",
    "seed",
    "modality",
    "split_id",
    "config_json",
    "mask_policy",
    "score_variant",
    "calib_method",
    "target_fpr",
    "frame_auc",
    "frame_ap",
    "best_f1",
    "f1_at_thr",
    "fpr_at_95tpr",
    "normal_frame_fpr",
    "normal_video_fpr",
    "false_alarms_per_hour",
    "event_recall",
    "mean_delay_frames",
    "threshold",
    "wall_seconds",
    "notes",
    "anomaly_type",
]


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def load_completed_run_keys(results_csv: Path) -> set:
    if not results_csv.exists():
        return set()
    keys = set()
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = (row.get("run_id"), str(row.get("fold")), str(row.get("seed")), row.get("modality"), row.get("anomaly_type"))
            keys.add(k)
    return keys


def append_result_rows(results_csv: Path, rows: List[Dict[str, Any]]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in rows:
            clean_row = {col: r.get(col, "") for col in RESULTS_COLUMNS}
            writer.writerow(clean_row)


def run_experiment_fold(
    run_id: str,
    group: str,
    fold_info: Dict[str, Any],
    seed: int,
    cache_lookup: Dict[str, List[Path]],
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    select_metric: str = "val_ratio",
    input_noise: float = 0.0,
    patience: Optional[int] = None,
    freeze_fusion_epoch: Optional[int] = None,
    calib_method: str = "std",
    target_fpr: float = 0.05,
    smoothing_window: int = 5,
    modalities: Sequence[str] = ("rgb+ir", "rgb", "ir"),
    mask_policy: str = "baseline",
    score_variant: str = "mse",
    notes: str = "",
    device: str = "cpu",
    results_csv: Path = Path("experiments/results.csv"),
) -> List[Dict[str, Any]]:
    """Execute one fold for a given seed and record all modality and anomaly-type rows."""
    fold_id = fold_info["fold_id"]
    completed_keys = load_completed_run_keys(results_csv)

    # Check if overall row for this fold/seed already exists
    if all((run_id, str(fold_id), str(seed), mod, "macro") in completed_keys for mod in modalities):
        logger.info("Skipping already completed run: run_id=%s, fold=%s, seed=%d", run_id, fold_id, seed)
        return []

    t0 = time.time()
    git_commit = get_git_commit()
    config_dict = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
        "select_metric": select_metric, "input_noise": input_noise, "patience": patience,
        "freeze_fusion_epoch": freeze_fusion_epoch, "smoothing_window": smoothing_window,
    }

    fold_dir = ensure_dir(output_dir / f"fold_{fold_id}_seed_{seed}")

    # Check if this fold and seed have already been completed in results_csv
    if Path(results_csv).exists():
        try:
            existing_df = pd.read_csv(results_csv)
            if not existing_df.empty and "run_id" in existing_df.columns:
                done = existing_df[
                    (existing_df["run_id"] == run_id) &
                    (existing_df["fold"] == fold_id) &
                    (existing_df["seed"] == seed)
                ]
                if len(done) >= len(modalities):
                    logger.info(
                        "Fold %d seed %d for %s already completed (%d rows); skipping.",
                        fold_id, seed, run_id, len(done),
                    )
                    return []
        except Exception as e:
            logger.warning("Could not check existing results in %s: %s", results_csv, e)

    # Gather cache paths for this fold
    train_paths = [p for s in fold_info["train"] for p in cache_lookup.get(s, [])]
    val_paths = [p for s in fold_info["val"] for p in cache_lookup.get(s, [])]
    calib_paths = [p for s in fold_info["calib"] for p in cache_lookup.get(s, [])]
    test_paths = [p for s in fold_info["test"] for p in cache_lookup.get(s, [])]

    # Verify scene disjointness
    splits = {
        "train": [p.stem for p in train_paths],
        "val": [p.stem for p in val_paths],
        "calib": [p.stem for p in calib_paths],
        "test": [p.stem for p in test_paths],
    }
    audit_splits(splits, fail_on_overlap=True)

    if not train_paths or not val_paths:
        raise ValueError(f"Insufficient caches for fold {fold_id}: train={len(train_paths)}, val={len(val_paths)}")

    # 1. Train model
    train_ds = CachedFeatureDataset(train_paths, sequence_length=16, sequence_stride=1)
    val_ds = CachedFeatureDataset(val_paths, sequence_length=16, sequence_stride=1)

    model = VigiLensModel(device=device, load_extractors=False)
    train_res = train_cached(
        model, train_ds, val_ds, device=device,
        epochs=epochs, batch_size=batch_size, learning_rate=lr,
        weight_decay=weight_decay, amp=False,
        checkpoint_dir=fold_dir / "checkpoints",
        config_dict=config_dict,
        seed=seed,
        input_noise=input_noise,
        select_metric=select_metric,
        patience=patience,
        freeze_fusion_epoch=freeze_fusion_epoch,
    )

    # Load best checkpoint
    best_ckpt = train_res["best_checkpoint"]
    model.load_checkpoint(best_ckpt)
    model.to(device)

    # 2. Calibrate threshold on disjoint calibration scenes
    calibrator = AnomalyCalibrator(model)
    calib_caches = calib_paths if calib_paths else val_paths
    calib_errors = calibrator.collect_errors_from_caches(calib_caches, device=device)
    calib_info = calibrator.compute_threshold(calib_errors, method=calib_method)
    threshold = float(calib_info["threshold"])
    calib_json = fold_dir / "calibration.json"
    calibrator.save_calibration(calib_info, calib_json)

    # 3. Generate Tier A pseudo-anomalies on held-out test scenes
    pseudo_dir = fold_dir / "pseudo_anomalies"
    pseudo_manifest = generate_tier_a_dataset(
        cache_paths=test_paths,
        output_dir=pseudo_dir,
        types=TIER_A_TYPES,
        copies_per_clip=1,
        seed=seed,
    )

    # Also include the normal test clips in evaluation manifest
    eval_videos = list(pseudo_manifest["videos"])
    for p in test_paths:
        eval_videos.append({
            "sequence_id": p.stem,
            "cache": str(p.resolve()),
            "label": "normal",
            "anomaly_type": "normal",
        })

    eval_manifest_path = fold_dir / "eval_manifest.json"
    with open(eval_manifest_path, "w", encoding="utf-8") as f:
        json.dump({"videos": eval_videos}, f, indent=2)

    wall_seconds = round(time.time() - t0, 2)
    output_rows = []

    # 4. Evaluate across modalities
    for mod in modalities:
        eval_out = fold_dir / f"eval_{mod}"
        eval_summary = evaluate_manifest(
            model,
            manifest_path=eval_manifest_path,
            output_dir=eval_out,
            calibration_path=calib_json,
            threshold=threshold,
            smoothing_window=smoothing_window,
            device=device,
            modality=mod,
        )

        m = eval_summary["metrics"]
        ev = eval_summary.get("event_metrics", {})
        fa = eval_summary.get("false_alarm_rate", {})
        macro_auc = eval_summary.get("macro_frame_auc", m.get("auc", float("nan")))

        # Headline macro row
        row_macro = {
            "run_id": run_id,
            "group": group,
            "git_commit": git_commit,
            "fold": fold_id,
            "seed": seed,
            "modality": mod,
            "split_id": f"fold_{fold_id}",
            "config_json": json.dumps(config_dict),
            "mask_policy": mask_policy,
            "score_variant": score_variant,
            "calib_method": calib_method,
            "target_fpr": target_fpr,
            "frame_auc": macro_auc,
            "frame_ap": m.get("ap", float("nan")),
            "best_f1": m.get("best_f1", float("nan")),
            "f1_at_thr": m.get("f1", float("nan")),
            "fpr_at_95tpr": m.get("fpr_at_95tpr", float("nan")),
            "normal_frame_fpr": fa.get("frame_fpr", m.get("fpr", float("nan"))),
            "normal_video_fpr": eval_summary.get("video_metrics", {}).get("fpr", float("nan")),
            "false_alarms_per_hour": fa.get("false_alarms_per_hour", float("nan")),
            "event_recall": ev.get("event_recall", float("nan")),
            "mean_delay_frames": ev.get("mean_delay_frames", float("nan")),
            "threshold": threshold,
            "wall_seconds": wall_seconds,
            "notes": notes,
            "anomaly_type": "macro",
        }
        output_rows.append(row_macro)

        # Per anomaly-type rows
        by_type = eval_summary.get("by_anomaly_type", {})
        for atype, tm in by_type.items():
            row_type = dict(row_macro)
            row_type["anomaly_type"] = atype
            row_type["frame_auc"] = tm.get("auc", float("nan"))
            row_type["frame_ap"] = tm.get("ap", float("nan"))
            row_type["best_f1"] = tm.get("best_f1", float("nan"))
            row_type["f1_at_thr"] = tm.get("f1", float("nan"))
            row_type["fpr_at_95tpr"] = tm.get("fpr_at_95tpr", float("nan"))
            output_rows.append(row_type)

    append_result_rows(results_csv, output_rows)
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-scene-out VigiLens experiment across folds and seeds")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", default="experiments")
    parser.add_argument("--folds-json", default="experiments/splits/folds.json")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="experiments/runs")
    parser.add_argument("--folds", default=None, help="Comma-separated fold indices (default: all folds)")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    parser.add_argument("--modalities", default="rgb+ir,rgb,ir", help="Comma-separated modalities")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--select-metric", choices=["val_ratio", "val_loss"], default="val_ratio")
    parser.add_argument("--input-noise", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--freeze-fusion-epoch", type=int, default=None)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--notes", default="")
    parser.add_argument("--results-csv", default="experiments/results.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.folds_json, "r", encoding="utf-8") as f:
        folds_data = json.load(f)

    all_folds = folds_data["folds"]
    if args.folds:
        selected_fold_ids = set(int(x.strip()) for x in args.folds.split(","))
        all_folds = [f for f in all_folds if f["fold_id"] in selected_fold_ids]

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    modalities = [m.strip() for m in args.modalities.split(",")]

    # Map scenes to discovered caches
    cache_lookup: Dict[str, List[Path]] = {}
    for p in Path(args.cache_dir).rglob("*.pt"):
        sc = p.stem.split("_")[0]
        cache_lookup.setdefault(sc, []).append(p)

    out_base = Path(args.output_dir) / args.run_id
    total_runs = len(all_folds) * len(seeds)
    logger.info("Starting experiment %s: %d folds x %d seeds = %d runs", args.run_id, len(all_folds), len(seeds), total_runs)

    for fold in all_folds:
        for seed in seeds:
            run_experiment_fold(
                run_id=args.run_id,
                group=args.group,
                fold_info=fold,
                seed=seed,
                cache_lookup=cache_lookup,
                output_dir=out_base,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                select_metric=args.select_metric,
                input_noise=args.input_noise,
                patience=args.patience,
                freeze_fusion_epoch=args.freeze_fusion_epoch,
                smoothing_window=args.smoothing_window,
                modalities=modalities,
                notes=args.notes,
                results_csv=Path(args.results_csv),
            )

    print(f"\n[Experiment {args.run_id} Complete] Results recorded in {args.results_csv}")


if __name__ == "__main__":
    main()
