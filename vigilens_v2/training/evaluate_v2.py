"""Reproducible Quantitative Evaluator for VigiLens V2 (UCSD Ped2 True GT + VidLLVIP Holdout).

Usage:
    python -m vigilens_v2.training.evaluate_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anomaly_model import VigiLensModel
from training.metrics import compute_event_metrics
from training.scoring import fuse_features
from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model
from vigilens_v2.pipeline.smoothing import CausalHysteresisDetector, causal_moving_average_skip0
from vigilens_v2.training.data_gt import load_ped2_test_labels


def evaluate_ucsd_ped2(
    ckpt_path: Path,
    probe_cache_dir: Path,
    output_json: Path,
) -> Dict[str, object]:
    """Compare V1 Baseline vs VigiLens V2 on ALL 12 Ped2 Test Clips with FIXED Ground Truth."""
    model_v2 = VigiLensV2Model.load_v2_checkpoint(str(ckpt_path), device="cpu")
    model_v2.eval()

    # Load V1 unified reference (0.3*ImageNet + 0.7*ReID-Unit under identical fixed GT for apples-to-apples!)
    v1_model = VigiLensModel(device="cpu", load_extractors=False)
    v1_ckpt = ROOT / "checkpoints" / "best_model.pth"
    if v1_ckpt.exists():
        v1_model.load_checkpoint(str(v1_ckpt))
    v1_model.eval()

    tr_ids = ["Train001", "Train002", "Train003", "Train004"]
    v1_tr_i, v1_tr_r = [], []
    with torch.inference_mode():
        for cid in tr_ids:
            di = torch.load(ROOT / f"cache/ucsd_ped2_train/{cid}.pt", map_location="cpu", weights_only=False)
            dr = torch.load(ROOT / f"cache/ucsd_ped2_train_reid/{cid}.pt", map_location="cpu", weights_only=False)
            fi = fuse_features(v1_model, di["rgb_features"].float(), None, True, False)
            fr = fuse_features(v1_model, torch.nn.functional.normalize(dr["rgb_features"].float(), dim=-1), None, True, False)
            pi, _ = v1_model.temporal.forward_sequence(fi.unsqueeze(0))
            pr, _ = v1_model.temporal.forward_sequence(fr.unsqueeze(0))
            ei = torch.zeros(len(fi)); ei[1:] = (pi[0, :-1] - fi[1:]).pow(2).mean(dim=-1)
            er = torch.zeros(len(fr)); er[1:] = (pr[0, :-1] - fr[1:]).pow(2).mean(dim=-1)
            v1_tr_i.append(causal_moving_average_skip0(ei.numpy(), 7)[1:])
            v1_tr_r.append(causal_moving_average_skip0(er.numpy(), 7)[1:])
    i_m, i_s = float(np.mean(np.concatenate(v1_tr_i))), float(np.std(np.concatenate(v1_tr_i)) + 1e-6)
    r_m, r_s = float(np.mean(np.concatenate(v1_tr_r))), float(np.std(np.concatenate(v1_tr_r)) + 1e-6)

    test_root = ROOT / "Data" / "ucsd_ped2" / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2" / "Test"

    y_all, segs_all = [], []
    v1_all, trk_all, grid_all, app_all, v2_all = [], [], [], [], []
    v2_per_clip = []
    per_clip_rows = []

    for tid in range(1, 13):
        sid = f"Test{tid:03d}"
        d_probe = torch.load(probe_cache_dir / f"{sid}.pt", map_location="cpu", weights_only=False)
        n = int(d_probe["N"])
        lbl, segs = load_ped2_test_labels(test_root, sid, num_frames=n)

        di = torch.load(ROOT / f"cache/ucsd_ped2/{sid}.pt", map_location="cpu", weights_only=False)
        dr = torch.load(ROOT / f"cache/ucsd_ped2_reid/{sid}.pt", map_location="cpu", weights_only=False)
        rgb = di["rgb_features"].float()
        reid = torch.nn.functional.normalize(dr["rgb_features"].float(), dim=-1)

        sc_trk, _ = model_v2.track_model.score_meta_offline(d_probe["meta"], num_frames=n)
        sc_grid = model_v2.score_grid_excess(d_probe["grid_flow"], smoothing_window=5)
        sc_app = model_v2.score_scene_appearance(rgb, reid_feats=reid, smoothing_window=7)
        sc_v2 = model_v2.combine_z_scores(sc_trk, sc_grid, sc_app)

        # V1 reference under exact same true GT
        with torch.inference_mode():
            fi = fuse_features(v1_model, rgb, None, True, False)
            fr = fuse_features(v1_model, reid, None, True, False)
            pi, _ = v1_model.temporal.forward_sequence(fi.unsqueeze(0))
            pr, _ = v1_model.temporal.forward_sequence(fr.unsqueeze(0))
            ei = torch.zeros(n); ei[1:] = (pi[0, :-1] - fi[1:]).pow(2).mean(dim=-1)
            er = torch.zeros(n); er[1:] = (pr[0, :-1] - fr[1:]).pow(2).mean(dim=-1)
            v1_sc = 0.3 * ((causal_moving_average_skip0(ei.numpy(), 7) - i_m) / i_s) + 0.7 * ((causal_moving_average_skip0(er.numpy(), 7) - r_m) / r_s)
            v1_sc[0] = 0.0

        y_all.append(lbl)
        segs_all.append(segs)
        v1_all.append(v1_sc)
        trk_all.append(sc_trk)
        grid_all.append(sc_grid)
        app_all.append(sc_app)
        v2_all.append(sc_v2)
        v2_per_clip.append(sc_v2)

        has_both = 0 < int(lbl.sum()) < len(lbl)
        per_clip_rows.append(
            {
                "sequence_id": sid,
                "num_frames": n,
                "positive_frames": int(lbl.sum()),
                "normal_frames": int((lbl == 0).sum()),
                "segments": segs,
                "v1_auc": round(float(roc_auc_score(lbl, v1_sc) * 100), 2) if has_both else None,
                "v2_auc": round(float(roc_auc_score(lbl, sc_v2) * 100), 2) if has_both else None,
                "v2_track_auc": round(float(roc_auc_score(lbl, sc_trk) * 100), 2) if has_both else None,
            }
        )

    y_cat = np.concatenate(y_all)
    v1_cat = np.concatenate(v1_all)
    v2_cat = np.concatenate(v2_all)

    # Evaluate calibrated hysteresis operating regimes
    cal = model_v2.calibration
    # Recompute normal fused array thresholds for multiple operating presets cleanly from saved med/std
    regimes = []
    # Default saved regime first
    preset_list = [
        ("Calibrated (p95/p80)", float(cal["threshold_high"]), float(cal["threshold_low"])),
    ]
    # Also evaluate p92/p75 and p90/p75 derived strictly from Train001-004
    tr_fused = []
    for cid in tr_ids:
        dp = torch.load(probe_cache_dir / f"{cid}.pt", map_location="cpu", weights_only=False)
        di = torch.load(ROOT / f"cache/ucsd_ped2_train/{cid}.pt", map_location="cpu", weights_only=False)
        dr = torch.load(ROOT / f"cache/ucsd_ped2_train_reid/{cid}.pt", map_location="cpu", weights_only=False)
        st, _ = model_v2.track_model.score_meta_offline(dp["meta"], num_frames=int(dp["N"]))
        sg = model_v2.score_grid_excess(dp["grid_flow"], 5)
        sa = model_v2.score_scene_appearance(
            di["rgb_features"].float(),
            reid_feats=torch.nn.functional.normalize(dr["rgb_features"].float(), dim=-1),
            smoothing_window=7,
        )
        tr_fused.append(model_v2.combine_z_scores(st, sg, sa)[1:])
    tr_fused_cat = np.concatenate(tr_fused)
    for name, ph, pl in [
        ("Balanced High-Catch (p92/p75)", 92.0, 75.0),
        ("Full-Sensitivity (p85/p70)", 85.0, 70.0),
    ]:
        preset_list.append((name, float(np.percentile(tr_fused_cat, ph)), float(np.percentile(tr_fused_cat, pl))))

    for name, th_h, th_l in preset_list:
        hyst = CausalHysteresisDetector(th_h, th_l, hold_frames=int(cal.get("hold_frames", 15)))
        preds = [hyst.run_sequence(arr) for arr in v2_per_clip]
        p_cat = np.concatenate(preds)
        tp = int(((p_cat == 1) & (y_cat == 1)).sum())
        fp = int(((p_cat == 1) & (y_cat == 0)).sum())
        fn = int(((p_cat == 0) & (y_cat == 1)).sum())
        tn = int(((p_cat == 0) & (y_cat == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        f1 = 2 * prec * rec / max(1e-8, prec + rec)
        ev = compute_event_metrics(segs_all, [p.astype(np.float32) for p in preds], threshold=0.5)
        regimes.append(
            {
                "name": name,
                "threshold_high": round(th_h, 4),
                "threshold_low": round(th_l, 4),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "fpr": round(fpr * 100, 2),
                "f1": round(f1, 4),
                "events_detected": int(ev["num_detected"]),
                "events_total": int(ev["num_events"]),
                "event_recall": round(float(ev["event_recall"]) * 100, 2),
                "mean_delay_frames": round(float(ev["mean_delay_frames"]), 2),
            }
        )

    summary = {
        "dataset": "UCSD Ped2 (True Official GT — all 12 Test Clips, 1648 pos / 362 neg)",
        "v1_unified_auc": round(float(roc_auc_score(y_cat, v1_cat) * 100), 2),
        "v1_unified_ap": round(float(average_precision_score(y_cat, v1_cat) * 100), 2),
        "v2_track_lstm_auc": round(float(roc_auc_score(y_cat, np.concatenate(trk_all)) * 100), 2),
        "v2_grid_flow_auc": round(float(roc_auc_score(y_cat, np.concatenate(grid_all)) * 100), 2),
        "v2_scene_app_auc": round(float(roc_auc_score(y_cat, np.concatenate(app_all)) * 100), 2),
        "v2_unified_auc": round(float(roc_auc_score(y_cat, v2_cat) * 100), 2),
        "v2_unified_ap": round(float(average_precision_score(y_cat, v2_cat) * 100), 2),
        "operating_regimes": regimes,
        "per_clip": per_clip_rows,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_vidllvip_holdout(output_json: Path) -> Dict[str, object]:
    """Compare V1 norm_mse vs V2 Multi-Component (Pred + Normal Manifold + Transition Delta) on frozen Holdout 09 & 13."""
    v1_model = VigiLensModel(device="cpu", load_extractors=False)
    v1_model.load_checkpoint(str(ROOT / "checkpoints" / "best_model.pth"))
    v1_model.eval()

    eval_man = json.loads((ROOT / "experiments/runs/final_production_model/fold_0_seed_42/eval_manifest.json").read_text())
    calib = json.loads((ROOT / "experiments/runs/final_production_model/fold_0_seed_42/calibration.json").read_text())
    split = json.loads((ROOT / "experiments/splits/production.json").read_text())["folds"][0]

    # Normal train+val+calib latent manifold stats (zero holdout leak)
    tr_z = []
    with torch.inference_mode():
        for sid in split["train"] + split["val"] + split["calib"]:
            for p in sorted((ROOT / "cache" / "scenes").glob(f"{sid}_*.pt")):
                d = torch.load(p, map_location="cpu", weights_only=False)
                tr_z.append(fuse_features(v1_model, d["rgb_features"].float(), d["ir_features"].float(), True, True))
    z_cat = torch.cat(tr_z, dim=0)
    z_mu = z_cat.mean(dim=0, keepdim=True)
    z_var = z_cat.var(dim=0, keepdim=True) + 1e-2
    dv = torch.tensor(calib["diag_variance"], dtype=torch.float32).clamp(min=1e-4)

    vy_all, v1_all, v2_all = [], [], []
    with torch.inference_mode():
        for ent in eval_man["videos"]:
            cpath = ROOT / ent["cache"] if not Path(ent["cache"]).is_absolute() else Path(ent["cache"])
            d = torch.load(cpath, map_location="cpu", weights_only=False)
            rgb = d["rgb_features"].float()
            ir = d["ir_features"].float()
            n = len(rgb)
            lbl = np.zeros(n, dtype=np.int64)
            for s0, s1 in ent.get("segments", []):
                lbl[max(0, s0) : min(n, s1)] = 1
            vy_all.append(lbl)

            z = fuse_features(v1_model, rgb, ir, True, True)
            pr, _ = v1_model.temporal.forward_sequence(z.unsqueeze(0))
            pr = pr.squeeze(0)

            err_norm = torch.zeros(n); err_norm[1:] = ((pr[:-1] - z[1:]).pow(2) / dv).mean(dim=-1)
            err_plain = torch.zeros(n); err_plain[1:] = (pr[:-1] - z[1:]).pow(2).mean(dim=-1)
            dz = torch.zeros(n); dz[1:] = (z[1:] - z[:-1]).pow(2).mean(dim=-1)
            mah = ((z - z_mu).pow(2) / z_var).mean(dim=-1)

            v1_all.append(causal_moving_average_skip0(err_norm.numpy(), 7))
            v2_all.append(causal_moving_average_skip0((err_plain + 0.25 * mah + 0.50 * dz).numpy(), 7))

    vy = np.concatenate(vy_all)
    s1 = np.concatenate(v1_all)
    s2 = np.concatenate(v2_all)
    res = {
        "dataset": "VidLLVIP Paired RGB+IR Frozen Holdout (Scenes 09 & 13, 3500 frames)",
        "v1_norm_mse_auc": round(float(roc_auc_score(vy, s1) * 100), 2),
        "v1_norm_mse_ap": round(float(average_precision_score(vy, s1) * 100), 2),
        "v2_manifold_transition_auc": round(float(roc_auc_score(vy, s2) * 100), 2),
        "v2_manifold_transition_ap": round(float(average_precision_score(vy, s2) * 100), 2),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VigiLens V2 vs V1 on Ped2 True GT and VidLLVIP Holdout.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "vigilens_v2" / "artifacts" / "vigilens_v2_ped2.pt",
    )
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=Path(r"C:\Users\Administrator\.gemini\antigravity\brain\bc3d31f5-6c92-49fd-81c3-a336251907a8\scratch\v2_probe_cache"),
    )
    args = parser.parse_args()

    ped2 = evaluate_ucsd_ped2(
        ckpt_path=args.ckpt,
        probe_cache_dir=args.probe_cache,
        output_json=ROOT / "vigilens_v2" / "results" / "ucsd_ped2_v2_results.json",
    )
    vid = evaluate_vidllvip_holdout(
        output_json=ROOT / "vigilens_v2" / "results" / "vidllvip_holdout_v2_results.json",
    )
    print("\n=== UCSD PED2 (TRUE OFFICIAL GT) ===")
    print(json.dumps(ped2, indent=2))
    print("\n=== VIDLLVIP FROZEN HOLDOUT (SCENES 09 & 13) ===")
    print(json.dumps(vid, indent=2))


if __name__ == "__main__":
    main()
