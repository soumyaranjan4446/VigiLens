import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from models.anomaly_model import VigiLensModel
from training.scoring import fuse_features, score_fused_sequence
from training.metrics import compute_event_metrics

# 1. Load Model
device = "cpu"
model = VigiLensModel(device=device, load_extractors=False)
model.load_checkpoint(str(ROOT / "checkpoints" / "best_model.pth"))
model.to(device)
model.eval()

# 2. Load Manifest
with open(ROOT / "data" / "ucsd_ped2" / "manifest.json", "r", encoding="utf-8") as f:
    mani = json.load(f)

segments = [entry["segments"] for entry in mani["videos"]]
num_events = sum(len(s) for s in segments)

# 3. Calibration Clips for finding site stats & threshold
calib_clips = ["Train001", "Train002", "Train003", "Train004"]
train_img_dir = ROOT / "cache" / "ucsd_ped2_train"
train_reid_dir = ROOT / "cache" / "ucsd_ped2_train_reid"

calib_s_img = []
calib_s_reid = []

for c_name in calib_clips:
    # ImageNet
    d_img = torch.load(train_img_dir / f"{c_name}.pt", map_location="cpu", weights_only=False)
    rgb_img = d_img["rgb_features"].float()
    with torch.inference_mode():
        fused = fuse_features(model, rgb_img, None, rgb_available=True, ir_available=False)
        _, s_img = score_fused_sequence(model, fused, smoothing_window=7, score_variant="mse")
    calib_s_img.append(s_img)
    
    # ReID Unit-Norm
    d_reid = torch.load(train_reid_dir / f"{c_name}.pt", map_location="cpu", weights_only=False)
    rgb_reid = F.normalize(d_reid["rgb_features"].float(), p=2, dim=-1)
    with torch.inference_mode():
        fused = fuse_features(model, rgb_reid, None, rgb_available=True, ir_available=False)
        _, s_reid = score_fused_sequence(model, fused, smoothing_window=7, score_variant="mse")
    calib_s_reid.append(s_reid)

calib_img_all = np.concatenate(calib_s_img)
calib_reid_all = np.concatenate(calib_s_reid)

mu_img, std_img = float(np.mean(calib_img_all)), float(np.std(calib_img_all))
mu_reid, std_reid = float(np.mean(calib_reid_all)), float(np.std(calib_reid_all))

print(f"Calibration stats (ImageNet): mu={mu_img:.4f}, std={std_img:.4f}")
print(f"Calibration stats (ReID-Unit): mu={mu_reid:.4f}, std={std_reid:.4f}")

# 4. Score Test Clips
test_seq_img = []
test_seq_reid = []
labels_by_seq = []

for entry in mani["videos"]:
    sid = entry["sequence_id"]
    # ImageNet
    d_img = torch.load(ROOT / "cache" / "ucsd_ped2" / f"{sid}.pt", map_location="cpu", weights_only=False)
    rgb_img = d_img["rgb_features"].float()
    with torch.inference_mode():
        fused = fuse_features(model, rgb_img, None, rgb_available=True, ir_available=False)
        _, s_img = score_fused_sequence(model, fused, smoothing_window=7, score_variant="mse")
    test_seq_img.append(s_img)
    
    # ReID Unit-Norm
    d_reid = torch.load(ROOT / "cache" / "ucsd_ped2_reid" / f"{sid}.pt", map_location="cpu", weights_only=False)
    rgb_reid = F.normalize(d_reid["rgb_features"].float(), p=2, dim=-1)
    with torch.inference_mode():
        fused = fuse_features(model, rgb_reid, None, rgb_available=True, ir_available=False)
        _, s_reid = score_fused_sequence(model, fused, smoothing_window=7, score_variant="mse")
    test_seq_reid.append(s_reid)
    
    lbl = np.zeros(len(rgb_img), dtype=int)
    for seg in entry["segments"]:
        lbl[seg[0]:seg[1]] = 1
    labels_by_seq.append(lbl)

y_true = np.concatenate(labels_by_seq)

def evaluate_system(alpha=0.6, p_high=95.0, p_low=80.0, H=15):
    # Standardize calib
    calib_z_img = (calib_img_all - mu_img) / std_img
    calib_z_reid = (calib_reid_all - mu_reid) / std_reid
    calib_ens = (1.0 - alpha) * calib_z_img + alpha * calib_z_reid
    
    th_high = float(np.percentile(calib_ens, p_high))
    th_low = float(np.percentile(calib_ens, p_low))
    
    # Standardize test and compute hysteresis
    test_ens_seqs = []
    test_preds_seqs = []
    
    for s_i, s_r in zip(test_seq_img, test_seq_reid):
        z_i = (s_i - mu_img) / std_img
        z_r = (s_r - mu_reid) / std_reid
        ens = (1.0 - alpha) * z_i + alpha * z_r
        test_ens_seqs.append(ens)
        
        # Hysteresis
        pred = np.zeros_like(ens, dtype=int)
        hold = 0
        for t in range(len(ens)):
            if ens[t] >= th_high:
                hold = H
                pred[t] = 1
            elif hold > 0 and ens[t] >= th_low:
                hold -= 1
                pred[t] = 1
            else:
                hold = 0
        test_preds_seqs.append(pred)
        
    y_scores = np.concatenate(test_ens_seqs)
    y_preds = np.concatenate(test_preds_seqs)
    
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    tp = int(np.sum((y_preds == 1) & (y_true == 1)))
    fp = int(np.sum((y_preds == 1) & (y_true == 0)))
    tn = int(np.sum((y_preds == 0) & (y_true == 0)))
    fn = int(np.sum((y_preds == 0) & (y_true == 1)))
    
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    
    # Event metrics
    ev = compute_event_metrics(segments, test_preds_seqs, threshold=0.5)
    
    return {
        "alpha": alpha,
        "th_high": th_high,
        "th_low": th_low,
        "H": H,
        "auc": auc,
        "ap": ap,
        "precision": prec,
        "recall": rec,
        "fpr": fpr,
        "f1": f1,
        "event_recall": ev["event_recall"],
        "caught_events": ev["num_detected"],
        "total_events": ev["num_events"],
        "mean_delay_sec": ev["mean_delay_frames"] / 10.0 if not np.isnan(ev["mean_delay_frames"]) else 0.0,
        "mean_delay_frames": ev["mean_delay_frames"] if not np.isnan(ev["mean_delay_frames"]) else 0.0,
    }

print("\n" + "=" * 90)
print("GRID SEARCH ACROSS HYPERPARAMETERS (ALPHA, HYSTERESIS WINDOW, PERCENTILE)")
print("=" * 90)

configs = [
    (0.5, 95.0, 80.0, 10),
    (0.5, 95.0, 80.0, 15),
    (0.6, 95.0, 80.0, 10),
    (0.6, 95.0, 80.0, 15),
    (0.6, 95.0, 85.0, 15),
    (0.6, 96.0, 85.0, 15),
    (0.7, 95.0, 80.0, 10),
    (0.7, 95.0, 80.0, 15),
    (0.7, 96.0, 85.0, 15),
]

for alpha, ph, pl, H in configs:
    r = evaluate_system(alpha=alpha, p_high=ph, p_low=pl, H=H)
    print(f"a={r['alpha']:.1f} H={r['H']:2d} p=({ph:.0f},{pl:.0f}) | AUC: {r['auc']*100:.1f}% | Prec: {r['precision']*100:.1f}% | Rec: {r['recall']*100:.1f}% | FPR: {r['fpr']*100:.2f}% | F1: {r['f1']:.4f} | EvRec: {r['event_recall']*100:.1f}% ({r['caught_events']}/{r['total_events']}) | Delay: {r['mean_delay_sec']:.2f}s")
