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

device = "cpu"
model = VigiLensModel(device=device, load_extractors=False)
model.load_checkpoint(str(ROOT / "checkpoints" / "best_model.pth"))
model.to(device)
model.eval()

with open(ROOT / "data" / "ucsd_ped2" / "manifest.json", "r", encoding="utf-8") as f:
    mani = json.load(f)

# Calib stats
mu_img, std_img = 0.0896, 0.0080
mu_reid, std_reid = 0.1080, 0.0093

# Calculate threshold from Train001-004
calib_clips = ["Train001", "Train002", "Train003", "Train004"]
calib_ens = []
for c_name in calib_clips:
    d_i = torch.load(ROOT / "cache" / "ucsd_ped2_train" / f"{c_name}.pt", map_location="cpu", weights_only=False)
    d_r = torch.load(ROOT / "cache" / "ucsd_ped2_train_reid" / f"{c_name}.pt", map_location="cpu", weights_only=False)
    with torch.inference_mode():
        f_i = fuse_features(model, d_i["rgb_features"].float(), None, rgb_available=True, ir_available=False)
        _, s_i = score_fused_sequence(model, f_i, smoothing_window=7)
        f_r = fuse_features(model, F.normalize(d_r["rgb_features"].float(), p=2, dim=-1), None, rgb_available=True, ir_available=False)
        _, s_r = score_fused_sequence(model, f_r, smoothing_window=7)
    z_i = (s_i - mu_img) / std_img
    z_r = (s_r - mu_reid) / std_reid
    ens = 0.3 * z_i + 0.7 * z_r
    calib_ens.append(ens)
calib_ens = np.concatenate(calib_ens)
th_high = float(np.percentile(calib_ens, 95.0))
th_low = float(np.percentile(calib_ens, 80.0))

print(f"Ensemble Thresholds: th_high={th_high:.4f}, th_low={th_low:.4f}")
print("=" * 85)
print(f"{'Sequence':10s} | {'Anomaly Type':20s} | {'AUC':7s} | {'AP':7s} | {'Prec':7s} | {'Rec':7s} | {'Status'}")
print("=" * 85)

for entry in mani["videos"]:
    sid = entry["sequence_id"]
    d_i = torch.load(ROOT / "cache" / "ucsd_ped2" / f"{sid}.pt", map_location="cpu", weights_only=False)
    d_r = torch.load(ROOT / "cache" / "ucsd_ped2_reid" / f"{sid}.pt", map_location="cpu", weights_only=False)
    with torch.inference_mode():
        f_i = fuse_features(model, d_i["rgb_features"].float(), None, rgb_available=True, ir_available=False)
        _, s_i = score_fused_sequence(model, f_i, smoothing_window=7)
        f_r = fuse_features(model, F.normalize(d_r["rgb_features"].float(), p=2, dim=-1), None, rgb_available=True, ir_available=False)
        _, s_r = score_fused_sequence(model, f_r, smoothing_window=7)
    z_i = (s_i - mu_img) / std_img
    z_r = (s_r - mu_reid) / std_reid
    ens = 0.3 * z_i + 0.7 * z_r
    
    lbl = np.zeros(len(ens), dtype=int)
    for seg in entry["segments"]:
        lbl[seg[0]:seg[1]] = 1
        
    # Hysteresis
    pred = np.zeros_like(ens, dtype=int)
    hold = 0
    for t in range(len(ens)):
        if ens[t] >= th_high:
            hold = 15
            pred[t] = 1
        elif hold > 0 and ens[t] >= th_low:
            hold -= 1
            pred[t] = 1
        else:
            hold = 0
            
    # Metrics
    has_pos = bool(np.any(lbl == 1))
    auc_str = f"{roc_auc_score(lbl, ens)*100:.1f}%" if has_pos and np.any(lbl == 0) else "N/A"
    ap_str = f"{average_precision_score(lbl, ens)*100:.1f}%" if has_pos and np.any(lbl == 0) else "N/A"
    tp = np.sum((pred == 1) & (lbl == 1))
    fp = np.sum((pred == 1) & (lbl == 0))
    fn = np.sum((pred == 0) & (lbl == 1))
    prec_str = f"{tp/(tp+fp)*100:.1f}%" if (tp+fp) else "0.0%"
    rec_str = f"{tp/(tp+fn)*100:.1f}%" if (tp+fn) else "0.0%"
    
    detected = False
    for seg in entry["segments"]:
        if np.any(pred[seg[0]:seg[1]] == 1):
            detected = True
            
    status = "DETECTED" if detected else ("CLEAN (NORMAL)" if not entry["segments"] else "MISSED")
    desc = entry.get("anomaly_type", "Normal" if not entry["segments"] else "Anomaly")
    print(f"{sid:10s} | {desc:20s} | {auc_str:7s} | {ap_str:7s} | {prec_str:7s} | {rec_str:7s} | {status}")
