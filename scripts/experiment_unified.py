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

# Load model
model = VigiLensModel(device="cpu", load_extractors=False)
model.load_checkpoint(str(ROOT / "checkpoints" / "best_model.pth"))
model.to("cpu")
model.eval()

with open(ROOT / "data" / "ucsd_ped2" / "manifest.json", "r", encoding="utf-8") as f:
    mani = json.load(f)

segments = [entry["segments"] for entry in mani["videos"]]

def get_scores(cache_dir, norm_mode="none"):
    seq_scores = []
    all_labels = []
    for entry in mani["videos"]:
        sid = entry["sequence_id"]
        cache_file = ROOT / cache_dir / f"{sid}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        rgb = data["rgb_features"].float()
        if norm_mode == "unit":
            rgb = F.normalize(rgb, p=2, dim=-1)
        with torch.inference_mode():
            fused = fuse_features(model, rgb, None, rgb_available=True, ir_available=False)
            raw, smoothed = score_fused_sequence(model, fused, smoothing_window=7)
        seq_scores.append(smoothed)
        
        n_frames = len(rgb)
        lbl = np.zeros(n_frames, dtype=int)
        for seg in entry["segments"]:
            lbl[seg[0]:seg[1]] = 1
        all_labels.append(lbl)
    return seq_scores, all_labels

seq_img, labels_list = get_scores("cache/ucsd_ped2", norm_mode="none")
seq_reid_unit, _ = get_scores("cache/ucsd_ped2_reid", norm_mode="unit")
seq_reid_raw, _ = get_scores("cache/ucsd_ped2_reid", norm_mode="none")

y_true = np.concatenate(labels_list)
s_img = np.concatenate(seq_img)
s_reid_u = np.concatenate(seq_reid_unit)
s_reid_r = np.concatenate(seq_reid_raw)

# Standardize (Z-score)
def z_score(arr):
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-6)

z_img = z_score(s_img)
z_reid_u = z_score(s_reid_u)
z_reid_r = z_score(s_reid_r)

print("=== ENSEMBLE EVALUATION (ImageNet + ReID Unit-Norm) ===")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ens = (1 - alpha) * z_img + alpha * z_reid_u
    auc = roc_auc_score(y_true, ens)
    ap = average_precision_score(y_true, ens)
    print(f"Alpha={alpha:.1f} (ReID weight): ROC-AUC = {auc*100:.2f}%, AP = {ap*100:.2f}%")

print("\n=== ENSEMBLE EVALUATION (ImageNet + ReID Raw) ===")
for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    ens = (1 - alpha) * z_img + alpha * z_reid_r
    auc = roc_auc_score(y_true, ens)
    ap = average_precision_score(y_true, ens)
    print(f"Alpha={alpha:.1f} (ReID weight): ROC-AUC = {auc*100:.2f}%, AP = {ap*100:.2f}%")
