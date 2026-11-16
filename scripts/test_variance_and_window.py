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

model = VigiLensModel(device="cpu", load_extractors=False)
model.load_checkpoint(str(ROOT / "checkpoints" / "best_model.pth"))
model.to("cpu")
model.eval()

with open(ROOT / "data" / "ucsd_ped2" / "manifest.json", "r", encoding="utf-8") as f:
    mani = json.load(f)

# Compute residual variance on Train001-Train004 with Unit-Norm
calib_clips = ["Train001", "Train002", "Train003", "Train004"]
train_cache_dir = ROOT / "cache" / "ucsd_ped2_train_reid"

residuals = []
for c_name in calib_clips:
    pt_path = train_cache_dir / f"{c_name}.pt"
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    rgb = F.normalize(data["rgb_features"].float(), p=2, dim=-1)
    with torch.inference_mode():
        fused = fuse_features(model, rgb, None, rgb_available=True, ir_available=False)
        chunk = fused.unsqueeze(0)
        preds, _ = model.temporal.forward_sequence(chunk, None)
        preds = preds.squeeze(0)
        res = preds[:-1] - fused[1:]
        residuals.append(res)

all_res = torch.cat(residuals, dim=0)
diag_var = (all_res ** 2).mean(dim=0).cpu().numpy()
print(f"Computed unit-norm Re-ID residual variance (dim={len(diag_var)}), mean var={diag_var.mean():.4f}")

# Evaluate norm_mse vs standard mse across smoothing windows
for w in [5, 7, 9, 11, 15]:
    scores_mse = []
    scores_norm_mse = []
    labels = []
    for entry in mani["videos"]:
        sid = entry["sequence_id"]
        cache_file = ROOT / "cache" / "ucsd_ped2_reid" / f"{sid}.pt"
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
        rgb = F.normalize(data["rgb_features"].float(), p=2, dim=-1)
        with torch.inference_mode():
            fused = fuse_features(model, rgb, None, rgb_available=True, ir_available=False)
            _, s_mse = score_fused_sequence(model, fused, smoothing_window=w, score_variant="mse")
            _, s_nmse = score_fused_sequence(model, fused, smoothing_window=w, score_variant="norm_mse", diag_variance=diag_var)
        scores_mse.append(s_mse)
        scores_norm_mse.append(s_nmse)
        lbl = np.zeros(len(rgb), dtype=int)
        for seg in entry["segments"]:
            lbl[seg[0]:seg[1]] = 1
        labels.append(lbl)
    
    y_true = np.concatenate(labels)
    auc_mse = roc_auc_score(y_true, np.concatenate(scores_mse))
    auc_nmse = roc_auc_score(y_true, np.concatenate(scores_norm_mse))
    ap_mse = average_precision_score(y_true, np.concatenate(scores_mse))
    ap_nmse = average_precision_score(y_true, np.concatenate(scores_norm_mse))
    print(f"Window {w:2d} | MSE: AUC={auc_mse*100:.2f}%, AP={ap_mse*100:.2f}% | NormMSE: AUC={auc_nmse*100:.2f}%, AP={ap_nmse*100:.2f}%")
