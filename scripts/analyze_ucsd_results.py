import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from training.metrics import compute_event_metrics, compute_false_alarm_rate

with open("data/ucsd_ped2/manifest.json", "r", encoding="utf-8") as f:
    mani = json.load(f)

df = pd.read_csv("Evaluation/results_ucsd_ped2/scores.csv")
segments = [entry["segments"] for entry in mani["videos"]]

all_scores = []
for entry in mani["videos"]:
    sid = entry["sequence_id"]
    sub = df[df["sequence_id"] == sid]
    all_scores.append(sub["smoothed_score"].values)

normal_scores = df[df["label"] == 0]["smoothed_score"].values
p95_th = float(np.percentile(normal_scores, 95))
mean_3std_th = float(np.mean(normal_scores) + 3 * np.std(normal_scores))

thresholds = [
    ("Frozen VidLLVIP Production Threshold", 3.2326),
    ("Site-Calibrated (Normal 95th Percentile)", p95_th),
    ("Site-Calibrated (Normal Mean + 3 Std)", mean_3std_th),
    ("Optimal F1 Threshold", 0.2619),
]

print("=" * 80)
print("UCSD Ped2 Zero-Shot Evaluation Detailed Report")
print("=" * 80)
for name, th in thresholds:
    ev = compute_event_metrics(segments, all_scores, threshold=th)
    fa = compute_false_alarm_rate(normal_scores, threshold=th, fps=10.0)
    pred = (df["smoothed_score"].values >= th).astype(int)
    labels = df["label"].values
    tp = int(np.sum((pred == 1) & (labels == 1)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\n[{name}] Threshold: {th:.4f}")
    print(f"  Frame Metrics : Precision={prec:6.2%}, Frame Recall={rec:6.2%}, F1={f1:6.4f}, FPR={fa['frame_fpr']:6.2%}")
    print(f"  Event Metrics : {ev['num_detected']}/{ev['num_events']} events detected ({ev['event_recall']:6.2%}), Mean Delay = {ev['mean_delay_frames']:.1f} frames")
    print(f"  False Alarms  : {fa['false_alarm_frames']} frames ({fa['false_alarms_per_hour']:.1f} / hr @ 10fps)")
