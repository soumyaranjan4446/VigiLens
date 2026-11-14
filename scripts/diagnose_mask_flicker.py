"""
Diagnose YOLO mask drops and flicker in VigiLens.

Measures:
1. Frequency of YOLO person drops (person_found == False) across normal scenes.
2. Feature delta norms ||z_t - z_{t-1}|| on transition frames vs stable frames.
3. Correlation between mask drops/transitions and prediction error spikes (> threshold).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from scipy import stats

from models.anomaly_model import VigiLensModel
from training.scoring import score_cache_file


def analyze_scene_flicker(model: VigiLensModel,
                          cache_paths: List[Path],
                          threshold: float = 0.5,
                          smoothing_window: int = 5) -> Dict:
    total_frames = 0
    total_drops = 0
    drop_transitions = 0  # 1 -> 0 transitions
    return_transitions = 0  # 0 -> 1 transitions

    all_raw_scores = []
    all_smoothed_scores = []
    all_person_found = []
    all_transitions = []
    all_delta_norms = []

    model.eval()

    for p in cache_paths:
        data = torch.load(p, map_location="cpu")
        rgb = data["rgb_features"].float()
        ir = data["ir_features"].float()
        T = rgb.shape[0]

        if "person_found" in data:
            pf = data["person_found"].cpu().numpy().astype(bool)
        else:
            pf = np.ones(T, dtype=bool)

        # Compute fused features to measure feature deltas
        with torch.no_grad():
            fused = model.fusion(rgb, ir).cpu().numpy()  # (T, 512)

        # Delta norms ||z_t - z_{t-1}||
        delta_norms = np.zeros(T, dtype=np.float32)
        delta_norms[1:] = np.linalg.norm(fused[1:] - fused[:-1], axis=1)

        # Score the file
        res = score_cache_file(model, p, smoothing_window=smoothing_window, device="cpu")
        raw = res["raw_scores"]
        smoothed = res["smoothed_scores"]

        # Transitions
        transitions = np.zeros(T, dtype=np.int32)
        for t in range(1, T):
            if pf[t - 1] and not pf[t]:
                transitions[t] = -1  # drop
                drop_transitions += 1
            elif not pf[t - 1] and pf[t]:
                transitions[t] = 1   # return
                return_transitions += 1

        total_frames += T
        total_drops += int(np.sum(~pf))

        all_raw_scores.append(raw)
        all_smoothed_scores.append(smoothed)
        all_person_found.append(pf)
        all_transitions.append(transitions)
        all_delta_norms.append(delta_norms)

    raw_arr = np.concatenate(all_raw_scores)
    smooth_arr = np.concatenate(all_smoothed_scores)
    pf_arr = np.concatenate(all_person_found)
    trans_arr = np.concatenate(all_transitions)
    delta_arr = np.concatenate(all_delta_norms)

    is_transition = (trans_arr != 0)
    is_stable = (trans_arr == 0)

    # Spike definitions
    raw_spikes = (raw_arr >= threshold)
    smooth_spikes = (smooth_arr >= threshold)

    # Point biserial correlations
    corr_trans_raw, p_trans_raw = stats.pointbiserialr(is_transition.astype(int), raw_arr)
    corr_trans_smooth, p_trans_smooth = stats.pointbiserialr(is_transition.astype(int), smooth_arr)
    corr_pf_raw, p_pf_raw = stats.pointbiserialr(pf_arr.astype(int), raw_arr)

    # Delta norms comparison
    mean_delta_trans = float(np.mean(delta_arr[is_transition])) if np.any(is_transition) else 0.0
    mean_delta_stable = float(np.mean(delta_arr[is_stable])) if np.any(is_stable) else 0.0

    # Probability of spike given transition vs stable
    p_spike_given_trans = float(np.mean(raw_spikes[is_transition])) if np.any(is_transition) else 0.0
    p_spike_given_stable = float(np.mean(raw_spikes[is_stable])) if np.any(is_stable) else 0.0

    return {
        "total_frames": total_frames,
        "person_missing_fraction": float(total_drops / max(1, total_frames)),
        "drop_transitions": drop_transitions,
        "return_transitions": return_transitions,
        "mean_delta_transition": mean_delta_trans,
        "mean_delta_stable": mean_delta_stable,
        "delta_ratio": (mean_delta_trans / max(1e-6, mean_delta_stable)),
        "corr_transition_raw_score": float(corr_trans_raw),
        "corr_transition_smooth_score": float(corr_trans_smooth),
        "p_spike_given_transition": p_spike_given_trans,
        "p_spike_given_stable": p_spike_given_stable,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose YOLO mask drops and flicker")
    parser.add_argument("--cache-dir", default="cache/scenes")
    parser.add_argument("--checkpoint", default="experiments/runs/baseline_v0/fold_0_seed_42/checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    paths = sorted(Path(args.cache_dir).glob("*.pt"))
    if not paths:
        print("No cache files found in", args.cache_dir)
        return

    model = VigiLensModel(device="cpu", load_extractors=False)
    if Path(args.checkpoint).exists():
        model.load_checkpoint(args.checkpoint)
    model.eval()

    report = analyze_scene_flicker(model, paths, threshold=args.threshold)
    print("\n=== YOLO Mask Flicker & Drop Diagnostic Report ===")
    print(f"Total Analyzed Frames: {report['total_frames']}")
    print(f"Person Missing Fraction: {report['person_missing_fraction']:.2%}")
    print(f"Drop Transitions (1->0): {report['drop_transitions']}")
    print(f"Return Transitions (0->1): {report['return_transitions']}")
    print(f"Mean ||Delta z|| on Transitions: {report['mean_delta_transition']:.4f}")
    print(f"Mean ||Delta z|| on Stable Frames: {report['mean_delta_stable']:.4f}")
    print(f"Transition Delta Spike Ratio: {report['delta_ratio']:.2f}x")
    print(f"Correlation (Transition vs Raw Error): r = {report['corr_transition_raw_score']:.4f}")
    print(f"Correlation (Transition vs Smoothed Error): r = {report['corr_transition_smooth_score']:.4f}")
    print(f"P(Error Spike | Mask Transition): {report['p_spike_given_transition']:.2%}")
    print(f"P(Error Spike | Stable Tracking): {report['p_spike_given_stable']:.2%}")


if __name__ == "__main__":
    main()
