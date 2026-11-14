"""
Binary anomaly-detection metrics.

All metrics are frame-level.  Scores are treated as anomaly scores: a larger
value means "more anomalous".  The module works even when the evaluation set
contains only normal frames (AUC/AP are then reported as ``NaN``), because a
normal-only smoke test is still useful for measuring the false-positive rate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:  # scikit-learn is in requirements.txt, but keep a clear error if missing.
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover - exercised only in broken installs
    average_precision_score = None
    precision_recall_curve = None
    roc_auc_score = None


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    tn = int(np.sum((predictions == 0) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """False-positive rate at the smallest threshold reaching ``target_tpr``."""
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    threshold = float(np.quantile(positives, 1.0 - target_tpr))
    return float(np.mean(negatives >= threshold))


def compute_binary_metrics(labels,
                           scores,
                           threshold: Optional[float] = None,
                           target_tpr: float = 0.95) -> Dict[str, float]:
    """
    Compute frame-level AUC, AP, threshold metrics, and FPR@95%TPR.

    Args:
        labels: Binary labels (0 = normal, 1 = anomaly)
        scores: Anomaly scores
        threshold: Optional decision threshold.  If omitted, the threshold
            that maximises F1 on this data is used and also returned.
        target_tpr: Target recall for the FPR-at-TPR metric

    Returns:
        Dictionary of metrics.  Values that are undefined (for example AUC on
        a single-class set) are ``NaN``.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError(f"labels and scores must have the same length; got {labels.size} and {scores.size}")
    if labels.size == 0:
        raise ValueError("labels and scores are empty")

    scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(np.float64).max, neginf=0.0)
    unique_labels = np.unique(labels)
    has_both_classes = unique_labels.size == 2

    auc = float("nan")
    ap = float("nan")
    if has_both_classes and roc_auc_score is not None:
        auc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))

    best_threshold = float("nan")
    best_f1 = float("nan")
    if precision_recall_curve is not None and has_both_classes:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
        if thresholds.size:
            best_index = int(np.argmax(f1[:-1])) if f1.size > thresholds.size else int(np.argmax(f1))
            best_index = min(best_index, thresholds.size - 1)
            best_threshold = float(thresholds[best_index])
            best_f1 = float(f1[best_index])

    if threshold is None:
        threshold = best_threshold if np.isfinite(best_threshold) else 0.5
    threshold = float(threshold)

    predictions = (scores >= threshold).astype(np.int64)
    counts = _confusion(labels, predictions)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision_at_threshold = _safe_div(tp, tp + fp)
    recall_at_threshold = _safe_div(tp, tp + fn)
    f1_at_threshold = _safe_div(
        2.0 * precision_at_threshold * recall_at_threshold,
        precision_at_threshold + recall_at_threshold,
    )
    accuracy = _safe_div(tp + tn, labels.size)
    fpr = _safe_div(fp, fp + tn)
    tpr = _safe_div(tp, tp + fn)
    fnr = _safe_div(fn, fn + tp)

    return {
        "auc": auc,
        "ap": ap,
        "threshold": threshold,
        "best_f1_threshold": best_threshold,
        "best_f1": best_f1,
        "precision": precision_at_threshold,
        "recall": recall_at_threshold,
        "f1": f1_at_threshold,
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "fpr_at_95tpr": _fpr_at_tpr(labels, scores, target_tpr),
        "num_frames": int(labels.size),
        "num_normal": int(np.sum(labels == 0)),
        "num_anomaly": int(np.sum(labels == 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "has_both_classes": bool(has_both_classes),
    }


def compute_video_metrics(video_labels,
                          video_scores,
                          threshold: Optional[float] = None) -> Dict[str, float]:
    """Compute AUC/AP using one maximum score per video."""
    return compute_binary_metrics(video_labels, video_scores, threshold=threshold)


def compute_event_metrics(segments_per_video: Sequence[Sequence[Sequence[int]]],
                          scores_per_video: Sequence[np.ndarray],
                          threshold: float) -> Dict[str, Any]:
    """
    Compute event-level recall and mean detection delay in frames.

    An anomalous segment [start, end) counts as detected if any frame inside
    the segment reaches or exceeds the threshold.  The detection delay is the
    difference between the first flagged frame and the segment start.

    Args:
        segments_per_video: For each video, a list of [start, end] intervals.
        scores_per_video: For each video, a 1-D array of anomaly scores.
        threshold: Operating decision threshold.

    Returns:
        Dictionary with num_events, num_detected, event_recall, and mean_delay_frames.
    """
    total_events = 0
    detected_events = 0
    delays: List[int] = []

    for segments, scores in zip(segments_per_video, scores_per_video):
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        for seg in segments:
            if len(seg) != 2:
                continue
            start, end = int(seg[0]), int(seg[1])
            start = max(0, start)
            end = min(scores.size, end)
            if end <= start:
                continue

            total_events += 1
            seg_scores = scores[start:end]
            flagged = np.where(seg_scores >= threshold)[0]
            if flagged.size > 0:
                detected_events += 1
                delay = int(flagged[0])  # Frame offset from start
                delays.append(delay)

    event_recall = float(detected_events / max(total_events, 1)) if total_events > 0 else float("nan")
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    return {
        "num_events": total_events,
        "num_detected": detected_events,
        "event_recall": event_recall,
        "mean_delay_frames": mean_delay,
    }


def compute_false_alarm_rate(normal_scores: np.ndarray,
                             threshold: float,
                             fps: float = 25.0) -> Dict[str, float]:
    """
    Compute frame false-positive rate and false alarms per hour on normal video.

    Args:
        normal_scores: 1-D array of scores on normal frames.
        threshold: Operating decision threshold.
        fps: Video frame rate (default: 25.0 fps for VidLLVIP).

    Returns:
        Dictionary with normal_frames, false_alarm_frames, frame_fpr, and false_alarms_per_hour.
    """
    scores = np.asarray(normal_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return {
            "normal_frames": 0,
            "false_alarm_frames": 0,
            "frame_fpr": float("nan"),
            "false_alarms_per_hour": float("nan"),
        }

    flags = int(np.sum(scores >= threshold))
    fps = max(float(fps), 1.0)
    total_hours = (scores.size / fps) / 3600.0
    fa_per_hour = float(flags / max(total_hours, 1e-6))
    frame_fpr = float(flags / scores.size)

    return {
        "normal_frames": int(scores.size),
        "false_alarm_frames": flags,
        "frame_fpr": frame_fpr,
        "false_alarms_per_hour": fa_per_hour,
    }


def bootstrap_video_metric(per_video_labels: Sequence[np.ndarray],
                           per_video_scores: Sequence[np.ndarray],
                           metric_name: str = "auc",
                           threshold: Optional[float] = None,
                           num_bootstraps: int = 500,
                           ci: float = 0.95,
                           seed: int = 42) -> Dict[str, float]:
    """
    Compute cluster bootstrap confidence intervals resampling over whole videos.

    Avoids over-optimistic i.i.d. frame assumptions across correlated frames of the same video.
    """
    n_vids = len(per_video_labels)
    if n_vids == 0:
        return {"point_estimate": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    # Point estimate
    all_l = np.concatenate(per_video_labels)
    all_s = np.concatenate(per_video_scores)
    point = compute_binary_metrics(all_l, all_s, threshold=threshold).get(metric_name, float("nan"))

    rng = np.random.default_rng(seed)
    boot_vals: List[float] = []

    for _ in range(num_bootstraps):
        sampled_indices = rng.choice(n_vids, size=n_vids, replace=True)
        samp_labels = np.concatenate([per_video_labels[i] for i in sampled_indices])
        samp_scores = np.concatenate([per_video_scores[i] for i in sampled_indices])
        res = compute_binary_metrics(samp_labels, samp_scores, threshold=threshold)
        val = res.get(metric_name, float("nan"))
        if np.isfinite(val):
            boot_vals.append(float(val))

    if not boot_vals:
        return {"point_estimate": point, "ci_lower": float("nan"), "ci_upper": float("nan")}

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_vals, 100.0 * alpha))
    upper = float(np.percentile(boot_vals, 100.0 * (1.0 - alpha)))

    return {
        "point_estimate": point,
        "ci_lower": lower,
        "ci_upper": upper,
    }
