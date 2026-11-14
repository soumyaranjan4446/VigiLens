"""Tests for frame-level anomaly metrics."""

import math

import numpy as np

from training.metrics import (
    bootstrap_video_metric,
    compute_binary_metrics,
    compute_event_metrics,
    compute_false_alarm_rate,
    compute_video_metrics,
)


def test_perfect_separation():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)

    assert metrics["auc"] == 1.0
    assert metrics["ap"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["fpr"] == 0.0
    assert metrics["tpr"] == 1.0
    assert metrics["fnr"] == 0.0
    assert metrics["fpr_at_95tpr"] == 0.0
    assert metrics["tp"] == 2 and metrics["fp"] == 0
    assert metrics["tn"] == 2 and metrics["fn"] == 0
    assert metrics["num_frames"] == 4


def test_threshold_confusion_values():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.6, 0.4, 0.9]
    metrics = compute_binary_metrics(labels, scores, threshold=0.5)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["fpr"] == 0.5
    assert metrics["tpr"] == 0.5
    assert metrics["fnr"] == 0.5
    assert metrics["fpr_at_95tpr"] == 0.5
    assert metrics["tp"] == 1 and metrics["fp"] == 1
    assert metrics["tn"] == 1 and metrics["fn"] == 1


def test_single_class_returns_nan_auc_but_valid_fpr():
    labels = [0, 0, 0]
    scores = [0.1, 0.2, 0.3]
    metrics = compute_binary_metrics(labels, scores, threshold=0.2)

    assert math.isnan(metrics["auc"])
    assert math.isnan(metrics["ap"])
    assert math.isnan(metrics["fpr_at_95tpr"])
    assert metrics["has_both_classes"] is False
    assert metrics["fpr"] == 2.0 / 3.0
    assert metrics["tp"] == 0
    assert metrics["fp"] == 2
    assert metrics["tn"] == 1


def test_auto_threshold_uses_best_f1():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_binary_metrics(labels, scores, threshold=None)
    assert 0.2 < metrics["threshold"] <= 0.8
    assert metrics["best_f1"] == 1.0


def test_metrics_reject_invalid_inputs():
    try:
        compute_binary_metrics([0, 1], [0.1])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Length mismatch should raise")

    try:
        compute_binary_metrics([], [])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Empty input should raise")


def test_video_metrics_use_one_score_per_video():
    metrics = compute_video_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["auc"] == 1.0
    assert metrics["num_frames"] == 4


def test_compute_event_metrics_hand_computed():
    # Video 1: segment [10, 20). Flagged at index 12 -> delay = 2 frames.
    scores1 = np.zeros(30)
    scores1[12] = 0.8
    scores1[15] = 0.9

    # Video 2: segment [5, 15). Never reaches threshold 0.5.
    scores2 = np.zeros(25)
    scores2[8] = 0.3

    res = compute_event_metrics(
        segments_per_video=[[[10, 20]], [[5, 15]]],
        scores_per_video=[scores1, scores2],
        threshold=0.5,
    )

    assert res["num_events"] == 2
    assert res["num_detected"] == 1
    assert res["event_recall"] == 0.5
    assert res["mean_delay_frames"] == 2.0


def test_compute_false_alarm_rate_hand_computed():
    normal_scores = np.zeros(100)
    normal_scores[:4] = 0.6  # 4 false alarms above 0.5
    res = compute_false_alarm_rate(normal_scores, threshold=0.5, fps=25.0)

    assert res["normal_frames"] == 100
    assert res["false_alarm_frames"] == 4
    assert res["frame_fpr"] == 0.04
    # (100 / 25) / 3600 = 4 / 3600 hr -> 4 / (4 / 3600) = 3600 fa/hr
    assert math.isclose(res["false_alarms_per_hour"], 3600.0)


def test_bootstrap_video_metric():
    labels = [np.array([0, 0, 1, 1]), np.array([0, 1])]
    scores = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.2, 0.7])]
    res = bootstrap_video_metric(labels, scores, metric_name="auc", num_bootstraps=50, seed=42)

    assert np.isfinite(res["point_estimate"])
    assert res["ci_lower"] <= res["point_estimate"] <= res["ci_upper"]

