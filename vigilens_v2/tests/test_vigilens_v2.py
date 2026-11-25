"""Comprehensive unit tests for VigiLens V2 (GT fix, causal smoothing, hysteresis, tracker, TrackDynamicsModel, VigiLensV2Model, InferencePipelineV2)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from vigilens_v2.models.motion_encoder import CausalMotionFlowEncoder
from vigilens_v2.models.track_model import TrackDynamicsModel
from vigilens_v2.models.tracker import CausalIoUTracker, TrackDetection
from vigilens_v2.models.vigilens_v2_model import VigiLensV2Model
from vigilens_v2.pipeline.smoothing import CausalHysteresisDetector, causal_moving_average_skip0
from vigilens_v2.pipeline.streaming_v2 import InferencePipelineV2
from vigilens_v2.training.data_gt import load_ped2_test_labels, parse_ucsdped2_m


def test_causal_ma_skip0_eliminates_warmup_dilution():
    raw = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    out = causal_moving_average_skip0(raw, window=5)
    assert out[0] == 0.0
    # Unlike V1, t=1..5 are NOT dragged down by raw[0]=0.0
    assert np.allclose(out[1:], 1.0, atol=1e-6)


def test_causal_ma_skip0_strict_causality():
    a = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 10.0], dtype=np.float32)
    b = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 99.0], dtype=np.float32)
    assert np.allclose(causal_moving_average_skip0(a, 5)[:5], causal_moving_average_skip0(b, 5)[:5])


def test_hysteresis_hold_and_drop():
    det = CausalHysteresisDetector(threshold_high=1.0, threshold_low=0.5, hold_frames=3)
    seq = np.array([0.2, 1.2, 0.7, 0.6, 0.6, 0.6, 0.2, 0.7], dtype=np.float32)
    preds = det.run_sequence(seq)
    # t=1 triggers (hold=3), t=2..4 sustain (>=0.5, hold decrements 3->2->1->0), t=5 drops because hold==0 even though 0.6>=0.5
    assert list(preds) == [0, 1, 1, 1, 1, 0, 0, 0]


def test_ped2_gt_loader_fixes_framexxx_bmp():
    test_root = Path("Data/ucsd_ped2/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test")
    if not test_root.exists():
        pytest.skip("UCSD Ped2 dataset not present locally")
    # Test006, Test009, Test012 use frameXXX.bmp masks that V1 missed!
    l6, seg6 = load_ped2_test_labels(test_root, "Test006", num_frames=180)
    l9, seg9 = load_ped2_test_labels(test_root, "Test009", num_frames=120)
    l12, seg12 = load_ped2_test_labels(test_root, "Test012", num_frames=180)
    assert l6.sum() == 159 and seg6 == [[0, 159]]
    assert l9.sum() == 120 and seg9 == [[0, 120]]
    assert l12.sum() == 93 and seg12 == [[87, 180]]

    m_parsed = parse_ucsdped2_m(test_root / "UCSDped2.m")
    assert len(m_parsed) == 12
    assert m_parsed["Test006"] == [(0, 159)]


def test_causal_iou_tracker_preserves_ids_and_resets():
    trk = CausalIoUTracker(iou_thresh=0.2, max_center_dist=0.15, max_missed=2)
    b1 = np.array([[0.10, 0.20, 0.20, 0.45], [0.60, 0.20, 0.70, 0.45]], dtype=np.float32)
    d1 = trk.update(b1)
    assert [d.track_id for d in d1] == [1, 2]

    # Slightly shifted next frame should preserve exact IDs
    b2 = np.array([[0.12, 0.20, 0.22, 0.45], [0.61, 0.20, 0.71, 0.45]], dtype=np.float32)
    d2 = trk.update(b2)
    assert [d.track_id for d in d2] == [1, 2]
    assert d2[0].displacement_norm > d2[1].displacement_norm

    trk.reset()
    d3 = trk.update(b1[:1])
    assert [d.track_id for d in d3] == [1]


def test_track_dynamics_streaming_vs_offline_parity():
    model = TrackDynamicsModel(in_dim=8, hidden_dim=32, smoothing_window=5, min_track_len=3)
    model.eval()
    # Synthetic track 7 moving across frames 1..8
    meta_rows = []
    for f in range(1, 9):
        meta_rows.append([f, 7, 0.2, 0.4, 0.3, 0.6, 0.5, 0.2, 0.1, 1.2 + 0.1 * f, 1.4, 0.9, 1.1])
    meta = np.asarray(meta_rows, dtype=np.float32)

    off_scores, off_per_trk = model.score_meta_offline(meta, num_frames=10)

    model.reset_stream_state()
    str_scores = []
    for f in range(1, 9):
        sc, ts = model.step_frame_tracks(meta[f - 1 : f])
        str_scores.append((f, sc, ts))

    for f, sc, ts in str_scores:
        if f >= 3:
            assert abs(sc - off_scores[f]) < 1e-5
            assert abs(ts[7] - off_per_trk[f][7]) < 1e-5


def test_motion_encoder_and_v2_model_step():
    enc = CausalMotionFlowEncoder(flow_height=120, flow_width=180, grid_rows=12, grid_cols=18)
    tracks = [TrackDetection(track_id=1, box_norm=(0.2, 0.3, 0.35, 0.7), conf=0.9, cls_id=0, displacement_norm=2.5)]
    f0 = np.zeros((120, 180, 3), dtype=np.uint8)
    f1 = np.zeros((120, 180, 3), dtype=np.uint8)
    f1[:, 20:40] = 200

    g0, m0 = enc.compute_frame_flow_and_track_stats(f0, tracks, frame_index=0)
    assert g0.shape == (12, 18) and np.allclose(g0, 0.0)
    g1, m1 = enc.compute_frame_flow_and_track_stats(f1, tracks, frame_index=1)
    assert m1.shape == (1, 13)

    mot_2048 = enc.encode_motion_2048(g1[None, ...], np.zeros((1, 8), dtype=np.float32))
    assert mot_2048.shape == (1, 2048)

    v2 = VigiLensV2Model()
    v2.eval()
    v2.reset_stream_state()
    rgb = torch.randn(2048)
    out0 = v2.step_frame(0, g0, m0, rgb)
    out1 = v2.step_frame(1, g1, m1, rgb)
    assert out0["score"] == 0.0
    assert isinstance(out1["score"], float)


def test_render_overlay_draws_red_and_green_boxes():
    img = np.zeros((240, 360, 3), dtype=np.uint8)
    tracks = [
        TrackDetection(track_id=1, box_norm=(0.1, 0.2, 0.2, 0.6), conf=0.9, cls_id=0),
        TrackDetection(track_id=2, box_norm=(0.5, 0.3, 0.7, 0.8), conf=0.9, cls_id=1),
    ]
    out = InferencePipelineV2.render_overlay(
        frame_bgr=img,
        tracks=tracks,
        track_scores={1: 0.1, 2: 3.5},
        anom_ids=[2],
        frame_score=2.4,
        is_anomalous=True,
        threshold_high=1.23,
        fps=12.0,
        frame_idx=42,
    )
    assert out.shape == (480, 720, 3)
    # RED box edge around track 2 (x1=360, y1=144..384 in upscaled 720x480)
    assert int(out[240, 360, 2]) > 200
    assert int(out[240, 360, 0]) < 100


def test_dual_visual_extractor_handles_ndarray_outputs():
    from vigilens_v2.models.feature_extractor import DualVisualExtractor

    ext = DualVisualExtractor.__new__(DualVisualExtractor)
    ext.reid_scale = 1.0
    out_np = ext._to_1d_tensor(np.ones((2048,), dtype=np.float32))
    out_t = ext._to_1d_tensor(torch.ones((1, 2048), dtype=torch.float64))
    assert out_np.shape == (2048,) and out_np.dtype == torch.float32
    assert out_t.shape == (2048,) and out_t.dtype == torch.float32


def test_isolated_pipelines_do_not_leak_state():
    enc1 = CausalMotionFlowEncoder(flow_height=120, flow_width=180, grid_rows=12, grid_cols=18)
    enc2 = CausalMotionFlowEncoder(flow_height=120, flow_width=180, grid_rows=12, grid_cols=18, proj_2048=enc1.proj_2048)
    f0 = np.zeros((120, 180, 3), dtype=np.uint8)
    f1 = np.full((120, 180, 3), 255, dtype=np.uint8)
    g1, _ = enc1.compute_frame_flow_and_track_stats(f0, [], frame_index=0)
    assert enc2._prev_gray is None
    g2, _ = enc2.compute_frame_flow_and_track_stats(f1, [], frame_index=1)
    assert np.allclose(g1, 0.0) and np.allclose(g2, 0.0)
