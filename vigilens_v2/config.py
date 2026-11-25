"""Configuration dataclasses for VigiLens V2."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class TrackerConfig:
    yolo_weights: str = "yolov8n-seg.pt"
    conf_thresh: float = 0.25
    imgsz: int = 320
    iou_match_thresh: float = 0.22
    max_center_dist_frac: float = 0.14
    max_missed_frames: int = 3


@dataclass
class MotionConfig:
    flow_height: int = 240
    flow_width: int = 360
    grid_rows: int = 12
    grid_cols: int = 18
    min_track_len_for_score: int = 3
    smoothing_window_track: int = 7
    smoothing_window_grid: int = 5
    smoothing_window_scene: int = 7


@dataclass
class ModelV2Config:
    input_size: Tuple[int, int] = (224, 224)
    fusion_hidden_dim: int = 256
    fusion_output_dim: int = 512
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    track_feature_dim: int = 8
    track_hidden_dim: int = 64
    v1_checkpoint: str = "checkpoints/best_model.pth"
    reid_weights: str = "checkpoints/resnet50_market1501.pt"
    v2_checkpoint: str = "vigilens_v2/artifacts/vigilens_v2_ped2.pt"


@dataclass
class DecisionConfig:
    # Balanced ensemble weights across VigiLens V2 streams (sum = 1.0 on RGB single-cam)
    weight_track_kin: float = 0.56
    weight_flow_grid: float = 0.24
    weight_scene_app: float = 0.20
    weight_track_geom: float = 0.12
    weight_track_lstm_err: float = 0.06
    # Dual-threshold hysteresis defaults (frames at ~10-25 FPS)
    hysteresis_hold_frames: int = 15
    percentile_high: float = 95.0
    percentile_low: float = 80.0


@dataclass
class V2Config:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    model: ModelV2Config = field(default_factory=ModelV2Config)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    project_root: str = str(Path(__file__).resolve().parent.parent)


def get_v2_config() -> V2Config:
    return V2Config()
