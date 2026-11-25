"""Causal YOLOv8n-seg object detector and Hungarian IoU tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class TrackDetection:
    track_id: int
    box_norm: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0, 1]
    conf: float
    cls_id: int
    displacement_norm: float = 0.0  # per-frame normalized displacement * 100


@dataclass
class _TrackState:
    track_id: int
    box_norm: Tuple[float, float, float, float]
    missed: int = 0
    hits: int = 1


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between boxes A (M, 4) and B (N, 4) in [0, 1]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0:1].T, b[:, 1:2].T, b[:, 2:3].T, b[:, 3:4].T
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = np.maximum(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1) * (by2 - by1))
    union = np.maximum(1e-8, area_a + area_b - inter)
    return (inter / union).astype(np.float32)


def _center_dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ac = np.stack([0.5 * (a[:, 0] + a[:, 2]), 0.5 * (a[:, 1] + a[:, 3])], axis=1)
    bc = np.stack([0.5 * (b[:, 0] + b[:, 2]), 0.5 * (b[:, 1] + b[:, 3])], axis=1)
    diff = ac[:, None, :] - bc[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


class CausalIoUTracker:
    """Pure causal multi-object tracker using IoU + centroid proximity gating."""

    def __init__(
        self,
        iou_thresh: float = 0.22,
        max_center_dist: float = 0.14,
        max_missed: int = 3,
    ):
        self.iou_thresh = float(iou_thresh)
        self.max_center_dist = float(max_center_dist)
        self.max_missed = int(max_missed)
        self._next_id = 1
        self._active: Dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._active.clear()

    def update(
        self,
        boxes_norm: np.ndarray,
        confs: Optional[np.ndarray] = None,
        cls_ids: Optional[np.ndarray] = None,
    ) -> List[TrackDetection]:
        """Match normalized boxes (N, 4) in [x1, y1, x2, y2] causally to active tracks."""
        boxes = np.asarray(boxes_norm, dtype=np.float32).reshape(-1, 4) if len(boxes_norm) else np.zeros((0, 4), dtype=np.float32)
        n_det = len(boxes)
        if confs is None:
            confs = np.ones(n_det, dtype=np.float32)
        if cls_ids is None:
            cls_ids = np.zeros(n_det, dtype=np.int64)

        active_ids = list(self._active.keys())
        matched_trk = set()
        matched_det = set()
        results: List[TrackDetection] = []

        if active_ids and n_det > 0:
            trk_boxes = np.array([self._active[tid].box_norm for tid in active_ids], dtype=np.float32)
            iou = _box_iou_matrix(trk_boxes, boxes)
            dist = _center_dist_matrix(trk_boxes, boxes)
            score = iou - 0.5 * dist
            # Large negative sentinel for invalid pairs
            invalid = (iou < self.iou_thresh) & (dist > self.max_center_dist)

            if linear_sum_assignment is not None:
                cost = np.where(invalid, 1e6, -score)
                r_ind, c_ind = linear_sum_assignment(cost)
                pairs = [(int(r), int(c)) for r, c in zip(r_ind, c_ind) if not invalid[r, c]]
            else:
                pairs = []
                flat = np.argsort(-score.ravel())
                used_r, used_c = set(), set()
                for idx in flat:
                    r, c = divmod(int(idx), n_det)
                    if r in used_r or c in used_c or invalid[r, c]:
                        continue
                    used_r.add(r)
                    used_c.add(c)
                    pairs.append((r, c))

            for r, c in pairs:
                tid = active_ids[r]
                prev_box = self._active[tid].box_norm
                cur_box = tuple(float(v) for v in boxes[c])
                dx = 0.5 * ((cur_box[0] + cur_box[2]) - (prev_box[0] + prev_box[2]))
                dy = 0.5 * ((cur_box[1] + cur_box[3]) - (prev_box[1] + prev_box[3]))
                dt_factor = float(1 + self._active[tid].missed)
                disp = float(np.hypot(dx, dy) / dt_factor) * 100.0
                self._active[tid].box_norm = cur_box
                self._active[tid].missed = 0
                self._active[tid].hits += 1
                matched_trk.add(tid)
                matched_det.add(c)
                results.append(
                    TrackDetection(
                        track_id=tid,
                        box_norm=cur_box,
                        conf=float(confs[c]),
                        cls_id=int(cls_ids[c]),
                        displacement_norm=disp,
                    )
                )

        # Increment missed count & prune expired tracks
        for tid in active_ids:
            if tid not in matched_trk:
                self._active[tid].missed += 1
                if self._active[tid].missed > self.max_missed:
                    del self._active[tid]

        # Spawn new tracks for unmatched detections
        for c in range(n_det):
            if c in matched_det:
                continue
            tid = self._next_id
            self._next_id += 1
            cur_box = tuple(float(v) for v in boxes[c])
            self._active[tid] = _TrackState(track_id=tid, box_norm=cur_box, missed=0, hits=1)
            results.append(
                TrackDetection(
                    track_id=tid,
                    box_norm=cur_box,
                    conf=float(confs[c]),
                    cls_id=int(cls_ids[c]),
                    displacement_norm=0.0,
                )
            )

        results.sort(key=lambda d: d.track_id)
        return results


class YOLOCausalTracker:
    """YOLOv8n-seg detector paired with CausalIoUTracker and union mask generation."""

    def __init__(
        self,
        weights_path: str = "yolov8n-seg.pt",
        conf_thresh: float = 0.25,
        imgsz: int = 320,
        iou_match_thresh: float = 0.22,
        max_center_dist_frac: float = 0.14,
        max_missed_frames: int = 3,
        device: str = "cpu",
        yolo_model=None,
    ):
        if yolo_model is None:
            from ultralytics import YOLO

            self.yolo = YOLO(weights_path)
        else:
            self.yolo = yolo_model
        self.device = device
        self.conf_thresh = float(conf_thresh)
        self.imgsz = int(imgsz)
        self.tracker = CausalIoUTracker(
            iou_thresh=iou_match_thresh,
            max_center_dist=max_center_dist_frac,
            max_missed=max_missed_frames,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[TrackDetection], np.ndarray]:
        """Run detection, causal track association, and build binary foreground mask (H, W)."""
        h, w = frame_bgr.shape[:2]
        res = self.yolo.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            verbose=False,
            device=self.device,
        )[0]

        fg_mask = np.zeros((h, w), dtype=np.float32)
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(np.int64)
            # Filter non-trivial classes: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7), skateboard(36) or any moving foreground
            boxes_norm = np.stack(
                [
                    np.clip(xyxy[:, 0] / max(1, w), 0.0, 1.0),
                    np.clip(xyxy[:, 1] / max(1, h), 0.0, 1.0),
                    np.clip(xyxy[:, 2] / max(1, w), 0.0, 1.0),
                    np.clip(xyxy[:, 3] / max(1, h), 0.0, 1.0),
                ],
                axis=1,
            ).astype(np.float32)
            if res.masks is not None and res.masks.data is not None and len(res.masks.data) > 0:
                m_union = (res.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.float32)
                fg_mask = cv2.resize(m_union, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                for x1, y1, x2, y2 in xyxy.astype(int):
                    fg_mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1.0
        else:
            boxes_norm = np.zeros((0, 4), dtype=np.float32)
            confs = np.zeros(0, dtype=np.float32)
            cls_ids = np.zeros(0, dtype=np.int64)

        tracks = self.tracker.update(boxes_norm, confs=confs, cls_ids=cls_ids)
        return tracks, fg_mask
