"""Fixed UCSD Ped2 Ground Truth loader (handles both XXX.bmp and frameXXX.bmp).

Fixes the V1 prepare_ucsd_ped2.py bug where Test006_gt, Test009_gt, and Test012_gt
were missed because their pixel-mask filenames follow `frameXXX.bmp` instead of `XXX.bmp`,
and cross-checks intervals against the official `UCSDped2.m` annotation script.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def natural_frame_sort_key(path: Path) -> Tuple[int, str]:
    """Extract numeric frame index from filenames like 001.bmp or frame001.bmp."""
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def parse_ucsdped2_m(m_file: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse official 1-based inclusive MATLAB ground-truth intervals into 0-based [start, end)."""
    if not m_file.exists():
        return {}
    text = m_file.read_text(encoding="utf-8", errors="ignore")
    # Format lines like: TestVideoFile{end+1}.gt_frame = [1, 61:180]; or [1:180];
    intervals: Dict[str, List[Tuple[int, int]]] = {}
    matches = re.findall(r"gt_frame\s*=\s*\[([^\]]+)\]", text)
    for idx, expr in enumerate(matches, start=1):
        sid = f"Test{idx:03d}"
        segs: List[Tuple[int, int]] = []
        for token in re.split(r"[,;\s]+", expr.strip()):
            if ":" not in token:
                continue
            parts = token.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_1b = int(parts[0])
                end_1b = int(parts[1])
                segs.append((max(0, start_1b - 1), end_1b))
        intervals[sid] = segs
    return intervals


def load_ped2_test_labels(
    test_root: Path,
    sequence_id: str,
    num_frames: Optional[int] = None,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Load binary per-frame GT labels and [start, end) segments for a Ped2 test clip.

    Priority:
      1. Pixel-mask BMPs inside `<sequence_id>_gt/` (both `*.bmp` excluding `._*` metadata).
      2. Fallback to `UCSDped2.m` if mask directory is missing.
    """
    test_root = Path(test_root)
    gt_dir = test_root / f"{sequence_id}_gt"
    bmp_files = []
    if gt_dir.exists():
        bmp_files = sorted(
            [p for p in gt_dir.glob("*.bmp") if not p.name.startswith("._")],
            key=natural_frame_sort_key,
        )

    if bmp_files:
        n = int(num_frames) if num_frames is not None else len(bmp_files)
        labels = np.zeros(n, dtype=np.int64)
        for i in range(min(n, len(bmp_files))):
            mask = cv2.imread(str(bmp_files[i]), cv2.IMREAD_GRAYSCALE)
            if mask is not None and bool((mask > 0).any()):
                labels[i] = 1
    else:
        m_intervals = parse_ucsdped2_m(test_root / "UCSDped2.m")
        if sequence_id not in m_intervals:
            raise FileNotFoundError(
                f"Neither BMP masks ({gt_dir}) nor UCSDped2.m interval found for {sequence_id}"
            )
        segs_raw = m_intervals[sequence_id]
        inferred_n = max((end for _, end in segs_raw), default=0)
        n = int(num_frames) if num_frames is not None else inferred_n
        labels = np.zeros(n, dtype=np.int64)
        for s0, s1 in segs_raw:
            labels[max(0, s0) : min(n, s1)] = 1

    segments: List[List[int]] = []
    in_seg = False
    seg_start = 0
    for i, val in enumerate(labels):
        if val == 1 and not in_seg:
            in_seg = True
            seg_start = i
        elif val == 0 and in_seg:
            in_seg = False
            segments.append([seg_start, i])
    if in_seg:
        segments.append([seg_start, len(labels)])

    return labels, segments
