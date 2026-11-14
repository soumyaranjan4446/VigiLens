"""Tests for split leakage audit."""

import pytest

from scripts.audit_splits import (
    LeakageError,
    audit_splits,
    extract_scene_id,
    generate_splits_markdown,
)


def test_extract_scene_id():
    assert extract_scene_id("01_0000_0005.mp4") == "01"
    assert extract_scene_id("13_0005_0010_rgb.mp4") == "13"
    assert extract_scene_id("09_0000_0005.pt") == "09"
    assert extract_scene_id("train_000_rgb.mp4") == "train"
    assert extract_scene_id("normal_001.pt") == "normal"


def test_audit_splits_disjoint_passes():
    splits = {
        "train": ["01_0000_0005", "01_0005_0010", "02_0000_0005"],
        "val": ["03_0000_0005", "03_0005_0010"],
        "test": ["04_0000_0005", "05_0000_0005"],
    }
    result = audit_splits(splits, fail_on_overlap=True)
    assert result["passed"] is True
    assert len(result["overlaps"]) == 0
    assert result["splits"]["train"]["num_scenes"] == 2
    assert result["splits"]["val"]["num_scenes"] == 1
    assert result["splits"]["test"]["num_scenes"] == 2


def test_audit_splits_leakage_fails():
    # Scene 13 in both val and test (as happened in baseline run)
    splits = {
        "train": ["01_0000_0005", "02_0000_0005"],
        "val": ["12_0000_0005", "13_0000_0005", "14_0000_0005"],
        "test": ["09_0000_0005", "13_0005_0010"],
    }
    with pytest.raises(LeakageError) as exc_info:
        audit_splits(splits, fail_on_overlap=True)

    err = str(exc_info.value)
    assert "val" in err and "test" in err
    assert "13" in err


def test_generate_splits_markdown():
    splits = {
        "train": ["01_0000_0005"],
        "val": ["02_0000_0005"],
        "test": ["03_0000_0005"],
    }
    result = audit_splits(splits, fail_on_overlap=True)
    md = generate_splits_markdown(result)
    assert "PASSED (Zero Leakage)" in md
    assert "| `train` | 1 | 1 | `01` |" in md


def test_folds_json_disjointness():
    import json
    from pathlib import Path
    folds_path = Path("experiments/splits/folds.json")
    if not folds_path.exists():
        pytest.skip("experiments/splits/folds.json not yet created")

    with open(folds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    holdout = data["frozen_holdout"]
    assert len(holdout) > 0

    for fold in data["folds"]:
        splits = {
            "train": [f"{s}_0000" for s in fold["train"]],
            "val": [f"{s}_0000" for s in fold["val"]],
            "calib": [f"{s}_0000" for s in fold["calib"]],
            "test": [f"{s}_0000" for s in fold["test"]],
            "holdout": [f"{s}_0000" for s in holdout],
        }
        res = audit_splits(splits, fail_on_overlap=True)
        assert res["passed"] is True
        assert len(res["overlaps"]) == 0
