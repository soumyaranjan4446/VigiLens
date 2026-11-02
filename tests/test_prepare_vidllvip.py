"""Tests for scripts/prepare_vidllvip.py."""

import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_vidllvip import (
    discover_vidllvip_dirs,
    discover_vidllvip_pairs,
    extract_scene_id,
    prepare_vidllvip,
    split_by_scenes,
)


def _make_dummy_mp4(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"dummy mp4 content")
    return path


def test_extract_scene_id():
    assert extract_scene_id("01_0000_0005.mp4") == "01"
    assert extract_scene_id("12_0010_0015.mp4") == "12"
    assert extract_scene_id("sceneA_part1.mp4") == "sceneA"


def test_discover_vidllvip_dirs(tmp_path):
    root = tmp_path / "vidllvip"
    vi = root / "dataset" / "vi"
    ir = root / "dataset" / "ir"
    vi.mkdir(parents=True)
    ir.mkdir(parents=True)

    found_vi, found_ir = discover_vidllvip_dirs(root)
    assert found_vi == vi
    assert found_ir == ir


def test_discover_pairs_and_split(tmp_path):
    root = tmp_path / "vidllvip"
    vi_dir = root / "vi"
    ir_dir = root / "ir"

    # Create scene 01, scene 02, scene 03
    for scene in ["01", "02", "03"]:
        for interval in ["0000_0005", "0005_0010"]:
            name = f"{scene}_{interval}.mp4"
            _make_dummy_mp4(vi_dir / name)
            _make_dummy_mp4(ir_dir / name)

    pairs = discover_vidllvip_pairs(vi_dir, ir_dir)
    assert len(pairs) == 6

    # Test scene splitting with explicit val_scenes
    train_pairs, val_pairs = split_by_scenes(pairs, val_scenes=["03"])
    assert len(train_pairs) == 4
    assert len(val_pairs) == 2
    assert all(item[3] != "03" for item in train_pairs)
    assert all(item[3] == "03" for item in val_pairs)


def test_prepare_vidllvip_full_flow(tmp_path):
    src_dir = tmp_path / "dataset"
    vi_dir = src_dir / "dataset" / "vi"
    ir_dir = src_dir / "dataset" / "ir"

    # Create 3 scenes
    for scene in ["01", "02", "03"]:
        for interval in ["0000_0005", "0005_0010"]:
            name = f"{scene}_{interval}.mp4"
            _make_dummy_mp4(vi_dir / name)
            _make_dummy_mp4(ir_dir / name)

    out_dir = tmp_path / "Data"
    manifest_path = tmp_path / "manifest.json"

    counts = prepare_vidllvip(
        input_path=src_dir,
        output_dir=out_dir,
        val_scenes=["03"],
        copy_files=True,
        eval_manifest=manifest_path,
    )

    assert counts["train"] == 4
    assert counts["val"] == 2

    # Check files created
    assert (out_dir / "train" / "rgb" / "01_0000_0005_rgb.mp4").exists()
    assert (out_dir / "train" / "ir" / "01_0000_0005_ir.mp4").exists()
    assert (out_dir / "val" / "rgb" / "03_0000_0005_rgb.mp4").exists()
    assert (out_dir / "val" / "ir" / "03_0000_0005_ir.mp4").exists()

    # Check manifest
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert len(data["videos"]) == 2
    assert data["videos"][0]["label"] == "normal"


def test_prepare_vidllvip_from_zip(tmp_path):
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("dataset/vi/01_0000_0005.mp4", b"dummy")
        zf.writestr("dataset/ir/01_0000_0005.mp4", b"dummy")

    out_dir = tmp_path / "Data_zip"
    counts = prepare_vidllvip(
        input_path=zip_path,
        output_dir=out_dir,
        all_train=True,
        copy_files=True,
    )
    assert counts["train"] == 1
    assert counts["val"] == 0
    assert (out_dir / "train" / "rgb" / "01_0000_0005_rgb.mp4").exists()
    assert (out_dir / "train" / "ir" / "01_0000_0005_ir.mp4").exists()
