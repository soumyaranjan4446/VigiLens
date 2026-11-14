"""Tests for pseudo-anomaly generator."""

import json
from pathlib import Path

import torch

from scripts.make_pseudo_anomalies import (
    TIER_A_TYPES,
    apply_tier_a_anomaly,
    generate_tier_a_dataset,
)


def _make_dummy_cache(num_frames: int = 60, feat_dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.randn(num_frames, feat_dim, generator=g)
    ir = torch.randn(num_frames, feat_dim, generator=g)
    return {
        "rgb_features": rgb,
        "ir_features": ir,
        "has_ir": True,
        "sequence_id": f"dummy_{seed}",
        "fps": 20.0,
        "frame_indices": list(range(num_frames)),
    }


def test_tier_a_types_preserve_shape_and_align_labels():
    for anomaly_type in TIER_A_TYPES:
        data = _make_dummy_cache(num_frames=60, seed=1)
        donor = _make_dummy_cache(num_frames=60, seed=2)
        perturbed, segments = apply_tier_a_anomaly(
            data,
            anomaly_type=anomaly_type,
            donor_data=donor,
            min_len=10,
            max_len=20,
            min_warmup=15,
            seed=42,
        )

        assert perturbed["rgb_features"].shape == data["rgb_features"].shape
        assert perturbed["ir_features"].shape == data["ir_features"].shape
        assert len(segments) == 1
        start, end = segments[0]
        assert 15 <= start < end <= 60

        labels = perturbed["labels"]
        assert labels.shape == (60,)
        assert labels[:start].sum() == 0
        assert labels[start:end].sum() == (end - start)
        assert labels[end:].sum() == 0


def test_tier_a_seed_reproducibility():
    data = _make_dummy_cache(num_frames=50, seed=0)
    p1, s1 = apply_tier_a_anomaly(data, "freeze", seed=123)
    p2, s2 = apply_tier_a_anomaly(data, "freeze", seed=123)
    assert s1 == s2
    assert torch.allclose(p1["rgb_features"], p2["rgb_features"])


def test_generate_tier_a_dataset(tmp_path):
    cache_dir = tmp_path / "src"
    cache_dir.mkdir()
    p1 = cache_dir / "clip1.pt"
    p2 = cache_dir / "clip2.pt"
    torch.save(_make_dummy_cache(num_frames=50, seed=1), p1)
    torch.save(_make_dummy_cache(num_frames=50, seed=2), p2)

    out_dir = tmp_path / "pseudo"
    manifest = generate_tier_a_dataset(
        cache_paths=[p1, p2],
        output_dir=out_dir,
        types=["reverse", "freeze"],
        copies_per_clip=1,
        seed=42,
    )

    assert manifest["is_proxy_benchmark"] is True
    assert len(manifest["videos"]) == 4  # 2 clips * 2 types
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "reverse").is_dir()
    assert (out_dir / "freeze").is_dir()

    # Verify sidecar json
    v0 = manifest["videos"][0]
    sidecar_path = Path(v0["cache"]).with_suffix(".json")
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["is_pseudo_anomaly"] is True
