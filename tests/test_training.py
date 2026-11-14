"""Tests for cached training, validation, and checkpointing."""

import json

import torch

from models.anomaly_model import VigiLensModel
from training.dataset import CachedFeatureDataset
from training.train import (
    build_cached_datasets,
    discover_caches,
    train_cached,
)
from training.validate import validate_cached


def _dataset(paths, sequence_length=4):
    return CachedFeatureDataset(
        paths,
        sequence_length=sequence_length,
        sequence_stride=1,
        rgb_dropout_prob=0.0,
        ir_dropout_prob=0.0,
    )


def test_validate_cached_returns_real_errors(tiny_model, make_cache, tmp_path):
    first = make_cache(tmp_path / "a.pt", num_frames=12, has_ir=True)
    second = make_cache(tmp_path / "b.pt", num_frames=8, has_ir=False, seed=1)
    result = validate_cached(
        tiny_model,
        [first, second],
        device="cpu",
        sequence_length=4,
        batch_size=2,
        amp=False,
    )

    # 12 frames -> 9 sequences, 8 frames -> 5 sequences, 3 errors each.
    assert result["num_frames"] == (9 + 5) * 3
    assert result["num_batches"] == 7
    assert result["errors"].shape == (42,)
    assert result["persistence_errors"].shape == (42,)
    assert torch.isfinite(torch.tensor(result["val_loss"]))
    assert "val_ratio" in result
    assert "persistence_mse" in result
    assert torch.isfinite(torch.tensor(result["val_ratio"]))
    assert torch.isfinite(torch.tensor(result["persistence_mse"]))


def test_build_cached_datasets_and_discover(tiny_model, make_cache, tmp_path):
    cache_dir = tmp_path / "cache"
    make_cache(cache_dir / "train" / "a.pt", num_frames=10, has_ir=True)
    make_cache(cache_dir / "val" / "b.pt", num_frames=8, has_ir=True, seed=1)

    train_dataset, val_dataset = build_cached_datasets(
        cache_dir, sequence_length=4, sequence_stride=1,
        rgb_dropout_prob=0.0, ir_dropout_prob=0.0,
    )
    assert len(train_dataset) == 7
    assert len(val_dataset) == 5
    assert len(discover_caches(cache_dir)) == 2


def test_train_cached_one_epoch_writes_checkpoints(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)
    checkpoint_dir = tmp_path / "checkpoints"

    result = train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        amp=False,
        checkpoint_dir=checkpoint_dir,
        config_dict={"test": True},
    )

    assert result["history"][0]["epoch"] == 1
    assert (checkpoint_dir / "best_model.pth").exists()
    assert (checkpoint_dir / "last_model.pth").exists()
    assert (checkpoint_dir / "training_history.json").exists()
    with open(checkpoint_dir / "training_history.json", "r", encoding="utf-8") as handle:
        history = json.load(handle)
    assert len(history) == 1

    fresh = VigiLensModel(device="cpu", load_extractors=False)
    metadata = fresh.load_checkpoint(checkpoint_dir / "best_model.pth")
    assert metadata["epoch"] == 1
    assert metadata["config"] == {"test": True}
    for key, value in tiny_model.state_dict().items():
        assert torch.allclose(fresh.state_dict()[key], value)


def test_checkpoint_round_trip_without_extractors(tmp_path):
    model = VigiLensModel(device="cpu", load_extractors=False)
    assert model.rgb_extractor is None
    assert model.ir_extractor is None

    path = tmp_path / "checkpoint.pth"
    model.save_checkpoint(str(path), epoch=3, loss=0.25, config={"device": "cpu"})
    loaded = VigiLensModel(device="cpu", load_extractors=False)
    metadata = loaded.load_checkpoint(str(path))

    assert metadata["epoch"] == 3
    assert metadata["loss"] == 0.25
    for key, value in model.state_dict().items():
        assert torch.allclose(loaded.state_dict()[key], value)

    try:
        model.save_checkpoint(str(path), 1, 0.1, {}, include_encoders=True)
    except RuntimeError as exc:
        assert "load_extractors=True" in str(exc)
    else:
        raise AssertionError("include_encoders should require extractors")


def test_train_cached_prevents_representation_collapse(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=16, has_ir=True, seed=0)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=12, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)

    train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=5,
        batch_size=4,
        learning_rate=1e-3,
        amp=False,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    data = torch.load(val_cache, map_location="cpu", weights_only=True)
    tiny_model.eval()
    with torch.no_grad():
        z = tiny_model.fusion(data["rgb_features"].float(), data["ir_features"].float(), True, True)
    assert float(z.std(dim=0).mean().item()) > 0.1


def test_train_cached_with_input_noise(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)

    result = train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=1,
        batch_size=2,
        input_noise=0.1,
        amp=False,
        checkpoint_dir=tmp_path / "checkpoints_noise",
    )
    assert result["history"][0]["epoch"] == 1
    assert "val_ratio" in result["history"][0]
    assert "persistence_mse" in result["history"][0]


def test_train_cached_select_metric_val_ratio(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)
    checkpoint_dir = tmp_path / "checkpoints_val_ratio"

    result = train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=2,
        batch_size=2,
        select_metric="val_ratio",
        amp=False,
        checkpoint_dir=checkpoint_dir,
    )
    assert result["select_metric"] == "val_ratio"
    assert torch.isfinite(torch.tensor(result["best_metric"]))
    best_ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location="cpu", weights_only=False)
    assert best_ckpt["extra"]["select_metric"] == "val_ratio"
    assert "best_metric" in best_ckpt["extra"]


def test_train_cached_freeze_fusion_epoch(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)

    train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=2,
        batch_size=2,
        freeze_fusion_epoch=1,
        amp=False,
        checkpoint_dir=tmp_path / "checkpoints_freeze",
    )
    # After epoch 1 (which was the frozen epoch threshold), fusion parameters should be frozen
    assert all(not p.requires_grad for p in tiny_model.fusion.parameters())
    # But temporal parameters should remain trainable
    assert any(p.requires_grad for p in tiny_model.temporal.parameters())


def test_train_cached_patience_early_stopping(tiny_model, make_cache, tmp_path):
    train_cache = make_cache(tmp_path / "cache" / "train" / "a.pt", num_frames=12, has_ir=True)
    val_cache = make_cache(tmp_path / "cache" / "val" / "b.pt", num_frames=10, has_ir=True, seed=1)
    train_dataset = _dataset([train_cache], sequence_length=4)
    val_dataset = _dataset([val_cache], sequence_length=4)

    # Set patience=1 with epochs=10: should stop after a couple epochs if metric does not improve
    result = train_cached(
        tiny_model,
        train_dataset,
        val_dataset,
        device="cpu",
        epochs=10,
        batch_size=2,
        patience=1,
        amp=False,
        checkpoint_dir=tmp_path / "checkpoints_patience",
    )
    # Should stop well before 10 epochs
    assert len(result["history"]) < 10
