"""Tests for data modules: synthetic dataset and MVTec loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from visual_inspector.data.mvtec import MVTecDataset, check_mvtec_data, get_mvtec_transforms
from visual_inspector.data.synthetic import SyntheticDataset, save_synthetic_dataset

# ──────────────────────────── Synthetic Dataset ────────────────────────────


class TestSyntheticDataset:
    def test_train_split_length(self) -> None:
        ds = SyntheticDataset(split="train", n_good=20)
        assert len(ds) == 20

    def test_val_split_length(self) -> None:
        ds = SyntheticDataset(split="val_good", n_good=20)
        assert len(ds) == max(2, 20 // 5)  # 4

    def test_test_split_length(self) -> None:
        ds = SyntheticDataset(split="test", n_good=10, n_anomalous=6)
        assert len(ds) == 10 + 6

    def test_train_all_good(self) -> None:
        ds = SyntheticDataset(split="train", n_good=10)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["label"] == 0

    def test_test_has_anomalies(self) -> None:
        ds = SyntheticDataset(split="test", n_good=5, n_anomalous=5)
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert sum(labels) == 5

    def test_image_shape(self) -> None:
        ds = SyntheticDataset(split="train", image_size=64, crop_size=56)
        sample = ds[0]
        assert sample["image"].shape == (3, 56, 56)

    def test_mask_shape(self) -> None:
        ds = SyntheticDataset(split="test", image_size=64, crop_size=56)
        sample = ds[0]
        assert sample["mask"].shape == (1, 56, 56)

    def test_mask_binary(self) -> None:
        ds = SyntheticDataset(split="test", n_good=5, n_anomalous=5)
        for i in range(len(ds)):
            mask = ds[i]["mask"]
            unique = torch.unique(mask)
            assert all(v in [0.0, 1.0] for v in unique.tolist())

    def test_determinism(self) -> None:
        ds1 = SyntheticDataset(split="train", seed=123)
        ds2 = SyntheticDataset(split="train", seed=123)
        assert torch.allclose(ds1[0]["image"], ds2[0]["image"])

    def test_different_seeds_differ(self) -> None:
        ds1 = SyntheticDataset(split="train", seed=1)
        ds2 = SyntheticDataset(split="train", seed=2)
        assert not torch.allclose(ds1[0]["image"], ds2[0]["image"])


# ──────────────────────────── Synthetic → Disk ────────────────────────────


class TestSaveSyntheticDataset:
    def test_save_creates_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_synthetic_dataset(tmpdir, n_good=5, n_anomalous=4, image_size=32)
            root = Path(tmpdir) / "synthetic"
            assert (root / "train" / "good").exists()
            assert len(list((root / "train" / "good").glob("*.png"))) == 5
            assert (root / "test" / "good").exists()
            assert (root / "test" / "scratch").exists()
            assert (root / "test" / "blob").exists()
            assert (root / "ground_truth" / "scratch").exists()
            assert (root / "ground_truth" / "blob").exists()


# ──────────────────────────── MVTec Loader (on disk synthetic) ────────────


class TestMVTecDatasetOnSynthetic:
    """Test MVTecDataset by loading saved synthetic data in MVTec format."""

    @pytest.fixture
    def synthetic_root(self, tmp_path: Path) -> Path:
        save_synthetic_dataset(tmp_path, n_good=10, n_anomalous=6, image_size=64)
        return tmp_path

    def test_train_split(self, synthetic_root: Path) -> None:
        ds = MVTecDataset(
            root=synthetic_root,
            category="synthetic",
            split="train",
            image_size=64,
            crop_size=56,
        )
        assert len(ds) > 0
        assert all(ds[i]["label"] == 0 for i in range(len(ds)))

    def test_val_good_split(self, synthetic_root: Path) -> None:
        ds = MVTecDataset(
            root=synthetic_root,
            category="synthetic",
            split="val_good",
            image_size=64,
            crop_size=56,
        )
        assert len(ds) > 0
        assert all(ds[i]["label"] == 0 for i in range(len(ds)))

    def test_train_val_disjoint(self, synthetic_root: Path) -> None:
        train_ds = MVTecDataset(root=synthetic_root, category="synthetic", split="train")
        val_ds = MVTecDataset(root=synthetic_root, category="synthetic", split="val_good")
        train_paths = {s["path"] for s in [train_ds[i] for i in range(len(train_ds))]}
        val_paths = {s["path"] for s in [val_ds[i] for i in range(len(val_ds))]}
        assert train_paths.isdisjoint(val_paths)

    def test_test_split_has_labels(self, synthetic_root: Path) -> None:
        ds = MVTecDataset(
            root=synthetic_root,
            category="synthetic",
            split="test",
            image_size=64,
            crop_size=56,
        )
        assert len(ds) > 0
        labels = [ds[i]["label"] for i in range(len(ds))]
        assert 0 in labels
        assert 1 in labels

    def test_test_masks_correct_shape(self, synthetic_root: Path) -> None:
        ds = MVTecDataset(
            root=synthetic_root,
            category="synthetic",
            split="test",
            image_size=64,
            crop_size=56,
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["mask"].shape == (1, 56, 56)


# ──────────────────────────── Transforms ────────────────────────────


class TestTransforms:
    def test_image_transform_output_shape(self) -> None:
        t = get_mvtec_transforms(image_size=256, crop_size=224, is_mask=False)
        from PIL import Image

        img = Image.new("RGB", (300, 300), (128, 128, 128))
        out = t(img)
        assert out.shape == (3, 224, 224)

    def test_mask_transform_output_shape(self) -> None:
        t = get_mvtec_transforms(image_size=256, crop_size=224, is_mask=True)
        from PIL import Image

        mask = Image.new("L", (300, 300), 0)
        out = t(mask)
        assert out.shape == (1, 224, 224)


# ──────────────────────────── Data Check CLI ────────────────────────────


class TestDataCheck:
    def test_data_check_missing(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        missing_path = tmp_path / "nonexistent"
        check_mvtec_data(missing_path)
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "❌" in captured.out

    def test_data_check_with_synthetic(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        # Save synthetic data to look like MVTec
        save_synthetic_dataset(tmp_path, n_good=5, n_anomalous=4, image_size=32)
        # The synthetic category won't match MVTEC_CATEGORIES, but should not crash
        check_mvtec_data(tmp_path)
        captured = capsys.readouterr()
        # Should report that standard categories are missing
        assert "NOT FOUND" in captured.out or "Missing" in captured.out or "0/" in captured.out
