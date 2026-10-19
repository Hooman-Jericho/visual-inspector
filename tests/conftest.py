"""Shared test fixtures."""

from __future__ import annotations

import pytest
import torch

from visual_inspector.config import (
    AppConfig,
    BackboneConfig,
    DataConfig,
    OutputConfig,
    PatchCoreConfig,
)


@pytest.fixture
def smoke_config() -> AppConfig:
    """Config for fast tests: small images, no pretrained weights."""
    return AppConfig(
        seed=42,
        device="cpu",
        data=DataConfig(
            root="__synthetic__",
            category="synthetic",
            image_size=64,
            crop_size=56,
            val_good_fraction=0.10,
            num_workers=0,
        ),
        backbone=BackboneConfig(
            name="wide_resnet50_2",
            layers=["layer2", "layer3"],
            pretrained=False,
        ),
        patchcore=PatchCoreConfig(
            d_target=256,
            coreset_ratio=0.50,
            projection_dim=64,
            k_neighbors=1,
            sigma_blur=2.0,
            threshold_percentile=99.0,
        ),
        output=OutputConfig(artifacts_dir="artifacts", results_dir="results"),
    )


@pytest.fixture
def synthetic_images() -> torch.Tensor:
    """Small batch of random images for shape tests."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 56, 56)
