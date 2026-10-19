"""Tests for visualization and heatmap overlay utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from visual_inspector.viz.heatmap import (
    apply_jet_colormap,
    create_comparison_image,
    create_heatmap_overlay,
)


class TestHeatmapViz:
    """Test suite for visualization generation."""

    def test_apply_jet_colormap(self) -> None:
        """Test Jet colormap mapping on a 2D float array in [0, 1]."""
        arr = np.linspace(0.0, 1.0, 100).reshape(10, 10).astype(np.float32)
        rgb = apply_jet_colormap(arr)
        assert rgb.shape == (10, 10, 3)
        assert rgb.dtype == np.uint8
        # Check boundary values
        assert rgb[0, 0, 2] > 100  # Blue at zero
        assert rgb[-1, -1, 0] > 100  # Red at one

    def test_create_heatmap_overlay_numpy(self) -> None:
        """Test overlay generation from numpy image and map."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        anom_map = np.random.rand(64, 64).astype(np.float32)
        overlay = create_heatmap_overlay(img, anom_map, alpha=0.5)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (64, 64)

    def test_create_heatmap_overlay_torch_and_pil(self) -> None:
        """Test overlay generation from PyTorch tensor and PIL image."""
        pil_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        tensor_map = torch.rand(1, 16, 16)
        overlay = create_heatmap_overlay(pil_img, tensor_map, alpha=0.4)
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (32, 32)

    def test_create_comparison_image_with_mask(self) -> None:
        """Test 3-panel comparison image generation with mask."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        anom_map = np.zeros((32, 32), dtype=np.float32)
        anom_map[12:18, 12:18] = 0.9

        canvas = create_comparison_image(
            image=img,
            mask=mask,
            anomaly_map=anom_map,
            score=0.85,
            threshold=0.50,
            label="test_defect",
        )
        assert isinstance(canvas, Image.Image)
        # Expected width: 32*3 + 4*4 = 112, height: 32 + 40 + 8 = 80
        assert canvas.width > 96
        assert canvas.height > 32

    def test_create_comparison_image_without_mask(self) -> None:
        """Test 3-panel comparison image without ground truth mask."""
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        anom_map = np.zeros((32, 32), dtype=np.float32)
        canvas = create_comparison_image(image=img, mask=None, anomaly_map=anom_map)
        assert isinstance(canvas, Image.Image)

    def test_unsupported_image_type_raises(self) -> None:
        """Test that invalid image types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            create_heatmap_overlay("invalid_type", np.zeros((10, 10)))  # type: ignore[arg-type]
