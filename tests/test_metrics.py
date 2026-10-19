"""Unit tests for evaluation metrics: Image AUROC, Pixel AUROC, and AU-PRO."""

from __future__ import annotations

import numpy as np

from visual_inspector.eval.metrics import (
    compute_au_pro,
    compute_image_auroc,
    compute_pixel_auroc,
)


class TestImageAUROC:
    def test_separable_case(self) -> None:
        """Perfect separation yields 1.0."""
        labels = [0, 0, 0, 1, 1, 1]
        scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        auroc = compute_image_auroc(labels, scores)
        assert abs(auroc - 1.0) < 1e-6

    def test_inverted_case(self) -> None:
        """Completely inverted scores yield 0.0."""
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]
        auroc = compute_image_auroc(labels, scores)
        assert abs(auroc - 0.0) < 1e-6

    def test_random_case(self) -> None:
        """Random prediction yields approx 0.5."""
        rng = np.random.RandomState(42)
        labels = [0] * 500 + [1] * 500
        scores = rng.rand(1000).tolist()
        auroc = compute_image_auroc(labels, scores)
        assert 0.45 <= auroc <= 0.55

    def test_single_class_edge_case(self) -> None:
        """Single class input doesn't crash and returns 1.0."""
        assert compute_image_auroc([0, 0, 0], [0.1, 0.2, 0.3]) == 1.0


class TestPixelAUROC:
    def test_perfect_localization(self) -> None:
        """Perfect map localization yields 1.0."""
        masks = np.zeros((2, 10, 10))
        masks[1, 3:7, 3:7] = 1.0

        maps = np.zeros((2, 10, 10))
        maps[1, 3:7, 3:7] = 10.0  # high scores inside defect

        auroc = compute_pixel_auroc(masks, maps)
        assert abs(auroc - 1.0) < 1e-6

    def test_random_pixel_maps(self) -> None:
        """Random anomaly maps yield ~0.5."""
        rng = np.random.RandomState(42)
        masks = np.zeros((4, 20, 20))
        masks[2:, 5:15, 5:15] = 1.0

        maps = rng.rand(4, 20, 20)
        auroc = compute_pixel_auroc(masks, maps)
        assert 0.45 <= auroc <= 0.55


class TestAUPRO:
    def test_perfect_overlap_gives_one(self) -> None:
        """When defect region has strictly highest scores, AU-PRO is 1.0."""
        masks = np.zeros((1, 10, 10))
        masks[0, 2:5, 2:5] = 1.0  # 9 defect pixels

        maps = np.zeros((1, 10, 10))
        maps[0, 2:5, 2:5] = 5.0  # perfectly highlights the defect

        au_pro = compute_au_pro(masks, maps, max_fpr=0.30)
        assert abs(au_pro - 1.0) < 1e-4

    def test_hand_computed_tiny_case(self) -> None:
        """Hand-computed 4x4 case with 1 defect component."""
        # 4x4 image, total 16 pixels.
        # Defect component: 2x2 square (4 pixels). Normal pixels: 12.
        masks = np.zeros((1, 4, 4))
        masks[0, 0:2, 0:2] = 1.0

        # Anomaly map: defect has scores [10, 9, 8, 7].
        # 2 normal pixels have scores [6, 5], others have score 0.
        maps = np.zeros((1, 4, 4))
        maps[0, 0, 0] = 10.0
        maps[0, 0, 1] = 9.0
        maps[0, 1, 0] = 8.0
        maps[0, 1, 1] = 7.0
        maps[0, 2, 2] = 6.0  # false positive 1 (1/12 = 0.083 FPR)
        maps[0, 2, 3] = 5.0  # false positive 2 (2/12 = 0.167 FPR)

        # At th=7: all 4 defect pixels detected -> PRO=1.0. FPR=0 (no normal pixels >= 7).
        # So for FPR from 0 to 0.3, PRO is 1.0 throughout!
        au_pro = compute_au_pro(masks, maps, max_fpr=0.30)
        assert abs(au_pro - 1.0) < 1e-3

    def test_no_defects_returns_one(self) -> None:
        """When no anomalous regions exist in test set, returns 1.0."""
        masks = np.zeros((2, 10, 10))
        maps = np.zeros((2, 10, 10))
        assert compute_au_pro(masks, maps) == 1.0
