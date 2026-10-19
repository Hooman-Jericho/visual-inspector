"""Baseline model: global average pooled WRN-50-2 + kNN (image-level only).

A simpler baseline for comparison: uses the same backbone but pools
features globally instead of using patch-level embeddings. This gives
image-level anomaly detection without localization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from visual_inspector.config import AppConfig, get_env_info
from visual_inspector.data.synthetic import SyntheticDataset
from visual_inspector.models.backbone import FeatureExtractor
from visual_inspector.models.knn import get_knn_backend

logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline: global-average-pooled WRN-50-2 (layer3) + kNN.

    Image-level anomaly detection only (no localization maps).

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = config.resolve_device()

        self.feature_extractor = FeatureExtractor(
            backbone_name=config.backbone.name,
            layers=["layer3"],  # Only layer3 for baseline
            pretrained=config.backbone.pretrained,
        ).to(self.device)

        self.knn = get_knn_backend(prefer_faiss=True)
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float = 0.0
        self._is_fitted: bool = False

    def _extract_global_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global features via global average pooling.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            Global features of shape (B, C).
        """
        features: dict[str, torch.Tensor] = self.feature_extractor(images.to(self.device))  # type: ignore[assignment]
        f3 = features["layer3"]  # (B, C, H, W)
        # Global average pooling
        global_feat: torch.Tensor = f3.mean(dim=[2, 3])  # (B, C)
        return global_feat

    def fit(
        self, train_loader: DataLoader | None = None, val_loader: DataLoader | None = None
    ) -> None:
        """Fit the baseline model."""
        if train_loader is None or val_loader is None:
            train_loader, val_loader = self._create_data_loaders()

        logger.info("Baseline: extracting training features...")
        all_features: list[torch.Tensor] = []
        for batch in train_loader:
            feats = self._extract_global_features(batch["image"])
            all_features.append(feats.cpu())

        self.memory_bank = torch.cat(all_features, dim=0)
        logger.info(f"  Memory bank: {self.memory_bank.shape}")

        self.knn.fit(self.memory_bank)

        # Threshold from val-good
        val_scores: list[float] = []
        for batch in val_loader:
            scores = self.score(batch["image"])
            val_scores.extend(scores)

        percentile = self.config.patchcore.threshold_percentile
        self.threshold = float(np.percentile(val_scores, percentile))
        self._is_fitted = True
        logger.info(f"  Threshold: {self.threshold:.4f}")

    def score(self, images: torch.Tensor) -> list[float]:
        """Compute image-level anomaly scores."""
        assert self.memory_bank is not None
        feats = self._extract_global_features(images)
        distances, _ = self.knn.query(feats.cpu(), k=self.config.patchcore.k_neighbors)
        return distances.mean(dim=1).tolist()

    def predict(self, images: torch.Tensor) -> tuple[list[bool], list[float]]:
        """Predict anomalies using fitted threshold."""
        scores = self.score(images)
        predictions = [s > self.threshold for s in scores]
        return predictions, scores

    def save(self, path: str | Path) -> None:
        """Save baseline model."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.memory_bank is not None
        torch.save(self.memory_bank, path / "memory_bank.pt")
        if not self.config.backbone.pretrained:
            torch.save(
                self.feature_extractor.backbone.state_dict(),
                path / "backbone.pt",
            )
        meta: dict[str, Any] = {
            "model_type": "baseline",
            "config_hash": self.config.config_hash(),
            "threshold": self.threshold,
            "memory_bank_shape": list(self.memory_bank.shape),
            "env": get_env_info(),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path, config: AppConfig) -> BaselineModel:
        """Load a saved baseline model."""
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        model = cls(config)
        model.memory_bank = torch.load(path / "memory_bank.pt", weights_only=True)
        backbone_path = path / "backbone.pt"
        if backbone_path.exists():
            state_dict = torch.load(backbone_path, weights_only=True)
            model.feature_extractor.backbone.load_state_dict(state_dict)
        model.threshold = meta["threshold"]
        model.knn.fit(model.memory_bank)
        model._is_fitted = True
        return model

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and val data loaders."""
        cfg = self.config
        train_ds: Dataset
        val_ds: Dataset
        if cfg.data.root == "__synthetic__":
            train_ds = SyntheticDataset(
                split="train",
                image_size=cfg.data.image_size,
                crop_size=cfg.data.crop_size,
                seed=cfg.seed,
            )
            val_ds = SyntheticDataset(
                split="val_good",
                image_size=cfg.data.image_size,
                crop_size=cfg.data.crop_size,
                seed=cfg.seed,
            )
        else:
            from visual_inspector.data.mvtec import MVTecDataset

            train_ds = MVTecDataset(
                root=cfg.data.root, category=cfg.data.category, split="train", seed=cfg.seed
            )
            val_ds = MVTecDataset(
                root=cfg.data.root, category=cfg.data.category, split="val_good", seed=cfg.seed
            )
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        return train_loader, val_loader
