"""PatchCore model -- faithful implementation from scratch.

Reference: Roth et al., Towards Total Recall in Industrial Anomaly Detection, CVPR 2022.

Pipeline:
1. Extract patch-level features from WRN-50-2 (layer2 + layer3).
2. Local aggregation: AvgPool(3,1,1) on each map, upsample layer3 to layer2 size,
   concatenate, reduce channels to d_target via adaptive average pooling.
3. Coreset: greedy k-center subsampling of training patch embeddings.
4. Scoring: kNN distance, reshape, upsample, Gaussian blur -> anomaly map.
5. Image score = max(anomaly map).
6. Threshold: percentile of val-good image scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset

from visual_inspector.config import AppConfig, get_env_info
from visual_inspector.data.synthetic import SyntheticDataset
from visual_inspector.models.backbone import FeatureExtractor
from visual_inspector.models.coreset import greedy_kcenter
from visual_inspector.models.knn import get_knn_backend

logger = logging.getLogger(__name__)


class PatchCoreModel:
    """PatchCore anomaly detection model.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = config.resolve_device()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone_name=config.backbone.name,
            layers=config.backbone.layers,
            pretrained=config.backbone.pretrained,
        ).to(self.device)

        # kNN backend
        self.knn = get_knn_backend(prefer_faiss=True)

        # State
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float = 0.0
        self._patch_grid_h: int = 0
        self._patch_grid_w: int = 0
        self._is_fitted: bool = False

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and aggregate patch-level features from a batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            Patch embeddings of shape (B*h*w, d_target).
        """
        features = self.feature_extractor(images.to(self.device))
        return self._aggregate_features(features)

    def _aggregate_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate multi-scale features into patch embeddings.

        Steps:
        1. AvgPool2d(3,1,1) on each feature map (local neighborhood aggregation).
        2. Bilinear upsample layer3 to layer2 spatial size.
        3. Concatenate along channel dimension.
        4. Reduce channels to d_target via adaptive average pooling over channels.

        Args:
            features: Dict mapping layer names to feature tensors.

        Returns:
            Patch embeddings of shape (B*H*W, d_target).
        """
        layer_names = self.config.backbone.layers
        assert len(layer_names) == 2, "Expected exactly 2 layers"

        f1 = features[layer_names[0]]  # layer2: (B, C1, H1, W1)
        f2 = features[layer_names[1]]  # layer3: (B, C2, H2, W2)

        # Local aggregation
        avg_pool = torch.nn.AvgPool2d(3, 1, 1)
        f1 = avg_pool(f1)
        f2 = avg_pool(f2)

        # Upsample layer3 to layer2 spatial size
        f2 = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)

        # Concatenate
        concat = torch.cat([f1, f2], dim=1)  # (B, C1+C2, H, W)
        B, C, H, W = concat.shape

        self._patch_grid_h = H
        self._patch_grid_w = W

        # Reduce to d_target via adaptive average pooling over channels
        # Reshape to (B*H*W, C) → (B*H*W, 1, C) → adaptive_avg_pool1d → (B*H*W, d_target)
        d_target = self.config.patchcore.d_target
        patches = concat.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)

        if d_target < C:
            # Adaptive average pooling over the channel dimension
            patches = patches.unsqueeze(1)  # (B*H*W, 1, C)
            patches = F.adaptive_avg_pool1d(patches, d_target)  # (B*H*W, 1, d_target)
            patches = patches.squeeze(1)  # (B*H*W, d_target)
        # If C <= d_target, keep as-is (happens with random weights / small configs)

        return patches

    def fit(
        self, train_loader: DataLoader | None = None, val_loader: DataLoader | None = None
    ) -> None:
        """Fit the PatchCore model.

        Steps:
        1. Extract patch features from all training-good images.
        2. Apply coreset subsampling.
        3. Build kNN index.
        4. Compute threshold from val-good scores.

        Args:
            train_loader: DataLoader for training-good images. If None, created from config.
            val_loader: DataLoader for validation-good images. If None, created from config.
        """
        cfg = self.config

        if train_loader is None or val_loader is None:
            train_loader, val_loader = self._create_data_loaders()

        logger.info("Extracting training features...")
        all_patches = self._collect_patches(train_loader)
        logger.info(f"  Collected {all_patches.shape[0]} patches of dim {all_patches.shape[1]}")

        # Coreset subsampling
        ratio = cfg.patchcore.coreset_ratio
        n_select = max(1, int(all_patches.shape[0] * ratio))
        logger.info(f"  Coreset: selecting {n_select}/{all_patches.shape[0]} (ratio={ratio})")

        if ratio < 1.0:
            indices = greedy_kcenter(
                all_patches,
                n_select,
                projection_dim=cfg.patchcore.projection_dim,
                seed=cfg.seed,
            )
            self.memory_bank = all_patches[indices].cpu()
        else:
            self.memory_bank = all_patches.cpu()

        logger.info(f"  Memory bank: {self.memory_bank.shape}")

        # Build kNN index
        self.knn.fit(self.memory_bank)

        # Compute threshold from val-good scores
        logger.info("Computing threshold from val-good images...")
        val_scores = self._score_loader(val_loader)
        percentile = cfg.patchcore.threshold_percentile
        self.threshold = float(np.percentile(val_scores, percentile))
        logger.info(f"  Threshold (p{percentile}): {self.threshold:.4f}")

        self._is_fitted = True

    def _collect_patches(self, loader: DataLoader) -> torch.Tensor:
        """Collect all patch embeddings from a data loader."""
        all_patches: list[torch.Tensor] = []
        for batch in loader:
            images = batch["image"]
            patches = self._extract_patch_features(images)
            all_patches.append(patches.cpu())
        return torch.cat(all_patches, dim=0)

    def _score_loader(self, loader: DataLoader) -> list[float]:
        """Compute image-level anomaly scores for all images in a loader."""
        scores: list[float] = []
        for batch in loader:
            images = batch["image"]
            batch_scores, _ = self.score(images)
            scores.extend(batch_scores)
        return scores

    def score(self, images: torch.Tensor) -> tuple[list[float], torch.Tensor]:
        """Compute anomaly scores and maps for a batch of images.

        Args:
            images: Tensor of shape (B, 3, H, W).

        Returns:
            image_scores: List of per-image anomaly scores (max of map).
            anomaly_maps: Tensor of shape (B, 1, crop_H, crop_W).
        """
        assert self.memory_bank is not None, "Model must be fitted before scoring"

        patches = self._extract_patch_features(images)
        B = images.shape[0]
        H, W = self._patch_grid_h, self._patch_grid_w
        crop_h, crop_w = self.config.data.crop_size, self.config.data.crop_size

        # kNN distances
        distances, _ = self.knn.query(patches.cpu(), k=self.config.patchcore.k_neighbors)
        # Use mean distance across k neighbors for each patch
        patch_scores = distances.mean(dim=1)  # (B*H*W,)

        # Reshape to spatial grid
        score_map = patch_scores.reshape(B, 1, H, W).float()

        # Upsample to image size
        score_map = F.interpolate(
            score_map, size=(crop_h, crop_w), mode="bilinear", align_corners=False
        )

        # Gaussian blur
        sigma = self.config.patchcore.sigma_blur
        anomaly_maps_np = score_map.squeeze(1).numpy()  # (B, crop_h, crop_w)
        for i in range(B):
            anomaly_maps_np[i] = gaussian_filter(anomaly_maps_np[i], sigma=sigma)

        anomaly_maps = torch.from_numpy(anomaly_maps_np).unsqueeze(1)  # (B, 1, H, W)

        # Image score = max of anomaly map
        image_scores = [float(anomaly_maps[i].max()) for i in range(B)]

        return image_scores, anomaly_maps

    def predict(self, images: torch.Tensor) -> tuple[list[bool], list[float], torch.Tensor]:
        """Predict anomalies (binary) using the fitted threshold.

        Returns:
            predictions: List of bools (True = anomalous).
            scores: List of float anomaly scores.
            anomaly_maps: Tensor of anomaly maps.
        """
        scores, anomaly_maps = self.score(images)
        predictions = [s > self.threshold for s in scores]
        return predictions, scores, anomaly_maps

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Saves memory_bank and metadata as separate files.
        No pickle of arbitrary objects — only tensors and JSON.

        Args:
            path: Directory to save the model in.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save memory bank
        assert self.memory_bank is not None
        torch.save(self.memory_bank, path / "memory_bank.pt", _use_new_zipfile_serialization=True)

        # Save backbone weights only if not pretrained (e.g. for deterministic offline tests)
        if not self.config.backbone.pretrained:
            torch.save(
                self.feature_extractor.backbone.state_dict(),
                path / "backbone.pt",
                _use_new_zipfile_serialization=True,
            )

        # Save metadata
        meta: dict[str, Any] = {
            "config_hash": self.config.config_hash(),
            "threshold": self.threshold,
            "memory_bank_shape": list(self.memory_bank.shape),
            "patch_grid": [self._patch_grid_h, self._patch_grid_w],
            "env": get_env_info(),
            "config": self.config.model_dump(),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, config: AppConfig | None = None) -> PatchCoreModel:
        """Load a saved model.

        Args:
            path: Directory containing saved model files.
            config: Config to use. If None, reconstructed from meta.json.

        Returns:
            Loaded PatchCoreModel.
        """
        path = Path(path)

        # Load metadata
        with open(path / "meta.json") as f:
            meta = json.load(f)

        if config is None:
            config = AppConfig(**meta["config"])

        model = cls(config)

        # Load memory bank (weights_only=True for safety)
        model.memory_bank = torch.load(path / "memory_bank.pt", weights_only=True)
        model.threshold = meta["threshold"]
        model._patch_grid_h = meta["patch_grid"][0]
        model._patch_grid_w = meta["patch_grid"][1]

        # Restore backbone weights if present (for non-pretrained models)
        backbone_path = path / "backbone.pt"
        if backbone_path.exists():
            state_dict = torch.load(backbone_path, weights_only=True)
            model.feature_extractor.backbone.load_state_dict(state_dict)

        # Rebuild kNN index
        model.knn.fit(model.memory_bank)
        model._is_fitted = True

        logger.info(f"Model loaded from {path}")
        return model

    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and val data loaders from config."""
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
                root=cfg.data.root,
                category=cfg.data.category,
                split="train",
                image_size=cfg.data.image_size,
                crop_size=cfg.data.crop_size,
                val_good_fraction=cfg.data.val_good_fraction,
                seed=cfg.seed,
            )
            val_ds = MVTecDataset(
                root=cfg.data.root,
                category=cfg.data.category,
                split="val_good",
                image_size=cfg.data.image_size,
                crop_size=cfg.data.crop_size,
                val_good_fraction=cfg.data.val_good_fraction,
                seed=cfg.seed,
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        return train_loader, val_loader
