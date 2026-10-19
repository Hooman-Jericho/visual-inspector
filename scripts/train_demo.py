"""Train demo categories and save weights into artifacts/ for API serving."""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from visual_inspector.config import AppConfig
from visual_inspector.data.synthetic import SyntheticDataset
from visual_inspector.models.patchcore import PatchCoreModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_demo_categories(artifacts_root: Path = Path("artifacts")) -> None:
    """Train demo models for 'synthetic' and 'bottle' demo testing."""
    artifacts_root.mkdir(parents=True, exist_ok=True)

    categories = ["synthetic", "bottle"]

    for cat in categories:
        cat_dir = artifacts_root / cat
        logger.info(f"Training demo model for '{cat}' -> {cat_dir}")

        config = AppConfig(seed=42, device="cpu")
        config.data.image_size = 128
        config.data.crop_size = 112
        config.backbone.pretrained = False
        config.patchcore.d_target = 256
        config.patchcore.coreset_ratio = 0.10

        train_ds: Dataset = SyntheticDataset(
            split="train", n_good=20, image_size=128, crop_size=112, seed=42
        )
        val_ds: Dataset = SyntheticDataset(
            split="val_good", n_good=10, image_size=128, crop_size=112, seed=42
        )

        model = PatchCoreModel(config)
        model.fit(DataLoader(train_ds, batch_size=8), DataLoader(val_ds, batch_size=8))

        model.save(cat_dir)
        logger.info(f"Demo model '{cat}' saved successfully to {cat_dir}.")


if __name__ == "__main__":
    train_demo_categories()
