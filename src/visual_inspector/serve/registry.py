"""Model registry for lazy-loading and caching PatchCore models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from visual_inspector.data.mvtec import MVTEC_CATEGORIES
from visual_inspector.models.patchcore import PatchCoreModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe lazy-loading model registry."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._loaded_models: dict[str, PatchCoreModel] = {}
        self.known_categories: set[str] = set(MVTEC_CATEGORIES) | {"synthetic"}

    def is_known(self, category: str) -> bool:
        """Check if category is a recognized MVTec category."""
        return category in self.known_categories

    def is_trained(self, category: str) -> bool:
        """Check if trained weights and metadata exist on disk for category."""
        cat_dir = self.artifacts_dir / category
        return (cat_dir / "meta.json").exists() and (
            (cat_dir / "memory_bank.pt").exists() or (cat_dir / "model.pt").exists()
        )

    def get_model(self, category: str) -> PatchCoreModel:
        """Get model for category, loading from disk on demand."""
        if not self.is_known(category):
            raise KeyError(
                f"Unknown category '{category}'. Supported: {sorted(self.known_categories)}"
            )

        if category in self._loaded_models:
            return self._loaded_models[category]

        cat_dir = self.artifacts_dir / category
        if not self.is_trained(category):
            raise FileNotFoundError(
                f"Model for category '{category}' is not trained yet. "
                f"Hint: Run 'make train-demo' or 'visual-inspector fit --config configs/smoke.yaml' first."
            )

        logger.info(f"Loading PatchCore model for category '{category}' from {cat_dir}...")
        model = PatchCoreModel.load(cat_dir)
        self._loaded_models[category] = model
        return model

    def register_in_memory(self, category: str, model: PatchCoreModel) -> None:
        """Register an already instantiated model (e.g. for testing or demo)."""
        self._loaded_models[category] = model

    def list_categories(self) -> list[dict[str, Any]]:
        """List all known categories with their trained and loaded status."""
        result = []
        for cat in sorted(self.known_categories):
            result.append(
                {
                    "category": cat,
                    "trained": self.is_trained(cat),
                    "loaded": cat in self._loaded_models,
                }
            )
        return result

    def loaded_category_names(self) -> list[str]:
        """List names of currently loaded categories."""
        return list(self._loaded_models.keys())
