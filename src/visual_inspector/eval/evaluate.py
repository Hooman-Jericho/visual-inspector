"""Evaluation pipeline for PatchCore and baselines on MVTec AD."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from visual_inspector.config import AppConfig
from visual_inspector.data.mvtec import MVTecDataset
from visual_inspector.data.synthetic import SyntheticDataset
from visual_inspector.eval.env import get_full_env_info
from visual_inspector.eval.metrics import (
    compute_au_pro,
    compute_image_auroc,
    compute_pixel_auroc,
)
from visual_inspector.models.baseline import BaselineModel
from visual_inspector.models.patchcore import PatchCoreModel

logger = logging.getLogger(__name__)

PROFILE_CATEGORIES = {
    "smoke": ["bottle", "hazelnut", "carpet"],
    "default": ["bottle", "carpet", "hazelnut", "screw", "transistor"],
    "full": [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ],
}


def evaluate_model(
    model: PatchCoreModel | BaselineModel,
    test_loader: DataLoader,
    category: str,
    profile: str,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted model on test dataset.

    Args:
        model: Fitted PatchCore or Baseline model.
        test_loader: DataLoader yielding test images, labels, and masks.
        category: Name of the evaluated category.
        profile: Evaluation profile name.
        output_dir: Directory to save the resulting JSON file.

    Returns:
        Dictionary of evaluation results.
    """
    all_labels: list[int] = []
    all_image_scores: list[float] = []
    all_masks: list[np.ndarray] = []
    all_anomaly_maps: list[np.ndarray] = []

    is_patchcore = isinstance(model, PatchCoreModel)

    for batch in test_loader:
        images = batch["image"]
        labels = (
            batch["label"].tolist() if isinstance(batch["label"], torch.Tensor) else batch["label"]
        )
        masks = batch["mask"].cpu().numpy()

        all_labels.extend(labels)
        all_masks.extend(masks)

        if isinstance(model, PatchCoreModel):
            pc_scores, maps = model.score(images)
            all_image_scores.extend(pc_scores)
            all_anomaly_maps.extend(maps.cpu().numpy())
        else:
            base_scores = model.score(images)
            all_image_scores.extend(base_scores)

    # Compute metrics
    image_auroc = compute_image_auroc(all_labels, all_image_scores)

    pixel_auroc = 0.0
    au_pro = 0.0
    if is_patchcore and all_anomaly_maps:
        masks_arr = np.array(all_masks)
        maps_arr = np.array(all_anomaly_maps)
        pixel_auroc = compute_pixel_auroc(masks_arr, maps_arr)
        au_pro = compute_au_pro(masks_arr, maps_arr, max_fpr=0.30)

    result_data: dict[str, Any] = {
        "category": category,
        "profile": profile,
        "model_type": "patchcore" if is_patchcore else "baseline",
        "metrics": {
            "image_auroc": round(image_auroc, 4),
            "pixel_auroc": round(pixel_auroc, 4) if is_patchcore else None,
            "au_pro": round(au_pro, 4) if is_patchcore else None,
            "threshold": round(float(model.threshold), 4),
            "n_test": len(all_labels),
            "n_good": sum(1 for lbl in all_labels if lbl == 0),
            "n_anomalous": sum(1 for lbl in all_labels if lbl == 1),
        },
        "config": model.config.model_dump(),
        "config_hash": model.config.config_hash(),
        "seed": model.config.seed,
        "env": get_full_env_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{category}.json"
        with open(out_file, "w") as f:
            json.dump(result_data, f, indent=2, default=str)
        logger.info(f"Saved evaluation result to {out_file}")

    return result_data


def run_evaluation(
    config_path: str | Path,
    profile: str = "smoke",
    category: str | None = None,
    max_train: int | None = None,
    use_synthetic: bool = False,
) -> list[dict[str, Any]]:
    """Run evaluation over a category profile.

    Args:
        config_path: Path to YAML config.
        profile: Profile name (smoke, default, full).
        category: Single category to evaluate, or None for all categories in profile.
        max_train: Maximum training images to use (for fast smoke test).
        use_synthetic: Force use of synthetic dataset if MVTec is absent.

    Returns:
        List of result dictionaries.
    """
    config = AppConfig.from_yaml(config_path)
    categories = [category] if category else PROFILE_CATEGORIES.get(profile, ["bottle"])
    output_dir = Path(config.output.results_dir) / profile

    results = []

    for cat in categories:
        cat_config = config.model_copy(deep=True)
        cat_config.data.category = cat

        # Determine data sources
        data_root = Path(cat_config.data.root)
        cat_dir = data_root / cat

        if not cat_dir.exists() or use_synthetic:
            # Fallback to synthetic dataset if MVTec is missing
            logger.warning(
                f"MVTec directory for '{cat}' not found at {cat_dir}. Using synthetic dataset for evaluation pipeline."
            )
            train_ds: Dataset = SyntheticDataset(
                split="train",
                n_good=max_train or 20,
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                seed=cat_config.seed,
            )
            val_ds: Dataset = SyntheticDataset(
                split="val_good",
                n_good=max_train or 20,
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                seed=cat_config.seed,
            )
            test_ds: Dataset = SyntheticDataset(
                split="test",
                n_good=10,
                n_anomalous=10,
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                seed=cat_config.seed,
            )
        else:
            train_ds = MVTecDataset(
                root=data_root,
                category=cat,
                split="train",
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                val_good_fraction=cat_config.data.val_good_fraction,
                seed=cat_config.seed,
            )
            val_ds = MVTecDataset(
                root=data_root,
                category=cat,
                split="val_good",
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                val_good_fraction=cat_config.data.val_good_fraction,
                seed=cat_config.seed,
            )
            test_ds = MVTecDataset(
                root=data_root,
                category=cat,
                split="test",
                image_size=cat_config.data.image_size,
                crop_size=cat_config.data.crop_size,
                seed=cat_config.seed,
            )

        # Slice train if max_train is set
        if (
            max_train is not None
            and hasattr(train_ds, "samples")
            and len(train_ds.samples) > max_train
        ):  # type: ignore[attr-defined]
            train_ds.samples = train_ds.samples[:max_train]  # type: ignore[attr-defined]

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

        # Train model
        model = PatchCoreModel(cat_config)
        model.fit(train_loader, val_loader)

        # Evaluate
        res = evaluate_model(
            model, test_loader, category=cat, profile=profile, output_dir=output_dir
        )
        results.append(res)

    return results
