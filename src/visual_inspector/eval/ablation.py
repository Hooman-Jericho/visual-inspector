"""Ablation experiments: coreset ratio, random seeds, and baselines."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from visual_inspector.config import AppConfig
from visual_inspector.data.synthetic import SyntheticDataset
from visual_inspector.eval.env import get_full_env_info
from visual_inspector.eval.evaluate import evaluate_model
from visual_inspector.models.baseline import BaselineModel
from visual_inspector.models.patchcore import PatchCoreModel

logger = logging.getLogger(__name__)


def run_coreset_ablation(
    config_path: str | Path,
    category: str = "bottle",
    ratios: tuple[float, ...] = (0.01, 0.10, 1.00),
    output_dir: str | Path = "results/ablation",
) -> list[dict[str, Any]]:
    """Evaluate performance across different coreset subsampling ratios."""
    config = AppConfig.from_yaml(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Use synthetic data for repeatable offline ablation
    train_ds: Dataset = SyntheticDataset(
        split="train",
        n_good=30,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        seed=config.seed,
    )
    val_ds: Dataset = SyntheticDataset(
        split="val_good",
        n_good=10,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        seed=config.seed,
    )
    test_ds: Dataset = SyntheticDataset(
        split="test",
        n_good=15,
        n_anomalous=15,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        seed=config.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    for r in ratios:
        cfg = config.model_copy(deep=True)
        cfg.patchcore.coreset_ratio = r

        model = PatchCoreModel(cfg)
        model.fit(train_loader, val_loader)

        assert model.memory_bank is not None
        mem_bytes = model.memory_bank.element_size() * model.memory_bank.nelement()
        mem_mb = round(mem_bytes / (1024 * 1024), 3)

        res = evaluate_model(
            model, test_loader, category=f"{category}_r{r:.2f}", profile="ablation"
        )
        res["metrics"]["memory_bank_mb"] = mem_mb
        res["metrics"]["coreset_ratio"] = r

        out_file = output_dir / f"coreset_r{r:.2f}.json"
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2, default=str)

        results.append(res)

    return results


def run_seeds_evaluation(
    config_path: str | Path,
    seeds: tuple[int, ...] = (42, 123, 999),
    output_dir: str | Path = "results/ablation",
) -> dict[str, Any]:
    """Run evaluation across multiple random seeds to measure stability."""
    config = AppConfig.from_yaml(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_aurocs = []
    pixel_aurocs = []
    au_pros = []

    for seed in seeds:
        cfg = config.model_copy(deep=True)
        cfg.seed = seed

        train_ds: Dataset = SyntheticDataset(split="train", n_good=25, seed=seed)
        val_ds: Dataset = SyntheticDataset(split="val_good", n_good=10, seed=seed)
        test_ds: Dataset = SyntheticDataset(split="test", n_good=10, n_anomalous=10, seed=seed)

        model = PatchCoreModel(cfg)
        model.fit(DataLoader(train_ds, batch_size=8), DataLoader(val_ds, batch_size=8))

        res = evaluate_model(
            model, DataLoader(test_ds, batch_size=8), category="seed_run", profile="seeds"
        )
        image_aurocs.append(res["metrics"]["image_auroc"])
        pixel_aurocs.append(res["metrics"]["pixel_auroc"])
        au_pros.append(res["metrics"]["au_pro"])

    summary = {
        "seeds": list(seeds),
        "image_auroc": {
            "mean": round(float(np.mean(image_aurocs)), 4),
            "std": round(float(np.std(image_aurocs)), 4),
            "values": image_aurocs,
        },
        "pixel_auroc": {
            "mean": round(float(np.mean(pixel_aurocs)), 4),
            "std": round(float(np.std(pixel_aurocs)), 4),
            "values": pixel_aurocs,
        },
        "au_pro": {
            "mean": round(float(np.mean(au_pros)), 4),
            "std": round(float(np.std(au_pros)), 4),
            "values": au_pros,
        },
        "env": get_full_env_info(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(output_dir / "seeds.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_default_profile(
    config_path: str | Path,
    output_dir: str | Path = "results/default",
) -> list[dict[str, Any]]:
    """Run the default profile for PatchCore (r=0.10), no_coreset (r=1.0), and baseline."""
    config = AppConfig.from_yaml(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from visual_inspector.eval.evaluate import PROFILE_CATEGORIES

    categories = PROFILE_CATEGORIES["default"]
    results = []

    for cat in categories:
        cfg = config.model_copy(deep=True)
        cfg.data.category = cat

        # Generate dataset
        train_ds: Dataset = SyntheticDataset(split="train", n_good=25, seed=cfg.seed)
        val_ds: Dataset = SyntheticDataset(split="val_good", n_good=10, seed=cfg.seed)
        test_ds: Dataset = SyntheticDataset(split="test", n_good=15, n_anomalous=15, seed=cfg.seed)

        train_l = DataLoader(train_ds, batch_size=8)
        val_l = DataLoader(val_ds, batch_size=8)
        test_l = DataLoader(test_ds, batch_size=8)

        # 1. Standard PatchCore (r=0.10)
        p_model = PatchCoreModel(cfg)
        p_model.fit(train_l, val_l)
        res_p = evaluate_model(
            p_model, test_l, category=cat, profile="default", output_dir=output_dir
        )
        results.append(res_p)

        # 2. PatchCore no coreset (r=1.00)
        cfg_no_core = cfg.model_copy(deep=True)
        cfg_no_core.patchcore.coreset_ratio = 1.00
        nc_model = PatchCoreModel(cfg_no_core)
        nc_model.fit(train_l, val_l)
        res_nc = evaluate_model(
            nc_model,
            test_l,
            category=f"{cat}_no_coreset",
            profile="default",
            output_dir=output_dir,
        )
        results.append(res_nc)

        # 3. Baseline global kNN
        b_model = BaselineModel(cfg)
        b_model.fit(train_l, val_l)
        res_b = evaluate_model(
            b_model,
            test_l,
            category=f"{cat}_baseline",
            profile="default",
            output_dir=output_dir,
        )
        results.append(res_b)

    return results
