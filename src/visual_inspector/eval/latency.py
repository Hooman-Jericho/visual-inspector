"""Latency benchmarking for PatchCore components: backbone, kNN, postprocessing."""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from visual_inspector.config import AppConfig
from visual_inspector.eval.env import get_full_env_info
from visual_inspector.models.patchcore import PatchCoreModel

logger = logging.getLogger(__name__)


def measure_patchcore_latency(
    model: PatchCoreModel,
    n_warmup: int = 20,
    n_runs: int = 200,
    input_size: int = 224,
) -> dict[str, Any]:
    """Measure inference latency with component breakdown.

    Batch size 1, fixed dummy input.

    Args:
        model: Fitted PatchCoreModel with memory bank.
        n_warmup: Number of warmup runs.
        n_runs: Number of timed runs.
        input_size: Image resolution (default: 224).

    Returns:
        Latency summary dict with median and p95 for each stage.
    """
    assert model.memory_bank is not None, "Model must be fitted"

    dummy_input = torch.randn(1, 3, input_size, input_size).to(model.device)

    # 1. Warmup
    for _ in range(n_warmup):
        _ = model.score(dummy_input)

    backbone_times: list[float] = []
    knn_times: list[float] = []
    postprocess_times: list[float] = []
    total_times: list[float] = []

    B = 1
    H, W = model._patch_grid_h, model._patch_grid_w
    crop_h = crop_w = model.config.data.crop_size
    sigma = model.config.patchcore.sigma_blur

    for _ in range(n_runs):
        t0 = time.perf_counter()

        # Step A: Backbone & Aggregation
        t_a0 = time.perf_counter()
        patches = model._extract_patch_features(dummy_input)
        t_a1 = time.perf_counter()

        # Step B: kNN query
        t_b0 = time.perf_counter()
        distances, _ = model.knn.query(patches.cpu(), k=model.config.patchcore.k_neighbors)
        t_b1 = time.perf_counter()

        # Step C: Post-process (spatial reshape, upsample, Gaussian blur, max)
        t_c0 = time.perf_counter()
        patch_scores = distances.mean(dim=1)
        score_map = patch_scores.reshape(B, 1, H, W).float()
        score_map = F.interpolate(
            score_map, size=(crop_h, crop_w), mode="bilinear", align_corners=False
        )
        anomaly_maps_np = score_map.squeeze(1).numpy()
        for i in range(B):
            anomaly_maps_np[i] = gaussian_filter(anomaly_maps_np[i], sigma=sigma)
        _ = float(anomaly_maps_np.max())
        t_c1 = time.perf_counter()

        t_end = time.perf_counter()

        backbone_times.append((t_a1 - t_a0) * 1000.0)
        knn_times.append((t_b1 - t_b0) * 1000.0)
        postprocess_times.append((t_c1 - t_c0) * 1000.0)
        total_times.append((t_end - t0) * 1000.0)

    return {
        "n_runs": n_runs,
        "n_warmup": n_warmup,
        "batch_size": 1,
        "memory_bank_size": list(model.memory_bank.shape),
        "backbone_ms": {
            "median": round(float(np.median(backbone_times)), 2),
            "p95": round(float(np.percentile(backbone_times, 95)), 2),
        },
        "knn_ms": {
            "median": round(float(np.median(knn_times)), 2),
            "p95": round(float(np.percentile(knn_times, 95)), 2),
        },
        "postprocess_ms": {
            "median": round(float(np.median(postprocess_times)), 2),
            "p95": round(float(np.percentile(postprocess_times, 95)), 2),
        },
        "total_ms": {
            "median": round(float(np.median(total_times)), 2),
            "p95": round(float(np.percentile(total_times, 95)), 2),
        },
    }


def run_latency_benchmark(
    config_path: str | Path,
    output_path: str | Path = "results/latency.json",
    coreset_ratios: tuple[float, ...] = (0.01, 0.10, 1.00),
    n_warmup: int = 20,
    n_runs: int = 200,
) -> dict[str, Any]:
    """Benchmark inference latency across multiple coreset ratios.

    Args:
        config_path: Path to configuration YAML.
        output_path: Path to save the latency JSON results.
        coreset_ratios: Coreset ratios to evaluate.
        n_warmup: Warmup iterations.
        n_runs: Timed iterations.

    Returns:
        Latency benchmark results dictionary.
    """
    config = AppConfig.from_yaml(config_path)

    # Synthetic dataset for building realistic memory bank size
    from torch.utils.data import DataLoader

    from visual_inspector.data.synthetic import SyntheticDataset

    train_ds = SyntheticDataset(
        split="train",
        n_good=40,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        seed=config.seed,
    )
    val_ds = SyntheticDataset(
        split="val_good",
        n_good=10,
        image_size=config.data.image_size,
        crop_size=config.data.crop_size,
        seed=config.seed,
    )

    ratio_results: dict[str, Any] = {}

    for ratio in coreset_ratios:
        cfg = config.model_copy(deep=True)
        cfg.patchcore.coreset_ratio = ratio

        model = PatchCoreModel(cfg)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
        model.fit(train_loader, val_loader)

        # Compute memory bank size in MB
        assert model.memory_bank is not None
        mem_bytes = model.memory_bank.element_size() * model.memory_bank.nelement()
        mem_mb = round(mem_bytes / (1024 * 1024), 3)

        latency_data = measure_patchcore_latency(
            model,
            n_warmup=n_warmup,
            n_runs=n_runs,
            input_size=config.data.crop_size,
        )
        latency_data["memory_bank_mb"] = mem_mb
        ratio_results[f"ratio_{ratio}"] = latency_data

    results = {
        "benchmark": "inference_latency",
        "ratios": ratio_results,
        "env": get_full_env_info(),
        "config_hash": config.config_hash(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Latency benchmark saved to {out_p}")
    return results
