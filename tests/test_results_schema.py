"""JSON schema validation tests for all generated results files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_latency_json_schema() -> None:
    """Validate schema of results/latency.json."""
    latency_file = Path("results/latency.json")
    if not latency_file.exists():
        pytest.skip("results/latency.json not yet generated")

    with open(latency_file) as f:
        data = json.load(f)

    assert data.get("benchmark") == "inference_latency"
    assert "ratios" in data
    assert isinstance(data["ratios"], dict)
    assert len(data["ratios"]) > 0

    for _ratio_key, ratio_data in data["ratios"].items():
        assert "backbone_ms" in ratio_data
        assert "knn_ms" in ratio_data
        assert "postprocess_ms" in ratio_data
        assert "total_ms" in ratio_data

        for stage in ["backbone_ms", "knn_ms", "postprocess_ms", "total_ms"]:
            assert "median" in ratio_data[stage]
            assert "p95" in ratio_data[stage]
            assert isinstance(ratio_data[stage]["median"], (int, float))
            assert isinstance(ratio_data[stage]["p95"], (int, float))
            assert ratio_data[stage]["median"] >= 0

        assert "memory_bank_mb" in ratio_data
        assert isinstance(ratio_data["memory_bank_mb"], (int, float))

    assert "env" in data
    assert "python_version" in data["env"]
    assert "torch_version" in data["env"]
    assert "cpu_model" in data["env"]
    assert "timestamp" in data


def test_evaluation_json_schemas() -> None:
    """Validate schema of all results/{profile}/*.json files."""
    results_dir = Path("results")
    eval_files = list(results_dir.glob("*/*.json"))

    if not eval_files:
        pytest.skip("No evaluation JSON files found in results/*/")

    for file_path in eval_files:
        if file_path.name == "latency.json" or file_path.name == "seeds.json":
            continue

        with open(file_path) as f:
            data = json.load(f)

        assert "category" in data, f"Missing category in {file_path}"
        assert "profile" in data, f"Missing profile in {file_path}"
        assert "metrics" in data, f"Missing metrics in {file_path}"

        metrics = data["metrics"]
        assert "image_auroc" in metrics, f"Missing image_auroc in {file_path}"
        assert isinstance(metrics["image_auroc"], (int, float))
        assert 0.0 <= metrics["image_auroc"] <= 1.0, f"Invalid AUROC in {file_path}"

        assert "threshold" in metrics, f"Missing threshold in {file_path}"
        assert isinstance(metrics["threshold"], (int, float))

        assert "config_hash" in data, f"Missing config_hash in {file_path}"
        assert len(data["config_hash"]) > 0

        assert "env" in data, f"Missing env in {file_path}"
        assert "python_version" in data["env"]
        assert "torch_version" in data["env"]
        assert "timestamp" in data


def test_seeds_json_schema() -> None:
    """Validate schema of results/ablation/seeds.json if present."""
    seeds_file = Path("results/ablation/seeds.json")
    if not seeds_file.exists():
        pytest.skip("results/ablation/seeds.json not found")

    with open(seeds_file) as f:
        data = json.load(f)

    assert "seeds" in data
    assert len(data["seeds"]) >= 2
    assert "image_auroc" in data
    assert "mean" in data["image_auroc"]
    assert "std" in data["image_auroc"]
    assert 0.0 <= data["image_auroc"]["mean"] <= 1.0
