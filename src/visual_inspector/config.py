"""Configuration management with Pydantic + YAML and config hashing."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    root: str = "data/mvtec_ad"
    category: str = "bottle"
    image_size: int = 256
    crop_size: int = 224
    val_good_fraction: float = 0.10
    num_workers: int = 4


class BackboneConfig(BaseModel):
    """Feature extractor configuration."""

    name: str = "wide_resnet50_2"
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])
    pretrained: bool = True


class PatchCoreConfig(BaseModel):
    """PatchCore algorithm configuration."""

    d_target: int = 1024
    coreset_ratio: float = 0.10
    projection_dim: int = 128
    k_neighbors: int = 1
    sigma_blur: float = 4.0
    threshold_percentile: float = 99.0


class OutputConfig(BaseModel):
    """Output paths configuration."""

    artifacts_dir: str = "artifacts"
    results_dir: str = "results"


class AppConfig(BaseModel):
    """Top-level application configuration."""

    seed: int = 42
    device: str = "auto"

    data: DataConfig = Field(default_factory=DataConfig)
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    patchcore: PatchCoreConfig = Field(default_factory=PatchCoreConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def config_hash(self) -> str:
        """Compute a deterministic hash of the config for reproducibility tracking."""
        serialized = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:12]

    def resolve_device(self) -> torch.device:
        """Resolve 'auto' device to the best available."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)


def get_env_info() -> dict:
    """Capture environment info for reproducibility."""
    info: dict = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torchvision

        info["torchvision_version"] = torchvision.__version__
    except ImportError:
        info["torchvision_version"] = "not installed"

    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)

    # Try to get git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["git_commit"] = "unknown"

    return info


DeviceType = Literal["auto", "cpu", "cuda", "mps"]
