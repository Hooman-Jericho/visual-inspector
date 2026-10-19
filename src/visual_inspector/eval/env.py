"""Environment and hardware capture for reproducible evaluations."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

import torch
import torchvision


def get_cpu_info() -> str:
    """Get CPU model name across platforms."""
    if platform.system() == "Windows":
        cpu = os.environ.get("PROCESSOR_IDENTIFIER")
        if cpu:
            return cpu.strip()
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return commit
    except Exception:
        return "unknown"


def is_faiss_available() -> bool:
    """Check if faiss-cpu is available and working."""
    try:
        import faiss  # noqa: F401

        return True
    except ImportError:
        return False


def get_full_env_info() -> dict[str, Any]:
    """Capture comprehensive environment details."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu_model": get_cpu_info(),
        "cpu_threads": os.cpu_count() or 1,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "faiss_available": is_faiss_available(),
        "git_commit": get_git_commit(),
    }
