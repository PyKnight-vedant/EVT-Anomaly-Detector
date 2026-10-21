"""Step 18: Reproducibility helpers — seeding, device selection, run manifests."""
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms where available (may slow things down slightly).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def log_manifest(results_dir: str, extra: dict) -> str:
    """Write hardware/software versions + hyperparameters to a JSON manifest."""
    manifest = {
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy_version": np.__version__,
        "git_commit": _git_commit(),
        **extra,
    }
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_manifest_{manifest['timestamp']}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
