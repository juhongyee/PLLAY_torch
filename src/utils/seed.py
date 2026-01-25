from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """
    Returned state for reproducibility bookkeeping.
    """
    seed: int
    deterministic: bool


def set_seed(seed: int, deterministic: bool = False) -> SeedState:
    """
    Fix random seeds for:
      - Python random
      - NumPy
      - PyTorch (CPU + CUDA)

    Args:
      seed: non-negative int
      deterministic: if True, enforce deterministic algorithms (slower but reproducible)

    Returns:
      SeedState
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed}")

    # Python / OS-level
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        enable_determinism()

    return SeedState(seed=seed, deterministic=deterministic)


def enable_determinism() -> None:
    """
    Best-effort deterministic settings for PyTorch.

    Notes:
      - Some ops may still be nondeterministic depending on hardware / CUDA / cuDNN versions.
      - This can reduce performance.
    """
    # cuDNN determinism (conv/recurrent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch deterministic algorithms (may raise if an op has no deterministic impl)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Older torch versions might not support this or raise for unsupported ops.
        pass

    # CUBLAS deterministic behavior (for some matmul paths)
    # This is recommended by PyTorch for deterministic behavior on CUDA.
    # Needs to be set before CUDA context init ideally, but setting here is still useful.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
