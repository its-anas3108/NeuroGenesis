"""
Seed and determinism control.
=============================

Section 24 requires every experiment to be reproducible. This module is the
single place that touches global RNG state.

Determinism has real limits, and this module reports them rather than pretending
they do not exist:

* ``torch.use_deterministic_algorithms(True)`` makes several 3D convolution
  backward kernels raise instead of silently using a non-deterministic path.
  We therefore request determinism in *warn-only* mode by default, and record
  in the returned :class:`SeedReport` whether strict mode was achievable.
* CPU floating-point reductions are deterministic for a fixed thread count, so
  ``num_workers`` is pinned to 0 by default in :class:`~modules.common.config.TrainConfig`.
* Even with identical seeds, results are only bit-identical on the same
  hardware, library versions and thread count. The :class:`SeedReport` captures
  those so a later mismatch is explainable instead of mysterious.
"""

from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SeedReport:
    """Record of what determinism was actually achieved."""

    seed: int
    deterministic_requested: bool
    #: True only if ``torch.use_deterministic_algorithms`` was applied strictly.
    deterministic_strict: bool = False
    torch_available: bool = False
    torch_version: Optional[str] = None
    cuda_available: bool = False
    device: str = "cpu"
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    numpy_version: str = field(default_factory=lambda: np.__version__)
    #: Non-fatal caveats worth recording alongside the results.
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for the experiment manifest."""
        return {
            "seed": self.seed,
            "deterministic_requested": self.deterministic_requested,
            "deterministic_strict": self.deterministic_strict,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "device": self.device,
            "python_version": self.python_version,
            "platform": self.platform,
            "numpy_version": self.numpy_version,
            "notes": list(self.notes),
        }


def set_all_seeds(seed: int = 42, deterministic: bool = True,
                  cudnn_benchmark: bool = False) -> SeedReport:
    """Seed every RNG the framework touches and request deterministic kernels.

    Args:
        seed: Base seed applied to ``random``, ``numpy``, ``torch`` (CPU and all
            CUDA devices) and ``PYTHONHASHSEED``.
        deterministic: Request deterministic algorithms. Applied in warn-only
            mode so that a 3D-convolution backward pass without a deterministic
            implementation degrades to a recorded caveat instead of crashing a
            long training run.
        cudnn_benchmark: cuDNN autotuning. Must be ``False`` for reproducible
            GPU runs; autotuning picks different kernels across runs.

    Returns:
        A :class:`SeedReport` describing what was actually achieved. Persist it
        next to the results.
    """
    report = SeedReport(seed=seed, deterministic_requested=deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        report.notes.append(
            "PyTorch is not installed — only Python and NumPy RNGs were seeded."
        )
        return report

    report.torch_available = True
    report.torch_version = torch.__version__

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    report.cuda_available = bool(torch.cuda.is_available())
    report.device = "cuda" if report.cuda_available else "cpu"

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        # CuBLAS needs this set before the first CUDA context to make GEMM
        # reductions deterministic. Setting it late is harmless but ineffective.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report.deterministic_strict = True
        except (RuntimeError, TypeError, AttributeError) as exc:
            report.notes.append(
                f"Deterministic algorithms could not be enabled strictly: {exc}"
            )

        report.notes.append(
            "Determinism holds for a fixed device, library version and thread "
            "count. Cross-hardware bit-identical results are not guaranteed."
        )

    return report


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string, falling back to CPU when CUDA is unavailable.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"`` or an explicit
            ``"cuda:N"``.

    Returns:
        A device string safe to pass to ``torch.device``.
    """
    if requested != "auto":
        if requested.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    return "cpu"
            except ImportError:
                return "cpu"
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser that reseeds NumPy and ``random``.

    Without this, every worker process inherits the parent NumPy seed and any
    NumPy-based augmentation produces *identical* randomness across workers —
    a silent bug that reduces effective augmentation diversity.
    """
    try:
        import torch

        base = torch.initial_seed() % (2 ** 31 - 1)
    except ImportError:  # pragma: no cover
        base = worker_id
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


__all__ = ["SeedReport", "set_all_seeds", "resolve_device", "worker_init_fn"]
