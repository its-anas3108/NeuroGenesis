"""
Output directory layout (Section 25).
=====================================

One function creates the whole ``outputs/`` tree and returns a name -> Path map,
so no module has to build paths by string concatenation. Every artifact the
framework writes goes to a key defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

#: Logical artifact name -> path relative to the outputs root.
OUTPUT_LAYOUT: Dict[str, str] = {
    # Per-subject pipeline artifacts
    "patient": "patient",
    "preprocessing": "preprocessing",
    "roi": "roi",
    "roi_patches": "roi/patches",
    "roi_masks": "roi/masks",
    "cnn_embeddings": "cnn_embeddings",
    "graph": "graph",
    # NeuroProp-X
    "neuropropx": "neuropropx",
    "vulnerability": "neuropropx/vulnerability",
    "adaptive_adjacency": "neuropropx/adaptive_adjacency",
    "propagation": "neuropropx/propagation",
    "enriched_graph": "neuropropx/enriched_graph",
    # Model outputs
    "predictions": "predictions",
    "stage_tgt": "stage_tgt",
    "roi_ranking": "roi_ranking",
    "xai": "xai",
    # Analysis
    "statistics": "statistics",
    "ablation": "ablation",
    "baselines": "baselines",
    "tables": "tables",
    "figures": "figures",
    "reports": "reports",
    # Bookkeeping
    "experiments": "experiments",
    "splits": "splits",
    "checkpoints": "checkpoints",
    "scalers": "scalers",
    "logs": "logs",
    "state": "state",
}


def create_output_dirs(outputs_root: Path, create: bool = True) -> Dict[str, Path]:
    """Build (and optionally create) the full output directory tree.

    Args:
        outputs_root: Root ``outputs/`` directory.
        create: When ``True``, create every directory. Pass ``False`` from
            read-only consumers such as the dashboard, which must not create
            empty directories that then look like completed pipeline stages.

    Returns:
        Mapping from logical artifact name to absolute :class:`Path`.
    """
    root = Path(outputs_root)
    dirs = {name: root / rel for name, rel in OUTPUT_LAYOUT.items()}
    dirs["root"] = root
    if create:
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def subject_dir(outputs_root: Path, key: str, subject_id: str,
                create: bool = True) -> Path:
    """Return the per-subject subdirectory under a given artifact key.

    Args:
        outputs_root: Root ``outputs/`` directory.
        key: A key of :data:`OUTPUT_LAYOUT`.
        subject_id: Subject / session identifier, e.g. ``"OAS1_0001_MR1"``.
        create: Create the directory when ``True``.

    Raises:
        KeyError: If ``key`` is not a known artifact key — catching layout
            typos here rather than scattering files into stray directories.
    """
    if key not in OUTPUT_LAYOUT:
        raise KeyError(
            f"Unknown output key {key!r}. Known keys: {sorted(OUTPUT_LAYOUT)}"
        )
    path = Path(outputs_root) / OUTPUT_LAYOUT[key] / subject_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["OUTPUT_LAYOUT", "create_output_dirs", "subject_dir"]
