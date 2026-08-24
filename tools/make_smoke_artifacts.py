"""
Smoke-test artifact generator — SYNTHETIC DATA, NOT RESULTS.
===========================================================

Generates the cached artifacts that the model, training, XAI, statistics and
dashboard stages consume, so the whole code path can be exercised while
``dataset/OASIS/`` holds no MRI volumes.

.. danger::

   **Everything this script produces is synthetic.** The ROI patches are
   parametric blobs, not brains. Any accuracy, F1 or AUC measured on them
   describes the synthetic generator, not Alzheimer's disease. The script writes
   a ``SMOKE_TEST.json`` marker into the output root, and every consumer —
   dashboard, report generator, results tables — checks for that marker and
   refuses to present the numbers as findings.

What is real and what is not
----------------------------

* **Real:** the cohort, the subject IDs, the CDR-derived CN/MCI/AD labels and
  the class imbalance. These come from ``dataset/oasis_cross-sectional.csv``.
* **Synthetic:** the imaging. ROI patches are ellipsoidal blobs whose radius and
  intensity texture vary with the subject's stage, so a correctly wired model
  *should* learn them. That is the point: if the pipeline cannot fit an
  obviously-learnable signal, the pipeline is broken. If it can, the pipeline is
  wired correctly — and nothing more has been shown.

Why a deliberately learnable signal
-----------------------------------

The generator injects a monotone CN -> MCI -> AD effect (progressively smaller
blobs with progressively higher intensity entropy) at a configurable effect
size. A pipeline that reaches chance accuracy on this is misconfigured; a
pipeline that fits it is merely functional. Effect size is a parameter so that
the "does the code work" question and the "is the task hard" question stay
separate.

Usage::

    python tools/make_smoke_artifacts.py --out outputs_smoke --n-subjects 60
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.logging_utils import get_logger, setup_logging  # noqa: E402
from modules.common.roi_constants import ROI_ORDER, STAGE_INDEX  # noqa: E402
from modules.m01_dataset import build_cohort  # noqa: E402
from modules.m04_feature_extraction import (  # noqa: E402
    MorphometricFeatureExtractor,
    save_features,
)

logger = get_logger(__name__)

#: Written into the output root. Consumers must check for this file.
MARKER_NAME = "SMOKE_TEST.json"

#: Per-ROI susceptibility to the injected stage effect. Broca and Insula are
#: given the strongest effect so that a correctly wired ROI-ranking stage should
#: recover them; this is a property of the generator, not a finding.
ROI_SUSCEPTIBILITY: Dict[str, float] = {
    "Broca_Area": 1.00,
    "Wernicke_Area": 0.55,
    "Insula": 0.85,
    "Inferior_Frontal_Gyrus": 0.70,
    "Superior_Temporal_Gyrus": 0.40,
}


def synth_patch(
    rng: np.random.Generator,
    stage_index: int,
    susceptibility: float,
    effect_size: float,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
) -> np.ndarray:
    """Generate one synthetic ROI patch with a stage-dependent effect.

    Args:
        rng: Seeded generator.
        stage_index: 0 = CN, 1 = MCI, 2 = AD.
        susceptibility: How strongly this ROI responds to the stage effect.
        effect_size: Global effect magnitude. ``0.0`` makes stages
            indistinguishable, which is useful for confirming that a reported
            accuracy really does come from the signal.
        patch_size: Patch shape.

    Returns:
        Float32 array in ``[0, 1]``.
    """
    d, h, w = patch_size
    patch = np.zeros((d, h, w), dtype=np.float32)

    # Monotone stage effect: radius shrinks, texture roughens.
    shrink = 1.0 - effect_size * susceptibility * (stage_index / 2.0)
    base_radius = 14.0 * shrink * float(rng.normal(1.0, 0.05))
    roughness = 0.10 + effect_size * susceptibility * (stage_index / 2.0) * 0.35

    cz, cy, cx = d / 2.0, h / 2.0, w / 2.0
    zz, yy, xx = np.ogrid[:d, :h, :w]
    radius = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    # Smooth intensity falloff so the tissue mask has a real boundary rather
    # than a hard step, which would make surface area degenerate.
    interior = np.clip(1.0 - (radius / max(base_radius, 1.0)) ** 3, 0.0, 1.0)
    texture = rng.normal(0.0, roughness, size=(d, h, w)).astype(np.float32)
    patch = (0.55 * interior + texture * interior).astype(np.float32)
    return np.clip(patch, 0.0, 1.0)


def generate(
    out_root: Path,
    cohort: pd.DataFrame,
    n_subjects: Optional[int] = None,
    effect_size: float = 0.6,
    seed: int = 12345,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
) -> Dict[str, object]:
    """Generate patch tensors and a feature table for a subset of the cohort.

    Sessions are taken **stratified by stage** so that all three classes are
    represented even at small ``n_subjects``.

    Args:
        out_root: Output root, e.g. ``outputs_smoke``.
        cohort: Cohort table from :func:`modules.m01_dataset.build_cohort`.
        n_subjects: Number of sessions to generate. ``None`` uses all 235.
        effect_size: Injected stage-effect magnitude.
        seed: RNG seed.
        patch_size: ROI patch shape.

    Returns:
        A manifest dict, also written to ``SMOKE_TEST.json``.
    """
    out_root = Path(out_root)
    rng = np.random.default_rng(seed)

    if n_subjects is not None and n_subjects < len(cohort):
        parts: List[pd.DataFrame] = []
        for stage, group in cohort.groupby("stage"):
            share = max(2, int(round(n_subjects * len(group) / len(cohort))))
            parts.append(group.head(min(share, len(group))))
        selected = pd.concat(parts).sort_values("session_id").reset_index(drop=True)
    else:
        selected = cohort.copy()

    extractor = MorphometricFeatureExtractor(out_root / "features_workdir")
    feature_frames: List[pd.DataFrame] = []
    n_patches = 0

    for _, row in selected.iterrows():
        session_id = str(row["session_id"])
        stage_index = STAGE_INDEX[str(row["stage"])]
        # Seed per session so re-running regenerates identical artifacts.
        sub_rng = np.random.default_rng(
            abs(hash((seed, session_id))) % (2 ** 32)
        )

        patches: Dict[str, np.ndarray] = {}
        for roi in ROI_ORDER:
            patches[roi] = synth_patch(
                sub_rng, stage_index, ROI_SUSCEPTIBILITY[roi],
                effect_size, patch_size,
            )

        tensor = np.stack([patches[r] for r in ROI_ORDER]).astype(np.float32)
        patch_dir = out_root / "roi" / "patches" / session_id
        patch_dir.mkdir(parents=True, exist_ok=True)
        np.save(patch_dir / f"{session_id}_roi_tensor.npy", tensor)
        n_patches += 1

        etiv = row.get("eTIV")
        try:
            etiv_value = float(etiv)
        except (TypeError, ValueError):
            etiv_value = None
        feature_frames.append(
            extractor.extract_subject(patches, session_id, etiv=etiv_value)
        )

    features = pd.concat(feature_frames, ignore_index=True)
    report = extractor.audit(features)
    feature_dir = out_root / "features"
    save_features(features, report, feature_dir)

    cohort_dir = out_root / "patient"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(cohort_dir / "cohort.csv", index=False)

    manifest = {
        "is_smoke_test": True,
        "warning": (
            "SYNTHETIC DATA. The ROI patches in this output tree are parametric "
            "blobs, not brain images. Any metric computed from them describes "
            "the synthetic generator and MUST NOT be reported as a research "
            "result or as evidence about Alzheimer's disease."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/make_smoke_artifacts.py",
        "real_components": [
            "subject IDs", "CDR-derived CN/MCI/AD labels", "class imbalance",
            "eTIV values",
        ],
        "synthetic_components": [
            "ROI patch volumes", "all morphometric features derived from them",
        ],
        "n_sessions": int(len(selected)),
        "n_patch_tensors": n_patches,
        "effect_size": effect_size,
        "seed": seed,
        "patch_size": list(patch_size),
        "roi_susceptibility": dict(ROI_SUSCEPTIBILITY),
        "stage_counts": {
            str(k): int(v) for k, v in selected["stage"].value_counts().items()
        },
        "feature_quality": report.to_dict(),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / MARKER_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    logger.info(
        "Smoke artifacts written to %s: %d session(s), stage counts %s",
        out_root, len(selected), manifest["stage_counts"],
    )
    return manifest


def is_smoke_output(out_root: Path) -> Optional[Dict[str, object]]:
    """Return the smoke-test manifest if ``out_root`` holds synthetic artifacts.

    Every consumer that displays or reports metrics calls this and, when it
    returns non-``None``, must render an explicit synthetic-data warning instead
    of presenting the numbers as findings.

    Args:
        out_root: An outputs root directory.

    Returns:
        The manifest, or ``None`` if the tree is not marked as a smoke test.
    """
    marker = Path(out_root) / MARKER_NAME
    if not marker.exists():
        return None
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A marker that cannot be parsed still means "this tree is synthetic".
        return {"is_smoke_test": True, "warning": "unparseable smoke-test marker"}


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SYNTHETIC smoke-test artifacts (not results)."
    )
    parser.add_argument("--out", type=Path, default=Path("outputs_smoke"),
                        help="Output root directory.")
    parser.add_argument("--n-subjects", type=int, default=60,
                        help="Number of sessions to generate (stratified).")
    parser.add_argument("--effect-size", type=float, default=0.6,
                        help="Injected stage-effect magnitude; 0 = no signal.")
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    setup_logging()
    cfg = NeuroGenesisConfig()
    cohort, _ = build_cohort(cfg.paths, cfg.data)

    manifest = generate(
        out_root=args.out,
        cohort=cohort,
        n_subjects=args.n_subjects,
        effect_size=args.effect_size,
        seed=args.seed,
    )
    print(json.dumps(
        {k: v for k, v in manifest.items() if k != "feature_quality"}, indent=2
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
