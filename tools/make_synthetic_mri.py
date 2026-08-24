"""
Synthetic T1 volume generator — SYNTHETIC IMAGING, NOT REAL MRI.
================================================================

Writes NIfTI volumes named after **real** OASIS-1 session IDs so that the
imaging half of the pipeline (M1-M8) can be exercised end to end when no real
MRI is available.

.. danger::

   **The volumes are synthetic.** They are parametric ellipsoids with simulated
   skull, brain tissue and speech-region blobs — not brain images. Any metric
   derived from them describes this generator. What running the pipeline on them
   *does* prove is that loading, QC, N4 correction, WM-peak normalization,
   CLAHE, anisotropic diffusion, skull stripping, resampling, Harvard-Oxford
   atlas localization, patch extraction and feature extraction all execute and
   produce well-formed artifacts.

   The subject IDs and their CDR-derived CN/MCI/AD labels are real, taken from
   ``dataset/oasis_cross-sectional.csv``, so the cohort and split machinery
   operate on genuine metadata.

This differs from ``tools/make_smoke_artifacts.py``, which starts at the ROI
patch stage and skips imaging entirely. Use this one to validate M1-M8.

Usage::

    python tools/make_synthetic_mri.py --out dataset/OASIS_synthetic --n 9
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
from modules.common.roi_constants import STAGE_INDEX  # noqa: E402
from modules.m01_dataset import build_cohort  # noqa: E402

logger = get_logger(__name__)

MARKER_NAME = "SYNTHETIC_MRI.json"

#: Approximate left-hemisphere MNI coordinates (mm) of the five speech regions.
#:
#: These matter. A first version placed blobs at arbitrary voxel indices and
#: wrote an identity affine, so the volume carried no MNI-like world
#: coordinates. The Harvard-Oxford atlas is defined in MNI space, so registering
#: it onto such a volume put every mask in one corner: Wernicke's area came back
#: with zero voxels and STG with 63, while the pipeline itself was working
#: correctly. Placing the blobs at real MNI coordinates and writing a centred
#: affine is what makes the phantom actually exercise atlas registration.
REGION_MNI_MM: Dict[str, Tuple[float, float, float]] = {
    "Broca_Area": (-48.0, 20.0, 12.0),            # IFG pars opercularis
    "Wernicke_Area": (-52.0, -48.0, 12.0),        # posterior STG
    "Insula": (-38.0, 4.0, 2.0),                  # insular cortex
    "Inferior_Frontal_Gyrus": (-48.0, 26.0, 8.0),
    "Superior_Temporal_Gyrus": (-58.0, -20.0, 4.0),
}


def mni_affine(shape: Tuple[int, int, int]) -> np.ndarray:
    """Return a 1 mm RAS affine that places the volume centre at world origin.

    A real T1 carries an affine whose translation puts the anatomy near the MNI
    origin, which is what lets an MNI-space atlas be resampled onto it. An
    identity affine claims the corner voxel sits at world (0, 0, 0), and the
    atlas then lands almost entirely outside the brain.
    """
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = -np.asarray(shape, dtype=np.float64) / 2.0
    return affine


def mni_to_voxel(coord: Tuple[float, float, float],
                 affine: np.ndarray) -> Tuple[int, int, int]:
    """Convert an MNI millimetre coordinate to a voxel index."""
    voxel = np.linalg.inv(affine) @ np.array([*coord, 1.0])
    return tuple(int(round(v)) for v in voxel[:3])


def synth_volume(
    shape: Tuple[int, int, int],
    stage_index: int,
    effect_size: float,
    rng: np.random.Generator,
    affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Generate one synthetic T1-like volume with a stage-dependent effect.

    Args:
        shape: Volume dimensions.
        stage_index: 0 = CN, 1 = MCI, 2 = AD.
        effect_size: Magnitude of the injected stage effect; 0 removes it.
        rng: Seeded generator.
        affine: Voxel-to-world affine, used to place the speech-region
            blobs at their MNI coordinates. Defaults to a centred 1 mm RAS
            affine.

    Returns:
        Float32 volume with a skull shell, brain tissue, ventricles and five
        speech-region blobs whose size shrinks with stage index.
    """
    nx, ny, nz = shape
    volume = np.zeros(shape, dtype=np.float32)
    affine = mni_affine(shape) if affine is None else np.asarray(affine)

    zz, yy, xx = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    cx, cy, cz = nx / 2.0, ny / 2.0, nz / 2.0

    # Ellipsoidal head, then brain inside it.
    head = (((zz - cx) / (nx * 0.42)) ** 2
            + ((yy - cy) / (ny * 0.44)) ** 2
            + ((xx - cz) / (nz * 0.40)) ** 2)
    brain_radius = 0.86 - effect_size * 0.05 * (stage_index / 2.0)
    brain = (((zz - cx) / (nx * 0.42 * brain_radius)) ** 2
             + ((yy - cy) / (ny * 0.44 * brain_radius)) ** 2
             + ((xx - cz) / (nz * 0.40 * brain_radius)) ** 2)

    volume[head <= 1.0] = 0.92          # skull / scalp: bright shell
    volume[brain <= 1.0] = 0.55         # white matter
    # Grey-matter rim.
    rim = (brain > 0.72) & (brain <= 1.0)
    volume[rim] = 0.40

    # Ventricles enlarge with stage — the classic coarse AD signature.
    ventricle_scale = 1.0 + effect_size * 0.55 * (stage_index / 2.0)
    ventricle = (((zz - cx) / (nx * 0.075 * ventricle_scale)) ** 2
                 + ((yy - cy) / (ny * 0.115 * ventricle_scale)) ** 2
                 + ((xx - cz) / (nz * 0.055 * ventricle_scale)) ** 2)
    volume[ventricle <= 1.0] = 0.08

    # Speech-region blobs at real MNI coordinates, so the Harvard-Oxford
    # atlas actually finds tissue where it expects each region to be.
    for name, mni in REGION_MNI_MM.items():
        bx, by, bz = mni_to_voxel(mni, affine)
        radius = 11.0 * (1.0 - effect_size * 0.30 * (stage_index / 2.0))
        radius *= float(rng.normal(1.0, 0.04))
        blob = ((zz - bx) ** 2 + (yy - by) ** 2 + (xx - bz) ** 2) < radius ** 2
        inside = blob & (brain <= 1.0)
        volume[inside] = 0.62 + 0.06 * rng.normal(0.0, 1.0, size=inside.sum())

        # Mirror to the right hemisphere so bilateral regions are populated.
        mirror = ((zz - (nx - bx)) ** 2 + (yy - by) ** 2
                  + (xx - bz) ** 2) < radius ** 2
        inside_m = mirror & (brain <= 1.0)
        volume[inside_m] = 0.60 + 0.06 * rng.normal(0.0, 1.0, size=inside_m.sum())

    # A smooth multiplicative bias field, so N4 correction has something to do.
    bias = (1.0
            + 0.22 * (zz / nx - 0.5)
            + 0.16 * (yy / ny - 0.5)
            - 0.12 * (xx / nz - 0.5))
    volume = volume * bias.astype(np.float32)

    # Rician-ish noise on tissue only, leaving air quiet.
    tissue = volume > 0.02
    volume[tissue] += rng.normal(0.0, 0.028, size=int(tissue.sum())).astype(
        np.float32
    )
    return np.clip(volume, 0.0, None).astype(np.float32)


def generate(
    out_dir: Path,
    cohort: pd.DataFrame,
    n_subjects: int = 9,
    shape: Tuple[int, int, int] = (176, 208, 176),
    effect_size: float = 0.8,
    seed: int = 4242,
) -> Dict[str, object]:
    """Write synthetic volumes for a stratified subset of real session IDs.

    Args:
        out_dir: Destination directory, e.g. ``dataset/OASIS_synthetic``.
        cohort: Cohort table from :func:`modules.m01_dataset.build_cohort`.
        n_subjects: Number of sessions to generate.
        shape: Volume dimensions.
        effect_size: Injected stage-effect magnitude.
        seed: RNG seed.

    Returns:
        A manifest dict, also written to ``SYNTHETIC_MRI.json``.

    Raises:
        ImportError: If nibabel is unavailable.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "nibabel is required to write NIfTI volumes. "
            "Install it with `pip install nibabel`."
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_stage = max(1, n_subjects // 3)
    selected = pd.concat(
        [group.head(per_stage) for _, group in cohort.groupby("stage")]
    ).sort_values("session_id").reset_index(drop=True)

    # A centred 1 mm RAS affine, as a real T1 carries. Identity would put
    # the corner voxel at world origin and mis-register the atlas.
    affine = mni_affine(shape)
    written: List[Dict[str, object]] = []

    for _, row in selected.iterrows():
        session_id = str(row["session_id"])
        stage = str(row["stage"])
        # Seed per session so re-running reproduces identical volumes.
        rng = np.random.default_rng(abs(hash((seed, session_id))) % (2 ** 32))
        volume = synth_volume(
            shape, STAGE_INDEX[stage], effect_size, rng, affine
        )

        path = out_dir / f"{session_id}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine), path)
        written.append({
            "session_id": session_id,
            "stage": stage,
            "path": path.as_posix(),
            "shape": list(volume.shape),
            "intensity_range": [float(volume.min()), float(volume.max())],
            "affine": affine.tolist(),
        })
        logger.info("wrote %s (%s) %s", path.name, stage, volume.shape)

    manifest = {
        "is_synthetic_mri": True,
        "warning": (
            "SYNTHETIC IMAGING. These NIfTI volumes are parametric phantoms, "
            "not brain images. Running the pipeline on them proves the imaging "
            "code executes and produces well-formed artifacts; it says nothing "
            "about Alzheimer's disease. Any metric derived from them MUST NOT "
            "be reported as a research result."
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/make_synthetic_mri.py",
        "real_components": ["session IDs", "CDR-derived CN/MCI/AD labels"],
        "synthetic_components": ["all imaging data"],
        "n_volumes": len(written),
        "shape": list(shape),
        "affine": affine.tolist(),
        "region_mni_mm": {k: list(v) for k, v in REGION_MNI_MM.items()},
        "effect_size": effect_size,
        "seed": seed,
        "volumes": written,
    }
    (out_dir / MARKER_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SYNTHETIC T1 volumes for real OASIS-1 session IDs."
    )
    parser.add_argument("--out", type=Path,
                        default=Path("dataset/OASIS_synthetic"))
    parser.add_argument("--n", type=int, default=9,
                        help="Number of sessions (stratified across stages).")
    parser.add_argument("--effect-size", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--shape", type=int, nargs=3,
                        default=[176, 208, 176])
    args = parser.parse_args()

    setup_logging()
    cfg = NeuroGenesisConfig()
    cohort, _ = build_cohort(cfg.paths, cfg.data)

    manifest = generate(
        out_dir=args.out, cohort=cohort, n_subjects=args.n,
        shape=tuple(args.shape), effect_size=args.effect_size, seed=args.seed,
    )
    print(f"\nWrote {manifest['n_volumes']} synthetic volume(s) to {args.out}")
    for entry in manifest["volumes"]:
        print(f"  {entry['session_id']}  {entry['stage']:<4s} "
              f"{tuple(entry['shape'])}")
    print(f"\n{manifest['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
