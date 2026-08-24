"""
M1-M5 — MRI loading, QC, preprocessing, skull stripping, standardization.
========================================================================

Wraps the **preserved** ``preprocessing/`` package into one per-subject
pipeline that records every intermediate stage.

The underlying implementations are unchanged. This module adds three things the
refactor needs:

1. **Per-stage artifact recording**, so Sections 41-45 of the dashboard can show
   the input, output, numeric summary and downloadable file for each stage
   rather than only the final volume.
2. **Explicit dependency reporting.** The imaging chain needs ``nibabel``,
   ``SimpleITK``, ``nilearn`` and ``scikit-image``, none of which is a pure-Python
   package. :func:`check_imaging_dependencies` reports exactly which are missing
   so the failure is one clear message rather than an ``ImportError`` from three
   frames down.
3. **QC gating that is recorded, not silent.** A subject below the QC threshold
   is either skipped with the reason attached, or processed with the failure
   flagged — never processed as though it had passed.

Every heavy import is deferred into the method that needs it, so this module is
importable (and the dashboard and tests run) on a machine with no imaging stack.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.common.config import PreprocessConfig
from modules.common.logging_utils import get_logger
from modules.common.run_state import RunStateTracker

logger = get_logger(__name__)

#: Package -> what breaks without it.
IMAGING_DEPENDENCIES: Dict[str, str] = {
    "nibabel": "reading and writing NIfTI volumes (M1, all NIfTI outputs)",
    "SimpleITK": "N4 bias correction, anisotropic diffusion, resampling (M3, M5)",
    "nilearn": "Harvard-Oxford atlas download and brain masking (M4, M6)",
    "skimage": "CLAHE and marching-cubes surface area (M3, M8)",
}


def check_imaging_dependencies() -> Dict[str, Any]:
    """Report which imaging packages are available.

    Returns:
        ``{"available": {...}, "missing": [...], "can_run": bool, "message": str}``
    """
    import importlib

    available: Dict[str, bool] = {}
    for package in IMAGING_DEPENDENCIES:
        try:
            importlib.import_module(package)
            available[package] = True
        except ImportError:
            available[package] = False

    missing = [p for p, ok in available.items() if not ok]
    if missing:
        details = "; ".join(f"{p} ({IMAGING_DEPENDENCIES[p]})" for p in missing)
        message = (
            f"The imaging pipeline cannot run: {len(missing)} required "
            f"package(s) are missing - {details}. Install them with "
            "`pip install -r requirements.txt`."
        )
    else:
        message = "All imaging dependencies are available."

    return {
        "available": available,
        "missing": missing,
        "can_run": not missing,
        "message": message,
    }


@dataclass
class StageArtifact:
    """One recorded preprocessing stage."""

    name: str
    description: str
    #: Shape of the volume this stage produced.
    shape: Optional[Tuple[int, ...]] = None
    #: ``(min, max)`` intensity range after the stage.
    intensity_range: Optional[Tuple[float, float]] = None
    mean_intensity: Optional[float] = None
    #: Count of non-background voxels.
    nonzero_voxels: Optional[int] = None
    seconds: Optional[float] = None
    #: Written NIfTI path, when saving was enabled.
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "description": self.description,
            "shape": list(self.shape) if self.shape else None,
            "intensity_range": (
                list(self.intensity_range) if self.intensity_range else None
            ),
            "mean_intensity": self.mean_intensity,
            "nonzero_voxels": self.nonzero_voxels,
            "seconds": self.seconds,
            "path": self.path,
        }


@dataclass
class SubjectPreprocessingResult:
    """Complete per-subject preprocessing outcome."""

    subject_id: str
    source_path: str
    succeeded: bool = False
    #: M1 metadata: dimensions, voxel spacing, orientation, affine.
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: M2 quality-control metrics.
    qc: Dict[str, Any] = field(default_factory=dict)
    qc_passed: Optional[bool] = None
    #: M3-M5 stage artifacts in execution order.
    stages: List[StageArtifact] = field(default_factory=list)
    #: Final standardised volume path.
    final_path: Optional[str] = None
    brain_mask_path: Optional[str] = None
    #: Affine of the final standardised volume. M6 needs the **resampled**
    #: affine, not the original one, or the atlas is registered to the wrong grid.
    final_affine: Optional[List[List[float]]] = None
    #: White-matter peak found by KDE normalization, for the dashboard.
    wm_peak: Optional[float] = None
    #: Voxel counts before and after skull stripping.
    brain_voxels_before: Optional[int] = None
    brain_voxels_after: Optional[int] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "subject_id": self.subject_id,
            "source_path": self.source_path,
            "succeeded": self.succeeded,
            "metadata": self.metadata,
            "qc": self.qc,
            "qc_passed": self.qc_passed,
            "stages": [s.to_dict() for s in self.stages],
            "final_path": self.final_path,
            "brain_mask_path": self.brain_mask_path,
            "final_affine": self.final_affine,
            "wm_peak": self.wm_peak,
            "brain_voxels_before": self.brain_voxels_before,
            "brain_voxels_after": self.brain_voxels_after,
            "error": self.error,
            "warnings": list(self.warnings),
        }

    def save(self, out_dir: Path) -> Path:
        """Write the result manifest to JSON."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.subject_id}_preprocessing.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


class PreprocessingPipeline:
    """Per-subject MRI preprocessing over the preserved ``preprocessing`` package.

    Args:
        cfg: Preprocessing configuration.
        outputs_root: Root outputs directory.
        tracker: Optional run-state tracker for M1-M5.
    """

    def __init__(
        self,
        cfg: Optional[PreprocessConfig] = None,
        outputs_root: Path = Path("outputs"),
        tracker: Optional[RunStateTracker] = None,
    ) -> None:
        self.cfg = cfg or PreprocessConfig()
        self.outputs_root = Path(outputs_root)
        self.tracker = tracker
        self._checked = False

    def _require_imaging(self) -> None:
        """Raise a single clear error if the imaging stack is incomplete."""
        if self._checked:
            return
        status = check_imaging_dependencies()
        if not status["can_run"]:
            raise ImportError(status["message"])
        self._checked = True

    def _stage(self, code: str, subject_id: str):
        """Return a tracker stage context, or a no-op when untracked."""
        if self.tracker is None:
            from contextlib import nullcontext

            return nullcontext(None)
        return self.tracker.stage(code, subject_id)

    @staticmethod
    def _describe(volume: np.ndarray) -> Dict[str, Any]:
        """Summarise a volume numerically."""
        finite = volume[np.isfinite(volume)]
        return {
            "shape": tuple(int(s) for s in volume.shape),
            "intensity_range": (
                (float(finite.min()), float(finite.max())) if finite.size
                else (0.0, 0.0)
            ),
            "mean_intensity": float(finite.mean()) if finite.size else 0.0,
            "nonzero_voxels": int(np.count_nonzero(volume > 1e-6)),
        }

    def run(self, subject_id: str, mri_path: Path) -> SubjectPreprocessingResult:
        """Run M1-M5 for one subject.

        Args:
            subject_id: Session identifier.
            mri_path: Path to the T1 volume.

        Returns:
            A :class:`SubjectPreprocessingResult`. On failure ``succeeded`` is
            ``False`` and ``error`` carries the reason; the exception is not
            re-raised so a batch run continues past one bad scan.
        """
        result = SubjectPreprocessingResult(
            subject_id=subject_id, source_path=Path(mri_path).as_posix()
        )
        try:
            self._require_imaging()
        except ImportError as exc:
            result.error = str(exc)
            return result

        from preprocessing.artifact_detector import ArtifactDetector
        from preprocessing.loader import MRILoader
        from preprocessing.normalization import MRINormalizer
        from preprocessing.resize import MRIResizer
        from preprocessing.skull_strip import SkullStripper

        processed_dir = self.outputs_root / "preprocessing" / subject_id
        processed_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ── M1: load ──────────────────────────────────────────────────
            with self._stage("M1", subject_id) as record:
                loader = MRILoader(
                    dataset_dir=Path(mri_path).parent, output_dir=processed_dir
                )
                scan = loader.load_single(Path(mri_path))
                volume = np.asarray(scan["data"], dtype=np.float32)
                # The affine is carried through every stage: skull
                # stripping, resampling and atlas registration all need it,
                # and resampling replaces it.
                affine = np.asarray(
                    scan.get("affine", np.eye(4)), dtype=np.float64
                )
                result.metadata = {
                    k: v for k, v in scan.get("metadata", {}).items()
                    if k != "data"
                }
                result.metadata.update(self._describe(volume))
                if record is not None:
                    record.record_metric("shape", list(volume.shape))
                    record.record_metric("affine", affine.tolist())
                    record.record_artifact("source", Path(mri_path))

            # ── M2: quality control ───────────────────────────────────────
            with self._stage("M2", subject_id) as record:
                detector = ArtifactDetector(
                    output_dir=processed_dir,
                    min_quality_score=self.cfg.min_quality_score,
                )
                report = detector.run_qc(volume, subject_id)
                result.qc = {
                    k: v for k, v in vars(report).items()
                    if not k.startswith("_")
                }
                score = float(getattr(report, "quality_score", 0.0))
                result.qc_passed = score >= self.cfg.min_quality_score
                if record is not None:
                    record.record_metric("quality_score", score)
                    record.record_metric("passed", result.qc_passed)

            if not result.qc_passed and self.cfg.skip_failed_qc:
                reason = (
                    f"QC score {result.qc.get('quality_score')} is below the "
                    f"threshold {self.cfg.min_quality_score}; the subject was "
                    "skipped as configured (skip_failed_qc=True)."
                )
                result.warnings.append(reason)
                result.error = reason
                if self.tracker is not None:
                    for code in ("M3", "M4", "M5"):
                        self.tracker.mark_skipped(code, reason, subject_id)
                return result
            if not result.qc_passed:
                result.warnings.append(
                    f"QC score {result.qc.get('quality_score')} is below the "
                    f"threshold {self.cfg.min_quality_score}. The subject was "
                    "processed anyway (skip_failed_qc=False); downstream results "
                    "for it should be treated as low confidence."
                )

            # ── M3: intensity preprocessing chain ─────────────────────────
            with self._stage("M3", subject_id) as record:
                normalizer = MRINormalizer(
                    output_dir=processed_dir,
                    gaussian_sigma=self.cfg.gaussian_sigma,
                    aniso_iterations=self.cfg.aniso_iterations,
                    aniso_conductance=self.cfg.aniso_conductance,
                    clahe_clip_limit=self.cfg.clahe_clip_limit,
                )
                # normalize_wm_peak returns (volume, wm_peak); the others
                # return a bare volume. The flag records which is which so
                # the tuple is unpacked rather than stored as a volume.
                chain = (
                    ("n4_bias_corrected", "N4 bias field correction",
                     normalizer.apply_n4_bias_correction, False),
                    ("wm_normalized", "White-matter KDE peak normalization",
                     normalizer.normalize_wm_peak, True),
                    ("clahe", "CLAHE adaptive histogram equalization",
                     normalizer.apply_clahe, False),
                    ("anisotropic_diffusion",
                     "Perona-Malik edge-preserving denoising",
                     normalizer.apply_anisotropic_diffusion, False),
                    ("intensity_normalized", "Min-max intensity normalization",
                     normalizer.normalize_minmax, False),
                )
                current = volume
                result.stages.append(StageArtifact(
                    name="original", description="Loaded T1 volume",
                    **self._describe(current),
                ))
                for name, description, function, returns_tuple in chain:
                    t0 = time.perf_counter()
                    try:
                        produced = function(current)
                        if returns_tuple:
                            produced, extra = produced
                            result.wm_peak = float(extra)
                        current = np.asarray(produced, dtype=np.float32)
                        artifact = StageArtifact(
                            name=name, description=description,
                            seconds=round(time.perf_counter() - t0, 3),
                            **self._describe(current),
                        )
                    except Exception as exc:  # noqa: BLE001
                        # One optional enhancement failing must not lose the
                        # whole subject; the stage is recorded as skipped.
                        result.warnings.append(
                            f"Stage {name} failed ({exc}); the previous volume "
                            "was carried forward unchanged."
                        )
                        artifact = StageArtifact(
                            name=name,
                            description=f"{description} - FAILED: {exc}",
                            seconds=round(time.perf_counter() - t0, 3),
                            **self._describe(current),
                        )
                    result.stages.append(artifact)
                if record is not None:
                    record.record_metric(
                        "stages", [s.name for s in result.stages]
                    )

            # ── M4: skull stripping ───────────────────────────────────────
            with self._stage("M4", subject_id) as record:
                result.brain_voxels_before = int(np.count_nonzero(current > 1e-6))
                stripper = SkullStripper(
                    output_dir=processed_dir, morph_radius=self.cfg.morph_radius
                )
                stripped, mask = stripper.strip(
                    current, subject_id, affine, save=self.cfg.save_nifti
                )
                current = np.asarray(stripped, dtype=np.float32)
                if self.cfg.save_nifti:
                    mask_path = self._save_volume(
                        np.asarray(mask, dtype=np.float32), processed_dir,
                        f"{subject_id}_brain_mask", affine,
                    )
                    if mask_path is not None:
                        result.brain_mask_path = mask_path.as_posix()
                result.brain_voxels_after = int(np.count_nonzero(current > 1e-6))
                result.stages.append(StageArtifact(
                    name="skull_stripped", description="Brain extraction",
                    **self._describe(current),
                ))
                if record is not None:
                    record.record_metric("voxels_before", result.brain_voxels_before)
                    record.record_metric("voxels_after", result.brain_voxels_after)
                    record.record_metric(
                        "brain_fraction",
                        result.brain_voxels_after
                        / max(result.brain_voxels_before, 1),
                    )

            # ── M5: spatial standardization ───────────────────────────────
            with self._stage("M5", subject_id) as record:
                resizer = MRIResizer(
                    output_dir=processed_dir,
                    target_shape=self.cfg.target_shape,
                    interpolator=self.cfg.interpolator,
                )
                original_shape = current.shape
                resampled, new_affine = resizer.resample(
                    current, affine, subject_id, save=self.cfg.save_nifti
                )
                current = np.asarray(resampled, dtype=np.float32)
                # Resampling replaces the affine. Everything downstream,
                # most importantly the atlas registration in M6, must use
                # the new one or the ROI masks land on the wrong grid.
                affine = np.asarray(new_affine, dtype=np.float64)
                result.final_affine = affine.tolist()
                result.stages.append(StageArtifact(
                    name="standardized",
                    description=f"Resampled to {self.cfg.target_shape}",
                    **self._describe(current),
                ))
                if record is not None:
                    record.record_metric("shape_before", list(original_shape))
                    record.record_metric("shape_after", list(current.shape))
                    record.record_metric("affine_after", affine.tolist())

            if self.cfg.save_nifti:
                path = self._save_volume(
                    current, processed_dir, f"{subject_id}_standardized",
                    affine,
                )
                result.final_path = path.as_posix() if path else None

            result.succeeded = True
            return result

        except Exception as exc:  # noqa: BLE001
            logger.error("Preprocessing failed for %s: %s", subject_id, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result

    @staticmethod
    def _save_volume(volume: np.ndarray, out_dir: Path, name: str,
                     affine: Optional[np.ndarray] = None) -> Optional[Path]:
        """Write a volume as ``.nii.gz`` with the affine that describes it.

        The affine is passed in rather than defaulted to identity. After
        resampling the voxel-to-world mapping has genuinely changed, and writing
        identity here would silently mis-register the volume against the atlas in
        M6 while every shape and intensity check still looked correct. Returns
        ``None`` rather than raising if nibabel is unavailable.
        """
        try:
            import nibabel as nib
        except ImportError:
            return None
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name}.nii.gz"
        matrix = np.eye(4) if affine is None else np.asarray(affine)
        nib.save(nib.Nifti1Image(volume.astype(np.float32), matrix), path)
        return path


__all__ = [
    "IMAGING_DEPENDENCIES",
    "check_imaging_dependencies",
    "StageArtifact",
    "SubjectPreprocessingResult",
    "PreprocessingPipeline",
]
