"""
M6-M7 — Harvard-Oxford speech ROI localization and patch extraction.
===================================================================

Wraps the **preserved** ``segmentation/roi_extraction.py`` and
``preprocessing/roi_crop.py`` so that ROI localization is a clean, reusable
module (Section 4) and so that each step's intermediate output is recorded for
the dashboard.

The atlas is the anatomical localization mechanism, not the research
contribution. NeuroProp-X begins after this module has produced the five ROI
patches.

The five ROIs are fixed by :data:`~modules.common.roi_constants.ROI_ORDER`:
Broca's area, Wernicke's area, the insula, the inferior frontal gyrus and the
superior temporal gyrus. The patch tensor axis follows that order, and
:func:`save_patch_tensor` re-checks it on every write — a transposed ROI axis
produces a perfectly plausible tensor whose regions are silently mislabelled,
which no downstream number would reveal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.common.config import PreprocessConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_METADATA, ROI_ORDER
from modules.common.run_state import RunStateTracker

logger = get_logger(__name__)


@dataclass
class ROIResult:
    """Localization and patch-extraction outcome for one ROI."""

    roi_name: str
    voxel_count: int = 0
    volume_mm3: float = 0.0
    #: ``((z0, z1), (y0, y1), (x0, x1))`` bounding box in the standardised grid.
    bounding_box: Optional[List[List[int]]] = None
    centroid: Optional[List[float]] = None
    patch_shape: Optional[List[int]] = None
    mask_path: Optional[str] = None
    patch_path: Optional[str] = None
    #: True when the atlas produced no voxels for this region.
    empty: bool = False
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "roi_name": self.roi_name,
            "label": ROI_METADATA.get(self.roi_name, {}).get("label", ""),
            "voxel_count": self.voxel_count,
            "volume_mm3": self.volume_mm3,
            "bounding_box": self.bounding_box,
            "centroid": self.centroid,
            "patch_shape": self.patch_shape,
            "mask_path": self.mask_path,
            "patch_path": self.patch_path,
            "empty": self.empty,
            "note": self.note,
        }


@dataclass
class SegmentationResult:
    """Complete M6-M7 outcome for one subject."""

    subject_id: str
    succeeded: bool = False
    atlas_name: Optional[str] = None
    rois: List[ROIResult] = field(default_factory=list)
    tensor_shape: Optional[List[int]] = None
    tensor_path: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def n_empty(self) -> int:
        """Number of ROIs the atlas failed to populate."""
        return sum(1 for r in self.rois if r.empty)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "subject_id": self.subject_id,
            "succeeded": self.succeeded,
            "atlas_name": self.atlas_name,
            "roi_order": list(ROI_ORDER),
            "rois": [r.to_dict() for r in self.rois],
            "n_empty_rois": self.n_empty,
            "tensor_shape": self.tensor_shape,
            "tensor_path": self.tensor_path,
            "error": self.error,
            "warnings": list(self.warnings),
        }

    def save(self, out_dir: Path) -> Path:
        """Write the result manifest to JSON."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.subject_id}_segmentation.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def save_patch_tensor(
    patches: Dict[str, np.ndarray],
    out_dir: Path,
    subject_id: str,
    patch_size: Tuple[int, int, int] = (48, 48, 48),
) -> Tuple[Path, np.ndarray]:
    """Stack per-ROI patches into the ``(5, 48, 48, 48)`` tensor and save it.

    Args:
        patches: ``{roi_name: (D, H, W) array}``.
        out_dir: Destination directory.
        subject_id: Session identifier.
        patch_size: Expected patch shape.

    Returns:
        ``(path, tensor)``.

    Raises:
        KeyError: If an ROI from :data:`ROI_ORDER` is absent. A missing region
            must not be silently zero-filled: the CNN would then train on a
            blank volume labelled with the subject's stage.
        ValueError: If a patch has an unexpected shape.
    """
    stacked: List[np.ndarray] = []
    for roi in ROI_ORDER:
        if roi not in patches:
            raise KeyError(
                f"{subject_id}: no patch for ROI {roi!r}. The tensor axis must "
                f"follow ROI_ORDER {ROI_ORDER}; zero-filling a missing region "
                "would train the encoder on a blank volume."
            )
        patch = np.asarray(patches[roi], dtype=np.float32)
        if tuple(patch.shape) != tuple(patch_size):
            raise ValueError(
                f"{subject_id}/{roi}: patch shape {patch.shape} != expected "
                f"{patch_size}"
            )
        stacked.append(patch)

    tensor = np.stack(stacked).astype(np.float32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{subject_id}_roi_tensor.npy"
    np.save(path, tensor)
    return path, tensor


class ROIPipeline:
    """Harvard-Oxford ROI localization and ROI-centric patch extraction.

    Args:
        cfg: Preprocessing configuration (atlas name, patch size, context pad).
        outputs_root: Root outputs directory.
        tracker: Optional run-state tracker for M6-M7.
        atlas_dir: Nilearn atlas cache directory. Forwarded to the
            extractor's ``data_dir`` so the atlas is downloaded once into
            the project rather than into the user's home directory.
    """

    def __init__(
        self,
        cfg: Optional[PreprocessConfig] = None,
        outputs_root: Path = Path("outputs"),
        tracker: Optional[RunStateTracker] = None,
        atlas_dir: Optional[Path] = None,
    ) -> None:
        self.cfg = cfg or PreprocessConfig()
        self.outputs_root = Path(outputs_root)
        self.tracker = tracker
        #: Nilearn atlas cache. Passed to the extractor as `data_dir`.
        self.atlas_dir = Path(atlas_dir) if atlas_dir else None

    def _stage(self, code: str, subject_id: str):
        """Return a tracker stage context, or a no-op when untracked."""
        if self.tracker is None:
            from contextlib import nullcontext

            return nullcontext(None)
        return self.tracker.stage(code, subject_id)

    def run(
        self,
        subject_id: str,
        volume: np.ndarray,
        affine: Optional[np.ndarray] = None,
    ) -> Tuple[SegmentationResult, Dict[str, np.ndarray]]:
        """Localize the five speech ROIs and extract their patches.

        Args:
            subject_id: Session identifier.
            volume: Standardised brain volume from M5.
            affine: Voxel-to-world affine. Falls back to identity, which is
                correct for a volume already resampled onto the common grid.

        Returns:
            ``(result, patches)`` where ``patches`` maps ROI name to its
            ``(48, 48, 48)`` array. On failure ``patches`` is empty and
            ``result.error`` carries the reason.
        """
        from modules.m02_preprocessing.pipeline import check_imaging_dependencies

        result = SegmentationResult(
            subject_id=subject_id, atlas_name=self.cfg.atlas_name
        )
        status = check_imaging_dependencies()
        if not status["can_run"]:
            result.error = status["message"]
            return result, {}

        from preprocessing.roi_crop import ROICropper
        from segmentation.roi_extraction import ROIExtractor

        roi_dir = self.outputs_root / "roi" / "masks" / subject_id
        patch_dir = self.outputs_root / "roi" / "patches" / subject_id
        affine = np.eye(4) if affine is None else np.asarray(affine)

        try:
            # ── M6: atlas ROI localization ────────────────────────────────
            with self._stage("M6", subject_id) as record:
                extractor = ROIExtractor(
                    output_dir=roi_dir,
                    atlas_name=self.cfg.atlas_name,
                    data_dir=self.atlas_dir,
                )
                masks = extractor.extract_all(
                    volume, affine, subject_id, save=self.cfg.save_nifti
                )
                if record is not None:
                    record.record_metric("atlas", self.cfg.atlas_name)
                    record.record_metric("n_rois", len(masks))
                    record.record_metric("roi_names", sorted(masks))

            voxel_volume = float(abs(np.linalg.det(affine[:3, :3]))) or 1.0

            for roi in ROI_ORDER:
                mask = masks.get(roi)
                entry = ROIResult(roi_name=roi)
                if mask is None:
                    entry.empty = True
                    entry.note = "The atlas returned no mask for this region."
                    result.warnings.append(
                        f"{roi}: no atlas mask was produced."
                    )
                else:
                    binary = np.asarray(mask) > 0
                    entry.voxel_count = int(binary.sum())
                    entry.volume_mm3 = float(entry.voxel_count * voxel_volume)
                    if entry.voxel_count == 0:
                        entry.empty = True
                        entry.note = (
                            "The atlas mask is empty for this subject. This "
                            "usually means the atlas and the subject volume are "
                            "misaligned."
                        )
                        result.warnings.append(f"{roi}: atlas mask is empty.")
                    else:
                        coords = np.argwhere(binary)
                        entry.bounding_box = [
                            [int(coords[:, a].min()), int(coords[:, a].max())]
                            for a in range(3)
                        ]
                        entry.centroid = [
                            float(coords[:, a].mean()) for a in range(3)
                        ]
                result.rois.append(entry)

            # ── M7: ROI-centric patch extraction ──────────────────────────
            with self._stage("M7", subject_id) as record:
                cropper = ROICropper(
                    output_dir=patch_dir,
                    patch_size=self.cfg.patch_size,
                    context_pad=self.cfg.context_pad,
                    tensor_dir=patch_dir,
                )
                # `ROICropper.extract_all` stacks patches in `masks.keys()`
                # order and silently substitutes zeros for an ROI it fails on.
                # Both are unacceptable here: the tensor axis must follow
                # ROI_ORDER, and a blank patch labelled with the subject's stage
                # would corrupt training invisibly. So each ROI is cropped
                # individually, in canonical order, and a failure is recorded
                # rather than filled in.
                patches: Dict[str, np.ndarray] = {}
                by_name = {entry.roi_name: entry for entry in result.rois}
                for roi in ROI_ORDER:
                    mask = masks.get(roi)
                    if mask is None:
                        result.warnings.append(
                            f"{roi}: no atlas mask, so no patch was extracted."
                        )
                        continue
                    try:
                        patch, meta = cropper.extract_single(
                            volume, np.asarray(mask), roi, subject_id
                        )
                    except Exception as exc:  # noqa: BLE001
                        result.warnings.append(
                            f"{roi}: patch extraction failed ({exc})."
                        )
                        if roi in by_name:
                            by_name[roi].note = f"patch extraction failed: {exc}"
                        continue
                    patches[roi] = np.asarray(patch, dtype=np.float32)
                    if roi in by_name:
                        by_name[roi].patch_shape = list(patches[roi].shape)
                        if isinstance(meta, dict) and meta.get("bbox"):
                            by_name[roi].note = f"bbox={meta['bbox']}"

                path, tensor = save_patch_tensor(
                    patches, patch_dir, subject_id, self.cfg.patch_size
                )
                result.tensor_path = path.as_posix()
                result.tensor_shape = list(tensor.shape)
                if record is not None:
                    record.record_metric("tensor_shape", list(tensor.shape))
                    record.record_metric("roi_order", list(ROI_ORDER))
                    record.record_artifact("roi_tensor", path)

            if result.n_empty:
                result.warnings.append(
                    f"{result.n_empty} of {N_ROI} ROI(s) were empty. Features "
                    "for those regions will be degenerate and the subject "
                    "should be reviewed before inclusion."
                )
            result.succeeded = True
            return result, patches

        except Exception as exc:  # noqa: BLE001
            logger.error("ROI extraction failed for %s: %s", subject_id, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result, {}


__all__ = [
    "ROIResult",
    "SegmentationResult",
    "save_patch_tensor",
    "ROIPipeline",
]
