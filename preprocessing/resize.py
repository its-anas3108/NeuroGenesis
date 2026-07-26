"""
NeuroGenesis — Spatial Resampling Module
==========================================
Module: preprocessing/resize.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Resamples MRI volumes to a common isotropic spatial resolution using
SimpleITK's high-quality BSpline or linear interpolation.

Spatial standardisation is essential for:
    - Enabling batch processing across OASIS subjects with varying acquisitions
    - Ensuring ROI patch tensors are comparable across subjects
    - Providing consistent input dimensions for NeuroProp-X

The resampling preserves the affine/physical coordinate system so that
downstream ROI masks (generated in standard MNI space) can be correctly
aligned to the resampled volume.

Usage:
    resizer  = MRIResizer(target_shape=(128, 128, 128),
                          output_dir=Path("outputs/processed"))
    resized  = resizer.resample(data, affine, patient_id="OAS1_0001")
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class MRIResizer:
    """
    Resample MRI volumes to a common target shape using SimpleITK.

    SimpleITK is preferred over SciPy/skimage resizing because it:
    - Correctly handles anisotropic voxel spacing
    - Supports BSpline interpolation for sub-voxel accuracy
    - Preserves physical coordinates (spacing, origin, direction)

    Args:
        target_shape : Desired output shape as (X, Y, Z). Default (128, 128, 128).
        interpolator : SimpleITK interpolation mode.
                       'linear'  → sitkLinear (fast, good quality)
                       'bspline' → sitkBSpline (slower, higher quality)
                       'nearest' → sitkNearestNeighbor (for binary masks)
        output_dir   : Directory to save resampled volumes and figures.
    """

    INTERPOLATORS = {
        "linear":  sitk.sitkLinear,
        "bspline": sitk.sitkBSpline,
        "nearest": sitk.sitkNearestNeighbor,
    }

    def __init__(
        self,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        interpolator: str = "linear",
        output_dir: Path = Path("outputs/processed"),
    ) -> None:
        if interpolator not in self.INTERPOLATORS:
            raise ValueError(
                f"Unknown interpolator '{interpolator}'. "
                f"Choose from: {list(self.INTERPOLATORS)}"
            )

        self.target_shape = tuple(target_shape)
        self.interpolator_key = interpolator
        self.interpolator = self.INTERPOLATORS[interpolator]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[MRIResizer] Initialised │ target={self.target_shape} │ "
            f"interpolation={interpolator} │ output={self.output_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def resample(
        self,
        data:       np.ndarray,
        affine:     np.ndarray,
        patient_id: str,
        save:       bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample a 3-D MRI volume to *target_shape*.

        The new voxel spacing is computed to preserve the physical field-of-view:
            new_spacing[i] = old_spacing[i] * old_shape[i] / target_shape[i]

        Args:
            data       : Float32 input array (X, Y, Z).
            affine     : 4×4 affine matrix encoding voxel spacing and origin.
            patient_id : Subject identifier.
            save       : If *True*, save resampled NIfTI and comparison figure.

        Returns:
            Tuple of (resampled_data, new_affine) both as float32/float64 arrays.
        """
        logger.info(
            f"[MRIResizer] Resampling [{patient_id}] "
            f"{data.shape} → {self.target_shape}"
        )

        try:
            from nilearn.image import resample_img

            old_shape = data.shape[:3]
            spacing_orig = tuple(abs(float(affine[i, i])) for i in range(3))
            new_spacing = tuple(
                spacing_orig[i] * old_shape[i] / self.target_shape[i]
                for i in range(3)
            )

            new_affine = affine.copy()
            for i in range(3):
                new_affine[i, i] = np.sign(affine[i, i]) * new_spacing[i]

            img_in = nib.Nifti1Image(data.astype(np.float32), affine)
            interp = "continuous" if self.interpolator_key != "nearest" else "nearest"
            res_img = resample_img(
                img_in,
                target_affine=new_affine,
                target_shape=self.target_shape,
                interpolation=interp,
                force_resample=True,
            )
            resampled_data = res_img.get_fdata().astype(np.float32)

            logger.info(
                f"[MRIResizer] ✓ {patient_id} │ "
                f"{old_shape} → {resampled_data.shape} │ "
                f"spacing {spacing_orig} → {new_spacing} mm"
            )

            if save:
                self._save_nifti(resampled_data, new_affine, patient_id)
                self._save_figure(data, resampled_data, patient_id,
                                  spacing_orig, new_spacing)

            return resampled_data, new_affine

        except Exception as exc:
            logger.error(f"[MRIResizer] Resampling failed for {patient_id}: {exc}")
            raise RuntimeError(f"Resampling failed: {patient_id}") from exc

    def resample_mask(
        self,
        mask:       np.ndarray,
        affine:     np.ndarray,
        patient_id: str,
        roi_name:   str = "mask",
    ) -> np.ndarray:
        """
        Resample a binary ROI mask to *target_shape* using nearest-neighbour.

        Nearest-neighbour interpolation is mandatory for binary masks to
        avoid introducing non-binary values at boundaries.

        Args:
            mask       : Binary float32 mask (X, Y, Z).
            affine     : Affine matrix of the mask volume.
            patient_id : Subject identifier.
            roi_name   : Name of the ROI for log messages.

        Returns:
            Binary float32 mask resampled to *target_shape*.
        """
        orig_interp = self.interpolator
        self.interpolator = sitk.sitkNearestNeighbor
        resampled, _ = self.resample(mask, affine, f"{patient_id}_{roi_name}", save=False)
        self.interpolator = orig_interp
        return (resampled > 0.5).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save_nifti(
        self,
        data:      np.ndarray,
        affine:    np.ndarray,
        patient_id: str,
    ) -> str:
        path = self.output_dir / f"{patient_id}_resampled_{self.target_shape[0]}iso.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        logger.info(f"[MRIResizer] NIfTI saved → {path}")
        return str(path)

    def _save_figure(
        self,
        original:     np.ndarray,
        resampled:    np.ndarray,
        patient_id:   str,
        orig_spacing: tuple,
        new_spacing:  tuple,
    ) -> str:
        """Save a before/after resampling comparison figure."""
        cz_orig = original.shape[2] // 2
        cz_new  = resampled.shape[2] // 2

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ Spatial Resampling │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        panels = [
            (np.rot90(original[:, :, cz_orig]),
             f"Original {original.shape[:3]}\nSpacing {[f'{s:.2f}' for s in orig_spacing]} mm",
             "#8b949e"),
            (np.rot90(resampled[:, :, cz_new]),
             f"Resampled {resampled.shape[:3]}\nSpacing {[f'{s:.2f}' for s in new_spacing]} mm",
             "#3fb950"),
        ]

        for ax, (sl, title, color) in zip(axes, panels):
            im = ax.imshow(sl, cmap="bone", aspect="auto")
            ax.set_title(title, color=color, fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_resampling.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[MRIResizer] Figure saved → {save_path}")
        return str(save_path)
