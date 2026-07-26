"""
NeuroGenesis — Skull Stripping Module
======================================
Module: preprocessing/skull_strip.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Implements brain extraction (skull stripping) using two complementary
approaches with automatic fallback:

    Primary  : Nilearn NiftiMasker with EPI brain mask
    Fallback : SimpleITK Otsu thresholding + morphological cleanup

Both approaches require no external binaries (FSL, FreeSurfer, HD-BET),
making the pipeline portable across systems.

Integration Hook:
    SkullStripper.replace_backend(fn) — Drop in HD-BET or FSL BET
    by providing a callable that accepts (data, affine) → masked_data.

Usage:
    stripper = SkullStripper(output_dir=Path("outputs/processed"))
    masked   = stripper.strip(data, patient_id="OAS1_0001", affine=affine)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


class SkullStripper:
    """
    Brain extraction (skull stripping) with pluggable backends.

    The default pipeline uses a two-stage approach:
    1. Otsu thresholding to create an initial brain mask
    2. Morphological closing + largest-connected-component selection
       to clean up the mask

    A Nilearn-based approach is attempted first if nilearn is available,
    as it tends to produce more anatomically accurate masks.

    Args:
        output_dir       : Directory for saving stripped volumes and figures.
        morph_radius     : Radius (voxels) for morphological close operation.
        opening_radius   : Radius for morphological opening (noise removal).
    """

    def __init__(
        self,
        output_dir:     Path,
        morph_radius:   int = 4,
        opening_radius: int = 2,
    ) -> None:
        self.output_dir     = Path(output_dir)
        self.morph_radius   = morph_radius
        self.opening_radius = opening_radius

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Custom backend hook — None means use built-in logic
        self._custom_backend: Optional[Callable] = None

        logger.info(
            f"[SkullStripper] Initialised │ output={self.output_dir} │ "
            f"morph_radius={morph_radius}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — main interface
    # ──────────────────────────────────────────────────────────────────────────

    def strip(
        self,
        data:       np.ndarray,
        patient_id: str,
        affine:     np.ndarray,
        save:       bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform skull stripping on a 3-D MRI volume.

        Args:
            data       : Float32 MRI array (X, Y, Z).
            patient_id : Subject identifier.
            affine     : Affine matrix for saving NIfTI outputs.
            save       : Save stripped volume and figure to *output_dir*.

        Returns:
            Tuple of (masked_data, binary_brain_mask) both float32 arrays.
        """
        logger.info(f"[SkullStripper] Starting skull strip for [{patient_id}]")

        # ── Custom backend override ────────────────────────────────────────
        if self._custom_backend is not None:
            logger.info("[SkullStripper] Using custom backend.")
            masked, mask = self._custom_backend(data, affine)
            self._post_process(data, masked, mask, patient_id, affine, save)
            return masked, mask

        # ── Try Nilearn first ──────────────────────────────────────────────
        masked, mask = self._strip_with_nilearn(data, patient_id)

        if masked is None:
            logger.info("[SkullStripper] Nilearn unavailable — using SimpleITK fallback.")
            masked, mask = self._strip_with_sitk(data)

        if masked is None:
            logger.error(
                f"[SkullStripper] Both backends failed for {patient_id}. "
                "Returning original data."
            )
            return data.copy(), np.ones_like(data, dtype=np.float32)

        self._post_process(data, masked, mask, patient_id, affine, save)
        logger.info(f"[SkullStripper] ✓ Skull strip complete for [{patient_id}]")

        return masked, mask

    def replace_backend(self, backend_fn: Callable) -> None:
        """
        Replace the built-in skull stripping with a custom backend.

        This hook is designed for seamless integration of HD-BET, FSL BET,
        or any other skull stripping tool in Phase 2.

        Args:
            backend_fn : Callable with signature:
                         ``(data: np.ndarray, affine: np.ndarray)
                           → (masked: np.ndarray, mask: np.ndarray)``

        Example::

            def hd_bet_backend(data, affine):
                # Call HD-BET via subprocess or API
                ...
                return masked_data, binary_mask

            stripper.replace_backend(hd_bet_backend)

        .. note::
            Set to ``None`` to revert to the built-in backends.
        """
        self._custom_backend = backend_fn
        logger.info(
            f"[SkullStripper] Custom backend registered: {getattr(backend_fn, '__name__', str(backend_fn))}"
        )

    def clear_backend(self) -> None:
        """Remove any custom backend and revert to built-in logic."""
        self._custom_backend = None
        logger.info("[SkullStripper] Reverted to built-in backend.")

    # ──────────────────────────────────────────────────────────────────────────
    # Private — Nilearn backend
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_with_nilearn(
        self,
        data:       np.ndarray,
        patient_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Attempt skull stripping using Nilearn's brain masking utilities.

        Uses ``nilearn.masking.compute_brain_mask`` on a temporary NIfTI
        image. Falls back gracefully if nilearn is not installed.

        Args:
            data       : Float32 array (X, Y, Z).
            patient_id : For log messages.

        Returns:
            Tuple of (masked_data, mask) or (None, None) on failure.
        """
        try:
            from nilearn import masking as nlm
            from nilearn.image import new_img_like
            import tempfile, os

            # Build a temporary NIfTI for nilearn
            affine_eye = np.eye(4)
            tmp_img = nib.Nifti1Image(data, affine_eye)

            brain_mask = nlm.compute_brain_mask(tmp_img, threshold=0.2)
            mask_data = brain_mask.get_fdata().astype(np.float32)

            # Ensure shape matches and mask is non-empty
            if mask_data.shape != data.shape or mask_data.sum() < 500:
                logger.warning(
                    f"[SkullStripper] Nilearn mask invalid or empty ({mask_data.sum()} voxels). "
                    "Using SimpleITK fallback."
                )
                return None, None

            masked = (data * mask_data).astype(np.float32)
            logger.debug(
                f"[SkullStripper] Nilearn: {int(mask_data.sum())} brain voxels "
                f"({mask_data.mean()*100:.1f}% of volume) for {patient_id}"
            )
            return masked, mask_data

        except ImportError:
            logger.debug("[SkullStripper] nilearn not available.")
            return None, None
        except Exception as exc:
            logger.warning(f"[SkullStripper] Nilearn skull strip failed: {exc}")
            return None, None

    # ──────────────────────────────────────────────────────────────────────────
    # Private — SimpleITK backend (fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _strip_with_sitk(
        self,
        data: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Skull strip using SimpleITK Otsu threshold + morphological cleanup.

        Algorithm:
            1. Otsu thresholding → binary brain/background mask
            2. Binary morphological closing (fills holes inside brain)
            3. Morphological opening (removes small noise fragments)
            4. Largest connected component selection
            5. Apply mask to data

        Args:
            data : Float32 MRI array (X, Y, Z).

        Returns:
            Tuple of (masked_data, mask) or (None, None) on failure.
        """
        try:
            sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
            sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

            # Otsu threshold
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(0)
            otsu.SetOutsideValue(1)
            mask_sitk = otsu.Execute(sitk_img)
            mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)

            # Morphological close (fill holes)
            close_filter = sitk.BinaryMorphologicalClosingImageFilter()
            close_filter.SetKernelRadius(self.morph_radius)
            close_filter.SetForegroundValue(1)
            mask_sitk = close_filter.Execute(mask_sitk)

            # Morphological open (remove debris)
            open_filter = sitk.BinaryMorphologicalOpeningImageFilter()
            open_filter.SetKernelRadius(self.opening_radius)
            open_filter.SetForegroundValue(1)
            mask_sitk = open_filter.Execute(mask_sitk)

            # Keep only the largest connected component
            cc_filter = sitk.ConnectedComponentImageFilter()
            cc_img = cc_filter.Execute(mask_sitk)
            label_map = sitk.RelabelComponentImageFilter()
            label_map.SetMinimumObjectSize(500)
            cc_clean = label_map.Execute(cc_img)
            mask_sitk = sitk.BinaryThreshold(cc_clean, 1, 1, 1, 0)

            mask_data = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)
            masked = (data * mask_data).astype(np.float32)

            logger.debug(
                f"[SkullStripper] SimpleITK: {int(mask_data.sum())} brain voxels "
                f"({mask_data.mean()*100:.1f}% of volume)"
            )
            return masked, mask_data

        except Exception as exc:
            logger.error(f"[SkullStripper] SimpleITK backend failed: {exc}")
            return None, None

    # ──────────────────────────────────────────────────────────────────────────
    # Private — post-processing (save + visualise)
    # ──────────────────────────────────────────────────────────────────────────

    def _post_process(
        self,
        original:   np.ndarray,
        masked:     np.ndarray,
        mask:       np.ndarray,
        patient_id: str,
        affine:     np.ndarray,
        save:       bool,
    ) -> None:
        """Save stripped NIfTI and comparison figure."""
        if not save:
            return

        # Save NIfTI
        nib.save(
            nib.Nifti1Image(masked, affine),
            str(self.output_dir / f"{patient_id}_skull_stripped.nii.gz")
        )
        nib.save(
            nib.Nifti1Image(mask, affine),
            str(self.output_dir / f"{patient_id}_brain_mask.nii.gz")
        )

        # Save figure
        self._save_figure(original, masked, mask, patient_id)

    def _save_figure(
        self,
        original:   np.ndarray,
        masked:     np.ndarray,
        mask:       np.ndarray,
        patient_id: str,
    ) -> str:
        """Generate and save a skull stripping before/after figure."""
        cz = original.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ Skull Stripping │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        panels = [
            (np.rot90(original[:, :, cz]), "Original MRI",    "bone"),
            (np.rot90(masked[:, :, cz]),   "Skull Stripped",  "bone"),
            (np.rot90(mask[:, :, cz]),     "Brain Mask",      "Greens"),
        ]
        colors = ["#8b949e", "#3fb950", "#58a6ff"]

        for ax, (sl, title, cmap), color in zip(axes, panels, colors):
            im = ax.imshow(sl, cmap=cmap, aspect="auto")
            ax.set_title(title, color=color, fontsize=11, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_skull_stripping.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[SkullStripper] Figure saved → {save_path}")
        return str(save_path)
