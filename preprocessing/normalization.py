"""
NeuroGenesis — Novel MRI Normalization Pipeline
================================================
Module: preprocessing/normalization.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Novel Contributions (5 techniques beyond standard pipelines):

    1. N4 Bias Field Correction
       Removes smooth intensity gradients introduced by MRI scanner RF coils.
       Uses the industry-standard N4ITK algorithm (SimpleITK backend).
       Critical for accurate tissue segmentation and inter-subject comparison.

    2. White Matter Peak Normalization
       Normalises each scan to its WM intensity peak (estimated via KDE on
       the intensity histogram) rather than global min-max. This makes
       intensities biologically comparable across sessions and subjects —
       essential for tracking atrophy over time.

    3. CLAHE (Contrast-Limited Adaptive Histogram Equalisation)
       Applied slice-by-slice along the axial axis with adaptive tile grids.
       Enhances local tissue contrast inside speech ROIs without global
       intensity distortion. Rarely applied on volumetric MRI in the
       literature.

    4. Anisotropic Diffusion Filtering
       Edge-preserving denoising using the Perona-Malik model. Unlike
       Gaussian smoothing, this preserves anatomical boundaries at Broca/
       Wernicke area borders while removing thermal noise — critical for
       clean ROI crop patches.

    5. Gaussian Smoothing (baseline comparison)
       Standard Gaussian lowpass filter included as a baseline against
       which anisotropic diffusion is compared in visualisations.

Usage:
    normalizer = MRINormalizer(output_dir=Path("outputs/processed"))
    result     = normalizer.run_full_pipeline(data, patient_id="OAS1_0001",
                                              affine=affine)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from skimage import exposure

logger = logging.getLogger(__name__)


class MRINormalizer:
    """
    Multi-stage MRI intensity normalization and noise reduction.

    Each stage generates a before/after figure for visual verification.
    All intermediate NIfTI volumes are optionally saved to disk.

    Args:
        output_dir       : Directory where processed volumes and figures are saved.
        clahe_clip_limit : Clip limit for CLAHE contrast enhancement (0–1).
        clahe_nbins      : Number of histogram bins for CLAHE.
        gaussian_sigma   : Standard deviation for Gaussian smoothing (mm).
        aniso_iterations : Gradient anisotropic diffusion iterations.
        aniso_conductance: Conductance parameter for anisotropic diffusion.
        aniso_time_step  : Time step for anisotropic diffusion solver.
    """

    def __init__(
        self,
        output_dir: Path,
        clahe_clip_limit:  float = 0.03,
        clahe_nbins:       int   = 256,
        gaussian_sigma:    float = 1.0,
        aniso_iterations:  int   = 10,
        aniso_conductance: float = 3.0,
        aniso_time_step:   float = 0.0625,
    ) -> None:
        self.output_dir       = Path(output_dir)
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_nbins      = clahe_nbins
        self.gaussian_sigma   = gaussian_sigma
        self.aniso_iterations = aniso_iterations
        self.aniso_conductance= aniso_conductance
        self.aniso_time_step  = aniso_time_step

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"[MRINormalizer] Initialised │ output={self.output_dir} │ "
            f"gaussian_sigma={gaussian_sigma} │ aniso_iter={aniso_iterations}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — full pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        data:       np.ndarray,
        patient_id: str,
        affine:     np.ndarray,
        save_nifti: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the complete normalisation pipeline in sequence:

            N4 Bias Correction → WM Peak Normalisation → CLAHE →
            Anisotropic Diffusion → Min-Max Normalisation

        Args:
            data       : Raw float32 MRI array (X, Y, Z).
            patient_id : Subject identifier for file naming.
            affine     : Affine matrix (preserved in saved NIfTI files).
            save_nifti : If *True*, save each stage as a .nii.gz file.

        Returns:
            Dictionary with keys corresponding to each processed stage:
                ``n4``, ``wm_norm``, ``clahe``, ``aniso``, ``final``
        """
        logger.info(f"[MRINormalizer] ▶ Full pipeline started for [{patient_id}]")
        results: Dict[str, Any] = {"patient_id": patient_id, "stages": {}}

        # ── Stage 1: N4 Bias Field Correction ─────────────────────────────
        logger.info(f"[MRINormalizer]   Stage 1/5 — N4 Bias Field Correction")
        n4_data = self.apply_n4_bias_correction(data, patient_id)
        results["stages"]["n4"] = n4_data
        self._save_comparison_figure(data, n4_data, patient_id,
                                     "N4 Bias Field Correction",
                                     "Raw", "N4 Corrected", "RdBu_r", "gray")
        if save_nifti:
            self._save_nifti(n4_data, affine, patient_id, "01_n4_corrected")

        # ── Stage 2: WM Peak Normalisation ────────────────────────────────
        logger.info(f"[MRINormalizer]   Stage 2/5 — White Matter Peak Normalisation")
        wm_data, wm_peak = self.normalize_wm_peak(n4_data, patient_id)
        results["stages"]["wm_norm"] = wm_data
        results["wm_peak"] = wm_peak
        self._save_comparison_figure(n4_data, wm_data, patient_id,
                                     "WM-Peak Normalisation",
                                     "N4 Corrected", f"WM-Norm (peak={wm_peak:.0f})",
                                     "gray", "gray")
        if save_nifti:
            self._save_nifti(wm_data, affine, patient_id, "02_wm_norm")

        # ── Stage 3: CLAHE ────────────────────────────────────────────────
        logger.info(f"[MRINormalizer]   Stage 3/5 — CLAHE Enhancement")
        clahe_data = self.apply_clahe(wm_data, patient_id)
        results["stages"]["clahe"] = clahe_data
        self._save_comparison_figure(wm_data, clahe_data, patient_id,
                                     "CLAHE Contrast Enhancement",
                                     "WM-Norm", "CLAHE Enhanced", "gray", "gray")
        if save_nifti:
            self._save_nifti(clahe_data, affine, patient_id, "03_clahe")

        # ── Stage 4: Anisotropic Diffusion ────────────────────────────────
        logger.info(f"[MRINormalizer]   Stage 4/5 — Anisotropic Diffusion Filtering")
        aniso_data = self.apply_anisotropic_diffusion(clahe_data, patient_id)
        results["stages"]["aniso"] = aniso_data

        # Also compute Gaussian for comparison figure
        gauss_data = self.apply_gaussian_smoothing(clahe_data)
        self._save_triway_comparison(
            clahe_data, gauss_data, aniso_data, patient_id,
            "Denoising Comparison",
            "CLAHE Input", f"Gaussian (σ={self.gaussian_sigma})",
            f"Anisotropic Diffusion ({self.aniso_iterations} iter)"
        )
        if save_nifti:
            self._save_nifti(aniso_data, affine, patient_id, "04_aniso_diffusion")

        # ── Stage 5: Final Min-Max Normalisation ──────────────────────────
        logger.info(f"[MRINormalizer]   Stage 5/5 — Min-Max Normalisation [0, 1]")
        final_data = self.normalize_minmax(aniso_data)
        results["stages"]["final"] = final_data
        results["final"] = final_data
        self._save_comparison_figure(aniso_data, final_data, patient_id,
                                     "Final Min-Max Normalisation [0,1]",
                                     "Aniso Filtered", "Normalised [0,1]",
                                     "gray", "gray")
        if save_nifti:
            self._save_nifti(final_data, affine, patient_id, "05_final_normalized")

        # ── Intensity histogram comparison ────────────────────────────────
        self._save_intensity_histograms(
            data, n4_data, wm_data, clahe_data, aniso_data, final_data, patient_id
        )

        logger.info(
            f"[MRINormalizer] ✓ Pipeline complete for [{patient_id}] │ "
            f"output={self.output_dir}"
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Stage methods (public — can be called independently)
    # ──────────────────────────────────────────────────────────────────────────

    def apply_n4_bias_correction(
        self,
        data:       np.ndarray,
        patient_id: str = "unknown",
    ) -> np.ndarray:
        """
        Apply N4 Bias Field Correction using SimpleITK.

        The N4ITK algorithm iteratively estimates and removes the smooth
        multiplicative bias field introduced by MRI scanner RF coils.

        Args:
            data       : Input float32 MRI array (X, Y, Z).
            patient_id : Used for log messages.

        Returns:
            N4-corrected float32 array of the same shape.
        """
        try:
            sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
            sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

            # Compute brain mask (non-zero voxels) for N4 fitting
            mask = sitk.OtsuThreshold(sitk_img, 0, 1, 200)

            n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
            n4_filter.SetMaximumNumberOfIterations([50, 50, 50, 50])
            n4_filter.SetConvergenceThreshold(0.001)

            corrected = n4_filter.Execute(sitk_img, mask)
            result = sitk.GetArrayFromImage(corrected).astype(np.float32)

            logger.debug(f"[MRINormalizer] N4 complete for {patient_id}")
            return result

        except Exception as exc:
            logger.warning(
                f"[MRINormalizer] N4 failed for {patient_id} ({exc}). "
                "Returning original data."
            )
            return data.copy()

    def normalize_wm_peak(
        self,
        data:       np.ndarray,
        patient_id: str = "unknown",
    ) -> Tuple[np.ndarray, float]:
        """
        Normalise intensity using the White Matter peak as reference.

        Estimates the WM peak as the dominant mode of the intensity histogram
        for voxels in the upper 40–90% of intensity range (WM territory),
        using Gaussian KDE. The volume is then divided by this peak so that
        WM intensities are approximately unity across subjects.

        Args:
            data       : Input float32 array.
            patient_id : Used for log messages.

        Returns:
            Tuple of (WM-normalised array, WM peak intensity value).
        """
        try:
            nonzero = data[data > 0].flatten()

            # Restrict to WM intensity territory (40th–90th percentile)
            p40, p90 = np.percentile(nonzero, [40, 90])
            wm_range = nonzero[(nonzero >= p40) & (nonzero <= p90)]

            if len(wm_range) < 100:
                wm_peak = float(np.median(nonzero))
                logger.debug(f"[MRINormalizer] KDE failed for {patient_id} — using median")
            else:
                # Gaussian KDE on WM range
                kde = gaussian_kde(wm_range, bw_method="silverman")
                x_eval = np.linspace(wm_range.min(), wm_range.max(), 500)
                wm_peak = float(x_eval[np.argmax(kde(x_eval))])

            normalised = data / (wm_peak + 1e-10)
            normalised = normalised.astype(np.float32)

            logger.debug(
                f"[MRINormalizer] WM peak for {patient_id}: {wm_peak:.2f} → "
                f"normalised range [{normalised.min():.3f}, {normalised.max():.3f}]"
            )
            return normalised, wm_peak

        except Exception as exc:
            logger.warning(
                f"[MRINormalizer] WM normalisation failed ({exc}). "
                "Falling back to min-max."
            )
            return self.normalize_minmax(data), 1.0

    def apply_clahe(
        self,
        data:       np.ndarray,
        patient_id: str = "unknown",
    ) -> np.ndarray:
        """
        Apply CLAHE slice-by-slice along the axial axis.

        Each axial slice is enhanced independently with adaptive histogram
        equalisation to boost local tissue contrast without global distortion.

        Args:
            data       : Float32 array (X, Y, Z). Expected in [0, 1].
            patient_id : Used for log messages.

        Returns:
            CLAHE-enhanced float32 array of the same shape.
        """
        try:
            # Ensure data is in [0, 1] before CLAHE
            data_01 = self.normalize_minmax(data)
            result = np.zeros_like(data_01)

            for z in range(data_01.shape[2]):
                sl = data_01[:, :, z]
                result[:, :, z] = exposure.equalize_adapthist(
                    sl,
                    clip_limit=self.clahe_clip_limit,
                    nbins=self.clahe_nbins,
                )

            result = result.astype(np.float32)
            logger.debug(f"[MRINormalizer] CLAHE applied ({data_01.shape[2]} slices) for {patient_id}")
            return result

        except Exception as exc:
            logger.warning(
                f"[MRINormalizer] CLAHE failed for {patient_id} ({exc}). "
                "Returning WM-normalised data."
            )
            return data.copy()

    def apply_anisotropic_diffusion(
        self,
        data:       np.ndarray,
        patient_id: str = "unknown",
    ) -> np.ndarray:
        """
        Apply Gradient Anisotropic Diffusion (Perona-Malik) via SimpleITK.

        Unlike Gaussian smoothing, anisotropic diffusion diffuses noise within
        homogeneous regions while stopping at intensity edges (anatomical
        boundaries). This is critical for maintaining sharp ROI boundaries.

        Args:
            data       : Input float32 array (X, Y, Z).
            patient_id : Used for log messages.

        Returns:
            Smoothed float32 array with preserved edges.
        """
        try:
            sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
            sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

            filter_ = sitk.GradientAnisotropicDiffusionImageFilter()
            filter_.SetNumberOfIterations(self.aniso_iterations)
            filter_.SetConductanceParameter(self.aniso_conductance)
            filter_.SetTimeStep(self.aniso_time_step)

            result_sitk = filter_.Execute(sitk_img)
            result = sitk.GetArrayFromImage(result_sitk).astype(np.float32)

            logger.debug(
                f"[MRINormalizer] Anisotropic diffusion done "
                f"({self.aniso_iterations} iter) for {patient_id}"
            )
            return result

        except Exception as exc:
            logger.warning(
                f"[MRINormalizer] Anisotropic diffusion failed for {patient_id} ({exc}). "
                "Falling back to Gaussian."
            )
            return self.apply_gaussian_smoothing(data)

    def apply_gaussian_smoothing(
        self, data: np.ndarray
    ) -> np.ndarray:
        """
        Apply isotropic Gaussian smoothing via SciPy.

        Used as a baseline comparator against anisotropic diffusion and
        as a fallback when SimpleITK is unavailable.

        Args:
            data : Input float32 array.

        Returns:
            Gaussian-smoothed float32 array.
        """
        return gaussian_filter(data, sigma=self.gaussian_sigma).astype(np.float32)

    def normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """
        Normalise intensity values to the range [0, 1].

        Args:
            data : Input float32 array.

        Returns:
            Min-max normalised float32 array.
        """
        vmin, vmax = float(data.min()), float(data.max())
        if vmax - vmin < 1e-10:
            logger.warning("[MRINormalizer] Uniform volume — min-max undefined; returning zeros.")
            return np.zeros_like(data)
        return ((data - vmin) / (vmax - vmin)).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Private — figure helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save_comparison_figure(
        self,
        before:     np.ndarray,
        after:      np.ndarray,
        patient_id: str,
        stage_name: str,
        label_before: str,
        label_after:  str,
        cmap_before:  str = "gray",
        cmap_after:   str = "gray",
    ) -> str:
        """Generate and save a before/after comparison figure for one stage."""
        cz = before.shape[2] // 2
        sl_before = np.rot90(before[:, :, cz])
        sl_after  = np.rot90(after[:, :, cz])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ {stage_name} │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        # Before
        im0 = axes[0].imshow(sl_before, cmap=cmap_before, aspect="auto")
        axes[0].set_title(f"Before: {label_before}", color="#58a6ff", fontsize=10)
        axes[0].axis("off")
        plt.colorbar(im0, ax=axes[0], fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        # After
        im1 = axes[1].imshow(sl_after, cmap=cmap_after, aspect="auto")
        axes[1].set_title(f"After: {label_after}", color="#3fb950", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        # Difference map
        diff = np.abs(sl_after.astype(np.float32) - sl_before.astype(np.float32))
        im2 = axes[2].imshow(diff, cmap="hot", aspect="auto")
        axes[2].set_title("Absolute Difference", color="#e3b341", fontsize=10)
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        plt.tight_layout()
        safe_name = stage_name.lower().replace(" ", "_")
        save_path = self.output_dir / f"{patient_id}_{safe_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.debug(f"[MRINormalizer] Figure saved → {save_path}")
        return str(save_path)

    def _save_triway_comparison(
        self,
        original:   np.ndarray,
        gauss:      np.ndarray,
        aniso:      np.ndarray,
        patient_id: str,
        stage_name: str,
        label_a:    str,
        label_b:    str,
        label_c:    str,
    ) -> str:
        """Generate a three-way comparison: original, Gaussian, Anisotropic."""
        cz = original.shape[2] // 2
        slices = [
            (np.rot90(original[:, :, cz]), label_a, "gray"),
            (np.rot90(gauss[:, :, cz]),    label_b, "gray"),
            (np.rot90(aniso[:, :, cz]),    label_c, "gray"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ {stage_name} │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        palette = ["#8b949e", "#58a6ff", "#3fb950"]
        for ax, (sl, label, cmap), color in zip(axes, slices, palette):
            im = ax.imshow(sl, cmap=cmap, aspect="auto")
            ax.set_title(label, color=color, fontsize=10)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.04, shrink=0.8).ax.tick_params(colors="white")

        plt.tight_layout()
        safe_name = stage_name.lower().replace(" ", "_")
        save_path = self.output_dir / f"{patient_id}_{safe_name}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        return str(save_path)

    def _save_intensity_histograms(
        self,
        raw:    np.ndarray,
        n4:     np.ndarray,
        wm:     np.ndarray,
        clahe:  np.ndarray,
        aniso:  np.ndarray,
        final:  np.ndarray,
        patient_id: str,
    ) -> str:
        """Save a multi-panel intensity histogram comparison across all stages."""
        stages = [
            (raw,   "Raw",        "#8b949e"),
            (n4,    "N4 Corrected","#58a6ff"),
            (wm,    "WM-Norm",    "#d2a8ff"),
            (clahe, "CLAHE",      "#e3b341"),
            (aniso, "Anisotropic","#3fb950"),
            (final, "Final [0,1]","#ff7b72"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8), facecolor="#0d1117")
        axes = axes.flatten()
        fig.suptitle(
            f"NeuroGenesis │ Intensity Histogram Pipeline │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        for ax, (vol, label, color) in zip(axes, stages):
            vals = vol[vol > 0].flatten()
            ax.hist(vals, bins=128, color=color, alpha=0.8, density=True)
            ax.set_facecolor("#161b22")
            ax.set_title(label, color=color, fontsize=11)
            ax.tick_params(colors="white", labelsize=7)
            ax.set_xlabel("Intensity", color="#8b949e", fontsize=8)
            ax.set_ylabel("Density", color="#8b949e", fontsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_intensity_histograms.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[MRINormalizer] Histogram figure saved → {save_path}")
        return str(save_path)

    def _save_nifti(
        self,
        data:      np.ndarray,
        affine:    np.ndarray,
        patient_id: str,
        stage_tag:  str,
    ) -> str:
        """Save a processed volume as a NIfTI file."""
        img = nib.Nifti1Image(data, affine)
        path = self.output_dir / f"{patient_id}_{stage_tag}.nii.gz"
        nib.save(img, str(path))
        logger.debug(f"[MRINormalizer] NIfTI saved → {path}")
        return str(path)
