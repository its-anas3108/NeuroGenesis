"""
NeuroGenesis — Speech ROI Extraction
======================================
Module: segmentation/roi_extraction.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Extracts five speech-related brain regions of interest (ROIs) using the
Harvard-Oxford cortical atlas (via Nilearn). The atlas is automatically
downloaded on first use and cached locally.

Extracted Speech ROIs:
    1. Broca Area        — IFG pars triangularis + pars opercularis
       (Brodmann areas 44, 45 — speech production)
    2. Wernicke Area     — Posterior superior temporal gyrus
       (Brodmann area 22 — speech comprehension)
    3. Insula            — Insular cortex
       (Articulatory planning, phonological processing)
    4. Inferior Frontal Gyrus (IFG) — Full IFG complex
       (Syntactic processing)
    5. Superior Temporal Gyrus (STG) — Full STG
       (Auditory-verbal processing)

For each ROI the module generates:
    - Binary mask (NIfTI)
    - Overlay on MRI
    - Region statistics (volume, voxel count, mean intensity)

The ROI-Centric Pipeline feeds these masks directly to ROICropper.

Usage:
    extractor = ROIExtractor(output_dir=Path("outputs/segmented"))
    masks     = extractor.extract_all(data, affine, patient_id="OAS1_0001")
    # masks → dict: roi_name → binary float32 mask array
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ROI definitions: label substrings that identify each region in the atlas
# ──────────────────────────────────────────────────────────────────────────────

ROI_DEFINITIONS: Dict[str, List[str]] = {
    "Broca_Area": [
        "inferior frontal gyrus, pars triangularis",
        "inferior frontal gyrus, pars opercularis",
    ],
    "Wernicke_Area": [
        "superior temporal gyrus, posterior division",
        "planum temporale",
    ],
    "Insula": [
        "insular cortex",
    ],
    "Inferior_Frontal_Gyrus": [
        "inferior frontal gyrus",
        "frontal opercular cortex",
    ],
    "Superior_Temporal_Gyrus": [
        "superior temporal gyrus",
        "heschl",
        "planum polare",
    ],
}

# Display colours for visualisation (RGB 0–1)
ROI_COLORS: Dict[str, Tuple[float, float, float]] = {
    "Broca_Area":               (0.345, 0.647, 1.000),   # blue
    "Wernicke_Area":            (0.247, 0.725, 0.314),   # green
    "Insula":                   (0.886, 0.706, 0.255),   # amber
    "Inferior_Frontal_Gyrus":   (0.847, 0.329, 0.329),   # red
    "Superior_Temporal_Gyrus":  (0.671, 0.329, 0.847),   # purple
}


class ROIExtractor:
    """
    Atlas-based extraction of speech-related brain ROIs.

    Uses the Nilearn Harvard-Oxford cortical atlas. The atlas is
    probabilistic; this class thresholds it at 25% probability to
    obtain binary masks, then resamples to the subject's MRI space.

    Args:
        output_dir    : Directory for masks, overlays, and statistics.
        atlas_name    : Nilearn atlas identifier. Default is the 25% threshold
                        maximum probability map at 2 mm resolution.
        combine_bilateral : If *True*, combine left and right hemisphere labels.
    """

    def __init__(
        self,
        output_dir: Path,
        atlas_name: str = "cort-maxprob-thr25-2mm",
        combine_bilateral: bool = True,
        data_dir: Optional[Path] = None,
    ) -> None:
        self.output_dir        = Path(output_dir)
        self.atlas_name        = atlas_name
        self.combine_bilateral = combine_bilateral
        self.data_dir          = Path(data_dir) if data_dir is not None else (Path.cwd() / "dataset" / "nilearn_data")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._atlas_img:    Optional[Any] = None
        self._atlas_data:   Optional[np.ndarray] = None
        self._atlas_labels: Optional[List[str]] = None

        logger.info(
            f"[ROIExtractor] Initialised │ atlas={atlas_name} │ "
            f"output={self.output_dir} │ data_dir={self.data_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def load_atlas(self) -> None:
        """
        Fetch and cache the Harvard-Oxford cortical atlas via Nilearn.

        Downloads automatically on first call; subsequent calls use the
        Nilearn cache. Requires an active internet connection
        on first run.

        Raises:
            ImportError : If nilearn is not installed.
            RuntimeError: If atlas cannot be fetched.
        """
        try:
            from nilearn import datasets as nlds
        except ImportError as exc:
            raise ImportError(
                "nilearn is required for ROI extraction. "
                "Install with: pip install nilearn"
            ) from exc

        logger.info(f"[ROIExtractor] Loading Harvard-Oxford atlas ({self.atlas_name})…")
        try:
            import ssl
            try:
                ssl._create_default_https_context = ssl._create_unverified_context
            except AttributeError:
                pass

            atlas = nlds.fetch_atlas_harvard_oxford(self.atlas_name, data_dir=str(self.data_dir.resolve()))
            maps = atlas["maps"]
            # Newer nilearn versions return a Nifti1Image directly;
            # older versions return a file-path string.
            if isinstance(maps, nib.Nifti1Image):
                self._atlas_img = maps
            else:
                self._atlas_img = nib.load(maps)
            self._atlas_data   = np.asarray(self._atlas_img.dataobj, dtype=np.int32)
            self._atlas_labels = atlas["labels"]
            logger.info(
                f"[ROIExtractor] Atlas loaded │ "
                f"{len(self._atlas_labels)} regions │ "
                f"shape={self._atlas_data.shape}"
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load atlas: {exc}") from exc

    def extract_all(
        self,
        data:       np.ndarray,
        affine:     np.ndarray,
        patient_id: str,
        save:       bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract all five speech-related ROI masks for a subject.

        The atlas is resampled to the subject's MRI space before extraction.

        Args:
            data       : Preprocessed float32 MRI array (X, Y, Z).
            affine     : Subject MRI affine matrix.
            patient_id : Subject identifier.
            save       : Save masks and visualisations to *output_dir*.

        Returns:
            Dict mapping ROI name → binary float32 mask (same shape as data).
        """
        if self._atlas_img is None:
            self.load_atlas()

        logger.info(f"[ROIExtractor] Extracting ROIs for [{patient_id}]")

        # Resample atlas to subject space
        atlas_resampled = self._resample_atlas_to_subject(data, affine)

        masks: Dict[str, np.ndarray] = {}
        stats: Dict[str, Dict] = {}

        for roi_name, label_patterns in ROI_DEFINITIONS.items():
            try:
                mask = self._build_roi_mask(atlas_resampled, label_patterns)
                masks[roi_name] = mask
                stats[roi_name] = self._compute_roi_stats(mask, data, roi_name, affine)

                logger.info(
                    f"[ROIExtractor]   ✓ {roi_name:28s} │ "
                    f"voxels={int(mask.sum()):6d} │ "
                    f"vol={stats[roi_name]['volume_mm3']:.1f} mm³"
                )

                if save:
                    self._save_mask_nifti(mask, affine, patient_id, roi_name)

            except Exception as exc:
                logger.warning(
                    f"[ROIExtractor]   ✗ Could not extract {roi_name}: {exc}"
                )
                masks[roi_name] = np.zeros(data.shape[:3], dtype=np.float32)
                stats[roi_name] = {"error": str(exc)}

        if save:
            self._save_all_overlays(data, masks, patient_id)
            self._save_combined_overlay(data, masks, patient_id)
            self._save_stats_figure(stats, patient_id)

        return masks

    def extract_single(
        self,
        data:          np.ndarray,
        affine:        np.ndarray,
        patient_id:    str,
        roi_name:      str,
        save:          bool = True,
    ) -> np.ndarray:
        """
        Extract a single named ROI mask.

        Args:
            data       : Preprocessed MRI float32 array.
            affine     : Affine matrix.
            patient_id : Subject identifier.
            roi_name   : Must be one of the keys in ROI_DEFINITIONS.
            save       : Save mask and overlay figure.

        Returns:
            Binary float32 mask array (same shape as data).

        Raises:
            KeyError: If *roi_name* is not in ROI_DEFINITIONS.
        """
        if roi_name not in ROI_DEFINITIONS:
            raise KeyError(
                f"Unknown ROI '{roi_name}'. "
                f"Available: {list(ROI_DEFINITIONS.keys())}"
            )

        if self._atlas_img is None:
            self.load_atlas()

        atlas_resampled = self._resample_atlas_to_subject(data, affine)
        mask = self._build_roi_mask(atlas_resampled, ROI_DEFINITIONS[roi_name])

        if save:
            self._save_mask_nifti(mask, affine, patient_id, roi_name)

        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # Private — mask building
    # ──────────────────────────────────────────────────────────────────────────

    def _resample_atlas_to_subject(
        self,
        data:   np.ndarray,
        affine: np.ndarray,
    ) -> np.ndarray:
        """
        Resample the atlas label volume to the subject MRI space.

        Uses nearest-neighbour interpolation to preserve integer labels.

        Args:
            data   : Subject MRI array (target shape).
            affine : Subject affine matrix (target space).

        Returns:
            Integer label array of shape data.shape[:3].
        """
        try:
            from nilearn.image import resample_to_img
        except ImportError:
            # Fallback: use scipy zoom (less accurate but works without nilearn)
            return self._resample_atlas_scipy(data)

        # Create a temporary target image from the subject data
        subject_img  = nib.Nifti1Image(data, affine)
        atlas_img_rs = resample_to_img(
            self._atlas_img,
            subject_img,
            interpolation="nearest",
            copy=True,
        )
        return np.asarray(atlas_img_rs.dataobj, dtype=np.int32)

    def _resample_atlas_scipy(self, data: np.ndarray) -> np.ndarray:
        """Fallback atlas resampling using scipy zoom (nearest-neighbour)."""
        from scipy.ndimage import zoom as spzoom
        atlas = self._atlas_data
        zoom_factors = tuple(
            data.shape[i] / atlas.shape[i] for i in range(3)
        )
        resampled = spzoom(atlas, zoom_factors, order=0)  # order=0 → nearest
        # Crop or pad to exact target shape
        result = np.zeros(data.shape[:3], dtype=np.int32)
        slices = tuple(slice(0, min(resampled.shape[i], data.shape[i])) for i in range(3))
        result[slices] = resampled[slices]
        return result

    def _build_roi_mask(
        self,
        atlas_resampled: np.ndarray,
        label_patterns:  List[str],
    ) -> np.ndarray:
        """
        Build a binary mask from atlas label indices matching *label_patterns*.

        Label matching is case-insensitive substring matching, so partial
        names (e.g. 'triangularis') will match 'Inferior Frontal Gyrus,
        pars triangularis'.

        Args:
            atlas_resampled : Integer atlas label array (subject space).
            label_patterns  : List of label substrings to match.

        Returns:
            Binary float32 mask (1 inside ROI, 0 outside).
        """
        matched_indices: List[int] = []

        for pattern in label_patterns:
            for idx, label in enumerate(self._atlas_labels):
                if pattern.lower() in label.lower():
                    if idx not in matched_indices:
                        matched_indices.append(idx)

        if not matched_indices:
            logger.warning(
                f"[ROIExtractor] No atlas labels matched patterns: {label_patterns}"
            )
            return np.zeros(atlas_resampled.shape[:3], dtype=np.float32)

        mask = np.zeros(atlas_resampled.shape[:3], dtype=np.float32)
        for idx in matched_indices:
            mask[atlas_resampled == idx] = 1.0

        logger.debug(
            f"[ROIExtractor] Patterns {label_patterns} → "
            f"indices {matched_indices} → {int(mask.sum())} voxels"
        )
        return mask

    # ──────────────────────────────────────────────────────────────────────────
    # Private — statistics
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_roi_stats(
        self,
        mask:       np.ndarray,
        data:       np.ndarray,
        roi_name:   str,
        affine:     np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute region-level statistics for one ROI.

        Args:
            mask     : Binary float32 mask.
            data     : MRI intensity array.
            roi_name : ROI label.
            affine   : Affine matrix (for voxel volume).

        Returns:
            Dict with region statistics.
        """
        voxel_vol = float(abs(np.linalg.det(affine[:3, :3])))
        voxel_count = int(mask.sum())
        volume_mm3 = voxel_count * voxel_vol

        masked_vals = data[mask > 0]
        stats = {
            "roi_name":      roi_name,
            "voxel_count":   voxel_count,
            "volume_mm3":    round(volume_mm3, 2),
            "mean_intensity": round(float(masked_vals.mean()), 4) if len(masked_vals) else 0.0,
            "std_intensity":  round(float(masked_vals.std()),  4) if len(masked_vals) else 0.0,
            "max_intensity":  round(float(masked_vals.max()),  4) if len(masked_vals) else 0.0,
            "min_intensity":  round(float(masked_vals.min()),  4) if len(masked_vals) else 0.0,
        }
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    # Private — save helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _save_mask_nifti(
        self,
        mask:       np.ndarray,
        affine:     np.ndarray,
        patient_id: str,
        roi_name:   str,
    ) -> str:
        path = self.output_dir / f"{patient_id}_{roi_name}_mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, affine), str(path))
        logger.debug(f"[ROIExtractor] Mask saved → {path}")
        return str(path)

    def _save_all_overlays(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
    ) -> None:
        """Save one overlay figure per ROI (axial + sagittal + coronal)."""
        for roi_name, mask in masks.items():
            self._save_roi_overlay_figure(data, mask, patient_id, roi_name)

    def _save_roi_overlay_figure(
        self,
        data:       np.ndarray,
        mask:       np.ndarray,
        patient_id: str,
        roi_name:   str,
    ) -> str:
        """Save a three-plane ROI overlay for one region."""
        color = ROI_COLORS.get(roi_name, (1.0, 0.5, 0.0))
        cx, cy, cz = [s // 2 for s in data.shape[:3]]

        slices = [
            (np.rot90(data[cx, :, :]),  np.rot90(mask[cx, :, :]),  f"Sagittal (x={cx})"),
            (np.rot90(data[:, cy, :]),  np.rot90(mask[:, cy, :]),  f"Coronal  (y={cy})"),
            (np.rot90(data[:, :, cz]),  np.rot90(mask[:, :, cz]),  f"Axial    (z={cz})"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ {roi_name.replace('_', ' ')} │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        for ax, (sl_mri, sl_mask, title) in zip(axes, slices):
            ax.imshow(sl_mri, cmap="bone", aspect="auto", alpha=1.0)
            if sl_mask.any():
                overlay = np.zeros((*sl_mask.shape, 4), dtype=np.float32)
                overlay[sl_mask > 0] = [*color, 0.55]
                ax.imshow(overlay, aspect="auto")
            ax.set_title(title, color="#58a6ff", fontsize=10)
            ax.axis("off")

        # Legend
        patch = mpatches.Patch(color=color, alpha=0.7,
                               label=roi_name.replace("_", " "))
        fig.legend(handles=[patch], loc="lower center", ncol=1,
                   framealpha=0.2, labelcolor="white", fontsize=9)

        plt.tight_layout()
        safe = roi_name.lower()
        save_path = self.output_dir / f"{patient_id}_{safe}_overlay.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        return str(save_path)

    def _save_combined_overlay(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
    ) -> str:
        """Save a single figure with all ROIs overlaid simultaneously."""
        cz = data.shape[2] // 2
        sl_mri = np.rot90(data[:, :, cz])

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="#0d1117")
        ax.imshow(sl_mri, cmap="bone", aspect="auto")

        legend_patches = []
        for roi_name, mask in masks.items():
            color = ROI_COLORS.get(roi_name, (1.0, 0.5, 0.0))
            sl_mask = np.rot90(mask[:, :, cz])
            if sl_mask.any():
                overlay = np.zeros((*sl_mask.shape, 4), dtype=np.float32)
                overlay[sl_mask > 0] = [*color, 0.60]
                ax.imshow(overlay, aspect="auto")
            legend_patches.append(
                mpatches.Patch(color=color, alpha=0.7,
                               label=roi_name.replace("_", " "))
            )

        ax.legend(handles=legend_patches, loc="lower right",
                  framealpha=0.3, labelcolor="white", fontsize=9)
        ax.set_title(
            f"All Speech ROIs │ Axial z={cz} │ {patient_id}",
            color="white", fontsize=12, fontweight="bold"
        )
        ax.axis("off")

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_all_rois_combined.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[ROIExtractor] Combined overlay saved → {save_path}")
        return str(save_path)

    def _save_stats_figure(
        self,
        stats:      Dict[str, Dict],
        patient_id: str,
    ) -> str:
        """Save a bar chart of ROI volumes and voxel counts."""
        rois   = [r for r in stats if "error" not in stats[r]]
        volumes = [stats[r]["volume_mm3"]  for r in rois]
        voxels  = [stats[r]["voxel_count"] for r in rois]
        colors  = [ROI_COLORS.get(r, (0.5, 0.5, 0.5)) for r in rois]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
        fig.suptitle(
            f"NeuroGenesis │ ROI Statistics │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        short_names = [r.replace("_", "\n") for r in rois]

        for ax, (vals, ylabel, title) in zip(axes, [
            (volumes, "Volume (mm³)",  "ROI Volume"),
            (voxels,  "Voxel Count",   "ROI Voxel Count"),
        ]):
            bars = ax.bar(short_names, vals, color=colors, alpha=0.85)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="white", labelsize=8)
            ax.set_xlabel("ROI", color="#8b949e", fontsize=9)
            ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
            ax.set_title(title, color="#58a6ff", fontsize=11)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:,.0f}",
                    ha="center", va="bottom", color="white", fontsize=7
                )

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_roi_statistics.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[ROIExtractor] Statistics figure saved → {save_path}")
        return str(save_path)
