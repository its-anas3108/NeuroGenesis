"""
NeuroGenesis — Module 3: Speech Region Segmentation
===================================================
File   : backend/modules/03_segmentation/speech_segmenter.py
Purpose: Segment 5 Speech-Related Brain Regions:
         1. Broca's Area
         2. Wernicke's Area
         3. Insula
         4. Inferior Frontal Gyrus (IFG)
         5. Superior Temporal Gyrus (STG)
         Generates binary masks, colored overlays, independent ROI images, and statistics.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation.roi_extraction import ROIExtractor, ROI_DEFINITIONS

logger = logging.getLogger(__name__)

SPEECH_REGIONS = [
    "Broca_Area",
    "Wernicke_Area",
    "Insula",
    "Inferior_Frontal_Gyrus",
    "Superior_Temporal_Gyrus"
]

ROI_COLOR_MAP = {
    "Broca_Area": "#E63946",            # Vibrant Red
    "Wernicke_Area": "#1D3557",         # Deep Blue
    "Insula": "#2A9D8F",                # Teal
    "Inferior_Frontal_Gyrus": "#F4A261",# Warm Amber
    "Superior_Temporal_Gyrus": "#9C27B0"# Purple
}

class SpeechRegionSegmenter:
    """
    Speech Region Segmentation Module
    Segments the 5 speech-related brain regions and exports masks, overlays, and ROI statistics.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir) / "segmented"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = ROIExtractor(output_dir=self.output_dir)

    def segment_speech_regions(
        self,
        volume: np.ndarray,
        affine: np.ndarray,
        subject_id: str
    ) -> Dict[str, Any]:
        """
        Extract masks, ROI patches, statistics, and visualizations for all 5 speech regions.
        """
        logger.info(f"--- Segmenting Speech Regions for {subject_id} ---")
        
        # Call underlying atlas extractor
        masks = self.extractor.extract_all(volume=volume, affine=affine, patient_id=subject_id)

        # Generate individual ROI stats
        roi_stats: Dict[str, Dict[str, Any]] = {}
        roi_patches: Dict[str, np.ndarray] = {}

        voxel_volume_mm3 = float(np.abs(np.linalg.det(affine[:3, :3])))

        for roi_name, mask in masks.items():
            voxel_count = int(np.sum(mask > 0))
            roi_volume_mm3 = float(voxel_count * voxel_volume_mm3)
            
            roi_voxels = volume[mask > 0]
            mean_intensity = float(np.mean(roi_voxels)) if len(roi_voxels) > 0 else 0.0
            std_intensity = float(np.std(roi_voxels)) if len(roi_voxels) > 0 else 0.0

            roi_stats[roi_name] = {
                "voxel_count": voxel_count,
                "volume_mm3": roi_volume_mm3,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "color_hex": ROI_COLOR_MAP.get(roi_name, "#333333")
            }

            # Save individual ROI mask NIfTI
            subj_roi_dir = self.output_dir / subject_id
            subj_roi_dir.mkdir(parents=True, exist_ok=True)
            roi_nii_path = subj_roi_dir / f"{subject_id}_{roi_name}_mask.nii.gz"
            nib.save(nib.Nifti1Image((mask > 0).astype(np.uint8), affine), str(roi_nii_path))

        # Generate Multi-Region Colored Overlay Figure
        self._generate_colored_overlay(subject_id, volume, masks)

        logger.info(f"Successfully segmented {len(masks)} speech ROIs for {subject_id}.")

        return {
            "masks": masks,
            "roi_stats": roi_stats,
            "speech_regions": list(masks.keys())
        }

    def _generate_colored_overlay(
        self,
        subject_id: str,
        volume: np.ndarray,
        masks: Dict[str, np.ndarray]
    ):
        """Generate 3-plane RGB overlay figure of speech ROIs on top of MRI."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"NeuroGenesis Speech Regions Segmentation — {subject_id}", fontsize=14, fontweight="bold")

        mid_slices = [s // 2 for s in volume.shape]

        # Background MRI
        ax_sag = np.rot90(volume[mid_slices[0], :, :])
        ax_cor = np.rot90(volume[:, mid_slices[1], :])
        ax_axi = np.rot90(volume[:, :, mid_slices[2]])

        axes[0].imshow(ax_sag, cmap="gray")
        axes[0].set_title("Sagittal View")
        axes[1].imshow(ax_cor, cmap="gray")
        axes[1].set_title("Coronal View")
        axes[2].imshow(ax_axi, cmap="gray")
        axes[2].set_title("Axial View")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_file = self.output_dir / subject_id / f"{subject_id}_speech_overlay.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
