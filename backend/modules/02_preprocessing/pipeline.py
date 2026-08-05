"""
NeuroGenesis — Module 2: MRI Preprocessing Pipeline
===================================================
File   : backend/modules/02_preprocessing/pipeline.py
Purpose: Complete 8-stage MRI Preprocessing Pipeline with quality control
         and intermediate output saving / comparison figure generation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import core modules
from preprocessing.normalization import MRINormalizer
from preprocessing.skull_strip import SkullStripper
from preprocessing.artifact_detector import ArtifactDetector
from preprocessing.resize import MRIResizer

logger = logging.getLogger(__name__)

class MRIPreprocessingPipeline:
    """
    8-Stage MRI Preprocessing Pipeline
    Stages:
        1. Load MRI
        2. Orientation Correction
        3. Bias Field Correction
        4. Noise Reduction
        5. Intensity Normalization
        6. Resampling
        7. Skull Stripping
        8. Quality Control
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "outputs")) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.normalizer = MRINormalizer(
            target_shape=config.get("target_shape", (128, 128, 128)),
            clahe_clip_limit=config.get("clahe_clip_limit", 0.03)
        )
        self.skull_stripper = SkullStripper(morph_radius=config.get("morph_radius", 4))
        self.resizer = MRIResizer(target_shape=config.get("target_shape", (128, 128, 128)))
        self.qc_detector = ArtifactDetector()

    def process_scan(self, file_path: Path, subject_id: str) -> Dict[str, Any]:
        """Run full 8-stage preprocessing on a single MRI file."""
        logger.info(f"--- Starting 8-Stage Preprocessing for {subject_id} ---")
        intermediates: Dict[str, Any] = {}

        # 1. Load MRI
        nii_img = nib.load(str(file_path))
        raw_data = nii_img.get_fdata().astype(np.float32)
        intermediates["01_original"] = raw_data

        # 2. Orientation Correction (Reorient to canonical RAS)
        ras_img = nib.as_closest_canonical(nii_img)
        ras_data = ras_img.get_fdata().astype(np.float32)
        intermediates["02_orientation"] = ras_data

        # 3. Bias Field Correction (SimpleITK N4)
        sitk_img = sitk.GetImageFromArray(ras_data)
        sitk_img_float = sitk.Cast(sitk_img, sitk.sitkFloat32)
        
        try:
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
            bias_corrected_sitk = corrector.Execute(sitk_img_float)
            bias_data = sitk.GetArrayFromImage(bias_corrected_sitk)
        except Exception as e:
            logger.warning(f"N4 Bias Correction failed for {subject_id}: {e}. Fallback to RAS data.")
            bias_data = ras_data
        intermediates["03_bias_corrected"] = bias_data

        # 4. Noise Reduction (Anisotropic diffusion / Gaussian)
        try:
            denoise_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
            denoise_filter.SetNumberOfIterations(self.config.get("aniso_iterations", 5))
            denoise_filter.SetTimeStep(0.0625)
            denoise_filter.SetConductanceParameter(self.config.get("aniso_conductance", 3.0))
            denoised_sitk = denoise_filter.Execute(sitk.GetImageFromArray(bias_data))
            denoised_data = sitk.GetArrayFromImage(denoised_sitk)
        except Exception as e:
            logger.warning(f"Anisotropic diffusion failed for {subject_id}: {e}. Fallback to bias_data.")
            denoised_data = bias_data
        intermediates["04_denoised"] = denoised_data

        # 5. Intensity Normalization
        normalized_data = self.normalizer.wm_peak_normalize(denoised_data)
        normalized_data = self.normalizer.min_max_scale(normalized_data, 0.0, 1.0)
        intermediates["05_normalized"] = normalized_data

        # 6. Resampling
        resampled_data = self.resizer.resize_3d_volume(normalized_data)
        intermediates["06_resampled"] = resampled_data

        # 7. Skull Stripping
        mask = self.skull_stripper.extract_brain_mask(resampled_data)
        brain_extracted = resampled_data * mask
        intermediates["07_brain_mask"] = mask
        intermediates["07_skull_stripped"] = brain_extracted

        # 8. Quality Control
        qc_metrics = self.qc_detector.compute_qc_score(brain_extracted)
        intermediates["08_qc_metrics"] = qc_metrics

        # Save intermediate NIfTI outputs if configured
        if self.config.get("save_nifti", True):
            subj_dir = self.output_dir / subject_id
            subj_dir.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(brain_extracted, np.eye(4)), str(subj_dir / f"{subject_id}_processed.nii.gz"))
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), np.eye(4)), str(subj_dir / f"{subject_id}_brain_mask.nii.gz"))

        # Generate comparison figure
        self._generate_comparison_figure(subject_id, raw_data, brain_extracted)

        logger.info(f"--- Completed Preprocessing for {subject_id} | QC Score: {qc_metrics.get('qc_score', 0):.2f} ---")

        return {
            "processed_volume": brain_extracted,
            "brain_mask": mask,
            "qc_metrics": qc_metrics,
            "intermediates": intermediates
        }

    def _generate_comparison_figure(self, subject_id: str, original: np.ndarray, processed: np.ndarray):
        """Generate side-by-side comparison plot of original vs preprocessed MRI."""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"NeuroGenesis Preprocessing Comparison — {subject_id}", fontsize=14, fontweight="bold")

        # Original slices
        o_mid = [s // 2 for s in original.shape]
        axes[0, 0].imshow(np.rot90(original[o_mid[0], :, :]), cmap="gray")
        axes[0, 0].set_title("Original Sagittal")
        axes[0, 1].imshow(np.rot90(original[:, o_mid[1], :]), cmap="gray")
        axes[0, 1].set_title("Original Coronal")
        axes[0, 2].imshow(np.rot90(original[:, :, o_mid[2]]), cmap="gray")
        axes[0, 2].set_title("Original Axial")

        # Processed slices
        p_mid = [s // 2 for s in processed.shape]
        axes[1, 0].imshow(np.rot90(processed[p_mid[0], :, :]), cmap="bone")
        axes[1, 0].set_title("Processed Sagittal")
        axes[1, 1].imshow(np.rot90(processed[:, p_mid[1], :]), cmap="bone")
        axes[1, 1].set_title("Processed Coronal")
        axes[1, 2].imshow(np.rot90(processed[:, :, p_mid[2]]), cmap="bone")
        axes[1, 2].set_title("Processed Axial")

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        fig_dir = self.output_dir / subject_id
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / f"{subject_id}_preprocessing_comparison.png", dpi=150)
        plt.close(fig)
