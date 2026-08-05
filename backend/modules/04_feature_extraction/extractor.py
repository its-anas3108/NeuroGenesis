"""
NeuroGenesis — Module 4: Morphological Feature Extraction
=========================================================
File   : backend/modules/04_feature_extraction/extractor.py
Purpose: Extract Regional Volume, Grey Matter Volume, Surface Area, Mean Intensity,
         Standard Deviation, Entropy, Texture Features, and Atrophy Index per speech ROI.
         Export CSV, JSON, DataFrame and generate feature comparison plots.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.measure import marching_cubes, mesh_surface_area
    from skimage.feature import graycomatrix, graycoprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from features.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class MorphologicalFeatureExtractor:
    """
    Morphological & Atrophy Feature Extraction Module.
    Extracts rich quantitative features across all 5 speech regions.
    """

    def __init__(self, output_dir: Path, voxel_volume_mm3: float = 1.0):
        self.output_dir = Path(output_dir) / "features"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_volume_mm3 = voxel_volume_mm3
        self.core_extractor = FeatureExtractor(output_dir=self.output_dir, voxel_volume_mm3=voxel_volume_mm3)

    def extract_features(
        self,
        masks: Dict[str, np.ndarray],
        volume: np.ndarray,
        subject_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Extract complete feature matrix for all speech ROIs in a scan.
        """
        logger.info(f"--- Extracting Morphological Features for {subject_id} ---")
        eTIV = float(metadata.get("etiv", 1450.0)) if metadata and metadata.get("etiv") else 1450.0

        records: List[Dict[str, Any]] = []

        for roi_name, mask in masks.items():
            tissue_mask = (mask > 0) & (volume > 0.05)
            roi_voxels = volume[tissue_mask]
            if len(roi_voxels) == 0:
                continue

            # 1. Morphometric
            voxel_count = int(np.sum(tissue_mask))
            regional_volume = float(voxel_count * self.voxel_volume_mm3)
            
            # Grey matter volume (intensity > threshold)
            gm_mask = tissue_mask & (volume > 0.30)
            gm_volume = float(np.sum(gm_mask) * self.voxel_volume_mm3)

            # Surface Area
            surface_area = 0.0
            if HAS_SKIMAGE and voxel_count > 10:
                try:
                    verts, faces, _, _ = marching_cubes(tissue_mask.astype(np.float32), level=0.5)
                    surface_area = float(mesh_surface_area(verts, faces))
                except Exception:
                    surface_area = float(voxel_count ** (2.0 / 3.0) * 6.0)
            else:
                surface_area = float(voxel_count ** (2.0 / 3.0) * 6.0)

            # Cortical Thickness
            cortical_thickness = float(regional_volume / (surface_area + 1e-6)) if surface_area > 0 else 2.5


            # 2. Intensity & Entropy
            mean_intensity = float(np.mean(roi_voxels))
            std_intensity = float(np.std(roi_voxels))
            
            # Histogram entropy
            hist, _ = np.histogram(roi_voxels, bins=32, density=True)
            hist = hist[hist > 0]
            voxel_entropy = float(-np.sum(hist * np.log2(hist))) if len(hist) > 0 else 0.0

            # 3. Texture Features (GLCM)
            texture_contrast = 0.0
            texture_homogeneity = 0.0
            if HAS_SKIMAGE and len(roi_voxels) > 100:
                try:
                    # Quantize 0-1 float to 0-15 int
                    quant = np.clip(np.floor(volume * 15.99), 0, 15).astype(np.uint8)
                    mid_z = volume.shape[2] // 2
                    slice_mask = mask[:, :, mid_z]
                    if np.sum(slice_mask) > 10:
                        slice_quant = quant[:, :, mid_z]
                        glcm = graycomatrix(slice_quant, distances=[1], angles=[0], levels=16, symmetric=True, normed=True)
                        texture_contrast = float(graycoprops(glcm, 'contrast')[0, 0])
                        texture_homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
                except Exception:
                    pass

            # 4. Atrophy Index (Regional Volume / eTIV)
            atrophy_index = float(regional_volume / eTIV) if eTIV > 0 else 0.0

            records.append({
                "subject_id": subject_id,
                "roi_name": roi_name,
                "regional_volume_mm3": regional_volume,
                "gm_volume_mm3": gm_volume,
                "surface_area_mm2": surface_area,
                "cortical_thickness_mm": cortical_thickness,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "voxel_entropy": voxel_entropy,
                "texture_contrast": texture_contrast,
                "texture_homogeneity": texture_homogeneity,
                "atrophy_index": atrophy_index
            })

        df = pd.DataFrame(records)

        # Save CSV & JSON
        out_dir = self.output_dir / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{subject_id}_morphological_features.csv"
        json_path = out_dir / f"{subject_id}_morphological_features.json"
        
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(records, f, indent=2)

        # Generate Feature Bar Chart
        self._plot_feature_bars(df, subject_id, out_dir)

        logger.info(f"Extracted {len(df)} ROI feature records for {subject_id}.")
        return df

    def _plot_feature_bars(self, df: pd.DataFrame, subject_id: str, out_dir: Path):
        """Plot comparison bar charts for regional volume and atrophy index."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Speech ROI Morphometrics — {subject_id}", fontsize=14, fontweight="bold")

        rois = df["roi_name"].values
        vols = df["regional_volume_mm3"].values
        atrophy = df["atrophy_index"].values

        axes[0].barh(rois, vols, color="#2A9D8F")
        axes[0].set_xlabel("Regional Volume (mm³)")
        axes[0].set_title("Volume per Speech ROI")

        axes[1].barh(rois, atrophy, color="#E63946")
        axes[1].set_xlabel("Atrophy Index (ROI Vol / eTIV)")
        axes[1].set_title("Regional Atrophy Index")

        plt.tight_layout()
        plt.savefig(out_dir / f"{subject_id}_feature_summary.png", dpi=150)
        plt.close(fig)
