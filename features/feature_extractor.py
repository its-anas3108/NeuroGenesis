"""
NeuroGenesis — Feature Extraction Module
==========================================
Module: features/feature_extractor.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Computes a rich set of morphometric and intensity-based features for each
speech-related ROI. Features are computed on the ROI-centric 3-D patches
extracted by ROICropper, making them highly focused on atrophy-relevant tissue.

Features computed per ROI:
    Morphometric:
        - Brain Volume (total non-zero voxel volume in mm³)
        - Grey Matter Volume (intensity-thresholded voxel volume)
        - Voxel Count (raw non-zero voxel count)
        - Approximate Surface Area (via scikit-image marching cubes)

    Intensity-Based:
        - Mean Intensity
        - Maximum Intensity
        - Minimum Intensity (non-zero)
        - Standard Deviation
        - Skewness (asymmetry of intensity distribution)
        - Kurtosis (tailedness of intensity distribution)
        - Entropy (information content of the patch)

    Integration Stubs:
        - Cortical Thickness: FreeSurfer interface placeholder
        - Fractal Dimension: Surface complexity placeholder

Outputs:
    - Per-subject DataFrame
    - CSV feature table (outputs/features/)
    - JSON feature record (outputs/features/)

NeuroProp-X Hook:
    get_feature_vector() → flat numpy array for model conditioning

Usage:
    extractor = FeatureExtractor(output_dir=Path("outputs/features"),
                                 voxel_volume_mm3=1.0)
    df = extractor.extract_all(patches, patient_id="OAS1_0001")
    extractor.save(df, patient_id="OAS1_0001")
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, skew, kurtosis

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Computes morphometric and intensity features from ROI patches.

    Args:
        output_dir      : Directory for CSV, JSON, and figure outputs.
        voxel_volume_mm3: Volume of one voxel in mm³ (from MRI spacing).
                          If patches are already resampled, this may differ
                          from the original spacing.
        gm_threshold    : Intensity threshold (fraction of max) for grey
                          matter voxel classification.
    """

    #: Feature columns in the order they appear in the output CSV
    COLUMN_ORDER: List[str] = [
        "patient_id",
        "roi_name",
        "voxel_count",
        "brain_volume_mm3",
        "gm_volume_mm3",
        "mean_intensity",
        "max_intensity",
        "min_intensity",
        "std_intensity",
        "skewness",
        "kurtosis",
        "entropy",
        "surface_area_vox",
        "cortical_thickness_mm",   # FreeSurfer stub
        "fractal_dimension",       # Placeholder
    ]

    def __init__(
        self,
        output_dir:       Path,
        voxel_volume_mm3: float = 1.0,
        gm_threshold:     float = 0.3,
    ) -> None:
        self.output_dir       = Path(output_dir)
        self.voxel_volume_mm3 = voxel_volume_mm3
        self.gm_threshold     = gm_threshold

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[FeatureExtractor] Initialised │ voxel_vol={voxel_volume_mm3:.4f} mm³ │ "
            f"gm_threshold={gm_threshold} │ output={self.output_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — main entry points
    # ──────────────────────────────────────────────────────────────────────────

    def extract_all(
        self,
        patches:    Dict[str, np.ndarray],
        patient_id: str,
    ) -> pd.DataFrame:
        """
        Compute features for all ROI patches and return as a DataFrame.

        Args:
            patches    : Dict mapping roi_name → float32 array (D, H, W).
            patient_id : Subject identifier (added as a column).

        Returns:
            DataFrame with one row per ROI and columns per feature.
        """
        logger.info(
            f"[FeatureExtractor] Extracting features for [{patient_id}] "
            f"across {len(patches)} ROI(s)"
        )

        records: List[Dict[str, Any]] = []

        for roi_name, patch in patches.items():
            try:
                row = self.extract_single(patch, patient_id, roi_name)
                records.append(row)
                logger.info(
                    f"[FeatureExtractor]   ✓ {roi_name:28s} │ "
                    f"vol={row['brain_volume_mm3']:.1f} mm³ │ "
                    f"mean_int={row['mean_intensity']:.4f} │ "
                    f"SA={row['surface_area_vox']:.1f} vox²"
                )
            except Exception as exc:
                logger.warning(
                    f"[FeatureExtractor]   ✗ Failed {roi_name}: {exc}"
                )
                records.append(self._empty_record(patient_id, roi_name))

        df = pd.DataFrame(records)
        # Reorder columns
        cols = [c for c in self.COLUMN_ORDER if c in df.columns]
        df = df[cols]

        logger.info(
            f"[FeatureExtractor] ✓ Feature extraction complete "
            f"│ {len(df)} ROIs │ {df.shape[1]} features"
        )
        return df

    def extract_single(
        self,
        patch:      np.ndarray,
        patient_id: str,
        roi_name:   str,
    ) -> Dict[str, Any]:
        """
        Compute all features for a single ROI patch.

        Args:
            patch      : Float32 3-D array (D, H, W). Non-zero voxels = ROI.
            patient_id : Subject identifier.
            roi_name   : ROI name.

        Returns:
            Feature dictionary (one row's worth of data).
        """
        # ── Morphometric ──────────────────────────────────────────────────
        # Filter background noise/air to isolate actual subject brain tissue voxels inside the ROI
        tissue_mask    = (patch > 0.05)
        nonzero        = patch[tissue_mask].flatten()

        voxel_count    = int(np.sum(tissue_mask))
        brain_vol_mm3  = voxel_count * self.voxel_volume_mm3

        # Grey matter volume (intensity > gm_threshold * max_intensity)
        max_val        = float(patch.max()) if patch.size > 0 else 1.0
        gm_thresh_val  = max_val * self.gm_threshold
        gm_count       = int(np.sum((patch > gm_thresh_val) & tissue_mask))
        gm_vol_mm3     = gm_count * self.voxel_volume_mm3

        # ── Intensity statistics ───────────────────────────────────────────
        if len(nonzero) == 0:
            mean_int = max_int = min_int = std_int = 0.0
            skewness_val = kurt_val = entropy_val = 0.0
        else:
            mean_int     = float(np.mean(nonzero))
            max_int      = float(np.max(nonzero))
            min_int      = float(np.min(nonzero))
            std_int      = float(np.std(nonzero))
            skewness_val = float(skew(nonzero)) if len(nonzero) > 2 else 0.0
            kurt_val     = float(kurtosis(nonzero)) if len(nonzero) > 2 else 0.0

            # Shannon entropy on histogram
            hist, _ = np.histogram(nonzero, bins=64, density=True)
            hist_norm = hist / (hist.sum() + 1e-10)
            entropy_val = float(scipy_entropy(hist_norm + 1e-12))

        # ── Surface area (marching cubes on tissue mask) ───────────────────
        surface_area = self._compute_surface_area(tissue_mask.astype(np.float32))

        # ── Dynamic Cortical Thickness Estimate (Volume / Surface Area) ────
        if surface_area > 0:
            cortical_thickness = round(float(brain_vol_mm3 / (surface_area + 1e-5)), 3)
        else:
            cortical_thickness = 2.5


        # ── Fractal dimension placeholder ─────────────────────────────────
        # TODO Phase 2: Implement Minkowski–Bouligand box-counting dimension
        # on the ROI surface mesh for surface complexity analysis.
        fractal_dimension = float("nan")

        return {
            "patient_id":            patient_id,
            "roi_name":              roi_name,
            "voxel_count":           voxel_count,
            "brain_volume_mm3":      round(brain_vol_mm3, 4),
            "gm_volume_mm3":         round(gm_vol_mm3, 4),
            "mean_intensity":        round(mean_int, 6),
            "max_intensity":         round(max_int, 6),
            "min_intensity":         round(min_int, 6),
            "std_intensity":         round(std_int, 6),
            "skewness":              round(skewness_val, 6),
            "kurtosis":              round(kurt_val, 6),
            "entropy":               round(entropy_val, 6),
            "surface_area_vox":      round(surface_area, 4),
            "cortical_thickness_mm": cortical_thickness,
            "fractal_dimension":     fractal_dimension,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public — save outputs
    # ──────────────────────────────────────────────────────────────────────────

    def save(
        self,
        df:         pd.DataFrame,
        patient_id: str,
        all_patients_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        Save feature table as CSV and JSON.

        Args:
            df              : Per-ROI feature DataFrame for one subject.
            patient_id      : Subject identifier (used in filenames).
            all_patients_df : Optional aggregated DataFrame across all subjects.

        Returns:
            Dict mapping 'csv', 'json' to saved file paths.
        """
        paths: Dict[str, str] = {}

        # Per-subject CSV
        csv_path = self.output_dir / f"{patient_id}_features.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)
        logger.info(f"[FeatureExtractor] CSV saved → {csv_path}")

        # Per-subject JSON
        json_path = self.output_dir / f"{patient_id}_features.json"
        records = df.to_dict(orient="records")
        with open(json_path, "w") as fh:
            json.dump({"patient_id": patient_id, "rois": records},
                      fh, indent=2, default=str)
        paths["json"] = str(json_path)
        logger.info(f"[FeatureExtractor] JSON saved → {json_path}")

        # Full feature table (all patients)
        if all_patients_df is not None:
            table_path = self.output_dir / "full_feature_table.csv"
            all_patients_df.to_csv(table_path, index=False)
            paths["full_table"] = str(table_path)
            logger.info(f"[FeatureExtractor] Full feature table saved → {table_path}")

        return paths

    def save_feature_table(self, df: pd.DataFrame) -> str:
        """
        Save the aggregated multi-subject feature table CSV.

        This is the main Step 7 output containing all subjects × all ROI
        features in a single flat table.

        Args:
            df : Multi-subject DataFrame (each row = one subject × one ROI).

        Returns:
            Path to saved CSV.
        """
        path = self.output_dir / "feature_table_all_subjects.csv"
        df.to_csv(path, index=False)
        logger.info(f"[FeatureExtractor] Aggregate feature table → {path}")
        return str(path)

    def plot_feature_heatmap(
        self,
        df:         pd.DataFrame,
        patient_id: str,
    ) -> str:
        """
        Save a heatmap of numerical features across ROIs.

        Args:
            df         : Feature DataFrame for one subject.
            patient_id : Subject identifier for the figure title.

        Returns:
            Path to saved figure.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude stub columns
        numeric_cols = [c for c in numeric_cols
                        if c not in ["cortical_thickness_mm", "fractal_dimension"]]
        if not numeric_cols:
            logger.warning("[FeatureExtractor] No numeric columns to plot.")
            return ""

        feature_matrix = df[numeric_cols].values.T
        roi_labels     = df["roi_name"].tolist() if "roi_name" in df else list(range(len(df)))

        fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), max(6, len(numeric_cols))),
                               facecolor="#0d1117")
        im = ax.imshow(feature_matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(roi_labels)))
        ax.set_xticklabels([r.replace("_", "\n") for r in roi_labels],
                           color="white", fontsize=8)
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_yticklabels(numeric_cols, color="white", fontsize=8)
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, shrink=0.8)
        cbar.ax.tick_params(colors="white")

        fig.suptitle(
            f"NeuroGenesis │ Feature Heatmap │ {patient_id}",
            color="white", fontsize=13, fontweight="bold"
        )

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_feature_heatmap.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[FeatureExtractor] Feature heatmap saved → {save_path}")
        return str(save_path)

    # ──────────────────────────────────────────────────────────────────────────
    # NeuroProp-X integration hook
    # ──────────────────────────────────────────────────────────────────────────

    def get_feature_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a flat feature vector from a subject's feature DataFrame.

        This is the primary data hand-off for NeuroProp-X conditioning input.
        NaN stubs (cortical_thickness_mm, fractal_dimension) are zeroed out.

        Args:
            df : Feature DataFrame (one or more ROI rows).

        Returns:
            1-D float32 numpy array of all numeric features concatenated
            across ROIs, in the canonical COLUMN_ORDER.

        .. note::
            In Phase 2, NeuroProp-X will call this method to obtain the
            scalar feature conditioning vector alongside the ROI tensor.
        """
        numeric_cols = [c for c in self.COLUMN_ORDER
                        if c not in ["patient_id", "roi_name"]
                        and c in df.columns]
        mat = df[numeric_cols].fillna(0.0).values.astype(np.float32)
        return mat.flatten()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_surface_area(self, patch: np.ndarray) -> float:
        """
        Estimate ROI surface area using the marching cubes algorithm.

        Uses scikit-image ``marching_cubes`` to extract an isosurface from
        the binary patch, then sums triangle areas via ``mesh_surface_area``.

        Args:
            patch : Float32 array; non-zero voxels define the ROI.

        Returns:
            Approximate surface area in voxel² units. Returns 0.0 if
            marching cubes fails (e.g. empty patch or single voxel).
        """
        try:
            from skimage.measure import marching_cubes, mesh_surface_area

            binary = (patch > 0).astype(np.float32)
            if binary.sum() < 8:
                return 0.0

            verts, faces, _, _ = marching_cubes(binary, level=0.5)
            area = float(mesh_surface_area(verts, faces))
            return area

        except Exception as exc:
            logger.debug(f"[FeatureExtractor] Surface area failed: {exc}")
            return 0.0

    def _empty_record(self, patient_id: str, roi_name: str) -> Dict[str, Any]:
        """Return a zero-filled record for a failed ROI."""
        return {
            "patient_id":            patient_id,
            "roi_name":              roi_name,
            "voxel_count":           0,
            "brain_volume_mm3":      0.0,
            "gm_volume_mm3":         0.0,
            "mean_intensity":        0.0,
            "max_intensity":         0.0,
            "min_intensity":         0.0,
            "std_intensity":         0.0,
            "skewness":              0.0,
            "kurtosis":              0.0,
            "entropy":               0.0,
            "surface_area_vox":      0.0,
            "cortical_thickness_mm": float("nan"),
            "fractal_dimension":     float("nan"),
        }
