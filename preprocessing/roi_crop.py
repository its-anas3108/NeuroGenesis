"""
NeuroGenesis — ROI-Centric Patch Extraction (Novel Module)
============================================================
Module: preprocessing/roi_crop.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Novel Contribution:
    Instead of feeding the full brain volume to downstream models, this
    module extracts compact 3-D bounding-box patches around each speech-
    related ROI. The patches are:

        1. Cropped from the skull-stripped volume at each ROI's bounding box
        2. Padded with a configurable contextual margin
        3. Resampled to a uniform patch size (default 48 × 48 × 48)
        4. Saved individually as .nii.gz files
        5. Stacked into a (N_rois, D, H, W) tensor saved as .npy

    This ROI-centric representation:
        • Reduces input size by ~6× compared to whole-brain
        • Focuses the model on speech-atrophy-relevant tissue only
        • Enables multi-branch architectures in NeuroProp-X where each
          ROI branch processes its own patch independently

    The final tensor shape is the primary NeuroProp-X Phase 2 input.

Usage:
    cropper = ROICropper(patch_size=(48, 48, 48),
                         output_dir=Path("outputs/roi_patches"),
                         tensor_dir=Path("outputs/tensors"))

    tensor = cropper.extract_all(
        data       = preprocessed_brain,        # (X, Y, Z)
        masks      = roi_masks_dict,             # {roi_name: binary_mask}
        patient_id = "OAS1_0001",
        affine     = affine,
    )
    # tensor.shape → (5, 48, 48, 48)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


class ROICropper:
    """
    Extract, resize, and stack 3-D patches around speech-related ROIs.

    This module implements the novel ROI-Centric Processing Pipeline
    described in the NeuroGenesis Phase 1 design document. The resulting
    tensor is the canonical input for NeuroProp-X.

    Args:
        patch_size  : Uniform output size for each ROI patch (D, H, W).
        context_pad : Voxel padding added around each ROI bounding box.
        output_dir  : Directory for per-ROI .nii.gz patch files.
        tensor_dir  : Directory for stacked .npy tensors.
    """

    def __init__(
        self,
        patch_size:  Tuple[int, int, int] = (48, 48, 48),
        context_pad: int = 4,
        output_dir:  Path = Path("outputs/roi_patches"),
        tensor_dir:  Path = Path("outputs/tensors"),
    ) -> None:
        self.patch_size  = tuple(patch_size)
        self.context_pad = context_pad
        self.output_dir  = Path(output_dir)
        self.tensor_dir  = Path(tensor_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensor_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[ROICropper] Initialised │ patch_size={self.patch_size} │ "
            f"context_pad={context_pad} │ output={self.output_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — main entry point
    # ──────────────────────────────────────────────────────────────────────────

    def extract_all(
        self,
        data:       np.ndarray,
        masks:      Dict[str, np.ndarray],
        patient_id: str,
        affine:     np.ndarray,
        save:       bool = True,
    ) -> np.ndarray:
        """
        Extract, resize, and stack patches for all ROIs.

        Args:
            data       : Skull-stripped, preprocessed float32 array (X, Y, Z).
            masks      : Dict mapping ROI name → binary float32 mask (X, Y, Z).
            patient_id : Subject identifier.
            affine     : Affine matrix (used for saving NIfTI patches).
            save       : If *True*, save individual patches and the tensor.

        Returns:
            Stacked numpy array of shape (N_rois, D, H, W) where
            N_rois = len(masks) and D, H, W = patch_size.
        """
        logger.info(
            f"[ROICropper] Extracting {len(masks)} ROI patch(es) for [{patient_id}]"
        )

        patches: Dict[str, np.ndarray] = {}
        roi_metadata: Dict[str, dict] = {}

        for roi_name, mask in masks.items():
            try:
                patch, meta = self.extract_single(
                    data, mask, roi_name, patient_id
                )
                patches[roi_name] = patch
                roi_metadata[roi_name] = meta

                if save:
                    self._save_patch_nifti(patch, affine, patient_id, roi_name)

                logger.info(
                    f"[ROICropper]   ✓ {roi_name:25s} │ "
                    f"bbox={meta['bbox']} │ "
                    f"patch={patch.shape}"
                )

            except Exception as exc:
                logger.warning(
                    f"[ROICropper]   ✗ Failed to extract {roi_name} for "
                    f"{patient_id}: {exc}. Using zeros."
                )
                patches[roi_name] = np.zeros(self.patch_size, dtype=np.float32)
                roi_metadata[roi_name] = {"error": str(exc)}

        # ── Stack into tensor ──────────────────────────────────────────────
        roi_names_ordered = list(patches.keys())
        tensor = np.stack([patches[n] for n in roi_names_ordered], axis=0)
        # tensor.shape → (N_rois, D, H, W)

        if save:
            self._save_tensor(tensor, roi_names_ordered, patient_id)
            self._save_mosaic_figure(patches, patient_id)

        logger.info(
            f"[ROICropper] ✓ Tensor shape {tensor.shape} saved for [{patient_id}]"
        )
        return tensor

    def extract_single(
        self,
        data:       np.ndarray,
        mask:       np.ndarray,
        roi_name:   str,
        patient_id: str,
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract and resize a single ROI patch.

        Algorithm:
            1. Find bounding box of non-zero mask voxels
            2. Add contextual padding (clipped to volume boundaries)
            3. Crop the brain volume at the padded bounding box
            4. Resize the crop to *patch_size* using zoom

        Args:
            data       : Full brain float32 array (X, Y, Z).
            mask       : Binary ROI mask (same shape as data).
            roi_name   : ROI identifier for logging.
            patient_id : Subject identifier for logging.

        Returns:
            Tuple of (patch float32 array of shape patch_size, metadata dict).

        Raises:
            ValueError: If the ROI mask is empty (zero voxels).
        """
        # ── Find bounding box ──────────────────────────────────────────────
        nz = np.argwhere(mask > 0)
        if len(nz) == 0:
            raise ValueError(f"Empty mask for ROI '{roi_name}' in {patient_id}")

        mins = nz.min(axis=0)
        maxs = nz.max(axis=0)
        bbox_raw = tuple(zip(mins.tolist(), maxs.tolist()))

        # ── Add contextual padding ─────────────────────────────────────────
        shape = np.array(data.shape[:3])
        lo = np.maximum(mins - self.context_pad, 0)
        hi = np.minimum(maxs + self.context_pad + 1, shape)
        bbox_padded = tuple(zip(lo.tolist(), hi.tolist()))

        # ── Crop ───────────────────────────────────────────────────────────
        crop = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()

        # ── Mask the crop (zero out non-brain voxels within the bbox) ──────
        mask_crop = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        crop = crop * mask_crop

        # ── Resize to target patch size ────────────────────────────────────
        zoom_factors = tuple(
            self.patch_size[i] / (crop.shape[i] + 1e-10)
            for i in range(3)
        )
        patch = zoom(crop, zoom_factors, order=1).astype(np.float32)

        # Ensure exact shape (zoom can introduce off-by-one)
        patch = self._enforce_shape(patch)

        metadata = {
            "roi_name":     roi_name,
            "bbox":         bbox_raw,
            "bbox_padded":  bbox_padded,
            "crop_shape":   crop.shape,
            "patch_shape":  patch.shape,
            "zoom_factors": zoom_factors,
            "n_roi_voxels": int(nz.shape[0]),
            "mean_intensity": float(patch[patch > 0].mean()) if (patch > 0).any() else 0.0,
        }

        return patch, metadata

    # ──────────────────────────────────────────────────────────────────────────
    # NeuroProp-X integration hook
    # ──────────────────────────────────────────────────────────────────────────

    def get_neuroprox_tensor(
        self, patient_id: str
    ) -> Optional[np.ndarray]:
        """
        Load the pre-computed NeuroProp-X input tensor for a subject.

        This is the primary data hand-off point from Phase 1 → NeuroProp-X.

        Returns:
            numpy array of shape (N_rois, D, H, W) or *None* if not found.

        .. note::
            In Phase 2, NeuroProp-X will call this method to obtain the
            ROI-centric input representation for each subject.
        """
        tensor_path = self.tensor_dir / f"{patient_id}_roi_tensor.npy"
        if not tensor_path.exists():
            logger.error(f"[ROICropper] Tensor not found: {tensor_path}")
            return None

        tensor = np.load(str(tensor_path))
        logger.debug(f"[ROICropper] Loaded tensor {tensor.shape} for {patient_id}")
        return tensor

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _enforce_shape(self, patch: np.ndarray) -> np.ndarray:
        """
        Ensure the patch is exactly *patch_size* by cropping or zero-padding.

        Args:
            patch : Array from zoom, may be off by 1 voxel.

        Returns:
            Array of exactly *patch_size*.
        """
        result = np.zeros(self.patch_size, dtype=np.float32)
        slices_src = tuple(slice(0, min(patch.shape[i], self.patch_size[i])) for i in range(3))
        slices_dst = tuple(slice(0, min(patch.shape[i], self.patch_size[i])) for i in range(3))
        result[slices_dst] = patch[slices_src]
        return result

    def _save_patch_nifti(
        self,
        patch:      np.ndarray,
        affine:     np.ndarray,
        patient_id: str,
        roi_name:   str,
    ) -> str:
        """Save a single ROI patch as a NIfTI file."""
        path = self.output_dir / f"{patient_id}_{roi_name}_patch.nii.gz"
        nib.save(nib.Nifti1Image(patch, affine), str(path))
        return str(path)

    def _save_tensor(
        self,
        tensor:    np.ndarray,
        roi_names: List[str],
        patient_id: str,
    ) -> str:
        """Save the stacked ROI tensor and a companion text manifest."""
        tensor_path = self.tensor_dir / f"{patient_id}_roi_tensor.npy"
        manifest_path = self.tensor_dir / f"{patient_id}_roi_tensor_manifest.txt"

        np.save(str(tensor_path), tensor)

        with open(manifest_path, "w") as f:
            f.write(f"Patient ID : {patient_id}\n")
            f.write(f"Tensor shape: {tensor.shape}  [N_rois, D, H, W]\n")
            f.write(f"Patch size  : {self.patch_size}\n\n")
            f.write("ROI Index Mapping:\n")
            for i, name in enumerate(roi_names):
                f.write(f"  [{i}] {name}\n")

        logger.info(f"[ROICropper] Tensor saved → {tensor_path}")
        return str(tensor_path)

    def _save_mosaic_figure(
        self,
        patches:    Dict[str, np.ndarray],
        patient_id: str,
    ) -> str:
        """
        Save a mosaic figure showing the central axial slice of each ROI patch.
        """
        n = len(patches)
        cols = min(n, 5)
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 4, rows * 4 + 1),
                                 facecolor="#0d1117")

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(
            f"NeuroGenesis │ ROI-Centric Patches ({self.patch_size}) │ {patient_id}",
            color="white", fontsize=13, fontweight="bold", y=1.01
        )

        # Colormap per ROI
        cmaps = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
        for idx, (roi_name, patch) in enumerate(patches.items()):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            cz = patch.shape[2] // 2
            sl = np.rot90(patch[:, :, cz])
            cmap = cmaps[idx % len(cmaps)]

            im = ax.imshow(sl, cmap=cmap, aspect="auto")
            ax.set_title(roi_name.replace("_", "\n"), color="white",
                         fontsize=9, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.8).ax.tick_params(
                colors="white", labelsize=6
            )

        # Hide unused subplots
        for idx in range(len(patches), rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].set_visible(False)

        plt.tight_layout()
        save_path = self.output_dir / f"{patient_id}_roi_patches_mosaic.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="#0d1117", edgecolor="none")
        plt.close(fig)
        logger.info(f"[ROICropper] Mosaic figure saved → {save_path}")
        return str(save_path)
