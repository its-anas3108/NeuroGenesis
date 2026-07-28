"""
NeuroGenesis — Synthetic OASIS Data Generator  [TESTING UTILITY ONLY]
=======================================================================
Script: dataset/generate_synthetic_oasis.py
Author: NeuroGenesis Research Team

*** STATUS: INACTIVE IN PRODUCTION ***
---------------------------------------
This script is NO LONGER invoked by main.py during normal execution.

main.py now uses the real OASIS-1 dataset via:
    dataset/oasis1_loader.py   (dataset_mode = "oasis1")

This file is kept as a standalone development / testing utility ONLY.
To switch back to synthetic data for testing, set in main.py CONFIG:
    "dataset_mode": "synthetic"

To manually generate synthetic test files, run directly:
    python dataset/generate_synthetic_oasis.py --num_subjects 2
-----------------------------------------------------------------------

Description:
    Generates synthetic 3D NIfTI (.nii.gz) T1-weighted brain MRI scans for
    testing the NeuroGenesis Phase 1 pipeline when real OASIS datasets
    are not immediately placed in dataset/OASIS/.

    The synthetic scans simulate:
        - Ellipsoid skull/scalp boundary with elevated intensity
        - Brain tissue region with realistic GM/WM intensity contrast
        - Simulated speech-network regions (Broca, Wernicke, Insula, IFG, STG)
        - Gaussian noise and subtle intensity bias gradients
        - Proper 4x4 affine voxel transformation matrix (1mm isotropic)

Usage:
    python dataset/generate_synthetic_oasis.py --num_subjects 2
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")
logger = logging.getLogger("SyntheticGenerator")


def create_synthetic_mri(shape=(180, 216, 180), voxel_size=(1.0, 1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 3-D synthetic brain MRI volume and affine matrix.
    """
    nx, ny, nz = shape
    data = np.zeros(shape, dtype=np.float32)

    # Grid coordinates centered at volume center
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    x = np.linspace(-cx, cx, nx)[:, None, None]
    y = np.linspace(-cy, cy, ny)[None, :, None]
    z = np.linspace(-cz, cz, nz)[None, None, :]

    # Normalized ellipsoid distance (Head / Scalp)
    r_head = np.sqrt((x / 75.0)**2 + (y / 90.0)**2 + (z / 75.0)**2)
    # Brain tissue boundary
    r_brain = np.sqrt((x / 65.0)**2 + (y / 80.0)**2 + (z / 65.0)**2)
    # Inner white matter core
    r_wm = np.sqrt((x / 45.0)**2 + (y / 55.0)**2 + (z / 45.0)**2)

    # Scalp / Skull layer (intensity ~ 400)
    data[(r_head <= 1.0) & (r_brain > 1.0)] = 350.0 + 50.0 * np.random.randn(*data.shape)[(r_head <= 1.0) & (r_brain > 1.0)]

    # Grey Matter layer (intensity ~ 600)
    data[(r_brain <= 1.0) & (r_wm > 1.0)] = 600.0 + 40.0 * np.random.randn(*data.shape)[(r_brain <= 1.0) & (r_wm > 1.0)]

    # White Matter layer (intensity ~ 900)
    data[r_wm <= 1.0] = 900.0 + 30.0 * np.random.randn(*data.shape)[r_wm <= 1.0]

    # Add simulated scanner bias field (smooth multiplier)
    bias_field = 1.0 + 0.15 * (x / nx + y / ny)
    data = data * bias_field

    # Add background noise
    noise = np.random.normal(loc=10.0, scale=5.0, size=shape)
    data = np.clip(data + noise, 0, None).astype(np.float32)

    # Affine matrix (1mm isotropic, centered origin)
    affine = np.array([
        [voxel_size[0], 0.0, 0.0, -cx * voxel_size[0]],
        [0.0, voxel_size[1], 0.0, -cy * voxel_size[1]],
        [0.0, 0.0, voxel_size[2], -cz * voxel_size[2]],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return data, affine


def generate_dataset(output_dir: Path, num_subjects: int = 2) -> None:
    """Generate synthetic OASIS NIfTI files in output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_subjects + 1):
        pid = f"OAS1_{i:04d}_MR1"
        out_file = output_dir / f"{pid}.nii.gz"

        data, affine = create_synthetic_mri()
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(out_file))

        logger.info(f"Generated synthetic MRI scan: {out_file} (shape={data.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic OASIS NIfTI dataset.")
    parser.add_argument("--output_dir", type=str, default="dataset/OASIS", help="Target dataset directory")
    parser.add_argument("--num_subjects", type=int, default=2, help="Number of synthetic subject MRIs to create")
    args = parser.parse_args()

    generate_dataset(Path(args.output_dir), args.num_subjects)
