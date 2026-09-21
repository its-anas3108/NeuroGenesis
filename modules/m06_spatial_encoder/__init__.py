"""ROI patch dataset.

The 3D CNN encoder that used to live in this package has been removed; the
model is morphometry-only. This package now holds only the shared dataset
infrastructure (:class:`ROIPatchDataset` and friends), which the morphometric
branch and preprocessing pipeline still use to locate/load cached ROI patch
tensors.
"""

from modules.m06_spatial_encoder.patch_dataset import (
    ROIPatchDataset,
    Sample,
    collate_samples,
    make_loader,
    patch_tensor_path,
)

__all__ = [
    "ROIPatchDataset",
    "Sample",
    "collate_samples",
    "make_loader",
    "patch_tensor_path",
]
