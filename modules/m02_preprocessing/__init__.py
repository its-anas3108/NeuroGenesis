"""M1-M5 — MRI loading, QC, preprocessing, skull stripping, standardization.

Wraps the preserved ``preprocessing/`` package. Heavy imaging imports are
deferred, so this module is importable without nibabel/SimpleITK/nilearn.
"""

from modules.m02_preprocessing.pipeline import (
    IMAGING_DEPENDENCIES,
    PreprocessingPipeline,
    StageArtifact,
    SubjectPreprocessingResult,
    check_imaging_dependencies,
)

__all__ = [
    "IMAGING_DEPENDENCIES",
    "check_imaging_dependencies",
    "StageArtifact",
    "SubjectPreprocessingResult",
    "PreprocessingPipeline",
]
