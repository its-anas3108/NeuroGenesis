"""M01 — dataset layer: OASIS-1 (trainable) and ADNI (inference-only)."""

from modules.m01_dataset.adni_manager import (
    ADNIDataManager,
    ADNISeries,
)
from modules.m01_dataset.adni_manager import DATASET_NAME as ADNI_DATASET_NAME
from modules.m01_dataset.adni_manager import (
    DATASET_SOURCE as ADNI_DATASET_SOURCE,
)
from modules.m01_dataset.cohort import (
    CohortReport,
    build_cohort,
    discover_mri_files,
    save_cohort,
)
from modules.m01_dataset.integrity import (
    DatasetIntegrityError,
    IntegrityCheck,
    IntegrityReport,
    check_dataset_integrity,
)
from modules.m01_dataset.labels import (
    DEFAULT_CDR_TO_STAGE,
    LabelReport,
    class_weights,
    extract_subject_id,
    map_labels,
)
from modules.m01_dataset.oasis1_manager import (
    DATASET_NAME,
    DATASET_SOURCE,
    VOLUME_KINDS,
    OASIS1DataManager,
    OASIS1Session,
)
from modules.m01_dataset.splits import (
    SplitManifest,
    make_subject_split,
    repeated_subject_splits,
)
from modules.m01_dataset.validation import (
    ValidationReport,
    save_validation,
    validate_oasis1,
)

__all__ = [
    "DATASET_NAME",
    "DATASET_SOURCE",
    "VOLUME_KINDS",
    "OASIS1DataManager",
    "OASIS1Session",
    "ADNI_DATASET_NAME",
    "ADNI_DATASET_SOURCE",
    "ADNIDataManager",
    "ADNISeries",
    "ValidationReport",
    "validate_oasis1",
    "save_validation",
    "DatasetIntegrityError",
    "IntegrityCheck",
    "IntegrityReport",
    "check_dataset_integrity",
    "DEFAULT_CDR_TO_STAGE",
    "LabelReport",
    "map_labels",
    "class_weights",
    "extract_subject_id",
    "CohortReport",
    "build_cohort",
    "discover_mri_files",
    "save_cohort",
    "SplitManifest",
    "make_subject_split",
    "repeated_subject_splits",
]
