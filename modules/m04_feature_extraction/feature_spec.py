"""
Morphometric feature specification (Section 6).
==============================================

The canonical, ordered list of per-ROI features that form the node feature
vector, with an explicit definition for each one.

Two rules govern this list, both from Section 6:

**Nothing here is a silently invented clinical feature.** Every entry is either
computed directly from the ROI patch by the preserved
``features.feature_extractor.FeatureExtractor``, or derived from those values and
the OASIS-1 metadata table by a formula stated in :data:`FEATURE_DEFINITIONS`.
The pre-refactor codebase contained invented features such as "Naming Accuracy
Index" and "Speech Fluency Score" that OASIS-1 does not measure; none of those
appear here or anywhere in the active framework.

**A placeholder is dropped, not shipped as a number.** The original extractor
emitted ``fractal_dimension`` as ``float("nan")`` with a "TODO Phase 2" comment.
Carrying a NaN column into a node feature vector would either poison the
gradients or be quietly imputed to a constant, so it is excluded. It is recorded
in :data:`EXCLUDED_FEATURES` with the reason.
"""

from __future__ import annotations

from typing import Dict, Final, List

#: Features produced directly by the preserved ROI patch extractor.
RAW_FEATURES: Final[List[str]] = [
    "voxel_count",
    "brain_volume_mm3",
    "gm_volume_mm3",
    "mean_intensity",
    "std_intensity",
    "skewness",
    "kurtosis",
    "entropy",
    "surface_area_vox",
    "cortical_thickness_mm",
]

#: Features derived from the raw features and the OASIS-1 metadata table.
DERIVED_FEATURES: Final[List[str]] = [
    "gm_fraction",
    "normalized_volume",
    "surface_to_volume_ratio",
    "atrophy_index",
]

#: The full ordered node feature vector. Every ROI-indexed feature tensor in the
#: framework uses this order.
FEATURE_ORDER: Final[List[str]] = RAW_FEATURES + DERIVED_FEATURES

N_FEATURES: Final[int] = len(FEATURE_ORDER)

#: Features requiring a cohort-level reference fitted on the **training split
#: only**. Computing these before splitting would leak test-set statistics.
COHORT_REFERENCED_FEATURES: Final[List[str]] = ["atrophy_index"]

#: Features requiring the OASIS-1 metadata table (specifically ``eTIV``).
METADATA_DEPENDENT_FEATURES: Final[List[str]] = [
    "normalized_volume",
    "atrophy_index",
]

#: Deliberately excluded, with reasons.
EXCLUDED_FEATURES: Final[Dict[str, str]] = {
    "fractal_dimension": (
        "The pre-refactor extractor returned float('nan') with a 'TODO Phase 2' "
        "comment; box-counting dimension was never implemented. A NaN column "
        "cannot be used as a node feature, and imputing it would fabricate a "
        "value."
    ),
    "max_intensity": (
        "After min-max intensity normalization the ROI maximum is 1.0 for almost "
        "every patch, making the column near-constant and uninformative."
    ),
    "min_intensity": (
        "After background masking the ROI minimum sits at the tissue threshold "
        "for almost every patch, making the column near-constant."
    ),
}

#: Definition, unit and provenance of every feature in :data:`FEATURE_ORDER`.
FEATURE_DEFINITIONS: Final[Dict[str, Dict[str, str]]] = {
    "voxel_count": {
        "label": "Voxel count",
        "unit": "voxels",
        "definition": "Number of voxels in the ROI patch above the tissue "
                      "threshold (intensity > 0.05).",
        "source": "ROI patch",
        "category": "volumetric",
    },
    "brain_volume_mm3": {
        "label": "Regional volume",
        "unit": "mm^3",
        "definition": "Tissue voxel count multiplied by the voxel volume.",
        "source": "ROI patch",
        "category": "volumetric",
    },
    "gm_volume_mm3": {
        "label": "Grey-matter volume",
        "unit": "mm^3",
        "definition": "Volume of voxels whose intensity exceeds "
                      "gm_threshold x patch maximum. An intensity-threshold "
                      "proxy for grey matter, not a tissue-class segmentation.",
        "source": "ROI patch",
        "category": "volumetric",
    },
    "mean_intensity": {
        "label": "Mean intensity",
        "unit": "normalized",
        "definition": "Mean intensity over tissue voxels after preprocessing.",
        "source": "ROI patch",
        "category": "intensity",
    },
    "std_intensity": {
        "label": "Intensity SD",
        "unit": "normalized",
        "definition": "Standard deviation of intensity over tissue voxels.",
        "source": "ROI patch",
        "category": "intensity",
    },
    "skewness": {
        "label": "Intensity skewness",
        "unit": "dimensionless",
        "definition": "Third standardised moment of the tissue intensity "
                      "distribution.",
        "source": "ROI patch",
        "category": "textural",
    },
    "kurtosis": {
        "label": "Intensity kurtosis",
        "unit": "dimensionless",
        "definition": "Fourth standardised moment (excess) of the tissue "
                      "intensity distribution.",
        "source": "ROI patch",
        "category": "textural",
    },
    "entropy": {
        "label": "Intensity entropy",
        "unit": "nats",
        "definition": "Shannon entropy of the 64-bin tissue intensity "
                      "histogram. Higher values indicate a more heterogeneous "
                      "intensity distribution.",
        "source": "ROI patch",
        "category": "textural",
    },
    "surface_area_vox": {
        "label": "Surface area",
        "unit": "voxels^2",
        "definition": "Isosurface area of the tissue mask via marching cubes.",
        "source": "ROI patch (requires scikit-image)",
        "category": "morphometric",
    },
    "cortical_thickness_mm": {
        "label": "Cortical thickness (proxy)",
        "unit": "mm",
        "definition": "Regional volume divided by surface area. A "
                      "volume-to-surface proxy, NOT a FreeSurfer cortical "
                      "thickness measurement.",
        "source": "ROI patch",
        "category": "morphometric",
    },
    "gm_fraction": {
        "label": "Grey-matter fraction",
        "unit": "fraction",
        "definition": "gm_volume_mm3 / brain_volume_mm3 within the ROI. "
                      "Head-size invariant tissue composition.",
        "source": "derived",
        "category": "volumetric",
    },
    "normalized_volume": {
        "label": "eTIV-normalized volume",
        "unit": "fraction",
        "definition": "brain_volume_mm3 / eTIV, using the estimated total "
                      "intracranial volume from the OASIS-1 metadata table. "
                      "Removes head-size variation, which otherwise dominates "
                      "raw regional volume.",
        "source": "derived (requires eTIV)",
        "category": "volumetric",
    },
    "surface_to_volume_ratio": {
        "label": "Surface-to-volume ratio",
        "unit": "1/voxel",
        "definition": "surface_area_vox / brain_volume_mm3. Increases as a "
                      "region becomes more convoluted or fragmented relative "
                      "to its bulk.",
        "source": "derived",
        "category": "morphometric",
    },
    "atrophy_index": {
        "label": "Atrophy index",
        "unit": "dimensionless",
        "definition": "1 - (normalized_volume / training-cohort median "
                      "normalized_volume for the same ROI), clipped to "
                      "[-1, 1]. Positive values indicate a smaller "
                      "head-size-normalized regional volume than the training "
                      "cohort median; negative values indicate a larger one. "
                      "The reference median is fitted on the TRAINING SPLIT "
                      "ONLY.",
        "source": "derived, cohort-referenced (training split only)",
        "category": "volumetric",
    },
}


def validate_spec() -> None:
    """Assert the specification is internally consistent.

    Raises:
        ValueError: If a feature lacks a definition, a definition names an
            unknown feature, or a name appears in both the included and
            excluded lists.
        """
    missing = [f for f in FEATURE_ORDER if f not in FEATURE_DEFINITIONS]
    if missing:
        raise ValueError(f"Features without a definition: {missing}")
    extra = [f for f in FEATURE_DEFINITIONS if f not in FEATURE_ORDER]
    if extra:
        raise ValueError(f"Definitions for unknown features: {extra}")
    overlap = set(FEATURE_ORDER) & set(EXCLUDED_FEATURES)
    if overlap:
        raise ValueError(
            f"Features appear as both included and excluded: {sorted(overlap)}"
        )
    if len(set(FEATURE_ORDER)) != len(FEATURE_ORDER):
        raise ValueError("FEATURE_ORDER contains duplicates")


def describe_features() -> str:
    """Return a human-readable table of every feature and its definition."""
    lines = [f"Node feature vector: {N_FEATURES} features per ROI", ""]
    for name in FEATURE_ORDER:
        d = FEATURE_DEFINITIONS[name]
        lines.append(f"  {name}")
        lines.append(f"    label      : {d['label']} [{d['unit']}]")
        lines.append(f"    category   : {d['category']}  ({d['source']})")
        lines.append(f"    definition : {d['definition']}")
    lines.append("")
    lines.append("Excluded:")
    for name, reason in EXCLUDED_FEATURES.items():
        lines.append(f"  {name}: {reason}")
    return "\n".join(lines)


validate_spec()

__all__ = [
    "RAW_FEATURES",
    "DERIVED_FEATURES",
    "FEATURE_ORDER",
    "N_FEATURES",
    "COHORT_REFERENCED_FEATURES",
    "METADATA_DEPENDENT_FEATURES",
    "EXCLUDED_FEATURES",
    "FEATURE_DEFINITIONS",
    "validate_spec",
    "describe_features",
]
