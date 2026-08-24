"""
Canonical speech-ROI and disease-stage constants.
=================================================

Single source of truth for:

* the five speech-related ROIs and their **canonical ordering**,
* their display labels and anatomical metadata,
* the ordered CN -> MCI -> AD disease-stage vocabulary.

Every tensor axis in the framework that indexes ROIs uses :data:`ROI_ORDER`,
and every tensor axis that indexes stages uses :data:`STAGE_ORDER`. Nothing
else in the codebase is permitted to define its own ordering — mismatched ROI
or stage ordering silently corrupts adjacency matrices, attention maps and
confusion matrices, and is not detectable from the numbers alone.

The ROI identifiers are kept byte-identical to ``graph.graph_builder.SPEECH_NODES``
so that the pre-existing (and preserved) Harvard-Oxford extraction, feature
extraction and graph construction code remains directly compatible.
"""

from __future__ import annotations

from typing import Dict, Final, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Speech ROIs
# ──────────────────────────────────────────────────────────────────────────────

#: Canonical ROI ordering. **Do not reorder** — every ROI-indexed tensor axis,
#: adjacency matrix, attention matrix and vulnerability vector in the framework
#: follows this exact order.
ROI_ORDER: Final[List[str]] = [
    "Broca_Area",
    "Wernicke_Area",
    "Insula",
    "Inferior_Frontal_Gyrus",
    "Superior_Temporal_Gyrus",
]

N_ROI: Final[int] = len(ROI_ORDER)

#: Position of each ROI in :data:`ROI_ORDER`.
ROI_INDEX: Final[Dict[str, int]] = {name: i for i, name in enumerate(ROI_ORDER)}

#: Short display names used in figures, tables and the dashboard.
ROI_SHORT: Final[Dict[str, str]] = {
    "Broca_Area": "Broca",
    "Wernicke_Area": "Wernicke",
    "Insula": "Insula",
    "Inferior_Frontal_Gyrus": "IFG",
    "Superior_Temporal_Gyrus": "STG",
}

#: Reverse lookup from short display name to canonical identifier.
ROI_FROM_SHORT: Final[Dict[str, str]] = {v: k for k, v in ROI_SHORT.items()}

#: Anatomical/functional metadata, used for report text and dashboard panels.
ROI_METADATA: Final[Dict[str, Dict[str, str]]] = {
    "Broca_Area": {
        "label": "Broca's Area",
        "region": "IFG pars triangularis + pars opercularis",
        "function": "Speech production, phonological encoding",
        "hemisphere": "Left",
        "brodmann": "44, 45",
        "color": "#388bfd",
    },
    "Wernicke_Area": {
        "label": "Wernicke's Area",
        "region": "Posterior superior temporal gyrus + planum temporale",
        "function": "Speech comprehension, auditory word recognition",
        "hemisphere": "Left",
        "brodmann": "22",
        "color": "#3fb950",
    },
    "Insula": {
        "label": "Insula",
        "region": "Insular cortex",
        "function": "Articulatory planning, phonological awareness",
        "hemisphere": "Bilateral",
        "brodmann": "13, 14",
        "color": "#e3b341",
    },
    "Inferior_Frontal_Gyrus": {
        "label": "Inferior Frontal Gyrus",
        "region": "Full inferior frontal gyrus / frontal operculum",
        "function": "Syntactic processing, verbal working memory",
        "hemisphere": "Left",
        "brodmann": "44, 45, 47",
        "color": "#ff7b72",
    },
    "Superior_Temporal_Gyrus": {
        "label": "Superior Temporal Gyrus",
        "region": "Superior temporal gyrus",
        "function": "Auditory-verbal processing, spectrotemporal analysis",
        "hemisphere": "Bilateral",
        "brodmann": "22, 41, 42",
        "color": "#d2a8ff",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Disease stages
# ──────────────────────────────────────────────────────────────────────────────

#: Ordered disease-stage vocabulary. The order encodes clinical severity and is
#: relied upon by the ordinal stage-order regularizer and by the Stage-TGT
#: stage-token sequence — it is an *ordered* scale, not an arbitrary label set.
STAGE_ORDER: Final[List[str]] = ["CN", "MCI", "AD"]

N_STAGE: Final[int] = len(STAGE_ORDER)

#: Stage name -> integer class index used by all classifiers and loss functions.
STAGE_INDEX: Final[Dict[str, int]] = {s: i for i, s in enumerate(STAGE_ORDER)}

#: Integer class index -> stage name.
STAGE_FROM_INDEX: Final[Dict[int, str]] = {i: s for i, s in enumerate(STAGE_ORDER)}

#: Human-readable stage descriptions for reports.
STAGE_DESCRIPTION: Final[Dict[str, str]] = {
    "CN": "Cognitively normal",
    "MCI": "Mild cognitive impairment / very mild dementia",
    "AD": "Alzheimer's-type dementia",
}

#: Plot colours per stage, consistent across every figure in the framework.
STAGE_COLOR: Final[Dict[str, str]] = {
    "CN": "#3fb950",
    "MCI": "#e3b341",
    "AD": "#ff7b72",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def roi_short(name: str) -> str:
    """Return the short display name for a canonical ROI identifier.

    Falls back to the input string if the identifier is unknown, so that
    logging and figure labelling never raise on an unexpected ROI name.
    """
    return ROI_SHORT.get(name, name)


def stage_pairs() -> List[Tuple[str, str]]:
    """Return the three stage contrasts required by the statistical analysis.

    Returns:
        ``[("CN", "MCI"), ("MCI", "AD"), ("CN", "AD")]`` — the order in which
        contrasts are reported in Table 5.
    """
    return [("CN", "MCI"), ("MCI", "AD"), ("CN", "AD")]


def validate_roi_order(names: List[str]) -> None:
    """Assert that ``names`` matches :data:`ROI_ORDER` exactly.

    Args:
        names: ROI identifiers in the order used by some external array.

    Raises:
        ValueError: If the ordering differs in any position. The message names
            the offending index, because a silent ROI transposition produces
            plausible-looking but wrong adjacency and attention matrices.
    """
    if list(names) != ROI_ORDER:
        raise ValueError(
            "ROI ordering mismatch — every ROI-indexed axis must follow "
            f"ROI_ORDER.\n  expected: {ROI_ORDER}\n  received: {list(names)}"
        )
