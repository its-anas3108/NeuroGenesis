"""
OASIS-1 CDR -> CN/MCI/AD label mapping (Section 11).
====================================================

Section 11 forbids assuming a label mapping without checking the actual OASIS-1
metadata. The mapping implemented here was derived from the real
``dataset/oasis_cross-sectional.csv`` shipped with this repository, whose
measured CDR distribution is:

===== ======== =========
CDR   Stage    Sessions
===== ======== =========
0.0   CN       135
0.5   MCI      70
1.0   AD       28
2.0   AD       2
NaN   *none*   201
===== ======== =========

Two points that materially affect every downstream claim:

**1. CDR 0.5 is mapped to MCI.** In OASIS-1 documentation CDR 0.5 denotes "very
mild dementia". It is the conventional stand-in for the MCI stage in
cross-sectional OASIS-1 work, but it is not an independent clinical MCI
diagnosis. The report and dashboard must therefore describe this class as
"MCI / very mild dementia", which :data:`STAGE_DESCRIPTION` already does.

**2. The 201 missing-CDR sessions are excluded by default.** They are not
unlabelled AD-risk cases; they are young subjects (18-40) for whom CDR was
never administered. Assigning them CN would roughly triple the CN class while
making age almost perfectly predictive of stage, so a model could reach high
accuracy without learning anything about pathology. ``missing_cdr_policy="cn"``
exists purely so that this confound can be demonstrated deliberately in a
sensitivity analysis, and :func:`map_labels` returns a warning whenever it is
used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_INDEX, STAGE_ORDER

logger = get_logger(__name__)

#: Default mapping, keyed by the string form of the CDR value.
DEFAULT_CDR_TO_STAGE: Dict[str, str] = {
    "0.0": "CN",
    "0.5": "MCI",
    "1.0": "AD",
    "2.0": "AD",
    "3.0": "AD",  # not present in OASIS-1, included for completeness
}


@dataclass
class LabelReport:
    """Audit trail of a label-mapping pass (Section 11 validation requirement)."""

    n_input_rows: int = 0
    n_labeled: int = 0
    n_missing_cdr: int = 0
    n_unmapped_cdr: int = 0
    n_excluded_age: int = 0
    #: Stage -> session count after mapping.
    stage_counts: Dict[str, int] = field(default_factory=dict)
    #: Stage -> unique-subject count after mapping.
    subject_counts: Dict[str, int] = field(default_factory=dict)
    #: CDR values present in the data but absent from the mapping.
    unmapped_cdr_values: List[float] = field(default_factory=list)
    missing_cdr_policy: str = "exclude"
    warnings: List[str] = field(default_factory=list)

    @property
    def imbalance_ratio(self) -> Optional[float]:
        """Ratio of the largest to smallest stage count, or ``None`` if empty.

        Kept explicit because Section 11 requires class imbalance to stay
        visible rather than be quietly absorbed by a weighted loss.
        """
        counts = [c for c in self.stage_counts.values() if c > 0]
        if len(counts) < 2:
            return None
        return max(counts) / min(counts)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot for the experiment manifest."""
        return {
            "n_input_rows": self.n_input_rows,
            "n_labeled": self.n_labeled,
            "n_missing_cdr": self.n_missing_cdr,
            "n_unmapped_cdr": self.n_unmapped_cdr,
            "n_excluded_age": self.n_excluded_age,
            "stage_counts": dict(self.stage_counts),
            "subject_counts": dict(self.subject_counts),
            "unmapped_cdr_values": list(self.unmapped_cdr_values),
            "missing_cdr_policy": self.missing_cdr_policy,
            "imbalance_ratio": self.imbalance_ratio,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Multi-line human-readable summary for logs and the dashboard."""
        lines = [
            f"Input rows            : {self.n_input_rows}",
            f"Labeled sessions      : {self.n_labeled}",
            f"Missing CDR           : {self.n_missing_cdr} "
            f"(policy: {self.missing_cdr_policy})",
        ]
        if self.n_excluded_age:
            lines.append(f"Excluded by age filter: {self.n_excluded_age}")
        if self.n_unmapped_cdr:
            lines.append(
                f"Unmapped CDR values   : {self.n_unmapped_cdr} "
                f"{self.unmapped_cdr_values}"
            )
        for stage in STAGE_ORDER:
            lines.append(
                f"  {stage:<4s} sessions={self.stage_counts.get(stage, 0):<5d} "
                f"subjects={self.subject_counts.get(stage, 0)}"
            )
        ratio = self.imbalance_ratio
        if ratio is not None:
            lines.append(f"Class imbalance ratio : {ratio:.2f}:1 (max:min)")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def _cdr_key(value: Any) -> Optional[str]:
    """Normalise a raw CDR cell to a mapping key, or ``None`` if missing.

    OASIS CSVs have been seen with CDR written as ``0``, ``0.0``, ``"0.5"`` and
    blank. Normalising through ``float`` then formatting with one decimal makes
    all of those hit the same key.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return None


def _is_missing_key(value: Any) -> bool:
    """Return whether a mapped CDR key represents "no assessment".

    This cannot be a plain ``value is None`` test. ``Series.map`` does not
    preserve ``None``: on a float64 column pandas infers a ``str``/object result
    dtype and converts the returned ``None`` into ``float('nan')``. An identity
    check against ``None`` therefore misses every unassessed session and
    misreports all 201 of them as carrying an *unmapped* CDR value, which reads
    as a data problem rather than as the documented exclusion it is.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def extract_subject_id(session_id: str) -> str:
    """Strip the OASIS session suffix to obtain the subject identifier.

    ``"OAS1_0001_MR1" -> "OAS1_0001"``. This is the unit that splits operate
    on: OASIS-1 contains 20 ``MR2`` rescans of already-present subjects, and
    letting a subject's two sessions land in different splits would leak.

    Args:
        session_id: A session ID such as ``"OAS1_0001_MR1"``.

    Returns:
        The subject ID, or ``session_id`` unchanged if it carries no ``_MR``
        suffix.
    """
    text = str(session_id).strip()
    idx = text.rfind("_MR")
    return text[:idx] if idx > 0 else text


def map_labels(
    metadata: pd.DataFrame,
    cdr_to_stage: Optional[Dict[str, str]] = None,
    missing_cdr_policy: str = "exclude",
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    id_column: str = "ID",
    cdr_column: str = "CDR",
    age_column: str = "Age",
) -> Tuple[pd.DataFrame, LabelReport]:
    """Map CDR values to CN/MCI/AD stage labels with full validation.

    Args:
        metadata: The OASIS-1 cross-sectional table.
        cdr_to_stage: CDR-string -> stage mapping. Defaults to
            :data:`DEFAULT_CDR_TO_STAGE`.
        missing_cdr_policy: ``"exclude"`` (drop unassessed sessions) or
            ``"cn"`` (label them CN — age-confounded, warned about).
        min_age: Optional inclusive lower age bound.
        max_age: Optional inclusive upper age bound.
        id_column: Session-ID column name.
        cdr_column: CDR column name.
        age_column: Age column name.

    Returns:
        ``(labeled_df, report)`` where ``labeled_df`` adds ``subject_id``,
        ``session_id``, ``stage`` and ``label`` columns and contains only rows
        that received a stage.

    Raises:
        KeyError: If a required column is missing — surfacing a schema mismatch
            immediately rather than silently producing an empty cohort.
        ValueError: If ``missing_cdr_policy`` is not recognised.
    """
    mapping = dict(cdr_to_stage or DEFAULT_CDR_TO_STAGE)
    if missing_cdr_policy not in ("exclude", "cn"):
        raise ValueError(
            f"missing_cdr_policy must be 'exclude' or 'cn', got {missing_cdr_policy!r}"
        )

    for col in (id_column, cdr_column):
        if col not in metadata.columns:
            raise KeyError(
                f"Metadata is missing the required column {col!r}. "
                f"Present columns: {list(metadata.columns)}"
            )

    report = LabelReport(
        n_input_rows=len(metadata),
        missing_cdr_policy=missing_cdr_policy,
    )

    df = metadata.copy()
    df["session_id"] = df[id_column].astype(str).str.strip()
    df["subject_id"] = df["session_id"].map(extract_subject_id)

    keys = df[cdr_column].map(_cdr_key)
    df["cdr_key"] = keys

    stages: List[Optional[str]] = []
    unmapped: set = set()
    n_missing = 0
    for key in keys:
        if _is_missing_key(key):
            n_missing += 1
            stages.append("CN" if missing_cdr_policy == "cn" else None)
        elif key in mapping:
            stages.append(mapping[key])
        else:
            stages.append(None)
            try:
                unmapped.add(float(key))
            except (TypeError, ValueError):
                # An unparseable key is still an unmapped value; record it as
                # text rather than dropping the fact that it occurred.
                unmapped.add(str(key))

    df["stage"] = stages
    report.n_missing_cdr = n_missing
    report.n_unmapped_cdr = int(sum(
        1 for k, s in zip(keys, stages)
        if not _is_missing_key(k) and s is None
    ))
    report.unmapped_cdr_values = sorted(unmapped, key=str)

    if report.n_unmapped_cdr:
        report.warnings.append(
            f"{report.n_unmapped_cdr} session(s) carry CDR values outside the "
            f"configured mapping {report.unmapped_cdr_values} and were dropped. "
            f"Extend data.cdr_to_stage if these should be labeled."
        )
    if missing_cdr_policy == "cn" and report.n_missing_cdr:
        report.warnings.append(
            f"{report.n_missing_cdr} session(s) with no CDR assessment were "
            "labeled CN. In OASIS-1 these are young subjects who were never "
            "clinically assessed, so age becomes strongly predictive of stage. "
            "Results under this policy are age-confounded and must be reported "
            "as a sensitivity analysis, not as the main result."
        )

    labeled = df[df["stage"].notna()].copy()

    # Age filtering is applied after mapping so the report can distinguish
    # "no CDR" from "filtered out by age".
    if (min_age is not None or max_age is not None) and age_column in labeled.columns:
        before = len(labeled)
        ages = pd.to_numeric(labeled[age_column], errors="coerce")
        keep = pd.Series(True, index=labeled.index)
        if min_age is not None:
            keep &= ages >= min_age
        if max_age is not None:
            keep &= ages <= max_age
        # A non-numeric or absent age cannot be verified against the bound, so
        # it is dropped rather than assumed to pass.
        keep &= ages.notna()
        labeled = labeled[keep].copy()
        report.n_excluded_age = before - len(labeled)
    elif (min_age is not None or max_age is not None):
        report.warnings.append(
            f"Age filter requested but column {age_column!r} is absent; "
            "no age filtering was applied."
        )

    labeled["label"] = labeled["stage"].map(STAGE_INDEX).astype(int)

    report.n_labeled = len(labeled)
    report.stage_counts = {
        s: int((labeled["stage"] == s).sum()) for s in STAGE_ORDER
    }
    report.subject_counts = {
        s: int(labeled.loc[labeled["stage"] == s, "subject_id"].nunique())
        for s in STAGE_ORDER
    }

    empty = [s for s, c in report.stage_counts.items() if c == 0]
    if empty:
        report.warnings.append(
            f"Stage(s) {empty} have zero sessions after mapping. "
            "Three-class classification is not possible with an empty class."
        )
    small = [s for s, c in report.stage_counts.items() if 0 < c < 30]
    if small:
        report.warnings.append(
            f"Stage(s) {small} have fewer than 30 sessions. Point estimates on a "
            "single split are not reportable; use repeated stratified evaluation "
            "with confidence intervals."
        )

    logger.info("Label mapping complete:\n%s", report.summary())
    return labeled, report


def class_weights(labels: np.ndarray, n_classes: int = len(STAGE_ORDER)
                  ) -> np.ndarray:
    """Compute inverse-frequency class weights normalised to mean 1.

    Args:
        labels: Integer class labels of the **training split only**. Computing
            weights over the full dataset would leak test-set composition into
            the loss.
        n_classes: Number of classes.

    Returns:
        Float64 array of length ``n_classes``. Absent classes receive weight
        ``0.0`` rather than an infinite weight, which would make the loss NaN.
    """
    labels = np.asarray(labels).astype(int).ravel()
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return weights


__all__ = [
    "DEFAULT_CDR_TO_STAGE",
    "LabelReport",
    "extract_subject_id",
    "map_labels",
    "class_weights",
]
