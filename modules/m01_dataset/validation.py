"""
OASIS-1 dataset validation (Section 6).
=======================================

Runs before any training and answers one question: *which real OASIS-1 sessions
are actually usable, and why is each excluded one excluded?*

Checks performed
----------------

* number of MRI volumes found, and unique subjects
* duplicate sessions and repeat-visit subjects
* unreadable or truncated volumes
* invalid dimensionality, unexpected shape, unexpected voxel size
* orientation
* sessions with no metadata row, and metadata rows with no volume
* missing labels (CDR not assessed)
* class distribution over the labelled, imaged cohort
* missing clinical variables

Outputs
-------

``outputs/dataset_validation/oasis1_validation_report.json``
``outputs/dataset_validation/oasis1_dataset_summary.csv``

Nothing here repairs, imputes or substitutes. A session that fails is excluded
and counted; the report states the reason for every exclusion so the final study
size is explainable rather than merely asserted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_ORDER
from modules.m01_dataset.labels import extract_subject_id, map_labels
from modules.m01_dataset.oasis1_manager import (
    DATASET_NAME,
    DATASET_SOURCE,
    OASIS1DataManager,
)
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Clinical columns the OASIS-1 cross-sectional table is expected to carry.
EXPECTED_CLINICAL_COLUMNS: Tuple[str, ...] = (
    "ID", "M/F", "Hand", "Age", "Educ", "SES", "MMSE", "CDR",
    "eTIV", "nWBV", "ASF",
)


@dataclass
class ValidationReport:
    """Outcome of validating the uploaded OASIS-1 dataset."""

    dataset_source: str = DATASET_NAME
    source_description: str = DATASET_SOURCE
    is_synthetic: bool = False
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    oasis1_root: Optional[str] = None
    volume_kind: Optional[str] = None

    # ── Imaging ───────────────────────────────────────────────────────────
    n_volumes_found: int = 0
    n_sessions: int = 0
    n_unique_subjects: int = 0
    n_readable: int = 0
    n_unreadable: int = 0
    n_usable: int = 0
    unreadable_sessions: List[Dict[str, str]] = field(default_factory=list)
    shape_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    orientations: Dict[str, int] = field(default_factory=dict)
    voxel_sizes: Dict[str, int] = field(default_factory=dict)
    shapes: Dict[str, int] = field(default_factory=dict)
    dtypes: Dict[str, int] = field(default_factory=dict)

    # ── Subjects ──────────────────────────────────────────────────────────
    subjects_with_repeat_sessions: Dict[str, int] = field(default_factory=dict)

    # ── Metadata ──────────────────────────────────────────────────────────
    metadata_csv: Optional[str] = None
    n_metadata_rows: int = 0
    missing_clinical_columns: List[str] = field(default_factory=list)
    clinical_missing_counts: Dict[str, int] = field(default_factory=dict)
    n_sessions_without_metadata: int = 0
    sessions_without_metadata: List[str] = field(default_factory=list)
    n_metadata_without_volume: int = 0
    metadata_without_volume: List[str] = field(default_factory=list)

    # ── Labels ────────────────────────────────────────────────────────────
    n_labelled: int = 0
    n_missing_cdr: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_counts_imaged: Dict[str, int] = field(default_factory=dict)
    subject_counts_imaged: Dict[str, int] = field(default_factory=dict)

    # ── Verdict ───────────────────────────────────────────────────────────
    passed: bool = False
    is_subset: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def n_analysable(self) -> int:
        """Sessions that are imaged, readable and labelled — the study size."""
        return sum(self.class_counts_imaged.get(s, 0) for s in STAGE_ORDER)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        payload = {k: v for k, v in self.__dict__.items()}
        payload["n_analysable"] = self.n_analysable
        return payload

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"DATASET          : {self.dataset_source}",
            f"SOURCE           : {self.source_description}",
            f"SYNTHETIC DATA   : DISABLED (is_synthetic={self.is_synthetic})",
            f"ROOT             : {self.oasis1_root}",
            f"VOLUME KIND      : {self.volume_kind}",
            "",
            f"MRI volumes found: {self.n_volumes_found}",
            f"Sessions         : {self.n_sessions}  "
            f"(unique subjects {self.n_unique_subjects})",
            f"Readable         : {self.n_readable}   "
            f"Unreadable: {self.n_unreadable}",
            f"Usable volumes   : {self.n_usable}",
            "",
            f"Metadata rows    : {self.n_metadata_rows}",
            f"Labelled sessions: {self.n_labelled}  "
            f"(missing CDR: {self.n_missing_cdr})",
            f"Sessions w/o metadata : {self.n_sessions_without_metadata}",
            f"Metadata w/o volume   : {self.n_metadata_without_volume}",
            "",
            "Analysable cohort (imaged AND labelled):",
        ]
        for stage in STAGE_ORDER:
            lines.append(
                f"   {stage:<4s} sessions={self.class_counts_imaged.get(stage, 0):<5d}"
                f" subjects={self.subject_counts_imaged.get(stage, 0)}"
            )
        lines.append(f"   TOTAL = {self.n_analysable}")
        lines.append("")
        lines.append(f"VERDICT          : {'PASS' if self.passed else 'FAIL'}")
        if self.is_subset:
            lines.append("MODE             : OASIS-1 SUBSET MODE")
        for e in self.errors:
            lines.append(f"ERROR  : {e}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def validate_oasis1(
    manager: OASIS1DataManager,
    metadata_csv: Optional[Path] = None,
    cdr_to_stage: Optional[Dict[str, str]] = None,
    missing_cdr_policy: str = "exclude",
    deep: bool = True,
    expected_total_sessions: int = 436,
) -> Tuple[ValidationReport, pd.DataFrame]:
    """Validate the uploaded OASIS-1 dataset.

    Args:
        manager: A configured :class:`OASIS1DataManager`.
        metadata_csv: The OASIS-1 cross-sectional CSV. Defaults to the
            manager's.
        cdr_to_stage: CDR-to-stage mapping.
        missing_cdr_policy: Passed to the label mapper.
        deep: Read voxel data during validation, not just headers.
        expected_total_sessions: The full OASIS-1 cross-sectional session count,
            used only to decide whether this is a subset (Section 18).

    Returns:
        ``(report, summary_table)``.
    """
    report = ValidationReport(
        oasis1_root=manager.root.as_posix(),
        volume_kind=manager.volume_kind,
    )

    # ── Imaging ───────────────────────────────────────────────────────────
    if not manager.root_exists():
        report.errors.append(
            f"OASIS-1 root does not exist: {manager.root}"
        )
        return report, pd.DataFrame()

    table = manager.index(validated=True, deep=deep)
    report.n_volumes_found = len(table)
    report.n_sessions = int(table["session_id"].nunique()) if not table.empty else 0
    report.n_unique_subjects = (
        int(table["subject_id"].nunique()) if not table.empty else 0
    )

    if table.empty:
        report.errors.append(
            "No OASIS-1 volumes were discovered. "
            + manager.missing_data_message()
        )
        return report, table

    report.n_readable = int(table["readable"].sum())
    report.n_unreadable = int((~table["readable"]).sum())
    report.n_usable = int(table["usable"].sum())

    for _, row in table[~table["usable"]].iterrows():
        report.unreadable_sessions.append({
            "session_id": str(row["session_id"]),
            "path": str(row["mri_path"]),
            "error": str(row["error"]),
        })

    for column, target in (
        ("orientation", report.orientations),
        ("voxel_size_mm", report.voxel_sizes),
        ("shape", report.shapes),
        ("dtype", report.dtypes),
    ):
        counts = table[column].fillna("unknown").value_counts()
        target.update({str(k): int(v) for k, v in counts.items()})

    anomalies = table[table["n_warnings"] > 0]
    for _, row in anomalies.iterrows():
        report.shape_anomalies.append({
            "session_id": str(row["session_id"]),
            "shape": str(row["shape"]),
            "voxel_size_mm": str(row["voxel_size_mm"]),
            "warnings": str(row["warnings"]),
        })

    repeats = table.groupby("subject_id")["session_id"].nunique()
    report.subjects_with_repeat_sessions = {
        str(k): int(v) for k, v in repeats[repeats > 1].items()
    }

    # ── Metadata ──────────────────────────────────────────────────────────
    csv_path = Path(metadata_csv) if metadata_csv else manager.metadata_csv
    if csv_path is None or not Path(csv_path).exists():
        report.errors.append(
            f"OASIS-1 metadata CSV not found at {csv_path}. It supplies the CDR "
            "values that define the CN/MCI/AD labels; without it no labelled "
            "cohort exists."
        )
        return report, table

    report.metadata_csv = Path(csv_path).as_posix()
    metadata = pd.read_csv(csv_path)
    report.n_metadata_rows = len(metadata)

    report.missing_clinical_columns = [
        c for c in EXPECTED_CLINICAL_COLUMNS if c not in metadata.columns
    ]
    if report.missing_clinical_columns:
        report.warnings.append(
            f"Metadata is missing expected clinical column(s): "
            f"{report.missing_clinical_columns}"
        )
    for column in EXPECTED_CLINICAL_COLUMNS:
        if column in metadata.columns:
            report.clinical_missing_counts[column] = int(
                metadata[column].isna().sum()
            )

    metadata_sessions = set(metadata["ID"].astype(str).str.strip()) \
        if "ID" in metadata.columns else set()
    imaged_sessions = set(table["session_id"].astype(str))

    without_metadata = sorted(imaged_sessions - metadata_sessions)
    without_volume = sorted(metadata_sessions - imaged_sessions)
    report.n_sessions_without_metadata = len(without_metadata)
    report.sessions_without_metadata = without_metadata[:50]
    report.n_metadata_without_volume = len(without_volume)
    report.metadata_without_volume = without_volume[:50]

    if without_metadata:
        report.warnings.append(
            f"{len(without_metadata)} imaged session(s) have no metadata row "
            "and cannot be labelled or used."
        )
    if without_volume:
        report.warnings.append(
            f"{len(without_volume)} metadata row(s) have no imaging volume."
        )

    # ── Labels ────────────────────────────────────────────────────────────
    labelled, label_report = map_labels(
        metadata, cdr_to_stage=cdr_to_stage,
        missing_cdr_policy=missing_cdr_policy,
    )
    report.n_labelled = label_report.n_labeled
    report.n_missing_cdr = label_report.n_missing_cdr
    report.class_counts = dict(label_report.stage_counts)
    report.warnings.extend(label_report.warnings)

    usable_sessions = set(table.loc[table["usable"], "session_id"].astype(str))
    analysable = labelled[labelled["session_id"].astype(str).isin(usable_sessions)]
    report.class_counts_imaged = {
        s: int((analysable["stage"] == s).sum()) for s in STAGE_ORDER
    }
    report.subject_counts_imaged = {
        s: int(analysable.loc[analysable["stage"] == s, "subject_id"].nunique())
        for s in STAGE_ORDER
    }

    # ── Verdict ───────────────────────────────────────────────────────────
    if report.n_usable == 0:
        report.errors.append("No usable OASIS-1 volume passed validation.")
    empty_classes = [s for s in STAGE_ORDER
                     if report.class_counts_imaged.get(s, 0) == 0]
    if empty_classes:
        report.errors.append(
            f"Stage(s) {empty_classes} have no imaged, labelled session. "
            "Three-class classification is impossible."
        )
    too_small = [s for s in STAGE_ORDER
                 if 0 < report.class_counts_imaged.get(s, 0) < 2]
    if too_small:
        report.errors.append(
            f"Stage(s) {too_small} have fewer than 2 sessions; a stratified "
            "train/val/test split is impossible."
        )

    report.is_subset = report.n_sessions < expected_total_sessions
    if report.is_subset:
        report.warnings.append(
            f"OASIS-1 SUBSET MODE: {report.n_sessions} of the expected "
            f"{expected_total_sessions} cross-sectional sessions are present. "
            "Results describe this subset, not the complete OASIS-1 dataset."
        )
    if report.n_unreadable:
        report.warnings.append(
            f"{report.n_unreadable} volume(s) could not be read and were "
            "excluded. See unreadable_sessions for the reason on each."
        )
    smallest = min(
        (v for v in report.class_counts_imaged.values() if v > 0), default=0
    )
    if 0 < smallest < 30:
        report.warnings.append(
            f"The smallest analysable class has {smallest} session(s). Point "
            "estimates on a single split are not reportable; use repeated "
            "stratified evaluation with confidence intervals."
        )

    report.passed = not report.errors
    logger.info("OASIS-1 validation:\n%s", report.summary())
    return report, table


def save_validation(
    report: ValidationReport,
    table: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Path]:
    """Write the validation report and dataset summary CSV.

    Args:
        report: The validation report.
        table: The per-session index.
        out_dir: Destination, conventionally
            ``outputs/dataset_validation``.

    Returns:
        Mapping of logical name -> written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "oasis1_validation_report.json"
    report_path.write_text(json.dumps(json_safe(report.to_dict()), indent=2, default=str),
                           encoding="utf-8")

    summary_path = out_dir / "oasis1_dataset_summary.csv"
    if not table.empty:
        table.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(columns=["session_id", "subject_id", "mri_path", "usable",
                              "error"]).to_csv(summary_path, index=False)

    logger.info("Validation artifacts written to %s", out_dir)
    return {"report_json": report_path, "summary_csv": summary_path}


__all__ = [
    "EXPECTED_CLINICAL_COLUMNS",
    "ValidationReport",
    "validate_oasis1",
    "save_validation",
]
