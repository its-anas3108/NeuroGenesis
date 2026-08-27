"""
OASIS-1 cohort assembly.
========================

Joins three things into one session-level table that the rest of the framework
consumes:

1. the OASIS-1 cross-sectional metadata CSV (clinical variables and CDR),
2. the CDR -> CN/MCI/AD mapping from :mod:`modules.m01_dataset.labels`,
3. the MRI files actually present on disk.

The third point matters. This repository ships the real OASIS-1 metadata table
but no imaging data, so a cohort can be fully *defined* while remaining
un-*processable*. :class:`CohortReport` keeps those two facts separate — the
number of labelled sessions and the number of sessions with an MRI file present
— so that no downstream stage can mistake "235 labelled subjects" for "235
subjects ready to train on".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from modules.common.config import DataConfig, PathsConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_ORDER
from modules.m01_dataset.labels import LabelReport, extract_subject_id, map_labels
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Recognised NIfTI / Analyze / MGZ extensions, longest first so that
#: ``.nii.gz`` is matched before ``.gz``.
MRI_SUFFIXES: Tuple[str, ...] = (".nii.gz", ".nii", ".hdr", ".img", ".mgz")

#: Matches an OASIS-1 session identifier anywhere in a path, e.g.
#: ``.../disc1/OAS1_0001_MR1/mri/orig/001.mgz``.
_SESSION_RE = re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE)


@dataclass
class CohortReport:
    """Summary of cohort assembly, kept deliberately explicit about gaps."""

    metadata_csv: Optional[str] = None
    mri_dir: Optional[str] = None
    #: Dataset identity and OASIS-1 provenance (Section 20).
    dataset_source: str = "OASIS-1"
    oasis1_root: Optional[str] = None
    volume_kind: Optional[str] = None
    n_metadata_rows: int = 0
    n_labeled_sessions: int = 0
    n_labeled_subjects: int = 0
    n_mri_files_found: int = 0
    #: Labelled sessions for which an MRI file was located.
    n_sessions_with_mri: int = 0
    #: MRI files whose session ID is absent from the metadata table.
    n_mri_without_label: int = 0
    stage_counts: Dict[str, int] = field(default_factory=dict)
    #: Stage counts restricted to sessions that actually have an MRI file.
    stage_counts_with_mri: Dict[str, int] = field(default_factory=dict)
    label_report: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_processable(self) -> bool:
        """True only if at least one labelled session has an MRI file on disk."""
        return self.n_sessions_with_mri > 0

    @property
    def can_train(self) -> bool:
        """True if every stage has at least two imaged sessions.

        Two per class is the bare minimum for a stratified split to place a
        member of each class in more than one partition. This is a structural
        feasibility check, not a statement that the sample is adequate.
        """
        if not self.stage_counts_with_mri:
            return False
        return all(self.stage_counts_with_mri.get(s, 0) >= 2 for s in STAGE_ORDER)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot."""
        return {
            "dataset_source": self.dataset_source,
            "oasis1_root": self.oasis1_root,
            "volume_kind": self.volume_kind,
            "is_synthetic": False,
            "metadata_csv": self.metadata_csv,
            "mri_dir": self.mri_dir,
            "n_metadata_rows": self.n_metadata_rows,
            "n_labeled_sessions": self.n_labeled_sessions,
            "n_labeled_subjects": self.n_labeled_subjects,
            "n_mri_files_found": self.n_mri_files_found,
            "n_sessions_with_mri": self.n_sessions_with_mri,
            "n_mri_without_label": self.n_mri_without_label,
            "stage_counts": dict(self.stage_counts),
            "stage_counts_with_mri": dict(self.stage_counts_with_mri),
            "is_processable": self.is_processable,
            "can_train": self.can_train,
            "label_report": self.label_report,
            "warnings": list(self.warnings),
        }

    def summary(self) -> str:
        """Human-readable summary for logs and the dashboard."""
        lines = [
            f"Dataset               : {self.dataset_source}",
            f"OASIS-1 root          : {self.oasis1_root}",
            f"Volume kind           : {self.volume_kind}",
            f"Metadata CSV          : {self.metadata_csv}",
            f"MRI directory         : {self.mri_dir}",
            f"Metadata rows         : {self.n_metadata_rows}",
            f"Labeled sessions      : {self.n_labeled_sessions} "
            f"({self.n_labeled_subjects} subjects)",
            f"  stage counts        : "
            + ", ".join(f"{s}={self.stage_counts.get(s, 0)}" for s in STAGE_ORDER),
            f"MRI files found       : {self.n_mri_files_found}",
            f"Sessions with MRI     : {self.n_sessions_with_mri}",
            f"  stage counts        : "
            + ", ".join(
                f"{s}={self.stage_counts_with_mri.get(s, 0)}" for s in STAGE_ORDER
            ),
            f"Processable           : {self.is_processable}",
            f"Trainable             : {self.can_train}",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def discover_mri_files(mri_dir: Path) -> Dict[str, Path]:
    """Recursively locate one MRI volume per OASIS-1 session.

    Handles the flat, BIDS and FreeSurfer-disc layouts that OASIS-1 ships in by
    matching the session ID anywhere in the path.

    When several candidate files map to the same session (typical for
    FreeSurfer directories, which contain ``orig``, ``T1``, ``brainmask`` and
    more), the shortest path is chosen. Shorter paths sit nearer the session
    root and correspond to the primary acquisition rather than a derived
    volume, and the choice is logged so it can be audited.

    Args:
        mri_dir: Directory to scan recursively.

    Returns:
        ``session_id -> path``. Empty if the directory is absent or has no
        recognised volumes.
    """
    mri_dir = Path(mri_dir)
    if not mri_dir.exists():
        logger.warning("MRI directory does not exist: %s", mri_dir)
        return {}

    candidates: Dict[str, List[Path]] = {}
    for path in sorted(mri_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not any(name.endswith(sfx) for sfx in MRI_SUFFIXES):
            continue
        match = _SESSION_RE.search(path.as_posix())
        session_id = match.group(1).upper() if match else path.stem.split(".")[0]
        candidates.setdefault(session_id, []).append(path)

    resolved: Dict[str, Path] = {}
    for session_id, paths in candidates.items():
        chosen = min(paths, key=lambda p: (len(p.as_posix()), p.as_posix()))
        resolved[session_id] = chosen
        if len(paths) > 1:
            logger.debug(
                "Session %s has %d candidate volumes; selected %s",
                session_id, len(paths), chosen.name,
            )

    logger.info("Discovered %d MRI volume(s) across %d session(s) under %s",
                sum(len(v) for v in candidates.values()), len(resolved), mri_dir)
    return resolved


def build_cohort(
    paths: PathsConfig,
    data: DataConfig,
) -> Tuple[pd.DataFrame, CohortReport]:
    """Assemble the session-level cohort table.

    Args:
        paths: Path configuration (metadata CSV and MRI directory).
        data: Data configuration (label mapping, age filter).

    Returns:
        ``(cohort_df, report)``. ``cohort_df`` columns:

        ``session_id``, ``subject_id``, ``stage``, ``label``, ``mri_path``
        (``None`` when absent), ``has_mri``, plus every original metadata
        column (``Age``, ``M/F``, ``MMSE``, ``CDR``, ``eTIV``, ``nWBV``, ...).

    Raises:
        FileNotFoundError: If the metadata CSV is missing. Without CDR values
            there are no labels, so proceeding would be meaningless.
    """
    csv_path = Path(paths.metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"OASIS-1 metadata CSV not found at {csv_path}. This file supplies "
            "the CDR values that define the CN/MCI/AD labels; the cohort cannot "
            "be built without it."
        )

    metadata = pd.read_csv(csv_path)
    report = CohortReport(
        metadata_csv=csv_path.as_posix(),
        mri_dir=Path(paths.mri_dir).as_posix(),
        n_metadata_rows=len(metadata),
    )

    labeled, label_report = map_labels(
        metadata,
        cdr_to_stage=data.cdr_to_stage,
        missing_cdr_policy=data.missing_cdr_policy,
        min_age=data.min_age,
        max_age=data.max_age,
    )
    report.label_report = label_report.to_dict()
    report.warnings.extend(label_report.warnings)
    report.n_labeled_sessions = len(labeled)
    report.n_labeled_subjects = int(labeled["subject_id"].nunique())
    report.stage_counts = dict(label_report.stage_counts)

    # Route discovery through OASIS1DataManager when an OASIS-1 root is
    # configured. This is the single point where the experiment's MRI
    # data enters: the manager returns only validated real OASIS-1
    # volumes. The legacy directory glob remains for the code paths that
    # operate on already-extracted patch trees.
    oasis_root = getattr(paths, "oasis1_root", None)
    if oasis_root:
        from modules.m01_dataset.oasis1_manager import OASIS1DataManager

        manager = OASIS1DataManager(
            oasis1_root=Path(oasis_root),
            volume_kind=getattr(data, "oasis1_volume_kind", "t88_gfc"),
            metadata_csv=csv_path,
        )
        index = manager.usable_index(
            deep=getattr(data, "deep_validation", True)
        )
        available = {
            str(row["session_id"]): Path(row["mri_path"])
            for _, row in index.iterrows()
        }
        report.dataset_source = "OASIS-1"
        report.oasis1_root = Path(oasis_root).as_posix()
        report.volume_kind = manager.volume_kind
        report.mri_dir = Path(oasis_root).as_posix()
        logger.info(
            "Cohort MRI source: validated OASIS-1 index (%d usable "
            "session(s)) from %s", len(available), oasis_root,
        )
    else:
        available = discover_mri_files(Path(paths.mri_dir))
    report.n_mri_files_found = len(available)

    labeled = labeled.copy()
    labeled["mri_path"] = labeled["session_id"].map(
        lambda s: available[s].as_posix() if s in available else None
    )
    labeled["has_mri"] = labeled["mri_path"].notna()

    report.n_sessions_with_mri = int(labeled["has_mri"].sum())
    with_mri = labeled[labeled["has_mri"]]
    report.stage_counts_with_mri = {
        s: int((with_mri["stage"] == s).sum()) for s in STAGE_ORDER
    }

    labeled_ids = set(labeled["session_id"])
    orphans = [s for s in available if s not in labeled_ids]
    report.n_mri_without_label = len(orphans)

    if not available:
        report.warnings.append(
            f"No MRI volumes were found under {paths.mri_dir}. The cohort is "
            "defined from metadata but nothing can be preprocessed, trained or "
            "evaluated until T1 volumes are placed there."
        )
    elif report.n_sessions_with_mri < report.n_labeled_sessions:
        report.warnings.append(
            f"{report.n_labeled_sessions - report.n_sessions_with_mri} of "
            f"{report.n_labeled_sessions} labeled session(s) have no MRI file "
            "on disk and cannot be processed."
        )
    if orphans:
        report.warnings.append(
            f"{len(orphans)} MRI file(s) have no matching row in the metadata "
            f"table and are therefore unlabeled: {sorted(orphans)[:5]}"
        )
    if available and not report.can_train:
        report.warnings.append(
            "At least one stage has fewer than 2 imaged sessions; a stratified "
            "train/val/test split over all three classes is not yet possible."
        )

    keep_first = [
        "session_id", "subject_id", "stage", "label", "mri_path", "has_mri",
    ]
    ordered = keep_first + [c for c in labeled.columns if c not in keep_first]
    cohort = labeled[ordered].reset_index(drop=True)

    logger.info("Cohort assembled:\n%s", report.summary())
    return cohort, report


def save_cohort(cohort: pd.DataFrame, report: CohortReport,
                out_dir: Path) -> Dict[str, Path]:
    """Persist the cohort table and its report.

    Args:
        cohort: Cohort table from :func:`build_cohort`.
        report: The accompanying :class:`CohortReport`.
        out_dir: Destination directory.

    Returns:
        Mapping of logical name -> written path.
    """
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cohort.csv"
    json_path = out_dir / "cohort_report.json"
    cohort.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json_safe(report.to_dict()), indent=2), encoding="utf-8")

    logger.info("Cohort saved: %s", csv_path)
    return {"cohort_csv": csv_path, "report_json": json_path}


__all__ = [
    "MRI_SUFFIXES",
    "CohortReport",
    "discover_mri_files",
    "build_cohort",
    "save_cohort",
]
