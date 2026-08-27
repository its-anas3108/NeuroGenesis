"""
Dataset integrity guard (Section 8).
====================================

The hard stop between the data layer and training. It asserts, immediately
before any model sees data:

* ``dataset_source == "OASIS-1"``
* ``synthetic_data_enabled == False``
* every training example traces back to a validated OASIS-1 record
* no synthetic marker is present anywhere in the run
* no subject appears in more than one split

If any assertion fails the guard raises
:class:`DatasetIntegrityError`, the offending records are written to the log and
to an integrity report, and training does not start.

Why a positive assertion and not just synthetic detection
---------------------------------------------------------

Detecting a synthetic marker catches the case where someone points the run at a
generated tree. It does **not** catch a sample that reached the training set from
some other path — a stray cached artifact, a leftover file from an earlier run,
a hand-edited feature table. So the guard works the other way round: every
training session ID must be present in the validated OASIS-1 index. Anything that
is not is rejected by name, whatever its origin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

from modules.common.logging_utils import get_logger
from modules.m01_dataset.oasis1_manager import DATASET_NAME
from modules.common.serialization import json_safe

logger = get_logger(__name__)


class DatasetIntegrityError(RuntimeError):
    """Raised when a non-OASIS-1 sample would enter the experiment."""


@dataclass
class IntegrityCheck:
    """One named check and its outcome."""

    name: str
    passed: bool
    detail: str = ""
    offenders: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "offenders": self.offenders[:50],
            "n_offenders": len(self.offenders),
        }


@dataclass
class IntegrityReport:
    """Result of the pre-training integrity gate."""

    dataset_source: str = DATASET_NAME
    synthetic_data_enabled: bool = False
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    checks: List[IntegrityCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every check passed."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[IntegrityCheck]:
        """The checks that failed."""
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "dataset_source": self.dataset_source,
            "synthetic_data_enabled": self.synthetic_data_enabled,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failed": len(self.failures),
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"DATASET          : {self.dataset_source}",
            f"SYNTHETIC DATA   : "
            f"{'ENABLED' if self.synthetic_data_enabled else 'DISABLED'}",
            "",
        ]
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{mark}] {check.name}")
            if check.detail:
                lines.append(f"         {check.detail}")
            if check.offenders:
                lines.append(f"         offenders: {check.offenders[:5]}")
        lines.append("")
        lines.append(
            "INTEGRITY: " + ("PASSED" if self.passed else "FAILED")
        )
        return "\n".join(lines)

    def save(self, out_dir: Path) -> Path:
        """Write the report to ``dataset_integrity_report.json``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "dataset_integrity_report.json"
        path.write_text(json.dumps(json_safe(self.to_dict()), indent=2), encoding="utf-8")
        return path


def check_dataset_integrity(
    training_session_ids: Sequence[str],
    validated_index: pd.DataFrame,
    dataset_source: str,
    synthetic_data_enabled: bool,
    split_assignment: Optional[Dict[str, str]] = None,
    subject_of: Optional[Dict[str, str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    out_dir: Optional[Path] = None,
    raise_on_failure: bool = True,
) -> IntegrityReport:
    """Assert that only validated OASIS-1 data is about to be used.

    Args:
        training_session_ids: Every session that will be seen during training,
            validation or test.
        validated_index: The validated OASIS-1 index; must carry ``session_id``
            and ``usable``.
        dataset_source: The configured dataset name.
        synthetic_data_enabled: The configured synthetic-data flag.
        split_assignment: ``session_id -> split``, for the leakage check.
        subject_of: ``session_id -> subject_id``, for the leakage check.
        provenance: Run provenance; a synthetic marker here fails the gate.
        out_dir: Where to write the integrity report.
        raise_on_failure: Raise :class:`DatasetIntegrityError` on any failure.

    Returns:
        The :class:`IntegrityReport`.

    Raises:
        DatasetIntegrityError: If any check fails and ``raise_on_failure``.
    """
    report = IntegrityReport(
        dataset_source=dataset_source,
        synthetic_data_enabled=bool(synthetic_data_enabled),
    )

    # 1. dataset_source == "OASIS-1"
    report.checks.append(IntegrityCheck(
        name="dataset_source is OASIS-1",
        passed=dataset_source == DATASET_NAME,
        detail=f"configured dataset_source = {dataset_source!r}",
    ))

    # 2. synthetic data disabled
    report.checks.append(IntegrityCheck(
        name="synthetic data disabled",
        passed=not synthetic_data_enabled,
        detail="data.allow_synthetic_data must be False for research runs",
    ))

    # 3. run provenance carries no synthetic marker
    is_synth = bool(provenance and provenance.get("is_synthetic"))
    report.checks.append(IntegrityCheck(
        name="run provenance is not synthetic",
        passed=not is_synth,
        detail=(provenance or {}).get("kind", "real")
        if provenance else "no provenance marker present",
    ))

    # 4. every training session is in the validated OASIS-1 index
    if validated_index is None or validated_index.empty:
        report.checks.append(IntegrityCheck(
            name="all samples originate from validated OASIS-1 records",
            passed=False,
            detail="the validated OASIS-1 index is empty",
        ))
    else:
        usable = set(
            validated_index.loc[
                validated_index.get(
                    "usable", pd.Series(True, index=validated_index.index)
                ), "session_id",
            ].astype(str)
        )
        requested = [str(s) for s in training_session_ids]
        offenders = sorted(set(requested) - usable)
        report.checks.append(IntegrityCheck(
            name="all samples originate from validated OASIS-1 records",
            passed=not offenders,
            detail=(
                f"{len(requested) - len(offenders)}/{len(requested)} session(s) "
                "matched a validated OASIS-1 record"
            ),
            offenders=offenders,
        ))

    # 5. session IDs look like OASIS-1 identifiers
    malformed = sorted(
        s for s in map(str, training_session_ids)
        if not s.upper().startswith("OAS1_")
    )
    report.checks.append(IntegrityCheck(
        name="session identifiers are OASIS-1 formatted",
        passed=not malformed,
        detail="every training session ID must begin with 'OAS1_'",
        offenders=malformed,
    ))

    # 6. no subject spans two splits
    if split_assignment and subject_of:
        by_subject: Dict[str, Set[str]] = {}
        for session, split in split_assignment.items():
            subject = subject_of.get(str(session))
            if subject is None:
                continue
            by_subject.setdefault(subject, set()).add(split)
        leaked = sorted(s for s, splits in by_subject.items() if len(splits) > 1)
        report.checks.append(IntegrityCheck(
            name="no subject appears in more than one split",
            passed=not leaked,
            detail=f"{len(by_subject)} subject(s) checked",
            offenders=leaked,
        ))
    else:
        report.checks.append(IntegrityCheck(
            name="no subject appears in more than one split",
            passed=True,
            detail="not checked here; the split manifest performs its own "
                   "disjointness verification",
        ))

    if out_dir is not None:
        report.save(Path(out_dir))

    if not report.passed:
        for check in report.failures:
            logger.error("DATASET INTEGRITY FAILURE: %s — %s",
                         check.name, check.detail)
            for offender in check.offenders[:20]:
                logger.error("   offending record: %s", offender)
        if raise_on_failure:
            names = ", ".join(c.name for c in report.failures)
            first = report.failures[0]
            raise DatasetIntegrityError(
                "DATASET INTEGRITY FAILURE:\n"
                "Non-OASIS-1 sample detected.\n\n"
                f"Failed check(s): {names}\n"
                f"First failure: {first.name} — {first.detail}\n"
                f"Offending records ({len(first.offenders)}): "
                f"{first.offenders[:10]}\n\n"
                "Training has been stopped. This project accepts the real "
                "OASIS-1 dataset only."
            )
    else:
        logger.info("Dataset integrity check passed:\n%s", report.summary())

    return report


__all__ = [
    "DatasetIntegrityError",
    "IntegrityCheck",
    "IntegrityReport",
    "check_dataset_integrity",
]
