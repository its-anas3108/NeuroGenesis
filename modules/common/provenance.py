"""
Data provenance detection (Section 28).
=======================================

Single source of truth for the question *"is anything in this outputs tree
derived from synthetic data?"* — asked by the dashboard, the report generator,
the figure writers and the results tables before they render any number.

Two independent generators can put synthetic data into a run, and both must be
detected:

``tools/make_smoke_artifacts.py``
    Writes ``SMOKE_TEST.json`` into the **outputs** root. It skips imaging
    entirely and fabricates ROI patches directly.

``tools/make_synthetic_mri.py``
    Writes ``SYNTHETIC_MRI.json`` into the **dataset** directory. The imaging
    pipeline then runs for real on phantom volumes, so the outputs tree looks
    exactly like a genuine run and carries no marker of its own.

The second case is the dangerous one. Its artifacts are produced by the real
preprocessing chain, real atlas registration and real feature extraction, so
nothing about their shape or content reveals that the underlying images were
parametric ellipsoids. :func:`stamp_outputs` therefore copies the provenance
into the outputs tree at preprocessing time, and every consumer reads it through
:func:`detect_provenance`.

A tree with no marker is reported as ``real``. That default is safe only because
both generators stamp unconditionally; nothing else in the codebase writes a
marker, so an unmarked tree genuinely did not come from a generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Written into the outputs root by the smoke-artifact generator.
SMOKE_MARKER = "SMOKE_TEST.json"

#: Written into the dataset directory by the synthetic-MRI generator.
SYNTHETIC_MRI_MARKER = "SYNTHETIC_MRI.json"

#: Written into the outputs root by :func:`stamp_outputs`.
PROVENANCE_MARKER = "DATA_PROVENANCE.json"


@dataclass
class Provenance:
    """What kind of data produced the artifacts in an outputs tree."""

    #: ``"real"`` | ``"synthetic_patches"`` | ``"synthetic_mri"``
    kind: str = "real"
    warning: str = ""
    #: The raw marker payload(s) found.
    details: Dict[str, Any] = field(default_factory=dict)
    #: Where each marker was found.
    sources: List[str] = field(default_factory=list)

    @property
    def is_synthetic(self) -> bool:
        """True when any part of this tree derives from generated data."""
        return self.kind != "real"

    @property
    def banner(self) -> str:
        """Short banner text for figures and dashboard headers."""
        if not self.is_synthetic:
            return ""
        if self.kind == "synthetic_mri":
            return "SYNTHETIC PHANTOM MRI - NOT A RESEARCH RESULT"
        return "SYNTHETIC SMOKE-TEST DATA - NOT A RESEARCH RESULT"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "kind": self.kind,
            "is_synthetic": self.is_synthetic,
            "warning": self.warning,
            "banner": self.banner,
            "sources": list(self.sources),
            "details": self.details,
        }


def _read(path: Path) -> Optional[Dict[str, Any]]:
    """Read a marker file, returning ``None`` if absent or malformed."""
    if not Path(path).exists():
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # An unreadable marker still means the tree was marked. Returning None
        # here would silently upgrade it to "real", which is the one direction
        # this function must never fail in.
        return {"unparseable": True}


def detect_provenance(outputs_root: Path,
                      mri_dir: Optional[Path] = None) -> Provenance:
    """Determine whether an outputs tree derives from synthetic data.

    Args:
        outputs_root: The outputs directory to classify.
        mri_dir: Optional dataset directory, checked as a fallback when the
            outputs tree carries no stamped marker of its own.

    Returns:
        A :class:`Provenance`. Synthetic findings take precedence: if any marker
        is present the result is synthetic, never real.
    """
    outputs_root = Path(outputs_root)
    provenance = Provenance()

    stamped = _read(outputs_root / PROVENANCE_MARKER)
    if stamped and stamped.get("kind") in ("synthetic_mri", "synthetic_patches"):
        provenance.kind = stamped["kind"]
        provenance.warning = stamped.get("warning", "")
        provenance.details["stamped"] = stamped
        provenance.sources.append((outputs_root / PROVENANCE_MARKER).as_posix())

    smoke = _read(outputs_root / SMOKE_MARKER)
    if smoke:
        provenance.kind = "synthetic_patches"
        provenance.warning = smoke.get("warning", provenance.warning)
        provenance.details["smoke_test"] = smoke
        provenance.sources.append((outputs_root / SMOKE_MARKER).as_posix())

    if mri_dir is not None:
        synthetic = _read(Path(mri_dir) / SYNTHETIC_MRI_MARKER)
        if synthetic:
            # Synthetic imaging outranks synthetic patches in the label because
            # it is the harder case to notice: the outputs look like a real run.
            provenance.kind = "synthetic_mri"
            provenance.warning = synthetic.get("warning", provenance.warning)
            provenance.details["synthetic_mri"] = synthetic
            provenance.sources.append(
                (Path(mri_dir) / SYNTHETIC_MRI_MARKER).as_posix()
            )

    return provenance


def stamp_outputs(outputs_root: Path, mri_dir: Optional[Path] = None
                  ) -> Optional[Path]:
    """Copy synthetic provenance from the dataset into the outputs tree.

    Called at the start of preprocessing. Without it, a run over phantom MRI
    produces an outputs tree indistinguishable from a real one, and every
    downstream consumer would render its numbers unmarked.

    Args:
        outputs_root: Outputs directory to stamp.
        mri_dir: Dataset directory to inspect.

    Returns:
        The written marker path, or ``None`` when the source data is real and
        no stamp is needed.
    """
    provenance = detect_provenance(outputs_root, mri_dir)
    if not provenance.is_synthetic:
        return None

    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)
    path = outputs_root / PROVENANCE_MARKER
    path.write_text(json.dumps(provenance.to_dict(), indent=2), encoding="utf-8")
    logger.warning(
        "This outputs tree is stamped as %s. %s",
        provenance.kind, provenance.warning,
    )
    return path


__all__ = [
    "SMOKE_MARKER",
    "SYNTHETIC_MRI_MARKER",
    "PROVENANCE_MARKER",
    "Provenance",
    "detect_provenance",
    "stamp_outputs",
]
