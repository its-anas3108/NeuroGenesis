"""
OASIS1DataManager — the single source of MRI data for the experiment.
=====================================================================

Discovers, validates and indexes the **real** OASIS-1 cross-sectional dataset.
Nothing in this module generates data. If the dataset is absent or unusable it
says so and refuses to proceed; it never substitutes anything.

Dataset
-------

OASIS-1 cross-sectional, Washington University / OASIS.
Official source: <https://sites.wustl.edu/oasisbrains/home/oasis-1/>

The archives are supplied locally by the user. This module never downloads.

Layout it understands
---------------------

The canonical OASIS-1 disc layout, at any nesting depth::

    <root>/discN/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/
        OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.{img,hdr}
        OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.{img,hdr}

Volume selection
----------------

``t88_gfc`` (default)
    Atlas-registered to Talairach-88, 1 mm isotropic, averaged across the
    subject's acquisitions, gain-field and N4 corrected. **Skull present**, so
    the pipeline's own skull-stripping stage runs as designed.

``t88_masked_gfc``
    The same volume with OASIS's brain mask already applied. Selectable, but not
    the default, because it would make the skull-stripping stage a no-op and
    silently change what the pipeline does.

The T88 volumes are used rather than ``RAW/`` or ``SUBJ_111/`` because the
Harvard-Oxford atlas is defined in a standard space: a T88 volume carries an
atlas-centred affine, so ROI localization lands on the correct anatomy without
a separate registration step.

Format adaptations
------------------

Three properties of the real files require handling, and this module is the only
place they are handled:

1. **Analyze `.img`/`.hdr` pairs, not NIfTI.** The `.hdr` is the canonical path;
   nibabel reads the `.img` alongside it. Treating both as volumes would
   double-count every session.
2. **A singleton 4th axis.** Volumes load as ``(176, 208, 176, 1)``. Downstream
   stages expect 3-D. Only a *trailing axis of extent 1* is squeezed; a genuinely
   4-D volume is rejected rather than silently reduced.
3. **Big-endian int16 (`>i2`).** Cast to native float32 for SimpleITK and torch.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_ORDER
from modules.m01_dataset.labels import extract_subject_id

logger = get_logger(__name__)

#: Canonical dataset identity. Asserted by the integrity guard before training.
DATASET_NAME = "OASIS-1"
DATASET_SOURCE = ("Washington University / OASIS — "
                  "https://sites.wustl.edu/oasisbrains/home/oasis-1/")

#: Matches an OASIS-1 session directory, e.g. ``OAS1_0001_MR1``.
SESSION_DIR_RE = re.compile(r"^OAS1_\d{4}_MR\d+$")

#: Matches a session ID anywhere inside a path string.
SESSION_ANY_RE = re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE)

#: Selectable volume kinds -> the filename fragment that identifies them.
VOLUME_KINDS: Dict[str, str] = {
    "t88_gfc": "_t88_gfc",
    "t88_masked_gfc": "_t88_masked_gfc",
}

#: Expected properties of an OASIS-1 T88 volume. Deviations are recorded as
#: validation findings rather than silently accepted.
EXPECTED_SHAPE: Tuple[int, int, int] = (176, 208, 176)
EXPECTED_ZOOMS: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class OASIS1Session:
    """One discovered OASIS-1 imaging session."""

    session_id: str
    subject_id: str
    #: The ``.hdr`` path. nibabel reads the paired ``.img`` from it.
    volume_path: Optional[str] = None
    volume_kind: Optional[str] = None
    disc: Optional[str] = None
    #: Companion ``.img`` path, recorded for the integrity check.
    image_path: Optional[str] = None

    # ── Populated by validation ───────────────────────────────────────────
    readable: bool = False
    shape: Optional[List[int]] = None
    dtype: Optional[str] = None
    zooms: Optional[List[float]] = None
    orientation: Optional[str] = None
    affine: Optional[List[List[float]]] = None
    nonzero_voxels: Optional[int] = None
    intensity_range: Optional[List[float]] = None
    file_bytes: Optional[int] = None
    #: Why this session cannot be used, if it cannot.
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when the volume was read and passed structural validation."""
        return self.readable and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


class OASIS1DataManager:
    """Discover, validate and index the real OASIS-1 dataset.

    Args:
        oasis1_root: Directory holding the extracted OASIS-1 discs. Supplied by
            configuration; never hard-coded.
        volume_kind: Key of :data:`VOLUME_KINDS`.
        metadata_csv: The OASIS-1 cross-sectional metadata table, which carries
            the CDR values that define the labels.

    Raises:
        ValueError: If ``volume_kind`` is not recognised.
    """

    def __init__(
        self,
        oasis1_root: Path,
        volume_kind: str = "t88_gfc",
        metadata_csv: Optional[Path] = None,
    ) -> None:
        if volume_kind not in VOLUME_KINDS:
            raise ValueError(
                f"Unknown volume_kind {volume_kind!r}. "
                f"Available: {sorted(VOLUME_KINDS)}"
            )
        self.root = Path(oasis1_root)
        self.volume_kind = volume_kind
        self.fragment = VOLUME_KINDS[volume_kind]
        self.metadata_csv = Path(metadata_csv) if metadata_csv else None
        self._sessions: Optional[List[OASIS1Session]] = None

    # ── Presence ──────────────────────────────────────────────────────────

    def root_exists(self) -> bool:
        """Whether the configured OASIS-1 root exists."""
        return self.root.exists() and self.root.is_dir()

    def missing_data_message(self) -> str:
        """The message to show when real OASIS-1 data is not available."""
        return (
            "REAL OASIS-1 DATA REQUIRED\n\n"
            f"No usable OASIS-1 volumes were found under: {self.root}\n\n"
            "This project accepts the real OASIS-1 cross-sectional dataset "
            "only. Obtain it from "
            "https://sites.wustl.edu/oasisbrains/home/oasis-1/ and extract the "
            "discs, then set paths.oasis1_root to the extraction directory.\n\n"
            "Synthetic, mock and substitute datasets are disabled for research "
            "execution and will not be used."
        )

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self, refresh: bool = False) -> List[OASIS1Session]:
        """Recursively locate one volume per OASIS-1 session.

        Keys on the ``.hdr`` file, because nibabel reads an Analyze pair from
        its header; globbing both extensions would register every session twice.

        Args:
            refresh: Re-scan even if a previous scan is cached.

        Returns:
            Discovered sessions, sorted by session ID. Empty if the root is
            absent or holds no matching volumes.
        """
        if self._sessions is not None and not refresh:
            return self._sessions

        if not self.root_exists():
            logger.error("OASIS-1 root does not exist: %s", self.root)
            self._sessions = []
            return self._sessions

        found: Dict[str, OASIS1Session] = {}
        duplicates: Dict[str, List[str]] = {}

        for header in sorted(self.root.rglob("*.hdr")):
            name = header.name
            # Exclude the masked variant when the plain one is requested: the
            # plain fragment is a substring of the masked filename.
            if self.fragment not in name:
                continue
            if (self.volume_kind == "t88_gfc"
                    and "_masked_gfc" in name):
                continue

            match = SESSION_ANY_RE.search(header.as_posix())
            if not match:
                logger.warning("Skipping file with no OASIS session ID: %s",
                               header)
                continue
            session_id = match.group(1).upper()

            image = header.with_suffix(".img")
            disc = next(
                (p for p in header.parts if p.lower().startswith("disc")), None
            )

            if session_id in found:
                duplicates.setdefault(session_id, []).append(header.as_posix())
                continue

            found[session_id] = OASIS1Session(
                session_id=session_id,
                subject_id=extract_subject_id(session_id),
                volume_path=header.as_posix(),
                image_path=image.as_posix() if image.exists() else None,
                volume_kind=self.volume_kind,
                disc=disc,
            )

        for session_id, extra in duplicates.items():
            found[session_id].warnings.append(
                f"{len(extra)} additional volume(s) matched this session; the "
                f"first was used. Others: {extra[:2]}"
            )

        self._sessions = [found[k] for k in sorted(found)]
        logger.info(
            "OASIS-1 discovery: %d session(s) of kind %r under %s",
            len(self._sessions), self.volume_kind, self.root,
        )
        return self._sessions

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, sessions: Optional[List[OASIS1Session]] = None,
                 deep: bool = True) -> List[OASIS1Session]:
        """Read each volume's header (and optionally its data) and record facts.

        Args:
            sessions: Sessions to validate; defaults to :meth:`discover`.
            deep: Also load the voxel array to measure intensity range and
                non-zero count. Slower, but it is the only way to detect a
                truncated or all-zero volume, which a header check cannot.

        Returns:
            The same session objects, populated in place.
        """
        sessions = sessions if sessions is not None else self.discover()
        if not sessions:
            return sessions

        try:
            import nibabel as nib
        except ImportError:
            for session in sessions:
                session.error = (
                    "nibabel is not installed, so OASIS-1 volumes cannot be "
                    "read. Install it with `pip install nibabel`."
                )
            return sessions

        for session in sessions:
            path = Path(session.volume_path) if session.volume_path else None
            if path is None or not path.exists():
                session.error = f"volume file not found: {session.volume_path}"
                continue
            if session.image_path is None:
                session.error = (
                    "Analyze header has no companion .img file; the pair is "
                    "incomplete."
                )
                continue

            try:
                session.file_bytes = Path(session.image_path).stat().st_size
                image = nib.load(str(path))
                shape = tuple(int(s) for s in image.shape)
                session.shape = list(shape)
                session.dtype = str(image.get_data_dtype())
                zooms = tuple(float(z) for z in image.header.get_zooms()[:3])
                session.zooms = list(zooms)
                session.affine = np.asarray(image.affine).tolist()
                try:
                    session.orientation = "".join(nib.aff2axcodes(image.affine))
                except Exception:  # noqa: BLE001 - orientation is informational
                    session.orientation = None

                spatial = self._spatial_shape(shape)
                if spatial is None:
                    session.error = (
                        f"unexpected volume dimensionality {shape}; expected a "
                        "3-D volume or a 4-D volume with a trailing singleton "
                        "axis"
                    )
                    continue

                if spatial != EXPECTED_SHAPE:
                    session.warnings.append(
                        f"spatial shape {spatial} differs from the expected "
                        f"OASIS-1 T88 shape {EXPECTED_SHAPE}"
                    )
                if not all(abs(z - e) < 1e-3
                           for z, e in zip(zooms, EXPECTED_ZOOMS)):
                    session.warnings.append(
                        f"voxel size {zooms} differs from the expected "
                        f"{EXPECTED_ZOOMS} mm"
                    )

                if deep:
                    data = np.asanyarray(image.dataobj)
                    finite = np.isfinite(data)
                    if not finite.all():
                        session.warnings.append(
                            f"{int((~finite).sum())} non-finite voxel(s)"
                        )
                    session.nonzero_voxels = int((data > 0).sum())
                    session.intensity_range = [
                        float(np.nanmin(data)), float(np.nanmax(data))
                    ]
                    if session.nonzero_voxels == 0:
                        session.error = "volume contains no non-zero voxels"
                        continue
                    if session.intensity_range[1] <= session.intensity_range[0]:
                        session.error = (
                            "volume has a degenerate intensity range "
                            f"{session.intensity_range}"
                        )
                        continue

                session.readable = True

            except Exception as exc:  # noqa: BLE001 - any read failure excludes
                session.error = f"{type(exc).__name__}: {exc}"

        usable = sum(1 for s in sessions if s.usable)
        logger.info("OASIS-1 validation: %d/%d session(s) usable",
                    usable, len(sessions))
        return sessions

    @staticmethod
    def _spatial_shape(shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int]]:
        """Return the 3-D spatial shape, or ``None`` if the volume is not 3-D-able.

        A trailing axis of extent 1 is a packaging artifact of the OASIS Analyze
        files and is dropped. Any other 4-D shape carries real data on the extra
        axis and must not be silently collapsed.
        """
        if len(shape) == 3:
            return shape  # type: ignore[return-value]
        if len(shape) == 4 and shape[3] == 1:
            return shape[:3]  # type: ignore[return-value]
        return None

    # ── Loading ───────────────────────────────────────────────────────────

    @staticmethod
    def load_volume(volume_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load one OASIS-1 volume as a 3-D float32 array plus its affine.

        The single place the three OASIS format adaptations are applied:
        the Analyze pair is read from its header, a trailing singleton axis is
        squeezed, and big-endian int16 is cast to native float32.

        Args:
            volume_path: Path to the ``.hdr`` file.

        Returns:
            ``(volume, affine)`` with ``volume`` of shape ``(X, Y, Z)``.

        Raises:
            ValueError: If the volume is not 3-D-able.
        """
        import nibabel as nib

        image = nib.load(str(volume_path))
        data = np.asanyarray(image.dataobj)

        if data.ndim == 4 and data.shape[3] == 1:
            data = data[..., 0]
        elif data.ndim != 3:
            raise ValueError(
                f"{volume_path}: expected a 3-D volume or a 4-D volume with a "
                f"trailing singleton axis; got shape {data.shape}"
            )

        # `astype` also resolves the big-endian byte order to native.
        return data.astype(np.float32), np.asarray(image.affine, dtype=np.float64)

    # ── Index ─────────────────────────────────────────────────────────────

    def index(self, validated: bool = True, deep: bool = True) -> pd.DataFrame:
        """Return the session index as a DataFrame.

        Args:
            validated: Run validation before indexing.
            deep: Passed through to :meth:`validate`.

        Returns:
            One row per discovered session, with the discovery and validation
            facts. Includes unusable sessions, flagged as such, so exclusions
            are visible rather than silent.
        """
        sessions = self.discover()
        if validated:
            sessions = self.validate(sessions, deep=deep)
        if not sessions:
            return pd.DataFrame(columns=[
                "session_id", "subject_id", "mri_path", "usable", "error",
            ])

        rows = []
        for session in sessions:
            rows.append({
                "session_id": session.session_id,
                "subject_id": session.subject_id,
                "mri_path": session.volume_path,
                "image_path": session.image_path,
                "volume_kind": session.volume_kind,
                "disc": session.disc,
                "usable": session.usable,
                "readable": session.readable,
                "shape": str(session.shape) if session.shape else None,
                "dtype": session.dtype,
                "voxel_size_mm": str(session.zooms) if session.zooms else None,
                "orientation": session.orientation,
                "nonzero_voxels": session.nonzero_voxels,
                "intensity_min": (session.intensity_range[0]
                                  if session.intensity_range else None),
                "intensity_max": (session.intensity_range[1]
                                  if session.intensity_range else None),
                "file_bytes": session.file_bytes,
                "error": session.error,
                "n_warnings": len(session.warnings),
                "warnings": "; ".join(session.warnings) or None,
                "dataset_source": DATASET_NAME,
            })
        return pd.DataFrame(rows)

    def usable_index(self, deep: bool = True) -> pd.DataFrame:
        """Return only the sessions that passed validation."""
        table = self.index(validated=True, deep=deep)
        if table.empty:
            return table
        return table[table["usable"]].reset_index(drop=True)

    def provenance(self) -> Dict[str, Any]:
        """Return the dataset provenance block recorded with every artifact."""
        return {
            "dataset_source": DATASET_NAME,
            "source_description": DATASET_SOURCE,
            "oasis1_root": self.root.as_posix(),
            "volume_kind": self.volume_kind,
            "volume_description": (
                "PROCESSED/MPRAGE/T88_111 — Talairach-88 atlas space, 1 mm "
                "isotropic, averaged across acquisitions, N4 and gain-field "
                "corrected"
                + (", OASIS brain mask applied"
                   if self.volume_kind == "t88_masked_gfc"
                   else ", skull present")
            ),
            "is_synthetic": False,
            "preprocessing_note": (
                "OASIS applied N4 bias-field and gain-field correction before "
                "distribution (filename components 'n4' and 'gfc'). The "
                "pipeline's M3 stage applies N4 again; this duplication is "
                "harmless but is recorded rather than hidden."
            ),
        }


__all__ = [
    "DATASET_NAME",
    "DATASET_SOURCE",
    "VOLUME_KINDS",
    "EXPECTED_SHAPE",
    "EXPECTED_ZOOMS",
    "OASIS1Session",
    "OASIS1DataManager",
]
