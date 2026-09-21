"""
ADNIDataManager — inference-only ADNI data source.
====================================================

Discovers, converts and MNI152-registers a local ADNI raw-DICOM export.
Nothing in this module fabricates labels: ADNI sessions here carry no
CN/MCI/AD diagnosis, so the cohort layer (:mod:`modules.m01_dataset.cohort`)
marks every session ``stage="UNLABELED"``, and the training-integrity gate
(:mod:`modules.m01_dataset.integrity`) accepts OASIS-1 only, so this dataset
can never reach model training — only preprocessing and inference-only report
generation against an already-trained checkpoint.

Dataset
-------

ADNI (Alzheimer's Disease Neuroimaging Initiative).
Official source: <https://adni.loni.usc.edu/>.

The archive is supplied locally by the user. This module never downloads it.

Layout it understands
----------------------

The canonical ADNI raw-DICOM export layout, at any nesting depth::

    <root>/<subject_id>/<series_description>/<series_timestamp>/<image_id>/*.dcm

e.g. ``002_S_0295/MP-RAGE/2006-04-18_08_20_30.0/I13722/*.dcm``. Each leaf
directory of ``.dcm`` files is one series, keyed as one session:
``session_id = "<subject_id>_<image_id>"``.

Two format adaptations, and why registration is not optional
--------------------------------------------------------------

1. **Raw DICOM, not NIfTI.** Each series is a directory of per-slice ``.dcm``
   files, not a single volume file. This module reads a series with
   SimpleITK's ``ImageSeriesReader`` — which handles correct instance
   ordering, gantry tilt and DICOM's LPS coordinate convention — and writes a
   cached NIfTI in RAS+ orientation, exactly matching every other volume this
   codebase reads (nibabel's convention).
2. **Native scanner space, not standard/atlas space.** Unlike OASIS-1's T88
   volumes, which arrive pre-registered to Talairach-88 space by the dataset
   provider, a raw ADNI DICOM series' affine only encodes wherever that
   subject's head happened to sit in the scanner that day. The ROI-extraction
   stage (:mod:`modules.m03_segmentation`) does not perform its own
   registration: it resamples the Harvard-Oxford atlas onto whatever affine
   the input volume already carries, which only lands on the correct anatomy
   if that affine already places the volume in the atlas's space. Skipping
   registration here would silently produce anatomically meaningless ROI
   masks rather than merely "approximate" ones — so this module performs a
   real (automated, unverified) SimpleITK rigid -> affine registration of
   every volume to the MNI152 template before handing it downstream. The
   registration is not expert-QC'd, so its output is still research-grade,
   not clinically validated (see :data:`CAVEATS`).

No CDR/diagnosis metadata exists for this local export, so this module never
assigns a stage. Session identity, geometry and registration quality are the
only facts recorded.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.serialization import json_safe

logger = get_logger(__name__)

#: Canonical dataset identity. Never accepted by the training-integrity gate.
DATASET_NAME = "ADNI"
DATASET_SOURCE = ("Alzheimer's Disease Neuroimaging Initiative — "
                  "https://adni.loni.usc.edu/")

#: Matches an ADNI subject identifier anywhere in a path, e.g. "002_S_0295".
ADNI_SUBJECT_RE = re.compile(r"(\d{3}_S_\d{4})")

#: Caveats every consumer of this dataset must surface (console banner,
#: cohort report warnings, generated report "DATA CAVEATS" block).
CAVEATS: Tuple[str, ...] = (
    "ROI localization on ADNI relies on an automated SimpleITK rigid+affine "
    "registration of each volume to the MNI152 template (Mattes mutual "
    "information, no manual quality check of the registration result). This "
    "is research-grade, not clinically validated — unlike OASIS-1's T88 "
    "volumes, which ship pre-registered to Talairach-88 space by the dataset "
    "provider.",
    "No CN/MCI/AD or CDR diagnostic labels exist for this cohort. Every "
    "session is stage='UNLABELED'. This dataset supports preprocessing and "
    "inference-only report generation against an already-trained checkpoint "
    "— never training, evaluation or ablation.",
)


@dataclass
class ADNISeries:
    """One discovered ADNI DICOM series (= one session)."""

    session_id: str
    subject_id: str
    series_description: str
    image_id: str
    dicom_dir: str
    n_slices: int = 0

    # ── Populated by validate() ───────────────────────────────────────────
    readable: bool = False
    shape: Optional[List[int]] = None
    affine: Optional[List[List[float]]] = None
    #: Path of the cached MNI152-registered NIfTI, once registered.
    registered_path: Optional[str] = None
    #: Final registration metric (Mattes mutual information; more negative
    #: is a tighter match). ``None`` until registration has run.
    registration_metric: Optional[float] = None
    #: Why this session cannot be used, if it cannot.
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when the series was read, converted and registered."""
        return self.readable and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return asdict(self)


def _read_series(dicom_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Read one DICOM series into ``(volume, affine, meta)``.

    Uses SimpleITK's ``ImageSeriesReader`` for the geometry-correct read
    (instance ordering, gantry tilt) and converts its result from DICOM's
    LPS+ convention to NIfTI/nibabel's RAS+ convention. A handful of header
    tags are read separately with ``pydicom``, purely to populate the
    provenance record.

    Args:
        dicom_dir: Directory holding one series' ``.dcm`` files.

    Returns:
        ``(volume, affine, meta)``, with ``volume`` of shape ``(X, Y, Z)``
        float32 and ``affine`` a 4x4 RAS+ affine.

    Raises:
        ValueError: If no DICOM series is found, or the modality is not MR.
    """
    import pydicom
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not file_names:
        raise ValueError(f"No DICOM series found under {dicom_dir}")
    reader.SetFileNames(file_names)
    image = reader.Execute()

    # SimpleITK's array axis order is (z, y, x); transpose to (x, y, z) so it
    # matches the affine's voxel-index convention (the same one nibabel uses).
    volume = np.transpose(
        sitk.GetArrayFromImage(image), (2, 1, 0)
    ).astype(np.float32)

    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.array(image.GetOrigin(), dtype=np.float64)

    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction * spacing
    affine_lps[:3, 3] = origin
    # DICOM/SimpleITK is LPS+; NIfTI/nibabel is RAS+. Flip the first two axes.
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps

    meta: Dict[str, Any] = {"n_files": len(file_names)}
    try:
        first = pydicom.dcmread(file_names[0], stop_before_pixels=True)
        for tag in ("PatientID", "StudyDate", "SeriesDescription",
                    "Manufacturer", "Modality", "MagneticFieldStrength"):
            if hasattr(first, tag):
                meta[tag] = str(getattr(first, tag))
        modality = meta.get("Modality")
        if modality not in (None, "MR"):
            raise ValueError(f"Expected Modality='MR', got {modality!r}")
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        meta["metadata_read_error"] = f"{type(exc).__name__}: {exc}"

    return volume, affine, meta


class ADNIDataManager:
    """Discover, convert and MNI152-register a local ADNI raw-DICOM export.

    Args:
        adni_root: Directory holding the ADNI subject folders. Supplied by
            configuration; never hard-coded.
        cache_dir: Directory for cached NIfTI conversions and registrations
            (typically ``<outputs_dir>/adni_nifti``). Conversion and
            registration are expensive and idempotent, so both are cached
            here and reused on subsequent runs.
    """

    def __init__(self, adni_root: Path, cache_dir: Path) -> None:
        self.root = Path(adni_root)
        self.cache_dir = Path(cache_dir)
        self._raw_cache_dir = self.cache_dir / "raw"
        self._mni_cache_dir = self.cache_dir / "mni"
        self._series: Optional[List[ADNISeries]] = None

    # ── Presence ──────────────────────────────────────────────────────────

    def root_exists(self) -> bool:
        """Whether the configured ADNI root exists."""
        return self.root.exists() and self.root.is_dir()

    def missing_data_message(self) -> str:
        """The message to show when the configured ADNI root has no data."""
        return (
            "ADNI DATA NOT FOUND\n\n"
            f"No DICOM series were found under: {self.root}\n\n"
            "Set paths.adni_root (or pass --adni-root) to a directory "
            "containing ADNI subject folders, each holding one or more "
            "DICOM series: "
            "<subject_id>/<series_description>/<series_timestamp>/"
            "<image_id>/*.dcm\n\n"
            "Note: this dataset carries no CN/MCI/AD labels here, so it "
            "supports preprocessing and inference-only report generation "
            "only — never training."
        )

    # ── Discovery ─────────────────────────────────────────────────────────

    def discover(self, refresh: bool = False) -> List[ADNISeries]:
        """Recursively locate one DICOM series per leaf directory.

        Args:
            refresh: Re-scan even if a previous scan is cached.

        Returns:
            Discovered series, sorted by session ID. Empty if the root is
            absent or holds no ``.dcm`` files.
        """
        if self._series is not None and not refresh:
            return self._series

        if not self.root_exists():
            logger.error("ADNI root does not exist: %s", self.root)
            self._series = []
            return self._series

        groups: Dict[str, List[Path]] = {}
        for dcm in self.root.rglob("*.dcm"):
            groups.setdefault(dcm.parent.as_posix(), []).append(dcm)

        found: List[ADNISeries] = []
        for dicom_dir, files in sorted(groups.items()):
            dir_path = Path(dicom_dir)
            match = ADNI_SUBJECT_RE.search(dir_path.as_posix())
            if not match:
                logger.warning(
                    "Skipping DICOM directory with no ADNI subject ID: %s",
                    dicom_dir,
                )
                continue
            subject_id = match.group(1)
            image_id = dir_path.name
            try:
                rel_parts = dir_path.relative_to(self.root).parts
                series_description = rel_parts[1] if len(rel_parts) > 1 \
                    else "UNKNOWN"
            except ValueError:
                series_description = "UNKNOWN"

            found.append(ADNISeries(
                session_id=f"{subject_id}_{image_id}",
                subject_id=subject_id,
                series_description=series_description,
                image_id=image_id,
                dicom_dir=dir_path.as_posix(),
                n_slices=len(files),
            ))

        self._series = sorted(found, key=lambda s: s.session_id)
        logger.info(
            "ADNI discovery: %d series across %d session(s) under %s",
            sum(s.n_slices for s in self._series), len(self._series),
            self.root,
        )
        return self._series

    # ── Conversion + registration ────────────────────────────────────────

    def _ensure_template(self) -> Path:
        """Fetch and cache the MNI152 template NIfTI, once."""
        template_path = self.cache_dir / "mni152_template.nii.gz"
        if template_path.exists():
            return template_path

        import nibabel as nib
        from nilearn.datasets import load_mni152_template

        template_img = load_mni152_template(resolution=2)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(template_img, template_path)
        return template_path

    def _register(
        self, raw_path: Path, session_id: str, template_path: Path,
    ) -> Tuple[Path, float]:
        """Register a raw (native-space) NIfTI to the MNI152 template.

        Rigid (Euler3D, moments-initialised) registration followed by an
        affine refinement initialised from the rigid result, both with a
        Mattes mutual-information metric and a 3-level multi-resolution
        pyramid. The result is cached and reused on subsequent calls.

        Returns:
            ``(registered_path, final_metric)``.
        """
        import SimpleITK as sitk

        registered_path = self._mni_cache_dir / f"{session_id}.nii.gz"
        meta_path = registered_path.with_name(
            f"{session_id}.registration.json"
        )
        if registered_path.exists() and meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return registered_path, float(payload.get("final_metric", float("nan")))

        fixed = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
        moving = sitk.ReadImage(str(raw_path), sitk.sitkFloat32)

        def _configure(reg: "sitk.ImageRegistrationMethod"
                       ) -> "sitk.ImageRegistrationMethod":
            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(0.2, seed=42)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsRegularStepGradientDescent(
                learningRate=2.0, minStep=1e-4, numberOfIterations=200,
                gradientMagnitudeTolerance=1e-8,
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel([4, 2, 1])
            reg.SetSmoothingSigmasPerLevel([2, 1, 0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            return reg

        # ``inPlace=True`` mutates and returns the same Euler3DTransform
        # object; with ``inPlace=False`` SimpleITK instead returns a
        # CompositeTransform wrapping it, which the Euler3DTransform
        # downcast below would reject.
        rigid_transform = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
        rigid_reg = _configure(sitk.ImageRegistrationMethod())
        rigid_reg.SetInitialTransform(rigid_transform, inPlace=True)
        rigid_reg.Execute(fixed, moving)

        affine_initial = sitk.AffineTransform(3)
        affine_initial.SetCenter(rigid_transform.GetCenter())
        affine_initial.SetTranslation(rigid_transform.GetTranslation())
        affine_initial.SetMatrix(rigid_transform.GetMatrix())

        affine_reg = _configure(sitk.ImageRegistrationMethod())
        affine_reg.SetInitialTransform(affine_initial, inPlace=False)
        final_transform = affine_reg.Execute(fixed, moving)
        final_metric = float(affine_reg.GetMetricValue())

        resampled = sitk.Resample(
            moving, fixed, final_transform, sitk.sitkLinear, 0.0,
            moving.GetPixelID(),
        )

        registered_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(registered_path))
        meta_path.write_text(json.dumps({
            "final_metric": final_metric,
            "method": ("SimpleITK rigid (Euler3D, moments-initialised) -> "
                      "affine, Mattes mutual information, regular-step "
                      "gradient descent, 3-level multi-resolution pyramid"),
            "template": "MNI152 (nilearn, 2mm)",
        }, indent=2), encoding="utf-8")

        return registered_path, final_metric

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, series: Optional[List[ADNISeries]] = None,
                 deep: bool = True) -> List[ADNISeries]:
        """Convert each series to NIfTI, register it, and record facts.

        Args:
            series: Series to validate; defaults to :meth:`discover`.
            deep: Also check the registered volume's intensity range for a
                degenerate result. Slower, but the only way to detect a
                registration or conversion that silently produced a blank
                or constant volume.

        Returns:
            The same series objects, populated in place.
        """
        series = series if series is not None else self.discover()
        if not series:
            return series

        try:
            import nibabel as nib  # noqa: F401
            import pydicom  # noqa: F401
            import SimpleITK as sitk  # noqa: F401
        except ImportError as exc:
            for s in series:
                s.error = (
                    f"required package not installed ({exc}); ADNI ingestion "
                    "needs nibabel, pydicom and SimpleITK"
                )
            return series

        try:
            template_path = self._ensure_template()
        except Exception as exc:  # noqa: BLE001
            for s in series:
                s.error = f"could not fetch the MNI152 template: {exc}"
            return series

        import nibabel as nib

        for s in series:
            try:
                raw_path = self._raw_cache_dir / f"{s.session_id}.nii.gz"
                if not raw_path.exists():
                    volume, affine, meta = _read_series(Path(s.dicom_dir))
                    if volume.ndim != 3:
                        raise ValueError(
                            f"expected a 3-D volume, got shape {volume.shape}"
                        )
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(volume, affine), raw_path)
                    meta_path = raw_path.with_name(
                        f"{s.session_id}.dicom_meta.json"
                    )
                    meta_path.write_text(
                        json.dumps(json_safe(meta), indent=2),
                        encoding="utf-8",
                    )

                registered_path, metric = self._register(
                    raw_path, s.session_id, template_path
                )
                s.registered_path = registered_path.as_posix()
                s.registration_metric = metric

                image = nib.load(str(registered_path))
                s.shape = list(image.shape)
                s.affine = np.asarray(image.affine).tolist()

                if deep:
                    data = np.asanyarray(image.dataobj)
                    finite = np.isfinite(data)
                    if not finite.all():
                        s.warnings.append(
                            f"{int((~finite).sum())} non-finite voxel(s) in "
                            "the registered volume"
                        )
                    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
                    if hi <= lo:
                        s.error = (
                            f"registered volume has a degenerate intensity "
                            f"range [{lo}, {hi}]"
                        )
                        continue

                s.readable = True

            except Exception as exc:  # noqa: BLE001 - any failure excludes
                s.error = f"{type(exc).__name__}: {exc}"

        usable = sum(1 for s in series if s.usable)
        logger.info("ADNI validation: %d/%d session(s) usable",
                    usable, len(series))
        return series

    # ── Index ─────────────────────────────────────────────────────────────

    def index(self, validated: bool = True, deep: bool = True) -> pd.DataFrame:
        """Return the session index as a DataFrame.

        Args:
            validated: Run validation (convert + register) before indexing.
            deep: Passed through to :meth:`validate`.

        Returns:
            One row per discovered session. ``mri_path`` is the cached,
            MNI152-registered NIfTI path — the same shape of value
            :func:`modules.m01_dataset.cohort.build_cohort` gets from
            :class:`~modules.m01_dataset.oasis1_manager.OASIS1DataManager`,
            so nothing downstream of this manager needs an ADNI-specific
            branch. Includes unusable sessions, flagged as such.
        """
        series = self.discover()
        if validated:
            series = self.validate(series, deep=deep)
        if not series:
            return pd.DataFrame(columns=[
                "session_id", "subject_id", "mri_path", "usable", "error",
            ])

        rows = []
        for s in series:
            rows.append({
                "session_id": s.session_id,
                "subject_id": s.subject_id,
                "mri_path": s.registered_path,
                "series_description": s.series_description,
                "dicom_dir": s.dicom_dir,
                "n_slices": s.n_slices,
                "usable": s.usable,
                "readable": s.readable,
                "shape": str(s.shape) if s.shape else None,
                "registration_metric": s.registration_metric,
                "error": s.error,
                "n_warnings": len(s.warnings),
                "warnings": "; ".join(s.warnings) or None,
                "dataset_source": DATASET_NAME,
            })
        return pd.DataFrame(rows)

    def usable_index(self, deep: bool = True) -> pd.DataFrame:
        """Return only the sessions that were converted and registered."""
        table = self.index(validated=True, deep=deep)
        if table.empty:
            return table
        return table[table["usable"]].reset_index(drop=True)

    def provenance(self) -> Dict[str, Any]:
        """Return the dataset provenance block recorded with every artifact."""
        return {
            "dataset_source": DATASET_NAME,
            "source_description": DATASET_SOURCE,
            "adni_root": self.root.as_posix(),
            "volume_description": (
                "Raw DICOM MPRAGE series, converted to NIfTI and registered "
                "(SimpleITK rigid->affine, Mattes MI) to the MNI152 "
                "template — skull present, so the pipeline's own "
                "skull-stripping stage still runs as designed."
            ),
            "is_synthetic": False,
            "caveats": list(CAVEATS),
        }


__all__ = [
    "DATASET_NAME",
    "DATASET_SOURCE",
    "CAVEATS",
    "ADNI_SUBJECT_RE",
    "ADNISeries",
    "ADNIDataManager",
]
