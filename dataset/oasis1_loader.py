"""
NeuroGenesis — OASIS-1 Dataset Loader
======================================
File   : dataset/oasis1_loader.py
Purpose: Replace synthetic MRI input with real OASIS-1 data.

What this file does
-------------------
1. Recursively finds all .nii / .nii.gz files under a given MRI root directory.
2. Extracts the OASIS-1 subject ID from each filename.
3. Auto-discovers the OASIS-1 metadata CSV (or uses an explicit path).
4. Matches each MRI file to its CSV row using the subject ID.
5. Validates the dataset and prints a summary report.
6. Returns data in a format that feeds directly into the existing MRILoader
   interface — no other pipeline file needs to change.

Usage (called from main.py stage_load when dataset_mode == "oasis1")
---------------------------------------------------------------------
    from dataset.oasis1_loader import OASIS1Loader

    loader = OASIS1Loader(
        mri_root   = Path("dataset/oasis_raw"),
        csv_path   = None,          # None => auto-discover first .csv under dataset/
        dataset_dir= Path("dataset"),
    )
    pairs = loader.load()           # list of (Path, dict) — one per matched subject

Output format per subject pair
-------------------------------
    filepath : pathlib.Path   — absolute path to the .nii / .nii.gz file
    metadata : dict           — clinical/demographic fields from the OASIS CSV:
        {
            "age"       : int | None,
            "gender"    : str | None,   # "M" or "F"
            "education" : int | None,   # years of education
            "handedness": str | None,   # "R" or "L"
            "mmse"      : int | None,   # Mini-Mental State Examination
            "cdr"       : float | None, # Clinical Dementia Rating
            "ses"       : int | None,   # Socioeconomic Status
            "etiv"      : float | None, # Estimated Total Intracranial Volume
            "nwbv"      : float | None, # Normalized Whole Brain Volume
            "asf"        : float | None, # Atlas Scaling Factor
            # ... plus any extra CSV columns verbatim
            "metadata_matched": True | False,
        }

Compatibility guarantee
-----------------------
The existing MRILoader.load_single() is called on each filepath exactly as
before.  After loading, the clinical metadata dict is merged into
loader.loaded_scans[pid]["metadata"] — every downstream stage is unaware of
this enrichment.

Extensibility
-------------
Subclass `BaseDatasetLoader` to add OASIS-2 / OASIS-3 / ADNI / IXI support
without touching any pipeline file.
"""

import logging
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OASIS-1 CSV column → normalised key mapping
# The CSV uses cryptic short headers; we map them to readable names.
# Keys are lowercase versions of possible column headers found in the wild.
# ---------------------------------------------------------------------------
_COL_MAP: Dict[str, str] = {
    # Subject identifier
    "id":           "subject_id",
    # Demographics
    "m/f":          "gender",
    "gender":       "gender",
    "hand":         "handedness",
    "handedness":   "handedness",
    "age":          "age",
    "educ":         "education",
    "education":    "education",
    "ses":          "ses",
    # Clinical scores
    "mmse":         "mmse",
    "cdr":          "cdr",
    # Brain morphometry
    "etiv":         "etiv",
    "nwbv":         "nwbv",
    "asf":          "asf",
}

# Regex that matches OASIS-1 subject IDs embedded in filenames
# e.g. "OAS1_0001_MR1.nii.gz"  ->  "OAS1_0001_MR1"
_OASIS1_ID_PATTERN = re.compile(r"(OAS1_\d{4}_MR\d+)", re.IGNORECASE)


# ===========================================================================
# Abstract base — extend here for OASIS-2, ADNI, IXI, etc.
# ===========================================================================

class BaseDatasetLoader(ABC):
    """Abstract dataset loader interface."""

    @abstractmethod
    def discover_mri_files(self) -> List[Path]:
        """Return sorted list of all valid MRI file paths."""

    @abstractmethod
    def load_csv_metadata(self) -> pd.DataFrame:
        """Return a DataFrame of clinical/demographic records."""

    @abstractmethod
    def load(self) -> List[Tuple[Path, Dict]]:
        """
        Return a list of (mri_filepath, clinical_metadata_dict) pairs,
        one per subject that will be processed.
        """


# ===========================================================================
# OASIS-1 concrete implementation
# ===========================================================================

class OASIS1Loader(BaseDatasetLoader):
    """
    Dataset loader for the OASIS-1 cross-sectional MRI dataset.

    Parameters
    ----------
    mri_root : Path
        Root directory that contains the .nii / .nii.gz files
        (searched recursively).
    csv_path : Path | None
        Explicit path to the OASIS-1 metadata CSV.  If None, the loader
        auto-discovers the first .csv file found directly under
        ``dataset_dir``.
    dataset_dir : Path | None
        Parent directory used only for CSV auto-discovery (usually
        ``dataset/``).  Ignored when ``csv_path`` is supplied.
    max_subjects : int | None
        Cap the number of subjects returned (useful for validation runs).
        None means return all matched subjects.
    """

    #: File extensions accepted as NIfTI MRI volumes
    VALID_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii")

    def __init__(
        self,
        mri_root:    Path,
        csv_path:    Optional[Path] = None,
        dataset_dir: Optional[Path] = None,
        max_subjects: Optional[int] = None,
    ) -> None:
        self.mri_root     = Path(mri_root)
        self.csv_path     = Path(csv_path) if csv_path else None
        self.dataset_dir  = Path(dataset_dir) if dataset_dir else self.mri_root.parent
        self.max_subjects = max_subjects

        # Populated during load()
        self._mri_files:    List[Path]       = []
        self._metadata_df:  Optional[pd.DataFrame] = None
        self._matched:      int = 0
        self._unmatched:    int = 0
        self._duplicate_ids: List[str] = []

        logger.info(
            f"[OASIS1Loader] Initialised | mri_root={self.mri_root} | "
            f"csv_path={self.csv_path or '(auto-discover)'} | "
            f"max_subjects={max_subjects}"
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader implementation
    # ------------------------------------------------------------------
    #: File extensions accepted as MRI volumes
    VALID_EXTENSIONS: Tuple[str, ...] = (".nii.gz", ".nii", ".hdr", ".img")

    def discover_mri_files(self) -> List[Path]:
        """
        Recursively scan *mri_root* for .nii, .nii.gz, and Analyze .hdr/.img files.

        Deduplicates per subject ID (e.g. OAS1_0001_MR1) prioritizing structural
        T1-weighted scans over segmentations.
        """
        if not self.mri_root.exists():
            logger.error(
                f"[OASIS1Loader] MRI root directory not found: {self.mri_root}\n"
                "  Create it and place your OASIS-1 MRI files inside."
            )
            return []

        def _get_priority(fpath: Path) -> int:
            s = str(fpath).lower()
            if s.endswith(".nii.gz"):
                return 1
            elif s.endswith(".nii"):
                return 2
            elif "mpr-1_anon.hdr" in s or "mpr_n4_anon_sbj_111.hdr" in s:
                return 3
            elif "raw" in s and s.endswith(".hdr"):
                return 4
            elif s.endswith(".hdr") and "fseg" not in s:
                return 5
            elif s.endswith(".hdr"):
                return 6
            elif s.endswith(".img") and "fseg" not in s:
                return 7
            return 10

        found_by_sub: Dict[str, Tuple[int, Path]] = {}

        for fpath in self.mri_root.rglob("*"):
            if not fpath.is_file():
                continue
            if not any(str(fpath).lower().endswith(ext) for ext in self.VALID_EXTENSIONS):
                continue

            match = _OASIS1_ID_PATTERN.search(str(fpath))
            if match:
                sid = match.group(1).upper()
                prio = _get_priority(fpath)
                if sid not in found_by_sub or prio < found_by_sub[sid][0]:
                    found_by_sub[sid] = (prio, fpath)

        files = [p for _, p in sorted(found_by_sub.values(), key=lambda item: item[1])]
        logger.info(f"[OASIS1Loader] Found {len(files)} subject MRI scan(s) in {self.mri_root}")
        return files

    def load_csv_metadata(self) -> pd.DataFrame:
        """
        Load the OASIS-1 metadata CSV or Excel file.

        If ``csv_path`` was not supplied, auto-discover the first .csv / .xlsx file
        under ``dataset_dir``. Normalises column names using ``_COL_MAP``.
        """
        csv_path = self.csv_path

        if csv_path is None:
            # Auto-discover: look for any .csv or .xlsx directly in dataset_dir
            candidates = sorted(self.dataset_dir.glob("*.csv")) + sorted(self.dataset_dir.glob("*.xlsx")) + sorted(self.dataset_dir.glob("*.xls"))
            if not candidates:
                candidates = sorted(self.dataset_dir.rglob("*.csv")) + sorted(self.dataset_dir.rglob("*.xlsx"))
            if candidates:
                csv_path = candidates[0]
                logger.info(f"[OASIS1Loader] Auto-discovered metadata file: {csv_path}")
            else:
                logger.warning(
                    f"[OASIS1Loader] No metadata file (CSV/XLSX) found under {self.dataset_dir}. "
                    "Clinical metadata will not be available."
                )
                return pd.DataFrame()

        try:
            if str(csv_path).lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(str(csv_path))
            else:
                df = pd.read_csv(str(csv_path))
            logger.info(
                f"[OASIS1Loader] Loaded metadata file: {csv_path.name} "
                f"({len(df)} rows, {len(df.columns)} columns)"
            )
        except Exception as exc:
            logger.error(f"[OASIS1Loader] Failed to read metadata file {csv_path}: {exc}")
            return pd.DataFrame()

        # Normalise column names: strip whitespace, apply map
        df.columns = [c.strip() for c in df.columns]
        rename_map = {
            col: _COL_MAP[col.lower()]
            for col in df.columns
            if col.lower() in _COL_MAP
        }
        df = df.rename(columns=rename_map)

        # Ensure a subject_id column exists (may have been renamed above)
        if "subject_id" not in df.columns:
            # Last resort: look for any column whose values match OASIS-1 IDs
            for col in df.columns:
                sample = df[col].dropna().astype(str)
                if sample.str.match(r"OAS1_\d{4}", case=False).any():
                    df = df.rename(columns={col: "subject_id"})
                    logger.info(
                        f"[OASIS1Loader] Using column '{col}' as subject_id"
                    )
                    break

        if "subject_id" not in df.columns:
            logger.error(
                "[OASIS1Loader] Could not identify subject ID column in CSV. "
                "Metadata matching will be skipped."
            )
            return pd.DataFrame()

        df["subject_id"] = df["subject_id"].astype(str).str.strip()

        # Check for duplicates
        dupes = df[df["subject_id"].duplicated(keep=False)]["subject_id"].unique().tolist()
        if dupes:
            logger.warning(
                f"[OASIS1Loader] {len(dupes)} duplicate subject_id(s) in CSV: {dupes}"
            )
            self._duplicate_ids = dupes
            df = df.drop_duplicates(subset="subject_id", keep="first")

        return df

    def load(self) -> List[Tuple[Path, Dict]]:
        """
        Main entry point called from main.py.

        Steps:
          1. Discover MRI files.
          2. Load & normalise the CSV.
          3. Extract subject IDs from filenames.
          4. Match MRI ↔ CSV rows.
          5. Print validation summary.
          6. Return list of (filepath, clinical_metadata_dict).
        """
        self._mri_files   = self.discover_mri_files()
        self._metadata_df = self.load_csv_metadata()

        # Build CSV lookup:  subject_id (normalised) → row dict
        csv_lookup: Dict[str, Dict] = {}
        if not self._metadata_df.empty and "subject_id" in self._metadata_df.columns:
            for _, row in self._metadata_df.iterrows():
                sid = str(row["subject_id"]).strip().upper()
                csv_lookup[sid] = row.to_dict()

        # Match each MRI file to its metadata record
        pairs: List[Tuple[Path, Dict]] = []
        for fpath in self._mri_files:
            subject_id = self._extract_subject_id(fpath)
            sid_upper  = subject_id.upper()

            clinical: Dict = {"metadata_matched": False}

            if sid_upper in csv_lookup:
                raw_row = csv_lookup[sid_upper]
                clinical = self._build_clinical_dict(raw_row)
                clinical["metadata_matched"] = True
                self._matched += 1
            else:
                logger.warning(
                    f"[OASIS1Loader] No CSV match for subject '{subject_id}' "
                    f"(file: {fpath.name})"
                )
                self._unmatched += 1

            pairs.append((fpath, clinical))

        # Apply subject cap AFTER counting all matches
        if self.max_subjects is not None:
            pairs = pairs[: self.max_subjects]
            logger.info(
                f"[OASIS1Loader] max_subjects={self.max_subjects} — "
                f"capped at {len(pairs)} subject(s)"
            )

        self._print_validation_report(len(self._mri_files))
        return pairs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_subject_id(filepath: Path) -> str:
        """
        Derive OASIS-1 subject ID from a file path.

        Priority:
          1. Regex match for ``OAS1_NNNN_MRN`` pattern in the full path string.
          2. Strip known NIfTI extensions from the bare filename.
        """
        path_str = str(filepath)
        match = _OASIS1_ID_PATTERN.search(path_str)
        if match:
            return match.group(1).upper()

        # Fallback: strip extensions
        name = filepath.name
        for ext in (".nii.gz", ".nii"):
            if name.lower().endswith(ext):
                return name[: -len(ext)]
        return name

    @staticmethod
    def _build_clinical_dict(row: Dict) -> Dict:
        """
        Convert a raw CSV row dict to the standardised clinical metadata dict.

        All fields are cast to their expected Python types.  Missing or
        unreadable values become None and are logged at DEBUG level.
        """
        def _safe(cast, key):
            val = row.get(key)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return None
            try:
                return cast(val)
            except (ValueError, TypeError):
                return None

        clinical = {
            # Demographics
            "age":        _safe(int,   "age"),
            "gender":     _safe(str,   "gender"),
            "education":  _safe(int,   "education"),
            "handedness": _safe(str,   "handedness"),
            "ses":        _safe(int,   "ses"),
            # Clinical scores
            "mmse":       _safe(int,   "mmse"),
            "cdr":        _safe(float, "cdr"),
            # Brain morphometry (OASIS-specific)
            "etiv":       _safe(float, "etiv"),
            "nwbv":       _safe(float, "nwbv"),
            "asf":        _safe(float, "asf"),
        }

        # Also carry the full raw row so no information is lost
        clinical["raw_csv_row"] = {
            k: (None if (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in row.items()
        }

        return clinical

    def _print_validation_report(self, total_mri: int) -> None:
        """Print the dataset validation summary to stdout and the log."""
        sep = "-" * 50
        lines = [
            "",
            sep,
            "  OASIS-1 Dataset Validation",
            sep,
            f"  MRI files detected   : {total_mri}",
            f"  Metadata records     : "
            f"{len(self._metadata_df) if not self._metadata_df.empty else 0}",
            f"  Matched subjects     : {self._matched}",
            f"  Unmatched subjects   : {self._unmatched}",
        ]
        if self._duplicate_ids:
            lines.append(
                f"  Duplicate IDs in CSV : {len(self._duplicate_ids)} "
                f"(kept first occurrence)"
            )
        if self.max_subjects is not None:
            lines.append(
                f"  Processing cap       : {self.max_subjects} subject(s) "
                "(max_subjects limit)"
            )
        lines += [sep, ""]

        report = "\n".join(lines)
        print(report)
        logger.info(report)
