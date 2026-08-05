"""
NeuroGenesis — Module 1: Dataset Manager
========================================
File   : backend/modules/01_dataset/dataset_manager.py
Purpose: Generic DatasetManager supporting Patient, Session, MRI, Metadata.
         Supports OASIS-1 currently, designed with interfaces for OASIS-3 extension.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Metadata:
    subject_id: str
    age: Optional[float] = None
    gender: Optional[str] = None
    handedness: Optional[str] = None
    education: Optional[float] = None
    ses: Optional[float] = None
    mmse: Optional[float] = None
    cdr: Optional[float] = None
    etiv: Optional[float] = None
    nwbv: Optional[float] = None
    asf: Optional[float] = None
    raw_dict: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MRIScan:
    scan_id: str
    file_path: Path
    modality: str = "T1w"
    session_id: Optional[str] = None
    shape: Optional[tuple] = None
    metadata: Optional[Metadata] = None

@dataclass
class Session:
    session_id: str
    patient_id: str
    scans: List[MRIScan] = field(default_factory=list)
    age_at_session: Optional[float] = None
    cdr_at_session: Optional[float] = None

@dataclass
class Patient:
    patient_id: str
    sessions: List[Session] = field(default_factory=list)
    gender: Optional[str] = None
    handedness: Optional[str] = None

class BaseDatasetManager(ABC):
    """Abstract Base Class for NeuroGenesis Dataset Managers."""

    @abstractmethod
    def discover_scans(self) -> List[MRIScan]:
        """Discover all MRI scans in dataset."""
        pass

    @abstractmethod
    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get structured Patient object."""
        pass

    @abstractmethod
    def list_patient_ids(self) -> List[str]:
        """Return list of patient IDs."""
        pass

class OASIS1DatasetManager(BaseDatasetManager):
    """
    OASIS-1 Dataset Manager Implementation.
    Handles cross-sectional single-session OASIS-1 dataset.
    """

    def __init__(self, mri_root: Path, csv_path: Optional[Path] = None):
        self.mri_root = Path(mri_root)
        self.csv_path = Path(csv_path) if csv_path else None
        self._df_metadata: Optional[pd.DataFrame] = None
        self._patients: Dict[str, Patient] = {}
        self._scans: List[MRIScan] = []
        self._initialize()

    def _initialize(self):
        """Discover metadata CSV and populate patients/scans."""
        if self.csv_path and self.csv_path.exists():
            self._load_csv(self.csv_path)
        else:
            # Auto-discover CSV
            possible_csvs = list(self.mri_root.parent.glob("*.csv")) + list(self.mri_root.glob("*.csv"))
            if possible_csvs:
                self._load_csv(possible_csvs[0])
            else:
                logger.warning(f"No OASIS-1 CSV metadata file found near {self.mri_root}")

        self.discover_scans()

    def _load_csv(self, path: Path):
        try:
            self._df_metadata = pd.read_csv(path)
            # Normalize column names
            self._df_metadata.columns = [c.strip().lower() for c in self._df_metadata.columns]
            logger.info(f"Loaded OASIS-1 metadata with {len(self._df_metadata)} records from {path}")
        except Exception as e:
            logger.error(f"Failed to load CSV at {path}: {e}")

    def discover_scans(self) -> List[MRIScan]:
        self._scans = []
        self._patients = {}

        if not self.mri_root.exists():
            logger.warning(f"MRI root path {self.mri_root} does not exist.")
            return self._scans

        # Look for NIfTI files recursively
        nii_files = sorted(list(self.mri_root.rglob("*.nii")) + list(self.mri_root.rglob("*.nii.gz")))
        
        for file in nii_files:
            sid = file.stem.replace(".nii", "")
            meta = self._get_metadata_for_subject(sid)
            scan = MRIScan(
                scan_id=sid,
                file_path=file,
                modality="T1w",
                session_id=sid,
                metadata=meta
            )
            self._scans.append(scan)

            patient_id = sid.split("_")[1] if "_" in sid else sid
            if patient_id not in self._patients:
                self._patients[patient_id] = Patient(
                    patient_id=patient_id,
                    gender=meta.gender if meta else None,
                    handedness=meta.handedness if meta else None
                )
            
            session = Session(
                session_id=sid,
                patient_id=patient_id,
                scans=[scan],
                age_at_session=meta.age if meta else None,
                cdr_at_session=meta.cdr if meta else None
            )
            self._patients[patient_id].sessions.append(session)

        logger.info(f"OASIS1DatasetManager discovered {len(self._scans)} scans across {len(self._patients)} subjects.")
        return self._scans

    def _get_metadata_for_subject(self, subject_id: str) -> Metadata:
        meta = Metadata(subject_id=subject_id)
        if self._df_metadata is not None:
            id_col = "id" if "id" in self._df_metadata.columns else "subject_id"
            if id_col in self._df_metadata.columns:
                match = self._df_metadata[self._df_metadata[id_col].astype(str).str.upper() == subject_id.upper()]
                if not match.empty:
                    row = match.iloc[0].to_dict()
                    meta.age = float(row.get("age")) if pd.notnull(row.get("age")) else None
                    meta.gender = str(row.get("m/f", row.get("gender"))) if pd.notnull(row.get("m/f", row.get("gender"))) else None
                    meta.handedness = str(row.get("hand", row.get("handedness"))) if pd.notnull(row.get("hand", row.get("handedness"))) else None
                    meta.education = float(row.get("educ")) if pd.notnull(row.get("educ")) else None
                    meta.ses = float(row.get("ses")) if pd.notnull(row.get("ses")) else None
                    meta.mmse = float(row.get("mmse")) if pd.notnull(row.get("mmse")) else None
                    meta.cdr = float(row.get("cdr")) if pd.notnull(row.get("cdr")) else None
                    meta.etiv = float(row.get("etiv")) if pd.notnull(row.get("etiv")) else None
                    meta.nwbv = float(row.get("nwbv")) if pd.notnull(row.get("nwbv")) else None
                    meta.asf = float(row.get("asf")) if pd.notnull(row.get("asf")) else None
                    meta.raw_dict = row
        return meta

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        return self._patients.get(patient_id)

    def list_patient_ids(self) -> List[str]:
        return list(self._patients.keys())

class OASIS3DatasetManager(BaseDatasetManager):
    """
    OASIS-3 Dataset Manager Interface.
    Placeholder architecture for longitudinal OASIS-3 integration.
    """

    def __init__(self, mri_root: Path, csv_path: Optional[Path] = None):
        self.mri_root = Path(mri_root)
        self.csv_path = Path(csv_path) if csv_path else None
        logger.info("OASIS3DatasetManager initialized (Longitudinal framework ready).")

    def discover_scans(self) -> List[MRIScan]:
        logger.info("OASIS-3 longitudinal scan discovery placeholder called.")
        return []

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        return None

    def list_patient_ids(self) -> List[str]:
        return []
