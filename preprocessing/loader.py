"""
NeuroGenesis — MRI Data Loader
================================
Module: preprocessing/loader.py
Author: NeuroGenesis Research Team
Phase : 1 (Preprocessing Pipeline)

Responsibilities:
    - Recursively scan dataset directory for .nii / .nii.gz files
    - Load each MRI using nibabel
    - Extract and display metadata (patient ID, shape, voxel dims, dtype)
    - Visualize axial, sagittal, coronal tri-plane slices
    - Save tri-plane images to outputs/original/
    - Build a metadata DataFrame for downstream use
    - Provide NeuroProp-X integration hook

Usage:
    loader = MRILoader(dataset_dir=Path("dataset/OASIS"),
                       output_dir=Path("outputs/original"))
    files  = loader.scan_dataset()
    scans  = loader.load_all(files)
    loader.visualize_all_triplanes()
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/HPC environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Module-level logger — configured by main.py's logging setup
# ──────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


class MRILoader:
    """
    Handles MRI file discovery, loading, metadata extraction,
    and tri-plane visualization for the NeuroGenesis pipeline.

    Attributes:
        dataset_dir  : Root directory containing MRI files.
        output_dir   : Directory where tri-plane images are saved.
        loaded_scans : Dict mapping patient_id → scan entry dict.
        metadata_df  : Pandas DataFrame of all subject metadata.
    """

    #: File extensions that are considered valid NIfTI/Analyze files
    SUPPORTED_EXTENSIONS: List[str] = [".nii.gz", ".nii", ".hdr", ".img"]

    def __init__(self, dataset_dir: Path, output_dir: Path) -> None:
        """
        Initialise MRILoader.

        Args:
            dataset_dir : Path to the dataset root (e.g. dataset/OASIS).
            output_dir  : Path where visualisation images will be saved.
        """
        self.dataset_dir: Path = Path(dataset_dir)
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_scans: Dict[str, Dict[str, Any]] = {}
        self.metadata_df: Optional[pd.DataFrame] = None

        logger.info(
            f"[MRILoader] Initialised │ dataset={self.dataset_dir} │ "
            f"output={self.output_dir}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — discovery
    # ──────────────────────────────────────────────────────────────────────────

    def scan_dataset(self) -> List[Path]:
        """
        Recursively scan *dataset_dir* for NIfTI files.

        The search prioritises .nii.gz over .nii so that compressed archives
        are not double-counted when both forms exist.

        Returns:
            Sorted list of unique MRI file paths found.

        Raises:
            FileNotFoundError: If *dataset_dir* does not exist on disk.
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}\n"
                "Please place your OASIS .nii/.nii.gz files there."
            )

        discovered: Dict[str, Path] = {}

        # Search in extension priority order so .nii.gz wins over .nii
        for ext in self.SUPPORTED_EXTENSIONS:
            for fpath in sorted(self.dataset_dir.rglob(f"*{ext}")):
                key = str(fpath)
                if key not in discovered:
                    discovered[key] = fpath

        mri_files = sorted(discovered.values())
        logger.info(f"[MRILoader] Found {len(mri_files)} MRI file(s) in {self.dataset_dir}")

        if not mri_files:
            logger.warning(
                "[MRILoader] No MRI files found. "
                "Place .nii or .nii.gz files under dataset/OASIS/ and re-run."
            )

        return mri_files

    # ──────────────────────────────────────────────────────────────────────────
    # Public — loading
    # ──────────────────────────────────────────────────────────────────────────

    def load_single(self, filepath: Path) -> Dict[str, Any]:
        """
        Load a single NIfTI file and extract rich metadata.

        Args:
            filepath : Path to the .nii or .nii.gz file.

        Returns:
            Scan entry dictionary with keys:
                ``image``     — nibabel Nifti1Image object
                ``data``      — float32 numpy array (X, Y, Z[, T])
                ``affine``    — 4×4 affine matrix
                ``header``    — nibabel image header
                ``metadata``  — dict of serialisable metadata

        Raises:
            RuntimeError: If nibabel cannot load the file.
        """
        patient_id = self._extract_patient_id(filepath)
        logger.info(f"[MRILoader] Loading [{patient_id}] ← {filepath.name}")

        try:
            img: nib.Nifti1Image = nib.load(str(filepath))
            data: np.ndarray = img.get_fdata(dtype=np.float32)
            header = img.header
            affine: np.ndarray = img.affine

            # Squeeze out singleton time dimension if present
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[:, :, :, 0]
                logger.debug(f"[MRILoader] Squeezed 4D→3D for {patient_id}")

            zooms = header.get_zooms()
            voxel_dims = tuple(round(float(z), 4) for z in zooms[:3])
            voxel_vol_mm3 = float(np.prod(voxel_dims))

            metadata: Dict[str, Any] = {
                "patient_id":      patient_id,
                "filepath":        str(filepath),
                "shape":           list(data.shape),
                "voxel_dims_mm":   list(voxel_dims),
                "voxel_vol_mm3":   voxel_vol_mm3,
                "dtype":           str(data.dtype),
                "min_intensity":   float(np.nanmin(data)),
                "max_intensity":   float(np.nanmax(data)),
                "mean_intensity":  float(np.nanmean(data)),
                "std_intensity":   float(np.nanstd(data)),
                "total_voxels":    int(data.size),
                "non_zero_voxels": int(np.count_nonzero(data)),
            }

            scan_entry: Dict[str, Any] = {
                "image":    img,
                "data":     data,
                "affine":   affine,
                "header":   header,
                "metadata": metadata,
            }

            self.loaded_scans[patient_id] = scan_entry

            logger.info(
                f"[MRILoader] ✓ {patient_id} │ shape={data.shape} │ "
                f"voxel={voxel_dims} mm │ dtype={data.dtype} │ "
                f"intensity=[{metadata['min_intensity']:.2f}, "
                f"{metadata['max_intensity']:.2f}]"
            )

            return scan_entry

        except Exception as exc:
            logger.error(f"[MRILoader] ✗ Failed to load {filepath}: {exc}")
            raise RuntimeError(f"Could not load MRI file: {filepath}") from exc

    def load_all(self, files: Optional[List[Path]] = None) -> Dict[str, Dict]:
        """
        Load every MRI file and aggregate metadata.

        Args:
            files : Explicit list of paths. If *None*, calls :meth:`scan_dataset`.

        Returns:
            ``loaded_scans`` dictionary (patient_id → scan entry).
        """
        if files is None:
            files = self.scan_dataset()

        failed: List[str] = []

        for filepath in files:
            try:
                self.load_single(filepath)
            except Exception as exc:
                logger.warning(f"[MRILoader] Skipping {filepath.name}: {exc}")
                failed.append(str(filepath))

        if failed:
            logger.warning(f"[MRILoader] {len(failed)} file(s) skipped due to errors.")

        self._build_metadata_df()
        return self.loaded_scans

    # ──────────────────────────────────────────────────────────────────────────
    # Public — metadata
    # ──────────────────────────────────────────────────────────────────────────

    def print_metadata_table(self) -> None:
        """Print a formatted metadata summary table to stdout."""
        if self.metadata_df is None or self.metadata_df.empty:
            logger.warning("[MRILoader] No metadata to display — run load_all() first.")
            return

        display_cols = [
            "patient_id", "shape", "voxel_dims_mm", "dtype",
            "min_intensity", "max_intensity", "mean_intensity",
        ]
        cols = [c for c in display_cols if c in self.metadata_df.columns]

        print("\n" + "═" * 90)
        print("  NeuroGenesis ─── Loaded MRI Dataset Metadata")
        print("═" * 90)
        print(self.metadata_df[cols].to_string(index=False))
        print("═" * 90 + "\n")

    def save_metadata_json(self, output_path: Path) -> None:
        """
        Serialise all subject metadata to a JSON file.

        Args:
            output_path : Destination JSON path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = [
            entry["metadata"]
            for entry in self.loaded_scans.values()
        ]

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)

        logger.info(f"[MRILoader] Metadata JSON saved → {output_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public — visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def visualize_triplane(
        self,
        patient_id: str,
        save: bool = True,
    ) -> Optional[str]:
        """
        Generate a publication-quality tri-plane (axial/sagittal/coronal) figure.

        The figure uses a dark theme consistent with the NeuroGenesis design
        system and is saved at 300 DPI for publication use.

        Args:
            patient_id : Subject to visualise.
            save       : If *True*, write PNG to *output_dir*.

        Returns:
            Absolute path of the saved PNG, or *None* if not saved.
        """
        if patient_id not in self.loaded_scans:
            logger.error(f"[MRILoader] Patient '{patient_id}' not loaded.")
            return None

        data = self.loaded_scans[patient_id]["data"]
        if data.ndim != 3:
            logger.warning(f"[MRILoader] Skipping tri-plane for {patient_id} — not 3D.")
            return None

        cx, cy, cz = [s // 2 for s in data.shape]

        # ── Figure layout ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(18, 6), facecolor="#0d1117")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.04)

        plane_data = [
            (np.rot90(data[cx, :, :]), f"Sagittal  (x = {cx})"),
            (np.rot90(data[:, cy, :]), f"Coronal   (y = {cy})"),
            (np.rot90(data[:, :, cz]), f"Axial     (z = {cz})"),
        ]

        axes = [fig.add_subplot(gs[i]) for i in range(3)]

        for ax, (sl, title) in zip(axes, plane_data):
            im = ax.imshow(sl, cmap="bone", aspect="auto", interpolation="bilinear")
            ax.set_title(title, color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
            ax.axis("off")
            cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
            cbar.ax.tick_params(colors="white", labelsize=7)
            cbar.outline.set_edgecolor("#30363d")

        fig.suptitle(
            f"NeuroGenesis │ Original MRI │ Patient: {patient_id}",
            color="white",
            fontsize=15,
            fontweight="bold",
            y=1.03,
            fontfamily="monospace",
        )

        # Metadata annotation
        meta = self.loaded_scans[patient_id]["metadata"]
        annotation = (
            f"Shape: {meta['shape']}   │   "
            f"Voxel: {meta['voxel_dims_mm']} mm   │   "
            f"Dtype: {meta['dtype']}   │   "
            f"Intensity: [{meta['min_intensity']:.1f}, {meta['max_intensity']:.1f}]"
        )
        fig.text(
            0.5, -0.02, annotation,
            ha="center", color="#8b949e", fontsize=9, fontfamily="monospace"
        )

        plt.tight_layout()

        save_path: Optional[Path] = None
        if save:
            save_path = self.output_dir / f"{patient_id}_triplane_original.png"
            fig.savefig(
                save_path, dpi=300, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none"
            )
            logger.info(f"[MRILoader] Tri-plane saved → {save_path}")

        plt.close(fig)
        return str(save_path) if save_path else None

    def visualize_all_triplanes(self) -> List[str]:
        """
        Visualise tri-plane views for every loaded patient.

        Returns:
            List of saved PNG file paths.
        """
        saved_paths: List[str] = []
        for patient_id in self.loaded_scans:
            path = self.visualize_triplane(patient_id, save=True)
            if path:
                saved_paths.append(path)
        return saved_paths

    def visualize_slice_gallery(
        self,
        patient_id: str,
        axis: int = 2,
        n_slices: int = 12,
        save: bool = True,
    ) -> Optional[str]:
        """
        Generate a multi-slice gallery along a chosen axis.

        Args:
            patient_id : Target subject.
            axis       : 0=sagittal, 1=coronal, 2=axial.
            n_slices   : Number of evenly-spaced slices to show.
            save       : Save figure to disk.

        Returns:
            Path of saved figure, or *None*.
        """
        if patient_id not in self.loaded_scans:
            logger.error(f"[MRILoader] Patient '{patient_id}' not in loaded scans.")
            return None

        data = self.loaded_scans[patient_id]["data"]
        depth = data.shape[axis]
        indices = np.linspace(int(depth * 0.1), int(depth * 0.9), n_slices, dtype=int)

        axis_name = {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis]
        cols = 4
        rows = int(np.ceil(n_slices / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                                 facecolor="#0d1117")
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            sl = np.take(data, idx, axis=axis)
            axes[i].imshow(np.rot90(sl), cmap="bone", aspect="auto")
            axes[i].set_title(f"{axis_name} [{idx}]", color="#58a6ff",
                              fontsize=8, pad=4)
            axes[i].axis("off")

        # Hide unused subplots
        for j in range(len(indices), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"NeuroGenesis │ {axis_name} Slice Gallery │ {patient_id}",
            color="white", fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()

        save_path: Optional[Path] = None
        if save:
            save_path = self.output_dir / f"{patient_id}_{axis_name.lower()}_gallery.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor="#0d1117", edgecolor="none")
            logger.info(f"[MRILoader] Gallery saved → {save_path}")

        plt.close(fig)
        return str(save_path) if save_path else None

    # ──────────────────────────────────────────────────────────────────────────
    # NeuroProp-X integration hook
    # ──────────────────────────────────────────────────────────────────────────

    def get_raw_volume(self, patient_id: str) -> Optional[np.ndarray]:
        """
        Return the raw 3-D float32 volume for a subject.

        This is the primary data hand-off point for downstream modules.

        Args:
            patient_id : Subject identifier.

        Returns:
            float32 numpy array of shape (X, Y, Z), or *None* if not loaded.
        """
        if patient_id not in self.loaded_scans:
            logger.error(f"[MRILoader] '{patient_id}' not found in loaded scans.")
            return None
        return self.loaded_scans[patient_id]["data"]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_patient_id(self, filepath: Path) -> str:
        """
        Derive a clean patient ID string from a file path.

        Strips extensions and matches OASIS-1 patient ID patterns (e.g. OAS1_0001_MR1).

        Args:
            filepath : Path to NIfTI / Analyze file.

        Returns:
            Patient ID string.
        """
        import re
        name = filepath.name
        match = re.search(r"(OAS1_\d{4}_MR\d+)", name, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        for ext in [".nii.gz", ".nii", ".hdr", ".img"]:
            if name.lower().endswith(ext):
                return name[: -len(ext)]
        return name

    def _build_metadata_df(self) -> None:
        """Compile all subject metadata into a Pandas DataFrame."""
        records = [
            {k: v for k, v in entry["metadata"].items()}
            for entry in self.loaded_scans.values()
        ]
        if records:
            self.metadata_df = pd.DataFrame(records)
            logger.info(
                f"[MRILoader] Metadata DataFrame built — "
                f"{len(self.metadata_df)} subject(s)"
            )
