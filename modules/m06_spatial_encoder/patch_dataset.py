"""
ROI patch dataset (Sections 23, 29).
====================================

A ``torch.utils.data.Dataset`` over the cached per-subject artifacts:

* ``(N_ROI, 48, 48, 48)`` ROI patch tensors produced by M7,
* ``(N_ROI, N_FEATURES)`` standardised morphometric features produced by M8,
* the CN/MCI/AD label from the cohort table.

Leakage control
---------------

This class is read-only with respect to statistics. It never computes a mean, a
median or a scaler — those all come pre-fitted from
:class:`~modules.m04_feature_extraction.scaler.MorphometricScaler`, which was fit
on the training split. A dataset that normalised its own contents would leak the
split it was constructed for.

Construction is by **explicit session list**, which is how a split manifest is
consumed. There is no "give me everything and I will split it" path, because that
is where subject-level leakage gets introduced.

Missing patches
---------------

A session whose patch tensor is absent is *excluded* and named in
:attr:`ROIPatchDataset.skipped`, never silently replaced by zeros. A zero patch
would train the CNN on a blank volume labelled AD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_ROI, ROI_ORDER, STAGE_INDEX
from modules.common.seeds import worker_init_fn

logger = get_logger(__name__)


@dataclass
class Sample:
    """One subject's inputs and label."""

    session_id: str
    subject_id: str
    label: int
    #: ``(N_ROI, N_FEATURES)`` standardised morphometric features.
    morph: torch.Tensor
    #: ``(N_ROI, D, H, W)`` ROI patches, or ``None`` when patches are not used.
    patches: Optional[torch.Tensor] = None
    #: ``(N_ROI, embed_dim)`` precomputed CNN embeddings, or ``None``.
    cnn_embedding: Optional[torch.Tensor] = None


def patch_tensor_path(outputs_root: Path, session_id: str) -> Path:
    """Return the canonical path of a session's ROI patch tensor."""
    return (Path(outputs_root) / "roi" / "patches" / session_id
            / f"{session_id}_roi_tensor.npy")


def embedding_path(outputs_root: Path, session_id: str) -> Path:
    """Return the canonical path of a session's cached CNN embeddings."""
    return (Path(outputs_root) / "cnn_embeddings" / session_id
            / f"{session_id}_cnn_embeddings.npy")


class ROIPatchDataset(Dataset):
    """Dataset over an explicit list of sessions.

    Args:
        session_ids: Sessions to include, typically one split of a
            :class:`~modules.m01_dataset.splits.SplitManifest`.
        cohort: Cohort table providing ``session_id``, ``subject_id`` and
            ``label``.
        morph_array: ``(len(session_ids), N_ROI, N_FEATURES)`` standardised
            features, aligned to ``session_ids`` in order.
        outputs_root: Root ``outputs/`` directory, for locating cached tensors.
        load_patches: Load ROI patch volumes. Set ``False`` for the
            morphometry-only variants (A0-A5), which never touch the patches.
        load_embeddings: Load cached CNN embeddings instead of raw patches. Lets
            the graph stage train without re-running the CNN each epoch.
        patch_size: Expected patch shape, validated on load.
        strict: Raise on a missing artifact instead of skipping the session.

    Raises:
        ValueError: If ``morph_array`` does not align with ``session_ids``, or if
            every session ends up skipped.
    """

    def __init__(
        self,
        session_ids: Sequence[str],
        cohort: pd.DataFrame,
        morph_array: np.ndarray,
        outputs_root: Path,
        load_patches: bool = True,
        load_embeddings: bool = False,
        patch_size: Tuple[int, int, int] = (48, 48, 48),
        strict: bool = False,
    ) -> None:
        self.outputs_root = Path(outputs_root)
        self.load_patches = load_patches
        self.load_embeddings = load_embeddings
        self.patch_size = tuple(patch_size)
        self.strict = strict

        requested = [str(s) for s in session_ids]
        morph_array = np.asarray(morph_array, dtype=np.float32)
        if morph_array.shape[0] != len(requested):
            raise ValueError(
                f"morph_array has {morph_array.shape[0]} rows but "
                f"{len(requested)} session_ids were given; the two must be "
                "aligned in order or features would be attached to the wrong "
                "subjects."
            )
        if morph_array.shape[1] != N_ROI:
            raise ValueError(
                f"morph_array ROI axis is {morph_array.shape[1]}, expected "
                f"{N_ROI} ({ROI_ORDER})"
            )

        lookup = cohort.set_index("session_id")
        self.records: List[Dict[str, Any]] = []
        self.skipped: Dict[str, str] = {}

        for i, sid in enumerate(requested):
            if sid not in lookup.index:
                self._skip(sid, "absent from the cohort table")
                continue
            row = lookup.loc[sid]
            if isinstance(row, pd.DataFrame):  # duplicate session_id
                row = row.iloc[0]

            patch_file = patch_tensor_path(self.outputs_root, sid)
            embed_file = embedding_path(self.outputs_root, sid)

            if load_patches and not patch_file.exists():
                self._skip(sid, f"ROI patch tensor not found: {patch_file}")
                continue
            if load_embeddings and not embed_file.exists():
                self._skip(sid, f"CNN embeddings not found: {embed_file}")
                continue

            self.records.append({
                "session_id": sid,
                "subject_id": str(row.get("subject_id", sid)),
                "label": int(row["label"]),
                "morph_index": i,
                "patch_file": patch_file if load_patches else None,
                "embed_file": embed_file if load_embeddings else None,
            })

        self.morph_array = morph_array
        self.n_features = int(morph_array.shape[2])

        if not self.records:
            raise ValueError(
                f"No usable sessions. {len(self.skipped)} of {len(requested)} "
                "were skipped. First reasons: "
                f"{list(self.skipped.items())[:3]}"
            )
        if self.skipped:
            logger.warning(
                "ROIPatchDataset skipped %d of %d session(s); first: %s",
                len(self.skipped), len(requested),
                list(self.skipped.items())[:3],
            )

    def _skip(self, session_id: str, reason: str) -> None:
        """Record a skipped session, or raise in strict mode."""
        if self.strict:
            raise FileNotFoundError(f"{session_id}: {reason}")
        self.skipped[session_id] = reason

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of usable sessions."""
        return len(self.records)

    def __getitem__(self, index: int) -> Sample:
        """Load one sample.

        Raises:
            ValueError: If a loaded patch tensor has an unexpected shape, which
                would misalign the ROI axis.
        """
        rec = self.records[index]
        morph = torch.from_numpy(self.morph_array[rec["morph_index"]])

        patches = None
        if rec["patch_file"] is not None:
            array = np.load(rec["patch_file"])
            expected = (N_ROI,) + self.patch_size
            if tuple(array.shape) != expected:
                raise ValueError(
                    f"{rec['session_id']}: patch tensor has shape "
                    f"{tuple(array.shape)}, expected {expected}"
                )
            patches = torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32)
            )

        embedding = None
        if rec["embed_file"] is not None:
            array = np.load(rec["embed_file"])
            if array.ndim != 2 or array.shape[0] != N_ROI:
                raise ValueError(
                    f"{rec['session_id']}: embeddings have shape "
                    f"{tuple(array.shape)}, expected ({N_ROI}, embed_dim)"
                )
            embedding = torch.from_numpy(
                np.ascontiguousarray(array, dtype=np.float32)
            )

        return Sample(
            session_id=rec["session_id"],
            subject_id=rec["subject_id"],
            label=rec["label"],
            morph=morph,
            patches=patches,
            cnn_embedding=embedding,
        )

    # ── Introspection ─────────────────────────────────────────────────────

    def labels(self) -> np.ndarray:
        """Return the integer labels of the usable sessions, in order."""
        return np.array([r["label"] for r in self.records], dtype=np.int64)

    def session_ids(self) -> List[str]:
        """Return the usable session IDs, in order."""
        return [r["session_id"] for r in self.records]

    def subject_ids(self) -> List[str]:
        """Return the usable subject IDs, in order."""
        return [r["subject_id"] for r in self.records]

    def class_counts(self) -> Dict[str, int]:
        """Return ``{stage: count}`` for the usable sessions."""
        labels = self.labels()
        return {
            stage: int((labels == idx).sum())
            for stage, idx in STAGE_INDEX.items()
        }

    def summary(self) -> Dict[str, Any]:
        """Return a description of the dataset."""
        return {
            "n_sessions": len(self),
            "n_subjects": len(set(self.subject_ids())),
            "n_skipped": len(self.skipped),
            "skipped": dict(list(self.skipped.items())[:20]),
            "class_counts": self.class_counts(),
            "n_features": self.n_features,
            "loads_patches": self.load_patches,
            "loads_embeddings": self.load_embeddings,
            "patch_size": list(self.patch_size),
            "roi_order": list(ROI_ORDER),
        }


def collate_samples(batch: List[Sample]) -> Dict[str, Any]:
    """Collate :class:`Sample` objects into batched tensors.

    A custom collate is needed because ``patches`` and ``cnn_embedding`` may be
    ``None``, which the default collate cannot handle.

    Args:
        batch: List of samples.

    Returns:
        Dict with ``morph``, ``patches``, ``cnn_embedding``, ``labels``,
        ``session_ids`` and ``subject_ids``. Optional tensors are ``None`` when
        absent from the samples.
    """
    return {
        "morph": torch.stack([s.morph for s in batch]),
        "patches": (
            torch.stack([s.patches for s in batch])
            if batch and batch[0].patches is not None else None
        ),
        "cnn_embedding": (
            torch.stack([s.cnn_embedding for s in batch])
            if batch and batch[0].cnn_embedding is not None else None
        ),
        "labels": torch.tensor([s.label for s in batch], dtype=torch.long),
        "session_ids": [s.session_id for s in batch],
        "subject_ids": [s.subject_id for s in batch],
    }


def make_loader(
    dataset: ROIPatchDataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """Build a reproducible :class:`~torch.utils.data.DataLoader`.

    Shuffling is driven by an explicitly seeded generator rather than the global
    RNG, so the batch order is reproducible independently of whatever else has
    consumed randomness in the process.

    Args:
        dataset: The dataset.
        batch_size: Batch size.
        shuffle: Shuffle each epoch. ``True`` for training only.
        num_workers: Worker processes. ``0`` keeps the run single-threaded and
            bit-reproducible.
        seed: Seed for the shuffling generator.
        drop_last: Drop a final partial batch. Worth enabling for training when
            batch normalisation is active, since a batch of one makes batch-norm
            statistics degenerate.

    Returns:
        A configured DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_samples,
        generator=generator if shuffle else None,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        drop_last=drop_last,
    )


__all__ = [
    "Sample",
    "ROIPatchDataset",
    "collate_samples",
    "make_loader",
    "patch_tensor_path",
    "embedding_path",
]
