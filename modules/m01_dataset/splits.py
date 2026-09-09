"""
Subject-wise stratified splitting (Sections 11, 23).
====================================================

Data-leakage control is mandatory. Two rules are enforced here structurally,
not by convention:

**1. Splitting is on subjects, never sessions.** OASIS-1 contains 20 ``MR2``
rescans of subjects who also have an ``MR1`` session. Splitting rows would put
two scans of the same brain on both sides of the train/test boundary, and the
resulting test accuracy would measure scan-to-scan reproducibility rather than
generalisation. :func:`make_subject_split` therefore partitions unique
``subject_id`` values and only afterwards expands to sessions.

**2. Split assignment happens before any statistic is fitted.** Feature scalers,
class weights and SRVE/prototype initialisation all consume the training split
returned here. Nothing in this module looks at feature values at all, so it
cannot leak them.

Every split is written to a manifest JSON (Section 23) recording the seed, the
fractions, the resulting counts and the exact subject lists, so a result can be
re-derived months later.

A subject-level stratum is required for stratification. When a subject has
several sessions with different stages, the **most advanced** stage is used, and
the choice is recorded in the manifest. On the default OASIS-1 cohort this
never triggers (all 235 labelled sessions belong to distinct subjects), but the
rule must be defined for it to be safe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.common.logging_utils import get_logger
from modules.common.roi_constants import STAGE_INDEX, STAGE_ORDER

logger = get_logger(__name__)


@dataclass
class SplitManifest:
    """A reproducible record of one train/val/test partition."""

    seed: int
    val_fraction: float
    test_fraction: float
    train_subjects: List[str] = field(default_factory=list)
    val_subjects: List[str] = field(default_factory=list)
    test_subjects: List[str] = field(default_factory=list)
    train_sessions: List[str] = field(default_factory=list)
    val_sessions: List[str] = field(default_factory=list)
    test_sessions: List[str] = field(default_factory=list)
    #: split -> stage -> session count. This is Table 1.
    session_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: split -> stage -> subject count.
    subject_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: Subjects whose sessions disagreed on stage, and the stage assigned.
    stratum_conflicts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: Repeat index when produced by :func:`repeated_subject_splits`, or the
    #: fold index when produced by :func:`stratified_subject_folds`.
    repeat: Optional[int] = None
    #: Fold index, set only under cross-validation.
    fold: Optional[int] = None
    #: Total fold count, set only under cross-validation.
    n_folds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "seed": self.seed,
            "repeat": self.repeat,
            "fold": self.fold,
            "n_folds": self.n_folds,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "stratum_rule": "most advanced stage across a subject's sessions",
            "session_counts": self.session_counts,
            "subject_counts": self.subject_counts,
            "stratum_conflicts": self.stratum_conflicts,
            "warnings": self.warnings,
            "subjects": {
                "train": self.train_subjects,
                "val": self.val_subjects,
                "test": self.test_subjects,
            },
            "sessions": {
                "train": self.train_sessions,
                "val": self.val_sessions,
                "test": self.test_sessions,
            },
        }

    def save(self, path: Path) -> Path:
        """Write the manifest to ``path`` as indented JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        logger.info("Split manifest written: %s", path)
        return path

    def to_frames(
        self, cohort: "pd.DataFrame"
    ) -> Dict[str, "pd.DataFrame"]:
        """Return one row-per-session DataFrame per split (Section 9).

        Each row carries ``subject_id``, ``session_id``, ``MRI_path``,
        ``label`` and ``split``, so the split files are self-contained
        and auditable without re-deriving anything.

        Args:
            cohort: The cohort table the split was built from.

        Returns:
            ``{"train": df, "val": df, "test": df}``.
        """
        assignment = self.assignment()
        indexed = cohort.set_index(
            cohort["session_id"].astype(str), drop=False
        )
        frames: Dict[str, pd.DataFrame] = {}
        for split in ("train", "val", "test"):
            rows = []
            for session in getattr(self, f"{split}_sessions"):
                if session not in indexed.index:
                    continue
                record = indexed.loc[session]
                if isinstance(record, pd.DataFrame):
                    record = record.iloc[0]
                rows.append({
                    "subject_id": record.get("subject_id"),
                    "session_id": session,
                    "MRI_path": record.get("mri_path"),
                    "stage": record.get("stage"),
                    "label": record.get("label"),
                    "split": split,
                    "dataset_source": "OASIS-1",
                })
            frames[split] = pd.DataFrame(rows)
        return frames

    def save_split_csvs(
        self, cohort: "pd.DataFrame", out_dir: Path
    ) -> Dict[str, Path]:
        """Write ``oasis1_{train,val,test}.csv`` (Section 9)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}
        for split, frame in self.to_frames(cohort).items():
            path = out_dir / f"oasis1_{split}.csv"
            frame.to_csv(path, index=False)
            written[split] = path
        logger.info("Split CSVs written to %s", out_dir)
        return written

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        """Reload a manifest previously written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=payload["seed"],
            repeat=payload.get("repeat"),
            fold=payload.get("fold"),
            n_folds=payload.get("n_folds"),
            val_fraction=payload["val_fraction"],
            test_fraction=payload["test_fraction"],
            train_subjects=payload["subjects"]["train"],
            val_subjects=payload["subjects"]["val"],
            test_subjects=payload["subjects"]["test"],
            train_sessions=payload["sessions"]["train"],
            val_sessions=payload["sessions"]["val"],
            test_sessions=payload["sessions"]["test"],
            session_counts=payload.get("session_counts", {}),
            subject_counts=payload.get("subject_counts", {}),
            stratum_conflicts=payload.get("stratum_conflicts", {}),
            warnings=payload.get("warnings", []),
        )

    def assignment(self) -> Dict[str, str]:
        """Return ``session_id -> {"train","val","test"}``."""
        out: Dict[str, str] = {}
        for split, sessions in (
            ("train", self.train_sessions),
            ("val", self.val_sessions),
            ("test", self.test_sessions),
        ):
            for s in sessions:
                out[s] = split
        return out

    def table1(self) -> pd.DataFrame:
        """Return Table 1: dataset and split distribution."""
        rows = []
        for split in ("train", "val", "test"):
            counts = self.session_counts.get(split, {})
            row = {"Split": split.capitalize()}
            row.update({s: counts.get(s, 0) for s in STAGE_ORDER})
            row["Total"] = sum(counts.get(s, 0) for s in STAGE_ORDER)
            rows.append(row)
        total = {"Split": "Total"}
        for s in STAGE_ORDER:
            total[s] = sum(r[s] for r in rows)
        total["Total"] = sum(r["Total"] for r in rows)
        rows.append(total)
        return pd.DataFrame(rows)

    def verify_disjoint(self) -> List[str]:
        """Check that no subject and no session appears in two splits.

        Returns:
            A list of leakage descriptions. Empty means the split is clean.
        """
        problems: List[str] = []
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        subj = {
            "train": set(self.train_subjects),
            "val": set(self.val_subjects),
            "test": set(self.test_subjects),
        }
        sess = {
            "train": set(self.train_sessions),
            "val": set(self.val_sessions),
            "test": set(self.test_sessions),
        }
        for a, b in pairs:
            overlap = subj[a] & subj[b]
            if overlap:
                problems.append(
                    f"LEAKAGE: {len(overlap)} subject(s) in both {a} and {b}: "
                    f"{sorted(overlap)[:5]}"
                )
            overlap_s = sess[a] & sess[b]
            if overlap_s:
                problems.append(
                    f"LEAKAGE: {len(overlap_s)} session(s) in both {a} and {b}: "
                    f"{sorted(overlap_s)[:5]}"
                )
        return problems


def _subject_strata(cohort: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Reduce a session-level cohort to one stratum row per subject.

    Returns:
        ``(subject_df, conflicts)`` where ``subject_df`` has columns
        ``subject_id`` and ``stratum``, and ``conflicts`` maps any subject whose
        sessions disagreed to the stage that was assigned.
    """
    conflicts: Dict[str, str] = {}
    records = []
    for subject_id, group in cohort.groupby("subject_id", sort=True):
        stages = list(group["stage"].unique())
        # Most advanced stage wins: a subject who has progressed to AD in any
        # session should be stratified as AD, not diluted into CN.
        stratum = max(stages, key=lambda s: STAGE_INDEX[s])
        if len(stages) > 1:
            conflicts[str(subject_id)] = stratum
        records.append({"subject_id": subject_id, "stratum": stratum})
    return pd.DataFrame(records), conflicts


def _stratified_partition(
    subjects: List[str],
    strata: List[str],
    val_fraction: float,
    test_fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str], List[str]]:
    """Partition subjects into train/val/test, stratified by stage.

    Allocation is per-stratum so that even a 30-subject AD class is represented
    in all three splits. Sizes are computed with ``round`` and then clamped so
    that no split can swallow an entire stratum: with AD n=30 and a 0.20 test
    fraction, naive flooring can leave zero AD subjects in validation, which
    silently makes balanced accuracy undefined for that class.
    """
    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    by_stratum: Dict[str, List[str]] = {}
    for subj, stratum in zip(subjects, strata):
        by_stratum.setdefault(stratum, []).append(subj)

    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        rng.shuffle(members)
        n = len(members)

        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))

        # Guarantee at least one subject per split whenever the stratum is big
        # enough to allow it.
        if n >= 3:
            n_test = max(1, min(n_test, n - 2))
            n_val = max(1, min(n_val, n - n_test - 1))
        else:
            n_test = min(n_test, max(0, n - 1))
            n_val = min(n_val, max(0, n - n_test - 1))

        test.extend(members[:n_test])
        val.extend(members[n_test:n_test + n_val])
        train.extend(members[n_test + n_val:])

    return sorted(train), sorted(val), sorted(test)


def make_subject_split(
    cohort: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 42,
    repeat: Optional[int] = None,
) -> SplitManifest:
    """Create a leakage-free, stage-stratified, subject-wise split.

    Args:
        cohort: Session-level table with ``subject_id``, ``session_id`` and
            ``stage`` columns (as produced by
            :func:`modules.m01_dataset.labels.map_labels`).
        val_fraction: Fraction of subjects held out for validation.
        test_fraction: Fraction of subjects held out for test.
        seed: RNG seed.
        repeat: Optional repeat index, recorded in the manifest.

    Returns:
        A :class:`SplitManifest`.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If the fractions do not leave a training set, or if the
            resulting split leaks (a defensive check that should never fire).
    """
    for col in ("subject_id", "session_id", "stage"):
        if col not in cohort.columns:
            raise KeyError(
                f"Cohort is missing required column {col!r}. "
                f"Present: {list(cohort.columns)}"
            )
    if not 0.0 < (val_fraction + test_fraction) < 1.0:
        raise ValueError(
            "val_fraction + test_fraction must lie strictly in (0, 1); got "
            f"{val_fraction} + {test_fraction}"
        )

    subject_df, conflicts = _subject_strata(cohort)
    rng = np.random.default_rng(seed)

    train_s, val_s, test_s = _stratified_partition(
        subject_df["subject_id"].tolist(),
        subject_df["stratum"].tolist(),
        val_fraction,
        test_fraction,
        rng,
    )

    return _assemble_manifest(
        cohort, train_s, val_s, test_s,
        seed=seed, repeat=repeat,
        val_fraction=val_fraction, test_fraction=test_fraction,
        conflicts=conflicts,
    )


def _assemble_manifest(
    cohort: pd.DataFrame,
    train_s: List[str],
    val_s: List[str],
    test_s: List[str],
    seed: int,
    repeat: Optional[int],
    val_fraction: float,
    test_fraction: float,
    conflicts: Dict[str, str],
    fold: Optional[int] = None,
    n_folds: Optional[int] = None,
) -> SplitManifest:
    """Expand a subject-level partition into a validated session-level manifest.

    Shared by the random-split and cross-validation schemes so both produce
    identical bookkeeping: session lists, per-class counts, empty-class
    warnings, and the disjointness assertion that must hold before any model
    sees the data.
    """
    manifest = SplitManifest(
        seed=seed,
        repeat=repeat,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        train_subjects=train_s,
        val_subjects=val_s,
        test_subjects=test_s,
        stratum_conflicts=conflicts,
    )
    if fold is not None:
        manifest.fold = fold
        manifest.n_folds = n_folds

    subj_to_split = {s: "train" for s in train_s}
    subj_to_split.update({s: "val" for s in val_s})
    subj_to_split.update({s: "test" for s in test_s})

    sessions = {"train": [], "val": [], "test": []}
    for _, row in cohort.iterrows():
        split = subj_to_split.get(row["subject_id"])
        if split is not None:
            sessions[split].append(str(row["session_id"]))

    manifest.train_sessions = sorted(sessions["train"])
    manifest.val_sessions = sorted(sessions["val"])
    manifest.test_sessions = sorted(sessions["test"])

    for split, subjects in (("train", train_s), ("val", val_s), ("test", test_s)):
        sub = cohort[cohort["subject_id"].isin(subjects)]
        manifest.session_counts[split] = {
            s: int((sub["stage"] == s).sum()) for s in STAGE_ORDER
        }
        manifest.subject_counts[split] = {
            s: int(sub.loc[sub["stage"] == s, "subject_id"].nunique())
            for s in STAGE_ORDER
        }

    for split in ("train", "val", "test"):
        empty = [s for s, c in manifest.session_counts[split].items() if c == 0]
        if empty:
            manifest.warnings.append(
                f"{split} split contains no {empty} session(s); per-class "
                f"metrics for {empty} are undefined on this split."
            )
    if conflicts:
        manifest.warnings.append(
            f"{len(conflicts)} subject(s) had sessions with differing stages; "
            "each was stratified by its most advanced stage."
        )

    leaks = manifest.verify_disjoint()
    if leaks:
        raise ValueError("Split construction produced leakage:\n" + "\n".join(leaks))

    label = f"fold {fold + 1}/{n_folds}" if fold is not None \
        else f"seed={seed}"
    logger.info(
        "Subject-wise split (%s): train=%d/%d val=%d/%d test=%d/%d "
        "(subjects/sessions)",
        label, len(train_s), len(manifest.train_sessions),
        len(val_s), len(manifest.val_sessions),
        len(test_s), len(manifest.test_sessions),
    )
    for w in manifest.warnings:
        logger.warning("%s", w)
    return manifest


def stratified_subject_folds(
    cohort: pd.DataFrame,
    n_folds: int = 5,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Iterator[SplitManifest]:
    """Yield ``n_folds`` subject-wise, stage-stratified cross-validation folds.

    Why folds rather than repeated random draws
    -------------------------------------------

    ``repeated_subject_splits`` samples an independent 20% test set each repeat.
    Every subject is therefore tested a random number of times -- some several
    times, some never -- so the spread across repeats mixes real model variance
    with the accident of who happened to be held out. Under k-fold, **every
    subject is tested exactly once per pass**, so the k test sets tile the
    cohort and the mean is an estimate over all 235 subjects rather than over a
    resample of them.

    That matters most for the smallest class. With AD n=30 and k=5, each fold
    holds 6 AD subjects and all 30 are evaluated exactly once, instead of the
    1-per-split the previous scheme gave at a 0.20 test fraction on a
    30-subject pilot.

    Construction
    ------------

    Subjects -- never sessions -- are assigned to folds round-robin *within
    each stage stratum*, so class balance is near-identical across folds and no
    subject can appear in two folds. Validation is then carved out of the
    remaining training pool only, so the test fold is untouched by model
    selection.

    Args:
        cohort: Session-level table with ``subject_id``, ``session_id`` and
            ``stage``.
        n_folds: Number of folds. 5 gives a 20% test fold.
        val_fraction: Validation size as a fraction of the **whole** cohort;
            it is drawn from the training pool, so the within-pool fraction is
            scaled up accordingly.
        seed: RNG seed for fold assignment.

    Yields:
        One :class:`SplitManifest` per fold, in fold order.

    Raises:
        ValueError: If ``n_folds`` is less than 2 or exceeds the smallest
            stratum, which would leave a fold with no subject of that class.
    """
    for col in ("subject_id", "session_id", "stage"):
        if col not in cohort.columns:
            raise KeyError(
                f"Cohort is missing required column {col!r}. "
                f"Present: {list(cohort.columns)}"
            )
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2; got {n_folds}")

    subject_df, conflicts = _subject_strata(cohort)
    stratum_of = dict(zip(subject_df["subject_id"].astype(str),
                          subject_df["stratum"]))

    by_stratum: Dict[str, List[str]] = {}
    for subject_id, stratum in stratum_of.items():
        by_stratum.setdefault(str(stratum), []).append(str(subject_id))

    smallest = min((len(v) for v in by_stratum.values()), default=0)
    if smallest < n_folds:
        raise ValueError(
            f"n_folds={n_folds} exceeds the smallest stage stratum "
            f"({smallest} subject(s)); some fold would contain no subject of "
            "that class and its per-class metrics would be undefined."
        )

    rng = np.random.default_rng(seed)
    fold_of: Dict[str, int] = {}
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        rng.shuffle(members)
        # Round-robin within the stratum keeps every fold's class mix within
        # one subject of every other fold's.
        for position, subject_id in enumerate(members):
            fold_of[subject_id] = position % n_folds

    # Validation comes out of the training pool, which is (1 - 1/k) of the
    # cohort, so scale the requested whole-cohort fraction up to a within-pool
    # fraction.
    pool_share = 1.0 - (1.0 / n_folds)
    pool_val_fraction = min(0.5, float(val_fraction) / pool_share)

    for fold in range(n_folds):
        test_s = sorted([s for s, f in fold_of.items() if f == fold])
        pool = sorted([s for s, f in fold_of.items() if f != fold])
        train_s, val_s, _ = _stratified_partition(
            pool,
            [str(stratum_of[s]) for s in pool],
            val_fraction=pool_val_fraction,
            test_fraction=0.0,
            rng=np.random.default_rng(seed + 1000 + fold),
        )
        yield _assemble_manifest(
            cohort, train_s, val_s, test_s,
            seed=seed, repeat=fold,
            val_fraction=val_fraction,
            test_fraction=1.0 / n_folds,
            conflicts=conflicts,
            fold=fold, n_folds=n_folds,
        )


def repeated_subject_splits(
    cohort: pd.DataFrame,
    n_repeats: int = 10,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    base_seed: int = 42,
) -> Iterator[SplitManifest]:
    """Yield ``n_repeats`` independent subject-wise splits.

    Required by Sections 17 and 26 (Table 9): with AD n=30, a single split's
    point estimate is not reportable, so ablation and model comparison run over
    repeated splits and report mean +/- SD with confidence intervals.

    Args:
        cohort: Session-level cohort table.
        n_repeats: Number of repeats.
        val_fraction: Validation fraction per repeat.
        test_fraction: Test fraction per repeat.
        base_seed: Repeat ``r`` uses seed ``base_seed + r``, so the whole
            sequence is reproducible from one number.

    Yields:
        One :class:`SplitManifest` per repeat.
    """
    for r in range(n_repeats):
        yield make_subject_split(
            cohort,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=base_seed + r,
            repeat=r,
        )


__all__ = [
    "SplitManifest",
    "make_subject_split",
    "stratified_subject_folds",
    "repeated_subject_splits",
]
