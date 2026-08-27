"""
Training loop (Sections 11, 24, 29).
====================================

A single trainer used identically by the full model, every baseline and every
ablation rung, so that a difference between them is attributable to the model
rather than to how it was trained.

Guarantees this class enforces
------------------------------

* **Class weights come from the training split only.** They are computed inside
  :meth:`Trainer.fit` from ``train_loader``'s labels, never from the full
  dataset. This is the most easily overlooked leak in an imbalanced study: a
  weight vector derived from all labels encodes the test set's composition.
* **Model selection uses validation only.** The best checkpoint is the epoch
  with the best validation ``monitor`` metric. The test split is touched exactly
  once, by :meth:`Trainer.evaluate`, after training has finished.
* **Balanced accuracy is the default selection metric.** With CN 88 / MCI 46 /
  AD 20 training sessions, plain accuracy would select a CN-biased model.
* **Every epoch is recorded.** :attr:`Trainer.history` holds per-epoch train and
  validation losses and metrics, which is what the training-curve figures plot.
  Nothing in the figures is synthesised.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules.common.config import NeuroGenesisConfig
from modules.common.logging_utils import get_logger
from modules.common.roi_constants import N_STAGE, STAGE_ORDER
from modules.common.seeds import resolve_device
from modules.m01_dataset.labels import class_weights as compute_class_weights
from modules.model import ModelOutput, NeuroGenesisModel
from modules.training.losses import LossBreakdown, NeuroGenesisLoss
from modules.training.metrics import ClassificationMetrics, compute_metrics
from modules.common.serialization import json_safe

logger = get_logger(__name__)


@dataclass
class EpochRecord:
    """One epoch's losses and metrics."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    val_balanced_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None
    #: Validation ``L_order``; lower means a better-trained stage geometry.
    val_order_loss: Optional[float] = None
    loss_terms: Dict[str, float] = field(default_factory=dict)
    lr: Optional[float] = None
    seconds: Optional[float] = None
    is_best: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_balanced_accuracy": self.val_balanced_accuracy,
            "val_macro_f1": self.val_macro_f1,
            "val_order_loss": self.val_order_loss,
            "loss_terms": dict(self.loss_terms),
            "lr": self.lr,
            "seconds": self.seconds,
            "is_best": self.is_best,
        }


@dataclass
class TrainingResult:
    """Outcome of a completed training run."""

    history: List[EpochRecord]
    best_epoch: int
    best_value: Optional[float]
    monitor: str
    checkpoint_path: Optional[Path] = None
    class_weights: Optional[List[float]] = None
    train_class_counts: Dict[str, int] = field(default_factory=dict)
    stopped_early: bool = False
    total_seconds: float = 0.0
    #: Prototype geometry measured on the selected checkpoint, or ``None`` when
    #: the variant has no Stage-TGT. Surfaced so that a weakly separated
    #: propensity score can be recognised as under-trained geometry rather than
    #: mistaken for a finding.
    stage_geometry: Optional[Dict[str, Any]] = None
    #: Number of epochs whose selection was decided by the geometry tie-break.
    n_geometry_tie_breaks: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "monitor": self.monitor,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
            "checkpoint_path": (
                self.checkpoint_path.as_posix() if self.checkpoint_path else None
            ),
            "class_weights": self.class_weights,
            "train_class_counts": dict(self.train_class_counts),
            "stopped_early": self.stopped_early,
            "total_seconds": self.total_seconds,
            "stage_geometry": self.stage_geometry,
            "n_geometry_tie_breaks": self.n_geometry_tie_breaks,
            "n_epochs_run": len(self.history),
            "history": [e.to_dict() for e in self.history],
            "warnings": list(self.warnings),
        }

    def curves(self) -> Dict[str, List[Optional[float]]]:
        """Return training-curve series for the figures."""
        return {
            "epoch": [e.epoch for e in self.history],
            "train_loss": [e.train_loss for e in self.history],
            "val_loss": [e.val_loss for e in self.history],
            "train_accuracy": [e.train_accuracy for e in self.history],
            "val_accuracy": [e.val_accuracy for e in self.history],
            "val_balanced_accuracy": [
                e.val_balanced_accuracy for e in self.history
            ],
        }


class Trainer:
    """Train and evaluate a :class:`~modules.model.NeuroGenesisModel`.

    Args:
        model: The model to train.
        cfg: Framework configuration.
        device: Torch device string, or ``"auto"``.
    """

    def __init__(
        self,
        model: NeuroGenesisModel,
        cfg: Optional[NeuroGenesisConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or NeuroGenesisConfig()
        self.device = resolve_device(device or self.cfg.train.device)
        self.model = model.to(self.device)
        self.criterion: Optional[NeuroGenesisLoss] = None
        self.history: List[EpochRecord] = []

    # ── Batch plumbing ────────────────────────────────────────────────────

    def _forward(self, batch: Dict[str, Any],
                 return_trace: bool = False) -> ModelOutput:
        """Move a batch to the device and run the model.

        Precomputed CNN embeddings take precedence over raw patches when both
        are present: they are cheaper and, being produced in eval mode, are
        deterministic.
        """
        morph = batch["morph"].to(self.device)
        patches = (
            batch["patches"].to(self.device)
            if batch.get("patches") is not None else None
        )
        embeddings = (
            batch["cnn_embedding"].to(self.device)
            if batch.get("cnn_embedding") is not None else None
        )
        return self.model(
            morph_features=morph,
            patches=patches if embeddings is None else None,
            cnn_embeddings=embeddings,
            return_trace=return_trace,
        )

    # ── Fit ───────────────────────────────────────────────────────────────

    @staticmethod
    def _git_commit() -> Optional[str]:
        """Return the current git commit, or ``None`` outside a repo.

        Recorded in every checkpoint so a trained model can be traced back
        to the exact code that produced it (Section 15).
        """
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
                cwd=Path(__file__).resolve().parent.parent.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:  # noqa: BLE001 - provenance is best-effort
            pass
        return None

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_path: Optional[Path] = None,
        epochs: Optional[int] = None,
        verbose: bool = True,
        dataset_provenance: Optional[Dict[str, Any]] = None,
        split_file: Optional[Path] = None,
    ) -> TrainingResult:
        """Train the model, selecting the best epoch on the validation split.

        Args:
            train_loader: Training loader.
            val_loader: Validation loader. Without it no model selection is
                possible and the final epoch is kept, which is recorded as a
                warning in the result.
            checkpoint_path: Where to save the best checkpoint.
            epochs: Override ``cfg.train.epochs``.
            verbose: Log per-epoch progress.

        Returns:
            A :class:`TrainingResult`.

        Raises:
            ValueError: If the training loader is empty.
        """
        tc = self.cfg.train
        n_epochs = int(epochs if epochs is not None else tc.epochs)
        warnings: List[str] = []

        train_labels = self._collect_labels(train_loader)
        if train_labels.size == 0:
            raise ValueError("The training loader yielded no samples.")

        counts = {
            stage: int((train_labels == i).sum())
            for i, stage in enumerate(STAGE_ORDER)
        }
        weights = compute_class_weights(train_labels, N_STAGE)
        missing = [s for s, c in counts.items() if c == 0]
        if missing:
            warnings.append(
                f"Training split contains no {missing} session(s); those "
                "classes cannot be learned and receive zero weight."
            )

        self.criterion = NeuroGenesisLoss(
            cfg=self.cfg.loss,
            class_weights=weights,
            d_model=self.model.fusion.out_dim,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_epochs, 1)
        )

        if val_loader is None:
            warnings.append(
                "No validation loader was supplied, so no model selection was "
                "performed and the final epoch's weights were kept. Reported "
                "test metrics from such a run are optimistic."
            )

        monitor = tc.monitor
        best_value: Optional[float] = None
        best_epoch = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None
        # Validation L_order at the currently selected epoch; used only to break
        # ties on the monitored metric, never to override it.
        best_order: Optional[float] = None
        n_tie_breaks = 0
        patience = 0
        stopped_early = False
        self.history = []
        t_start = time.perf_counter()

        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc, terms = self._train_epoch(
                train_loader, optimizer, tc.grad_clip
            )

            record = EpochRecord(
                epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                loss_terms=terms, lr=float(optimizer.param_groups[0]["lr"]),
            )

            if val_loader is not None:
                val_loss, val_metrics, val_terms = self._evaluate_loader(
                    val_loader
                )
                record.val_loss = val_loss
                record.val_accuracy = val_metrics.accuracy
                record.val_balanced_accuracy = val_metrics.balanced_accuracy
                record.val_macro_f1 = val_metrics.macro_f1
                record.val_order_loss = val_terms.get("order")

                current = val_metrics.get(monitor)
                if current is not None and np.isfinite(current):
                    improved = best_value is None or current > best_value
                    tie_broken = False
                    if (not improved and tc.tie_break_on_geometry
                            and best_value is not None
                            and abs(current - best_value) <= tc.tie_break_tolerance
                            and record.val_order_loss is not None
                            and best_order is not None
                            and record.val_order_loss < best_order):
                        # Equal classification quality, better-trained stage
                        # geometry: prefer this epoch. Nothing is traded away.
                        improved = True
                        tie_broken = True

                    if improved:
                        best_value, best_epoch = current, epoch
                        best_order = record.val_order_loss
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                        record.is_best = True
                        n_tie_breaks += int(tie_broken)
                        # A tie-break is not evidence of classification
                        # progress, so it must not reset early-stopping
                        # patience, or a plateau could train indefinitely.
                        if not tie_broken:
                            patience = 0
                        else:
                            patience += 1
                    else:
                        patience += 1
                else:
                    patience += 1

            record.seconds = time.perf_counter() - t0
            self.history.append(record)

            if verbose and (epoch == 1 or epoch % 10 == 0 or record.is_best
                            or epoch == n_epochs):
                logger.info(
                    "epoch %3d/%d | train loss %.4f acc %.3f | val loss %s "
                    "bal-acc %s macro-F1 %s%s",
                    epoch, n_epochs, train_loss, train_acc,
                    "n/a" if record.val_loss is None else f"{record.val_loss:.4f}",
                    "n/a" if record.val_balanced_accuracy is None
                    else f"{record.val_balanced_accuracy:.3f}",
                    "n/a" if record.val_macro_f1 is None
                    else f"{record.val_macro_f1:.3f}",
                    "  <- best" if record.is_best else "",
                )

            scheduler.step()

            if (val_loader is not None and tc.early_stopping_patience > 0
                    and patience >= tc.early_stopping_patience):
                stopped_early = True
                logger.info(
                    "Early stopping at epoch %d (%d epochs without improving "
                    "%s).", epoch, patience, monitor,
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best epoch %d (%s = %.4f)",
                        best_epoch, monitor, best_value)
        elif val_loader is not None:
            warnings.append(
                f"The monitored metric {monitor!r} was undefined in every "
                "epoch, so no best checkpoint could be selected and the final "
                "epoch's weights were kept."
            )

        geometry: Optional[Dict[str, Any]] = None
        if getattr(self.model, "stage_tgt", None) is not None:
            geometry = self.model.stage_tgt.prototypes.prototype_geometry()
            if not geometry.get("ordering_respected", False):
                warnings.append(
                    "The learned stage prototypes do not respect the "
                    "CN < MCI < AD ordering at the selected checkpoint. "
                    "Stage-transition propensity and AD-associated propensity "
                    "are therefore weakly separated and should be read as "
                    "under-trained geometry, not as a finding. Increase "
                    "loss.lambda_order, train for more epochs, or raise "
                    "train.early_stopping_patience."
                )

        saved: Optional[Path] = None
        if checkpoint_path is not None:
            # Section 15: a checkpoint must record enough to identify the
            # exact data, split, code and settings that produced it, so a
            # model trained on the wrong data can never be mistaken for one
            # trained on the real dataset.
            n_train = len(getattr(train_loader, "dataset", []) or [])
            n_val = len(getattr(val_loader, "dataset", []) or []) \
                if val_loader is not None else 0
            saved = self.model.save_checkpoint(
                Path(checkpoint_path),
                extra={
                    "dataset": (dataset_provenance or {}).get(
                        "dataset_source", "OASIS-1"
                    ),
                    "dataset_provenance": dataset_provenance,
                    "split_file": (
                        Path(split_file).as_posix() if split_file else None
                    ),
                    "n_train_sessions": n_train,
                    "n_val_sessions": n_val,
                    "subject_count": n_train + n_val,
                    "seed": self.cfg.repro.seed,
                    "training_date": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "git_commit": self._git_commit(),
                    "hyperparameters": {
                        "train": asdict(self.cfg.train),
                        "loss": asdict(self.cfg.loss),
                        "graph_learning": asdict(self.cfg.graph_learning),
                        "neuropropx": asdict(self.cfg.neuropropx),
                        "spatial_encoder": asdict(self.cfg.spatial_encoder),
                        "stage_tgt": asdict(self.cfg.stage_tgt),
                        "fusion": asdict(self.cfg.fusion),
                    },
                    "monitor": monitor,
                    "best_epoch": best_epoch,
                    "best_value": best_value,
                    "validation_metrics": {
                        "monitor": monitor,
                        "best_value": best_value,
                        "accuracy": (
                            self.history[best_epoch - 1].val_accuracy
                            if 0 < best_epoch <= len(self.history) else None
                        ),
                        "balanced_accuracy": (
                            self.history[best_epoch - 1].val_balanced_accuracy
                            if 0 < best_epoch <= len(self.history) else None
                        ),
                        "macro_f1": (
                            self.history[best_epoch - 1].val_macro_f1
                            if 0 < best_epoch <= len(self.history) else None
                        ),
                    },
                    "test_metrics": None,  # filled by evaluate()
                    "class_weights": weights.tolist(),
                    "train_class_counts": counts,
                },
            )

        return TrainingResult(
            history=self.history,
            best_epoch=best_epoch,
            best_value=best_value,
            monitor=monitor,
            checkpoint_path=saved,
            class_weights=weights.tolist(),
            train_class_counts=counts,
            stopped_early=stopped_early,
            total_seconds=time.perf_counter() - t_start,
            stage_geometry=geometry,
            n_geometry_tie_breaks=n_tie_breaks,
            warnings=warnings,
        )

    def _train_epoch(self, loader: DataLoader,
                     optimizer: torch.optim.Optimizer,
                     grad_clip: float) -> Tuple[float, float, Dict[str, float]]:
        """Run one training epoch. Returns (loss, accuracy, mean loss terms)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0
        term_sums: Dict[str, float] = {}

        for batch in loader:
            targets = batch["labels"].to(self.device)
            optimizer.zero_grad(set_to_none=True)
            output = self._forward(batch)
            breakdown: LossBreakdown = self.criterion(output, targets)
            breakdown.total.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(breakdown.total.detach())
            n_batches += 1
            preds = output.classification.predictions.detach()
            correct += int((preds == targets).sum())
            total += int(targets.numel())
            for k, v in breakdown.to_dict().items():
                term_sums[k] = term_sums.get(k, 0.0) + v

        divisor = max(n_batches, 1)
        return (
            total_loss / divisor,
            correct / max(total, 1),
            {k: v / divisor for k, v in term_sums.items()},
        )

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_loader(
        self, loader: DataLoader
    ) -> Tuple[Optional[float], ClassificationMetrics, Dict[str, float]]:
        """Evaluate a loader in eval mode.

        Returns:
            ``(mean loss, metrics, mean per-term losses)``. The per-term means
            are what the geometry tie-break reads.
        """
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        probs: List[np.ndarray] = []
        total_loss = 0.0
        n_batches = 0
        term_sums: Dict[str, float] = {}

        for batch in loader:
            targets = batch["labels"].to(self.device)
            output = self._forward(batch)
            if self.criterion is not None:
                breakdown = self.criterion(output, targets)
                total_loss += float(breakdown.total)
                n_batches += 1
                for k, v in breakdown.to_dict().items():
                    term_sums[k] = term_sums.get(k, 0.0) + v
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(
                output.classification.predictions.cpu().numpy().tolist()
            )
            probs.append(output.classification.probabilities.cpu().numpy())

        divisor = max(n_batches, 1)
        loss = total_loss / n_batches if n_batches else None
        prob_array = np.concatenate(probs, axis=0) if probs else None
        terms = {k: v / divisor for k, v in term_sums.items()}
        return loss, compute_metrics(y_true, y_pred, prob_array), terms

    @torch.no_grad()
    def evaluate(self, loader: DataLoader,
                 collect_predictions: bool = True) -> Dict[str, Any]:
        """Evaluate on a held-out split.

        Args:
            loader: The split to evaluate. For the test split this must be
                called exactly once, after training.
            collect_predictions: Also return per-subject predictions, which the
                report and XAI stages consume.

        Returns:
            ``{"loss", "metrics", "predictions"}``.
        """
        self.model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        probs: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            targets = batch["labels"].to(self.device)
            output = self._forward(batch)
            if self.criterion is not None:
                total_loss += float(self.criterion(output, targets).total)
                n_batches += 1

            p = output.classification.probabilities.cpu().numpy()
            preds = output.classification.predictions.cpu().numpy()
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            probs.append(p)

            if collect_predictions:
                for i, sid in enumerate(batch["session_ids"]):
                    row: Dict[str, Any] = {
                        "session_id": sid,
                        "subject_id": batch["subject_ids"][i],
                        "true_stage": STAGE_ORDER[int(batch["labels"][i])],
                        "predicted_stage": STAGE_ORDER[int(preds[i])],
                        "correct": bool(int(preds[i]) == int(batch["labels"][i])),
                    }
                    for j, stage in enumerate(STAGE_ORDER):
                        row[f"p_{stage}"] = float(p[i, j])
                    conf = output.classification.confidence(i)
                    row["margin"] = conf["margin"]
                    row["normalized_entropy"] = conf["normalized_entropy"]
                    if output.propensity is not None:
                        prop = output.propensity.report(i)
                        row["ad_associated_propensity"] = \
                            prop["ad_associated_propensity"]
                        row["stage_transition_propensity"] = \
                            prop["stage_transition_propensity"]
                        row["advanced_stage_alignment"] = \
                            prop["advanced_stage_alignment"]
                    rows.append(row)

        prob_array = np.concatenate(probs, axis=0) if probs else None
        metrics = compute_metrics(y_true, y_pred, prob_array)
        return {
            "loss": total_loss / n_batches if n_batches else None,
            "metrics": metrics,
            "predictions": rows if collect_predictions else None,
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _collect_labels(loader: DataLoader) -> np.ndarray:
        """Read every label from a loader without running the model.

        Prefers the dataset's own label accessor so no forward pass or data
        loading is needed; falls back to iterating the loader.
        """
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "labels"):
            return np.asarray(dataset.labels(), dtype=np.int64)
        labels: List[int] = []
        for batch in loader:
            labels.extend(batch["labels"].numpy().tolist())
        return np.asarray(labels, dtype=np.int64)

    @staticmethod
    def save_result(result: TrainingResult, path: Path) -> Path:
        """Write a training result to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(result.to_dict()), indent=2), encoding="utf-8")
        logger.info("Training result written: %s", path)
        return path


__all__ = ["EpochRecord", "TrainingResult", "Trainer"]
