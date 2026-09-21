"""
Validation-only hyperparameter and capacity selection (Sections 10, 15, 18).
============================================================================

Chooses among candidate configurations using **mean validation balanced
accuracy across the cross-validation folds** and nothing else. Test metrics are
not computed here at all -- not recorded and discarded, but never calculated --
so there is no path by which a test score could influence the choice.

That matters more than the usual hygiene argument. Capacity is the single
largest lever on this cohort: the full model carries roughly 2,300 parameters
per training subject, of which 46% sit in the CNN branch and 5.3% in
NeuroProp-X. Choosing that on test would make every downstream number a
selection artefact rather than a measurement.

Usage::

    python tools/select_hyperparameters.py \
        --configs config_full_capA.json config_full_capB.json config_full_capC.json \
        --variant A7 --epochs 60

The winner is written to ``<outputs>/selection/hyperparameter_selection.json``
alongside every candidate's per-fold validation scores, so the decision is
auditable rather than asserted. Run ``--mode ablation`` with the winning config
afterwards to produce the reported test metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.common.logging_utils import get_logger, log_banner, setup_logging  # noqa: E402
from modules.common.seeds import set_all_seeds  # noqa: E402
from modules.common.serialization import dump_json  # noqa: E402

logger = get_logger(__name__)


@dataclass
class CandidateResult:
    """One configuration evaluated across every fold."""

    name: str
    config_path: str
    n_parameters: int
    fold_scores: List[Optional[float]] = field(default_factory=list)
    fold_epochs: List[int] = field(default_factory=list)
    seconds: float = 0.0
    error: Optional[str] = None

    @property
    def valid(self) -> List[float]:
        """Fold scores that actually produced a number."""
        return [s for s in self.fold_scores if s is not None]

    @property
    def mean(self) -> Optional[float]:
        """Mean validation score, or ``None`` if no fold produced one."""
        return float(np.mean(self.valid)) if self.valid else None

    @property
    def sd(self) -> Optional[float]:
        """Standard deviation across folds."""
        return float(np.std(self.valid, ddof=1)) if len(self.valid) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "config_path": self.config_path,
            "n_parameters": self.n_parameters,
            "selection_metric": "validation balanced accuracy (mean over folds)",
            "mean": self.mean,
            "sd": self.sd,
            "fold_scores": self.fold_scores,
            "fold_best_epochs": self.fold_epochs,
            "n_folds_completed": len(self.valid),
            "seconds": round(self.seconds, 1),
            "error": self.error,
        }


def evaluate_candidate(
    config_path: Path,
    variant: str,
    epochs: Optional[int],
    max_folds: Optional[int],
) -> CandidateResult:
    """Train ``variant`` on every fold and return its validation scores.

    Only the validation split is scored. The test split is built by the dataset
    factory (the fold manifest defines it) but never evaluated here.
    """
    import run as driver
    from modules.m04_feature_extraction.feature_spec import FEATURE_ORDER
    from modules.m01_dataset.splits import stratified_subject_folds
    from modules.m06_spatial_encoder.patch_dataset import make_loader
    from modules.model import build_model
    from modules.training.trainer import Trainer

    cfg = NeuroGenesisConfig.from_file(config_path)
    ctx = driver.Context(cfg)
    cohort = ctx.trainable_cohort()

    model_probe = build_model(
        len(FEATURE_ORDER), cfg, variant, list(FEATURE_ORDER)
    )
    result = CandidateResult(
        name=config_path.stem,
        config_path=str(config_path),
        n_parameters=sum(
            p.numel() for p in model_probe.parameters() if p.requires_grad
        ),
    )

    folds = list(stratified_subject_folds(
        cohort,
        n_folds=cfg.data.n_folds,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.repro.seed,
    ))
    if max_folds:
        folds = folds[:max_folds]

    started = time.perf_counter()
    for split in folds:
        set_all_seeds(
            cfg.repro.seed + 1000 * (split.fold or 0),
            deterministic=cfg.repro.deterministic,
            cudnn_benchmark=cfg.repro.cudnn_benchmark,
        )
        try:
            train_ds, val_ds, _ = ctx.build_datasets(split)
            model = build_model(
                len(FEATURE_ORDER), cfg, variant, list(FEATURE_ORDER)
            )
            trainer = Trainer(model, cfg)
            fit = trainer.fit(
                make_loader(
                    train_ds, cfg.train.batch_size, shuffle=True,
                    seed=cfg.repro.seed + (split.fold or 0),
                    drop_last=len(train_ds) > 2 * cfg.train.batch_size,
                ),
                make_loader(val_ds, cfg.train.batch_size),
                epochs=epochs,
                verbose=False,
            )
            result.fold_scores.append(
                None if fit.best_value is None else float(fit.best_value)
            )
            result.fold_epochs.append(int(fit.best_epoch))
            logger.info(
                "  %-22s fold %d/%d: val %s = %s (epoch %d)",
                result.name, (split.fold or 0) + 1, len(folds),
                fit.monitor,
                "n/a" if fit.best_value is None else f"{fit.best_value:.4f}",
                fit.best_epoch,
            )
        except Exception as exc:  # noqa: BLE001 - one fold must not end the search
            logger.error("  %s fold %s FAILED: %s",
                         result.name, split.fold, exc)
            result.fold_scores.append(None)
            result.error = f"{type(exc).__name__}: {exc}"
    result.seconds = time.perf_counter() - started
    return result


def main() -> int:
    """Run the selection and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True,
                        help="Candidate config files.")
    parser.add_argument("--variant", default="A7",
                        help="Variant to select the configuration for.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the epoch budget for the search.")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Evaluate only the first N folds. Use to size the "
                             "search; the final selection should use all.")
    parser.add_argument("--outputs", type=Path, default=None,
                        help="Where to write the report. Defaults to the first "
                             "config's outputs_dir.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    log_banner(logger, f"Validation-only selection for {args.variant}")

    results: List[CandidateResult] = []
    for config_path in args.configs:
        if not config_path.exists():
            logger.error("No such config: %s", config_path)
            continue
        logger.info("Evaluating %s", config_path.name)
        results.append(evaluate_candidate(
            config_path, args.variant, args.epochs, args.max_folds
        ))

    scored = [r for r in results if r.mean is not None]
    if not scored:
        print("\nNo candidate produced a validation score. Nothing selected.")
        return 1
    winner = max(scored, key=lambda r: r.mean)

    print(f"\n{'candidate':<24}{'params':>10}{'val bal-acc':>14}"
          f"{'sd':>9}{'folds':>7}{'minutes':>10}")
    for r in sorted(scored, key=lambda r: -(r.mean or 0)):
        mark = "  <-- selected" if r is winner else ""
        sd = "     n/a" if r.sd is None else f"{r.sd:>8.4f}"
        print(f"{r.name:<24}{r.n_parameters:>10,}{r.mean:>14.4f}{sd}"
              f"{len(r.valid):>7}{r.seconds / 60:>10.1f}{mark}")

    outputs = args.outputs or Path(
        NeuroGenesisConfig.from_file(args.configs[0]).paths.outputs_dir
    )
    path = dump_json(
        {
            "variant": args.variant,
            "selection_metric": "mean validation balanced accuracy across folds",
            "test_data_used_for_selection": False,
            "selected": winner.name,
            "selected_config": winner.config_path,
            "candidates": [r.to_dict() for r in results],
        },
        Path(outputs) / "selection" / "hyperparameter_selection.json",
    )
    print(f"\nSelected : {winner.name}  ({winner.n_parameters:,} parameters)")
    print(f"Report   : {path}")
    print("\nTest metrics were never computed during this search. Run "
          f"`--mode ablation --config {winner.config_path}` to produce them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
