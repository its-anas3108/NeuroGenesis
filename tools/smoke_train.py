"""
Smoke training run over synthetic artifacts — validates the code path.
=====================================================================

Trains a chosen model variant on the artifacts produced by
``tools/make_smoke_artifacts.py`` and prints the training curve, test metrics and
per-subject predictions.

.. danger::

   The artifacts are synthetic. Metrics printed here validate that the pipeline
   is correctly wired; they say nothing about Alzheimer's disease and must never
   be reported as results. The script refuses to run against an output tree that
   is *not* marked as a smoke test, so it cannot be mistaken for the real
   evaluation path (use ``run.py --mode train_full`` for that).

Usage::

    python tools/smoke_train.py --variant A7 --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig
from modules.common.logging_utils import setup_logging
from modules.common.seeds import set_all_seeds
from modules.m01_dataset import make_subject_split
from modules.m04_feature_extraction import FEATURE_ORDER, MorphometricScaler
from modules.m06_spatial_encoder.patch_dataset import ROIPatchDataset, make_loader
from modules.model import build_model
from modules.training import Trainer
from tools.make_smoke_artifacts import is_smoke_output


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Smoke training run (synthetic data).")
    ap.add_argument("--out", type=Path, default=Path("outputs_smoke"))
    ap.add_argument("--variant", default="A7")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args()

    setup_logging()
    marker = is_smoke_output(args.out)
    if marker is None:
        print(
            f"REFUSING TO RUN: {args.out} is not marked as a smoke-test output "
            "tree. This script only runs against synthetic artifacts; use "
            "run.py --mode train_full for real data.",
            file=sys.stderr,
        )
        return 2

    set_all_seeds(args.seed)
    cfg = NeuroGenesisConfig()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.early_stopping_patience = max(args.epochs // 2, 4)

    cohort = pd.read_csv(args.out / "patient" / "cohort.csv")
    features = pd.read_csv(args.out / "features" / "morphometric_features.csv")

    split = make_subject_split(
        cohort, val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction, seed=args.split_seed,
    )
    leaks = split.verify_disjoint()
    print("leakage check:", leaks or "CLEAN")
    print(split.table1().to_string(index=False))

    scaler = MorphometricScaler().fit(
        features, train_session_ids=split.train_sessions
    )
    features = scaler.transform(scaler.add_atrophy_index(features))

    spec_uses_cnn = build_model(
        len(FEATURE_ORDER), cfg, args.variant, FEATURE_ORDER
    ).spec.use_cnn

    def make_ds(sessions):
        array, _ = scaler.to_tensor_array(features, sessions)
        return ROIPatchDataset(
            sessions, cohort, array, args.out, load_patches=spec_uses_cnn
        )

    train_ds, val_ds, test_ds = (
        make_ds(split.train_sessions), make_ds(split.val_sessions),
        make_ds(split.test_sessions),
    )
    print("class counts | train", train_ds.class_counts(),
          "val", val_ds.class_counts(), "test", test_ds.class_counts())

    train_loader = make_loader(train_ds, cfg.train.batch_size, shuffle=True,
                               seed=args.seed, drop_last=len(train_ds) > 16)
    val_loader = make_loader(val_ds, cfg.train.batch_size)
    test_loader = make_loader(test_ds, cfg.train.batch_size)

    model = build_model(len(FEATURE_ORDER), cfg, args.variant, FEATURE_ORDER)
    print(f"variant {args.variant}: {model.n_parameters():,} parameters, "
          f"CNN branch={spec_uses_cnn}")

    trainer = Trainer(model, cfg)
    result = trainer.fit(
        train_loader, val_loader,
        checkpoint_path=args.out / "checkpoints" / f"{args.variant}.pt",
    )
    print(f"\nbest epoch {result.best_epoch} | {result.monitor} = "
          f"{result.best_value if result.best_value is None else round(result.best_value, 4)}")
    print("class weights", [round(w, 3) for w in result.class_weights or []],
          "from", result.train_class_counts)
    for w in result.warnings:
        print("WARNING:", w)
    if result.stage_geometry is not None:
        geo = result.stage_geometry
        print("prototype pairwise:",
              {k: round(v, 3) for k, v in geo["pairwise_distances"].items()})
        print("CN < MCI < AD ordering respected:", geo["ordering_respected"],
              "| geometry tie-breaks used:", result.n_geometry_tie_breaks)

    curves = result.curves()
    print("train loss:", round(curves["train_loss"][0], 4), "->",
          round(curves["train_loss"][-1], 4))
    print("train acc :", round(curves["train_accuracy"][0], 4), "->",
          round(curves["train_accuracy"][-1], 4))
    print("loss terms (last epoch):",
          {k: round(v, 4) for k, v in result.history[-1].loss_terms.items()})

    evaluation = trainer.evaluate(test_loader)
    print("\n=== TEST METRICS ON SYNTHETIC DATA - NOT A RESEARCH RESULT ===")
    print(evaluation["metrics"].summary())
    print("\nconfusion matrix (rows = true CN/MCI/AD):")
    print(evaluation["metrics"].confusion_matrix)

    preds = pd.DataFrame(evaluation["predictions"])
    columns = [c for c in (
        "session_id", "true_stage", "predicted_stage", "p_CN", "p_MCI", "p_AD",
        "ad_associated_propensity", "stage_transition_propensity",
    ) if c in preds.columns]
    print("\n" + preds[columns].head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
