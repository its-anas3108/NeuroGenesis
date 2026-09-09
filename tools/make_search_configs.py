"""
Generate the candidate configurations for validation-only model selection.
==========================================================================

The search is staged rather than a full grid, because a full grid over
capacity x learning rate x dropout x weight decay is 24 configurations, and at
five folds each that is 120 trainings of the full model on a CPU-only machine.
Staging spends the budget where the leverage is:

``--stage capacity``
    Three widths. This is the dominant lever on this cohort: the model as built
    carries ~2,300 parameters per training subject.

``--stage regularisation --base <winner>``
    Dropout x weight decay around the winning capacity. At n=151 training
    subjects, regularisation strength is the next-largest effect.

``--stage lr --base <winner>``
    Learning rate, last, because its useful range depends on the two above.

Every generated config keeps ``split_scheme = "folds"`` and writes to one
outputs tree, so the selection report sits beside the runs it compares.

Usage::

    python tools/make_search_configs.py --stage capacity
    python tools/make_search_configs.py --stage regularisation --base config_full_capB.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from modules.common.config import NeuroGenesisConfig  # noqa: E402
from modules.m04_feature_extraction.feature_spec import FEATURE_ORDER  # noqa: E402
from modules.model import build_model  # noqa: E402

#: Widths for the capacity stage. capA is the model as built.
CAPACITY: Dict[str, Dict[str, Any]] = {
    "capA": {},
    "capB": {
        "spatial_encoder.embed_dim": 32,
        "graph_learning.hidden_dim": 32,
        "fusion.hidden_dim": 64,
        "fusion.out_dim": 32,
        "stage_tgt.d_model": 32,
        "stage_tgt.ff_dim": 64,
        "stage_tgt.n_layers": 1,
    },
    "capC": {
        "spatial_encoder.embed_dim": 16,
        "spatial_encoder.channels": [8, 16, 32],
        "graph_learning.hidden_dim": 24,
        "fusion.hidden_dim": 48,
        "fusion.out_dim": 24,
        "stage_tgt.d_model": 24,
        "stage_tgt.ff_dim": 48,
        "stage_tgt.n_layers": 1,
    },
}

#: Dropout x weight decay. Heavier regularisation is included because the
#: model is over-parameterised relative to 151 training subjects whatever
#: capacity wins.
REGULARISATION: Dict[str, Dict[str, Any]] = {
    "reg_d20_w1e4": {"graph_learning.dropout": 0.2, "train.weight_decay": 1e-4},
    "reg_d20_w1e2": {"graph_learning.dropout": 0.2, "train.weight_decay": 1e-2},
    "reg_d40_w1e3": {"graph_learning.dropout": 0.4, "train.weight_decay": 1e-3},
    "reg_d50_w1e2": {"graph_learning.dropout": 0.5, "train.weight_decay": 1e-2},
}

#: Learning rate, explored last.
LEARNING_RATE: Dict[str, Dict[str, Any]] = {
    "lr_3e4": {"train.lr": 3e-4},
    "lr_1e3": {"train.lr": 1e-3},
    "lr_3e3": {"train.lr": 3e-3},
}

STAGES = {
    "capacity": CAPACITY,
    "regularisation": REGULARISATION,
    "lr": LEARNING_RATE,
}


def apply(payload: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``payload`` with dotted ``section.field`` overrides applied."""
    out = json.loads(json.dumps(payload))
    for path, value in overrides.items():
        section, field = path.split(".")
        out.setdefault(section, {})[field] = value
    return out


def main() -> int:
    """Write one config per candidate in the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--base", type=Path, default=Path("config_oasis1.json"),
                        help="Config to derive from. For the later stages pass "
                             "the previous stage's winner.")
    parser.add_argument("--outputs", default="outputs_oasis1",
                        help="Shared outputs tree for the search.")
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    print(f"{'config':<34}{'A7 params':>11}{'per subject':>13}")

    for name, overrides in STAGES[args.stage].items():
        payload = apply(base, overrides)
        payload["paths"]["outputs_dir"] = args.outputs
        payload["paths"]["experiment_id"] = f"oasis1_full_{name}"
        payload["data"]["split_scheme"] = "folds"
        payload["data"]["n_folds"] = 5
        path = _ROOT / f"config_full_{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        cfg = NeuroGenesisConfig.from_file(path)
        problems = cfg.validate()
        if problems:
            raise SystemExit(f"{path.name} is invalid: {problems}")
        model = build_model(
            len(FEATURE_ORDER), cfg, "A7", list(FEATURE_ORDER)
        )
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{path.name:<34}{n:>11,}{n / 151:>13,.0f}")

    print(f"\nStage '{args.stage}': {len(STAGES[args.stage])} candidate(s) "
          f"-> {args.outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
