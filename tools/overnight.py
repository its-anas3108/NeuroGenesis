"""
Unattended end-to-end run: preprocessing -> selection -> ablation -> report.
===========================================================================

Runs the whole remaining experiment with nobody watching, against a wall-clock
deadline, and records what it did. Three properties matter more than speed here:

**It never fabricates.** Every step's outcome is recorded as it happened. A step
that fails is written down as failed, with its stderr, and the run continues to
the steps that do not depend on it. A step that is skipped for lack of time is
written down as skipped. The summary therefore distinguishes "measured X" from
"did not measure X", which is the distinction the final report depends on.

**It sizes the search against measured cost, not a guess.** One fold of the full
model is timed first. The capacity search is then scoped to what actually fits
before the deadline -- all five folds if it fits, fewer if not, and skipped
entirely (falling back to the middle capacity) if even that does not. On a
CPU-only machine that difference is hours.

**Cheap and irreplaceable work goes first.** The stage-wise statistics need only
the feature table, take about a minute, and do not depend on any model, so they
run immediately after preprocessing. If the night goes wrong after that, the
most defensible result is already on disk.

Usage::

    python tools/overnight.py --pid <preprocess_pid> --deadline 07:15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

LOG = _ROOT / "overnight.log"


def say(message: str) -> None:
    """Log to stdout and to the run log with a timestamp."""
    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@dataclass
class Step:
    """One executed stage of the overnight run."""

    name: str
    status: str = "pending"       # ok | failed | skipped
    seconds: float = 0.0
    detail: str = ""
    stdout_tail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable form."""
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 1),
            "detail": self.detail,
            "stdout_tail": self.stdout_tail[-4000:],
        }


@dataclass
class Journal:
    """Everything the run did, written after every step."""

    started: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    deadline: str = ""
    steps: List[Step] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    def save(self) -> Path:
        """Persist the journal so a crash cannot lose the record."""
        from modules.common.serialization import dump_json

        return dump_json(
            {
                "started": self.started,
                "deadline": self.deadline,
                "updated": datetime.now().isoformat(timespec="seconds"),
                "facts": self.facts,
                "steps": [s.to_dict() for s in self.steps],
            },
            _ROOT / "outputs_oasis1" / "overnight_journal.json",
        )


def run(journal: Journal, name: str, command: List[str],
        timeout: Optional[float] = None) -> Step:
    """Execute one subprocess step, recording its outcome either way."""
    step = Step(name=name)
    journal.steps.append(step)
    say(f"START  {name}")
    say(f"       $ {' '.join(command)}")
    slug = "".join(c if c.isalnum() else "_" for c in name)[:48]
    log_dir = _ROOT / "outputs_oasis1" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    step_log = log_dir / f"overnight_{slug}.log"
    started = time.perf_counter()
    try:
        # A file, not a pipe: preprocessing spawns its own process pool and
        # captured pipes kill it on Windows with no diagnostic at all.
        with step_log.open("w", encoding="utf-8", errors="replace") as sink:
            proc = subprocess.run(
                command, cwd=str(_ROOT), stdout=sink,
                stderr=subprocess.STDOUT, timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        step.seconds = time.perf_counter() - started
        try:
            step.stdout_tail = step_log.read_text(
                encoding="utf-8", errors="replace"
            )[-4000:]
        except OSError:
            step.stdout_tail = ""
        if proc.returncode == 0:
            step.status = "ok"
            say(f"OK     {name}  ({step.seconds / 60:.1f} min)")
        else:
            step.status = "failed"
            step.detail = f"exit {proc.returncode} — see {step_log.name}"
            say(f"FAILED {name}  exit {proc.returncode}")
            for line in step.stdout_tail.strip().splitlines()[-8:]:
                say(f"       {line}")
    except subprocess.TimeoutExpired:
        step.seconds = time.perf_counter() - started
        step.status = "failed"
        step.detail = f"timed out after {step.seconds / 60:.0f} min"
        say(f"FAILED {name}  timed out")
    except Exception as exc:  # noqa: BLE001 - the run must continue
        step.seconds = time.perf_counter() - started
        step.status = "failed"
        step.detail = f"{type(exc).__name__}: {exc}"
        say(f"FAILED {name}  {step.detail}")
    journal.save()
    return step


def skip(journal: Journal, name: str, why: str) -> Step:
    """Record a step that was deliberately not run."""
    step = Step(name=name, status="skipped", detail=why)
    journal.steps.append(step)
    say(f"SKIP   {name}  -- {why}")
    journal.save()
    return step


#: Below this many sessions in the feature table, the run refuses to continue.
#: A partial or stale table must never be mistaken for the cohort -- an earlier
#: single-subject test left a 1-session table on disk and it was picked up as if
#: it were the full 235.
MIN_SESSIONS = 200


FEATURES = (_ROOT / "outputs_oasis1" / "features"
            / "morphometric_features.csv")


def feature_sessions() -> int:
    """Sessions currently covered by the feature table, 0 if absent."""
    if not FEATURES.exists():
        return 0
    try:
        import pandas as pd

        frame = pd.read_csv(FEATURES)
        column = ("subject_id" if "subject_id" in frame.columns
                  else frame.columns[0])
        return int(frame[column].nunique())
    except Exception:  # noqa: BLE001 - a half-written file just reads as 0
        return 0


def await_preprocessing(journal: "Journal", timeout_hours: float) -> "Step":
    """Block until the feature table covers the cohort.

    The feature table is the completion signal: ``mode_preprocess`` writes it
    only after every subject has been accounted for. Waiting on the artifact
    rather than on a process is what makes this robust -- identifying the
    pool's parent among its own workers by command line is guesswork, and
    getting it wrong killed a worker and broke the pool once already.
    """
    step = Step(name="await preprocessing (feature table >= "
                     f"{MIN_SESSIONS} sessions)")
    journal.steps.append(step)
    say(f"START  {step.name}")
    started = time.perf_counter()
    last = -1
    while time.perf_counter() - started < timeout_hours * 3600:
        n = feature_sessions()
        if n != last:
            say(f"       feature table: {n} session(s)")
            last = n
        if n >= MIN_SESSIONS:
            step.status = "ok"
            step.seconds = time.perf_counter() - started
            step.detail = f"{n} sessions"
            say(f"OK     {step.name}  ({step.seconds / 60:.1f} min)")
            journal.save()
            return step
        time.sleep(120)
    step.status = "failed"
    step.seconds = time.perf_counter() - started
    step.detail = (f"timed out after {timeout_hours} h with "
                   f"{feature_sessions()} session(s)")
    say(f"FAILED {step.name}  {step.detail}")
    journal.save()
    return step


def feature_facts() -> Dict[str, Any]:
    """Read the size of the produced feature table."""
    import pandas as pd

    path = _ROOT / "outputs_oasis1" / "features" / "morphometric_features.csv"
    if not path.exists():
        return {"features_present": False}
    frame = pd.read_csv(path)
    column = "subject_id" if "subject_id" in frame.columns else frame.columns[0]
    return {
        "features_present": True,
        "feature_rows": int(len(frame)),
        "feature_sessions": int(frame[column].nunique()),
    }


def manifest_facts() -> Dict[str, Any]:
    """Read the Section 24 preprocessing accounting."""
    path = (_ROOT / "outputs_oasis1" / "preprocessing"
            / "preprocessing_manifest.json")
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n_requested": payload.get("n_requested"),
        "n_succeeded": payload.get("n_succeeded"),
        "n_failed": payload.get("n_failed"),
        "class_counts_processed": payload.get("class_counts_processed"),
        "failures": (payload.get("failures") or [])[:20],
    }


def probe_seconds(journal: Journal) -> Optional[float]:
    """Time one fold of the full model, to size the search that follows."""
    step = run(journal, "timing probe: A7 capB, 1 fold", [
        sys.executable, "tools/select_hyperparameters.py",
        "--configs", "config_full_capB.json",
        "--variant", "A7", "--max-folds", "1",
    ], timeout=3 * 3600)
    if step.status != "ok":
        return None
    report = (_ROOT / "outputs_oasis1" / "selection"
              / "hyperparameter_selection.json")
    if not report.exists():
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    for candidate in payload.get("candidates", []):
        if candidate.get("seconds"):
            return float(candidate["seconds"])
    return None


def main() -> int:
    """Run everything, then write the journal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-hours", type=float, default=6.0,
                        dest="wait_hours",
                        help="How long to wait for preprocessing to finish.")
    parser.add_argument("--deadline", default="07:15",
                        help="Wall-clock HH:MM by which to stop starting new "
                             "long steps.")
    args = parser.parse_args()

    hour, minute = (int(v) for v in args.deadline.split(":"))
    deadline = datetime.now().replace(hour=hour, minute=minute, second=0,
                                      microsecond=0)
    if deadline <= datetime.now():
        deadline += timedelta(days=1)

    journal = Journal(deadline=deadline.isoformat(timespec="seconds"))
    say("=" * 68)
    say(f"overnight run — deadline {deadline.strftime('%Y-%m-%d %H:%M')}")
    say("=" * 68)

    def remaining() -> float:
        """Seconds left before the deadline."""
        return (deadline - datetime.now()).total_seconds()

    # ── 1. preprocessing ─────────────────────────────────────────────────
    await_preprocessing(journal, args.wait_hours)
    journal.facts.update(manifest_facts())
    journal.facts.update(feature_facts())
    journal.save()
    say(f"facts: {json.dumps(journal.facts, default=str)[:400]}")

    if not journal.facts.get("features_present"):
        say("no feature table was produced — nothing downstream can run.")
        journal.save()
        return 1

    sessions = int(journal.facts.get("feature_sessions") or 0)
    if sessions < MIN_SESSIONS:
        say(f"the feature table covers only {sessions} session(s), below the "
            f"{MIN_SESSIONS} required. Refusing to train on a partial cohort "
            "and report it as the full one.")
        journal.facts["stopped_because"] = (
            f"feature table covered {sessions} sessions, minimum "
            f"{MIN_SESSIONS}"
        )
        journal.save()
        return 1
    say(f"feature table covers {sessions} session(s) — proceeding")

    # ── 2. statistics: cheap, model-free, most defensible result ─────────
    run(journal, "stage-wise statistics", [
        sys.executable, "run.py", "--mode", "statistics",
        "--config", "config_full_capB.json",
    ], timeout=3600)

    # ── 3. time one fold, then size the capacity search ──────────────────
    per_fold = probe_seconds(journal)
    journal.facts["a7_fold_seconds"] = per_fold
    journal.save()

    selected = "config_full_capB.json"
    if per_fold is None:
        skip(journal, "capacity selection", "the timing probe did not complete")
    else:
        say(f"one A7 fold costs {per_fold / 60:.1f} min; "
            f"{remaining() / 3600:.1f} h remain before the deadline")
        # Reserve time for the ablation, training, xai, figures and report.
        reserve = max(2.0 * 3600, 12 * per_fold)
        budget = remaining() - reserve
        folds = 0
        for candidate_folds in (5, 3, 2, 1):
            if 3 * candidate_folds * per_fold <= budget:
                folds = candidate_folds
                break
        if folds == 0:
            skip(journal, "capacity selection",
                 f"3 candidates x 1 fold needs "
                 f"{3 * per_fold / 3600:.1f} h but only "
                 f"{budget / 3600:.1f} h is free after reserving the ablation; "
                 f"proceeding with capB, the middle capacity")
        else:
            step = run(journal, f"capacity selection ({folds} fold(s))", [
                sys.executable, "tools/select_hyperparameters.py",
                "--configs", "config_full_capA.json",
                "config_full_capB.json", "config_full_capC.json",
                "--variant", "A7", "--max-folds", str(folds),
            ], timeout=max(600.0, budget))
            if step.status == "ok":
                report = (_ROOT / "outputs_oasis1" / "selection"
                          / "hyperparameter_selection.json")
                if report.exists():
                    payload = json.loads(report.read_text(encoding="utf-8"))
                    chosen = payload.get("selected_config")
                    if chosen and Path(chosen).exists():
                        selected = chosen
                        journal.facts["selection"] = {
                            "selected": payload.get("selected"),
                            "candidates": [
                                {"name": c["name"], "mean": c["mean"],
                                 "n_parameters": c["n_parameters"]}
                                for c in payload.get("candidates", [])
                            ],
                        }
    # ── 3b. regularisation, if the budget stretches to it ────────────────
    # Second-largest lever after capacity: the winning model is still
    # over-parameterised relative to 151 training subjects, so how hard it is
    # regularised matters more than the learning rate.
    if per_fold is not None and "selection" in journal.facts:
        reserve = max(2.0 * 3600, 12 * per_fold)
        budget = remaining() - reserve
        folds = 0
        for candidate_folds in (5, 3, 2):
            if 4 * candidate_folds * per_fold <= budget:
                folds = candidate_folds
                break
        if folds == 0:
            skip(journal, "regularisation selection",
                 f"4 candidates need at least "
                 f"{8 * per_fold / 3600:.1f} h but only "
                 f"{budget / 3600:.1f} h is free after reserving the ablation")
        else:
            gen = run(journal, "generate regularisation candidates", [
                sys.executable, "tools/make_search_configs.py",
                "--stage", "regularisation", "--base", selected,
            ], timeout=600)
            if gen.status == "ok":
                step = run(journal,
                           f"regularisation selection ({folds} fold(s))", [
                               sys.executable,
                               "tools/select_hyperparameters.py",
                               "--configs",
                               "config_full_reg_d20_w1e4.json",
                               "config_full_reg_d20_w1e2.json",
                               "config_full_reg_d40_w1e3.json",
                               "config_full_reg_d50_w1e2.json",
                               "--variant", "A7",
                               "--max-folds", str(folds),
                           ], timeout=max(600.0, budget))
                if step.status == "ok":
                    report = (_ROOT / "outputs_oasis1" / "selection"
                              / "hyperparameter_selection.json")
                    if report.exists():
                        payload = json.loads(
                            report.read_text(encoding="utf-8")
                        )
                        chosen = payload.get("selected_config")
                        if chosen and Path(chosen).exists():
                            selected = chosen
                            journal.facts["selection_regularisation"] = {
                                "selected": payload.get("selected"),
                                "candidates": [
                                    {"name": c["name"], "mean": c["mean"]}
                                    for c in payload.get("candidates", [])
                                ],
                            }

    journal.facts["config_used"] = selected
    journal.save()
    say(f"configuration for the reported run: {selected}")

    # ── 4. the ablation that produces the headline numbers ───────────────
    if remaining() < 1800:
        skip(journal, "ablation", "less than 30 minutes before the deadline")
    else:
        run(journal, "ablation A0-A5,A7 + A7_no_struct + baselines", [
            sys.executable, "run.py", "--mode", "ablation",
            "--config", selected,
            "--variants", "A0,A1,A2,A3,A4,A5,A7,A7_no_struct",
        ], timeout=max(1800.0, remaining() - 1500))

    # ── 5. a checkpoint for XAI and the report ───────────────────────────
    if remaining() < 1200:
        skip(journal, "train_full", "less than 20 minutes before the deadline")
    else:
        run(journal, "train_full (checkpoint for XAI and report)", [
            sys.executable, "run.py", "--mode", "train_full",
            "--config", selected,
        ], timeout=max(1200.0, remaining() - 900))

    # ── 6. explanations, figures, report ─────────────────────────────────
    for name, argv, need in (
        ("xai", ["--mode", "xai", "--config", selected], 900),
        ("figures", ["--mode", "figures", "--config", selected], 300),
        ("report", ["--mode", "report", "--config", selected], 300),
    ):
        if remaining() < need:
            skip(journal, name, "not enough time before the deadline")
            continue
        run(journal, name, [sys.executable, "run.py", *argv],
            timeout=max(float(need), remaining()))

    # ── 7. summary ───────────────────────────────────────────────────────
    path = journal.save()
    say("=" * 68)
    for step in journal.steps:
        say(f"  {step.status.upper():<8}{step.name}  "
            f"({step.seconds / 60:.1f} min)")
    say(f"journal: {path}")
    say("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
