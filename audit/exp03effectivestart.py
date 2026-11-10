"""EXP-03: one positional effective start per stage (HYP-04, Q2).

The decided reading (Q2): a series' effective start is df.index[lookback - 1] of its
date-filtered, preprocessed frame, and every stage uses it: the realistic suppression
(trigger.md:158), the extraction filter, the reference trimming (trigger.md:279) and the curve
index, which starts at the earliest start over the processed series. Per series: the
positional start beside the calendar date ini_date + BDay(lookback - 1) the predecessor's
stages used; every candidate the realistic loop evaluates, with its suppression reason and the
start the loop passes (a recorded run on the raw input); ideal triggers of a run copy beginning
before their series' start; the saved reference rows; the first date of each mode's curve.

Structure:
1. main
2. Measurements
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
from pathlib import Path

import pandas as pd

import pipelinekit
from src.utils.common.constants import TRIGGER_BEGIN_DATES
from src.utils.common.schema import AppConfig
from src.utils.trigger.realistic import RealisticModeMixin

EXPERIMENT = "03"
SUPPRESSED = "before_effective_start"
MODES = ("ideal", "realistic")
REFERENCE_PREFIX, TRIGGERS_PREFIX = "/reference_series_", "/triggers_"
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)


def main() -> None:
    """The positional start per series, then each stage's start against it (Q2)."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    lookback = config.trigger.replay.lookback_window
    calendar = pd.Timestamp(config.read.ini_date) + pd.offsets.BDay(lookback - 1)
    series = pipelinekit.input_series(config, folder / "input")
    realistic = pipelinekit.concrete(config, "realistic")
    found = {"calendar": day(calendar), "series": {}}
    for sid, frame in sorted(series.items()):
        positional = frame.index[lookback - 1]
        evaluated = evaluated_candidates(sid, frame, realistic)
        found["series"][sid] = {"positional": day(positional), "gap_rows": int(((frame.index >= calendar) & (frame.index < positional)).sum()),
                                "loop_starts": sorted({day(start) for _, _, start in evaluated}),
                                "misjudged": [[day(date), reason] for date, reason, _ in evaluated
                                              if (reason == SUPPRESSED) != (date < positional)],
                                "suppressed_before_start": sum(reason == SUPPRESSED for _, reason, _ in evaluated)}
    positional = {sid: pd.Timestamp(item["positional"]) for sid, item in found["series"].items()}
    earliest = day(min(positional.values()))
    found["ideal_triggers_before_start"] = early_triggers(args.run_copy / PARAMS / "ideal" / "trigger" / "0.h5", positional)
    found["reference_rows"] = reference_rows(args.run_copy / PARAMS / "realistic" / "trigger" / "0.h5")
    found["curve_first"] = {mode: curve_first(args.run_copy / PARAMS / mode / "performance" / "0.h5") for mode in MODES}
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    starts = {"loop": {sid: item["loop_starts"] for sid, item in found["series"].items()}, "curves": found["curve_first"]}
    CTX.verdict("one effective start (Q2): the realistic loop starts at each series' positional start, both curves at the earliest",
                starts == {"loop": {sid: [item["positional"]] for sid, item in found["series"].items()},
                           "curves": {mode: earliest for mode in MODES}},
                {"loop": "each series' positional start", "curves": earliest}, starts)
    CTX.verdict("realistic: a candidate is suppressed before_effective_start exactly when it precedes its positional start (Q2)",
                not any(item["misjudged"] for item in found["series"].values()), "none misjudged",
                {sid: item["misjudged"] for sid, item in found["series"].items() if item["misjudged"]})
    CTX.verdict("ideal: no trigger kept that begins before its series' positional start (Q2)",
                not found["ideal_triggers_before_start"], "none", found["ideal_triggers_before_start"])
    reference_starts = {key: first for key, (first, _) in found["reference_rows"].items()}
    expected_starts = {key: found["series"][key[len(REFERENCE_PREFIX):]]["positional"] for key in reference_starts}
    CTX.verdict("reference rows start at their series' positional start (Q2, trigger.md:279)",
                bool(reference_starts) and reference_starts == expected_starts, expected_starts, reference_starts)
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# MEASUREMENTS
# =============================================================================


def evaluated_candidates(series_id: str, frame: pd.DataFrame, config: AppConfig) -> list:
    """(candidate date, reason or None, the start the loop passes) of every candidate the realistic loop evaluates
    (recorded run)."""
    original = RealisticModeMixin._check_suppression_conditions
    decisions = []

    def recording(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = original(self, i, candidate_date, trigger_type, state, effective_start_date)
        decisions.append((pd.Timestamp(candidate_date), reason, pd.Timestamp(effective_start_date)))
        return reason

    RealisticModeMixin._check_suppression_conditions = recording
    try:
        pipelinekit.analyze(series_id, frame, config)
    finally:
        RealisticModeMixin._check_suppression_conditions = original
    return decisions


def early_triggers(path: Path, positional: dict) -> list:
    """Persisted ideal triggers beginning before their series' positional start."""
    found = []
    with pd.HDFStore(path, mode="r") as store:
        for key in sorted(store.keys()):
            if key.startswith(TRIGGERS_PREFIX):
                sid = key[len(TRIGGERS_PREFIX):]
                begins = pd.to_datetime(store.get(key)[TRIGGER_BEGIN_DATES])
                found += [(sid, day(date)) for date in begins if date < positional[sid]]
    return found


def reference_rows(path: Path) -> dict:
    """First date and row count of every saved reference series."""
    with pd.HDFStore(path, mode="r") as store:
        frames = {key: store.get(key) for key in sorted(store.keys()) if key.startswith(REFERENCE_PREFIX)}
    return {key: [day(frame.index[0]), len(frame)] for key, frame in frames.items()}


def curve_first(path: Path) -> str:
    """First date of a performance file's /total_variation (the curve index's start)."""
    with pd.HDFStore(path, mode="r") as store:
        return day(store.get("/total_variation")["date"].iloc[0])


def day(value: object) -> str:
    """ISO date of a timestamp-like value."""
    return str(pd.Timestamp(value).date())


if __name__ == "__main__":
    main()
