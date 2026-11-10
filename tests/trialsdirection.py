"""Regression for trials_to_true counted per direction (Q8).

(1) calculate_metrics on two rows frames whose plus and minus rows alternate: one without
series_id (the per-series trigger shape) and one with series_id and two plus rows sharing a begin
date in different series. Expected: each direction counted on its own, in (trigger_begin_dates,
series_id) sequence. (2) --task all on a realistic copy that keeps series 0 and 1, per-series
printouts and plots on as shipped: the persisted /signals.trials_to_true equals the per-direction
count over the whole table with that key; the run logs no ERROR record and every
"Processed N items, successful: M" line has N = M. (3) --task trigger signal, then the
/series_<id> of one signal series removed from the scratch signal store, then --task
performance: that series' rows, now without enrichment, still carry the per-direction value.
The unfixed code counts both directions together, per series, and leaves NaN on rows of a
series missing from the signal store.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.performance.perseries_.metrics import calculate_metrics

KEPT_SERIES = {"trigger.replay.mode": "realistic", "read.skip_series": ["2", "3", "4", "5"]}
PROCESSED = re.compile(r"Processed (\d+) items, successful: (\d+)")
ERROR_MARK = "ERROR"
TRIALS = "trials_to_true"
RTOL = 1e-6


def main() -> None:
    """The three cases, then the summary (return code 1 on any FAIL)."""
    in_process_counts()
    with testkit.scratch(KIT) as folder:
        (folder / "full").mkdir()
        persisted_table(folder / "full")
        (folder / "partial").mkdir()
        unenriched_rows(folder / "partial")
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def in_process_counts() -> None:
    """Case (1): calculate_metrics counts each direction on its own, ties broken by series_id."""
    days = pd.bdate_range("2022-01-03", periods=6)
    alternating = pd.DataFrame({"trigger_begin_dates": days, "is_plus_trigger": [True, False] * 3,
                                "is_true": [False, False, True, True, True, False]})
    tied = pd.DataFrame({"series_id": ["b", "a", "a", "c"], "trigger_begin_dates": [days[0], days[0], days[1], days[2]],
                         "is_plus_trigger": [True, True, True, False], "is_true": [False, True, True, True]})
    for name, frame in (("without series_id", alternating), ("with series_id and a tied begin", tied)):
        produced = calculate_metrics(pd.DataFrame(), frame, {TRIALS})
        expected = direction_counts(produced)
        KIT.verdict(f"case (1) {name}: calculate_metrics gives per-direction counts",
                    bool(np.allclose(produced[TRIALS].to_numpy(dtype=float), expected.to_numpy())),
                    expected.tolist(), produced[TRIALS].tolist())


def persisted_table(folder: Path) -> None:
    """Case (2): the persisted column over the whole /signals table, and a clean log of the per-series consumers."""
    config_file = testkit.config_copy(KIT, folder, strategy=dict(KEPT_SERIES))
    child = testkit.run_pipeline(KIT, folder, "all", config_file, ["all"])
    lines = testkit.plain_lines(child.text)
    processed = [(int(total), int(done)) for total, done in (match.groups() for match in map(PROCESSED.search, lines) if match)]
    errors = [line for line in lines if ERROR_MARK in line]
    KIT.verdict("case (2): the run returns 0 with no ERROR record and N = M on every Processed line",
                child.return_code == 0 and not errors and bool(processed) and all(total == done for total, done in processed),
                "return code 0, no ERROR, N = M", {"return_code": child.return_code, "errors": errors[:2], "processed": processed})
    check_table("case (2)", read_signals(config_file))


def unenriched_rows(folder: Path) -> None:
    """Case (3): rows of a series removed from the signal store keep a per-direction count."""
    config_file = testkit.config_copy(KIT, folder, strategy=dict(KEPT_SERIES))
    first = testkit.run_pipeline(KIT, folder, "triggersignal", config_file, ["trigger", "signal"])
    signal_file = testkit.output_file(config_file, "signal")
    with pd.HDFStore(str(signal_file), mode="a") as store:
        removed = sorted(key for key in store.keys() if key.startswith("/series_"))[0]
        store.remove(removed)
    second = testkit.run_pipeline(KIT, folder, "performance", config_file, ["performance"])
    signals = read_signals(config_file)
    rows = signals[signals["series_id"] == removed[len("/series_"):]]
    KIT.verdict(f"case (3): the rows of {removed} lost their enrichment (non-vacuity)",
                first.return_code == 0 and second.return_code == 0 and len(rows) > 0 and rows["variation"].isna().all(),
                "return code 0 twice, rows without variation", {"return_codes": [first.return_code, second.return_code], "rows": len(rows)})
    check_table("case (3)", signals, rows.index)


# =============================================================================
# HELPERS
# =============================================================================


def check_table(case: str, signals: pd.DataFrame, rows: Optional[pd.Index] = None) -> None:
    """The persisted trials_to_true (of the given rows, else all) against the per-direction count over the table."""
    expected = direction_counts(signals)
    labels = signals.index if rows is None else rows
    persisted = signals.loc[labels, TRIALS].to_numpy(dtype=float)
    differing = [int(label) for label, value, wanted in zip(labels, persisted, expected.loc[labels])
                 if not np.isclose(value, wanted, rtol=RTOL)]
    KIT.verdict(f"{case}: /signals.trials_to_true equals the per-direction count over the whole table "
                f"({len(labels)} rows)", len(labels) > 0 and not differing, "no row differs",
                {"differing_rows": differing[:6], "persisted_sum": float(np.nansum(persisted)),
                 "expected_sum": float(expected.loc[labels].sum())})


def direction_counts(rows: pd.DataFrame) -> pd.Series:
    """Trials to each true event per direction, indexed like rows: plus and minus rows apart, each counted in
    (trigger_begin_dates, series_id) sequence (series_id when present); 0 on false rows."""
    keys = ["trigger_begin_dates", "series_id"] if "series_id" in rows.columns else ["trigger_begin_dates"]
    counts = pd.Series(0.0, index=rows.index)
    for plus in (True, False):
        count = 1
        for label, value in rows[rows["is_plus_trigger"] == plus].sort_values(keys, kind="mergesort")["is_true"].items():
            counts[label] = float(count) if value else 0.0
            count = 1 if value else count + 1
    return counts


def read_signals(config_file: Path) -> pd.DataFrame:
    """The performance /signals of the copy (realistic), read in mode r."""
    with pd.HDFStore(str(testkit.output_file(config_file, "performance")), mode="r") as store:
        return store.get("/signals")


if __name__ == "__main__":
    main()
