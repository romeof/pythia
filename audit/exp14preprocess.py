"""EXP-14: the preprocess drop rules and the consecutive-block report (R-T3, HYP-17).

Synthetic series for the NaN check and each shipped rule of trigger.validity (nonnegative,
nonzero, not_all_equal) through production Preprocess.process_series, and a three-row bad block on consecutive
rows with dates missing inside it: documents "consecutive rows" (trigger.md:91) against
the row rule of _find_consecutive_blocks (preprocess.py:170-196; F-20).

Structure:
1. main
2. Cases
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

import numpy as np
import pandas as pd

import pipelinekit
from src.utils.common.schema import AppConfig
import synthetic

EXPERIMENT = "14"
SIZE = 20
START = "2022-01-03"
RULE_ROWS = {"NaN": 3, "nonnegative": 5, "nonzero": 7, "not_all_equal": 9}
GAP_BLOCK = ("2022-01-13", "2022-01-14", "2022-01-17")
CALENDAR_BLOCK = ("2022-01-25", "2022-01-26", "2022-01-27")
CONSECUTIVE = "Consecutive Issues"


def main() -> None:
    """Each drop rule removes its row; a block of rows across missing dates against a calendar-day block."""
    _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    found = {"rules": check_rules(config, folder / "rules"), "blocks": check_blocks(config, folder)}
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


def base_frame() -> pd.DataFrame:
    """A clean frame of SIZE rows from START."""
    return synthetic.daily_frame(100.0 + np.arange(SIZE, dtype=float), START)


def check_rules(config: AppConfig, folder: Path) -> dict:
    """One planted row per rule; exactly those rows are dropped."""
    frame = base_frame().astype({"var5": "float64"})
    frame.iloc[RULE_ROWS["NaN"], frame.columns.get_loc("var2")] = np.nan
    frame.iloc[RULE_ROWS["nonnegative"], frame.columns.get_loc("var1")] = -1.0
    frame.iloc[RULE_ROWS["nonzero"], frame.columns.get_loc("var5")] = 0.0
    row = RULE_ROWS["not_all_equal"]
    frame.iloc[row, [frame.columns.get_loc(column) for column in ("var0", "var1", "var2")]] = frame["var3"].iat[row]
    CTX.guard(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cleaned = pipelinekit.preprocessor_of(config, folder / "0.h5").process_series("rules", frame)
    dropped = sorted(frame.index.get_loc(date) for date in frame.index.difference(cleaned.index))
    CTX.verdict("each drop rule removes exactly its planted row", dropped == sorted(RULE_ROWS.values()),
                sorted(RULE_ROWS.values()), dropped)
    return {"dropped_rows": dropped}


def check_blocks(config: AppConfig, folder: Path) -> dict:
    """Three bad rows across missing dates versus three calendar-consecutive bad days."""
    found = {}
    for name, dates in (("gap", GAP_BLOCK), ("calendar", CALENDAR_BLOCK)):
        frame = synthetic.daily_frame(100.0 + np.arange(30, dtype=float), START)
        frame.loc[pd.to_datetime(list(dates)), "var5"] = 0
        target = folder / name
        CTX.guard(target)
        target.mkdir(parents=True, exist_ok=True)
        preprocessor = pipelinekit.preprocessor_of(config, target / "0.h5")
        preprocessor.process_series(name, frame)
        report = preprocessor.report_path.read_text(encoding="utf-8")
        found[name] = {"dates": list(dates), "reported": CONSECUTIVE in report}
    threshold = config.trigger.consecutive_rows_with_bad_preprocessed_data
    CTX.verdict("calendar-consecutive block of three is reported", found["calendar"]["reported"], True, found["calendar"])
    CTX.verdict(f"three-row block across missing dates is reported (consecutive rows, threshold {threshold})",
                found["gap"]["reported"], True, found["gap"])
    return found


if __name__ == "__main__":
    main()
