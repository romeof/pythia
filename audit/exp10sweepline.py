"""EXP-10: the weight constraint's sweep line (Q4, R-P3).

Unit cases on production sweep_line_filter under the decided reading (Q4, performance.md:141):
end events before begin events on the same date, a zero-length signal's own end after its day's begin
events, so it counts on its day. Cases: a signal beginning on the day another ends, the removal of the
newest active signal, same-day begin events in insertion sequence, and a zero-length signal followed
by a later one (a plain ends-first rule would keep the zero-length signal active and remove
the later one). The begins-first reading of the audited revision is checked as no longer observed wherever
it gives another result.

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

import pandas as pd

from src.utils.performance.gradient_.validation import sweep_line_filter

DAYS = pd.bdate_range("2022-01-03", periods=12)
CASES = {
    "begin on the day another ends": {
        "bounds": {1: (DAYS[0], DAYS[4]), 2: (DAYS[4], DAYS[8]), 3: (DAYS[4], DAYS[7])}, "weight": 0.5, "maximum": 1.0,
        "begins_first": {3}, "decided": set()},
    "newest active signal removed": {
        "bounds": {1: (DAYS[0], DAYS[9]), 2: (DAYS[1], DAYS[9]), 3: (DAYS[2], DAYS[9])}, "weight": 0.4, "maximum": 1.0,
        "begins_first": {3}, "decided": {3}},
    "same-day begin events in insertion sequence": {
        "bounds": {1: (DAYS[0], DAYS[5]), 2: (DAYS[0], DAYS[6])}, "weight": 0.6, "maximum": 1.0,
        "begins_first": {2}, "decided": {2}},
    "zero-length signal, then a later one": {
        "bounds": {1: (DAYS[2], DAYS[2]), 2: (DAYS[5], DAYS[8])}, "weight": 1.0, "maximum": 1.0,
        "begins_first": set(), "decided": set()},
}


def main() -> None:
    """Each case: the removed signals against the decided reading; the begins-first reading where it differs."""
    _parse_args()
    found = {}
    for name, case in CASES.items():
        removed, peak = sweep_line_filter(case["bounds"], {gid: {} for gid in case["bounds"]}, case["weight"], case["maximum"])
        found[name] = {"removed": sorted(removed), "max_concurrent": peak}
        CTX.verdict(f"{name}: as coded (end events before begin events, a zero-length signal's end after its day's begin events, Q4, "
                    f"performance.md:141)", removed == case["decided"], sorted(case["decided"]), sorted(removed))
        agrees = case["begins_first"] == case["decided"]
        CTX.verdict(f"{name}: the begins-first reading of the audited revision is no longer observed where it differs (Q4)",
                    agrees or removed != case["begins_first"],
                    "same result" if agrees else f"not {sorted(case['begins_first'])}", sorted(removed))
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
