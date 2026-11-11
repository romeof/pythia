"""Regression for the gate's scope, its pre-filter and its missing-column error in the pipeline (C4, C5, C6).

Dual --task trigger runs, each with the pre-filter on and off, on scratch copies reading synthetic three-series inputs:
200 daily rows of a sine (var3 = var4 = 100 + 10 sin(2 pi t / 40), var0 = 0.5 x var3, var1 and var2 1 % below and
above, var5 1e6), series 0 the reference, the items var12 >= 0.83 and var15 >= 0.73.
File A (C4): the three series hold identical values, var12 0.2 and var15 1.0, and the var12 item exempts series 0 and
2: in each mode no /triggers_1, while /triggers_0 and /triggers_2 are present and equal across the two runs, and
/series_1 is absent with the pre-filter on only.
File B (C6): the var12 item exempts series 0 and every series holds var12 1.0; series 0 and 1 hold var15 0.5 on every
bar, series 2 on its first 100 bars and 1.0 after: in each mode the key sets of the two runs differ by /series_1
alone, /triggers_2 holds a row and /series_0 is present in both, and every /triggers_* and /reference_series_* frame
is equal across the runs (DataFrame.equals); the run with the pre-filter on holds one INFO record per mode naming the
item var15 >= 0.73 that skips series 1, the other none.
File C (C5): the var12 item exempts series 0, which lacks var12, while series 1 holds every column: with the
pre-filter on and off the run returns non-zero and one Task failed record carries the exact message
"series 0 lacks the configured gate columns ['var12']". Detection checks only the items that apply to a series, so
the producer's check of every configured column is the one that fails; a pre-filter that skipped a reference series
or ran before that check, or a gate that ignored exempt_series, fails a verdict.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROWS, PERIOD = 200, 40
DATES = pd.date_range("2020-01-01", periods=ROWS, freq="D")
SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
SERIES = ("0", "1", "2")
PASSING, FAILING_VAR12, FAILING_VAR15 = 1.0, 0.2, 0.5
VAR15_ITEM = testkit.gate_item("var15", ">=", 0.73)
SWITCHES = {"on": True, "off": False}
SKIP_RECORD = "Skipping series 1: gate item var15 >= 0.73 passes on no row of the read window"
MISSING_MESSAGE = "series 0 lacks the configured gate columns ['var12']"
SERIES_KEY, TRIGGERS_PREFIX, REFERENCE_PREFIX = "/series_", "/triggers_", "/reference_series_"


def main() -> None:
    """Files A, B and C, then the summary (return code 1 on any FAIL)."""
    exempt_scope()
    prefilter_scope()
    missing_column()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def exempt_scope() -> None:
    """File A (C4): series 1 fails the var12 item on every bar and gets no trigger; the exempt series 2 and the exempt
    reference series 0, with the same values, keep theirs, with the pre-filter on and off."""
    frames = {series_id: series_frame(np.full(ROWS, FAILING_VAR12), np.full(ROWS, PASSING)) for series_id in SERIES}
    with testkit.scratch(KIT) as folder:
        runs = switched_runs(folder, "A", frames, [testkit.gate_item("var12", ">=", 0.83, ["0", "2"]), VAR15_ITEM])
    returned_zero("A", runs)
    for mode in testkit.MODES:
        stores = {switch: run[mode] for switch, run in runs.items()}
        present = {switch: {key: key in store for key in ("/triggers_0", "/triggers_1", "/triggers_2", "/series_1")}
                   for switch, store in stores.items()}
        expected = {switch: {"/triggers_0": True, "/triggers_1": False, "/triggers_2": True, "/series_1": not prefilter}
                    for switch, prefilter in SWITCHES.items()}
        KIT.verdict(f"file A {mode}: no /triggers_1, /triggers_0 and /triggers_2 present, /series_1 absent with the "
                    f"pre-filter on only", present == expected, expected, present)
        equal = {key: equal_across(stores, key) for key in ("/triggers_0", "/triggers_2")}
        KIT.verdict(f"file A {mode}: /triggers_0 and /triggers_2 equal across the two runs", all(equal.values()),
                    {key: True for key in equal}, equal)


def prefilter_scope() -> None:
    """File B (C6): with the pre-filter on only the non-reference series that var15 passes on no bar is skipped; the
    trigger and reference frames are equal across the runs."""
    second_half = np.where(np.arange(ROWS) >= ROWS // 2, PASSING, FAILING_VAR15)
    var15 = {"0": np.full(ROWS, FAILING_VAR15), "1": np.full(ROWS, FAILING_VAR15), "2": second_half}
    frames = {series_id: series_frame(np.full(ROWS, PASSING), var15[series_id]) for series_id in SERIES}
    with testkit.scratch(KIT) as folder:
        runs = switched_runs(folder, "B", frames, [testkit.gate_item("var12", ">=", 0.83, ["0"]), VAR15_ITEM])
    returned_zero("B", runs)
    for mode in testkit.MODES:
        on, off = runs["on"][mode], runs["off"][mode]
        difference = {"off only": sorted(set(off) - set(on)), "on only": sorted(set(on) - set(off))}
        KIT.verdict(f"file B {mode}: the key sets of the two runs differ by /series_1 alone", difference ==
                    {"off only": ["/series_1"], "on only": []}, {"off only": ["/series_1"], "on only": []}, difference)
        held = {switch: {"/triggers_2 rows": len(run[mode].get("/triggers_2", ())), "/series_0": "/series_0" in run[mode]}
                for switch, run in runs.items()}
        KIT.verdict(f"file B {mode}: non-vacuity, /triggers_2 holds a row and /series_0 is present in both runs",
                    all(found["/triggers_2 rows"] > 0 and found["/series_0"] for found in held.values()),
                    "rows > 0, present", held)
        compared = sorted(key for key in set(on) | set(off) if key.startswith((TRIGGERS_PREFIX, REFERENCE_PREFIX)))
        unequal = [key for key in compared if not equal_across({"on": on, "off": off}, key)]
        KIT.verdict(f"file B {mode}: every /triggers_* and /reference_series_* frame equal across the runs",
                    bool(compared) and not unequal, {"compared": ">= 1", "unequal": []},
                    {"compared": compared, "unequal": unequal})
    records = {switch: sum(SKIP_RECORD in line for line in testkit.plain_lines(run["capture"]))
               for switch, run in runs.items()}
    expected = {"on": len(testkit.MODES), "off": 0}
    KIT.verdict("file B: one INFO record per mode names the item var15 >= 0.73 skipping series 1, with the pre-filter "
                "on only", records == expected, expected, records)


def missing_column() -> None:
    """File C (C5): a configured gate column absent from the exempt reference series fails the run, with the
    pre-filter on and off, with one Task failed record carrying the exact message."""
    frames = {series_id: series_frame(np.full(ROWS, PASSING), np.full(ROWS, PASSING)) for series_id in ("0", "1")}
    frames["0"] = frames["0"].drop(columns=["var12"])
    with testkit.scratch(KIT) as folder:
        runs = switched_runs(folder, "C", frames, [testkit.gate_item("var12", ">=", 0.83, ["0"]), VAR15_ITEM])
    for switch, run in runs.items():
        records = [line for line in testkit.plain_lines(run["capture"])
                   if testkit.HANDLER_PREFIX in line and MISSING_MESSAGE in line]
        KIT.verdict(f"file C, pre-filter {switch}: return != 0, one '{testkit.HANDLER_PREFIX}' record carries "
                    f"'{MISSING_MESSAGE}'", run["return_code"] not in (0, None) and len(records) == 1,
                    {"return_code": "!= 0", "records": 1}, {"return_code": run["return_code"], "records": records})


# =============================================================================
# HELPERS
# =============================================================================


def series_frame(var12: np.ndarray, var15: np.ndarray) -> pd.DataFrame:
    """The sine series with the given var12 and var15."""
    levels = 100.0 + 10.0 * np.sin(2.0 * np.pi * np.arange(ROWS) / PERIOD)
    return pd.DataFrame({"var0": 0.5 * levels, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels,
                         "var4": levels, "var5": np.full(ROWS, 1.0e6), "var12": var12, "var15": var15}, index=DATES)


def switched_runs(folder: Path, label: str, frames: dict[str, pd.DataFrame], items: list[dict]) -> dict[str, dict]:
    """The input in the folder, then per pre-filter switch a dual --task trigger run on a copy reading it, each copy
    in its own folder (the output name never holds the switch): its return code, capture and the frames of each
    mode's trigger store (an empty mapping when the store is absent)."""
    raw = folder / SYNTHETIC_INPUT
    testkit.write_raw_input(raw, frames)
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    runs = {}
    for switch, prefilter in SWITCHES.items():
        (folder / switch).mkdir()
        config_file = testkit.config_copy(KIT, folder / switch, strategy={
            "read.raw_data": f"./{relative}", "read.ini_date": day(DATES[0]), "read.end_date": day(DATES[-1]),
            "read.reference_series": ["0"], "trigger.replay.mode": testkit.DUAL, "trigger.gate.items": items,
            "trigger.gate.prefilter": prefilter})
        child = testkit.run_pipeline(KIT, folder / switch, f"{label}_{switch}", config_file, ["trigger"])
        stores = {mode: stored_frames(testkit.mode_output(config_file, mode, "trigger")) for mode in testkit.MODES}
        runs[switch] = {"return_code": child.return_code, "capture": child.text, **stores}
    return runs


def returned_zero(label: str, runs: dict[str, dict]) -> None:
    """Both runs of the file return 0."""
    codes = {switch: run["return_code"] for switch, run in runs.items()}
    KIT.verdict(f"file {label}: --task trigger returns 0 with the pre-filter on and off", codes == {"on": 0, "off": 0},
                {"on": 0, "off": 0}, codes)


def stored_frames(path: Path) -> dict[str, pd.DataFrame]:
    """Every frame of a store by key; an empty mapping when the store is absent."""
    return testkit.store_frames(path) if path.is_file() else {}


def equal_across(stores: dict[str, dict], key: str) -> bool:
    """The key's frame is present in every run's store and equal across them."""
    frames = [store.get(key) for store in stores.values()]
    return all(frame is not None for frame in frames) and all(frame.equals(frames[0]) for frame in frames[1:])


def day(date: pd.Timestamp) -> str:
    """A date as the read section writes it."""
    return date.strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
