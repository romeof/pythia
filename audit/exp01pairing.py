"""EXP-01: ideal-mode pairing (HYP-01, Q16), end_reason on cleaned rows (HYP-02), plateau extrema (HYP-22).

1. Synthetic n = 97 fixtures [max, min, max] and its mirror through analyze_series in ideal
   mode without a gate (pipelinekit.GATE_OFF): the classified sequence first, then the pairs and
   active flags against both Q16 readings (B: unmatched items removed, the decided reading as
   coded; A: trailing excess removed by count, the audited revision's reading, checked as no longer observed).
2. A plateau maximum, without a gate: find_peaks reports it; R-T13 expects a minus begin there.
3. The persisted ideal series of a run copy re-paired with reading B (the extrema masked by the
   gate on the previous bar of the copy's items, rising-edge dedup, each begin with the first
   strictly later unmatched end, first-wins overlap removal); every difference reported with
   its mechanism; end_reason rows counted against end rows (HYP-02).
4. Real-data ties: on every in-range series and every realistic window, the turning points
   find_peaks reports (computed here with the strategy's parameters) against the source root's
   minima and maxima (_classify_extrema at the audited revision, _find_extrema from the F-01 fix on);
   series 3 through analyze_series as coded and with each turning point typed by the
   find_peaks call that reported it.
Planted-defect control: a persisted series with one begin flag cleared must show that pair in
the comparison of section 3.

Structure:
1. main
2. Synthetic fixtures
3. Persisted ideal series
4. Real-data ties
5. Pairing helpers
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import pipelinekit
import synthetic
from src.utils.common.constants import (
    ACTIVE_TRIGGER_MINUS_BEGIN, ACTIVE_TRIGGER_PLUS_BEGIN, END_IDEAL_MODE, END_REASON,
    TRIGGER_BEGIN_DATES, TRIGGER_END_DATES, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, TRIGGER_PLUS_BEGIN,
    TRIGGER_PLUS_END, TURNING_POINTS, VAR3,
)
from src.utils.common.schema import AppConfig, GateItem
from src.utils.trigger.gate import applicable_items, begin_allowed
from src.utils.trigger.outcome import TriggerData
from src.utils.trigger.strategies import extrema

EXPERIMENT = "01"
FIXTURE_SIZE = 97
FIXTURE_SAMPLES = (16, 48, 80)
FIXTURE_START = "2020-01-01"
PLATEAU_SAMPLE = 16
TIE_SERIES = "3"
PLUS, MINUS = "plus", "minus"
FLAGS = {PLUS: (TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END), MINUS: (TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END)}
ACTIVE = {PLUS: ACTIVE_TRIGGER_PLUS_BEGIN, MINUS: ACTIVE_TRIGGER_MINUS_BEGIN}
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)


def main() -> None:
    """Fixtures, plateau, persisted series, ties; one RESULTS digest for the worktree/base comparison."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config_file = pipelinekit.write_config_copy(CTX, folder / "config")
    shipped = pipelinekit.load_copy(config_file)
    ideal, realistic = pipelinekit.concrete(shipped, "ideal"), pipelinekit.concrete(shipped, "realistic")
    ungated = pipelinekit.with_values(ideal, pipelinekit.GATE_OFF)
    results = {
        "fixtures": check_fixtures(ungated),
        "plateau": check_plateau(ungated),
        "persisted": check_persisted(ideal, args.run_copy),
        "ties": check_ties(shipped, realistic, folder),
    }
    digest = hashlib.md5(json.dumps(results, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    CTX.log.info("RESULTS %s %s", digest, json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy (the labelled run whose ideal series are re-paired) and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# SYNTHETIC FIXTURES
# =============================================================================


def check_fixtures(config: AppConfig) -> dict:
    """[max, min, max] and its mirror against reading A (as traced) and reading B."""
    x = np.linspace(0.0, 3.0 * np.pi, FIXTURE_SIZE)
    first, middle, last = FIXTURE_SAMPLES
    cases = {
        "max-min-max": (100.0 + 10.0 * np.sin(x), [(MINUS, first), (PLUS, middle), (MINUS, last)],
                        {(MINUS, first, middle)}, {PLUS: [middle], MINUS: [last]},
                        {(MINUS, first, middle), (PLUS, middle, last)}, {PLUS: [], MINUS: [last]}),
        "min-max-min": (100.0 - 10.0 * np.sin(x), [(PLUS, first), (MINUS, middle), (PLUS, last)],
                        {(PLUS, first, middle)}, {MINUS: [middle], PLUS: [last]},
                        {(PLUS, first, middle), (MINUS, middle, last)}, {MINUS: [], PLUS: [last]}),
    }
    found = {}
    for name, (values, sequence, pairs_a, active_a, pairs_b, active_b) in cases.items():
        frame = synthetic.daily_frame(values, FIXTURE_START)
        classified = classified_sequence(config, frame)
        CTX.verdict(f"fixture {name}: classified sequence", classified == sequence, sequence, classified)
        out = pipelinekit.analyze(name, frame, config)
        pairs, active = flag_pairs(out), active_rows(out)
        CTX.verdict(f"fixture {name}: the count-based result of reading A is no longer observed (Q16)",
                    not (pairs == pairs_a and active == active_a),
                    {"not pairs": sorted(pairs_a), "not active": active_a}, {"pairs": sorted(pairs), "active": active})
        CTX.verdict(f"fixture {name}: reading B as coded (unmatched items removed, Q16)", pairs == pairs_b and active == active_b,
                    {"pairs": sorted(pairs_b), "active": active_b}, {"pairs": sorted(pairs), "active": active})
        found[name] = {"pairs": sorted(pairs), "active": active}
    return found


def check_plateau(config: AppConfig) -> dict:
    """A two-sample plateau at a maximum: find_peaks reports it, R-T13 expects a minus begin there."""
    values = 100.0 + 10.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, FIXTURE_SIZE))
    values[PLATEAU_SAMPLE + 1] = values[PLATEAU_SAMPLE]
    strategy = pipelinekit.strategy_of(config)
    turning = np.union1d(*find_peaks_extrema(strategy, values)).tolist()
    minima, maxima = root_extrema(strategy, values)
    CTX.verdict("plateau: reported by find_peaks", PLATEAU_SAMPLE in turning, f"{PLATEAU_SAMPLE} in turning points", turning)
    CTX.verdict("plateau: classified as a maximum (R-T13)", PLATEAU_SAMPLE in maxima.tolist(),
                f"{PLATEAU_SAMPLE} in maxima", {"maxima": maxima.tolist(), "minima": minima.tolist()})
    out = pipelinekit.analyze("plateau", synthetic.daily_frame(values, FIXTURE_START), config)
    begin = int(out[TRIGGER_MINUS_BEGIN].iat[PLATEAU_SAMPLE]) + int(out[TRIGGER_PLUS_BEGIN].iat[PLATEAU_SAMPLE])
    marked = int(out[TURNING_POINTS].iat[PLATEAU_SAMPLE])
    CTX.verdict("plateau: turning point marked and a begin generated", marked == 1 and begin == 1,
                "turning_points 1 and a begin", {"turning_points": marked, "begin": begin})
    return {"turning": turning, "maxima": maxima.tolist(), "minima": minima.tolist(), "begin": begin}


def classified_sequence(config: AppConfig, frame: pd.DataFrame) -> list:
    """(direction of the begin, row) of every classified extremum, by row."""
    marked = pipelinekit.strategy_of(config)._generate_trigger_begin(frame.copy())
    begins = [(PLUS, int(row)) for row in np.flatnonzero(marked[TRIGGER_PLUS_BEGIN].to_numpy() > 0)]
    begins += [(MINUS, int(row)) for row in np.flatnonzero(marked[TRIGGER_MINUS_BEGIN].to_numpy() > 0)]
    return sorted(begins, key=lambda item: item[1])


# =============================================================================
# PERSISTED IDEAL SERIES
# =============================================================================


def check_persisted(config: AppConfig, run_copy: Path) -> dict:
    """Reading-B pairs of every persisted ideal series against the persisted pairs."""
    path = run_copy / PARAMS / "ideal" / "trigger" / "0.h5"
    strategy = pipelinekit.strategy_of(config)
    items = list(config.trigger.gate.items)
    found, compared = {}, 0
    with pd.HDFStore(path, mode="r") as store:
        series_keys = sorted(key for key in store.keys() if key.startswith(harness.SERIES_KEY_PREFIX))
        for key in series_keys:
            frame = store.get(key)
            sid = key[len(harness.SERIES_KEY_PREFIX):]
            begins, ends = derived_flags(sid, frame, strategy, items)
            independent = reading_b_pairs(begins, ends)
            persisted = flag_pairs(frame)
            compared += len(independent) + len(persisted)
            reason_rows = int((frame[END_REASON] == END_IDEAL_MODE).sum())
            end_rows = int(((frame[TRIGGER_PLUS_END] > 0) | (frame[TRIGGER_MINUS_END] > 0)).sum())
            only_reading_b, only_persisted = pair_differences(independent, persisted)
            found[sid] = {"only_reading_b": only_reading_b, "only_persisted": only_persisted,
                          "mechanism": balance_mechanism(begins, ends), "end_reason_rows": reason_rows, "end_rows": end_rows,
                          "first_last": first_last(begins, frame.index)}
            CTX.log.info("series %s: %s", sid, json.dumps(found[sid], default=str))
            CTX.verdict(f"series {sid}: persisted pairs equal reading-B pairs", not found[sid]["only_reading_b"]
                        and not found[sid]["only_persisted"], "no difference",
                        {k: found[sid][k] for k in ("only_reading_b", "only_persisted", "mechanism")})
            CTX.verdict(f"series {sid}: end_reason only on end rows (HYP-02)", reason_rows == end_rows, end_rows, reason_rows)
    CTX.verdict("persisted: non-vacuity", compared > 0 and len(series_keys) == 6, "6 series, pairs compared > 0",
                {"series": len(series_keys), "pairs": compared})
    sample = _series(path, min(found))
    independent = reading_b_pairs(*derived_flags(min(found), sample, strategy, items))
    common = sorted(independent & flag_pairs(sample))
    removed = common[0] if common else None
    only_reading_b, only_persisted = (pair_differences(independent, flag_pairs(_without_begin(sample, removed)))
                                      if removed else ([], []))
    CTX.verdict("control: a pair whose begin is cleared in the persisted flags is flagged",
                removed is not None and removed in only_reading_b, "the cleared pair among the differences",
                {"removed": removed, "only_reading_b": only_reading_b, "only_persisted": only_persisted})
    return found


def _series(path: Path, sid: str) -> pd.DataFrame:
    """One persisted series frame."""
    with pd.HDFStore(path, mode="r") as store:
        return store.get(f"{harness.SERIES_KEY_PREFIX}{sid}")


def pair_differences(independent: set, persisted: set) -> tuple[list, list]:
    """The pairs only the independent pairing holds and those only the persisted flags hold."""
    return sorted(independent - persisted), sorted(persisted - independent)


def _without_begin(frame: pd.DataFrame, pair: tuple) -> pd.DataFrame:
    """A copy of the frame with the begin flag of one (direction, begin row, end row) pair cleared: a planted lost trigger."""
    direction, begin, _ = pair
    planted = frame.copy()
    planted.iloc[begin, planted.columns.get_loc(FLAGS[direction][0])] = 0
    return planted


def derived_flags(series_id: str, df: pd.DataFrame, strategy, items: Sequence[GateItem]) -> tuple[dict, dict]:
    """Begin and end flags re-derived with the strategy's own extrema and the gate on the previous bar (the items
    that apply to the series, read on the persisted frame's gate columns)."""
    values = df[VAR3].to_numpy(dtype=float)
    minima, maxima = root_extrema(strategy, values)
    allowed = begin_allowed(series_id, df, applicable_items(series_id, items))
    begins = {PLUS: _flags(len(values), minima) & allowed, MINUS: _flags(len(values), maxima) & allowed}
    return begins, {PLUS: begins[MINUS].copy(), MINUS: begins[PLUS].copy()}


def _flags(size: int, rows: np.ndarray) -> np.ndarray:
    """Boolean flags at the given rows."""
    flags = np.zeros(size, dtype=bool)
    flags[np.asarray(rows, dtype=int)] = True
    return flags


def first_last(begins: dict, index: pd.DatetimeIndex) -> dict:
    """Type and date of the first and last classified extremum (a minimum begins plus)."""
    rows = sorted([(int(row), "minimum") for row in np.flatnonzero(begins[PLUS])]
                  + [(int(row), "maximum") for row in np.flatnonzero(begins[MINUS])])
    if not rows:
        return {}
    return {"first": (rows[0][1], str(index[rows[0][0]].date())), "last": (rows[-1][1], str(index[rows[-1][0]].date())),
            "maxima": int(begins[MINUS].sum()), "minima": int(begins[PLUS].sum())}


def balance_mechanism(begins: dict, ends: dict) -> dict:
    """Per direction: the deduplicated begin and end counts and whether an end precedes the first begin."""
    found = {}
    for direction in (PLUS, MINUS):
        rows_begin, rows_end = _deduplicated(begins[direction], ends[direction])
        found[direction] = {"begins": len(rows_begin), "ends": len(rows_end),
                            "leading_end": bool(rows_end and (not rows_begin or rows_end[0] < rows_begin[0]))}
    return found


# =============================================================================
# REAL-DATA TIES
# =============================================================================


def check_ties(shipped: AppConfig, realistic: AppConfig, folder: Path) -> dict:
    """Untyped find_peaks turning points on full series and on realistic windows; series 3 as coded vs typed by
    find_peaks."""
    strategy = pipelinekit.strategy_of(realistic)
    lookback = realistic.trigger.replay.lookback_window
    series = pipelinekit.input_series(shipped, folder / "input")
    found = {}
    for sid, frame in sorted(series.items()):
        values = frame[VAR3].to_numpy(dtype=float)
        full = _unclassified(strategy, values)
        windows = {}
        for end in range(lookback - 1, len(values)):
            start = end - lookback + 1
            hits = [start + int(row) for row in _unclassified(strategy, values[start:end + 1])]
            if hits:
                windows[end] = hits
        dates = sorted({str(frame.index[row].date()) for hits in windows.values() for row in hits})
        found[sid] = {"full": [str(frame.index[row].date()) for row in full], "windows": len(windows),
                      "window_rows": [min(windows), max(windows)] if windows else [], "points": dates}
        CTX.log.info("ties series %s: %s", sid, json.dumps(found[sid]))
    CTX.verdict("ties: no unclassified turning point on any full series",
                not any(item["full"] for item in found.values()), "none", {sid: item["full"] for sid, item in found.items()})
    CTX.verdict("ties: no unclassified turning point in any realistic window",
                not any(item["windows"] for item in found.values()), "none",
                {sid: (item["windows"], item["points"]) for sid, item in found.items() if item["windows"]})
    found["series3_effect"] = typed_comparison(realistic, series[TIE_SERIES])
    return found


def _unclassified(strategy, values: np.ndarray) -> list[int]:
    """Turning points find_peaks reports (computed here) that the source root types as neither minimum nor maximum."""
    turning = np.union1d(*find_peaks_extrema(strategy, values))
    minima, maxima = root_extrema(strategy, values)
    return sorted(set(turning.tolist()) - set(minima.tolist()) - set(maxima.tolist()))


def root_extrema(strategy, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minima and maxima as the source root types them: from the F-01 fix on, _find_extrema types each point by its
    find_peaks call; at the audited revision, _classify_extrema types _find_turning_points by strict neighbour comparisons."""
    if hasattr(strategy, "_find_extrema"):
        return strategy._find_extrema(values)
    return strategy._classify_extrema(values, strategy._find_turning_points(values))


def find_peaks_extrema(strategy, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minima and maxima of find_peaks with the strategy's parameters, computed here and independent of the source root:
    their union is the set of turning points the root must type."""
    prominence = strategy.prominence_threshold * (np.max(data) - np.min(data))
    maxima, _ = find_peaks(data, distance=strategy.min_distance, prominence=prominence, width=strategy.width)
    minima, _ = find_peaks(-data, distance=strategy.min_distance, prominence=prominence, width=strategy.width)
    return minima, maxima


def typed_comparison(config: AppConfig, frame: pd.DataFrame) -> dict:
    """Series 3 in realistic mode as coded and with each point typed by the find_peaks call that reported it, in the
    frame path (_generate_trigger_begin) and, on a root that has it, in the array path the realistic loop reads
    (_begin_candidate_rows)."""
    coded = pipelinekit.analyze(TIE_SERIES, frame, config)
    original = extrema.ExtremaTrigger._generate_trigger_begin
    array_path = hasattr(extrema.ExtremaTrigger, "_begin_candidate_rows")
    original_rows = extrema.ExtremaTrigger._begin_candidate_rows if array_path else None
    extrema.ExtremaTrigger._generate_trigger_begin = _typed_by_find_peaks
    if array_path:
        extrema.ExtremaTrigger._begin_candidate_rows = _typed_rows
    try:
        typed = pipelinekit.analyze(TIE_SERIES, frame, config)
    finally:
        extrema.ExtremaTrigger._generate_trigger_begin = original
        if array_path:
            extrema.ExtremaTrigger._begin_candidate_rows = original_rows
    helper = TriggerData()
    pairs = {name: _extracted(helper, out) for name, out in (("coded", coded), ("typed", typed))}
    found = {"begins_only_coded": _rows_only(coded, typed, (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN)),
             "begins_only_typed": _rows_only(typed, coded, (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN)),
             "ends_only_coded": _rows_only(coded, typed, (TRIGGER_PLUS_END, TRIGGER_MINUS_END)),
             "ends_only_typed": _rows_only(typed, coded, (TRIGGER_PLUS_END, TRIGGER_MINUS_END)),
             "pairs_only_coded": sorted(pairs["coded"] - pairs["typed"]),
             "pairs_only_typed": sorted(pairs["typed"] - pairs["coded"]), "pairs_coded": len(pairs["coded"])}
    CTX.log.info("series 3 typed comparison: %s", json.dumps(found, default=str))
    CTX.verdict("series 3: begins, ends and pairs unchanged by typing each point by its find_peaks call",
                not any(value for key, value in found.items() if key != "pairs_coded"), "no difference", found)
    return found


def _typed_by_find_peaks(self, df: pd.DataFrame) -> pd.DataFrame:
    """The begin generation with each turning point typed by the find_peaks call that reported it."""
    minima, maxima = find_peaks_extrema(self, df[VAR3].values)
    df = self._initialize_columns(df, extrema.BEGIN_COLUMNS)
    return self._mark_triggers(df, minima, maxima, np.unique(np.concatenate([maxima, minima])))


def _typed_rows(self, var3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The array path's three row sets with each turning point typed by the find_peaks call that reported it: minima,
    maxima and their union."""
    minima, maxima = find_peaks_extrema(self, var3)
    return minima, maxima, np.union1d(minima, maxima)


def _rows_only(left: pd.DataFrame, right: pd.DataFrame, columns: tuple) -> list:
    """(column, date) flagged in left and not in right."""
    return [(column, str(date.date())) for column in columns
            for date in left.index[(left[column].to_numpy() > 0) & ~(right[column].to_numpy() > 0)]]


def _extracted(helper: TriggerData, out: pd.DataFrame) -> set:
    """(begin, end, direction) of the triggers the extraction pairs from the flags."""
    triggers = helper._extract_triggers_from_trigger_columns(TIE_SERIES, out)
    if triggers is None:
        return set()
    return {(str(begin.date()), str(end.date()), bool(plus)) for begin, end, plus in
            zip(triggers[TRIGGER_BEGIN_DATES], triggers[TRIGGER_END_DATES], triggers["is_plus_trigger"])}


# =============================================================================
# PAIRING HELPERS
# =============================================================================


def flag_pairs(frame: pd.DataFrame) -> set:
    """(direction, begin row, end row): each begin flag with the first strictly later unused end flag."""
    pairs = set()
    for direction, (begin_col, end_col) in FLAGS.items():
        ends = np.flatnonzero(frame[end_col].to_numpy() > 0).tolist()
        used = set()
        for begin in np.flatnonzero(frame[begin_col].to_numpy() > 0):
            end = next((row for row in ends if row > begin and row not in used), None)
            if end is not None:
                used.add(end)
                pairs.add((direction, int(begin), int(end)))
    return pairs


def active_rows(frame: pd.DataFrame) -> dict:
    """Rows carrying each active-trigger flag."""
    return {direction: np.flatnonzero(frame[column].to_numpy() > 0).tolist() for direction, column in ACTIVE.items()}


def reading_b_pairs(begins: dict, ends: dict) -> set:
    """Q16 reading B: dedup on the triggers-only timeline, 1:1 pairing, first-wins overlap removal."""
    candidates = []
    for direction in (PLUS, MINUS):
        rows_begin, rows_end = _deduplicated(begins[direction], ends[direction])
        used = set()
        for begin in rows_begin:
            end = next((row for row in rows_end if row > begin and row not in used), None)
            if end is not None:
                used.add(end)
                candidates.append((begin, end, direction))
    accepted = []
    for begin, end, direction in sorted(candidates, key=lambda item: item[0]):
        if not accepted or begin >= accepted[-1][1]:
            accepted.append((begin, end, direction))
    return {(direction, begin, end) for begin, end, direction in accepted}


def _deduplicated(begin_flags: np.ndarray, end_flags: np.ndarray) -> tuple[list, list]:
    """Rising edges (0 to 1) of begin and end flags on the triggers-only timeline, as row numbers."""
    rows = np.flatnonzero(begin_flags | end_flags)
    if rows.size == 0:
        return [], []
    states = np.stack([begin_flags[rows], end_flags[rows]], axis=1).astype(np.int8)
    rising = states.copy()
    rising[1:] = ((states[1:] - states[:-1]) == 1).astype(np.int8)
    return rows[rising[:, 0] == 1].tolist(), rows[rising[:, 1] == 1].tolist()


if __name__ == "__main__":
    main()
