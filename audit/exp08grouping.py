"""EXP-08: signal grouping (R-S1..S3, Q15, Q20, HYP-19, HYP-26, A4(b)).

An independent replay of the documented grouping rule (signal.md:58-75,156-175) on the
persisted trigger tables of a run copy, compared per row with the persisted /signals
(group_id, is_signal, corr_to_signal, slope, num_series, series_ids). The replay follows the
decided precedence of each mode: ideal takes eligible signals by signal_end, newest first among
equal ends (Q15, as coded); realistic by (signal_begin, group_id), never by signal_end (Q20).
The persisted partition is judged against the decided replay, with each alternative's delta
logged (ideal: oldest first, point-in-time; realistic: the end precedence and, for Q15, the
equal-end insertion rule, which the point-in-time key must leave without effect). A4(b): for
every realistic trigger, production SignalSelection rebuilds the prefix up to it with the ends
of the signals it can join moved past every end in reversed sequence; its assignment and every
earlier one must stay. Final-state invariants, an independent numpy Pearson and polyfit sample,
and the HYP-19 synthetic case on production SignalSelection end the experiment. Planted
controls: a changed persisted group_id and a precedence-sensitive synthetic case.

Structure:
1. main
2. Inputs from the run copy
3. Replay
4. Comparisons and invariants
5. HYP-19 cache case
6. Controls
7. Selection parameters (signal_params, selection_of)
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import bisect
import dataclasses
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import pipelinekit
from src.utils.common.constants import (
    CORR_TO_SIGNAL, GROUP_ID, IS_PLUS_TRIGGER, IS_SIGNAL, IS_TRUE, NUM_SERIES, SERIES_ID, SERIES_IDS, SLOPE,
    TRIGGER_BEGIN_DATES, TRIGGER_END_DATES, VAR3,
)
from src.utils.common.schema import AppConfig, SignalConfig
from src.utils.signal.selection import SignalSelection

EXPERIMENT = "08"
MODES = ("ideal", "realistic")
FLOAT32_TOLERANCE = 1e-5
SAMPLE_STEP = 10
PRECEDENCE_CODE, PRECEDENCE_OLDEST, PRECEDENCE_BEGIN = "code", "oldest-first", "point-in-time"
PRECEDENCE_BEGIN_OLDEST = "point-in-time, oldest-first insertion"
DECIDED = {"ideal": PRECEDENCE_CODE, "realistic": PRECEDENCE_BEGIN}  # Q15: ideal by signal_end, newest first; Q20
POINT_IN_TIME_KEY = "point_in_time"  # the point-in-time rule of Q20 among the replay's parameters
SIGNAL_KEYS = ("corr_threshold", "period_back", "signal_criterion")  # the signal section's selection parameters
END_SHIFTS = (1, 5)  # k: moved ends at latest_end + BDay(k + reversed rank)
CONTROL_PREFIXES = 5
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)


@dataclasses.dataclass
class ActiveSignal:
    """One active signal of the replay."""

    group_id: int
    begin: pd.Timestamp
    end: pd.Timestamp
    series_id: str
    slope: float


def main() -> None:
    """Both modes: decided replay, comparisons, alternatives, invariants, sample; then A4(b), HYP-19 and the controls."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    params = signal_params(config, config.trigger.replay.mode)
    CTX.log.info("signal parameters %s", params)
    results = {}
    for mode in MODES:
        series, triggers, persisted = load_inputs(args.run_copy / PARAMS / mode)
        replayed = replay(series, triggers, params, DECIDED[mode])
        results[mode] = {"rows": compare_rows(mode, persisted, replayed),
                         "precedence": precedence_verdicts(mode, series, triggers, params, persisted, replayed),
                         "precedence_sensitive_triggers": precedence_sensitive(series, triggers, params),
                         "invariants": invariants(mode, persisted, series, params),
                         "sample": numpy_sample(mode, persisted, series, params)}
        if mode == "realistic":
            results[mode]["variant_table"] = write_variant(folder / "variant_realistic.h5", replayed)
    results["invariance"] = check_point_in_time_invariance(args.run_copy, config)
    results["hyp19"] = check_cache_orientation(params)
    check_controls(args.run_copy, params)
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# INPUTS FROM THE RUN COPY
# =============================================================================


def load_inputs(mode_folder: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Per-series var3 (as Signal stores it), all triggers sorted by (begin, series_id), and the persisted /signals."""
    series, parts = {}, []
    with pd.HDFStore(mode_folder / "trigger" / "0.h5", mode="r") as store:
        for key in sorted(store.keys()):
            if key.startswith(harness.SERIES_KEY_PREFIX):
                series[key[len(harness.SERIES_KEY_PREFIX):]] = store.get(key)[VAR3].dropna().astype(float)
            elif key.startswith("/triggers_"):
                part = store.get(key).copy()
                part[SERIES_ID] = key[len("/triggers_"):]
                parts.append(part)
    triggers = pd.concat(parts, ignore_index=True).sort_values([TRIGGER_BEGIN_DATES, SERIES_ID], kind="mergesort")
    with pd.HDFStore(mode_folder / "signal" / "0.h5", mode="r") as store:
        persisted = store.get("/signals")
    return series, triggers.reset_index(drop=True), persisted


# =============================================================================
# REPLAY
# =============================================================================


def replay(series: dict, triggers: pd.DataFrame, params: dict, precedence: str, passing: Optional[list] = None) -> pd.DataFrame:
    """The documented grouping rule; precedence selects the eligible-signal sequence (code, oldest-first, point-in-time,
    point-in-time with oldest-first insertion); passing, when given, receives each trigger's count of eligible signals
    passing the threshold."""
    period_back, threshold = params["period_back"], params["corr_threshold"]
    trimmed = trim(series, triggers, period_back)
    rows = triggers.copy()
    slopes = [window_slope(trimmed.get(sid), begin, period_back)
              for sid, begin in zip(rows[SERIES_ID], rows[TRIGGER_BEGIN_DATES])]
    active, groups, corrs, signal_row, updated, counter = [], [0] * len(rows), [np.nan] * len(rows), {}, set(), 0
    for idx, (sid, begin, end) in enumerate(zip(rows[SERIES_ID], rows[TRIGGER_BEGIN_DATES], rows[TRIGGER_END_DATES])):
        active = [signal for signal in active if signal.end > begin]
        eligible = [signal for signal in _by_precedence(active, precedence) if signal.begin <= begin < signal.end]
        passed = [(signal, r) for signal in eligible for r in [pearson(trimmed, signal.series_id, sid, begin, period_back)]
                  if r is not None and abs(r) >= threshold]
        if passing is not None:
            passing.append(len(passed))
        joined = passed[0] if passed else None
        if joined is None:
            counter += 1
            signal = ActiveSignal(counter, begin, end, sid, slopes[idx] if np.isfinite(slopes[idx]) else np.nan)
            _insert(active, signal, precedence)
            signal_row[counter], groups[idx], corrs[idx] = idx, counter, 1.0
            continue
        signal, r = joined
        groups[idx], corrs[idx] = signal.group_id, r
        if params["signal_criterion"] == "earliest_begin_highest_slope" and _replaces(signal, sid, begin, slopes[idx]):
            old_end = signal.end
            signal.series_id, signal.slope, signal.end = sid, slopes[idx], end
            signal_row[signal.group_id] = idx
            updated.add(signal.group_id)
            if end != old_end:
                active.remove(signal)
                _insert(active, signal, precedence)
    rows[GROUP_ID], rows[SLOPE], rows[CORR_TO_SIGNAL] = groups, slopes, corrs
    rows[IS_SIGNAL] = [idx in set(signal_row.values()) for idx in range(len(rows))]
    for gid in sorted(updated):
        signal_sid = rows.at[signal_row[gid], SERIES_ID]
        for idx in rows.index[rows[GROUP_ID] == gid]:
            member = rows.at[idx, SERIES_ID]
            rows.at[idx, CORR_TO_SIGNAL] = 1.0 if member == signal_sid else pearson(
                trimmed, signal_sid, member, rows.at[idx, TRIGGER_BEGIN_DATES], period_back)
    return aggregates(rows)


def _by_precedence(active: list, precedence: str) -> list:
    """Eligible-signal sequence: the active list's own (by end) or by (signal_begin, group_id)."""
    point_in_time = precedence in (PRECEDENCE_BEGIN, PRECEDENCE_BEGIN_OLDEST)
    return sorted(active, key=lambda signal: (signal.begin, signal.group_id)) if point_in_time else active


def _insert(active: list, signal: ActiveSignal, precedence: str) -> None:
    """Keep the list sorted by end: a new end goes before equal ends (code) or after them (oldest-first)."""
    ends = [item.end for item in active]
    oldest_first = precedence in (PRECEDENCE_OLDEST, PRECEDENCE_BEGIN_OLDEST)
    slot = bisect.bisect_right(ends, signal.end) if oldest_first else bisect.bisect_left(ends, signal.end)
    active.insert(slot, signal)


def _replaces(signal: ActiveSignal, sid: str, begin: pd.Timestamp, slope: float) -> bool:
    """Same begin date and a higher slope (ties: the smaller series id) replace the signal."""
    if not np.isfinite(slope) or begin != signal.begin:
        return False
    return np.isnan(signal.slope) or slope > signal.slope or (slope == signal.slope and sid < signal.series_id)


def trim(series: dict, triggers: pd.DataFrame, period_back: int) -> dict:
    """Every series cut to [earliest begin - BDay(period_back), latest end]; empty series dropped."""
    start = triggers[TRIGGER_BEGIN_DATES].min() - pd.offsets.BDay(period_back)
    end = triggers[TRIGGER_END_DATES].max()
    cut = {sid: points[(points.index >= start) & (points.index <= end)] for sid, points in series.items()}
    return {sid: points for sid, points in cut.items() if len(points)}


def window(values: pd.Series, begin: pd.Timestamp, period_back: int):
    """Row indices of the period_back + 1 points ending at the last index <= begin, None below two points."""
    end = int(np.searchsorted(values.index.values, np.datetime64(begin, "ns"), side="right")) - 1
    start = max(0, end - period_back)
    return (start, end) if end >= 0 and end - start + 1 >= 2 else None


def window_slope(values, begin: pd.Timestamp, period_back: int) -> float:
    """|slope| of y / y_last - 1 over the window (np.polyfit degree 1), NaN when not computable."""
    bounds = None if values is None else window(values, begin, period_back)
    if bounds is None:
        return np.nan
    y = values.to_numpy()[bounds[0]:bounds[1] + 1]
    finite = np.isfinite(y)
    if not np.isfinite(y[-1]) or finite.sum() < 2:
        return np.nan
    slope = np.polyfit(np.arange(len(y), dtype=float)[finite], y[finite] / y[-1] - 1.0, 1)[0]
    return float(abs(slope)) if np.isfinite(slope) else np.nan


def pearson(trimmed: dict, signal_sid: str, member_sid: str, begin: pd.Timestamp, period_back: int):
    """Pearson r of var3 levels over the signal's window, on the dates both series hold."""
    signal, member = trimmed.get(signal_sid), trimmed.get(member_sid)
    bounds = None if signal is None or member is None else window(signal, begin, period_back)
    if bounds is None:
        return None
    dates = signal.index[bounds[0]:bounds[1] + 1].intersection(member.index)
    a, b = signal.loc[dates].to_numpy(), member.loc[dates].to_numpy()
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 2:
        return None
    r = np.corrcoef(a[finite], b[finite])[0, 1]
    return None if np.isnan(r) else float(r)


def aggregates(rows: pd.DataFrame) -> pd.DataFrame:
    """num_series and series_ids per group, series sorted by decreasing maximum slope."""
    members = {}
    for gid, group in rows.groupby(GROUP_ID):
        maxima = group.groupby(SERIES_ID)[SLOPE].max().sort_values(ascending=False, kind="mergesort")
        ranked = [sid for sid in maxima.index if pd.notna(maxima[sid])]
        members[gid] = ranked + [sid for sid in pd.unique(group[SERIES_ID]) if sid not in ranked]
    rows = rows.assign(**{NUM_SERIES: rows[GROUP_ID].map(lambda gid: len(members[gid])).astype(int),
                          SERIES_IDS: rows[GROUP_ID].map(lambda gid: ",".join(members[gid]))})
    return rows.sort_values([GROUP_ID, TRIGGER_BEGIN_DATES, SERIES_ID]).reset_index(drop=True)


# =============================================================================
# COMPARISONS AND INVARIANTS
# =============================================================================


def compare_rows(mode: str, persisted: pd.DataFrame, replayed: pd.DataFrame) -> dict:
    """The row comparison with its non-vacuity and equality verdicts."""
    found = row_differences(persisted, replayed)
    CTX.log.info("%s: row comparison %s", mode, json.dumps(found, default=str))
    CTX.verdict(f"{mode}: non-vacuity", found["matched"] > 0 and found["matched"] == found["rows"], "every row matched",
                {k: found[k] for k in ("rows", "matched", "only_persisted", "only_replay")})
    CTX.verdict(f"{mode}: persisted grouping equals the documented replay", not found["difference_count"], "no difference",
                {"count": found["difference_count"], "first": found["differences"][:5]})
    return found


def row_differences(persisted: pd.DataFrame, replayed: pd.DataFrame) -> dict:
    """Per row, keyed by (series_id, begin): the replay against the persisted /signals."""
    left = persisted.set_index([SERIES_ID, TRIGGER_BEGIN_DATES])
    right = replayed.set_index([SERIES_ID, TRIGGER_BEGIN_DATES])
    common = left.index.intersection(right.index)
    differences = []
    for key in common:
        a, b = left.loc[key], right.loc[key]
        for column in (GROUP_ID, IS_SIGNAL, NUM_SERIES, SERIES_IDS):
            if str(a[column]) != str(b[column]):
                differences.append((key[0], str(key[1].date()), column, str(a[column]), str(b[column])))
        for column in (CORR_TO_SIGNAL, SLOPE):
            if not _near(a[column], b[column]):
                differences.append((key[0], str(key[1].date()), column, float(a[column]), float(b[column])))
    return {"rows": len(persisted), "matched": len(common), "only_persisted": len(left.index.difference(right.index)),
            "only_replay": len(right.index.difference(left.index)), "differences": differences[:20],
            "difference_count": len(differences), "groups": int(persisted[GROUP_ID].nunique())}


def _near(a: float, b: float) -> bool:
    """Equal within float32 storage, NaN equal to NaN."""
    if pd.isna(a) or pd.isna(b):
        return pd.isna(a) and pd.isna(b)
    return abs(float(a) - float(b)) <= FLOAT32_TOLERANCE * max(1.0, abs(float(b)))


def partition_delta(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """Groups compared by member sets and signals by (series_id, begin), ids relabelled apart."""
    groups = [_members(frame) for frame in (left, right)]
    signals = [set(zip(frame.loc[frame[IS_SIGNAL], SERIES_ID], frame.loc[frame[IS_SIGNAL], TRIGGER_BEGIN_DATES].astype(str)))
               for frame in (left, right)]
    found = {"groups_only_left": len(groups[0] - groups[1]), "groups_only_right": len(groups[1] - groups[0]),
             "signals_only_left": sorted(signals[0] - signals[1])[:10], "signals_only_right": sorted(signals[1] - signals[0])[:10],
             "signal_rows_changed": len(signals[0] ^ signals[1])}
    found["equal"] = not found["groups_only_left"] and not found["groups_only_right"] and not found["signal_rows_changed"]
    return found


def precedence_verdicts(mode: str, series: dict, triggers: pd.DataFrame, params: dict, persisted: pd.DataFrame,
                        replayed: pd.DataFrame) -> dict:
    """The persisted partition against the decided replay of the mode (Q15, Q20), each alternative's delta logged as the
    verdict's actual value."""
    decided = partition_delta(persisted, replayed)
    if mode == "realistic":
        found = {"end_precedence": partition_delta(replayed, replay(series, triggers, params, PRECEDENCE_CODE)),
                 "insertion": partition_delta(replayed, replay(series, triggers, params, PRECEDENCE_BEGIN_OLDEST))}
        CTX.verdict("realistic: no ties under (signal_begin, group_id): the point-in-time replay is independent of the "
                    "equal-end insertion rule (Q15)", found["insertion"]["equal"], "equal partition", found["insertion"])
        CTX.verdict("realistic: the persisted partition equals the point-in-time replay (Q20); the end precedence's delta "
                    "logged", decided["equal"], "equal partition",
                    {"persisted_vs_decided": decided, "end_precedence": found["end_precedence"]})
    else:
        found = {"oldest_first": partition_delta(replayed, replay(series, triggers, params, PRECEDENCE_OLDEST)),
                 "point_in_time": partition_delta(replayed, replay(series, triggers, params, PRECEDENCE_BEGIN))}
        CTX.verdict("ideal: the persisted partition equals the end-precedence, newest-first replay (Q15 as coded); the "
                    "oldest-first delta logged", decided["equal"], "equal partition",
                    {"persisted_vs_decided": decided, "oldest_first": found["oldest_first"]})
        CTX.verdict("ideal: the persisted partition equals the end-precedence replay (Q20 keeps it in ideal mode); the "
                    "point-in-time delta logged", decided["equal"], "equal partition",
                    {"persisted_vs_decided": decided, "point_in_time": found["point_in_time"]})
    CTX.log.info("%s: decided precedence %s, persisted delta %s, alternatives %s", mode, DECIDED[mode],
                 json.dumps(decided, default=str), json.dumps(found, default=str))
    return {"decided": decided, **found}


def _members(frame: pd.DataFrame) -> set:
    """Each group as the frozen set of its (series_id, begin) members."""
    return {frozenset(zip(group[SERIES_ID], group[TRIGGER_BEGIN_DATES].astype(str))) for _, group in frame.groupby(GROUP_ID)}


def precedence_sensitive(series: dict, triggers: pd.DataFrame, params: dict) -> int:
    """Non-vacuity of the variant: triggers whose eligible final signals the two keys rank differently."""
    replayed = replay(series, triggers, params, PRECEDENCE_CODE)
    signals = replayed[replayed[IS_SIGNAL]]
    count = 0
    for begin in triggers[TRIGGER_BEGIN_DATES]:
        eligible = signals[(signals[TRIGGER_BEGIN_DATES] <= begin) & (signals[TRIGGER_END_DATES] > begin)]
        by_end = list(eligible.sort_values([TRIGGER_END_DATES, GROUP_ID], ascending=[True, False])[GROUP_ID])
        by_begin = list(eligible.sort_values([TRIGGER_BEGIN_DATES, GROUP_ID])[GROUP_ID])
        count += int(len(eligible) > 1 and by_end != by_begin)
    return count


def invariants(mode: str, persisted: pd.DataFrame, series: dict, params: dict) -> dict:
    """One signal per group, signal corr 1.0, consistent aggregates, replaced signals and their members' r."""
    per_group = persisted.groupby(GROUP_ID)
    signals_per_group = per_group[IS_SIGNAL].sum()
    signal_rows = persisted[persisted[IS_SIGNAL]]
    consistent = all(int(group[NUM_SERIES].iloc[0]) == group[SERIES_ID].nunique()
                     and set(str(group[SERIES_IDS].iloc[0]).split(",")) == set(group[SERIES_ID]) for _, group in per_group)
    replaced = sorted(int(gid) for gid, group in per_group
                      if group.loc[group[IS_SIGNAL], TRIGGER_BEGIN_DATES].min() > group[TRIGGER_BEGIN_DATES].min()
                      or not group[IS_SIGNAL].iloc[group[TRIGGER_BEGIN_DATES].argmin()])
    below = [(int(gid), str(sid), float(r)) for gid, sid, r in
             zip(persisted[GROUP_ID], persisted[SERIES_ID], persisted[CORR_TO_SIGNAL])
             if pd.notna(r) and abs(r) < params["corr_threshold"]]
    found = {"one_signal_per_group": bool((signals_per_group == 1).all()),
             "signal_corr_one": bool(np.allclose(signal_rows[CORR_TO_SIGNAL].astype(float), 1.0)),
             "aggregates_consistent": consistent, "groups_with_replaced_signal": replaced[:20],
             "members_below_threshold": below[:20], "members_without_corr": int(persisted[CORR_TO_SIGNAL].isna().sum())}
    CTX.log.info("%s: invariants %s", mode, json.dumps(found, default=str))
    CTX.verdict(f"{mode}: final-state invariants (one signal per group, signal r = 1, aggregates)",
                found["one_signal_per_group"] and found["signal_corr_one"] and found["aggregates_consistent"], "all hold", found)
    return found


def numpy_sample(mode: str, persisted: pd.DataFrame, series: dict, params: dict) -> dict:
    """Independent numpy Pearson and slope on every 10th row, the first and last rows and the extreme r rows."""
    ranked = persisted.sort_values([GROUP_ID, TRIGGER_BEGIN_DATES, SERIES_ID]).reset_index(drop=True)
    sampled_rows = set(range(0, len(ranked), SAMPLE_STEP)) | {0, len(ranked) - 1,
                                                            int(ranked[CORR_TO_SIGNAL].astype(float).idxmin()),
                                                            int(ranked[CORR_TO_SIGNAL].astype(float).idxmax())}
    triggers = ranked[[SERIES_ID, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES]]
    trimmed = trim(series, triggers, params["period_back"])
    signal_of = {int(gid): sid for gid, sid in zip(ranked.loc[ranked[IS_SIGNAL], GROUP_ID], ranked.loc[ranked[IS_SIGNAL], SERIES_ID])}
    misses = []
    for row_number in sorted(sampled_rows):
        row = ranked.iloc[row_number]
        sid, begin = row[SERIES_ID], row[TRIGGER_BEGIN_DATES]
        slope = window_slope(trimmed.get(sid), begin, params["period_back"])
        signal_sid = signal_of[int(row[GROUP_ID])]
        r = 1.0 if signal_sid == sid else pearson(trimmed, signal_sid, sid, begin, params["period_back"])
        if not _near(row[SLOPE], slope) or not _near(row[CORR_TO_SIGNAL], np.nan if r is None else r):
            misses.append((row_number, sid, str(begin.date()), float(row[SLOPE]), slope, float(row[CORR_TO_SIGNAL]), r))
    CTX.verdict(f"{mode}: numpy Pearson and polyfit sample ({len(sampled_rows)} rows)", bool(sampled_rows) and not misses,
                "all within float32", misses[:5])
    return {"sampled": len(sampled_rows), "misses": misses[:10]}


def write_variant(path: Path, variant: pd.DataFrame) -> dict:
    """The point-in-time variant table for EXP-09, with its MD5."""
    CTX.guard(path)
    if path.exists():
        harness.refuse(CTX.log, "variant", f"{path} already exists")
    table = variant.copy()
    table[SERIES_IDS] = table[SERIES_IDS].astype(str)
    with pd.HDFStore(str(path), mode="w") as store:
        store.put("/signals", table, format="fixed")
    found = {"path": str(path), "md5": harness.file_md5(path), "rows": len(table), "signals": int(table[IS_SIGNAL].sum())}
    CTX.log.info("variant table %s", json.dumps(found))
    return found


# =============================================================================
# A4(B) POINT-IN-TIME INVARIANCE
# =============================================================================


def check_point_in_time_invariance(run_copy: Path, config: AppConfig) -> dict:
    """A4(b): production SignalSelection on the realistic trigger tables with this root's realistic signal parameters.

    G0 is the unmodified build. For every trigger t, M(t) holds the triggers processed before t whose [begin, end)
    covers t's begin and that are a group signal in G0 or share their group signal's begin date (the same-begin
    replacement); for k in END_SHIFTS their ends move to latest_end + BDay(k + reversed rank), distinct, past every end
    and reversing their original end sequence. The prefix up to t is rebuilt; the group_id of every trigger up to and
    including t must equal G0's (a moved end keeps a signal eligible for later triggers, so later groups are not
    compared). A trigger with an empty M(t) needs no rebuild: its prefix is unchanged, which the control checks.
    """
    params = signal_params(config, "realistic")
    series, triggers, _ = load_inputs(run_copy / PARAMS / "realistic")
    groups = build_groups(params, series, triggers)
    keys = list(zip(triggers[SERIES_ID], triggers[TRIGGER_BEGIN_DATES]))
    g0 = [int(groups.at[key, GROUP_ID]) for key in keys]
    signal_rows = groups[groups[IS_SIGNAL]]
    signal_begin = {int(gid): begin for gid, begin in zip(signal_rows[GROUP_ID],
                                                          signal_rows.index.get_level_values(TRIGGER_BEGIN_DATES))}
    begins, ends = triggers[TRIGGER_BEGIN_DATES].to_numpy(), triggers[TRIGGER_END_DATES].to_numpy()
    movable = np.array([bool(groups.at[key, IS_SIGNAL]) or begin == signal_begin[gid]
                        for key, begin, gid in zip(keys, triggers[TRIGGER_BEGIN_DATES], g0)])
    latest_end = triggers[TRIGGER_END_DATES].max()
    differing, rebuilt = [], 0
    for t in range(len(triggers)):
        moved = [u for u in np.flatnonzero(movable[:t]) if begins[u] <= begins[t] < ends[u]]
        if not moved:
            continue
        sequence = sorted(moved, key=lambda u: (ends[u], u))
        for shift in END_SHIFTS:
            prefix = triggers.iloc[:t + 1].copy()
            for rank, u in enumerate(sequence):
                prefix.iat[u, prefix.columns.get_loc(TRIGGER_END_DATES)] = latest_end + pd.offsets.BDay(shift + len(sequence) - 1 - rank)
            moved_groups = build_groups(params, series, prefix)
            changed = [(keys[u][0], str(pd.Timestamp(keys[u][1]).date()), g0[u], int(moved_groups.at[keys[u], GROUP_ID]))
                       for u in range(t + 1) if int(moved_groups.at[keys[u], GROUP_ID]) != g0[u]]
            rebuilt += 1
            if changed:
                differing.append({"trigger": t, "k": shift, "moved": len(moved), "changed": changed[:4]})
    passing: list = []
    replay(series, triggers, params, PRECEDENCE_BEGIN if params.get(POINT_IN_TIME_KEY) else PRECEDENCE_CODE, passing)
    found = {"triggers": len(triggers), "rebuilds": rebuilt, "differing": len(differing), "first": differing[:5],
             "two_or_more_passing": int(sum(count >= 2 for count in passing)), "point_in_time": params.get(POINT_IN_TIME_KEY)}
    CTX.log.info("A4(b) invariance: %s", json.dumps(found, default=str))
    CTX.verdict("A4(b) non-vacuity: realistic triggers with >= 2 eligible signals passing corr_threshold in G0",
                found["two_or_more_passing"] > 0, "> 0", found["two_or_more_passing"])
    CTX.verdict(f"A4(b) point-in-time invariance (Q20): moving the ends of every signal active at a trigger's begin past "
                f"every end, reversed (k = {END_SHIFTS}), leaves the group of that trigger and of every earlier one",
                rebuilt > 0 and not differing, "no group_id changed", {key: found[key] for key in ("rebuilds", "differing", "first")})
    check_prefix_control(params, series, triggers, g0, keys)
    return found


def build_groups(params: dict, series: dict, triggers: pd.DataFrame) -> pd.DataFrame:
    """Production SignalSelection over the triggers, indexed by (series_id, trigger_begin_dates)."""
    selection = selection_of(params)
    for sid, values in sorted(series.items()):
        selection.add_series_values(sid, values.to_frame(VAR3))
    selection.add_series_triggers("all", triggers)
    return selection.build_signal_window_groups().set_index([SERIES_ID, TRIGGER_BEGIN_DATES])


def check_prefix_control(params: dict, series: dict, triggers: pd.DataFrame, g0: list, keys: list) -> None:
    """Unmodified prefixes up to evenly spaced triggers give G0's group_ids (the skip of empty M(t) rests on it)."""
    cuts = sorted({int(cut) for cut in np.linspace(0, len(triggers) - 1, CONTROL_PREFIXES)})
    changed = {}
    for t in cuts:
        prefix_groups = build_groups(params, series, triggers.iloc[:t + 1])
        changed[t] = [u for u in range(t + 1) if int(prefix_groups.at[keys[u], GROUP_ID]) != g0[u]]
    CTX.verdict("A4(b) control: unmodified prefixes reproduce G0's group_ids", not any(changed.values()),
                "no difference", {t: rows[:4] for t, rows in changed.items()})


# =============================================================================
# HYP-19 CACHE CASE
# =============================================================================


def check_cache_orientation(params: dict) -> dict:
    """Two series on differing indexes: fresh r per orientation differs; a cached value overrides the reverse one."""
    dates = pd.bdate_range("2021-01-04", periods=20)
    a = pd.DataFrame({VAR3: 100.0 + np.arange(20) ** 1.5}, index=dates)
    b = pd.DataFrame({VAR3: 50.0 + np.sin(np.arange(18))}, index=dates.delete([15, 17]))
    begin = dates[-1]
    begins = np.array([begin.to_datetime64()], dtype="datetime64[ns]")
    pb = params["period_back"]

    def fresh(first: str, second: str) -> tuple:
        selection = selection_of(params)
        selection.add_series_values("A", a)
        selection.add_series_values("B", b)
        bounds = selection._precompute_window_bounds(["A", "B"], begins, pb)
        return selection, bounds, selection._lookup_or_compute_correlation(first, second, begin, pb, bounds)

    selection, bounds, r_ab = fresh("A", "B")
    _, _, r_ba = fresh("B", "A")
    found = {"r_ab_fresh": r_ab, "r_ba_fresh": r_ba}
    runnable = r_ab is not None and r_ba is not None and not np.isclose(r_ab, r_ba)
    CTX.verdict("HYP-19: fresh orientations differ on differing indexes (the case can run)", runnable,
                "two different non-None values", found)
    if runnable:
        found["r_ba_after_ab"] = selection._lookup_or_compute_correlation("B", "A", begin, pb, bounds)
        CTX.verdict("HYP-19: the reverse orientation computed on the same instance equals its fresh value",
                    np.isclose(found["r_ba_after_ab"], r_ba), r_ba, found["r_ba_after_ab"])
    else:
        CTX.log.info("HYP-19 case could not run: %s", found)
    return found


# =============================================================================
# CONTROLS
# =============================================================================


def check_controls(run_copy: Path, params: dict) -> None:
    """A changed persisted group_id is flagged; a precedence-sensitive case changes its assignment."""
    series, triggers, persisted = load_inputs(run_copy / PARAMS / "ideal")
    planted = persisted.copy()
    planted.loc[planted.index[0], GROUP_ID] = int(planted[GROUP_ID].max()) + 1
    found = row_differences(planted, replay(series, triggers, params, PRECEDENCE_CODE))
    CTX.verdict("control: a changed persisted group_id is flagged", any(item[2] == GROUP_ID for item in found["differences"]),
                "flagged", found["differences"][:2])
    dates = pd.bdate_range("2022-01-03", periods=30)
    rising = np.linspace(100.0, 130.0, 30)
    dip = 100.0 + np.concatenate([[0.0, -10.0, 0.0], np.arange(1.0, 28.0)])
    sensitive = {"X": pd.Series(rising, index=dates), "Y": pd.Series(dip, index=dates),
                 "Z": pd.Series(dip + 0.01 * np.sin(np.arange(30)), index=dates)}
    rows = pd.DataFrame({SERIES_ID: ["X", "Y", "Z"], TRIGGER_BEGIN_DATES: [dates[0], dates[2], dates[5]],
                         TRIGGER_END_DATES: [dates[25], dates[10], dates[20]], IS_PLUS_TRIGGER: True, IS_TRUE: True})
    loose = dict(params, corr_threshold=0.5)
    by_end = replay(sensitive, rows, loose, PRECEDENCE_CODE)
    by_begin = replay(sensitive, rows, loose, PRECEDENCE_BEGIN)
    z_end = int(by_end.loc[by_end[SERIES_ID] == "Z", GROUP_ID].iloc[0])
    z_begin = int(by_begin.loc[by_begin[SERIES_ID] == "Z", GROUP_ID].iloc[0])
    CTX.verdict("control: a precedence-sensitive case changes its assignment under the point-in-time key", z_end != z_begin,
                "different group for Z", {"by_end": z_end, "by_begin": z_begin})


# =============================================================================
# SELECTION PARAMETERS
# =============================================================================


def signal_params(config: AppConfig, replay_mode: str) -> dict:
    """The selection parameters the replay reads: the signal section, and the point-in-time rule SignalSelection
    derives from the replay mode (true for realistic)."""
    return {**{key: getattr(config.signal, key) for key in SIGNAL_KEYS}, POINT_IN_TIME_KEY: replay_mode == "realistic"}


def selection_of(params: dict) -> SignalSelection:
    """Production SignalSelection on the replay's parameters, the point-in-time rule given as the replay mode."""
    signal = SignalConfig(**{key: params[key] for key in SIGNAL_KEYS})
    return SignalSelection(signal, "realistic" if params[POINT_IN_TIME_KEY] else "ideal")


if __name__ == "__main__":
    main()
