"""EXP-09: the gradient recomputed from the documented formulas (R-P2..P6, R-P8, R-P10, HYP-26, HYP-31).

An explicit oracle, written from performance.md and metrics.md, recomputes from the
performance /signals and the signal file's /series_<id> trigger_value, on the curve index of the
observed dates (the trigger file's series and the references, W5) from the earliest effective start
index[lookback - 1] over the trigger file's series (Q2) to end_date:
the signal segments, the weight filter (its own sweep), compounding with attenuation and deduction,
the terminal attenuations, the reference curves (from the persisted /reference_series_*, trimmed to
the index start), every gradient metric and the signal breakdown, in both modes. Comparisons use the code's readings
(ends before begins on the same date, a zero-length segment's end after its day's begins (Q4),
per-direction trials_to_true counted in (trigger_begin_dates, series_id) sequence (Q8), the
last-year attenuation applied to the curve at its last index date and subtracted once for the
strategy's post-attenuation metrics (Q17), the breakdown over the signals
that build the curve (Q21)); the alternatives (Q4 begins first, Q8, Q17 and Q21) run separately with their
deltas; the oracle reruns on EXP-08's point-in-time variant (HYP-26).
In-process sweep: the production builders are called with each variation_method, a non-zero
boundary_deduction_pct and a non-zero mpv, and compared with the oracle. Planted control: a
perturbed persisted value must be flagged.

Structure:
1. main
2. Inputs
3. Oracle: segments, filter, compounding, attenuation, references
4. Oracle: metrics and breakdown
5. Comparisons
6. Variants and sweep
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pipelinekit
from src.utils.common.common import tall_to_wide, variation_pct
from src.utils.common.constants import (
    GROUP_ID, IS_PLUS_TRIGGER, IS_SIGNAL, IS_TRUE, SERIES_ID, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES,
    TRIGGER_END_VALUES, TRIGGER_START_VALUES, TRIGGER_VALUE, VAR3,
)
from src.utils.common.schema import AppConfig
from src.utils.performance.gradient import (
    build_actual_variation_pair, build_global_index, build_signal_segments_cumulative_pct_with_maxweight_constraint,
)
from src.utils.performance.gradient_.metrics import build_gradient_metrics_dataframe, compute_gradient_columns

EXPERIMENT = "09"
MODES = ("ideal", "realistic")
METHODS = ("multiplicative_fixed", "multiplicative_total_with_active_signals",
           "multiplicative_total_before_active_signals")
SWEEP_DEDUCTION_PCT = 0.1
SWEEP_MPV = 0.0002
RTOL, ATOL = 1e-5, 1e-6
SAME_BEGIN_FIRST, SAME_END_FIRST = "begin-first", "end-first"
STRATEGY, REFERENCE = "strategy", "reference {ref_id}"
DIRECTIONS = {"plus+minus": None, "plus": True, "minus": False}
SUBDIVISIONS = {"true+false": None, "true": True, "false": False}
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)


@dataclasses.dataclass(frozen=True)
class Settings:
    """The gradient parameters of one evaluation."""

    weight: float
    max_total: float
    method: str
    attenuation_rate: float
    attenuation_month: int
    deduction_pct: float
    mpv: float
    end_date: str
    lookback: int
    max_references: int
    display_signals: tuple
    display_gradient: tuple


def main() -> None:
    """Both modes against the persisted tables; alternatives; the variant; the sweep; the control."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    settings = settings_of(config)
    CTX.log.info("settings %s", dataclasses.asdict(settings))
    results = {}
    for mode in MODES:
        inputs = load_inputs(args.run_copy / PARAMS / mode, settings)
        computed = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings,
                            SAME_END_FIRST)
        results[mode] = {"comparison": compare(mode, computed, inputs["persisted"]),
                         "alternatives": alternatives(mode, inputs, settings, computed)}
    results["variant"] = point_in_time_variant(args, settings)
    results["sweep"] = sweep(load_inputs(args.run_copy / PARAMS / "realistic", settings), settings)
    check_control(load_inputs(args.run_copy / PARAMS / "realistic", settings), settings)
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy, --variant (EXP-08's table), --variant-label and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--variant-label", required=True)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


def settings_of(config: AppConfig) -> Settings:
    """The gradient settings the Performance task reads (display_order_gradient's schema default is metrics.json's
    key sequence)."""
    gradient, metrics = config.performance.gradient, config.performance.metrics
    return Settings(gradient.weight, gradient.concurrent_weight_limit, gradient.variation_method,
                    gradient.yearly_attenuation_fraction, gradient.attenuation_month, gradient.boundary_deduction_pct, gradient.mpv,
                    config.read.end_date, config.trigger.replay.lookback_window, gradient.max_references,
                    tuple(metrics.display_order_signals), tuple(metrics.display_order_gradient))


# =============================================================================
# INPUTS
# =============================================================================


def load_inputs(mode_folder: Path, settings: Settings) -> dict:
    """Performance /signals and tables, the signal file's series and references, the trigger file's series, the
    earliest effective start over them (Q2), and the curve index: the observed dates of the trigger file's series
    and of the first max_references references (pipelinekit.observed_dates, W5) from that start to end_date."""
    trigger_file, signal_file = mode_folder / "trigger" / "0.h5", mode_folder / "signal" / "0.h5"
    persisted = {}
    with pd.HDFStore(mode_folder / "performance" / "0.h5", mode="r") as store:
        for key in sorted(store.keys()):
            persisted[key] = store.get(key)
    with pd.HDFStore(signal_file, mode="r") as store:
        series = {key[len(harness.SERIES_KEY_PREFIX):]: store.get(key) for key in sorted(store.keys())
                  if key.startswith(harness.SERIES_KEY_PREFIX)}
        references = {key[len(pipelinekit.REFERENCE_KEY_PREFIX):]: store.get(key) for key in sorted(store.keys())
                      if key.startswith(pipelinekit.REFERENCE_KEY_PREFIX)}
    with pd.HDFStore(trigger_file, mode="r") as store:
        trigger_series = {key[len(harness.SERIES_KEY_PREFIX):]: store.get(key) for key in sorted(store.keys())
                          if key.startswith(harness.SERIES_KEY_PREFIX)}
    start = min(frame.index[settings.lookback - 1] for frame in trigger_series.values())
    observed = pipelinekit.observed_dates(trigger_file, signal_file, settings.max_references)
    return {"signals": persisted["/signals"], "series": series, "references": references, "persisted": persisted,
            "trigger_series": trigger_series, "start": start, "observed": observed,
            "index": pipelinekit.observed_index(observed, start, settings.end_date)}


# =============================================================================
# ORACLE: SEGMENTS, FILTER, COMPOUNDING, ATTENUATION, REFERENCES
# =============================================================================


def evaluate(signals: pd.DataFrame, series: dict, references: dict, idx: pd.DatetimeIndex, settings: Settings,
             same_date: str, *, last_year_applied: bool = True, breakdown_rows: str = "filtered") -> dict:
    """Every quantity the Performance task persists, recomputed on the curve index idx."""
    signal_rows = signals[signals[IS_SIGNAL]].sort_values([TRIGGER_BEGIN_DATES, GROUP_ID], kind="mergesort")
    segs = segments(signal_rows, series, idx)
    kept = sweep_filter(segs, settings, same_date)
    pure, _ = compound(kept, idx, settings, attenuation=False)
    attenuated, _ = compound(kept, idx, settings, attenuation=True)
    terminal = last_year_attenuation(attenuated, idx, settings.attenuation_rate)
    if last_year_applied:
        attenuated = attenuated.copy()
        attenuated.iloc[-1] -= terminal
    refs = {ref_id: reference_curve(frame, idx) for ref_id, frame in sorted(references.items())}
    ref_terminal = {ref_id: settings.attenuation_rate * max(0.0, float(curve.iloc[-1])) for ref_id, curve in refs.items()}
    rows = signal_rows if breakdown_rows == "all" else signal_rows[signal_rows[GROUP_ID].isin(list(kept))]
    return {"idx": idx, "segments": segs, "kept": list(kept), "total": pure, "attenuated": attenuated, "terminal": terminal,
            "references": refs, "reference_terminal": ref_terminal, "breakdown_rows": rows,
            "metrics": gradient_metrics(pure, attenuated, 0.0 if last_year_applied else terminal, refs, ref_terminal, settings),
            "breakdown": breakdown(rows, series, settings.display_signals)}


def segments(signal_rows: pd.DataFrame, series: dict, idx: pd.DatetimeIndex) -> dict:
    """gid -> (pct on the global days of [begin, end], sign, begin, end, metadata) (performance.md:116-133)."""
    found = {}
    for _, row in signal_rows.iterrows():
        start = float(row[TRIGGER_START_VALUES])
        frame = series.get(str(row[SERIES_ID]))
        if pd.isna(start) or start == 0.0 or frame is None or TRIGGER_VALUE not in frame.columns:
            continue
        begin, end = pd.Timestamp(row[TRIGGER_BEGIN_DATES]), pd.Timestamp(row[TRIGGER_END_DATES])
        values = frame.loc[(frame.index >= begin) & (frame.index <= end), TRIGGER_VALUE]
        if values.empty:
            continue
        pct = variation_pct(values, start).reindex(idx[(idx >= begin) & (idx <= end)]).ffill().dropna()
        if pct.empty:
            continue
        pct.iloc[0] = 0.0
        found[int(row[GROUP_ID])] = {"pct": pct, "sign": 1 if bool(row[IS_PLUS_TRIGGER]) else -1, "row": row}
    return found


def sweep_filter(segs: dict, settings: Settings, same_date: str) -> dict:
    """Concurrent configured weight capped: on a violation the newest active signal goes (performance.md:135-145).

    The sort priority of each event: begin 0 and end 1 when begins come first; end 0 and begin 1 when ends come
    first (Q4), except a zero-length segment (begin = end), whose end gets 2, after its day's begins.
    """
    events = []
    zero_length = 0
    for rank, (gid, seg) in enumerate(segs.items()):
        begin, end = seg["pct"].index.min(), seg["pct"].index.max()
        zero_length += int(begin == end)
        begin_rank, end_rank = (0, 1) if same_date == SAME_BEGIN_FIRST else (1, 2 if begin == end else 0)
        events += [(begin, begin_rank, rank, gid, "begin"), (end, end_rank, rank, gid, "end")]
    CTX.log.info("sweep %s: %d segments, %d zero-length", same_date, len(segs), zero_length)
    active, removed = {}, set()
    for _, _, rank, gid, kind in sorted(events):
        if gid in removed:
            continue
        if kind == "end":
            active.pop(gid, None)
            continue
        active[gid] = rank
        if len(active) * settings.weight > settings.max_total:
            newest = max(active, key=active.get)
            del active[newest]
            removed.add(newest)
    return {gid: seg for gid, seg in segs.items() if gid not in removed}


def period_rates(pct: pd.Series) -> pd.Series:
    """r_t = (1 + pct_t/100) / (1 + pct_{t-1}/100) - 1, pct_{-1} = 0."""
    previous = pct.shift(1).fillna(0.0)
    return (1.0 + pct / 100.0) / (1.0 + previous / 100.0) - 1.0


def compound(kept: dict, idx: pd.DatetimeIndex, settings: Settings, *, attenuation: bool) -> tuple[pd.Series, float]:
    """Geometric per-event compounding, locked weights by method, yearly attenuations interleaved (performance.md:147-357)."""
    events = sorted(({"rates": period_rates(seg["pct"]).reindex(idx).fillna(0.0) * seg["sign"],
                      "start": seg["pct"].index[0], "end": seg["pct"].index[-1]} for seg in kept.values()),
                    key=lambda event: event["start"])
    schedule = attenuation_schedule(idx, settings.attenuation_month) if attenuation and settings.attenuation_rate > 0 else []
    total = pd.Series(0.0, index=idx)
    processed, attenuated_amount, pending = [], 0.0, 0
    for event in events:
        while pending < len(schedule) and schedule[pending]["date"] <= event["start"]:
            attenuated_amount += apply_attenuation(total, schedule[pending], settings.attenuation_rate)
            pending += 1
        cumulative = (1.0 + event["rates"]).cumprod() - 1.0
        if settings.deduction_pct > 0:
            keep = 1.0 - settings.deduction_pct / 100.0
            cumulative = (1.0 + cumulative) * keep - 1.0
            after = cumulative.index >= event["end"]
            cumulative[after] = (1.0 + cumulative[after]) * keep - 1.0
        locked = settings.weight * start_level(settings.method, event, idx, total, processed, attenuated_amount)
        total = total + locked * cumulative
        processed.append({"end": event["end"], "cumulative": cumulative, "locked": locked})
    while pending < len(schedule):
        attenuated_amount += apply_attenuation(total, schedule[pending], settings.attenuation_rate)
        pending += 1
    return total, attenuated_amount


def start_level(method: str, event: dict, idx: pd.DatetimeIndex, total: pd.Series, processed: list, attenuated: float) -> float:
    """V(start): 1; 1 + total at the start's own row; 1 + ended events - attenuations so far."""
    row = int(idx.searchsorted(event["start"], side="right")) - 1
    if method == METHODS[0] or row < 0:
        return 1.0
    if method == METHODS[1]:
        return 1.0 + float(total.iloc[row])
    return 1.0 + sum(item["locked"] * float(item["cumulative"].iloc[row]) for item in processed
                     if item["end"] < event["start"]) - attenuated


def attenuation_schedule(idx: pd.DatetimeIndex, month: int) -> list:
    """Every calendar year but the last: attenuated on the first index day on or after day 1 of month in the next year."""
    years = sorted(set(idx.year))
    found = []
    for year in years[:-1]:
        days = idx[idx.year == year]
        later = idx[idx >= pd.Timestamp(year + 1, month, 1)]
        found.append({"date": later[0] if len(later) else idx[-1], "start": days[0], "end": days[-1]})
    return found


def apply_attenuation(total: pd.Series, item: dict, rate: float) -> float:
    """rate x max(0, year increase) deducted from the attenuation date onward; the amount."""
    amount = rate * max(0.0, float(total[item["end"]]) - float(total[item["start"]]))
    total[total.index >= item["date"]] -= amount
    return amount


def last_year_attenuation(curve: pd.Series, idx: pd.DatetimeIndex, rate: float) -> float:
    """The last calendar year's attenuation: applied at the curve's last index date (Q17), or only marked (the audited revision)."""
    days = idx[idx.year == idx[-1].year]
    if rate <= 0 or len(days) < 2:
        return 0.0
    return rate * max(0.0, float(curve[days[-1]]) - float(curve[days[0]]))


def reference_curve(frame: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    """Single-series compounding of var3 from its first persisted row on the curve index (performance.md:298-300;
    the rows before the index start trimmed, Q2)."""
    values = pd.Series(frame[VAR3].to_numpy(dtype=float), index=pd.to_datetime(frame.index)).dropna()
    values = values[values.index >= idx[0]]
    rates = period_rates(variation_pct(values, float(values.iloc[0]))).reindex(idx).fillna(0.0)
    return (1.0 + rates).cumprod() - 1.0


# =============================================================================
# ORACLE: METRICS AND BREAKDOWN
# =============================================================================


def gradient_metrics(pure: pd.Series, attenuated: pd.Series, terminal: float, refs: dict, ref_terminal: dict,
                     settings: Settings) -> dict:
    """Column -> metric -> value (metrics.md); acr_after_attenuation from the attenuated curves with the terminal attenuations not yet
    in them: the strategy's when its curve lacks the last-year attenuation (terminal 0 once applied, Q17), the references'."""
    post = attenuated.copy()
    post.iloc[-1] -= terminal
    columns = {STRATEGY: dict(standalone(pure * 100.0, settings.mpv), acr_after_attenuation=acr(post * 100.0))}
    for ref_id, curve in refs.items():
        ref_post = curve.copy()
        ref_post.iloc[-1] -= ref_terminal[ref_id]
        values = dict(standalone(curve * 100.0, settings.mpv), acr_after_attenuation=acr(ref_post * 100.0))
        values.update(relative(pure * 100.0, curve * 100.0, settings.mpv))
        columns[REFERENCE.format(ref_id=ref_id)] = values
    return columns


def annualization(index: pd.DatetimeIndex) -> tuple[float, float]:
    """m = (N - 1) / n_years, n_years = days / 365.25."""
    years = (index[-1] - index[0]).days / 365.25
    return ((len(index) - 1) / years, years) if years >= 1e-6 else (1.0, 0.0)


def acr(cumulative_pct: pd.Series) -> float:
    """(1 + V)^(1 / n_years) - 1, in percent; -100 when V is at or below -100 %."""
    _, years = annualization(cumulative_pct.index)
    base = 1.0 + float(cumulative_pct.iloc[-1]) / 100.0
    if years < 1e-6:
        return 0.0
    return -100.0 if base <= 0 else (base ** (1.0 / years) - 1.0) * 100.0


def rates_of(cumulative_pct: np.ndarray) -> np.ndarray:
    """R_t = (1 + C_t/100) / (1 + C_{t-1}/100) - 1."""
    level = 1.0 + cumulative_pct / 100.0
    return level[1:] / level[:-1] - 1.0


def standalone(cumulative_pct: pd.Series, mpv: float) -> dict:
    """acr, std, std_dw, snr, snr_dw, max_fall, fall_dur, below_max_dur (metrics.md, standalone table)."""
    m, _ = annualization(cumulative_pct.index)
    r = rates_of(cumulative_pct.to_numpy(dtype=float))
    std = float(np.std(r, ddof=1))
    downside = math.sqrt(float(np.mean(np.minimum(r - mpv, 0.0) ** 2)))
    level = 1.0 + cumulative_pct.to_numpy(dtype=float) / 100.0
    running_max = np.maximum.accumulate(level)
    decline = (level - running_max) / running_max
    deepest = int(np.argmin(decline))
    start = deepest
    while start > 0 and level[start] < running_max[start]:
        start -= 1
    recovery = deepest
    while recovery < len(level) - 1 and level[recovery] < running_max[deepest]:
        recovery += 1
    runs, current = 0, 0
    for below in level < running_max:
        current = current + 1 if below else 0
        runs = max(runs, current)
    return {"acr": acr(cumulative_pct), "std": std * math.sqrt(m) * 100.0, "std_dw": downside * math.sqrt(m) * 100.0,
            "snr": (float(np.mean(r)) - mpv) / std * math.sqrt(m) if std > 1e-15 else np.nan,
            "snr_dw": (float(np.mean(r)) - mpv) / downside * math.sqrt(m) if downside > 1e-15 else np.nan,
            "max_fall": abs(float(decline.min())) * 100.0, "fall_dur": float(recovery - start), "below_max_dur": float(runs)}


def relative(strategy_pct: pd.Series, reference_pct: pd.Series, mpv: float) -> dict:
    """rho, sens, sens_dw, exv, exv_dw, ref_snr, ref_snr_dw on the common index (metrics.md, relative table)."""
    common = strategy_pct.index.intersection(reference_pct.index)
    rs, rr = rates_of(strategy_pct.loc[common].to_numpy(dtype=float)), rates_of(reference_pct.loc[common].to_numpy(dtype=float))
    m, _ = annualization(common)
    cov = np.cov(rs, rr, ddof=1)
    sens = cov[0, 1] / cov[1, 1]
    down = rr < mpv
    cov_dw = np.cov(rs[down], rr[down], ddof=1)
    sens_dw = cov_dw[0, 1] / cov_dw[1, 1]
    excess = rs - rr
    return {"rho": cov[0, 1] / math.sqrt(cov[0, 0] * cov[1, 1]), "sens": sens, "sens_dw": sens_dw,
            "exv": (np.mean(rs) - sens * np.mean(rr) - mpv * (1 - sens)) * m * 100.0,
            "exv_dw": (np.mean(rs[down]) - sens_dw * np.mean(rr[down]) - mpv * (1 - sens_dw)) * m * 100.0,
            "ref_snr": np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(m),
            "ref_snr_dw": np.mean(excess) / math.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * math.sqrt(m)}


def breakdown(rows: pd.DataFrame, series: dict, display: tuple, *, trials: str = "per-direction") -> dict:
    """(row key, column) -> value: signal counts, then each display metric per direction and subdivision."""
    found = {}
    for direction, plus in DIRECTIONS.items():
        subset = rows if plus is None else rows[rows[IS_PLUS_TRIGGER] == plus]
        subset = subset.sort_values(TRIGGER_BEGIN_DATES, kind="mergesort")
        outcome = subset[IS_TRUE].to_numpy(dtype=bool)
        variations = np.where(subset[IS_PLUS_TRIGGER], 1.0, -1.0) * variation_pct(
            subset[TRIGGER_END_VALUES].to_numpy(dtype=np.float32), subset[TRIGGER_START_VALUES].to_numpy(dtype=np.float32))
        waits = keyed_waits(subset) if trials == "per-direction" else subset["trials_to_true"].to_numpy(dtype=float)
        columns = {"variation": variations, "duration": durations(subset, series), "trials_to_true": waits}
        for name, keep in SUBDIVISIONS.items():
            mask = np.ones(len(subset), bool) if keep is None else outcome == keep
            found[("Signals", f"{direction}_{name}")] = float(mask.sum())
            for key in display:
                metric, operation = composite_keys()[key]
                values = columns[metric][mask if not (operation == "avgt" and keep is None) else outcome]
                found[(key, f"{direction}_{name}")] = aggregate(operation, values)
    return found


def composite_keys() -> dict:
    """Composite display key -> (metric, operation), from the perseries metric specification (metrics.md:25-40)."""
    spec = json.loads((CTX.source_root / "src" / "utils" / "performance" / "perseries_" / "metrics.json").read_text(encoding="utf-8"))
    return {f"{metric}_{operation}": (metric, operation) for metric, item in spec.items() for operation in item["operations"]}


def durations(rows: pd.DataFrame, series: dict) -> np.ndarray:
    """Row-index difference of each trigger's end and begin in its own series index (metrics.md:35)."""
    return np.array([series[str(sid)].index.get_loc(end) - series[str(sid)].index.get_loc(begin)
                     for sid, begin, end in zip(rows[SERIES_ID], rows[TRIGGER_BEGIN_DATES], rows[TRIGGER_END_DATES])],
                    dtype=float)


def keyed_waits(subset: pd.DataFrame) -> np.ndarray:
    """wait_counts in (trigger_begin_dates, series_id) sequence, the tie key of Q8, aligned to the subset's rows."""
    sequence = subset.reset_index(drop=True).sort_values([TRIGGER_BEGIN_DATES, SERIES_ID], kind="mergesort").index.to_numpy()
    found = np.zeros(len(subset))
    found[sequence] = wait_counts(subset[IS_TRUE].to_numpy(dtype=bool)[sequence])
    return found


def wait_counts(outcome: np.ndarray) -> np.ndarray:
    """Trials to each true event (including it) since the previous true one; 0 on false rows."""
    found, count = np.zeros(len(outcome)), 1
    for row, value in enumerate(outcome):
        found[row], count = (count, 1) if value else (0.0, count + 1)
    return found


def aggregate(operation: str, values: np.ndarray) -> float:
    """avg, avgt, std (ddof 1, 0 below two values), std_dw (RMS of negatives), mult (compounded)."""
    if len(values) == 0:
        return 0.0
    if operation in ("avg", "avgt"):
        return float(np.mean(values))
    if operation == "std":
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if operation == "std_dw":
        return float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    return float((np.prod(1.0 + values / 100.0) - 1.0) * 100.0)


# =============================================================================
# COMPARISONS
# =============================================================================


def compare(mode: str, computed: dict, persisted: dict) -> dict:
    """Every persisted gradient table against the oracle (float32 storage tolerance)."""
    found = {}
    total = persisted["/total_variation"].set_index("date")["total_variation"]
    found["total_variation"] = _series_check(f"{mode}: /total_variation", computed["total"], total)
    attenuated = persisted["/total_variation_attenuated"].set_index("date")["total_variation_attenuated"]
    found["total_variation_attenuated"] = _series_check(f"{mode}: /total_variation_attenuated", computed["attenuated"], attenuated)
    segs = persisted["/gradient_signal_segments"]
    found["segments"] = _check(f"{mode}: /gradient_signal_segments", all(
        _near_series(computed["segments"][int(gid)]["pct"], group.set_index("date")["pct"])
        for gid, group in segs.groupby("group_id")) and set(segs["group_id"]) == set(computed["kept"]),
        "every kept segment equal", {"persisted": int(segs["group_id"].nunique()), "kept": len(computed["kept"])})
    sequence = persisted["/gradient_filtered_sequence"].sort_values("sequence_index")["group_id"].tolist()
    found["filtered_sequence"] = _check(f"{mode}: /gradient_filtered_sequence", sequence == computed["kept"],
                                        computed["kept"][:5], {"persisted": sequence[:5], "count": len(sequence)})
    refs = persisted["/gradient_reference_variation"]
    found["references"] = _check(f"{mode}: /gradient_reference_variation", all(
        _near_series(computed["references"][str(ref_id)], group.set_index("date")["variation"])
        for ref_id, group in refs.groupby("reference_id")), "equal", sorted(set(refs["reference_id"])))
    attenuations = dict(zip(persisted["/gradient_terminal_attenuations"]["series_id"], persisted["/gradient_terminal_attenuations"]["amount"]))
    expected_attenuations = {"__strategy__": computed["terminal"], **computed["reference_terminal"]}
    found["terminal_attenuations"] = _check(f"{mode}: /gradient_terminal_attenuations", all(
        _near(expected_attenuations[key], value) for key, value in attenuations.items()) and set(attenuations) == set(expected_attenuations),
        expected_attenuations, attenuations)
    metrics = tall_to_wide(persisted["/gradient_metrics_table"], "metric_key", "column_header", "value")
    misses = [(column, key, float(metrics.at[key, column]), computed["metrics"][column].get(key))
              for column in metrics.columns for key in metrics.index if not key.startswith("_")
              and not _near(computed["metrics"][column].get(key, np.nan), metrics.at[key, column])]
    found["metrics"] = _check(f"{mode}: /gradient_metrics_table (every metric of every column)", not misses,
                              "all within float32", misses[:6])
    persisted_breakdown = persisted["/signals_breakdown_table"]
    misses = [(row, column, value, computed["breakdown"].get((row, column)))
              for row, column, value in zip(persisted_breakdown["row_key"], persisted_breakdown["column_header"],
                                            persisted_breakdown["value"])
              if not _near(computed["breakdown"].get((row, column), np.nan), value)]
    found["breakdown"] = _check(f"{mode}: /signals_breakdown_table", not misses and len(persisted_breakdown) > 0,
                                "all cells equal", misses[:6])
    CTX.log.info("%s: terminal total_variation %.6f, attenuated %.6f, terminal attenuation %.6f, kept %d of %d", mode,
                 computed["total"].iloc[-1], computed["attenuated"].iloc[-1], computed["terminal"], len(computed["kept"]),
                 len(computed["segments"]))
    return found


def _series_check(name: str, expected: pd.Series, persisted: pd.Series) -> dict:
    """A curve against its persisted float32 copy."""
    return _check(name, _near_series(expected, persisted), "equal within float32",
                  {"points": len(persisted), "last_expected": float(expected.iloc[-1]), "last_persisted": float(persisted.iloc[-1])})


def _near_series(expected: pd.Series, persisted: pd.Series) -> bool:
    """Same dates and values within float32 storage tolerance."""
    persisted = persisted.copy()
    persisted.index = pd.to_datetime(persisted.index)
    return expected.index.equals(persisted.index) and bool(np.allclose(expected.to_numpy(dtype=float),
                                                                       persisted.to_numpy(dtype=float), rtol=RTOL, atol=ATOL))


def _near(expected: float, persisted: float) -> bool:
    """Scalars within tolerance, NaN equal to NaN."""
    if pd.isna(expected) or pd.isna(persisted):
        return bool(pd.isna(expected) and pd.isna(persisted))
    return bool(np.isclose(float(expected), float(persisted), rtol=RTOL, atol=ATOL))


def _check(name: str, passed: bool, expected: object, actual: object) -> dict:
    """A verdict whose result is also returned."""
    CTX.verdict(name, passed, expected, actual)
    return {"passed": passed, "actual": actual}


# =============================================================================
# VARIANTS AND SWEEP
# =============================================================================


def alternatives(mode: str, inputs: dict, settings: Settings, coded: dict) -> dict:
    """Q4 begins first (the audited revision), Q17 the last-year attenuation only marked (the audited revision; deltas decided minus marked), Q21
    every is_signal row (the audited revision; values the audited revision, decided), Q8 persisted per-series trials_to_true over the decided
    breakdown rows."""
    found = {}
    begin_first = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings,
                           SAME_BEGIN_FIRST)
    found["Q4"] = {"kept_delta": sorted(set(begin_first["kept"]) ^ set(coded["kept"])),
                   "terminal_delta": float(begin_first["total"].iloc[-1] - coded["total"].iloc[-1])}
    marked = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings, SAME_END_FIRST,
                      last_year_applied=False)
    found["Q17"] = {"attenuated_last_delta": float(coded["attenuated"].iloc[-1] - marked["attenuated"].iloc[-1]),
                    "acr_after_attenuation": (marked["metrics"][STRATEGY]["acr_after_attenuation"], coded["metrics"][STRATEGY]["acr_after_attenuation"])}
    every = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings, SAME_END_FIRST,
                     breakdown_rows="all")
    found["Q21"] = [[*key, every["breakdown"][key], coded["breakdown"][key]] for key in coded["breakdown"]
                    if not _near(every["breakdown"][key], coded["breakdown"][key])]
    per_series = breakdown(coded["breakdown_rows"], inputs["series"], settings.display_signals, trials="persisted")
    found["Q8"] = [[*key, coded["breakdown"][key], per_series[key]] for key in coded["breakdown"]
                   if key[0].startswith("trials_to_true") and not _near(coded["breakdown"][key], per_series[key])]
    CTX.log.info("%s alternatives: %s", mode, json.dumps(found, default=str))
    return found


def point_in_time_variant(args: argparse.Namespace, settings: Settings) -> dict:
    """The realistic oracle on EXP-08's point-in-time signals (trigger values from the trigger file)."""
    if not args.variant.is_file():
        harness.refuse(CTX.log, "variant", f"EXP-08 variant table {args.variant} is missing")
    with pd.HDFStore(str(args.variant), mode="r") as store:
        variant = store.get("/signals")
    inputs = load_inputs(args.run_copy / PARAMS / "realistic", settings)
    missing = sorted(set(variant.loc[variant[IS_SIGNAL], SERIES_ID]) - set(inputs["trigger_series"]))
    if missing:
        harness.refuse(CTX.log, "variant", f"variant signal series {missing} absent from the trigger file")
    coded = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings, SAME_END_FIRST)
    varied = evaluate(variant, inputs["trigger_series"], inputs["references"], inputs["index"], settings, SAME_END_FIRST)
    keyed = [frame.set_index([SERIES_ID, TRIGGER_BEGIN_DATES])[[GROUP_ID, IS_SIGNAL]] for frame in (inputs["signals"], variant)]
    rows = keyed[0].join(keyed[1], how="outer", lsuffix="_persisted", rsuffix="_variant")
    differing = rows[(rows[f"{GROUP_ID}_persisted"] != rows[f"{GROUP_ID}_variant"])
                     | (rows[f"{IS_SIGNAL}_persisted"] != rows[f"{IS_SIGNAL}_variant"])]
    found = {"variant_md5": harness.file_md5(args.variant), "variant_label": args.variant_label,
             "signals": (int(inputs["signals"][IS_SIGNAL].sum()), int(variant[IS_SIGNAL].sum())),
             "rows_differing": len(differing),
             "terminal_total_variation": (float(coded["total"].iloc[-1]), float(varied["total"].iloc[-1])),
             "acr": (coded["metrics"][STRATEGY]["acr"], varied["metrics"][STRATEGY]["acr"])}
    CTX.log.info("HYP-26 variant: %s", json.dumps(found))
    CTX.verdict("HYP-26 (Q20): the point-in-time variant equals the persisted realistic grouping, so the terminal "
                "total_variation is unchanged", not found["rows_differing"] and _near(*found["terminal_total_variation"]),
                {"rows_differing": 0, "terminal": found["terminal_total_variation"][0]},
                {"rows_differing": found["rows_differing"], "terminal": found["terminal_total_variation"][1]})
    return found


def sweep(inputs: dict, settings: Settings) -> dict:
    """Production builders with each method, a deduction and an mpv, against the oracle (the index of the observed
    dates from the earliest effective start, build_global_index as changed by W1 and, on this branch, by W5;
    compute_gradient_columns without the strategy's terminal attenuation, already in the attenuated curve, as changed by
    W11)."""
    idx = build_global_index(inputs["observed"], inputs["start"], settings.end_date)
    signals_data, _ = build_signal_segments_cumulative_pct_with_maxweight_constraint(
        inputs["signals"], inputs["series"], idx, settings.weight, settings.max_total)
    found = {}
    for method in METHODS:
        varied = dataclasses.replace(settings, method=method, deduction_pct=SWEEP_DEDUCTION_PCT, mpv=SWEEP_MPV)
        pure, attenuated, terminal = build_actual_variation_pair(signals_data, varied.weight, idx, method, varied.attenuation_rate,
                                                                 varied.attenuation_month, boundary_deduction_pct=varied.deduction_pct)
        oracle = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], varied, SAME_END_FIRST)
        references = {ref_id: curve for ref_id, curve in oracle["references"].items()}
        pre, post = compute_gradient_columns(pure, attenuated, references, oracle["reference_terminal"],
                                             selection=list(varied.display_gradient), mpv=varied.mpv)
        table = build_gradient_metrics_dataframe(pre, post, list(varied.display_gradient))
        misses = [(column, key) for column in table.columns for key in table.index if not key.startswith("_")
                  and not _near(oracle["metrics"][column].get(key, np.nan), table.at[key, column])]
        curves = _near_series(oracle["total"], pure) and _near_series(oracle["attenuated"], attenuated) and _near(oracle["terminal"], terminal)
        found[method] = {"curves": curves, "metric_misses": misses[:6], "terminal": float(pure.iloc[-1])}
        CTX.verdict(f"sweep {method} (deduction {SWEEP_DEDUCTION_PCT}, mpv {SWEEP_MPV}): production equals the oracle",
                    curves and not misses, "curves and metrics equal", found[method])
    return found


def check_control(inputs: dict, settings: Settings) -> None:
    """A perturbed persisted /total_variation point must be flagged."""
    computed = evaluate(inputs["signals"], inputs["series"], inputs["references"], inputs["index"], settings, SAME_END_FIRST)
    total = inputs["persisted"]["/total_variation"].set_index("date")["total_variation"].copy()
    total.iloc[len(total) // 2] += 0.01
    CTX.verdict("control: a perturbed persisted curve point is flagged", not _near_series(computed["total"], total),
                "flagged", "difference detected" if not _near_series(computed["total"], total) else "missed")


if __name__ == "__main__":
    main()
