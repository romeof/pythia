"""Weekly-bar oracle: the shipped weekly workflow and a synthetic weekly input with hand-derived values, both modes.

Two kinds of expectation, never mixed: Part A's are measured figures, a regression pin; Part B's are constants
derived by hand from the fixture below, and no value read back from an output serves as one.

Part A, workflows/extrema_weekly.yaml. It validates under the dev and prod profiles and names the parameters folder
var12ge0_83x0-var15ge0_73__w7md1p0_05; its leaves differ from workflows/extrema.yaml in exactly read.raw_data
"../data/2026_09_25_1/0/raw/0.h5", read.end_date "2021-09-04", trigger.strategy.min_distance 1,
trigger.replay.lookback_window 10, trigger.replay.min_duration_periods 1 and trigger.replay.end.max_duration_periods
100; neither file's text holds the word day or days. A copy of extrema.yaml with the weekly overrides (those six
leaves, read from the weekly YAML, and the shipped gate items and pre-filter over the copy's GATE_OFF), run --task all
under the prod profile on the shipped weekly input, reproduces the weekly run R6, measured on this code on 2026-11-06
and never derived by hand: return 0; is_signal rows of the signal store's /signals 32 ideal and 57 realistic; the
reference curve on 131 Saturdays from 2019-03-09 to 2021-09-04; reference acr 38.942, std 22.055 and max_fall 27.250
within 5e-4, half a unit of their third decimal, and fall_dur 16, in both modes. A failing pin is a finding, never a
re-baseline.

Part B, a synthetic input. Calendar: weeks k = 0..77, Saturday S_k = 2020-01-11 + 7k days (S_9 2020-03-14, S_45
2020-11-21, S_50 2020-12-26, S_51 2021-01-02, S_73 2021-06-05, S_77 2021-07-03); daily rows Monday S_k - 5 to
Friday S_k - 1, row index d = 0..389. Knots of series 0: v_-1 = 97.5, v_k = v_(k-1) + s_k with s_k = 2.5 (k <= 14),
-3.75 (15-20), 5 (21-32), -6.25 (33-40), 7.5 (41-56), -8.75 (57-66), 10 (67-77): maxima v_14 = 135, v_32 = 172.5,
v_56 = 242.5, minima v_20 = 112.5, v_40 = 122.5, v_66 = 155; v_9 = 122.5, v_77 = 265. Row j = 1..5 of week k: var3 =
v_(k-1) + j s_k / 5 (Friday on the knot), series 1 = 2 x series 0 - 100; var0 = the previous row's var3 of the same
series + 0.5, on its own rows (first row 98.0 / 95.5); var1 = min(var0, var3) - 1, var2 = max(var0, var3) + 1,
var4 = var3, var5 = 1000 + d, var12 = var15 = 0.9. Every increment is a multiple of 0.25, so float32 keeps every
value. Anomalies, in both series unless named: (i) Monday 2020-06-29 (week 25) is an uncounted row, var5 0 and var0 =
var1 = var2 = var3 = var4 = Friday 2020-06-26's var3; (ii) Friday 2020-06-26 (week 24) is a counted row with var5 0;
(iii) series 1 lacks the five rows of week 45, so its Monday 2020-11-23 has var0 205.0 + 0.5; (iv) the rows of week 40
hold var12 0.5. The slope magnitudes grow leg by leg, so the 10-bar window ending at e + 1 reports each extremum e
(width at half prominence 0.5 + s_r / (2 s_l) > 1), and no prominence filters: an ideal one is a full leg of at least
22.5 against 5 % of the range 165, a realistic one the single right step, above 5 % of the window's range.

Bars (weekly_bars, the upstream rules D1-D5 for the eight columns read): a row belongs to the bar of the Saturday on
or after it; a row is counted unless var5 is 0 and var0 = var1 = var2 = var3 (D3), and a bar uses its counted rows,
or all its rows when none counts; var0 is the first used row's, var1 the minimum, var2 the maximum, var3, var4, var12
and var15 the last used row's, var5 the sum. Series 0 gives 78 bars S_0..S_77, series 1 77 (no S_45); the var5 sums
keep the daily totals and stay within 16,777,216. Bar 24 holds Friday 2020-06-26's var2, var3, var4 (133.5, 132.5,
132.5; series 1 166, 165, 165), since that zero-var5 row counts (without it Thursday's 132.5, 131.5, 131.5), and the
Monday-Thursday var5 sum 4486; bar 25 holds Tuesday's var0 (133.0; series 1 165.5), since the uncounted Monday is
skipped (it would give 132.5 and 165.0), the Tuesday-Friday sum 4510 and var3 on the knot 137.5 (175). Every bar's
var3 is its knot.

Run: one dual --task all under the prod profile on a copy with the weekly overrides and read.ini_date 2020-01-11,
read.end_date 2021-07-03; as shipped: following_row, weight 0.2, multiplicative_total_before_active_signals, attenuation 0.25 in
month 6, period_back 5, corr_threshold 0.6, the gate var12 >= 0.83 exempting series 0 and var15 >= 0.73.
Hand-derived:
- tv(k), the recorded trigger_value, is the next bar's var0 = var3(k) + 0.5 (v_k + 0.5 for series 0, 2 v_k - 99.5
  for series 1, across the gap week and the uncounted Monday too), the last bar's var3 on the last bar; it differs
  from var3(k) and var0(k), so a same-row reading of either fails;
- triggers (weeks begin -> end, direction, tv values, all true; durations in rows of the series' own index): ideal,
  both series, 14->20 minus, 20->32 plus, 32->40 minus, 40->56 plus, 56->66 minus (series 0 135.5->113.0,
  113.0->173.0, 173.0->123.0, 123.0->243.0, 243.0->155.5, 6, 12, 8, 16, 10 rows; series 1 170.5->125.5,
  125.5->245.5, 245.5->145.5, 145.5->385.5, 385.5->210.5, 6, 12, 8, 15, 10 rows), end_reason ideal_mode on each end
  and a plus begin still active at week 66; realistic, each begin one bar later: series 0 15->21 minus 131.75->118.0,
  21->33 plus 118.0->166.75, 33->41 minus 166.75->130.5, 41->57 plus 130.5->234.25, 57->67 minus 234.25->165.5 (6,
  12, 8, 16, 10 rows), series 1 15->21 minus 163.0->135.5, 21->33 plus 135.5->233.0, 33->67 minus 233.0->230.5 (6,
  12, 33 rows), end_reason opposite_trigger on each end and a plus begin still active at week 67; every duration is
  below 100, so no max_duration;
- the gate reads the previous bar: bar 40 fails var12 for series 1 (series 0 is exempt), so ideal keeps its begin on
  S_40 (bar 39 passes) and realistic has none on S_41, where series 0 begins; the blocked candidate is suppressed, the
  week-56 minus candidate repeats the active trigger's type, and the week-66 plus candidate ends it on S_67;
- groups: the two series begin together (every ideal begin, the realistic S_15, S_21 and S_33) and correlate 1 as
  affine copies, series 1 the steeper; ideal 5 groups of both series, signal series 1; realistic 15->21 and 21->33
  likewise, then series 1's 33->67 (signal) with series 0's 33->41, 41->57 and 57->67 (each begins inside [S_33,
  S_67)); num_series 2; is_signal rows 5 and 3; the performance store's per-trigger duration and variation
  (variation_pct of end over start, negated for a minus trigger) on series 1's rows, the only series it enriches;
- curve: index S_9..S_77, 69 Saturdays (the reference keeps S_45), n_years = 476 / 365.25, m = 68 / n_years =
  365.25 / 7; pure(t) = the sum over the begun events (series 1's signals) of locked_i x cum_i(t): cum_i(t) is the
  product of tv(r) / tv(r') (plus) or 2 - tv(r) / tv(r') (minus) over series 1's rows r after the begin up to min(t,
  end), r' series 1's row before r (week 44 before week 46), minus 1 (on S_45, absent in series 1, every cum keeps its
  S_44 value), and locked_i = 0.2 x (1 + the locked_j x cum_j of the events ended strictly before its begin), no attenuation
  applied before any begin: ideal locked 0.2, 0.2, 0.2135338, 0.2517808, 0.2788104, pure(S_44) = pure(S_45) 0.497879,
  pure(S_50) 0.653620, pure(S_51) 0.679577, pure(S_77) 1.023360; realistic locked 0.2, 0.2, 0.2071923, pure(S_44) =
  pure(S_45) 0.193219, pure(S_50) 0.120736, pure(S_51) 0.113221, pure(S_77) 0.151981; flat after the last end;
- attenuations: A = 0.25 x (pure(S_50) - pure(S_9)) (ideal 0.163405, realistic 0.030184) from S_73, the first date the
  attenuated curve differs, and the terminal T = 0.25 x max(0, pure(S_77) - A - pure(S_51)) (0.045095, 0.002144) on S_77;
- reference: level(S) = v3(S) / 122.5, v3(S) the var3 of the last counted row on or before S in series 0's daily rows
  (each the knot; 122.5 is Friday 2020-03-13's); acr = variation_pct((265 / 122.5) ^ (1 / n_years), 1) = 80.776; std
  = the ddof-1 standard deviation of variation_pct(v_k, v_(k-1)) over k = 10..77 x sqrt(m) = 28.833478; max_fall =
  -variation_pct(155, 242.5) = 36.082474; terminal attenuation 0.25 x (265 / 122.5 - 1) = 0.290816;
- daily side: the max_fall of series 0's daily levels from Friday 2020-03-13 to Friday 2021-07-02 is 36.082474 (linear
  between the knots), and the weekly max_fall does not exceed it; the span of those levels, 476 days, gives the curve's
  n_years;
- breakdown (Signals row): ideal 5 (plus 2, minus 3), realistic 3 (plus 1, minus 2), all true.

Structure:
1. main
2. Part A: the weekly workflow and the R6 run
3. Part B: the synthetic bars and the mode verdicts
4. The R6 verdicts, builders and readers, shared with the record drivers
5. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.common.common import variation_pct
from src.utils.common.config import load_config, validate_config
from src.utils.common.constants import GRADIENT_REFERENCE_HEADER_TEMPLATE, GRADIENT_STRATEGY_SENTINEL
from src.utils.common.strategyname import get_strategy_parameters

# Part A: the weekly workflow, its six weekly leaves (A1) and its parameters folder
WEEKLY_FILE = "extrema_weekly.yaml"
PROD_PROFILE = "prod"
WEEKLY_PARAMETERS = "var12ge0_83x0-var15ge0_73__w7md1p0_05"
WEEKLY_LEAVES = {"read.raw_data": "../data/2026_09_25_1/0/raw/0.h5", "read.end_date": "2021-09-04",
                 "trigger.strategy.min_distance": 1, "trigger.replay.lookback_window": 10,
                 "trigger.replay.min_duration_periods": 1, "trigger.replay.end.max_duration_periods": 100}
ABSENT = "<absent>"  # a leaf one YAML lacks
DAY_WORD = re.compile(r"\bdays?\b")
# R6, measured on this code on 2026-11-06, never derived by hand: is_signal rows per mode, the reference curve's dates,
# the reference metrics within R6_ATOL, half a unit of their third decimal
R6_SIGNALS = {"ideal": 32, "realistic": 57}
R6_CURVE = {"dates": 131, "first": "2019-03-09", "last": "2021-09-04", "saturdays": True}
R6_METRICS = {"acr": 38.942, "std": 22.055, "max_fall": 27.250, "fall_dur": 16.0}
R6_ATOL = 5e-4
REFERENCE_ID = "0"  # the shipped read.reference_series, which no weekly copy overrides
FLOAT32_RTOL = 1e-6  # a float32-persisted value against its float64 expectation
SATURDAY = 5  # DatetimeIndex.dayofweek
DATE_FORMAT = "%Y-%m-%d"

# Part B: the calendar, the knots of series 0 and the affine map of each series, the anomalies
SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
SATURDAYS = pd.date_range("2020-01-11", periods=78, freq="7D")  # S_0..S_77
ROWS_PER_WEEK = 5  # Monday to Friday
EPOCH = pd.Timestamp("1970-01-01")  # a Thursday: day 2 is a Saturday
EPOCH_SATURDAY = 2
LEG_ENDS = (14, 20, 32, 40, 56, 66, 77)  # the last week of each leg
LEG_SLOPES = (2.5, -3.75, 5.0, -6.25, 7.5, -8.75, 10.0)  # s_k per week of the leg
WEEK_SLOPES = np.repeat(LEG_SLOPES, np.diff((-1,) + LEG_ENDS))  # s_0..s_77
FIRST_KNOT = 97.5  # v_-1
KNOTS = FIRST_KNOT + np.r_[0.0, np.cumsum(WEEK_SLOPES)]  # v_-1..v_77: v_k is KNOTS[k + 1]
AFFINE = {"0": (1.0, 0.0), "1": (2.0, -100.0)}  # series: (factor, offset) on series 0's var3
VAR0_STEP = 0.5  # var0 = the previous row's var3 + VAR0_STEP, so tv(k) = var3(k) + VAR0_STEP
BAND = 1.0  # var1 and var2 lie BAND below min(var0, var3) and above max(var0, var3)
VAR5_BASE = 1000.0  # var5 = VAR5_BASE + d
GATE_PASS, GATE_FAIL = 0.9, 0.5
GATE_WEEK = 40  # var12 GATE_FAIL in both series
ZERO_VAR5_ROW = 24 * ROWS_PER_WEEK + 4  # Friday 2020-06-26, counted with var5 0
UNCOUNTED_ROW = 25 * ROWS_PER_WEEK  # Monday 2020-06-29, flat at the Friday before it with var5 0
GAP_SERIES, GAP_WEEK = "1", 45
FLOAT32_SUM_BOUND = 2 ** 24  # the largest integer float32 holds with every integer below it
# The anomalies' bars per series: bar 24 holds the zero-var5 Friday's var2-var4 and the Monday-Thursday var5 sum;
# bar 25 holds Tuesday's var0 (the uncounted Monday skipped), the Tuesday-Friday sum and var3 on the knot
ANOMALY_BARS = {"0": {24: {"var2": 133.5, "var3": 132.5, "var4": 132.5, "var5": 4486.0},
                      25: {"var0": 133.0, "var3": 137.5, "var5": 4510.0}},
                "1": {24: {"var2": 166.0, "var3": 165.0, "var4": 165.0, "var5": 4486.0},
                      25: {"var0": 165.5, "var3": 175.0, "var5": 4510.0}}}

# Part B expectations: per mode the end reason, the week of the plus begin still active at the end and per series its
# triggers (begin week, end week, plus, duration in rows of the series' own index)
PLUS, MINUS = True, False
EXPECTED = {
    "ideal": ("ideal_mode", 66, {
        "0": [(14, 20, MINUS, 6), (20, 32, PLUS, 12), (32, 40, MINUS, 8), (40, 56, PLUS, 16), (56, 66, MINUS, 10)],
        "1": [(14, 20, MINUS, 6), (20, 32, PLUS, 12), (32, 40, MINUS, 8), (40, 56, PLUS, 15), (56, 66, MINUS, 10)]}),
    "realistic": ("opposite_trigger", 67, {
        "0": [(15, 21, MINUS, 6), (21, 33, PLUS, 12), (33, 41, MINUS, 8), (41, 57, PLUS, 16), (57, 67, MINUS, 10)],
        "1": [(15, 21, MINUS, 6), (21, 33, PLUS, 12), (33, 67, MINUS, 33)]}),
}
# mode: the begin week the gate decides and whether series 1's plus begin stands there (series 0, exempt, begins)
GATE_BEGIN = {"ideal": (40, True), "realistic": (41, False)}
SIGNAL_SERIES = "1"
# mode: the groups in begin sequence, each (series, begin week, end week) of its signal, then its other members
GROUPS = {
    "ideal": [[("1", begin, end), ("0", begin, end)] for begin, end in ((14, 20), (20, 32), (32, 40), (40, 56), (56, 66))],
    "realistic": [[("1", 15, 21), ("0", 15, 21)], [("1", 21, 33), ("0", 21, 33)],
                  [("1", 33, 67), ("0", 33, 41), ("0", 41, 57), ("0", 57, 67)]],
}
NUM_SERIES = 2
CURVE_FIRST_WEEK = 9  # index[lookback_window - 1]
SPAN_DAYS = 476  # S_9 to S_77, and Friday 2020-03-13 to Friday 2021-07-02 on the daily side
N_YEARS = SPAN_DAYS / 365.25
ANNUAL_PERIODS = 365.25 / 7  # m: 68 intervals over N_YEARS
WEIGHT = 0.2
ATTENUATION_RATE = 0.25
YEAR_LAST_WEEK, YEAR_FIRST_WEEK, ATTENUATION_WEEK = 50, 51, 73  # 2020's last curve date, 2021's first, the 2020 attenuation date
GAP_NEIGHBOUR_WEEK = 44  # the curve date before series 1's gap week
MAX_WEEK, DEEPEST_WEEK = 56, 66  # the reference's deepest fall from the running maximum
TOLERANCE = 1e-5
CURVE_ATOL = 1e-7


def main() -> None:
    """Part A, the weekly workflow and its run, then Part B, the synthetic weekly input, then the summary (return
    code 1 on any FAIL)."""
    texts = {name: (KIT.root / "workflows" / name).read_text(encoding="utf-8")
             for name in (testkit.STRATEGY_FILE, WEEKLY_FILE)}
    shipped, weekly = (yaml.safe_load(texts[name]) for name in (testkit.STRATEGY_FILE, WEEKLY_FILE))
    workflow_verdicts(texts, shipped, weekly)
    overrides = weekly_overrides(shipped, weekly)
    weekly_run(overrides)
    synthetic_run(overrides)
    KIT.finish()


# =============================================================================
# PART A: THE WEEKLY WORKFLOW AND THE R6 RUN
# =============================================================================


def workflow_verdicts(texts: dict[str, str], shipped: dict, weekly: dict) -> None:
    """The weekly YAML validates under both profiles and names its parameters folder; its leaves differ from the
    shipped YAML's in exactly WEEKLY_LEAVES; neither text holds the word day or days."""
    weekly_file = str(KIT.root / "workflows" / WEEKLY_FILE)
    names = {profile: get_strategy_parameters(validate_config(load_config(weekly_file, profile)).trigger)
             for profile in (testkit.DEV_PROFILE, PROD_PROFILE)}
    KIT.verdict(f"{WEEKLY_FILE} validates under both profiles and names {WEEKLY_PARAMETERS}",
                set(names.values()) == {WEEKLY_PARAMETERS}, WEEKLY_PARAMETERS, names)
    shipped_leaves, weekly_leaves = dict(testkit.leaves(shipped)), dict(testkit.leaves(weekly))
    differing = {key: weekly_leaves.get(key, ABSENT) for key in sorted(shipped_leaves.keys() | weekly_leaves.keys())
                 if shipped_leaves.get(key, ABSENT) != weekly_leaves.get(key, ABSENT)}
    KIT.verdict(f"{WEEKLY_FILE} differs from {testkit.STRATEGY_FILE} in exactly the six weekly leaves",
                differing == WEEKLY_LEAVES, WEEKLY_LEAVES, differing)
    words = {name: DAY_WORD.findall(text) for name, text in texts.items()}
    KIT.verdict("neither workflow's text holds the word day or days (the comments count bars)",
                not any(words.values()), {name: [] for name in texts}, words)


def weekly_overrides(shipped: dict, weekly: dict) -> dict:
    """The dotted overrides of a weekly copy of the shipped YAML: the weekly YAML's leaves that differ from the
    shipped ones, then the shipped values of the keys GATE_OFF names (the gate items and the pre-filter), restored
    over the copy's GATE_OFF."""
    shipped_leaves = dict(testkit.leaves(shipped))
    differing = {key: value for key, value in testkit.leaves(weekly) if shipped_leaves.get(key, ABSENT) != value}
    return {**differing, **{key: shipped_leaves[key] for key in testkit.GATE_OFF}}


def weekly_run(overrides: dict) -> None:
    """R6: a copy of the shipped YAML with the weekly overrides, run --task all under the prod profile on the shipped
    weekly input, returns 0 and reproduces the measured figures in both modes."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=overrides)
        child = testkit.run_pipeline(KIT, folder, "weekly", config_file, ["all"], profile=PROD_PROFILE)
        KIT.verdict("R6: the weekly --task all returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            r6_verdicts(mode, *(testkit.store_frames(testkit.mode_output(config_file, mode, task))
                                for task in ("signal", "performance")))


# =============================================================================
# PART B: THE SYNTHETIC BARS AND THE MODE VERDICTS
# =============================================================================


def synthetic_run(overrides: dict) -> None:
    """The synthetic daily rows and their bars, the bars' verdicts, one dual --task all on a copy with the weekly
    overrides that reads the bars, then both modes' verdicts."""
    daily = {series_id: daily_frame(series_id) for series_id in AFFINE}
    bars = {series_id: weekly_bars(frame) for series_id, frame in daily.items()}
    bar_verdicts(daily, bars)
    with testkit.scratch(KIT) as folder:
        raw = folder / SYNTHETIC_INPUT
        testkit.write_raw_input(raw, bars)
        read = {"read.raw_data": f"./{Path(os.path.relpath(raw, Path.cwd())).as_posix()}",
                "read.ini_date": SATURDAYS[0].strftime(DATE_FORMAT), "read.end_date": SATURDAYS[-1].strftime(DATE_FORMAT)}
        config_file = testkit.config_copy(KIT, folder, strategy={**overrides, **read})
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"], profile=PROD_PROFILE)
        KIT.verdict("the synthetic dual --task all returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            mode_verdicts(mode, {task: testkit.store_frames(testkit.mode_output(config_file, mode, task))
                                 for task in testkit.TASKS}, daily["0"])


def bar_verdicts(daily: dict[str, pd.DataFrame], bars: dict[str, pd.DataFrame]) -> None:
    """Before the run: the Saturday stamps, the var5 sums and their float32 bound, the anomalies' bars, every var3 on
    its knot."""
    for series_id, frame in bars.items():
        knots = bar_levels(series_id)
        expected = {"bars": len(knots), "saturdays": True, "as hand-derived": True}
        stamps = {"bars": len(frame), "saturdays": bool((frame.index.dayofweek == SATURDAY).all()),
                  "as hand-derived": list(frame.index) == list(knots.index)}
        KIT.verdict(f"series {series_id}: {len(knots)} bars, each stamped on its Saturday", stamps == expected, expected,
                    stamps)
        sums = {"daily": float(daily[series_id]["var5"].sum()), "weekly": float(frame["var5"].astype(float).sum()),
                "largest bar": float(frame["var5"].max())}
        KIT.verdict(f"series {series_id}: the bars' var5 keeps the daily total, every bar within {FLOAT32_SUM_BOUND}",
                    sums["weekly"] == sums["daily"] and sums["largest bar"] <= FLOAT32_SUM_BOUND,
                    {"weekly": sums["daily"], "largest bar": f"<= {FLOAT32_SUM_BOUND}"}, sums)
        found = {week: {column: float(frame.at[SATURDAYS[week], column]) for column in columns}
                 for week, columns in ANOMALY_BARS[series_id].items()}
        KIT.verdict(f"series {series_id}: bar 24 holds the zero-var5 Friday's var2-var4 (that row counts), bar 25 "
                    "Tuesday's var0 (the uncounted Monday skipped)", found == ANOMALY_BARS[series_id],
                    ANOMALY_BARS[series_id], found)
        found_knots = frame["var3"].to_numpy(dtype=float)
        KIT.verdict(f"series {series_id}: every bar's var3 is its knot", np.array_equal(found_knots, knots.to_numpy()),
                    knots.to_numpy()[[0, 24, 25, -1]].tolist(), found_knots[[0, 24, 25, -1]].tolist())


def mode_verdicts(mode: str, stores: dict[str, dict[str, pd.DataFrame]], daily: pd.DataFrame) -> None:
    """One mode: each series' recording, triggers, end reasons and active begin, the gate, the groups, the
    per-trigger metrics and the breakdown, the curve and the reference."""
    trigger, signal, performance = (stores[task] for task in testkit.TASKS)
    frames = {series_id: trigger[f"/series_{series_id}"] for series_id in AFFINE}
    for series_id, frame in frames.items():
        frame.index = pd.DatetimeIndex(frame.index)
        series_verdicts(series_id, frame, mode, trigger[f"/triggers_{series_id}"])
    week, stands = GATE_BEGIN[mode]
    begins = {series_id: int(frame["trigger_plus_begin"].at[SATURDAYS[week]]) for series_id, frame in frames.items()}
    expected = {"0": 1, "1": int(stands)}
    KIT.verdict(f"{mode}: series 1's plus begin on S_{week} {'stands' if stands else 'is blocked'}, since the gate "
                f"reads bar {week - 1}, which {'passes' if stands else 'fails var12'}; series 0, exempt, begins there",
                begins == expected, expected, begins)
    group_verdicts(mode, signal["/signals"])
    signal_metric_verdicts(mode, performance)
    metrics = reference_metrics(performance)
    attenuations = performance["/gradient_terminal_attenuations"].set_index("series_id")["amount"].astype(float)
    curve_verdicts(mode, performance, metrics, attenuations.get(GRADIENT_STRATEGY_SENTINEL, np.nan))
    reference_verdicts(mode, reference_levels(performance), metrics, attenuations.get(REFERENCE_ID, np.nan), daily)


def series_verdicts(series_id: str, series: pd.DataFrame, mode: str, triggers: pd.DataFrame) -> None:
    """One series of one mode: trigger_value under following_row, the triggers, end_reason exactly on their ends, the
    plus begin still active at the end."""
    reason, active_week, _ = EXPECTED[mode]
    index = series.index
    expected_values = trigger_values(series_id)
    found = series["trigger_value"].to_numpy(dtype=float)
    same_row = {column: bool(np.any(expected_values.to_numpy()[:-1] == series[column].to_numpy(dtype=float)[:-1]))
                for column in ("var3", "var0")}
    KIT.verdict(f"{mode} series {series_id}: trigger_value is the next bar's var0 (var3 + {VAR0_STEP}), var3 on the "
                "last bar, never the same bar's var3 or var0",
                list(index) == list(expected_values.index) and np.array_equal(found, expected_values.to_numpy())
                and not any(same_row.values()),
                {"values": expected_values.to_numpy()[[0, 24, -2, -1]].tolist(), "same row": {"var3": False, "var0": False}},
                {"values": found[[0, 24, -2, -1]].tolist(), "same row": same_row})
    rows = sorted((*weeks([begin, end]), bool(plus), float(start), float(end_value), bool(true))
                  for begin, end, plus, start, end_value, true in triggers[
                      ["trigger_begin_dates", "trigger_end_dates", "is_plus_trigger", "trigger_start_values",
                       "trigger_end_values", "is_true"]].itertuples(index=False))
    expected = expected_triggers(series_id, mode)
    KIT.verdict(f"{mode} series {series_id}: the triggers in begin sequence (begin week, end week, plus, start, end, "
                "true)", rows == expected, expected, rows)
    ends = {end: reason for _, end, _, _, _, _ in expected}
    reasons = {week: value for week, value in zip(weeks(index), series["end_reason"]) if value != ""}
    KIT.verdict(f"{mode} series {series_id}: end_reason {reason} exactly on the trigger ends", reasons == ends, ends,
                reasons)
    active = {direction: weeks(index[series[f"active_trigger_{direction}_begin"].to_numpy() > 0])
              for direction in ("plus", "minus")}
    KIT.verdict(f"{mode} series {series_id}: a plus begin still active at week {active_week}",
                active == {"plus": [active_week], "minus": []}, {"plus": [active_week], "minus": []}, active)


def group_verdicts(mode: str, table: pd.DataFrame) -> None:
    """The signal store's groups, each its signal then its other members, num_series and the is_signal rows."""
    groups = []
    for _, group in table.groupby("group_id", sort=True):
        members = [((str(series_id), *weeks([begin, end])), bool(flag)) for series_id, begin, end, flag in group[
            ["series_id", "trigger_begin_dates", "trigger_end_dates", "is_signal"]].itertuples(index=False)]
        groups.append([member for member, flag in members if flag]
                      + sorted(member for member, flag in members if not flag))
    found = {"groups": sorted(groups), "num_series": sorted(set(table["num_series"].astype(int))),
             "is_signal": int(table["is_signal"].sum())}
    expected = {"groups": GROUPS[mode], "num_series": [NUM_SERIES], "is_signal": len(GROUPS[mode])}
    KIT.verdict(f"{mode}: the groups, each its signal (series {SIGNAL_SERIES}) first, num_series {NUM_SERIES}",
                found == expected, expected, found)


def signal_metric_verdicts(mode: str, performance: dict[str, pd.DataFrame]) -> None:
    """The performance store's per-trigger duration and variation on series 1's rows, the breakdown's Signals row."""
    table = performance["/signals"]
    rows = table[table["series_id"].astype(str) == SIGNAL_SERIES].sort_values("trigger_begin_dates")
    found = [(*weeks([begin, end]), float(duration), float(variation)) for begin, end, duration, variation in rows[
        ["trigger_begin_dates", "trigger_end_dates", "duration", "variation"]].itertuples(index=False)]
    triggers = expected_triggers(SIGNAL_SERIES, mode)
    expected = [(begin, end, float(duration), variation_pct(end_value, start) * (1.0 if plus else -1.0))
                for (begin, end, plus, start, end_value, _), (*_, duration)
                in zip(triggers, EXPECTED[mode][2][SIGNAL_SERIES])]
    KIT.verdict(f"{mode}: series {SIGNAL_SERIES}'s per-trigger duration (rows) and variation (%)",
                len(found) == len(expected) and all(item[:3] == wanted[:3]
                                                    and np.isclose(item[3], wanted[3], rtol=TOLERANCE)
                                                    for item, wanted in zip(found, expected)), expected, found)
    plus = sum(1 for _, _, is_plus, *_ in triggers if is_plus)
    minus = len(triggers) - plus
    wanted = {"plus+minus_true+false": plus + minus, "plus+minus_true": plus + minus, "plus+minus_false": 0,
              "plus_true+false": plus, "plus_true": plus, "plus_false": 0, "minus_true+false": minus,
              "minus_true": minus, "minus_false": 0}
    breakdown = performance["/signals_breakdown_table"]
    counts = breakdown[breakdown["row_key"] == "Signals"].set_index("column_header")["value"].astype(float).to_dict()
    KIT.verdict(f"{mode}: the breakdown counts {plus + minus} signals, plus {plus}, minus {minus}, all true",
                counts == {key: float(value) for key, value in wanted.items()}, wanted, counts)


def curve_verdicts(mode: str, performance: dict[str, pd.DataFrame], metrics: dict[str, float],
                   terminal_attenuation: float) -> None:
    """The curve index, n_years and m; pure(t) before the first begin, at every event end, on the flat tail, around
    the gap week and the year boundary; the attenuated curve from S_73 and the strategy's terminal attenuation."""
    total = performance["/total_variation"].set_index("date")["total_variation"].astype(float)
    total.index = pd.DatetimeIndex(total.index)
    span = {"_n_years": metrics.get("_n_years", np.nan),
            "m": metrics.get("_n_intervals", np.nan) / metrics.get("_n_years", np.nan)}
    expected = {"dates": len(SATURDAYS) - CURVE_FIRST_WEEK, "_n_years": N_YEARS, "m": ANNUAL_PERIODS}
    KIT.verdict(f"{mode}: the curve index is S_{CURVE_FIRST_WEEK}..S_77 ({expected['dates']} Saturdays), n_years "
                f"{SPAN_DAYS} / 365.25 and m 365.25 / 7", list(total.index) == list(SATURDAYS[CURVE_FIRST_WEEK:])
                and np.allclose([span["_n_years"], span["m"]], [N_YEARS, ANNUAL_PERIODS], rtol=FLOAT32_RTOL, atol=0.0),
                expected, {"dates": len(total), **span})
    events = expected_triggers(SIGNAL_SERIES, mode)
    first_begin, last_end = events[0][0], events[-1][1]
    checked = sorted({*range(CURVE_FIRST_WEEK, first_begin + 1), *(end for _, end, *_ in events),
                      *range(last_end, len(SATURDAYS)), GAP_NEIGHBOUR_WEEK, GAP_WEEK, YEAR_LAST_WEEK, YEAR_FIRST_WEEK})
    pure = expected_curve(mode, SATURDAYS[checked])
    found = total.reindex(pure.index).to_numpy()
    KIT.verdict(f"{mode}: the curve equals pure(t) before the first begin, at every event end, on the flat tail and on "
                f"weeks {GAP_NEIGHBOUR_WEEK}, {GAP_WEEK}, {YEAR_LAST_WEEK}, {YEAR_FIRST_WEEK}",
                np.allclose(found, pure.to_numpy(), rtol=TOLERANCE, atol=CURVE_ATOL),
                dict(zip(checked, np.round(pure.to_numpy(), 6).tolist())), dict(zip(checked, np.round(found, 6).tolist())))
    first, year_last, year_first, last = (pure[SATURDAYS[week]] for week in (CURVE_FIRST_WEEK, YEAR_LAST_WEEK,
                                                                              YEAR_FIRST_WEEK, len(SATURDAYS) - 1))
    annual = ATTENUATION_RATE * max(0.0, year_last - first)
    terminal = ATTENUATION_RATE * max(0.0, last - annual - year_first)
    attenuated = performance["/total_variation_attenuated"].set_index("date")["total_variation_attenuated"].astype(float)
    attenuated.index = pd.DatetimeIndex(attenuated.index)
    differing = attenuated.index[attenuated.to_numpy() != total.reindex(attenuated.index).to_numpy()]
    tail = SATURDAYS[ATTENUATION_WEEK:]
    expected_tail = pure.reindex(tail).to_numpy() - annual
    expected_tail[-1] -= terminal
    found_attenuations = {"first differing date": str(differing[0].date()) if len(differing) else None,
                          "tail": np.round(attenuated.reindex(tail).to_numpy(), 6).tolist(), "terminal": terminal_attenuation}
    expected_attenuations = {"first differing date": str(SATURDAYS[ATTENUATION_WEEK].date()),
                             "tail": np.round(expected_tail, 6).tolist(), "terminal": terminal}
    KIT.verdict(f"{mode}: the attenuated curve departs on S_{ATTENUATION_WEEK} by A = {ATTENUATION_RATE} x (pure(S_{YEAR_LAST_WEEK}) - "
                f"pure(S_{CURVE_FIRST_WEEK})) and on S_77 also by the terminal T",
                found_attenuations["first differing date"] == expected_attenuations["first differing date"]
                and np.allclose(attenuated.reindex(tail).to_numpy(), expected_tail, rtol=TOLERANCE, atol=CURVE_ATOL)
                and np.isclose(terminal_attenuation, terminal, rtol=TOLERANCE), {**expected_attenuations, "A": annual}, found_attenuations)


def reference_verdicts(mode: str, levels: pd.Series, metrics: dict[str, float], terminal_attenuation: float,
                       daily: pd.DataFrame) -> None:
    """The reference identity against series 0's daily rows, its acr, std, max_fall and terminal attenuation, and the daily side:
    the daily max_fall, bounding the weekly one, and the daily span, SPAN_DAYS days, whose n_years equals the curve's."""
    daily_levels = last_counted_levels(daily, levels.index)
    ratios = daily_levels.to_numpy() / daily_levels.iat[0]
    KIT.verdict(f"{mode}: the reference level on every curve date is the var3 ratio of series 0's last counted daily "
                f"rows (within {FLOAT32_RTOL} relative)",
                len(levels) == len(SATURDAYS) - CURVE_FIRST_WEEK and np.allclose(levels.to_numpy(), ratios,
                                                                                  rtol=FLOAT32_RTOL, atol=0.0),
                ratios[[0, -1]].tolist(), levels.to_numpy()[[0, -1]].tolist())
    knots = bar_levels("0")
    curve = knots.to_numpy()[CURVE_FIRST_WEEK:]
    expected = {"acr": variation_pct((curve[-1] / curve[0]) ** (1.0 / N_YEARS), 1.0),
                "std": float(np.std(variation_pct(curve[1:], curve[:-1]), ddof=1)) * np.sqrt(ANNUAL_PERIODS),
                "max_fall": -variation_pct(knots.iat[DEEPEST_WEEK], knots.iat[MAX_WEEK]),
                "terminal attenuation": ATTENUATION_RATE * (curve[-1] / curve[0] - 1.0)}
    found = {**{key: metrics.get(key, np.nan) for key in ("acr", "std", "max_fall")}, "terminal attenuation": terminal_attenuation}
    KIT.verdict(f"{mode}: the reference acr, std, max_fall and terminal attenuation",
                all(np.isclose(found[key], value, rtol=TOLERANCE) for key, value in expected.items()), expected, found)
    daily_decline = daily_max_fall(daily, levels.index[0], levels.index[-1])
    decline = {"daily": daily_decline, "weekly": found["max_fall"]}
    KIT.verdict(f"{mode}: the daily max_fall is {expected['max_fall']:.6f} and bounds the weekly one (x (1 + {FLOAT32_RTOL}))",
                np.isclose(daily_decline, expected["max_fall"], rtol=FLOAT32_RTOL)
                and found["max_fall"] <= daily_decline * (1.0 + FLOAT32_RTOL),
                {"daily": expected["max_fall"], "weekly": f"<= {expected['max_fall']:.6f}"}, decline)
    days = (daily_levels.index[-1] - daily_levels.index[0]).days
    span = {"days": days, "daily n_years": days / 365.25, "_n_years": metrics.get("_n_years", np.nan)}
    KIT.verdict(f"{mode}: the daily levels span {SPAN_DAYS} days, and their n_years equals the curve's",
                days == SPAN_DAYS and np.isclose(span["daily n_years"], span["_n_years"], rtol=FLOAT32_RTOL, atol=0.0),
                {"days": SPAN_DAYS, "daily n_years": N_YEARS, "_n_years": N_YEARS}, span)


# =============================================================================
# THE R6 VERDICTS, BUILDERS AND READERS, SHARED WITH THE RECORD DRIVERS
# =============================================================================


def r6_verdicts(mode: str, signal: dict[str, pd.DataFrame], performance: dict[str, pd.DataFrame]) -> None:
    """One mode's R6 figures: the is_signal rows of the signal store, the reference curve's Saturdays and the
    reference metrics of the performance store."""
    count = int(signal["/signals"]["is_signal"].sum())
    KIT.verdict(f"R6 {mode}: {R6_SIGNALS[mode]} is_signal rows", count == R6_SIGNALS[mode], R6_SIGNALS[mode], count)
    dates = reference_levels(performance).index
    found = {"dates": len(dates), "first": str(dates[0].date()), "last": str(dates[-1].date()),
             "saturdays": bool((dates.dayofweek == SATURDAY).all())}
    KIT.verdict(f"R6 {mode}: the reference curve holds {R6_CURVE['dates']} Saturdays from {R6_CURVE['first']} to "
                f"{R6_CURVE['last']}", found == R6_CURVE, R6_CURVE, found)
    metrics = reference_metrics(performance)
    values = {key: metrics.get(key, np.nan) for key in R6_METRICS}
    KIT.verdict(f"R6 {mode}: the reference acr, std, max_fall and fall_dur within {R6_ATOL}",
                all(abs(values[key] - value) <= R6_ATOL for key, value in R6_METRICS.items()), R6_METRICS, values)


def reference_levels(performance: dict[str, pd.DataFrame]) -> pd.Series:
    """The reference curve of a performance store as levels, 1 + variation, indexed by date: the REFERENCE_ID rows of
    /gradient_reference_variation (reference_id, date, variation)."""
    curves = performance["/gradient_reference_variation"]
    rows = curves[curves["reference_id"].astype(str) == REFERENCE_ID]
    return pd.Series(1.0 + rows["variation"].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["date"]))


def reference_metrics(performance: dict[str, pd.DataFrame]) -> dict[str, float]:
    """metric_key -> value of the reference column of a performance store's /gradient_metrics_table (metric_key,
    column_header, value)."""
    table = performance["/gradient_metrics_table"]
    rows = table[table["column_header"] == GRADIENT_REFERENCE_HEADER_TEMPLATE.format(ref_id=REFERENCE_ID)]
    return dict(zip(rows["metric_key"], rows["value"].astype(float)))


def daily_frame(series_id: str) -> pd.DataFrame:
    """The daily rows of one series, Monday to Friday of every week: var3 on the line between the knots through the
    series' affine map, the uncounted Monday flat at the Friday before it; var0 the previous row's var3 + VAR0_STEP on
    the series' own rows; var1 and var2 BAND outside min and max of var0 and var3; var4 = var3; var5 VAR5_BASE + the row
    index, 0 on the zero-var5 Friday and the uncounted Monday; var12 GATE_FAIL in the gate week, else GATE_PASS, var15
    GATE_PASS; series 1 without its gap week."""
    week, step = np.divmod(np.arange(len(SATURDAYS) * ROWS_PER_WEEK), ROWS_PER_WEEK)
    path = KNOTS[week] + (step + 1) * WEEK_SLOPES[week] / ROWS_PER_WEEK
    path[UNCOUNTED_ROW] = path[UNCOUNTED_ROW - 1]
    rows = np.flatnonzero(week != GAP_WEEK) if series_id == GAP_SERIES else np.arange(week.size)
    factor, offset = AFFINE[series_id]
    var3 = factor * path[rows] + offset
    var0 = np.r_[factor * KNOTS[0] + offset, var3[:-1]] + VAR0_STEP
    var1, var2 = np.minimum(var0, var3) - BAND, np.maximum(var0, var3) + BAND
    flat = rows == UNCOUNTED_ROW
    var0[flat] = var1[flat] = var2[flat] = var3[flat]
    var5 = np.where(np.isin(rows, (ZERO_VAR5_ROW, UNCOUNTED_ROW)), 0.0, VAR5_BASE + rows)
    dates = SATURDAYS[week[rows]] - pd.to_timedelta(ROWS_PER_WEEK - step[rows], unit="D")
    return pd.DataFrame({"var0": var0, "var1": var1, "var2": var2, "var3": var3, "var4": var3, "var5": var5,
                         "var12": np.where(week[rows] == GATE_WEEK, GATE_FAIL, GATE_PASS),
                         "var15": np.full(rows.size, GATE_PASS)}, index=dates)


def weekly_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """The bars of a daily frame by the upstream rules D1-D5 for the eight columns read: a row belongs to the bar of the
    Saturday on or after it; a bar uses its counted rows (D3), or all its rows when none counts; var0 is the first used
    row's, var1 the minimum, var2 the maximum, var3, var4, var12 and var15 the last used row's (the row giving var4;
    the fixture holds no NaN), var5 the float64 sum cast to float32."""
    days = (frame.index - EPOCH).days.to_numpy()
    stamps = days + (EPOCH_SATURDAY - days) % 7
    counted = counted_rows(frame)
    used = counted | ~pd.Series(counted).groupby(stamps).transform("any").to_numpy()
    rows = frame[used]
    groups = rows.groupby(stamps[used])
    sums = rows["var5"].astype("float64").groupby(stamps[used]).sum()
    bars = pd.DataFrame({"var0": groups["var0"].first(), "var1": groups["var1"].min(), "var2": groups["var2"].max(),
                         "var3": groups["var3"].last(), "var4": groups["var4"].last(), "var5": sums.astype("float32"),
                         "var12": groups["var12"].last(), "var15": groups["var15"].last()})
    bars.index = EPOCH + pd.to_timedelta(bars.index, unit="D")
    return bars


def daily_max_fall(frame: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> float:
    """The maximum fall from the running maximum in percent of a daily frame's var3 levels from first to last: the last counted
    row's on or before first, then every row's after it up to last, an uncounted row repeating the last counted level,
    which moves neither the running maximum nor a decline; the maximum of -variation_pct(level, running maximum)."""
    dates = pd.DatetimeIndex([first]).append(frame.index[(frame.index > first) & (frame.index <= last)])
    levels = last_counted_levels(frame, dates).to_numpy(dtype=float)
    return float(np.max(-variation_pct(levels, np.maximum.accumulate(levels))))


def last_counted_levels(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """var3 of the last counted row (D3) on or before each date, indexed by that row's date; a date before the first
    counted row raises."""
    counted = frame.loc[counted_rows(frame), "var3"]
    rows = counted.index.searchsorted(dates, side="right") - 1
    if (rows < 0).any():
        KIT.log.error("no counted row on or before %s", dates[rows < 0][0])
        raise ValueError(f"no counted row on or before {dates[rows < 0][0]}")
    return counted.iloc[rows]


def counted_rows(frame: pd.DataFrame) -> np.ndarray:
    """D3: every row counts but one with var5 0 and var0 = var1 = var2 = var3."""
    flat = (frame["var0"] == frame["var1"]) & (frame["var1"] == frame["var2"]) & (frame["var2"] == frame["var3"])
    return ~((frame["var5"] == 0) & flat).to_numpy()


# =============================================================================
# HELPERS
# =============================================================================


def expected_curve(mode: str, dates: pd.DatetimeIndex) -> pd.Series:
    """pure(t) at the dates: the sum over series 1's signals of locked_i x cum_i(t), each locked weight WEIGHT x (1 +
    the locked_j x cum_j of the events ended strictly before its begin); no attenuation precedes a begin."""
    values = trigger_values(SIGNAL_SERIES)
    events = [(SATURDAYS[begin], SATURDAYS[end], plus)
              for begin, end, plus, *_ in expected_triggers(SIGNAL_SERIES, mode)]
    locked = []
    for begin, _, _ in events:
        ended = sum(weight * event_variation(values, *event, event[1])
                    for event, weight in zip(events, locked) if event[1] < begin)
        locked.append(WEIGHT * (1.0 + ended))
    return pd.Series([sum(weight * event_variation(values, *event, date) for event, weight in zip(events, locked))
                      for date in dates], index=dates)


def event_variation(values: pd.Series, begin: pd.Timestamp, end: pd.Timestamp, plus: bool,
                    date: pd.Timestamp) -> float:
    """cum_i at a date: the product of tv(r) / tv(r') (plus) or 2 - tv(r) / tv(r') (minus) over the rows r after the
    begin up to min(date, end), r' the row before r in the series' own index, minus 1; 0 up to the begin, flat after
    the end, and on a date the series lacks the value of its last row before it."""
    rows = values[(values.index >= begin) & (values.index <= min(date, end))].to_numpy()
    ratios = rows[1:] / rows[:-1]
    return float(np.prod(ratios if plus else 2.0 - ratios)) - 1.0


def expected_triggers(series_id: str, mode: str) -> list[tuple[int, int, bool, float, float, bool]]:
    """The hand-derived triggers of one series and mode: (begin week, end week, plus, tv at the begin, tv at the end,
    true), every one true."""
    values = trigger_values(series_id)
    return [(begin, end, plus, float(values[SATURDAYS[begin]]), float(values[SATURDAYS[end]]), True)
            for begin, end, plus, _ in EXPECTED[mode][2][series_id]]


def trigger_values(series_id: str) -> pd.Series:
    """The following_row trigger values of one series' bars by Saturday: var3 + VAR0_STEP, the next bar's var0 (the
    previous row's var3 + VAR0_STEP), and var3 on the last bar."""
    levels = bar_levels(series_id)
    return pd.Series(np.r_[levels.to_numpy()[:-1] + VAR0_STEP, levels.iat[-1]], index=levels.index)


def bar_levels(series_id: str) -> pd.Series:
    """var3 of one series' bars by Saturday, the knots through its affine map: v_k for series 0, 2 v_k - 100 for
    series 1, which lacks its gap week."""
    factor, offset = AFFINE[series_id]
    levels = pd.Series(factor * KNOTS[1:] + offset, index=SATURDAYS)
    return levels.drop(SATURDAYS[GAP_WEEK]) if series_id == GAP_SERIES else levels


def weeks(dates) -> list[int]:
    """The week k of each Saturday S_k, -1 for a date off the calendar."""
    return SATURDAYS.get_indexer(pd.DatetimeIndex(dates)).tolist()


if __name__ == "__main__":
    main()
