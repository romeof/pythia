"""EXP-24: per-series metrics (R-P1, Q8, HYP-16).

variation_pct against hand-computed values first; then an independent recomputation of
variation, duration (row-index difference in the series index) and trials_to_true
(metrics.md:25-40) for (a) the persisted enrichment columns of the performance /signals of a
run copy and (b) calculate_metrics applied to /triggers_0 and /series_0 of its trigger file.
trials_to_true is counted per direction (Q8): plus and minus rows apart, each in
(trigger_begin_dates, series_id) sequence, over the whole /signals table (every row, enriched
or not) and over the series' rows for /triggers_0; NaN and inf probes through the fastmath kernel
_multiplicative_compound and int8 boundary probes for _calculate_per_row_wait_times_jit.
Tolerance: float32 storage (relative 1e-6) and the kernels' reassociated reductions (1e-5).
Planted control: one perturbed persisted variation must be flagged.

Structure:
1. main
2. variation_pct
3. Enrichment and calculate_metrics
4. Kernel probes
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
from src.utils.common.common import variation_pct
from src.utils.common.constants import (
    IS_PLUS_TRIGGER, IS_TRUE, SERIES_ID, TRIGGER_BEGIN_DATES, TRIGGER_END_DATES, TRIGGER_END_VALUES, TRIGGER_START_VALUES,
)
from src.utils.common.schema import AppConfig
from src.utils.performance.perseries_.metrics import _extract_needed_columns, calculate_metrics, get_metric_config
from src.utils.performance.perseries_.metricscalculations import (
    _calculate_per_row_wait_times_jit, _multiplicative_compound, calculate_trials_to_true,
)

EXPERIMENT = "24"
MODES = ("ideal", "realistic")
RTOL, ATOL = 1e-5, 1e-6
PARAMS = harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR)
HAND_CASES = (((110.0, 100.0), 10.0), ((90.0, 100.0), -10.0), ((100.0, 100.0), 0.0), ((150.0, 120.0), 25.0),
              ((0.0, 50.0), -100.0))


def main() -> None:
    """variation_pct, enrichment per mode, calculate_metrics on series 0, kernel probes, control."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    results = {"variation_pct": check_variation_pct()}
    for mode in MODES:
        results[mode] = check_enrichment(mode, args.run_copy / PARAMS / mode)
    results["calculate_metrics"] = check_calculate_metrics(args.run_copy / PARAMS / "realistic", config)
    results["kernels"] = check_kernels()
    check_control(args.run_copy / PARAMS / "realistic")
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# VARIATION_PCT
# =============================================================================


def check_variation_pct() -> list:
    """(current / var0 - 1) * 100 on hand-computed scalars and on an array."""
    found = [(current, var0, variation_pct(current, var0), expected) for (current, var0), expected in HAND_CASES]
    array = variation_pct(np.array([110.0, 90.0]), np.array([100.0, 100.0]))
    CTX.verdict("variation_pct: hand-computed scalars", all(np.isclose(value, expected) for _, _, value, expected in found),
                [expected for _, expected in HAND_CASES], [value for _, _, value, _ in found])
    CTX.verdict("variation_pct: element-wise on arrays", bool(np.allclose(array, [10.0, -10.0])), [10.0, -10.0], array.tolist())
    return found


# =============================================================================
# ENRICHMENT AND CALCULATE_METRICS
# =============================================================================


def expected_columns(rows: pd.DataFrame, index: pd.DatetimeIndex, waits: pd.Series,
                     precision: type = np.float32) -> pd.DataFrame:
    """variation (in the given precision) and duration of one series' rows (metrics.md:25-40), with their
    trials_to_true taken from the per-direction counts of the scope they belong to (Q8)."""
    rows = rows.sort_values(TRIGGER_BEGIN_DATES, kind="mergesort")
    sign = np.where(rows[IS_PLUS_TRIGGER], 1.0, -1.0).astype(precision)
    variation = sign * variation_pct(rows[TRIGGER_END_VALUES].to_numpy(dtype=np.float32).astype(precision),
                                     rows[TRIGGER_START_VALUES].to_numpy(dtype=np.float32).astype(precision))
    duration = [index.get_loc(end) - index.get_loc(begin) for begin, end in zip(rows[TRIGGER_BEGIN_DATES], rows[TRIGGER_END_DATES])]
    return pd.DataFrame({"variation": variation, "duration": np.array(duration, dtype=float),
                         "trials_to_true": waits.loc[rows.index].to_numpy(dtype=float)}, index=rows.index)


def direction_waits(rows: pd.DataFrame) -> pd.Series:
    """Trials to each true event per direction (Q8), indexed like rows: plus and minus rows apart, each counted in
    (trigger_begin_dates, series_id) sequence, series_id when the column exists; 0 on false rows."""
    keys = [TRIGGER_BEGIN_DATES, SERIES_ID] if SERIES_ID in rows.columns else [TRIGGER_BEGIN_DATES]
    waits = pd.Series(0.0, index=rows.index)
    for plus in (True, False):
        count = 1
        for label, value in rows[rows[IS_PLUS_TRIGGER] == plus].sort_values(keys, kind="mergesort")[IS_TRUE].items():
            waits[label] = float(count) if value else 0.0
            count = 1 if value else count + 1
    return waits


def check_enrichment(mode: str, mode_folder: Path) -> dict:
    """The persisted enrichment of /signals against the per-series recomputation (the code's Q8 scope)."""
    with pd.HDFStore(mode_folder / "performance" / "0.h5", mode="r") as store:
        signals = store.get("/signals")
    with pd.HDFStore(mode_folder / "signal" / "0.h5", mode="r") as store:
        indexes = {key[len(harness.SERIES_KEY_PREFIX):]: store.get(key).index for key in sorted(store.keys())
                   if key.startswith(harness.SERIES_KEY_PREFIX)}
    enriched = signals[signals[SERIES_ID].isin(indexes)]
    table_waits = direction_waits(signals)
    misses, compared, deviations = [], 0, []
    for sid, rows in enriched.groupby(SERIES_ID):
        expected = expected_columns(rows, indexes[sid], table_waits)
        exact = expected_columns(rows, indexes[sid], table_waits, np.float64)["variation"].to_numpy()
        deviations += list(np.abs(rows.loc[expected.index, "variation"].to_numpy(dtype=float) - exact))
        for column in expected.columns:
            compared += len(rows)
            bad = ~np.isclose(rows.loc[expected.index, column].to_numpy(dtype=float), expected[column].to_numpy(), rtol=RTOL, atol=ATOL)
            misses += [(sid, column, int(row)) for row in np.flatnonzero(bad)]
    outside = signals[~signals[SERIES_ID].isin(indexes)]
    unenriched = int(outside["variation"].notna().sum())
    compared += len(outside)
    misses += [("outside the signal file", "trials_to_true", int(row)) for row in np.flatnonzero(~np.isclose(
        outside["trials_to_true"].to_numpy(dtype=float), table_waits.loc[outside.index].to_numpy(), rtol=RTOL, atol=ATOL))]
    per_direction = int(table_waits.sum())
    found = {"rows": len(signals), "enriched_rows": len(enriched), "compared_cells": compared, "misses": misses[:10],
             "variation_float32_vs_float64_max_abs": float(max(deviations)),
             "rows_of_series_outside_the_signal_file_with_values": unenriched,
             "trials_sum_per_series": float(enriched["trials_to_true"].sum()), "trials_sum_per_direction": per_direction}
    CTX.log.info("%s enrichment: %s", mode, json.dumps(found, default=str))
    CTX.verdict(f"{mode}: non-vacuity", compared > 0, "> 0 cells", compared)
    CTX.verdict(f"{mode}: persisted variation (float32 arithmetic), duration, and trials_to_true per direction over the "
                f"whole table (Q8) equal the recomputation", not misses, "no miss", misses[:5])
    CTX.log.info("%s: float32 variation arithmetic deviates from float64 by at most %.3g percentage points", mode,
                 found["variation_float32_vs_float64_max_abs"])
    return found


def check_calculate_metrics(mode_folder: Path, config: AppConfig) -> dict:
    """calculate_metrics on /triggers_0 with /series_0 against the recomputation."""
    with pd.HDFStore(mode_folder / "trigger" / "0.h5", mode="r") as store:
        triggers, series = store.get("/triggers_0"), store.get("/series_0")
    needed = _extract_needed_columns(get_metric_config(), config.performance.metrics.display_order_signals)
    produced = calculate_metrics(series, triggers, needed)
    expected = expected_columns(triggers, series.index, direction_waits(triggers)).reset_index(drop=True)
    misses = [(column, int(row)) for column in sorted(needed)
              for row in np.flatnonzero(~np.isclose(produced[column].to_numpy(dtype=float),
                                                          expected[column].to_numpy(), rtol=RTOL, atol=ATOL))]
    CTX.verdict("calculate_metrics on /triggers_0: equals the recomputation, trials_to_true per direction (Q8)",
                not misses and len(produced) > 0,
                f"{len(produced)} rows equal", misses[:5])
    return {"rows": len(produced), "needed": sorted(needed), "misses": misses[:10], "dtypes": {c: str(produced[c].dtype) for c in needed}}


# =============================================================================
# KERNEL PROBES
# =============================================================================


def check_kernels() -> dict:
    """NaN and inf through the fastmath compound; int8 boundaries through the wait-count kernel."""
    probes = {"finite": [10.0, 20.0, -5.0], "nan": [10.0, np.nan], "inf": [10.0, np.inf], "minus_inf": [10.0, -np.inf],
              "at_minus_100": [10.0, -100.0], "below_minus_100": [-150.0]}
    compound = {name: float(_multiplicative_compound(np.array(values, dtype=np.float32))) for name, values in probes.items()}
    reference = {name: float((np.prod(1.0 + np.array(values) / 100.0) - 1.0) * 100.0) for name, values in probes.items()}
    waits = {"empty": calculate_trials_to_true(np.zeros(0, dtype=np.int8)).tolist(),
             "all_false": calculate_trials_to_true(np.zeros(4, dtype=np.int8)).tolist(),
             "all_true": calculate_trials_to_true(np.ones(4, dtype=np.int8)).tolist(),
             "kernel_all_false": _calculate_per_row_wait_times_jit(np.zeros(4, dtype=np.int8)).tolist()}
    CTX.log.info("compound probes %s; plain numpy %s; wait probes %s", compound, reference, waits)
    CTX.verdict("fastmath compound: finite input equals the plain product", np.isclose(compound["finite"], reference["finite"], rtol=RTOL),
                reference["finite"], compound["finite"])
    CTX.verdict("fastmath compound: NaN propagates as in plain numpy", np.isnan(compound["nan"]), "NaN", compound["nan"])
    CTX.verdict("fastmath compound: +inf as in plain numpy", np.isposinf(compound["inf"]), "inf", compound["inf"])
    CTX.verdict("fastmath compound: -inf clamps to -100 %", compound["minus_inf"] == -100.0, -100.0, compound["minus_inf"])
    CTX.verdict("wait-count kernel: empty, all false, all true",
                waits["empty"] == [] and waits["all_false"] == [0.0] * 4 and waits["all_true"] == [1.0] * 4,
                {"empty": [], "all_false": [0.0] * 4, "all_true": [1.0] * 4}, waits)
    return {"compound": compound, "plain": reference, "waits": waits}


def check_control(mode_folder: Path) -> None:
    """A perturbed persisted variation must be flagged."""
    with pd.HDFStore(mode_folder / "performance" / "0.h5", mode="r") as store:
        signals = store.get("/signals")
    with pd.HDFStore(mode_folder / "signal" / "0.h5", mode="r") as store:
        sid = sorted(key for key in store.keys() if key.startswith(harness.SERIES_KEY_PREFIX))[0][len(harness.SERIES_KEY_PREFIX):]
        index = store.get(f"{harness.SERIES_KEY_PREFIX}{sid}").index
    rows = signals[signals[SERIES_ID] == sid].copy()
    rows.iloc[0, rows.columns.get_loc("variation")] += 1.0
    expected = expected_columns(rows, index, direction_waits(signals))
    flagged = not np.allclose(rows.loc[expected.index, "variation"].to_numpy(dtype=float), expected["variation"], rtol=RTOL, atol=ATOL)
    CTX.verdict("control: a perturbed persisted variation is flagged", flagged, "flagged", flagged)


if __name__ == "__main__":
    main()
