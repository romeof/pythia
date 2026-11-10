"""
Gradient-level metric orchestration and configuration.

Contains the top-level orchestrator that computes gradient metrics for the
strategy and each reference, plus helpers and the config loader.

Structure (strict top-down — callee immediately after first caller):
1. GradientColumn                  dataclass
2. compute_gradient_columns        NEW — builds pre/post-attenuation GradientColumn lists
3. build_gradient_metrics_dataframe NEW — wide DataFrame from GradientColumn lists
4. _pair_columns_by_header         private — pairs pre/post columns by header
5. shared_annualization            the (N-1, years) pair every column of 3's DataFrame shares, checked
6. calculate_all_gradient_metrics  orchestrator — called by compute_gradient_columns
7. get_gradient_metric_config      called first inside orchestrator
8. _metrics_by_group               called second inside orchestrator
9. _compute_standalone_metrics     called third inside orchestrator
10. _compute_semi_deviation        called by 9 (standalone) and 11 (relative)
11. _compute_relative_metrics      called fourth inside orchestrator
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ...common.config import load_json_config
from ...common.constants import (
    METRIC_ACR,
    METRIC_BELOW_MAX_DUR,
    METRIC_REF_SNR,
    METRIC_REF_SNR_DW,
    METRIC_EXV,
    METRIC_EXV_DW,
    METRIC_SNR,
    METRIC_SNR_DW,
    METRIC_MAX_FALL,
    METRIC_FALL_DUR,
    METRIC_RHO,
    METRIC_SENS,
    METRIC_SENS_DW,
    METRIC_STD,
    METRIC_STD_DW,
    METRIC_GROUP_STANDALONE,
    METRIC_GROUP_RELATIVE,
    GRADIENT_STRATEGY_HEADER,
    GRADIENT_REFERENCE_HEADER_TEMPLATE,
    FROM_DICT_ORIENT,
)
from .metricscalculations import (
    _cumulative_pct_to_periodic_variations,
    _compute_annualization_factor,
    _compute_gradient_and_running_max,
    _calculate_acr,
    _calculate_max_fall,
    _calculate_fall_dur,
    _calculate_below_max_dur,
)

# Set of all known metric keys — validated against metrics.json at import time
_ALL_METRIC_KEYS: Set[str] = {
    METRIC_ACR, METRIC_STD, METRIC_STD_DW,
    METRIC_SNR, METRIC_SNR_DW,
    METRIC_MAX_FALL, METRIC_FALL_DUR, METRIC_BELOW_MAX_DUR,
    METRIC_RHO, METRIC_SENS, METRIC_SENS_DW,
    METRIC_EXV, METRIC_EXV_DW, METRIC_REF_SNR, METRIC_REF_SNR_DW,
}


# =============================================================================
# RETURN TYPE
# =============================================================================


@dataclass(frozen=True)
class GradientColumn:
    """One column in the gradient metrics summary table.

    header: column header string (GRADIENT_STRATEGY_HEADER or "reference {ref_id}").
    values: metric_key → float; absent key = blank cell in the renderer.
             fall_dur / below_max_dur are int-valued but stored as float (int formatting
             is decided by the renderer via metric_config["format"]).
    n_intervals: N-1 intervals for this column's standalone series (numerator of m).
    n_years: Observation span in years (denominator of m). m = n_intervals / n_years.
    """

    header: str
    values: Dict[str, float]
    n_intervals: int = 0
    n_years: float = 0.0


# =============================================================================
# COLUMN PAIR BUILDERS
# =============================================================================


def compute_gradient_columns(
    gradient_series: Optional[pd.Series],
    attenuation_adjusted: Optional[pd.Series],
    reference_variation_map: Optional[Dict[str, pd.Series]],
    reference_terminal_attenuations: Optional[Dict[str, float]],
    selection: List[str],
    mpv: float = 0.0,
) -> Tuple[List["GradientColumn"], Optional[List["GradientColumn"]]]:
    """Build pre-attenuation and post-attenuation GradientColumn lists.

    Converts decimal cumulative series to percent (×100) before metric
    computation. The post-attenuation path takes the strategy from attenuation_adjusted,
    which already carries every attenuation, and lowers each reference's iloc[-1]
    by its terminal attenuation. When attenuation is inactive (`attenuation_adjusted is None`),
    the post-attenuation list is returned as None.

    Args:
        gradient_series: Pre-attenuation cumulative variation (decimal, e.g. 0.42 = 42%).
        attenuation_adjusted: Attenuation-adjusted cumulative variation (decimal), or None.
        reference_variation_map: Pre-attenuation reference cumulative variation (decimal).
        reference_terminal_attenuations: Terminal attenuation amount per reference (decimal),
            applied once on its total accumulated increase.
        selection: Sequence of metric keys (validated at config-init time).
        mpv: Minimal positive variation rate (decimal). Default 0.

    Returns:
        (preattenuation_columns, postattenuation_columns). postattenuation_columns is None when
        attenuation_adjusted is None (no attenuation active).
    """
    pct_grad = _series_to_pct(gradient_series)
    pct_ref_map = _ref_map_to_pct(reference_variation_map)

    preattenuation_columns = calculate_all_gradient_metrics(
        pct_grad, pct_ref_map, selection=selection, mpv=mpv
    )

    if attenuation_adjusted is None:
        return preattenuation_columns, None

    attenuated_ref_map = _apply_terminal_attenuations_to_map(
        reference_variation_map, reference_terminal_attenuations
    )
    postattenuation_columns = calculate_all_gradient_metrics(
        _series_to_pct(attenuation_adjusted),
        _ref_map_to_pct(attenuated_ref_map),
        selection=selection,
        mpv=mpv,
    )
    return preattenuation_columns, postattenuation_columns


def build_gradient_metrics_dataframe(
    preattenuation_columns: List["GradientColumn"],
    postattenuation_columns: Optional[List["GradientColumn"]],
    display_order_gradient: List[str],
) -> pd.DataFrame:
    """Build a wide numeric DataFrame from pre-attenuation and post-attenuation GradientColumn lists.

    Rows correspond to `display_order_gradient` items; columns correspond to
    column headers in `preattenuation_columns` (strategy + references in sorted order).
    Metrics with `postattenuation_source` in metrics.json take their value from the
    post-attenuation column; all others take from the pre-attenuation column.
    Relative metrics (group == "relative") are NaN for the strategy column.

    Args:
        preattenuation_columns: Pre-attenuation GradientColumn list (strategy + references).
        postattenuation_columns: Post-attenuation GradientColumn list or None when no attenuation.
        display_order_gradient: Ordered metric keys to render as rows (index).

    Returns:
        DataFrame with index name "metric_key", columns = column headers,
        dtype float32. Shape: len(display_order_gradient) × len(preattenuation_columns).
    """
    if not preattenuation_columns:
        return pd.DataFrame()

    metric_config = get_gradient_metric_config()
    paired = _pair_columns_by_header(preattenuation_columns, postattenuation_columns)

    col_headers = [pre.header for pre, _ in paired]
    rows: Dict[str, Dict[str, float]] = {}

    for key in display_order_gradient:
        cfg = metric_config.get(key, {})
        postattenuation_src: Optional[str] = cfg.get("postattenuation_source")
        row: Dict[str, float] = {}
        for pre_col, post_col in paired:
            if postattenuation_src is not None:
                # acr_after_attenuation: take 'acr' from post-attenuation column (NaN if no attenuation)
                val = post_col.values.get(postattenuation_src) if post_col is not None else float("nan")
            else:
                val = pre_col.values.get(key)
            row[pre_col.header] = float("nan") if val is None else float(val)
        rows[key] = row

    # Sentinel rows for the renderer: annualization info per column.
    # Keys start with "_" so they are never in display_order_gradient and are
    # handled separately by shared_annualization. Stored in the
    # DataFrame (and therefore in HDF5) so the replot path gets them for free.
    rows["_n_intervals"] = {pre.header: float(pre.n_intervals) for pre, _ in paired}
    rows["_n_years"] = {pre.header: float(pre.n_years) for pre, _ in paired}

    df = pd.DataFrame.from_dict(rows, orient=FROM_DICT_ORIENT, columns=col_headers)
    df.index.name = "metric_key"
    return df.astype("float32")


def _pair_columns_by_header(
    preattenuation_columns: List["GradientColumn"],
    postattenuation_columns: Optional[List["GradientColumn"]],
) -> List[Tuple["GradientColumn", Optional["GradientColumn"]]]:
    """Match each pre-attenuation column to its post-attenuation counterpart by header.

    Returns None for post-attenuation when no attenuation is active — either postattenuation_columns
    is None, or it is the same object as preattenuation_columns (no attenuation computed).
    """
    if postattenuation_columns is None or postattenuation_columns is preattenuation_columns:
        return [(col, None) for col in preattenuation_columns]
    post_by_header = {col.header: col for col in postattenuation_columns}
    return [(col, post_by_header.get(col.header)) for col in preattenuation_columns]


def shared_annualization(gradient_metrics_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """The (N-1, years) pair every column shares, read from the sentinel rows build_gradient_metrics_dataframe
    writes; None when the DataFrame lacks either row (an empty DataFrame has neither).

    All columns must share identical annualization parameters — they are derived from the same global index. Any
    discrepancy signals a data inconsistency and raises ValueError before a misleading table or a persisted result.
    """
    if "_n_intervals" not in gradient_metrics_df.index or "_n_years" not in gradient_metrics_df.index:
        return None
    col_headers = list(gradient_metrics_df.columns)
    n_intervals = gradient_metrics_df.loc["_n_intervals"]
    n_years = gradient_metrics_df.loc["_n_years"]
    ref_n = float(n_intervals[col_headers[0]])
    ref_y = float(n_years[col_headers[0]])
    for header in col_headers[1:]:
        col_n = float(n_intervals[header])
        col_y = float(n_years[header])
        # N-1 counts are integers stored as float32; nearest-integer slack tolerance.
        if abs(col_n - ref_n) > 0.5 or abs(col_y - ref_y) > 1e-4:
            raise ValueError(
                f"Annualization parameters differ across columns: "
                f"'{col_headers[0]}' has N-1={int(ref_n)}, years={ref_y:.4f}; "
                f"'{header}' has N-1={int(col_n)}, years={col_y:.4f}. "
                "All series must share the same observation period."
            )
    return ref_n, ref_y


def _series_to_pct(series: Optional[pd.Series]) -> Optional[pd.Series]:
    """Scale a decimal cumulative series to percent (×100) for metric calculations."""
    return series * 100.0 if series is not None else None


def _ref_map_to_pct(
    ref_map: Optional[Dict[str, pd.Series]],
) -> Optional[Dict[str, pd.Series]]:
    """Scale all series in a reference map from decimal to percent."""
    return (
        {k: v * 100.0 for k, v in ref_map.items() if v is not None}
        if ref_map
        else None
    )


def _apply_terminal_attenuations_to_map(
    ref_map: Optional[Dict[str, pd.Series]],
    attenuations: Optional[Dict[str, float]],
) -> Optional[Dict[str, pd.Series]]:
    """Return reference map with each series' last point reduced by its attenuation amount."""
    if not ref_map or not attenuations:
        return ref_map
    return {
        ref_id: _apply_terminal_attenuation(s, attenuations.get(ref_id, 0.0))
        for ref_id, s in ref_map.items()
    }


def _apply_terminal_attenuation(
    series: Optional[pd.Series], attenuation: float
) -> Optional[pd.Series]:
    """Return a copy of series with iloc[-1] reduced by attenuation (for metrics only)."""
    if series is None or len(series) == 0 or attenuation <= 0.0:
        return series
    out = series.copy()
    out.iloc[-1] = float(out.iloc[-1]) - attenuation
    return out


# =============================================================================
# ORCHESTRATOR
# =============================================================================


def calculate_all_gradient_metrics(
    gradient_series: Optional[pd.Series],
    reference_variation_map: Optional[Dict[str, pd.Series]],
    selection: List[str],
    mpv: float = 0.0,
) -> List[GradientColumn]:
    """Compute selected gradient metrics for the strategy and each reference.

    Args:
        gradient_series: Cumulative variation (%) indexed by observed date.
        reference_variation_map: Dict mapping reference IDs to cumulative
            variation series (%). Used for standalone and relative metrics.
        selection: Ordered list of metric keys from display_order_gradient.
            Only listed metrics are computed. Validated at config-init time
            (unknown keys raise ValueError before this function is called).
        mpv: Minimal positive variation rate (decimal). Default 0.

    Returns:
        List of GradientColumn, one strategy column then one per reference
        (sorted by reference ID). An empty list is returned when there is
        insufficient data.
    """
    if gradient_series is None or len(gradient_series) < 2:
        return []

    metric_config = get_gradient_metric_config()
    standalone_all, relative_all = _metrics_by_group(metric_config)
    selection_set = set(selection)
    standalone_wanted: Set[str] = selection_set & standalone_all
    relative_wanted: Set[str] = selection_set & relative_all

    cumulative_pct_strat = gradient_series.values.astype(np.float64)
    idx_strat = gradient_series.index

    columns: List[GradientColumn] = []

    # Strategy column — standalone metrics only
    _, n_years_strat = _compute_annualization_factor(idx_strat)
    columns.append(GradientColumn(
        header=GRADIENT_STRATEGY_HEADER,
        values=_compute_standalone_metrics(
            cumulative_pct_strat, idx_strat, standalone_wanted, mpv
        ),
        n_intervals=len(idx_strat) - 1,
        n_years=n_years_strat,
    ))

    ref_map = reference_variation_map or {}
    for ref_id in sorted(ref_map.keys()):
        ref_series = ref_map[ref_id]
        if ref_series is None or len(ref_series) < 2:
            continue
        cumulative_pct_ref = ref_series.values.astype(np.float64)

        # Standalone metrics on the reference itself
        ref_vals: Dict[str, float] = _compute_standalone_metrics(
            cumulative_pct_ref, ref_series.index, standalone_wanted, mpv
        )

        # Strategy-vs-this-reference relative metrics
        if relative_wanted:
            common_idx = idx_strat.intersection(ref_series.index)
            if len(common_idx) >= 2:
                strat_aligned = gradient_series.loc[common_idx].values.astype(np.float64)
                ref_aligned = ref_series.loc[common_idx].values.astype(np.float64)
                r_s = _cumulative_pct_to_periodic_variations(strat_aligned)
                r_ref = _cumulative_pct_to_periodic_variations(ref_aligned)
                m_aligned, _ = _compute_annualization_factor(common_idx)
                ref_vals.update(_compute_relative_metrics(
                    r_s, r_ref, m_aligned, relative_wanted, mpv
                ))

        _, n_years_ref = _compute_annualization_factor(ref_series.index)
        columns.append(GradientColumn(
            header=GRADIENT_REFERENCE_HEADER_TEMPLATE.format(ref_id=ref_id),
            values=ref_vals,
            n_intervals=len(ref_series.index) - 1,
            n_years=n_years_ref,
        ))

    return columns


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================


def get_gradient_metric_config() -> Dict:
    """Load metric configuration from metrics.json located alongside this module.

    Also asserts that every METRIC_* constant in _ALL_METRIC_KEYS appears in
    the loaded JSON (fail-fast: detects key drift between constants.py and JSON).
    """
    config = load_json_config(__file__)
    missing = _ALL_METRIC_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"gradient_/metrics.json is missing keys that are defined in "
            f"constants.py: {sorted(missing)}. Update metrics.json."
        )
    return config


# =============================================================================
# GROUP PARTITIONER
# =============================================================================


def _metrics_by_group(metric_config: Dict) -> Tuple[Set[str], Set[str]]:
    """Partition metric keys into (standalone, relative) per metrics.json `group` field."""
    standalone = {
        k for k, v in metric_config.items() if v["group"] == METRIC_GROUP_STANDALONE
    }
    relative = {
        k for k, v in metric_config.items() if v["group"] == METRIC_GROUP_RELATIVE
    }
    return standalone, relative


# =============================================================================
# STANDALONE METRICS HELPER
# =============================================================================


def _compute_standalone_metrics(
    cumulative_pct: np.ndarray,
    index: pd.DatetimeIndex,
    wanted: Set[str],
    mpv: float,
) -> Dict[str, float]:
    """Compute standalone gradient metrics — only those in `wanted`.

    Shared-computation groups:
    - acr: uses only cumulative_pct[-1], no periodic variations needed.
    - Dispersion group (snr, snr_dw, std, std_dw): periodic variations computed
      once; std_r and sigma_d each computed at most once and reused.
    - Fall group (max_fall, fall_dur, below_max_dur): gradient_arr + running_max
      computed once and shared.
    """
    out: Dict[str, float] = {}
    if not wanted:
        return out

    m, n_years = _compute_annualization_factor(index)
    sqrt_m = math.sqrt(m) if m > 0 else 0.0

    if METRIC_ACR in wanted:
        out[METRIC_ACR] = _calculate_acr(cumulative_pct[-1] / 100.0, n_years)

    # Dispersion group — periodic variations computed only when group is needed
    dispersion_wanted = wanted & {METRIC_SNR, METRIC_SNR_DW, METRIC_STD, METRIC_STD_DW}
    if dispersion_wanted:
        variations = _cumulative_pct_to_periodic_variations(cumulative_pct)
        if len(variations) >= 2:
            need_std_r  = bool(wanted & {METRIC_SNR, METRIC_STD})
            need_sigma_d = bool(wanted & {METRIC_SNR_DW, METRIC_STD_DW})
            need_mean   = bool(wanted & {METRIC_SNR, METRIC_SNR_DW})
            mean_r: float = float(np.mean(variations)) if need_mean else 0.0
            std_r: float  = 0.0
            sigma_d: float = 0.0
            if need_std_r:
                std_r = float(np.std(variations, ddof=1))
            if need_sigma_d:
                sigma_d = _compute_semi_deviation(variations, mpv)
            if METRIC_SNR in wanted:
                out[METRIC_SNR] = (
                    (mean_r - mpv) / std_r * sqrt_m if std_r > 1e-15 else float("nan")
                )
            if METRIC_SNR_DW in wanted:
                out[METRIC_SNR_DW] = (
                    (mean_r - mpv) / sigma_d * sqrt_m if sigma_d > 1e-15 else float("nan")
                )
            if METRIC_STD in wanted:
                out[METRIC_STD] = std_r * sqrt_m * 100.0
            if METRIC_STD_DW in wanted:
                out[METRIC_STD_DW] = sigma_d * sqrt_m * 100.0

    # Fall group — gradient_arr + running_max computed once and shared
    if wanted & {METRIC_MAX_FALL, METRIC_FALL_DUR, METRIC_BELOW_MAX_DUR}:
        gradient_arr, running_max = _compute_gradient_and_running_max(cumulative_pct)
        if METRIC_MAX_FALL in wanted:
            out[METRIC_MAX_FALL] = _calculate_max_fall(gradient_arr, running_max)
        if METRIC_FALL_DUR in wanted:
            out[METRIC_FALL_DUR] = float(_calculate_fall_dur(gradient_arr, running_max))
        if METRIC_BELOW_MAX_DUR in wanted:
            out[METRIC_BELOW_MAX_DUR] = float(_calculate_below_max_dur(gradient_arr, running_max))

    return out


def _compute_semi_deviation(arr: np.ndarray, threshold: float = 0.0) -> float:
    """Downside semi-deviation of arr relative to threshold.

    Returns sqrt(mean(min(arr - threshold, 0)²)).
    """
    downside = np.minimum(arr - threshold, 0.0)
    return math.sqrt(float(np.mean(downside * downside)))


# =============================================================================
# RELATIVE METRICS HELPER
# =============================================================================


def _compute_relative_metrics(
    r_s: np.ndarray,
    r_ref: np.ndarray,
    m: float,
    wanted: Set[str],
    mpv: float,
) -> Dict[str, float]:
    """Compute strategy-vs-reference metrics via three independent caching paths.

    Path 1 — covariance path (rho, sens, exv):
        One np.cov 2×2 call gives var_s, var_ref, cov_s_ref.
        mean_s / mean_ref computed only when METRIC_EXV is wanted.
    Path 2 — excess path (ref_snr, ref_snr_dw):
        excess = r_s - r_ref allocated once; shared by both metrics.
    Path 3 — downside path (sens_dw, exv_dw):
        One mask + one np.cov on the downside subset.
    Paths are fully independent — ref_snr-only selection does NOT trigger np.cov.
    """
    out: Dict[str, float] = {}
    if not wanted or len(r_s) < 2:
        return out

    sqrt_m = math.sqrt(m)

    # --- Path 1: covariance path (rho, sens, exv) ---
    cov_wanted = wanted & {METRIC_RHO, METRIC_SENS, METRIC_EXV}
    if cov_wanted:
        # np.cov 2×2: [0,0]=var(r_s), [1,1]=var(r_ref), [0,1]=cov(r_s,r_ref)
        cov_matrix = np.cov(r_s, r_ref, ddof=1)
        var_s     = float(cov_matrix[0, 0])
        var_ref   = float(cov_matrix[1, 1])
        cov_s_ref = float(cov_matrix[0, 1])
        if METRIC_RHO in wanted:
            out[METRIC_RHO] = (
                cov_s_ref / math.sqrt(var_s * var_ref)
                if var_s > 1e-30 and var_ref > 1e-30
                else float("nan")
            )
        if wanted & {METRIC_SENS, METRIC_EXV}:
            sens_val = cov_s_ref / var_ref if var_ref > 1e-30 else float("nan")
            if METRIC_SENS in wanted:
                out[METRIC_SENS] = sens_val
            if METRIC_EXV in wanted and not math.isnan(sens_val):
                # mean_s / mean_ref allocated only when exv is wanted
                mean_s = float(np.mean(r_s))
                mean_ref = float(np.mean(r_ref))
                out[METRIC_EXV] = (
                    mean_s - sens_val * mean_ref - mpv * (1.0 - sens_val)
                ) * m * 100.0

    # --- Path 2: excess path (ref_snr, ref_snr_dw) — independent of Path 1 ---
    if wanted & {METRIC_REF_SNR, METRIC_REF_SNR_DW}:
        excess  = r_s - r_ref
        mean_ex = float(np.mean(excess))
        if METRIC_REF_SNR in wanted:
            std_ex = float(np.std(excess, ddof=1))
            out[METRIC_REF_SNR] = (
                mean_ex / std_ex * sqrt_m if std_ex > 1e-15 else float("nan")
            )
        if METRIC_REF_SNR_DW in wanted:
            sigma_d_ex = _compute_semi_deviation(excess)
            out[METRIC_REF_SNR_DW] = (
                mean_ex / sigma_d_ex * sqrt_m if sigma_d_ex > 1e-15 else float("nan")
            )

    # --- Path 3: downside subset path (sens_dw, exv_dw) ---
    if wanted & {METRIC_SENS_DW, METRIC_EXV_DW}:
        mask = r_ref < mpv
        if int(np.sum(mask)) >= 2:
            r_s_dw     = r_s[mask]
            r_ref_dw   = r_ref[mask]
            cov_dw     = np.cov(r_s_dw, r_ref_dw, ddof=1)
            var_ref_dw = float(cov_dw[1, 1])
            sens_dw_val = (
                float(cov_dw[0, 1]) / var_ref_dw if var_ref_dw > 1e-30 else float("nan")
            )
            if METRIC_SENS_DW in wanted:
                out[METRIC_SENS_DW] = sens_dw_val
            if METRIC_EXV_DW in wanted and not math.isnan(sens_dw_val):
                # Uses downside-subset means — corrected semantics (previously
                # used full-sample means mixed with downside-subset sens_dw).
                mean_s_dw = float(np.mean(r_s_dw))
                mean_ref_dw = float(np.mean(r_ref_dw))
                out[METRIC_EXV_DW] = (
                    mean_s_dw - sens_dw_val * mean_ref_dw - mpv * (1.0 - sens_dw_val)
                ) * m * 100.0
        else:
            if METRIC_SENS_DW in wanted:
                out[METRIC_SENS_DW] = float("nan")
            if METRIC_EXV_DW in wanted:
                out[METRIC_EXV_DW] = float("nan")

    return out
