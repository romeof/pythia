"""
Gradient plot reconstruction from a persisted performance HDF5 file.

Reads all gradient plot data written by Performance._persist_gradient_plot_data
and calls plot_gradient with precomputed tables to reproduce the gradient figure.

Structure (strict top-down — callee immediately after first caller):
1. GradientReplotParams      typed dataclass for scalar/list params
2. replot_gradient_from_hdf5 starting point
3. load_gradient_replot_params typed load of /gradient_params + /gradient_param_sequences
4. rebuild_signals_data_dict reassemble signals_data from /gradient_signal_segments
5. _load_reference_variation_map load /gradient_reference_variation
6. _load_terminal_attenuations     load /gradient_terminal_attenuations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..common.common import tall_to_wide
from ..common.constants import (
    GRADIENT_PARAM_SEQ_DISPLAY_GRADIENT,
    GRADIENT_PARAM_SEQ_DISPLAY_SIGNALS,
    GRADIENT_PARAM_SEQ_SIGNAL_PLOTS_WHITELIST,
    GRADIENT_STRATEGY_SENTINEL,
    GROUP_ID,
    IS_SIGNAL,
    HDF5_GRADIENT_FILTERED_SEQUENCE,
    HDF5_GRADIENT_METRICS_TABLE,
    HDF5_GRADIENT_PARAM_SEQUENCES,
    HDF5_GRADIENT_PARAMS,
    HDF5_GRADIENT_REFERENCE_VARIATION,
    HDF5_GRADIENT_SIGNAL_SEGMENTS,
    HDF5_GRADIENT_TERMINAL_ATTENUATIONS,
    HDF5_SIGNALS_BREAKDOWN_TABLE,
    SERIES_ID,
)
from ..common.hdf5manager import hdf5_manager
from .plots.gradient import plot_gradient

logger = logging.getLogger(__name__)


# =============================================================================
# TYPED PARAMS DATACLASS
# =============================================================================


@dataclass(frozen=True)
class GradientReplotParams:
    """Typed scalar and list parameters required to reconstruct the gradient figure.

    All list-valued params are persisted in /gradient_param_sequences; all
    scalars are in /gradient_params. Loaded via load_gradient_replot_params.
    """

    weight: float
    concurrent_weight_limit: float
    variation_method: str
    yearly_attenuation_fraction: float
    attenuation_month: int
    boundary_deduction_pct: float
    mpv: float
    max_signal_plots: int
    display_order_signals: List[str] = field(default_factory=list)
    display_order_gradient: List[str] = field(default_factory=list)
    signal_plots_whitelist: List[str] = field(default_factory=list)


# =============================================================================
# STARTING POINT
# =============================================================================


def replot_gradient_from_hdf5(
    hdf5_path: Path,
    save_path: Optional[Path] = None,
    close_figure: bool = True,
) -> None:
    """Reconstruct and save the gradient figure from a performance HDF5 file.

    Derives the output directory from the HDF5 path via PlotBase.plots_dir_from_output.
    Writes `gradient_hd5.png` to `.../performance/plots/gradient/` when save_path
    is not explicitly supplied.

    Args:
        hdf5_path: Path to the performance output HDF5 (e.g. .../performance/0.h5).
        save_path: Override output path. Derived from hdf5_path when None.
        close_figure: Whether to close the figure after saving. Default True.
            Pass False in tests to keep the figure alive for artist inspection.
    """
    from .plots.common import PlotBase

    hdf5_path = Path(hdf5_path)
    if save_path is None:
        plots_dir = PlotBase.plots_dir_from_output(hdf5_path)
        gradient_dir = plots_dir / "gradient"
        gradient_dir.mkdir(parents=True, exist_ok=True)
        save_path = gradient_dir / "gradient_hd5.png"

    with hdf5_manager.open_store(hdf5_path, "r") as store:
        # Pre-load key set once (O(1) lookups) — used to guard every optional read
        # so absent keys produce no error/warning log for legitimately missing data
        # (no signals → no /gradient_signal_segments; no attenuation → no /total_variation_attenuated, etc.)
        keys_set = hdf5_manager.get_keys_set(store)

        def _read_if_present(key: str) -> Optional[pd.DataFrame]:
            """Read key only when it exists in the store; silently return None otherwise."""
            norm = key if key.startswith("/") else f"/{key}"
            return hdf5_manager.get_dataframe_fast(store, key) if norm in keys_set else None

        params = load_gradient_replot_params(store, keys_set)

        # Load /signals + /gradient_filtered_sequence to get filtered_signal_sequence
        signals_df = hdf5_manager.get_dataframe(store, "/signals")
        filtered_seq_df = _read_if_present(HDF5_GRADIENT_FILTERED_SEQUENCE)
        if filtered_seq_df is not None and signals_df is not None:
            # Filter to IS_SIGNAL rows only: signals_df has one row per (group_id, series_id)
            # pair; merging without this filter fans out to multiple rows per group.
            signal_rows = signals_df[signals_df[IS_SIGNAL] == True] if IS_SIGNAL in signals_df.columns else signals_df  # noqa: E712
            filtered_seq_df = filtered_seq_df.sort_values("sequence_index").reset_index(drop=True)
            filtered_signal_sequence = (
                filtered_seq_df[[GROUP_ID]]
                .merge(signal_rows, on=GROUP_ID, how="left")
                .reset_index(drop=True)
            )
        else:
            filtered_signal_sequence = pd.DataFrame()

        # Rebuild signals_data from /gradient_signal_segments (absent when no signals passed filter)
        segments_df = _read_if_present(HDF5_GRADIENT_SIGNAL_SEGMENTS)
        signals_data = rebuild_signals_data_dict(segments_df, filtered_signal_sequence)

        # Load gradient series
        tv_df = _read_if_present("/total_variation")
        gradient_series: Optional[pd.Series] = None
        if tv_df is not None and "total_variation" in tv_df.columns:
            gradient_series = pd.Series(
                tv_df["total_variation"].values.astype(float),
                index=pd.to_datetime(tv_df["date"]),
            )

        # Load attenuation-adjusted series (only present when yearly_attenuation_fraction > 0)
        attenuation_adjusted: Optional[pd.Series] = None
        if params.yearly_attenuation_fraction > 0:
            tld_df = _read_if_present("/total_variation_attenuated")
            if tld_df is not None and "total_variation_attenuated" in tld_df.columns:
                attenuation_adjusted = pd.Series(
                    tld_df["total_variation_attenuated"].values.astype(float),
                    index=pd.to_datetime(tld_df["date"]),
                )

        reference_variation_map = _load_reference_variation_map(store, keys_set)
        # The curve index: /total_variation's dates, else the reference curves' (each spans the whole index)
        global_idx = (gradient_series.index if gradient_series is not None else pd.DatetimeIndex(
            sorted({date for curve in reference_variation_map.values() for date in curve.index})))
        terminal_attenuation_amount, reference_terminal_attenuations = _load_terminal_attenuations(store, keys_set)

        # Load precomputed summary tables — tall → wide via pivot (absent when no signals)
        gradient_metrics_df: Optional[pd.DataFrame] = None
        gm_tall = _read_if_present(HDF5_GRADIENT_METRICS_TABLE)
        if gm_tall is not None:
            gradient_metrics_df = tall_to_wide(
                gm_tall, "metric_key", "column_header", "value"
            )

        signals_breakdown_df: Optional[pd.DataFrame] = None
        sb_tall = _read_if_present(HDF5_SIGNALS_BREAKDOWN_TABLE)
        if sb_tall is not None:
            signals_breakdown_df = tall_to_wide(
                sb_tall, "row_key", "column_header", "value"
            )

        # Load strategy name
        sn_df = _read_if_present("/strategy_name")
        strategy_name: Optional[str] = None
        if sn_df is not None and "strategy_name" in sn_df.columns:
            strategy_name = str(sn_df["strategy_name"].iloc[0])

    plot_gradient(
        signals_data=signals_data,
        filtered_signal_sequence=filtered_signal_sequence,
        gradient_series=gradient_series,
        global_idx=global_idx,
        reference_variation_map=reference_variation_map,
        strategy_name=strategy_name,
        emit_signal_plots=False,
        signal_plots_whitelist=params.signal_plots_whitelist,
        max_signal_plots=params.max_signal_plots,
        save_path=str(save_path),
        show=False,
        display_order_signals=params.display_order_signals,
        display_order_gradient=params.display_order_gradient,
        attenuation_adjusted_gradient=attenuation_adjusted,
        terminal_attenuation_amount=terminal_attenuation_amount,
        reference_terminal_attenuations=reference_terminal_attenuations,
        mpv=params.mpv,
        gradient_metrics_table=gradient_metrics_df,
        signals_breakdown_table=signals_breakdown_df,
        close_figure=close_figure,
    )
    logger.info("Gradient figure written to %s", save_path)


# =============================================================================
# PARAMETER LOADER
# =============================================================================


def load_gradient_replot_params(store: Any, keys_set: set) -> GradientReplotParams:
    """Load and validate gradient replot parameters from a store in read mode.

    Reads /gradient_params (scalar row) and /gradient_param_sequences (list
    values, absent when every list is empty). Fails fast with a descriptive error
    when /gradient_params is absent.

    Args:
        store: HDFStore in read mode.
        keys_set: The store's keys, read once by the caller.

    Returns:
        Validated GradientReplotParams dataclass.

    Raises:
        KeyError: If /gradient_params is absent from the store.
        ValueError: If required scalar fields are missing from the params row.
    """
    params_df = (
        hdf5_manager.get_dataframe_fast(store, HDF5_GRADIENT_PARAMS)
        if HDF5_GRADIENT_PARAMS in keys_set else None
    )
    if params_df is None or params_df.empty:
        raise KeyError(
            f"{HDF5_GRADIENT_PARAMS} is absent from the HDF5 store. "
            "Regenerate the performance output with the current pipeline version."
        )

    row = params_df.iloc[0]

    def _req(col: str) -> Any:
        if col not in row.index:
            raise ValueError(
                f"Required param column {col!r} missing from {HDF5_GRADIENT_PARAMS}. "
                "Regenerate the performance output."
            )
        return row[col]

    # Load list-valued params from /gradient_param_sequences (absent when every list is empty)
    seq_df = (
        hdf5_manager.get_dataframe_fast(store, HDF5_GRADIENT_PARAM_SEQUENCES)
        if HDF5_GRADIENT_PARAM_SEQUENCES in keys_set else None
    )
    display_signals: List[str] = []
    display_gradient: List[str] = []
    signal_plots_whitelist: List[str] = []
    if seq_df is not None and not seq_df.empty:
        for kind, target in [
            (GRADIENT_PARAM_SEQ_DISPLAY_SIGNALS, display_signals),
            (GRADIENT_PARAM_SEQ_DISPLAY_GRADIENT, display_gradient),
            (GRADIENT_PARAM_SEQ_SIGNAL_PLOTS_WHITELIST, signal_plots_whitelist),
        ]:
            subset = seq_df[seq_df["kind"] == kind].sort_values("sequence_index")
            target.extend(str(v) for v in subset["name"].tolist())

    return GradientReplotParams(
        weight=float(_req("weight")),
        concurrent_weight_limit=float(_req("concurrent_weight_limit")),
        variation_method=str(_req("variation_method")),
        yearly_attenuation_fraction=float(_req("yearly_attenuation_fraction")),
        attenuation_month=int(_req("attenuation_month")),
        boundary_deduction_pct=float(_req("boundary_deduction_pct")),
        mpv=float(_req("mpv")),
        max_signal_plots=int(_req("max_signal_plots")),
        display_order_signals=display_signals,
        display_order_gradient=display_gradient,
        signal_plots_whitelist=signal_plots_whitelist,
    )


# =============================================================================
# DATA REASSEMBLY HELPERS
# =============================================================================


def rebuild_signals_data_dict(
    segments_df: Optional[pd.DataFrame],
    filtered_signal_sequence: pd.DataFrame,
) -> Dict[int, Dict[str, Any]]:
    """Reassemble signals_data dict from /gradient_signal_segments.

    Single groupby over segments_df — no per-group rescans. Metadata columns
    (series_id, is_plus_trigger, is_true, trigger_start_values, trigger_end_values)
    are joined from filtered_signal_sequence by group_id.

    Args:
        segments_df: DataFrame with (group_id, date, pct) columns or None.
        filtered_signal_sequence: The signal sequence DataFrame (joined from /signals
            and /gradient_filtered_sequence).

    Returns:
        Dict mapping group_id (int) to signal data dict with keys:
        group_id, pct (pd.Series indexed by date), sign (int), series_id,
        start_val, end_val, is_plus_trigger, is_true,
        trigger_start_values, trigger_end_values.
    """
    from ..common.constants import (
        IS_PLUS_TRIGGER,
        IS_TRUE,
        SERIES_ID,
        TRIGGER_END_VALUES,
        TRIGGER_START_VALUES,
    )

    if segments_df is None or segments_df.empty or filtered_signal_sequence.empty:
        return {}

    # Build metadata lookup from filtered_signal_sequence
    meta_cols = [
        GROUP_ID, SERIES_ID, IS_PLUS_TRIGGER, IS_TRUE,
        TRIGGER_START_VALUES, TRIGGER_END_VALUES,
    ]
    available = [c for c in meta_cols if c in filtered_signal_sequence.columns]
    meta_df = (
        filtered_signal_sequence[available]
        .drop_duplicates(subset=[GROUP_ID])
        .set_index(GROUP_ID)
    )

    signals_data: Dict[int, Dict[str, Any]] = {}

    for gid, group in segments_df.groupby(GROUP_ID):
        gid = int(gid)
        dates = pd.to_datetime(group["date"].values)
        pct_vals = group["pct"].values.astype(float)
        pct = pd.Series(pct_vals, index=dates)

        meta = meta_df.loc[gid] if gid in meta_df.index else pd.Series(dtype=object)
        is_plus = bool(meta.get(IS_PLUS_TRIGGER, True))
        sign = +1 if is_plus else -1

        signals_data[gid] = {
            GROUP_ID: gid,
            "pct": pct,
            "sign": sign,
            SERIES_ID: str(meta.get(SERIES_ID, "")),
            "start_val": float(meta.get(TRIGGER_START_VALUES, float("nan"))),
            "end_val": float(meta.get(TRIGGER_END_VALUES, float("nan"))),
            IS_PLUS_TRIGGER: is_plus,
            IS_TRUE: bool(meta.get(IS_TRUE, False)),
            TRIGGER_START_VALUES: float(meta.get(TRIGGER_START_VALUES, float("nan"))),
            TRIGGER_END_VALUES: float(meta.get(TRIGGER_END_VALUES, float("nan"))),
        }

    return signals_data


def _load_reference_variation_map(store: Any, keys_set: set) -> Dict[str, pd.Series]:
    """Load /gradient_reference_variation from a store in read mode.

    Returns dict mapping reference_id to cumulative variation Series indexed
    by DatetimeIndex. Empty dict when the key is absent.
    """
    norm = HDF5_GRADIENT_REFERENCE_VARIATION if HDF5_GRADIENT_REFERENCE_VARIATION.startswith("/") else f"/{HDF5_GRADIENT_REFERENCE_VARIATION}"
    if norm not in keys_set:
        return {}
    ref_df = hdf5_manager.get_dataframe_fast(store, HDF5_GRADIENT_REFERENCE_VARIATION)
    if ref_df.empty:
        return {}

    out: Dict[str, pd.Series] = {}
    for ref_id, group in ref_df.groupby("reference_id"):
        group = group.sort_values("date")
        out[str(ref_id)] = pd.Series(
            group["variation"].values.astype(float),
            index=pd.to_datetime(group["date"].values),
        )
    return out


def _load_terminal_attenuations(
    store: Any,
    keys_set: set,
) -> Tuple[float, Dict[str, float]]:
    """Load /gradient_terminal_attenuations from a store in read mode.

    Returns (strategy_terminal_attenuation, {reference_id: attenuation_amount}).
    Both are zero/empty when the key is absent (no attenuation active).
    """
    norm = HDF5_GRADIENT_TERMINAL_ATTENUATIONS if HDF5_GRADIENT_TERMINAL_ATTENUATIONS.startswith("/") else f"/{HDF5_GRADIENT_TERMINAL_ATTENUATIONS}"
    if norm not in keys_set:
        return 0.0, {}
    attenuation_df = hdf5_manager.get_dataframe_fast(store, HDF5_GRADIENT_TERMINAL_ATTENUATIONS)
    if attenuation_df.empty:
        return 0.0, {}

    terminal_attenuation_amount = 0.0
    reference_terminal_attenuations: Dict[str, float] = {}
    for _, row in attenuation_df.iterrows():
        sid = str(row[SERIES_ID])
        amt = float(row["amount"])
        if sid == GRADIENT_STRATEGY_SENTINEL:
            terminal_attenuation_amount = amt
        else:
            reference_terminal_attenuations[sid] = amt

    return terminal_attenuation_amount, reference_terminal_attenuations
