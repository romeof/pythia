"""
Performance Luigi task for per-series metrics and analysis, and gradient computation.

This module provides the Performance task that orchestrates:
- Reading trigger HDF5 for per-series trigger plots (perseries_trigger)
- Reading signal HDF5 for per-series signal plots (perseries_signal) and gradient
- Computing metrics, printing, and plotting
- Generating gradient plot from signals
- Persisting enriched signals table and gradient data

Structure:
1. Performance Luigi task with output/run
2. Config caching, the effective start and the series' observed dates (_load_curve_dates)
3. Per-series performance (trigger and signal producer-consumer from HDF5)
4. Gradient (computation and plotting)
5. Persistence
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import luigi
import pandas as pd

from src.utils.common.common import (
    effective_start,
    wide_to_tall,
)
from src.utils.common.hdf5manager import hdf5_manager, stream_trigger_series
from src.utils.common.schema import PerformanceConfig, ReadConfig, RuntimeConfig, TriggerConfig
from src.tasks.sections import SectionParameter
from src.utils.common.strategyname import get_strategy_name
from src.utils.common.producerconsumer import (
    ProducerConsumerMixin,
    ProducerOutlet,
    create_producer_with_sentinel,
    logged_series_failure,
)
from src.utils.performance.gradient import (
    build_global_index,
    build_signal_segments_cumulative_pct_with_maxweight_constraint,
    build_actual_variation_pair,
    build_reference_terminal_attenuations,
    build_reference_variation_series_map,
)
from src.utils.performance.gradient_.metrics import (
    build_gradient_metrics_dataframe,
    compute_gradient_columns,
    get_gradient_metric_config,
    shared_annualization,
)
from src.utils.performance.perseries_.metrics import (
    build_signals_breakdown_dataframe,
    calculate_metrics,
    get_metric_config,
    prepare_direction_subset,
    trials_to_true_per_direction,
    _extract_needed_columns,
)
from src.utils.performance.perseries import process_perseries
from src.utils.performance.plots.common import PlotBase
from src.utils.performance.plots.gradient import plot_gradient
from src.utils.common.printframe import print_dataframe_output
from src.utils.performance.prints.gradient import (
    print_gradient_plot_values,
    print_total_variation,
)

from src.utils.common.constants import (
    DIRECTIONS,
    GRADIENT_PARAM_EFFECTIVE_START,
    GRADIENT_PARAM_SEQ_DISPLAY_GRADIENT,
    GRADIENT_PARAM_SEQ_DISPLAY_SIGNALS,
    GRADIENT_PARAM_SEQ_SIGNAL_PLOTS_WHITELIST,
    GRADIENT_STRATEGY_SENTINEL,
    GROUP_ID,
    HDF5_GRADIENT_FILTERED_SEQUENCE,
    HDF5_GRADIENT_METRICS_TABLE,
    HDF5_GRADIENT_PARAM_SEQUENCES,
    HDF5_GRADIENT_PARAMS,
    HDF5_GRADIENT_REFERENCE_VARIATION,
    HDF5_GRADIENT_SIGNAL_SEGMENTS,
    HDF5_GRADIENT_TERMINAL_ATTENUATIONS,
    HDF5_REFERENCE_PREFIX,
    HDF5_SIGNALS_BREAKDOWN_TABLE,
    HDF5_TABLE_INDEX_COLUMN,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    NUM_SERIES,
    VAR3,
    SERIES_ID,
    SERIES_IDS,
    SIGNAL_TABLE_COLUMNS,
    TRIALS_TO_TRUE,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)

logger = logging.getLogger(__name__)


class Performance(ProducerConsumerMixin, luigi.Task):
    """Luigi task for computing per-series metrics and aggregate gradient analysis."""

    read = SectionParameter(ReadConfig)
    trigger = SectionParameter(TriggerConfig)
    performance = SectionParameter(PerformanceConfig)
    runtime = SectionParameter(RuntimeConfig)
    target_file = luigi.Parameter()
    trigger_file = luigi.Parameter()
    signal_file = luigi.Parameter()

    # =========================================================================
    # LUIGI INTERFACE
    # =========================================================================

    def output(self):
        """Define task output (pipeline.py's _run_task cleans the folder before the build)."""
        return luigi.LocalTarget(self.target_file, format=luigi.format.Nop)

    def run(self):
        """Execute the performance task."""
        trigger_file = Path(self.trigger_file)
        signal_file = Path(self.signal_file)

        if not trigger_file.exists():
            raise FileNotFoundError(f"No trigger data found at {trigger_file}")
        if not signal_file.exists():
            raise FileNotFoundError(f"No signal data found at {signal_file}")

        self._cache_config()
        self._effective_start, self._series_dates = self._load_curve_dates(trigger_file)
        strategy_name = get_strategy_name(self.trigger)
        plots_dir = PlotBase.plots_dir_from_output(Path(self.target_file))

        # Per-series performance sub-task
        if self._trigger_printout_io or self._trigger_plot_debug:
            total_series, successful_series = (
                self._run_perseries_trigger_performance(
                    trigger_file, strategy_name, plots_dir
                )
            )
            logger.info(
                f"Perseries_trigger: Read {total_series} series, "
                f"successfully processed {successful_series} series."
            )

        # signals_df: accumulated signal metadata from Signal task HDF5
        # signal_series: time series DataFrames passed through from Trigger/Preprocessing
        signals_df, signal_series, reference_series, total_series, successful_series = (
            self._run_perseries_signal_performance(
                signal_file, strategy_name, plots_dir
            )
        )
        logger.info(
            f"Perseries_signal: Read {total_series} series, "
            f"successfully processed {successful_series} series."
        )

        # Gradient sub-task — also returns precomputed metric DataFrames and
        # raw gradient inputs needed for persistence.
        (
            gradient_series, attenuation_adjusted,
            gradient_metrics_df, signals_breakdown_df,
            filtered_signal_sequence, signals_data,
            reference_variation_map, terminal_attenuation_amount,
            reference_terminal_attenuations,
        ) = self._run_gradient(
            signals_df, signal_series, reference_series, strategy_name, plots_dir
        )

        # Persist enriched signals table and gradient data
        self._persist(
            signals_df, strategy_name, gradient_series, attenuation_adjusted,
            gradient_metrics_df, signals_breakdown_df,
            filtered_signal_sequence, signals_data,
            reference_variation_map, terminal_attenuation_amount,
            reference_terminal_attenuations,
            Path(self.target_file),
        )

    # =========================================================================
    # CONFIG CACHING
    # =========================================================================

    def _cache_config(self) -> None:
        """Cache all config reads at initialization, the metric configuration first."""
        self._metric_config = get_metric_config()
        # Per-series trigger flags
        self._trigger_printout_io = self.performance.perseries_trigger.printout_io
        self._trigger_full_output = self.performance.perseries_trigger.full_output
        self._trigger_plot_debug = self.performance.perseries_trigger.plot_debug
        # Per-series signal flags
        self._signal_printout_io = self.performance.perseries_signal.printout_io
        self._signal_full_output = self.performance.perseries_signal.full_output
        self._signal_plot_debug = self.performance.perseries_signal.plot_debug
        # Gradient flags
        self._gradient_printout_io = self.performance.gradient.printout_io
        self._gradient_full_output = self.performance.gradient.full_output
        # Gradient params
        self._weight = self.performance.gradient.weight
        self._concurrent_weight_limit = self.performance.gradient.concurrent_weight_limit
        self._variation_method = self.performance.gradient.variation_method
        self._yearly_attenuation_fraction = self.performance.gradient.yearly_attenuation_fraction
        self._attenuation_month = self.performance.gradient.attenuation_month
        self._boundary_deduction_pct = self.performance.gradient.boundary_deduction_pct
        self._mpv = self.performance.gradient.mpv
        self._gradient_plot_debug = self.performance.gradient.plot_debug
        self._gradient_printout_debug = self.performance.gradient.printout_debug
        self._max_signal_plots = self.performance.gradient.max_plots
        self._signal_plots_whitelist = list(self.performance.gradient.plots_whitelist)
        self._max_references = self.performance.gradient.max_references
        # Fail-fast: reference_series count must not exceed max_references.
        # schema.py declares reference_series: List[str] = Field(default_factory=list), so the list is never
        # absent (empty when the key is).
        reference_series_ids = self.read.reference_series
        if len(reference_series_ids) > self._max_references:
            raise ValueError(
                f"read.reference_series defines {len(reference_series_ids)} series "
                f"but performance.gradient.max_references is {self._max_references}. "
                f"Reduce reference_series or increase max_references."
            )
        # Shared params
        self._lookback = self.trigger.replay.lookback_window
        self._display_order_signals = list(self.performance.metrics.display_order_signals)
        self._display_order_gradient = list(self.performance.metrics.display_order_gradient)
        # Fail-fast: all display_order_gradient keys must exist in metrics.json
        valid_gradient_keys = set(get_gradient_metric_config().keys())
        unknown_keys = [k for k in self._display_order_gradient if k not in valid_gradient_keys]
        if unknown_keys:
            raise ValueError(
                f"Unknown gradient metric keys in display_order_gradient: {unknown_keys}. "
                f"Valid keys: {sorted(valid_gradient_keys)}"
            )
        self._needed_signal_columns = _extract_needed_columns(
            self._metric_config, self._display_order_signals
        )
        # Date range (cached here for _run_gradient)
        self._ini_date = self.read.ini_date
        self._end_date = self.read.end_date

    def _load_curve_dates(self, trigger_file: Path) -> Tuple[Optional[pd.Timestamp], List[pd.DatetimeIndex]]:
        """Earliest effective start over the trigger store's processed series (/series_<id>), the curve index's start,
        and each series' dates, the observed dates the curve index unites with the references'.

        Reads each series' whole index column (the frames are format='table'), never a whole frame; a series
        without rows is skipped, as iter_series skips it. Returns (None, []) for a store without series. A
        stored series shorter than the lookback (a trigger file written under another lookback) raises.
        """
        starts: List[pd.Timestamp] = []
        dates: List[pd.DatetimeIndex] = []
        short: Dict[str, int] = {}
        with hdf5_manager.open_store(trigger_file, "r") as store:
            for series_id, key in hdf5_manager.iter_series_keys(store):
                column = pd.DatetimeIndex(store.select_column(key, HDF5_TABLE_INDEX_COLUMN))
                if len(column) == 0:
                    continue
                dates.append(column)
                start = effective_start(column, self._lookback)
                if start is None:
                    short[series_id] = len(column)
                else:
                    starts.append(start)
        if short:
            raise ValueError(
                f"Trigger file {trigger_file} holds series shorter than "
                f"trigger.replay.lookback_window {self._lookback}: "
                f"{', '.join(f'{sid} ({short[sid]} rows)' for sid in sorted(short))}"
            )
        return (min(starts) if starts else None), dates

    # =========================================================================
    # PER-SERIES PERFORMANCE
    # =========================================================================

    def _run_perseries_trigger_performance(
        self,
        trigger_file: Path,
        strategy_name: str,
        plots_dir: Path,
    ) -> Tuple[int, int]:
        """Run producer-consumer pipeline for per-series trigger processing."""
        trigger_plots_dir = plots_dir / "trigger"
        trigger_plots_dir.mkdir(parents=True, exist_ok=True)

        return self._run_hdf5_producer_consumer(
            hdf5_file=trigger_file,
            stream_func=stream_trigger_series,
            consumer_func=lambda item: self._trigger_consumer(
                item, strategy_name, trigger_plots_dir
            ),
        )

    def _run_hdf5_producer_consumer(
        self,
        hdf5_file: Path,
        stream_func: Callable[[pd.HDFStore], Iterator],
        consumer_func: Callable[[Any], bool],
    ) -> Tuple[int, int]:
        """Run producer-consumer pipeline streaming items from an HDF5 file.

        Generic helper that reads an HDF5 store in read mode, iterates via stream_func,
        and feeds items through the producer-consumer pipeline.

        Args:
            hdf5_file: Path to the HDF5 file
            stream_func: Callable that takes an HDFStore in read mode and yields items
            consumer_func: Callable that processes each item, returns success bool

        Returns:
            Tuple of (total_items, successful_items)
        """
        def _stream_to_queue(queue: ProducerOutlet) -> None:
            with hdf5_manager.open_store(hdf5_file, "r") as store:
                for item in stream_func(store):
                    queue.put(item)

        def producer_func(queue: ProducerOutlet) -> None:
            return create_producer_with_sentinel(
                lambda: _stream_to_queue(queue), queue
            )

        return self.run_producer_consumer(
            producer_func=producer_func,
            consumer_func=consumer_func,
        )

    def _trigger_consumer(
        self,
        item: Tuple[str, pd.DataFrame, Optional[pd.DataFrame]],
        strategy_name: str,
        trigger_plots_dir: Path,
    ) -> bool:
        """Consumer: delegate to process_perseries for trigger data."""
        with logged_series_failure(item[0]):
            series_id, series_df, rows_df = item
            process_perseries(
                series_id=series_id,
                series_df=series_df,
                rows_df=rows_df,
                strategy_name=strategy_name,
                metric_config=self._metric_config,
                display_order_signals=self._display_order_signals,
                needed_columns=self._needed_signal_columns,
                plots_dir=trigger_plots_dir,
                printout_io=self._trigger_printout_io,
                plot_debug=self._trigger_plot_debug,
                print_sub_task_name="TRIGGER SERIES METRICS",
                full_output=self._trigger_full_output,
            )
        return True

    def _run_perseries_signal_performance(
        self,
        signal_file: Path,
        strategy_name: str,
        plots_dir: Path,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.Series], int, int]:
        """Load signal HDF5 metadata, stream series for enrichment and perseries.

        Always runs: enrichment and series accumulation are needed for gradient.
        Per-series print/plot is flag-guarded internally in _signal_consumer.

        Returns:
            Tuple of (signals_df, signal_series, reference_series, total, success)
        """
        # Batch load: signals table and reference series (small, needed upfront)
        signals_df, reference_series = self._load_signal_metadata(signal_file)

        if signals_df is None or signals_df.empty:
            logger.warning("No signals found in signal HDF5")
            empty = pd.DataFrame(columns=SIGNAL_TABLE_COLUMNS).astype({IS_PLUS_TRIGGER: bool, IS_TRUE: bool, IS_SIGNAL: bool})
            return empty, {}, {}, 0, 0

        # Producer-consumer: stream signal series for enrichment + perseries_signal
        signal_plots_dir = plots_dir / "signal"
        signal_plots_dir.mkdir(parents=True, exist_ok=True)

        enriched_parts: List[pd.DataFrame] = []
        signal_series: Dict[str, pd.DataFrame] = {}

        # Pre-build lookup dict: O(M) once instead of O(N*M) repeated .astype(str) scans
        signals_by_series: Dict[str, pd.DataFrame] = {
            str(sid): group for sid, group in signals_df.groupby(SERIES_ID)
        }

        total, success = self._run_hdf5_producer_consumer(
            hdf5_file=signal_file,
            stream_func=hdf5_manager.iter_series,
            consumer_func=lambda item: self._signal_consumer(
                item, signals_by_series, enriched_parts, signal_series,
                strategy_name, signal_plots_dir,
            ),
        )

        # Reassemble enriched signals_df; trials-to-true per direction over every row of the table
        signals_df = self._reassemble_enriched_signals(signals_df, enriched_parts)
        if TRIALS_TO_TRUE in self._needed_signal_columns:
            signals_df[TRIALS_TO_TRUE] = trials_to_true_per_direction(signals_df)

        return signals_df, signal_series, reference_series, total, success

    def _load_signal_metadata(
        self, signal_file: Path
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.Series]]:
        """Load signals table and reference series from signal HDF5 (no series data).

        An absent /signals node gives None (the documented empty path); a store that cannot be read raises.
        """
        with hdf5_manager.open_store(signal_file, "r") as store:
            signals_df = store.get("/signals") if "/signals" in store.keys() else None
            reference_series = self._load_reference_series_from_store(store)
        return signals_df, reference_series

    @staticmethod
    def _load_reference_series_from_store(store: pd.HDFStore) -> Dict[str, pd.Series]:
        """Load reference series from a store in read mode."""
        prefix = "/" + HDF5_REFERENCE_PREFIX
        reference_series: Dict[str, pd.Series] = {}
        for key in sorted(hdf5_manager.get_keys_set(store)):
            if not key.startswith(prefix):
                continue
            df = hdf5_manager.get_dataframe_fast(store, key)
            if len(df) == 0:
                continue
            if VAR3 not in df.columns:
                continue
            ts = pd.Series(
                df[VAR3].values, index=pd.to_datetime(df.index)
            ).dropna()
            series_id = key[len(prefix):]
            reference_series[series_id] = ts
        return reference_series

    def _signal_consumer(
        self,
        item: Tuple[str, pd.DataFrame],
        signals_by_series: Dict[str, pd.DataFrame],
        enriched_parts: List[pd.DataFrame],
        signal_series: Dict[str, pd.DataFrame],
        strategy_name: str,
        signal_plots_dir: Path,
    ) -> bool:
        """Consumer: enrich signals, run perseries_signal, accumulate for gradient."""
        with logged_series_failure(item[0]):
            series_id, series_df = item
            series_id_str = str(series_id)

            # Accumulate for gradient (must store all series regardless of signal status)
            signal_series[series_id_str] = series_df

            series_rows = signals_by_series.get(series_id_str)
            enriched = (
                calculate_metrics(series_df, series_rows, self._needed_signal_columns - {TRIALS_TO_TRUE})
                if series_rows is not None and not series_rows.empty
                else pd.DataFrame()
            )
            if not enriched.empty:
                enriched_parts.append(enriched)

            # Per-series signal processing (IS_SIGNAL rows only)
            # Recomputes metrics on the IS_SIGNAL subset so trials-to-true
            # reflects only selected signals (not inflated by non-IS_SIGNAL rows).
            if self._signal_printout_io or self._signal_plot_debug:
                signal_rows = (
                    enriched[enriched[IS_SIGNAL]]
                    if not enriched.empty
                    else pd.DataFrame()
                )
                if not signal_rows.empty:
                    process_perseries(
                        series_id=series_id_str,
                        series_df=series_df,
                        rows_df=signal_rows,
                        strategy_name=strategy_name,
                        metric_config=self._metric_config,
                        display_order_signals=self._display_order_signals,
                        needed_columns=self._needed_signal_columns,
                        plots_dir=signal_plots_dir,
                        printout_io=self._signal_printout_io,
                        plot_debug=self._signal_plot_debug,
                        print_sub_task_name="SIGNAL SERIES METRICS",
                        full_output=self._signal_full_output,
                    )

            return True

    def _reassemble_enriched_signals(
        self,
        original_signals_df: pd.DataFrame,
        enriched_parts: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """Reassemble signals_df from enriched parts, keeping unenriched rows."""
        if not enriched_parts:
            return original_signals_df

        enriched_df = pd.concat(enriched_parts)
        # Both enriched_df and original_signals_df share the same SERIES_ID dtype
        # (enriched parts are subsets of original_signals_df passed through
        # calculate_metrics, which does not alter SERIES_ID). No .astype(str) needed.
        enriched_series_ids = set(enriched_df[SERIES_ID].unique())

        # Keep original rows for series not found in signal HDF5
        unenriched = original_signals_df[
            ~original_signals_df[SERIES_ID].isin(enriched_series_ids)
        ]

        result = pd.concat([enriched_df, unenriched])
        result = result.sort_values(
            [GROUP_ID, TRIGGER_BEGIN_DATES, SERIES_ID], kind="mergesort"
        ).reset_index(drop=True)
        return result

    # =========================================================================
    # GRADIENT
    # =========================================================================

    def _run_gradient(
        self,
        signals_df: pd.DataFrame,
        signal_series: Dict[str, pd.DataFrame],
        reference_series: Dict[str, pd.Series],
        strategy_name: str,
        plots_dir: Path,
    ) -> Tuple[
        Optional[pd.Series], Optional[pd.Series],
        pd.DataFrame, pd.DataFrame,
        pd.DataFrame, Dict[int, Dict],
        Dict[str, pd.Series], float,
        Dict[str, float],
    ]:
        """Compute gradient variation from signals; plot it when performance.gradient.plot_debug is on.

        Returns:
            Tuple of (gradient_series, attenuation_adjusted, gradient_metrics_df,
            signals_breakdown_df, filtered_signal_sequence, signals_data,
            reference_variation_map, terminal_attenuation_amount,
            reference_terminal_attenuations).
        """
        # Safety-net cap: enforce max_references on HDF5-loaded series
        # (guards against files written by a prior run with a different config).
        if self._max_references == 0:
            reference_series = {}
        elif len(reference_series) > self._max_references:
            sorted_ids = sorted(reference_series.keys())[:self._max_references]
            reference_series = {sid: reference_series[sid] for sid in sorted_ids}

        # Build the curve index from the observed dates of the processed series and the references via the
        # shared helper (Fix #9).
        global_idx = build_global_index(
            [*self._series_dates, *(ts.index for ts in reference_series.values())],
            self._effective_start, self._end_date,
        )

        # Print signal events separated by direction (plus+minus, plus, minus),
        # each with trials-to-true recomputed on the direction-specific subset.
        if self._gradient_printout_io and not signals_df.empty:
            self._print_gradient_input_signals(signals_df)

        # Signal segments (middle panel) + filtered signals for the gradient.
        signals_data, filtered_signal_sequence = (
            build_signal_segments_cumulative_pct_with_maxweight_constraint(
                signals_df, signal_series, global_idx,
                self._weight, self._concurrent_weight_limit,
            )
        )

        # Pure gradient + attenuation-adjusted gradient (single conversion pass).
        # terminal_attenuation_amount is the last-year attenuation, already in attenuation_adjusted's last item.
        gradient_series, attenuation_adjusted, terminal_attenuation_amount = (
            build_actual_variation_pair(
                signals_data=signals_data,
                weight=self._weight,
                idx=global_idx,
                variation_method=self._variation_method,
                yearly_attenuation_fraction=self._yearly_attenuation_fraction,
                attenuation_month=self._attenuation_month,
                boundary_deduction_pct=self._boundary_deduction_pct,
            )
        )

        # References: pre-attenuation cumulative series + terminal attenuation amounts.
        reference_variation_map = build_reference_variation_series_map(
            reference_series=reference_series or {},
            global_idx=global_idx,
        )
        reference_terminal_attenuations = build_reference_terminal_attenuations(
            reference_variation_map, self._yearly_attenuation_fraction
        )

        # Fix C: branch on the rate, not on object identity — safe regardless
        # of build_actual_variation_pair's "same object when rate==0" contract.
        attenuation_adjusted_for_plot = (
            attenuation_adjusted if self._yearly_attenuation_fraction > 0 else None
        )

        # Build metric DataFrames once — shared by plot_gradient and _persist.
        # compute_gradient_columns handles pct scaling and terminal attenuation logic.
        preattenuation_columns, postattenuation_columns = compute_gradient_columns(
            gradient_series,
            attenuation_adjusted_for_plot,
            reference_variation_map,
            reference_terminal_attenuations,
            selection=self._display_order_gradient,
            mpv=self._mpv,
        )
        gradient_metrics_df = build_gradient_metrics_dataframe(
            preattenuation_columns, postattenuation_columns, self._display_order_gradient
        )

        # The breakdown counts the signals that build the curve (the weight-filtered rows).
        signals_breakdown_df = build_signals_breakdown_dataframe(
            filtered_signal_sequence.dropna(subset=[IS_TRUE, IS_PLUS_TRIGGER], how="any"), self._metric_config, self._display_order_signals
        )

        # The columns' shared annualization, checked on every run: the figure's table renders it only under the gate.
        shared_annualization(gradient_metrics_df)

        # Plot — pass precomputed DataFrames to avoid recomputing metric columns.
        if self._gradient_plot_debug:
            gradient_plots_dir = plots_dir / "gradient"
            gradient_plots_dir.mkdir(parents=True, exist_ok=True)
            gradient_file = gradient_plots_dir / "gradient.png"
            plot_gradient(
                signals_data=signals_data,
                filtered_signal_sequence=filtered_signal_sequence,
                gradient_series=gradient_series,
                global_idx=global_idx,
                reference_variation_map=reference_variation_map,
                strategy_name=strategy_name,
                emit_signal_plots=True,
                signal_plots_whitelist=self._signal_plots_whitelist,
                max_signal_plots=self._max_signal_plots,
                plots_dir=str(gradient_plots_dir),
                save_path=str(gradient_file),
                show=False,
                display_order_signals=self._display_order_signals,
                display_order_gradient=self._display_order_gradient,
                attenuation_adjusted_gradient=attenuation_adjusted_for_plot,
                terminal_attenuation_amount=terminal_attenuation_amount,
                reference_terminal_attenuations=reference_terminal_attenuations,
                mpv=self._mpv,
                gradient_metrics_table=gradient_metrics_df,
                signals_breakdown_table=signals_breakdown_df,
            )

        if self._gradient_printout_io and gradient_series is not None:
            print_total_variation(gradient_series)
        if self._gradient_printout_debug and gradient_series is not None:
            print_gradient_plot_values(
                gradient_series,
                attenuation_adjusted_gradient=attenuation_adjusted_for_plot,
                reference_variation_map=reference_variation_map,
            )

        return (
            gradient_series, attenuation_adjusted,
            gradient_metrics_df, signals_breakdown_df,
            filtered_signal_sequence, signals_data,
            reference_variation_map, terminal_attenuation_amount,
            reference_terminal_attenuations,
        )

    def _print_gradient_input_signals(self, signals_df: pd.DataFrame) -> None:
        """Print gradient input signals separated by direction (plus+minus, plus, minus).

        Each direction table has trials-to-true recomputed on its own subset,
        so wait times reflect only events within that direction.
        """
        signal_rows = signals_df[signals_df[IS_SIGNAL]]
        if signal_rows.empty:
            return

        for direction in DIRECTIONS:
            subset = prepare_direction_subset(signal_rows, direction)
            if subset.empty:
                continue
            print_dataframe_output(
                task_name="PERFORMANCE",
                sub_task_name=f"GRADIENT INPUT SIGNALS - {direction.upper()}",
                series_id=None,
                df=subset,
                full_output=self._gradient_full_output,
            )

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _persist(
        self,
        signals_df: pd.DataFrame,
        strategy_name: str,
        gradient_series: Optional[pd.Series],
        attenuation_adjusted: Optional[pd.Series],
        gradient_metrics_df: pd.DataFrame,
        signals_breakdown_df: pd.DataFrame,
        filtered_signal_sequence: pd.DataFrame,
        signals_data: Dict[int, Dict],
        reference_variation_map: Dict[str, pd.Series],
        terminal_attenuation_amount: float,
        reference_terminal_attenuations: Dict[str, float],
        out_file: Path,
    ) -> None:
        """Persist enriched signals table and all gradient plot data to HDF5."""
        metric_cols = sorted(
            {v.get("column", str(k)) for k, v in self._metric_config.items()}
        )

        schema = {
            "float32": [*metric_cols, TRIGGER_START_VALUES, TRIGGER_END_VALUES],
            "bool": [IS_PLUS_TRIGGER, IS_TRUE, IS_SIGNAL],
            "int32": [GROUP_ID, NUM_SERIES],
            "str": [SERIES_IDS, SERIES_ID],
        }

        with hdf5_manager.open_store(out_file, "w") as store:
            hdf5_manager.put_dataframe_normalized(
                store, "/signals", signals_df, format="table", schema=schema
            )

            if strategy_name:
                strategy_df = pd.DataFrame(
                    {"strategy_name": [str(strategy_name)]}
                )
                hdf5_manager.put_dataframe_normalized(
                    store, "/strategy_name", strategy_df, format="table"
                )

            self._persist_gradient_plot_data(
                store=store,
                gradient_series=gradient_series,
                attenuation_adjusted=attenuation_adjusted,
                gradient_metrics_df=gradient_metrics_df,
                signals_breakdown_df=signals_breakdown_df,
                filtered_signal_sequence=filtered_signal_sequence,
                signals_data=signals_data,
                reference_variation_map=reference_variation_map,
                terminal_attenuation_amount=terminal_attenuation_amount,
                reference_terminal_attenuations=reference_terminal_attenuations,
            )

    def _persist_gradient_plot_data(
        self,
        store: Any,
        gradient_series: Optional[pd.Series],
        attenuation_adjusted: Optional[pd.Series],
        gradient_metrics_df: pd.DataFrame,
        signals_breakdown_df: pd.DataFrame,
        filtered_signal_sequence: pd.DataFrame,
        signals_data: Dict[int, Dict],
        reference_variation_map: Dict[str, pd.Series],
        terminal_attenuation_amount: float,
        reference_terminal_attenuations: Dict[str, float],
    ) -> None:
        """Write all gradient plot reconstruction data to the HDF5 store in write mode.

        Split from _persist to keep each method focused (SRP).
        The curves, segments and filtered sequence use format='fixed'; the reference
        curves, attenuations, parameters and metric tables use format='table'.
        """
        # /total_variation — pre-attenuation only (Fix B: single column, no redundant attenuation copy)
        if gradient_series is not None and len(gradient_series) > 0:
            tv_df = pd.DataFrame({
                "date": gradient_series.index,
                "total_variation": gradient_series.values.astype("float32"),
            })
            hdf5_manager.put_dataframe_normalized(
                store, "/total_variation", tv_df, format="fixed"
            )

        # /total_variation_attenuated — only when attenuation is active (Fix B)
        if attenuation_adjusted is not None and len(attenuation_adjusted) > 0 and self._yearly_attenuation_fraction > 0:
            tld_df = pd.DataFrame({
                "date": attenuation_adjusted.index,
                "total_variation_attenuated": attenuation_adjusted.values.astype("float32"),
            })
            hdf5_manager.put_dataframe_normalized(
                store, "/total_variation_attenuated", tld_df, format="fixed"
            )

        # /gradient_reference_variation — long-form (reference_id, date, variation)
        if reference_variation_map:
            parts = []
            for ref_id, series in reference_variation_map.items():
                if series is not None and len(series) > 0:
                    parts.append(pd.DataFrame({
                        "reference_id": ref_id,
                        "date": series.index,
                        "variation": series.values.astype("float32"),
                    }))
            if parts:
                ref_df = pd.concat(parts, ignore_index=True)
                ref_df["reference_id"] = ref_df["reference_id"].astype(str)
                hdf5_manager.put_dataframe_normalized(
                    store, HDF5_GRADIENT_REFERENCE_VARIATION, ref_df, format="table"
                )

        # /gradient_signal_segments — long-form (group_id, date, pct)
        if signals_data:
            parts = []
            for gid, gdata in signals_data.items():
                pct = gdata.get("pct")
                if pct is not None and len(pct) > 0:
                    parts.append(pd.DataFrame({
                        GROUP_ID: gid,
                        "date": pct.index,
                        "pct": pct.values.astype("float32"),
                    }))
            if parts:
                seg_df = pd.concat(parts, ignore_index=True)
                seg_df[GROUP_ID] = seg_df[GROUP_ID].astype("int32")
                hdf5_manager.put_dataframe_normalized(
                    store, HDF5_GRADIENT_SIGNAL_SEGMENTS, seg_df, format="fixed"
                )

        # /gradient_filtered_sequence — (sequence_index, group_id)
        if not filtered_signal_sequence.empty:
            seq_df = pd.DataFrame({
                "sequence_index": pd.array(
                    range(len(filtered_signal_sequence)), dtype="int32"
                ),
                GROUP_ID: filtered_signal_sequence[GROUP_ID].values.astype("int32"),
            })
            hdf5_manager.put_dataframe_normalized(
                store, HDF5_GRADIENT_FILTERED_SEQUENCE, seq_df, format="fixed"
            )

        # /gradient_terminal_attenuations — only when attenuation is active
        if self._yearly_attenuation_fraction > 0:
            attenuation_rows = [
                {SERIES_ID: GRADIENT_STRATEGY_SENTINEL, "amount": float(terminal_attenuation_amount)}
            ]
            for ref_id, amt in reference_terminal_attenuations.items():
                attenuation_rows.append({SERIES_ID: str(ref_id), "amount": float(amt)})
            attenuation_df = pd.DataFrame(attenuation_rows)
            attenuation_df["amount"] = attenuation_df["amount"].astype("float32")
            attenuation_df[SERIES_ID] = attenuation_df[SERIES_ID].astype(str)
            hdf5_manager.put_dataframe_normalized(
                store, HDF5_GRADIENT_TERMINAL_ATTENUATIONS, attenuation_df, format="table"
            )

        # /gradient_params — scalar replot parameters (one row)
        params_df = pd.DataFrame([{
            "weight": float(self._weight),
            "concurrent_weight_limit": float(self._concurrent_weight_limit),
            "variation_method": self._variation_method,
            "yearly_attenuation_fraction": float(self._yearly_attenuation_fraction),
            "attenuation_month": int(self._attenuation_month),
            "boundary_deduction_pct": float(self._boundary_deduction_pct),
            "mpv": float(self._mpv),
            "ini_date": self._ini_date,
            "end_date": self._end_date,
            "lookback": int(self._lookback),
            "max_signal_plots": int(self._max_signal_plots),
            GRADIENT_PARAM_EFFECTIVE_START: (
                "" if self._effective_start is None else self._effective_start.strftime("%Y-%m-%d")
            ),
        }])
        hdf5_manager.put_dataframe_normalized(
            store, HDF5_GRADIENT_PARAMS, params_df, format="table"
        )

        # /gradient_param_sequences — (kind, sequence_index, name) for list-valued params
        seq_rows = []
        for kind, items in [
            (GRADIENT_PARAM_SEQ_DISPLAY_SIGNALS, self._display_order_signals),
            (GRADIENT_PARAM_SEQ_DISPLAY_GRADIENT, self._display_order_gradient),
            (GRADIENT_PARAM_SEQ_SIGNAL_PLOTS_WHITELIST, self._signal_plots_whitelist),
        ]:
            for idx, name in enumerate(items):
                seq_rows.append({"kind": kind, "sequence_index": idx, "name": str(name)})
        if seq_rows:
            param_seqs_df = pd.DataFrame(seq_rows)
            param_seqs_df["sequence_index"] = param_seqs_df["sequence_index"].astype("int32")
            param_seqs_df["kind"] = param_seqs_df["kind"].astype(str)
            param_seqs_df["name"] = param_seqs_df["name"].astype(str)
            hdf5_manager.put_dataframe_normalized(
                store, HDF5_GRADIENT_PARAM_SEQUENCES, param_seqs_df, format="table"
            )

        # /gradient_metrics_table — tall form (metric_key, column_header, value)
        if gradient_metrics_df is not None and not gradient_metrics_df.empty:
            tall_gm = wide_to_tall(
                gradient_metrics_df, "metric_key", "column_header", "value"
            )
            hdf5_manager.put_dataframe_normalized(
                store, HDF5_GRADIENT_METRICS_TABLE, tall_gm, format="table"
            )

        # /signals_breakdown_table — tall form (row_key, column_header, value)
        if signals_breakdown_df is not None and not signals_breakdown_df.empty:
            tall_sb = wide_to_tall(
                signals_breakdown_df, "row_key", "column_header", "value"
            )
            hdf5_manager.put_dataframe_normalized(
                store, HDF5_SIGNALS_BREAKDOWN_TABLE, tall_sb, format="table"
            )
