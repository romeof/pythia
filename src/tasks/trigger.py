"""
Trigger Luigi task for time series trigger detection and extraction with trigger direction and outcome.

This module provides the Trigger task that orchestrates:
- Reading raw time series data
- Preprocessing (cleaning)
- Trigger detection via configurable strategies
- Trigger extraction with direction (is_plus_trigger) and outcome (is_true)
- HDF5 output with trigger data and extracted triggers

Structure:
1. Trigger Luigi task: output/run, task setup, the producer-consumer pipeline, trigger extraction and
   persistence, series filtering (the gate pre-filter) and reference handling
"""

import logging
from pathlib import Path
from typing import Optional, Set, Tuple

import luigi
import pandas as pd

from src.utils.common.common import effective_start
from src.utils.common.hdf5manager import hdf5_manager
from src.utils.common.producerconsumer import (
    ProducerConsumerMixin,
    create_producer_with_sentinel,
    logged_series_failure,
)
from src.utils.common.schema import ReadConfig, RuntimeConfig, TriggerConfig
from src.utils.preprocess.preprocess import Preprocess
from src.utils.preprocess.read import Read
from src.tasks.sections import SectionParameter
from src.utils.common.constants import (
    HDF5_REFERENCE_PREFIX,
    HDF5_TRIGGERS_PREFIX,
    IS_PLUS_TRIGGER,
    IS_TRUE,
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)
from src.utils.trigger.factory import create_strategy
from src.utils.trigger.gate import applicable_items, blocking_item, check_gate_columns, gate_columns
from src.utils.trigger.outcome import TriggerData
from src.utils.trigger.prints.detection import print_trigger_outcome_output

logger = logging.getLogger(__name__)


class Trigger(ProducerConsumerMixin, luigi.Task):
    """Luigi task for detecting time series triggers and extracting triggers."""

    read = SectionParameter(ReadConfig)
    trigger = SectionParameter(TriggerConfig)
    runtime = SectionParameter(RuntimeConfig)
    target_file = luigi.Parameter()

    # =========================================================================
    # LUIGI INTERFACE
    # =========================================================================

    def output(self):
        """Define task output (pipeline.py's _run_task cleans the folder before the build)."""
        return luigi.LocalTarget(self.target_file, format=luigi.format.Nop)

    def run(self):
        """Execute the trigger task."""
        trigger_strategy = self._create_strategy()

        trigger_helper = TriggerData()
        self._reference_series = self._load_reference_series()
        self._gate_items = list(self.trigger.gate.items)
        self._gate_prefilter = self.trigger.gate.prefilter
        self._gate_columns = gate_columns(self._gate_items)
        self._lookback_window = self.trigger.replay.lookback_window
        self._trigger_printout_io = self.trigger.strategy.printout_io
        self._full_output = self.trigger.strategy.full_output

        self._run_producer_consumer_pipeline(trigger_strategy, trigger_helper)

    # =========================================================================
    # TASK SETUP
    # =========================================================================

    def _create_strategy(self):
        """Create the trigger detection algorithm instance."""
        try:
            return create_strategy(self.trigger, Path(self.target_file))
        except Exception as e:
            logger.error("Error creating trigger detection algorithm: %s", e)
            raise

    def _load_reference_series(self) -> Set[str]:
        """Load reference series set from config."""
        reference_series = set(self.read.reference_series)
        if reference_series:
            sorted_ids = sorted(reference_series)
            logger.info(
                f"{len(reference_series)} reference series enabled: {' and '.join(sorted_ids)}"
            )
        else:
            logger.info("No reference_series configured; raw saving disabled")
        return reference_series

    # =========================================================================
    # PRODUCER-CONSUMER PIPELINE
    # =========================================================================

    def _run_producer_consumer_pipeline(
        self, trigger_strategy, trigger_helper: TriggerData
    ) -> None:
        """Run the producer-consumer pipeline with shared HDF5 store."""

        def producer_func(queue):
            return create_producer_with_sentinel(
                lambda: self._producer_worker(queue, trigger_strategy), queue
            )

        output_file = Path(self.target_file)
        with hdf5_manager.open_store(output_file, "w") as store:

            def consumer_func_with_store(item):
                return self._consumer_worker(
                    item, trigger_strategy, trigger_helper, store
                )

            total_series, successful_series = self.run_producer_consumer(
                producer_func=producer_func,
                consumer_func=consumer_func_with_store,
            )

        logger.info(
            f"Read {total_series} series, successfully processed {successful_series} series."
        )

    def _producer_worker(self, queue, trigger_strategy) -> None:
        """Producer: stream, preprocess, and emit series."""
        reader = Read(self.read, self._gate_columns, self.trigger.read.printout_io)
        output_file = Path(self.target_file)
        preprocessor = Preprocess(self.trigger.consecutive_rows_with_bad_preprocessed_data, self.trigger.validity,
                                  self.trigger.preprocess.printout_io, output_file)

        # before _emit_series where preprocessing is executed: every configured gate column, whether or not the
        # pre-filter is on, then whether _should_process_series at all
        for series_id, df in reader.iter_series():
            check_gate_columns(series_id, df, self._gate_columns)
            if not self._should_process_series(series_id, df):
                continue

            self._emit_series(series_id, df, preprocessor, queue)

    def _consumer_worker(
        self,
        batch: Tuple[str, Optional[pd.DataFrame], pd.DataFrame],
        trigger_strategy,
        trigger_helper: TriggerData,
        store,
    ) -> bool:
        """Consumer: detect triggers, extract triggers, and save to HDF5."""
        with logged_series_failure(batch[0]):
            series_id, raw_df, preprocessed_df = batch
            start = effective_start(preprocessed_df.index, self._lookback_window)
            detected_df = trigger_strategy.analyze_series(
                series_id, preprocessed_df, store=store
            )

            if raw_df is not None:
                self._save_reference_series(series_id, raw_df, start, store)

            if detected_df is not None:
                triggers_df = self._extract_triggers(series_id, detected_df, start, trigger_helper)
                if triggers_df is not None and len(triggers_df) > 0:
                    if self._trigger_printout_io:
                        print_trigger_outcome_output(series_id, triggers_df, full_output=self._full_output)
                    self._persist_triggers(series_id, triggers_df, store)


            logger.info(f"Successfully processed series {series_id}")
            return True

    # =========================================================================
    # TRIGGER EXTRACTION
    # =========================================================================

    def _extract_triggers(
        self,
        series_id: str,
        detected_df: pd.DataFrame,
        start: pd.Timestamp,
        trigger_helper: TriggerData,
    ) -> Optional[pd.DataFrame]:
        """Extract triggers from trigger data.

        Filters out triggers whose begin date falls before the series' effective start
        (index[lookback_window - 1] of its preprocessed frame). Ideal mode detects triggers
        on the full series with no burn-in awareness, so the filter removes its early ones;
        realistic mode suppresses those candidates at the same date
        (before_effective_start), so the filter keeps all of its triggers.
        """
        triggers_df = trigger_helper._extract_triggers_from_trigger_columns(series_id, detected_df)
        if triggers_df is None or len(triggers_df) == 0:
            logger.warning(f"No triggers extracted for series {series_id}")
            return None

        before = len(triggers_df)
        triggers_df = triggers_df[
            pd.to_datetime(triggers_df[TRIGGER_BEGIN_DATES]) >= start
        ].reset_index(drop=True)

        if len(triggers_df) < before:
            logger.info(
                f"Series {series_id}: filtered {before - len(triggers_df)} "
                f"trigger(s) starting before effective start "
                f"({start.date()})"
            )

        if len(triggers_df) == 0:
            return None

        return triggers_df

    def _persist_triggers(self, series_id: str, triggers_df: pd.DataFrame, store) -> None:
        """Persist triggers DataFrame to HDF5."""

        schema = {
            "drop": [SERIES_ID],
            "float32": [TRIGGER_START_VALUES, TRIGGER_END_VALUES],
            "bool": [IS_PLUS_TRIGGER, IS_TRUE],
        }
        hdf5_manager.put_dataframe_normalized(
            store, f"{HDF5_TRIGGERS_PREFIX}{series_id}", triggers_df, format="fixed", schema=schema
        )

    # =========================================================================
    # SERIES FILTERING
    # =========================================================================

    def _should_process_series(self, series_id: str, df: pd.DataFrame) -> bool:
        """The gate pre-filter (trigger.gate.prefilter) on the date-filtered raw frame: False, with one INFO record
        naming the item, when one item that applies to the series passes on no row of the read window; True with the
        pre-filter off and for a reference series, which it never skips.
        """
        if not self._gate_prefilter or series_id in self._reference_series:
            return True
        item = blocking_item(df, applicable_items(series_id, self._gate_items))
        if item is None:
            return True
        logger.info(
            f"Skipping series {series_id}: gate item {item.column} {item.op} {item.threshold} "
            f"passes on no row of the read window"
        )
        return False

    def _emit_series(
        self, series_id: str, df: pd.DataFrame, preprocessor: Preprocess, queue
    ) -> None:
        """Preprocess and emit a series to the queue."""
        raw_df = df if series_id in self._reference_series else None
        processed_df = preprocessor.process_series(series_id, df)
        queue.put((series_id, raw_df, processed_df))

    # =========================================================================
    # REFERENCE HANDLING
    # =========================================================================

    def _save_reference_series(
        self, series_id: str, raw_df: pd.DataFrame, start: Optional[pd.Timestamp], store
    ) -> None:
        """Save reference series trimmed to the series' effective start for aligned comparison.

        The start is index[lookback_window - 1] of the series' preprocessed frame, the date
        the gradient's global index starts from when this series is the earliest; a
        preprocessed frame shorter than lookback_window (start None) leaves the raw frame
        untrimmed.
        """
        if start is not None:
            raw_df = raw_df.loc[raw_df.index >= start]

        hdf5_manager.put_dataframe_normalized(
            store, f"{HDF5_REFERENCE_PREFIX}{series_id}", raw_df, format="fixed"
        )
