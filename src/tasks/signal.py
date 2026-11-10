"""
Signal Luigi task for trigger grouping and signal selection.

This module provides the Signal task that orchestrates:
- Reading trigger HDF5 output (series DataFrames and triggers) and collect series values and trigger information
- Grouping triggers into signal windows by time overlap and correlation
- Selecting representative signals per group
- HDF5 output with groups table, signal series, and reference pass-through

Structure:
1. Signal Luigi task with output/run
2. Config caching
3. Producer-consumer pipeline
4. Finalization and persistence
"""

import logging
from pathlib import Path
from typing import Set

import luigi
import pandas as pd

from src.utils.common.hdf5manager import hdf5_manager, stream_trigger_series
from src.utils.common.schema import RuntimeConfig, SignalConfig, TriggerConfig
from src.tasks.sections import SectionParameter
from src.utils.common.strategyname import get_strategy_name
from src.utils.common.producerconsumer import (
    ProducerConsumerMixin,
    create_producer_with_sentinel,
    logged_series_failure,
)
from src.utils.signal.prints.selection import print_groups
from src.utils.signal.selection import SignalSelection
from src.utils.common.constants import (
    CORR_TO_SIGNAL,
    GROUP_ID,
    IS_PLUS_TRIGGER,
    IS_SIGNAL,
    IS_TRUE,
    NUM_SERIES,
    SERIES_ID,
    SERIES_IDS,
    SLOPE,
    TRIGGER_END_VALUES,
    TRIGGER_START_VALUES,
)

logger = logging.getLogger(__name__)


class Signal(ProducerConsumerMixin, luigi.Task):
    """Luigi task for grouping triggers and selecting representative signals."""

    trigger = SectionParameter(TriggerConfig)
    signal = SectionParameter(SignalConfig)
    runtime = SectionParameter(RuntimeConfig)
    target_file = luigi.Parameter()
    trigger_file = luigi.Parameter()

    # =========================================================================
    # LUIGI INTERFACE
    # =========================================================================

    def output(self):
        """Define task output (pipeline.py's _run_task cleans the folder before the build)."""
        return luigi.LocalTarget(self.target_file, format=luigi.format.Nop)

    def run(self):
        """Execute the signal task."""
        trigger_file = Path(self.trigger_file)
        if not trigger_file.exists():
            raise FileNotFoundError(f"No trigger data found at {trigger_file}")

        self._cache_config()
        strategy_name = get_strategy_name(self.trigger)
        collector = SignalSelection(self.signal, self.trigger.replay.mode)

        total_series, successful_series = self._run_producer_consumer_pipeline(
            trigger_file, collector
        )
        logger.info(
            f"Read {total_series} series, "
            f"successfully processed {successful_series} series."
        )

        self._finalize(collector, trigger_file, strategy_name)

    # =========================================================================
    # CONFIG CACHING
    # =========================================================================

    def _cache_config(self) -> None:
        """Cache all config reads at initialization."""
        self._input_printout_io = self.signal.input.printout_io
        self._input_full_output = self.signal.input.full_output
        self._selection_printout_io = self.signal.selection.printout_io
        self._selection_full_output = self.signal.selection.full_output

    # =========================================================================
    # PRODUCER-CONSUMER PIPELINE
    # =========================================================================

    def _run_producer_consumer_pipeline(self, trigger_file: Path, collector: SignalSelection):
        """Run the producer-consumer pipeline reading trigger HDF5."""

        def producer_func(queue):
            return create_producer_with_sentinel(
                lambda: self._producer_worker(trigger_file, queue), queue
            )

        return self.run_producer_consumer(
            producer_func=producer_func,
            consumer_func=lambda item: self._consumer_worker(item, collector),
        )

    def _producer_worker(self, trigger_file: Path, queue) -> None:
        """Producer: stream series and triggers from trigger HDF5."""
        with hdf5_manager.open_store(trigger_file, "r") as store:
            for series_id, series_df, triggers_df in stream_trigger_series(store):
                queue.put((series_id, series_df, triggers_df))

    def _consumer_worker(self, item, collector: SignalSelection) -> bool:
        """Consumer: accumulate series values and triggers into the collector."""
        with logged_series_failure(item[0]):
            series_id, series_df, triggers_df = item
            collector.add_series_values(series_id, series_df)

            if triggers_df is not None and len(triggers_df) > 0:
                # Add series_id column since trigger HDF5 drops it during persistence
                triggers_df = triggers_df.copy()
                triggers_df[SERIES_ID] = str(series_id)
                collector.add_series_triggers(series_id, triggers_df)

            return True

    # =========================================================================
    # FINALIZATION AND PERSISTENCE
    # =========================================================================

    def _finalize(
        self,
        collector: SignalSelection,
        trigger_file: Path,
        strategy_name: str,
    ) -> None:
        """Build groups and persist signal HDF5."""
        groups_df = collector.build_signal_window_groups(
            printout_io=self._input_printout_io,
            full_output=self._input_full_output,
        )
        logger.info(f"Generated {len(groups_df)} group rows")

        if self._selection_printout_io:
            print_groups(groups_df, full_output=self._selection_full_output)

        # Identify signal series IDs (only is_signal=True rows)
        signal_series_ids = set(
            groups_df.loc[groups_df[IS_SIGNAL], SERIES_ID].astype(str)
        )

        self._persist(groups_df, trigger_file, signal_series_ids, strategy_name)

    def _persist(
        self,
        groups_df: pd.DataFrame,
        trigger_file: Path,
        signal_series_ids: Set[str],
        strategy_name: str,
    ) -> None:
        """Persist signal HDF5 with groups, signal series, and references."""
        out_file = Path(self.target_file)

        schema = {
            "float32": [
                TRIGGER_START_VALUES,
                TRIGGER_END_VALUES,
                SLOPE,
                CORR_TO_SIGNAL,
            ],
            "bool": [IS_PLUS_TRIGGER, IS_TRUE, IS_SIGNAL],
            "int32": [GROUP_ID, NUM_SERIES],
            "str": [SERIES_IDS, SERIES_ID],
        }

        with hdf5_manager.open_store(out_file, "w") as out_store:
            hdf5_manager.put_dataframe_normalized(
                out_store, "/signals", groups_df, format="table", schema=schema
            )

            self._pass_through_series(out_store, trigger_file, signal_series_ids)

            if strategy_name:
                strategy_df = pd.DataFrame({"strategy_name": [str(strategy_name)]})
                hdf5_manager.put_dataframe_normalized(
                    out_store, "/strategy_name", strategy_df, format="table"
                )

    def _pass_through_series(
        self,
        out_store: pd.HDFStore,
        trigger_file: Path,
        signal_series_ids: Set[str],
    ) -> None:
        """Copy signal series and reference series from trigger HDF5 to output."""
        with hdf5_manager.open_store(trigger_file, "r") as trig_store:
            for key in sorted(hdf5_manager.get_keys_set(trig_store)):
                is_signal_series = (
                    key.startswith("/series_")
                    and "reference" not in key
                    and key.removeprefix("/series_") in signal_series_ids
                )
                is_reference = "reference" in key

                if not is_signal_series and not is_reference:
                    continue

                df = hdf5_manager.get_dataframe_fast(trig_store, key)
                hdf5_manager.put_dataframe_normalized(out_store, key, df, format="table")

