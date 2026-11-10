"""Regression for the failure paths: one in-process case per production raise (D-7; R-X3).

Each case calls the function with a failing input through testkit.raised and asserts the
exception's type and a fragment of its message; each verdict label names the function, so the
failure-path sweep finds a test for every raise. The cases cover every raise the sweep listed as
named by no test, the raises W1-W6 added (logged_series_failure, the two of
create_producer_consumer_pipeline, ProducerOutlet.put, plot_series, plot_window_with_triggers),
the gate's (check_gate_columns; the refused GateItem op and column) and the re-raise sites
(Trigger._create_strategy, open_store, _get_git_output). Four record or
chain checks complement them: logged_series_failure leaves with the block's own exception and one
ERROR record naming series 7 (testkit.RecordCollector at ERROR on the root logger, to which the
module loggers propagate); the producer-failure raise of the pipeline's finally, reached only off
the normal path (a producer that puts one item and raises OSError while the consumer raises
ValueError on it), leaves the OSError with the ValueError as its __context__; _get_git_output,
its subprocess.run patched to write b"\\xff" through the file behind stdout as a child would,
raises UnicodeDecodeError and logs the ERROR record naming the command. A raise turned into a
return fails its case.

Structure:
1. main
2. Case table
3. Record and chain cases
4. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import copy
import logging
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from typing import Callable, Iterator

import h5py
import numpy as np
import pandas as pd
from pydantic import ValidationError

import pipeline
import src.tasks.trigger as trigger_module
import src.utils.performance.gradient_.metrics as gradient_metrics_module
from src.tasks.performance import Performance
from src.utils.common.common import select_trigger_value_vector, tall_to_wide
from src.utils.common.config import load_json_config, validate_config
from src.utils.common.hdf5manager import hdf5_manager
from src.utils.common.outputs import _get_git_output
from src.utils.common.producerconsumer import (
    ProducerOutlet, ProducerStopped, create_producer_consumer_pipeline, logged_series_failure,
)
from src.utils.common.schema import AppConfig, strategy_class
from src.utils.performance.gradient_.compounding import multiplicative_variation
from src.utils.performance.gradient_.metrics import get_gradient_metric_config, shared_annualization
from src.utils.performance.perseries_.metrics import get_metric_config, prepare_direction_subset, resolve_signal_selection
from src.utils.performance.perseries_.metricscalculations import _apply_operation
from src.utils.performance.plots.perseries import plot_series
from src.utils.performance.replot import load_gradient_replot_params
from src.utils.preprocess.read import Read
from src.utils.trigger.gate import check_gate_columns
from src.utils.trigger.plots.detection import plot_window_with_triggers

PLANTED = "planted"
DATES = pd.bdate_range("2020-01-01", periods=5)
CONSUMER_ERROR = "planted consumer failure"
GIT_COMMAND = ["git", "rev-parse", "HEAD"]
UNDECODABLE = b"\xff"


def main() -> None:
    """The case table, then the record and chain cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        config = testkit.loaded_config(config_file)
        for label, call, expected_type, fragment in case_table(folder, config, config_file):
            exc = testkit.raised(call)
            KIT.verdict(f"{label}: {expected_type.__name__} naming {fragment!r}",
                        isinstance(exc, expected_type) and fragment in str(exc), f"{expected_type.__name__}: {fragment}",
                        repr(exc)[:240])
    series_record()
    producer_failure_chain()
    undecodable_git_output()
    KIT.finish()


# =============================================================================
# CASE TABLE
# =============================================================================


def case_table(folder: Path, config: dict, config_file: Path) -> list[tuple[str, Callable[[], object], type, str]]:
    """(label naming the function, the failing call, the exception type, a fragment of its message)."""
    absent = folder / "absent" / "0.h5"
    return [
        ("pipeline._create_task: an unknown task",
         lambda: pipeline._create_task(testkit.validated_config(config_file), PLANTED), ValueError,
         f"Unknown task: {PLANTED}"),
        ("Performance._cache_config: more reference series than max_references",
         lambda: performance_task(edited(config, {"performance.gradient.max_references": 0}))._cache_config(), ValueError,
         "max_references is 0"),
        ("Performance._cache_config: an unknown display_order_gradient key",
         lambda: performance_task(edited(config, {"performance.metrics.display_order_gradient": [PLANTED]}))._cache_config(),
         ValueError, "Unknown gradient metric keys"),
        ("Performance._load_curve_dates: a stored series shorter than the lookback",
         lambda: short_store_task(config)._load_curve_dates(short_store(folder)), ValueError,
         "holds series shorter than"),
        ("Trigger._create_strategy: the strategy cannot be built (re-raised)", lambda: failing_strategy(config), OSError,
         testkit.PLANTED_ERROR),
        ("common.tall_to_wide: repeated (row, column) pairs",
         lambda: tall_to_wide(pd.DataFrame({"r": ["a", "a"], "c": ["x", "x"], "v": [1.0, 2.0]}), "r", "c", "v"), ValueError,
         "duplicate"),
        ("config.load_json_config: no metrics.json beside the caller", lambda: load_json_config(str(folder / "caller.py")),
         RuntimeError, "Failed to load metric configuration"),
        ("common.select_trigger_value_vector: an unknown trigger_value mode",
         lambda: select_trigger_value_vector(value_frame(), PLANTED, primary_col="var3", start_col="var0"), ValueError,
         f"Unknown trigger_value mode: {PLANTED}"),
        ("schema ValidityConfig._two_distinct_columns: not_all_equal of one distinct column",
         lambda: validate_config(edited(config, {"trigger.validity.not_all_equal": ["var0", "var0"]})), ValidationError,
         "not_all_equal lists one distinct column"),
        ("schema SignalConfig._corr_range: corr_threshold above 1",
         lambda: validate_config(edited(config, {"signal.corr_threshold": 1.5})), ValidationError,
         "corr_threshold must be in [0.0, 1.0]"),
        ("schema PerformanceGradientConfig._mpv_range: mpv above 0.25",
         lambda: validate_config(edited(config, {"performance.gradient.mpv": 0.3})), ValidationError,
         "mpv must be in [0.0, 0.25]"),
        ("schema PerformanceGradientConfig._weight_range: weight above 1",
         lambda: validate_config(edited(config, {"performance.gradient.weight": 1.5})), ValidationError,
         "weight must be in (0.0, 1.0]"),
        ("schema PerformanceGradientConfig._concurrent_weight_limit_range: concurrent_weight_limit above 1",
         lambda: validate_config(edited(config, {"performance.gradient.concurrent_weight_limit": 1.5})), ValidationError,
         "concurrent_weight_limit must be in (0.0, 1.0]"),
        ("schema strategy_class: a name outside the registry", lambda: strategy_class(PLANTED), KeyError,
         f"Unknown strategy: {PLANTED}"),
        ("schema GateItem.op: an op outside >= and <=",
         lambda: validate_config(edited(config, {"trigger.gate.items": [testkit.gate_item("var12", ">", 0.5)]})),
         ValidationError, "Input should be '>=' or '<='"),
        ("schema GateItem.column: a column that is no positional input column",
         lambda: validate_config(edited(config, {"trigger.gate.items": [testkit.gate_item("var0x", ">=", 0.5)]})),
         ValidationError, "String should match pattern"),
        ("gate.check_gate_columns: a frame without the configured gate column var12",
         lambda: check_gate_columns("0", value_frame(), ["var12"]), ValueError,
         "series 0 lacks the configured gate columns ['var12']"),
        ("compounding.multiplicative_variation: an unknown variation method",
         lambda: multiplicative_variation([], 0.2, DATES, PLANTED), ValueError, f"Unknown variation method: {PLANTED}"),
        ("gradient metrics get_gradient_metric_config: a metrics.json without the declared keys",
         lambda: gradient_config_without_keys(), ValueError, "is missing keys"),
        ("perseries metrics resolve_signal_selection: an unknown composite key",
         lambda: resolve_signal_selection(get_metric_config(), [PLANTED]), ValueError,
         f"Unknown signal metric composite key '{PLANTED}'"),
        ("perseries metrics prepare_direction_subset: an unknown direction",
         lambda: prepare_direction_subset(pd.DataFrame(), PLANTED), ValueError, f"Unknown direction: {PLANTED}"),
        ("metricscalculations._apply_operation: an unknown operation",
         lambda: _apply_operation(PLANTED, np.array([1.0], dtype=np.float32)), ValueError, f"Unknown operation: {PLANTED}"),
        ("plots plot_series: a frame without var3",
         lambda: plot_series("0", value_frame().drop(columns=["var3"]), pd.DataFrame({"x": [1]}), save_path=None, show=False),
         ValueError, "Expected 'var3' column"),
        ("gradient metrics shared_annualization: annualization differing across columns",
         lambda: shared_annualization(differing_annualization()), ValueError,
         "Annualization parameters differ across columns"),
        ("replot load_gradient_replot_params: no /gradient_params", lambda: load_gradient_replot_params(None, set()),
         KeyError, "/gradient_params is absent"),
        ("replot load_gradient_replot_params._req: a /gradient_params without a required column",
         lambda: params_without_columns(folder), ValueError, "Required param column"),
        ("Read._decode_index: a group without index_encoding", lambda: decoded_index(folder, config, None), ValueError,
         "index_encoding attribute not found"),
        ("Read._decode_index: an unknown index encoding", lambda: decoded_index(folder, config, "epoch_weeks"), ValueError,
         "Unknown index encoding: epoch_weeks"),
        ("RealisticModeMixin._window_bounds: an unknown window_type",
         lambda: testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, "trigger"))
         ._window_bounds(10, {"window_type": PLANTED, "lookback": 5}), ValueError, f"Unknown window_type: {PLANTED}"),
        ("plots plot_window_with_triggers: a window without var3",
         lambda: plot_window_with_triggers(series_id="0", window_df=value_frame().drop(columns=["var3"]), trigger_cols=[],
                                           iteration=1, current_date=DATES[0], debug_dir=str(folder / "debug")),
         ValueError, "'var3' column not found"),
        ("create_producer_consumer_pipeline: a producer that ends without the sentinel",
         lambda: create_producer_consumer_pipeline(lambda outlet: None, lambda item: True, queue_get_timeout=1), RuntimeError,
         "the producer ended without putting the sentinel"),
        ("ProducerOutlet.put: an item after the stop", lambda: stopped_outlet().put(1), ProducerStopped,
         "the pipeline stopped its producer"),
        ("HDF5Manager.open_store: an absent store (re-raised)", lambda: store_with_block(absent), FileNotFoundError, "0.h5"),
    ]


# =============================================================================
# RECORD AND CHAIN CASES
# =============================================================================


def series_record() -> None:
    """logged_series_failure: the block's own exception leaves, with one ERROR record naming the series."""
    with collected_errors() as collector:
        exc = testkit.raised(lambda: failing_block("7"))
    KIT.verdict("logged_series_failure: the block's OSError leaves", isinstance(exc, OSError) and testkit.PLANTED_ERROR in str(exc),
                testkit.PLANTED_ERROR, repr(exc))
    records = [message for message in collector.messages if "series 7" in message]
    KIT.verdict("logged_series_failure: one ERROR record names series 7", len(records) == 1, 1, collector.messages)


def producer_failure_chain() -> None:
    """create_producer_consumer_pipeline: the producer's OSError raised from the finally, the consumer's ValueError as context."""
    exc = testkit.raised(lambda: create_producer_consumer_pipeline(putting_then_failing, failing_consumer, queue_get_timeout=1))
    context = getattr(exc, "__context__", None)
    KIT.verdict("create_producer_consumer_pipeline: the producer's OSError, the consumer's ValueError as its __context__",
                isinstance(exc, OSError) and isinstance(context, ValueError) and CONSUMER_ERROR in str(context),
                {"raised": "OSError", "context": f"ValueError: {CONSUMER_ERROR}"}, {"raised": repr(exc), "context": repr(context)})


def undecodable_git_output() -> None:
    """_get_git_output: output that is not UTF-8 raises UnicodeDecodeError, logged at ERROR naming the command."""
    original = subprocess.run

    def writing(cmd, *args, **kwargs):
        os.write(kwargs["stdout"].fileno(), UNDECODABLE)
        return subprocess.CompletedProcess(cmd, 0)

    subprocess.run = writing
    try:
        with collected_errors() as collector:
            exc = testkit.raised(lambda: _get_git_output(list(GIT_COMMAND)))
    finally:
        subprocess.run = original
    KIT.verdict("_get_git_output: output that is not UTF-8 raises UnicodeDecodeError (re-raised)",
                isinstance(exc, UnicodeDecodeError), "UnicodeDecodeError", repr(exc))
    records = [message for message in collector.messages if " ".join(GIT_COMMAND) in message]
    KIT.verdict("_get_git_output: one ERROR record names the command", len(records) == 1, 1, collector.messages)


# =============================================================================
# HELPERS
# =============================================================================


def edited(config: dict, overrides: dict) -> dict:
    """A deep copy of the configuration with dotted-key overrides set."""
    copied = copy.deepcopy(config)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = copied
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return copied


def performance_task(config: dict) -> Performance:
    """The Performance task pipeline._create_task builds from the validated configuration (its files never read by
    the cases, which call _cache_config without run())."""
    return pipeline._create_task(AppConfig.model_validate(config), "performance")


def short_store_task(config: dict) -> Performance:
    """A Performance task whose lookback, as _cache_config sets it, exceeds the short store's rows."""
    task = performance_task(config)
    task._lookback = task.trigger.replay.lookback_window
    return task


def short_store(folder: Path) -> Path:
    """A trigger store whose /series_0 (format table) holds three rows."""
    path = folder / "short" / "0.h5"
    path.parent.mkdir(parents=True)
    with pd.HDFStore(str(path), mode="w") as store:
        store.put("/series_0", pd.DataFrame({"var3": [1.0, 2.0, 3.0]}, index=DATES[:3]), format="table")
    return path


def failing_strategy(config: dict) -> object:
    """Trigger._create_strategy with create_strategy planted to raise."""
    original = trigger_module.create_strategy

    def planted(*args: object, **kwargs: object) -> object:
        raise OSError(testkit.PLANTED_ERROR)

    trigger_module.create_strategy = planted
    try:
        return pipeline._create_task(AppConfig.model_validate(config), "trigger")._create_strategy()
    finally:
        trigger_module.create_strategy = original


def value_frame() -> pd.DataFrame:
    """Five rows of var0 and var3."""
    return pd.DataFrame({"var0": np.arange(5, dtype=float), "var3": np.arange(5, dtype=float) + 1.0}, index=DATES)


def gradient_config_without_keys() -> dict:
    """get_gradient_metric_config with its JSON loader planted to return no key."""
    original = gradient_metrics_module.load_json_config
    gradient_metrics_module.load_json_config = lambda caller_file: {}
    try:
        return get_gradient_metric_config()
    finally:
        gradient_metrics_module.load_json_config = original


def differing_annualization() -> pd.DataFrame:
    """A gradient metrics table whose two columns carry different N-1 and years rows."""
    return pd.DataFrame({"strategy": [10.0, 1.0, 0.1], "reference 0": [12.0, 1.5, 0.2]}, index=["_n_intervals", "_n_years", "acr"])


def params_without_columns(folder: Path) -> object:
    """load_gradient_replot_params on a store whose /gradient_params holds only weight."""
    path = folder / "params" / "0.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(str(path), mode="w") as store:
        store.put("/gradient_params", pd.DataFrame({"weight": [0.2]}), format="table")
    with pd.HDFStore(str(path), mode="r") as store:
        return load_gradient_replot_params(store, set(store.keys()))


def decoded_index(folder: Path, config: dict, encoding: object) -> pd.DatetimeIndex:
    """Read._decode_index on a group with the given index_encoding (None: no such attribute)."""
    path = folder / f"index_{encoding}.h5"
    with h5py.File(path, "w") as sink:
        group = sink.create_group("0")
        if encoding is not None:
            group.attrs["index_encoding"] = encoding
    with h5py.File(path, "r") as source:
        typed = AppConfig.model_validate(config)
        return Read(typed.read, [], typed.trigger.read.printout_io)._decode_index(source["0"])


def stopped_outlet() -> ProducerOutlet:
    """An outlet the pipeline has stopped."""
    outlet = ProducerOutlet(Queue())
    outlet.stop()
    return outlet


def store_with_block(path: Path) -> None:
    """open_store on the path in mode r, used in a with block."""
    with hdf5_manager.open_store(path, "r"):
        return None


def failing_block(series_id: str) -> None:
    """A block that raises inside logged_series_failure."""
    with logged_series_failure(series_id):
        raise OSError(testkit.PLANTED_ERROR)


def putting_then_failing(outlet: ProducerOutlet) -> None:
    """A producer that puts one item, then raises."""
    outlet.put(1)
    raise OSError(testkit.PLANTED_ERROR)


def failing_consumer(item: object) -> bool:
    """A consumer that raises on its item."""
    raise ValueError(CONSUMER_ERROR)


@contextmanager
def collected_errors() -> Iterator[testkit.RecordCollector]:
    """A RecordCollector at ERROR on the root logger inside the block."""
    collector = testkit.RecordCollector(logging.ERROR)
    root = logging.getLogger()
    root.addHandler(collector)
    try:
        yield collector
    finally:
        root.removeHandler(collector)


if __name__ == "__main__":
    main()
