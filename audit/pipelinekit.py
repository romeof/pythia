"""Shared in-process helpers of the audit experiments that run pythia code.

Imported only after the experiment's preamble has put its source root first on
sys.path, so every src import below resolves to that root.

Structure:
1. Config copies and the startup refusal (startup_refusal, StartupOutcome; GATE_OFF and gate_item, the gate
   overrides)
2. Input series and the observed dates of the curve index (observed_dates, observed_index)
3. Strategy runs
"""
import sys

sys.dont_write_bytecode = True

import dataclasses
import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import yaml

import harness
from src.utils.common.config import validate_config
from src.utils.common.constants import SUPPRESSION_BEFORE_EFFECTIVE_START, SUPPRESSION_TEMPORAL_CONSTRAINT, VAR3, VAR4
from src.utils.common.outputs import clean_task_output_directory, get_output_file_name
from src.utils.common.schema import AppConfig
from src.utils.common.strategyname import get_strategy_name
from src.utils.preprocess.preprocess import Preprocess
from src.utils.preprocess.read import Read
from src.utils.trigger.factory import create_strategy
from src.utils.trigger.gate import gate_columns
from src.utils.trigger.realistic import RealisticModeMixin

STRATEGY_FILE = "extrema.yaml"
DEV_FILE = "dev.yaml"
OUTPUT_DIR = "out"
REPORT_STEM = "input"
COMPARER = "exp13compare.py"
WORKFLOWS = harness.REAL_WORKTREE / "workflows"
REFUSAL_ERROR = "ValidationError"  # the startup refusal's exception, named in the child's capture
TASK_START_LINE = "Trigger task start"  # logged once the Trigger task begins (pipeline.py), absent after a refusal
REFERENCE_KEY_PREFIX = "/reference_series_"
# No gate item and the pre-filter off, the counterpart of the regressions' testkit.GATE_OFF: the overrides of every
# configuration that runs detection on a synthetic var0-var5 frame, which holds none of the shipped items' columns
GATE_OFF = {"trigger.gate.items": [], "trigger.gate.prefilter": False}


# =============================================================================
# CONFIG COPIES AND THE STARTUP REFUSAL
# =============================================================================


@dataclasses.dataclass(frozen=True)
class StartupOutcome:
    """A Trigger launch on a config copy the validation should refuse: the child and the three facts of a refusal."""

    child: harness.ChildResult
    failed: bool
    validation_error: bool
    task_started: bool


def startup_refusal(ctx: harness.Context, folder: Path, overrides: dict) -> StartupOutcome:
    """The Trigger task on a config copy with the dotted overrides, written into the unused folder: a return code
    outside (0, None), the validation error in the capture and no task start line make a startup refusal; logged,
    judged by the caller."""
    config_file = write_config_copy(ctx, folder, strategy=overrides)
    child, _, _ = ctx.run_pipeline("pipeline", strategy=strategy_argument(config_file), tasks=["trigger"], quiet_luigi=True,
                                   hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder, overrides=overrides)
    outcome = StartupOutcome(child, child.return_code not in (0, None), REFUSAL_ERROR in child.text,
                             TASK_START_LINE in child.text)
    ctx.log.info("startup refusal %s: return code %s, %s %s, task started %s (%s)", folder.name, child.return_code,
                 REFUSAL_ERROR, outcome.validation_error, outcome.task_started, child.log_path.name)
    return outcome


def write_config_copy(ctx: harness.Context, folder: Path, *, strategy: Optional[dict] = None,
                      dev: Optional[dict] = None, drop_strategy: tuple = (), drop_dev: tuple = ()) -> Path:
    """Copies of workflows\\extrema.yaml and dev.yaml in folder, with outputs under folder\\out.

    Dotted-key overrides and removals apply to each copy; read.output_root always points
    under the folder (relative to the worktree cwd). Returns the strategy copy's path.
    """
    ctx.guard(folder / STRATEGY_FILE, folder / DEV_FILE, folder / OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    rel = folder.resolve().relative_to(harness.REAL_WORKTREE).as_posix()
    strategy_doc = _edited(WORKFLOWS / STRATEGY_FILE, {"read.output_root": f"./{rel}/{OUTPUT_DIR}", **(strategy or {})},
                           drop_strategy)
    dev_doc = _edited(WORKFLOWS / DEV_FILE, dev or {}, drop_dev)
    for name, document in ((STRATEGY_FILE, strategy_doc), (DEV_FILE, dev_doc)):
        with open(folder / name, "x", encoding="utf-8") as sink:
            yaml.safe_dump(document, sink, sort_keys=False)
    ctx.log.info("config copy %s: overrides %s, dropped %s; dev overrides %s, dropped %s",
                 folder, strategy, list(drop_strategy), dev, list(drop_dev))
    return folder / STRATEGY_FILE


def strategy_argument(config_file: Path) -> str:
    """The --strategy value pipeline.py turns into workflows/<value>.yaml for this copy."""
    return Path(os.path.relpath(config_file.with_suffix(""), WORKFLOWS)).as_posix()


def _edited(source: Path, overrides: dict, removals: tuple) -> dict:
    """A YAML document with dotted-key overrides set and dotted keys removed."""
    return edited_document(yaml.safe_load(source.read_text(encoding="utf-8")), overrides, removals)


def edited_document(document: dict, overrides: dict, removals: tuple) -> dict:
    """A mapping with dotted-key overrides set (missing parents created) and dotted keys removed, in place."""
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = document
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    for dotted in removals:
        *parents, leaf = dotted.split(".")
        node = document
        for part in parents:
            node = node[part]
        del node[leaf]
    return document


def load_copy(config_file: Path) -> AppConfig:
    """The validated configuration of a copy (harness.load_root_config: the dev profile), as pipeline.py validates
    it: the model with the schema defaults applied (W7, Q11)."""
    config, _ = harness.load_root_config(str(config_file))
    return validate_config(config)


def concrete(config: AppConfig, mode: str) -> AppConfig:
    """The configuration with trigger.replay.mode set to one concrete mode, pipeline.py's own mode copy."""
    import pipeline

    return pipeline._clone_config_with_mode(config, mode)


def with_values(config: AppConfig, overrides: dict) -> AppConfig:
    """The configuration with dotted-key values set, validated again, so an experiment's variant passes the schema
    as a copy's YAML would."""
    return AppConfig.model_validate(edited_document(config.model_dump(), overrides, ()))


def gate_item(column: str, op: str, threshold: float, exempt_series: Sequence[str] = ()) -> dict:
    """One trigger.gate.items mapping as an override takes it: the threshold a Python float and the exempt ids a list,
    the types yaml.safe_dump writes (write_config_copy; it refuses a numpy scalar)."""
    return {"column": column, "op": op, "threshold": float(threshold), "exempt_series": list(exempt_series)}


def output_file(config_file: Path, mode: str, task: str) -> Path:
    """The task output a copy resolves to in one concrete mode (absolute)."""
    config = concrete(load_copy(config_file), mode)
    return get_output_file_name(config.read, config.trigger, task).resolve()


def store_keys(path: Path) -> Optional[list[str]]:
    """Sorted keys of an output store read in mode r, None when the file is absent."""
    if not path.is_file():
        return None
    with pd.HDFStore(str(path), mode="r") as store:
        return sorted(store.keys())


# =============================================================================
# INPUT SERIES AND THE OBSERVED DATES OF THE CURVE INDEX
# =============================================================================


def raw_series(config: AppConfig) -> dict[str, pd.DataFrame]:
    """Every series as Read yields it (date filter and skip_series applied, before preprocessing)."""
    return dict(reader_of(config).iter_series())


def input_series(config: AppConfig, report_folder: Path) -> dict[str, pd.DataFrame]:
    """Every series read and preprocessed as the Trigger producer does (report under report_folder)."""
    preprocessor = preprocessor_of(config, report_folder / f"{REPORT_STEM}.h5")
    return {series_id: preprocessor.process_series(series_id, df) for series_id, df in raw_series(config).items()}


def reader_of(config: AppConfig) -> Read:
    """The Read the Trigger producer builds: the read section, the columns trigger.gate.items names and
    trigger.read.printout_io."""
    return Read(config.read, gate_columns(config.trigger.gate.items), config.trigger.read.printout_io)


def preprocessor_of(config: AppConfig, output_file: Path) -> Preprocess:
    """The Preprocess the Trigger producer builds, its report beside output_file."""
    return Preprocess(config.trigger.consecutive_rows_with_bad_preprocessed_data, config.trigger.validity,
                      config.trigger.preprocess.printout_io, output_file)


def observed_dates(trigger_file: Path, signal_file: Path, max_references: int) -> list[pd.DatetimeIndex]:
    """The observed dates the curve index unites, by their two definitions: each processed series' index in the
    trigger file's /series_* frames, and each reference's var3 dates without its missing values in the signal file's
    /reference_series_* frames, the first max_references references in sorted sequence."""
    with pd.HDFStore(str(trigger_file), mode="r") as store:
        dates = [pd.DatetimeIndex(store.get(key).index) for key in sorted(store.keys())
                 if key.startswith(harness.SERIES_KEY_PREFIX)]
    with pd.HDFStore(str(signal_file), mode="r") as store:
        references = [store.get(key) for key in sorted(store.keys()) if key.startswith(REFERENCE_KEY_PREFIX)]
    kept = [frame for frame in references if len(frame) and VAR3 in frame.columns][:max_references]
    return dates + [pd.DatetimeIndex(pd.to_datetime(frame[VAR3].dropna().index)) for frame in kept]


def observed_index(dates: list[pd.DatetimeIndex], start: pd.Timestamp, end_date: str) -> pd.DatetimeIndex:
    """The sorted, duplicate-free union of the observed dates within [start at midnight, end_date]."""
    union = pd.DatetimeIndex(sorted({date for index in dates for date in index}))
    return union[(union >= start.normalize()) & (union <= pd.Timestamp(end_date))]


# =============================================================================
# STRATEGY RUNS
# =============================================================================


class DiscardStore:
    """Throwaway store for analyze_series: records the keys it receives, keeps no data."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def put(self, key: str, value: pd.DataFrame, **kwargs: object) -> None:
        """Accept a frame and discard it."""
        self.keys.append(key)


def analyze(series_id: str, df: pd.DataFrame, config: AppConfig) -> Optional[pd.DataFrame]:
    """analyze_series of the configured strategy on a fresh copy of the frame, with a discarding store."""
    return strategy_of(config).analyze_series(series_id, df.copy(), store=DiscardStore())


def strategy_of(config: AppConfig):
    """The configured trigger strategy as the Trigger task builds it: create_strategy on the trigger section, which
    holds the recording mode, its output file the copy's trigger output (which analyze's discarding store never
    writes)."""
    return create_strategy(config.trigger, get_output_file_name(config.read, config.trigger, "trigger"))


def luigi_build(config_file: Path, task_arguments: list[str], *, quiet_luigi: bool = True) -> list[list]:
    """pipeline.py's config, logging, modes and tasks, one luigi.build per task after the cleanup _run_task performs
    first (Q12; on the audited revision output() cleans as well): [mode, task, return value]."""
    import luigi
    import pipeline

    config = load_copy(config_file)
    pipeline._setup_logging(config, str(config_file), harness.DEV_PROFILE, quiet_luigi,
                            get_strategy_name(config.trigger))
    modes = pipeline._resolve_modes(config)
    results = []
    for mode in modes:
        mode_config = pipeline._clone_config_with_mode(config, mode) if len(modes) > 1 else config
        for task_name in pipeline._resolve_tasks(task_arguments):
            task = pipeline._create_task(mode_config, task_name)
            clean_task_output_directory(task.target_file)
            results.append([mode, task_name, bool(luigi.build([task], local_scheduler=True))])
    return results


def compare_trees(ctx: harness.Context, role: str, left: Path, right: Path, *, h5_only: bool) -> dict:
    """EXP-13 on two trees in a child of its own; its verdict and comparison line."""
    args = [str(harness.HARNESS_DIR / COMPARER), str(left), str(right)] + (["--h5-only"] if h5_only else [])
    child = ctx.launch_child(role, args, timeout_s=harness.CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED,
                             audit_child=True)
    return {"return_code": child.return_code, "identical": "PASS identical value by value" in child.text,
            "compared": last_prefixed(child.text, "COMPARED "), "log": child.log_path.name,
            "differences": [line.split("DIFF ", 1)[1] for line in child.text.splitlines() if " DIFF " in line][:10]}


def last_prefixed(text: str, prefix: str) -> str:
    """Payload of the last line that carries the prefix, empty when none does."""
    lines = [line.split(prefix, 1)[1].strip() for line in text.splitlines() if prefix in line]
    return lines[-1] if lines else ""


def pipeline_main(arguments: list[str]) -> None:
    """pipeline.main() of the source root with the given command-line arguments (for patched children)."""
    import pipeline

    sys.argv = ["pipeline.py", *arguments]
    pipeline.main()


def realistic_gate_rows(series_id: str, df: pd.DataFrame, config: AppConfig) -> list[int]:
    """Rows at which the realistic loop evaluates the gate for a candidate (recorded run): the calls of
    _check_suppression_conditions that pass its two checks before the gate, the effective start and the temporal
    constraint."""
    original = RealisticModeMixin._check_suppression_conditions
    rows: list[int] = []

    def recording(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = original(self, i, candidate_date, trigger_type, state, effective_start_date)
        if reason not in (SUPPRESSION_BEFORE_EFFECTIVE_START, SUPPRESSION_TEMPORAL_CONSTRAINT):
            rows.append(i)
        return reason

    RealisticModeMixin._check_suppression_conditions = recording
    try:
        analyze(series_id, df, config)
    finally:
        RealisticModeMixin._check_suppression_conditions = original
    return sorted(set(rows))


def binding_gate(series: dict[str, pd.DataFrame], rows: dict[str, list[int]]) -> tuple[float, dict[str, list[int]]]:
    """A binding var4 item over the candidate rows: its threshold, the lower middle value of gate_input over every
    series' candidate rows (a value of one of them, so the item compares exactly in the column's dtype), and per series
    the candidate rows the item var4 >= threshold blocks, those whose gate_input lies below it."""
    inputs = sorted(gate_input(frame, row) for sid, frame in sorted(series.items()) for row in rows[sid])
    threshold = inputs[(len(inputs) - 1) // 2]
    blocked = {sid: [row for row in rows[sid] if gate_input(frame, row) < threshold] for sid, frame in sorted(series.items())}
    return threshold, blocked


def gate_input(df: pd.DataFrame, row: int) -> float:
    """var4 on the bar before a candidate row, what the audit's binding var4 items compare (a candidate row is never
    row 0: find_peaks marks no end sample, and the realistic loop starts at lookback_window - 1)."""
    return float(df[VAR4].iat[row - 1])
