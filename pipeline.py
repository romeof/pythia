"""
Main pipeline starting point for the Pythia time series analysis framework.

This script orchestrates the execution of Luigi tasks for:
- Trigger: Read, preprocessing, trigger detection, event extraction with trigger direction and outcome
- Signal: Signal selection
- Performance: Per-series analysis and gradient computation

Structure:
1. main
2. Argument parsing
3. Configuration setup (the merged config, the coloured log format, logging)
4. Pipeline execution (the tasks of each mode, their outputs and failures)
5. Finalization
"""

import argparse
import sys
import logging
import time
from datetime import datetime
from typing import List, Tuple

import luigi

from src.tasks.performance import Performance
from src.tasks.signal import Signal
from src.tasks.trigger import Trigger
from src.utils.common.config import load_config, validate_config
from src.utils.common.schema import AppConfig
from src.utils.common.outputs import (
    clean_task_output_directory,
    copy_config_to_output,
    get_output_file_name,
    remove_provenance_files,
    remove_task_output_directory,
)
from src.utils.common.constants import (
    PROFILE_DEFAULT,
    PROFILES,
    REPLAY_DUAL,
    REPLAY_IDEAL,
    REPLAY_REALISTIC,
    STRATEGY_EXTREMA,
    TASK_FAILURE_PREFIX,
    CLI_TRUE_VALUES,
)
from src.utils.common.strategyname import get_strategy_name

logger = logging.getLogger(__name__)


def main():
    """Main starting point for the pipeline."""
    args = _parse_arguments()
    config, config_file, profile = _setup_config(args)
    _setup_logging(config, config_file, profile, args.quiet_luigi,
                   get_strategy_name(config.trigger))
    luigi.Task.event_handler(luigi.Event.FAILURE)(_log_task_failure)

    tasks_to_run = _resolve_tasks(args.task)

    remove_provenance_files(config_file, config.read, config.trigger)
    last_config = _run_pipeline(config, tasks_to_run)

    _log_separator()
    _finalize_pipeline(last_config, config_file, profile)

    _log_separator()


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a specific Luigi task from the time series pipeline."
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=["trigger", "signal", "performance", "all"],
        default=["trigger"],
        help="Choose which task(s) to run. Use 'all' to run all tasks in sequence.",
    )
    parser.add_argument("--strategy", default=STRATEGY_EXTREMA, help="Choose trigger strategy")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=PROFILE_DEFAULT,
        help="Choose the switch profile merged beside the strategy YAML: 'dev' (its printouts and plots) or 'prod' "
        "(neither; default).",
    )

    def _parse_bool_or_default(v: str) -> bool:
        return str(v).lower() in CLI_TRUE_VALUES

    parser.add_argument(
        "--quiet-luigi",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_or_default,
        help="Suppress Luigi logging (default: True). Pass 'False' to enable Luigi logs.",
    )
    return parser.parse_args()


# =============================================================================
# CONFIGURATION SETUP
# =============================================================================


def _setup_config(args: argparse.Namespace) -> Tuple[AppConfig, str, str]:
    """Load and validate configuration; return it with the strategy YAML's path and the profile, the run arguments
    the logging and the provenance copy read."""
    config_file = f"workflows/{args.strategy}.yaml"
    config = load_config(config_file, args.profile)
    config = validate_config(config)
    return config, config_file, args.profile


class _ColoredFormatter(logging.Formatter):
    """Formatter that applies ANSI colors based on log level."""

    _RESET = "\033[0m"
    _COLORS = {
        logging.WARNING: "\033[38;5;208m",   # dark orange
        logging.ERROR: "\033[31m",           # red
        logging.CRITICAL: "\033[31m",        # red
    }

    def format(self, record: logging.LogRecord) -> str:
        if not record.getMessage():
            return ""
        msg = super().format(record)
        color = self._COLORS.get(record.levelno)
        if color:
            return f"{color}{msg}{self._RESET}"
        return msg


def _setup_logging(config: AppConfig, config_file: str, profile: str, quiet_luigi: bool,
                   strategy_name: str) -> None:
    """Configure logging for the pipeline; log the config, the profile and the strategy name."""
    if quiet_luigi:
        for logger_name in [
            "luigi",
            "luigi-interface",
            "luigi.worker",
            "luigi.scheduler",
            "luigi.task_register",
            "luigi.cmdline",
        ]:
            logging.getLogger(logger_name).disabled = True

    handler = logging.StreamHandler()
    handler.setFormatter(
        _ColoredFormatter("%(levelname)s %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(handler)

    # Suppress verbose third-party library logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    enable_info = config.general.enable_info

    if not enable_info:
        logging.disable(logging.INFO)

    sys.stderr.write("\n")
    logger.info(
        f"Using strategy with config: {config_file}, profile {profile}, "
        f"with INFO logging {'enabled' if enable_info else 'disabled'}."
    )
    logger.info(f"Strategy name: {strategy_name}")


def _log_task_failure(task: luigi.Task, exception: BaseException) -> None:
    """Luigi FAILURE handler: the cause of a failed run(), its only record in quiet mode; it never raises."""
    logger.error(f"{TASK_FAILURE_PREFIX} {task.task_id}: {type(exception).__name__}: {exception}",
                 exc_info=(type(exception), exception, exception.__traceback__))


def _resolve_tasks(task_args: List[str]) -> List[str]:
    """Resolve task arguments to actual task list."""
    if "all" in task_args:
        return ["trigger", "signal", "performance"]
    return task_args


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================


def _run_pipeline(config: AppConfig, tasks_to_run: List[str]) -> AppConfig:
    """Run the pipeline with specified tasks, expanding dual mode to individual runs.

    Before the first build the output folder of every (mode, task) pair the run will build is removed, so a run
    killed before its end leaves no previous output of a task it was to run; a task that fails leaves its pair
    and the later ones pending, and their folders are removed on the way out. Finished tasks keep theirs.

    Returns:
        The last resolved mode config, used by finalization for the provenance copy.
    """
    pipeline_start_time = time.time()
    logger.info(f"Pipeline start - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _log_separator()

    modes = _resolve_modes(config)
    mode_configs = [_clone_config_with_mode(config, mode) if len(modes) > 1 else config for mode in modes]
    pending = [(mode_config, task_name) for mode_config in mode_configs for task_name in tasks_to_run]
    _remove_task_outputs(pending)
    try:
        for mode, mode_config in zip(modes, mode_configs):
            mode_start_time = time.time()
            if len(modes) > 1:
                logger.info(f"Running mode: {mode}")
            for task_name in tasks_to_run:
                _run_task(mode_config, task_name)
                pending.pop(0)
            if len(modes) > 1:
                mode_duration = time.time() - mode_start_time
                logger.info(f"{mode.capitalize()} mode end - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(
                    f"{mode.capitalize()} mode duration: {mode_duration:.2f} seconds ({mode_duration / 60:.2f} minutes)"
                )
                logger.info(f"Tasks executed: {', '.join(tasks_to_run)}")
                _log_separator()
    finally:
        _remove_task_outputs(pending)

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time

    logger.info(f"Pipeline end - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(
        f"Total pipeline duration: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)"
    )
    logger.info(f"Tasks executed: {', '.join(tasks_to_run)}")
    return mode_configs[-1]


def _resolve_modes(config: AppConfig) -> List[str]:
    """Expand 'dual' mode into the two concrete modes; return a list of modes to run."""
    mode = config.trigger.replay.mode
    if mode == REPLAY_DUAL:
        return [REPLAY_IDEAL, REPLAY_REALISTIC]
    return [mode]


def _clone_config_with_mode(config: AppConfig, mode: str) -> AppConfig:
    """The configuration with trigger.replay.mode set to a concrete mode: a model_copy of the trigger section and its
    replay section, every other section shared."""
    replay = config.trigger.replay.model_copy(update={"mode": mode})
    return config.model_copy(update={"trigger": config.trigger.model_copy(update={"replay": replay})})


def _remove_task_outputs(pairs: List[Tuple[AppConfig, str]]) -> None:
    """Remove the output folder of every (mode config, task) pair; a folder that cannot be removed raises at once."""
    for mode_config, task_name in pairs:
        remove_task_output_directory(get_output_file_name(mode_config.read, mode_config.trigger, task_name))


def _run_task(config: AppConfig, task_name: str) -> None:
    """Run a single task with timing and logging, after removing its output folder so it recomputes."""
    logger.info(
        f"{task_name.capitalize()} task start - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    start_time = time.time()

    task = _create_task(config, task_name)
    clean_task_output_directory(task.target_file)
    if not luigi.build([task], local_scheduler=True):
        logger.error(f"{task_name.capitalize()} task failed; rerun with --quiet-luigi False for Luigi's own report")
        raise RuntimeError(f"{task_name} task failed")

    end_time = time.time()
    duration = end_time - start_time

    logger.info(
        f"{task_name.capitalize()} task end - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    logger.info(f"Duration: {duration:.2f} seconds ({duration / 60:.2f} minutes)")
    _log_separator()


def _create_task(config: AppConfig, task_name: str) -> luigi.Task:
    """Create the appropriate Luigi task based on task name, each given the sections it reads."""
    read, trigger, runtime = config.read, config.trigger, config.runtime
    trigger_file = str(get_output_file_name(read, trigger, "trigger"))
    if task_name == "trigger":
        return Trigger(read=read, trigger=trigger, runtime=runtime, target_file=trigger_file)

    elif task_name == "signal":
        return Signal(
            trigger=trigger,
            signal=config.signal,
            runtime=runtime,
            target_file=str(get_output_file_name(read, trigger, "signal")),
            trigger_file=trigger_file,
        )

    elif task_name == "performance":
        return Performance(
            read=read,
            trigger=trigger,
            performance=config.performance,
            runtime=runtime,
            target_file=str(get_output_file_name(read, trigger, "performance")),
            trigger_file=trigger_file,
            signal_file=str(get_output_file_name(read, trigger, "signal")),
        )

    raise ValueError(f"Unknown task: {task_name}")


def _log_separator() -> None:
    """Log a blank line for visual separation."""
    sys.stderr.write("\n")


# =============================================================================
# FINALIZATION
# =============================================================================


def _finalize_pipeline(config: AppConfig, config_file: str, profile: str) -> None:
    """Finalize pipeline: copy the provenance files."""
    copy_config_to_output(config_file, profile, config.read, config.trigger)


if __name__ == "__main__":
    main()
