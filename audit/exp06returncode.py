"""EXP-06: return code and failure visibility (HYP-08, HYP-24, HYP-27, KNOWN (b)).

Case 1: an absent input inside data\\exp\\06 (the <date>/<dataset>/raw/<file> depth),
        --task trigger and --task all, each quiet and with --quiet-luigi False, each run on
        its own output root.
Case 2: the real input with performance.gradient.max_references 0 in the dev copy: the
        Performance task raises in its config check.
Case 3: the real input with series 0 only and one gate item, var4 >= twice series 0's highest
        var4, which fails on every row: no trigger, no signal.
For the failing builds a child calls luigi.build in-process and logs its return values.

Structure:
1. main
2. Cases
3. One run and its observations
4. Child command
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

import pipelinekit
from src.utils.common.constants import GATE_OP_GE, VAR4

EXPERIMENT = "06"
KNOWN_B = "FileNotFoundError: [Errno 2] Unable to synchronously"
STORE_LINE = "Error opening HDF5 store"
NO_SIGNALS = "No signals found in signal HDF5"
IS_SIGNAL_ERROR = "KeyError: 'is_signal'"
MAX_REFERENCES_ERROR = "max_references is 0"
FAILED_SUMMARY = "This progress looks :("
CHILD_TIMEOUT_S = 900.0
MODES = ("ideal", "realistic")
TASKS = ("trigger", "signal", "performance")
KEPT_SERIES = "0"
GATE_MARGIN = 2.0


def main() -> None:
    """Child command when requested, else the three cases."""
    args = _parse_args()
    if args.child_build is not None:
        CTX.log.info("LUIGI_BUILD %s", json.dumps(pipelinekit.luigi_build(Path(args.child_build), args.tasks)))
        return
    folder = CTX.label_dir(EXPERIMENT)
    results = {"case1": case_absent_input(folder / "case1"), "case2": case_max_references(folder / "case2"),
               "case3": case_zero_triggers(folder / "case3")}
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root; --child-build CONFIG --tasks ... for the luigi.build child."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument("--child-build")
    parser.add_argument("--tasks", nargs="+", default=["all"])
    return parser.parse_args()


# =============================================================================
# CASES
# =============================================================================


def case_absent_input(folder: Path) -> dict:
    """KNOWN (b) generalised: an absent input returns 0 with empty outputs; Performance fails out of sight."""
    found = {}
    for tasks in (["trigger"], ["all"]):
        for quiet in (True, False):
            name = f"{tasks[0]}_{'quiet' if quiet else 'verbose'}"
            run_folder = folder / name
            absent = run_folder / "in" / harness.DATA_OUTPUT_DIR / "0" / "raw" / "0.h5"
            raw_data = "./" + absent.resolve().relative_to(harness.REAL_WORKTREE).as_posix()
            config_file = pipelinekit.write_config_copy(CTX, run_folder, strategy={"read.raw_data": raw_data})
            found[name] = observe(name, config_file, tasks, quiet, run_folder, {"read.raw_data": raw_data})
            CTX.verdict(f"case 1 {name}: input absent as planted", not absent.exists() and found[name]["known_b"] >= 1,
                        "absent, KNOWN (b) line", {"exists": absent.exists(), "known_b": found[name]["known_b"]})
            CTX.verdict(f"case 1 {name}: observed per the trace (return code 0, trigger stores without keys)",
                        found[name]["return_code"] == 0 and all(keys == [] for keys in found[name]["keys"]["trigger"].values()),
                        "return code 0, empty trigger stores", {"return_code": found[name]["return_code"], "trigger": found[name]["keys"]["trigger"]})
            CTX.verdict(f"case 1 {name}: intent (R-X3) return != 0", found[name]["return_code"] not in (0, None), "return != 0", found[name]["return_code"])
        if tasks == ["all"]:
            build = child_build(found["all_quiet"]["config"], tasks)
            found["all_build"] = build
            CTX.verdict("case 1 all: signal stores hold only /strategy_name; no performance store; luigi.build False for Performance",
                        all(keys == ["/strategy_name"] for keys in found["all_quiet"]["keys"]["signal"].values())
                        and all(keys is None for keys in found["all_quiet"]["keys"]["performance"].values())
                        and [item[2] for item in build if item[1] == "performance"] == [False, False],
                        "signal ['/strategy_name'], performance None, [False, False]",
                        {"signal": found["all_quiet"]["keys"]["signal"], "performance": found["all_quiet"]["keys"]["performance"],
                         "build": build})
            CTX.verdict("case 1 all: the Performance KeyError is visible only with --quiet-luigi False",
                        found["all_quiet"]["is_signal_error"] == 0 and found["all_verbose"]["is_signal_error"] > 0
                        and found["all_verbose"]["failed_summaries"] > 0, "hidden when quiet, shown verbose",
                        {name: (found[name]["is_signal_error"], found[name]["failed_summaries"]) for name in ("all_quiet", "all_verbose")})
    return found


def case_max_references(folder: Path) -> dict:
    """A schema-valid max_references of 0 with one reference series: Performance raises, the run returns 0."""
    found = {}
    for quiet in (True, False):
        name = "quiet" if quiet else "verbose"
        config_file = pipelinekit.write_config_copy(CTX, folder / name, strategy={"trigger.replay.mode": "realistic"},
                                                    dev={"performance.gradient.max_references": 0})
        found[name] = observe(f"case2 {name}", config_file, ["all"], quiet, folder / name,
                              {"performance.gradient.max_references": 0})
    found["build"] = child_build(found["quiet"]["config"], ["all"])
    CTX.verdict("case 2: observed per the trace (ValueError, no performance store, return code 0, luigi.build False)",
                found["quiet"]["return_code"] == 0 and found["verbose"]["max_references_error"] > 0
                and found["quiet"]["keys"]["performance"]["realistic"] is None
                and [item[2] for item in found["build"] if item[1] == "performance"] == [False],
                "ValueError, None, return code 0, [False]", {"build": found["build"], "return_code": found["quiet"]["return_code"],
                                                      "error_lines": found["verbose"]["max_references_error"]})
    CTX.verdict("case 2: intent (R-X3) return != 0", found["quiet"]["return_code"] not in (0, None), "return != 0", found["quiet"]["return_code"])
    return found


def case_zero_triggers(folder: Path) -> dict:
    """A var4 item no row of series 0 passes: no trigger, no /signals, Performance's KeyError."""
    probe_file = pipelinekit.write_config_copy(CTX, folder / "probe", strategy={"read.skip_series": ["1", "2", "3", "4", "5"]})
    config = pipelinekit.load_copy(probe_file)
    frame = pipelinekit.raw_series(config)[KEPT_SERIES]
    highest = float(np.max(frame[VAR4].to_numpy(dtype=float)))
    threshold = GATE_MARGIN * highest
    CTX.log.info("case 3: series %s highest var4 %s, item var4 >= %s", KEPT_SERIES, highest, threshold)
    found = {"highest": highest, "threshold": threshold}
    for quiet in (True, False):
        name = "quiet" if quiet else "verbose"
        overrides = {"trigger.replay.mode": "realistic", "read.skip_series": ["1", "2", "3", "4", "5"],
                     "trigger.gate.items": [pipelinekit.gate_item(VAR4, GATE_OP_GE, threshold)]}
        config_file = pipelinekit.write_config_copy(CTX, folder / name, strategy=overrides)
        found[name] = observe(f"case3 {name}", config_file, ["all"], quiet, folder / name, overrides)
    keys = found["quiet"]["keys"]
    CTX.verdict("case 3: non-vacuity (no /triggers_* key in the trigger store)",
                keys["trigger"]["realistic"] is not None and not any(key.startswith("/triggers_") for key in keys["trigger"]["realistic"]),
                "a trigger store without /triggers_*", keys["trigger"]["realistic"])
    CTX.verdict("case 3: observed per the trace (no /signals, warning, KeyError, no performance store, return code 0)",
                "/signals" not in (keys["signal"]["realistic"] or []) and found["quiet"]["no_signals"] > 0
                and found["verbose"]["is_signal_error"] > 0 and keys["performance"]["realistic"] is None
                and found["quiet"]["return_code"] == 0, "as traced", {"keys": keys, "return_code": found["quiet"]["return_code"]})
    CTX.verdict("case 3: intent (R-P9) the documented empty outputs with return code 0",
                keys["performance"]["realistic"] is not None and found["quiet"]["return_code"] == 0
                and found["verbose"]["is_signal_error"] == 0, "a performance store, return code 0",
                {"performance": keys["performance"]["realistic"], "return_code": found["quiet"]["return_code"]})
    return found


# =============================================================================
# ONE RUN AND ITS OBSERVATIONS
# =============================================================================


def observe(name: str, config_file: Path, tasks: list[str], quiet: bool, folder: Path, overrides: dict) -> dict:
    """Run pipeline.py on the copy and record return code, decisive lines and every output store's keys."""
    child, _, origin = CTX.run_pipeline("pipeline", strategy=pipelinekit.strategy_argument(config_file), tasks=tasks,
                                        quiet_luigi=quiet, hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder,
                                        overrides=overrides, timeout_s=CHILD_TIMEOUT_S)
    text = child.text
    found = {"config": str(config_file), "return_code": child.return_code, "timed_out": child.timed_out, "origin": origin["confirmed"],
             "known_b": text.count(KNOWN_B), "store_line": text.count(STORE_LINE), "no_signals": text.count(NO_SIGNALS),
             "is_signal_error": text.count(IS_SIGNAL_ERROR), "max_references_error": text.count(MAX_REFERENCES_ERROR),
             "failed_summaries": text.count(FAILED_SUMMARY), "error_lines": len(harness.error_lines(harness.plain_lines(text))),
             "log": child.log_path.name,
             "keys": {task: {mode: pipelinekit.store_keys(pipelinekit.output_file(config_file, mode, task)) for mode in MODES}
                      for task in TASKS}}
    CTX.log.info("%s: %s", name, json.dumps(found, default=str))
    return found


# =============================================================================
# CHILD COMMAND
# =============================================================================


def child_build(config_file: str, tasks: list[str]) -> list:
    """luigi.build in a child of this script; its [mode, task, return value] records."""
    child = CTX.launch_child("build", [str(Path(__file__).resolve()), "--child-build", config_file, "--tasks", *tasks,
                                       "--source-root", str(CTX.source_root)],
                             timeout_s=CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED, audit_child=True)
    return json.loads(pipelinekit.last_prefixed(child.text, "LUIGI_BUILD ") or "null")


if __name__ == "__main__":
    main()
