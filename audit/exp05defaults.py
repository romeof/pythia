"""EXP-05: schema defaults, strict validation and merge precedence (HYP-05, Q11, R-X2, R-X6).

(a) A realistic config copy omitting trigger.replay.window_type (the schema declares a
    default, which Q11 applies through the validated model): validation, then the Trigger task
    through luigi.build in a child, then pipeline.py --task trigger on the copy.
(b) A copy with one unknown key, launched through pipelinekit.startup_refusal: a ValidationError before any
    task and a non-zero return code.
(c) A leaf key set differently in the strategy and dev copies: the merged config holds the
    dev.yaml value, while mapping values merge key by key.

Structure:
1. main
2. Case (a)
3. Case (b)
4. Case (c)
5. Child command
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
import re
from pathlib import Path

import pipelinekit
from src.utils.common.config import validate_config

EXPERIMENT = "05"
MISSING_KEY = "trigger.replay.window_type"
MISSING_LINE = f"Missing required configuration key: {MISSING_KEY}"
INPUT_MISSING = "[Errno 2]"
UNKNOWN_KEY = "read.planted_unknown_key"
LEAF_KEY = "trigger.strategy.printout_io"
CHILD_TIMEOUT_S = 600.0
SERIES_COUNT = 6
FRAME = re.compile(r'File "([^"]+)", line \d+')
NO_FILE_FRAME = "<"  # <string> and <frozen runpy>: the frames harness.ORIGIN_CHILD's launch adds, with no file behind them


def main() -> None:
    """Child command when requested, else cases (a), (b), (c)."""
    args = _parse_args()
    if args.child_build is not None:
        child_build(Path(args.child_build))
        return
    folder = CTX.label_dir(EXPERIMENT)
    results = {"a": check_omitted_default(folder / "a"), "b": check_unknown_key(folder / "b"),
               "c": check_merge_precedence(folder / "c")}
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root, and --child-build CONFIG for the luigi.build child."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument("--child-build")
    return parser.parse_args()


# =============================================================================
# CASE (A): AN OMITTED KEY WITH A SCHEMA DEFAULT
# =============================================================================


def check_omitted_default(folder: Path) -> dict:
    """Validation passes, every series fails at runtime, the task and the run report success."""
    config_file = pipelinekit.write_config_copy(CTX, folder, strategy={"trigger.replay.mode": "realistic"},
                                                drop_strategy=(MISSING_KEY,))
    merged, _ = harness.load_root_config(str(config_file))
    validate_config(merged)
    CTX.verdict("(a) validation passes without the key", "window_type" not in merged["trigger"]["replay"],
                "valid, key absent from the merged dict", sorted(merged["trigger"]["replay"]))
    build = CTX.launch_child("build", [str(Path(__file__).resolve()), "--child-build", str(config_file),
                                       "--source-root", str(CTX.source_root)],
                             timeout_s=CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED, audit_child=True)
    returned = json.loads(pipelinekit.last_prefixed(build.text, "LUIGI_BUILD ") or "null")
    output = pipelinekit.output_file(config_file, "realistic", "trigger")
    keys_after_build = pipelinekit.store_keys(output)
    child, _, origin = CTX.run_pipeline("pipeline", strategy=pipelinekit.strategy_argument(config_file), tasks=["trigger"],
                                        quiet_luigi=True, hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder,
                                        overrides={"trigger.replay.mode": "realistic"})
    keys_after_run = pipelinekit.store_keys(output)
    found = {"build_return_code": build.return_code, "build_missing_lines": build.text.count(MISSING_LINE),
             "luigi_build": returned, "keys_after_build": keys_after_build, "run_return_code": child.return_code,
             "run_missing_lines": child.text.count(MISSING_LINE), "run_errno_lines": child.text.count(INPUT_MISSING),
             "keys_after_run": keys_after_run, "origin": origin["confirmed"]}
    CTX.log.info("(a) %s", json.dumps(found, default=str))
    CTX.verdict("(a) the input was read (no [Errno 2])", found["run_errno_lines"] == 0, 0, found["run_errno_lines"])
    CTX.verdict("(a) observed per the trace: one missing-key line per series, empty outputs, success reported",
                found["build_missing_lines"] == SERIES_COUNT and found["run_missing_lines"] == SERIES_COUNT
                and returned == [["realistic", "trigger", True]] and keys_after_run == [] and child.return_code == 0,
                {"missing_lines": SERIES_COUNT, "luigi_build": [["realistic", "trigger", True]], "keys": [], "return_code": 0}, found)
    series_keys = [key for key in keys_after_run or [] if key.startswith("/series_")]
    CTX.verdict("(a) intent (Q11): the run completes with the schema default, return code 0 and every series written",
                child.return_code == 0 and len(series_keys) == SERIES_COUNT, {"return_code": 0, "series": SERIES_COUNT},
                {"return_code": child.return_code, "series": len(series_keys)})
    return found


# =============================================================================
# CASE (B): AN UNKNOWN KEY
# =============================================================================


def check_unknown_key(folder: Path) -> dict:
    """A ValidationError before any task, a non-zero return code, frames from the source root."""
    outcome = pipelinekit.startup_refusal(CTX, folder, {UNKNOWN_KEY: 1})
    text = outcome.child.text
    frames = [Path(match) for match in FRAME.findall(text)
              if "site-packages" not in match and not match.startswith(NO_FILE_FRAME)]
    foreign = [str(frame) for frame in frames if not frame.resolve().is_relative_to(CTX.source_root.resolve())]
    found = {"return_code": outcome.child.return_code, "validation_error": outcome.validation_error,
             "extra_forbidden": "extra_forbidden" in text or "Extra inputs are not permitted" in text,
             "task_started": outcome.task_started, "frames": len(frames), "foreign_frames": foreign}
    CTX.log.info("(b) %s", json.dumps(found))
    CTX.verdict("(b) origin: every pythia traceback frame under the source root", bool(frames) and not foreign,
                "frames under the source root", found)
    CTX.verdict("(b) ValidationError before any task, return != 0", found["validation_error"] and found["extra_forbidden"]
                and not found["task_started"] and outcome.failed, "validation error, return != 0", found)
    return found


# =============================================================================
# CASE (C): MERGE PRECEDENCE
# =============================================================================


def check_merge_precedence(folder: Path) -> dict:
    """The dev.yaml value wins for a leaf present in both files; mappings merge key by key."""
    config_file = pipelinekit.write_config_copy(CTX, folder, strategy={LEAF_KEY: True}, dev={LEAF_KEY: False})
    merged, _ = harness.load_root_config(str(config_file))
    leaf = merged["trigger"]["strategy"]["printout_io"]
    kept = {"strategy_name": merged["trigger"]["strategy"].get("strategy_name"), "min_length": merged["trigger"].get("min_length"),
            "dev_read_section": "read" in merged["trigger"]}
    CTX.verdict("(c) leaf in both files: the dev.yaml value wins", leaf is False, False, leaf)
    CTX.verdict("(c) mappings merge key by key", kept == {"strategy_name": "extrema", "min_length": 5, "dev_read_section": True},
                {"strategy_name": "extrema", "min_length": 5, "dev_read_section": True}, kept)
    return {"leaf": leaf, "kept": kept}


# =============================================================================
# CHILD COMMAND
# =============================================================================


def child_build(config_file: Path) -> None:
    """Child: the Trigger task through luigi.build; its return values logged."""
    CTX.log.info("LUIGI_BUILD %s", json.dumps(pipelinekit.luigi_build(config_file, ["trigger"])))


if __name__ == "__main__":
    main()
