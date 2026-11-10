"""EXP-18: the preprocess report under the platform's ANSI code page (HYP-12).

1. A synthetic series with three bad rows on calendar-consecutive dates goes through
   Preprocess.process_series in two children: an ANSI child (-X utf8=0, PYTHONUTF8 removed,
   its code page recorded) and a UTF-8 child. Intent: identical report and frame in both.
2. pipeline --task trigger, without a gate (pipelinekit.GATE_OFF), on a copy of the six input
   series (written by synthetic.py) with var5 = 0 on three calendar-consecutive weekdays of
   series 0, once per environment. Intent: six series in both trigger outputs.

Structure:
1. main
2. Preprocess children
3. Pipeline children
4. Child commands
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import hashlib
import json
import locale
from pathlib import Path

import numpy as np
import pandas as pd

import pipelinekit
import synthetic
from src.utils.common.constants import VAR5

EXPERIMENT = "18"
BAD_DATES = ("2020-06-09", "2020-06-10", "2020-06-11")
SYNTHETIC_SIZE = 60
SYNTHETIC_START = "2020-05-04"
SERIES_ID = "0"
CONSECUTIVE = "Consecutive Issues"
ENCODE_ERROR = "UnicodeEncodeError"
CHARACTER = "\\u2265"
REPORT_SYMBOL = "≥"
PREPROCESS_ERROR = "Error preprocessing series 0"
CHILD_TIMEOUT_S = 900.0
MODE = "realistic"


def main() -> None:
    """Child commands when requested, else the preprocess children and the pipeline children."""
    args = _parse_args()
    if args.pipeline_args is not None:
        CTX.log.info("child code page %s, UTF-8 mode %d", locale.getpreferredencoding(False), sys.flags.utf8_mode)
        pipelinekit.pipeline_main(json.loads(args.pipeline_args))
        return
    if args.child_preprocess is not None:
        child_preprocess(Path(args.child_preprocess))
        return
    folder = CTX.label_dir(EXPERIMENT)
    config_file = pipelinekit.write_config_copy(CTX, folder / "config", strategy={"trigger.replay.mode": MODE})
    results = {"preprocess": check_preprocess(folder, config_file), "pipeline": check_pipeline(folder)}
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root; children: --child-preprocess FOLDER (with --config) or --pipeline-args JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument("--child-preprocess")
    parser.add_argument("--config")
    parser.add_argument(harness.PIPELINE_ARGS_OPTION)
    return parser.parse_args()


# =============================================================================
# PREPROCESS CHILDREN
# =============================================================================


def check_preprocess(folder: Path, config_file: Path) -> dict:
    """The same synthetic series under the ANSI code page and under UTF-8."""
    found = {}
    for name, ansi in (("ansi", True), ("utf8", False)):
        target = folder / f"preprocess_{name}"
        CTX.guard(target)
        target.mkdir(parents=True, exist_ok=True)
        child = CTX.launch_child(f"preprocess{name}", [str(Path(__file__).resolve()), "--child-preprocess", str(target),
                                                        "--config", str(config_file), "--source-root", str(CTX.source_root)],
                                 timeout_s=harness.CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED, ansi=ansi, audit_child=True)
        found[name] = {"return_code": child.return_code, "encoding": child.text.splitlines()[0] if ansi else "UTF-8 mode",
                       "result": pipelinekit.last_prefixed(child.text, "PREPROCESS "),
                       "encode_error": ENCODE_ERROR in child.text and CHARACTER in child.text,
                       "error_line": [line for line in child.text.splitlines() if ENCODE_ERROR in line][-1:]}
        CTX.log.info("preprocess %s: %s", name, json.dumps(found[name]))
    utf8 = json.loads(found["utf8"]["result"] or "{}")
    CTX.verdict("UTF-8 child: the report holds the consecutive block", utf8.get("consecutive") is True and found["utf8"]["return_code"] == 0,
                "Consecutive Issues, return code 0", found["utf8"])
    CTX.verdict("ANSI child: the code page cannot encode the report's symbol (the case can bite)",
                found["ansi"]["encoding"].startswith(harness.CHILD_ENCODING_PREFIX) and not _encodable(found["ansi"]["encoding"]),
                "a code page without U+2265", found["ansi"]["encoding"])
    CTX.verdict("ANSI child: observed per the trace (UnicodeEncodeError on U+2265 in _write_report)",
                found["ansi"]["return_code"] not in (0, None) and found["ansi"]["encode_error"], "UnicodeEncodeError",
                found["ansi"]["error_line"])
    CTX.verdict("intent: identical report and frame in both environments", found["ansi"]["result"] == found["utf8"]["result"]
                and found["ansi"]["return_code"] == 0, found["utf8"]["result"], found["ansi"]["result"] or found["ansi"]["error_line"])
    return found


def _encodable(record: str) -> bool:
    """Whether the recorded code page can encode the report's symbol."""
    return bool(REPORT_SYMBOL.encode(record[len(harness.CHILD_ENCODING_PREFIX):], "ignore"))


def synthetic_frame() -> pd.DataFrame:
    """A daily frame with var5 = 0 on three calendar-consecutive weekdays."""
    frame = synthetic.daily_frame(100.0 + np.sin(np.arange(SYNTHETIC_SIZE, dtype=float)), SYNTHETIC_START)
    frame.loc[pd.to_datetime(list(BAD_DATES)), VAR5] = 0
    return frame


# =============================================================================
# PIPELINE CHILDREN
# =============================================================================


def check_pipeline(folder: Path) -> dict:
    """--task trigger on six series, series 0 with a three-day var5 = 0 block, per environment."""
    source = pipelinekit.raw_series(pipelinekit.load_copy(folder / "config" / pipelinekit.STRATEGY_FILE))
    planted = {sid: synthetic.SyntheticSeries(_with_block(sid, frame)) for sid, frame in sorted(source.items())}
    input_file = synthetic.write_input(CTX, folder / "input" / harness.DATA_OUTPUT_DIR / "0" / "raw" / "0.h5", planted)
    raw_data = "./" + input_file.resolve().relative_to(harness.REAL_WORKTREE).as_posix()
    found = {}
    for name, ansi in (("ansi", True), ("utf8", False)):
        run_folder = folder / f"pipeline_{name}"
        overrides = {"trigger.replay.mode": MODE, "read.raw_data": raw_data, **pipelinekit.GATE_OFF}
        config_file = pipelinekit.write_config_copy(CTX, run_folder, strategy=overrides)
        child, _, origin = CTX.run_pipeline("pipeline", strategy=pipelinekit.strategy_argument(config_file), tasks=["trigger"],
                                            quiet_luigi=True, hashseed=harness.DEFAULT_HASHSEED, allowed_root=run_folder,
                                            overrides=overrides, timeout_s=CHILD_TIMEOUT_S, ansi=ansi,
                                            child_argv=[str(Path(__file__).resolve()), "--source-root", str(CTX.source_root)])
        keys = pipelinekit.store_keys(pipelinekit.output_file(config_file, MODE, "trigger")) or []
        found[name] = {"return_code": child.return_code, "origin": origin["confirmed"], "preprocess_error": child.text.count(PREPROCESS_ERROR),
                       "series": sorted(key for key in keys if key.startswith(harness.SERIES_KEY_PREFIX)),
                       "code_page": pipelinekit.last_prefixed(child.text, "child code page ")}
        CTX.log.info("pipeline %s: %s", name, json.dumps(found[name]))
    CTX.verdict("pipeline: the input was read in both environments (UTF-8 run has six series)", len(found["utf8"]["series"]) == 6,
                6, found["utf8"]["series"])
    CTX.verdict("pipeline ANSI: observed per the trace (series 0 dropped, error logged, return code 0)",
                f"{harness.SERIES_KEY_PREFIX}{SERIES_ID}" not in found["ansi"]["series"] and found["ansi"]["preprocess_error"] > 0
                and found["ansi"]["return_code"] == 0, "series 0 missing, return code 0", found["ansi"])
    CTX.verdict("pipeline: intent six series in both trigger outputs", found["ansi"]["series"] == found["utf8"]["series"],
                found["utf8"]["series"], found["ansi"]["series"])
    return found


def _with_block(series_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Series 0 with var5 = 0 on the three bad dates; the others unchanged."""
    if series_id != SERIES_ID:
        return frame
    planted = frame.copy()
    planted.loc[pd.to_datetime(list(BAD_DATES)), VAR5] = 0
    return planted


# =============================================================================
# CHILD COMMANDS
# =============================================================================


def child_preprocess(folder: Path) -> None:
    """Child: Preprocess.process_series on the synthetic frame; a digest of the frame and report."""
    args = _parse_args()
    preprocessor = pipelinekit.preprocessor_of(pipelinekit.load_copy(Path(args.config)), folder / "0.h5")
    cleaned = preprocessor.process_series(SERIES_ID, synthetic_frame())
    report = preprocessor.report_path.read_text(encoding="utf-8")
    CTX.log.info("PREPROCESS %s", json.dumps({
        "rows": len(cleaned), "frame_md5": hashlib.md5(pd.util.hash_pandas_object(cleaned).values.tobytes()).hexdigest(),
        "consecutive": CONSECUTIVE in report, "report_md5": hashlib.md5(report.encode("utf-8")).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
