"""Regression for a run's outputs across consoles (D-9 with hashseeds.py; F-04).

Two dual --task all runs, each on its own fresh copy of the shipped configuration over the same
synthetic input (testkit.write_raw_input: one 160-row series whose var5 is zero on three adjacent
rows, a bad block the preprocess report names under its "Consecutive Issues (>= 3 rows)" heading
with the sign U+2265): one under UTF-8 (testkit.run_pipeline), one an ANSI child (-X utf8=0,
PYTHONUTF8 removed) whose code begins with testkit.ANSI_CHILD_BOOTSTRAP, which records its code page
before pipeline.main(). The case first requires that the ANSI code page cannot encode the sign (else
it cannot bite and fails as "could not run"); then both runs return 0, both reports name the block,
each mode's reports are byte-equal, and every task file holds the same keys and frames, value by
value. A report written in the code page fails the ANSI run at its first block line.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
ROWS = 160
BAD_ROWS = [70, 71, 72]
SIGN = chr(0x2265)
BLOCK_HEADING = "Consecutive Issues"
REPORT_FILE = "0_preprocess_report.txt"
ANSI_CODE = testkit.ANSI_CHILD_BOOTSTRAP + (
    "\nimport pipeline\n"
    "sys.argv = ['pipeline.py', '--task', 'all', '--strategy', sys.argv[4], '--profile', testkit.DEV_PROFILE]\n"
    "pipeline.main()\n")


def main() -> None:
    """The two runs, the bite precondition and the comparisons, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as utf8_folder, testkit.scratch(KIT) as ansi_folder:
        utf8_config, ansi_config = (fresh_copy(folder) for folder in (utf8_folder, ansi_folder))
        utf8 = testkit.run_pipeline(KIT, utf8_folder, "utf8", utf8_config, ["all"])
        ansi = testkit.run_child(KIT, ansi_folder, "ansi", ["-c", ANSI_CODE, *testkit.child_args(KIT),
                                                            testkit.strategy_argument(ansi_config)],
                                 timeout_s=testkit.PIPELINE_TIMEOUT_S, ansi=True)
        code_page = testkit.plain_lines(ansi.text)[0][len(testkit.CHILD_ENCODING_PREFIX):] if ansi.text else ""
        bites = bool(code_page) and testkit.raised(lambda: SIGN.encode(code_page)) is not None
        KIT.verdict("the ANSI code page cannot encode the report's sign (else the case could not run)", bites, True,
                    code_page or "could not run")
        KIT.verdict("both runs return 0", utf8.return_code == 0 and ansi.return_code == 0, {"utf8": 0, "ansi": 0},
                    {"utf8": utf8.return_code, "ansi": ansi.return_code, "timed_out": [utf8.timed_out, ansi.timed_out]})
        for mode in testkit.MODES:
            reports = {name: report_bytes(config, mode) for name, config in (("utf8", utf8_config), ("ansi", ansi_config))}
            named = {name: f"{BLOCK_HEADING} ({SIGN}".encode("utf-8") in data for name, data in reports.items()}
            KIT.verdict(f"{mode}: both reports name the bad block, byte-equal",
                        all(named.values()) and reports["utf8"] == reports["ansi"],
                        {"named": {"utf8": True, "ansi": True}, "equal": True},
                        {"named": named, "equal": reports["utf8"] == reports["ansi"]})
            for task in testkit.TASKS:
                differing = store_differences(testkit.mode_output(utf8_config, mode, task),
                                              testkit.mode_output(ansi_config, mode, task))
                KIT.verdict(f"{mode} {task}: the ANSI run's task file equals the UTF-8 run's value by value", not differing,
                            "no difference", differing[:5])
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def fresh_copy(folder: Path) -> Path:
    """The synthetic input inside the folder and a shipped-configuration copy that reads it."""
    raw = folder / SYNTHETIC_INPUT
    levels = 100.0 + 10.0 * np.sin(np.linspace(0.0, 8.0 * np.pi, ROWS))
    frame = pd.DataFrame({"var0": levels + 0.5, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels, "var4": levels,
                          "var5": np.full(ROWS, 1.0e6)}, index=pd.bdate_range("2020-01-02", periods=ROWS))
    frame.iloc[BAD_ROWS, frame.columns.get_loc("var5")] = 0.0
    testkit.write_raw_input(raw, {"0": frame})
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    return testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{relative}"})


def report_bytes(config_file: Path, mode: str) -> bytes:
    """The mode's preprocess report beside its trigger output, empty when absent."""
    report = testkit.mode_output(config_file, mode, "trigger").parent / REPORT_FILE
    return report.read_bytes() if report.is_file() else b""


def store_differences(left: Path, right: Path) -> list[str]:
    """Missing files, differing key sets and keys whose frames differ, value by value (NaN equal to NaN)."""
    if not (left.is_file() and right.is_file()):
        return [f"missing: {[str(path) for path in (left, right) if not path.is_file()]}"]
    with pd.HDFStore(str(left), mode="r") as first, pd.HDFStore(str(right), mode="r") as second:
        keys = sorted(first.keys())
        if keys != sorted(second.keys()):
            return [f"keys {keys} vs {sorted(second.keys())}"]
        return [key for key in keys if not first.get(key).equals(second.get(key))]


if __name__ == "__main__":
    main()
