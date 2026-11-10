"""EXP-19: the replay validation suite run unmodified, with one planted defect per test (HYP-13, D-1).

The suite <source root>/tests/replayvalidation.py runs from a launcher written under
data/exp/19 (MD5 logged) that puts <source root>/tests and <source root> first on sys.path,
prints the imported src origin and calls runpy.run_path on the suite with run_name "__main__",
so the suite's testkit preamble roots at the source root. Ten runs through the child launcher,
each with its own recorded seed, must give identical outcomes (return code, the suite's PASS/FAIL
verdict per test, its SUMMARY line, traceback lines). Then one planted non-raising value defect
per test, applied by its own launcher before the suite starts (a raising one crashes the suite
and proves nothing), must turn that test's verdict to FAIL.

Structure:
1. main
2. Launchers
3. Runs and outcomes
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

EXPERIMENT = "19"
TEST_FILE = Path("tests") / "replayvalidation.py"
ORIGIN_PREFIX = "SRC_ORIGIN "
LAUNCHER = """import runpy
import sys

sys.dont_write_bytecode = True
sys.path[:0] = [{tests!r}, {root!r}]
import src

print("{prefix}" + src.__file__)
{planted}
runpy.run_path({suite!r}, run_name="__main__")
"""
SEEDS = tuple(range(1, 11))
TESTS = ("(1) parity", "(2) no future rows", "(3) rolling vs expanding", "(4) boundary")
VERDICT_LINE = re.compile(r" (?P<status>PASS|FAIL) (?P<test>\(\d\) [a-z -]+): ")
SUMMARY_LINE = re.compile(r"SUMMARY \d+ PASS, \d+ FAIL")
TRACEBACK_TEXT = "Traceback (most recent call last)"
PASS_STATUS, FAIL_STATUS = "PASS", "FAIL"
# test name: (what the planted defect does, code the launcher runs before the suite)
PLANTED = {
    "(1) parity": ("ideal mode zeroes every minus begin after the analysis", """
from src.utils.trigger.ideal import IdealModeMixin
_planted_original = IdealModeMixin._analyze_series_ideal
def _planted(self, series_id, df):
    df = _planted_original(self, series_id, df)
    df["trigger_minus_begin"] = 0
    return df
IdealModeMixin._analyze_series_ideal = _planted
"""),
    "(2) no future rows": ("the realistic window includes row i+1 (RealisticModeMixin._window_bounds)", """
from src.utils.trigger.realistic import RealisticModeMixin
_planted_original = RealisticModeMixin._window_bounds
def _planted(self, i, config):
    return _planted_original(self, i, config)[0], i + 2
RealisticModeMixin._window_bounds = _planted
"""),
    "(3) rolling vs expanding": ("every window starts at row 0, so rolling equals expanding", """
from src.utils.trigger.realistic import RealisticModeMixin
RealisticModeMixin._window_bounds = lambda self, i, config: (0, i + 1)
"""),
    "(4) boundary": ("realistic mode zeroes every begin after the analysis", """
from src.utils.trigger.realistic import RealisticModeMixin
_planted_original = RealisticModeMixin._analyze_series_realistic
def _planted(self, series_id, df):
    df = _planted_original(self, series_id, df)
    df["trigger_plus_begin"] = 0
    df["trigger_minus_begin"] = 0
    return df
RealisticModeMixin._analyze_series_realistic = _planted
"""),
}


def main() -> None:
    """Ten seeded runs of the suite, then one run per planted defect."""
    _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    root = CTX.source_root.resolve()
    CTX.log.info("suite %s: MD5 %s", root / TEST_FILE, harness.file_md5(root / TEST_FILE))
    path = write_script(folder / "launcher.py", launcher(root, ""))
    outcomes = {seed: outcome(CTX.launch_child(f"suite{seed}_", [str(path)], timeout_s=harness.CHILD_TIMEOUT_S,
                                               hashseed=seed), root) for seed in SEEDS}
    first = outcomes[SEEDS[0]]
    CTX.verdict("ten seeded runs: identical outcomes", all(item == first for item in outcomes.values()), first,
                {seed: item for seed, item in outcomes.items() if item != first})
    CTX.verdict("every test reaches its verdict (a SUMMARY line, no traceback)",
                bool(first["summary"]) and not first["tracebacks"], "a SUMMARY line, no traceback",
                {"summary": first["summary"], "tracebacks": first["tracebacks"]})
    CTX.verdict("the suite imports src from the source root", first["origin_ok"], str(root), first["origin"])
    for test in TESTS:
        CTX.verdict(f"{test} passes on the audited code", first["tests"].get(test) == PASS_STATUS, PASS_STATUS,
                    first["tests"].get(test))
    planted = {test: planted_run(test, folder, root) for test in TESTS}
    for test, item in planted.items():
        caught = item["tests"].get(test) == FAIL_STATUS and first["tests"].get(test) == PASS_STATUS
        CTX.verdict(f"{test} fails on its planted non-raising defect ({PLANTED[test][0]}) and passes without it", caught,
                    "FAIL with the defect, PASS without", {"with": item["tests"].get(test), "without": first["tests"].get(test),
                                                          "tracebacks": item["tracebacks"]})
    CTX.log.info("RESULTS %s", json.dumps({"outcome": first, "planted": planted}, sort_keys=True))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# LAUNCHERS
# =============================================================================


def launcher(root: Path, planted: str) -> str:
    """A launcher: the tests folder and the source root first on sys.path, the src origin printed, the planted code,
    then the unmodified suite as __main__."""
    return LAUNCHER.format(tests=str(root / TEST_FILE.parent), root=str(root), prefix=ORIGIN_PREFIX, planted=planted,
                           suite=str(root / TEST_FILE))


def write_script(path: Path, text: str) -> Path:
    """A derived script written exclusively under the experiment folder, its MD5 logged."""
    CTX.guard(path)
    with open(path, "x", encoding="utf-8", newline="\n") as sink:
        sink.write(text)
    CTX.log.info("derived script %s: MD5 %s", path, harness.file_md5(path))
    return path


def planted_run(test: str, folder: Path, root: Path) -> dict:
    """The suite run once from a launcher that applies one planted defect first."""
    path = write_script(folder / f"planted{TESTS.index(test) + 1}.py", launcher(root, PLANTED[test][1]))
    found = outcome(CTX.launch_child(f"planted{TESTS.index(test) + 1}_", [str(path)], timeout_s=harness.CHILD_TIMEOUT_S,
                                     hashseed=harness.DEFAULT_HASHSEED), root)
    CTX.log.info("PLANTED %s (%s): %s", test, PLANTED[test][0], json.dumps(found, sort_keys=True))
    return found


# =============================================================================
# RUNS AND OUTCOMES
# =============================================================================


def outcome(child: harness.ChildResult, root: Path) -> dict:
    """Return code, the suite's verdict per test, its SUMMARY line, traceback lines and the src origin of one run."""
    lines = child.text.splitlines()
    tests = {match["test"]: match["status"] for match in map(VERDICT_LINE.search, lines) if match}
    origin = pipelinekit.last_prefixed(child.text, ORIGIN_PREFIX)
    return {"return_code": child.return_code, "timed_out": child.timed_out, "tests": tests,
            "summary": [match.group(0) for match in map(SUMMARY_LINE.search, lines) if match],
            "tracebacks": [line.strip() for line in lines if TRACEBACK_TEXT in line],
            "origin": origin, "origin_ok": bool(origin) and Path(origin).is_relative_to(root)}


if __name__ == "__main__":
    main()
