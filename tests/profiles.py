"""Regression for the run profiles and the gradient figure's plot_debug gate (C-2, R2).

One synthetic one-series input of 200 daily rows (testkit.sine_frame through testkit.write_raw_input), read whole by
dual config copies that hold prod.yaml beside dev.yaml.
(a) testkit.run_pipeline(..., ["all"], profile="prod"): zero .png under the scratch output; the strategy folder holds
extrema.yaml, prod.yaml and git_info.txt and no dev.yaml; src/varsreadout.txt keeps its size and modification time
(or its absence); the first log record names profile prod.
(b) profile "dev" with performance.gradient.plot_debug false: in each mode no gradient.png and no group_*_signal_*.png,
while plots/trigger/*.png and plots/signal/*.png exist (non-vacuity: the dev profile's other plot switches are on);
src/varsreadout.txt keeps its size and modification time (or its absence), as in (a): no run writes or removes it,
so a stale copy in the tree the suite runs from survives.
(c) pipeline.py --task trigger on the copy of (a) without --profile: the first log record names profile prod and the
run copies prod.yaml (the default profile; the run's start removed the copy of (a)).
(d) The shipped workflows/dev.yaml and workflows/prod.yaml hold the same keys and the same value at every leaf whose
key is not a switch (printout_io, printout_debug, full_output, plot_debug), so the profiles
cannot part a stored value; every switch of prod.yaml is false.
(e) testkit.planted_failure with src.tasks.performance:shared_annualization planted, --task all under the prod
profile: one "Task failed" record names the planted error and the return code is non-zero, so a production run, which
draws no figure, still runs the annualization check (without the call in _run_gradient the child returns 0).
Before C-2 no profile exists: every run writes the PNGs, the reads log and dev.yaml, and pipeline.py rejects --profile.
Until the reads log's retirement the dev run rewrites the file, so (b)'s reads-log verdict fails.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
import re
from pathlib import Path
from typing import Optional

import yaml

SYNTHETIC_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
FIRST_DAY = "2020-01-01"
ROWS = 200
PROD_PROFILE = "prod"
GIT_FILE = "git_info.txt"
READS_LOG = Path("src") / "varsreadout.txt"
SWITCHES = ("printout_io", "printout_debug", "full_output", "plot_debug")
GRADIENT_FIGURE = "gradient.png"
SIGNAL_PLOT = "group_*_signal_*.png"
FIRST_RECORD = re.compile(r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL) ")
ANNUALIZATION_TARGET = "src.tasks.performance:shared_annualization"


def main() -> None:
    """The five cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = synthetic_copy(folder)
        production_run(folder, config_file)
        default_profile(folder, config_file)
    with testkit.scratch(KIT) as folder:
        figure_gate(folder, synthetic_copy(folder, {"performance.gradient.plot_debug": False}))
    shipped_profiles()
    testkit.planted_failure(KIT, target=ANNUALIZATION_TARGET, task="all", profile=PROD_PROFILE)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def production_run(folder: Path, config_file: Path) -> None:
    """(a) The production profile writes no PNG, copies prod.yaml only and leaves the reads log untouched."""
    before = reads_log_stamp()
    child = testkit.run_pipeline(KIT, folder, "prod", config_file, ["all"], profile=PROD_PROFILE)
    KIT.verdict("(a) prod --task all: return code 0", child.return_code == 0, 0,
                {"return_code": child.return_code, "timed_out": child.timed_out})
    pngs = sorted(str(path.relative_to(folder)) for path in (folder / "out").rglob("*.png"))
    KIT.verdict("(a) prod: zero .png under the scratch output", not pngs, [], pngs[:5])
    present = provenance(config_file)
    expected = sorted((testkit.STRATEGY_FILE, testkit.PROD_FILE, GIT_FILE))
    KIT.verdict(f"(a) prod: the strategy folder holds {expected} and no {testkit.DEV_FILE}", present == expected,
                expected, present)
    after = reads_log_stamp()
    KIT.verdict(f"(a) prod: {READS_LOG.as_posix()} keeps its size and modification time", after == before,
                before, after)
    first = first_record(child.text)
    KIT.verdict("(a) prod: the first log record names profile prod", f"profile {PROD_PROFILE}," in first,
                f"profile {PROD_PROFILE}", first)


def default_profile(folder: Path, config_file: Path) -> None:
    """(c) A run that names no profile runs the production profile and copies prod.yaml."""
    child = testkit.run_child(KIT, folder, "default", [str(KIT.root / "pipeline.py"), "--task", "trigger", "--strategy",
                                                       testkit.strategy_argument(config_file)],
                              timeout_s=testkit.PIPELINE_TIMEOUT_S)
    first = first_record(child.text)
    KIT.verdict("(c) no --profile: return code 0, the first log record names profile prod",
                child.return_code == 0 and f"profile {PROD_PROFILE}," in first,
                {"return_code": 0, "first record": f"profile {PROD_PROFILE}"},
                {"return_code": child.return_code, "first record": first})
    present = provenance(config_file)
    KIT.verdict(f"(c) no --profile: the run copies {testkit.PROD_FILE}, not {testkit.DEV_FILE}",
                testkit.PROD_FILE in present and testkit.DEV_FILE not in present,
                {testkit.PROD_FILE: True, testkit.DEV_FILE: False},
                {testkit.PROD_FILE: testkit.PROD_FILE in present, testkit.DEV_FILE: testkit.DEV_FILE in present})


def figure_gate(folder: Path, config_file: Path) -> None:
    """(b) performance.gradient.plot_debug false: no gradient figure and no per-signal plot, while the per-series
    plots of the dev profile exist; the dev run, as the prod run, leaves the reads log untouched."""
    before = reads_log_stamp()
    child = testkit.run_pipeline(KIT, folder, "devgate", config_file, ["all"])
    KIT.verdict("(b) dev, gradient plot_debug false, --task all: return code 0", child.return_code == 0, 0,
                {"return_code": child.return_code, "timed_out": child.timed_out})
    after = reads_log_stamp()
    KIT.verdict(f"(b) dev: {READS_LOG.as_posix()} keeps its size and modification time", after == before,
                before, after)
    for mode in testkit.MODES:
        plots = testkit.mode_output(config_file, mode, "performance").parent / "plots"
        gated = sorted(path.name for pattern in (GRADIENT_FIGURE, SIGNAL_PLOT) for path in plots.rglob(pattern))
        KIT.verdict(f"(b) {mode}: no {GRADIENT_FIGURE} and no {SIGNAL_PLOT}", not gated, [], gated[:5])
        drawn = {kind: len(list((plots / kind).glob("*.png"))) for kind in ("trigger", "signal")}
        KIT.verdict(f"(b) {mode}: non-vacuity, plots/trigger/*.png and plots/signal/*.png exist",
                    all(drawn.values()), "both > 0", drawn)


def shipped_profiles() -> None:
    """(d) The shipped profiles part only on their switches, and every switch of prod.yaml is off."""
    dev, prod = (dict(testkit.leaves(yaml.safe_load((KIT.root / "workflows" / name).read_text(encoding="utf-8"))))
                 for name in (testkit.DEV_FILE, testkit.PROD_FILE))
    KIT.verdict(f"(d) {testkit.DEV_FILE} and {testkit.PROD_FILE} hold the same keys", dev.keys() == prod.keys(),
                "equal key sets", {"only in dev": sorted(dev.keys() - prod.keys()),
                                   "only in prod": sorted(prod.keys() - dev.keys())})
    differing = sorted(key for key in dev.keys() & prod.keys() if key.rsplit(".", 1)[-1] not in SWITCHES
                       and dev[key] != prod[key])
    KIT.verdict(f"(d) every leaf but the switches {list(SWITCHES)} holds the same value in both profiles",
                not differing, [], differing)
    switched_on = sorted(key for key, value in prod.items() if key.rsplit(".", 1)[-1] in SWITCHES and value is not False)
    KIT.verdict(f"(d) every switch of {testkit.PROD_FILE} is false", not switched_on, [], switched_on)


# =============================================================================
# HELPERS
# =============================================================================


def synthetic_copy(folder: Path, dev: Optional[dict] = None) -> Path:
    """The one-series sine input inside the folder and a config copy that reads all of its dates."""
    raw = folder / SYNTHETIC_INPUT
    frame = testkit.sine_frame(ROWS, FIRST_DAY)
    testkit.write_raw_input(raw, {"0": frame})
    relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
    return testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{relative}",
                                                      "read.ini_date": frame.index[0].strftime("%Y-%m-%d"),
                                                      "read.end_date": frame.index[-1].strftime("%Y-%m-%d")}, dev=dev)


def provenance(config_file: Path) -> list[str]:
    """The sorted names of the YAML and git files in the copy's strategy folder."""
    strategy_dir = testkit.output_file(config_file, "trigger").parents[2]
    return sorted(path.name for path in strategy_dir.iterdir() if path.is_file())


def reads_log_stamp() -> Optional[tuple[int, int]]:
    """The reads log's (size, modification time in ns), None when absent."""
    path = KIT.root / READS_LOG
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def first_record(text: str) -> str:
    """The capture's first log record (a line that begins with a level name), empty when none."""
    return next((line for line in testkit.plain_lines(text) if FIRST_RECORD.match(line)), "")


if __name__ == "__main__":
    main()
