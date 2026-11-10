"""Bounded measurement launch of the pythia pipeline: one label, one input, one profile.

Usage (cwd = the worktree, PYTHONUTF8=1):
    python audit\\perflaunch.py <label> --input <absolute .h5> --profile {dev,prod} [--ini-date D] [--end-date D]
                                [--audit-config <folder>] [--task all]

It refuses a label that already has a config, output or record folder. It writes data\\cfg\\<label>\\extrema.yaml, a
copy of workflows\\extrema.yaml with read.raw_data, read.output_root (./data/out/<label>) and the given dates, and
data\\cfg\\<label>\\<profile>.yaml, a copy of workflows\\<profile>.yaml, each checked key by key; with --audit-config
the strategy copy must equal that folder's extrema.yaml at every leaf but read.raw_data and read.output_root, and a
dev profile copy that folder's dev.yaml at every leaf, or the launch is refused. Holding the harness lock, it starts
pipeline.py on the copy as the project's launchers start it (the base interpreter with __PYVENV_LAUNCHER__, UTF-8,
hash seed 0, bytecode allowed), stdout to stdout.log, stderr stamped on arrival to stderr.log, and samples the
child's memory and CPU and the system's CPU every 0.2 s to samples.tsv; at 15 min or 6 GiB of RSS it ends the
child's tree. The record data\\perf\\<label>\\summary.json holds the launch's identity, outcome, times, memory, CPU,
outputs, the log counts and the stderr tail. It runs outside the harness, so the measured
child carries no harness marker.

Structure:
1. main
2. Config copies
3. Launch: child, stderr reader, sampling loop, bounds
4. Record
"""
import sys

sys.dont_write_bytecode = True

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
import yaml

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE))

import harness
import pipelinekit
from src.utils.common.constants import PROFILES, REPLAY_IDEAL, REPLAY_REALISTIC

LOG = logging.getLogger("perflaunch")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
WORKFLOWS = WORKTREE / "workflows"
CFG_DIR = WORKTREE / "data" / "cfg"
OUT_DIR = WORKTREE / "data" / "out"
PERF_DIR = WORKTREE / "data" / "perf"
STRATEGY_FILE = "extrema.yaml"
TASK_CHOICES = ("trigger", "signal", "performance", "all")
SKIPPED_LEAVES = ("read.raw_data", "read.output_root")
MODES = (REPLAY_IDEAL, REPLAY_REALISTIC)
OUTSIDE = "-"
TIME_BOUND_S = 15 * 60.0
MEMORY_BOUND_B = 6 * 1024 ** 3
SAMPLE_PERIOD_S = 0.2
SURVIVOR_WAIT_S = 30.0
CONTENDED_CORES = 1.0
PERCENT = 100.0
HASHSEED = "0"
STDERR_TAIL_LINES = 40
FIELD_SEPARATOR = "\t"
MIB = 2 ** 20
JSON_INDENT = 1
UTF8_ON = "1"
STDOUT_NAME, STDERR_NAME, SAMPLES_NAME, SUMMARY_NAME = "stdout.log", "stderr.log", "samples.tsv", "summary.json"
TOOL_LOG_NAME = "perflaunch.log"
OUTCOME_COMPLETED, OUTCOME_STOPPED, OUTCOME_FAILED = "completed", "stopped", "failed"
STOP_TIME, STOP_MEMORY = "time", "memory"
MODE_LINE = re.compile(r"Running mode: (?P<mode>[a-z]+)")
TASK_START = re.compile(r"(?P<task>[A-Z][a-z]+) task start - ")
TASK_END = re.compile(r"(?P<task>[A-Z][a-z]+) task end - ")
DURATION = re.compile(r"\bDuration: (?P<seconds>[0-9.]+) seconds")
MODE_DURATION = re.compile(r"(?P<mode>[A-Z][a-z]+) mode duration: (?P<seconds>[0-9.]+) seconds")
TOTAL_DURATION = re.compile(r"Total pipeline duration: (?P<seconds>[0-9.]+) seconds")
# substring counts: pipeline.py colours the level name, so a word boundary before WARNING never matches
LOG_COUNTS = {"warning_lines": "WARNING ", "maxticks": "Locator.MAXTICKS", "mismatch_between": "mismatch between",
              "fell_back": "fell back to date-aligned", "error_lines": "ERROR "}
PNG_SUFFIX = ".png"
GRADIENT_PNG = "gradient.png"
SIGNAL_PLOT = re.compile(r"^group_.*_signal_.*\.png$")
SERIES_PLOT_FOLDERS = ("trigger", "signal")


def main() -> int:
    """Parse, refuse an existing label, write and check the copies, then launch under the lock and record."""
    args = _parse_args()
    if not sys.flags.utf8_mode:
        raise RuntimeError("perflaunch runs under PYTHONUTF8=1, as the copies and the record are UTF-8")
    folders = {"cfg": CFG_DIR / args.label, "out": OUT_DIR / args.label, "perf": PERF_DIR / args.label}
    existing = [str(folder) for folder in folders.values() if folder.exists()]
    if existing:
        raise RuntimeError(f"label {args.label} already has {existing}")
    folders["cfg"].mkdir(parents=True)
    folders["perf"].mkdir(parents=True)
    _attach_log(folders["perf"] / TOOL_LOG_NAME)
    copies = write_copies(args, folders["cfg"])
    pinned = harness._pinned_interpreter(LOG)
    lock = harness.acquire_lock(harness.lock_path(WORKTREE), LOG)
    try:
        identity = launch_identity(args, copies, pinned)
        run = run_child(args, pinned, folders["perf"], identity)
    finally:
        os.close(lock)
    record = summarize(folders, identity, run)
    tmp = folders["perf"] / f"{SUMMARY_NAME}.tmp"
    tmp.write_text(json.dumps(record, indent=JSON_INDENT, default=str), encoding="utf-8")
    os.replace(tmp, folders["perf"] / SUMMARY_NAME)
    LOG.info("%s: %s in %.1f s, largest sampled RSS %.0f MiB", args.label, record["outcome"], record["times"]["wall_s"],
             record["memory"]["largest_sampled_rss_b"] / MIB)
    return 0


def _parse_args() -> argparse.Namespace:
    """label, --input, --profile, the dates, --audit-config and --task."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--ini-date")
    parser.add_argument("--end-date")
    parser.add_argument("--audit-config", type=Path)
    parser.add_argument("--task", nargs="+", choices=TASK_CHOICES, default=["all"])
    args = parser.parse_args()
    if not (args.input.is_absolute() and args.input.is_file()):
        parser.error(f"--input must name an existing file by its absolute path: {args.input}")
    return args


def _attach_log(path: Path) -> None:
    """The command's stderr and the label's perflaunch.log."""
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)


# =============================================================================
# 2. CONFIG COPIES
# =============================================================================


def write_copies(args: argparse.Namespace, folder: Path) -> dict:
    """The strategy and profile copies, each equal to its workflows\\ source but for its overrides, and with
    --audit-config equal to that folder's files but for the skipped leaves."""
    strategy = {"read.raw_data": str(args.input), "read.output_root": f"./data/out/{args.label}"}
    strategy.update({key: value for key, value in (("read.ini_date", args.ini_date), ("read.end_date", args.end_date))
                     if value is not None})
    profile_file = f"{args.profile}.yaml"
    for name, overrides in ((STRATEGY_FILE, strategy), (profile_file, {})):
        with open(folder / name, "x", encoding="utf-8") as sink:
            yaml.safe_dump(pipelinekit._edited(WORKFLOWS / name, overrides, ()), sink, sort_keys=False)
        expected = {**harness.flatten_yaml(_parsed(WORKFLOWS / name)), **overrides}
        _refuse_differences(folder / name, harness.flatten_yaml(_parsed(folder / name)), expected, ())
    compared = {}
    if args.audit_config is not None:
        compared[STRATEGY_FILE] = list(SKIPPED_LEAVES)
        _refuse_differences(folder / STRATEGY_FILE, harness.flatten_yaml(_parsed(folder / STRATEGY_FILE)),
                            harness.flatten_yaml(_parsed(args.audit_config / STRATEGY_FILE)), SKIPPED_LEAVES)
        if args.profile == harness.DEV_PROFILE:
            compared[profile_file] = []
            _refuse_differences(folder / profile_file, harness.flatten_yaml(_parsed(folder / profile_file)),
                                harness.flatten_yaml(_parsed(args.audit_config / profile_file)), ())
    return {"strategy_overrides": strategy, "profile_file": profile_file,
            "audit_config": None if args.audit_config is None else str(args.audit_config),
            "audit_config_skipped_leaves": compared}


def _refuse_differences(copy: Path, actual: dict, expected: dict, skipped: tuple) -> None:
    """Refuse the launch when a leaf outside the skipped ones differs."""
    differing = [key for key in sorted(set(actual) | set(expected))
                 if key not in skipped and actual.get(key, "<absent>") != expected.get(key, "<absent>")]
    if differing:
        LOG.error("%s differs at %s", copy, differing)
        raise RuntimeError(f"{copy} differs at {differing}")


def _parsed(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# =============================================================================
# 3. LAUNCH
# =============================================================================


def launch_identity(args: argparse.Namespace, copies: dict, pinned: Path) -> dict:
    """The launch's identity before the child starts: label, worktree, commit, branch and status, host, interpreter."""
    memory = psutil.virtual_memory()
    return {"label": args.label, "worktree": str(WORKTREE), "commit": _git("rev-parse", "HEAD"),
            "branch": _git("symbolic-ref", "--short", "HEAD"), "status": _git("status", "--porcelain"),
            "input": str(args.input), "profile": args.profile, "tasks": args.task, **copies,
            "interpreter": sys._base_executable, "venv_interpreter": str(pinned), "cores": psutil.cpu_count(),
            "ram_total_b": memory.total, "ram_available_b": memory.available, "start": _now()}


def run_child(args: argparse.Namespace, pinned: Path, folder: Path, identity: dict) -> dict:
    """Start pipeline.py on the copy, read its stderr, sample until it ends or a bound stops it; end its tree
    however this ends."""
    strategy = Path(os.path.relpath(CFG_DIR / args.label / STRATEGY_FILE, WORKFLOWS)).with_suffix("").as_posix()
    argv = [sys._base_executable, "pipeline.py", "--task", *args.task, "--strategy", strategy,
            "--profile", args.profile]
    env = {name: value for name, value in os.environ.items() if name != harness.BYTECODE_ENV}
    env.update({harness.UTF8_ENV: UTF8_ON, harness.HASHSEED_ENV: HASHSEED, harness.VENV_LAUNCHER_ENV: str(pinned)})
    identity["argv"] = argv
    psutil.cpu_percent(None)
    started_wall, started = time.time(), time.monotonic()
    with open(folder / STDOUT_NAME, "xb") as sink:
        child = subprocess.Popen(argv, cwd=WORKTREE, env=env, stdin=subprocess.DEVNULL, stdout=sink,
                                 stderr=subprocess.PIPE)
    root = harness._root_identity(child.pid, started_wall)
    reader = StderrReader(child.stderr, folder / STDERR_NAME)
    reader.start()
    try:
        process = psutil.Process(child.pid)
        identity.update(child_cwd=process.cwd(), child_cmdline=process.cmdline())
        LOG.info("%s: child %s started in %s: %s", args.label, root, identity["child_cwd"], identity["child_cmdline"])
        sampled = sample_until_end(child, process, reader, folder / SAMPLES_NAME, started)
        if sampled["stopped"]:
            harness.end_tree(root, LOG, "bound")
    finally:
        sweep = harness.end_tree(root, LOG, "final")
    return_code = child.wait(timeout=SURVIVOR_WAIT_S)
    wall = time.monotonic() - started
    reader.join(timeout=SURVIVOR_WAIT_S)
    if sweep.survivors:
        raise RuntimeError(f"survivors after the kill: {sweep.survivors}")
    if reader.is_alive() or not reader.ended:
        raise RuntimeError("the stderr reader did not reach the end of the stream")
    identity["end"] = _now()
    return {"return_code": return_code, "wall_s": wall, "started_wall": started_wall, **sampled}


class StderrReader(threading.Thread):
    """Reads the child's stderr in lines split on newline, decodes each as UTF-8 with backslash escapes for invalid
    bytes, stamps it with its arrival time into stderr.log and marks the normal end of the stream."""

    def __init__(self, stream, path: Path) -> None:
        super().__init__(name="perflaunch-stderr", daemon=True)
        self.stream = stream
        self.path = path
        self.ended = False

    def run(self) -> None:
        with open(self.path, "x", encoding="utf-8", buffering=1) as sink:
            for raw in iter(self.stream.readline, b""):
                text = raw.decode("utf-8", errors="backslashreplace").rstrip("\r\n")
                sink.write(f"{time.time():.3f}{FIELD_SEPARATOR}{text}\n")
        self.ended = True


def sample_until_end(child: subprocess.Popen, process: psutil.Process, reader: StderrReader, path: Path,
                     started: float) -> dict:
    """RSS, peak working set and CPU times of the child and the system's CPU every 0.2 s until the child ends or
    reaches 15 min or 6 GiB of RSS; a stderr reader stopped without its end mark fails the launch."""
    largest_rss, peak_wset, cpu_times, stopped, elapsed = 0, None, None, "", 0.0
    system_cpu = []
    with open(path, "x", encoding="utf-8") as sink:
        while child.poll() is None:
            try:
                memory = process.memory_info()
                cpu_times = process.cpu_times()
            except psutil.NoSuchProcess:
                if child.poll() is None:
                    raise
                break
            system_cpu.append(psutil.cpu_percent(None))
            largest_rss, peak_wset = max(largest_rss, memory.rss), memory.peak_wset
            sink.write(FIELD_SEPARATOR.join(str(value) for value in (
                f"{time.time():.3f}", memory.rss, memory.peak_wset, cpu_times.user, cpu_times.system,
                system_cpu[-1])) + "\n")
            if not reader.is_alive() and not reader.ended:
                raise RuntimeError("the stderr reader stopped before the end of the stream")
            elapsed = time.monotonic() - started
            if elapsed >= TIME_BOUND_S or memory.rss >= MEMORY_BOUND_B:
                stopped = STOP_TIME if elapsed >= TIME_BOUND_S else STOP_MEMORY
                LOG.info("stopped at %.1f s, RSS %d B: %s", elapsed, memory.rss, stopped)
                break
            time.sleep(SAMPLE_PERIOD_S)
    return {"stopped": stopped, "stopped_at_s": elapsed if stopped else None, "largest_rss_b": largest_rss,
            "last_peak_wset_b": peak_wset, "cpu_user_s": None if cpu_times is None else cpu_times.user,
            "cpu_system_s": None if cpu_times is None else cpu_times.system,
            "system_cpu_mean_pct": statistics.fmean(system_cpu) if system_cpu else None, "samples": len(system_cpu)}


# =============================================================================
# 4. RECORD
# =============================================================================


def summarize(folders: dict, identity: dict, run: dict) -> dict:
    """Outcome, the stop, times, memory, CPU, outputs, log counts and the stderr tail."""
    lines = _stamped_lines(folders["perf"] / STDERR_NAME)
    parsed = parse_stderr(lines, run["started_wall"])
    outcome = (OUTCOME_STOPPED if run["stopped"] else OUTCOME_COMPLETED if run["return_code"] == 0
               else OUTCOME_FAILED)
    core_use = (None if run["cpu_user_s"] is None
                else (run["cpu_user_s"] + run["cpu_system_s"]) / run["wall_s"])
    other = (None if run["system_cpu_mean_pct"] is None or core_use is None
             else run["system_cpu_mean_pct"] / PERCENT * identity["cores"] - core_use)
    samples = _rss_samples(folders["perf"] / SAMPLES_NAME)
    in_progress = parsed["running"]
    return {
        "identity": identity, "outcome": outcome, "return_code": None if run["stopped"] else run["return_code"],
        "stop": None if not run["stopped"] else {
            "reason": run["stopped"], "elapsed_s": run["stopped_at_s"],
            "task_in_progress": None if in_progress is None else {"mode": in_progress["mode"],
                                                                  "task": in_progress["task"],
                                                                  "start_s": in_progress["start_s"]}},
        "times": {"wall_s": run["wall_s"], "total_logged_s": parsed["total_s"], "modes_logged_s": parsed["modes"],
                  "pairs": parsed["pairs"]},
        "memory": {"largest_sampled_rss_b": run["largest_rss_b"], "last_sampled_peak_wset_b": run["last_peak_wset_b"],
                   "pairs": [{"mode": pair["mode"], "task": pair["task"],
                              "largest_sampled_rss_b": _largest_within(samples, run["started_wall"], pair)}
                             for pair in parsed["pairs"] + ([in_progress] if in_progress else [])]},
        "cpu": {"user_s": run["cpu_user_s"], "system_s": run["cpu_system_s"], "core_use": core_use,
                "system_cpu_mean_pct": run["system_cpu_mean_pct"], "other_cores": other,
                "contended": other is not None and other > CONTENDED_CORES, "samples": run["samples"]},
        "outputs": _outputs(folders["out"]),
        "log_counts": parsed["counts"],
        "stderr_tail": [text for _, text in lines[-STDERR_TAIL_LINES:]],
    }


def _stamped_lines(path: Path) -> list[tuple[float, str]]:
    """(arrival time, text) of every stderr line."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stamp, _, text = line.partition(FIELD_SEPARATOR)
        lines.append((float(stamp), text))
    return lines


def parse_stderr(lines: list[tuple[float, str]], started_wall: float) -> dict:
    """Mode, task and duration lines, unanchored; each pair with its start and end offsets; the pair still running at
    the end of the stream; the log counts."""
    mode, current = OUTSIDE, None
    parsed = {"pairs": [], "modes": {}, "total_s": None, "running": None,
              "counts": {name: 0 for name in LOG_COUNTS} | {"lines": len(lines)}}
    for stamp, text in lines:
        for name, fragment in LOG_COUNTS.items():
            parsed["counts"][name] += fragment in text
        if match := MODE_LINE.search(text):
            mode = match["mode"]
        elif match := TASK_START.search(text):
            current = {"mode": mode, "task": match["task"].lower(), "start_s": stamp - started_wall}
        elif (match := TASK_END.search(text)) and current is not None:
            current["end_s"] = stamp - started_wall
        elif (match := DURATION.search(text)) and current is not None and "end_s" in current:
            current["logged_s"] = float(match["seconds"])
            parsed["pairs"].append(current)
            current = None
        elif match := MODE_DURATION.search(text):
            parsed["modes"][match["mode"].lower()] = float(match["seconds"])
        elif match := TOTAL_DURATION.search(text):
            parsed["total_s"] = float(match["seconds"])
    parsed["running"] = current
    return parsed


def _rss_samples(path: Path) -> list[tuple[float, int]]:
    """(time, RSS) of every sample."""
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(FIELD_SEPARATOR)
        samples.append((float(fields[0]), int(fields[1])))
    return samples


def _largest_within(samples: list[tuple[float, int]], started_wall: float, pair: dict) -> Optional[int]:
    """The largest RSS sample between a pair's start and end stamps (to the last sample for a pair still running)."""
    start = started_wall + pair["start_s"]
    end = started_wall + pair.get("end_s", float("inf"))
    return max((rss for stamp, rss in samples if start <= stamp <= end), default=None)


def _outputs(out: Path) -> dict:
    """Files and bytes of the launch's outputs and its PNGs by kind (per-series, per-signal, gradient, other)."""
    files = [path for path in sorted(out.rglob("*")) if path.is_file()] if out.is_dir() else []
    pngs = [path for path in files if path.suffix == PNG_SUFFIX]
    kinds = {"per_series": 0, "per_signal": 0, "gradient": 0, "other": 0}
    for path in pngs:
        if path.name == GRADIENT_PNG:
            kinds["gradient"] += 1
        elif SIGNAL_PLOT.match(path.name):
            kinds["per_signal"] += 1
        elif path.parent.name in SERIES_PLOT_FOLDERS and path.parent.parent.name == "plots":
            kinds["per_series"] += 1
        else:
            kinds["other"] += 1
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files), "png_files": len(pngs),
            "png_by_kind": kinds}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=WORKTREE, capture_output=True, text=True, encoding="utf-8",
                          check=True).stdout.strip()


def _now() -> str:
    """Local time, ISO 8601 with milliseconds."""
    return datetime.now().isoformat(timespec="milliseconds")


if __name__ == "__main__":
    raise SystemExit(main())
