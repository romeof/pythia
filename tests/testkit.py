"""Shared test logic of the regression scripts under tests/: plain scripts that return 1 on any FAIL.

It holds only what several regressions share: the preamble (logger, source root), PASS/FAIL
reporting with the return-1 summary, the scratch folder and its config copies (each without a gate, GATE_OFF,
unless a regression sets one; gate_item: one gate item override; validated_config: its typed
sections; strategy: the trigger strategy of a validated copy), the replay section
of a config found by content (replay_key) and the dotted path of one of its values in the root's
shipped YAML (replay_path), the dotted leaves of a YAML document (leaves, the family's one flattener), the
output of one mode of a copy (mode_output), a synthetic raw input
in the reader's layout (write_raw_input) and a synthetic sine series to write in it (sine_frame),
the keys and the frames of an output store, the child launcher with its timeout and process-tree
kill, the traceback blocks of a capture (traceback_blocks), the child side of the source-root rule
with its bootstraps (ANSI_CHILD_BOOTSTRAP for an ANSI child), the planted-failure check and its planted child
(planted_run), the in-process observation of an expected exception (raised) and of log records
(RecordCollector), and the oracle literals more than one regression uses. It imports only the
standard library, the pinned packages and the source root's src: the regression family keeps its own
launcher, preamble and kill logic, so a fix commit stays self-contained. Windows-only, the
pipeline's supported platform: on another platform it raises at import rather than ship an
unexecuted kill path.

Structure:
1. Oracle literals, bounds and names
2. Preamble and reporting
3. Scratch folder, config copies, the replay section and the YAML leaves
4. Child launch, timeout, the process-tree kill and the capture's traceback blocks
5. Planted-failure check
6. In-process observation
7. Child side
"""
import sys

sys.dont_write_bytecode = True

import dataclasses
import importlib
import locale
import logging
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, NoReturn, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import psutil
import yaml

if TYPE_CHECKING:
    from src.utils.common.schema import AppConfig

if sys.platform != "win32":
    raise RuntimeError("tests/testkit.py runs on Windows only, the pipeline's supported platform")

# Oracle literals: independent of production, so a base copy fails on behaviour, not on an ImportError
HANDLER_PREFIX = "Task failed"
ERROR_PREFIX = "ERROR "
PLANTED_ERROR = "planted failure of the regression"

PIPELINE_TIMEOUT_S = 900.0
CHILD_TIMEOUT_S = 300.0
HELPER_BOUND_S = 300.0
SURVIVOR_BOUND_S = 30.0
POLL_INTERVAL_S = 0.2
GONE_POLL_INTERVAL_S = 0.25
TIME_TOLERANCE_S = 1e-6
DEFAULT_HASHSEED = 0

UTF8_ENV = "PYTHONUTF8"
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"
HASHSEED_ENV = "PYTHONHASHSEED"
VENV_LAUNCHER_ENV = "__PYVENV_LAUNCHER__"
CHILD_ENCODING_PREFIX = "ENCODING="
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
TRACEBACK_LINE = "Traceback (most recent call last):"
CONTEXT_LINE = "During handling of the above exception, another exception occurred:"
STRATEGY_FILE, DEV_FILE = "extrema.yaml", "dev.yaml"
DEV_PROFILE, PROD_FILE = "dev", "prod.yaml"  # the profile every regression names, and the other profile's file
TASKS = ("trigger", "signal", "performance")
MODES = ("ideal", "realistic")
DUAL = "dual"
REPLAY_MODE_KEY = "mode"  # the one key that marks the replay section among the trigger subsections
ONE_SERIES = {"trigger.replay.mode": "realistic", "read.skip_series": ["1", "2", "3", "4", "5"]}
# The base overrides of every config copy, set before a regression's own, which win: no gate item and the pre-filter
# off, since the shipped items read var12 and var15, which neither the shipped input nor a var0-var5 frame holds
GATE_OFF = {"trigger.gate.items": [], "trigger.gate.prefilter": False}
# A queue consumer's failure record for series 0, the one series ONE_SERIES keeps
SERIES_RECORD = "Error processing series 0:"
# sine_frame: a sine of SINE_PERIOD_ROWS rows and SINE_AMPLITUDE around SINE_LEVEL, drifting by SINE_DRIFT per row
SINE_PERIOD_ROWS = 20
SINE_LEVEL = 100.0
SINE_AMPLITUDE = 10.0
SINE_DRIFT = 0.01
SINE_VAR5 = 1.0e6
CHILD_BOOTSTRAP = ("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, sys.argv[1]); import testkit; "
                   "testkit.child_preamble(sys.argv[2], sys.argv[3]); ")
# An ANSI child records its code page first, the record decode_capture requires, then bootstraps
ANSI_CHILD_BOOTSTRAP = ("import sys, locale; sys.stdout.write('" + CHILD_ENCODING_PREFIX
                        + "' + locale.getpreferredencoding(False) + '\\n'); sys.stdout.flush(); " + CHILD_BOOTSTRAP)


# =============================================================================
# PREAMBLE AND REPORTING
# =============================================================================


@dataclasses.dataclass
class Kit:
    """A regression's logger, source root and verdict counts."""

    log: logging.Logger
    root: Path
    name: str
    counts: dict = dataclasses.field(default_factory=lambda: {"PASS": 0, "FAIL": 0})

    def verdict(self, name: str, passed: bool, expected: object, actual: object) -> bool:
        """One PASS/FAIL line with expected and actual values."""
        outcome = "PASS" if passed else "FAIL"
        self.counts[outcome] += 1
        self.log.info("%s %s: expected %s; actual %s", outcome, name, expected, actual)
        return passed

    def finish(self) -> NoReturn:
        """The summary line; return code 1 on any FAIL."""
        self.log.info("SUMMARY %d PASS, %d FAIL", self.counts["PASS"], self.counts["FAIL"])
        sys.exit(1 if self.counts["FAIL"] else 0)


def preamble(script_file: str) -> Kit:
    """The regression's ASCII logger and its source root (the parent of its tests folder) first on sys.path,
    asserted; UTF-8 mode is required because the parent imports src (README:132)."""
    if sys.flags.utf8_mode != 1:
        raise RuntimeError(f"{Path(script_file).name} imports src and needs UTF-8 mode (-X utf8 or PYTHONUTF8=1)")
    root = Path(script_file).resolve().parents[1]
    insert_source_root(root)
    name = Path(script_file).stem
    return Kit(_stdout_logger(name), root, name)


def insert_source_root(root: Path) -> None:
    """Put the root first on sys.path, import src and assert that it comes from there."""
    sys.path.insert(0, str(root))
    import src

    origin = Path(src.__file__).resolve()
    if not origin.is_relative_to(Path(root).resolve()):
        raise ImportError(f"src imported from {origin}, not from {root}")


def _stdout_logger(name: str) -> logging.Logger:
    """A logger with one ASCII stdout handler."""
    log = logging.getLogger(f"tests.{name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_AsciiFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    return log


class _AsciiFormatter(logging.Formatter):
    """ASCII-only records, so a regression's output is readable under any console code page."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).encode("ascii", "backslashreplace").decode("ascii")


# =============================================================================
# SCRATCH FOLDER AND CONFIG COPIES
# =============================================================================


@contextmanager
def scratch(kit: Kit) -> Iterator[Path]:
    """A fresh folder under <cwd>/data (created when missing, as in a fresh checkout), removed afterwards."""
    data = Path.cwd() / "data"
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data, prefix=f"{kit.name}_") as folder:
        kit.log.info("scratch folder %s", folder)
        yield Path(folder)


def config_copy(kit: Kit, folder: Path, *, strategy: Optional[dict] = None, dev: Optional[dict] = None,
                drop_strategy: tuple = (), drop_dev: tuple = ()) -> Path:
    """Copies of the root's workflows/extrema.yaml and of both profile YAMLs in folder/config: the strategy and
    dev.yaml copies with dotted overrides and removals, prod.yaml as shipped (load_config reads the profile a run
    names beside the strategy YAML); read.output_root always lies inside the scratch folder, and the strategy copy
    takes GATE_OFF before the regression's own overrides. Returns the strategy copy's path."""
    target = folder / "config"
    target.mkdir()
    output_root = Path(os.path.relpath(folder / "out", Path.cwd())).as_posix()
    for name, overrides, removals in ((STRATEGY_FILE, {"read.output_root": f"./{output_root}", **GATE_OFF, **(strategy or {})},
                                       drop_strategy), (DEV_FILE, dev or {}, drop_dev), (PROD_FILE, {}, ())):
        document = _edited(kit.root / "workflows" / name, overrides, removals)
        with open(target / name, "x", encoding="utf-8") as sink:
            yaml.safe_dump(document, sink, sort_keys=False)
    kit.log.info("config copy %s: overrides %s, dropped %s; dev overrides %s, dropped %s",
                 target, strategy, list(drop_strategy), dev, list(drop_dev))
    assert_outputs_inside(kit, target / STRATEGY_FILE, folder)
    return target / STRATEGY_FILE


def _edited(source: Path, overrides: dict, removals: tuple) -> dict:
    """A YAML document with dotted-key overrides set and dotted keys removed."""
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
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


def gate_item(column: str, op: str, threshold: float, exempt_series: Sequence[str] = ()) -> dict:
    """One trigger.gate.items mapping as a config copy's overrides take it: the threshold a Python float and the
    exempt ids a list, the types yaml.safe_dump writes (it refuses a numpy scalar)."""
    return {"column": column, "op": op, "threshold": float(threshold), "exempt_series": list(exempt_series)}


def assert_outputs_inside(kit: Kit, config_file: Path, folder: Path) -> None:
    """Every task output of the copy, in each mode it runs, lies inside the scratch folder (get_output_file_name per
    task). The read, replay and strategy sections are built without validation (model_construct), so a copy a
    regression plants invalid is checked too and its ValidationError left to the regression."""
    from src.utils.common.outputs import get_output_file_name
    from src.utils.common.schema import ReadConfig, ReplayConfig, StrategyConfig, TriggerConfig

    config = loaded_config(config_file)
    section = replay_key(config)
    mode = config["trigger"][section][REPLAY_MODE_KEY]
    read = ReadConfig.model_construct(**config["read"])
    strategy = StrategyConfig.model_construct(**config["trigger"]["strategy"])
    outside = []
    for concrete in (MODES if mode == DUAL else (mode,)):
        replay = ReplayConfig.model_construct(**{**config["trigger"][section], REPLAY_MODE_KEY: concrete})
        trigger = TriggerConfig.model_construct(replay=replay, strategy=strategy)
        outside += [str(path) for task in TASKS
                    if not (path := Path(get_output_file_name(read, trigger, task)).resolve()).is_relative_to(folder.resolve())]
    if outside:
        kit.log.error("outputs outside the scratch folder %s: %s", folder, outside)
        raise RuntimeError(f"outputs outside the scratch folder {folder}: {outside}")


def loaded_config(config_file: Path) -> dict:
    """A copy's merged config (the dev profile)."""
    from src.utils.common.config import load_config

    return load_config(str(config_file), DEV_PROFILE)


def validated_config(config_file: Path) -> "AppConfig":
    """A copy's merged config validated (validate_config) into the typed sections the tasks, the strategy and the
    readers take."""
    from src.utils.common.config import validate_config

    return validate_config(loaded_config(config_file))


def strategy(config: "AppConfig", output_file: Path):
    """The trigger strategy of a validated copy, writing to output_file: create_strategy on its trigger section, which
    holds the recording mode trigger.trigger_value."""
    from src.utils.trigger.factory import create_strategy

    return create_strategy(config.trigger, output_file)


def replay_path(kit: Kit, leaf: str) -> str:
    """The dotted path of a value of the replay section, the section found by content in the root's shipped YAML
    (replay_key), so that a regression's override reaches the replay section of any tree."""
    shipped = yaml.safe_load((kit.root / "workflows" / STRATEGY_FILE).read_text(encoding="utf-8"))
    return f"trigger.{replay_key(shipped)}.{leaf}"


def replay_key(config: dict) -> str:
    """The key of the one trigger subsection that holds mode, found by content so that a copy of any tree's YAML is
    read alike; a ValueError names the holders when there is not exactly one."""
    holders = sorted(key for key, value in config["trigger"].items() if isinstance(value, dict) and REPLAY_MODE_KEY in value)
    if len(holders) != 1:
        raise ValueError(f"exactly one trigger subsection must hold {REPLAY_MODE_KEY!r}; holders: {holders}")
    return holders[0]


def leaves(document: dict, prefix: str = "") -> Iterator[tuple[str, object]]:
    """(dotted key, value) of every leaf of a YAML document, the regression family's one flattener: an empty
    mapping has no leaf and is dropped."""
    for key, value in document.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from leaves(value, f"{dotted}.")
            continue
        yield dotted, value


def output_file(config_file: Path, task: str) -> Path:
    """The task output a copy with a concrete mode resolves to."""
    from src.utils.common.outputs import get_output_file_name

    config = validated_config(config_file)
    return Path(get_output_file_name(config.read, config.trigger, task))


def mode_output(config_file: Path, mode: str, task: str) -> Path:
    """The task output of one concrete mode of a copy: its output path with the mode folder replaced by the mode."""
    path = output_file(config_file, task)
    return path.parents[2] / mode / task / path.name


def store_keys(path: Path) -> Optional[list[str]]:
    """Sorted keys of a store read in mode r, None when the file is absent."""
    if not path.is_file():
        return None
    with pd.HDFStore(str(path), mode="r") as store:
        return sorted(store.keys())


def store_frames(path: Path) -> dict[str, pd.DataFrame]:
    """Every frame of a store read in mode r, by key."""
    with pd.HDFStore(str(path), mode="r") as store:
        return {key: store.get(key) for key in sorted(store.keys())}


def write_raw_input(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    """A synthetic raw input in the reader's layout at a new path: one group per series with epoch-day indexes and
    the frame's columns as float32 datasets listed in its columns attribute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as sink:
        for series_id, frame in frames.items():
            group = sink.create_group(series_id)
            group.attrs["index_encoding"] = "epoch_days"
            group.attrs["columns"] = list(frame.columns)
            group["index_days"] = (frame.index - pd.Timestamp("1970-01-01")).days.to_numpy(dtype="int32")
            for column in frame.columns:
                group[column] = frame[column].to_numpy(dtype="float32")


def sine_frame(rows: int, first_day: str) -> pd.DataFrame:
    """A synthetic series on rows daily dates from first_day: var0, var3 and var4 the drifting sine of the SINE_*
    names, var1 and var2 at 0.99 and 1.01 of it, var5 SINE_VAR5."""
    steps = np.arange(rows)
    levels = SINE_LEVEL + SINE_AMPLITUDE * np.sin(2.0 * np.pi * steps / SINE_PERIOD_ROWS) + SINE_DRIFT * steps
    return pd.DataFrame({"var0": levels, "var1": levels * 0.99, "var2": levels * 1.01, "var3": levels, "var4": levels,
                         "var5": np.full(rows, SINE_VAR5)}, index=pd.date_range(first_day, periods=rows, freq="D"))


def strategy_argument(config_file: Path) -> str:
    """The --strategy value pipeline.py turns into workflows/<value>.yaml for this copy (cwd-relative)."""
    return Path(os.path.relpath(config_file.with_suffix(""), Path.cwd() / "workflows")).as_posix()


# =============================================================================
# CHILD LAUNCH, TIMEOUT AND THE PROCESS-TREE KILL
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Identity:
    """A process identity: pid plus creation time, so a reused pid never matches."""

    pid: int
    create_time: float


@dataclasses.dataclass(frozen=True)
class Child:
    """One bounded child: return code (None after a timeout), its decoded capture and duration."""

    return_code: Optional[int]
    timed_out: bool
    duration_s: float
    text: str
    log_path: Path


def run_pipeline(kit: Kit, folder: Path, role: str, config_file: Path, tasks: list[str], *,
                 timeout_s: float = PIPELINE_TIMEOUT_S, hashseed: int = DEFAULT_HASHSEED, ansi: bool = False,
                 profile: str = DEV_PROFILE) -> Child:
    """<root>/pipeline.py on the copy under the profile, from the inherited cwd (the worktree), under the hash seed."""
    return run_child(kit, folder, role, [str(kit.root / "pipeline.py"), "--task", *tasks,
                                         "--strategy", strategy_argument(config_file), "--profile", profile],
                     timeout_s=timeout_s, hashseed=hashseed, ansi=ansi)


def run_child(kit: Kit, folder: Path, role: str, args: list[str], *, timeout_s: float = CHILD_TIMEOUT_S,
              hashseed: int = DEFAULT_HASHSEED, ansi: bool = False) -> Child:
    """A Python child of the venv's base interpreter (UTF-8 unless ANSI, no bytecode, the seed), its output in a
    scratch file decoded once it has ended."""
    out_path = folder / f"{role}.log"
    argv = [sys._base_executable, "-B"] + (["-X", "utf8=0"] if ansi else []) + args
    started = time.monotonic()
    return_code, timed_out = launch(argv, env=child_environment(hashseed, ansi), cwd=Path.cwd(), out_path=out_path,
                                  timeout_s=timeout_s, log=kit.log)
    child = Child(return_code, timed_out, time.monotonic() - started, decode_capture(out_path, ansi, kit.log), out_path)
    kit.log.info("child %s: return code %s, timed out %s, %.1f s", role, return_code, timed_out, child.duration_s)
    return child


def child_environment(hashseed: int, ansi: bool) -> dict:
    """The inherited environment with the venv launch, no bytecode, the seed and UTF-8 (removed for ANSI)."""
    env = {name: value for name, value in os.environ.items() if not (ansi and name == UTF8_ENV)}
    env.update({BYTECODE_ENV: "1", HASHSEED_ENV: str(hashseed), VENV_LAUNCHER_ENV: sys.executable})
    if not ansi:
        env[UTF8_ENV] = "1"
    return env


def launch(argv: list[str], *, env: dict, cwd: Path, out_path: Path, timeout_s: float,
           log: logging.Logger) -> tuple[Optional[int], bool]:
    """Run a child with its output in a file; at the deadline and however the wait ends, end its whole tree.

    A timeout is a timeout, never a return code: the first value is then None. The root keeps its launch time as
    creation time unless the snapshot shows its pid under this process, so a pid reused at once is never walked.
    """
    started_wall, started = time.time(), time.monotonic()
    with open(out_path, "xb") as sink:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT)
    found = snapshot().get(proc.pid)
    root = Identity(proc.pid, found[1] if found is not None and found[0] == os.getpid() else started_wall)
    log.info("child root %s: %s", root, argv)
    timed_out, deadline_survivors = False, []
    try:
        while proc.poll() is None:
            if time.monotonic() - started >= timeout_s:
                timed_out = True
                deadline_survivors = end_tree(root, log, "deadline")[1]
                break
            time.sleep(POLL_INTERVAL_S)
    finally:
        members, final_survivors = end_tree(root, log, "final")
    return_code = proc.wait(timeout=SURVIVOR_BOUND_S)
    if deadline_survivors or final_survivors:
        log.error("processes alive after the kill: %s", deadline_survivors + final_survivors)
        raise RuntimeError(f"processes alive after the kill: {deadline_survivors + final_survivors}")
    if members:
        log.error("processes outlived the child %s: %s", root, members)
        raise RuntimeError(f"processes outlived the child {root}: {members}")
    return (None if timed_out else return_code), timed_out


def end_tree(root: Identity, log: logging.Logger, phase: str) -> tuple[list[Identity], list[Identity]]:
    """Kill every live process of the root's tree, then its late children; (members killed, survivors)."""
    members = walk(snapshot(), [root])
    if not members:
        return [], []
    log.info("%s kill: %s", phase, members)
    taskkill(members, log)
    late = [member for member in walk(snapshot(), members) if member not in members]
    if late:
        log.info("%s kill of late children: %s", phase, late)
        taskkill(late, log)
    return members + late, survivors_of(members + late, SURVIVOR_BOUND_S)


def snapshot() -> dict[int, tuple[int, float]]:
    """pid -> (ppid, create_time) of every process, read fresh (stale cached objects cleared)."""
    parents = psutil._ppid_map()
    psutil.process_iter.cache_clear()
    found = {}
    for proc in psutil.process_iter(attrs=["pid", "create_time"], ad_value=None):
        pid, created = proc.info["pid"], proc.info["create_time"]
        if created is not None and pid in parents:
            found[pid] = (parents[pid], created)
    return found


def walk(snap: dict[int, tuple[int, float]], starts: list[Identity]) -> list[Identity]:
    """Live processes reachable from the starts by parent links (a child is never older than its parent)."""
    children: dict[int, list[int]] = {}
    for pid, (ppid, _) in snap.items():
        children.setdefault(ppid, []).append(pid)
    found, seen, stack = [], set(), list(starts)
    while stack:
        member = stack.pop()
        if member in seen:
            continue
        seen.add(member)
        holder = snap.get(member.pid)
        alive = holder is not None and abs(holder[1] - member.create_time) <= TIME_TOLERANCE_S
        if alive:
            found.append(member)
        reuse_time = holder[1] if holder is not None and not alive else None
        for pid in sorted(children.get(member.pid, [])):
            created = snap[pid][1]
            if pid == member.pid or created < member.create_time or (reuse_time is not None and created >= reuse_time):
                continue
            stack.append(Identity(pid, created))
    return sorted(found, key=dataclasses.astuple)


def survivors_of(members: list[Identity], bound_s: float) -> list[Identity]:
    """Members still alive (same pid and creation time) once the bound has elapsed."""
    deadline = time.monotonic() + bound_s
    while True:
        snap = snapshot()
        alive = [member for member in members
                 if member.pid in snap and abs(snap[member.pid][1] - member.create_time) <= TIME_TOLERANCE_S]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(GONE_POLL_INTERVAL_S)


def taskkill(members: list[Identity], log: logging.Logger) -> int:
    """Kill exactly the given pids with one taskkill /F, bounded, its output in a file; non-zero when a member
    ended meanwhile."""
    argv = ["taskkill", "/F"] + [part for member in members for part in ("/PID", str(member.pid))]
    with tempfile.TemporaryFile() as out:
        try:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=HELPER_BOUND_S)
        except subprocess.TimeoutExpired:
            log.error("taskkill exceeded its %.0f s bound: %s", HELPER_BOUND_S, argv)
            raise
        out.seek(0)
        log.info("taskkill return code %d: %s", done.returncode, out.read().decode("ascii", "backslashreplace").strip()[-400:])
    return done.returncode


def decode_capture(path: Path, ansi: bool, log: logging.Logger) -> str:
    """A capture decoded strictly: UTF-8, or the parent's ANSI code page after the child's own record of it."""
    raw = path.read_bytes()
    encoding = "utf-8"
    if ansi:
        encoding = locale.getencoding()
        first = raw.split(b"\n", 1)[0].decode("ascii", "backslashreplace").strip()
        if first != f"{CHILD_ENCODING_PREFIX}{encoding}":
            log.error("capture %s recorded %r, parent code page is %s", path.name, first, encoding)
            raise RuntimeError(f"{path.name} recorded {first!r}, parent code page is {encoding}")
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        log.error("capture %s is not valid %s: %s", path.name, encoding, exc)
        raise


def traceback_blocks(text: str) -> list[tuple[str, bool]]:
    """One item per traceback line of a capture, in sequence: the block's exception line (the first non-blank
    unindented line after it) and whether the interpreter's context line is the last non-blank line before it."""
    lines = plain_lines(text)
    blocks = []
    for row, line in enumerate(lines):
        if line.rstrip() != TRACEBACK_LINE:
            continue
        exception = next((later.rstrip() for later in lines[row + 1:] if later.strip() and not later[0].isspace()), "")
        before = next((earlier.rstrip() for earlier in reversed(lines[:row]) if earlier.strip()), "")
        blocks.append((exception, before == CONTEXT_LINE))
    return blocks


def plain_lines(text: str) -> list[str]:
    """A capture's lines with ANSI escape sequences removed (pipeline.py colours WARNING and ERROR records)."""
    return ANSI_ESCAPE.sub("", text).splitlines()


def child_args(kit: Kit) -> list[str]:
    """The argv a CHILD_BOOTSTRAP program reads first: this tests folder, the source root, the venv interpreter."""
    return [str(Path(__file__).resolve().parent), str(kit.root), sys.executable]


# =============================================================================
# PLANTED-FAILURE CHECK
# =============================================================================


def planted_failure(kit: Kit, *, target: str, task: str, key_prefix: str = "", store_folder: str = "",
                    profile: str = DEV_PROFILE) -> Child:
    """A planted_run child on a one-series realistic copy with the real input, under the profile. Exactly one record
    with the handler prefix must name the planted error, then return != 0. Returns the child, its capture decoded."""
    with scratch(kit) as folder:
        config_file = config_copy(kit, folder, strategy=dict(ONE_SERIES))
        child = planted_run(kit, folder, "planted", config_file, target=target, task=task, key_prefix=key_prefix,
                            store_folder=store_folder, profile=profile)
        records = [line for line in plain_lines(child.text) if HANDLER_PREFIX in line and PLANTED_ERROR in line]
        kit.verdict(f"{target} planted under --task {task}: one '{HANDLER_PREFIX}' record names the planted error",
                    len(records) == 1, 1, records)
        kit.verdict(f"{target} planted under --task {task}: return != 0", child.return_code not in (0, None),
                    "return != 0", {"return_code": child.return_code, "timed_out": child.timed_out})
    return child


def planted_run(kit: Kit, folder: Path, role: str, config_file: Path, *, target: str, task: str, key_prefix: str = "",
                store_folder: str = "", timeout_s: float = PIPELINE_TIMEOUT_S, profile: str = DEV_PROFILE) -> Child:
    """A child in the folder patches target (module:Class.method) to raise a planted OSError, then runs
    pipeline.main() with the task on the copy under the profile; with a key prefix only calls carrying a key with
    that prefix raise, with a store folder only calls on a store whose file lies in a folder of that name (trigger,
    signal, performance), the others delegate. One role per child: each capture is created exclusively."""
    code = CHILD_BOOTSTRAP + "testkit.plant_and_run(sys.argv[4:])"
    return run_child(kit, folder, role, ["-c", code, *child_args(kit), target, key_prefix, store_folder, task,
                                         strategy_argument(config_file), profile], timeout_s=timeout_s)


# =============================================================================
# IN-PROCESS OBSERVATION
# =============================================================================


def raised(call: Callable[[], object]) -> Optional[BaseException]:
    """The exception an in-process call raises, None when it returns: the call runs in a one-worker thread pool
    whose future holds the exception, so a regression observes it without an except clause."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(call).exception()


class RecordCollector(logging.Handler):
    """Keeps the messages of the records it receives at or above its level."""

    def __init__(self, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self.messages: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


# =============================================================================
# CHILD SIDE
# =============================================================================


def child_preamble(root: str, interpreter: str) -> None:
    """A child's checks: the venv is active and src comes from the given source root."""
    if not (os.path.samefile(sys.executable, interpreter) and sys.prefix != sys.base_prefix):
        raise RuntimeError(f"{sys.executable} (prefix {sys.prefix}) is not the active venv {interpreter}")
    insert_source_root(Path(root))


def plant_and_run(argv: list[str]) -> None:
    """Child of the planted-failure check: patch the target, then pipeline.main() with the task, strategy and
    profile."""
    target, key_prefix, store_folder, task, strategy, profile = argv
    module_name, _, qualified = target.partition(":")
    owner_name, _, attribute = qualified.rpartition(".")
    owner = importlib.import_module(module_name)
    for part in owner_name.split(".") if owner_name else ():
        owner = getattr(owner, part)
    original = getattr(owner, attribute)

    def planted(*args: object, **kwargs: object) -> object:
        values = (*args, *kwargs.values())
        keyed = any(isinstance(value, str) and value.lstrip("/").startswith(key_prefix) for value in values)
        stored = any(Path(str(getattr(value, "filename", "") or "")).parent.name == store_folder for value in values)
        if (not key_prefix or keyed) and (not store_folder or stored):
            raise OSError(PLANTED_ERROR)
        return original(*args, **kwargs)

    setattr(owner, attribute, planted)
    import pipeline

    sys.argv = ["pipeline.py", "--task", task, "--strategy", strategy, "--profile", profile]
    pipeline.main()
