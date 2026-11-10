"""Audit harness: the single starting point of every evidence launch of the pythia audit.

Every Python process whose output is evidence starts here, so each launch has a fixed
environment, a never-reused log, an evidence record (commit, branch, tree status, launch,
environment, file identities, source root) and a child launch that never outlives it.

Commands (one per invocation; --run FILE names the run's identity file, required by every command but
--self-test, which plants its own; --worktree re-roots any command onto a self-test fixture):
    --label L --timeout S [--hashseed N] [--mode M] SCRIPT [ARGS]   launch an evidence script
    --guard PATH [PATH ...]                                       reparse-point guard
    --self-test                                                   the planted cases below
    --select-seeds                                                choose the two run hash seeds
    --manifest LABEL ROOT                                         manifest of a copy or export
    --resume                                                      the resume rule (refused in the master tree)
    --mutate F-NN COMMIT SPEC                                     a planted-mutation export

Audit scripts import this module for their preamble, their child launcher and the
output-root probe that precedes every pipeline child.

Structure:
1. Command line, run file and dispatch
2. Script launch
3. Non-script commands (guard, manifest, seed selection, resume)
4. Evidence gate: record, modes, identities, source roots, manifests
5. Reparse-point guard and lock
6. Child launch, timeouts and the process-tree kill
7. Audit-script side: preamble, context (rule_data and rule_text read the rules folder), child launcher, output-root
   probe, replay section (replay_key), name map (rename_map, renamed_keys, kept_frame, renamed_frame,
   renamed_document), YAML flattener (flatten_yaml)
8. Self-test
9. Resume helpers
10. Base helpers
"""
import sys

sys.dont_write_bytecode = True

import argparse
import dataclasses
import hashlib
import importlib.metadata
import io
import itertools
import json
import locale
import logging
import msvcrt
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, NoReturn, Optional, TypeVar

HARNESS_DIR = Path(__file__).resolve().parent
HARNESS_FILE = HARNESS_DIR / "harness.py"
REAL_WORKTREE = HARNESS_DIR.parent
RUN_FILE_KEYS = ("branches", "verification_file", "report_path")  # the --run file: a JSON object of exactly these

AUDIT_DIR = "audit"
TESTS_DIR = "tests"
TESTKIT_FILE = "testkit.py"
EXPORT_PATHS = ("pipeline.py", "src", "tests", "workflows")
PIPELINE_FILE = "pipeline.py"
DEV_PROFILE = "dev"  # the profile an audit launch names (load_root_config)
TESTKIT_FILE = "testkit.py"
FIX_PREFIXES = ("src/", "tests/")
AUDIT_REFERENCE = re.compile(r"\baudit\b")

MODE_EVIDENCE = "evidence"
MODE_PRE_FIX = "pre-fix"
MODE_PRE_COMMIT = "pre-commit"
MODES = (MODE_EVIDENCE, MODE_PRE_FIX, MODE_PRE_COMMIT)
PRE_COMMIT_SCRIPT_STEM = "exp20sweeps"
COMMIT_KIND_AUDIT = "audit"
COMMIT_KIND_FIX = "fix"

COMMAND_SCRIPT = "script"
COMMAND_GUARD = "guard"
COMMAND_SELF_TEST = "selftest"
COMMAND_SELECT_SEEDS = "selectseeds"
COMMAND_MANIFEST = "manifest"
COMMAND_RESUME = "resume"
COMMAND_MUTATE = "mutate"
UNRECORDED_COMMANDS = (COMMAND_RESUME, COMMAND_GUARD)
HARNESS_STEM = "harness"

MARKER_ENV = "PYTHIA_AUDIT_HARNESS"
LABEL_ENV = "PYTHIA_AUDIT_LABEL"
SOURCE_ROOT_ENV = "PYTHIA_AUDIT_SOURCE_ROOT"
WORKTREE_ENV = "PYTHIA_AUDIT_WORKTREE"
RUN_FILE_ENV = "PYTHIA_AUDIT_RUN_FILE"
INTERPRETER_ENV = "PYTHIA_AUDIT_INTERPRETER"
ANSI_CHILD_ENV = "PYTHIA_AUDIT_ANSI_CHILD"
PIPELINE_ARGS_OPTION = "--pipeline-args"
VENV_LAUNCHER_ENV = "__PYVENV_LAUNCHER__"
UTF8_ENV = "PYTHONUTF8"
BYTECODE_ENV = "PYTHONDONTWRITEBYTECODE"
HASHSEED_ENV = "PYTHONHASHSEED"
DEFAULT_HASHSEED = 0
RECORDED_EXACT = frozenset({"TZ", "LANG", "LANGUAGE"})
RECORDED_PREFIXES = ("PYTHON", "NUMBA_", "LUIGI_", "OMP_", "MKL_", "OPENBLAS_", "MPL", "LC_")
SECRET_FRAGMENTS = ("KEY", "TOKEN", "SECRET", "PASS", "AUTH", "CRED")
RECORDED_PACKAGES = (
    "pandas", "numpy", "h5py", "tables", "luigi", "scipy", "numba",
    "psutil", "pydantic", "hdf5plugin", "matplotlib", "PyYAML",
)

HELPER_BOUND_S = 300.0
SURVIVOR_BOUND_S = 30.0
PIPELINE_TIMEOUT_S = 900.0
CHILD_TIMEOUT_S = 300.0
POLL_INTERVAL_S = 0.2
GONE_POLL_INTERVAL_S = 0.25
TIME_TOLERANCE_S = 1e-6
LOCK_BYTES = 1
CREATE_SUSPENDED = 0x00000004  # Win32 creation flag, not exported by subprocess: the child runs only once resumed

LABEL_PATTERN = re.compile(r"^(?P<stem>[a-z0-9_]+)-(?P<point>[A-Za-z0-9_]+)-(?P<seq>[0-9]+)$")
LAUNCH_LOG_PATTERN = re.compile(r"^[^.]+\.log$")
RECORD_PREFIX = "RECORD "
LOCK_REFUSAL = "REFUSED[lock]"
MUT_FINDING = re.compile(r"^F-[0-9]{2}$")
MANIFEST_BASE = "base"
MANIFEST_REF = "ref"
RUN_COPY_PREFIX = "run"
FORBIDDEN_OUTPUT_PARTS = ("ref", "base", "mut")
DATA_OUTPUT_DIR = "2026_09_25_0"
DRAFTS_DIR = "drafts"
PRE_RESUME_PREFIX = "pre_"
RESUME_PREFIX = "resume_"
CACHE_DIR = "__pycache__"
NUMBA_CACHE_SUFFIXES = (".nbi", ".nbc")  # numba's cache=True index and data files beside the compiled module
LINE_ENDING_SETTINGS = ("core.autocrlf", "core.eol")

REFERENCE_PARAMS = Path(DATA_OUTPUT_DIR) / "0" / "extrema" / "var12ge0_83x0-var15ge0_73__w7md5p0_05"
SEED_FILES = tuple(f"{mode}/{task}/0.h5" for mode in ("ideal", "realistic") for task in ("trigger", "signal"))
SERIES_KEY_PREFIX = "/series_"
SEED_CANDIDATES = tuple(range(1, 33))
SEED_SEQUENCES_PREFIX = "SEQUENCES "
PROBE_PREFIX = "PROBE "
ORIGIN_PREFIX = "ORIGIN "  # the line a pipeline child reports src's origin on (ORIGIN_CHILD, preamble)
ORIGIN_NOT_IMPORTED = "not-imported"  # the origin of a child that ended before importing src
# The pipeline child of Context.run_pipeline (python -c ORIGIN_CHILD <source root> <pipeline args JSON>): sys.path[0],
# the empty cwd path -c puts there, becomes the source root, so the import path equals a python <root>\pipeline.py
# launch; the source root's pipeline.py runs as __main__, and the finally reports src's origin, also after a failure.
ORIGIN_CHILD = (
    "import json, runpy, sys\n"
    "if sys.path[0] != '':\n"
    "    raise RuntimeError(f'ORIGIN_CHILD expects the empty cwd path first on sys.path, not {sys.path[0]!r}')\n"
    "sys.path[0] = sys.argv[1]\n"
    "sys.argv = [sys.argv[1] + '/pipeline.py', *json.loads(sys.argv[2])]\n"
    "try:\n"
    "    runpy.run_path(sys.argv[0], run_name='__main__')\n"
    "finally:\n"
    "    src = sys.modules.get('src')\n"
    f"    sys.stdout.write({ORIGIN_PREFIX!r} + (src.__file__ if src is not None else {ORIGIN_NOT_IMPORTED!r}) + '\\n')\n"
    "    sys.stdout.flush()\n"
)
CHILD_ENCODING_PREFIX = "ENCODING="
TESTKIT_RESULT_PREFIX = "TESTKIT_RESULT "
NOTICE_PREFIX = "NOTICE "
RULE_DATA_FILE = "terminology_data.json"  # in a --rules folder: the exclusions, controls and report_title of the rule
RULE_DATA_RESULT_PREFIX = "RULE_DATA_RESULT "
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
PROCESSED_LINE = re.compile(r"Processed (\d+) items, successful: (\d+)")
ERROR_MARK = "ERROR"
REPLAY_MODE_KEY = "mode"  # the one key that marks the replay section among the trigger subsections
GATE_SECTION_KEY = "gate"  # the trigger section's gate (trigger.gate), a field of TriggerConfig on a root that holds it
SECTION = TypeVar("SECTION")  # a configuration section model the probe builds (_probe_section)
NAME_MAP_PARTS = ("keys", "columns", "values", "yaml_keys")
NAME_MAP_DROP = "drop_columns"  # the name map's list of the column labels its frames lose before the rename

FIXTURE_TIMEOUT_S = 180.0
FIXTURE_HANG_TIMEOUT_S = 8.0
FIXTURE_RETURN_CODE = 3
PLANTED_SURVIVOR_BOUND_S = 1.0
RESTART_STAMP_S = 1.0  # resolution of the resume stamp, so two restarts of one case get distinct folders
LOCK_WAIT_MIN_S = 8.0
LOCK_WAIT_MAX_S = 40.0
HELPER_TEST_BOUND_S = 3.0
HELPER_TEST_MARGIN_S = 15.0
FIXTURE_IDENTITY = ("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid")
FIXTURE_BRANCHES = ("planted-first", "planted-second", "planted-third")  # the planted run's; a fixture starts on the first
FIXTURE_VERIFICATION_FILE = "verification_planted.txt"  # the planted run's record
FIXTURE_REPORT_NAME = "report.md"  # the planted run's report, in the fixture repository
FIXTURE_RUN_NAME = "run.json"  # the planted run file, beside the fixture repository
# probed copies: a prominence_threshold spelt as an int and as a float (validated, both 1.0), a key validation refuses
PROBE_COPIES = {"int": {"trigger.strategy.prominence_threshold": 1},
                "float": {"trigger.strategy.prominence_threshold": 1.0}, "invalid": {"read.planted_unknown_key": 1}}


# =============================================================================
# 1. COMMAND LINE, RUN FILE AND DISPATCH
# =============================================================================


def main() -> int:
    """Read the run file, check the fixture override and the interpreter, then run exactly one command."""
    args = _parse_cli(sys.argv[1:])
    console = _console_logger()
    run = None if args.self_test and args.run is None else read_run_file(args.run, console)
    roots = _resolve_roots(args.worktree, run, console)
    pinned = _pinned_interpreter(console)
    if args.guard:
        return command_guard(roots, pinned, args.guard, console, args.mode)
    if args.self_test:
        return command_self_test(roots, pinned, console, args.mode)
    if args.select_seeds:
        return command_select_seeds(roots, pinned, console)
    if args.manifest:
        return command_manifest(roots, pinned, args.manifest[0], Path(args.manifest[1]), console)
    if args.resume:
        return command_resume(roots, pinned, console)
    if args.mutate:
        return command_mutate(roots, pinned, *args.mutate, console)
    return command_script(roots, pinned, args, console)


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """Parse the harness options; everything after the script path belongs to the script."""
    parser = argparse.ArgumentParser(prog="harness.py", allow_abbrev=False)
    parser.add_argument("--run", type=Path, help="the run's identity file (every command but --self-test)")
    parser.add_argument("--worktree", type=Path, help="self-test fixture root (fixture cases only)")
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--guard", nargs="+", type=Path, metavar="PATH")
    command.add_argument("--self-test", action="store_true")
    command.add_argument("--select-seeds", action="store_true")
    command.add_argument("--manifest", nargs=2, metavar=("LABEL", "ROOT"))
    command.add_argument("--resume", action="store_true")
    command.add_argument("--mutate", nargs=3, metavar=("FINDING", "COMMIT", "SPEC"))
    parser.add_argument("--label")
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--hashseed", type=int, default=DEFAULT_HASHSEED)
    parser.add_argument("--mode", choices=MODES, default=MODE_EVIDENCE)
    parser.add_argument("script", nargs="?", type=Path)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    non_script = args.guard or args.self_test or args.select_seeds or args.manifest or args.resume or args.mutate
    if non_script and args.script is not None:
        parser.error("a non-script command takes no script")
    if not non_script and (args.script is None or args.label is None or args.timeout is None):
        parser.error("a script launch needs --label, --timeout and a script")
    return args


def read_run_file(path: Optional[Path], log: logging.Logger) -> "RunIdentity":
    """The run's identity from its --run file, a JSON object of exactly RUN_FILE_KEYS: branches (the HEAD branches
    its evidence may come from, a non-empty list of names), verification_file (the verifier's record, a
    worktree-relative path with forward slashes, as git names it) and report_path (absolute). Absent or malformed:
    REFUSED[run-file]."""
    if path is None or not path.is_file():
        refuse(log, "run-file", f"no run file {path}: every command but --self-test names one with --run FILE")
    try:
        document = json.loads(path.read_bytes())
    except ValueError as exc:
        log.error("REFUSED[run-file]: %s is no JSON document: %s", path, exc)
        raise Refusal("run-file", f"{path} is no JSON document: {exc}") from exc
    if not _run_document_valid(document):
        refuse(log, "run-file", f"{path} is not an object of {', '.join(RUN_FILE_KEYS)} in their shapes: {document!r}")
    return RunIdentity(path.resolve(), tuple(document["branches"]), document["verification_file"],
                       Path(document["report_path"]))


def _run_document_valid(document: object) -> bool:
    """Exactly RUN_FILE_KEYS; branches a non-empty list of non-empty strings; verification_file a non-empty relative
    path without a backslash; report_path an absolute path."""
    if not (isinstance(document, dict) and sorted(document) == sorted(RUN_FILE_KEYS)):
        return False
    branches, record, report = (document[key] for key in RUN_FILE_KEYS)
    return (isinstance(branches, list) and bool(branches) and all(isinstance(name, str) and name for name in branches)
            and isinstance(record, str) and bool(record) and "\\" not in record and not Path(record).anchor
            and isinstance(report, str) and Path(report).is_absolute())


def _resolve_roots(override: Optional[Path], run: Optional["RunIdentity"], console: logging.Logger) -> "Roots":
    """Return the real roots, or a fixture's after the override check (console-only refusal); both carry the run."""
    real = Roots(REAL_WORKTREE, run)
    if override is None:
        return real
    if not _contained(real.exp00, override):
        refuse(console, "worktree", f"{override} does not lie under {real.exp00}")
    code, out = run_helper(["git", "-C", str(override), "rev-parse", "--show-toplevel"],
                           cwd=REAL_WORKTREE, log=console, check=False)
    if code != 0 or not same_path(Path(out.strip()), override):
        refuse(console, "worktree", f"{override} is not its own repository's top level ({out.strip()})")
    return Roots(override.resolve(), run)


def _pinned_interpreter(log: logging.Logger) -> Path:
    """Return the master tree's .venv interpreter and assert that this process runs it."""
    _, out = run_helper(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                        cwd=REAL_WORKTREE, log=log)
    pinned = Path(out.strip()).parent / ".venv" / "Scripts" / "python.exe"
    if not os.path.samefile(sys.executable, pinned):
        refuse(log, "interpreter", f"{sys.executable} is not the pinned {pinned}")
    return pinned


# =============================================================================
# 2. SCRIPT LAUNCH
# =============================================================================


def command_script(roots: "Roots", pinned: Path, args: argparse.Namespace, console: logging.Logger) -> int:
    """Launch one evidence script: guard, launch log, lock, gate, child, outcome."""
    script = args.script.resolve()
    label = args.label
    _check_label(label, script, console)
    launch_log = roots.logs / f"{label}.log"
    script_log = roots.logs / f"{label}.script.log"
    guard_or_refuse(roots, [launch_log, script_log, roots.lock], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    _refuse_reused_label(roots, label, console)
    log = _attach_launch_log(console, launch_log)
    lock_fd = acquire_lock(roots.lock, log)
    try:
        gate = evidence_gate(roots, pinned, log, command=COMMAND_SCRIPT, mode=args.mode, label=label,
                             script=script, script_argv=args.script_args,
                             timeout_s=args.timeout, hashseed=args.hashseed)
        result = launch([sys._base_executable, "-B", str(script), *args.script_args],
                        env=gate.child_env, cwd=REAL_WORKTREE, out_path=script_log,
                        timeout_s=args.timeout, log=log)
    finally:
        os.close(lock_fd)
    return _finish_launch(log, label, result)


def _check_label(label: str, script: Path, console: logging.Logger) -> None:
    """A label is <script stem>-<run point>-<sequence number>; the run point has no hyphen."""
    match = LABEL_PATTERN.match(label)
    if match is None or match["stem"] != script.stem:
        refuse(console, "label", f"{label!r} is not {script.stem}-<run point>-<n>")


def _refuse_reused_label(roots: "Roots", label: str, console: logging.Logger) -> None:
    """Refuse a label whose launch log exists, also among the logs a resume moved away."""
    names = {f"{label}.log", f"{label}.script.log"}
    if any(path.name in names for path in _all_log_files(roots)):
        refuse(console, "label-exists", f"a log of {label} already exists")


def _all_log_files(roots: "Roots") -> list[Path]:
    """Every file directly under the logs folder and at any depth under its pre-resume folders (a later restart moves
    the earlier pre-resume folders into its own)."""
    if not roots.logs.is_dir():
        return []
    files = [path for path in roots.logs.iterdir() if path.is_file()]
    for folder in sorted(roots.logs.glob(f"{PRE_RESUME_PREFIX}*")):
        files.extend(path for path in folder.rglob("*") if path.is_file())
    return sorted(files)


def _finish_launch(log: logging.Logger, label: str, result: "LaunchResult") -> int:
    """Record the outcome; a timeout is recorded as a timeout and never as a return code."""
    _log_event(log, "end", outcome="timeout" if result.timed_out else "returned",
               return_code=result.return_code, duration_s=round(result.duration_s, 3))
    if result.timed_out:
        refuse(log, "timeout", f"{label} exceeded its timeout after {result.duration_s:.1f} s")
    log.info("%s finished with return code %d after %.1f s", label, result.return_code, result.duration_s)
    return result.return_code


# =============================================================================
# 3. NON-SCRIPT COMMANDS
# =============================================================================


def command_guard(roots: "Roots", pinned: Path, targets: list[Path], console: logging.Logger, mode: str) -> int:
    """Guard the targets; only after the check passes create data\\logs and write the log. Mode pre-fix admits
    tracked changes confined to tests\\ (copying a regression whose fix extends tests/testkit.py into data\\base)."""
    if mode == MODE_PRE_COMMIT:
        refuse(console, "mode", "the guard runs in evidence or pre-fix mode")
    guard_or_refuse(roots, targets, console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_GUARD)
    evidence_gate(roots, pinned, log, command=COMMAND_GUARD, mode=mode, label=label)
    log.info("guard passed for %s", [str(target) for target in targets])
    return 0


def command_manifest(roots: "Roots", pinned: Path, manifest_label: str, root: Path, console: logging.Logger) -> int:
    """Write the manifest of a copy or an export right after it was made."""
    guard_or_refuse(roots, [roots.manifests / f"{manifest_label}.json"], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_MANIFEST + _point_suffix(manifest_label))
    evidence_gate(roots, pinned, log, command=COMMAND_MANIFEST, mode=MODE_EVIDENCE, label=label)
    write_manifest(roots, log, manifest_label, root.resolve())
    return 0


def command_select_seeds(roots: "Roots", pinned: Path, console: logging.Logger) -> int:
    """Choose s1 and s2 whose /series_* key sequences differ in each of the four reference files."""
    guard_or_refuse(roots, [roots.logs], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_SELECT_SEEDS)
    evidence_gate(roots, pinned, log, command=COMMAND_SELECT_SEEDS, mode=MODE_EVIDENCE, label=label)
    files = [roots.ref / REFERENCE_PARAMS / name for name in SEED_FILES]
    s1, s2 = select_seed_pair(Context(log, roots, label, REAL_WORKTREE, pinned), files, SERIES_KEY_PREFIX)
    log.info("SEEDS s1=%d s2=%d", s1, s2)
    return 0


def select_seed_pair(context: "Context", files: list[Path], prefix: str, first: Optional[int] = None) -> tuple[int, int]:
    """Two candidate seeds whose sequences of the keys with the prefix differ in every file, after the
    planted controls; with first, the pair starts with that seed (EXP-22's second run after its first)."""
    sequences = {seed: _seed_sequences(context, seed, files) for seed in SEED_CANDIDATES}
    _check_seed_controls(context.log, sequences, prefix)
    candidates = (itertools.combinations(SEED_CANDIDATES, 2) if first is None
                  else ((first, seed) for seed in SEED_CANDIDATES if seed != first))
    pair = next(((a, b) for a, b in candidates if seeds_differ(sequences[a], sequences[b], prefix)), None)
    if pair is None:
        refuse(context.log, "seeds", f"no candidate pair among {SEED_CANDIDATES} differs in every file (first {first})")
    for seed in pair:
        context.log.info("seed %d %s* sequences: %s", seed, prefix, json.dumps(_prefixed_sequences(sequences[seed], prefix)))
    return pair


def _seed_sequences(context: "Context", seed: int, files: list[Path]) -> dict[str, list[str]]:
    """Set iteration order of each file's stored key list in a child running under the seed."""
    code = ("import sys; sys.dont_write_bytecode = True; "
            f"sys.path.insert(0, {str(HARNESS_DIR)!r}); import harness; harness.seed_sequence_main(sys.argv[1:])")
    child = context.launch_child(f"seed{seed}_", ["-c", code, *[str(path) for path in files]],
                                 timeout_s=CHILD_TIMEOUT_S, hashseed=seed)
    if child.return_code != 0:
        refuse(context.log, "seeds", f"seed child {seed} returned {child.return_code}: {child.log_path}")
    return json.loads(_prefixed_line(child.text, SEED_SEQUENCES_PREFIX))


def seed_sequence_main(argv: list[str]) -> None:
    """Child side of the seed selection: log list(set(store.keys())) of each file (mode r)."""
    import pandas as pd

    sequences = {}
    for name in argv:
        with pd.HDFStore(name, mode="r") as store:
            sequences[name] = list(set(store.keys()))
    _stdout_logger("seedsequence").info("%s%s", SEED_SEQUENCES_PREFIX, json.dumps(sequences))


def seeds_differ(sequences_a: dict[str, list[str]], sequences_b: dict[str, list[str]], prefix: str) -> bool:
    """True when the order of the keys with the prefix differs in every file (/series_*: the keys iter_series takes)."""
    keys_a, keys_b = _prefixed_sequences(sequences_a, prefix), _prefixed_sequences(sequences_b, prefix)
    return sorted(keys_a) == sorted(keys_b) and all(keys_a[name] != keys_b[name] for name in keys_a)


def _prefixed_sequences(sequences: dict[str, list[str]], prefix: str) -> dict[str, list[str]]:
    """The keys with the prefix of each file, in set iteration order."""
    return {name: [key for key in keys if key.startswith(prefix)] for name, keys in sequences.items()}


def _check_seed_controls(log: logging.Logger, sequences: dict[int, dict[str, list[str]]], prefix: str) -> None:
    """Planted controls: equal seeds and sequences that differ only outside the prefix are rejected."""
    first = sequences[SEED_CANDIDATES[0]]
    swapped = {name: _swap_outside(keys, prefix) for name, keys in first.items()}
    for case, accepted in (("equal seeds", seeds_differ(first, first, prefix)),
                           (f"sequences differing outside {prefix}* only", seeds_differ(first, swapped, prefix))):
        if accepted:
            refuse(log, "seed-control", f"planted pair accepted: {case}")
        log.info("PASS seed control rejects %s", case)


def _swap_outside(keys: list[str], prefix: str) -> list[str]:
    """Reverse the keys without the prefix among themselves, keeping every prefixed slot."""
    others = [key for key in keys if not key.startswith(prefix)][::-1]
    if len(others) < 2:
        others = others + ["/planted_a", "/planted_b"]
    rest = iter(others)
    swapped = [key if key.startswith(prefix) else next(rest) for key in keys]
    return swapped + list(rest)


def command_resume(roots: "Roots", pinned: Path, console: logging.Logger) -> int:
    """The resume rule: prove the recorded artefacts valid in place, else restart at step 1. Refused in the master tree,
    whose tracked changes a resume reverts and whose outputs a failed proof deletes."""
    if same_path(roots.worktree, Path(_master_tree())):
        refuse(console, "master-tree", f"{roots.worktree} is the master tree: a resume reverts its tracked changes and, "
                                       f"on a failed proof, deletes {roots.outputs}")
    guard_or_refuse(roots, [roots.logs, roots.lock], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_RESUME)
    lock_fd = acquire_lock(roots.lock, log)
    try:
        evidence_gate(roots, pinned, log, command=COMMAND_RESUME, mode=MODE_EVIDENCE, label=label)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs = recorded_runs(roots, log, exclude_label=label)
        failures = proof_failures(roots, log, runs)
        if not failures:
            _save_and_restore(roots, log, stamp, keep=held_paths(roots, log, runs))
            _delete_unfinished(roots, log)
            log.info("RESUME proof passed for %d recorded runs: continue within the first unfinished step", len(runs))
            return 0
        log.info("RESUME proof failed: %s", failures)
        _save_and_restore(roots, log, stamp, keep=set())
        _move_untracked(roots, log, roots.logs / f"{RESUME_PREFIX}{stamp}")
        pre = roots.logs / f"{PRE_RESUME_PREFIX}{stamp}"
        _move_earlier_logs(roots, log, pre, keep_label=label, stamp=stamp)
        _move_report_and_drafts(roots, log, pre)
        _delete_outputs(roots, log)
        log.info("RESUME restart at step 1, accepting the resumed HEAD")
        return 0
    finally:
        os.close(lock_fd)


def command_mutate(roots: "Roots", pinned: Path, finding: str, commit: str, spec_file: str, console: logging.Logger) -> int:
    """Export a fix commit afresh under data\\mut\\<F-NN>\\<m> and apply one recorded mutation (Mutation runs).

    The spec (JSON) names the mutation number m, the files it declares and its scripted replacements; the
    pristine and the mutated manifests are both written, and they must differ in exactly the declared files.
    """
    spec = json.loads(Path(spec_file).read_text(encoding="utf-8"))
    if not MUT_FINDING.match(finding) or not str(spec.get("number", "")).isdigit():
        refuse(console, "mutate", f"finding {finding!r} or mutation number {spec.get('number')!r} malformed")
    target = roots.mut / finding / str(spec["number"])
    guard_or_refuse(roots, [target], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_MUTATE + _point_suffix(f"{finding}_{spec['number']}"))
    evidence_gate(roots, pinned, log, command=COMMAND_MUTATE, mode=MODE_EVIDENCE, label=label)
    _log_event(log, "mutation", finding=finding, commit=_git_text(roots, log, "rev-parse", commit), spec=spec)
    if target.exists():
        refuse(log, "mutate", f"{target} exists: each mutation gets a fresh export")
    export_commit(roots, log, commit, target)
    pristine = write_manifest(roots, log, mutation_manifest_label(finding, str(spec["number"]), mutated=False), target)
    for change in spec["changes"]:
        apply_change(log, target, change)
    mutated = write_manifest(roots, log, mutation_manifest_label(finding, str(spec["number"]), mutated=True), target)
    changed = manifest_changes(pristine, mutated)
    if changed != sorted(spec["files"]):
        refuse(log, "mutate", f"the mutated manifest differs in {changed}, the spec declares {sorted(spec['files'])}")
    log.info("MUTATION %s %s applied to %s: %s", finding, spec["number"], target, changed)
    return 0


def export_commit(roots: "Roots", log: logging.Logger, commit: str, target: Path) -> None:
    """git archive of the commit's exported paths, extracted with zipfile into a new folder."""
    target.mkdir(parents=True)
    archive = target / f"export_{_git_text(roots, log, 'rev-parse', '--short', commit)}.zip"
    run_helper(["git", "archive", "--format=zip", "-o", str(archive), commit, "--", *EXPORT_PATHS], cwd=roots.worktree, log=log)
    with zipfile.ZipFile(archive) as source:
        source.extractall(target)


def apply_change(log: logging.Logger, root: Path, change: dict) -> None:
    """One scripted replacement on LF-normalised text of a file inside the export; the old text must occur exactly once."""
    path = root / change["path"]
    if not _contained(root, path):
        refuse(log, "mutate", f"{change['path']} resolves outside the export {root}")
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    old, new = change["old"].replace("\r\n", "\n"), change["new"].replace("\r\n", "\n")
    if text.count(old) != 1:
        refuse(log, "mutate", f"{change['path']}: the replaced text occurs {text.count(old)} times, not once")
    path.write_bytes(text.replace(old, new).encode("utf-8"))


def manifest_changes(pristine: Path, mutated: Path) -> list[str]:
    """Paths whose size or MD5 differs between two manifests, or that only one lists."""
    before, after = ({item["path"]: (item["size"], item["md5"]) for item in json.loads(path.read_text(encoding="utf-8"))["files"]}
                     for path in (pristine, mutated))
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def _point_suffix(text: str) -> str:
    """A run point fragment from free text: letters, digits and underscores only."""
    return re.sub(r"[^A-Za-z0-9_]", "", text)


def _new_command_log(roots: "Roots", console: logging.Logger, point: str) -> tuple[logging.Logger, str]:
    """Create the launch log of a non-script command under the next free harness-<point>-<n> label."""
    taken = [LABEL_PATTERN.match(path.name[:-len(".log")]) for path in _all_log_files(roots)
             if path.name.endswith(".log")]
    numbers = [int(match["seq"]) for match in taken
               if match is not None and match["stem"] == HARNESS_STEM and match["point"] == point]
    label = f"{HARNESS_STEM}-{point}-{max(numbers, default=0) + 1}"
    return _attach_launch_log(console, roots.logs / f"{label}.log"), label


# =============================================================================
# 4. EVIDENCE GATE
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """A run's identity, read from its --run file (read_run_file): the HEAD branches its evidence may come from, the
    verifier's record (the one tracked change an evidence launch admits, which a resume never reverts) and the report
    draft a failed resume proof moves aside."""

    file: Path
    branches: tuple[str, ...]
    verification_file: str
    report_path: Path


@dataclasses.dataclass(frozen=True)
class Roots:
    """Worktree-relative locations and the run's identity (None only for a self-test without --run); a fixture
    override re-roots all of them."""

    worktree: Path
    run: Optional[RunIdentity]

    @property
    def data(self) -> Path:
        return self.worktree / "data"

    @property
    def logs(self) -> Path:
        return self.data / "logs"

    @property
    def manifests(self) -> Path:
        return self.logs / "manifests"

    @property
    def lock(self) -> Path:
        return lock_path(self.worktree)

    @property
    def exp(self) -> Path:
        return self.data / "exp"

    @property
    def exp00(self) -> Path:
        return self.exp / "00"

    @property
    def base(self) -> Path:
        return self.data / "base"

    @property
    def mut(self) -> Path:
        return self.data / "mut"

    @property
    def drafts(self) -> Path:
        return self.data / DRAFTS_DIR

    @property
    def ref(self) -> Path:
        return self.data / "ref"

    @property
    def outputs(self) -> Path:
        return self.data / DATA_OUTPUT_DIR


@dataclasses.dataclass(frozen=True)
class GateRecord:
    """The logged evidence record and, for a script launch, the child's environment."""

    record: dict
    child_env: Optional[dict]


def evidence_gate(roots: Roots, pinned: Path, log: logging.Logger, *, command: str, mode: str, label: str,
                  script: Optional[Path] = None, script_argv: tuple = (),
                  timeout_s: Optional[float] = None, hashseed: int = DEFAULT_HASHSEED) -> GateRecord:
    """Record HEAD, branch, tree status, launch, environment, identities and source root."""
    record = {
        "label": label, "command": command, "mode": mode, "argv": sys.argv, "start": _now(),
        "harness_pid": os.getpid(), "harness_create_time": _own_create_time(),
        "worktree": str(roots.worktree), "run": None if roots.run is None else dataclasses.asdict(roots.run),
        "python": sys.version, "packages": _package_versions(),
        "head": _git_text(roots, log, "rev-parse", "HEAD"),
        "branch": _git_text(roots, log, "symbolic-ref", "--short", "HEAD"),
    }
    if roots.run is not None and record["branch"] not in roots.run.branches:  # a self-test without --run judges no branch
        _refuse_with_record(log, record, "branch",
                            f"HEAD is on {record['branch']}, not on {' or '.join(roots.run.branches)}")
    if mode == MODE_PRE_COMMIT and (script is None or script.stem != PRE_COMMIT_SCRIPT_STEM):
        _refuse_with_record(log, record, "pre-commit-script", f"{script} may not run in pre-commit mode")
    record["tree_status"] = _tree_status(roots, log, mode)
    if not record["tree_status"]["admitted"] and command != COMMAND_RESUME:
        _refuse_with_record(log, record, "tree", f"tree not admitted in {mode} mode: {record['tree_status']}")
    record["identities"] = identities(roots, log)
    record["line_endings"] = line_ending_settings(roots, log)
    child_env = None
    if script is None:
        record["environment"] = environment_record(dict(os.environ), {}, [], pinned)
    else:
        root, kind = _source_root(roots, log, script, script_argv, record)
        record.update(script=_display_path(roots, script), source_root=str(root), source_root_kind=kind,
                      timeout_s=timeout_s, hashseed=hashseed,
                      copies=[_copy_identity(roots, log, path) for path in _script_copies(roots, script)])
        child_env, record["environment"] = child_environment(
            pinned, hashseed=hashseed, markers=_markers(roots, label, root, pinned))
        if kind != "worktree":
            mismatches = manifest_mismatches(roots, _manifest_label(roots, root, kind), root)
            record["manifest_check"] = mismatches
            if mismatches:
                _refuse_with_record(log, record, "manifest", f"{root}: {mismatches[:5]}")
    _log_record(log, record)
    return GateRecord(record, child_env)


def _refuse_with_record(log: logging.Logger, record: dict, code: str, detail: str) -> NoReturn:
    """Log the partial record so the refusal is cited with its context, then refuse."""
    _log_record(log, record)
    refuse(log, code, detail)


def _tree_status(roots: Roots, log: logging.Logger, mode: str) -> dict:
    """Evaluate the working tree against the launch mode."""
    if mode == MODE_PRE_COMMIT:
        staged = _git_z(roots, log, "diff", "--cached", "--name-only", "-z")
        kind, reason = _classify_staged(roots, log, staged)
        return {"admitted": kind is not None, "staged": staged, "commit_kind": kind, "reason": reason,
                "tree_id": _git_text(roots, log, "write-tree")}
    changes = porcelain_records(roots, log)
    record_file = None if roots.run is None else roots.run.verification_file
    offending = [f"{xy} {path}" for xy, path in changes
                 if not (path == record_file or (mode == MODE_PRE_FIX and path.startswith(f"{TESTS_DIR}/")))]
    return {"admitted": not offending, "changes": [f"{xy} {path}" for xy, path in changes], "offending": offending}


def _classify_staged(roots: Roots, log: logging.Logger, staged: list[str]) -> tuple[Optional[str], str]:
    """An audit commit stages only audit\\ paths; a fix commit only src/, tests/ and pipeline.py."""
    if not staged:
        return None, "nothing staged"
    if all(path.startswith(f"{AUDIT_DIR}/") for path in staged):
        return COMMIT_KIND_AUDIT, ""
    if not all(path.startswith(FIX_PREFIXES) or path == PIPELINE_FILE for path in staged):
        return None, "staged paths mix the audit and fix sets or lie outside both"
    _, diff = run_helper(["git", "diff", "--cached", "-U0", "--no-color", "--no-ext-diff"],
                         cwd=roots.worktree, log=log)
    added = [line for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
    if any(AUDIT_REFERENCE.search(line) for line in added):
        return None, "the staged fix diff references audit"
    return COMMIT_KIND_FIX, ""


def porcelain_records(roots: Roots, log: logging.Logger) -> list[tuple[str, str]]:
    """(XY, path) of every tracked change; a rename or copy contributes both of its paths."""
    tokens = _git_z(roots, log, "status", "--porcelain", "-z", "--untracked-files=no")
    records, rest = [], iter(tokens)
    for token in rest:
        xy, path = token[:2], token[3:]
        records.append((xy, path))
        if "R" in xy or "C" in xy:
            records.append((xy, next(rest)))
    return records


def identities(roots: Roots, log: logging.Logger) -> dict[str, Optional[str]]:
    """Blob id of every tracked and every untracked, non-ignored file under audit\\ and tests\\."""
    tracked = _git_z(roots, log, "ls-files", "-z", "--", AUDIT_DIR, TESTS_DIR)
    untracked = _git_z(roots, log, "ls-files", "-z", "--others", "--exclude-standard", "--", AUDIT_DIR, TESTS_DIR)
    present = sorted(path for path in set(tracked) | set(untracked) if (roots.worktree / path).is_file())
    found = dict(zip(present, hash_paths(roots, log, present)))
    found.update({path: None for path in sorted(set(tracked) - set(present))})
    return dict(sorted(found.items()))


def hash_paths(roots: Roots, log: logging.Logger, paths: list[str]) -> list[str]:
    """Blob ids of files at their own path: one git hash-object --stdin-paths call."""
    if not paths:
        return []
    _, out = run_helper(["git", "hash-object", "--stdin-paths"], cwd=roots.worktree, log=log,
                        stdin_text="".join(f"{path}\n" for path in paths))
    blobs = out.split()
    if len(blobs) != len(paths):
        refuse(log, "identity", f"hash-object returned {len(blobs)} ids for {len(paths)} paths")
    return blobs


def line_ending_settings(roots: Roots, log: logging.Logger) -> dict[str, Optional[str]]:
    """The git settings that convert line endings in the worktree (None when unset): a CRLF working copy hashes as its
    LF blob only under them, so the identities depend on them."""
    found = {}
    for name in LINE_ENDING_SETTINGS:
        code, out = run_helper(["git", "config", "--get", name], cwd=roots.worktree, log=log, check=False)
        found[name] = out.strip() if code == 0 else None
    return found


def copy_blob(roots: Roots, log: logging.Logger, file: Path, as_path: str) -> str:
    """Blob id of a copy at another location, filtered as the path it copies."""
    return _git_text(roots, log, "hash-object", f"--path={as_path}", str(file))


def _script_copies(roots: Roots, script: Path) -> list[Path]:
    """The launched script and the testkit beside it, when they lie outside audit\\ and tests\\."""
    candidates = [script, script.parent / TESTKIT_FILE]
    return [path for path in candidates if path.is_file() and not _at_own_path(roots, path)]


def _at_own_path(roots: Roots, path: Path) -> bool:
    """True for a file under the worktree's audit\\ or tests\\ folders (already an identity)."""
    rel = _relative_to(roots.worktree, path)
    return rel is not None and rel.parts[0] in (AUDIT_DIR, TESTS_DIR)


def _copy_identity(roots: Roots, log: logging.Logger, path: Path) -> dict:
    """Identity of a copy: its location, the repository path it copies and its blob id."""
    as_path = copy_as_path(roots, path)
    return {"file": _display_path(roots, path), "as": as_path, "blob": copy_blob(roots, log, path, as_path)}


def copy_as_path(roots: Roots, path: Path) -> str:
    """Repository path a copy stands for: relative to its export root, or audit/<name> for a draft."""
    rel = _relative_to(roots.worktree, path)
    if rel is None:
        return path.name
    parts = rel.parts
    if parts[:2] == ("data", "base"):
        return "/".join(parts[2:])
    if parts[:2] == ("data", "mut") and len(parts) > 4:
        return "/".join(parts[4:])
    if parts[:2] == ("data", DRAFTS_DIR):
        return f"{AUDIT_DIR}/{path.name}"
    return "/".join(parts)


def _source_root(roots: Roots, log: logging.Logger, script: Path, script_argv: tuple, record: dict) -> tuple[Path, str]:
    """The tree whose src a launch runs: --source-root, else the script's nearest pipeline.py ancestor."""
    explicit = option_value(list(script_argv), "--source-root")
    root = Path(explicit) if explicit is not None else nearest_pipeline_root(script)
    if root is None:
        _refuse_with_record(log, record, "source-root", f"no {PIPELINE_FILE} above {script}")
    root = root.resolve()
    kind = root_kind(roots, root)
    if kind is None:
        _refuse_with_record(log, record, "source-root", f"{root} is not the worktree, data\\base or a mutation export")
    return root, kind


def option_value(argv: list[str], name: str) -> Optional[str]:
    """Value of --name X or --name=X in an argument list, else None."""
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith(f"{name}="):
            return token[len(name) + 1:]
    return None


def nearest_pipeline_root(path: Path) -> Optional[Path]:
    """The nearest ancestor of a file that holds pipeline.py."""
    return next((folder for folder in path.resolve().parents if (folder / PIPELINE_FILE).is_file()), None)


def root_kind(roots: Roots, root: Path) -> Optional[str]:
    """worktree, base or mut for an admitted source root, else None."""
    if same_path(root, roots.worktree):
        return "worktree"
    if same_path(root, roots.base):
        return "base"
    if (same_path(root.parent.parent, roots.mut) and MUT_FINDING.match(root.parent.name)
            and root.name.isdigit()):
        return "mut"
    return None


def _manifest_label(roots: Roots, root: Path, kind: str) -> str:
    """The manifest a source root is checked against: base, or the mutated manifest of an export."""
    if kind == "base":
        return MANIFEST_BASE
    return mutation_manifest_label(root.parent.name, root.name, mutated=True)


def mutation_manifest_label(finding: str, number: str, *, mutated: bool) -> str:
    """Manifest label of a mutation export: pristine before the mutation, mutated after it."""
    return f"mut_{finding}_{number}_{'mutated' if mutated else 'pristine'}"


def write_manifest(roots: Roots, log: logging.Logger, label: str, root: Path) -> Path:
    """Relative path, size and MD5 of every file under root, in the one format the check reads."""
    target = roots.manifests / f"{label}.json"
    guard_or_refuse(roots, [target], log)
    roots.manifests.mkdir(parents=True, exist_ok=True)
    files = [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "md5": file_md5(path)}
             for path in sorted(root.rglob("*")) if path.is_file()]
    with open(target, "x", encoding="utf-8") as sink:
        json.dump({"label": label, "root": str(root), "files": files}, sink, indent=1)
    log.info("manifest %s: %d files, %d bytes, MD5 %s", target.name, len(files),
             sum(item["size"] for item in files), file_md5(target))
    return target


def manifest_mismatches(roots: Roots, label: str, root: Path) -> list[str]:
    """Missing manifest, missing files, MD5 mismatches and unrecorded files of a root against its manifest; side
    files and copies of the worktree's tests\\ files stay outside it (each launched copy is an identity of its record)."""
    manifest = roots.manifests / f"{label}.json"
    if not manifest.is_file():
        return [f"missing manifest {manifest.name}"]
    items = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    problems = []
    for item in items:
        path = root / item["path"]
        if not path.is_file():
            problems.append(f"missing {item['path']}")
        elif file_md5(path) != item["md5"]:
            problems.append(f"changed {item['path']}")
    recorded = {item["path"] for item in items}
    present = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    return problems + [f"unrecorded {rel}" for rel in present if rel not in recorded and not _outside_manifest(roots, rel)]


def _outside_manifest(roots: Roots, rel: str) -> bool:
    """A file a root's manifest need not list: numba's cache files under __pycache__ or a copy of a worktree tests\\
    file (the testkit included) directly under the root's tests\\."""
    parts = rel.split("/")
    if CACHE_DIR in parts and Path(rel).suffix in NUMBA_CACHE_SUFFIXES:
        return True
    return (len(parts) == 2 and parts[0] == TESTS_DIR
            and (parts[1] == TESTKIT_FILE or (roots.worktree / TESTS_DIR / parts[1]).is_file()))


def _markers(roots: Roots, label: str, source_root: Path, pinned: Path) -> dict[str, str]:
    """Variables an audit child reads: the marker (the harness folder), label, roots, interpreter and the run file
    (absent only under a self-test without --run, whose own children read none)."""
    markers = {MARKER_ENV: str(HARNESS_DIR), LABEL_ENV: label, SOURCE_ROOT_ENV: str(source_root),
               WORKTREE_ENV: str(roots.worktree), INTERPRETER_ENV: str(pinned)}
    if roots.run is not None:
        markers[RUN_FILE_ENV] = str(roots.run.file)
    return markers


def child_environment(pinned: Path, *, hashseed: int, markers: Optional[dict[str, str]] = None,
                      ansi: bool = False, overrides: Optional[dict[str, str]] = None,
                      removals: tuple = ()) -> tuple[dict, dict]:
    """A child's environment (pinned venv launch, UTF-8 unless ANSI, no bytecode, seed) and its record."""
    inherited = dict(os.environ)
    settings = {BYTECODE_ENV: "1", HASHSEED_ENV: str(hashseed), VENV_LAUNCHER_ENV: str(pinned)}
    if not ansi:
        settings[UTF8_ENV] = "1"
    settings.update(markers or {})
    settings.update(overrides or {})
    removed = sorted(({UTF8_ENV} if ansi else set()) | set(removals))
    env = {name: value for name, value in inherited.items() if name not in removed}
    env.update(settings)
    return env, environment_record(inherited, settings, removed, pinned)


def environment_record(inherited: dict, settings: dict, removed: list, pinned: Path) -> dict:
    """Set and removed variables, the interpreter, inherited names, and only harmless inherited values."""
    return {"set": dict(sorted(settings.items())), "removed": removed, "interpreter": str(pinned),
            "inherited_names": sorted(inherited),
            "inherited_values": {name: inherited[name] for name in sorted(inherited) if _value_recorded(name)}}


def _value_recorded(name: str) -> bool:
    """Values of behaviour-changing variables are recorded unless the name looks like a credential."""
    upper = name.upper()
    behavioural = upper in RECORDED_EXACT or upper.startswith(RECORDED_PREFIXES)
    return behavioural and not any(fragment in upper for fragment in SECRET_FRAGMENTS)


def _package_versions() -> dict[str, str]:
    """Installed versions of the packages the pipeline and the audit use."""
    return {name: importlib.metadata.version(name) for name in RECORDED_PACKAGES}


# =============================================================================
# 5. REPARSE-POINT GUARD AND LOCK
# =============================================================================


def guard_or_refuse(roots: Roots, targets: list[Path], log: logging.Logger) -> None:
    """Refuse a target under data that crosses a reparse point or resolves outside data."""
    problems = [problem for target in targets for problem in guard_problems(roots.worktree, roots.data, target)]
    if problems:
        refuse(log, "guard", "; ".join(problems))


def guard_problems(worktree: Path, data_root: Path, target: Path) -> list[str]:
    """Both guard predicates for one target; a relative target is taken from the cwd, which is the worktree."""
    target = Path(target).absolute()
    if _relative_to(worktree, target, resolve=False) is None:
        return [f"{target} lies outside the worktree {worktree}"]
    problems = [f"reparse point {path}" for path in reparse_components(worktree, target)]
    if not _contained(data_root, target):
        problems.append(f"{target} resolves outside {data_root}")
    return problems


def reparse_components(base: Path, target: Path) -> list[Path]:
    """Existing components from base down to the target's deepest existing ancestor that are reparse points."""
    rel = _relative_to(base, target, resolve=False)
    if rel is None:
        return [target]
    components = [base] + [base.joinpath(*rel.parts[:depth]) for depth in range(1, len(rel.parts) + 1)]
    hits = []
    for component in components:
        if not os.path.lexists(component):
            break
        if os.lstat(component).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            hits.append(component)
    return hits


def _contained(root: Path, target: Path) -> bool:
    """The containment predicate: target resolves (strict=False) under the resolved root."""
    return Path(target).resolve(strict=False).is_relative_to(Path(root).resolve(strict=False))


def lock_path(worktree: Path) -> Path:
    """A worktree's harness lock, data\\logs\\harness.lock: Roots.lock, and the lock of the callers outside the
    harness that take it without a run file (perflaunch, the bounded suite run)."""
    return Roots(worktree, None).logs / "harness.lock"


def acquire_lock(path: Path, log: logging.Logger) -> int:
    """Lock byte 0 of the lock file at path (never written); LK_LOCK gives up after ten one-second retries."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_BINARY)
    started = time.monotonic()
    try:
        msvcrt.locking(fd, msvcrt.LK_LOCK, LOCK_BYTES)
    except OSError as exc:
        log.error("%s: %s held elsewhere: %s (errno %s) after %.1f s",
                  LOCK_REFUSAL, path, exc, exc.errno, time.monotonic() - started)
        os.close(fd)
        raise
    log.info("lock acquired: %s after %.1f s", path, time.monotonic() - started)
    return fd


# =============================================================================
# 6. CHILD LAUNCH, TIMEOUTS AND THE PROCESS-TREE KILL
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Identity:
    """A process identity: pid plus creation time, so a reused pid never matches."""

    pid: int
    create_time: float


@dataclasses.dataclass(frozen=True)
class Sweep:
    """One end-of-tree pass: walked members, late children and survivors."""

    members: list
    late: list
    survivors: list


@dataclasses.dataclass(frozen=True)
class LaunchResult:
    """Outcome of one bounded child launch; return_code is None after a timeout."""

    return_code: Optional[int]
    timed_out: bool
    duration_s: float
    root: Identity
    deadline_sweep: Optional[Sweep]
    final_sweep: Sweep


def launch(argv: list[str], *, env: dict, cwd: Path, out_path: Path, timeout_s: float,
           log: logging.Logger, on_poll: Optional[Callable[[], None]] = None,
           confirm_env: Optional[dict[str, str]] = None) -> LaunchResult:
    """Run a child with its output in a file; at the deadline and however the wait ends, end its whole tree.

    A child whose environment is confirmed starts suspended and runs only after the confirmation, so even a
    child that fails at once is read first and its return code is recorded like any other.
    """
    started_wall, started = time.time(), time.monotonic()
    with open(out_path, "xb") as sink:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=sink, stderr=subprocess.STDOUT,
                                creationflags=CREATE_SUSPENDED if confirm_env else 0)
    root = _root_identity(proc.pid, started_wall)
    _log_event(log, "root", pid=root.pid, create_time=root.create_time, argv=argv, out=str(out_path))
    deadline_sweep, timed_out = None, False
    try:
        if confirm_env:
            _confirm_child_env(log, root, confirm_env)
        while proc.poll() is None:
            if on_poll is not None:
                on_poll()
            if time.monotonic() - started >= timeout_s:
                timed_out = True
                deadline_sweep = end_tree(root, log, "deadline")
                break
            time.sleep(POLL_INTERVAL_S)
    finally:
        final_sweep = end_tree(root, log, "final")
    if deadline_sweep is not None and deadline_sweep.survivors:
        refuse(log, "survivors", f"alive after the deadline kill: {deadline_sweep.survivors}")
    return_code = proc.wait(timeout=SURVIVOR_BOUND_S)
    duration = time.monotonic() - started
    if final_sweep.members or final_sweep.late:
        refuse(log, "orphans", f"processes outlived {root}: {final_sweep.members + final_sweep.late}")
    return LaunchResult(None if timed_out else return_code, timed_out, duration, root, deadline_sweep, final_sweep)


def _root_identity(pid: int, started_wall: float) -> Identity:
    """The launched root's identity; a root gone before the snapshot keeps its launch time as bound, and so does a pid
    the snapshot shows under another parent (reused at once), so the walk never takes a foreign process for the root."""
    found = snapshot().get(pid)
    return Identity(pid, found[1] if found is not None and found[0] == os.getpid() else started_wall)


def _confirm_child_env(log: logging.Logger, root: Identity, expected: dict[str, str]) -> None:
    """Read the suspended root's environment block, refuse a value other than the one launched, then resume the root."""
    import psutil

    process = psutil.Process(root.pid)
    found = process.as_dict(attrs=["environ"], ad_value=None)["environ"] or {}
    actual = {name: found.get(name) for name in sorted(expected)}
    _log_event(log, "child-env", pid=root.pid, values=actual)
    if actual != dict(sorted(expected.items())):
        refuse(log, "child-env", f"child {root.pid} runs with {actual}, expected {expected}")
    process.resume()


def end_tree(root: Identity, log: logging.Logger, phase: str) -> Sweep:
    """Kill every live process of the root's tree, then its late children; report survivors."""
    members = walk(snapshot(), [root])
    _log_event(log, phase, members=[dataclasses.astuple(member) for member in members])
    if not members:
        return Sweep([], [], [])
    _taskkill(members, log)
    late = [member for member in walk(snapshot(), members) if member not in members]
    if late:
        _log_event(log, f"{phase}-late", members=[dataclasses.astuple(member) for member in late])
        _taskkill(late, log)
    survivors = survivors_of(members + late, SURVIVOR_BOUND_S)
    if survivors:
        log.error("survivors after the %s kill: %s", phase, survivors)
    return Sweep(members, late, survivors)


def snapshot() -> dict[int, tuple[int, float]]:
    """pid -> (ppid, create_time) of every process, read fresh (stale cached objects cleared).

    The parents come from one call of the map psutil's own children() walks: on Windows,
    reading ppid per process rebuilds that map for every process (1.8 s per snapshot, measured).
    """
    import psutil

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
        alive = holder is not None and _same_time(holder[1], member.create_time)
        if alive:
            found.append(member)
        reuse_time = holder[1] if holder is not None and not alive else None
        for pid in sorted(children.get(member.pid, [])):
            created = snap[pid][1]
            if pid == member.pid or created < member.create_time:
                continue
            if reuse_time is not None and created >= reuse_time:
                continue
            stack.append(Identity(pid, created))
    return sorted(found, key=dataclasses.astuple)


def survivors_of(members: list[Identity], bound_s: float) -> list[Identity]:
    """Members still alive (same pid and creation time) once the bound has elapsed."""
    deadline = time.monotonic() + bound_s
    while True:
        snap = snapshot()
        alive = [member for member in members
                 if member.pid in snap and _same_time(snap[member.pid][1], member.create_time)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(GONE_POLL_INTERVAL_S)


def _taskkill(members: list[Identity], log: logging.Logger) -> int:
    """Kill exactly the given pids with one taskkill /F; non-zero when a member ended meanwhile."""
    argv = ["taskkill", "/F"] + [part for member in members for part in ("/PID", str(member.pid))]
    code, out = run_helper(argv, cwd=REAL_WORKTREE, log=log, check=False, strict=False)
    _log_event(log, "taskkill", pids=[member.pid for member in members], return_code=code, output=out.strip()[-400:])
    return code


# =============================================================================
# 7. AUDIT-SCRIPT SIDE: PREAMBLE, CONTEXT, CHILD LAUNCHER, OUTPUT-ROOT PROBE, REPLAY SECTION, NAME MAP, FLATTENER
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ChildResult:
    """A child launched by an audit script: return code (None on timeout), duration and decoded log."""

    role: str
    return_code: Optional[int]
    timed_out: bool
    duration_s: float
    log_path: Path
    text: str
    root: Identity


@dataclasses.dataclass(frozen=True)
class Probe:
    """The output files a config copy resolves to, per concrete mode and task, and the profile the probe loaded it
    with. Only a source root with this branch's modules can be probed (probe_main, load_root_config)."""

    config: dict
    outputs: dict
    profile: str


class Context:
    """What an audit script receives from its preamble: logger, roots, label, source root, launcher."""

    def __init__(self, log: logging.Logger, roots: Roots, label: str, source_root: Path, pinned: Path) -> None:
        self.log = log
        self.roots = roots
        self.label = label
        self.source_root = source_root
        self.pinned = pinned
        self._sequence = itertools.count(1)
        self._verdicts = {"PASS": 0, "FAIL": 0}

    def rule_data(self, rules: Optional[Path]) -> Optional[dict]:
        """The rule data of the terminology checks in a rules folder (RULE_DATA_FILE), or None after one NOTICE line
        when no folder is given or the folder lacks the file: the caller then skips the checks that read it. Controls
        that lack a reported or an excluded line raise: the terminology control would leave that outcome untested. A
        report_title other than a non-blank string raises: EXP-23 would skip its title check without a NOTICE line
        (null) or pass any heading (blank)."""
        path = None if rules is None else rules / RULE_DATA_FILE
        if path is not None and path.is_file():
            data = json.loads(self.rule_text(path))
            if {reported for _, reported in data["controls"]} != {True, False}:
                raise ValueError(f"{path} holds no valid controls: [line, reported] pairs with at least one reported "
                                 "(true) and one excluded (false) line")
            if not isinstance(data["report_title"], str) or not data["report_title"].strip():
                raise ValueError(f"{path} holds no valid report_title: a string with at least one non-blank character")
            return data
        self.log.info("%s%s: the checks that read %s are skipped", NOTICE_PREFIX,
                      "no --rules folder" if path is None else f"{path} is not a file", RULE_DATA_FILE)
        return None

    def rule_text(self, path: Path) -> str:
        """The text of a rules-folder file, after one line with its path and MD5: the folder lies outside the commit
        that an evidence record names. One read yields both, so the MD5 names the very text returned, decoded as
        read_text decodes it (UTF-8, universal newlines)."""
        content = path.read_bytes()
        self.log.info("rule file %s (MD5 %s)", path, hashlib.md5(content).hexdigest())
        return io.TextIOWrapper(io.BytesIO(content), encoding="utf-8").read()

    def guard(self, *targets: Path) -> None:
        """Every target under data passes the reparse-point guard first."""
        guard_or_refuse(self.roots, list(targets), self.log)

    def exp_dir(self, number: str) -> Path:
        """The experiment's own output folder data\\exp\\NN, guarded and created."""
        path = self.roots.exp / number
        self.guard(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def label_dir(self, number: str) -> Path:
        """This launch's folder data\\exp\\NN\\<base|tip>\\<label> (base: the run's source root is data\\base)."""
        kind = "base" if same_path(self.source_root, self.roots.base) else "tip"
        return self.exp_dir(f"{number}/{kind}/{self.label}")

    def verdict(self, name: str, passed: bool, expected: object, actual: object) -> bool:
        """Log one PASS/FAIL line with expected and actual values."""
        outcome = "PASS" if passed else "FAIL"
        self._verdicts[outcome] += 1
        self.log.info("%s %s: expected %s; actual %s", outcome, name, expected, actual)
        return passed

    def summary(self) -> None:
        """Log the PASS/FAIL counts of the script."""
        self.log.info("SUMMARY %d PASS, %d FAIL", self._verdicts["PASS"], self._verdicts["FAIL"])

    def write_manifest(self, label: str, root: Path) -> Path:
        """Manifest of a labelled copy, written by the harness's one manifest function."""
        return write_manifest(self.roots, self.log, label, root)

    def launch_child(self, role: str, python_args: list[str], *, timeout_s: float, hashseed: int,
                     ansi: bool = False, audit_child: bool = False,
                     env_overrides: Optional[dict[str, str]] = None, env_removals: tuple = (),
                     confirm_env: Optional[dict[str, str]] = None) -> ChildResult:
        """Launch a Python child under the pinned environment; its output goes to its own log."""
        out_path = self.roots.logs / f"{self.label}.{role}{next(self._sequence)}.log"
        self.guard(out_path)
        markers = _markers(self.roots, self.label, self.source_root, self.pinned) if audit_child else None
        if markers is not None and ansi:
            markers[ANSI_CHILD_ENV] = "1"
        env, env_record = child_environment(self.pinned, hashseed=hashseed, markers=markers, ansi=ansi,
                                            overrides=env_overrides, removals=env_removals)
        interpreter = [sys._base_executable, "-B"] + (["-X", "utf8=0"] if ansi else [])
        _log_event(self.log, "child", role=role, log_file=out_path.name, args=python_args, hashseed=hashseed,
                   timeout_s=timeout_s, ansi=ansi, environment={"set": env_record["set"], "removed": env_record["removed"]})
        result = launch(interpreter + python_args, env=env, cwd=REAL_WORKTREE, out_path=out_path,
                        timeout_s=timeout_s, log=self.log, confirm_env=confirm_env)
        text = decode_child_output(out_path, ansi, self.log)
        _log_event(self.log, "child-end", role=role, log_file=out_path.name, return_code=result.return_code,
                   timed_out=result.timed_out, duration_s=round(result.duration_s, 3))
        return ChildResult(role, result.return_code, result.timed_out, result.duration_s, out_path, text, result.root)

    def probe(self, role: str, config_file: str, overrides: dict, hashseed: int) -> Probe:
        """Output files of a config copy, computed by a child of the run's source root."""
        code = ("import sys; sys.dont_write_bytecode = True; "
                f"sys.path.insert(0, {str(HARNESS_DIR)!r}); import harness; harness.probe_main(sys.argv[1:])")
        child = self.launch_child(role, ["-c", code, str(self.source_root), config_file, json.dumps(overrides),
                                         str(self.pinned)], timeout_s=CHILD_TIMEOUT_S, hashseed=hashseed)
        if child.return_code != 0:
            refuse(self.log, "probe", f"probe child returned {child.return_code}: {child.log_path.name}")
        report = json.loads(_prefixed_line(child.text, PROBE_PREFIX))
        return Probe(report["config"], {mode: {task: Path(path) for task, path in tasks.items()}
                                        for mode, tasks in report["outputs"].items()}, report["profile"])

    def run_pipeline(self, role: str, *, strategy: str, tasks: list[str], quiet_luigi: bool, hashseed: int,
                     allowed_root: Path, overrides: Optional[dict] = None, timeout_s: float = PIPELINE_TIMEOUT_S,
                     child_argv: Optional[list[str]] = None, ansi: bool = False) -> tuple[ChildResult, Probe, dict]:
        """Probe, check the output root, run the pipeline of the source root, confirm its origin.

        Without child_argv the child is ORIGIN_CHILD, which runs the source root's pipeline.py as __main__; with it,
        the child is that audit script, which receives the pipeline arguments as JSON after --pipeline-args and calls
        pipeline.main() itself (patched children). Either reports src's origin on an ORIGIN line (origin_verdict).
        """
        config_file = f"workflows/{strategy}.yaml"
        probe = self.probe(f"{role}probe", config_file, overrides or {}, hashseed)
        outputs = [path for tasks_of in probe.outputs.values() for path in tasks_of.values()]
        refuse_output_root(self.roots, self.log, outputs, allowed_root)
        self.guard(*outputs)
        pipeline_args = (["--task", *tasks, "--strategy", strategy] + ([] if quiet_luigi else ["--quiet-luigi", "False"])
                         + ["--profile", probe.profile])
        args = (["-c", ORIGIN_CHILD, str(self.source_root), json.dumps(pipeline_args)] if child_argv is None
                else [*child_argv, PIPELINE_ARGS_OPTION, json.dumps(pipeline_args)])
        child = self.launch_child(role, args, timeout_s=timeout_s, hashseed=hashseed, ansi=ansi,
                                  audit_child=child_argv is not None, confirm_env={HASHSEED_ENV: str(hashseed)})
        origin = origin_verdict(self.source_root, child.text)
        _log_event(self.log, "origin", role=role, **origin)
        return child, probe, origin


def preamble(script_file: str, *, imports_src: bool) -> Context:
    """Assert the marker, the venv, the bytecode flag and, for in-process pythia, UTF-8 (a deliberate ANSI child
    records its code page instead), then for in-process pythia report src's origin on an ORIGIN line (as
    ORIGIN_CHILD does); set up logging and read the run file the harness names."""
    marker = os.environ.get(MARKER_ENV)
    if marker is None or not os.path.samefile(marker, HARNESS_DIR):
        raise Refusal("marker", f"{Path(script_file).name} runs only through audit/harness.py")
    pinned = Path(os.environ[INTERPRETER_ENV])
    if not (os.path.samefile(sys.executable, pinned) and sys.prefix != sys.base_prefix):
        raise Refusal("venv", f"{sys.executable} (prefix {sys.prefix}) is not the active pinned venv {pinned}")
    if not sys.dont_write_bytecode:
        raise Refusal("bytecode", f"{Path(script_file).name} must set sys.dont_write_bytecode first")
    if os.environ.get(ANSI_CHILD_ENV) == "1":
        sys.stdout.write(CHILD_ENCODING_PREFIX + locale.getpreferredencoding(False) + "\n")
        sys.stdout.flush()
    elif imports_src and sys.flags.utf8_mode != 1:
        raise Refusal("utf8", f"{Path(script_file).name} imports src and needs UTF-8 mode")
    source_root = Path(os.environ[SOURCE_ROOT_ENV])
    if imports_src:
        insert_source_root(source_root)
        sys.stdout.write(f"{ORIGIN_PREFIX}{sys.modules['src'].__file__}\n")
        sys.stdout.flush()
    log = _stdout_logger(Path(script_file).stem)
    run_file = os.environ.get(RUN_FILE_ENV)
    run = read_run_file(None if run_file is None else Path(run_file), log)
    return Context(log, Roots(Path(os.environ[WORKTREE_ENV]), run), os.environ[LABEL_ENV], source_root, pinned)


def insert_source_root(source_root: Path) -> None:
    """Put the source root first on sys.path, import src and assert that it comes from there."""
    sys.path.insert(0, str(source_root))
    import src

    origin = Path(src.__file__).resolve()
    if not origin.is_relative_to(source_root.resolve()):
        raise Refusal("src-origin", f"src imported from {origin}, not from {source_root}")


def probe_main(argv: list[str]) -> None:
    """Child side of the probe: load the config copy, assert overrides, report its output files. The output files come
    from the read, replay, strategy and gate sections validated as the pipeline validates them (_probe_section), so
    each value has the type the pipeline names its folders by; a section the copy holds invalid is built from its own
    values, so a copy an experiment plants with an unknown key or a bad value is probed too and its ValidationError
    left to the pipeline child. A copy missing a field the output files are named by (read.output_root, ...) is built
    without it, so get_output_file_name raises and the probe refuses (REFUSED[probe]); no experiment plants one. The
    gate section (trigger.gate) is built only on a root whose TriggerConfig holds it, so a root older than the gate
    still names its own folder; a copy without the section takes the schema default, and a copy whose gate section is
    invalid and holds items (an invalid item, an unknown key in trigger.gate) gets it from model_construct with the
    items left as mappings, which the output name cannot read, so the probe refuses (REFUSED[probe]); no experiment
    plants one. The source root must hold src.utils.common.outputs and the section models: a root older than this
    branch cannot be probed."""
    source_root, config_file, overrides_json, pinned = argv
    if not (os.path.samefile(sys.executable, pinned) and sys.prefix != sys.base_prefix):
        raise Refusal("venv", f"probe runs {sys.executable}, not the active pinned venv {pinned}")
    insert_source_root(Path(source_root))
    from src.utils.common.outputs import get_output_file_name
    from src.utils.common.schema import ReadConfig, ReplayConfig, StrategyConfig, TriggerConfig

    config, profile = load_root_config(config_file)
    for dotted, expected in sorted(json.loads(overrides_json).items()):
        actual = config
        for part in dotted.split("."):
            actual = actual[part]
        if actual != expected:
            raise Refusal("override", f"{dotted} is {actual!r}, expected {expected!r}")
    section = replay_key(config)
    mode = config["trigger"][section][REPLAY_MODE_KEY]
    read = _probe_section(ReadConfig, config["read"])
    sections = {"strategy": _probe_section(StrategyConfig, config["trigger"]["strategy"])}
    if GATE_SECTION_KEY in TriggerConfig.model_fields:
        from src.utils.common.schema import GateConfig

        sections[GATE_SECTION_KEY] = _probe_section(GateConfig, config["trigger"].get(GATE_SECTION_KEY, {}))
    outputs = {}
    for concrete in (["ideal", "realistic"] if mode == "dual" else [mode]):
        replay = _probe_section(ReplayConfig, {**config["trigger"][section], REPLAY_MODE_KEY: concrete})
        trigger = TriggerConfig.model_construct(replay=replay, **sections)
        outputs[concrete] = {task: str(get_output_file_name(read, trigger, task).resolve())
                             for task in ("trigger", "signal", "performance")}
    report = {"config": {"read": config["read"], "mode": mode}, "outputs": outputs, "profile": profile}
    _stdout_logger("probe").info("%s%s", PROBE_PREFIX, json.dumps(report))


def load_root_config(config_file: str) -> tuple[dict, str]:
    """The merged config of a copy as the imported source root's load_config reads it under the dev profile, and that
    profile. The root must hold src.utils.common.config: a root older than this branch cannot be loaded."""
    from src.utils.common.config import load_config

    return load_config(config_file, DEV_PROFILE), DEV_PROFILE


def replay_key(config: dict) -> str:
    """The key of the one trigger subsection that holds mode, found by content so that one tool reads the replay
    section of any tree; a ValueError names the holders when there is not exactly one."""
    holders = sorted(key for key, value in config["trigger"].items() if isinstance(value, dict) and REPLAY_MODE_KEY in value)
    if len(holders) != 1:
        raise ValueError(f"exactly one trigger subsection must hold {REPLAY_MODE_KEY!r}; holders: {holders}")
    return holders[0]


def _probe_section(model: type[SECTION], values: dict) -> SECTION:
    """A section validated (model_validate) when its values are valid, as the model's validator answers without
    raising (isinstance_python), else built from the copy's own values (model_construct)."""
    if model.__pydantic_validator__.isinstance_python(values):
        return model.model_validate(values)
    return model.model_construct(**values)


def refuse_output_root(roots: Roots, log: logging.Logger, outputs: list[Path], allowed_root: Path) -> None:
    """Refuse output files outside their allowed root or under data\\ref, run*, base or mut."""
    problems = output_root_problems(roots, outputs, allowed_root)
    if problems:
        refuse(log, "output-root", "; ".join(problems))


def output_root_problems(roots: Roots, outputs: list[Path], allowed_root: Path) -> list[str]:
    """The output-root rule for each output file."""
    problems = []
    for output in outputs:
        if not _contained(allowed_root, output):
            problems.append(f"{output} lies outside {allowed_root}")
        rel = _relative_to(roots.data, output)
        if rel is not None and (rel.parts[0] in FORBIDDEN_OUTPUT_PARTS or rel.parts[0].startswith(RUN_COPY_PREFIX)):
            problems.append(f"{output} lies under data\\{rel.parts[0]}")
    return problems


def origin_verdict(source_root: Path, text: str) -> dict:
    """The origin of src a pipeline child reported on the last ORIGIN line of its capture: confirmed when the file lies
    under <source root>\\src; not-imported and a capture without the line are unconfirmed."""
    reported = [line[len(ORIGIN_PREFIX):].strip() for line in text.splitlines() if line.startswith(ORIGIN_PREFIX)]
    origin = reported[-1] if reported else None
    confirmed = origin not in (None, ORIGIN_NOT_IMPORTED) and _contained(source_root / "src", Path(origin))
    return {"source_tree": str(source_root), "origin": origin, "confirmed": confirmed}


def decode_child_output(path: Path, ansi: bool, log: logging.Logger) -> str:
    """Decode a child's capture: UTF-8, or the parent's ANSI code page after the child's own record."""
    raw = path.read_bytes()
    if not ansi:
        return _strict_decode(raw, "utf-8", path, log)
    expected = locale.getencoding()
    first = raw.split(b"\n", 1)[0].decode("ascii", "backslashreplace").strip()
    if first != f"{CHILD_ENCODING_PREFIX}{expected}":
        refuse(log, "ansi-encoding", f"{path.name} recorded {first!r}, parent code page is {expected}")
    return _strict_decode(raw, expected, path, log)


def _strict_decode(raw: bytes, encoding: str, path: Path, log: logging.Logger) -> str:
    """Strict decoding; an undecodable capture is logged at ERROR and raised."""
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        log.error("capture %s is not valid %s: %s", path.name, encoding, exc)
        raise


def plain_lines(text: str) -> list[str]:
    """A capture's lines with ANSI escape sequences removed (pipeline.py:111-128 colours WARNING and ERROR
    records unconditionally), for substring matching."""
    return ANSI_ESCAPE.sub("", text).splitlines()


def error_lines(lines: list[str]) -> list[str]:
    """The lines that carry an ERROR record."""
    return [line for line in lines if ERROR_MARK in line]


def processed_counts(lines: list[str]) -> list[tuple[int, int]]:
    """(N, M) of every 'Processed N items, successful: M' line (producerconsumer.py:59)."""
    return [(int(match.group(1)), int(match.group(2))) for match in map(PROCESSED_LINE.search, lines) if match]


def _master_tree() -> str:
    """The master tree: the parent of the shared git directory."""
    _, out = run_helper(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                        cwd=REAL_WORKTREE, log=logging.getLogger(HARNESS_STEM))
    return str(Path(out.strip()).parent)


def rename_map(path: Path) -> dict:
    """A name map (the JSON of a --rename option): exactly the parts of NAME_MAP_PARTS, each an {old: new} mapping,
    "values" one per column, and NAME_MAP_DROP, a list of column labels; the map drops and renames in the older tree
    of a comparison."""
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    valid = (isinstance(loaded, dict) and sorted(loaded) == sorted((*NAME_MAP_PARTS, NAME_MAP_DROP))
             and all(isinstance(loaded[part], dict) for part in NAME_MAP_PARTS)
             and all(isinstance(pairs, dict) for pairs in loaded["values"].values())
             and isinstance(loaded[NAME_MAP_DROP], list) and all(isinstance(label, str) for label in loaded[NAME_MAP_DROP]))
    if not valid:
        raise ValueError(f"{path} is no name map: a JSON object of the mappings {NAME_MAP_PARTS} and the label list "
                         f"{NAME_MAP_DROP}")
    return loaded


def renamed_keys(keys: list[str], mapping: dict) -> dict[str, str]:
    """{mapped key: key as written} of a store's keys under the map's "keys"; two keys renamed onto one raise."""
    mapped: dict[str, str] = {}
    for key in keys:
        new = mapping["keys"].get(key, key)
        if new in mapped:
            raise ValueError(f"the name map renames {mapped[new]} and {key} to the one key {new}")
        mapped[new] = key
    return mapped


def kept_frame(stored, mapping: dict):
    """A stored DataFrame without the column labels of the map's NAME_MAP_DROP (a label it lacks ignored), the input
    unchanged; a Series is returned as stored."""
    import pandas as pd

    if isinstance(stored, pd.Series):
        return stored
    return stored.drop(columns=mapping[NAME_MAP_DROP], errors="ignore")


def renamed_frame(stored, mapping: dict):
    """A stored DataFrame or Series renamed, the input unchanged: the column labels (a Series' name) under "columns",
    then the string cells of each column "values" names, after that rename, under its pairs, every dtype kept."""
    import pandas as pd

    if isinstance(stored, pd.Series):
        name = mapping["columns"].get(stored.name, stored.name)
        return _renamed_cells(stored.rename(name), mapping["values"].get(name, {}))
    renamed = stored.rename(columns=mapping["columns"])
    for number, label in enumerate(renamed.columns):
        if label in mapping["values"]:
            renamed.isetitem(number, _renamed_cells(renamed.iloc[:, number], mapping["values"][label]))
    return renamed


def _renamed_cells(column, pairs: dict):
    """A column whose string cells the pairs name renamed, its dtype kept."""
    return column.map(lambda cell: pairs.get(cell, cell) if isinstance(cell, str) else cell).astype(column.dtype)


def renamed_document(document, mapping: dict):
    """A parsed YAML document with the keys of its mappings renamed at every depth under "yaml_keys", each mapping's
    key sequence kept; two keys of one mapping renamed onto one raise."""
    if isinstance(document, list):
        return [renamed_document(item, mapping) for item in document]
    if not isinstance(document, dict):
        return document
    renamed = {}
    for key, value in document.items():
        new = mapping["yaml_keys"].get(key, key)
        if new in renamed:
            raise ValueError(f"the name map renames two keys of one mapping to {new}")
        renamed[new] = renamed_document(value, mapping)
    return renamed


def flatten_yaml(document: dict, prefix: str = "") -> dict:
    """Dotted key -> leaf value of a YAML document, the audit family's one flattener: an empty mapping is a leaf
    (its value {}), so a key the document declares empty is kept."""
    flat = {}
    for key, value in document.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            flat.update(flatten_yaml(value, f"{dotted}."))
            continue
        flat[dotted] = value
    return flat


# =============================================================================
# 8. SELF-TEST
# =============================================================================


def command_self_test(roots: Roots, pinned: Path, console: logging.Logger, mode: str) -> int:
    """Run every planted case; a failed required outcome blocks evidence launches. Mode pre-fix admits a changed
    committed tests/testkit.py, whose hang case must pass before the changed file's first use."""
    if mode == MODE_PRE_COMMIT:
        refuse(console, "mode", "the self-test runs in evidence or pre-fix mode")
    guard_or_refuse(roots, [roots.logs], console)
    roots.logs.mkdir(parents=True, exist_ok=True)
    log, label = _new_command_log(roots, console, COMMAND_SELF_TEST)
    evidence_gate(roots, pinned, log, command=COMMAND_SELF_TEST, mode=mode, label=label)
    guard_or_refuse(roots, [roots.exp00], log)
    roots.exp00.mkdir(parents=True, exist_ok=True)
    suite = SelfTest(roots, pinned, log, label)
    before = suite.real_state()
    for case in suite.cases():
        log.info("CASE %s", case.__name__)
        case()
    suite.check("resume/real-worktree-unchanged", suite.real_state() == before,
                "tracked status, data listing and report MD5 unchanged by every fixture case")
    log.info("SELFTEST %d PASS, %d FAIL %s", suite.passed, len(suite.failed), suite.failed)
    return 1 if suite.failed else 0


class SelfTest:
    """Planted cases of section 6.1's self-test table, each in a fresh fixture under data\\exp\\00."""

    def __init__(self, roots: Roots, pinned: Path, log: logging.Logger, label: str) -> None:
        self.roots = roots
        self.pinned = pinned
        self.log = log
        self.label = label
        self.passed = 0
        self.failed: list[str] = []

    def cases(self) -> list[Callable[[], None]]:
        """Cases in table order."""
        return [
            self.case_gate_branch, self.case_gate_evidence_admitted, self.case_gate_evidence_refused,
            self.case_gate_pre_fix, self.case_gate_pre_commit_admitted, self.case_gate_pre_commit_refused,
            self.case_gate_commands, self.case_run_file, self.case_identity_crlf, self.case_identity_helpers,
            self.case_environment, self.case_environment_fast_return, self.case_source_root, self.case_manifests, self.case_guard,
            self.case_guard_commands, self.case_output_roots, self.case_probe_sections, self.case_rule_data,
            self.case_kill_hang, self.case_kill_walk,
            self.case_kill_root_parent, self.case_kill_survivors, self.case_kill_surviving_root,
            self.case_kill_poll_error, self.case_kill_orphan, self.case_kill_helper_bound,
            self.case_lock_killed_holder, self.case_lock_held, self.case_resume_passes,
            self.case_resume_fails, self.case_resume_ignored, self.case_resume_actions,
            self.case_resume_restart, self.case_resume_second_restart, self.case_master_tree, self.case_origin_child,
            self.case_preamble,
            self.case_preamble_control, self.case_preamble_venv, self.case_testkit_hang, self.case_mutate,
        ]

    def check(self, case: str, condition: bool, detail: str) -> None:
        """Log one PASS/FAIL line of a required outcome."""
        if condition:
            self.passed += 1
        else:
            self.failed.append(case)
        self.log.info("%s %s: %s", "PASS" if condition else "FAIL", case, detail)

    def real_state(self) -> dict:
        """Real worktree tracked status, data listing (without the self-test's logs) and report MD5."""
        listing = sorted(
            (path.relative_to(self.roots.data).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
            for path in self.roots.data.rglob("*")
            if path.is_file() and not path.name.startswith(self.label) and not _contained(self.roots.exp00, path))
        report = None if self.roots.run is None else self.roots.run.report_path
        report_md5 = file_md5(report) if report is not None and report.is_file() else None
        return {"status": porcelain_records(self.roots, self.log), "data": listing, "report": report_md5}

    # --- fixtures -------------------------------------------------------------

    def fixture(self, folder: Path) -> "Fixture":
        """A fixture repository on the planted run's first branch with the committed files the gate cases need; its run
        file lies beside it."""
        repo = folder / "repo"
        repo.mkdir()
        fixture = Fixture(self, folder, repo)
        fixture.git("init", "-q", "-b", FIXTURE_BRANCHES[0], str(repo))
        fixture.git("config", "core.autocrlf", "true")  # the real worktree's conversion, set here so no host setting decides it
        fixture.write(".gitignore", "data\n")
        fixture.write(PIPELINE_FILE, "# placeholder pipeline: makes the fixture a source root\n")
        fixture.write("ordinary.md", "ordinary\n")
        fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT)
        fixture.write(f"{AUDIT_DIR}/h.py", _FIXTURE_SCRIPT)
        fixture.write(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", _FIXTURE_SCRIPT)
        fixture.write(FIXTURE_VERIFICATION_FILE, "record\n")
        fixture.commit("fixture")
        return fixture

    def folder(self) -> tempfile.TemporaryDirectory:
        """A fresh temporary folder under data\\exp\\00 for one case."""
        return tempfile.TemporaryDirectory(dir=self.roots.exp00)

    # --- gate -----------------------------------------------------------------

    def case_gate_branch(self) -> None:
        """Each further accepted branch is admitted; another branch is refused in every mode."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            for number, branch in enumerate(FIXTURE_BRANCHES[1:], start=1):
                fixture.git("checkout", "-q", "-b", branch)
                run = fixture.script(f"{AUDIT_DIR}/h.py", f"h-accepted{number}-1")
                self.check(f"gate/accepted-branch/{branch}", run.code == 0, run.summary())
            fixture.git("checkout", "-q", "-b", "other")
            for number, mode in enumerate(MODES, start=1):
                run = fixture.script(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", f"exp20sweeps-branch-{number}", mode=mode)
                self.check(f"gate/other-branch/{mode}", run.refused("branch"), run.summary())

    def case_gate_evidence_admitted(self) -> None:
        """Evidence mode admits a clean tree and a tree whose only change is the verifier's record."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-clean-1")
            self.check("gate/evidence/clean", run.code == 0, run.summary())
            fixture.write(FIXTURE_VERIFICATION_FILE, "record modified\n")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-record-1")
            self.check("gate/evidence/record-only", run.code == 0, run.summary())

    def case_gate_evidence_refused(self) -> None:
        """Evidence mode refuses any other tracked change or staged file."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            plants = (("ordinary", ["ordinary.md"], False), ("ordinary+record", ["ordinary.md", FIXTURE_VERIFICATION_FILE], False),
                      ("staged-new", ["staged.md"], True), ("tests-modified", [f"{TESTS_DIR}/x.py"], False))
            for number, (name, paths, stage) in enumerate(plants, start=1):
                for path in paths:
                    fixture.write(path, f"planted change {name}\n")
                if stage:
                    fixture.git("add", *paths)
                run = fixture.script(f"{AUDIT_DIR}/h.py", f"h-dirty{number}-1")
                self.check(f"gate/evidence/{name}", run.refused("tree"), run.summary())
                fixture.reset()

    def case_gate_pre_fix(self) -> None:
        """pre-fix admits tracked changes confined to tests\\ only."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT + "# edited regression\n")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-prefixtests-1", mode=MODE_PRE_FIX)
            self.check("gate/pre-fix/tests-modified", run.code == 0, run.summary())
            fixture.reset()
            fixture.write("ordinary.md", "changed\n")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-prefixordinary-1", mode=MODE_PRE_FIX)
            self.check("gate/pre-fix/ordinary-modified", run.refused("tree"), run.summary())

    def case_gate_pre_commit_admitted(self) -> None:
        """pre-commit admits a staged fix diff (tree id = the commit's tree) and an audit diff."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write("src/y.py", "VALUE = 1\n")
            fixture.git("add", "src/y.py")
            run = fixture.script(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", "exp20sweeps-gatef01-1", mode=MODE_PRE_COMMIT)
            recorded = run.record().get("tree_status", {}).get("tree_id")
            fixture.commit("fix")
            committed = fixture.git("rev-parse", "HEAD^{tree}").strip()
            self.check("gate/pre-commit/fix", run.code == 0 and recorded == committed,
                       f"{run.summary()}; recorded tree {recorded}, commit tree {committed}")
            fixture.write(f"{AUDIT_DIR}/z.py", "# an audit experiment\n")
            fixture.git("add", f"{AUDIT_DIR}/z.py")
            run = fixture.script(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", "exp20sweeps-gatea1-1", mode=MODE_PRE_COMMIT)
            self.check("gate/pre-commit/audit-mentions-audit", run.code == 0, run.summary())

    def case_gate_pre_commit_refused(self) -> None:
        """pre-commit refuses mixed paths, outside paths, a fix referencing audit and another script."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            plants = (("mixed", {f"{AUDIT_DIR}/z.py": "# z\n", "src/y.py": "VALUE = 1\n"}),
                      ("outside", {"ordinary.md": "staged change\n"}),
                      ("fix-references-audit", {"src/y.py": "# see audit/z.py\n"}))
            for number, (name, files) in enumerate(plants, start=1):
                for path, text in files.items():
                    fixture.write(path, text)
                fixture.git("add", *files)
                run = fixture.script(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", f"exp20sweeps-gate{number}-1", mode=MODE_PRE_COMMIT)
                self.check(f"gate/pre-commit/{name}", run.refused("tree"), run.summary())
                fixture.reset()
            fixture.write("src/y.py", "VALUE = 1\n")
            fixture.git("add", "src/y.py")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-precommit-1", mode=MODE_PRE_COMMIT)
            self.check("gate/pre-commit/other-script", run.refused("pre-commit-script"), run.summary())

    def case_gate_commands(self) -> None:
        """Non-script commands, --resume, a reused label and the two invalid overrides."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            first = fixture.script(f"{AUDIT_DIR}/h.py", "h-once-1")
            again = fixture.script(f"{AUDIT_DIR}/h.py", "h-once-1")
            self.check("gate/label-exists", first.code == 0 and again.refused("label-exists"), again.summary())
            fixture.write("ordinary.md", "changed\n")
            base = Path(tmp) / "manifested"
            base.mkdir()
            (base / "f.txt").write_text("f\n", encoding="utf-8")
            run = fixture.harness("--manifest", "planted", str(base))
            self.check("gate/manifest-dirty-tree", run.refused("tree"), run.summary())
            run = fixture.harness("--resume")
            resume = fixture.newest_record("resume")
            self.check("gate/resume-dirty-tree", run.code == 0 and resume.get("tree_status", {}).get("admitted") is False,
                       f"{run.summary()}; recorded admitted={resume.get('tree_status', {}).get('admitted')}")
            outside = self.roots.exp / "planted_outside"
            run = self.override_run(outside)
            self.check("gate/worktree-outside", run.refused("worktree") and not outside.exists(), run.summary())
            inner = fixture.repo / "sub"
            inner.mkdir()
            run = self.override_run(inner)
            self.check("gate/worktree-not-top-level", run.refused("worktree") and not any(inner.iterdir()), run.summary())

    def override_run(self, worktree: Path) -> "FixtureRun":
        """A guard command under an override that must be refused before any write."""
        folder = Path(tempfile.mkdtemp(dir=self.roots.exp00))
        console = folder / "console.txt"
        result = launch([sys._base_executable, "-B", str(HARNESS_FILE), "--worktree", str(worktree), "--run",
                         str(plant_run_file(folder, worktree).file), "--guard", str(worktree / "data")],
                        env=fixture_env(self.pinned), cwd=REAL_WORKTREE,
                        out_path=console, timeout_s=FIXTURE_TIMEOUT_S, log=self.log)
        text = console.read_text(encoding="utf-8")
        shutil.rmtree(folder)
        return FixtureRun(result.return_code, result.timed_out, text)

    def case_run_file(self) -> None:
        """A command without --run, or whose run file is absent or malformed, is refused before its first write; the
        planted run file admits the launch, and the record names the run."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            valid = {"branches": list(FIXTURE_BRANCHES), "verification_file": FIXTURE_VERIFICATION_FILE,
                     "report_path": str(fixture.run.report_path)}
            malformed = (("not-json", "{"), ("not-an-object", "[]"),
                         ("missing-key", json.dumps({key: valid[key] for key in RUN_FILE_KEYS[:2]})),
                         ("extra-key", json.dumps({**valid, "planted": 1})),
                         ("empty-branches", json.dumps({**valid, "branches": []})),
                         ("absolute-record", json.dumps({**valid, "verification_file": str(Path(tmp) / "record.txt")})),
                         ("relative-report", json.dumps({**valid, "report_path": FIXTURE_REPORT_NAME})))
            refusals = [("no-option", []), ("absent-file", ["--run", str(Path(tmp) / "absent.json")])]
            for name, text in malformed:
                path = Path(tmp) / f"{name}.json"
                path.write_text(text, encoding="utf-8")
                refusals.append((f"malformed/{name}", ["--run", str(path)]))
            for number, (name, run_args) in enumerate(refusals, start=1):
                run = fixture.script(f"{AUDIT_DIR}/h.py", f"h-runfile{number}-1", run_args=run_args)
                self.check(f"run-file/{name}", run.refused("run-file") and not fixture.roots.logs.exists(), run.summary())
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-plantedrun-1")
            recorded = run.record().get("run") or {}
            self.check("run-file/planted", run.code == 0 and recorded.get("branches") == list(FIXTURE_BRANCHES)
                       and recorded.get("verification_file") == FIXTURE_VERIFICATION_FILE, f"{run.summary()}; {recorded}")

    # --- identity -------------------------------------------------------------

    def case_identity_crlf(self) -> None:
        """The record names the line-ending conversion; under it a CRLF working copy of an LF blob matches it and one
        changed byte does not."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            recorded = fixture.script(f"{AUDIT_DIR}/h.py", "h-crlf-1").record().get("line_endings", {})
            self.check("identity/line-endings-recorded", recorded.get("core.autocrlf") == "true", str(recorded))
            committed = fixture.git("rev-parse", f"HEAD:{TESTS_DIR}/x.py").strip()
            (fixture.repo / TESTS_DIR / "x.py").write_bytes(_FIXTURE_SCRIPT.replace("\n", "\r\n").encode("ascii"))
            crlf = identities(fixture.roots, self.log)[f"{TESTS_DIR}/x.py"]
            (fixture.repo / TESTS_DIR / "x.py").write_bytes(_FIXTURE_SCRIPT.replace("\n", "\r\n").replace("1", "2").encode("ascii"))
            changed = identities(fixture.roots, self.log)[f"{TESTS_DIR}/x.py"]
            self.check("identity/crlf-copy", crlf == committed, f"working {crlf}, blob {committed}")
            self.check("identity/one-byte", changed != committed, f"working {changed}, blob {committed}")

    def case_identity_helpers(self) -> None:
        """Untracked helpers are recorded, an edited committed helper differs, ignored files are not."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/helper.py", "HELPER = 1\n")
            fixture.write(f"{AUDIT_DIR}/data/ignored.py", "IGNORED = 1\n")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-helper-1")
            recorded = run.record().get("identities", {})
            self.check("identity/untracked-helper", f"{AUDIT_DIR}/helper.py" in recorded, sorted(recorded).__str__())
            self.check("identity/ignored-file", f"{AUDIT_DIR}/data/ignored.py" not in recorded, sorted(recorded).__str__())
            fixture.git("add", f"{AUDIT_DIR}/helper.py")
            fixture.commit("helper")
            fixture.write(f"{AUDIT_DIR}/helper.py", "HELPER = 2\n")
            fixture.git("add", f"{AUDIT_DIR}/helper.py")
            run = fixture.script(f"{AUDIT_DIR}/{PRE_COMMIT_SCRIPT_STEM}.py", "exp20sweeps-gatea2-1", mode=MODE_PRE_COMMIT)
            edited = run.record().get("identities", {}).get(f"{AUDIT_DIR}/helper.py")
            committed = fixture.git("rev-parse", f"HEAD:{AUDIT_DIR}/helper.py").strip()
            self.check("identity/edited-helper", run.code == 0 and edited not in (None, committed),
                       f"recorded {edited}, committed {committed}")

    # --- environment ----------------------------------------------------------

    def case_environment(self) -> None:
        """Credential-like and unlisted inherited variables are recorded by name only."""
        planted = {"PLANTED_SECRET": "plantedvalue7311", "LANGPLANT": "plantedvalue7312",
                   "NUMBA_PLANTED_TOKEN": "plantedvalue7313", "NUMBA_OPT": "2"}
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-env-1", env_extra=planted)
            environment = run.record().get("environment", {})
            names, values = environment.get("inherited_names", []), environment.get("inherited_values", {})
            written = run.console + "".join(path.read_text(encoding="utf-8") for path in fixture.log_files())
            leaked = [name for name in ("PLANTED_SECRET", "LANGPLANT", "NUMBA_PLANTED_TOKEN") if planted[name] in written]
            self.check("environment/names-only", run.code == 0 and all(name in names for name in planted) and not leaked
                       and not any(name in values for name in ("PLANTED_SECRET", "LANGPLANT", "NUMBA_PLANTED_TOKEN")),
                       f"leaked values of {leaked}; recorded values {sorted(values)}")
            self.check("environment/numba-opt-value", values.get("NUMBA_OPT") == "2", f"NUMBA_OPT={values.get('NUMBA_OPT')}")

    def case_environment_fast_return(self) -> None:
        """A confirmed child that returns at once is read before it runs and its return code is recorded; a wrong
        expected value is refused, so the read never comes after the return."""
        with self.folder() as tmp:
            run = self.harness_function("fast_return_main", [str(Path(tmp) / "out1.txt"), str(DEFAULT_HASHSEED)])
            events = {event["event"]: event for event in map(json.loads, _record_payloads(run.console))}
            self.check("environment/fast-return-recorded", run.code == 0
                       and events.get("child-env", {}).get("values") == {HASHSEED_ENV: str(DEFAULT_HASHSEED)}
                       and events.get("fast-return", {}).get("return_code") == FIXTURE_RETURN_CODE, run.summary())
            run = self.harness_function("fast_return_main", [str(Path(tmp) / "out2.txt"), str(DEFAULT_HASHSEED + 1)])
            self.check("environment/fast-return-read-first", run.refused("child-env"), run.summary())

    # --- source roots and manifests ---------------------------------------------

    def case_source_root(self) -> None:
        """A base copy is recognised by location with its identities; a changed or foreign root is refused."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.make_export(fixture.roots.base, MANIFEST_BASE)
            fixture.write("data/base/tests/x.py", _FIXTURE_SCRIPT)
            fixture.write(f"data/base/tests/{TESTKIT_FILE}", "# helper copy\n")
            run = fixture.script("data/base/tests/x.py", "x-base-1")
            record = run.record()
            copies = {copy["as"] for copy in record.get("copies", [])}
            self.check("source-root/base-by-location",
                       run.code == 0 and record.get("source_root_kind") == "base"
                       and copies == {f"{TESTS_DIR}/x.py", f"{TESTS_DIR}/{TESTKIT_FILE}"}, f"{run.summary()}; copies {copies}")
            fixture.write("data/base/src/m.py", "VALUE = 2\n")
            run = fixture.script("data/base/tests/x.py", "x-base-2")
            self.check("source-root/manifest-file-changed", run.refused("manifest"), run.summary())
            fixture.write("other/pipeline.py", "# a foreign root\n")
            fixture.write("other/s.py", _FIXTURE_SCRIPT)
            run = fixture.script("other/s.py", "s-foreign-1")
            self.check("source-root/outside-admitted", run.refused("source-root"), run.summary())

    def case_manifests(self) -> None:
        """Intact and mutated roots pass their manifests, side files and tests\\ copies outside them; changed,
        unrecorded, unmanifested and pristine-checked roots fail."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.make_export(fixture.roots.base, MANIFEST_BASE)
            fixture.write(f"data/base/{TESTS_DIR}/x.py", _FIXTURE_SCRIPT)
            run = fixture.script(f"data/base/{TESTS_DIR}/x.py", "x-intact-1")
            self.check("manifests/intact-root", run.code == 0, run.summary())
            fixture.write(f"data/base/src/{CACHE_DIR}/m.f-1.py311{NUMBA_CACHE_SUFFIXES[0]}", "numba cache\n")
            run = fixture.script(f"data/base/{TESTS_DIR}/x.py", "x-sidefiles-1")
            self.check("manifests/side-files-outside", run.code == 0, run.summary())
            fixture.write("data/base/src/extra.py", "EXTRA = 1\n")
            run = fixture.script(f"data/base/{TESTS_DIR}/x.py", "x-unrecorded-1")
            self.check("manifests/unrecorded-file", run.refused("manifest") and "unrecorded src/extra.py" in run.console,
                       run.summary())
            (fixture.roots.base / "src" / "extra.py").unlink()
            export = fixture.roots.mut / "F-01" / "1"
            fixture.make_export(export, mutation_manifest_label("F-01", "1", mutated=False))
            fixture.write("data/mut/F-01/1/src/m.py", "VALUE = 3\n")
            run = fixture.harness("--manifest", mutation_manifest_label("F-01", "1", mutated=True), str(export))
            fixture.write(f"data/mut/F-01/1/{TESTS_DIR}/x.py", _FIXTURE_SCRIPT)
            run = fixture.script(f"data/mut/F-01/1/{TESTS_DIR}/x.py", "x-mutated-1")
            self.check("manifests/mutated-root", run.code == 0, run.summary())
            pristine = manifest_mismatches(fixture.roots, mutation_manifest_label("F-01", "1", mutated=False), export)
            self.check("manifests/mutated-vs-pristine", pristine == ["changed src/m.py"], str(pristine))
            fixture.write("data/base/src/m.py", "VALUE = 9\n")
            run = fixture.script(f"data/base/{TESTS_DIR}/x.py", "x-changed-1")
            self.check("manifests/changed-byte", run.refused("manifest"), run.summary())
            fixture.write("data/mut/F-01/2/pipeline.py", "# export without manifests\n")
            fixture.write("data/mut/F-01/2/a.py", _FIXTURE_SCRIPT)
            run = fixture.script("data/mut/F-01/2/a.py", "a-nomanifest-1")
            self.check("manifests/missing-manifest", run.refused("manifest"), run.summary())

    # --- guard ----------------------------------------------------------------

    def case_guard(self) -> None:
        """Absent data and real folders pass; a junction fails each predicate."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            run = fixture.harness("--guard", str(fixture.roots.data / "logs" / "x.log"))
            self.check("guard/absent-data", run.code == 0 and fixture.roots.logs.is_dir(), run.summary())
            (fixture.roots.data / "real").mkdir()
            run = fixture.harness("--guard", str(fixture.roots.data / "real" / "x"))
            self.check("guard/real-directory", run.code == 0, run.summary())
            (fixture.roots.data / "target").mkdir()
            junction = fixture.roots.data / "j"
            make_junction(junction, fixture.roots.data / "target", self.log)
            hits = reparse_components(fixture.repo, junction / "x")
            run = fixture.harness("--guard", str(junction / "x"))
            os.rmdir(junction)
            self.check("guard/attribute-predicate", hits == [junction] and run.refused("guard"), f"hits {hits}; {run.summary()}")
            inside, outside = Path(tmp) / "in", Path(tmp) / "out"
            inside.mkdir()
            outside.mkdir()
            make_junction(inside / "j", outside, self.log)
            contained = _contained(inside, inside / "j" / "x")
            os.rmdir(inside / "j")
            self.check("guard/containment-predicate", not contained, f"contained={contained}")

    def case_guard_commands(self) -> None:
        """--self-test and --select-seeds refuse a data junction before they create data\\logs through it."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            make_junction(fixture.roots.data, outside, self.log)
            try:
                for command in ("--self-test", "--select-seeds"):
                    run = fixture.harness(command)
                    written = sorted(path.name for path in outside.iterdir())
                    self.check(f"guard/{command[2:]}-before-logs", run.refused("guard") and not written,
                               f"{run.summary()}; written through the junction {written}")
            finally:
                os.rmdir(fixture.roots.data)

    # --- output roots ---------------------------------------------------------

    def case_output_roots(self) -> None:
        """A config copy resolving outside its allowed root is refused; each allowed root is admitted."""
        with self.folder() as tmp:
            folder = Path(tmp)
            context = Context(self.log, self.roots, self.label, REAL_WORKTREE, self.pinned)
            cases = (("run", "./data", self.roots.outputs, True), ("experiment", "./data/exp/05/out", self.roots.exp / "05", True),
                     ("under-ref", "./data/ref/out", self.roots.exp / "05", False), ("outside", "../outside", self.roots.outputs, False))
            for number, (name, output_root, allowed, expect_admitted) in enumerate(cases, start=1):
                config_dir = folder / f"cfg{number}"
                write_config_copy(config_dir, {"read.output_root": output_root})
                strategy = os.path.relpath(config_dir / "extrema", REAL_WORKTREE / "workflows")
                probe = context.probe(f"selftestprobe{number}_", f"workflows/{strategy}.yaml",
                                      {"read.output_root": output_root}, DEFAULT_HASHSEED)
                outputs = [path for tasks in probe.outputs.values() for path in tasks.values()]
                problems = output_root_problems(self.roots, outputs, allowed)
                self.check(f"output-roots/{name}", (not problems) == expect_admitted, str(problems[:2]))

    def case_probe_sections(self) -> None:
        """The probe derives a copy's output files from its validated sections, as the pipeline does: a
        prominence_threshold of 1 and of 1.0 resolve to the same files; a copy with a key validation refuses is still
        probed, from its own values (a probe child that raised would end the self-test with REFUSED[probe])."""
        with self.folder() as tmp:
            folder = Path(tmp)
            context = Context(self.log, self.roots, self.label, REAL_WORKTREE, self.pinned)
            outputs = {}
            for number, (name, overrides) in enumerate(PROBE_COPIES.items(), start=1):
                config_dir = folder / name
                write_config_copy(config_dir, overrides)
                strategy = os.path.relpath(config_dir / "extrema", REAL_WORKTREE / "workflows")
                probe = context.probe(f"selftestsections{number}_", f"workflows/{strategy}.yaml", overrides,
                                      DEFAULT_HASHSEED)
                outputs[name] = [str(path) for tasks in probe.outputs.values() for path in tasks.values()]
            folders = {name: sorted({Path(path).parents[2].name for path in paths}) for name, paths in outputs.items()}
            self.check("probe-sections/validated-types", outputs["int"] == outputs["float"], str(folders))
            self.check("probe-sections/invalid-copy", bool(outputs["invalid"]), str(folders["invalid"]))

    # --- rule data ------------------------------------------------------------

    def case_rule_data(self) -> None:
        """Context.rule_data in a child: without a folder or without the file, one NOTICE line and None; valid rule data
        is returned after the line with its MD5; one-outcome controls and a null, empty or blank report_title raise."""
        valid = {"exclusions": [], "controls": [["planted one", True], ["planted two", False]], "report_title": "planted"}
        raising = (("controls-all-reported", "controls", [["planted one", True]]),
                   ("controls-all-excluded", "controls", [["planted two", False]]),
                   ("title-null", "report_title", None), ("title-empty", "report_title", ""),
                   ("title-blank", "report_title", " "))
        with self.folder() as tmp:
            folder = Path(tmp)
            for name, data in (("valid", valid), *((name, {**valid, key: value}) for name, key, value in raising)):
                (folder / name).mkdir()
                (folder / name / RULE_DATA_FILE).write_text(json.dumps(data), encoding="utf-8")
            for name, args in (("no-folder", []), ("absent-file", [str(folder)])):
                run = self.harness_function("rule_data_main", args)
                notices = run.console.count(NOTICE_PREFIX)
                self.check(f"rule-data/{name}", run.code == 0 and notices == 1
                           and f"{RULE_DATA_RESULT_PREFIX}null" in run.console, f"{run.summary()}; NOTICE lines {notices}")
            path = folder / "valid" / RULE_DATA_FILE
            run = self.harness_function("rule_data_main", [str(path.parent)])
            self.check("rule-data/valid", run.code == 0 and NOTICE_PREFIX not in run.console
                       and f"rule file {path} (MD5 {file_md5(path)})" in run.console
                       and f"{RULE_DATA_RESULT_PREFIX}{json.dumps(valid, sort_keys=True)}" in run.console, run.summary())
            for name, key, _ in raising:
                run = self.harness_function("rule_data_main", [str(folder / name)])
                self.check(f"rule-data/{name}", run.code not in (0, None)
                           and f"ValueError: {folder / name / RULE_DATA_FILE} holds no valid {key}" in run.console,
                           run.summary())

    # --- timeouts and kill ----------------------------------------------------

    def case_kill_hang(self) -> None:
        """A hanging script and its grandchild: recorded timeout, both in the deadline snapshot, neither alive."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/hang.py", _HANG_FIXTURE)
            pid_file = Path(tmp) / "pids.json"
            run = fixture.script(f"{AUDIT_DIR}/hang.py", "hang-timeout-1", timeout_s=FIXTURE_HANG_TIMEOUT_S, script_args=[str(pid_file)])
            pids = _read_pids(pid_file)
            deadline = run.events("deadline")
            members = {member[0] for event in deadline for member in event.get("members", [])}
            self.check("kill/hang-assertions", bool(pids), f"pid file {pids}")
            self.check("kill/hang-deadline-snapshot", bool(pids) and {pid.pid for pid in pids} <= members,
                       f"deadline members {sorted(members)}; reported {pids}")
            self.check("kill/hang-none-alive", bool(pids) and not survivors_of(pids, 0.0), f"alive {survivors_of(pids, 0.0)}")
            self.check("kill/hang-recorded-timeout", run.refused("timeout") and run.events("end")[-1:] != []
                       and run.events("end")[-1].get("outcome") == "timeout", run.summary())

    def case_kill_walk(self) -> None:
        """Synthetic snapshots: parent-relative rule, late child of a dead member, pid reuse."""
        root = Identity(100, 10.0)
        snap = {100: (1, 10.0), 200: (100, 5.0), 201: (100, 11.0)}
        found = walk(snap, [root])
        self.check("kill/walk-older-excluded", Identity(200, 5.0) not in found, str(found))
        self.check("kill/walk-younger-included", Identity(201, 11.0) in found, str(found))
        dead = Identity(300, 20.0)
        found = walk({301: (300, 21.0)}, [dead])
        self.check("kill/walk-late-child-of-dead-member", found == [Identity(301, 21.0)], str(found))
        reused = Identity(400, 30.0)
        found = walk({400: (7, 40.0), 401: (400, 41.0)}, [reused])
        self.check("kill/walk-reused-pid-and-child-excluded", found == [], str(found))

    def case_kill_root_parent(self) -> None:
        """A root pid that the snapshot shows under another parent keeps its launch time, so no walk takes it as alive."""
        import psutil

        foreign, started_wall = psutil.Process().ppid(), time.time()
        identity = _root_identity(foreign, started_wall)
        self.check("kill/root-identity-foreign-parent", identity == Identity(foreign, started_wall)
                   and walk(snapshot(), [identity]) == [], str(identity))

    def case_kill_survivors(self) -> None:
        """The survivor check lists a live sleeping process and the refusal names it."""
        sleeper = start_sleeper(self.pinned)
        try:
            listed = survivors_of([sleeper], 1.0)
            self.check("kill/survivors-listed", listed == [sleeper], str(listed))
            run = self.harness_function("refuse_survivors_main", [str(sleeper.pid), repr(sleeper.create_time)])
            self.check("kill/survivors-refused", run.refused("survivors"), run.summary())
        finally:
            end_tree(sleeper, self.log, "selftest-sleeper")

    def case_kill_surviving_root(self) -> None:
        """A timed-out root that outlives both kills is refused as a survivor before any wait on it."""
        with self.folder() as tmp:
            run = self.harness_function("surviving_root_main", [str(Path(tmp) / "out.txt")])
            self.check("kill/surviving-root-refused", run.refused("survivors") and "TimeoutExpired" not in run.console,
                       run.summary())

    def case_kill_poll_error(self) -> None:
        """A planted error in the poll loop after both pids are reported: the finally ends the tree."""
        with self.folder() as tmp:
            hang = Path(tmp) / "hang.py"
            hang.write_text(_HANG_FIXTURE, encoding="utf-8")
            pid_file = Path(tmp) / "pids.json"
            run = self.harness_function("planted_poll_error_main", [str(hang), str(pid_file), str(Path(tmp) / "out.txt")])
            pids = _read_pids(pid_file)
            self.check("kill/poll-error-raised", run.code not in (0, None) and "PlantedPollError" in run.console, run.summary())
            self.check("kill/poll-error-none-alive", bool(pids) and not survivors_of(pids, 0.0), f"reported {pids}")

    def case_kill_orphan(self) -> None:
        """A script that returns while its sleeping child lives: found, killed, logged, refused."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/orphan.py", _ORPHAN_FIXTURE)
            pid_file = Path(tmp) / "pids.json"
            run = fixture.script(f"{AUDIT_DIR}/orphan.py", "orphan-return-1", script_args=[str(pid_file)])
            pids = _read_pids(pid_file)
            logged = "ERROR" in fixture.launch_log_text("orphan-return-1")
            self.check("kill/orphan-refused", run.refused("orphans") and logged, run.summary())
            self.check("kill/orphan-killed", bool(pids) and not survivors_of(pids, 0.0), f"reported {pids}")

    def case_kill_helper_bound(self) -> None:
        """A helper command whose grandchild survives returns within its bound, logs and raises."""
        with self.folder() as tmp:
            pid_file = Path(tmp) / "pid.txt"
            started = time.monotonic()
            run = self.harness_function("helper_bound_main", [str(pid_file), str(HELPER_TEST_BOUND_S)])
            elapsed = time.monotonic() - started
            grandchild = _sleeper_identity(pid_file)
            self.check("kill/helper-bound",
                       run.code not in (0, None) and "TimeoutExpired" in run.console and "ERROR" in run.console
                       and elapsed < HELPER_TEST_BOUND_S + HELPER_TEST_MARGIN_S,
                       f"{run.summary()}; elapsed {elapsed:.1f} s")
            if grandchild is not None:
                end_tree(grandchild, self.log, "selftest-helper-grandchild")

    def harness_function(self, function: str, args: list[str]) -> "FixtureRun":
        """Run one harness function in a child, judged by return code and output text."""
        folder = Path(tempfile.mkdtemp(dir=self.roots.exp00))
        console = folder / "console.txt"
        code = ("import sys; sys.dont_write_bytecode = True; "
                f"sys.path.insert(0, {str(HARNESS_DIR)!r}); import harness; harness.{function}(sys.argv[1:])")
        result = launch([sys._base_executable, "-B", "-c", code, *args], env=fixture_env(self.pinned),
                        cwd=REAL_WORKTREE, out_path=console, timeout_s=FIXTURE_TIMEOUT_S, log=self.log)
        text = console.read_text(encoding="utf-8")
        shutil.rmtree(folder)
        return FixtureRun(result.return_code, result.timed_out, text)

    # --- lock -----------------------------------------------------------------

    def case_lock_killed_holder(self) -> None:
        """A lock holder killed with taskkill /T /F leaves no stale lock."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/sleep.py", _SLEEP_FIXTURE)
            holder, holder_id = fixture.start_background("sleep-hold-1", f"{AUDIT_DIR}/sleep.py")
            try:
                acquired = _wait_for_text(fixture.roots.logs / "sleep-hold-1.log", "lock acquired", FIXTURE_TIMEOUT_S)
                run_helper(["taskkill", "/T", "/F", "/PID", str(holder.pid)], cwd=REAL_WORKTREE, log=self.log, check=False, strict=False)
                holder.wait(timeout=SURVIVOR_BOUND_S)
            finally:
                end_tree(holder_id, self.log, "selftest-lock-holder")
            run = fixture.script(f"{AUDIT_DIR}/h.py", "h-afterkill-1")
            self.check("lock/killed-holder-released", acquired and run.code == 0, run.summary())

    def case_lock_held(self) -> None:
        """A launch against a held lock fails with the lock's own OSError after about ten seconds."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.roots.logs.mkdir(parents=True)
            fd = os.open(fixture.roots.lock, os.O_RDWR | os.O_CREAT | os.O_BINARY)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, LOCK_BYTES)
            try:
                started = time.monotonic()
                run = fixture.script(f"{AUDIT_DIR}/h.py", "h-locked-1")
                elapsed = time.monotonic() - started
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, LOCK_BYTES)
                os.close(fd)
            text = fixture.launch_log_text("h-locked-1")
            self.check("lock/held-oserror", run.code not in (0, None) and LOCK_REFUSAL in text and "errno" in text
                       and LOCK_WAIT_MIN_S <= elapsed <= LOCK_WAIT_MAX_S, f"elapsed {elapsed:.1f} s; {run.summary()}")

    # --- resume ---------------------------------------------------------------

    def case_resume_passes(self) -> None:
        """Untouched trees, re-committed scripts, held pre-fix files and the record pass the proof."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.script(f"{AUDIT_DIR}/h.py", "h-recorded-1")
            listing = fixture.listing()
            run = fixture.harness("--resume")
            moved = sorted(set(listing) - set(fixture.listing()))
            self.check("resume/untouched", fixture.resume_passed() and not moved, f"{run.summary()}; gone {moved}")
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
            fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-1")
            fixture.git("add", f"{AUDIT_DIR}/r.py")
            fixture.commit("r unchanged")
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT + "# second version\n")
            fixture.git("add", f"{AUDIT_DIR}/r.py")
            fixture.commit("r edited")
            fixture.harness("--resume")
            self.check("resume/recommitted-script", fixture.resume_passed(), fixture.newest_resume_text()[-300:])
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT + "# held edit\n")
            fixture.script(f"{AUDIT_DIR}/h.py", "h-prefix-1", mode=MODE_PRE_FIX)
            fixture.write(FIXTURE_VERIFICATION_FILE, "record modified\n")
            fixture.harness("--resume")
            held = (fixture.repo / TESTS_DIR / "x.py").read_text(encoding="utf-8").endswith("# held edit\n")
            diff = fixture.resume_diff()
            self.check("resume/held-pre-fix-file", fixture.resume_passed() and held and f"{TESTS_DIR}/x.py" not in diff,
                       f"held={held}; diff mentions it={f'{TESTS_DIR}/x.py' in diff}")
            record = (fixture.repo / FIXTURE_VERIFICATION_FILE).read_text(encoding="utf-8")
            self.check("resume/record-untouched", record == "record modified\n" and FIXTURE_VERIFICATION_FILE not in diff, repr(record))

    def case_resume_fails(self) -> None:
        """Uncommitted changed scripts, changed run copies, changed held files and base records fail the proof."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
            fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-1")
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT + "#")
            fixture.harness("--resume")
            self.check("resume/never-committed-changed", fixture.resume_failed(), fixture.newest_resume_text()[-300:])
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.make_export(fixture.roots.data / "run1", "run1")
            fixture.write("data/run1/src/m.py", "VALUE = 5\n")
            fixture.harness("--resume")
            self.check("resume/run-copy-changed", fixture.resume_failed(), fixture.newest_resume_text()[-300:])
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT + "# held edit\n")
            fixture.script(f"{AUDIT_DIR}/h.py", "h-prefix-1", mode=MODE_PRE_FIX)
            fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT + "# held edit!\n")
            fixture.harness("--resume")
            self.check("resume/held-file-changed", fixture.resume_failed(), fixture.newest_resume_text()[-300:])
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{TESTS_DIR}/reg.py", _FIXTURE_SCRIPT)
            fixture.script(f"{TESTS_DIR}/reg.py", "reg-base-1")
            fixture.write(f"{TESTS_DIR}/reg.py", _FIXTURE_SCRIPT + "#")
            fixture.git("add", f"{TESTS_DIR}/reg.py")
            fixture.commit("regression committed with one byte changed")
            fixture.script(f"{TESTS_DIR}/reg.py", "reg-post-1")
            fixture.harness("--resume")
            self.check("resume/base-not-superseded-by-post", fixture.resume_failed(), fixture.newest_resume_text()[-300:])

    def case_resume_ignored(self) -> None:
        """Draft launches and superseded records do not count in the proof."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write("data/drafts/d.py", _FIXTURE_SCRIPT)
            fixture.script("data/drafts/d.py", "d-draft-1")
            fixture.write("data/drafts/d.py", _FIXTURE_SCRIPT + "# edited draft\n")
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
            fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-1")
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT + "# final\n")
            fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-2")
            fixture.git("add", f"{AUDIT_DIR}/r.py")
            fixture.commit("r final")
            fixture.harness("--resume")
            self.check("resume/draft-and-superseded-ignored", fixture.resume_passed(), fixture.newest_resume_text()[-300:])

    def case_resume_actions(self) -> None:
        """A passing proof restores unheld tracked edits, deletes unmanifested copies and stops on a held lock."""
        for staged in (False, True):
            with self.folder() as tmp:
                fixture = self.fixture(Path(tmp))
                fixture.script(f"{AUDIT_DIR}/h.py", "h-recorded-1")
                fixture.write(f"{TESTS_DIR}/x.py", _FIXTURE_SCRIPT + "# unheld edit\n")
                if staged:
                    fixture.git("add", f"{TESTS_DIR}/x.py")
                fixture.harness("--resume")
                restored = (fixture.repo / TESTS_DIR / "x.py").read_text(encoding="utf-8") == _FIXTURE_SCRIPT
                staged_now = fixture.git("diff", "--cached", "--name-only").strip()
                self.check(f"resume/unheld-edit-restored/{'staged' if staged else 'unstaged'}",
                           fixture.resume_passed() and restored and not staged_now and f"{TESTS_DIR}/x.py" in fixture.resume_diff(),
                           f"restored={restored}; staged={staged_now!r}")
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write("data/run1/f.txt", "unmanifested copy\n")
            fixture.write("data/mut/F-01/1/f.txt", "unmanifested export\n")
            pristine = mutation_manifest_label("F-01", "2", mutated=False)
            fixture.make_export(fixture.roots.mut / "F-01" / "2", pristine)
            fixture.harness("--resume")
            gone = not any((fixture.roots.data / rel).exists() for rel in ("run1", "mut/F-01/1", "mut/F-01/2"))
            self.check("resume/unfinished-deleted", fixture.resume_passed() and gone, f"deleted={gone}")
            stale = (fixture.roots.manifests / f"{pristine}.json").exists()
            self.check("resume/unfinished-manifest-deleted", not stale, f"pristine manifest left={stale}")
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.roots.logs.mkdir(parents=True)
            fd = os.open(fixture.roots.lock, os.O_RDWR | os.O_CREAT | os.O_BINARY)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, LOCK_BYTES)
            try:
                run = fixture.harness("--resume")
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, LOCK_BYTES)
                os.close(fd)
            self.check("resume/held-lock-stops", run.refused("lock"), run.summary())

    def case_resume_restart(self) -> None:
        """A failing proof deletes read-only outputs and leaves the verifier's record alone."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
            fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-1")
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT + "#")
            read_only = fixture.roots.exp / "05" / "readonly.txt"
            fixture.write("data/exp/05/readonly.txt", "read-only\n")
            os.chmod(read_only, stat.S_IREAD)
            fixture.write(FIXTURE_VERIFICATION_FILE, "record modified\n")
            fixture.harness("--resume")
            record = (fixture.repo / FIXTURE_VERIFICATION_FILE).read_text(encoding="utf-8")
            self.check("resume/restart-deletes-read-only", fixture.resume_failed() and not read_only.exists(),
                       f"exists={read_only.exists()}")
            self.check("resume/restart-record-untouched",
                       record == "record modified\n" and FIXTURE_VERIFICATION_FILE not in fixture.resume_diff(), repr(record))

    def case_resume_second_restart(self) -> None:
        """A label used before a first restart stays taken after a second one, which moves the first's folder into its own."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            for number in (1, 2):
                fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
                fixture.script(f"{AUDIT_DIR}/r.py", f"r-wt-{number}")
                fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT + "#")
                time.sleep(RESTART_STAMP_S)
                fixture.harness("--resume")
            nested = [path.relative_to(fixture.roots.logs).as_posix() for path in fixture.roots.logs.rglob("r-wt-1.log")]
            fixture.write(f"{AUDIT_DIR}/r.py", _FIXTURE_SCRIPT)
            run = fixture.script(f"{AUDIT_DIR}/r.py", "r-wt-1")
            self.check("resume/second-restart-label-taken", run.refused("label-exists") and len(nested) == 1
                       and nested[0].count(PRE_RESUME_PREFIX) == 2, f"{run.summary()}; earlier log at {nested}")

    # --- master tree ----------------------------------------------------------

    def case_master_tree(self) -> None:
        """The two master-tree guards: on planted captures, the last ORIGIN line confirms a run whose src came from its
        own source root, and no capture naming a master tree's src, saying not-imported or holding no ORIGIN line does;
        --resume in a fixture planted as the master tree is refused before its first write, its tracked change kept.
        Every resume/* case is the control: its fixture is not the master tree, and the resume runs."""
        with self.folder() as tmp:
            worktree, master = Path(tmp) / "worktree", Path(tmp) / "master"
            own, foreign = worktree / "src" / "__init__.py", master / "src" / "__init__.py"
            captures = (("origin-own", f"{ORIGIN_PREFIX}{foreign}\nINFO planted\n{ORIGIN_PREFIX}{own}\n", True),
                        ("origin-master", f"{ORIGIN_PREFIX}{own}\n{ORIGIN_PREFIX}{foreign}\n", False),
                        ("origin-not-imported", f"{ORIGIN_PREFIX}{ORIGIN_NOT_IMPORTED}\nTraceback (planted)\n", False),
                        ("origin-missing", "INFO planted, no origin line\n", False))
            for name, text, expected in captures:
                origin = origin_verdict(worktree, text)
                self.check(f"master-tree/{name}", origin["confirmed"] == expected, str(origin))
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            rel, planted = "ordinary.md", "planted tracked change\n"
            fixture.write(rel, planted)
            listing = fixture.listing()
            run = self.harness_function("master_resume_main", [str(fixture.repo), str(fixture.run.file)])
            kept = (fixture.repo / rel).read_text(encoding="utf-8") == planted
            self.check("master-tree/resume-refused",
                       run.refused("master-tree") and kept and fixture.listing() == listing,
                       f"{run.summary()}; change kept={kept}")

    def case_origin_child(self) -> None:
        """ORIGIN_CHILD on two planted source roots: a pipeline.py that imports src and returns 3 runs as __main__ with
        the given arguments and reports its root's src; one that cannot compile reports not-imported, its SyntaxError
        the failure."""
        with self.folder() as tmp:
            for name, pipeline, code, confirmed in (
                    ("imported", "import sys\nimport src\nprint('NAME', __name__, sys.argv[1:])\nraise SystemExit(3)\n",
                     FIXTURE_RETURN_CODE, True),
                    ("not-imported", "def broken(:\n", 1, False)):
                root = Path(tmp) / name
                (root / "src").mkdir(parents=True)
                (root / "src" / "__init__.py").write_text("", encoding="utf-8")
                (root / PIPELINE_FILE).write_text(pipeline, encoding="utf-8")
                console = Path(tmp) / f"{name}.txt"
                result = launch([sys._base_executable, "-B", "-c", ORIGIN_CHILD, str(root), json.dumps(["--planted"])],
                                env=fixture_env(self.pinned), cwd=REAL_WORKTREE, out_path=console,
                                timeout_s=FIXTURE_TIMEOUT_S, log=self.log)
                text = console.read_text(encoding="utf-8")
                origin = origin_verdict(root, text)
                ran = "NAME __main__ ['--planted']" in text if confirmed else "SyntaxError" in text
                self.check(f"origin-child/{name}", result.return_code == code and origin["confirmed"] == confirmed and ran,
                           f"return code {result.return_code}; {origin}; ran={ran}")

    # --- preamble -------------------------------------------------------------

    def case_preamble(self) -> None:
        """The marker, the UTF-8 assertion and the first-statement bytecode flag."""
        with self.folder() as tmp:
            folder = Path(tmp)
            script = folder / "pre.py"
            script.write_text(_preamble_fixture(imports_src=False, flag=True, module="freshone"), encoding="utf-8")
            (folder / "freshone.py").write_text("VALUE = 1\n", encoding="utf-8")
            utf8_script = folder / "preutf.py"
            utf8_script.write_text(_preamble_fixture(imports_src=True, flag=True, module=None), encoding="utf-8")
            planted = self.preamble_env(folder)
            run = self.direct(folder, [str(script)], {k: v for k, v in planted.items() if k != MARKER_ENV})
            self.check("preamble/marker", run.refused("marker"), run.summary())
            run = self.direct(folder, [str(script)], {k: v for k, v in planted.items() if k != RUN_FILE_ENV})
            self.check("preamble/run-file", run.refused("run-file"), run.summary())
            run = self.direct(folder, ["-X", "utf8=0", str(utf8_script)], {k: v for k, v in planted.items() if k != UTF8_ENV})
            self.check("preamble/utf8", run.refused("utf8"), run.summary())
            run = self.direct(folder, [str(script)], {k: v for k, v in planted.items() if k != BYTECODE_ENV}, bytecode=False)
            pycs = sorted(path.name for path in folder.rglob("*.pyc"))
            self.check("preamble/no-pyc", run.code == 0 and not pycs, f"{run.summary()}; pyc {pycs}")

    def case_preamble_control(self) -> None:
        """Without the first-statement flag a fresh import leaves a .pyc (the check can bite)."""
        with self.folder() as tmp:
            folder = Path(tmp)
            script = folder / "control.py"
            script.write_text("import freshtwo\n", encoding="utf-8")
            (folder / "freshtwo.py").write_text("VALUE = 2\n", encoding="utf-8")
            planted = {k: v for k, v in self.preamble_env(folder).items() if k != BYTECODE_ENV}
            run = self.direct(folder, [str(script)], planted, bytecode=False)
            pycs = sorted(path.name for path in folder.rglob("*.pyc"))
            self.check("preamble/control-leaves-pyc", run.code == 0 and bool(pycs), f"{run.summary()}; pyc {pycs}")

    def case_preamble_venv(self) -> None:
        """The base executable without __PYVENV_LAUNCHER__ is refused by the venv assertion."""
        with self.folder() as tmp:
            folder = Path(tmp)
            script = folder / "venv.py"
            script.write_text(_preamble_fixture(imports_src=False, flag=True, module=None), encoding="utf-8")
            planted = {k: v for k, v in self.preamble_env(folder).items() if k != VENV_LAUNCHER_ENV}
            run = self.direct(folder, [str(script)], planted)
            self.check("preamble/venv", run.refused("venv"), run.summary())

    def preamble_env(self, folder: Path) -> dict:
        """Planted environment of a direct preamble start: what a harness command would set."""
        env = fixture_env(self.pinned)
        env.update(_markers(self.roots, f"pre-direct-1", REAL_WORKTREE, self.pinned))
        env.update({WORKTREE_ENV: str(folder), RUN_FILE_ENV: str(plant_run_file(folder, folder).file), UTF8_ENV: "1",
                    BYTECODE_ENV: "1"})
        return env

    def direct(self, folder: Path, args: list[str], env: dict, *, bytecode: bool = True) -> "FixtureRun":
        """A direct start of a fixture script (no harness command), judged by return code and output."""
        console = folder / f"direct{len(list(folder.glob('direct*.txt'))) + 1}.txt"
        argv = [sys._base_executable] + (["-B"] if bytecode else []) + args
        environ = {name: value for name, value in os.environ.items()
                   if name not in (BYTECODE_ENV, UTF8_ENV, MARKER_ENV, RUN_FILE_ENV)}
        environ.update(env)
        result = launch(argv, env=environ, cwd=REAL_WORKTREE, out_path=console, timeout_s=FIXTURE_TIMEOUT_S, log=self.log)
        return FixtureRun(result.return_code, result.timed_out, console.read_text(encoding="utf-8"))

    # --- testkit and mutation (step 4) ----------------------------------------

    def case_testkit_hang(self) -> None:
        """tests/testkit.py's own launcher on the hang fixture; the case counts only for a committed testkit
        with the blob recorded here."""
        testkit = REAL_WORKTREE / TESTS_DIR / TESTKIT_FILE
        if not testkit.is_file():
            self.log.info("NOT APPLICABLE testkit/hang: %s absent", testkit)
            return
        rel = f"{TESTS_DIR}/{TESTKIT_FILE}"
        blob = _git_text(self.roots, self.log, "hash-object", "--", rel)
        listed = _git_text(self.roots, self.log, "ls-tree", "HEAD", "--", rel)
        with self.folder() as tmp:
            folder = Path(tmp)
            (folder / "hang.py").write_text(_TESTKIT_HANG_FIXTURE, encoding="utf-8")
            pid_file, console = folder / "pids.json", folder / "driver.txt"
            env, _ = child_environment(self.pinned, hashseed=DEFAULT_HASHSEED)
            launch([sys._base_executable, "-B", "-c", _TESTKIT_DRIVER, str(testkit.parent), str(folder / "hang.py"),
                    str(pid_file), str(folder / "hang.log"), str(FIXTURE_HANG_TIMEOUT_S)],
                   env=env, cwd=REAL_WORKTREE, out_path=console, timeout_s=FIXTURE_TIMEOUT_S, log=self.log)
            text = console.read_text(encoding="utf-8")
            pids = _read_pids(pid_file)
            report = json.loads(_prefixed_line(text, TESTKIT_RESULT_PREFIX))
            self.check("testkit/hang-assertions", bool(pids), f"pid file {pids}")
            self.check("testkit/hang-deadline-snapshot", bool(pids) and "deadline kill" in text
                       and all(f"pid={pid.pid}," in text for pid in pids), f"reported {pids}")
            self.check("testkit/hang-none-alive", bool(pids) and not survivors_of(pids, 0.0), f"alive {survivors_of(pids, 0.0)}")
            self.check("testkit/hang-recorded-timeout", report == {"return_code": None, "timed_out": True}, str(report))
        committed = listed.split()[2] if listed else None
        self.log.info("testkit/hang: tested blob %s, committed blob %s: the case counts %s", blob, committed, blob == committed)

    def case_mutate(self) -> None:
        """The mutation function on an export of the fixture's HEAD: one occurrence admitted with both manifests;
        zero or two occurrences refused; a change outside the declared files refused; a path outside the export refused
        before anything outside it is read or written."""
        with self.folder() as tmp:
            fixture = self.fixture(Path(tmp))
            fixture.write("src/m.py", "VALUE = 1\nOTHER = 2\nREPEAT = 0\nREPEAT = 0\n")
            fixture.write("src/n.py", "NAME = 1\n")
            fixture.write("workflows/w.yaml", "key: 1\n")
            fixture.commit("mutable files")
            # from an export root data/mut/F-99/<n> up to the fixture's own tracked src/n.py
            escape = "/".join([".."] * (len(fixture.roots.mut.relative_to(fixture.repo).parts) + 2) + ["src", "n.py"])
            cases = (("admitted", ["src/m.py"], [("src/m.py", "VALUE = 1", "VALUE = 3")], None),
                     ("zero-occurrences", ["src/m.py"], [("src/m.py", "ABSENT = 1", "ABSENT = 2")], "mutate"),
                     ("two-occurrences", ["src/m.py"], [("src/m.py", "REPEAT = 0", "REPEAT = 1")], "mutate"),
                     ("undeclared-file", ["src/m.py"], [("src/m.py", "OTHER = 2", "OTHER = 4"), ("src/n.py", "NAME = 1", "NAME = 2")],
                      "mutate"),
                     ("outside-export", ["src/n.py"], [(escape, "NAME = 1", "NAME = 5")], "mutate"))
            for number, (name, files, changes, refusal) in enumerate(cases, start=1):
                spec = Path(tmp) / f"spec{number}.json"
                spec.write_text(json.dumps({"number": number, "files": files, "changes": [
                    {"path": path, "old": old, "new": new} for path, old, new in changes]}), encoding="utf-8")
                run = fixture.harness("--mutate", "F-99", "HEAD", str(spec))
                manifests = [fixture.roots.manifests / f"{mutation_manifest_label('F-99', str(number), mutated=flag)}.json"
                             for flag in (False, True)]
                if refusal is None:
                    self.check(f"mutate/{name}", run.code == 0 and all(path.is_file() for path in manifests)
                               and manifest_changes(*manifests) == files, run.summary())
                else:
                    self.check(f"mutate/{name}", run.refused(refusal), run.summary())
            outside = (fixture.repo / "src" / "n.py").read_text(encoding="utf-8")
            self.check("mutate/outside-export-untouched", outside == "NAME = 1\n", repr(outside))


@dataclasses.dataclass(frozen=True)
class FixtureRun:
    """A fixture launch judged by its return code (None after a timeout) and its output."""

    code: Optional[int]
    timed_out: bool
    console: str
    records: tuple = ()

    def refused(self, code: str) -> bool:
        """A refusal counts only with the reason the case expects."""
        return self.code not in (0, None) and f"REFUSED[{code}]" in self.console

    def summary(self) -> str:
        """Return code and the decisive output lines."""
        lines = [line for line in self.console.splitlines() if "REFUSED" in line or "Error" in line or "error" in line]
        return f"return code {self.code}, timed out {self.timed_out}: {' | '.join(lines[-3:])[:400]}"

    def record(self) -> dict:
        """The script launch's evidence record."""
        return next((json.loads(text) for text in self.records if "identities" in json.loads(text)), {})

    def events(self, name: str) -> list[dict]:
        """Logged events of the launch with the given name."""
        return [event for event in (json.loads(text) for text in self.records) if event.get("event") == name]


class Fixture:
    """A fixture repository and the launches made against it."""

    def __init__(self, suite: SelfTest, folder: Path, repo: Path) -> None:
        self.suite = suite
        self.folder = folder
        self.repo = repo
        self.run = plant_run_file(folder, repo)
        self.roots = Roots(repo, self.run)
        self._console = itertools.count(1)

    def git(self, *args: str) -> str:
        """A git command in the fixture."""
        _, out = run_helper(["git", *FIXTURE_IDENTITY, *args], cwd=self.repo if self.repo.exists() else self.folder,
                            log=self.suite.log)
        return out

    def write(self, rel: str, text: str) -> None:
        """Write a fixture file (LF line endings), creating its folders."""
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))

    def commit(self, message: str) -> None:
        """Commit everything in the fixture."""
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def reset(self) -> None:
        """Back to the committed tree: unstage, restore tracked files, drop staged new files."""
        self.git("reset", "-q", "--hard", "HEAD")

    def make_export(self, root: Path, manifest: str) -> None:
        """A minimal export (pipeline.py, src) with its manifest written through --manifest."""
        rel = root.relative_to(self.repo).as_posix()
        self.write(f"{rel}/{PIPELINE_FILE}", "# exported placeholder pipeline\n")
        self.write(f"{rel}/src/m.py", "VALUE = 1\n")
        self.harness("--manifest", manifest, str(root))

    def harness(self, *args: str, env_extra: Optional[dict] = None, timeout_s: float = FIXTURE_TIMEOUT_S,
                run_args: Optional[list[str]] = None) -> FixtureRun:
        """A harness command under --worktree <fixture> and --run <its planted run file> (run_args replace that
        option); its console goes to a file outside the repo."""
        console = self.folder / f"console{next(self._console)}.txt"
        env = fixture_env(self.suite.pinned)
        env.update(env_extra or {})
        run_option = ["--run", str(self.run.file)] if run_args is None else run_args
        result = launch([sys._base_executable, "-B", str(HARNESS_FILE), "--worktree", str(self.repo), *run_option, *args],
                        env=env, cwd=REAL_WORKTREE, out_path=console, timeout_s=timeout_s, log=self.suite.log)
        text = console.read_text(encoding="utf-8")
        label = option_value(list(args), "--label")
        return FixtureRun(result.return_code, result.timed_out, text, tuple(self.records(label)) if label else ())

    def script(self, rel: str, label: str, *, mode: str = MODE_EVIDENCE, env_extra: Optional[dict] = None,
               timeout_s: float = 60.0, script_args: Optional[list[str]] = None,
               run_args: Optional[list[str]] = None) -> FixtureRun:
        """A script launch in the fixture."""
        return self.harness("--label", label, "--timeout", str(timeout_s), "--mode", mode,
                            str(self.repo / rel), *(script_args or []), env_extra=env_extra, run_args=run_args)

    def start_background(self, label: str, rel: str) -> tuple[subprocess.Popen, Identity]:
        """A script launch left running in the background (lock cases)."""
        console = self.folder / f"console{next(self._console)}.txt"
        sink = open(console, "xb")
        proc = subprocess.Popen([sys._base_executable, "-B", str(HARNESS_FILE), "--worktree", str(self.repo),
                                 "--run", str(self.run.file), "--label", label, "--timeout", "600",
                                 str(self.repo / rel)],
                                env=fixture_env(self.suite.pinned), cwd=REAL_WORKTREE, stdin=subprocess.DEVNULL,
                                stdout=sink, stderr=subprocess.STDOUT)
        sink.close()
        return proc, _root_identity(proc.pid, time.time() - 1.0)

    def records(self, label: str) -> list[str]:
        """RECORD payloads of one fixture launch log."""
        path = self.roots.logs / f"{label}.log"
        return [] if not path.is_file() else _record_payloads(path.read_text(encoding="utf-8"))

    def newest_record(self, point: str) -> dict:
        """The newest record of a non-script command."""
        logs = sorted(self.roots.logs.glob(f"{HARNESS_STEM}-{point}-*.log"), key=lambda path: path.stat().st_mtime_ns)
        payloads = _record_payloads(logs[-1].read_text(encoding="utf-8")) if logs else []
        return next((json.loads(text) for text in payloads if "tree_status" in json.loads(text)), {})

    def newest_resume_text(self) -> str:
        """Text of the newest resume launch log."""
        logs = sorted(self.roots.logs.glob(f"{HARNESS_STEM}-{COMMAND_RESUME}-*.log"), key=lambda path: path.stat().st_mtime_ns)
        return logs[-1].read_text(encoding="utf-8") if logs else ""

    def resume_passed(self) -> bool:
        """The newest resume proved the recorded artefacts valid."""
        return "RESUME proof passed" in self.newest_resume_text()

    def resume_failed(self) -> bool:
        """The newest resume's proof failed and it restarted."""
        text = self.newest_resume_text()
        return "RESUME proof failed" in text and "RESUME restart at step 1" in text

    def resume_diff(self) -> str:
        """Text of the newest resume diff, empty when none was written."""
        diffs = sorted(self.roots.logs.glob(f"{RESUME_PREFIX}*.diff"), key=lambda path: path.stat().st_mtime_ns)
        return diffs[-1].read_text(encoding="utf-8") if diffs else ""

    def launch_log_text(self, label: str) -> str:
        """Text of one fixture launch log."""
        path = self.roots.logs / f"{label}.log"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def log_files(self) -> list[Path]:
        """Every file the fixture's launches wrote under its logs."""
        return sorted(path for path in self.roots.logs.rglob("*") if path.is_file()) if self.roots.logs.is_dir() else []

    def listing(self) -> list[str]:
        """Every path in the fixture repository except .git and the logs."""
        return sorted(path.relative_to(self.repo).as_posix() for path in self.repo.rglob("*")
                      if ".git" not in path.relative_to(self.repo).parts and not _contained(self.roots.logs, path))


def plant_run_file(folder: Path, repo: Path) -> RunIdentity:
    """The self-test's planted run for a fixture repository (or a planted worktree): its file, written into folder
    beside the repository, names the planted branches, the planted record and a report inside the repository."""
    run = RunIdentity(folder / FIXTURE_RUN_NAME, FIXTURE_BRANCHES, FIXTURE_VERIFICATION_FILE, repo / FIXTURE_REPORT_NAME)
    document = {"branches": list(run.branches), "verification_file": run.verification_file,
                "report_path": str(run.report_path)}
    run.file.write_text(json.dumps(document), encoding="utf-8")
    return run


def fixture_env(pinned: Path) -> dict:
    """Environment of a fixture harness start: the session's plus the pinned venv launch and UTF-8."""
    env = dict(os.environ)
    env.update({VENV_LAUNCHER_ENV: str(pinned), BYTECODE_ENV: "1", UTF8_ENV: "1"})
    return env


def write_config_copy(folder: Path, overrides: dict) -> None:
    """Copies of workflows\\extrema.yaml and dev.yaml with dotted-key overrides of the first."""
    import yaml

    folder.mkdir(parents=True)
    strategy = yaml.safe_load((REAL_WORKTREE / "workflows" / "extrema.yaml").read_text(encoding="utf-8"))
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = strategy
        for part in parents:
            node = node[part]
        node[leaf] = value
    (folder / "extrema.yaml").write_text(yaml.safe_dump(strategy, sort_keys=False), encoding="utf-8")
    shutil.copyfile(REAL_WORKTREE / "workflows" / "dev.yaml", folder / "dev.yaml")


def make_junction(link: Path, target: Path, log: logging.Logger) -> None:
    """A directory junction (no elevation needed), created with cmd /c mklink /J in list form."""
    run_helper(["cmd", "/c", "mklink", "/J", str(link), str(target)], cwd=REAL_WORKTREE, log=log, strict=False)


def start_sleeper(pinned: Path) -> Identity:
    """A sleeping process the self-test owns, by pid and creation time."""
    proc = subprocess.Popen([sys._base_executable, "-B", "-c", "import time; time.sleep(600)"],
                            env=fixture_env(pinned), cwd=REAL_WORKTREE, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return _root_identity(proc.pid, time.time() - 1.0)


def fast_return_main(argv: list[str]) -> None:
    """Child side of the fast-return case: a child that returns at once, launched with the seed value to confirm."""
    out_path, expected = Path(argv[0]), argv[1]
    log = _stdout_logger(HARNESS_STEM)
    env, _ = child_environment(Path(sys.executable), hashseed=DEFAULT_HASHSEED)
    result = launch([sys._base_executable, "-B", "-c", f"raise SystemExit({FIXTURE_RETURN_CODE})"], env=env,
                    cwd=REAL_WORKTREE, out_path=out_path, timeout_s=FIXTURE_TIMEOUT_S, log=log,
                    confirm_env={HASHSEED_ENV: expected})
    _log_event(log, "fast-return", return_code=result.return_code, timed_out=result.timed_out)


def rule_data_main(argv: list[str]) -> None:
    """Child side of the rule-data case: Context.rule_data on the folder argv[0] (no folder without an argument), its
    result on one RULE_DATA_RESULT line."""
    context = Context(_stdout_logger(HARNESS_STEM), Roots(REAL_WORKTREE, None), HARNESS_STEM, REAL_WORKTREE,
                      Path(sys.executable))
    data = context.rule_data(Path(argv[0]) if argv else None)
    context.log.info("%s%s", RULE_DATA_RESULT_PREFIX, json.dumps(data, sort_keys=True))


def refuse_survivors_main(argv: list[str]) -> None:
    """Child side of the survivor case: the survivor check refuses the given live member."""
    log = _stdout_logger(HARNESS_STEM)
    survivors = survivors_of([Identity(int(argv[0]), float(argv[1]))], 1.0)
    if survivors:
        refuse(log, "survivors", f"alive after the bound: {survivors}")


def surviving_root_main(argv: list[str]) -> None:
    """Child side of the surviving-root case: with the kill planted to leave every member alive, a timed-out launch
    refuses its survivors; the real kill then ends what this child started."""
    global _taskkill, SURVIVOR_BOUND_S
    log = _stdout_logger(HARNESS_STEM)
    env, _ = child_environment(Path(sys.executable), hashseed=DEFAULT_HASHSEED)
    real_taskkill, real_bound = _taskkill, SURVIVOR_BOUND_S

    def planted_no_kill(members: list[Identity], log: logging.Logger) -> int:
        return 0

    _taskkill, SURVIVOR_BOUND_S = planted_no_kill, PLANTED_SURVIVOR_BOUND_S
    try:
        launch([sys._base_executable, "-B", "-c", "import time; time.sleep(600)"], env=env, cwd=REAL_WORKTREE,
               out_path=Path(argv[0]), timeout_s=PLANTED_SURVIVOR_BOUND_S, log=log)
    finally:
        _taskkill, SURVIVOR_BOUND_S = real_taskkill, real_bound
        own = Identity(os.getpid(), _own_create_time())
        started = [member for member in walk(snapshot(), [own]) if member != own]
        if started:
            _taskkill(started, log)
            survivors_of(started, SURVIVOR_BOUND_S)


class PlantedPollError(RuntimeError):
    """The planted error of the poll-loop case."""


def planted_poll_error_main(argv: list[str]) -> None:
    """Child side of the poll-loop case: raise once the hang fixture reported two live pids."""
    hang, pid_file, out_path = (Path(arg) for arg in argv)
    log = _stdout_logger(HARNESS_STEM)
    pinned = Path(sys.executable)

    def plant() -> None:
        pids = _read_pids(pid_file)
        if pids and len(survivors_of(pids, 0.0)) == len(pids):
            raise PlantedPollError(f"planted error with live {pids}")

    env, _ = child_environment(pinned, hashseed=DEFAULT_HASHSEED,
                               markers=_markers(Roots(REAL_WORKTREE, None), "hang-poll-1", REAL_WORKTREE, pinned))
    launch([sys._base_executable, "-B", str(hang), str(pid_file)], env=env, cwd=REAL_WORKTREE, out_path=out_path,
           timeout_s=FIXTURE_TIMEOUT_S, log=log, on_poll=plant)


def helper_bound_main(argv: list[str]) -> None:
    """Child side of the helper case: cmd /c starts a sleeping grandchild that survives cmd's kill."""
    pid_file, bound = Path(argv[0]), float(argv[1])
    log = _stdout_logger(HARNESS_STEM)
    code = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(600)"
    run_helper(["cmd", "/c", sys._base_executable, "-B", "-c", code], cwd=REAL_WORKTREE, log=log, bound_s=bound)


def master_resume_main(argv: list[str]) -> None:
    """Child side of the master-tree case: harness.py --worktree argv[0] --run argv[1] --resume, with that fixture
    planted as the master tree, the one tree the resume rule refuses."""
    global _master_tree
    fixture, run_file = argv

    def planted_master_tree() -> str:
        return fixture

    _master_tree = planted_master_tree
    sys.argv = [str(HARNESS_FILE), "--worktree", fixture, "--run", run_file, "--resume"]
    main()


def _sleeper_identity(pid_file: Path) -> Optional[Identity]:
    """Identity of the process whose pid a fixture wrote, while it lives."""
    if not pid_file.is_file():
        return None
    pid = int(pid_file.read_text(encoding="utf-8"))
    found = snapshot().get(pid)
    return Identity(pid, found[1]) if found is not None else None


def _read_pids(pid_file: Path) -> list[Identity]:
    """Identities a hang or orphan fixture reported, empty while the file is absent."""
    if not pid_file.is_file() or not pid_file.read_text(encoding="utf-8").strip():
        return []
    reported = json.loads(pid_file.read_text(encoding="utf-8"))
    return [Identity(int(pid), float(created)) for pid, created in reported.values()]


def _wait_for_text(path: Path, text: str, bound_s: float) -> bool:
    """Poll a file until it contains the text or the bound elapses."""
    deadline = time.monotonic() + bound_s
    while time.monotonic() < deadline:
        if path.is_file() and text in path.read_text(encoding="utf-8"):
            return True
        time.sleep(POLL_INTERVAL_S)
    return False


def _preamble_fixture(*, imports_src: bool, flag: bool, module: Optional[str]) -> str:
    """A fixture script calling the shared preamble (the harness path is literal: fixtures are untracked)."""
    lines = ["import sys"]
    if flag:
        lines.append("sys.dont_write_bytecode = True")
    lines += [f"sys.path.insert(0, {str(HARNESS_DIR)!r})", "import harness",
              f"harness.preamble(__file__, imports_src={imports_src})"]
    if module:
        lines.append(f"import {module}")
    return "\n".join(lines) + "\n"


_FIXTURE_SCRIPT = """import sys
sys.dont_write_bytecode = True
print("fixture script 1 ran")
"""

_SLEEP_FIXTURE = """import sys
sys.dont_write_bytecode = True
import time
time.sleep(600)
"""

_HANG_FIXTURE = """import sys
sys.dont_write_bytecode = True
import json
import os
import subprocess
import time
import psutil
pinned = os.environ["PYTHIA_AUDIT_INTERPRETER"]
assert os.path.samefile(sys.executable, pinned), sys.executable
assert sys.prefix != sys.base_prefix, sys.prefix
env = dict(os.environ)
env["__PYVENV_LAUNCHER__"] = sys.executable
child = subprocess.Popen([sys._base_executable, "-B", "-c", "import time; time.sleep(600)"], env=env)
me, grandchild = psutil.Process(), psutil.Process(child.pid)
report = {"self": [me.pid, me.create_time()], "grandchild": [grandchild.pid, grandchild.create_time()]}
with open(sys.argv[1], "w", encoding="utf-8") as sink:
    json.dump(report, sink)
time.sleep(600)
"""

# the hang fixture for the testkit's launcher: the same assertions, the pinned interpreter as its second argument
_TESTKIT_HANG_FIXTURE = _HANG_FIXTURE.replace('pinned = os.environ["PYTHIA_AUDIT_INTERPRETER"]', "pinned = sys.argv[2]")

_TESTKIT_DRIVER = """import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import json
import logging
from pathlib import Path
import testkit
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("testkitdriver")
argv = [sys._base_executable, "-B", sys.argv[2], sys.argv[3], sys.executable]
code, timed_out = testkit.launch(argv, env=testkit.child_environment(0, False), cwd=Path.cwd(), out_path=Path(sys.argv[4]),
                                 timeout_s=float(sys.argv[5]), log=log)
log.info("TESTKIT_RESULT %s", json.dumps({"return_code": code, "timed_out": timed_out}))
"""

_ORPHAN_FIXTURE = """import sys
sys.dont_write_bytecode = True
import json
import os
import subprocess
import psutil
env = dict(os.environ)
env["__PYVENV_LAUNCHER__"] = sys.executable
child = subprocess.Popen([sys._base_executable, "-B", "-c", "import time; time.sleep(600)"], env=env)
orphan = psutil.Process(child.pid)
with open(sys.argv[1], "w", encoding="utf-8") as sink:
    json.dump({"orphan": [orphan.pid, orphan.create_time()]}, sink)
"""


# =============================================================================
# 9. RESUME HELPERS
# =============================================================================


@dataclasses.dataclass(frozen=True)
class RecordedRun:
    """A recorded run of the resume rule: its label and the identities it recorded."""

    label: str
    stem: str
    point: str
    seq: int
    identities: dict
    copies: tuple


def recorded_runs(roots: Roots, log: logging.Logger, *, exclude_label: str) -> list[RecordedRun]:
    """Logged commands except resume and guard, draft launches and superseded records."""
    runs = []
    for path in sorted(roots.logs.iterdir()) if roots.logs.is_dir() else []:
        if not path.is_file() or not LAUNCH_LOG_PATTERN.match(path.name):
            continue
        label = path.name[:-len(".log")]
        match = LABEL_PATTERN.match(label)
        payloads = [json.loads(text) for text in _record_payloads(path.read_text(encoding="utf-8"))]
        record = next((payload for payload in payloads if "identities" in payload), None)
        if label == exclude_label or match is None or record is None or record["command"] in UNRECORDED_COMMANDS:
            continue
        if str(record.get("script", "")).startswith(f"data/{DRAFTS_DIR}/"):
            continue
        runs.append(RecordedRun(label, match["stem"], match["point"], int(match["seq"]),
                                record["identities"], tuple(tuple(sorted(copy.items())) for copy in record.get("copies", []))))
    latest = {}
    for run in runs:
        key = (run.stem, run.point)
        if key not in latest or run.seq > latest[key].seq:
            latest[key] = run
    log.info("recorded runs: %d logged, %d after drafts and superseded records", len(runs), len(latest))
    return sorted(latest.values(), key=lambda run: run.label)


def proof_failures(roots: Roots, log: logging.Logger, runs: list[RecordedRun]) -> list[str]:
    """Runs that are no longer citable and run copies that no longer match their manifests."""
    commits = _git_text(roots, log, "rev-list", "HEAD").split()  # the gate holds HEAD to an accepted branch
    trees = {commit: _tree_blobs(roots, log, commit) for commit in commits}
    held_anywhere = {(path, blob) for blobs in trees.values() for path, blob in blobs.items()}
    working = identities(roots, log)
    failures = [run.label for run in runs if not _citable(roots, log, run, trees, held_anywhere, working)]
    for manifest in sorted(roots.manifests.glob("*.json")) if roots.manifests.is_dir() else []:
        label = manifest.stem
        if label.endswith("_pristine"):
            continue
        root = Path(json.loads(manifest.read_text(encoding="utf-8"))["root"])
        mismatches = manifest_mismatches(roots, label, root)
        if mismatches:
            failures.append(f"copy {label}: {mismatches[:3]}")
    return failures


def _citable(roots: Roots, log: logging.Logger, run: RecordedRun, trees: dict, held_anywhere: set, working: dict) -> bool:
    """One commit holds every recorded identity, except versions no commit holds and the working copy keeps."""
    items = [(path, blob, working.get(path)) for path, blob in sorted(run.identities.items())]
    for copy in run.copies:
        fields = dict(copy)
        file = roots.worktree / fields["file"]
        current = copy_blob(roots, log, file, fields["as"]) if file.is_file() else None
        items.append((fields["as"], fields["blob"], current))
    return any(all(blobs.get(path) == blob or ((path, blob) not in held_anywhere and current == blob)
                   for path, blob, current in items) for blobs in trees.values())


def _tree_blobs(roots: Roots, log: logging.Logger, commit: str) -> dict[str, str]:
    """path -> blob of audit\\ and tests\\ at one commit."""
    found = {}
    for line in _git_z(roots, log, "ls-tree", "-r", "-z", commit, "--", AUDIT_DIR, TESTS_DIR):
        meta, path = line.split("\t", 1)
        found[path] = meta.split()[2]
    return found


def held_paths(roots: Roots, log: logging.Logger, runs: list[RecordedRun]) -> set[str]:
    """Paths whose working copy equals the identity some recorded run recorded for them."""
    working = identities(roots, log)
    return {path for run in runs for path, blob in run.identities.items() if blob is not None and working.get(path) == blob}


def _save_and_restore(roots: Roots, log: logging.Logger, stamp: str, *, keep: set[str]) -> None:
    """Write tracked modifications (staged or not) to the resume diff, unstage and restore them."""
    paths = sorted({path for _, path in porcelain_records(roots, log)
                    if path != roots.run.verification_file and path not in keep})
    if not paths:
        return
    _, unstaged = run_helper(["git", "diff", "--", *paths], cwd=roots.worktree, log=log)
    _, staged = run_helper(["git", "diff", "--cached", "--", *paths], cwd=roots.worktree, log=log)
    target = roots.logs / f"{RESUME_PREFIX}{stamp}.diff"
    guard_or_refuse(roots, [target], log)
    with open(target, "x", encoding="utf-8") as sink:
        sink.write(unstaged + staged)
    run_helper(["git", "reset", "-q", "--", *paths], cwd=roots.worktree, log=log)
    in_head = _git_z(roots, log, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *paths)
    if in_head:
        run_helper(["git", "checkout", "--", *in_head], cwd=roots.worktree, log=log)
    log.info("saved and restored %s into %s", paths, target.name)


def _delete_unfinished(roots: Roots, log: logging.Logger) -> None:
    """Labelled copies, REF or BASE without a manifest, and exports without both manifests, with the one manifest such
    an export has (its pristine manifest would stop the export's next --mutate)."""
    unfinished = [path for path in sorted(roots.data.glob(f"{RUN_COPY_PREFIX}*"))
                  if path.is_dir() and not (roots.manifests / f"{path.name}.json").is_file()]
    for root, label in ((roots.ref, MANIFEST_REF), (roots.base, MANIFEST_BASE)):
        if root.is_dir() and not (roots.manifests / f"{label}.json").is_file():
            unfinished.append(root)
    stale = []
    for export in sorted(roots.mut.glob("F-*/*")) if roots.mut.is_dir() else []:
        manifests = [roots.manifests / f"{mutation_manifest_label(export.parent.name, export.name, mutated=state)}.json"
                     for state in (False, True)]
        if not all(path.is_file() for path in manifests):
            unfinished.append(export)
            stale += [path for path in manifests if path.is_file()]
    for path in unfinished:
        remove_tree(roots, log, path)
    for path in stale:
        guard_or_refuse(roots, [path], log)
        path.unlink()
        log.info("removed %s, the manifest of an unfinished export", path.name)


def _move_untracked(roots: Roots, log: logging.Logger, target: Path) -> None:
    """Move every untracked, non-ignored file into data\\logs\\resume_<timestamp>\\."""
    for rel in _git_z(roots, log, "ls-files", "-z", "--others", "--exclude-standard"):
        destination = target / rel
        guard_or_refuse(roots, [destination], log)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(roots.worktree / rel), str(destination))
        log.info("moved untracked %s", rel)


def _move_earlier_logs(roots: Roots, log: logging.Logger, pre: Path, *, keep_label: str, stamp: str) -> None:
    """Move every earlier file and folder of the logs except the lock and this resume's own files."""
    kept = {roots.lock.name, f"{keep_label}.log", f"{RESUME_PREFIX}{stamp}.diff", f"{RESUME_PREFIX}{stamp}", pre.name}
    guard_or_refuse(roots, [pre], log)
    pre.mkdir(parents=True, exist_ok=True)
    for path in sorted(roots.logs.iterdir()):
        if path.name not in kept:
            shutil.move(str(path), str(pre / path.name))


def _move_report_and_drafts(roots: Roots, log: logging.Logger, pre: Path) -> None:
    """Move the report draft and data\\drafts into the pre-resume folder."""
    report = roots.run.report_path
    if report.is_file():
        shutil.move(str(report), str(pre / report.name))
        log.info("moved the report draft %s", report)
    if roots.drafts.is_dir():
        shutil.move(str(roots.drafts), str(pre / DRAFTS_DIR))
        log.info("moved %s", roots.drafts)


def _delete_outputs(roots: Roots, log: logging.Logger) -> None:
    """Delete outputs, run copies, REF, BASE, experiments and mutation exports."""
    targets = [roots.outputs, roots.ref, roots.base, roots.exp, roots.mut]
    targets += [path for path in sorted(roots.data.glob(f"{RUN_COPY_PREFIX}*")) if path.is_dir()]
    for path in targets:
        if path.exists():
            remove_tree(roots, log, path)


def remove_tree(roots: Roots, log: logging.Logger, path: Path) -> None:
    """Guarded removal that clears read-only attributes (git writes its objects read-only)."""
    guard_or_refuse(roots, [path], log)
    shutil.rmtree(path, onerror=_clear_read_only)
    log.info("removed %s", path)


def _clear_read_only(function: Callable, path: str, _excinfo: object) -> None:
    """rmtree error handler: clear the read-only attribute and retry once (a second failure raises)."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


# =============================================================================
# 10. BASE HELPERS
# =============================================================================


class Refusal(RuntimeError):
    """A failed harness check; its code names the check in the logged reason."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"REFUSED[{code}]: {detail}")
        self.code = code


def refuse(log: logging.Logger, code: str, detail: str) -> NoReturn:
    """Log a refusal at ERROR, then raise it."""
    log.error("REFUSED[%s]: %s", code, detail)
    raise Refusal(code, detail)


def run_helper(argv: list[str], *, cwd: Path, log: logging.Logger, bound_s: float = HELPER_BOUND_S,
               stdin_text: Optional[str] = None, check: bool = True, strict: bool = True,
               out_path: Optional[Path] = None) -> tuple[int, str]:
    """A bounded helper command with its input and output in files, never pipes.

    With out_path, the output goes to that file (created exclusively) and is not returned:
    the caller reads it, as EXP-20's secret scan of git log -p does.
    """
    with (tempfile.TemporaryFile() as inp, tempfile.TemporaryFile() if out_path is None else open(out_path, "xb") as out,
          tempfile.TemporaryFile() as err):
        if stdin_text is not None:
            inp.write(stdin_text.encode("utf-8"))
            inp.seek(0)
        started = time.monotonic()
        try:
            done = subprocess.run(argv, cwd=cwd, stdin=inp, stdout=out, stderr=err, timeout=bound_s)
        except subprocess.TimeoutExpired:
            log.error("helper command exceeded its %.0f s bound after %.1f s: %s", bound_s, time.monotonic() - started, argv)
            raise
        out.seek(0)
        err.seek(0)
        raw, raw_err = (b"" if out_path is not None else out.read()), err.read()
    text = raw.decode("utf-8") if strict else raw.decode("ascii", "backslashreplace")
    if check and done.returncode != 0:
        detail = raw_err.decode("ascii", "backslashreplace").strip()
        refuse(log, "helper", f"{argv} returned {done.returncode}: {detail[-400:]}")
    return done.returncode, text


def _git_text(roots: Roots, log: logging.Logger, *args: str) -> str:
    """Stripped output of a git command in the roots' worktree."""
    return run_helper(["git", *args], cwd=roots.worktree, log=log)[1].strip()


def _git_z(roots: Roots, log: logging.Logger, *args: str) -> list[str]:
    """NUL-separated output of a git command in the roots' worktree."""
    return [token for token in run_helper(["git", *args], cwd=roots.worktree, log=log)[1].split("\0") if token]


def file_md5(path: Path) -> str:
    """MD5 of a file, streamed."""
    digest = hashlib.md5()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_path(left: Path, right: Path) -> bool:
    """Equality of two paths after resolution, case-insensitively as Windows compares them."""
    return os.path.normcase(str(Path(left).resolve(strict=False))) == os.path.normcase(str(Path(right).resolve(strict=False)))


def _relative_to(base: Path, path: Path, *, resolve: bool = True) -> Optional[Path]:
    """path relative to base (both resolved unless resolve is False), None when outside."""
    base_path, target = (Path(base).resolve(strict=False), Path(path).resolve(strict=False)) if resolve else (Path(base), Path(path))
    return target.relative_to(base_path) if target.is_relative_to(base_path) else None


def _display_path(roots: Roots, path: Path) -> str:
    """Worktree-relative POSIX path when inside the worktree, else the absolute path."""
    rel = _relative_to(roots.worktree, path)
    return rel.as_posix() if rel is not None else str(path)


def _same_time(left: float, right: float) -> bool:
    """Creation times of one process read twice."""
    return abs(left - right) <= TIME_TOLERANCE_S


def _own_create_time() -> float:
    """Creation time of this process."""
    import psutil

    return psutil.Process(os.getpid()).create_time()


def _now() -> str:
    """Local time with offset, ISO 8601."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log_record(log: logging.Logger, record: dict) -> None:
    """One machine-readable RECORD line (launch log only)."""
    log.info("%s%s", RECORD_PREFIX, json.dumps(record, sort_keys=True, ensure_ascii=True, default=str))


def _log_event(log: logging.Logger, event: str, **fields: object) -> None:
    """A RECORD line describing one event of a launch."""
    _log_record(log, {"event": event, **fields})


def _record_payloads(text: str) -> list[str]:
    """JSON payloads of the RECORD lines of a launch log."""
    marker = f" {RECORD_PREFIX}"
    return [line.split(marker, 1)[1] for line in text.splitlines() if marker in line]


def _prefixed_line(text: str, prefix: str) -> str:
    """Payload of the last line that carries the prefix (a child's report)."""
    payloads = [line.split(prefix, 1)[1] for line in text.splitlines() if prefix in line]
    if not payloads:
        raise Refusal("child-report", f"no line with {prefix.strip()} in the child's output")
    return payloads[-1]


class _AsciiFormatter(logging.Formatter):
    """Console formatter: ASCII-only output."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).encode("ascii", "backslashreplace").decode("ascii")


def _console_logger() -> logging.Logger:
    """The harness logger with its ASCII console handler (RECORD lines go to the launch log only)."""
    log = logging.getLogger(HARNESS_STEM)
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_AsciiFormatter("%(levelname)s harness: %(message)s"))
    handler.addFilter(lambda record: not record.getMessage().startswith(RECORD_PREFIX))
    log.addHandler(handler)
    return log


def _attach_launch_log(log: logging.Logger, path: Path) -> logging.Logger:
    """Add the launch log, created exclusively so no log is ever reused or appended to."""
    stream = open(path, "x", encoding="utf-8")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    return log


def _stdout_logger(name: str) -> logging.Logger:
    """One stdout handler for an audit child (its stdout is its log file)."""
    log = logging.getLogger(f"audit.{name}")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    return log


if __name__ == "__main__":
    sys.exit(main())
