"""EXP-20: AST and text sweeps of production code and of every script the audit adds.

1. the handler classifier: the eight classes of section 6.3, checked in table sequence
   (the first match wins), a contextlib.suppress block judged as its except-pass clause,
   with its planted control module;
2. the new-scripts scan (--new-scripts PATH[=AS] ...): bytecode statements first, the
   family preamble before any other work (provider and helper modules exempt), handler
   classes against the gate column, machine paths, print calls, inlined variation_pct,
   series_id/df parameter slots, one-word module names and the terminology rule;
3. the pre-commit gate (--staged): every handler or guard a staged diff adds or edits
   (a removal edits the one that holds the line before it), classified on the parent and
   on the staged post-image, plus the new-scripts scan of staged scripts and the rule
   checks of a staged fix diff;
4. the production sweeps (--sweeps, on --source-root): the HYP-09 handler inventory,
   subprocess timeouts, resource release, the HYP-24 store probe, the config, constant and
   HDF5-key contracts, the unused window_size and the realistic turning_points as window counts (Q14), the
   rule-conformance table, sinks, raise sites against the tests, concurrency, module
   state, secrets (redacted), the input's names and the dependency pins.

The terminology checks (the planted control, the new-scripts and gate verdicts, the scope sweep) read the rule data of
the --rules folder (harness.RULE_DATA_FILE: exclusions and control lines) beside atp_terminology.md; without that file
they are skipped after one NOTICE line, every other check unchanged.

Structure:
1. main
2. Controls
3. New-scripts scan
4. Pre-commit gate
5. Production sweeps
6. Handler classifier
7. Terminology scanner
8. Source helpers
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=False)

import argparse
import ast
import dataclasses
import importlib.metadata
import io
import json
import re
import shutil
import tokenize
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np
import pandas as pd
import yaml
from packaging.markers import Marker
from pydantic import BaseModel
from scipy.signal import find_peaks

CLASS_NAMES = {1: "re-raise", 2: "log-then-raise", 3: "deferred raise", 4: "control-flow handler",
               5: "legitimate skip", 6: "log-and-return guard", 7: "WARNING degradation", 8: "swallow"}
GATE_PASS = frozenset({1, 2, 3, 4, 5})
CLASS_DISPOSITIONS = {3: "Q18: compliant only under reading B", 4: "Q19: outside the rule only under reading B"}
ERROR_LEVELS = frozenset({"error", "exception", "critical", "fatal"})
WARNING_LEVELS = frozenset({"warning", "warn"})
CONTROL_FLOW_TYPES = frozenset({"Empty", "KeyError", "StopIteration"})
LOG_METHODS = ERROR_LEVELS | WARNING_LEVELS | frozenset({"info", "debug", "log"})
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
LOOPS = (ast.For, ast.AsyncFor, ast.While)
ESCAPES = (ast.Return, ast.Continue, ast.Break)
SUPPRESS_CALL = "suppress"  # contextlib.suppress, also imported by name
KIND_EXCEPT = "except"
KIND_GUARD = "guard"
KIND_SUPPRESS = "suppress"

# provider and imported helper modules: they run after a script's preamble, so they carry only the bytecode statements
FAMILY_MODULES = {"audit": frozenset({"harness.py", "pipelinekit.py", "synthetic.py"}), "tests": frozenset({"testkit.py"})}
TESTKIT_PATH = "tests/testkit.py"
# property: the functions that hold it in the harness (HARNESS_SIDE) and in tests/testkit.py (TESTKIT_SIDE)
HARNESS_SIDE = {"environment keys": ("child_environment",), "capture encoding": ("decode_child_output", "_strict_decode"),
                "bytecode flag": ("launch_child",), "source-root insertion and assertion": ("insert_source_root",),
                "logger setup": ("_stdout_logger",), "output-root rule": ("refuse_output_root", "output_root_problems", "_contained"),
                "launch": ("launch_child",), "kill and survivors": ("end_tree", "walk", "survivors_of", "_taskkill")}
TESTKIT_SIDE = {"environment keys": ("child_environment",), "capture encoding": ("decode_capture",),
                "bytecode flag": ("run_child",), "source-root insertion and assertion": ("insert_source_root",),
                "logger setup": ("_stdout_logger",), "output-root rule": ("assert_outputs_inside",),
                "launch": ("run_child",), "kill and survivors": ("end_tree", "walk", "survivors_of", "taskkill")}
LAUNCHER_TIMEOUTS = ("PIPELINE_TIMEOUT_S", "CHILD_TIMEOUT_S", "HELPER_BOUND_S", "SURVIVOR_BOUND_S", "POLL_INTERVAL_S",
                     "GONE_POLL_INTERVAL_S")
LAUNCHER_LOG_FORMAT = "'%(asctime)s %(levelname)s %(name)s: %(message)s'"
# property: (a testkit fragment, its planted replacement); each must change that property alone
LAUNCHER_MUTATIONS = {
    "environment keys": ('env.update({BYTECODE_ENV: "1", HASHSEED_ENV', "env.update({HASHSEED_ENV"),
    "timeouts": ("CHILD_TIMEOUT_S = 300.0", "CHILD_TIMEOUT_S = 30.0"),
    "capture encoding": ('encoding = "utf-8"', 'encoding = "latin-1"'),
    "bytecode flag": ('[sys._base_executable, "-B"]', "[sys._base_executable]"),
    "source-root insertion and assertion": ("if not origin.is_relative_to(Path(root).resolve()):", "if False:"),
    "logger setup": ("logging.StreamHandler(sys.stdout)", "logging.StreamHandler(sys.stderr)"),
    "output-root rule": ('raise RuntimeError(f"outputs outside the scratch folder', 'RuntimeError(f"outputs outside the scratch folder'),
    "launch": ('argv = [sys._base_executable, "-B"]', 'argv = [sys.executable, "-B"]'),
    "kill and survivors": ("members = walk(snapshot(), [root])", "members = [root]"),
}
# family: (the statements after the bytecode statements, the provider whose preamble follows them)
FAMILY_BOOTSTRAPS = {"audit": (("import os", "sys.path.insert(0, os.environ['PYTHIA_AUDIT_HARNESS'])", "import harness"), "harness"),
                     "tests": (("import testkit",), "testkit")}
BYTECODE_STATEMENTS = ("import sys", "sys.dont_write_bytecode = True")
MODULE_NAME = re.compile(r"^[a-z0-9]+\.py$")
MACHINE_PATH = re.compile(r"(?i)(?<![a-z0-9])[a-z]:[\\/]|[\\/]users[\\/]")
IGNORE_WHITESPACE = "-w"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FIX_PREFIXES = ("src/", "tests/")
FIX_FILES = ("pipeline.py",)
TERMINOLOGY_RULE = "atp_terminology.md"

CONTROL_SOURCE = '''
import contextlib
import logging
from contextlib import suppress
from queue import Empty
log = logging.getLogger(__name__)

def re_raise():
    try:
        work()
    except OSError:
        cleanup()
        raise

def record_then_raise():
    try:
        work()
    except OSError as exc:
        record(exc)
        raise

def warning_then_raise():
    try:
        work()
    except OSError:
        log.warning("cannot retry")
        raise

def log_then_raise():
    try:
        work()
    except OSError as exc:
        log.error("failed: %s", exc)
        raise

def log_record_raise():
    try:
        work()
    except OSError as exc:
        log.error("failed: %s", exc)
        record(exc)
        raise

def inner_loop_then_raise(items):
    try:
        work()
    except OSError:
        for item in items:
            if item:
                continue
            cleanup(item)
        raise

def deferred_raise(items):
    failures = []
    for item in items:
        try:
            work(item)
        except OSError as exc:
            log.error("item %s failed: %s", item, exc)
            failures.append(item)
            continue
    if failures:
        raise RuntimeError(failures)

def unrelated_later_raise(items, config):
    for item in items:
        try:
            work(item)
        except OSError as exc:
            log.error("item %s failed: %s", item, exc)
            continue
    if not config:
        raise ValueError(config)

def control_flow(queue):
    while True:
        try:
            item = queue.get(timeout=1)
        except Empty:
            continue
        handle(item)

def legitimate_skip(rows):
    for row in rows:
        try:
            parse(row)
        except ValueError:
            log.warning("skipping a documented malformed row %s", row)
            continue

def log_and_return():
    try:
        work()
    except OSError as exc:
        log.error("failed: %s", exc)
        return None

def warning_degradation():
    try:
        return load()
    except OSError as exc:
        log.warning("no data: %s", exc)
        return []

def swallow():
    try:
        work()
    except Exception:
        pass

def fallback_to_default(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def suppressed():
    with contextlib.suppress(OSError):
        work()

def suppressed_lookup(table, key):
    with suppress(KeyError):
        return table[key]

def error_then_conditional_return(flag):
    try:
        work()
    except OSError as exc:
        log.error("failed: %s", exc)
        if flag:
            return
        raise

def error_guard(frame):
    if frame is None:
        log.error("no frame")
        return None
    return frame

def warning_guard(frame):
    if frame.empty:
        log.warning("empty frame")
        return frame
    return frame
'''
CONTROL_PATH = "control/planted.py"
CONTROL_EXPECTED = {"re_raise": 1, "record_then_raise": 1, "warning_then_raise": 1, "log_then_raise": 2,
                    "log_record_raise": 2, "inner_loop_then_raise": 1, "deferred_raise": 3, "unrelated_later_raise": 8,
                    "control_flow": 4, "legitimate_skip": 5, "log_and_return": 6, "warning_degradation": 7, "swallow": 8,
                    "fallback_to_default": 8, "suppressed": 8, "suppressed_lookup": 4,
                    "error_then_conditional_return": 6, "error_guard": 6, "warning_guard": 7}

EXPERIMENT = "20"
PRODUCTION_PARTS = ("src", "pipeline.py")
SEPARATE_PARTS = ("tests", "scripts")
MIN_PRODUCTION_MODULES = 40
BROAD_CATCH = "Exception"
RELIED_DEFAULT = "performance.metrics.display_order_gradient"  # the key the shipped YAMLs omit (Q11)
EXPECTED_CLAUSES = {"total": 59, "broad": 53, "broad_files": 17, "typed": tuple(sorted((
    "src/tasks/trigger.py:321", "src/utils/common/common.py:270", "src/utils/common/producerconsumer.py:76",
    "src/utils/preprocess/read.py:162", "src/utils/preprocess/read.py:166", "src/utils/trigger/prints/realistic.py:155")))}
SUBPROCESS_CALLS = frozenset({"subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
                              "subprocess.Popen"})
RESOURCE_CALLS = frozenset({"open", "h5py.File", "pd.HDFStore", "HDFStore"})  # the open() builtin and the stores
RELEASE_CALL = ".close("
FIGURE_CALLS = frozenset({"plt.subplots", "plt.figure"})
THREAD_CALLS = frozenset({"Thread", "threading.Thread"})
CONCURRENCY_CALLS = THREAD_CALLS | frozenset({"threading.Lock", "threading.RLock", "Lock", "RLock", "Queue", "queue.Queue",
                                              "threading.Event", "Event", "threading.Condition", "threading.Semaphore",
                                              "ThreadPoolExecutor", "ProcessPoolExecutor", "multiprocessing.Pool"})
STORE_PROBE_CODE = "\n".join([
    "import sys",
    "sys.dont_write_bytecode = True",
    "sys.path.insert(0, sys.argv[1])",
    "import logging",
    "from pathlib import Path",
    "import harness",
    "harness.insert_source_root(Path(sys.argv[2]))",
    "logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s %(name)s: %(message)s')",
    "from src.utils.common.hdf5manager import hdf5_manager",
    "with hdf5_manager.open_store(Path(sys.argv[3]), 'r') as store:",
    "    logging.getLogger('probe').info('PROBE body ran: %d keys', len(store.keys()))",
    "    raise RuntimeError(sys.argv[4])",
])
PLANTED_PROBE_ERROR = "planted error raised inside the store body"
PROBE_RAN = "PROBE body ran"
STORE_LINE = "Error opening HDF5 store"
SCHEMA_FILE = "src/utils/common/schema.py"
SCHEMA_ROOT = "AppConfig"
CONSTANTS_FILE = "src/utils/common/constants.py"
YAML_FILES = ("extrema.yaml", "dev.yaml")
REQUIRE_PREFIX = "require_"
TYPED_SCHEMA = """
class AppConfig(BaseModel):
    read: ReadConfig
    trigger: TriggerConfig
    runtime: RuntimeConfig
class ReadConfig(BaseModel):
    raw_data: str
    skip_series: List[str] = Field(default_factory=list)
class TriggerConfig(BaseModel):
    min_length: int
    read: DebugFlags = Field(default_factory=lambda: DebugFlags())
    preprocess: DebugFlags = Field(default_factory=lambda: DebugFlags())
class DebugFlags(BaseModel):
    printout_io: bool = False
class RuntimeConfig(BaseModel):
    queue_maxsize: int
    def doubled(self):
        return 2 * self.queue_maxsize
"""
TYPED_SOURCE = """
def by_parameter(read: ReadConfig):
    return read.raw_data
def by_assignment(config: AppConfig):
    trigger: TriggerConfig = config.trigger
    return trigger.min_length
class Mixin:
    runtime: RuntimeConfig
    def size(self):
        return self.runtime.queue_maxsize
class Task:
    read = SectionParameter(ReadConfig)
    def run(self):
        return self.read.skip_series
def unannotated(trigger):
    return trigger.read.printout_io
def ambiguous(flags: DebugFlags):
    return flags.printout_io
def by_method(read: ReadConfig):
    return read.skip_series.copy()
def misspelt(read: ReadConfig):
    return read.raw_dta
def own_attributes(config: AppConfig, runtime: RuntimeConfig):
    return config.trigger.model_copy(), runtime.doubled()
def containers(pairs: List[Tuple[AppConfig, str]], maybe: Optional["ReadConfig"]):
    return pairs.sort(), maybe.raw_data
def reassigned(read: ReadConfig, config: AppConfig):
    read = config.trigger
    return read.min_length
def nested(read: ReadConfig):
    return lambda read: read.skip_series
"""
TYPED_EXPECTED = {"parameter annotation": ("read.raw_data", 3), "annotated assignment": ("trigger.min_length", 6),
                  "class-body annotation": ("runtime.queue_maxsize", 10), "SectionParameter attribute": ("read.skip_series", 14),
                  "a leaf with a method call": ("read.skip_series", 20),
                  "a string annotation inside Optional": ("read.raw_data", 26)}
TYPED_UNCOUNTED_LINE = 16  # the unannotated root's read
TYPED_AMBIGUOUS_LINE = 18  # the DebugFlags root's read
TYPED_OUTSIDE = ("read.raw_dta", 22)  # the misspelt read, its first prefix neither a model nor a leaf
TYPED_OWN_LINE = 24  # a pydantic method and a def of a schema class
TYPED_CONTAINER_LINE = 26  # a container annotation naming AppConfig binds no root: pairs.sort() is no read
TYPED_REBOUND_LINES = [29, 31]  # a root its function assigns again; a root a nested lambda's parameter shadows
OPTIONAL_NAMES = frozenset({"Optional", "typing.Optional"})
SEQUENCE_NAMES = frozenset({"Sequence", "typing.Sequence", "List", "typing.List", "list"})
SECTION_PARAMETER = "SectionParameter"
STRATEGY_NAME_FILE = "src/utils/common/strategyname.py"  # the one module that reads window_size (the folder name, R-T12)
DYNAMIC = "{}"
MANAGER_NAME = "hdf5_manager"
STORE_RECEIVER = re.compile(r"^(self\.)?([a-z0-9]+_)?store$|hdf5_manager$")
WRITE_METHODS = frozenset({"put", "put_dataframe_normalized", "append"})
KEY_METHODS = WRITE_METHODS | frozenset({"get", "get_dataframe", "get_dataframe_fast", "select", "get_storer"})
WINDOW_SIZE = "window_size"
VARIATION_PCT = "variation_pct"
TURNING_POINTS = "turning_points"
VAR3_COLUMN = "var3"
WINDOW_EXPANDING_TYPE = "expanding"
TRIGGER_VALUE_NAMES = ("TRIGGER_VALUE", "trigger_value")
DISPLAY_ORDER_NAMES = ("display_order_signals", "display_order_gradient")
TERMINOLOGY_SCOPE = ("src/**/*.py", "pipeline.py", "tests/*.py", "scripts/*.py", "workflows/*.yaml", "audit/*.py",
                     "docs/*.md", "docs/mmflowcharts/*.mmd", "README.md")
INIT_FILE = "__init__.py"
DOC_CONFIG_PATH = re.compile(r"`([a-z_]+(?:\.[a-z0-9_]+)+)`")
CONFIG_SECTIONS = frozenset({"read", "trigger", "signal", "performance", "general", "runtime"})
SINK_PATTERNS = {"shell=True": re.compile(r"shell\s*=\s*True"), "eval(": re.compile(r"\beval\("),
                 "exec(": re.compile(r"\bexec\("), "pickle": re.compile(r"\bpickle\b"),
                 "yaml.load(": re.compile(r"\byaml\.load\("), "marshal": re.compile(r"\bmarshal\b"),
                 "verify=False": re.compile(r"\bverify\s*=\s*False"), "mktemp": re.compile(r"\bmktemp\("),
                 "os.system": re.compile(r"\bos\.system\(")}
SINK_SQL = "string-formatted SQL"
SINK_PATH = "path concatenation with +"
SQL_TEXT = re.compile(r"(?i)\b(select|insert|update|delete)\b.*\b(from|into|set|where)\b")
PATH_NAME = re.compile(r"(?i)(path|dir|file|folder)")
PATH_LITERAL = re.compile(r"[\\/][^\\/]*\.[A-Za-z0-9]{1,5}$")
SINK_CONTROL = '''
import marshal, os, pickle, subprocess, tempfile, yaml
subprocess.run(command, shell=True)
eval(text)
exec(text)
pickle.loads(blob)
yaml.load(stream)
marshal.loads(blob)
session.get(url, verify=False)
tempfile.mktemp()
os.system(command)
query = f"SELECT name FROM items WHERE id = {item}"
target = output_dir + "/result.h5"
'''
MUTABLE_CALLS = frozenset({"list", "dict", "set", "defaultdict", "OrderedDict", "deque", "Counter"})
EMAIL_ADDRESS = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# reserved names (RFC 2606, RFC 6761): synthetic addresses such as the self-test's fixture identity, not personal data
RESERVED_EMAIL = re.compile(r"(?i)@(?:[a-z0-9-]+\.)*(?:example\.(?:com|net|org)|example|invalid|localhost|test)$")
SECRET_PATTERNS = {
    "private key header": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "JSON web token": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "credential assignment": re.compile(r"(?i)\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)\b\s*[:=]\s*"
                                        r"[\"']([^\"'\s]{6,})[\"']"),
    "e-mail address": EMAIL_ADDRESS,
}
# built from parts at run time, so this file holds no literal the patterns match
SECRET_CONTROL = {
    "private key header": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "AWS access key id": "key = " + "AKIA" + "Q" * 16,
    "GitHub token": "ghp_" + "x" * 36,
    "Slack token": "xoxb-" + "1" * 12,
    "Google API key": "AIza" + "B" * 35,
    "JSON web token": "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12,
    "credential assignment": "password = " + "'" + "planted" + "1" + "'",
    "e-mail address": "user" + "@" + "planted-control.net",
}
EXPORT_SIDE_PARTS = frozenset({"__pycache__"})
GIT_LOG_SCRATCH = "gitlog.txt"
COMMIT_LINE = "commit "
DIFF_FILE_LINE = "+++ b/"
SHORT_SHA = 7
GIT_HEADERS = ("Author:", "Commit:", "Date:", "Merge:")
REQUIREMENTS_FILE = "setup/requirements.txt"
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)[ \t]*(?:;[ \t]*(.+?))?[ \t]*$", re.M)


@dataclasses.dataclass(frozen=True)
class Disposition:
    """A documented condition that makes a WARNING skip legitimate (class 5)."""

    path: str
    function: str
    fragment: str
    citation: str


@dataclasses.dataclass(frozen=True)
class Site:
    """One classified handler or guard."""

    path: str
    line: int
    end_line: int
    function: str
    kind: str
    klass: int
    caught: str
    disposition: str

    @property
    def passes(self) -> bool:
        """The gate column of the class table."""
        return self.klass in GATE_PASS


@dataclasses.dataclass(frozen=True)
class Terms:
    """The terminology rule's lists, as word sequences, and its rule data in the file's sequence: the exclusions as
    (the words it may exclude, empty: every word; the compiled line pattern; the reason) and the control lines as
    (line, reported)."""

    sequences: tuple
    exempt: tuple
    exclusions: tuple
    controls: tuple


def main() -> None:
    """Controls first, then the requested scans; the terminology checks only with the rule data of --rules."""
    args = _parse_args()
    terms = load_terms(args.rules)
    dispositions = [_parse_disposition(spec) for spec in args.disposition]
    check_classifier_control()
    if args.sweeps:
        check_typed_read_control()
    if terms is not None:
        check_terminology_control(terms)
    for spec in args.new_scripts:
        path, _, as_path = spec.partition("=")
        file = Path(path).resolve()
        scan_new_script(file.read_text(encoding="utf-8"), as_path or harness.copy_as_path(CTX.roots, file), terms, dispositions)
    if args.staged:
        gate_staged(terms, dispositions)
    if args.sweeps:
        raw = (harness.REAL_WORKTREE / merged_yaml(CTX.source_root)["read"]["raw_data"]).resolve()
        run_sweeps(CTX.source_root, args.ref, raw, terms, dispositions)
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--rules (atp_terminology.md and its rule data), --new-scripts, --staged, --disposition, --sweeps with --ref, and
    --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--new-scripts", nargs="+", default=[], metavar="PATH[=AS]")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--disposition", nargs="+", default=[], metavar="PATH::FUNCTION::FRAGMENT=CITATION")
    parser.add_argument("--sweeps", action="store_true")
    parser.add_argument("--ref", type=Path, default=harness.REAL_WORKTREE / "data" / "ref" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


def _parse_disposition(spec: str) -> Disposition:
    """PATH::FUNCTION::FRAGMENT=CITATION."""
    target, _, citation = spec.rpartition("=")
    path, function, fragment = target.split("::")
    return Disposition(path, function, fragment, citation)


# =============================================================================
# CONTROLS
# =============================================================================


def check_classifier_control() -> None:
    """Every planted handler and guard lands in its class (planted-defect control of the classifier)."""
    planted = [Disposition(CONTROL_PATH, "legitimate_skip", "documented malformed", "control.md:1")]
    sites = classify_source(CONTROL_SOURCE, CONTROL_PATH, planted)
    found = {site.function: site.klass for site in sites}
    CTX.verdict("classifier control: non-vacuity", len(sites) == len(CONTROL_EXPECTED), len(CONTROL_EXPECTED), len(sites))
    for function, expected in CONTROL_EXPECTED.items():
        CTX.verdict(f"classifier control: {function}", found.get(function) == expected,
                    f"{expected} {CLASS_NAMES[expected]}", found.get(function))


def check_typed_read_control() -> None:
    """The typed-read resolution on a planted schema and module: each of the four binding forms counted, a leaf with a
    method call counted as the leaf, a misspelt chain read as a path outside the schema, a model's own attributes (a
    pydantic method, a def of a schema class) counted nothing, an unannotated root not counted, a root of a class at
    two prefixes (DebugFlags) counted nothing and reported ambiguous, a container annotation binding no root, a root
    its function or a nested scope binds again counted nothing and reported rebound."""
    schema = schema_paths(TYPED_SCHEMA)
    reads, ambiguous, rebound = typed_reads(ast.parse(TYPED_SOURCE), schema_models(TYPED_SCHEMA), schema,
                                            model_attributes(TYPED_SCHEMA))
    CTX.verdict("typed-read control: non-vacuity", len(schema) == 6 and bool(reads), "6 leaves, reads > 0",
                {"leaves": len(schema), "reads": reads})
    for form, (dotted, line) in TYPED_EXPECTED.items():
        CTX.verdict(f"typed-read control: {form} counted", (dotted, line) in reads, [dotted, line], reads)
    outside = [read for read in reads if read[0] not in schema]
    CTX.verdict("typed-read control: a misspelt chain read as a path outside the schema", outside == [TYPED_OUTSIDE],
                [TYPED_OUTSIDE], outside)
    own = [read for read in reads if read[1] == TYPED_OWN_LINE]
    CTX.verdict("typed-read control: a model's own attributes (a pydantic method, a def of a schema class) counted "
                "nothing", not own, [], own)
    uncounted = [read for read in reads if read[1] == TYPED_UNCOUNTED_LINE]
    CTX.verdict("typed-read control: an unannotated root counted nothing", not uncounted, [], uncounted)
    ambiguous_counted = [read for read in reads if read[1] == TYPED_AMBIGUOUS_LINE]
    CTX.verdict("typed-read control: a DebugFlags root reported ambiguous and counted nothing",
                ambiguous == [TYPED_AMBIGUOUS_LINE] and not ambiguous_counted, [TYPED_AMBIGUOUS_LINE],
                {"ambiguous": ambiguous, "counted": ambiguous_counted})
    container = [read for read in reads if read[1] == TYPED_CONTAINER_LINE and read[0] not in schema]
    CTX.verdict("typed-read control: a container annotation (List[Tuple[AppConfig, str]]) binds no root", not container,
                [], container)
    rebound_counted = [read for read in reads if read[1] in TYPED_REBOUND_LINES]
    CTX.verdict("typed-read control: a root bound again (an assignment, a nested lambda's parameter) reported rebound "
                "and counted nothing", rebound == TYPED_REBOUND_LINES and not rebound_counted, TYPED_REBOUND_LINES,
                {"rebound": rebound, "counted": rebound_counted})


def check_terminology_control(terms: Terms) -> None:
    """The rule data's planted lines: lifecycle and forbidden uses reported, ordinary uses excluded."""
    CTX.verdict("terminology control: non-vacuity", len(terms.sequences) > 30, "> 30 term sequences", len(terms.sequences))
    for line, reported in terms.controls:
        hits = [hit for hit in terminology_hits(line, "control", terms) if hit[3] == "reported"]
        CTX.verdict(f"terminology control: {line!r}", bool(hits) == reported, "reported" if reported else "excluded", hits)


# =============================================================================
# NEW-SCRIPTS SCAN
# =============================================================================


def scan_new_script(source: str, as_path: str, terms: Optional[Terms], dispositions: list[Disposition]) -> bool:
    """The new-scripts checks of one script, judged as the file it is or becomes (terminology only with terms)."""
    tree = ast.parse(source)
    statements = _without_docstring(tree.body)
    family = as_path.split("/", 1)[0]
    name = Path(as_path).name
    results = [
        CTX.verdict(f"{as_path}: one-word module name", bool(MODULE_NAME.match(name)), "lowercase word", name),
        CTX.verdict(f"{as_path}: bytecode statements first", _unparsed(statements[:2]) == BYTECODE_STATEMENTS,
                    BYTECODE_STATEMENTS, _unparsed(statements[:2])),
    ]
    if name not in FAMILY_MODULES.get(family, frozenset()):
        results.append(CTX.verdict(f"{as_path}: family preamble before any other work",
                                   _calls_preamble(statements, family), f"{family} bootstrap then preamble",
                                   _unparsed(statements[2:6])))
    sites = classify_source(source, as_path, dispositions)
    for site in sites:
        CTX.log.info("SITE %s", _site_text(site))
    failing = [_site_text(site) for site in sites if not site.passes]
    results += [
        CTX.verdict(f"{as_path}: handler classes pass the gate", not failing, "classes 1-5", failing),
        CTX.verdict(f"{as_path}: no machine path", not (found := _machine_paths(source)), "none", found),
        CTX.verdict(f"{as_path}: no print call", not (found := _print_calls(tree)), "none", found),
        CTX.verdict(f"{as_path}: variation_pct never inlined", not (found := _inlined_variation(tree)), "none", found),
        CTX.verdict(f"{as_path}: series_id first, df second", not (found := _parameter_slots(tree)), "none", found),
    ]
    if terms is not None:
        hits = terminology_hits(source, as_path, terms)
        for hit in hits:
            CTX.log.info("TERM %s:%d %s (%s)", *hit)
        reported = [f"{hit[0]}:{hit[1]} {hit[2]}" for hit in hits if hit[3] == "reported"]
        results.append(CTX.verdict(f"{as_path}: terminology", not reported, "no reported hit", reported))
    if as_path == TESTKIT_PATH:
        results.append(compare_launchers(harness.HARNESS_FILE.read_text(encoding="utf-8"), source))
    return all(results)


def compare_launchers(harness_source: str, testkit_source: str) -> bool:
    """The two child launchers (audit harness and tests/testkit.py) property by property (section 6.1), then one
    planted mutation of the testkit per property, each of which must change that property and no other."""
    reference = launcher_properties(harness_source, HARNESS_SIDE)
    actual = launcher_properties(testkit_source, TESTKIT_SIDE)
    results = [CTX.verdict(f"launchers: {name} equal in both families", actual[name] == reference[name], reference[name],
                           actual[name]) for name in sorted(reference)]
    for name, (old, new) in sorted(LAUNCHER_MUTATIONS.items()):
        if testkit_source.count(old) != 1:
            results.append(CTX.verdict(f"launcher control: {name} mutation site", False, "exactly one", testkit_source.count(old)))
            continue
        planted = launcher_properties(testkit_source.replace(old, new), TESTKIT_SIDE)
        changed = sorted(prop for prop in reference if planted[prop] != actual[prop])
        results.append(CTX.verdict(f"launcher control: a planted change of the {name} is flagged", changed == [name], [name], changed))
    return all(results)


def launcher_properties(source: str, side: dict) -> dict:
    """The comparable features of one family's launcher: the property names of LAUNCHER_SIDES, each from its functions."""
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    constants = {target.id: statement.value.value for statement in tree.body if isinstance(statement, ast.Assign)
                 and isinstance(statement.value, ast.Constant) for target in statement.targets if isinstance(target, ast.Name)}

    def nodes(prop: str) -> list[ast.AST]:
        return [node for name in side[prop] for node in ast.walk(functions[name])]

    def text(prop: str) -> str:
        return "\n".join(ast.unparse(functions[name]) for name in side[prop])

    return {
        "environment keys": sorted({constants[node.id] for node in nodes("environment keys")
                                    if isinstance(node, ast.Name) and node.id.endswith("_ENV") and node.id in constants}),
        "timeouts": {name: constants.get(name) for name in LAUNCHER_TIMEOUTS},
        "capture encoding": {"utf-8": "'utf-8'" in text("capture encoding"), "locale.getencoding": "locale.getencoding()" in text("capture encoding"),
                             "strict decode": any(isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "decode"
                                                  and len(node.args) == 1 and not node.keywords for node in nodes("capture encoding"))},
        "bytecode flag": any(isinstance(node, ast.Constant) and node.value == "-B" for node in nodes("bytecode flag")),
        "source-root insertion and assertion": {"insert": "sys.path.insert(0," in text("source-root insertion and assertion"),
                                                "assert": ".is_relative_to(" in text("source-root insertion and assertion")},
        "logger setup": {"stdout": "logging.StreamHandler(sys.stdout)" in text("logger setup"),
                         "level": "setLevel(logging.INFO)" in text("logger setup"), "propagate": "propagate = False" in text("logger setup"),
                         "format": LAUNCHER_LOG_FORMAT in text("logger setup")},
        "output-root rule": {"containment": ".is_relative_to(" in text("output-root rule"),
                             "refusal": any(isinstance(node, ast.Raise) or (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "refuse")
                                            for node in nodes("output-root rule"))},
        "launch": any(isinstance(node, ast.List) and node.elts and ast.unparse(node.elts[0]) == "sys._base_executable"
                      for node in nodes("launch")),
        "kill and survivors": {"tree walks": text("kill and survivors").count("walk(snapshot(),"),
                               "survivor bound": "survivors_of(" in text("kill and survivors") and "SURVIVOR_BOUND_S" in text("kill and survivors"),
                               "force by pid": "'/F'" in text("kill and survivors") and "'/PID'" in text("kill and survivors")},
    }


def _calls_preamble(statements: list[ast.stmt], family: str) -> bool:
    """The family's bootstrap, then an assignment or expression calling its provider's preamble."""
    if family not in FAMILY_BOOTSTRAPS:
        return False
    bootstrap, provider = FAMILY_BOOTSTRAPS[family]
    end = len(BYTECODE_STATEMENTS) + len(bootstrap)
    if _unparsed(statements[len(BYTECODE_STATEMENTS):end]) != bootstrap or len(statements) <= end:
        return False
    value = getattr(statements[end], "value", None)
    return (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name) and value.func.value.id == provider
            and value.func.attr == "preamble")


def _machine_paths(source: str) -> list[str]:
    """String constants and comments that hold a drive or user path."""
    found = [f"line {node.lineno}" for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Constant) and isinstance(node.value, str) and MACHINE_PATH.search(node.value)]
    found += [f"comment line {token.start[0]}" for token in _tokens(source)
              if token.type == tokenize.COMMENT and MACHINE_PATH.search(token.string)]
    return found


def _print_calls(tree: ast.Module) -> list[int]:
    """Lines of print() calls (results go through the logger)."""
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"]


def _inlined_variation(tree: ast.AST) -> list[int]:
    """Lines of (a / b - 1) * 100 written out instead of variation_pct (its own definition excepted)."""
    canonical = {id(inner) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == VARIATION_PCT
                 for inner in ast.walk(node)}
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)) or id(node) in canonical:
            continue
        for factor, other in ((node.left, node.right), (node.right, node.left)):
            if (_is_number(other, 100) and isinstance(factor, ast.BinOp) and isinstance(factor.op, ast.Sub)
                    and _is_number(factor.right, 1) and isinstance(factor.left, ast.BinOp)
                    and isinstance(factor.left.op, ast.Div)):
                found.append(node.lineno)
    return found


def _is_number(node: ast.AST, value: float) -> bool:
    """A numeric constant equal to value."""
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and node.value == value


def _parameter_slots(tree: ast.AST) -> list[str]:
    """Functions whose series_id is not the first parameter or whose df is not the second."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [arg.arg for arg in node.args.posonlyargs + node.args.args]
        if names and names[0] in ("self", "cls"):
            names = names[1:]
        if "series_id" in names and names.index("series_id") != 0:
            found.append(f"{node.name}:{node.lineno} series_id at {names.index('series_id')}")
        if "series_id" in names and "df" in names and names.index("df") != 1:
            found.append(f"{node.name}:{node.lineno} df at {names.index('df')}")
    return found


# =============================================================================
# PRE-COMMIT GATE
# =============================================================================


def gate_staged(terms: Optional[Terms], dispositions: list[Disposition]) -> None:
    """Classify every handler or guard the staged diff adds or edits; scan staged scripts."""
    staged = [path for path in _git("diff", "--cached", "--name-only", "-z").split("\0") if path]
    audit = all(path.startswith("audit/") for path in staged)
    fix = all(path.startswith(FIX_PREFIXES) or path in FIX_FILES for path in staged)
    CTX.log.info("staged paths (%s commit): %s", "audit" if audit else "fix" if fix else "mixed", staged)
    CTX.verdict("gate: staged paths form one commit kind", bool(staged) and (audit or fix), "audit or fix", staged)
    present = {path for path in _git("diff", "--cached", "--name-only", "-z", "--diff-filter=d").split("\0") if path}
    verdicts = []
    for path in staged:
        if not path.endswith(".py") or path not in present:
            continue
        post = _git("show", f":{path}")
        parent = _git("show", f"HEAD:{path}") if _in_head(path) else ""
        parent_lines, post_lines = _changed_lines(path, removal_points=True)
        parent_sites = [site for site in classify_source(parent, path, dispositions) if _touches(site, parent_lines)] if parent else []
        post_sites = [site for site in classify_source(post, path, dispositions) if _touches(site, post_lines)]
        for site in parent_sites:
            CTX.log.info("GATE parent %s", _site_text(site))
        for site in post_sites:
            CTX.log.info("GATE post %s", _site_text(site))
        failing = [_site_text(site) for site in post_sites if not site.passes]
        verdicts.append(CTX.verdict(f"gate: {path} post-image handlers", not failing, "classes 1-5", failing))
        if path.startswith(("audit/", "tests/")):
            verdicts.append(scan_new_script(post, path, terms, dispositions))
        elif fix:
            verdicts.append(_fix_rule_checks(path, post, _changed_lines(path, IGNORE_WHITESPACE)[1], terms))
    CTX.log.info("GATE %s", "PASS" if all(verdicts) and (audit or fix) else "FAIL")


def _fix_rule_checks(path: str, post: str, post_lines: set[int], terms: Optional[Terms]) -> bool:
    """Rule conformance of a production file's changed lines (section 5): lines whose content changed, because a
    re-indented line keeps its words (the handler gate above sees every changed line, indentation included); their
    terminology only with terms."""
    tree = ast.parse(post)
    inlined = [line for line in _inlined_variation(tree) if line in post_lines]
    misplaced = [item for item in _parameter_slots(tree) if int(item.split(":")[1].split()[0]) in post_lines]
    verdicts = [
        CTX.verdict(f"gate: {path} variation_pct never inlined", not inlined, "none", inlined),
        CTX.verdict(f"gate: {path} series_id first, df second", not misplaced, "none", misplaced),
    ]
    if terms is not None:
        changed = "\n".join(line if number in post_lines else "" for number, line in enumerate(post.splitlines(), start=1))
        reported = [f"{hit[0]}:{hit[1]} {hit[2]}" for hit in terminology_hits(changed, path, terms) if hit[3] == "reported"]
        verdicts.append(CTX.verdict(f"gate: {path} terminology of changed lines", not reported, "no reported hit", reported))
    return all(verdicts)


def _changed_lines(path: str, *flags: str, removal_points: bool = False) -> tuple[set[int], set[int]]:
    """Parent and post-image line numbers the staged diff removes or adds (with IGNORE_WHITESPACE: content changes only).
    With removal_points, a hunk that removes lines also marks the post-image line before its added ones (git starts an
    empty range at that line), so a handler that lost lines, its raise included, counts as edited."""
    parent, post = set(), set()
    for line in _git("diff", "--cached", "-U0", *flags, "--no-color", "--no-ext-diff", "--", path).splitlines():
        match = HUNK.match(line)
        if match:
            start, count = int(match.group(1)), int(match.group(2) or 1)
            parent.update(range(start, start + count))
            removed = count > 0
            start, count = int(match.group(3)), int(match.group(4) or 1)
            post.update(range(start, start + count))
            if removal_points and removed and (point := start - 1 if count else start) > 0:
                post.add(point)
    return parent, post


def _touches(site: Site, lines: set[int]) -> bool:
    """The site's span holds a changed line."""
    return any(site.line <= line <= site.end_line for line in lines)


def _in_head(path: str) -> bool:
    """The path exists in HEAD."""
    return bool(_git("ls-tree", "--name-only", "HEAD", "--", path).strip())


def _git(*args: str) -> str:
    """A bounded git command in the worktree."""
    return harness.run_helper(["git", *args], cwd=harness.REAL_WORKTREE, log=CTX.log)[1]


# =============================================================================
# PRODUCTION SWEEPS
# =============================================================================


def run_sweeps(root: Path, ref: Path, raw: Path, terms: Optional[Terms], dispositions: list[Disposition]) -> None:
    """Every sweep of the EXP-20 row on one source root (the worktree or data/base)."""
    production = python_sources(root, PRODUCTION_PARTS)
    CTX.verdict("sweeps: non-vacuity: production modules read", len(production) >= MIN_PRODUCTION_MODULES,
                f">= {MIN_PRODUCTION_MODULES}", len(production))
    check_inventory(inventory("production", production, dispositions), harness.same_path(root, CTX.roots.base))
    for part in SEPARATE_PARTS:
        inventory(part, python_sources(root, (part,)), dispositions)
    subprocess_timeouts(production)
    resource_release(production)
    store_probe(root, ref)
    schema = schema_paths(production[SCHEMA_FILE])
    config_contract(root, production, schema)
    constants_contract(production)
    key_contract(production)
    unused_parameter(production, ref, merged_yaml(root))
    rule_conformance(root, production, schema, terms)
    sinks(root)
    raise_sites(root, production)
    concurrency(production)
    module_state(production)
    duplicate_names(production)
    secrets(root)
    input_inventory(raw)
    dependency_audit(root)


def python_sources(root: Path, parts: tuple) -> dict[str, str]:
    """{posix path relative to the root: source} of the .py files under the parts, sorted; absent parts logged."""
    found = {}
    for part in parts:
        target = root / part
        if not target.exists():
            CTX.log.info("SWEEP %s: not in this source root", part)
        files = [target] if target.is_file() else sorted(target.rglob("*.py")) if target.is_dir() else []
        found.update({file.relative_to(root).as_posix(): file.read_text(encoding="utf-8") for file in files})
    return dict(sorted(found.items()))


def inventory(family: str, sources: dict[str, str], dispositions: list[Disposition]) -> list[Site]:
    """Every except clause and logging guard of a family, classified, one INVENTORY line each (HYP-09)."""
    sites = [site for path, source in sources.items() for site in classify_source(source, path, dispositions)]
    for site in sites:
        CTX.log.info("INVENTORY %s %s", family, _site_text(site))
    per_class = {CLASS_NAMES[klass]: sum(site.klass == klass for site in sites) for klass in sorted(CLASS_NAMES)}
    CTX.log.info("INVENTORY %s: %d modules, %d sites, %d except clauses; per class %s", family, len(sources), len(sites),
                 sum(site.kind == KIND_EXCEPT for site in sites), json.dumps(per_class))
    return sites


def check_inventory(sites: list[Site], at_base: bool) -> None:
    """The production clause counts: on data\\base a verdict against the plan's measurement of the audited revision, on any other
    root a logged measurement (the fixes remove clauses and shift lines); the sites outside the gate classes."""
    clauses = [site for site in sites if site.kind == KIND_EXCEPT]
    broad = [site for site in clauses if site.caught == BROAD_CATCH]
    counts = {"total": len(clauses), "broad": len(broad), "broad_files": len({site.path for site in broad}),
              "typed": tuple(sorted(f"{site.path}:{site.line}" for site in clauses if site.caught != BROAD_CATCH))}
    if at_base:
        CTX.verdict("inventory: production except clauses as measured (59: 53 Exception in 17 files, 6 typed)",
                    counts == EXPECTED_CLAUSES, EXPECTED_CLAUSES, counts)
    else:
        CTX.log.info("INVENTORY production except clauses (the measured 59 describe data\\base only): %s", counts)
    failing = [site for site in sites if not site.passes]
    CTX.verdict("inventory: every production handler and guard in classes 1-5 (atp_coding-style.md:30)", not failing,
                "none in classes 6-8", f"{len(failing)} sites in classes 6-8")


def subprocess_timeouts(sources: dict[str, str]) -> list[str]:
    """subprocess calls without a timeout keyword (02 non-negotiable 5; HYP-23)."""
    found = [f"{path}:{node.lineno} {_call_name(node)}" for path, source in sources.items() for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and _call_name(node) in SUBPROCESS_CALLS
             and not any(keyword.arg == "timeout" for keyword in node.keywords)]
    for item in found:
        CTX.log.info("SUBPROCESS %s without a timeout", item)
    CTX.verdict("every subprocess call bounded by a timeout", not found, "none unbounded", found)
    return found


def resource_release(sources: dict[str, str]) -> list[str]:
    """Resources acquired outside a with statement, resources held across a yield, figures and whether a finally
    block calls .close() on them, threads and their daemon flag (02 resource release on every path out)."""
    found = []
    for path, source in sources.items():
        tree = ast.parse(source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        managed = {id(item.context_expr) for node in ast.walk(tree) if isinstance(node, (ast.With, ast.AsyncWith))
                   for item in node.items}
        for node in ast.walk(tree):
            name = _call_name(node) if isinstance(node, ast.Call) else ""
            where = f"{path}:{getattr(node, 'lineno', 0)} in {_function_name(node, parents)}"
            if name in RESOURCE_CALLS and id(node) not in managed:
                found.append(f"{where}: {name} outside a with statement; release in a finally block: "
                             f"{_releases_in_finally(node, parents)}")
            elif name in FIGURE_CALLS:
                found.append(f"{where}: {name}; plt.close() in a finally block: {_releases_in_finally(node, parents)}")
            elif name in THREAD_CALLS:
                daemon = next((ast.unparse(keyword.value) for keyword in node.keywords if keyword.arg == "daemon"), "unset")
                found.append(f"{where}: {name} daemon={daemon}")
            elif isinstance(node, ast.With) and _has_any(node.body, (ast.Yield, ast.YieldFrom)):
                found.append(f"{where}: a with-resource held across a yield (released only when the generator ends)")
    for item in found:
        CTX.log.info("RESOURCE %s", item)
    CTX.verdict("resource sweep: non-vacuity", bool(found), "> 0 sites", len(found))
    return found


def _releases_in_finally(node: ast.AST, parents: dict) -> bool:
    """The enclosing function has a finally block whose code calls .close()."""
    function = _enclosing_function(node, parents)
    return function is not None and any(RELEASE_CALL in ast.unparse(part) for statement in _local_nodes(function.body)
                                        if isinstance(statement, ast.Try) for part in statement.finalbody)


def _enclosing_function(node: ast.AST, parents: dict) -> Optional[ast.AST]:
    """The nearest enclosing function definition."""
    current = parents.get(node)
    while current is not None and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
        current = parents.get(current)
    return current


def store_probe(root: Path, ref: Path) -> dict:
    """HYP-24: an error raised inside open_store(path, "r")'s body, on a scratch copy of a reference file, in a child."""
    copy = CTX.label_dir(EXPERIMENT) / "probe" / "0.h5"
    CTX.guard(copy)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ref / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / "realistic" / "trigger" / "0.h5", copy)
    child = CTX.launch_child("storeprobe", ["-c", STORE_PROBE_CODE, str(harness.HARNESS_DIR), str(root), str(copy),
                                            PLANTED_PROBE_ERROR], timeout_s=harness.CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED)
    store_lines = [line for line in harness.plain_lines(child.text) if STORE_LINE in line]
    ran = PROBE_RAN in child.text and child.return_code == 1
    CTX.verdict("HYP-24 probe: non-vacuity: the store was readable, its body ran and raised (return code 1)", ran,
                "body ran, return code 1", {"return_code": child.return_code, "log": child.log_path.name})
    CTX.verdict("HYP-24: an error raised in the store body is not reported as 'Error opening HDF5 store' (standard 08)",
                ran and not any(PLANTED_PROBE_ERROR in line for line in store_lines), f"no '{STORE_LINE}' line", store_lines)
    return {"return_code": child.return_code, "store_lines": store_lines}


def config_contract(root: Path, sources: dict[str, str], schema: dict[str, bool]) -> dict:
    """The typed reads and any require_* paths against the schema's fields and the merged YAML keys; the defaults code
    relies on (Q11)."""
    schema_source = sources[SCHEMA_FILE]
    reads = config_reads(sources, schema_models(schema_source), schema, model_attributes(schema_source))
    keys = merged_yaml_paths(root)
    patterns = [re.compile("^" + re.escape(path).replace(re.escape(DYNAMIC), r"[^.]+") + "$") for path in reads if DYNAMIC in path]
    outside = sorted(path for path in reads if DYNAMIC not in path and path not in schema)
    relied = {path: reads[path] for path, defaulted in sorted(schema.items()) if defaulted and path not in keys and path in reads}
    undefaulted = {path: sites for path, sites in sorted(reads.items())
                   if path in schema and not schema[path] and path not in keys}
    unread = sorted(path for path in keys if path not in reads and not any(pattern.match(path) for pattern in patterns))
    for path, sites in sorted(reads.items()):
        CTX.log.info("CONFIG read %s at %s", path, sites)
    CTX.log.info("CONFIG schema defaults absent from the YAML: %s", sorted(path for path, defaulted in schema.items()
                                                                          if defaulted and path not in keys))
    CTX.verdict("config: every typed read and require_* path is a schema field", not outside, "none outside", outside)
    CTX.verdict("config: every read of a schema key absent from the merged YAML has a schema default (Q11: the validated "
                f"model applies it; non-vacuity: {RELIED_DEFAULT} relied on)", not undefaulted and RELIED_DEFAULT in relied,
                {"undefaulted": {}, "relied": f"includes {RELIED_DEFAULT}"}, {"undefaulted": undefaulted, "relied": relied})
    CTX.verdict("config: every YAML key is read (a typed read or a require_* path)", not unread, "none unread", unread)
    return {"reads": len(reads), "outside": outside, "relied": relied, "undefaulted": undefaulted, "unread": unread}


def config_reads(sources: dict[str, str], models: dict[str, list[str]], schema: dict[str, bool],
                 own_attributes: frozenset[str]) -> dict[str, list[str]]:
    """{dotted path: [path:line]} of every typed read (typed_reads: a schema leaf, or a path outside the schema) and of
    every remaining require_*(config, path) call (formatted parts become {}); each read through a root of a class at
    two prefixes is logged on an AMBIGUOUS-ROOT line, each through a root name its function binds again on a
    REBOUND-ROOT line."""
    reads: dict[str, list[str]] = {}
    for path, source in sources.items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and _call_name(node).split(".")[-1].startswith(REQUIRE_PREFIX)
                    and len(node.args) >= 2 and (dotted := _rendered(node.args[1], {})) is not None):
                reads.setdefault(dotted, []).append(f"{path}:{node.lineno}")
        typed, ambiguous, rebound = typed_reads(tree, models, schema, own_attributes)
        for dotted, line in typed:
            reads.setdefault(dotted, []).append(f"{path}:{line}")
        for line in ambiguous:
            CTX.log.info("AMBIGUOUS-ROOT %s:%d: a root of a class at more than one prefix, counted nothing", path, line)
        for line in rebound:
            CTX.log.info("REBOUND-ROOT %s:%d: a root name its function binds again, counted nothing", path, line)
    return reads


def typed_reads(tree: ast.Module, models: dict[str, list[str]], schema: dict[str, bool],
                own_attributes: frozenset[str]) -> tuple[list[tuple[str, int]], list[int], list[int]]:
    """(dotted path, line) of every typed read of a module, the lines of the reads through an ambiguous root and those
    through a rebound root. A root is a name bound in its function by a parameter annotation or an annotated
    assignment, or self.<attribute> bound in its class by a class-body annotation or a class attribute
    SectionParameter(Model); the chain after it is resolved name by name from the class's one prefix (_chain_path).
    A root name the function or a nested scope binds again (_bindings) is rebound: its reads count nothing."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    prefixes = {prefix for found in models.values() for prefix in found}
    reads, ambiguous, rebound = set(), set(), set()
    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        owner = parents.get(function)
        attributes = _class_roots(owner, models) if isinstance(owner, ast.ClassDef) else {}
        roots = _function_roots(function, models)
        bindings = _bindings(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Attribute) or (isinstance(parents.get(node), ast.Attribute)
                                                       and parents[node].value is node):
                continue
            base, chain = _attribute_chain(node)
            if isinstance(base, ast.Name) and base.id in roots and bindings.get(base.id, 0) > 1:
                rebound.add(node.lineno)
                continue
            if isinstance(base, ast.Name) and base.id in roots:
                model = roots[base.id]
            elif isinstance(base, ast.Name) and base.id == "self" and chain and chain[0] in attributes:
                model, chain = attributes[chain[0]], chain[1:]
            else:
                continue
            if len(models[model]) > 1:
                ambiguous.add(node.lineno)
                continue
            dotted = _chain_path(models[model][0], chain, prefixes, schema, own_attributes)
            if dotted is not None:
                reads.add((dotted, node.lineno))
    return sorted(reads), sorted(ambiguous), sorted(rebound)


def _bindings(function: ast.AST) -> dict[str, int]:
    """{name: binding sites} in a function and its nested scopes: parameters (a nested def's or lambda's included),
    stores and deletions, nested def and class names, import aliases, except and match captures; an annotation
    without a value binds nothing."""
    declarations = {id(node.target) for node in ast.walk(function) if isinstance(node, ast.AnnAssign) and node.value is None}
    names = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in declarations:
            names.append(node.id)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function:
            names.append(node.name)
        elif isinstance(node, ast.alias):
            names.append(node.asname or node.name.split(".")[0])
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name:
            names.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.append(node.rest)
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _function_roots(function: ast.AST, models: dict[str, list[str]]) -> dict[str, str]:
    """{name: schema class} of a function's parameter annotations and annotated assignments naming one class, and of
    the target of a for loop or comprehension over a parameter annotated Sequence[class] or List[class] (the items of
    a list field, such as trigger.gate.items)."""
    arguments = function.args
    parameters = [arg for arg in arguments.posonlyargs + arguments.args + arguments.kwonlyargs if arg.annotation is not None]
    bound = {arg.arg: _annotated_model(arg.annotation, models) for arg in parameters}
    bound.update({node.target.id: _annotated_model(node.annotation, models) for node in ast.walk(function)
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)})
    sequences = {arg.arg: _sequence_model(arg.annotation, models) for arg in parameters}
    bound.update({node.target.id: sequences[node.iter.id] for node in ast.walk(function)
                  if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name)
                  and isinstance(node.iter, ast.Name) and node.iter.id in sequences})
    return {name: model for name, model in bound.items() if model is not None}


def _class_roots(owner: ast.ClassDef, models: dict[str, list[str]]) -> dict[str, str]:
    """{attribute: schema class} of a class body's annotations and SectionParameter(Model) attributes."""
    bound = {}
    for statement in owner.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            bound[statement.target.id] = _annotated_model(statement.annotation, models)
        elif (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
              and _call_name(statement.value) == SECTION_PARAMETER and statement.value.args):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = _annotated_model(statement.value.args[0], models)
    return {name: model for name, model in bound.items() if model is not None}


def _sequence_model(annotation: ast.AST, models: dict[str, list[str]]) -> Optional[str]:
    """The schema class an annotation names as the item of Sequence[class] or List[class], also as a string; None
    else."""
    annotation = _unquoted(annotation)
    return _annotated_model(annotation.slice, models) if _is_sequence(annotation) else None


def _is_sequence(annotation: ast.AST) -> bool:
    """An unquoted annotation Sequence[...] or List[...]."""
    return isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) in SEQUENCE_NAMES


def _annotated_model(annotation: ast.AST, models: dict[str, list[str]]) -> Optional[str]:
    """The schema class an annotation (or a class argument) names as the class itself, Optional[class] or class | None,
    each also as a string; None else, so a container such as List[Tuple[AppConfig, str]] binds no root."""
    annotation = _unquoted(annotation)
    if isinstance(annotation, ast.Subscript) and ast.unparse(annotation.value) in OPTIONAL_NAMES:
        return _annotated_model(annotation.slice, models)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        kept = [side for side in (annotation.left, annotation.right)
                if not (isinstance(side, ast.Constant) and side.value is None)]
        return _annotated_model(kept[0], models) if len(kept) == 1 else None
    return annotation.id if isinstance(annotation, ast.Name) and annotation.id in models else None


def _unquoted(annotation: ast.AST) -> ast.AST:
    """A string annotation parsed into its expression, as often as it is quoted; any other annotation as it is. A string
    that is no expression raises its SyntaxError, so a malformed annotation fails the sweep instead of binding nothing."""
    while isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    return annotation


def _attribute_chain(node: ast.Attribute) -> tuple[ast.AST, list[str]]:
    """The base of an attribute chain and its attribute names, in source sequence."""
    chain = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    return node, chain[::-1]


def _chain_path(prefix: str, chain: list[str], prefixes: set[str], schema: dict[str, bool],
                own_attributes: frozenset[str]) -> Optional[str]:
    """The path a typed chain reads, its names resolved one by one from its root's prefix: the first prefix that is a
    schema leaf (a leaf's own attribute, as read.raw_data.split, may follow), or the first that is neither a model nor
    a leaf, a path outside the schema; None for a chain that ends at a model (as config.read passed whole) or reaches a
    model's own attribute (model_attributes, as model_copy)."""
    path = prefix
    for name in chain:
        dotted = path + name
        if dotted in schema:
            return dotted
        if f"{dotted}." not in prefixes:
            return None if name in own_attributes else dotted
        path = f"{dotted}."
    return None


def schema_models(source: str) -> dict[str, list[str]]:
    """{schema class: [its dotted prefixes in the AppConfig tree]} (AppConfig at the empty prefix), from the walk
    schema_paths makes; a class two fields annotate holds two prefixes."""
    models = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    found: dict[str, list[str]] = {}

    def walk(model: str, prefix: str) -> None:
        found.setdefault(model, []).append(prefix)
        for statement in models[model].body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                nested = [node.id for node in ast.walk(statement.annotation) if isinstance(node, ast.Name) and node.id in models]
                if nested:
                    walk(nested[0], f"{prefix}{statement.target.id}.")

    walk(SCHEMA_ROOT, "")
    return found


def model_attributes(source: str) -> frozenset[str]:
    """The names a schema model answers beside its fields: pydantic.BaseModel's attributes and each def and assignment
    target of a schema class body (a typed chain that reaches one reads no field)."""
    defined = set()
    for node in (item for item in ast.parse(source).body if isinstance(item, ast.ClassDef)):
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(statement.name)
            elif isinstance(statement, ast.Assign):
                defined |= {target.id for target in statement.targets if isinstance(target, ast.Name)}
    return frozenset(dir(BaseModel)) | defined


def schema_paths(source: str) -> dict[str, bool]:
    """{dotted leaf path: has a default} of the schema's AppConfig tree, parsed from schema.py's models; a List[model]
    field is a leaf of its own default and its model's fields are leaves at the field's prefix."""
    models = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    found: dict[str, bool] = {}

    def walk(model: str, prefix: str, defaulted: bool) -> None:
        for statement in models[model].body:
            if not (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)):
                continue
            name, default = statement.target.id, defaulted or _has_default(statement.value)
            nested = [node.id for node in ast.walk(statement.annotation) if isinstance(node, ast.Name) and node.id in models]
            if not nested or _is_sequence(_unquoted(statement.annotation)):
                found[f"{prefix}{name}"] = default
            if nested:
                walk(nested[0], f"{prefix}{name}.", default)

    walk(SCHEMA_ROOT, "", False)
    return found


def _has_default(value: Optional[ast.AST]) -> bool:
    """A field value that supplies a default: a plain value, or Field(...) with default or default_factory."""
    if value is None:
        return False
    if isinstance(value, ast.Call) and _call_name(value) == "Field":
        return any(keyword.arg in ("default", "default_factory") for keyword in value.keywords) or bool(value.args)
    return True


def merged_yaml_paths(root: Path) -> set[str]:
    """Dotted leaf paths of the merged workflow YAML, a list leaf kept, and <list path>.<key> for every key of every
    mapping in a list leaf (the fields of the items of a List[model] field, such as trigger.gate.items)."""
    leaves = harness.flatten_yaml(merged_yaml(root))
    return set(leaves) | {f"{path}.{key}" for path, value in leaves.items() if isinstance(value, list)
                          for item in value if isinstance(item, dict) for key in item}


def merged_yaml(root: Path) -> dict:
    """workflows/extrema.yaml merged with dev.yaml (dev values win for leaf keys, R-X6)."""
    merged: dict = {}
    for name in YAML_FILES:
        _deep_merge(merged, yaml.safe_load((root / "workflows" / name).read_text(encoding="utf-8")))
    return merged


def _deep_merge(target: dict, source: dict) -> None:
    """Merge source into target; nested mappings merge, leaves of source win."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def constants_contract(sources: dict[str, str]) -> dict:
    """Each string constant of constants.py: its uses by name, and the same literal written out elsewhere."""
    constants = string_constants(sources[CONSTANTS_FILE])
    uses: dict[str, int] = {}
    literals: dict[str, list[str]] = {}
    values = {value for value, _ in constants.values()}
    for path, source in sources.items():
        tree = ast.parse(source)
        docstrings = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                uses[node.id] = uses.get(node.id, 0) + 1
            elif isinstance(node, ast.Attribute):
                uses[node.attr] = uses.get(node.attr, 0) + 1
            elif (path != CONSTANTS_FILE and isinstance(node, ast.Constant) and isinstance(node.value, str)
                  and node.value in values and id(node) not in docstrings):
                literals.setdefault(node.value, []).append(f"{path}:{node.lineno}")
    unused = sorted(name for name in constants if not uses.get(name))
    duplicated = {name: literals[value] for name, (value, _) in sorted(constants.items()) if value in literals}
    for name, sites in duplicated.items():
        CTX.log.info("CONSTANT %s = %r also written as a literal at %s", name, constants[name][0], sites)
    CTX.verdict("constants: every string constant of constants.py used by name", not unused, "none unused", unused)
    CTX.verdict("constants: no constant's literal written out elsewhere (single source of truth)", not duplicated, "none",
                f"{len(duplicated)} constants, {sum(map(len, duplicated.values()))} literal sites")
    return {"constants": len(constants), "unused": unused, "duplicated": len(duplicated)}


def string_constants(source: str) -> dict[str, tuple[str, int]]:
    """{NAME: (value, line)} of a module's top-level string assignments."""
    return {target.id: (statement.value.value, statement.lineno) for statement in ast.parse(source).body
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str) for target in statement.targets if isinstance(target, ast.Name)}


def key_contract(sources: dict[str, str]) -> dict:
    """HDF5 keys written against the code that names them elsewhere. Reads go through wrappers and prefix scans
    (replot.py:110-114, hdf5manager.py:183-188), so a written key counts as consumed when its literal, or a
    constant it is built from, appears outside a write call's key argument; store reads by literal key must name a
    written key; write and read sites whose key is a variable are listed."""
    constants = {name: value for name, (value, _) in string_constants(sources[CONSTANTS_FILE]).items()}
    written: dict[str, dict] = {}
    read: dict[str, list[str]] = {}
    dynamic, write_nodes, trees = [], set(), {path: ast.parse(source) for path, source in sources.items()}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in KEY_METHODS
                    and STORE_RECEIVER.search(ast.unparse(node.func.value))):
                continue
            argument = _key_argument(node)
            key = None if argument is None else _rendered(argument, constants)
            if key is None or key == DYNAMIC:
                dynamic.append(f"{path}:{node.lineno} {node.func.attr}")
                continue
            key = "/" + key.lstrip("/")
            if node.func.attr in WRITE_METHODS:
                write_nodes.update(id(inner) for inner in ast.walk(argument))
                item = written.setdefault(key, {"sites": [], "names": set()})
                item["sites"].append(f"{path}:{node.lineno}")
                item["names"] |= {inner.id for inner in ast.walk(argument) if isinstance(inner, ast.Name) and inner.id in constants}
            else:
                read.setdefault(key, []).append(f"{path}:{node.lineno}")
    consumers = {key: [] for key in written}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if id(node) in write_nodes:
                continue
            for key, item in written.items():
                named = isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in item["names"]
                literal = (isinstance(node, ast.Constant) and isinstance(node.value, str) and DYNAMIC not in key
                           and "/" + node.value.lstrip("/") == key)
                if named or literal:
                    consumers[key].append(f"{path}:{node.lineno}")
    patterns = {key: re.compile("^" + re.escape(key).replace(re.escape(DYNAMIC), ".+") + "$") for key in written}
    for key, item in sorted(written.items()):
        CTX.log.info("KEY written %s at %s; named elsewhere at %s", key, item["sites"], consumers[key])
    CTX.log.info("KEY dynamic sites (the key is a variable): %s", dynamic)
    unconsumed = sorted(key for key, sites in consumers.items() if not sites)
    unwritten = sorted(key for key in read if not any(pattern.match(key) for pattern in patterns.values()))
    CTX.verdict("keys: every written key named by consuming code", not unconsumed, "none", unconsumed)
    CTX.verdict("keys: every key read by literal is written", not unwritten, "none", unwritten)
    return {"written": sorted(written), "unconsumed": unconsumed, "unwritten": unwritten, "dynamic": len(dynamic)}


def _key_argument(call: ast.Call) -> Optional[ast.AST]:
    """The key argument: the second of an hdf5_manager helper (store, key, ...), the first of a store method."""
    receiver = ast.unparse(call.func.value)
    index = 1 if receiver.endswith(MANAGER_NAME) else 0
    return call.args[index] if len(call.args) > index else None


def _rendered(node: ast.AST, constants: dict[str, str]) -> Optional[str]:
    """A string expression as text: constants resolved, other formatted parts as {}; None when it is a bare variable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [_rendered(value.value, constants) if isinstance(value, ast.FormattedValue) else value.value
                 for value in node.values]
        return "".join(DYNAMIC if part is None else part for part in parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _rendered(node.left, constants), _rendered(node.right, constants)
        return None if left is None and right is None else f"{left or DYNAMIC}{right or DYNAMIC}"
    return None


def unused_parameter(sources: dict[str, str], ref: Path, config: dict) -> dict:
    """R-T12: every window_size occurrence; HYP-20: turning_points of the reference trigger files per mode, the realistic
    ones against the window counts find_peaks gives (Q14)."""
    sites = [f"{path}:{node.lineno} {type(node).__name__}{'(load)' if isinstance(getattr(node, 'ctx', None), ast.Load) else ''}"
             for path, source in sources.items() for node in ast.walk(ast.parse(source)) if _names(node, WINDOW_SIZE)]
    loads = [site for site in sites if "Attribute(load)" in site and not site.startswith(f"{STRATEGY_NAME_FILE}:")]
    CTX.log.info("WINDOW_SIZE occurrences: %s", sites)
    CTX.verdict(f"R-T12: no attribute read of window_size outside {STRATEGY_NAME_FILE} (the folder name)",
                not loads, "none", loads)
    found, frames = {}, {}
    for mode in ("ideal", "realistic"):
        path = ref / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / mode / "trigger" / "0.h5"
        with pd.HDFStore(str(path), mode="r") as store:
            frames[mode] = {key: store.get(key) for key in sorted(store.keys()) if key.startswith(harness.SERIES_KEY_PREFIX)}
        found[mode] = {key: int(frame[TURNING_POINTS].sum()) for key, frame in frames[mode].items()}
    replay, strategy = config["trigger"][harness.replay_key(config)], config["trigger"]["strategy"]
    differing, widest = {}, 0
    for key, frame in frames["realistic"].items():
        expected = window_turning_points(frame[VAR3_COLUMN].to_numpy(), replay["lookback_window"],
                                         replay["window_type"] == WINDOW_EXPANDING_TYPE, strategy)
        widest = max(widest, int(expected.max()))
        rows = np.flatnonzero(frame[TURNING_POINTS].to_numpy() != expected)
        if len(rows):
            differing[key] = [[int(row), int(frame[TURNING_POINTS].iat[row]), int(expected[row])] for row in rows[:4]]
    CTX.verdict("HYP-20 (Q14): realistic turning_points[i] is the count find_peaks reports in the window ending at row i "
                "(rows before lookback - 1 zero); non-vacuity: a window count above 1",
                not differing and widest > 1, {"differing": {}, "widest_count": "> 1"},
                {"differing [row, persisted, window count]": differing, "widest_count": widest, "totals": found["realistic"]})
    CTX.verdict("HYP-20 contrast: ideal turning_points non-zero", all(found["ideal"].values()), "non-zero per series", found["ideal"])
    return {"window_size": sites, "turning_points": found}


def window_turning_points(values: np.ndarray, lookback: int, expanding: bool, strategy: dict) -> np.ndarray:
    """Per row i >= lookback - 1, the turning points find_peaks reports (maxima of var3, then of -var3) in the window
    ending at i, with the strategy's distance, width and prominence relative to the window's range; 0 before."""
    counts = np.zeros(len(values), dtype=int)
    for row in range(lookback - 1, len(values)):
        window = values[0 if expanding else row - lookback + 1:row + 1]
        options = {"distance": strategy["min_distance"], "width": strategy["width"],
                   "prominence": strategy["prominence_threshold"] * float(window.max() - window.min())}
        counts[row] = len(np.union1d(find_peaks(window, **options)[0], find_peaks(-window, **options)[0]))
    return counts


def _names(node: ast.AST, name: str) -> bool:
    """A Name, Attribute, argument, keyword or string constant spelling the name."""
    return ((isinstance(node, ast.Name) and node.id == name) or (isinstance(node, ast.Attribute) and node.attr == name)
            or (isinstance(node, ast.arg) and node.arg == name) or (isinstance(node, ast.keyword) and node.arg == name)
            or (isinstance(node, ast.Constant) and node.value == name))


def rule_conformance(root: Path, production: dict[str, str], schema: dict[str, bool], terms: Optional[Terms]) -> dict:
    """The section-5 rule table: trigger-value read sites, display_order_* propagation, terminology (only with terms),
    docs config paths, coding-style checks."""
    for label, names in (("TRIGGERVALUE", TRIGGER_VALUE_NAMES), ("DISPLAYORDER", DISPLAY_ORDER_NAMES)):
        for path, source in production.items():
            tree = ast.parse(source)
            parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for node in ast.walk(tree):
                if any(_names(node, name) for name in names) and not isinstance(node, (ast.arg, ast.keyword)):
                    CTX.log.info("%s %s:%d %s %s", label, path, node.lineno, _function_name(node, parents), ast.unparse(node))
    reported = None
    if terms is not None:
        reported, excluded = {}, {}
        for path, text in sorted(terminology_sources(root).items()):
            hits = terminology_hits(text, path, terms)
            for hit in hits:
                CTX.log.info("TERM %s:%d %s (%s)", *hit)
                excluded[hit[3]] = excluded.get(hit[3], 0) + 1
            reported[path] = sum(hit[3] == "reported" for hit in hits)
        for _, _, reason in terms.exclusions:
            CTX.log.info("TERM-EXCLUDED %s: %d", reason, excluded.get(f"excluded: {reason}", 0))
        CTX.verdict("atp_terminology.md: no reported hit in its scope", not any(reported.values()), "none",
                    {path: count for path, count in reported.items() if count})
    unknown = docs_config_paths(root, schema)
    CTX.verdict("atp_documentation-style.md: every config path in docs/*.md is a schema path", not unknown, "none", unknown)
    style = python_sources(root, PRODUCTION_PARTS + SEPARATE_PARTS)
    slots = {path: found for path, source in style.items() if (found := _parameter_slots(ast.parse(source)))}
    inlined = {path: found for path, source in style.items() if (found := _inlined_variation(ast.parse(source)))}
    names = sorted(path for path in style if not MODULE_NAME.match(Path(path).name) and Path(path).name != INIT_FILE)
    CTX.verdict("atp_coding-style.md: series_id first, df second", not slots, "none", slots)
    CTX.verdict("atp_coding-style.md: variation_pct never inlined", not inlined, "none", inlined)
    CTX.verdict("atp_coding-style.md: one-word module names", not names, "none", names)
    return {"terminology": reported, "docs": unknown, "slots": slots, "inlined": inlined, "names": names}


def terminology_sources(root: Path) -> dict[str, str]:
    """The rule's scope: src/**/*.py, pipeline.py, tests/*.py, scripts/*.py, workflows/*.yaml, audit/*.py, docs/*.md,
    docs/mmflowcharts/*.mmd and README.md."""
    found = {}
    for pattern in TERMINOLOGY_SCOPE:
        for file in sorted(root.glob(pattern)):
            found[file.relative_to(root).as_posix()] = file.read_text(encoding="utf-8")
    return found


def docs_config_paths(root: Path, schema: dict[str, bool]) -> list[str]:
    """Backticked dotted config paths in docs/*.md that name neither a schema leaf nor a schema section."""
    sections = {".".join(path.split(".")[:depth]) for path in schema for depth in range(1, path.count(".") + 1)}
    unknown = []
    for file in sorted((root / "docs").glob("*.md")):
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            for dotted in DOC_CONFIG_PATH.findall(line):
                if dotted.split(".")[0] in CONFIG_SECTIONS and dotted not in schema and dotted not in sections:
                    unknown.append(f"docs/{file.name}:{number} {dotted}")
    return unknown


def sinks(root: Path) -> list[str]:
    """The 06 sink list over production, tests and scripts, after the planted control."""
    control = sink_hits(SINK_CONTROL, "control")
    for family in sorted(set(SINK_PATTERNS) | {SINK_SQL, SINK_PATH}):
        CTX.verdict(f"sink control: {family} flagged", any(hit.endswith(family) for hit in control), "flagged", control)
    hits = [hit for path, source in python_sources(root, PRODUCTION_PARTS + SEPARATE_PARTS).items() for hit in sink_hits(source, path)]
    for hit in hits:
        CTX.log.info("SINK %s", hit)
    CTX.verdict("06 sink sweep: no sink in production, tests or scripts", not hits, "none", hits)
    return hits


def sink_hits(source: str, path: str) -> list[str]:
    """path:line family of every literal sink pattern, string-formatted SQL and + path concatenation."""
    hits = [f"{path}:{number} {family}" for number, line in enumerate(source.splitlines(), start=1)
            for family, pattern in SINK_PATTERNS.items() if pattern.search(line)]
    for node in ast.walk(ast.parse(source)):
        if _formatted_sql(node):
            hits.append(f"{path}:{node.lineno} {SINK_SQL}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and (_path_like(node.left) or _path_like(node.right)):
            hits.append(f"{path}:{node.lineno} {SINK_PATH}")
    return hits


def _formatted_sql(node: ast.AST) -> bool:
    """An f-string, %-format, + or .format() whose literal text reads as SQL."""
    if isinstance(node, ast.JoinedStr):
        return bool(SQL_TEXT.search("".join(value.value for value in node.values if isinstance(value, ast.Constant))))
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mod, ast.Add)):
        return any(isinstance(side, ast.Constant) and isinstance(side.value, str) and SQL_TEXT.search(side.value)
                   for side in (node.left, node.right))
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
            and bool(SQL_TEXT.search(node.func.value.value)))


def _path_like(node: ast.AST) -> bool:
    """A name that reads as a filesystem path, or a string literal with a separator and a file extension."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        return bool(PATH_NAME.search(node.id if isinstance(node, ast.Name) else node.attr))
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and bool(PATH_LITERAL.search(node.value))


def raise_sites(root: Path, production: dict[str, str]) -> list[str]:
    """Every production raise with the test scripts that name its function (standard 05 failure-path sweep -> D)."""
    tests = python_sources(root, ("tests",))
    found = []
    for path, source in production.items():
        tree = ast.parse(source)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                function = _function_name(node, parents)
                raised = "re-raise" if node.exc is None else ast.unparse(getattr(node.exc, "func", node.exc))
                named = sorted(test for test, text in tests.items()
                               if re.search(rf"\b{re.escape(function.split('.')[-1])}\b", text))
                found.append(f"{path}:{node.lineno} {function} raises {raised}; tests naming it: {named or 'none'}")
    for item in found:
        CTX.log.info("RAISE %s", item)
    untested = [item for item in found if item.endswith("none")]
    CTX.verdict("failure paths: every production raise named by a test", not untested, "none untested",
                f"{len(untested)} of {len(found)} raise sites named by no test")
    return found


def concurrency(sources: dict[str, str]) -> list[str]:
    """Threads, pools, locks, queues and events, and numba parallel kernels (02 concurrency audit)."""
    found = [f"{path}:{node.lineno} {_call_name(node)}" for path, source in sources.items() for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and (_call_name(node) in CONCURRENCY_CALLS
                                                or any(keyword.arg == "parallel" for keyword in node.keywords))]
    for item in found:
        CTX.log.info("CONCURRENCY %s", item)
    CTX.verdict("concurrency sweep: non-vacuity", bool(found), "> 0 sites", len(found))
    return found


def module_state(sources: dict[str, str]) -> list[str]:
    """Module-level mutable bindings and global statements (09 module-level mutable state and global writes)."""
    found = []
    for path, source in sources.items():
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)) and _mutable(statement.value):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                found.append(f"{path}:{statement.lineno} {', '.join(ast.unparse(target) for target in targets)}")
        found += [f"{path}:{node.lineno} global {', '.join(node.names)}" for node in ast.walk(tree) if isinstance(node, ast.Global)]
    for item in found:
        CTX.log.info("MODULESTATE %s", item)
    CTX.verdict("module state sweep: non-vacuity", bool(found), "> 0 sites", len(found))
    return found


def duplicate_names(sources: dict[str, str]) -> dict:
    """Module-level functions of one name in several modules, with their parameters, and function locals that
    rebind an imported name: one name, two meanings (HYP-25; 01 consistency sweep)."""
    defined: dict[str, list[str]] = {}
    shadowed = []
    for path, source in sources.items():
        tree = ast.parse(source)
        imported = {alias.asname or alias.name.split(".")[0] for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.setdefault(node.name, []).append(f"{path}:{node.lineno} ({', '.join(arg.arg for arg in node.args.args)})")
        shadowed += [f"{path}:{inner.lineno} {function.name} rebinds {inner.id}" for function in ast.walk(tree)
                     if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) for inner in ast.walk(function)
                     if isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store) and inner.id in imported]
    duplicates = {name: sites for name, sites in sorted(defined.items()) if len({site.split(":")[0] for site in sites}) > 1}
    CTX.verdict("names: no module-level function name defined in two modules", not duplicates, "none", duplicates)
    CTX.verdict("names: no function local rebinds an imported name", not shadowed, "none", shadowed)
    return {"duplicates": duplicates, "shadowed": shadowed}


def _mutable(value: Optional[ast.AST]) -> bool:
    """A mutable literal or comprehension, a mutable container call, or an instance (a CamelCase call)."""
    if isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        name = _call_name(value).split(".")[-1]
        return name in MUTABLE_CALLS or name[:1].isupper()
    return False


def secrets(root: Path) -> list[str]:
    """Secret and personal-data patterns over the tree and, in the worktree, git log -p; hits as path:line, class
    and value length only, never the value or a hash of it (06 row)."""
    for name, text in SECRET_CONTROL.items():
        CTX.verdict(f"secret control: {name} flagged", any(hit.split(" ", 1)[1].startswith(name) for hit in secret_hits(text, "control")),
                    "flagged", len(text))
    hits = [hit for path, text in tree_texts(root).items() for hit in secret_hits(text, path)]
    if harness.same_path(root, harness.REAL_WORKTREE):
        hits += history_hits()
    else:
        CTX.log.info("SECRET git log -p: not applicable to an export without history")
    for hit in hits:
        CTX.log.info("SECRET %s", hit)
    CTX.verdict("06 secret sweep: no secret or personal-data hit", not hits, "none", len(hits))
    return hits


def secret_hits(text: str, path: str) -> list[str]:
    """path:line class (length N) of each match; the value itself is never kept."""
    return [f"{path}:{number} {name} (length {length})" for number, line in enumerate(text.splitlines(), start=1)
            for name, length in _line_secrets(line)]


def _line_secrets(line: str) -> list[tuple[str, int]]:
    """(class, value length) of each secret or personal-data match in one line; reserved-name addresses skipped."""
    return [(name, len(match.group(match.lastindex or 0))) for name, pattern in SECRET_PATTERNS.items()
            for match in pattern.finditer(line) if not (pattern is EMAIL_ADDRESS and RESERVED_EMAIL.search(match.group(0)))]


def tree_texts(root: Path) -> dict[str, str]:
    """Text files of the tree: in the worktree its tracked and untracked non-ignored files, in an export every file
    but its side files; binary files (a NUL byte) skipped."""
    if harness.same_path(root, harness.REAL_WORKTREE):
        names = [name for name in _git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0") if name]
    else:
        names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
                       and not any(part in EXPORT_SIDE_PARTS for part in path.relative_to(root).parts))
    found = {}
    for name in sorted(set(names)):
        data = (root / name).read_bytes()
        if b"\0" not in data:
            found[name] = data.decode("utf-8", "replace")
    return found


def history_hits() -> list[str]:
    """git log -p of HEAD into a scratch file under data/exp/20, scanned, then deleted; header e-mail lines counted apart."""
    scratch = CTX.label_dir(EXPERIMENT) / GIT_LOG_SCRATCH
    CTX.guard(scratch)
    harness.run_helper(["git", "log", "-p", "--no-color", "HEAD"], cwd=harness.REAL_WORKTREE, log=CTX.log, out_path=scratch)
    try:
        hits, commit, file, headers = [], "", "", 0
        for number, line in enumerate(scratch.read_bytes().decode("utf-8", "replace").splitlines(), start=1):
            if line.startswith(COMMIT_LINE):
                commit = line.split()[1][:SHORT_SHA]
            elif line.startswith(DIFF_FILE_LINE):
                file = line[len(DIFF_FILE_LINE):]
            if line.startswith(GIT_HEADERS):
                headers += bool(_line_secrets(line))
                continue
            hits += [f"{commit}:{file} (git log line {number}) {name} (length {length})" for name, length in _line_secrets(line)]
    finally:
        scratch.unlink()
    CTX.log.info("SECRET git log -p: %d header lines with an e-mail address (author and committer metadata)", headers)
    return hits


def input_inventory(raw: Path) -> dict:
    """06 data audit: the raw input's series count, column names and scalar names (no values)."""
    with h5py.File(raw, "r") as source:
        groups = sorted(source.keys())
        first = source[groups[0]]
        found = {"series": len(groups), "columns": [str(name) for name in first.attrs["columns"]],
                 "datasets": sorted(first.keys()), "scalars": sorted(first["scalars"].keys()) if "scalars" in first else []}
    CTX.log.info("INPUT %s", json.dumps(found))
    return found


def dependency_audit(root: Path) -> dict:
    """setup/requirements.txt pins against the installed versions, with purpose, licence and maintainer from METADATA."""
    requirements = root / REQUIREMENTS_FILE
    if not requirements.is_file():
        CTX.log.info("DEPENDENCY %s: not in this source root", REQUIREMENTS_FILE)
        return {}
    pinned_lines = PIN.findall(requirements.read_text(encoding="utf-8"))
    pins = {name: pinned for name, pinned, marker in pinned_lines if not marker or Marker(marker).evaluate()}
    CTX.log.info("DEPENDENCY pins whose marker excludes this platform: %s",
                 sorted(f"{name} ({marker})" for name, _, marker in pinned_lines if marker and not Marker(marker).evaluate()))
    installed = {_distribution_name(dist.metadata["Name"]): dist.metadata for dist in importlib.metadata.distributions()}
    mismatched = {}
    for name, pinned in sorted(pins.items()):
        meta = installed.get(_distribution_name(name))
        version = meta["Version"] if meta is not None else "not installed"
        if version != pinned:
            mismatched[name] = [pinned, version]
        if meta is not None:
            CTX.log.info("DEPENDENCY %s pinned %s installed %s | %s | licence %s | maintainer %s", name, pinned, version,
                         meta.get("Summary") or "not stated", _licence(meta), _maintainer(meta))
    unpinned = sorted(set(installed) - {_distribution_name(name) for name in pins})
    CTX.log.info("DEPENDENCY installed without a pin: %s", unpinned)
    CTX.verdict("dependencies: every pin matches the installed version", not mismatched, "all match", mismatched)
    return {"pins": len(pins), "mismatched": mismatched, "unpinned": unpinned}


def _distribution_name(name: str) -> str:
    """A distribution name normalised as PEP 503 does (lowercase, runs of -_. as one -)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _licence(meta: object) -> str:
    """License-Expression, else the first line of License, else a License classifier, else not stated."""
    if meta.get("License-Expression"):
        return meta.get("License-Expression")
    licence = (meta.get("License") or "").strip()
    if licence:
        return licence.splitlines()[0]
    classifiers = [value.split("::")[-1].strip() for value in meta.get_all("Classifier") or [] if value.startswith("License ::")]
    return classifiers[0] if classifiers else "not stated"


def _maintainer(meta: object) -> str:
    """Maintainer, else the names in Maintainer-email, else Author, else the names in Author-email, else not stated;
    e-mail addresses are never logged."""
    for field in ("Maintainer", "Maintainer-email", "Author", "Author-email"):
        names = EMAIL_ADDRESS.sub("", meta.get(field) or "").replace("<>", "").strip(" ,\"")
        if names:
            return names
    return "not stated"


def _call_name(node: ast.Call) -> str:
    """The dotted name of a call's function, as written."""
    return ast.unparse(node.func)


# =============================================================================
# HANDLER CLASSIFIER
# =============================================================================


def classify_source(source: str, path: str, dispositions: list[Disposition]) -> list[Site]:
    """Every except handler, every logging return/continue guard and every contextlib.suppress block of a module,
    classified."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            sites.append(_classify_handler(node, parents, source, path, dispositions))
        elif isinstance(node, ast.If) and not _inside(node, parents, ast.ExceptHandler):
            site = _classify_guard(node, parents, source, path, dispositions)
            if site is not None:
                sites.append(site)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            sites += [_classify_suppress(node, item.context_expr, parents, source, path, dispositions) for item in node.items
                      if isinstance(item.context_expr, ast.Call) and _type_name(item.context_expr.func) == SUPPRESS_CALL]
    return sorted(sites, key=lambda site: (site.line, site.kind))


def _classify_handler(handler: ast.ExceptHandler, parents: dict, source: str, path: str,
                      dispositions: list[Disposition]) -> Site:
    """Classes 1-8 for an except clause, first match wins."""
    body = handler.body
    error, warning = _has_log(body, ERROR_LEVELS), _has_log(body, WARNING_LEVELS)
    function = _function_name(handler, parents)
    citation = _citation(path, function, ast.get_source_segment(source, handler) or "", dispositions)
    if _ends_in_raise(body):
        klass = 2 if error else 1
    elif error and _deferred(handler, parents):
        klass = 3
    elif _catches_only(handler, CONTROL_FLOW_TYPES) and not error and not warning:
        klass = 4
    elif citation:
        klass = 5
    elif error and _has_any(body, (ast.Return,)):
        klass = 6
    elif warning and _has_any(body, (ast.Return, ast.Continue)):
        klass = 7
    else:
        klass = 8
    return Site(path, handler.lineno, handler.end_lineno, function, KIND_EXCEPT, klass, _caught(handler),
                citation if klass == 5 else CLASS_DISPOSITIONS.get(klass, ""))


def _classify_guard(node: ast.If, parents: dict, source: str, path: str, dispositions: list[Disposition]) -> Optional[Site]:
    """An if-block that logs at ERROR or WARNING and returns or continues without raising."""
    body = node.body
    if _has_any(body, (ast.Raise,)) or not any(isinstance(statement, (ast.Return, ast.Continue)) for statement in body):
        return None
    error, warning = _has_log(body, ERROR_LEVELS), _has_log(body, WARNING_LEVELS)
    if not (error or warning):
        return None
    function = _function_name(node, parents)
    citation = _citation(path, function, ast.get_source_segment(source, node) or "", dispositions)
    klass = 5 if citation else 6 if error else 7
    return Site(path, node.lineno, node.end_lineno, function, KIND_GUARD, klass, "ERROR" if error else "WARNING",
                citation if klass == 5 else "")


def _classify_suppress(node: ast.With | ast.AsyncWith, call: ast.Call, parents: dict, source: str, path: str,
                       dispositions: list[Disposition]) -> Site:
    """A suppress block as the except-pass clause of its types: class 4 for control-flow types only, else 5 or 8."""
    function = _function_name(node, parents)
    citation = _citation(path, function, ast.get_source_segment(source, node) or "", dispositions)
    control_flow = bool(call.args) and all(_type_name(arg) in CONTROL_FLOW_TYPES for arg in call.args)
    klass = 4 if control_flow else 5 if citation else 8
    return Site(path, node.lineno, node.end_lineno, function, KIND_SUPPRESS, klass,
                ", ".join(ast.unparse(arg) for arg in call.args), citation if klass == 5 else CLASS_DISPOSITIONS.get(klass, ""))


def _ends_in_raise(statements: list[ast.stmt]) -> bool:
    """Every path through the statements ends in raise (an escape from a nested block counts as a path without raise;
    a continue or break inside a loop of the statements ends that loop's iteration, not the statements)."""
    for statement in statements:
        if isinstance(statement, ast.Raise):
            return True
        if isinstance(statement, ESCAPES):
            return False
        if isinstance(statement, ast.If):
            if _leaves(statement.body + statement.orelse, ESCAPES):
                return False
            if statement.orelse and _ends_in_raise(statement.body) and _ends_in_raise(statement.orelse):
                return True
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            if _leaves(statement.body, ESCAPES):
                return False
            if _ends_in_raise(statement.body):
                return True
        elif isinstance(statement, (*LOOPS, ast.Try, ast.Match)):
            if _leaves([statement], ESCAPES):
                return False
    return False


def _deferred(handler: ast.ExceptHandler, parents: dict) -> bool:
    """A handler that records into a name and continues its loop, while a raise conditioned on that name follows the
    loop."""
    loop = _enclosing(handler, parents, LOOPS)
    if loop is None or _has_any(handler.body, (ast.Return,)) or not _leaves(handler.body, (ast.Continue,)):
        return False
    recorded = _recorded_names(handler.body)
    container = parents[loop]
    for field in ("body", "orelse", "finalbody"):
        siblings = getattr(container, field, None)
        if isinstance(siblings, list) and loop in siblings:
            return any(isinstance(node, ast.If) and _has_any(node.body, (ast.Raise,))
                       and recorded & {name.id for name in ast.walk(node.test) if isinstance(name, ast.Name)}
                       for node in _local_nodes(siblings[siblings.index(loop) + 1:]))
    return False


def _leaves(statements: list[ast.stmt], kinds: tuple) -> bool:
    """A statement of the kinds that leaves the statements: a return anywhere in their scope, a continue or break unless
    a loop body inside the statements holds it (a loop's else clause does not)."""
    for statement in statements:
        if isinstance(statement, kinds):
            return True
        if isinstance(statement, SCOPES):
            continue
        if isinstance(statement, LOOPS):
            returns = tuple(kind for kind in kinds if kind is ast.Return)
            if (returns and _leaves(statement.body, returns)) or _leaves(statement.orelse, kinds):
                return True
            continue
        if any(_leaves(block, kinds) for block in _blocks(statement)):
            return True
    return False


def _blocks(statement: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists a compound statement holds: bodies, else and finally clauses, handlers and match cases."""
    blocks = []
    for _, value in ast.iter_fields(statement):
        if not isinstance(value, list):
            continue
        if value and isinstance(value[0], ast.stmt):
            blocks.append(value)
        blocks += [item.body for item in value if isinstance(item, (ast.ExceptHandler, ast.match_case))]
    return blocks


def _recorded_names(statements: list[ast.stmt]) -> set[str]:
    """Names the statements record into: assignment targets (an item or attribute of a name included) and the receivers
    of their method calls other than logging calls."""
    names = set()
    for node in _local_nodes(statements):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
            names.add(node.value.id)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
              and node.func.attr not in LOG_METHODS):
            names.add(node.func.value.id)
    return names


def _catches_only(handler: ast.ExceptHandler, names: frozenset) -> bool:
    """The clause names only the given exception types."""
    if handler.type is None:
        return False
    caught = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return all(_type_name(node) in names for node in caught)


def _caught(handler: ast.ExceptHandler) -> str:
    """The clause's exception types as written."""
    return "bare" if handler.type is None else ast.unparse(handler.type)


def _type_name(node: ast.AST) -> str:
    """Name of an exception type expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _has_log(statements: list[ast.stmt], levels: frozenset) -> bool:
    """A logging call at one of the levels (x.error(...), x.log(logging.ERROR, ...))."""
    for node in _local_nodes(statements):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in levels:
                return True
            if node.func.attr == "log" and node.args and _type_name(node.args[0]).lower() in levels:
                return True
    return False


def _has_any(statements: list[ast.stmt], kinds: tuple) -> bool:
    """A node of the given kinds inside the statements, nested scopes excluded."""
    return any(isinstance(node, kinds) for node in _local_nodes(statements))


def _local_nodes(statements: list[ast.stmt]) -> Iterator[ast.AST]:
    """Every node of the statements without descending into nested functions or classes."""
    stack = list(statements)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, SCOPES))


def _enclosing(node: ast.AST, parents: dict, kinds: tuple) -> Optional[ast.AST]:
    """The nearest ancestor of the given kinds inside the node's own scope."""
    current = parents.get(node)
    while current is not None and not isinstance(current, SCOPES):
        if isinstance(current, kinds):
            return current
        current = parents.get(current)
    return None


def _inside(node: ast.AST, parents: dict, kind: type) -> bool:
    """The node lies inside an ancestor of the kind within its scope."""
    return _enclosing(node, parents, (kind,)) is not None


def _function_name(node: ast.AST, parents: dict) -> str:
    """Qualified name of the enclosing function (Class.method), or <module>."""
    names, current = [], parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        current = parents.get(current)
    return ".".join(reversed(names)) or "<module>"


def _citation(path: str, function: str, segment: str, dispositions: list[Disposition]) -> str:
    """The doc citation of a matching disposition, else empty."""
    return next((item.citation for item in dispositions
                 if item.path == path and function.split(".")[-1] == item.function and item.fragment in segment), "")


def _site_text(site: Site) -> str:
    """One logged line per site."""
    return (f"{site.path}:{site.line} {site.function} {site.kind}({site.caught}) class {site.klass} "
            f"{CLASS_NAMES[site.klass]} {'pass' if site.passes else 'fail'}{' - ' + site.disposition if site.disposition else ''}")


# =============================================================================
# TERMINOLOGY SCANNER
# =============================================================================


def load_terms(rules: Path) -> Optional[Terms]:
    """The dictionary terms, state and method names, forbidden and previously canonical names of the folder's
    atp_terminology.md, with the exclusions and control lines of its rule data; None when the folder lacks the rule
    data (the harness has then logged its NOTICE line)."""
    data = CTX.rule_data(rules)
    if data is None:
        return None
    text = CTX.rule_text(rules / TERMINOLOGY_RULE)
    sections = _rule_sections(text)
    found = []
    for line in sections["Canonical Dictionary"]:
        cells = line.split("|")
        if len(cells) > 2 and "`" in cells[1]:
            found += re.findall(r"`([^`]+)`", cells[1])
    for title in ("Trigger / Event State", "Variation Method Names", "Forbidden Terms",
                  "Previously-canonical (now forbidden) per-row value names"):
        for line in sections[title]:
            if "→" in line and not line.startswith("**Exempt"):
                found += re.findall(r"`([^`]+)`", re.sub(r"\(→[^)]*\)", "", line).split("→")[0])
    exempt_line = next(line for line in text.splitlines() if line.startswith("**Exempt"))
    exempt = tuple(term for term in re.findall(r"`([^`]+)`", exempt_line) if "_" in term)
    sequences = set()
    for term in found:
        words = _words(term.replace("*", ""))
        if words:
            sequences.add(tuple(words))
            if len(words) == 1 and words[0].endswith("s") and not words[0].endswith("ss"):
                sequences.add((words[0][:-1],))
    exclusions = tuple((frozenset(item["words"]), re.compile(item["pattern"]), item["reason"]) for item in data["exclusions"])
    controls = tuple((line, reported) for line, reported in data["controls"])
    CTX.log.info("terminology: %d term sequences %s, exempt %s", len(sequences), sorted(sequences), exempt)
    return Terms(tuple(sorted(sequences)), exempt, exclusions, controls)


def _rule_sections(text: str) -> dict[str, list[str]]:
    """Lines under each ## or ### heading of the rule file."""
    sections, current = {}, ""
    for line in text.splitlines():
        if line.startswith("#"):
            current = line.lstrip("#").strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def terminology_hits(source: str, path: str, terms: Terms) -> list[tuple]:
    """(path, line, term, status) of every term occurrence; ordinary uses marked excluded."""
    hits = []
    for number, line in enumerate(source.splitlines(), start=1):
        cleaned = line
        for exempt in terms.exempt:
            cleaned = re.sub(re.escape(exempt), " ", cleaned, flags=re.IGNORECASE)
        words = _words(cleaned)
        for sequence in terms.sequences:
            for start in range(len(words) - len(sequence) + 1):
                window = words[start:start + len(sequence)]
                if window[:-1] == list(sequence[:-1]) and window[-1] in _inflections(sequence[-1]):
                    hits.append((path, number, " ".join(window), _status(window[-1], line, terms.exclusions)))
    return hits


def _status(word: str, line: str, exclusions: tuple) -> str:
    """reported, or excluded with the reason of the first matching ordinary-language pattern."""
    for words, pattern, reason in exclusions:
        if (not words or word in words) and pattern.search(line):
            return f"excluded: {reason}"
    return "reported"


def _inflections(word: str) -> set[str]:
    """The word and its plain inflections: plus s, es, ed, d and ing (a final e dropped before ing and ed)."""
    forms = {word, f"{word}s", f"{word}es", f"{word}ed", f"{word}d", f"{word}ing"}
    if word.endswith("e"):
        forms |= {f"{word[:-1]}ing", f"{word[:-1]}ed"}
    return forms


def _words(text: str) -> list[str]:
    """Lowercase word tokens: split at non-alphanumerics, underscores and case changes."""
    words = []
    for chunk in re.findall(r"[A-Za-z0-9]+", text):
        words += [part.lower() for part in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", chunk)]
    return words


# =============================================================================
# SOURCE HELPERS
# =============================================================================


def _without_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Module statements after an optional docstring."""
    if statements and isinstance(statements[0], ast.Expr) and isinstance(getattr(statements[0], "value", None), ast.Constant):
        return statements[1:]
    return statements


def _unparsed(statements: list[ast.stmt]) -> tuple[str, ...]:
    """Normalised source of statements."""
    return tuple(ast.unparse(statement) for statement in statements)


def _tokens(source: str) -> list[tokenize.TokenInfo]:
    """Tokens of a module."""
    return list(tokenize.generate_tokens(io.StringIO(source).readline))


if __name__ == "__main__":
    main()
