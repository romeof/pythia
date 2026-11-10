"""EXP-26: structure readings of the step-3 audit (elegance and simplicity) over the files of one commit.

Every reading command takes the commit's files through git (git ls-tree, git cat-file --batch), never the working
tree, so a rerun at the same commit gives the same output; the tool's own file is left out of every scan. Outputs go to
the launch's folder data\\exp\\26\\tip\\<label>, tables as text files beside the script log.

Commands (one per launch):
    --references   every def and class of src, pipeline.py, scripts, tests and audit, every constants.py assignment
                   and every metrics.json key, with whole-word hit counts per file over the scanned roots (src, tests,
                   audit, scripts with the notebooks' cells, pipeline.py, workflows\\*.yaml, docs, README.md; string
                   literals included); the dynamic-use forms the tool recognises (dunder methods, overrides of a base
                   class's method, pydantic validators, the strategy class derived from its name); the names with no
                   reference in code beyond their definition (docs and README hits listed beside them), the src names
                   referenced only from tests or audit, the package re-exports no file imports through the package
                   (an importer that is itself an unused re-export does not count), the imported names whose binding no
                   load reaches (symtable's scopes), and the getattr sites and module:attribute strings for the
                   dead-code protocol
    --imports      the import graph of src and pipeline.py: module-level and in-function imports (a from-import of an
                   outside module recorded as the module and as each dotted name), cycles, whether an in-function import
                   would complete a cycle at module level, and per module the modules its import executes (package
                   __init__ files included) with the backend selection (any module holding a module-level
                   matplotlib.use call, whatever its argument), pyplot and hdf5plugin among them; a Mermaid graph per
                   package
    --topdown      per module and class, the functions defined before every caller in the same scope; for a module
                   with both a Structure docstring list and section banners, the two compared (a first section without
                   a banner aligned), and the definitions the list names against their definition sequence
    --cochange     every non-merge commit since --since but the shallow boundary: files and lines; file pairs by
                   co-change count, commits by file count, per-file churn, and per merged run its commits and files

Prototype checks (each reads commits through git and the launch records under data\\perf, never a checkout):
    --moved COMMIT OLD:NAME=NEW ...   with --base: each moved definition's ast.dump at the base against the commit,
                   and every global name it reads (symtable's global and free symbols of its tables) bound in both
                   modules by the same sequence of module-level bindings (try, if, with and for bodies and unpacking
                   targets included; a name bound only by star imports compares by their sources)
    --text BASE_A BASE_B CANDIDATE ... [--mask-with LABEL ...]   each candidate launch's stdout.log and unstamped
                   stderr.log against BASE_A's, aligned by difflib.SequenceMatcher (autojunk off); the mask = the
                   BASE_A lines unmatched against BASE_B or against any --mask-with base; a replace hunk of equal
                   length pairs its lines one to one and skips each pair whose BASE_A line is masked and whose
                   candidate line has the base line's shape (every launch label of the comparison as <label>, then
                   every digit run as #); every other unmatched line is reported; non-vacuous when the unmasked lines
                   are at least half of BASE_A's
    --importstate COMMIT [BASE_LABEL] [--declared MODULE ...]   per module of src and pipeline.py at the commit (git
                   archive into the label folder), a fresh child that imports it alone and records whether the import
                   raised (the type from the line after the last traceback's frames) and whether matplotlib,
                   matplotlib.pyplot and hdf5plugin are loaded, with the backend as stored
                   (get_backend(auto_select=False)); against BASE_LABEL's record while its tool blob is this one
                   (else the base commit is read again): a new raise fails, and so does a module that loads pyplot at
                   the commit with another plot state (matplotlib, pyplot, backend), except that a declared module's
                   is reported as declared; a declared module without that change fails; any other difference (an
                   hdf5plugin change included) is a named import-time change, a module at one commit only is listed
                   apart, and --module-moves OLD=NEW compares a moved module with the module it left
    --speed BASE ... -- CANDIDATE ...   every launch record completed with return code 0, else a failing verdict
                   naming it; per (mode, task) pair the logged seconds and the largest sampled RSS: medians, the base's
                   relative range, v = max(10 %, relative range) and the verdict, a pair without a sample reported
                   absent and a zero base median undefined (neither within v); contended launches named
    --self-test    planted sources, records and a scratch repository for every command, each planted case required
                   in the output and nothing else reported; importing perflaunch adds no logging handler and, in a
                   child, writes no file under data\\perf

Structure:
1. main
2. Tree reading
3. References
4. Imports
5. Top-down
6. Co-change
7. Prototype checks
8. Self-test
9. Shared helpers
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import ast
import builtins
import collections
import dataclasses
import difflib
import importlib.util
import itertools
import json
import logging
import re
import statistics
import symtable
import tarfile
from pathlib import Path
from typing import Optional

import perflaunch
from src.utils.common.common import variation_pct

EXPERIMENT = "26"
SELF_PATH = f"{harness.AUDIT_DIR}/{Path(__file__).name}"
DEFINITION_ROOTS = ("src/", "scripts/", "tests/", "audit/")
DEFINITION_FILES = ("pipeline.py",)
SCANNED_ROOTS = DEFINITION_ROOTS + ("docs/",)
SCANNED_FILES = DEFINITION_FILES + ("README.md",)
WORKFLOW_FILE = re.compile(r"^workflows/[^/]+\.yaml$")
OUTSIDE_PREFIXES = ("docs/", "README.md")
TEST_PREFIXES = ("tests/", "audit/")
PY_SUFFIX, NOTEBOOK_SUFFIX = ".py", ".ipynb"
CONSTANTS_FILE = "src/utils/common/constants.py"
METRICS_FILES = ("src/utils/performance/gradient_/metrics.json", "src/utils/performance/perseries_/metrics.json")
STRATEGIES_DIR = "src/utils/trigger/strategies/"
STRATEGY_SUFFIX = "Trigger"  # class name suffix: registered by schema.strategy_class
PACKAGE_FILE = "__init__.py"
SOURCE_PACKAGE = "src"
PIPELINE_MODULE = "pipeline"
TASK_MODULES = ("src.tasks.trigger", "src.tasks.signal", "src.tasks.performance", PIPELINE_MODULE)
BACKEND_CALL = "matplotlib.use"  # a module-level call of it selects the backend, whatever its argument
PYPLOT, HDF5PLUGIN = "matplotlib.pyplot", "hdf5plugin"
LITERAL_NAME = "Literal"  # typing's Literal[...]: its string items are values, not annotations
VALIDATOR_SUFFIX = "validator"  # pydantic's field_validator and model_validator
WORD = re.compile(r"\w+")
DUNDER = re.compile(r"^__\w+__$")
TARGET_STRING = re.compile(r"^[\w.]+:[\w.]+$")  # the module:attribute strings the regressions resolve
STRUCTURE_HEADING = "Structure:"
STRUCTURE_ITEM = re.compile(r"^\s*(\d+)\.\s+(.+)$")
BANNER_RULE = re.compile(r"^# ={5,}$")
BANNER_TITLE = re.compile(r"^# (?:\d+\.\s+)?(.+)$")
NUMBERED = re.compile(r"^\d+\.\s+")
LEADING_NAME = re.compile(r"^\s*([A-Za-z_]\w*)")
BATCH_FILE = "catfile.bin"
SINCE = "2026-05-23"
WIDE_COMMIT_FILES = 10
TOP_PAIRS = 30
TOP_CHURN = 30
RENAME_SUBJECT = re.compile(r"(?i)renam|terminolog|vocabular|wording|agnostic")
RECORD_MARK, FIELD_MARK = "\x1e", "\x1f"
RECORD_PLACEHOLDER, FIELD_PLACEHOLDER = "%x1e", "%x1f"  # git log --format writes the two marks for these
GIT_BOUND_S = 120.0
TEXT_FILES = (perflaunch.STDOUT_NAME, perflaunch.STDERR_NAME)
IMPORTSTATE_FILE = "importstate.json"
IMPORTSTATE_PREFIX = "IMPORTSTATE "
TRACEBACK_HEAD = "Traceback (most recent call last):"
RAISED_TYPE = re.compile(r"^([A-Za-z_][\w.]*)(:|$)")
NO_OUTPUT = "<no output>"
LABEL_SHAPE, DIGIT_SHAPE = "<label>", "#"
DIGIT_RUN = re.compile(r"[0-9]+")
ARCHIVE_FILE = "tree.tar"
ARCHIVED = ("src", "pipeline.py")
WATCHED_MODULES = ("matplotlib", PYPLOT, HDF5PLUGIN)
PLOT_STATE = ("matplotlib", PYPLOT, "backend")
CHILD_BOUND_S = 120.0
CHILD_HASHSEED = 0
BACKEND_ENV = "MPLBACKEND"  # removed from each import child, so the stored backend is the import's own choice
MIN_VARIATION_PCT = 10.0  # step 2's variation rule: v = max(10 %, relative range of the base launches)
PERCENT = 100.0
SPEED_METRICS = (("time", "times", "logged_s"), ("memory", "memory", "largest_sampled_rss_b"))
# the child that imports one module alone: a raise ends it with a traceback, read by its parent
IMPORT_CHILD = (
    "import sys\n"
    "sys.dont_write_bytecode = True\n"
    "import importlib, json, pathlib\n"
    "root, name = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]\n"
    "sys.path.insert(0, str(root))\n"
    "importlib.import_module(name)\n"
    "if 'src' in sys.modules and not pathlib.Path(sys.modules['src'].__file__).resolve().is_relative_to(root):\n"
    "    raise RuntimeError(f\"src imported from {sys.modules['src'].__file__}, not under {root}\")\n"
    "state = {item: item in sys.modules for item in sys.argv[3:]}\n"
    "state['backend'] = sys.modules['matplotlib'].get_backend(auto_select=False) if state['matplotlib'] else None\n"
    "sys.stdout.write('IMPORTSTATE ' + json.dumps(state) + '\\n')\n"
)
# the child that imports perflaunch alone (argv[1]: the audit folder), around which the self-test lists data\\perf
PERFLAUNCH_CHILD = "import sys\nsys.dont_write_bytecode = True\nsys.path.insert(0, sys.argv[1])\nimport perflaunch\n"


def main() -> None:
    """Run the one command the arguments name, then log the verdict summary."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    if args.self_test:
        self_test(folder)
    elif args.cochange:
        cochange(args.commit, args.since, folder, CTX.source_root)
    elif args.moved:
        moved(args.base, args.moved[0], args.moved[1:])
    elif args.text:
        text(args.text, args.mask_with)
    elif args.importstate:
        importstate(args.importstate[0], args.importstate[1] if len(args.importstate) > 1 else None,
                    args.module_moves, args.declared, folder)
    elif args.speed:
        speed(args.speed, args.candidates)
    else:
        files = read_tree(args.commit, folder)
        command = references if args.references else imports if args.imports else topdown
        command(files, folder)
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """One command; --commit for every reading command; --since for --cochange; --base for --moved;
    --module-moves and --declared for --importstate; the candidates after -- for --speed; --source-root for the
    harness."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    command = parser.add_mutually_exclusive_group(required=True)
    for name in ("--references", "--imports", "--topdown", "--cochange", "--self-test"):
        command.add_argument(name, action="store_true")
    command.add_argument("--moved", nargs="+", metavar="COMMIT OLD:NAME=NEW")
    command.add_argument("--text", nargs="+", metavar="LABEL")
    command.add_argument("--importstate", nargs="+", metavar="COMMIT [BASE_LABEL]")
    command.add_argument("--speed", nargs="+", metavar="BASE_LABEL")
    parser.add_argument("candidates", nargs="*", metavar="CANDIDATE_LABEL")
    parser.add_argument("--commit")
    parser.add_argument("--base")
    parser.add_argument("--module-moves", nargs="*", default=[], metavar="OLD=NEW")
    parser.add_argument("--declared", nargs="*", default=[], metavar="MODULE")
    parser.add_argument("--mask-with", nargs="*", default=[], metavar="LABEL")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    args = parser.parse_args()
    reading = args.references or args.imports or args.topdown or args.cochange
    if reading and args.commit is None:
        parser.error("every reading command needs --commit")
    if args.moved and (args.base is None or len(args.moved) < 2):
        parser.error("--moved needs --base, a commit and at least one OLD:NAME=NEW")
    if args.text and len(args.text) < 3:
        parser.error("--text needs two base labels and at least one candidate")
    if args.importstate and len(args.importstate) > 2:
        parser.error("--importstate takes a commit and at most one base label")
    if args.declared and not (args.importstate and len(args.importstate) == 2):
        parser.error("--declared needs --importstate with a base label")
    if args.speed and not args.candidates:
        parser.error("--speed needs candidate labels after --")
    return args


# =============================================================================
# 2. TREE READING
# =============================================================================


def read_tree(commit: str, folder: Path) -> dict[str, str]:
    """Text of every scanned file of the commit (a notebook as its cells' sources), by repository path."""
    full = _full_commit(commit)
    listing = _git("ls-tree", "-r", "-z", "--name-only", full)
    paths = sorted(path for path in listing.split("\0") if _scanned(path))
    batch = folder / BATCH_FILE
    harness.run_helper(["git", "cat-file", "--batch"], cwd=CTX.source_root, log=CTX.log, bound_s=GIT_BOUND_S,
                       stdin_text="".join(f"{full}:{path}\n" for path in paths), out_path=batch)
    raw, cursor, files = batch.read_bytes(), 0, {}
    for path in paths:
        header_end = raw.index(b"\n", cursor)
        header = raw[cursor:header_end].decode("ascii").split(" ")
        if len(header) != 3 or header[1] != "blob":
            raise RuntimeError(f"git cat-file gave {header} for {full}:{path}")
        start = header_end + 1
        content = raw[start:start + int(header[2])].decode("utf-8")
        files[path] = _notebook_text(content) if path.endswith(NOTEBOOK_SUFFIX) else content
        cursor = start + int(header[2]) + 1
    batch.unlink()
    CTX.log.info("TREE %s: %d files scanned (%s)", full, len(files), dict(collections.Counter(_root(p) for p in files)))
    return files


def _full_commit(commit: str) -> str:
    """The full hash a commit name resolves to in the source root."""
    return _git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()


def _git(*args: str, repo: Optional[Path] = None) -> str:
    """Output of a read-only git command in a repository, the source root unless one is named."""
    return harness.run_helper(["git", *args], cwd=repo or CTX.source_root, log=CTX.log, bound_s=GIT_BOUND_S)[1]


def _scanned(path: str) -> bool:
    """A file of the scanned roots, the tool's own file left out."""
    return path != SELF_PATH and (path.startswith(SCANNED_ROOTS) or path in SCANNED_FILES
                                  or bool(WORKFLOW_FILE.match(path)))


def _notebook_text(content: str) -> str:
    """The cells' sources of a notebook, joined, so that the JSON escapes never join two words."""
    return "\n".join("".join(cell["source"]) for cell in json.loads(content)["cells"])


def _root(path: str) -> str:
    """The first path component (the file name for a file at the top level)."""
    return path.split("/", 1)[0]


# =============================================================================
# 3. REFERENCES
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Definition:
    """One enumerated name: where it is defined and what kind of definition it is."""

    path: str
    qualname: str
    name: str
    line: int
    kind: str


def references(files: dict[str, str], folder: Path) -> dict:
    """The reference table, its candidates and the protocol's dynamic-use sites; returns the findings by kind."""
    counts = {path: collections.Counter(WORD.findall(text)) for path, text in files.items()}
    modules = {path: ast.parse(text, filename=path) for path, text in files.items() if _defines(path)}
    indexed = [item for path, tree in modules.items() for item in definitions(path, tree)]
    defined = [item for item, _, _ in indexed] + constants(modules.get(CONSTANTS_FILE)) + metric_keys(files)
    exports = reexports(modules)
    unused_exports = {(item["package_file"], item["name"]) for item in exports if not item["used"]}
    dynamic = dynamic_forms(modules, indexed)
    by_name = collections.defaultdict(list)
    for item in defined:
        by_name[item.name].append(item)
    rows, found = [], {"unreferenced": [], "dynamic": [], "tests_only": []}
    for item in defined:
        hits = _reference_hits(item.name, by_name[item.name], counts, unused_exports)
        code = {path: count for path, count in hits.items() if not path.startswith(OUTSIDE_PREFIXES)}
        outside = {path: count for path, count in hits.items() if path.startswith(OUTSIDE_PREFIXES)}
        production = {path: count for path, count in code.items() if not path.startswith(TEST_PREFIXES)}
        forms = dynamic.get(item, [])
        rows.append(f"{item.path}:{item.line}\t{item.kind}\t{item.qualname}\t{sum(code.values())}\t"
                    f"{len(by_name[item.name])}\t{';'.join(forms)}\t{_hit_text(code)}\t{_hit_text(outside)}")
        if not code:
            found["dynamic" if forms else "unreferenced"].append(item)
            CTX.log.info("%s %s:%d %s %s (defined %d times) dynamic=%s outside=%s",
                         "DYNAMIC" if forms else "UNREFERENCED", item.path, item.line, item.kind, item.qualname,
                         len(by_name[item.name]), forms, _hit_text(outside))
        elif item.path.startswith(SOURCE_PACKAGE + "/") and not production and not forms:
            found["tests_only"].append(item)
            CTX.log.info("TESTS-ONLY %s:%d %s %s hits=%s", item.path, item.line, item.kind, item.qualname,
                         _hit_text(code))
    _write(folder, "references.tsv", ["definition\tkind\tqualname\tcode_refs\tdefinitions_of_name\tdynamic\t"
                                      "code_hits\toutside_hits", *rows])
    for item in exports:
        CTX.log.info("REEXPORT %s %s from %s: %s, importers %s", item["package_file"], item["name"], item["source"],
                     "used" if item["used"] else "UNUSED", item["importers"])
    found["unused_exports"] = sorted(unused_exports)
    found["unused_imports"] = []
    for path, tree in sorted(modules.items()):
        for line, name, module_form in unused_imports(tree, files[path]):
            found["unused_imports"].append((path, name))
            CTX.log.info("UNUSED-IMPORT %s:%d %s%s", path, line, name, " (module import)" if module_form else "")
    _log_dynamic_sites(modules)
    per_root = collections.Counter(_root(item.path) for item in defined if item.kind not in ("constant", "metric"))
    CTX.log.info("DEFINITIONS %d def and class (%s), %d constants, %d metric keys", sum(per_root.values()),
                 dict(per_root), sum(item.kind == "constant" for item in defined),
                 sum(item.kind == "metric" for item in defined))
    CTX.log.info("CANDIDATES %d unreferenced, %d dynamic only, %d src names referenced only from tests or audit, "
                 "%d unused re-exports, %d unused imports", len(found["unreferenced"]), len(found["dynamic"]),
                 len(found["tests_only"]), len(unused_exports), len(found["unused_imports"]))
    CTX.verdict("references: non-vacuity", bool(defined) and bool(counts), "definitions and files > 0",
                {"definitions": len(defined), "files": len(counts)})
    return found


def definitions(path: str, tree: ast.Module) -> list[tuple[Definition, ast.AST, Optional[ast.ClassDef]]]:
    """Every def and class at any depth: its Definition (dotted name and kind: function, method, nested, class), its
    node and the class that owns it (None outside a class body)."""
    found = []

    def visit(node: ast.AST, scope: tuple, owner: Optional[ast.ClassDef]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if owner is not None else "nested" if scope else "function"
                found.append((Definition(path, ".".join(scope + (child.name,)), child.name, child.lineno, kind), child,
                              owner))
                visit(child, scope + (child.name,), None)
            elif isinstance(child, ast.ClassDef):
                found.append((Definition(path, ".".join(scope + (child.name,)), child.name, child.lineno, "class"),
                              child, owner))
                visit(child, scope + (child.name,), child)
            else:
                visit(child, scope, owner)

    visit(tree, (), None)
    return found


def constants(tree: Optional[ast.Module]) -> list[Definition]:
    """Every module-level assignment target of constants.py."""
    if tree is None:
        return []
    found = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            for name in (item for item in ast.walk(target) if isinstance(item, ast.Name)):
                found.append(Definition(CONSTANTS_FILE, name.id, name.id, node.lineno, "constant"))
    return found


def metric_keys(files: dict[str, str]) -> list[Definition]:
    """Every top-level key of the two metrics.json files."""
    found = []
    for path in METRICS_FILES:
        if path not in files:
            continue
        lines = files[path].splitlines()
        for key in json.loads(files[path]):
            line = next(number for number, text in enumerate(lines, start=1) if f'"{key}"' in text)
            found.append(Definition(path, key, key, line, "metric"))
    return found


def _reference_hits(name: str, sites: list[Definition], counts: dict, unused_exports: set) -> dict[str, int]:
    """Whole-word hits of a name per file beyond its definition sites: one token per def, class or constant site,
    every hit in the metrics.json files for a metric key, and every hit in a package file whose re-export of the
    name no file imports."""
    hits = {path: found[name] for path, found in counts.items() if found[name]}
    for site in sites:
        if site.kind == "metric":
            hits.pop(site.path, None)
        elif site.path in hits:
            hits[site.path] -= 1
    for package_file, exported in unused_exports:
        if exported == name:
            hits.pop(package_file, None)
    return {path: count for path, count in sorted(hits.items()) if count > 0}


def _hit_text(hits: dict[str, int]) -> str:
    """path:count items, comma-separated."""
    return ",".join(f"{path}:{count}" for path, count in hits.items())


def reexports(modules: dict[str, ast.Module]) -> list[dict]:
    """Every name a package __init__.py imports from inside its package and lists in __all__, with the files that
    import it through the package; an importer that is itself a package file counts only while its own re-export of
    the name is used (fixed point)."""
    names = {path: _module_name(path) for path in modules}
    found = []
    for path, tree in sorted(modules.items()):
        if not path.endswith(PACKAGE_FILE):
            continue
        listed = set(_exported(tree))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                source = _resolve_from(names[path], True, node.level, node.module)
                found += [{"package_file": path, "package": names[path], "name": alias.asname or alias.name,
                           "source": source, "importers": [], "used": False}
                          for alias in node.names if (alias.asname or alias.name) in listed]
    by_key = {(item["package"], item["name"]): item for item in found}
    for path, tree in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = _resolve_from(names[path], path.endswith(PACKAGE_FILE), node.level, node.module)
                for alias in node.names:
                    if (base, alias.name) in by_key:
                        by_key[(base, alias.name)]["importers"].append(path)
    changed = True
    while changed:
        changed = False
        for item in found:
            used = any(not importer.endswith(PACKAGE_FILE) or _reexport_used(by_key, names[importer], item["name"])
                       for importer in item["importers"])
            if used and not item["used"]:
                item["used"], changed = True, True
    return found


def _reexport_used(by_key: dict, package: str, name: str) -> bool:
    """True when a package file's own re-export of the name is used."""
    item = by_key.get((package, name))
    return item is not None and item["used"]


def unused_imports(tree: ast.Module, source: str) -> list[tuple[int, str, bool]]:
    """(line, bound name, module import) of every import whose binding no load reaches, as symtable resolves each
    scope's names. The binding is the module's for an import at module level or one its scope declares global, the
    nearest enclosing function's for one declared nonlocal, else the import's own scope's (_binding_table). A load
    reaches it in the scope that holds it, or in a scope nested there: for the module's binding where the name is
    global or local to a class body, an __all__ item and a name inside an annotation reaching it too; for a function's
    where the name is free with no function between binding it again (_loaded); from __future__ left out."""
    reached = set(_exported(tree)) | _annotation_names(tree)
    top = symtable.symtable(source, "<module>", "exec")
    found = []
    for node, scope in _scoped_imports(tree, ()):
        tables = _scope_tables(top, scope)
        if isinstance(node, ast.Import):
            bound = [(alias.asname or alias.name.split(".")[0], True) for alias in node.names]
        else:
            bound = [(alias.asname or alias.name, False) for alias in node.names
                     if alias.name != "*" and node.module != "__future__"]
        for name, module_form in bound:
            table = _binding_table(tables, name)
            if not (table is top and name in reached) and not _loaded(table, name):
                found.append((node.lineno, name, module_form))
    return sorted(found)


def _annotation_names(tree: ast.Module) -> set[str]:
    """Names inside the annotations of arguments, returns and annotated assignments, string annotations parsed (an
    annotation postponed by from __future__ import annotations holds no load symtable sees, yet names a module-level
    binding); the strings under a Literal[...] subscript are values and left out, and any other string that is no
    expression raises."""
    annotations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            arguments += [item for item in (node.args.vararg, node.args.kwarg) if item is not None]
            annotations += [item.annotation for item in arguments if item.annotation is not None]
            annotations += [node.returns] if node.returns is not None else []
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    texts = [text for annotation in annotations for text in _annotation_strings(annotation)]
    parsed = annotations + [ast.parse(text, mode="eval") for text in texts]
    return {item.id for node in parsed for item in ast.walk(node) if isinstance(item, ast.Name)}


def _annotation_strings(node: ast.AST) -> list[str]:
    """The string constants of an annotation, those under a Literal[...] subscript left out."""
    if isinstance(node, ast.Subscript) and (_dotted(node.value) or "").split(".")[-1] == LITERAL_NAME:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return [text for child in ast.iter_child_nodes(node) for text in _annotation_strings(child)]


def _scoped_imports(node: ast.AST, scope: tuple) -> list[tuple[ast.stmt, tuple]]:
    """Every import statement under a node, with the (name, line) chain of the defs and classes around it."""
    found = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            found.append((child, scope))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found += _scoped_imports(child, scope + ((child.name, child.lineno),))
        else:
            found += _scoped_imports(child, scope)
    return found


def _scope_tables(top: symtable.SymbolTable, scope: tuple) -> list[symtable.SymbolTable]:
    """The symbol tables from the module to the one a (name, line) chain of defs and classes ends at, each link the
    one child table of its parent with that name and line; a link that matches no table, or two, raises."""
    tables = [top]
    for name, line in scope:
        matches = [child for child in tables[-1].get_children()
                   if (child.get_name(), child.get_lineno()) == (name, line)]
        if len(matches) != 1:
            raise RuntimeError(f"{len(matches)} symbol tables named {name} at line {line}")
        tables.append(matches[0])
    return tables


def _binding_table(tables: list[symtable.SymbolTable], name: str) -> symtable.SymbolTable:
    """The table whose binding an import of a name sets, given the tables from the module to the import's scope: the
    module's for a name that scope declares global, the nearest enclosing function's that binds the name for one it
    declares nonlocal (symtable refuses a nonlocal without one), else the scope's own."""
    symbol = tables[-1].lookup(name)
    if symbol.is_declared_global():
        return tables[0]
    if not symbol.is_nonlocal():
        return tables[-1]
    return next(table for table in reversed(tables[:-1]) if table.get_type() == "function"
                and name in table.get_identifiers() and table.lookup(name).is_local())


def _loaded(table: symtable.SymbolTable, name: str) -> bool:
    """True when a load reaches the binding of a name in a table: a load in the table itself, or, for a module or a
    function (a class body's bindings reach no nested scope), in a nested table where symtable resolves the name to
    that binding. For a module: global, or local to a class body, which looks a name up in its own namespace and then
    in the module's, so reads the module's binding until its own runs (symtable records no sequence, so the load
    counts). For a function: free, the subtree of a function that binds the name again left out."""
    if table.lookup(name).is_referenced():
        return True
    kind = table.get_type()
    if kind == "class":
        return False
    pending = list(table.get_children())
    while pending:
        child = pending.pop()
        if name in child.get_identifiers():
            symbol = child.lookup(name)
            if kind == "module":
                reaches = symbol.is_global() or (child.get_type() == "class" and symbol.is_local())
            else:
                reaches = symbol.is_free()
            if symbol.is_referenced() and reaches:
                return True
            if kind == "function" and child.get_type() == "function" and symbol.is_local():
                continue
        pending += child.get_children()
    return False


def dynamic_forms(modules: dict[str, ast.Module],
                  indexed: list[tuple[Definition, ast.AST, Optional[ast.ClassDef]]]) -> dict[Definition, list[str]]:
    """The dynamic-use forms of each def and class: a dunder method, an override of a base class's method (a base
    of the tree by name, an imported one by import), a pydantic validator, the strategy class derived from its
    module's name."""
    classes = {node.name: (node, path) for path, tree in modules.items() for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef)}
    local = {SOURCE_PACKAGE} | {Path(path).stem for path in modules}
    tables = {path: _import_table(tree, local) for path, tree in modules.items()}
    prefixes = _getattr_prefixes(modules)
    found = {}
    for item, node, owner in indexed:
        forms = []
        if DUNDER.match(item.name):
            forms.append("dunder")
        if owner is not None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            forms += [f"override of {base}" for base in _overridden(item.name, owner, item.path, classes, tables)]
        if any(_decorator_name(decorator).endswith(VALIDATOR_SUFFIX) for decorator in getattr(node, "decorator_list", [])):
            forms.append("pydantic validator")
        if item.kind == "class" and item.path.startswith(STRATEGIES_DIR):
            module = Path(item.path).stem
            if item.name == "".join(part.capitalize() for part in module.split("_")) + STRATEGY_SUFFIX:
                forms.append("strategy class derived from trigger.strategy.strategy_name")
        forms += [f"getattr by prefix {prefix!r} ({item.path}:{line})" for prefix, line in prefixes[item.path]
                  if item.name.startswith(prefix)]
        if forms:
            found[item] = forms
    return found


def _getattr_prefixes(modules: dict[str, ast.Module]) -> dict[str, list[tuple[str, int]]]:
    """Per module, the constant prefix and line of every getattr whose name is an f-string starting with text."""
    found = collections.defaultdict(list)
    for path, tree in modules.items():
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and _dotted(node.func) == "getattr" and len(node.args) > 1
                    and isinstance(node.args[1], ast.JoinedStr) and node.args[1].values
                    and isinstance(node.args[1].values[0], ast.Constant)):
                found[path].append((node.args[1].values[0].value, node.lineno))
    return found


def _overridden(name: str, owner: ast.ClassDef, path: str, classes: dict, tables: dict) -> list[str]:
    """The bases of a method's class, walked transitively, that define the method's name: a base of the tree by its
    class name, an imported or builtin one by its resolved class; each base resolved with its own module's imports."""
    found, pending, seen = [], [(base, tables[path]) for base in owner.bases], set()
    while pending:
        base, imported = pending.pop(0)
        dotted = _dotted(base)
        if dotted is None or dotted in seen:
            continue
        seen.add(dotted)
        if dotted.split(".")[0] not in imported and dotted.split(".")[-1] in classes:
            node, base_path = classes[dotted.split(".")[-1]]
            if any(isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name for item in node.body):
                found.append(dotted)
            pending += [(item, tables[base_path]) for item in node.bases]
            continue
        resolved = _external_class(dotted, imported)
        if resolved is not None and hasattr(resolved, name):
            found.append(dotted)
    return found


def _import_table(tree: ast.Module, local: set) -> dict[str, tuple[str, list[str]]]:
    """Bound name -> (module to import, attribute path) of the module's imports from outside the tree (src and the
    tree's own top-level modules, such as harness and testkit, left out)."""
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            table.update({alias.asname or alias.name: (alias.name, []) for alias in node.names if "." not in alias.name})
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            table.update({alias.asname or alias.name: (node.module, [alias.name]) for alias in node.names})
    return {name: target for name, target in table.items() if target[0].split(".")[0] not in local}


def _external_class(dotted: str, imported: dict) -> Optional[object]:
    """The class an imported or builtin base names; None when it is neither or its module is not importable."""
    head, *rest = dotted.split(".")
    if head not in imported:
        return getattr(builtins, dotted, None) if not rest else None
    module, attributes = imported[head]
    if importlib.util.find_spec(module.split(".")[0]) is None:
        CTX.log.info("UNRESOLVED-BASE %s: module %s not importable here", dotted, module)
        return None
    resolved = importlib.import_module(module)
    for attribute in attributes + rest:
        resolved = getattr(resolved, attribute, None)
    return resolved


def _decorator_name(node: ast.AST) -> str:
    """The dotted name of a decorator, its call stripped ("" for any other expression)."""
    return _dotted(node.func if isinstance(node, ast.Call) else node) or ""


def _log_dynamic_sites(modules: dict[str, ast.Module]) -> None:
    """Every getattr call with its arguments and every module:attribute string, for the dead-code protocol."""
    for path, tree in sorted(modules.items()):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) == "getattr":
                CTX.log.info("GETATTR %s:%d %s", path, node.lineno, ast.unparse(node))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and TARGET_STRING.match(node.value):
                CTX.log.info("TARGET-STRING %s:%d %s", path, node.lineno, node.value)


# =============================================================================
# 4. IMPORTS
# =============================================================================


@dataclasses.dataclass(frozen=True)
class ImportSite:
    """One imported module of a module: the target, its line and the function it runs in (None at module level)."""

    module: str
    target: str
    line: int
    function: Optional[str]


def imports(files: dict[str, str], folder: Path) -> dict:
    """The import graph of src and pipeline.py; returns the cycles, in-function imports and closures."""
    paths = {_module_name(path): path for path in files
             if path.endswith(PY_SUFFIX) and (path.startswith(SOURCE_PACKAGE + "/") or path == DEFINITION_FILES[0])}
    known = set(paths)
    trees = {module: ast.parse(files[path]) for module, path in sorted(paths.items())}
    sites = [site for module, tree in trees.items()
             for site in import_sites(module, paths[module].endswith(PACKAGE_FILE), tree, known)]
    backend = sorted(module for module, tree in trees.items() if _selects_backend(tree))
    CTX.log.info("BACKEND %d modules hold a module-level %s call: %s", len(backend), BACKEND_CALL, backend)
    internal = [site for site in sites if site.target in known]
    edges = collections.defaultdict(set)
    for site in internal:
        if site.function is None:
            edges[site.module] |= {target for target in [site.target, *_parents(site.target, known)]
                                   if not (site.module + ".").startswith(target + ".")}
    cycles = [sorted(group) for group in _components(known, edges) if len(group) > 1
              or any(module in edges[module] for module in group)]
    for group in cycles:
        CTX.log.info("CYCLE %s", group)
    found = {"cycles": cycles, "in_function": [], "closures": {}}
    for site in (site for site in sites if site.function is not None):
        cyclic = site.target in known and site.module in _reach(site.target, edges)
        found["in_function"].append((site.module, site.function, site.target, cyclic))
        CTX.log.info("IN-FUNCTION %s:%d in %s imports %s; at module level it would %s", paths[site.module], site.line,
                     site.function, site.target, "complete a cycle" if cyclic else "complete no cycle")
    external = collections.defaultdict(set)
    for site in sites:
        if site.function is None and site.target not in known:
            external[site.module].add(site.target)
    rows = []
    for module in sorted(known):
        loaded = _closure(module, edges, known)
        watched = {"agg": not loaded.isdisjoint(backend),
                   "pyplot": any(PYPLOT in external[item] for item in loaded),
                   "matplotlib": any(target.split(".")[0] == "matplotlib" for item in loaded for target in external[item]),
                   "hdf5plugin": any(HDF5PLUGIN in external[item] for item in loaded)}
        found["closures"][module] = {"loaded": sorted(loaded), **watched}
        rows.append(f"{module}\t{len(loaded)}\t" + "\t".join(str(value) for value in watched.values()))
        if module in TASK_MODULES:
            CTX.log.info("CLOSURE %s: %d modules, %s: %s", module, len(loaded), watched, sorted(loaded))
    _write(folder, "closures.tsv", ["module\tmodules_loaded\tagg\tpyplot\tmatplotlib\thdf5plugin", *rows])
    _write(folder, "imports.tsv", ["module\ttarget\tline\tfunction"] + [
        f"{site.module}\t{site.target}\t{site.line}\t{site.function or ''}" for site in sites])
    packages = {name for name, path in paths.items() if path.endswith(PACKAGE_FILE)}
    _write(folder, "imports.mmd", mermaid(internal, packages))
    CTX.verdict("imports: non-vacuity", bool(internal), "internal imports > 0", len(internal))
    return found


def import_sites(module: str, is_package: bool, tree: ast.Module, known: set) -> list[ImportSite]:
    """Every import of a module, resolved: a from-import of a submodule of the tree names the submodule; a
    from-import of an outside module names the module and each dotted name, since any of them may be a submodule
    (from matplotlib import pyplot)."""
    found = []

    def visit(node: ast.AST, function: Optional[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, function or child.name)
            elif isinstance(child, ast.Import):
                found.extend(ImportSite(module, alias.name, child.lineno, function) for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                base = _resolve_from(module, is_package, child.level, child.module)
                dotted = [f"{base}.{alias.name}" for alias in child.names if alias.name != "*"]
                if base.split(".")[0] in (SOURCE_PACKAGE, PIPELINE_MODULE):
                    targets = {name if name in known else base for name in dotted} or {base}
                else:
                    targets = {base, *dotted}
                found.extend(ImportSite(module, target, child.lineno, function) for target in sorted(targets))
            else:
                visit(child, function)

    visit(tree, None)
    return found


def _selects_backend(tree: ast.Module) -> bool:
    """True when a statement of the module body calls matplotlib.use, whatever its argument."""
    return any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
               and _dotted(node.value.func) == BACKEND_CALL for node in tree.body)


def _parents(module: str, known: set) -> list[str]:
    """The regular packages an import of the module executes first (a package without __init__.py runs nothing)."""
    parts = module.split(".")
    return [".".join(parts[:index]) for index in range(1, len(parts)) if ".".join(parts[:index]) in known]


def _closure(module: str, edges: dict, known: set) -> set[str]:
    """The modules an import of the module executes: its packages, then every module-level target transitively."""
    loaded = set(_parents(module, known)) | {module}
    pending = list(loaded)
    while pending:
        for target in edges[pending.pop()]:
            if target not in loaded:
                loaded.add(target)
                pending.append(target)
    return loaded


def _reach(module: str, edges: dict) -> set[str]:
    """Every module reachable from the module over module-level edges."""
    seen, pending = set(), [module]
    while pending:
        for target in edges[pending.pop()]:
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def _components(nodes: set, edges: dict) -> list[set]:
    """Strongly connected components (Tarjan, iterative; link = the smallest index reachable on the stack)."""
    index, link, stack, on_stack, groups, counter = {}, {}, [], set(), [], itertools.count()
    for root in sorted(nodes):
        if root in index:
            continue
        work = [(root, iter(sorted(edges[root])))]
        index[root] = link[root] = next(counter)
        stack.append(root)
        on_stack.add(root)
        while work:
            node, targets = work[-1]
            target = next(targets, None)
            if target is not None:
                if target not in index:
                    index[target] = link[target] = next(counter)
                    stack.append(target)
                    on_stack.add(target)
                    work.append((target, iter(sorted(edges[target]))))
                elif target in on_stack:
                    link[node] = min(link[node], index[target])
                continue
            work.pop()
            if work:
                link[work[-1][0]] = min(link[work[-1][0]], link[node])
            if link[node] == index[node]:
                group = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    group.add(member)
                    if member == node:
                        break
                groups.append(group)
    return groups


def mermaid(internal: list[ImportSite], packages: set[str]) -> list[str]:
    """A package-level graph: one node per package, one edge per importing package and target package with the
    count of module-level imports (in-function imports dotted)."""
    weights = collections.Counter()
    for site in internal:
        weights[(_package_of(site.module, packages), _package_of(site.target, packages),
                 site.function is not None)] += 1
    lines = ["graph LR"]
    for (source, target, in_function), count in sorted(weights.items()):
        if source != target:
            arrow = "-.->" if in_function else "-->"
            lines.append(f"    {_node(source)} {arrow}|{count}| {_node(target)}")
    return lines


def _package_of(module: str, packages: set[str]) -> str:
    """The package a module belongs to (a package file to its own package), the src prefix dropped; pipeline for
    pipeline.py."""
    parts = module.split(".")
    if module in packages and len(parts) > 1:
        return ".".join(parts[1:])
    return ".".join(parts[1:-1]) if len(parts) > 2 else module


def _node(package: str) -> str:
    """A Mermaid node id with the package as its label."""
    return f'{re.sub(r"[^A-Za-z0-9]", "_", package)}["{package}"]'


# =============================================================================
# 5. TOP-DOWN
# =============================================================================


def topdown(files: dict[str, str], folder: Path) -> dict:
    """Callees defined before every caller, per scope, and the Structure lists against the banners."""
    found = {"callees": [], "structure": []}
    rows = []
    for path, text in sorted(files.items()):
        if not _defines(path):
            continue
        tree = ast.parse(text)
        scopes = [("<module>", tree.body, False)] + [(node.name, node.body, True) for node in ast.walk(tree)
                                                     if isinstance(node, ast.ClassDef)]
        for scope, body, in_class in scopes:
            for callee, line, caller, caller_line in early_callees(body, in_class):
                found["callees"].append((path, scope, callee))
                rows.append(f"{path}:{line}\t{scope}\t{callee}\t{caller}\t{caller_line}")
        listed, banners = structure_list(tree), banner_titles(text)
        bannered = listed[1:] if len(listed) == len(banners) + 1 else listed  # a first section (main) has no banner
        differing = [(index + 1, left, right) for index, (left, right)
                     in enumerate(itertools.zip_longest(bannered, banners, fillvalue=""))
                     if listed and banners and not _same_title(left, right)]
        named = listed_definitions(listed, tree)
        if differing or named != sorted(named, key=lambda item: item[1]):
            found["structure"].append(path)
            CTX.log.info("STRUCTURE %s: %d listed, %d banners; differing %s; listed definitions %s", path, len(listed),
                         len(banners), differing, named)
    for row in rows:
        CTX.log.info("EARLY-CALLEE %s", row.replace("\t", " | "))
    per_root = collections.Counter(_root(path) for path, _, _ in found["callees"])
    CTX.log.info("TOPDOWN %d callees defined before every caller (%s) in %d modules; %d Structure lists differ "
                 "from their banners", len(rows), dict(per_root), len({path for path, _, _ in found["callees"]}),
                 len(found["structure"]))
    _write(folder, "topdown.tsv", ["callee\tscope\tname\tfirst_caller\tcaller_line", *rows])
    CTX.verdict("topdown: non-vacuity", any(_defines(path) for path in files), "modules > 0",
                sum(_defines(path) for path in files))
    return found


def early_callees(body: list[ast.stmt], in_class: bool) -> list[tuple[str, int, str, int]]:
    """(callee, line, first caller, its line) of every function or method of a scope that every one of its users in
    the scope follows; a use outside a def or class body (decorator, default, a statement of the module or of the
    class body, which runs at definition time and reads plain names in both) exempts the callee."""
    defined = {node.name: index for index, node in enumerate(body)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    users, exempt = collections.defaultdict(list), set()
    for index, node in enumerate(body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner = node.body
            outer = (node.decorator_list + (node.bases if isinstance(node, ast.ClassDef) else
                                            node.args.defaults + [item for item in node.args.kw_defaults if item]))
        else:
            inner, outer = [], [node]
        for name in _used_names(outer, False) & defined.keys():
            exempt.add(name)
        for name in _used_names(inner, in_class) & defined.keys():
            if name != getattr(node, "name", None):
                users[name].append((index, node.name, node.lineno))
    found = []
    for name, index in defined.items():
        if name in exempt or not users[name] or any(user_index < index for user_index, _, _ in users[name]):
            continue
        first = min(users[name])
        found.append((name, body[index].lineno, first[1], first[2]))
    return found


def _used_names(nodes: list[ast.AST], in_class: bool) -> set[str]:
    """Names the given nodes use: plain names, or self.x and cls.x attributes inside the methods of a class."""
    found = set()
    for node in (item for root in nodes for item in ast.walk(root)):
        if not in_class and isinstance(node, ast.Name):
            found.add(node.id)
        elif in_class and isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in (
                "self", "cls"):
            found.add(node.attr)
    return found


def structure_list(tree: ast.Module) -> list[str]:
    """The numbered items after the Structure: line of the module docstring, each with its continuation lines."""
    lines = (ast.get_docstring(tree) or "").splitlines()
    stripped = [line.strip() for line in lines]
    if STRUCTURE_HEADING not in stripped:
        return []
    found = []
    for line in lines[stripped.index(STRUCTURE_HEADING) + 1:]:
        if not line.strip():
            break
        match = STRUCTURE_ITEM.match(line)
        if match:
            found.append(match[2].strip())
        elif found:
            found[-1] = f"{found[-1]} {line.strip()}"
    return found


def banner_titles(text: str) -> list[str]:
    """The titles of the section banners: a title line between two rule lines."""
    lines = text.splitlines()
    return [BANNER_TITLE.match(lines[index])[1].strip() for index in range(1, len(lines) - 1)
            if BANNER_RULE.match(lines[index - 1]) and BANNER_RULE.match(lines[index + 1])
            and BANNER_TITLE.match(lines[index])]


def _same_title(left: str, right: str) -> bool:
    """A Structure item and a banner title name the same section: one key starts the other."""
    left_key, right_key = _title_key(left), _title_key(right)
    return bool(left_key) and bool(right_key) and (left_key.startswith(right_key) or right_key.startswith(left_key))


def _title_key(title: str) -> str:
    """A title compared casefolded on its letters and digits, the item number and any parenthesis dropped."""
    return re.sub(r"[^0-9a-z]", "", re.sub(r"\(.*$", "", NUMBERED.sub("", title)).casefold())


def listed_definitions(listed: list[str], tree: ast.Module) -> list[tuple[str, int]]:
    """(name, definition line) of each module-level def or class the Structure list names, in the listed
    sequence (first mention)."""
    lines = {node.name: node.lineno for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    named = dict.fromkeys(word for item in listed for word in WORD.findall(item) if word in lines)
    return [(name, lines[name]) for name in named]


# =============================================================================
# 6. CO-CHANGE
# =============================================================================


def cochange(commit: str, since: str, folder: Path, repo: Path) -> dict:
    """Files per commit of a repository since the date, the shallow boundary left out; pairs, wide commits, churn,
    merged runs."""
    common = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir", repo=repo).strip())
    boundary = set((common / "shallow").read_text(encoding="utf-8").split()) if (common / "shallow").is_file() else set()
    commits = [item for item in _numstat_log(commit, since, repo, "--no-merges") if item["hash"] not in boundary]
    pairs, churn = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for item in commits:
        pairs.update(itertools.combinations(sorted(item["files"]), 2))
        for path, lines in item["files"].items():
            churn[path][0] += 1
            churn[path][1] += lines
    sizes = [len(item["files"]) for item in commits]
    CTX.log.info("COMMITS %d non-merge since %s (shallow boundary %s left out); files per commit median %s, mean %.1f",
                 len(commits), since, sorted(boundary), statistics.median(sizes) if sizes else None,
                 statistics.fmean(sizes) if sizes else 0.0)
    for item in sorted(commits, key=lambda found: -len(found["files"])):
        if len(item["files"]) >= WIDE_COMMIT_FILES:
            CTX.log.info("WIDE %s %s %d files%s: %s", item["hash"][:7], item["date"], len(item["files"]),
                         " (subject names a rename sweep)" if RENAME_SUBJECT.search(item["subject"]) else "",
                         item["subject"][:160])
    for (left, right), count in pairs.most_common(TOP_PAIRS):
        CTX.log.info("PAIR %d %s + %s", count, left, right)
    for path, (touched, lines) in sorted(churn.items(), key=lambda found: (-found[1][0], found[0]))[:TOP_CHURN]:
        CTX.log.info("CHURN %s: %d commits, %d lines", path, touched, lines)
    merges = _git("log", "--merges", "--first-parent", f"--format=%H{FIELD_PLACEHOLDER}%ad{FIELD_PLACEHOLDER}%s",
                  "--date=short", f"--since={since}", commit, repo=repo).splitlines()
    for merge in merges:
        merge_hash, date, subject = merge.split(FIELD_MARK)
        run = _numstat_log(f"{merge_hash}^1..{merge_hash}^2", None, repo, "--no-merges")
        files = set().union(*(item["files"] for item in run)) if run else set()
        wide = sorted(run, key=lambda found: -len(found["files"]))[:3]
        CTX.log.info("MERGED-RUN %s %s: %d commits, %d files; widest %s; %s", merge_hash[:7], date, len(run),
                     len(files), [(item["hash"][:7], len(item["files"])) for item in wide], subject[:100])
    _write(folder, "cochange.tsv", ["commit\tdate\tfiles\tlines\tsubject"] + [
        f"{item['hash'][:7]}\t{item['date']}\t{len(item['files'])}\t{sum(item['files'].values())}\t{item['subject']}"
        for item in commits])
    CTX.verdict("cochange: non-vacuity", bool(commits), "commits > 0", len(commits))
    return {"commits": commits, "pairs": pairs, "churn": {path: tuple(value) for path, value in churn.items()}}


def _numstat_log(revisions: str, since: Optional[str], repo: Path, *options: str) -> list[dict]:
    """hash, date, subject and {path: lines added and removed} of every commit git log lists, renames as paths."""
    argv = ["log", *options, "--no-renames", "--numstat", "--date=short",
            f"--format={RECORD_PLACEHOLDER}%H{FIELD_PLACEHOLDER}%ad{FIELD_PLACEHOLDER}%s"]
    argv += [f"--since={since}"] if since else []
    found = []
    for record in _git(*argv, revisions, repo=repo).split(RECORD_MARK)[1:]:
        header, *rows = record.strip("\n").split("\n")
        commit_hash, date, subject = header.split(FIELD_MARK)
        files = {}
        for row in (row for row in rows if row.strip()):
            added, removed, path = row.split("\t")
            files[path] = (0 if added == "-" else int(added)) + (0 if removed == "-" else int(removed))
        found.append({"hash": commit_hash, "date": date, "subject": subject, "files": files})
    return found


# =============================================================================
# 7. PROTOTYPE CHECKS
# =============================================================================


@dataclasses.dataclass(frozen=True)
class Move:
    """One moved definition: its path at the base, its dotted name, its path at the candidate."""

    old: str
    name: str
    new: str


def moved(base: str, commit: str, specs: list[str]) -> None:
    """--moved: the moved definitions and the global names they read, at the base against the commit."""
    moves = [_parse_move(spec) for spec in specs]
    before = {move.old: _show(base, move.old) for move in moves}
    after = {move.new: _show(commit, move.new) for move in moves}
    differences, counts = compare_moves(moves, before, after)
    for line in differences:
        CTX.log.info("MOVED-DIFF %s", line)
    CTX.log.info("MOVED %s at %s against %s: %s", [str(move) for move in moves], base, commit, counts)
    CTX.verdict("moved: non-vacuity", counts["definitions"] > 0 and counts["bindings"] > 0,
                "definitions and bindings compared > 0", counts)
    CTX.verdict("moved: definitions and the bindings they read identical", not differences, "no difference",
                differences)


def _parse_move(spec: str) -> Move:
    """OLD:NAME=NEW."""
    source, _, new = spec.partition("=")
    old, _, name = source.rpartition(":")
    return Move(old, name, new)


def _show(commit: str, path: str) -> str:
    """A file's text at a commit."""
    return _git("show", f"{commit}:{path}")


def compare_moves(moves: list[Move], before: dict[str, str], after: dict[str, str]) -> tuple[list[str], dict]:
    """Each move's definition dump at the base against the candidate, and the binding of every global name it
    reads in its old and its new module."""
    trees = {("base", path): ast.parse(text) for path, text in before.items()}
    trees.update({("candidate", path): ast.parse(text) for path, text in after.items()})
    differences, counts = [], {"definitions": 0, "bindings": 0}
    for move in moves:
        old_tree, new_tree = trees[("base", move.old)], trees[("candidate", move.new)]
        old_node, new_node = _find_definition(old_tree, move.name), _find_definition(new_tree, move.name)
        if old_node is None or new_node is None:
            differences.append(f"{move.name}: absent at the {'base' if old_node is None else 'candidate'}")
            continue
        counts["definitions"] += 1
        if ast.dump(old_node) != ast.dump(new_node):
            differences.append(f"{move.name}: its definition differs")
        bound = _bound_builtins(old_tree, move.old) | _bound_builtins(new_tree, move.new)
        for name in sorted(_global_reads(old_node, bound) | _global_reads(new_node, bound)):
            counts["bindings"] += 1
            left, right = _binding(name, old_tree, move.old), _binding(name, new_tree, move.new)
            if left != right:
                differences.append(f"{move.name}: {name} bound to {str(left)[:160]} at the base, "
                                   f"{str(right)[:160]} at the candidate")
    return differences, counts


def _find_definition(tree: ast.Module, dotted: str) -> Optional[ast.AST]:
    """The def or class a dotted name designates (Class.method for a method)."""
    body, node = tree.body, None
    for part in dotted.split("."):
        node = next((item for item in body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and item.name == part), None)
        if node is None:
            return None
        body = node.body
    return node


def _global_reads(node: ast.AST, kept: frozenset[str] = frozenset()) -> set[str]:
    """Names a definition reads from its module, as symtable classes them in the definition's own source: every
    referenced symbol of the definition's tables (its own and every nested one) that is global, or free while no
    function table of the definition binds it, and every name the definition's statement reads where it stands
    (decorators, defaults, bases, annotations); builtins left out, save the ones a module binds (kept)."""
    top = symtable.symtable(ast.unparse(node), "<definition>", "exec")
    found = {symbol.get_name() for symbol in top.get_symbols()
             if symbol.is_referenced() and symbol.get_name() != getattr(node, "name", None)}
    pending = [(table, frozenset()) for table in top.get_children()]
    while pending:
        table, enclosing = pending.pop()
        for symbol in table.get_symbols():
            free = symbol.is_free() and symbol.get_name() not in enclosing
            if symbol.is_referenced() and (symbol.is_global() or free):
                found.add(symbol.get_name())
        bound = enclosing | ({symbol.get_name() for symbol in table.get_symbols() if symbol.is_local()}
                             if table.get_type() == "function" else frozenset())
        pending += [(child, bound) for child in table.get_children()]
    return found - (set(dir(builtins)) - kept)


def _bound_builtins(tree: ast.Module, path: str) -> frozenset[str]:
    """The builtin names a module binds itself: an import, definition or assignment _binding finds."""
    return frozenset(name for name in dir(builtins) if _binding(name, tree, path)[0][0] not in ("star", "unbound"))


def _binding(name: str, tree: ast.Module, path: str) -> tuple:
    """Every module-level binding of a name in source sequence, the bodies of module-level try, if, with and for
    statements included: an import by the module it binds and its absolute target (import a.b binds a and imports
    a.b), a definition by its dump, an assignment by its value's dump (an unpacking target by the whole target, an
    augmented assignment by its operator), a for or with target by its source expression; a name bound by none of them
    compares as the sources of the module's star imports, and ("unbound",) when the module has none."""
    module, is_package = _module_name(path), path.endswith(PACKAGE_FILE)
    found, stars = [], []
    for node in _module_statements(tree.body):
        if isinstance(node, ast.Import):
            found += [("import", alias.name if alias.asname else alias.name.split(".")[0], alias.name)
                      for alias in node.names if (alias.asname or alias.name.split(".")[0]) == name]
        elif isinstance(node, ast.ImportFrom):
            source = _resolve_from(module, is_package, node.level, node.module)
            stars += [("star", source) for alias in node.names if alias.name == "*"]
            found += [("from", source, alias.name) for alias in node.names if (alias.asname or alias.name) == name]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            found.append(("definition", node.name, ast.dump(node)))
        elif isinstance(node, ast.Assign):
            found += [("assignment", ast.dump(target), ast.dump(node.value)) for target in node.targets
                      if name in _target_names(target)]
        elif isinstance(node, ast.AnnAssign) and name in _target_names(node.target):
            found.append(("assignment", name, ast.dump(node.value) if node.value is not None else None))
        elif isinstance(node, ast.AugAssign) and name in _target_names(node.target):
            found.append(("augmented", name, ast.dump(node.op), ast.dump(node.value)))
        elif isinstance(node, ast.For) and name in _target_names(node.target):
            found.append(("for", ast.dump(node.target), ast.dump(node.iter)))
        elif isinstance(node, ast.With):
            found += [("with", ast.dump(item.optional_vars), ast.dump(item.context_expr)) for item in node.items
                      if item.optional_vars is not None and name in _target_names(item.optional_vars)]
    return tuple(found or stars or [("unbound",)])


def _module_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """The statements of a module body in source sequence, each module-level try, if, with or for followed by the
    statements of its bodies (handlers, else and finally included)."""
    found = []
    for node in body:
        found.append(node)
        if isinstance(node, ast.Try):
            parts = [node.body, *(handler.body for handler in node.handlers), node.orelse, node.finalbody]
        elif isinstance(node, (ast.If, ast.For)):
            parts = [node.body, node.orelse]
        elif isinstance(node, ast.With):
            parts = [node.body]
        else:
            parts = []
        for part in parts:
            found += _module_statements(part)
    return found


def _target_names(target: ast.AST) -> set[str]:
    """The names an assignment target binds: every stored name, so each name of an unpacking target and none of an
    attribute or subscript target."""
    return {item.id for item in ast.walk(target) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}


def text(labels: list[str], mask_with: list[str]) -> None:
    """--text: per printed file, the mask from the two bases (joined by the lines each --mask-with base shows
    varying against the first), then every candidate against the first base."""
    base_a, base_b, *candidates = labels
    compared_labels = labels + mask_with
    for name in TEXT_FILES:
        lines = {label: _launch_lines(label, name) for label in compared_labels}
        mask = text_mask(lines, base_a, [base_b, *mask_with])
        for index in sorted(mask):
            CTX.log.info("MASKED %s %s:%d %s", name, base_a, index + 1, lines[base_a][index][:200])
        compared = len(lines[base_a]) - len(mask)
        CTX.verdict(f"text {name}: non-vacuity (unmasked lines at least half of {base_a}'s)",
                    compared > 0 and 2 * compared >= len(lines[base_a]), f">= {len(lines[base_a]) / 2:.0f}",
                    {"lines": len(lines[base_a]), "masked": len(mask), "compared": compared})
        for candidate in candidates:
            unmasked = unmasked_differences(lines[base_a], lines[candidate], mask, compared_labels)
            for line in unmasked:
                CTX.log.info("UNMASKED %s %s %s", name, candidate, line)
            CTX.verdict(f"text {name}: {candidate} against {base_a}", not unmasked, "no unmasked difference",
                        f"{len(unmasked)} unmasked lines")


def _launch_lines(label: str, name: str) -> list[str]:
    """The lines of a launch's stdout.log, or of its stderr.log with the arrival stamps stripped."""
    path = perflaunch.PERF_DIR / label / name
    if name == perflaunch.STDERR_NAME:
        return [line for _, line in perflaunch._stamped_lines(path)]
    return path.read_text(encoding="utf-8").splitlines()


def text_mask(lines: dict[str, list[str]], base_a: str, others: list[str]) -> set[int]:
    """Indices of the base_a lines unmatched against any of the other bases."""
    return set().union(*(masked_lines(lines[base_a], lines[other]) for other in others))


def masked_lines(base_a: list[str], base_b: list[str]) -> set[int]:
    """Indices of the base_a lines unmatched against base_b."""
    matcher = difflib.SequenceMatcher(None, base_a, base_b, autojunk=False)
    return {index for tag, first, last, _, _ in matcher.get_opcodes() if tag != "equal"
            for index in range(first, last)}


def unmasked_differences(base_a: list[str], candidate: list[str], mask: set[int], labels: list[str]) -> list[str]:
    """Every unmatched line of an alignment, except that a replace hunk of equal length pairs its lines one to one
    and skips each pair whose base_a line is masked and whose candidate line has the base line's shape."""
    matcher = difflib.SequenceMatcher(None, base_a, candidate, autojunk=False)
    found = []
    for tag, first, last, other_first, other_last in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and last - first == other_last - other_first:
            for index, other in zip(range(first, last), range(other_first, other_last)):
                if index not in mask or _shape(base_a[index], labels) != _shape(candidate[other], labels):
                    found += [f"-{index + 1}: {base_a[index][:200]}", f"+{other + 1}: {candidate[other][:200]}"]
            continue
        found += [f"-{index + 1}: {base_a[index][:200]}" for index in range(first, last)]
        found += [f"+{index + 1}: {candidate[index][:200]}" for index in range(other_first, other_last)]
    return found


def _shape(line: str, labels: list[str]) -> str:
    """A line with every launch label replaced by <label> (longest first), then every digit run by #: labels
    first, so that base-1 never becomes base-#."""
    for label in sorted(labels, key=len, reverse=True):
        line = line.replace(label, LABEL_SHAPE)
    return DIGIT_RUN.sub(DIGIT_SHAPE, line)


def importstate(commit: str, base_label: Optional[str], module_moves: list[str], declared: list[str],
                folder: Path) -> None:
    """--importstate: the commit's record; against the base label's record when one is named, with the declared
    modules' plot-state changes admitted."""
    blob = harness.copy_blob(CTX.roots, CTX.log, Path(__file__), SELF_PATH)
    full = _full_commit(commit)
    record = {"commit": full, "tool_blob": blob, "records": import_records(full, folder / "candidate")}
    (folder / IMPORTSTATE_FILE).write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    CTX.log.info("IMPORTSTATE-RECORD %s: %d modules, tool blob %s", full, len(record["records"]), blob)
    CTX.verdict("importstate: non-vacuity", bool(record["records"]), "modules recorded > 0", len(record["records"]))
    if base_label is None:
        return
    base = json.loads((folder.parent / base_label / IMPORTSTATE_FILE).read_text(encoding="utf-8"))
    if base["tool_blob"] != blob:
        CTX.log.info("IMPORTSTATE-BASE %s names tool blob %s, not %s: %s read again", base_label, base["tool_blob"],
                     blob, base["commit"])
        base = {**base, "tool_blob": blob, "records": import_records(base["commit"], folder / "base")}
    CTX.log.info("IMPORTSTATE-BASE %s at %s, tool blob %s", base_label, base["commit"], base["tool_blob"])
    found = judge_import_states(base["records"], record["records"], dict(item.split("=") for item in module_moves),
                                frozenset(declared))
    for kind in ("failing", "declared", "named", "apart"):
        for line in found[kind]:
            CTX.log.info("IMPORTSTATE-%s %s", kind.upper(), line)
    CTX.verdict("importstate: no import raises at the candidate that did not at the base, every module that loads "
                "pyplot keeps its plot state unless declared, and every declared module loads pyplot at the candidate "
                "with another plot state", not found["failing"], "none failing", found["failing"])


def import_records(commit: str, folder: Path) -> dict[str, dict]:
    """The import record of every module of src and pipeline.py at a commit, from its archived tree."""
    folder.mkdir(parents=True)
    archive = folder / ARCHIVE_FILE
    harness.run_helper(["git", "archive", "--format=tar", "-o", str(archive), commit, "--", *ARCHIVED],
                       cwd=CTX.source_root, log=CTX.log, bound_s=GIT_BOUND_S)
    tree = folder / "tree"
    with tarfile.open(archive) as bundle:
        bundle.extractall(tree, filter="data")
    archive.unlink()
    names = sorted(_module_name(path.relative_to(tree).as_posix()) for path in tree.rglob(f"*{PY_SUFFIX}"))
    return {name: module_record(tree, name) for name in names}


def module_record(tree: Path, name: str) -> dict:
    """One module imported alone by a fresh child: the loaded modules and backend, or the type it raised."""
    child = CTX.launch_child("import", ["-c", IMPORT_CHILD, str(tree), name, *WATCHED_MODULES],
                             timeout_s=CHILD_BOUND_S, hashseed=CHILD_HASHSEED, env_removals=(BACKEND_ENV,))
    if child.timed_out:
        raise RuntimeError(f"the import child of {name} timed out: {child.log_path.name}")
    if child.return_code != 0:
        return {"raised": raised_type(child.text, child.log_path.name)}
    line = next(line for line in child.text.splitlines() if line.startswith(IMPORTSTATE_PREFIX))
    return json.loads(line[len(IMPORTSTATE_PREFIX):])


def raised_type(text: str, log_name: str) -> str:
    """The exception type a failed child printed: the first unindented line after the frame lines of its last
    traceback, read up to its colon, so a message's continuation lines are never taken for the type; NO_OUTPUT when
    the child printed nothing. Output without a traceback, or a type line of another shape, raises."""
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return NO_OUTPUT
    heads = [index for index, line in enumerate(lines) if line.startswith(TRACEBACK_HEAD)]
    if not heads:
        raise RuntimeError(f"the failed import child printed no traceback: {log_name}")
    after = next((line for line in lines[heads[-1] + 1:] if line and not line[0].isspace()), "")
    match = RAISED_TYPE.match(after)
    if match is None:
        raise RuntimeError(f"no exception type after the last traceback of {log_name}: {after[:200]!r}")
    return match[1]


def judge_import_states(base: dict, candidate: dict, moves: dict[str, str],
                        declared: frozenset = frozenset()) -> dict[str, list[str]]:
    """Each module's candidate record against its base record (a moved module against the module it left): a new
    raise fails; a module that loads pyplot at the candidate with another plot state (matplotlib, pyplot, backend)
    fails unless it is declared, when it is listed as declared; a declared module without such a change fails, so a
    declaration never passes vacuously; any other difference (an hdf5plugin change included) is a named import-time
    change; a module at one commit only is listed apart."""
    origin = {moves.get(name, name): name for name in base}
    found = {"failing": [], "declared": [], "named": [], "apart": []}
    for name in sorted(set(candidate) | set(origin) | declared):
        if name not in candidate or name not in origin:
            where = "the candidate" if name in candidate else "the base" if name in origin else "neither commit"
            found["failing" if name in declared else "apart"].append(
                f"{name} only at {where}" + (" though declared" if name in declared else ""))
            continue
        before, after = base[origin[name]], candidate[name]
        line = f"{name} (base {origin[name]}): {before} -> {after}"
        plot_changed = [before.get(key) for key in PLOT_STATE] != [after.get(key) for key in PLOT_STATE]
        if before == after:
            if name in declared:
                found["failing"].append(f"declared but unchanged: {name}")
        elif after.get("raised") and not before.get("raised"):
            found["failing"].append(f"raises: {line}")
        elif after.get(PYPLOT) and plot_changed:
            found["declared" if name in declared else "failing"].append(
                line if name in declared else f"loads pyplot with another record: {line}")
        elif name in declared:
            found["failing"].append(f"declared but no pyplot plot-state change: {line}")
        else:
            found["named"].append(line)
    return found


def speed(base_labels: list[str], candidate_labels: list[str]) -> None:
    """--speed: every launch completed with return code 0; per (mode, task) pair, time and memory medians against
    the base's variation."""
    base = {label: _summary(label) for label in base_labels}
    candidate = {label: _summary(label) for label in candidate_labels}
    for label, record in {**base, **candidate}.items():
        if record["cpu"]["contended"]:
            CTX.log.info("CONTENDED %s: other cores %.2f", label, record["cpu"]["other_cores"])
    incomplete = incomplete_launches({**base, **candidate})
    CTX.verdict("speed: every launch completed with return code 0", not incomplete, "none incomplete", incomplete)
    rows = speed_rows(list(base.values()), list(candidate.values()))
    for row in rows:
        CTX.log.info("SPEED %s", row)
    CTX.verdict("speed: non-vacuity", bool(rows), "pairs compared > 0", len(rows))
    CTX.verdict("speed: every pair within v", all(row["within"] for row in rows), "all within v",
                [f"{row['metric']} {row['pair']}" for row in rows if not row["within"]])


def _summary(label: str) -> dict:
    """A launch record of perflaunch."""
    return json.loads((perflaunch.PERF_DIR / label / perflaunch.SUMMARY_NAME).read_text(encoding="utf-8"))


def incomplete_launches(records: dict[str, dict]) -> list[str]:
    """The labels, with outcome and return code, of the launch records that did not complete with return code 0."""
    return [f"{label}: outcome {record['outcome']}, return code {record['return_code']}"
            for label, record in records.items()
            if record["outcome"] != perflaunch.OUTCOME_COMPLETED or record["return_code"] != 0]


def speed_rows(base: list[dict], candidate: list[dict]) -> list[dict]:
    """Per metric and (mode, task) pair: the medians, the base's relative range, v and whether the candidate's
    median lies within v of the base's; a pair missing from a side or holding a value None is absent, and a zero
    base median makes the change undefined, neither within v."""
    rows = []
    for metric, section, field in SPEED_METRICS:
        before, after = _pair_values(base, section, field), _pair_values(candidate, section, field)
        for pair in sorted(set(before) | set(after)):
            if pair not in before or pair not in after or None in before[pair] or None in after[pair]:
                rows.append({"metric": metric, "pair": pair, "base": before.get(pair), "candidate": after.get(pair),
                             "within": False, "absent": True})
                continue
            reference, median = statistics.median(before[pair]), statistics.median(after[pair])
            if reference == 0:
                rows.append({"metric": metric, "pair": pair, "base": before[pair], "candidate": after[pair],
                             "within": False, "undefined": True})
                continue
            relative_range = (max(before[pair]) - min(before[pair])) / reference * PERCENT
            bound = max(MIN_VARIATION_PCT, relative_range)
            change = variation_pct(median, reference)
            rows.append({"metric": metric, "pair": pair, "base": before[pair], "candidate": after[pair],
                         "base_median": reference, "candidate_median": median,
                         "relative_range_pct": round(relative_range, 2), "v_pct": round(bound, 2),
                         "change_pct": round(change, 2), "within": abs(change) <= bound})
    return rows


def _pair_values(records: list[dict], section: str, field: str) -> dict[str, list[float]]:
    """mode/task -> the field's value in each record."""
    found = collections.defaultdict(list)
    for record in records:
        for pair in record[section]["pairs"]:
            found[f"{pair['mode']}/{pair['task']}"].append(pair[field])
    return found


# =============================================================================
# 8. SELF-TEST
# =============================================================================

PLANTED = {
    "src/__init__.py": "",
    "src/kit/__init__.py": "from .core import exported_only, used_function\n\n__all__ = ['used_function', 'exported_only']\n",
    "src/kit/core.py": (
        '"""Planted module.\n\nStructure:\n1. First part\n2. Second part\n"""\n'
        "import logging\nimport os\nimport sys\nfrom typing import Dict\n\n\n"
        "# =============================================================================\n"
        "# 1. FIRST PART\n"
        "# =============================================================================\n\n\n"
        "def early():\n    return 5\n\n\n"
        "def used_function():\n    return helper()\n\n\n"
        "def helper():\n    return early()\n\n\n"
        "def unused_function():\n    return 2\n\n\n"
        "def exported_only():\n    return 3\n\n\n"
        "def named_in_string():\n    return 4\n\n\n"
        "# =============================================================================\n"
        "# 2. THIRD PART\n"
        "# =============================================================================\n\n\n"
        "def lazy():\n    from .left import shared\n    return shared\n\n\n"
        "def run_case(case):\n    return getattr(sys.modules[__name__], f'_case_{case}')()\n\n\n"
        "def _case_one():\n    return 1\n\n\n"
        "class PlantedFormatter(logging.Formatter):\n"
        "    def format(self, record):\n        return ''\n\n"
        "    def __repr__(self):\n        return ''\n"),
    "src/kit/left.py": "from . import right\n\nshared = 1\nLINKED = (right,)\n",
    "src/kit/right.py": "from .left import shared\n\nCOPY = shared\n",
    "tests/user.py": "from src.kit import used_function\n\nTARGET = 'src.kit.core:named_in_string'\nused_function()\n"
                     "from src.kit.core import lazy, run_case\nlazy()\nrun_case('one')\n",
    "docs/notes.md": "unused_function is described here\n",
}
EXPECTED = {
    "unreferenced": {"unused_function", "exported_only", "PlantedFormatter"},
    "dynamic": {"format", "__repr__", "_case_one"},
    "tests_only": {"named_in_string", "lazy", "run_case"},
    "unused_exports": {("src/kit/__init__.py", "exported_only")},
    "unused_imports": {("src/kit/core.py", "os"), ("src/kit/core.py", "Dict")},
    "cycles": [["src.kit.left", "src.kit.right"]],
    "in_function": [("src.kit.core", "lazy", "src.kit.left", False)],
    "callees": {("src/kit/core.py", "<module>", "early")},
    "structure": ["src/kit/core.py"],
}
PLANTED_CONSTANTS = {
    CONSTANTS_FILE: "USED = 1\nUNUSED_CONSTANT = 2\n",
    METRICS_FILES[0]: '{"used_key": {}, "unused_key": {}}\n',
    "src/user.py": "from .utils.common.constants import USED\n\nKEY = 'used_key'\nVALUE = USED\n",
}
EXPECTED_CONSTANTS = {"UNUSED_CONSTANT", "unused_key"}
PLANTED_ANNOTATIONS = ('from typing import Dict, Literal\n\nx: Literal["two words"]\n\n\n'
                       'def f() -> "Dict[str, int]":\n    return {}\n')
EXPECTED_ANNOTATIONS = {"Literal", "Dict", "str", "int"}
PLANTED_SCOPES = (
    "from __future__ import annotations\n\nimport copy\nimport json\nimport shutil\nfrom os import sep\n"
    "from typing import Optional\n\n\n"
    "def listing(copies):\n    found = []\n    for copy in copies:\n        found.append(copy)\n    return found\n\n\n"
    "def dumped(value):\n    def inner():\n        return json.dumps(value)\n\n    return inner()\n\n\n"
    "def shadowing():\n    shutil = 1\n\n    def inner():\n        return shutil\n\n    return inner()\n\n\n"
    "def local():\n    import math\n    import re\n\n    def inner():\n        return re.escape('x')\n\n    return inner()\n\n\n"
    "def postponed(value: Optional[int]) -> None:\n    return None\n\n\n"
    "class Holder:\n    import glob\n    PATTERN = glob.escape('x')\n\n\n"
    "class Separator:\n    sep = sep\n\n\n"
    "def setup():\n    global textwrap\n    import textwrap\n\n\n"
    "def use():\n    return textwrap.dedent('x')\n\n\n"
    "def outer():\n    pprint = None\n\n    def setter():\n        nonlocal pprint\n        import pprint\n\n"
    "    setter()\n    return pprint\n\n\n"
    "def rebinding():\n    import string\n\n    def middle():\n        string = 1\n\n        def inner():\n"
    "            return string\n\n        return inner()\n\n    return middle()\n"
)
# copy read only as a local of that name, shutil shadowed, math never read, string read only past a function that binds
# it again; sep read by a class body that binds it, textwrap and pprint bound through global and nonlocal, all read
EXPECTED_SCOPES = {"copy", "shutil", "math", "string"}
PLANTED_CLOSURES = {
    "src/__init__.py": "",
    "src/named.py": "import matplotlib\n\nmatplotlib.use('Agg')\n",
    "src/constant.py": "import matplotlib\n\nBACKEND = 'Agg'\nmatplotlib.use(BACKEND)\n",
    "src/first.py": "from . import named\n",
    "src/second.py": "from .constant import BACKEND\n",
    "src/drawing.py": "from matplotlib import pyplot\n",
    "src/plain.py": "VALUE = 1\n",
}
EXPECTED_CLOSURES = {
    "src.first": {"agg": True, "pyplot": False, "matplotlib": True},
    "src.second": {"agg": True, "pyplot": False, "matplotlib": True},
    "src.drawing": {"agg": False, "pyplot": True, "matplotlib": True},
    "src.plain": {"agg": False, "pyplot": False, "matplotlib": False},
}
PLANTED_TOPDOWN = {
    "src/sequence.py": (
        '"""Planted module.\n\nStructure:\n1. second\n2. first\n"""\n\n\n'
        "def first():\n    return second()\n\n\n"
        "def second():\n    return 1\n\n\n"
        "class Planted:\n    @staticmethod\n    def helper():\n        return 1\n\n"
        "    VALUE = helper()\n\n"
        "    def method(self):\n        return self.helper()\n"),
}
EXPECTED_TOPDOWN = {"callees": set(), "structure": ["src/sequence.py"]}
PLANTED_MOVE_SOURCE = (
    "from .shared import helper\nimport math\nimport importlib{submodule}\nfrom {star} import *\n\n"
    "try:\n    GUARDED = {guarded}\nexcept ImportError:\n    GUARDED = 0\n"
    "PAIR_A, PAIR_B = 1, {pair}\nTWICE = 1\n{twice}LIMIT = {limit}\nlen = {length}\n\n\n"
    "def same(x):\n    return abs(helper(x)) + math.pi\n\n\n"
    "def changed(x):\n    return x + {step}\n\n\n"
    "def rebound(x):\n    return LIMIT * x\n\n\n"
    "def guarded():\n    return GUARDED\n\n\n"
    "def unpacked():\n    return PAIR_A\n\n\n"
    "def twice():\n    return TWICE\n\n\n"
    "def starred():\n    return STARRED\n\n\n"
    "def loader(name):\n    return importlib.util.find_spec(name)\n\n\n"
    "def shadowed(x):\n    total = LIMIT + x\n\n    def inner():\n        LIMIT = 2\n        return LIMIT\n\n"
    "    return total + inner()\n\n\n"
    "def builtin(x):\n    return len(x)\n"
)
PLANTED_MOVES = (
    {"src/old.py": PLANTED_MOVE_SOURCE.format(star="a", submodule=".util", guarded=1, pair=2, twice="TWICE = 2\n",
                                              limit=3, length=1, step=1)},
    {"src/new.py": PLANTED_MOVE_SOURCE.format(star="b", submodule="", guarded=5, pair=3, twice="", limit=4, length=2,
                                              step=2)},
    [f"src/old.py:{name}=src/new.py" for name in ("same", "changed", "rebound", "missing", "guarded", "unpacked",
                                                    "twice", "starred", "loader", "shadowed", "builtin")],
)
EXPECTED_MOVES = ("changed: its definition differs", "rebound: LIMIT bound to", "missing: absent at the base",
                  "guarded: GUARDED bound to", "unpacked: PAIR_A bound to", "twice: TWICE bound to",
                  "starred: STARRED bound to", "loader: importlib bound to", "shadowed: LIMIT bound to",
                  "builtin: len bound to")  # a module binding shadowing a builtin; same's abs, bound by neither, is not
EXPECTED_SHADOWED_READS = {"LIMIT"}  # symtable: the outer function reads the global LIMIT that inner rebinds locally
WARNING_CORE, WARNING_OTHER = "\x1b[33mWARNING\x1b[0m kit.core: truncated row", "\x1b[33mWARNING\x1b[0m kit.other: truncated row"
PLANTED_LINES = {"base-1": ["start 10:00 base-1", "row 1", "row 2", WARNING_CORE, "Duration: 1.5 seconds"],
                 "base-2": ["start 10:05 base-2", "row 1", "row 2", WARNING_CORE, "Duration: 1.7 seconds"]}
PLANTED_CANDIDATES = {
    "varying lines only": (["start 11:00 tip-1", "row 1", "row 2", WARNING_CORE, "Duration: 2.1 seconds"], []),
    "changed line": (["start 11:00 tip-1", "row 1", "row 9", WARNING_CORE, "Duration: 2.1 seconds"],
                     ["-3: row 2", "+3: row 9"]),
    "added line": (["start 11:00 tip-1", "row 1", "row 2", "row 3", WARNING_CORE, "Duration: 2.1 seconds"],
                   ["+4: row 3"]),
    "deleted line": (["start 11:00 tip-1", "row 1", WARNING_CORE, "Duration: 2.1 seconds"], ["-3: row 2"]),
    "logger name": (["start 11:00 tip-1", "row 1", "row 2", WARNING_OTHER, "Duration: 2.1 seconds"],
                    [f"-4: {WARNING_CORE}", f"+4: {WARNING_OTHER}"]),
    "another line beside a masked one": (["start 11:00 tip-1", "row 1", "row 2", WARNING_CORE, "ERROR boom"],
                                         ["-5: Duration: 1.5 seconds", "+5: ERROR boom"]),
}
PLANTED_TEXT_LABELS = ["base-1", "base-2", "tip-1"]
PLANTED_PACKAGE = {"src/__init__.py": "", "src/agg.py": "import matplotlib\n\nmatplotlib.use('Agg')\nimport matplotlib.pyplot\n",
                   "src/alone.py": "import matplotlib.pyplot\n", "src/filter.py": "import hdf5plugin\n",
                   "src/broken.py": "raise ImportError('planted')\n", "src/plain.py": "VALUE = 1\n",
                   "src/multiline.py": "raise ImportError('a\\nb')\n", "src/silent.py": "raise SystemExit(3)\n"}
EXPECTED_RECORDS = {
    "src": {"matplotlib": False, PYPLOT: False, HDF5PLUGIN: False, "backend": None},
    "src.agg": {"matplotlib": True, PYPLOT: True, HDF5PLUGIN: False, "backend": "Agg"},
    "src.alone": {"matplotlib": True, PYPLOT: True, HDF5PLUGIN: False, "backend": None},
    "src.filter": {"matplotlib": False, PYPLOT: False, HDF5PLUGIN: True, "backend": None},
    "src.broken": {"raised": "ImportError"},
    "src.plain": {"matplotlib": False, PYPLOT: False, HDF5PLUGIN: False, "backend": None},
    "src.multiline": {"raised": "ImportError"},
    "src.silent": {"raised": NO_OUTPUT},
}
PLAIN = EXPECTED_RECORDS["src.plain"]
PLANTED_STATES = (
    {"m.same": PLAIN, "m.raise": PLAIN, "m.pyplot": EXPECTED_RECORDS["src.agg"], "m.drop": EXPECTED_RECORDS["src.agg"],
     "m.filter": EXPECTED_RECORDS["src.filter"], "m.gone": PLAIN, "m.old": PLAIN,
     "m.declared": EXPECTED_RECORDS["src.alone"], "m.static": EXPECTED_RECORDS["src.agg"],
     "m.plugin": EXPECTED_RECORDS["src.agg"]},
    {"m.same": PLAIN, "m.raise": {"raised": "ImportError"}, "m.pyplot": EXPECTED_RECORDS["src.alone"], "m.drop": PLAIN,
     "m.filter": PLAIN, "m.new": PLAIN, "m.moved": PLAIN,
     "m.declared": EXPECTED_RECORDS["src.agg"], "m.static": EXPECTED_RECORDS["src.agg"],
     "m.plugin": {**EXPECTED_RECORDS["src.agg"], HDF5PLUGIN: True}},
    {"m.old": "m.moved"},
    frozenset({"m.declared", "m.static"}),
)
EXPECTED_STATES = {"failing": ["m.pyplot", "m.raise", "m.static"], "declared": ["m.declared"],
                   "named": ["m.drop", "m.filter", "m.plugin"], "apart": ["m.gone", "m.new"]}


def _planted_summary(times: dict, memory: dict, contended: bool) -> dict:
    """A completed launch record with the given logged seconds and largest sampled RSS per pair."""
    return {"outcome": perflaunch.OUTCOME_COMPLETED, "return_code": 0,
            "times": {"pairs": [{"mode": "ideal", "task": task, "logged_s": value} for task, value in times.items()]},
            "memory": {"pairs": [{"mode": "ideal", "task": task, "largest_sampled_rss_b": value}
                                 for task, value in memory.items()]},
            "cpu": {"contended": contended, "other_cores": 2.0 if contended else 0.1}}


# trigger time: base range 26.1 %, change +17.4 %, within v only through the range; trigger memory: base range 0,
# change +5 %, within v only through the 10 % floor
PLANTED_SPEED = (
    [_planted_summary({"trigger": 1.0, "performance": 10.0, "signal": 0.0},
                      {"trigger": 100, "performance": 200, "signal": 50}, False),
     _planted_summary({"trigger": 1.3, "performance": 10.1, "signal": 0.0},
                      {"trigger": 100, "performance": 200, "signal": 50}, False),
     _planted_summary({"trigger": 1.15, "performance": 10.2, "signal": 0.0},
                      {"trigger": 100, "performance": 200, "signal": 50}, False)],
    [_planted_summary({"trigger": 1.34, "performance": 12.0, "signal": 0.5},
                      {"trigger": 105, "performance": 300, "signal": None}, True),
     _planted_summary({"trigger": 1.36, "performance": 12.1, "signal": 0.5},
                      {"trigger": 105, "performance": 300, "signal": 50}, False),
     _planted_summary({"trigger": 1.35, "performance": 12.2, "signal": 0.5},
                      {"trigger": 105, "performance": 300, "signal": 50}, False)],
)
EXPECTED_SPEED = {"outside": {"time ideal/performance", "memory ideal/performance"},
                  "undefined": {"time ideal/signal"}, "absent": {"memory ideal/signal"}}
PLANTED_COCHANGE = (("one", {"a.py": "first\n", "b.py": "first\n"}), ("two", {"a.py": "first\nsecond\n"}))
EXPECTED_COCHANGE = {"commits": 2, "pairs": {("a.py", "b.py"): 1}, "churn": {"a.py": (2, 2), "b.py": (1, 1)}}


def self_test(folder: Path) -> None:
    """Run every command on its planted sources, records or scratch repository; each planted case must be reported
    and nothing else."""
    found = references(PLANTED, folder)
    CTX.verdict("self-test: unreferenced", {item.name for item in found["unreferenced"]} == EXPECTED["unreferenced"],
                EXPECTED["unreferenced"], {item.name for item in found["unreferenced"]})
    CTX.verdict("self-test: dynamic only", {item.name for item in found["dynamic"]} == EXPECTED["dynamic"],
                EXPECTED["dynamic"], {item.name for item in found["dynamic"]})
    CTX.verdict("self-test: tests only", {item.name for item in found["tests_only"]} == EXPECTED["tests_only"],
                EXPECTED["tests_only"], {item.name for item in found["tests_only"]})
    for key in ("unused_exports", "unused_imports"):
        CTX.verdict(f"self-test: {key}", set(found[key]) == EXPECTED[key], EXPECTED[key], set(found[key]))
    constant_found = references(PLANTED_CONSTANTS, _subfolder(folder, "constants"))
    unreferenced = {item.name for item in constant_found["unreferenced"]}
    CTX.verdict("self-test: an unreferenced constant and metric key", unreferenced == EXPECTED_CONSTANTS,
                EXPECTED_CONSTANTS, unreferenced)
    names = _annotation_names(ast.parse(PLANTED_ANNOTATIONS))
    CTX.verdict("self-test: annotation names, string annotations parsed, Literal values left out",
                names == EXPECTED_ANNOTATIONS, EXPECTED_ANNOTATIONS, names)
    scoped = {name for _, name, _ in unused_imports(ast.parse(PLANTED_SCOPES), PLANTED_SCOPES)}
    CTX.verdict("self-test: unused imports by scope (a same-named local, a shadowing function, an unread import in a "
                "function, a postponed annotation, a class body binding the name it reads, imports declared global and "
                "nonlocal, a nested function binding a function's import again)", scoped == EXPECTED_SCOPES,
                EXPECTED_SCOPES, scoped)
    graph = imports(PLANTED, folder)
    for key in ("cycles", "in_function"):
        CTX.verdict(f"self-test: {key}", graph[key] == EXPECTED[key], EXPECTED[key], graph[key])
    closures = imports(PLANTED_CLOSURES, _subfolder(folder, "closures"))["closures"]
    flags = {module: {key: closures[module][key] for key in expected} for module, expected in EXPECTED_CLOSURES.items()}
    CTX.verdict("self-test: closure flags (two backend selections, from matplotlib import pyplot)",
                flags == EXPECTED_CLOSURES, EXPECTED_CLOSURES, flags)
    sequence = topdown(PLANTED, folder)
    CTX.verdict("self-test: early callees", set(sequence["callees"]) == EXPECTED["callees"], EXPECTED["callees"],
                set(sequence["callees"]))
    CTX.verdict("self-test: Structure against banners", sequence["structure"] == EXPECTED["structure"],
                EXPECTED["structure"], sequence["structure"])
    listed = topdown(PLANTED_TOPDOWN, _subfolder(folder, "topdown"))
    actual = {"callees": set(listed["callees"]), "structure": listed["structure"]}
    CTX.verdict("self-test: a class-body caller exempts its callee; a Structure list against the definitions",
                actual == EXPECTED_TOPDOWN, EXPECTED_TOPDOWN, actual)
    history = cochange("HEAD", SINCE, _subfolder(folder, "cochange"), _scratch_repository(folder / "repository"))
    actual = {"commits": len(history["commits"]), "pairs": dict(history["pairs"]), "churn": history["churn"]}
    CTX.verdict("self-test: cochange of a scratch repository", actual == EXPECTED_COCHANGE, EXPECTED_COCHANGE, actual)
    differences, _ = compare_moves([_parse_move(spec) for spec in PLANTED_MOVES[2]], *PLANTED_MOVES[:2])
    CTX.verdict("self-test: moved", len(differences) == len(EXPECTED_MOVES) and all(
        any(line.startswith(prefix) for line in differences) for prefix in EXPECTED_MOVES), EXPECTED_MOVES, differences)
    reads = _global_reads(_find_definition(ast.parse(PLANTED_MOVES[0]["src/old.py"]), "shadowed"))
    CTX.verdict("self-test: global reads from symtable", reads == EXPECTED_SHADOWED_READS, EXPECTED_SHADOWED_READS,
                reads)
    mask = text_mask(PLANTED_LINES, "base-1", ["base-2"])
    CTX.verdict("self-test: text mask", mask == {0, 4}, {0, 4}, mask)
    joined = text_mask({**PLANTED_LINES, "tip-1": PLANTED_CANDIDATES["changed line"][0]}, "base-1", ["base-2", "tip-1"])
    CTX.verdict("self-test: text mask joined by a third base", joined == {0, 2, 4}, {0, 2, 4}, joined)
    for kind, (lines, expected) in PLANTED_CANDIDATES.items():
        actual = unmasked_differences(PLANTED_LINES["base-1"], lines, mask, PLANTED_TEXT_LABELS)
        CTX.verdict(f"self-test: text, {kind}", actual == expected, expected, actual)
    tree = folder / "planted"
    for path, source in PLANTED_PACKAGE.items():
        (tree / path).parent.mkdir(parents=True, exist_ok=True)
        (tree / path).write_text(source, encoding="utf-8")
    for name, expected in EXPECTED_RECORDS.items():
        actual = module_record(tree, name)
        CTX.verdict(f"self-test: import record of {name}", actual == expected, expected, actual)
    states = judge_import_states(*PLANTED_STATES)
    actual = {kind: [re.search(r"\bm\.\w+", line)[0] for line in lines] for kind, lines in states.items()}
    CTX.verdict("self-test: import-state judgement", actual == EXPECTED_STATES, EXPECTED_STATES, actual)
    incomplete = incomplete_launches({"done-1": PLANTED_SPEED[0][0], "stopped-1": {
        **PLANTED_SPEED[0][1], "outcome": perflaunch.OUTCOME_STOPPED, "return_code": None}})
    CTX.verdict("self-test: speed names a stopped launch", len(incomplete) == 1 and incomplete[0].startswith(
        "stopped-1:"), "stopped-1 alone", incomplete)
    rows = speed_rows(*PLANTED_SPEED)
    actual = {"outside": {f"{row['metric']} {row['pair']}" for row in rows if not row["within"]
                          and not row.get("undefined") and not row.get("absent")},
              "undefined": {f"{row['metric']} {row['pair']}" for row in rows if row.get("undefined")},
              "absent": {f"{row['metric']} {row['pair']}" for row in rows if row.get("absent")}}
    CTX.verdict("self-test: speed", actual == EXPECTED_SPEED and len(rows) == 6, EXPECTED_SPEED, actual)
    CTX.verdict("self-test: importing perflaunch adds no logging handler", not perflaunch.LOG.handlers and not any(
        isinstance(handler, logging.FileHandler) for handler in logging.getLogger().handlers), "no handler",
                perflaunch.LOG.handlers)
    before = _listing(perflaunch.PERF_DIR)
    child = CTX.launch_child("perflaunch", ["-c", PERFLAUNCH_CHILD, str(harness.HARNESS_DIR)],
                             timeout_s=CHILD_BOUND_S, hashseed=CHILD_HASHSEED)
    after = _listing(perflaunch.PERF_DIR)
    CTX.verdict("self-test: importing perflaunch in a child writes no file under data\\perf",
                child.return_code == 0 and before == after, "return code 0, listing unchanged",
                {"return_code": child.return_code, "files_before": len(before), "files_after": len(after),
                 "changed": sorted(set(before) ^ set(after))[:10]})


def _subfolder(folder: Path, name: str) -> Path:
    """A new folder inside the launch's folder, for the tables of one planted command."""
    path = folder / name
    path.mkdir()
    return path


def _scratch_repository(path: Path) -> Path:
    """A new repository holding the planted co-change commits, committed with the harness fixtures' identity."""
    path.mkdir()
    harness.run_helper(["git", "init", "-q", "-b", "main", str(path)], cwd=path, log=CTX.log, bound_s=GIT_BOUND_S)
    for message, contents in PLANTED_COCHANGE:
        for name, content in contents.items():
            (path / name).write_bytes(content.encode("utf-8"))
        harness.run_helper(["git", "add", "-A"], cwd=path, log=CTX.log, bound_s=GIT_BOUND_S)
        harness.run_helper(["git", *harness.FIXTURE_IDENTITY, "commit", "-q", "-m", message], cwd=path, log=CTX.log,
                           bound_s=GIT_BOUND_S)
    return path


def _listing(root: Path) -> list[tuple[str, int, int]]:
    """(path relative to the folder, size, modification time in ns) of every file under a folder."""
    return sorted((path.relative_to(root).as_posix(), path.stat().st_size, path.stat().st_mtime_ns)
                  for path in root.rglob("*") if path.is_file())


# =============================================================================
# 9. SHARED HELPERS
# =============================================================================


def _defines(path: str) -> bool:
    """A Python file whose definitions are enumerated."""
    return path.endswith(PY_SUFFIX) and (path.startswith(DEFINITION_ROOTS) or path in DEFINITION_FILES)


def _write(folder: Path, name: str, lines: list[str]) -> None:
    """One output table of the launch, logged with its path and line count."""
    path = folder / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    CTX.log.info("WROTE %s (%d lines)", path, len(lines))


def _module_name(path: str) -> str:
    """Dotted module name of a repository path (a package file names its package)."""
    parts = path[:-len(PY_SUFFIX)].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve_from(module: str, is_package: bool, level: int, target: Optional[str]) -> str:
    """The absolute module a from-import names: a relative level climbs from the module's package."""
    if level == 0:
        return target or ""
    base = module.split(".") if is_package else module.split(".")[:-1]
    base = base[:len(base) - (level - 1)]
    return ".".join(base + ([target] if target else []))


def _exported(tree: ast.Module) -> list[str]:
    """The string items of a module-level __all__."""
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__"
                                                 for target in node.targets)):
            return [item.value for item in ast.walk(node.value) if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)]
    return []


def _dotted(node: ast.AST) -> Optional[str]:
    """a.b.c of a Name or Attribute chain, None for any other expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


if __name__ == "__main__":
    main()
