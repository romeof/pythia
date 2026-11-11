"""EXP-00: run driver of the section 6.2 pipeline runs (RUN-1, RUN-2, RUN-T).

Runs the planted-defect control of the completeness check, launches pipeline.py of the
source root through the harness's child launcher (probe first), checks the run's
completeness on the pipeline child's own log, and copies data\\reference to the run's
never-reused label with a manifest.

Structure:
1. main
2. Planted control of the completeness check
3. Completeness check
4. Output digests, file lists and the labelled copy
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=False)

import argparse
import re
import shutil
from pathlib import Path

STILL_EXISTS = "still exists after deletion"
GOOD_SUMMARY = "This progress looks :)"
BAD_SUMMARIES = ("This progress looks :(", "This progress looks :|")
EXPECTED_PROCESSED = 8
EXPECTED_SUMMARIES = 12
EXPECTED_H5 = 6
EXPECTED_FILES = 201
TASK_FILE = "0.h5"
RUN_LABEL = re.compile(r"^run(tip)?[0-9]+$")
KIND_BASELINE = "baseline"
KIND_TIP = "tip"
STRATEGY = "extrema"
TASKS = ["all"]


def main() -> None:
    """Control first, then the run, its completeness check and its labelled copy."""
    args = _parse_args()
    check_planted_control()
    if not RUN_LABEL.match(args.run_label):
        harness.refuse(CTX.log, "run-label", f"{args.run_label} is not run<N> or runtip<N>")
    before = h5_digests(CTX.roots.outputs)
    child, probe, origin = CTX.run_pipeline("pipeline", strategy=STRATEGY, tasks=TASKS, quiet_luigi=args.quiet_luigi,
                                            hashseed=args.child_seed, allowed_root=CTX.roots.outputs)
    CTX.log.info("pipeline child %s: return code %s, timed out %s, %.1f s, seed %d, source root %s",
                 child.log_path.name, child.return_code, child.timed_out, child.duration_s, args.child_seed, CTX.source_root)
    CTX.verdict("origin", origin["confirmed"], f"src imported from {origin['source_tree']}", origin["origin"])
    CTX.verdict("return code", child.return_code == 0 and not child.timed_out, "return code 0", child.return_code)
    after = h5_digests(CTX.roots.outputs)
    problems = completeness_problems(child.text, expect_summaries=not args.quiet_luigi, h5_after=after, h5_before=before)
    CTX.verdict("completeness", not problems, "no problem", problems)
    if args.kind == KIND_BASELINE:
        check_file_list(CTX.roots.outputs, CTX.roots.ref / harness.DATA_OUTPUT_DIR)
    copy_run(args.run_label)
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-label, --child-seed, --quiet-luigi, --kind and --source-root (read by the harness)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--child-seed", type=int, required=True)
    parser.add_argument("--quiet-luigi", choices=["True", "False"], default="True")
    parser.add_argument("--kind", choices=[KIND_BASELINE, KIND_TIP], default=KIND_BASELINE)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    args = parser.parse_args()
    args.quiet_luigi = args.quiet_luigi == "True"
    return args


# =============================================================================
# PLANTED CONTROL
# =============================================================================


def check_planted_control() -> None:
    """Every planted defect is flagged; the clean synthetic log is not (non-vacuity)."""
    processed = [f"INFO src.utils.common.producerconsumer: Processed 6 items, successful: 6" for _ in range(EXPECTED_PROCESSED)]
    summaries = [GOOD_SUMMARY] * EXPECTED_SUMMARIES
    clean = processed + summaries
    digests_before = {f"m{index}/{TASK_FILE}": f"old{index}" for index in range(EXPECTED_H5)}
    digests_after = {path: digest.replace("old", "new") for path, digest in digests_before.items()}
    plants = {
        "coloured ERROR line": clean + ["\x1b[31mERROR src.tasks.trigger: planted failure\x1b[0m"],
        "still-exists warning": clean + [f"\x1b[38;5;208mWARNING src.utils.common.common: x {STILL_EXISTS}\x1b[0m"],
        "seven Processed lines": processed[1:] + summaries,
        "nine Processed lines": processed + processed[:1] + summaries,
        "one line with N != M": processed[1:] + ["Processed 6 items, successful: 5"] + summaries,
        "a :( summary": clean + ["This progress looks :("],
        "eleven :) lines": processed + summaries[1:],
        "thirteen :) lines": processed + summaries + summaries[:1],
    }
    CTX.verdict("control: clean synthetic log", not completeness_problems(
        "\n".join(clean), expect_summaries=True, h5_after=digests_after, h5_before=digests_before), "no problem", "checked")
    for name, lines in plants.items():
        found = completeness_problems("\n".join(lines), expect_summaries=True, h5_after=digests_after, h5_before=digests_before)
        CTX.verdict(f"control: {name}", bool(found), "flagged", found)
    identical = dict(digests_after, **{f"m0/{TASK_FILE}": digests_before[f"m0/{TASK_FILE}"]})
    found = completeness_problems("\n".join(clean), expect_summaries=True, h5_after=identical, h5_before=digests_before)
    CTX.verdict("control: 0.h5 byte-identical to its previous copy", bool(found), "flagged", found)


# =============================================================================
# COMPLETENESS CHECK
# =============================================================================


def completeness_problems(text: str, *, expect_summaries: bool, h5_after: dict, h5_before: dict) -> list[str]:
    """Section 6.1's completeness check on ANSI-stripped text with substring matches."""
    lines = harness.plain_lines(text)
    problems = []
    errors = harness.error_lines(lines)
    if errors:
        problems.append(f"{len(errors)} ERROR lines, first: {errors[0][:200]}")
    if any(STILL_EXISTS in line for line in lines):
        problems.append("check of the audited revision: an output folder survived its deletion (from Q12 on, a failed removal raises "
                        "and the return-code verdict reports it)")
    processed = harness.processed_counts(lines)
    if len(processed) != EXPECTED_PROCESSED:
        problems.append(f"{len(processed)} Processed lines, expected {EXPECTED_PROCESSED}")
    unequal = [f"Processed {total} items, successful: {successful}" for total, successful in processed if total != successful]
    if unequal:
        problems.append(f"Processed lines with N != M: {unequal}")
    if expect_summaries:
        good = sum(GOOD_SUMMARY in line for line in lines)
        if good != EXPECTED_SUMMARIES:
            problems.append(f"{good} '{GOOD_SUMMARY}' lines, expected {EXPECTED_SUMMARIES}")
        bad = [line.strip() for line in lines if any(summary in line for summary in BAD_SUMMARIES)]
        if bad:
            problems.append(f"failed or partial Luigi summaries: {bad[:3]}")
    identical = sorted(path for path, digest in h5_after.items() if h5_before.get(path) == digest)
    if identical:
        problems.append(f"skipped task: {identical} byte-identical to the previous copy")
    if len(h5_after) != EXPECTED_H5:
        problems.append(f"{len(h5_after)} task files {TASK_FILE}, expected {EXPECTED_H5}")
    return problems


# =============================================================================
# DIGESTS, FILE LISTS AND THE LABELLED COPY
# =============================================================================


def h5_digests(root: Path) -> dict[str, str]:
    """MD5 of every task file under a run's outputs, keyed by relative path."""
    if not root.is_dir():
        return {}
    return {path.relative_to(root).as_posix(): harness.file_md5(path) for path in sorted(root.rglob(TASK_FILE))}


def file_list(root: Path) -> list[str]:
    """Relative paths of every file under a tree."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def check_file_list(outputs: Path, reference: Path) -> None:
    """Baseline runs: 201 files and the reference copy's file list."""
    produced, expected = file_list(outputs), file_list(reference)
    CTX.verdict("file count", len(produced) == EXPECTED_FILES, EXPECTED_FILES, len(produced))
    missing, extra = sorted(set(expected) - set(produced)), sorted(set(produced) - set(expected))
    CTX.verdict("file list equals the reference copy", not missing and not extra, "no difference",
                {"missing": missing[:20], "extra": extra[:20]})


def copy_run(run_label: str) -> None:
    """Copy data\\reference to data\\<label>\\reference (never reused) and write its manifest."""
    destination = CTX.roots.data / run_label
    CTX.guard(destination)
    if destination.exists():
        harness.refuse(CTX.log, "run-label", f"{destination} already exists")
    shutil.copytree(CTX.roots.outputs, destination / harness.DATA_OUTPUT_DIR)
    CTX.write_manifest(run_label, destination)


if __name__ == "__main__":
    main()
