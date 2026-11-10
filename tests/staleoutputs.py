"""Regression for the outputs and provenance files a failed run leaves (G-7).

On a one-series realistic copy, one healthy --task all writes the task folders and the strategy
folder's three provenance files (extrema.yaml, dev.yaml, git_info.txt), a positive control; the
test copies that output tree inside its scratch folder, and before case (b) it removes
the tree and restores it from the copy, so every case starts from the healthy run's files.
Case (a): a testkit.planted_run child, --task all with a planted OSError in
SignalSelection.build_signal_window_groups: the trigger folder keeps 0.h5, the signal and
performance folders are absent, the strategy folder holds none of the three files, return != 0
(before G-7: the performance folder keeps the previous outputs, the three files remain).
Case (b): a child whose shutil.copy2 raises OSError for a destination named dev.yaml, after the
strategy YAML was copied, under --task trigger: copy_config_to_output leaves none of the three
files, return != 0 (before: the fresh strategy YAML and the previous dev.yaml and git_info.txt
remain). Each case also requires the planted error in its capture, so no other failure stands in for it.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import shutil
from pathlib import Path

GIT_FILE = "git_info.txt"
PROVENANCE_FILES = (testkit.STRATEGY_FILE, testkit.DEV_FILE, GIT_FILE)
OUTPUT_FOLDER = "out"
SAVED_FOLDER = "healthy"
SIGNAL_TARGET = "src.tasks.signal:SignalSelection.build_signal_window_groups"
STRATEGY_COPIED = "Analysis config copied to:"
COPY_CODE = testkit.CHILD_BOOTSTRAP + (
    "import shutil\n"
    "from pathlib import Path\n"
    "original = shutil.copy2\n"
    "def planted(source, destination, *args, **kwargs):\n"
    f"    if Path(destination).name == '{testkit.DEV_FILE}':\n"
    "        raise OSError(testkit.PLANTED_ERROR)\n"
    "    return original(source, destination, *args, **kwargs)\n"
    "shutil.copy2 = planted\n"
    "import pipeline\n"
    "sys.argv = ['pipeline.py', '--task', 'trigger', '--strategy', sys.argv[4], '--profile', testkit.DEV_PROFILE]\n"
    "pipeline.main()\n")


def main() -> None:
    """The healthy run, the two cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        healthy_run(folder, config_file)
        failed_task(folder, config_file)
        restore(folder)
        failed_copy(folder, config_file)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def healthy_run(folder: Path, config_file: Path) -> None:
    """The positive control: every task file and the three provenance files, then the saved copy of the tree."""
    child = testkit.run_pipeline(KIT, folder, "healthy", config_file, ["all"])
    found = {task: testkit.output_file(config_file, task).is_file() for task in testkit.TASKS}
    KIT.verdict("healthy --task all: return code 0, every task file", child.return_code == 0 and all(found.values()),
                {"return_code": 0, "task files": "all present"}, {"return_code": child.return_code, "task files": found})
    present = provenance_present(config_file)
    KIT.verdict("healthy --task all: the three provenance files", present == list(PROVENANCE_FILES),
                list(PROVENANCE_FILES), present)
    shutil.copytree(folder / OUTPUT_FOLDER, folder / SAVED_FOLDER)


def failed_task(folder: Path, config_file: Path) -> None:
    """Case (a): a failed Signal task keeps the trigger output only and no provenance file."""
    child = testkit.planted_run(KIT, folder, "signal", config_file, target=SIGNAL_TARGET, task="all")
    trigger_kept = testkit.output_file(config_file, "trigger").is_file()
    later = {task: testkit.output_file(config_file, task).parent.exists() for task in ("signal", "performance")}
    KIT.verdict("case (a): the trigger folder keeps its 0.h5, the signal and performance folders are absent",
                trigger_kept and not any(later.values()), {"trigger 0.h5": True, "signal": False, "performance": False},
                {"trigger 0.h5": trigger_kept, **later})
    no_provenance(config_file, child, "case (a)")


def failed_copy(folder: Path, config_file: Path) -> None:
    """Case (b): a copy that fails on dev.yaml after the strategy YAML leaves no provenance file."""
    child = testkit.run_child(KIT, folder, "copy", ["-c", COPY_CODE, *testkit.child_args(KIT),
                                                    testkit.strategy_argument(config_file)],
                              timeout_s=testkit.PIPELINE_TIMEOUT_S)
    copied = [line for line in testkit.plain_lines(child.text) if STRATEGY_COPIED in line]
    KIT.verdict("case (b): copy_config_to_output copied the strategy YAML before the planted dev.yaml failure",
                len(copied) == 1, 1, copied)
    no_provenance(config_file, child, "case (b): copy_config_to_output")


# =============================================================================
# HELPERS
# =============================================================================


def restore(folder: Path) -> None:
    """The healthy run's output tree, restored from the saved copy."""
    shutil.rmtree(folder / OUTPUT_FOLDER)
    shutil.copytree(folder / SAVED_FOLDER, folder / OUTPUT_FOLDER)


def provenance_present(config_file: Path) -> list[str]:
    """The provenance files present in the strategy folder of the copy, in PROVENANCE_FILES' sequence."""
    strategy_folder = testkit.output_file(config_file, "trigger").parents[2]
    return [name for name in PROVENANCE_FILES if (strategy_folder / name).is_file()]


def no_provenance(config_file: Path, child: testkit.Child, case: str) -> None:
    """None of the three provenance files and a non-zero return code, the planted error in the capture (so no other failure
    stands in for it)."""
    present = provenance_present(config_file)
    planted = any(testkit.PLANTED_ERROR in line for line in testkit.plain_lines(child.text))
    KIT.verdict(f"{case}: none of {list(PROVENANCE_FILES)}, return != 0 with the planted error",
                not present and child.return_code not in (0, None) and planted,
                {"present": [], "return_code": "!= 0", "planted": True},
                {"present": present, "return_code": child.return_code, "timed_out": child.timed_out, "planted": planted})


if __name__ == "__main__":
    main()
