"""Regression for the cleanup of each task's output folder (Q12).

(a) output() and .complete() of the Trigger task that pipeline._create_task builds from the copy's
validated sections, on an existing task folder that holds a planted file, leave the folder and the
file in place: output() only defines the target. (b)
--task trigger twice on a one-series realistic copy, with a stale file planted in the task folder
between the runs: the second run logs one "Cleaned previous outputs" line, rewrites 0.h5 and
removes the stale file. (c) The first run, on a fresh folder, logs none. (d) A planted 0.h5 that
this process keeps locked while --task trigger runs: Windows refuses to delete a file held
without delete sharing, so the removal fails and the run must return non-zero with PermissionError
and the file name 0.h5 in its output (the traceback prints the path through repr, so the full
path is not matched) instead of taking the planted file for the task's output. Its
precondition: a probe child that tries to delete the locked file fails with PermissionError.
(e) luigi.build of Trigger, then Signal, in this process and outside pipeline.py's _run_task, on
a one-series realistic copy whose output tree does not exist yet: each build succeeds and writes
its 0.h5. pipeline._create_task builds both tasks from the copy's validated sections, as in (a).
The unfixed tasks clean their folder inside output(), so (a) deletes the planted file, and a
folder that cannot be removed is only warned about, so (d) returns 0 with the planted file standing
as the task's output. Their output() also recreates the folder, so (e) passes on them. Once
output() stopped doing so (Q12), Trigger wrote its store into a folder that did not exist and its
build failed with FileNotFoundError, until open_store created the folder of a "w" write.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import luigi

import pipeline

CLEANED = "Cleaned previous outputs from"
STALE_NAME = "stale_planted.txt"
PLANTED_BYTES = b"planted bytes, not an HDF5 store"
UNLINK_CODE = "import os, sys; os.unlink(sys.argv[1])"
PERMISSION = "PermissionError"
LUIGI_LOG_LEVEL = "WARNING"  # keeps Luigi's failure records, drops its per-task DEBUG lines that print the whole config


def main() -> None:
    """The five cases, then the summary (return code 1 on any FAIL)."""
    target_untouched()
    rerun_cleanup()
    locked_output()
    direct_build()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def target_untouched() -> None:
    """Case (a): output() and complete() leave an existing task folder and its planted file in place."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        target = testkit.output_file(config_file, "trigger")
        target.parent.mkdir(parents=True)
        planted = target.parent / STALE_NAME
        planted.write_bytes(PLANTED_BYTES)
        task = pipeline._create_task(testkit.validated_config(config_file), "trigger")
        task.output()
        complete = task.complete()
        KIT.verdict("case (a): output() and complete() leave the task folder and its planted file in place",
                    target.parent.is_dir() and planted.is_file(), {"folder": True, "planted": True},
                    {"folder": target.parent.is_dir(), "planted": planted.is_file(), "complete": complete})


def rerun_cleanup() -> None:
    """Cases (b) and (c): no cleanup line on a fresh folder; on a rerun one line, a new 0.h5, the stale file gone."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        target = testkit.output_file(config_file, "trigger")
        first = testkit.run_pipeline(KIT, folder, "first", config_file, ["trigger"])
        first_lines = cleaned_lines(first)
        KIT.verdict("case (c): the first run, on a fresh folder, logs no cleanup line",
                    first.return_code == 0 and target.is_file() and not first_lines, {"return_code": 0, "cleanup_lines": 0},
                    {"return_code": first.return_code, "cleanup_lines": first_lines})
        written = target.stat().st_mtime_ns if target.is_file() else None
        stale = target.parent / STALE_NAME
        stale.write_bytes(PLANTED_BYTES)
        second = testkit.run_pipeline(KIT, folder, "second", config_file, ["trigger"])
        second_lines = cleaned_lines(second)
        rewritten = target.is_file() and target.stat().st_mtime_ns != written
        KIT.verdict("case (b): the second run logs one cleanup line, rewrites 0.h5 and removes the stale file",
                    second.return_code == 0 and len(second_lines) == 1 and rewritten and not stale.exists(),
                    {"return_code": 0, "cleanup_lines": 1, "rewritten": True, "stale": False},
                    {"return_code": second.return_code, "cleanup_lines": second_lines, "rewritten": rewritten,
                     "stale": stale.exists()})


def locked_output() -> None:
    """Case (d): a locked 0.h5 cannot be removed; the run fails loudly instead of reusing it."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        target = testkit.output_file(config_file, "trigger")
        target.parent.mkdir(parents=True)
        target.write_bytes(PLANTED_BYTES)
        with open(target, "rb"):
            probe = testkit.run_child(KIT, folder, "unlink", ["-c", UNLINK_CODE, str(target)])
            child = testkit.run_pipeline(KIT, folder, "locked", config_file, ["trigger"])
        KIT.verdict("case (d) precondition: a child cannot delete the 0.h5 this process keeps locked (PermissionError)",
                    probe.return_code not in (0, None) and PERMISSION in probe.text, {"return_code": "!= 0", "error": PERMISSION},
                    {"return_code": probe.return_code, "error": PERMISSION in probe.text})
        KIT.verdict(f"case (d): with 0.h5 locked, --task trigger returns non-zero with {PERMISSION} and {target.name} in "
                    f"its output", child.return_code not in (0, None) and PERMISSION in child.text and target.name in child.text,
                    {"return_code": "!= 0", "error": PERMISSION, "file": target.name},
                    {"return_code": child.return_code, "error": PERMISSION in child.text,
                     "planted_file_left": target.is_file() and target.read_bytes() == PLANTED_BYTES,
                     "last": testkit.plain_lines(child.text)[-2:]})


def direct_build() -> None:
    """Case (e): on a fresh output tree, a luigi.build of Trigger, then of Signal, outside _run_task, writes each 0.h5."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        config = testkit.validated_config(config_file)
        trigger, signal = testkit.output_file(config_file, "trigger"), testkit.output_file(config_file, "signal")
        fresh = not trigger.parent.exists() and not signal.parent.exists()
        tasks = (pipeline._create_task(config, "trigger"), pipeline._create_task(config, "signal"))
        built = [bool(luigi.build([task], local_scheduler=True, log_level=LUIGI_LOG_LEVEL)) for task in tasks]
        written = [trigger.is_file(), signal.is_file()]
        KIT.verdict("case (e): on a fresh output tree, luigi.build of Trigger, then of Signal, outside _run_task each "
                    "succeed and write their 0.h5", fresh and all(built) and all(written),
                    {"fresh": True, "built": [True, True], "written": [True, True]},
                    {"fresh": fresh, "built": built, "written": written})


def cleaned_lines(child: testkit.Child) -> list:
    """The cleanup records of a pipeline child."""
    return [line for line in testkit.plain_lines(child.text) if CLEANED in line]


if __name__ == "__main__":
    main()
