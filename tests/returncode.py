"""Regression for the return code and the failure cause of a failed task (KNOWN (b) generalised).

Case 1: --task trigger on a copy whose input is absent: the [Errno 2] cause is logged, the
realistic mode never starts after the ideal failure, and the run returns non-zero. Case 2: four
realistic copies whose task meets a guard (Signal without a trigger output, Performance without
upstream output, Performance with an empty trigger placeholder and no signal output, Trigger
with a planted OSError in create_strategy): each logs its guard line and returns non-zero. Case 3:
--task trigger on a copy with a planted OSError in Trigger._load_reference_series: exactly one
record with the failure handler's prefix names OSError and the planted message, then a non-zero
return code. The unfixed code logs (or hides) these failures and returns 0. (The planted failures replace
an omitted trigger.strategy.printout_io and an omitted read.reference_series, which the schema
defaults now fill, Q11.)

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
from pathlib import Path

ERRNO_LINE = "[Errno 2]"
REALISTIC_START = "Running mode: realistic"
ABSENT_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
REALISTIC = {"trigger.replay.mode": "realistic"}
NO_TRIGGER = "No trigger data found"
NO_SIGNAL = "No signal data found"
NO_STRATEGY = "Error creating trigger detection algorithm"
STRATEGY_TARGET = "src.utils.trigger.factory:create_strategy"
REFERENCE_TARGET = "src.tasks.trigger:Trigger._load_reference_series"
PLANTED_TYPE = "OSError"


def main() -> None:
    """The three cases, then the summary (return code 1 on any FAIL)."""
    absent_input()
    guards()
    failure_cause()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def absent_input() -> None:
    """Case 1: the cause is logged, no later mode runs, the return code is non-zero."""
    with testkit.scratch(KIT) as folder:
        raw = Path(os.path.relpath(folder / ABSENT_INPUT, Path.cwd())).as_posix()
        config_file = testkit.config_copy(KIT, folder, strategy={"read.raw_data": f"./{raw}"})
        child = testkit.run_pipeline(KIT, folder, "absent", config_file, ["trigger"])
        lines = testkit.plain_lines(child.text)
        KIT.verdict("case 1: the missing input's cause is logged", any(ERRNO_LINE in line for line in lines), ERRNO_LINE,
                    [line for line in lines if "ERROR" in line][:3])
        KIT.verdict("case 1: no realistic mode after the ideal failure", not any(REALISTIC_START in line for line in lines),
                    f"no '{REALISTIC_START}' line", sum(REALISTIC_START in line for line in lines))
        KIT.verdict("case 1: return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


def guards() -> None:
    """Case 2: each guard's line, then a non-zero return code."""
    for name, tasks, guard_line, prepare, target in (
            ("signal without trigger output", ["signal"], NO_TRIGGER, None, None),
            ("performance without upstream output", ["performance"], NO_TRIGGER, None, None),
            ("performance with an empty trigger placeholder", ["performance"], NO_SIGNAL, "trigger", None),
            ("trigger with a planted create_strategy failure", ["trigger"], NO_STRATEGY, None, STRATEGY_TARGET)):
        with testkit.scratch(KIT) as folder:
            config_file = testkit.config_copy(KIT, folder, strategy=dict(REALISTIC))
            if prepare is not None:
                placeholder = testkit.output_file(config_file, prepare)
                placeholder.parent.mkdir(parents=True)
                placeholder.touch()
            child = (testkit.run_pipeline(KIT, folder, "guard", config_file, tasks) if target is None
                     else testkit.planted_run(KIT, folder, "planted", config_file, target=target, task=tasks[0]))
            lines = testkit.plain_lines(child.text)
            KIT.verdict(f"case 2 ({name}): its guard line", any(guard_line in line for line in lines), guard_line,
                        [line for line in lines if "ERROR" in line][:3])
            KIT.verdict(f"case 2 ({name}): return != 0", child.return_code not in (0, None), "return != 0",
                        {"return_code": child.return_code, "timed_out": child.timed_out})


def failure_cause() -> None:
    """Case 3: exactly one handler record names OSError and the planted message, then a non-zero return code."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder)
        child = testkit.planted_run(KIT, folder, "planted", config_file, target=REFERENCE_TARGET, task="trigger")
        records = [line for line in testkit.plain_lines(child.text) if testkit.HANDLER_PREFIX in line]
        KIT.verdict(f"case 3: one '{testkit.HANDLER_PREFIX}' record naming {PLANTED_TYPE} and the planted message",
                    len(records) == 1 and PLANTED_TYPE in records[0] and testkit.PLANTED_ERROR in records[0], 1, records)
        KIT.verdict("case 3: return != 0", child.return_code not in (0, None), "return != 0",
                    {"return_code": child.return_code, "timed_out": child.timed_out})


if __name__ == "__main__":
    main()
