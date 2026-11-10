"""Regression for the luigi.build result contract that pipeline.py's _run_task reads (E-4).

In-process, under the pinned luigi 3.6.0: luigi.build([task], local_scheduler=True) returns False
for a task whose run() raises and True for one whose run() returns, although luigi/interface.py's
docstring promises True "even if tasks may fail". A task with a short repr (parameter names under 60
characters, str(task) under 200) carrying SectionParameter(ReadConfig) with one-character strings and empty
lists builds True: luigi's execution summary then puts the section in a set, which pydantic's own frozen hash
refuses on list fields (TypeError: unhashable type: 'list') and ConfigModel's content hash admits; the same
task with a RuntimeConfig in its ReadConfig parameter raises ValidationError. The parameter's parse returns a section
equal to the one its serialize wrote, and a task built from that JSON text equals the task built from the instance
(normalize's text branch). Through pipeline children on one-series
realistic copies: --task signal without trigger outputs returns != 0 with "signal task failed" in
its capture, and --task trigger returns 0. A _run_task that misread the result would let the failed
build pass and fail the completed one.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import luigi
from pydantic import ValidationError

from src.tasks.sections import SectionParameter
from src.utils.common.schema import ReadConfig, RuntimeConfig

FAILED_TASK_LINE = "signal task failed"


class FailingTask(luigi.Task):
    """A task whose run() raises."""

    def run(self) -> None:
        raise OSError(testkit.PLANTED_ERROR)


class CompletingTask(luigi.Task):
    """A task whose run() returns."""

    def run(self) -> None:
        return None


class SectionTask(luigi.Task):
    """A task with a short repr carrying one section."""

    r = SectionParameter(ReadConfig)

    def run(self) -> None:
        return None


def main() -> None:
    """The two in-process builds and the two pipeline children, then the summary (return code 1 on any FAIL)."""
    build_results()
    section_build()
    run_task_results()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def build_results() -> None:
    """luigi.build returns False for a failed run() and True for a completed one."""
    failed = luigi.build([FailingTask()], local_scheduler=True)
    KIT.verdict("luigi.build of a task whose run() raises returns False", failed is False, False, failed)
    completed = luigi.build([CompletingTask()], local_scheduler=True)
    KIT.verdict("luigi.build of a task whose run() returns returns True", completed is True, True, completed)


def section_build() -> None:
    """A short-repr task carrying a list-bearing section builds; a foreign model in the parameter is refused; parse
    returns what serialize wrote, and the section's JSON text builds a task equal to the instance's."""
    read = ReadConfig(raw_data="a", output_root="b", ini_date="c", end_date="d", reference_series=[], skip_series=[])
    task = SectionTask(r=read)
    shape = {"parameter names": sum(len(name) for name, _ in SectionTask.get_params()), "str(task)": len(str(task))}
    KIT.verdict("a SectionParameter task: short repr (names under 60, str(task) under 200)",
                shape["parameter names"] < 60 and shape["str(task)"] < 200, "< 60, < 200", shape)
    built = luigi.build([task], local_scheduler=True)
    KIT.verdict("luigi.build of the short-repr task carrying a list-bearing section returns True", built is True, True,
                built)
    exc = testkit.raised(lambda: SectionTask(r=RuntimeConfig(queue_maxsize=1, queue_get_timeout=1)))
    KIT.verdict("a RuntimeConfig in the ReadConfig parameter raises ValidationError", isinstance(exc, ValidationError),
                "ValidationError", repr(exc)[:240])
    text = SectionTask.r.serialize(read)
    parsed = SectionTask.r.parse(text)
    KIT.verdict("SectionParameter.parse of serialize's JSON text returns an equal ReadConfig",
                isinstance(parsed, ReadConfig) and parsed == read, repr(read)[:240], repr(parsed)[:240])
    from_text = SectionTask(r=text)
    KIT.verdict("a task built from the section's JSON text equals the task built from the instance",
                from_text == task and from_text.task_id == task.task_id, task.task_id, from_text.task_id)


def run_task_results() -> None:
    """_run_task fails the run on a failed build and passes a completed one."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        child = testkit.run_pipeline(KIT, folder, "signal", config_file, ["signal"])
        found = [line for line in testkit.plain_lines(child.text) if FAILED_TASK_LINE in line]
        KIT.verdict(f"--task signal without trigger outputs: return != 0 with '{FAILED_TASK_LINE}'",
                    child.return_code not in (0, None) and bool(found), {"return_code": "!= 0", "line": "present"},
                    {"return_code": child.return_code, "timed_out": child.timed_out, "line": found[-1:]})
        child = testkit.run_pipeline(KIT, folder, "trigger", config_file, ["trigger"])
        KIT.verdict("--task trigger returns 0", child.return_code == 0, 0, {"return_code": child.return_code, "timed_out": child.timed_out})


if __name__ == "__main__":
    main()
