"""Regression for a run whose triggers form no signal (HYP-27, R-P9, performance.md:79).

Case (a): --task all on a one-series realistic copy whose one gate item, var5 >= 1e18, fails on
every row, so the gate blocks every begin: no /triggers_* key (non-vacuity), the "No signals found in
signal HDF5" warning, no ERROR line, return code 0 and a performance store holding /strategy_name (the
documented empty outputs). Case (b), after (a)'s verdicts: a zero-byte file replaces the signal
output and --task performance runs on the same copy: no performance store and a non-zero return code (a
failed read must not take the empty path). The unfixed code hides a KeyError 'is_signal' in (a)
and takes the empty path on the unreadable file in (b).

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path

UNREACHABLE_VAR5 = 1.0e18
NO_SIGNALS = "No signals found in signal HDF5"
STRATEGY_KEY = "/strategy_name"
TRIGGERS_PREFIX = "/triggers_"
ERROR_MARK = "ERROR"


def main() -> None:
    """Both cases on one copy, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={
            **testkit.ONE_SERIES, "trigger.gate.items": [testkit.gate_item("var5", ">=", UNREACHABLE_VAR5)]})
        empty_path(folder, config_file)
        unreadable_signal_file(folder, config_file)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def empty_path(folder: Path, config_file: Path) -> None:
    """Case (a): no signal at all: the documented empty outputs and return code 0."""
    child = testkit.run_pipeline(KIT, folder, "zero", config_file, ["all"])
    lines = testkit.plain_lines(child.text)
    trigger_keys = testkit.store_keys(testkit.output_file(config_file, "trigger"))
    KIT.verdict("case (a): non-vacuity: a trigger store without /triggers_* keys",
                trigger_keys is not None and not any(key.startswith(TRIGGERS_PREFIX) for key in trigger_keys),
                "no /triggers_* key", trigger_keys)
    KIT.verdict(f"case (a): the '{NO_SIGNALS}' warning", any(NO_SIGNALS in line for line in lines), NO_SIGNALS,
                sum(NO_SIGNALS in line for line in lines))
    KIT.verdict("case (a): no ERROR line", not any(ERROR_MARK in line for line in lines), "none",
                [line for line in lines if ERROR_MARK in line][:3])
    KIT.verdict("case (a): return code 0", child.return_code == 0, 0, {"return_code": child.return_code, "timed_out": child.timed_out})
    performance_keys = testkit.store_keys(testkit.output_file(config_file, "performance"))
    KIT.verdict(f"case (a): a performance store holding {STRATEGY_KEY}",
                performance_keys is not None and STRATEGY_KEY in performance_keys, STRATEGY_KEY, performance_keys)


def unreadable_signal_file(folder: Path, config_file: Path) -> None:
    """Case (b): an unreadable signal file fails the task instead of taking the empty path."""
    testkit.output_file(config_file, "signal").write_bytes(b"")
    performance_file = testkit.output_file(config_file, "performance")
    child = testkit.run_pipeline(KIT, folder, "unreadable", config_file, ["performance"])
    KIT.verdict("case (b): no performance store", not performance_file.exists(), "absent", performance_file.exists())
    KIT.verdict("case (b): return != 0", child.return_code not in (0, None), "return != 0",
                {"return_code": child.return_code, "timed_out": child.timed_out})


if __name__ == "__main__":
    main()
