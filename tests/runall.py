"""Runs every regression script under tests/ and returns 1 when any of them fails (D-2).

The first verdict checks that the cwd is the repository root (the children inherit it, and the
regressions resolve their config copies from it); when it fails, the summary follows at once,
before any scratch folder or child exists. Then every tests/*.py except this script and
testkit.py, sorted by name, runs as a child bounded by testkit.PIPELINE_TIMEOUT_S, its capture
in this script's scratch folder. One verdict per script: the child returns 0 without a FAIL
verdict line and, unless it prints without testkit (test_metrics_parity.py), after its SUMMARY
line, so a testkit script that ends before KIT.finish() fails. The verdict holds the return code,
the timeout flag, the child's SUMMARY line and its FAIL verdict lines, plus the capture's last
30 lines when the return code is non-zero, the child timed out or no SUMMARY line exists: the only
record of the capture once the scratch folder is removed.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
from pathlib import Path

SKIPPED = ("runall.py", "testkit.py")
WITHOUT_TESTKIT = ("test_metrics_parity.py",)
SUMMARY_LINE = re.compile(r"SUMMARY \d+ PASS, \d+ FAIL")
FAIL_PREFIX = "FAIL "
TAIL_LINES = 30


def main() -> None:
    """The cwd check, then one child per regression script, then the summary (return code 1 on any FAIL)."""
    cwd = Path.cwd().resolve()
    if not KIT.verdict("the cwd is the repository root", cwd == KIT.root, str(KIT.root), str(cwd)):
        KIT.finish()
    scripts = sorted(path for path in (KIT.root / "tests").glob("*.py") if path.name not in SKIPPED)
    with testkit.scratch(KIT) as folder:
        for script in scripts:
            child = testkit.run_child(KIT, folder, script.stem, [str(script)], timeout_s=testkit.PIPELINE_TIMEOUT_S)
            lines = testkit.plain_lines(child.text)
            summary = [match.group(0) for match in map(SUMMARY_LINE.search, lines) if match]
            fails = [text for text in map(message, lines) if text.startswith(FAIL_PREFIX)]
            found = {"return_code": child.return_code, "timed_out": child.timed_out, "summary": summary[-1] if summary else None,
                     "fails": fails}
            if child.return_code != 0 or child.timed_out or not summary:
                found["tail"] = lines[-TAIL_LINES:]
            needs_summary = script.name not in WITHOUT_TESTKIT
            summary_rule = ", after its SUMMARY line" if needs_summary else ""
            KIT.verdict(f"{script.name}: returns 0 without a FAIL verdict{summary_rule}",
                        child.return_code == 0 and not fails and (bool(summary) or not needs_summary),
                        {"return_code": 0, "fails": [], "summary": "present" if needs_summary else "not required"}, found)
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def message(line: str) -> str:
    """The message of a testkit log line (the text after the logger name), else the whole line."""
    return line.split(": ", 1)[1] if ": " in line else line


if __name__ == "__main__":
    main()
