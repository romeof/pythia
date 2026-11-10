"""Regression for a tracked document that names a test script the tree no longer holds (README.md:134).

Every tracked *.md file outside plans/ (git ls-files; untracked files are not read) is searched
for tests/<name>.py paths, each resolved against the repository root; plans/ holds the planloop
runs' historical records, which name the scripts of their own time and stay unedited. One
verdict: every named script exists; non-vacuity: at least one is named (README's own test
commands). The unfixed tree names the replay validation suite that D-1 replaced with
tests/replayvalidation.py at README.md:134.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
import subprocess

TESTS_DIR = "tests"
DOCUMENTS = "*.md"
PLANS_DIR = "plans"
SCRIPT_REFERENCE = re.compile(rf"\b{TESTS_DIR}[/\\]((?:[\w-]+[/\\])*[\w-]+\.py)\b")
SEPARATOR = re.compile(r"[/\\]")


def main() -> None:
    """Every script a tracked document outside plans/ names against the tree, then the summary (return code 1 on any FAIL)."""
    named = [(f"{document}:{number}", name) for document in tracked_documents()
             for number, line in enumerate((KIT.root / document).read_text(encoding="utf-8").splitlines(), start=1)
             for name in SCRIPT_REFERENCE.findall(line)]
    missing = [f"{site} {TESTS_DIR}/{name}" for site, name in named
               if not KIT.root.joinpath(TESTS_DIR, *SEPARATOR.split(name)).is_file()]
    KIT.verdict(f"every {TESTS_DIR}/ script a tracked document outside {PLANS_DIR}/ names exists",
                bool(named) and not missing,
                {"named": ">= 1", "missing": []}, {"named": len(named), "missing": missing})
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def tracked_documents() -> list[str]:
    """Repository-relative paths of the tracked *.md files outside plans/, sorted; git bounded by testkit.HELPER_BOUND_S."""
    listed = subprocess.run(["git", "ls-files", "-z", "--", DOCUMENTS, f":(exclude){PLANS_DIR}/"], cwd=KIT.root,
                            stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=testkit.HELPER_BOUND_S)
    return sorted(path for path in listed.stdout.decode("utf-8").split("\0") if path)


if __name__ == "__main__":
    main()
