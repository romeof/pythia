"""Regression for the source of the git provenance (E-5).

In-process: a scratch folder holds its own one-commit repository and becomes the working
directory; _get_git_output(["git", "rev-parse", "HEAD"]) then returns the commit of the tree
holding the pipeline source (KIT.root), not the scratch repository's. The unfixed call runs git
in the working directory, so it returns the scratch commit and git_info.txt names code that did
not run.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import os
import subprocess

from src.utils.common.outputs import _get_git_output

HEAD_COMMAND = ["git", "rev-parse", "HEAD"]
SCRATCH_IDENTITY = ["-c", "user.name=gitsource", "-c", "user.email=gitsource@example.com"]


def main() -> None:
    """Run the call from inside a foreign repository, then the summary (return code 1 on any FAIL)."""
    source_head = git_text(KIT.root, HEAD_COMMAND)
    with testkit.scratch(KIT) as folder:
        git_text(folder, ["git", "init", "-q"])
        git_text(folder, ["git", *SCRATCH_IDENTITY, "commit", "-q", "--allow-empty", "-m", "scratch"])
        scratch_head = git_text(folder, HEAD_COMMAND)
        previous = os.getcwd()
        os.chdir(folder)
        try:
            recorded = _get_git_output(HEAD_COMMAND)
        finally:
            os.chdir(previous)
    KIT.verdict("git provenance names the source tree's commit, not the working directory's",
                recorded == source_head and recorded != scratch_head,
                {"recorded": source_head}, {"recorded": recorded, "scratch_head": scratch_head})
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def git_text(folder, command: list) -> str:
    """Run a git command in folder, bounded by testkit.HELPER_BOUND_S, and return its stripped stdout."""
    done = subprocess.run(command, cwd=folder, stdin=subprocess.DEVNULL, capture_output=True, check=True,
                          timeout=testkit.HELPER_BOUND_S)
    return done.stdout.decode("utf-8").strip()


if __name__ == "__main__":
    main()
