"""Regression for the store line of open_store (G-10, F-16).

The test writes a small store into its scratch folder with pd.HDFStore. Case (a): a child whose
body inside hdf5_manager.open_store(<that store>, "r") raises a planted OSError: its capture holds
no "Error opening HDF5 store" line, and it returns 1 with the planted message. Case (b): a child
calls open_store on 0.h5 in an absent folder: exactly one store line naming that path, then the
FileNotFoundError, return code 1 (both trees). The child logs to stderr, line-buffered like the
interpreter's traceback, so the capture keeps the sequence of both. The unfixed open_store logs
the body's error as a failure to create the store.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import pandas as pd

STORE_LINE = "Error opening HDF5 store"
FILE_ERROR = "FileNotFoundError"
UNCAUGHT_RETURN_CODE = 1
KEY = "/frame"
ABSENT_FOLDER = "absent"
CHILD_CODE = testkit.CHILD_BOOTSTRAP + (
    "import logging\n"
    "from pathlib import Path\n"
    "from src.utils.common.hdf5manager import hdf5_manager\n"
    "logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')\n"
    "with hdf5_manager.open_store(Path(sys.argv[4]), 'r') as store:\n"
    f"    raise OSError('{testkit.PLANTED_ERROR}')\n")


def main() -> None:
    """Both cases, then the summary (return code 1 on any FAIL)."""
    body_failure()
    absent_store()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def body_failure() -> None:
    """Case (a): an exception of the body leaves without a store line, the child returns 1 with the planted message."""
    with testkit.scratch(KIT) as folder:
        path = folder / "0.h5"
        with pd.HDFStore(str(path), mode="w") as store:
            store.put(KEY, pd.DataFrame({"var3": [1.0, 2.0]}))
        child = testkit.run_child(KIT, folder, "body", ["-c", CHILD_CODE, *testkit.child_args(KIT), str(path)])
        lines = testkit.plain_lines(child.text)
        store_lines = [line for line in lines if STORE_LINE in line]
        KIT.verdict(f"case (a): no '{STORE_LINE}' line for an exception of the body", not store_lines, "none", store_lines)
        planted = [line for line in lines if testkit.PLANTED_ERROR in line]
        KIT.verdict(f"case (a): return code {UNCAUGHT_RETURN_CODE} with the planted message",
                    child.return_code == UNCAUGHT_RETURN_CODE and bool(planted), {"return_code": UNCAUGHT_RETURN_CODE, "planted": "present"},
                    {"return_code": child.return_code, "timed_out": child.timed_out, "planted": planted[-1:]})


def absent_store() -> None:
    """Case (b): one store line naming the absent path, then the FileNotFoundError, return code 1."""
    with testkit.scratch(KIT) as folder:
        path = folder / ABSENT_FOLDER / "0.h5"
        child = testkit.run_child(KIT, folder, "absent", ["-c", CHILD_CODE, *testkit.child_args(KIT), str(path)])
        lines = testkit.plain_lines(child.text)
        store_rows = [row for row, line in enumerate(lines) if STORE_LINE in line]
        KIT.verdict(f"case (b): exactly one '{STORE_LINE}' line, naming {path.name} in the absent folder",
                    len(store_rows) == 1 and str(path) in lines[store_rows[0]], 1, [lines[row] for row in store_rows])
        raised = [row for row, line in enumerate(lines) if line.startswith(FILE_ERROR)]
        KIT.verdict(f"case (b): then the {FILE_ERROR}, return code {UNCAUGHT_RETURN_CODE}",
                    bool(store_rows) and bool(raised) and raised[-1] > store_rows[-1] and child.return_code == UNCAUGHT_RETURN_CODE,
                    {"return_code": UNCAUGHT_RETURN_CODE, "after the store line": FILE_ERROR},
                    {"return_code": child.return_code, "timed_out": child.timed_out, "raised": [lines[row] for row in raised][-1:]})


if __name__ == "__main__":
    main()
