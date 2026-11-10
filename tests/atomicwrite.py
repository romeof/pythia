"""Regression for the atomic HDF5 write (HYP-07, R-X4).

Crash case: a child opens a store through hdf5_manager.open_store(path, "w"), writes one frame,
records a marker line and ends with os._exit(97) inside the with body (an uncaught exception
would return 1 and take the cleanup path instead of the hard crash); no file may remain at the
path. Completed case: an in-process write leaves the frame readable at the path and no
temporary file beside it. The unfixed code writes in place, so the crash leaves a partial file.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import pandas as pd

from src.utils.common.hdf5manager import hdf5_manager

CRASH_RETURN_CODE = 97
MARKER = "ATOMICWRITE frame written"
KEY = "/frame"
FRAME = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
TEMP_SUFFIX = ".tmp.h5"
CRASH_CODE = testkit.CHILD_BOOTSTRAP + (
    "import os; from pathlib import Path; import pandas as pd; "
    "from src.utils.common.hdf5manager import hdf5_manager\n"
    "with hdf5_manager.open_store(Path(sys.argv[4]), 'w') as store:\n"
    f"    hdf5_manager.put_dataframe_normalized(store, '{KEY}', pd.DataFrame({{'value': [1.0, 2.0, 3.0]}}), format='fixed')\n"
    f"    sys.stdout.write('{MARKER}\\n'); sys.stdout.flush(); os._exit({CRASH_RETURN_CODE})\n")


def main() -> None:
    """Both cases, then the summary (return code 1 on any FAIL)."""
    crashed_write()
    completed_write()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def crashed_write() -> None:
    """A hard crash inside the write: the marker and return code 97, then no file at the path."""
    with testkit.scratch(KIT) as folder:
        path = folder / "0.h5"
        child = testkit.run_child(KIT, folder, "crash", ["-c", CRASH_CODE, *testkit.child_args(KIT), str(path)])
        KIT.verdict("crash: the frame was written before the crash (marker)", MARKER in child.text, MARKER,
                    child.text.strip().splitlines()[-1:] if child.text.strip() else [])
        KIT.verdict(f"crash: the child ended by os._exit({CRASH_RETURN_CODE})", child.return_code == CRASH_RETURN_CODE, CRASH_RETURN_CODE,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        KIT.verdict("crash: no file at the output path", not path.exists(), "absent",
                    sorted(item.name for item in folder.iterdir()))


def completed_write() -> None:
    """A completed write: the frame readable at the path and no temporary file beside it."""
    with testkit.scratch(KIT) as folder:
        path = folder / "0.h5"
        with hdf5_manager.open_store(path, "w") as store:
            hdf5_manager.put_dataframe_normalized(store, KEY, FRAME, format="fixed")
        with pd.HDFStore(str(path), mode="r") as store:
            readable = store.get(KEY).equals(FRAME)
        KIT.verdict("completed: the frame is readable at the output path", readable, True, readable)
        KIT.verdict("completed: no temporary file remains", not (folder / f"{path.stem}{TEMP_SUFFIX}").exists(), "absent",
                    sorted(item.name for item in folder.iterdir()))


if __name__ == "__main__":
    main()
