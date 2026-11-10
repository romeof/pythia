"""EXP-15: trigger_value selection (R-T5, R-T16).

Direct select_trigger_value_vector calls: the last-row fallback, a NaN var0[D+1] and an absent
var0 column (common.py:91-92, :73-75).

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common.common import select_trigger_value_vector
from src.utils.common.constants import TRIGGER_VALUE_FOLLOWING_ROW, TRIGGER_VALUE_DETECTION_ROW, VAR0, VAR3

FRAME = pd.DataFrame({VAR3: [10.0, 11.0, 12.0, 13.0], VAR0: [9.0, np.nan, 11.5, 12.5]},
                     index=pd.bdate_range("2022-01-03", periods=4))


def main() -> None:
    """The documented rules on a hand frame."""
    _parse_args()
    same = select_trigger_value_vector(FRAME, TRIGGER_VALUE_DETECTION_ROW, primary_col=VAR3, start_col=VAR0)
    after = select_trigger_value_vector(FRAME, TRIGGER_VALUE_FOLLOWING_ROW, primary_col=VAR3, start_col=VAR0)
    absent = select_trigger_value_vector(FRAME.drop(columns=[VAR0]), TRIGGER_VALUE_FOLLOWING_ROW, primary_col=VAR3, start_col=VAR0)
    CTX.verdict("detection_row records var3", np.allclose(same, FRAME[VAR3]), FRAME[VAR3].tolist(), same.tolist())
    CTX.verdict("following_row: var0[D+1], var3[D] where var0[D+1] is NaN and on the last row",
                bool(np.allclose(after, [10.0, 11.5, 12.5, 13.0])), [10.0, 11.5, 12.5, 13.0], after.tolist())
    CTX.verdict("following_row without a var0 column records var3 on every row", np.allclose(absent, FRAME[VAR3]),
                FRAME[VAR3].tolist(), absent.tolist())
    found = {"detection_row": same.tolist(), "following_row": after.tolist(), "absent_var0": absent.tolist()}
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


if __name__ == "__main__":
    main()
