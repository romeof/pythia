"""EXP-02: trigger decisions independent of the trigger_value recording mode (HYP-03).

Six series from the raw input (Read + Preprocess) go through analyze_series in ideal and
realistic mode under detection_row and following_row, with the shipped gate items and with one binding
item, var4 >= the lower middle value of var4 on the bar before each candidate row (the rows where each
mode evaluates the gate), which blocks the candidate rows below it. Expected by
atp_trigger-value-usage.md:12: begin, end and end_reason columns identical across the two
recording modes, because the gate reads bar i-1's gate columns whatever the recording mode (Q3).
Planted-defect control: one flipped begin in one mode's copy is flagged.

Structure:
1. main
2. Candidate rows and the binding threshold
3. Decision comparison
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipelinekit
from src.utils.common.constants import (
    END_REASON, GATE_OP_GE, TRIGGER_MINUS_BEGIN, TRIGGER_MINUS_END, TRIGGER_PLUS_BEGIN, TRIGGER_PLUS_END,
    TRIGGER_VALUE_FOLLOWING_ROW, TRIGGER_VALUE_DETECTION_ROW, VAR4,
)
from src.utils.common.schema import AppConfig

EXPERIMENT = "02"
MODES = ("ideal", "realistic")
RECORDINGS = (TRIGGER_VALUE_DETECTION_ROW, TRIGGER_VALUE_FOLLOWING_ROW)
DECISION_COLUMNS = (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN, TRIGGER_PLUS_END, TRIGGER_MINUS_END, END_REASON)
BEGIN_COLUMNS = (TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN)


def main() -> None:
    """Both item lists, both modes, both recording modes; control last."""
    _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    shipped = pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config"))
    shipped_items = [item.model_dump() for item in shipped.trigger.gate.items]
    series = pipelinekit.input_series(shipped, folder / "input")
    rows = candidate_rows(series, shipped, shipped_items)
    threshold, blocked = pipelinekit.binding_gate(series, rows)
    CTX.log.info("candidate rows per series: %s", {sid: len(found) for sid, found in rows.items()})
    CTX.log.info("binding item %s %s %s; candidate rows it blocks %s", VAR4, GATE_OP_GE, threshold, blocked)
    CTX.verdict("non-vacuity: candidate rows the binding item blocks",
                sum(len(found) for found in blocked.values()) > 0, "> 0",
                {sid: len(found) for sid, found in blocked.items()})
    results = {}
    for name, items in (("shipped items", shipped_items),
                        ("binding item", [pipelinekit.gate_item(VAR4, GATE_OP_GE, threshold)])):
        for mode in MODES:
            outputs = {recording: {sid: pipelinekit.analyze(sid, frame, configured(shipped, mode, recording, items))
                                   for sid, frame in sorted(series.items())} for recording in RECORDINGS}
            differences = {sid: decision_differences(outputs[RECORDINGS[0]][sid], outputs[RECORDINGS[1]][sid])
                           for sid in sorted(series)}
            compared = sum(len(frame) for frame in outputs[RECORDINGS[0]].values())
            results[f"{name}/{mode}"] = {sid: found for sid, found in differences.items() if found}
            CTX.log.info("%s, %s mode: %d rows compared, differences %s", name, mode, compared,
                         json.dumps(results[f"{name}/{mode}"], default=str))
            CTX.verdict(f"{name}, {mode} mode: decisions identical across recording modes (Q3)",
                        not results[f"{name}/{mode}"], "no difference",
                        {sid: found[:6] for sid, found in results[f"{name}/{mode}"].items()})
            CTX.verdict(f"{name}, {mode} mode: non-vacuity", compared > 0, "rows compared > 0", compared)
    check_control(series, shipped, shipped_items)
    CTX.log.info("RESULTS %s", json.dumps({"threshold": threshold, "blocked": blocked, "differences": results},
                                          sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# CANDIDATE ROWS AND THE BINDING THRESHOLD
# =============================================================================


def candidate_rows(series: dict[str, pd.DataFrame], shipped: AppConfig, items: list[dict]) -> dict[str, list[int]]:
    """Ideal begin rows (the gate is applied to every begin) and the realistic loop's gate rows under the items."""
    ideal = configured(shipped, "ideal", TRIGGER_VALUE_DETECTION_ROW, items)
    realistic = configured(shipped, "realistic", TRIGGER_VALUE_DETECTION_ROW, items)
    rows = {}
    for sid, frame in sorted(series.items()):
        marked = pipelinekit.strategy_of(ideal)._generate_trigger_begin(frame.copy())
        begins = np.flatnonzero((marked[TRIGGER_PLUS_BEGIN].to_numpy() > 0) | (marked[TRIGGER_MINUS_BEGIN].to_numpy() > 0))
        rows[sid] = sorted(set(begins.tolist()) | set(pipelinekit.realistic_gate_rows(sid, frame, realistic)))
    return rows


def configured(shipped: AppConfig, mode: str, recording: str, items: list[dict]) -> AppConfig:
    """The shipped config in one mode, recording mode and gate item list."""
    return pipelinekit.with_values(shipped, {"trigger.replay.mode": mode,
                                             "trigger.trigger_value": recording,
                                             "trigger.gate.items": items})


# =============================================================================
# DECISION COMPARISON
# =============================================================================


def decision_differences(left: pd.DataFrame, right: pd.DataFrame) -> list:
    """(column, date, left value, right value) of every differing decision cell."""
    found = []
    for column in DECISION_COLUMNS:
        differs = left[column].to_numpy() != right[column].to_numpy()
        found += [(column, str(left.index[row].date()), left[column].iat[row], right[column].iat[row])
                  for row in np.flatnonzero(differs)]
    return found


def check_control(series: dict[str, pd.DataFrame], shipped: AppConfig, items: list[dict]) -> None:
    """A flipped begin in one copy of an identical output must be flagged."""
    sid = min(series)
    config = configured(shipped, "realistic", TRIGGER_VALUE_DETECTION_ROW, items)
    output = pipelinekit.analyze(sid, series[sid], config)
    planted = copy.deepcopy(output)
    row = int(np.flatnonzero(planted[TRIGGER_PLUS_BEGIN].to_numpy() > 0)[0])
    planted.iat[row, planted.columns.get_loc(TRIGGER_PLUS_BEGIN)] = 0
    found = decision_differences(output, planted)
    CTX.verdict("control: one flipped begin is flagged", len(found) == 1 and found[0][0] == TRIGGER_PLUS_BEGIN,
                "one difference in trigger_plus_begin", found)


if __name__ == "__main__":
    main()
