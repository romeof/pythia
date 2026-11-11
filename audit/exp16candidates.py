"""EXP-16: realistic candidate selection and minimum-duration suppression (Q6, Q7).

A synthetic series (a plateau at 150, then a triangle wave of amplitude 1 around 100) hides
every wave extremum while the plateau is in the rolling window (prominence 2 against 5 % of a
range of 51) and shows them all on the row the plateau leaves it. (a) Q7: the row before that one
fails the gate item var5 >= 2, so the newest candidate of each type is suppressed there; under the decided
reading only the newest candidate of a type seen so far may activate, so no older one does
(the audited revision activated an older one on the next row). (b) Q6: with a minimum duration of eight rows,
the first opposite extremum is suppressed for good and the trigger ends at a later one; the
deferred reading is not observed. Wrappers record the loop's detected candidates, activations
and suppressions. --real-data runs the newest-seen recorder on every series of the reference
input in realistic mode and lists each activation whose candidate is older than the newest
detected candidate of its type seen so far (A3's attribution of Q7).

Structure:
1. main
2. The synthetic series and the recorded run
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

import pipelinekit
import synthetic
from src.utils.common.constants import (
    GATE_OP_GE, SUPPRESSION_MINIMUM_DURATION, TRIGGER_MINUS_BEGIN, TRIGGER_PLUS_BEGIN, VAR5,
)
from src.utils.common.schema import AppConfig
from src.utils.trigger.realistic import RealisticModeMixin

EXPERIMENT = "16"
REALISTIC = "realistic"
PLATEAU_ROWS, PLATEAU_VALUE = 60, 150.0
WAVE_LEVEL, WAVE_STEP, WAVE_PERIOD, WAVE_ROWS = 100.0, 0.25, 16, 160
START = "2019-01-01"
STARVED_VAR5, GATE_VAR5 = 1, 2.0  # var5 on the row before the gate row, and the item var5 >= 2 that row fails
MIN_DURATION = 8


def main() -> None:
    """(a) the newest-seen rule after a suppressed newer candidate; (b) the opposite extremum lost within the minimum duration; with
    --real-data, the older activations of the reference input."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    config = pipelinekit.concrete(pipelinekit.load_copy(pipelinekit.write_config_copy(CTX, folder / "config")), REALISTIC)
    if args.real_data:
        real_data(config, folder)
        CTX.summary()
        return
    locked = pipelinekit.concrete(pipelinekit.load_copy(pipelinekit.write_config_copy(
        CTX, folder / "minimumduration", strategy={"trigger.replay.min_duration_periods": MIN_DURATION,
                                                   **pipelinekit.GATE_OFF})), REALISTIC)
    frame = series()
    gate_row = PLATEAU_ROWS + config.trigger.replay.lookback_window - 1
    starved = frame.copy()
    starved.iloc[gate_row - 1, starved.columns.get_loc(VAR5)] = STARVED_VAR5
    gated = pipelinekit.with_values(config, {"trigger.gate.items": [pipelinekit.gate_item(VAR5, GATE_OP_GE, GATE_VAR5)]})
    found = {"q7": q7(gated, starved, gate_row), "q6": q6(locked, frame)}
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root and --real-data."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument("--real-data", action="store_true")
    return parser.parse_args()


def series() -> pd.DataFrame:
    """The plateau rows, then the triangle wave: level + step x (4 - |(t + 4) mod 16 - 8|), exact in float32."""
    quarter = WAVE_PERIOD // 4
    steps = quarter - np.abs((np.arange(WAVE_ROWS) + quarter) % WAVE_PERIOD - 2 * quarter)
    values = np.concatenate([np.full(PLATEAU_ROWS, PLATEAU_VALUE), WAVE_LEVEL + WAVE_STEP * steps])
    return synthetic.daily_frame(values, START)


def q7(config: AppConfig, frame: pd.DataFrame, gate_row: int) -> dict:
    """The newest candidates suppressed on the gate row, whose previous row is starved; every activation against the
    newest candidate seen."""
    events = recorded_events(config, frame)
    suppressed = [event for event in events if event["kind"] == "suppressed" and event["row"] == gate_row]
    activated = next((event for event in events if event["kind"] == "activated" and event["row"] > gate_row), None)
    newer = [event for event in suppressed if activated is not None and event["type"] == activated["type"]
             and activated["candidate_row"] < event["candidate_row"]]
    activations, older = older_activations("recorded", frame, config)
    found = {"gate_row": gate_row, "suppressed_at_gate": suppressed, "first_activation_after": activated,
             "newer_suppressed_of_its_type": newer, "activations": len(activations), "older_than_newest_seen": older}
    CTX.verdict("Q7 as coded (newest only): no activation's candidate is older than the newest detected candidate of its "
                "type seen so far", bool(activations) and not older, "activations, none older", found)
    CTX.verdict("Q7: the older activation of the audited revision (an older candidate after a suppressed newer one) is no longer "
                "observed", bool(suppressed) and not newer, "suppressed newer candidates, no older activation", found)
    return found


def q6(config: AppConfig, frame: pd.DataFrame) -> dict:
    """Opposite extrema suppressed within the minimum duration, whether any is activated later, and the end that happens."""
    events = recorded_events(config, frame)
    lost = [event for event in events if event["kind"] == "suppressed" and event["reason"] == SUPPRESSION_MINIMUM_DURATION]
    activations = [event for event in events if event["kind"] == "activated"]
    acted = {event["candidate_row"] for event in activations}
    never = [event for event in lost if event["candidate_row"] not in acted]
    found = {"min_duration": MIN_DURATION, "suppressed_within_min_duration": lost, "never_activated": never, "activations": activations[:4]}
    CTX.verdict("Q6 as coded: an opposite extremum seen within the minimum duration is suppressed for good", bool(lost) and len(never) == len(lost),
                "every one lost", found)
    CTX.verdict("Q6: the deferred reading is not observed: no opposite extremum lost within the minimum duration is acted on later "
                "(non-vacuous)", bool(lost) and not any(event["candidate_row"] in acted for event in lost),
                "every lost one never acted on", found)
    return found


def real_data(config: AppConfig, folder: Path) -> dict:
    """Per series of the reference input (realistic, shipped config): the activations whose candidate is older than
    the newest detected candidate of its type seen so far; the first such row bounds W5's first differing flag."""
    series = pipelinekit.input_series(config, folder / "input")
    found = {}
    for sid, frame in sorted(series.items()):
        activations, older = older_activations(sid, frame, config)
        found[sid] = {"activations": len(activations), "older": older, "first_row": older[0]["row"] if older else None}
        CTX.log.info("REAL %s %s", sid, json.dumps(found[sid], default=str))
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.verdict("real data: non-vacuity, activations recorded in every series",
                all(item["activations"] > 0 for item in found.values()), "> 0 each",
                {sid: item["activations"] for sid, item in found.items()})
    CTX.verdict("real data (Q7): no activation's candidate is older than the newest detected candidate of its type seen so far",
                not any(item["older"] for item in found.values()), "none",
                {sid: item["first_row"] for sid, item in found.items() if item["older"]})
    return found


def older_activations(series_id: str, frame: pd.DataFrame, config: AppConfig) -> tuple[list[dict], list[dict]]:
    """Every activation of one realistic run and those whose candidate is older than the newest candidate of its type
    detected so far (recorded run: _window_candidates and _activate_trigger wrapped). series_id is the id the run sees:
    the series' own id on real data, so a gate item's exempt_series applies to it as in the pipeline."""
    newest: dict = {}
    activations: list[dict] = []
    extract, activate = RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger

    def extracting(self, df, start, stop):
        plus, minus, count = extract(self, df, start, stop)
        for kind, detected in ((TRIGGER_PLUS_BEGIN, plus), (TRIGGER_MINUS_BEGIN, minus)):
            if detected:
                newest[kind] = max(detected) if kind not in newest else max(newest[kind], max(detected))
        return plus, minus, count

    def activation(self, df, i, current_idx, new_candidate, state):
        kind, candidate = new_candidate
        activations.append({"row": i, "type": kind, "candidate_row": int(frame.index.get_loc(candidate)),
                            "newest_row": int(frame.index.get_loc(newest[kind]))})
        return activate(self, df, i, current_idx, new_candidate, state)

    RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger = extracting, activation
    try:
        pipelinekit.analyze(series_id, frame, config)
    finally:
        RealisticModeMixin._window_candidates, RealisticModeMixin._activate_trigger = extract, activate
    return activations, [item for item in activations if item["candidate_row"] < item["newest_row"]]


def recorded_events(config: AppConfig, frame: pd.DataFrame) -> list[dict]:
    """Activations and suppressions of one realistic run: kind, row, type, candidate row and date, reason."""
    events: list[dict] = []
    check, pending, activate = (RealisticModeMixin._check_suppression_conditions, RealisticModeMixin._suppress_pending_opposite,
                                RealisticModeMixin._activate_trigger)

    def record(kind: str, row: int, trigger_type: str, candidate_date: pd.Timestamp, reason: str = "") -> None:
        events.append({"kind": kind, "row": row, "type": trigger_type, "candidate_row": int(frame.index.get_loc(candidate_date)),
                       "candidate": str(candidate_date.date()), "reason": reason})

    def checking(self, i, candidate_date, trigger_type, state, effective_start_date=None):
        reason = check(self, i, candidate_date, trigger_type, state, effective_start_date)
        if reason:
            record("suppressed", i, trigger_type, candidate_date, reason)
        return reason

    def pending_suppression(self, series_id, df, window_df, i, current_idx, current_value, pending_opposite, reason, *args):
        record("suppressed", i, pending_opposite[0], pending_opposite[1], reason)
        return pending(self, series_id, df, window_df, i, current_idx, current_value, pending_opposite, reason, *args)

    def activation(self, df, i, current_idx, new_candidate, state):
        record("activated", i, new_candidate[0], new_candidate[1])
        return activate(self, df, i, current_idx, new_candidate, state)

    RealisticModeMixin._check_suppression_conditions = checking
    RealisticModeMixin._suppress_pending_opposite = pending_suppression
    RealisticModeMixin._activate_trigger = activation
    try:
        pipelinekit.analyze("synthetic", frame, config)
    finally:
        RealisticModeMixin._check_suppression_conditions, RealisticModeMixin._suppress_pending_opposite = check, pending
        RealisticModeMixin._activate_trigger = activate
    return events


if __name__ == "__main__":
    main()
