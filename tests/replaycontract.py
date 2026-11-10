"""Regression for the replay contract: the replay section, its end key and the end_reason column (W1, W2, W8).

(a) The root's shipped workflows/extrema.yaml, parsed, holds trigger.replay.end.different_end_strategy and
trigger.consecutive_rows_with_bad_preprocessed_data at those paths, and validate_config accepts the merged
configuration load_config returns. (b) Every model reachable from the validator's root model, AppConfig,
through its fields' annotations forbids extra keys and is frozen; two validations of the configuration hash
equal, the read section with its list fields included; an attribute assignment raises ValidationError; and a
neutral sentinel key planted in trigger, in the replay section and in the replay section's one mapping-valued
field (both found by content) raises ValidationError: no retired key can pass, the sweep proving that none
remains. (c) --task trigger on a copy
keeping series 0, the shipped dual mode: per mode, /series_0's columns are the fifteen names below,
end_reason among them (compared as sets: the modes write their last trigger columns in different
sequences). (d) The V fixture of endmodes.py (var3 through the knots (0, 140), (40, 100), (60, 120),
(100, 80)) as a raw input on consecutive calendar days, realistic, a rolling 20-row window, minimum
duration 1, the find_peaks values endmodes.py pins, different_end_strategy the only end mode: --task
trigger returns 0 and /series_0's end_reason on row 61 is different_end_strategy, the plus trigger of row
41 ending at the window's plus end. A tree whose YAML spells the retired section or key fails (a) and its own
schema refuses the copy of (d); a tree that writes the retired column fails (c).

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import copy
import os
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError

from src.utils.common.config import load_config, validate_config
from src.utils.common.schema import AppConfig

SHIPPED_PATHS = ("trigger.replay.end.different_end_strategy", "trigger.consecutive_rows_with_bad_preprocessed_data")
SENTINEL = "planted_sentinel_key"
FORBID = "forbid"
KEEP_SERIES_ZERO = {"read.skip_series": ["1", "2", "3", "4", "5"]}
SERIES_KEY = "/series_0"
SERIES_COLUMNS = {"var0", "var1", "var2", "var3", "var4", "var5", "trigger_value", "end_reason",
                  "trigger_plus_begin", "trigger_minus_begin", "turning_points", "trigger_plus_end", "trigger_minus_end",
                  "active_trigger_plus_begin", "active_trigger_minus_begin"}
V_INPUT = Path("input") / "2026_01_01" / "0" / "raw" / "0.h5"
V_KNOTS = [(0, 140.0), (40, 100.0), (60, 120.0), (100, 80.0)]
V_DATES = pd.date_range("2020-01-01", periods=V_KNOTS[-1][0] + 1, freq="D")
# replay-section leaves of case (d), each set through testkit.replay_path
V_REPLAY = {"mode": "realistic", "lookback_window": 20, "window_type": "rolling", "min_duration_periods": 1,
            "end.max_duration_periods": None, "end.against_threshold_pct": None,
            "end.along_threshold_pct": None, "end.opposite_trigger_as_begin_strategy": False,
            "end.different_end_strategy": True}
V_SETTINGS = {"read.ini_date": V_DATES[0].strftime("%Y-%m-%d"), "read.end_date": V_DATES[-1].strftime("%Y-%m-%d"),
              "trigger.strategy.min_distance": 5, "trigger.strategy.prominence_threshold": 0.05, "trigger.strategy.width": 1}
END_COLUMN, END_ROW, END_REASON = "end_reason", 61, "different_end_strategy"


def main() -> None:
    """The shipped YAML, the schema, the persisted column and the end key, then the summary (return code 1 on any FAIL)."""
    shipped_file = KIT.root / "workflows" / testkit.STRATEGY_FILE
    shipped_paths(shipped_file)
    forbidden_extras(load_config(str(shipped_file), testkit.DEV_PROFILE))
    series_columns()
    end_key()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def shipped_paths(shipped_file: Path) -> None:
    """(a) The two paths exist in the parsed shipped YAML; validate_config accepts the merged configuration."""
    shipped = yaml.safe_load(shipped_file.read_text(encoding="utf-8"))
    present = {path: value_present(shipped, path) for path in SHIPPED_PATHS}
    KIT.verdict("(a) the shipped workflows/extrema.yaml holds both paths", all(present.values()),
                {path: True for path in SHIPPED_PATHS}, present)
    exc = testkit.raised(lambda: validate_config(load_config(str(shipped_file), testkit.DEV_PROFILE)))
    KIT.verdict("(a) validate_config accepts the merged shipped configuration", exc is None, "accepted", repr(exc)[:240])


def forbidden_extras(config: dict) -> None:
    """(b) Every reachable model forbids extra keys and is frozen; equal content hashes equal, a section with list fields
    included; an attribute assignment raises ValidationError; a sentinel in trigger, the replay section and its one
    mapping-valued field raises ValidationError."""
    models = reachable_models(AppConfig)
    lenient = sorted(model.__name__ for model in models if model.model_config.get("extra") != FORBID)
    KIT.verdict("(b) every model reachable from AppConfig forbids extra keys", len(models) > 1 and not lenient,
                {"models": "> 1", "lenient": []}, {"models": len(models), "lenient": lenient})
    mutable = sorted(model.__name__ for model in models if not model.model_config.get("frozen"))
    KIT.verdict("(b) every model reachable from AppConfig is frozen", not mutable, [], mutable)
    first, second = validate_config(copy.deepcopy(config)), validate_config(copy.deepcopy(config))
    hashes = {"config": hash(first) == hash(second), "read (list fields)": hash(first.read) == hash(second.read)}
    KIT.verdict("(b) equal content gives an equal hash, a section with list fields included", all(hashes.values()),
                True, hashes)
    exc = testkit.raised(lambda: setattr(first.read, "raw_data", SENTINEL))
    KIT.verdict("(b) an attribute assignment raises ValidationError", isinstance(exc, ValidationError), "ValidationError",
                repr(exc)[:240])
    section = testkit.replay_key(config)
    mappings = sorted(key for key, value in config["trigger"][section].items() if isinstance(value, dict))
    KIT.verdict("(b) the replay section holds exactly one mapping-valued field", len(mappings) == 1, 1, mappings)
    for name, path in (("trigger", ["trigger"]), ("the replay section", ["trigger", section]),
                       ("its mapping-valued field", ["trigger", section, *mappings[:1]])):
        planted = copy.deepcopy(config)
        node = planted
        for part in path:
            node = node[part]
        node[SENTINEL] = 1
        exc = testkit.raised(lambda: validate_config(planted))
        KIT.verdict(f"(b) a sentinel key in {name} raises ValidationError", isinstance(exc, ValidationError),
                    "ValidationError", repr(exc)[:240])


def series_columns() -> None:
    """(c) The dual --task trigger on series 0: per mode, /series_0's columns are the fifteen names."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(KEEP_SERIES_ZERO))
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["trigger"])
        KIT.verdict("(c) the dual --task trigger returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            columns = (set(testkit.store_frames(testkit.mode_output(config_file, mode, "trigger"))[SERIES_KEY].columns)
                       if child.return_code == 0 else None)
            KIT.verdict(f"(c) {mode}: {SERIES_KEY}'s columns are the fifteen names, {END_COLUMN} among them",
                        columns == SERIES_COLUMNS, sorted(SERIES_COLUMNS), sorted(columns) if columns else columns)


def end_key() -> None:
    """(d) The V fixture under different_end_strategy alone: --task trigger returns 0, then row 61 ends with it."""
    with testkit.scratch(KIT) as folder:
        raw = folder / V_INPUT
        testkit.write_raw_input(raw, {"0": v_frame()})
        relative = Path(os.path.relpath(raw, Path.cwd())).as_posix()
        overrides = {"read.raw_data": f"./{relative}", **V_SETTINGS,
                     **{testkit.replay_path(KIT, leaf): value for leaf, value in V_REPLAY.items()}}
        config_file = testkit.config_copy(KIT, folder, strategy=overrides)
        child = testkit.run_pipeline(KIT, folder, "vfixture", config_file, ["trigger"])
        found = None
        if child.return_code == 0:
            frame = testkit.store_frames(testkit.output_file(config_file, "trigger"))[SERIES_KEY]
            found = frame[END_COLUMN].iat[END_ROW] if END_COLUMN in frame.columns else None
        KIT.verdict(f"(d) --task trigger returns 0, then {SERIES_KEY}'s {END_COLUMN} on row {END_ROW} is {END_REASON}",
                    child.return_code == 0 and found == END_REASON, {"return_code": 0, END_COLUMN: END_REASON},
                    {"return_code": child.return_code, "timed_out": child.timed_out, END_COLUMN: found})


# =============================================================================
# HELPERS
# =============================================================================


def value_present(document: dict, path: str) -> bool:
    """A dotted path names a key at every level of a parsed YAML document."""
    node = document
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


def reachable_models(root: type) -> list[type]:
    """Every pydantic model reachable from root through its fields' annotations, root included, each once."""
    found: list[type] = []
    pending = [root]
    while pending:
        model = pending.pop()
        if model not in found:
            found.append(model)
            pending += [named for field in model.model_fields.values() for named in models_in(field.annotation)]
    return found


def models_in(annotation: object) -> list[type]:
    """The pydantic models an annotation names, through the arguments of Optional, Union, List and Dict."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for argument in typing.get_args(annotation) for model in models_in(argument)]


def v_frame() -> pd.DataFrame:
    """The V fixture: var3 = var4 = var0 through the knots, var1 and var2 1 % below and above, var5 1e6."""
    values = np.interp(np.arange(len(V_DATES)), *zip(*V_KNOTS)).astype("float32")
    return pd.DataFrame({"var0": values, "var1": values * np.float32(0.99), "var2": values * np.float32(1.01),
                         "var3": values, "var4": values, "var5": np.full(len(values), 1.0e6)}, index=V_DATES)


if __name__ == "__main__":
    main()
