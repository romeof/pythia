"""Regression for the configuration contract (D-10, with schemadefaults.py; F-06, Q11, E-7).

In-process on a scratch copy: every defaulted schema leaf is omitted in turn from the loaded
configuration and validate_config fills the declared default, listed below as the test's own
literals; a completeness verdict requires that list to name exactly the schema's defaulted
leaves. An unknown key under read raises ValidationError (testkit.raised), and so does
trigger.replay.lookback_window below trigger.min_length, naming both values (E-7: before,
validation passed and every series was skipped with a WARNING, the run returning 0). Pipeline
child: performance.gradient.max_references 0 with a reference fails the Performance task with
its ValueError, return != 0. A schema that ignored unknown keys would accept the planted key.

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
from typing import Optional

from pydantic import BaseModel, ValidationError

from src.utils.common.config import validate_config
from src.utils.common.schema import AppConfig

DEFAULTS = {
    "read.reference_series": [],
    "trigger.gate.items": [],
    "trigger.gate.prefilter": False,
    "trigger.validity.nonnegative": [],
    "trigger.validity.nonzero": [],
    "trigger.validity.not_all_equal": [],
    "trigger.strategy.printout_io": False,
    "trigger.strategy.printout_debug": False,
    "trigger.strategy.full_output": False,
    "trigger.strategy.plot_debug": False,
    "trigger.replay.window_type": "expanding",
    "trigger.replay.min_duration_periods": 1,
    "trigger.replay.end.max_duration_periods": None,
    "trigger.replay.end.against_threshold_pct": None,
    "trigger.replay.end.along_threshold_pct": None,
    "trigger.replay.end.opposite_trigger_as_begin_strategy": False,
    "trigger.replay.end.different_end_strategy": False,
    "trigger.read.printout_io": False,
    "trigger.preprocess.printout_io": False,
    "signal.input.printout_io": False,
    "signal.input.full_output": False,
    "signal.selection.printout_io": False,
    "signal.selection.full_output": False,
    "performance.perseries_trigger.printout_io": False,
    "performance.perseries_trigger.full_output": False,
    "performance.perseries_trigger.plot_debug": False,
    "performance.perseries_signal.printout_io": False,
    "performance.perseries_signal.full_output": False,
    "performance.perseries_signal.plot_debug": False,
    "performance.gradient.yearly_attenuation_fraction": 0.0,
    "performance.gradient.attenuation_month": 1,
    "performance.gradient.boundary_deduction_pct": 0.0,
    "performance.gradient.mpv": 0.0,
    "performance.gradient.printout_io": False,
    "performance.gradient.printout_debug": False,
    "performance.gradient.full_output": False,
    "performance.gradient.plot_debug": False,
    "performance.gradient.max_plots": 100,
    "performance.gradient.plots_whitelist": [],
    "performance.gradient.max_references": 3,
    "performance.metrics.display_order_gradient": ["acr", "acr_after_attenuation", "std", "std_dw", "snr", "snr_dw", "max_fall", "fall_dur",
                                                   "below_max_dur", "rho", "sens", "sens_dw", "exv", "exv_dw", "ref_snr", "ref_snr_dw"],
}
UNKNOWN_KEY = "read.planted_unknown_key"
SHORT_LOOKBACK = 3
LOOKBACK_ERROR = "trigger.replay.lookback_window (3) < trigger.min_length (5)"
NO_REFERENCES = {"performance.gradient.max_references": 0}
REFERENCES_ERROR = "max_references is 0"


def main() -> None:
    """The defaults, the unknown key and the two pipeline cases, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config = testkit.loaded_config(testkit.config_copy(KIT, folder))
        declared_defaults(config)
        unknown_key(config)
        short_lookback(config)
    no_references()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def declared_defaults(config: dict) -> None:
    """The list names every defaulted leaf; each omitted leaf validates to its declared default."""
    leaves = sorted(defaulted_leaves(AppConfig, ""))
    KIT.verdict("the test's defaults name exactly the schema's defaulted leaves", leaves == sorted(DEFAULTS), sorted(DEFAULTS),
                leaves)
    for path, expected in DEFAULTS.items():
        omitted = copy.deepcopy(config)
        *parents, leaf = path.split(".")
        node = omitted
        for part in parents:
            node = node[part]
        node.pop(leaf, None)
        found = value_at(validate_config(omitted), path)
        KIT.verdict(f"{path} omitted: validate_config fills {expected!r}", found == expected, expected, found)


def unknown_key(config: dict) -> None:
    """An unknown key under read raises ValidationError."""
    planted = copy.deepcopy(config)
    planted["read"][UNKNOWN_KEY.split(".")[-1]] = 1
    exc = testkit.raised(lambda: validate_config(planted))
    KIT.verdict(f"{UNKNOWN_KEY} raises ValidationError", isinstance(exc, ValidationError), "ValidationError", repr(exc)[:200])


def short_lookback(config: dict) -> None:
    """lookback_window below min_length raises ValidationError naming both values (E-7)."""
    planted = copy.deepcopy(config)
    planted["trigger"]["replay"]["lookback_window"] = SHORT_LOOKBACK
    exc = testkit.raised(lambda: validate_config(planted))
    KIT.verdict("lookback_window < min_length raises ValidationError naming both values",
                isinstance(exc, ValidationError) and LOOKBACK_ERROR in str(exc), {"raises": "ValidationError", "names": LOOKBACK_ERROR},
                repr(exc)[:240])


def no_references() -> None:
    """max_references 0 with a reference: the Performance task fails with its ValueError, return != 0."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES), dev=dict(NO_REFERENCES))
        child = testkit.run_pipeline(KIT, folder, "references", config_file, ["all"])
        records = [line for line in testkit.plain_lines(child.text)
                   if testkit.HANDLER_PREFIX in line and "ValueError" in line and REFERENCES_ERROR in line]
        KIT.verdict("max_references 0 with a reference: a Task failed record names the ValueError, return != 0",
                    len(records) == 1 and child.return_code not in (0, None), {"records": 1, "return_code": "!= 0"},
                    {"records": records, "return_code": child.return_code, "timed_out": child.timed_out})


# =============================================================================
# HELPERS
# =============================================================================


def defaulted_leaves(model: type, prefix: str) -> list[str]:
    """Dotted paths of the model's leaves that declare a default, at any depth."""
    found = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            found += defaulted_leaves(annotation, f"{prefix}{name}.")
        elif not field.is_required():
            found.append(f"{prefix}{name}")
    return found


def value_at(config: Optional[AppConfig], path: str) -> object:
    """The attribute at a dotted path of the validated model, None for a missing configuration."""
    node = config
    for part in path.split("."):
        node = getattr(node, part, None) if isinstance(node, BaseModel) else None
    return node


if __name__ == "__main__":
    main()
