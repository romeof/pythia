"""Regression for the gradient metric keys (W3, W8).

(a) src/utils/performance/gradient_/metrics.json holds exactly the sixteen keys below, in sequence, with
the labels below. (b) validate_config on the merged configuration of a copy of the shipped YAML, which
omits performance.metrics.display_order_gradient, fills that list with the sixteen keys in sequence.
(c) --task all on a copy keeping series 0, the shipped dual mode: per mode, /gradient_param_sequences'
display_gradient names are the sixteen keys in sequence, and /gradient_metrics_table's metric_key set is
them plus the two information rows _n_intervals and _n_years. A tree that still writes a retired key fails
all three.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import json
from pathlib import Path

from src.utils.common.config import validate_config

KEYS = ["acr", "acr_after_attenuation", "std", "std_dw", "snr", "snr_dw", "max_fall", "fall_dur", "below_max_dur", "rho", "sens", "sens_dw",
        "exv", "exv_dw", "ref_snr", "ref_snr_dw"]
LABELS = ["acr (%)", "acr (after attenuation) (%)", "std (%)", "std_dw (%)", "snr", "snr_dw", "max_fall (%)", "fall_dur", "below_max_dur",
          "rho", "sens", "sens_dw", "exv (%)", "exv_dw (%)", "ref_snr", "ref_snr_dw"]
INFO_KEYS = ["_n_intervals", "_n_years"]
METRICS_FILE = Path("src") / "utils" / "performance" / "gradient_" / "metrics.json"
DISPLAY_KEY = "display_order_gradient"
KEEP_SERIES_ZERO = {"read.skip_series": ["1", "2", "3", "4", "5"]}
SEQUENCES_KEY, METRICS_KEY = "/gradient_param_sequences", "/gradient_metrics_table"
DISPLAY_KIND = "display_gradient"


def main() -> None:
    """The metrics file, the schema default and both modes' persisted names, then the summary (return code 1 on any
    FAIL)."""
    metrics_file()
    with testkit.scratch(KIT) as folder:
        schema_default(testkit.loaded_config(testkit.config_copy(KIT, folder)))
    persisted_names()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def metrics_file() -> None:
    """(a) The keys of metrics.json in sequence, and their labels."""
    spec = json.loads((KIT.root / METRICS_FILE).read_text(encoding="utf-8"))
    found = {"keys": list(spec), "labels": [item.get("label") for item in spec.values()]}
    KIT.verdict(f"(a) {METRICS_FILE.as_posix()}: the sixteen keys in sequence and their labels",
                found == {"keys": KEYS, "labels": LABELS}, {"keys": KEYS, "labels": LABELS}, found)


def schema_default(config: dict) -> None:
    """(b) The shipped YAML omits the display list; validate_config fills the sixteen keys in sequence."""
    omitted = DISPLAY_KEY not in config["performance"]["metrics"]
    filled = getattr(validate_config(config).performance.metrics, DISPLAY_KEY)
    KIT.verdict(f"(b) the shipped copy omits performance.metrics.{DISPLAY_KEY}; validate_config fills the sixteen keys",
                omitted and filled == KEYS, {"omitted": True, DISPLAY_KEY: KEYS}, {"omitted": omitted, DISPLAY_KEY: filled})


def persisted_names() -> None:
    """(c) The dual --task all on series 0: per mode, the display sequence and the metric table's keys."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(KEEP_SERIES_ZERO))
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("(c) the dual --task all returns 0", child.return_code == 0, 0,
                    {"return_code": child.return_code, "timed_out": child.timed_out})
        for mode in testkit.MODES:
            found = {"display_gradient": None, "metric_keys": None}
            if child.return_code == 0:
                frames = testkit.store_frames(testkit.mode_output(config_file, mode, "performance"))
                sequences = frames[SEQUENCES_KEY]
                shown = sequences[sequences["kind"] == DISPLAY_KIND].sort_values("sequence_index")
                found = {"display_gradient": shown["name"].astype(str).tolist(),
                         "metric_keys": sorted(set(frames[METRICS_KEY]["metric_key"].astype(str)))}
            expected = {"display_gradient": KEYS, "metric_keys": sorted({*KEYS, *INFO_KEYS})}
            KIT.verdict(f"(c) {mode}: {SEQUENCES_KEY}'s {DISPLAY_KIND} names and {METRICS_KEY}'s metric keys",
                        found == expected, expected, found)


if __name__ == "__main__":
    main()
