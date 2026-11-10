"""Regression for the declared schema defaults (Q11).

(1) --task trigger on a one-series realistic copy without trigger.replay.window_type, whose
schema default is "expanding": the run returns 0 and writes /series_0. (2) In-process,
validate_config on the loaded copy returns the validated configuration, whose
trigger.replay.window_type is "expanding". The unfixed pipeline validates the configuration
but discards the model, so the read of the omitted key raises at use and the run returns
non-zero; its validate_config returns None.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from src.utils.common.config import validate_config
from src.utils.common.schema import AppConfig

OMITTED_KEY = "trigger.replay.window_type"
DEFAULT = "expanding"
SERIES_KEY = "/series_0"


def main() -> None:
    """Both cases on one copy, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES), drop_strategy=(OMITTED_KEY,))
        child = testkit.run_pipeline(KIT, folder, "default", config_file, ["trigger"])
        keys = testkit.store_keys(testkit.output_file(config_file, "trigger"))
        KIT.verdict(f"case (1): --task trigger without {OMITTED_KEY} returns 0 and writes {SERIES_KEY}",
                    child.return_code == 0 and keys is not None and SERIES_KEY in keys, {"return_code": 0, "keys": SERIES_KEY},
                    {"return_code": child.return_code, "keys": keys,
                     "missing_key_lines": sum(OMITTED_KEY in line for line in testkit.plain_lines(child.text))})
        validated = validate_config(testkit.loaded_config(config_file))
        window = validated.trigger.replay.window_type if isinstance(validated, AppConfig) else None
        KIT.verdict(f"case (2): validate_config returns the configuration with {OMITTED_KEY} = {DEFAULT!r}",
                    window == DEFAULT, DEFAULT, {"returned": type(validated).__name__, "window_type": window})
    KIT.finish()


if __name__ == "__main__":
    main()
