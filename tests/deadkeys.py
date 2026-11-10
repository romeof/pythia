"""Regression for the nine dev keys no code read (F-23).

validate_config on scratch copies: the shipped dev file validates (both trees, a positive
control); a dev file that sets trigger.read.printout_debug, one of the nine keys, raises
ValidationError, observed through testkit.raised, because the schema no longer declares it
(before F-23: accepted). The unreachable strategy branches and their typed reads are
measured by the config-contract sweep.

Structure:
1. main
2. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from typing import Optional

from pydantic import ValidationError

from src.utils.common.config import validate_config

DEAD_KEY = "trigger.read.printout_debug"


def main() -> None:
    """The shipped copy and the copy with the dead key, then the summary (return code 1 on any FAIL)."""
    exc = validation_failure(None)
    KIT.verdict("the shipped dev file validates", exc is None, None, repr(exc))
    exc = validation_failure({DEAD_KEY: False})
    KIT.verdict(f"a dev file with {DEAD_KEY} raises ValidationError", isinstance(exc, ValidationError), "ValidationError",
                repr(exc)[:300])
    KIT.finish()


# =============================================================================
# HELPERS
# =============================================================================


def validation_failure(dev: Optional[dict]) -> Optional[BaseException]:
    """What validate_config raises on a scratch copy with the dev overrides, None when it validates."""
    with testkit.scratch(KIT) as folder:
        config = testkit.loaded_config(testkit.config_copy(KIT, folder, dev=dev))
        return testkit.raised(lambda: validate_config(config))


if __name__ == "__main__":
    main()
