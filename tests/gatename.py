"""Regression for the output name of the gate list and the shipped default list (C7, R7).

The shipped workflows/extrema.yaml, loaded and validated directly (load_config and validate_config, since a config
copy empties the gate list): its items are var12 >= 0.83 exempting series "0" and var15 >= 0.73, its pre-filter is
on, and its parameters folder is var12ge0_83x0-var15ge0_73__w7md5p0_05. Validated config copies, each in its own
scratch folder: the parameters folder of every task output equals get_strategy_parameters; the shipped list set on
a copy names the shipped folder, and a second copy of it the same folder; a one-item list and a list differing from
the shipped one in exempt_series alone each name a folder of their own; an empty list names nogate__w7md5p0_05;
get_strategy_name prefixes extrema_; no name holds a character a Windows folder name forbids. A name that ignored
the items or their exempt_series, or wrote > or <, fails a verdict.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
from pathlib import Path

from src.utils.common.config import load_config, validate_config
from src.utils.common.strategyname import get_strategy_name, get_strategy_parameters

SHIPPED_ITEMS = [("var12", ">=", 0.83, ["0"]), ("var15", ">=", 0.73, [])]
SHIPPED_PARAMETERS = "var12ge0_83x0-var15ge0_73__w7md5p0_05"
EMPTY_PARAMETERS = "nogate__w7md5p0_05"
NAME_PREFIX = "extrema_"
# the characters a Windows folder name forbids
WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# case: the gate list a copy sets
LISTS = {
    "shipped": [testkit.gate_item("var12", ">=", 0.83, ["0"]), testkit.gate_item("var15", ">=", 0.73)],
    "shipped again": [testkit.gate_item("var12", ">=", 0.83, ["0"]), testkit.gate_item("var15", ">=", 0.73)],
    "one item": [testkit.gate_item("var15", ">=", 0.73)],
    "another exempt_series": [testkit.gate_item("var12", ">=", 0.83, ["0", "1"]), testkit.gate_item("var15", ">=", 0.73)],
    "empty": [],
}


def main() -> None:
    """The shipped default, then the names of the copies, then the summary (return code 1 on any FAIL)."""
    shipped_default()
    names = copy_names()
    KIT.verdict("the shipped list on a copy names the shipped folder, and a second copy of it the same folder",
                names["shipped"] == names["shipped again"] == SHIPPED_PARAMETERS, SHIPPED_PARAMETERS,
                {case: names[case] for case in ("shipped", "shipped again")})
    for case in ("one item", "another exempt_series"):
        KIT.verdict(f"a list with {case} names a folder of its own", names[case] not in (SHIPPED_PARAMETERS, EMPTY_PARAMETERS),
                    f"neither {SHIPPED_PARAMETERS} nor {EMPTY_PARAMETERS}", names[case])
    KIT.verdict("the one-item list and the list differing in exempt_series name different folders",
                names["one item"] != names["another exempt_series"], "different", names)
    KIT.verdict(f"an empty list names {EMPTY_PARAMETERS}", names["empty"] == EMPTY_PARAMETERS, EMPTY_PARAMETERS,
                names["empty"])
    forbidden = {case: WINDOWS_FORBIDDEN.findall(name) for case, name in names.items() if WINDOWS_FORBIDDEN.search(name)}
    KIT.verdict("no name holds a character a Windows folder name forbids", not forbidden, {}, forbidden)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def shipped_default() -> None:
    """The shipped YAML's gate list, pre-filter switch and parameters folder."""
    config = validate_config(load_config(str(KIT.root / "workflows" / testkit.STRATEGY_FILE), testkit.DEV_PROFILE))
    items = [(item.column, item.op, item.threshold, list(item.exempt_series)) for item in config.trigger.gate.items]
    found = {"items": items, "prefilter": config.trigger.gate.prefilter,
             "parameters": get_strategy_parameters(config.trigger)}
    expected = {"items": SHIPPED_ITEMS, "prefilter": True, "parameters": SHIPPED_PARAMETERS}
    KIT.verdict("the shipped workflows/extrema.yaml: the default items, the pre-filter on and the default name",
                found == expected, expected, found)


def copy_names() -> dict[str, str]:
    """Each case's parameters folder; one verdict per copy: every task output's parameters folder equals
    get_strategy_parameters, and get_strategy_name is that with the extrema_ prefix."""
    names = {}
    with testkit.scratch(KIT) as folder:
        for number, (case, items) in enumerate(LISTS.items()):
            (folder / str(number)).mkdir()
            config_file = testkit.config_copy(KIT, folder / str(number), strategy={"trigger.gate.items": items})
            trigger = testkit.validated_config(config_file).trigger
            names[case] = get_strategy_parameters(trigger)
            found = {"folders": sorted({parameters_folder(config_file, task) for task in testkit.TASKS}),
                     "strategy name": get_strategy_name(trigger)}
            expected = {"folders": [names[case]], "strategy name": f"{NAME_PREFIX}{names[case]}"}
            KIT.verdict(f"the {case} list: every task output lies in its parameters folder, the strategy name prefixes "
                        f"{NAME_PREFIX}", found == expected, expected, found)
    return names


# =============================================================================
# HELPERS
# =============================================================================


def parameters_folder(config_file: Path, task: str) -> str:
    """The parameters folder of a task output: above its mode and task folders."""
    return testkit.output_file(config_file, task).parents[2].name


if __name__ == "__main__":
    main()
