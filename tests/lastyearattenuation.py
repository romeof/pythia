"""Regression for the last-year attenuation applied to the curve (Q17).

(1) build_actual_variation_pair on two synthetic plus signals over the observed dates of 2020
and 2021 (multiplicative_fixed, weight 0.5, yearly_attenuation_fraction 0.25, attenuation_month 1): one increases
20% in 2020, the other 40% in 2021, so the curve's last-year increase is G = 0.5 * 0.40 = 0.20.
Expected: the attenuated curve's item before the last holds the value without the last-year
attenuation, 0.5 * 0.20 * (1 - 0.25) + G = 0.275; its last item holds 0.275 - 0.25 * G = 0.225; the
returned last-year amount is 0.25 * G = 0.05. (2) --task all on an ideal copy of the reference
input: the strategy's acr_after_attenuation in /gradient_metrics_table equals the acr of the persisted
/total_variation_attenuated (calculate_all_gradient_metrics with the persisted mpv); non-vacuity:
the strategy row of /gradient_terminal_attenuations is positive. The unfixed code leaves the last
year out of the attenuated curve: its last item stays 0.275 in (1), and in (2) acr_after_attenuation
differs from the acr of the persisted curve by the last-year attenuation.

Structure:
1. main
2. Cases
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.performance.gradient import build_actual_variation_pair
from src.utils.performance.gradient_.metrics import calculate_all_gradient_metrics

WEIGHT, RATE, FIRST_INCREASE, LAST_INCREASE = 0.5, 0.25, 0.20, 0.40
SPANS = (("2020-03-02", "2020-06-30", FIRST_INCREASE), ("2021-03-01", "2021-06-30", LAST_INCREASE))
STRATEGY_SENTINEL = "__strategy__"
RTOL = 1e-5


def main() -> None:
    """The two cases, then the summary (return code 1 on any FAIL)."""
    synthetic_curve()
    with testkit.scratch(KIT) as folder:
        persisted_metrics(folder)
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def synthetic_curve() -> None:
    """Case (1): only the attenuated curve's last item carries the last-year attenuation, whose amount is returned."""
    signals = {}
    for group_id, (*bounds, increase) in enumerate(SPANS):
        days = pd.bdate_range(*bounds)
        signals[group_id] = {"pct": pd.Series(np.linspace(0.0, 100.0 * increase, len(days)), index=days), "sign": 1,
                             "group_id": group_id}
    _, attenuated, terminal = build_actual_variation_pair(signals, WEIGHT, pd.bdate_range("2020-01-01", "2021-12-31"),
                                                          "multiplicative_fixed", yearly_attenuation_fraction=RATE, attenuation_month=1)
    year_increase = WEIGHT * LAST_INCREASE
    before = WEIGHT * FIRST_INCREASE * (1.0 - RATE) + year_increase
    expected = {"item before the last": before, "last item": before - RATE * year_increase}
    found = {"item before the last": float(attenuated.iloc[-2]), "last item": float(attenuated.iloc[-1])}
    KIT.verdict("case (1): the attenuated curve's last item is lowered by 0.25 * G, the item before it is not",
                all(np.isclose(found[key], wanted, rtol=RTOL) for key, wanted in expected.items()), expected, found)
    KIT.verdict("case (1): the returned last-year amount is 0.25 * G", bool(np.isclose(terminal, RATE * year_increase, rtol=RTOL)),
                RATE * year_increase, terminal)


def persisted_metrics(folder: Path) -> None:
    """Case (2): acr_after_attenuation is the acr of the persisted attenuated curve, whose last-year attenuation is positive."""
    config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "ideal"})
    child = testkit.run_pipeline(KIT, folder, "all", config_file, ["all"])
    KIT.verdict("case (2): the run returns 0", child.return_code == 0, 0, child.return_code)
    with pd.HDFStore(str(testkit.output_file(config_file, "performance")), mode="r") as store:
        table = store.get("/gradient_metrics_table")
        attenuated = store.get("/total_variation_attenuated")
        attenuations = store.get("/gradient_terminal_attenuations")
        mpv = float(store.get("/gradient_params")["mpv"].iloc[0])
    amount = float(attenuations.loc[attenuations["series_id"] == STRATEGY_SENTINEL, "amount"].iloc[0])
    KIT.verdict("case (2): the strategy row of /gradient_terminal_attenuations is positive (non-vacuity)", amount > 0.0,
                "> 0", amount)
    curve = pd.Series(attenuated["total_variation_attenuated"].to_numpy(dtype=float) * 100.0,
                      index=pd.to_datetime(attenuated["date"]))
    recomputed = calculate_all_gradient_metrics(curve, None, selection=["acr"], mpv=mpv)[0].values["acr"]
    persisted = float(table.loc[(table["metric_key"] == "acr_after_attenuation") & (table["column_header"] == "strategy"),
                                "value"].iloc[0])
    KIT.verdict("case (2): the strategy's acr_after_attenuation equals the acr of /total_variation_attenuated",
                bool(np.isclose(persisted, recomputed, rtol=RTOL)), recomputed, persisted)


if __name__ == "__main__":
    main()
