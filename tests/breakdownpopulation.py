"""Regression for the signal breakdown population (Q21).

--task all on a realistic copy that keeps series 0 and 1, with
performance.gradient.concurrent_weight_limit 0.2 (weight 0.2 as shipped, so one signal at a
time): the weight filter must drop signals (non-vacuity: /gradient_filtered_sequence holds fewer
rows than /signals has is_signal rows), and the Signals count of /signals_breakdown_table in
column plus+minus_true+false must equal the /gradient_filtered_sequence row count: the breakdown
counts the signals that build the curve. The unfixed code counts every is_signal row.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import pandas as pd

KEPT_SERIES = {"trigger.replay.mode": "realistic", "read.skip_series": ["2", "3", "4", "5"],
               "performance.gradient.concurrent_weight_limit": 0.2}


def main() -> None:
    """The run, the non-vacuity check and the count, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(KEPT_SERIES))
        child = testkit.run_pipeline(KIT, folder, "all", config_file, ["all"])
        KIT.verdict("the run returns 0", child.return_code == 0, 0, child.return_code)
        with pd.HDFStore(str(testkit.output_file(config_file, "performance")), mode="r") as store:
            filtered = len(store.get("/gradient_filtered_sequence"))
            selected = int(store.get("/signals")["is_signal"].sum())
            table = store.get("/signals_breakdown_table")
        KIT.verdict("the weight filter drops signals (non-vacuity)", 0 < filtered < selected,
                    "0 < filtered rows < is_signal rows", {"filtered": filtered, "is_signal": selected})
        counted = table.loc[(table["row_key"] == "Signals") & (table["column_header"] == "plus+minus_true+false"),
                            "value"].tolist()
        KIT.verdict("the breakdown counts the signals that build the curve", counted == [float(filtered)],
                    [float(filtered)], {"counted": counted, "is_signal": selected})
    KIT.finish()


if __name__ == "__main__":
    main()
