"""Regression for the same-day sweep of the weight constraint (Q4).

Production sweep_line_filter on the weekdays d0..d11. (1) A signal beginning on the day another
ends: bounds {1: (d0, d4), 2: (d4, d8), 3: (d4, d7)}, weight 0.5, maximum 1.0; end events come
before begin events on the same date, so signal 1 has left when 2 and 3 begin on d4 and nothing is
removed. (2) A zero-length signal: {1: (d2, d2), 2: (d5, d8)}, weight 1.0, maximum 1.0; its end
comes after its day's begin events, so it leaves on d2 and nothing is removed (a plain ends-first
rule would process its end before its begin, keep it active and remove signal 2). (3) The
other two cases keep their result: the newest active signal is removed ({1: (d0, d9),
2: (d1, d9), 3: (d2, d9)}, weight 0.4 -> {3}) and same-day begin events follow their rank
({1: (d0, d5), 2: (d0, d6)}, weight 0.6 -> {2}). The unfixed sweep processes begin events before
end events, so case (1) removes signal 3. The removal of case (3) logs one WARNING record on
src.utils.performance.gradient_.validation of exactly two lines, the violation and the signals still active, the
second without leading spaces (D3-19: the code before C3-15 padded it for a log format the handler never had).

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import logging

import pandas as pd

DAYS = pd.bdate_range("2022-01-03", periods=12)
VALIDATION_LOGGER = "src.utils.performance.gradient_.validation"
CASES = {
    "case (1) a signal begins on the day another ends": (
        {1: (DAYS[0], DAYS[4]), 2: (DAYS[4], DAYS[8]), 3: (DAYS[4], DAYS[7])}, 0.5, set()),
    "case (2) a zero-length signal, then a later one": ({1: (DAYS[2], DAYS[2]), 2: (DAYS[5], DAYS[8])}, 1.0, set()),
    "case (3) the newest active signal is removed": (
        {1: (DAYS[0], DAYS[9]), 2: (DAYS[1], DAYS[9]), 3: (DAYS[2], DAYS[9])}, 0.4, {3}),
    "case (3) same-day begin events follow their rank": ({1: (DAYS[0], DAYS[5]), 2: (DAYS[0], DAYS[6])}, 0.6, {2}),
}
MAXIMUM = 1.0
REMOVAL_CASE = "case (3) the newest active signal is removed"
# the removal record of REMOVAL_CASE, line by line
REMOVAL_LINES = [
    f"Removed signal group_id=3 (begin={DAYS[2].date()}, end={DAYS[9].date()}) - caused violation: "
    "weight=0.4000 - concurrent=3 = 1.2000 > 1.0000",
    f"Existing signals: group_id=1 (begin={DAYS[0].date()}, end={DAYS[9].date()}); "
    f"group_id=2 (begin={DAYS[1].date()}, end={DAYS[9].date()})",
]


def main() -> None:
    """Every case against its expected removals, the removal case's record line by line, then the summary (return
    code 1 on any FAIL)."""
    from src.utils.performance.gradient_.validation import sweep_line_filter

    for name, (bounds, weight, expected) in CASES.items():
        collector = testkit.RecordCollector(logging.WARNING)
        logging.getLogger(VALIDATION_LOGGER).addHandler(collector)
        try:
            removed, peak = sweep_line_filter(bounds, {gid: {} for gid in bounds}, weight, MAXIMUM)
        finally:
            logging.getLogger(VALIDATION_LOGGER).removeHandler(collector)
        KIT.verdict(f"{name}: removed signals", removed == expected, sorted(expected),
                    {"removed": sorted(removed), "max_concurrent": peak})
        if name == REMOVAL_CASE:
            records = [message.splitlines() for message in collector.messages]
            KIT.verdict(f"{name}: one WARNING record of the two lines, the second unpadded",
                        records == [REMOVAL_LINES], [REMOVAL_LINES], records)
    KIT.finish()


if __name__ == "__main__":
    main()
