"""Regression for the ideal pairing of begins and ends (Q16).

EXP-01's fixtures through ideal analyze_series on a scratch config copy: 97 samples of
100 + 10 sin(x), x in [0, 3 pi] ([max, min, max] with the maxima on rows 16 and 80 and the
minimum on row 48), and its mirror 100 - 10 sin(x) ([min, max, min]); var4 = var3 and var5 =
1e6, and the copy sets no gate item, so no begin is blocked. Expected for [max, min, max]:
the pairs (minus, 16, 48) and (plus, 48, 80), no active plus flag, the active minus flag only on
row 80 (the final begin with no end after it); the mirror with the directions swapped. Each
fixture holds a leading end (row 16's end of the other direction, with no begin before it) and
a trailing begin (row 80, the active flag). The pairing per signal type matches each begin with
the first end after it and drops an end with no unmatched begin before it and a final begin with
no end after it. The unfixed code first removes the trailing excess by count, which drops the end
of row 80 before the plus begin of row 48 can take it: no plus pair, and an active plus flag on
row 48. Overlapping pairs (D-5): the cleaning, _clean_duplicates, on hand-set flags with a plus
pair (2, 8) and a minus pair (5, 11) that begins inside it, and on the reverse, a minus pair
(1, 6) and a plus pair (3, 9): the first pair wins, the later overlapping one is removed. Ideal
analyze_series cannot produce such an overlap (each direction's ends are the other's begins, so
its pairs chain), hence the direct call. An inverted overlap check keeps both overlapping pairs
and drops the touching pair of the fixtures.

Structure:
1. main
2. Overlapping pairs
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import numpy as np
import pandas as pd


SAMPLES, FIRST, MIDDLE, LAST = 97, 16, 48, 80
OVERLAP_ROWS = 14
# case: (pair flags as (direction, begin row, end row), the pair the cleaning keeps)
OVERLAPS = {"a minus pair beginning inside a plus pair": ([("plus", 2, 8), ("minus", 5, 11)], {("plus", 2, 8)}),
            "a plus pair beginning inside a minus pair": ([("minus", 1, 6), ("plus", 3, 9)], {("minus", 1, 6)})}
FLAGS = {"plus": ("trigger_plus_begin", "trigger_plus_end"), "minus": ("trigger_minus_begin", "trigger_minus_end")}
ACTIVE = {"plus": "active_trigger_plus_begin", "minus": "active_trigger_minus_begin"}
CASES = {
    "[max, min, max]": (1.0, {("minus", FIRST, MIDDLE), ("plus", MIDDLE, LAST)}, {"plus": [], "minus": [LAST]}),
    "[min, max, min]": (-1.0, {("plus", FIRST, MIDDLE), ("minus", MIDDLE, LAST)}, {"plus": [LAST], "minus": []}),
}


def main() -> None:
    """Both fixtures through ideal analyze_series, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "ideal"})
        config, output_file = testkit.validated_config(config_file), testkit.output_file(config_file, "trigger")
        for name, (sign, expected_pairs, expected_active) in CASES.items():
            levels = (100.0 + sign * 10.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, SAMPLES))).astype("float32")
            frame = pd.DataFrame({column: levels for column in ("var0", "var1", "var2", "var3", "var4")},
                                 index=pd.bdate_range("2020-01-01", periods=SAMPLES))
            frame["var5"] = np.full(SAMPLES, 1.0e6)
            output = testkit.strategy(config, output_file).analyze_series("0", frame)
            pairs = pairs_of(output)
            active = {direction: np.flatnonzero(output[column].to_numpy() > 0).tolist() for direction, column in ACTIVE.items()}
            KIT.verdict(f"{name}: pairs (direction, begin row, end row)", pairs == expected_pairs, sorted(expected_pairs),
                        sorted(pairs))
            KIT.verdict(f"{name}: active flags", active == expected_active, expected_active, active)
        overlapping_pairs(testkit.strategy(config, output_file))
    KIT.finish()


# =============================================================================
# OVERLAPPING PAIRS
# =============================================================================


def overlapping_pairs(strategy: object) -> None:
    """The cleaning keeps the first of two overlapping pairs."""
    for name, (flags, expected) in OVERLAPS.items():
        frame = pd.DataFrame(0, index=pd.bdate_range("2020-01-01", periods=OVERLAP_ROWS),
                             columns=[column for columns in FLAGS.values() for column in columns])
        frame["end_reason"] = ""
        for direction, begin, end in flags:
            begin_col, end_col = FLAGS[direction]
            frame.iloc[begin, frame.columns.get_loc(begin_col)] = 1
            frame.iloc[end, frame.columns.get_loc(end_col)] = 1
        pairs = pairs_of(strategy._clean_duplicates(frame))
        KIT.verdict(f"overlap, {name}: the first pair wins", pairs == expected, sorted(expected), sorted(pairs))


# =============================================================================
# HELPERS
# =============================================================================


def pairs_of(output: pd.DataFrame) -> set:
    """(direction, begin row, end row): each direction's begin flags zipped with its end flags."""
    return {(direction, int(begin), int(end)) for direction, (begin_col, end_col) in FLAGS.items()
            for begin, end in zip(np.flatnonzero(output[begin_col].to_numpy() > 0),
                                  np.flatnonzero(output[end_col].to_numpy() > 0))}


if __name__ == "__main__":
    main()
