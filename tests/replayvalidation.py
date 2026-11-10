"""Replay validation suite: parity, no future rows, rolling vs expanding, boundary (D-1).

Every strategy comes from a scratch copy of the shipped config, so the live key
trigger.trigger_value is read as shipped; each copy sets only the mode, the
lookback (40) and the window type its test names. Every fixture holds the same levels in var0
to var4 and var5 = 1e6, and the copies set no gate item, so no begin is blocked. One verdict per
test:
(1) parity: the pairs of ideal analyze_series on a noisy sine (120 rows) equal an independent
pairing of _generate_trigger_begin's extrema: rising edges per signal type on the turning-point
timeline, each begin with the first later end of its type not taken by an earlier begin, then
first-wins removal of a pair that begins before the last kept pair ends.
(2) no future rows: realistic analyze_series of the same noisy sine cut at every row c >= 39
gives the full run's begins, ends, end_reason and turning_points on rows <= c; non-vacuity:
the full run holds a begin and an end, both counted in the verdict.
(3) rolling vs expanding: a rise from 100 to 200 and back over rows 0-20, then 200 rows of
150 + 2 sin(2 pi t / 10). The expanding window keeps the early range, whose prominence floor
0.05 x 100 = 5 exceeds every later peak's prominence (at most 4); the rolling window drops it
from row 60 on (floor 0.05 x 4 = 0.2), so the rolling run has more begins, both counted in the
verdict. From row 21 + 40 - 1 = 60 on, every rolling _select_window frame equals the frame of a
copy whose first 21 rows are shifted by 1000.
(4) boundary: realistic analyze_series processes a 40-row series, returns None for a 39-row
one and finds begins on a 200-row noisy sine.

Structure:
1. main
2. Tests
3. Fixtures and helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


LOOKBACK, SINE_ROWS, LONG_ROWS, SEED = 40, 120, 200, 42
RISE_ROWS, EXCURSION_ROWS, WAVE_ROWS, SHIFT = 11, 21, 200, 1000.0
COPIES = {
    "ideal": {"trigger.replay.mode": "ideal", "trigger.replay.lookback_window": LOOKBACK},
    "rolling": {"trigger.replay.mode": "realistic", "trigger.replay.lookback_window": LOOKBACK,
                "trigger.replay.window_type": "rolling"},
    "expanding": {"trigger.replay.mode": "realistic", "trigger.replay.lookback_window": LOOKBACK,
                  "trigger.replay.window_type": "expanding"},
}
FLAGS = {"plus": ("trigger_plus_begin", "trigger_plus_end"), "minus": ("trigger_minus_begin", "trigger_minus_end")}
BEGINS = ("trigger_plus_begin", "trigger_minus_begin")
ENDS = ("trigger_plus_end", "trigger_minus_end")
DECISIONS = (*BEGINS, *ENDS, "end_reason", "turning_points")


def main() -> None:
    """The four tests on scratch config copies, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        configs = {name: configured(folder / name, overrides) for name, overrides in COPIES.items()}
        parity(configs["ideal"])
        future_rows(configs["rolling"])
        windows(configs["rolling"], configs["expanding"])
        boundary(configs["rolling"])
    KIT.finish()


# =============================================================================
# TESTS
# =============================================================================


def parity(config: tuple) -> None:
    """Test (1): ideal analyze_series pairs against the independent pairing of _generate_trigger_begin's extrema."""
    frame = noisy_sine(SINE_ROWS)
    flagged = testkit.strategy(*config)._generate_trigger_begin(frame.copy())
    minima, maxima = (flagged[column].to_numpy() > 0 for column in BEGINS)
    expected = paired({"plus": minima, "minus": maxima}, {"plus": maxima, "minus": minima})
    found = output_pairs(analyzed(config, frame))
    KIT.verdict("(1) parity: ideal analyze_series pairs equal the pairing of _generate_trigger_begin's extrema",
                bool(expected) and found == expected, sorted(expected), sorted(found))


def future_rows(config: tuple) -> None:
    """Test (2): every cut at row c >= lookback - 1 gives the full run's decision columns on rows <= c."""
    frame = noisy_sine(SINE_ROWS)
    full = analyzed(config, frame)
    differences = {cut: changed(analyzed(config, frame.iloc[:cut + 1]), full.iloc[:cut + 1])
                   for cut in range(LOOKBACK - 1, SINE_ROWS)}
    differing = {cut: columns for cut, columns in differences.items() if columns}
    begins, ends = (int(full[list(columns)].to_numpy().sum()) for columns in (BEGINS, ENDS))
    KIT.verdict("(2) no future rows: every cut's begins, ends, end_reason and turning_points equal the full run's on "
                "rows up to the cut", begins > 0 and ends > 0 and not differing,
                {"differing cuts": {}, "begins": "> 0", "ends": "> 0"},
                {"differing cuts": dict(list(differing.items())[:6]), "cuts": SINE_ROWS - LOOKBACK + 1,
                 "begins": begins, "ends": ends})


def windows(rolling: tuple, expanding: tuple) -> None:
    """Test (3): the rolling run has more begins than the expanding one, and its windows from row
    EXCURSION_ROWS + LOOKBACK - 1 on do not depend on the first EXCURSION_ROWS rows."""
    levels = excursion_wave()
    frame = frame_of(levels)
    counts = {name: begin_count(analyzed(config, frame)) for name, config in (("rolling", rolling), ("expanding", expanding))}
    strategy = testkit.strategy(*rolling)
    window_settings = strategy._load_realistic_config(rolling[0].trigger)
    shifted = frame_of(np.concatenate([levels[:EXCURSION_ROWS] + SHIFT, levels[EXCURSION_ROWS:]]))
    differing = [i for i in range(EXCURSION_ROWS + LOOKBACK - 1, len(frame))
                 if not strategy._select_window(frame, i, window_settings).equals(
                     strategy._select_window(shifted, i, window_settings))]
    KIT.verdict(f"(3) rolling vs expanding: the rolling run has more begins, and its windows from row "
                f"{EXCURSION_ROWS + LOOKBACK - 1} on ignore the first {EXCURSION_ROWS} rows",
                counts["rolling"] > counts["expanding"] and not differing,
                {"rolling": "> expanding", "differing windows": []}, {**counts, "differing windows": differing[:6]})


def boundary(config: tuple) -> None:
    """Test (4): a lookback-row series is processed, a shorter one returns None, a 200-row noisy sine has begins."""
    found = {"lookback rows processed": analyzed(config, noisy_sine(LOOKBACK)) is not None,
             "lookback - 1 rows give None": analyzed(config, noisy_sine(LOOKBACK - 1)) is None,
             "begins on 200 rows": begin_count(analyzed(config, noisy_sine(LONG_ROWS)))}
    KIT.verdict("(4) boundary: n = lookback is processed, n = lookback - 1 returns None, n = 200 yields begins",
                found["lookback rows processed"] and found["lookback - 1 rows give None"] and found["begins on 200 rows"] > 0,
                {"lookback rows processed": True, "lookback - 1 rows give None": True, "begins on 200 rows": "> 0"}, found)


# =============================================================================
# FIXTURES AND HELPERS
# =============================================================================


def configured(folder: Path, overrides: dict) -> tuple:
    """A scratch copy of the shipped config in its own folder, with the given overrides: its validated sections
    and its trigger output, the two arguments of testkit.strategy."""
    folder.mkdir()
    config_file = testkit.config_copy(KIT, folder, strategy=overrides)
    return testkit.validated_config(config_file), testkit.output_file(config_file, "trigger")


def noisy_sine(rows: int) -> pd.DataFrame:
    """100 + 10 sin(x) over x in [0, 4 pi] plus normal noise of standard deviation 0.5 (seed 42)."""
    rng = np.random.default_rng(SEED)
    return frame_of(100.0 + 10.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, rows)) + rng.normal(0.0, 0.5, rows))


def excursion_wave() -> np.ndarray:
    """A rise from 100 to 200 and back over the first EXCURSION_ROWS rows, then WAVE_ROWS rows of
    150 + 2 sin(2 pi t / 10)."""
    rise = np.linspace(100.0, 200.0, RISE_ROWS)
    return np.concatenate([rise, rise[-2::-1], 150.0 + 2.0 * np.sin(2.0 * np.pi * np.arange(WAVE_ROWS) / 10.0)])


def frame_of(levels: np.ndarray) -> pd.DataFrame:
    """var0 to var4 = levels (float32) and var5 = 1e6 on weekdays from 2020-01-01."""
    frame = pd.DataFrame({column: levels.astype("float32") for column in ("var0", "var1", "var2", "var3", "var4")},
                         index=pd.bdate_range("2020-01-01", periods=len(levels)))
    frame["var5"] = np.full(len(levels), 1.0e6)
    return frame


def analyzed(config: tuple, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    """analyze_series of a fresh strategy on a copy of the frame, as series 0."""
    return testkit.strategy(*config).analyze_series("0", frame.copy())


def paired(begins: dict, ends: dict) -> set:
    """(signal type, begin row, end row): rising edges per type on the turning-point timeline, matched, then
    first-wins removal of a pair that begins before the last kept pair ends."""
    candidates = []
    for direction in FLAGS:
        rows = np.flatnonzero(begins[direction] | ends[direction])
        candidates += [(begin, end, direction) for begin, end in matched(rising(rows, begins[direction]),
                                                                          rising(rows, ends[direction]))]
    kept = []
    for begin, end, direction in sorted(candidates):
        if not kept or begin >= kept[-1][1]:
            kept.append((begin, end, direction))
    return {(direction, begin, end) for begin, end, direction in kept}


def rising(rows: np.ndarray, flags: np.ndarray) -> list:
    """The timeline rows where the flag turns on (on at the first row, or off at the row before)."""
    return [int(row) for k, row in enumerate(rows) if flags[row] and (k == 0 or not flags[rows[k - 1]])]


def output_pairs(frame: pd.DataFrame) -> set:
    """(signal type, begin row, end row) of an analysed frame's flags."""
    return {(direction, begin, end) for direction, (begin_col, end_col) in FLAGS.items()
            for begin, end in matched(np.flatnonzero(frame[begin_col].to_numpy() > 0).tolist(),
                                      np.flatnonzero(frame[end_col].to_numpy() > 0).tolist())}


def matched(begins: list, ends: list) -> list:
    """Each begin row with the first later end row not taken by an earlier begin."""
    pairs, taken = [], set()
    for begin in begins:
        end = next((row for row in ends if row > begin and row not in taken), None)
        if end is not None:
            taken.add(end)
            pairs.append((int(begin), int(end)))
    return pairs


def changed(cut_run: pd.DataFrame, full_rows: pd.DataFrame) -> list:
    """The decision columns whose values differ between a cut's run and the full run's rows up to the cut."""
    return [column for column in DECISIONS if not np.array_equal(cut_run[column].to_numpy(), full_rows[column].to_numpy())]


def begin_count(frame: pd.DataFrame) -> int:
    """Plus and minus begins of an analysed frame."""
    return int(frame[list(BEGINS)].to_numpy().sum())


if __name__ == "__main__":
    main()
