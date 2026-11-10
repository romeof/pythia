"""EXP-17: the replot against the persisted tables (R-P7, R-P10, plots, HYP-31).

Per mode, replot_gradient_from_hdf5 on data/run1's performance file with close_figure=False;
the figure stays registered (replot.py:185,205; plots/gradient.py:305-306) and is taken as
the one figure the call added. Its line arrays (strategy curve, after attenuation, references,
signal segments clipped at +-50) and the signal block of its summary text are compared with
the persisted tables; tall_to_wide round trips by row key and records the pivot's loss of the
row sequence (common.py:609-613); a scratch copy without /signals_breakdown_table takes the summary
table's fallback (Q21), whose rendered counts are compared with the persisted breakdown
(HYP-31). Non-vacuity: lines and summary text present in the added figure; planted control: a
copy with one perturbed /total_variation point must be flagged.

Structure:
1. main
2. Replot and figure readout
3. Comparisons
4. Scratch copies
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.common.common import tall_to_wide, wide_to_tall
from src.utils.common.constants import (
    DIRECTION_SUBDIVISION_COLUMNS, HDF5_GRADIENT_FILTERED_SEQUENCE, HDF5_GRADIENT_METRICS_TABLE,
    HDF5_GRADIENT_REFERENCE_VARIATION, HDF5_GRADIENT_SIGNAL_SEGMENTS, HDF5_SIGNALS_BREAKDOWN_TABLE, SIGNALS_COUNT_LABEL,
)
from src.utils.performance.replot import replot_gradient_from_hdf5

import matplotlib.pyplot as plt

EXPERIMENT = "17"
MODES = ("ideal", "realistic")
PERFORMANCE_FILE = Path("performance") / "0.h5"
FIGURE_FILE = "gradient_hd5.png"
CLIP = 50.0
AFTER_ATTENUATION_LABEL = "after attenuation"
REFERENCE_LABEL = "reference {}"
TABLES = {HDF5_GRADIENT_METRICS_TABLE: ("metric_key", "column_header", "value"),
          HDF5_SIGNALS_BREAKDOWN_TABLE: ("row_key", "column_header", "value")}
PLANTED_ROW, PLANTED_SHIFT = 100, 1.0


def main() -> None:
    """Both modes: figure vs tables, round trips, the fallback counts; then the planted control."""
    args = _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    found = {}
    for mode in MODES:
        source = args.run_copy / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / mode / PERFORMANCE_FILE
        persisted = read_tables(source)
        figure = replot(mode, source, folder / mode / FIGURE_FILE)
        found[mode] = {"lines": compare_lines(mode, figure, persisted), "summary": compare_summary(mode, figure, persisted),
                       "round_trips": round_trips(mode, persisted),
                       "fallback": fallback_counts(mode, source, folder / mode, persisted)}
    found["planted"] = planted_control(args.run_copy, folder)
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--run-copy and --source-root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-copy", type=Path, default=harness.REAL_WORKTREE / "data" / "run1" / harness.DATA_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# REPLOT AND FIGURE READOUT
# =============================================================================


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Every key of a performance file, read in mode r."""
    with pd.HDFStore(str(path), mode="r") as store:
        return {key: store.get(key) for key in sorted(store.keys())}


def replot(name: str, source: Path, save_path: Path) -> dict:
    """The figure the replot call adds: top and bottom line data, summary text; then the figure is released with plt.close()."""
    CTX.guard(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    before = set(plt.get_fignums())
    replot_gradient_from_hdf5(source, save_path=save_path, close_figure=False)
    added = sorted(set(plt.get_fignums()) - before)
    CTX.verdict(f"{name}: the replot call registers exactly one figure", len(added) == 1, 1, added)
    figure = plt.figure(added[0])
    top, bottom = figure.axes[0], figure.axes[1]
    readout = {"top": {line.get_label(): (pd.DatetimeIndex(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float))
                       for line in top.get_lines() if line.get_rasterized() and not line.get_label().startswith("_")},
               "bottom": [(pd.DatetimeIndex(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float))
                          for line in bottom.get_lines() if line.get_rasterized()],
               "texts": [text.get_text() for text in figure.texts]}
    plt.close(figure)
    return readout


# =============================================================================
# COMPARISONS
# =============================================================================


def compare_lines(mode: str, figure: dict, persisted: dict) -> dict:
    """Strategy, after-attenuation and reference curves, and the signal segments in the filtered sequence."""
    strategy = str(persisted["/strategy_name"]["strategy_name"].iloc[0])
    expected_top = {strategy: curve(persisted["/total_variation"], "total_variation"),
                    AFTER_ATTENUATION_LABEL: curve(persisted["/total_variation_attenuated"], "total_variation_attenuated")}
    for ref_id, group in persisted[HDF5_GRADIENT_REFERENCE_VARIATION].groupby("reference_id"):
        expected_top[REFERENCE_LABEL.format(ref_id)] = curve(group.sort_values("date"), "variation")
    expected_bottom = segments(persisted)
    CTX.verdict(f"{mode}: non-vacuity: plotted lines in both panels", bool(figure["top"]) and bool(figure["bottom"]),
                "> 0 each", {"top": sorted(figure["top"]), "bottom": len(figure["bottom"])})
    top_diff = sorted(label for label in set(expected_top) | set(figure["top"])
                      if label not in figure["top"] or label not in expected_top
                      or not same_curve(figure["top"][label], expected_top[label]))
    CTX.verdict(f"{mode}: top-panel curves equal the persisted tables", not top_diff, sorted(expected_top), {"differing": top_diff})
    bottom_diff = [index for index, (drawn, expected) in enumerate(zip(figure["bottom"], expected_bottom))
                   if not same_curve(drawn, expected)]
    CTX.verdict(f"{mode}: signal segments equal /gradient_signal_segments (clipped at +-{CLIP:g}) in the filtered sequence",
                len(figure["bottom"]) == len(expected_bottom) and not bottom_diff,
                f"{len(expected_bottom)} equal segments", {"drawn": len(figure["bottom"]), "differing": bottom_diff[:10]})
    return {"top": sorted(figure["top"]), "segments": len(figure["bottom"]), "top_differing": top_diff,
            "segments_differing": bottom_diff}


def curve(table: pd.DataFrame, column: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """A persisted curve as the replot plots it: dates and float values."""
    return pd.DatetimeIndex(pd.to_datetime(table["date"].values)), table[column].to_numpy().astype(float)


def segments(persisted: dict) -> list[tuple[pd.DatetimeIndex, np.ndarray]]:
    """The persisted segments of the sequence's groups, in the filtered sequence, clipped as plotted."""
    sequence = persisted[HDF5_GRADIENT_FILTERED_SEQUENCE].sort_values("sequence_index")["group_id"].tolist()
    grouped = dict(list(persisted[HDF5_GRADIENT_SIGNAL_SEGMENTS].groupby("group_id")))
    return [(pd.DatetimeIndex(pd.to_datetime(grouped[gid]["date"].values)),
             np.clip(grouped[gid]["pct"].to_numpy().astype(float), -CLIP, CLIP)) for gid in sequence if gid in grouped]


def same_curve(drawn: tuple, expected: tuple) -> bool:
    """Equal dates and equal values (NaN-aware), exact: both sides are the same float32 values."""
    return drawn[0].equals(expected[0]) and np.array_equal(drawn[1], expected[1], equal_nan=True)


def compare_summary(mode: str, figure: dict, persisted: dict) -> dict:
    """The rendered signal block (count row, then metric rows) against the persisted breakdown."""
    rendered = signal_rows("\n".join(figure["texts"]))
    expected = breakdown_rows(persisted[HDF5_SIGNALS_BREAKDOWN_TABLE])
    CTX.verdict(f"{mode}: non-vacuity: summary text with a signal block", bool(rendered), "> 0 rows", len(rendered))
    CTX.verdict(f"{mode}: rendered signal block equals /signals_breakdown_table", rendered == expected, expected[:1], rendered[:1])
    return {"rows": len(rendered), "counts": rendered[0] if rendered else None}


def signal_rows(text: str) -> list[list[str]]:
    """Rows of the summary text that end in nine numbers (the signal block), as their nine tokens."""
    rows = []
    for line in text.splitlines():
        tokens = line.split()[-len(DIRECTION_SUBDIVISION_COLUMNS):]
        if len(tokens) == len(DIRECTION_SUBDIVISION_COLUMNS) and all(is_number(token) for token in tokens):
            rows.append(tokens)
    return rows


def is_number(token: str) -> bool:
    """A token float() accepts without raising (nan included)."""
    return pd.notna(pd.to_numeric(token, errors="coerce")) or token == "nan"


def breakdown_rows(tall: pd.DataFrame) -> list[list[str]]:
    """The persisted breakdown as the summary table renders it: counts as integers, metrics with two decimals."""
    rows = []
    for row_key in dict.fromkeys(tall["row_key"]):
        cells = tall[tall["row_key"] == row_key].set_index("column_header")["value"]
        rows.append([f"{int(float(cells[column]))}" if row_key == SIGNALS_COUNT_LABEL else f"{float(cells[column]):.2f}"
                     for column in DIRECTION_SUBDIVISION_COLUMNS])
    return rows


def round_trips(mode: str, persisted: dict) -> dict:
    """tall_to_wide then wide_to_tall: the same (row, column, value) set; the row sequence as observed."""
    found = {}
    for key, (row_col, col_col, value_col) in TABLES.items():
        tall = persisted[key]
        wide = tall_to_wide(tall, row_col, col_col, value_col)
        back = wide_to_tall(wide, row_col, col_col, value_col)
        same = cells(tall, row_col, col_col, value_col) == cells(back, row_col, col_col, value_col)
        first_seen = list(dict.fromkeys(tall[row_col]))
        found[key] = {"cells_equal": same, "row_sequence_kept": list(wide.index) == first_seen,
                      "wide_rows": list(wide.index), "tall_rows": first_seen}
        CTX.verdict(f"{mode}: {key} round trip keeps every (row, column, value)", same, True, len(tall))
        CTX.log.info("OBSERVATION %s %s: row sequence kept by the pivot: %s (wide %s, tall %s)", mode, key,
                     found[key]["row_sequence_kept"], found[key]["wide_rows"], first_seen)
    return found


def cells(table: pd.DataFrame, row_col: str, col_col: str, value_col: str) -> dict:
    """{(row, column): value as a string}, so NaN compares equal to NaN."""
    return {(str(row), str(column)): repr(float(value)) for row, column, value in
            zip(table[row_col], table[col_col], table[value_col])}


def fallback_counts(mode: str, source: Path, folder: Path, persisted: dict) -> dict:
    """A copy without /signals_breakdown_table: the fallback's rendered counts against the persisted ones."""
    copy = scratch_copy(source, folder / "nobreakdown.h5")
    with pd.HDFStore(str(copy), mode="a") as store:
        store.remove(HDF5_SIGNALS_BREAKDOWN_TABLE)
    rendered = signal_rows("\n".join(replot(f"{mode} fallback", copy, folder / "nobreakdown" / FIGURE_FILE)["texts"]))
    expected = breakdown_rows(persisted[HDF5_SIGNALS_BREAKDOWN_TABLE])
    found = {"fallback_counts": rendered[0] if rendered else None, "persisted_counts": expected[0],
             "filtered_sequence_rows": len(persisted[HDF5_GRADIENT_FILTERED_SEQUENCE]),
             "is_signal_rows": int(persisted["/signals"]["is_signal"].sum())}
    CTX.verdict(f"{mode}: fallback summary counts equal the persisted breakdown (prints/gradient.py:129-131), both "
                "over the signals that build the curve (Q21)",
                bool(rendered) and rendered[0] == expected[0], expected[0], found)
    return found


# =============================================================================
# SCRATCH COPIES
# =============================================================================


def scratch_copy(source: Path, target: Path) -> Path:
    """A guarded byte copy of a run file under the experiment folder."""
    CTX.guard(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    CTX.log.info("scratch copy %s of %s (MD5 %s)", target, source, harness.file_md5(target))
    return target


def planted_control(run_copy: Path, folder: Path) -> dict:
    """One /total_variation point shifted in a realistic copy: its replot must differ from the original table."""
    source = run_copy / harness.REFERENCE_PARAMS.relative_to(harness.DATA_OUTPUT_DIR) / MODES[-1] / PERFORMANCE_FILE
    copy = scratch_copy(source, folder / "planted" / "0.h5")
    with pd.HDFStore(str(copy), mode="a") as store:
        table = store.get("/total_variation")
        storage = store.get_storer("/total_variation").format_type
        table.iloc[PLANTED_ROW, table.columns.get_loc("total_variation")] += np.float32(PLANTED_SHIFT)
        store.put("/total_variation", table, format=storage)
    figure = replot("planted", copy, folder / "planted" / FIGURE_FILE)
    original = read_tables(source)
    strategy = str(original["/strategy_name"]["strategy_name"].iloc[0])
    flagged = not same_curve(figure["top"][strategy], curve(original["/total_variation"], "total_variation"))
    CTX.verdict("planted control: a shifted /total_variation point is flagged", flagged, True,
                {"row": PLANTED_ROW, "shift": PLANTED_SHIFT, "storage": storage})
    return {"flagged": flagged}


if __name__ == "__main__":
    main()
