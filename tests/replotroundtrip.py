"""Regression for the replot round trip: every persisted key read back from the figure (D-11; F-17, R-P7).

One dual --task all child on a testkit.config_copy without overrides (the shipped configuration:
the whole input, reference 0, yearly_attenuation_fraction 0.25), so both performance files hold the twelve
keys of performance.md's table. Per mode, replot_gradient_from_hdf5(..., close_figure=False) on the
performance file, and every key the replot reads read back from the one figure the call adds (the
top panel's rasterized lines by label and its "o" markers, the bottom panel's rasterized lines and
annotations in drawing sequence, the figure's summary text):
- /total_variation: the solid black top line, dates and values, and its first and last dates as the
  two date labels below the bottom axis (the replot takes the curve index from it); /strategy_name:
  its label;
- /total_variation_attenuated: the dashed "after attenuation" line, drawn while /gradient_params'
  yearly_attenuation_fraction is positive;
- /gradient_reference_variation: one dashed "reference <id>" line per reference_id, its rows by date;
- /gradient_terminal_attenuations: the strategy's black marker at the attenuated line's last point exactly
  when its __strategy__ amount is positive, then each reference's marker in its line's color at its
  curve's last value minus its amount;
- /gradient_filtered_sequence and /gradient_signal_segments: one bottom segment per sequence row in
  sequence_index sequence, its group's dates and values clipped at +-50;
- /signals: each segment's color and line style (plots/common.py's style_for of its group's
  is_signal row) and its label "(<n>) [<series_id>] <p>%", n its place in the sequence and p
  compute_variation_pct of the row's trigger values rendered .0f;
- /gradient_params: yearly_attenuation_fraction through the "after attenuation" line;
- /gradient_param_sequences, /signals_breakdown_table, /gradient_metrics_table: the summary text,
  whose signal block holds the count row, then one row per display_signals item in sequence (counts
  as integers, the rest with two decimals), and whose gradient block holds the annualization line's
  N-1 and years, a header naming the metrics table's columns in their stored sequence, then one row
  per display_gradient item the table holds, its cells by the metric's format (int or two
  decimals, NaN blank).
Non-vacuity: every key present in both files, at least two sequence rows, the reference's line and
marker in both modes, and the strategy's marker drawn in ideal mode and absent in realistic mode.
A copy whose /gradient_filtered_sequence rows are reversed replots segment by segment and label by
label like the original; a copy without /signals_breakdown_table replots, its figure written, and
its fallback signal block's count row equals the persisted table's. Recorded exclusions, the
/gradient_params fields that reach no artist, each logged with its reason; a verdict requires
every /gradient_params column read back or recorded. A replot that dropped its sequence_index sort
draws the reversed copy's segments in reverse.

Structure:
1. main
2. Replot and figure readout
3. Key verdicts
4. Copies
5. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.performance.gradient import compute_variation_pct
from src.utils.performance.gradient_.metrics import get_gradient_metric_config
from src.utils.performance.replot import replot_gradient_from_hdf5

SEQUENCE_KEY = "/gradient_filtered_sequence"
SEGMENTS_KEY = "/gradient_signal_segments"
BREAKDOWN_KEY = "/signals_breakdown_table"
METRICS_KEY = "/gradient_metrics_table"
PERFORMANCE_KEYS = {"/signals", "/strategy_name", "/total_variation", "/total_variation_attenuated",
                    "/gradient_reference_variation", SEGMENTS_KEY, SEQUENCE_KEY, "/gradient_terminal_attenuations",
                    "/gradient_params", "/gradient_param_sequences", METRICS_KEY, BREAKDOWN_KEY}
FIGURE_FILE = "gradient_hd5.png"
CLIP = 50.0
AFTER_ATTENUATION = "after attenuation"
REFERENCE_LABEL = "reference {}"
STRATEGY_ATTENUATION = "__strategy__"
COUNT_ROW = "Signals"
# plots/common.py's style_for: (is_plus_trigger, is_true) -> (color, line style)
STYLES = {(True, True): ("blue", "-"), (True, False): ("blue", ":"), (False, True): ("red", "-"), (False, False): ("red", ":")}
SIGNAL_COLUMNS = [f"{direction}_{part}" for direction in ("plus+minus", "plus", "minus") for part in ("true+false", "true", "false")]
DATE_LABEL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SEGMENT_LABEL = re.compile(r"^\(\d+\) \[")
READ_BACK = ("yearly_attenuation_fraction",)
# /gradient_params fields that reach no artist, with the reason the test records
EXCLUDED = {"weight": "loaded by the replot, never passed",
            "variation_method": "loaded by the replot, never passed",
            "concurrent_weight_limit": "loaded by the replot, never passed",
            "attenuation_month": "loaded by the replot, never passed",
            "boundary_deduction_pct": "loaded by the replot, never passed",
            "max_signal_plots": "read only for standalone signal plots, which the replot never emits",
            "mpv": "read only without /gradient_metrics_table, which both files hold",
            "ini_date": "persisted, never read by the replot",
            "lookback": "persisted, never read by the replot",
            "effective_start": "persisted; the replot takes the curve dates from /total_variation",
            "end_date": "persisted; the replot takes the curve dates from /total_variation"}
WHITELIST_EXCLUSION = "the signal_plots_whitelist items: read only for standalone signal plots, which the replot never emits"


def main() -> None:
    """The dual run, the exclusions, both modes' key verdicts and copies, then the summary (return code 1 on any FAIL)."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder)
        child = testkit.run_pipeline(KIT, folder, "dual", config_file, ["all"])
        KIT.verdict("the dual --task all returns 0", child.return_code == 0, 0, {"return_code": child.return_code, "timed_out": child.timed_out})
        for name, reason in [*EXCLUDED.items(), ("signal_plots_whitelist", WHITELIST_EXCLUSION)]:
            KIT.log.info("EXCLUDED %s: %s", name, reason)
        strategy_markers = {}
        for mode in testkit.MODES:
            source = testkit.mode_output(config_file, mode, "performance")
            persisted = testkit.store_frames(source)
            KIT.verdict(f"{mode}: the performance file holds the twelve keys of performance.md's table",
                        set(persisted) == PERFORMANCE_KEYS, sorted(PERFORMANCE_KEYS), sorted(persisted))
            columns = set(persisted["/gradient_params"].columns)
            KIT.verdict(f"{mode}: every /gradient_params column read back or recorded as excluded",
                        columns == {*READ_BACK, *EXCLUDED}, sorted({*READ_BACK, *EXCLUDED}), sorted(columns))
            figure = replotted(mode, source, folder / mode / FIGURE_FILE)
            strategy_markers[mode] = curve_verdicts(mode, figure, persisted)
            segment_verdicts(mode, figure, persisted)
            summary_verdicts(mode, figure, persisted)
            reversed_copy(mode, source, folder / mode, figure)
            breakdown_fallback(mode, source, folder / mode, persisted)
        KIT.verdict("non-vacuity: the strategy's marker drawn in ideal mode, absent in realistic mode",
                    strategy_markers == {"ideal": True, "realistic": False}, {"ideal": True, "realistic": False},
                    strategy_markers)
    KIT.finish()


# =============================================================================
# REPLOT AND FIGURE READOUT
# =============================================================================


def replotted(name: str, source: Path, save_path: Path) -> dict:
    """The one figure the replot call adds, read out, then released with plt.close()."""
    before = set(plt.get_fignums())
    replot_gradient_from_hdf5(source, save_path=save_path, close_figure=False)
    added = sorted(set(plt.get_fignums()) - before)
    KIT.verdict(f"{name}: the replot call adds exactly one figure", len(added) == 1, 1, added)
    figure = plt.figure(added[-1] if added else None)
    top, bottom = figure.axes[:2] if len(figure.axes) >= 2 else (figure.add_subplot(2, 1, 1), figure.add_subplot(2, 1, 2))
    readout = {"top": {line.get_label(): line_data(line) for line in top.get_lines() if line.get_rasterized()},
               "markers": [(line.get_color(), naive(line.get_xdata()[0]), float(line.get_ydata()[0]))
                           for line in top.get_lines() if line.get_marker() == "o"],
               "segments": [line_data(line) for line in bottom.get_lines() if line.get_rasterized()],
               "labels": [text.get_text() for text in bottom.texts if SEGMENT_LABEL.match(text.get_text())],
               "dates": [text.get_text() for text in bottom.texts if DATE_LABEL.match(text.get_text())],
               "summary": "\n".join(text.get_text() for text in figure.texts),
               "saved": save_path.is_file() and save_path.stat().st_size > 0}
    plt.close(figure)
    return readout


def line_data(line) -> tuple:
    """A line's dates, float values, color and line style."""
    return pd.DatetimeIndex(line.get_xdata()), np.asarray(line.get_ydata(), dtype=float), line.get_color(), line.get_linestyle()


# =============================================================================
# KEY VERDICTS
# =============================================================================


def curve_verdicts(mode: str, figure: dict, persisted: dict) -> bool:
    """The top panel: curve, attenuated curve, reference curves and terminal markers; True when the strategy's marker is drawn."""
    strategy = str(persisted["/strategy_name"]["strategy_name"].iloc[0])
    expected = {strategy: (*curve(persisted["/total_variation"], "total_variation"), "black", "-")}
    params = persisted["/gradient_params"].iloc[0]
    attenuated = curve(persisted["/total_variation_attenuated"], "total_variation_attenuated")
    if float(params["yearly_attenuation_fraction"]) > 0.0:
        expected[AFTER_ATTENUATION] = (*attenuated, "black", "--")
    references = {str(ref_id): curve(group.sort_values("date"), "variation")
                  for ref_id, group in persisted["/gradient_reference_variation"].groupby("reference_id")}
    for ref_id, (dates, values) in references.items():
        drawn = figure["top"].get(REFERENCE_LABEL.format(ref_id))
        expected[REFERENCE_LABEL.format(ref_id)] = (dates, values, drawn[2] if drawn else "absent", "--")
    differing = sorted(label for label in set(expected) | set(figure["top"])
                       if label not in expected or label not in figure["top"] or not same_line(figure["top"][label], expected[label]))
    KIT.verdict(f"{mode}: /total_variation, /strategy_name, /total_variation_attenuated and /gradient_reference_variation: "
                "the top lines by label, dates, values, color and style", not differing and bool(references),
                sorted(expected), {"drawn": sorted(figure["top"]), "differing": differing})
    attenuations = dict(zip(persisted["/gradient_terminal_attenuations"]["series_id"].astype(str),
                            persisted["/gradient_terminal_attenuations"]["amount"].astype(float)))
    markers = []
    if attenuations.get(STRATEGY_ATTENUATION, 0.0) > 0.0:
        markers.append(("black", attenuated[1][-1]))
    markers += [(expected[REFERENCE_LABEL.format(ref_id)][2], references[ref_id][1][-1] - amount)
                for ref_id, amount in sorted(attenuations.items()) if ref_id != STRATEGY_ATTENUATION and amount > 0.0]
    drawn = [(color, y) for color, _, y in figure["markers"]]
    at_last = all(x == attenuated[0][-1] for color, x, _ in figure["markers"] if color == "black")
    KIT.verdict(f"{mode}: /gradient_terminal_attenuations: the strategy's black marker at the attenuated last point exactly when its "
                "amount is positive, each reference's in its line's color at its last value minus its amount",
                drawn == markers and at_last and any(color != "black" for color, _ in markers),
                {"markers": markers, "strategy amount": attenuations.get(STRATEGY_ATTENUATION)}, {"markers": drawn, "at last date": at_last})
    return any(color == "black" for color, _ in drawn)


def segment_verdicts(mode: str, figure: dict, persisted: dict) -> None:
    """The bottom panel: segments, their styles and labels in the filtered sequence; the two date labels."""
    sequence = persisted[SEQUENCE_KEY].sort_values("sequence_index")["group_id"].astype(int).tolist()
    grouped = {int(gid): frame for gid, frame in persisted[SEGMENTS_KEY].groupby("group_id")}
    signals = persisted["/signals"]
    rows = signals[signals["is_signal"]].set_index("group_id")
    expected, labels = [], []
    for place, gid in enumerate(sequence, start=1):
        row = rows.loc[gid]
        dates, values = curve(grouped[gid], "pct")
        expected.append((dates, np.clip(values, -CLIP, CLIP), *STYLES[(bool(row["is_plus_trigger"]), bool(row["is_true"]))]))
        pct = compute_variation_pct(float(row["trigger_start_values"]), float(row["trigger_end_values"]))
        labels.append(f"({place}) [{row['series_id']}] {pct:.0f}%")
    differing = [place for place, (drawn, wanted) in enumerate(zip(figure["segments"], expected), start=1)
                 if not same_line(drawn, wanted)]
    KIT.verdict(f"{mode}: /gradient_filtered_sequence, /gradient_signal_segments and /signals: one segment per sequence row "
                f"in sequence_index sequence, clipped at +-{CLIP:g}, styled by its signal row ({len(sequence)} rows)",
                len(sequence) >= 2 and len(figure["segments"]) == len(expected) and not differing,
                f"{len(expected)} equal segments", {"drawn": len(figure["segments"]), "differing": differing[:10]})
    KIT.verdict(f"{mode}: /signals: each segment's label (<n>) [<series_id>] <p>%", figure["labels"] == labels, labels[:3],
                figure["labels"][:3])
    dates = [day.strftime("%Y-%m-%d") for day in pd.to_datetime(persisted["/total_variation"]["date"]).iloc[[0, -1]]]
    KIT.verdict(f"{mode}: the two date labels below the axis are the first and last /total_variation dates",
                figure["dates"] == dates, dates, figure["dates"])


def summary_verdicts(mode: str, figure: dict, persisted: dict) -> None:
    """The summary text: the signal block and the gradient block against their tables and display lists."""
    sequences = persisted["/gradient_param_sequences"].sort_values("sequence_index")
    display = {kind: frame["name"].astype(str).tolist() for kind, frame in sequences.groupby("kind")}
    breakdown = cells(persisted[BREAKDOWN_KEY], "row_key")
    expected = [breakdown_row(breakdown, COUNT_ROW)] + [breakdown_row(breakdown, key) for key in display.get("display_signals", [])]
    rendered = signal_block(figure["summary"])
    KIT.verdict(f"{mode}: /gradient_param_sequences and /signals_breakdown_table: the signal block's count row, then one row "
                f"per display_signals item ({len(expected) - 1})", rendered == expected and len(expected) > 1, expected[:2],
                rendered[:2])
    table = persisted[METRICS_KEY]
    metrics = cells(table, "metric_key")
    headers = list(dict.fromkeys(table["column_header"].astype(str)))
    config = get_gradient_metric_config()
    keys = [key for key in display.get("display_gradient", []) if (key, headers[0]) in metrics]
    wanted = {"annualization": [f"N-1: {int(metrics[('_n_intervals', headers[0])])};",
                                f"years: {metrics[('_n_years', headers[0])]:.2f};"],
              "headers": headers,
              "rows": [[config[key]["label"]] + [metric_cell(metrics[(key, header)], config[key]["format"]) for header in headers]
                       for key in keys]}
    found = gradient_block(figure["summary"], headers, [config[key]["label"] for key in keys])
    KIT.verdict(f"{mode}: /gradient_metrics_table: the annualization line, the header in stored sequence and one row per "
                f"display_gradient item the table holds ({len(keys)}), cells by format", found == wanted and bool(keys),
                wanted, found)


# =============================================================================
# COPIES
# =============================================================================


def reversed_copy(mode: str, source: Path, folder: Path, original: dict) -> None:
    """A copy with reversed /gradient_filtered_sequence rows replots like the original."""
    copy = copied(source, folder / "reversed" / "0.h5")
    with pd.HDFStore(str(copy), mode="a") as store:
        frame = store.get(SEQUENCE_KEY)
        storage = store.get_storer(SEQUENCE_KEY).format_type
        store.put(SEQUENCE_KEY, frame.iloc[::-1], format=storage)
    figure = replotted(f"{mode} reversed", copy, folder / "reversed" / FIGURE_FILE)
    same = (len(figure["segments"]) == len(original["segments"])
            and all(same_line(drawn, wanted) for drawn, wanted in zip(figure["segments"], original["segments"])))
    KIT.verdict(f"{mode}: a copy with reversed /gradient_filtered_sequence rows replots segment by segment and label by label "
                "like the original", same and figure["labels"] == original["labels"], original["labels"][:3],
                {"labels": figure["labels"][:3], "segments equal": same})


def breakdown_fallback(mode: str, source: Path, folder: Path, persisted: dict) -> None:
    """A copy without /signals_breakdown_table replots, its figure written, its fallback count row the persisted one."""
    copy = copied(source, folder / "nobreakdown" / "0.h5")
    with pd.HDFStore(str(copy), mode="a") as store:
        store.remove(BREAKDOWN_KEY)
    figure = replotted(f"{mode} without {BREAKDOWN_KEY}", copy, folder / "nobreakdown" / FIGURE_FILE)
    rendered = signal_block(figure["summary"])
    expected = breakdown_row(cells(persisted[BREAKDOWN_KEY], "row_key"), COUNT_ROW)
    KIT.verdict(f"{mode}: a copy without {BREAKDOWN_KEY} replots, its figure written, its fallback count row the persisted one",
                figure["saved"] and rendered[:1] == [expected], {"saved": True, "counts": expected},
                {"saved": figure["saved"], "counts": rendered[0] if rendered else None})


def copied(source: Path, target: Path) -> Path:
    """A byte copy of a performance file at a new path."""
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    return target


# =============================================================================
# HELPERS
# =============================================================================


def curve(table: pd.DataFrame, column: str) -> tuple:
    """A persisted curve as the replot draws it: dates and float values."""
    return pd.DatetimeIndex(pd.to_datetime(table["date"].values)), table[column].to_numpy().astype(float)


def same_line(drawn: tuple, expected: tuple) -> bool:
    """Equal dates, equal values (NaN-aware: both sides hold the same float32 values), color and style."""
    return (drawn[0].equals(expected[0]) and np.array_equal(drawn[1], expected[1], equal_nan=True)
            and tuple(drawn[2:]) == tuple(expected[2:]))


def naive(value) -> pd.Timestamp:
    """A marker's x as a timezone-free timestamp (matplotlib returns it in UTC)."""
    stamp = pd.Timestamp(value)
    return stamp.tz_convert(None) if stamp.tzinfo is not None else stamp


def cells(table: pd.DataFrame, row_column: str) -> dict:
    """{(row key, column header): value} of a tall table."""
    return {(str(row), str(column)): float(value)
            for row, column, value in zip(table[row_column], table["column_header"], table["value"])}


def breakdown_row(breakdown: dict, row_key: str) -> list:
    """A breakdown row as the signal block renders it: counts as integers, the rest with two decimals."""
    return [(f"{int(value)}" if row_key == COUNT_ROW else f"{value:.2f}") if (value := breakdown.get((row_key, column))) is not None
            else "missing" for column in SIGNAL_COLUMNS]


def signal_block(text: str) -> list:
    """The rows of the summary text that end in nine numbers (the signal block), as their nine tokens."""
    rows = []
    for line in text.splitlines():
        tokens = line.split()[-len(SIGNAL_COLUMNS):]
        if len(tokens) == len(SIGNAL_COLUMNS) and all(pd.notna(pd.to_numeric(token, errors="coerce")) or token == "nan"
                                                      for token in tokens):
            rows.append(tokens)
    return rows


def metric_cell(value: float, metric_format: str) -> str:
    """A gradient table cell as rendered: blank for NaN, an integer or two decimals by the metric's format."""
    if np.isnan(value):
        return ""
    return f"{int(value)}" if metric_format == "int" else f"{value:.2f}"


def gradient_block(text: str, headers: list, labels: list) -> dict:
    """The annualization line's N-1 and years, the header's column names and the body rows (label, then each column's cell)."""
    lines = text.splitlines()
    start = next((number for number, line in enumerate(lines) if line.startswith(" N-1:")), None)
    if start is None or len(lines) < start + 3:
        return {"annualization": [], "headers": [], "rows": []}
    tokens = lines[start].split()
    header = lines[start + 2]
    edges, cursor = [], 0
    for name in headers:
        cursor = header.find(name, cursor) + len(name) if name in header[cursor:] else len(header) + 1
        edges.append(cursor)
    rows = []
    for line, label in zip(lines[start + 3:], labels):
        cell_texts = [line[:edge].split()[-1] if len(line) >= edge and line[edge - 1] != " " else "" for edge in edges]
        rows.append([label if line.startswith(label) else line[:len(label)]] + cell_texts)
    return {"annualization": [" ".join(tokens[0:2]), " ".join(tokens[2:4])],
            "headers": re.split(r"\s{2,}", header.strip()), "rows": rows}


if __name__ == "__main__":
    main()
