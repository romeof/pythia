"""
HDF5 reading utilities for time series data.

Streams (series_id, DataFrame) from an HDF5 file. Input files are
blosc:zstd-compressed HDF5; hdf5plugin is always imported to
register the Zstd filter before any h5py read.

Structure:
1. Read class: the public generator iter_series, series reading (_read_series, _decode_index), the date
   filter (_apply_date_filter)
2. Module-level helpers (_resolve_columns)
"""

from typing import AbstractSet, Iterator, List, Sequence, Tuple

import h5py
import hdf5plugin  # noqa: F401 — registers Zstd filter before any h5py reads
import pandas as pd

from ..common.schema import ReadConfig
from .prints.read import (
    print_read_date_filter_result,
    print_read_file_summary,
    print_read_series_input,
)
from ..common.constants import VALUE_COLUMNS


class Read:
    """HDF5 reader that streams time series data."""

    def __init__(self, read: ReadConfig, gate_columns: Sequence[str], printout_io: bool):
        """The read section's input file, date window and skipped series; the kept columns are the value columns
        var0-var5 and gate_columns, the columns trigger.gate.items names (an input may store further columns, which
        are dropped); printout_io is trigger.read.printout_io."""
        self.input_file = read.raw_data
        self.ini_date = read.ini_date
        self.end_date = read.end_date
        self.print_io = printout_io
        self.skip_series = set(read.skip_series)
        self.kept_columns = frozenset(VALUE_COLUMNS).union(gate_columns)

    # =========================================================================
    # PUBLIC GENERATOR
    # =========================================================================

    def iter_series(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Yields (series_id, DataFrame) one at a time, streaming from HDF5."""
        with h5py.File(self.input_file, "r") as file:
            if self.print_io:
                all_series = list(file.keys())
                series_to_process = [s for s in all_series if s not in self.skip_series]
                print_read_file_summary(
                    self.input_file, all_series, series_to_process, self.ini_date, self.end_date
                )

            for series_id in file.keys():
                if series_id in self.skip_series:
                    continue

                df = self._read_series(series_id, file)

                if self.print_io:
                    print_read_series_input(series_id, df)

                df = self._apply_date_filter(df)

                if self.print_io:
                    print_read_date_filter_result(df, self.ini_date, self.end_date)

                yield series_id, df

    # =========================================================================
    # SERIES READING
    # =========================================================================

    def _read_series(self, series_id: str, file: h5py.File) -> pd.DataFrame:
        group = file[series_id]
        index = self._decode_index(group)
        kept = _resolve_columns(group, self.kept_columns)
        return pd.DataFrame({var_name: group[var_name][()] for var_name in kept}, index=index)

    def _decode_index(self, group: h5py.Group) -> pd.DatetimeIndex:
        if "index_encoding" not in group.attrs:
            raise ValueError(
                "index_encoding attribute not found in HDF5 group. "
                "This file format is not supported."
            )
        encoding = group.attrs["index_encoding"]
        # The encoding names the unit of the stored timestamps, not the bar spacing: index_ts_s counts seconds
        # and index_days counts days. Daily bars and weekly bars (each dated on its Saturday,
        # W-SAT) are both stored as day counts, so both decode under epoch_days; the dates carry the spacing.
        if encoding == "epoch_seconds":
            return pd.to_datetime(group["index_ts_s"][()], unit="s")
        if encoding == "epoch_days":
            return pd.to_datetime(group["index_days"][()], unit="D")
        raise ValueError(f"Unknown index encoding: {encoding}")

    # =========================================================================
    # DATE FILTERING
    # =========================================================================

    def _apply_date_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.ini_date and self.ini_date != "none":
            df = df[df.index >= pd.to_datetime(self.ini_date)]
        if self.end_date and self.end_date != "none":
            df = df[df.index <= pd.to_datetime(self.end_date)]
        return df


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================


def _resolve_columns(group: h5py.Group, kept: AbstractSet[str]) -> List[str]:
    """Return the stored column names of the HDF5 group that kept holds, in stored order.

    Input files store positional dataset names (var0..varN); pythia keeps them end to end.
    """
    stored_names = list(group.attrs["columns"])
    return [name for name in stored_names if name in kept]
