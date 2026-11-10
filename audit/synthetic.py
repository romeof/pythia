"""Synthetic input files in the read.py layout (shared by EXP-18 and EXP-21).

Each series is one HDF5 group carrying the attributes read.py decodes (columns,
index_encoding), the index dataset of its encoding, one dataset per column and an
optional scalars subgroup whose datasets hold bytes or numpy scalars, as production
input stores them.

Structure:
1. SyntheticSeries
2. write_input
3. Daily frames
"""
import sys

sys.dont_write_bytecode = True

import dataclasses
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

import harness

ENCODING_DAYS = "epoch_days"
ENCODING_SECONDS = "epoch_seconds"
INDEX_DATASETS = {ENCODING_DAYS: "index_days", ENCODING_SECONDS: "index_ts_s"}
VALUE_COLUMNS = ("var0", "var1", "var2", "var3", "var4", "var5")


@dataclasses.dataclass(frozen=True)
class SyntheticSeries:
    """One synthetic series: its frame, its scalars and its index encoding (None: attribute absent)."""

    frame: pd.DataFrame
    scalars: dict = dataclasses.field(default_factory=dict)
    encoding: Optional[str] = ENCODING_DAYS


def write_input(ctx: harness.Context, path: Path, series: dict[str, SyntheticSeries]) -> Path:
    """Write the series into a new file at <...>/<date>/<dataset>/raw/<file>.h5."""
    ctx.guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as sink:
        for series_id, item in series.items():
            group = sink.create_group(series_id)
            if item.encoding is not None:
                group.attrs["index_encoding"] = item.encoding
            if item.encoding in INDEX_DATASETS:
                group[INDEX_DATASETS[item.encoding]] = _encoded_index(item.frame.index, item.encoding)
            group.attrs["columns"] = np.array(list(item.frame.columns), dtype=h5py.string_dtype())
            for column in item.frame.columns:
                group[column] = item.frame[column].to_numpy()
            if item.scalars:
                scalars = group.create_group("scalars")
                for name, value in item.scalars.items():
                    scalars[name] = value
    ctx.log.info("synthetic input %s: series %s, MD5 %s", path, sorted(series), harness.file_md5(path))
    return path


def _encoded_index(index: pd.DatetimeIndex, encoding: str) -> np.ndarray:
    """Days or seconds since the epoch, as production input stores the index."""
    unit = "D" if encoding == ENCODING_DAYS else "s"
    return index.values.astype(f"datetime64[{unit}]").astype("int64")


def daily_frame(values: np.ndarray, start: str, *, capacity: float = 1.0e6) -> pd.DataFrame:
    """Weekday frame with var0..var5 built from one value path (var5 constant capacity)."""
    index = pd.bdate_range(start=start, periods=len(values))
    frame = pd.DataFrame({column: values.astype("float32") for column in VALUE_COLUMNS[:5]}, index=index)
    frame["var1"] = (values * 0.99).astype("float32")
    frame["var2"] = (values * 1.01).astype("float32")
    frame["var5"] = np.full(len(values), capacity).astype("int32")
    return frame[list(VALUE_COLUMNS)]
