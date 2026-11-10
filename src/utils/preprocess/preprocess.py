"""
Preprocessing utilities for time series data cleaning.

This module provides the Preprocess class that cleans time series data:
it removes every row with a NaN in a value column (var0-var5) and the rows that fail the trigger.validity rules.
The check reads the value columns only, so a gate column (NaN where a bar holds no value) never removes a row.

Structure:
1. Preprocess class with public starting point process_series
2. Core cleaning logic _clean_series: the NaN check, then one pass over the validity rules
3. Helper methods for condition checking and issue detection
"""

import logging
import operator
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..common.constants import REPORT_ENCODING, VALUE_COLUMNS
from ..common.schema import ValidityConfig
from .prints.preprocess import print_preprocess_output

logger = logging.getLogger(__name__)

# The rows each trigger.validity rule fails, from one comparison per listed column on the frame's own index (a repeated
# date keeps its rows apart); a NaN compares false, so it fails no rule
FAILING_ROWS: Dict[str, Callable[[pd.DataFrame, List[str]], pd.Series]] = {
    "nonnegative": lambda df, columns: reduce(operator.or_, (df[column] < 0 for column in columns)),
    "nonzero": lambda df, columns: reduce(operator.or_, (df[column] == 0 for column in columns)),
    "not_all_equal": lambda df, columns: reduce(operator.and_, (df[column] == df[columns[0]] for column in columns[1:])),
}


class Preprocess:
    """Preprocessor for cleaning time series data before trigger detection."""

    def __init__(self, consecutive_threshold: int, validity: ValidityConfig, printout_io: bool, output_file: Path):
        """Initialize preprocessor with configuration.

        Args:
            consecutive_threshold: trigger.consecutive_rows_with_bad_preprocessed_data, the bad-row block length
                the report names
            validity: trigger.validity, the row validity rules applied after the NaN check, each non-empty one in
                the schema's field sequence
            printout_io: trigger.preprocess.printout_io
            output_file: Path for output file (used to derive report path)
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.report_path = output_file.parent / (
            output_file.stem + "_preprocess_report.txt"
        )
        self._init_report()

        self.consecutive_threshold = consecutive_threshold
        # (reason, failing-row test, columns) of each non-empty rule, in the schema's field sequence
        self.rules = [(f"Fails {rule} ({', '.join(columns)})", FAILING_ROWS[rule], list(columns))
                      for rule, columns in (("nonnegative", validity.nonnegative), ("nonzero", validity.nonzero),
                                            ("not_all_equal", validity.not_all_equal)) if columns]
        self.print_io = printout_io

    def _init_report(self) -> None:
        """Initialize report file with header."""
        with open(self.report_path, "w", encoding=REPORT_ENCODING) as f:
            f.write("Preprocessing Report\n")
            f.write("===================\n\n")

    # =========================================================================
    # PUBLIC STARTING POINT
    # =========================================================================

    def process_series(self, series_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single series: clean and append to report if issues found.

        Args:
            series_id: Identifier for the series
            df: DataFrame containing the time series

        Returns:
            Cleaned DataFrame
        """
        cleaned_df, report = self._clean_series(series_id, df)

        if report["issues"]:
            self._write_report(series_id, report)

        if self.print_io:
            self._print_output(series_id, cleaned_df, report)

        return cleaned_df

    # =========================================================================
    # CORE CLEANING LOGIC
    # =========================================================================

    def _clean_series(
        self, series_id: str, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """Clean a single time series by removing invalid rows.

        Args:
            series_id: Identifier for the series
            df: DataFrame containing the time series

        Returns:
            Tuple of (cleaned DataFrame, report dictionary)
        """
        report = {"perseries_issues": [], "consecutive_issues": [], "issues": False}

        mask = pd.Series(True, index=df.index)

        # The NaN check on the value columns var0-var5 (a gate column outside them never removes a row), then each
        # rule; each tests every row, so a row is reported under each reason it fails
        mask = self._apply_condition_check(df, mask, report,
                                           condition_mask=df[df.columns.intersection(VALUE_COLUMNS)].isna().any(axis=1),
                                           reason="Contains NaN values")
        for reason, failing_rows, columns in self.rules:
            mask = self._apply_condition_check(df, mask, report, condition_mask=failing_rows(df, columns), reason=reason)

        return df[mask], report

    # =========================================================================
    # CONDITION CHECKING METHODS
    # =========================================================================

    def _apply_condition_check(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        report: Dict,
        condition_mask: pd.Series,
        reason: str,
    ) -> pd.Series:
        """Apply a condition check: flag issues, record the reason's consecutive blocks and update mask.

        The blocks come from the condition mask's rows, never from a date lookup, so a repeated date
        cannot break the count.

        Args:
            df: DataFrame being checked
            mask: Current validity mask
            report: Report dictionary to update
            condition_mask: Boolean mask where True indicates invalid rows
            reason: Reason string for the issue

        Returns:
            Updated mask
        """
        if not condition_mask.any():
            return mask

        report["issues"] = True
        for date in df[condition_mask].index:
            report["perseries_issues"].append((date, reason))

        rows = np.flatnonzero(condition_mask.to_numpy())
        report["consecutive_issues"].extend(
            (start, end, reason) for start, end in self._find_consecutive_blocks(rows, df.index[rows])
        )

        return mask & ~condition_mask

    # =========================================================================
    # CONSECUTIVE ISSUES DETECTION
    # =========================================================================

    def _find_consecutive_blocks(self, rows: np.ndarray, dates: pd.Index) -> List[Tuple]:
        """Find blocks of consecutive rows that meet the threshold; a block breaks where rows are more than one apart.

        Args:
            rows: Ascending row numbers with issues
            dates: The dates of those rows, reported as each block's bounds

        Returns:
            List of (start_date, end_date) tuples for consecutive blocks
        """
        if len(rows) == 0:
            return []

        blocks = []
        start_idx = 0

        for i in range(1, len(rows)):
            if rows[i] - rows[i - 1] > 1:
                if i - start_idx >= self.consecutive_threshold:
                    blocks.append((dates[start_idx], dates[i - 1]))
                start_idx = i

        # Check final block
        if len(rows) - start_idx >= self.consecutive_threshold:
            blocks.append((dates[start_idx], dates[-1]))

        return blocks

    # =========================================================================
    # REPORTING AND OUTPUT
    # =========================================================================

    def _write_report(self, series_id: str, report: Dict) -> None:
        """Write issues to the report file."""
        with open(self.report_path, "a", encoding=REPORT_ENCODING) as f:
            f.write(f"\nSeries ID: {series_id}\n")
            f.write("-" * 50 + "\n")

            if report["perseries_issues"]:
                f.write("\nPerseries Issues:\n")
                for date, reason in report["perseries_issues"]:
                    f.write(f"Date: {date}, Reason: {reason}\n")

            if report["consecutive_issues"]:
                f.write(
                    f"\nConsecutive Issues (≥ {self.consecutive_threshold} rows):\n"
                )
                for start_date, end_date, reason in report["consecutive_issues"]:
                    f.write(f"From {start_date} to {end_date}, Reason: {reason}\n")

            f.write("\n")

    def _print_output(self, series_id: str, df: pd.DataFrame, report: Dict) -> None:
        """Print preprocessing output summary."""
        print_preprocess_output(series_id, df, report)
