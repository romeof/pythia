"""Regression for the row validity rules that preprocessing reads from trigger.validity (REQ3).

In-process on scratch config copies, Preprocess.process_series on testkit's daily sine series, each
case with its own report: (1) the shipped rules: one planted row per rule (a negative var1, a zero
var5, var0 to var3 equal) and a NaN var2 drop exactly those rows, each reported under its own
reason; (2) a NaN var1 beside equal var0, var2 and var3 is reported under the NaN reason only, as a
NaN compares false; (3) the block removed, so the schema's empty defaults apply: only the NaN row
drops; (4) nonzero listing var4 alone: a zero var4 drops, a zero var5 stays; (5) a rule listing
var6 raises ValidationError; (6) FAILING_ROWS holds one failing-row test per ValidityConfig field,
and with every field listing var0 and var1 Preprocess builds each field's rule, in field sequence:
a field added without a test or a read would otherwise apply no rule. A Preprocess that takes no
validity argument fails cases (1) to (4), and rules fixed in code drop the rows that (3) and (4) keep.

Structure:
1. main
2. Cases
3. Helpers
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.utils.common.schema import TriggerConfig, ValidityConfig
from src.utils.preprocess.preprocess import FAILING_ROWS, Preprocess

ROWS = 20
FIRST_DAY = "2022-01-03"
ISSUE_PREFIX = "Date: "
NAN_REASON = "Contains NaN values"
# planted row -> its reason under the shipped rules, in the check sequence
SHIPPED = {2: NAN_REASON, 5: "Fails nonnegative (var0, var1, var2, var3, var4, var5)", 8: "Fails nonzero (var5)",
           11: "Fails not_all_equal (var0, var1, var2, var3)"}
NAN_ROW, NEGATIVE_ROW, ZERO_ROW, EQUAL_ROW = SHIPPED
VAR4_ZERO_ROW, VAR5_ZERO_ROW = 4, 9
VAR4_RULE = {"trigger.validity.nonzero": ["var4"]}
VAR4_REASON = "Fails nonzero (var4)"
UNKNOWN_COLUMN = "var6"
# two distinct columns, so every rule, not_all_equal included, validates
EVERY_RULE_COLUMNS = ("var0", "var1")


def main() -> None:
    """The six cases, then the summary (return code 1 on any FAIL)."""
    shipped_rules()
    nan_beside_equal_values()
    no_rules()
    nonzero_var4()
    unknown_column()
    rules_follow_schema()
    KIT.finish()


# =============================================================================
# CASES
# =============================================================================


def shipped_rules() -> None:
    """Case (1): under the shipped rules the NaN row and one planted row per rule drop, each under its reason."""
    with testkit.scratch(KIT) as folder:
        frame = planted_frame()
        dropped, issues = cleaned(testkit.config_copy(KIT, folder), frame, folder)
        expected = issue_lines(frame, SHIPPED)
        KIT.verdict("case (1): the shipped rules drop the NaN row and one planted row per rule, each under its reason",
                    dropped == sorted(SHIPPED) and issues == expected, {"dropped": sorted(SHIPPED), "issues": expected},
                    {"dropped": dropped, "issues": issues})


def nan_beside_equal_values() -> None:
    """Case (2): a NaN var1 beside equal var0, var2 and var3 fails the NaN check only."""
    with testkit.scratch(KIT) as folder:
        frame = testkit.sine_frame(ROWS, FIRST_DAY)
        frame.iloc[NAN_ROW, [frame.columns.get_loc(column) for column in ("var0", "var2")]] = frame["var3"].iat[NAN_ROW]
        frame.iloc[NAN_ROW, frame.columns.get_loc("var1")] = np.nan
        dropped, issues = cleaned(testkit.config_copy(KIT, folder), frame, folder)
        expected = issue_lines(frame, {NAN_ROW: NAN_REASON})
        KIT.verdict("case (2): a NaN var1 beside equal var0, var2 and var3 is reported under the NaN reason only",
                    dropped == [NAN_ROW] and issues == expected, {"dropped": [NAN_ROW], "issues": expected},
                    {"dropped": dropped, "issues": issues})


def no_rules() -> None:
    """Case (3): without the trigger.validity block only the NaN row drops."""
    with testkit.scratch(KIT) as folder:
        frame = planted_frame()
        dropped, issues = cleaned(testkit.config_copy(KIT, folder, drop_strategy=("trigger.validity",)), frame, folder)
        expected = issue_lines(frame, {NAN_ROW: NAN_REASON})
        KIT.verdict("case (3): the block removed, the schema's empty rules drop the NaN row only",
                    dropped == [NAN_ROW] and issues == expected, {"dropped": [NAN_ROW], "issues": expected},
                    {"dropped": dropped, "issues": issues})


def nonzero_var4() -> None:
    """Case (4): nonzero listing var4 alone drops a zero var4 and keeps a zero var5."""
    with testkit.scratch(KIT) as folder:
        frame = testkit.sine_frame(ROWS, FIRST_DAY)
        frame.iloc[VAR4_ZERO_ROW, frame.columns.get_loc("var4")] = 0.0
        frame.iloc[VAR5_ZERO_ROW, frame.columns.get_loc("var5")] = 0.0
        dropped, issues = cleaned(testkit.config_copy(KIT, folder, strategy=dict(VAR4_RULE)), frame, folder)
        expected = issue_lines(frame, {VAR4_ZERO_ROW: VAR4_REASON})
        KIT.verdict("case (4): nonzero listing var4 drops the zero var4 row and keeps the zero var5 row",
                    dropped == [VAR4_ZERO_ROW] and issues == expected, {"dropped": [VAR4_ZERO_ROW], "issues": expected},
                    {"dropped": dropped, "issues": issues})


def unknown_column() -> None:
    """Case (5): a rule listing a column outside var0 to var5 raises ValidationError."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.validity.nonnegative": [UNKNOWN_COLUMN]})
        exc = testkit.raised(lambda: testkit.validated_config(config_file))
        KIT.verdict(f"case (5): a rule listing {UNKNOWN_COLUMN} raises ValidationError naming it",
                    isinstance(exc, ValidationError) and UNKNOWN_COLUMN in str(exc), f"ValidationError: {UNKNOWN_COLUMN}",
                    repr(exc)[:240])


def rules_follow_schema() -> None:
    """Case (6): FAILING_ROWS holds one failing-row test per ValidityConfig field, and with every field listing columns
    Preprocess builds each field's rule, in field sequence."""
    fields = list(ValidityConfig.model_fields)
    with testkit.scratch(KIT) as folder:
        trigger = testkit.validated_config(testkit.config_copy(KIT, folder)).trigger
        every_rule = ValidityConfig(**{field: list(EVERY_RULE_COLUMNS) for field in fields})
        reasons = [reason for reason, _, _ in preprocessor(trigger, every_rule, folder).rules]
    expected = [f"Fails {field} ({', '.join(EVERY_RULE_COLUMNS)})" for field in fields]
    KIT.verdict("case (6): FAILING_ROWS's keys equal ValidityConfig's fields, and Preprocess builds every field's rule "
                "in field sequence", set(FAILING_ROWS) == set(fields) and reasons == expected,
                {"tests": sorted(fields), "reasons": expected}, {"tests": sorted(FAILING_ROWS), "reasons": reasons})


# =============================================================================
# HELPERS
# =============================================================================


def planted_frame() -> pd.DataFrame:
    """The sine series with a NaN var2, a negative var1, a zero var5 and var0 to var3 equal, each on its own row."""
    frame = testkit.sine_frame(ROWS, FIRST_DAY)
    frame.iloc[NAN_ROW, frame.columns.get_loc("var2")] = np.nan
    frame.iloc[NEGATIVE_ROW, frame.columns.get_loc("var1")] = -1.0
    frame.iloc[ZERO_ROW, frame.columns.get_loc("var5")] = 0.0
    frame.iloc[EQUAL_ROW, [frame.columns.get_loc(column) for column in ("var0", "var1", "var2")]] = frame["var3"].iat[EQUAL_ROW]
    return frame


def cleaned(config_file: Path, frame: pd.DataFrame, folder: Path) -> tuple[list[int], list[str]]:
    """The rows process_series drops under the copy's validated trigger section, and its report's issue lines."""
    trigger = testkit.validated_config(config_file).trigger
    cleaner = preprocessor(trigger, trigger.validity, folder)
    kept = cleaner.process_series("s", frame)
    dropped = sorted(frame.index.get_loc(date) for date in frame.index.difference(kept.index))
    lines = cleaner.report_path.read_text(encoding="utf-8").splitlines()
    return dropped, [line for line in lines if line.startswith(ISSUE_PREFIX)]


def preprocessor(trigger: TriggerConfig, validity: ValidityConfig, folder: Path) -> Preprocess:
    """A Preprocess of the trigger section's block length and printout flag under the given rules, its report in the
    folder."""
    return Preprocess(trigger.consecutive_rows_with_bad_preprocessed_data, validity, trigger.preprocess.printout_io,
                      folder / "report" / "0.h5")


def issue_lines(frame: pd.DataFrame, reasons: dict[int, str]) -> list[str]:
    """The report's issue lines of the rows and their reasons, in the given sequence."""
    return [f"{ISSUE_PREFIX}{frame.index[row]}, Reason: {reason}" for row, reason in reasons.items()]


if __name__ == "__main__":
    main()
