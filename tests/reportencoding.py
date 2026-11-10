"""Regression for the preprocess report under the platform's ANSI code page (HYP-12).

An ANSI child (-X utf8=0, PYTHONUTF8 removed, no UTF-8 assertion) records its code page, then
builds Preprocess on a scratch config copy with its report in the scratch folder and runs
process_series on a synthetic series with three bad rows on calendar-consecutive dates (the
shipped threshold of three). The case first requires that the child's code page cannot encode
the report's "≥" (otherwise it cannot bite and fails as "could not run"); then no exception,
"Consecutive Issues" in the report and the frame without the three rows. The unfixed code raises
UnicodeEncodeError in the report write.

Structure:
1. main
2. The case
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import re

REPORT_FILE = "0_preprocess_report.txt"
CONSECUTIVE = "Consecutive Issues"
ROWS, BAD_ROWS = 20, 3
ENCODABLE, KEPT = "ENCODABLE=", "KEPT="
ERROR_LINE = re.compile(r"^\w*(Error|Exception)\b.*:")
CHILD_CODE = testkit.ANSI_CHILD_BOOTSTRAP + (
    "\n"
    "import numpy as np, pandas as pd\n"
    "from pathlib import Path\n"
    "from src.utils.preprocess.preprocess import Preprocess\n"
    "sign = chr(0x2265)\n"
    f"sys.stdout.write('{ENCODABLE}' + str(sign.encode(locale.getpreferredencoding(False), 'replace') != '?'.encode('ascii')) + '\\n')\n"
    "config = testkit.validated_config(Path(sys.argv[4]))\n"
    f"index = pd.bdate_range('2022-01-03', periods={ROWS})\n"
    f"values = 100.0 + np.arange({ROWS}, dtype=float)\n"
    "frame = pd.DataFrame({name: values for name in ('var0', 'var1', 'var2', 'var3', 'var4')}, index=index)\n"
    "frame['var1'], frame['var2'] = values * 0.99, values * 1.01\n"
    "frame['var5'] = 1000000\n"
    f"frame.iloc[1:1 + {BAD_ROWS}, frame.columns.get_loc('var5')] = 0\n"
    "kept = Preprocess(config.trigger.consecutive_rows_with_bad_preprocessed_data, config.trigger.validity,"
    " config.trigger.preprocess.printout_io, Path(sys.argv[5]) / '0.h5').process_series('s', frame)\n"
    f"sys.stdout.write('{KEPT}' + str(len(kept)) + '\\n')\n")


def main() -> None:
    """The ANSI case, then the summary (return code 1 on any FAIL)."""
    ansi_report()
    KIT.finish()


# =============================================================================
# THE CASE
# =============================================================================


def ansi_report() -> None:
    """Under the ANSI code page: the case can bite, no exception, the block reported, the bad rows dropped."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder)
        report_folder = folder / "report"
        report_folder.mkdir()
        child = testkit.run_child(KIT, folder, "ansi", ["-c", CHILD_CODE, *testkit.child_args(KIT), str(config_file),
                                                        str(report_folder)], ansi=True)
        lines = testkit.plain_lines(child.text)
        encodable, kept = (next((line[len(prefix):].strip() for line in lines if line.startswith(prefix)), None)
                           for prefix in (ENCODABLE, KEPT))
        KIT.verdict("the child's code page cannot encode the report's sign (else the case could not run)", encodable == "False",
                    "False", encodable or "could not run")
        KIT.verdict("process_series raised nothing", child.return_code == 0 and not any(ERROR_LINE.match(line) for line in lines),
                    "return code 0", {"return_code": child.return_code, "errors": [line for line in lines if ERROR_LINE.match(line)][-1:]})
        report = report_folder / REPORT_FILE
        text = report.read_bytes().decode("utf-8", "replace") if report.is_file() else ""
        KIT.verdict(f"the report holds '{CONSECUTIVE}'", CONSECUTIVE in text, CONSECUTIVE, f"{len(text)} characters")
        KIT.verdict(f"the frame lost exactly its {BAD_ROWS} bad rows", kept == str(ROWS - BAD_ROWS), ROWS - BAD_ROWS, kept)


if __name__ == "__main__":
    main()
