"""Regression for plateau extrema (HYP-22, R-T13).

A child builds the extrema strategy from a scratch config (ideal mode) and runs its begin
generation on EXP-01's plateau fixture: 100 + 10 sin(x), x in [0, 3 pi], 97 samples, with sample
17 set equal to sample 16, a two-sample plateau at the first maximum, which find_peaks reports at
its left middle sample 16. Expected: minus begins exactly at the maxima 16 and 80, a plus begin
exactly at the minimum 48, no row with both begins, and the rows with turning_points = 1 exactly
the begin rows. The unfixed code types points by strict neighbour comparisons, which the tie
fails, so the plateau gets turning_points = 1 and no begin.

Structure:
1. main
2. The case
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

import json

SAMPLES, PLATEAU = 97, 16
MAXIMA, MINIMA = [PLATEAU, 80], [48]
TURNING, MINUS, PLUS = "turning_points", "trigger_minus_begin", "trigger_plus_begin"
REPORT_PREFIX = "PLATEAU "
CHILD_CODE = testkit.CHILD_BOOTSTRAP + (
    "\nimport json\nfrom pathlib import Path\nimport numpy as np, pandas as pd\n"
    "config_file = Path(sys.argv[4])\n"
    f"values = 100.0 + 10.0 * np.sin(np.linspace(0.0, 3.0 * np.pi, {SAMPLES}))\n"
    f"values[{PLATEAU} + 1] = values[{PLATEAU}]\n"
    f"frame = pd.DataFrame({{'var3': values}}, index=pd.bdate_range('2022-01-03', periods={SAMPLES}))\n"
    "strategy = testkit.strategy(testkit.validated_config(config_file), testkit.output_file(config_file, 'trigger'))\n"
    "marked = strategy._generate_trigger_begin(frame)\n"
    f"report = {{name: np.flatnonzero(marked[name].to_numpy() > 0).tolist() for name in ('{TURNING}', '{MINUS}', '{PLUS}')}}\n"
    f"sys.stdout.write('{REPORT_PREFIX}' + json.dumps(report) + '\\n')\n")


def main() -> None:
    """The plateau case, then the summary (return code 1 on any FAIL)."""
    plateau_begin()
    KIT.finish()


# =============================================================================
# THE CASE
# =============================================================================


def plateau_begin() -> None:
    """Minus begins exactly at the maxima, the plateau's reported sample included, a plus begin exactly at the
    minimum, never both on one row; the turning points exactly the begin rows."""
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy={"trigger.replay.mode": "ideal"})
        child = testkit.run_child(KIT, folder, "plateau", ["-c", CHILD_CODE, *testkit.child_args(KIT), str(config_file)])
        lines = [line[len(REPORT_PREFIX):] for line in testkit.plain_lines(child.text) if line.startswith(REPORT_PREFIX)]
        report = json.loads(lines[-1]) if lines else {}
        KIT.verdict("non-vacuity: the child reported its begin generation", child.return_code == 0 and bool(report), "a report",
                    {"return_code": child.return_code, "report": report})
        minus, plus = report.get(MINUS), report.get(PLUS)
        KIT.verdict(f"minus begins exactly at the maxima, the plateau sample {PLATEAU} included", minus == MAXIMA, MAXIMA,
                    minus)
        KIT.verdict("a plus begin exactly at the minimum", plus == MINIMA, MINIMA, plus)
        both = sorted(set(minus or []) & set(plus or []))
        KIT.verdict("no row carries both begins", bool(report) and not both, "none", both)
        begins = sorted(set(minus or []) | set(plus or []))
        KIT.verdict("the rows with turning_points = 1 are exactly the begin rows", bool(report) and report[TURNING] == begins,
                    begins, report.get(TURNING))


if __name__ == "__main__":
    main()
