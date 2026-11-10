"""Regression for the replot script (B3-8 (a)).

scripts/gradientreplot.py on a healthy one-series run's performance file (testkit.ONE_SERIES, --task all, the dev
profile), in two children. First with that path and an absent one under a missing folder: the child returns
non-zero, its capture holds a FileNotFoundError naming the absent path, no gradient_hd5.png lies beside the healthy
file and the missing folder was not created, so the paths are checked before the first replot. Then with the
healthy path alone: the child returns 0 and writes plots/gradient/gradient_hd5.png beside it. The script before
C3-16 read one fixed path from its source and printed a missing file instead of raising.

Structure:
1. main
"""
import sys

sys.dont_write_bytecode = True

import testkit

KIT = testkit.preamble(__file__)

SCRIPT = ("scripts", "gradientreplot.py")
FIGURE = ("plots", "gradient", "gradient_hd5.png")
MISSING_FOLDER = "missing"


def main() -> None:
    """The run, the refused child, the replotting child, then the summary (return code 1 on any FAIL)."""
    script = str(KIT.root.joinpath(*SCRIPT))
    with testkit.scratch(KIT) as folder:
        config_file = testkit.config_copy(KIT, folder, strategy=dict(testkit.ONE_SERIES))
        run = testkit.run_pipeline(KIT, folder, "run", config_file, ["all"])
        KIT.verdict("the one-series run returns 0", run.return_code == 0, 0, run.return_code)
        healthy = testkit.output_file(config_file, "performance")
        figure = healthy.parent.joinpath(*FIGURE)
        absent = folder / MISSING_FOLDER / "performance" / "0.h5"
        refused = testkit.run_child(KIT, folder, "refused", [script, str(healthy), str(absent)])
        KIT.verdict("a missing path: return != 0", refused.return_code not in (0, None), "return != 0",
                    {"return_code": refused.return_code, "timed_out": refused.timed_out})
        named = "FileNotFoundError" in refused.text and str(absent) in refused.text
        KIT.verdict("a missing path: the FileNotFoundError names it", named, str(absent),
                    testkit.plain_lines(refused.text)[-1:])
        KIT.verdict("a missing path: no figure beside the healthy file", not figure.exists(), "absent", figure.exists())
        created = (folder / MISSING_FOLDER).exists()
        KIT.verdict("a missing path: the missing folder not created", not created, "absent", created)
        replotted = testkit.run_child(KIT, folder, "replotted", [script, str(healthy)])
        KIT.verdict("the healthy path alone: return code 0", replotted.return_code == 0, 0, replotted.return_code)
        KIT.verdict("the healthy path alone: the figure written beside it", figure.is_file(), str(figure),
                    figure.is_file())
    KIT.finish()


if __name__ == "__main__":
    main()
