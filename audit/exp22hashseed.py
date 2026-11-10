"""EXP-22: run-to-run determinism with two references under differing hash seeds (HYP-15).

A realistic config copy with read.reference_series ["0", "1"] runs --task all twice, under
data/exp/22/<kind>/<label>/s1 and s2: the first run under the harness's seed s1, the second
under the seed the harness's selection picks from the first run's signal file
(harness.select_seed_pair, first seed fixed), so that the iteration order of the /reference_series_* keys differs
(section 6.1, Hash seeds). The task .h5 files are compared with EXP-13. "Identical" counts only
when both runs completed (return code 0, no ERROR line, four Processed lines with N = M,
/gradient_reference_variation holding both reference ids), because a failed Performance task
returns 0 and writes no .h5 (HYP-08), and when the recorded reference-key orders differ.
Planted control: a copy of the first run's performance file without the second reference id
must fail the completeness check.

Structure:
1. main
2. Runs and the completeness check
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

import pandas as pd

import pipelinekit

EXPERIMENT = "22"
MODE = "realistic"
REFERENCES = ["0", "1"]
FIRST_SEED = 1
REFERENCE_PREFIX = "/reference_series_"
REFERENCE_TABLE = "/gradient_reference_variation"
EXPECTED_PROCESSED = 4  # realistic: Trigger, Signal and Performance's two sub-pipelines


def main() -> None:
    """Two runs, the seed chosen between them, the comparison and the planted control."""
    _parse_args()
    folder = CTX.label_dir(EXPERIMENT)
    first = run(folder / "s1", FIRST_SEED)
    signal_file = pipelinekit.output_file(first["config"], MODE, "signal")
    _, second_seed = harness.select_seed_pair(CTX, [signal_file], REFERENCE_PREFIX, first=FIRST_SEED)
    second = run(folder / "s2", second_seed)
    comparison = pipelinekit.compare_trees(CTX, "compare", first["root"], second["root"], h5_only=True)
    both_complete = not first["problems"] and not second["problems"]
    CTX.verdict("non-vacuity: both runs complete (return code 0, no ERROR, four Processed N = M, both reference ids)", both_complete,
                "no problem", {"s1": first["problems"], "s2": second["problems"]})
    CTX.verdict(f"seeds {FIRST_SEED} and {second_seed}: task .h5 files identical value by value",
                both_complete and comparison["identical"], "identical", comparison)
    found = {"seeds": [FIRST_SEED, second_seed], "s1": first, "s2": second, "comparison": comparison,
             "differing_keys": differing_keys(first["config"], second["config"]), "planted": planted_control(folder, first)}
    CTX.log.info("RESULTS %s", json.dumps(found, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root only."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    return parser.parse_args()


# =============================================================================
# RUNS AND THE COMPLETENESS CHECK
# =============================================================================


def run(folder: Path, seed: int) -> dict:
    """--task all on a realistic two-reference copy under the seed; its completeness problems."""
    overrides = {"trigger.replay.mode": MODE, "read.reference_series": REFERENCES}
    config_file = pipelinekit.write_config_copy(CTX, folder, strategy=overrides)
    child, _, origin = CTX.run_pipeline(f"pipeline{seed}_", strategy=pipelinekit.strategy_argument(config_file), tasks=["all"],
                                        quiet_luigi=True, hashseed=seed, allowed_root=folder, overrides=overrides)
    performance = pipelinekit.output_file(config_file, MODE, "performance")
    found = {"config": config_file, "root": folder / pipelinekit.OUTPUT_DIR, "return_code": child.return_code, "log": child.log_path.name,
             "origin": origin["confirmed"], "problems": completeness_problems(child.text, child.return_code, performance)}
    CTX.log.info("run s%s (seed %d): %s", folder.name[1:], seed, json.dumps(found, default=str))
    return found


def completeness_problems(text: str, return_code: object, performance: Path) -> list[str]:
    """Return code 0, no ERROR line, four Processed lines with N = M, both reference ids in the reference table."""
    lines = harness.plain_lines(text)
    problems = [] if return_code == 0 else [f"return code {return_code}"]
    errors = harness.error_lines(lines)
    if errors:
        problems.append(f"{len(errors)} ERROR lines, first: {errors[0][:200]}")
    processed = harness.processed_counts(lines)
    if len(processed) != EXPECTED_PROCESSED or any(total != successful for total, successful in processed):
        problems.append(f"Processed lines {processed}, expected {EXPECTED_PROCESSED} with N = M")
    ids = reference_ids(performance)
    if ids != REFERENCES:
        problems.append(f"{REFERENCE_TABLE} reference ids {ids}, expected {REFERENCES}")
    return problems


def reference_ids(performance: Path) -> list[str]:
    """Sorted reference ids of the performance file's reference table (empty when the file or key is absent)."""
    if pipelinekit.store_keys(performance) is None or REFERENCE_TABLE not in pipelinekit.store_keys(performance):
        return []
    with pd.HDFStore(str(performance), mode="r") as store:
        return sorted(set(store.get(REFERENCE_TABLE)["reference_id"].astype(str)))


def differing_keys(first_config: Path, second_config: Path) -> dict:
    """Per task file, each key whose frames differ, classed 'row permutation only' when they are equal after sorting
    by every column, else 'content'."""
    found = {}
    for task in ("trigger", "signal", "performance"):
        paths = [pipelinekit.output_file(config, MODE, task) for config in (first_config, second_config)]
        with pd.HDFStore(str(paths[0]), mode="r") as left, pd.HDFStore(str(paths[1]), mode="r") as right:
            for key in sorted(set(left.keys()) & set(right.keys())):
                a, b = left.get(key), right.get(key)
                if not a.equals(b):
                    found[f"{task}{key}"] = "row permutation only" if sorted_rows(a).equals(sorted_rows(b)) else "content"
    CTX.verdict("every difference is a row permutation (same rows)", all(kind == "row permutation only" for kind in found.values()),
                "row permutation only", found)
    return found


def sorted_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame's rows sorted by every column; a DatetimeIndex is kept as a column, any other index (row numbers) dropped."""
    flat = frame.reset_index(drop=not isinstance(frame.index, pd.DatetimeIndex))
    return flat.sort_values(list(flat.columns), kind="mergesort").reset_index(drop=True)


def planted_control(folder: Path, first: dict) -> dict:
    """The first run's performance file without reference id 1 must fail the completeness check."""
    source = pipelinekit.output_file(first["config"], MODE, "performance")
    target = folder / "planted" / "0.h5"
    CTX.guard(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    with pd.HDFStore(str(target), mode="a") as store:
        table = store.get(REFERENCE_TABLE)
        storage = store.get_storer(REFERENCE_TABLE).format_type
        store.put(REFERENCE_TABLE, table[table["reference_id"].astype(str) != REFERENCES[-1]], format=storage)
    child_text = (CTX.roots.logs / first["log"]).read_text(encoding="utf-8")
    problems = completeness_problems(child_text, first["return_code"], target)
    CTX.verdict("planted control: a performance file without the second reference id fails the check",
                any(REFERENCE_TABLE in problem for problem in problems), "a reference-id problem", problems)
    return {"problems": problems}


if __name__ == "__main__":
    main()
