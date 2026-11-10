"""EXP-07: partial writes, crash consumption and soft write failures (HYP-07, HYP-09, HYP-28, HYP-29, HYP-30).

Crashes (standard 02 kill test, deterministic): a child patches Trigger._consumer_worker to
call os._exit(97) after k series were written (k = 0: the store created, nothing written; 3; 6: all
written, the store not yet released) and runs pipeline.main() for --task trigger on a realistic copy.
Per crash: the crashed 0.h5 (MD5, readable keys, read in a child), then --task signal alone
on it (no requires(), exists() only), then a rerun of --task trigger compared with an
uninterrupted run (_run_task cleans the task folder before the build, Q12).
Soft cases: a child plants an OSError inside one handler's try and runs pipeline.main():
s1 Performance._persist_gradient_plot_data, s2 Preprocess.process_series for series 1,
s3 HDF5Manager.put_dataframe_normalized for reference keys, and the HYP-09 read sites:
s4 HDFStore.get for /series_1 (get_dataframe_fast's handler), s5 get_dataframe_fast for
/series_1 (iter_series's handler), s6 HDFStore.get for /signals (get_dataframe's handler),
s7 HDFStore.keys on the trigger file (get_keys_set's handler), s8 get_dataframe_fast for the
reference key of the signal file (Performance's reference loader).

Structure:
1. main
2. Crashes
3. Soft cases
4. Child commands
"""
import sys

sys.dont_write_bytecode = True

import os

sys.path.insert(0, os.environ["PYTHIA_AUDIT_HARNESS"])
import harness

CTX = harness.preamble(__file__, imports_src=True)

import argparse
import json
from pathlib import Path

import pandas as pd

import pipelinekit
from src.tasks.performance import Performance
from src.tasks.trigger import Trigger
from src.utils.common.hdf5manager import HDF5Manager
from src.utils.preprocess.preprocess import Preprocess

EXPERIMENT = "07"
CRASH_RETURN_CODE = 97
CRASH_POINTS = (0, 3, 6)
MODE = "realistic"
REALISTIC = {"trigger.replay.mode": MODE}
CHILD_TIMEOUT_S = 900.0
PLANTED = "planted OSError"
SOFT_CASES = {
    "s1": (["all"], "Error writing performance HDF5"),
    "s2": (["trigger"], "Error preprocessing series 1"),
    "s3": (["trigger"], "Error saving reference raw data for series 0"),
    "s4": (["all"], "Error retrieving DataFrame from key '/series_1'"),
    "s5": (["all"], "Error loading series from /series_1"),
    "s6": (["all"], "Error retrieving DataFrame from key '/signals'"),
    "s7": (["all"], "Error retrieving keys set from HDF5 store"),
    "s8": (["all"], "Skipping malformed reference series at HDF5 key /reference_series_0"),
}
TASKS = ("trigger", "signal", "performance")


def main() -> None:
    """Child commands when requested, else the clean run, the crashes and the soft cases."""
    args = _parse_args()
    if args.pipeline_args is not None:
        child_run(args)
        return
    if args.child_keys is not None:
        CTX.log.info("KEYS %s", json.dumps(pipelinekit.store_keys(Path(args.child_keys))))
        return
    folder = CTX.label_dir(EXPERIMENT)
    clean = run_unpatched(folder / "clean", ["trigger"], "clean")
    results = {"clean": clean, "crashes": {k: check_crash(folder / f"crash{k}", k, folder / "clean") for k in CRASH_POINTS},
               "soft": {case: check_soft(folder / f"soft{case[1:]}", case) for case in SOFT_CASES}}
    CTX.log.info("RESULTS %s", json.dumps(results, sort_keys=True, default=str))
    CTX.summary()


def _parse_args() -> argparse.Namespace:
    """--source-root; child options --child-crash K | --child-soft CASE with --pipeline-args, --child-keys PATH."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, default=harness.REAL_WORKTREE)
    parser.add_argument("--child-crash", type=int)
    parser.add_argument("--child-soft", choices=sorted(SOFT_CASES))
    parser.add_argument("--child-keys")
    parser.add_argument(harness.PIPELINE_ARGS_OPTION)
    return parser.parse_args()


def run_unpatched(folder: Path, tasks: list[str], name: str, config_file: Path = None) -> dict:
    """pipeline.py on a realistic copy (written when not given); return code, decisive lines, store keys."""
    config_file = config_file or pipelinekit.write_config_copy(CTX, folder, strategy=REALISTIC)
    child, _, origin = CTX.run_pipeline("pipeline", strategy=pipelinekit.strategy_argument(config_file), tasks=tasks,
                                        quiet_luigi=True, hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder,
                                        overrides=REALISTIC, timeout_s=CHILD_TIMEOUT_S)
    return observation(name, config_file, child, origin)


def observation(name: str, config_file: Path, child: harness.ChildResult, origin: dict) -> dict:
    """Return code, ERROR and Processed lines, and the keys of every task store of the copy."""
    lines = harness.plain_lines(child.text)
    found = {"config": str(config_file), "return_code": child.return_code, "timed_out": child.timed_out, "origin": origin["confirmed"],
             "errors": [line.strip()[:200] for line in harness.error_lines(lines)][:8],
             "processed": harness.processed_counts(lines),
             "keys": {task: pipelinekit.store_keys(pipelinekit.output_file(config_file, MODE, task)) for task in TASKS},
             "log": child.log_path.name}
    CTX.log.info("%s: %s", name, json.dumps(found, default=str))
    return found


# =============================================================================
# CRASHES
# =============================================================================


def check_crash(folder: Path, k: int, clean_folder: Path) -> dict:
    """A crash after k written series; the crashed file read alone, consumed by Signal, healed by a rerun."""
    config_file = pipelinekit.write_config_copy(CTX, folder, strategy=REALISTIC)
    child, _, origin = CTX.run_pipeline("crash", strategy=pipelinekit.strategy_argument(config_file), tasks=["trigger"],
                                        quiet_luigi=True, hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder,
                                        overrides=REALISTIC, timeout_s=CHILD_TIMEOUT_S,
                                        child_argv=[str(Path(__file__).resolve()), "--child-crash", str(k),
                                               "--source-root", str(CTX.source_root)])
    trigger_file = pipelinekit.output_file(config_file, MODE, "trigger")
    crashed = {"return_code": child.return_code, "md5": harness.file_md5(trigger_file) if trigger_file.is_file() else None,
               "readable": read_keys(trigger_file)}
    CTX.verdict(f"crash k={k}: the child ended by the planted crash", child.return_code == CRASH_RETURN_CODE, CRASH_RETURN_CODE, child.return_code)
    signal = run_unpatched(folder, ["signal"], f"crash k={k} --task signal", config_file)
    healed = run_unpatched(folder, ["trigger"], f"crash k={k} rerun", config_file)
    comparison = pipelinekit.compare_trees(CTX, "compare", trigger_file.parent,
                                           pipelinekit.output_file(clean_folder / pipelinekit.STRATEGY_FILE, MODE, "trigger").parent,
                                           h5_only=True)
    found = {"crashed": crashed, "signal_after_crash": signal, "rerun": healed, "rerun_vs_clean": comparison}
    CTX.log.info("crash k=%d: %s", k, json.dumps(found, default=str))
    CTX.verdict(f"crash k={k}: intent (R-X4) no partial 0.h5 left at the output path", crashed["md5"] is None,
                "absent", crashed)
    CTX.verdict(f"crash k={k}: intent a separately started Signal does not consume a partial file as complete",
                signal["return_code"] not in (0, None) or signal["keys"]["signal"] is None, "fails or writes nothing",
                {"return_code": signal["return_code"], "signal_keys": signal["keys"]["signal"], "errors": signal["errors"][:2]})
    CTX.verdict(f"crash k={k}: a rerun of the task self-heals (_run_task cleans first, Q12; equal to the clean run)",
                comparison["identical"],
                "identical", comparison)
    return found


def read_keys(path: Path) -> dict:
    """Keys of a possibly partial store, read in a child (an unreadable file ends that child)."""
    if not path.is_file():
        return {"exists": False}
    child = CTX.launch_child("keys", [str(Path(__file__).resolve()), "--child-keys", str(path),
                                      "--source-root", str(CTX.source_root)],
                             timeout_s=harness.CHILD_TIMEOUT_S, hashseed=harness.DEFAULT_HASHSEED, audit_child=True)
    keys = pipelinekit.last_prefixed(child.text, "KEYS ")
    return {"exists": True, "return_code": child.return_code, "keys": json.loads(keys) if keys else None,
            "error": [line for line in child.text.splitlines() if "Error" in line][-1:]}


# =============================================================================
# SOFT CASES
# =============================================================================


def check_soft(folder: Path, case: str) -> dict:
    """One planted OSError inside a handler's try: what base reports, what intent requires."""
    tasks, swallow_line = SOFT_CASES[case]
    config_file = pipelinekit.write_config_copy(CTX, folder, strategy=REALISTIC)
    child, _, origin = CTX.run_pipeline("soft", strategy=pipelinekit.strategy_argument(config_file), tasks=tasks,
                                        quiet_luigi=True, hashseed=harness.DEFAULT_HASHSEED, allowed_root=folder,
                                        overrides=REALISTIC, timeout_s=CHILD_TIMEOUT_S,
                                        child_argv=[str(Path(__file__).resolve()), "--child-soft", case,
                                               "--source-root", str(CTX.source_root)])
    found = observation(f"soft {case}", config_file, child, origin)
    found["swallow_line"] = child.text.count(swallow_line)
    found["planted_seen"] = child.text.count(PLANTED)
    CTX.verdict(f"soft {case}: the planted failure reached the handler (its swallow line logged)",
                found["swallow_line"] > 0, swallow_line, found["swallow_line"])
    CTX.verdict(f"soft {case}: intent (R-X3) the run fails loudly", child.return_code not in (0, None), "return != 0",
                {"return_code": child.return_code, "keys": found["keys"], "processed": found["processed"]})
    return found


# =============================================================================
# CHILD COMMANDS
# =============================================================================


def child_run(args: argparse.Namespace) -> None:
    """Child: install the crash or the soft plant, then pipeline.main() with the probe-checked arguments."""
    if args.child_crash is not None:
        install_crash(args.child_crash)
    if args.child_soft is not None:
        install_soft(args.child_soft)
    pipelinekit.pipeline_main(json.loads(args.pipeline_args))


def install_crash(k: int) -> None:
    """Trigger._consumer_worker ends the process with os._exit after k written series (k = 0: before any)."""
    original = Trigger._consumer_worker
    calls = [0]

    def crashing(self, batch, trigger_strategy, trigger_helper, store):
        if k == 0:
            os._exit(CRASH_RETURN_CODE)
        result = original(self, batch, trigger_strategy, trigger_helper, store)
        calls[0] += 1
        if calls[0] == k:
            os._exit(CRASH_RETURN_CODE)
        return result

    Trigger._consumer_worker = crashing


def install_soft(case: str) -> None:
    """Plant one OSError at the site the case names."""
    getattr(sys.modules[__name__], f"_plant_{case}")()


def _plant_s1() -> None:
    def failing(self, **kwargs):
        raise OSError(f"{PLANTED} s1: _persist_gradient_plot_data")
    Performance._persist_gradient_plot_data = failing


def _plant_s2() -> None:
    original = Preprocess.process_series

    def failing(self, series_id, df):
        if series_id == "1":
            raise OSError(f"{PLANTED} s2: process_series {series_id}")
        return original(self, series_id, df)
    Preprocess.process_series = failing


def _plant_s3() -> None:
    original = HDF5Manager.put_dataframe_normalized

    def failing(self, store, key, df, format, schema=None):
        if key.startswith("reference_series_"):
            raise OSError(f"{PLANTED} s3: put {key}")
        return original(self, store, key, df, format=format, schema=schema)
    HDF5Manager.put_dataframe_normalized = failing


def _plant_s4() -> None:
    original = pd.HDFStore.get

    def failing(self, key):
        if key == "/series_1":
            raise OSError(f"{PLANTED} s4: HDFStore.get {key}")
        return original(self, key)
    pd.HDFStore.get = failing


def _plant_s5() -> None:
    original = HDF5Manager.get_dataframe_fast

    def failing(self, store, key):
        if key == "/series_1":
            raise OSError(f"{PLANTED} s5: get_dataframe_fast {key}")
        return original(self, store, key)
    HDF5Manager.get_dataframe_fast = failing


def _plant_s6() -> None:
    original = pd.HDFStore.get

    def failing(self, key):
        if key == "/signals":
            raise OSError(f"{PLANTED} s6: HDFStore.get {key}")
        return original(self, key)
    pd.HDFStore.get = failing


def _plant_s7() -> None:
    original = pd.HDFStore.keys

    def failing(self, include="pandas"):
        if Path(self.filename).parent.name == "trigger" and self._mode == "r":
            raise OSError(f"{PLANTED} s7: HDFStore.keys {self.filename}")
        return original(self, include)
    pd.HDFStore.keys = failing


def _plant_s8() -> None:
    original = HDF5Manager.get_dataframe_fast

    def failing(self, store, key):
        if key.startswith("/reference_series_") and Path(store.filename).parent.name == "signal":
            raise OSError(f"{PLANTED} s8: get_dataframe_fast {key}")
        return original(self, store, key)
    HDF5Manager.get_dataframe_fast = failing


if __name__ == "__main__":
    main()
