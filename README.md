# Pythia

Pythia is a domain-agnostic framework for detecting directional events in time series, grouping correlated events into signals, and measuring aggregate performance through cumulative variation analysis.

Inputs are positional (`series_id`, `var0`-`var5` and the gate columns `trigger.gate.items` names); the input requirements below are all pythia assumes. It applies to any field that produces timestamped numerical measurements meeting the requirements below, such as sensor data in particle physics.

## Input requirements

- Each series holds numeric `var0`-`var5` on a datetime index of observed dates; daily bars and weekly bars stamped on Saturdays (W-SAT) are the tested resolutions, and every horizon parameter counts bars, never calendar days.
- `var3` drives trigger detection and the end checks.
- `var3` and the recorded column (`var3` or `var0`, as `trigger.trigger_value` selects) must be positive: the end checks divide by the begin value of `var3`, each event's variation by that of the recorded column.
- The validity rules come from `trigger.validity`: preprocessing removes a row that fails one, and a row holding a NaN in `var0`-`var5` always.
- Each series holds every gate column `trigger.gate.items` names, read beside `var0`-`var5`; a NaN there marks a bar without a value. A gate column outside `var0`-`var5` never removes a row; a value column an item names is cleaned as before. A series without a configured gate column fails the run: the trigger task logs one `Task failed` record whose `ValueError` names the column and the series, no later task or mode runs, and the process returns a non-zero code.

## Pipeline

Three sequential tasks orchestrated by Luigi:

```
Raw HDF5 → [Trigger] → [Signal] → [Performance]
```

**Trigger** — Reads series from HDF5, preprocesses them (removes rows that fail the validity rules), and detects directional events (trigger begins and ends) using a configurable strategy; a begin needs its previous bar to pass every gate item (`trigger.gate.items`) that applies to the series. Supports two replay modes: `realistic` (incremental, no future rows) and `ideal` (full-series visibility).

**Signal** — Groups triggers from multiple series based on temporal overlap and Pearson correlation. Selects one representative signal per group.

**Performance** — Computes per-series metrics (mean variation, dispersion, duration, positive rate) and an aggregate cumulative variation curve from all signals, including annualized compound rate, maximum fall from the running maximum, and fall and below-maximum durations.

## Quick start (fresh clone)

1. Install **Python 3.11.x**, 64-bit (exactly 3.11: the pins are the environment that produced the reference outputs and are validated only on 3.11).
2. Clone to a **short path**. On Windows without long-path support (`LongPathsEnabled = 0`, the default), keep the clone folder at **100 characters or fewer** (e.g. `C:\work\pythia`): the deepest file in `.venv` adds 149 characters to it and Windows fails at 260.
3. Run the setup script for your platform (see [Setup](#setup)).
4. Copy the input data next to the clone (see [Input data](#input-data)) — it is **not** in git.
5. From the repo root, with the venv active, run `python pipeline.py --task all --strategy extrema --profile dev` (config and data paths are relative to the current directory; `--profile dev` writes the plots and printouts, the default `prod` none).

## Input data

Input HDF5 files are not in git; they sit outside the repo, in the `data/` folder beside it. The default workflow `workflows/extrema.yaml` (`read.raw_data`) reads exactly one file, and the weekly workflow `workflows/extrema_weekly.yaml` reads its weekly counterpart; copy them there yourself, at these paths relative to the repo root:

| Workflow | File | Size | MD5 |
|---|---|---|---|
| `extrema` (daily bars) | `../data/2026_09_25_0/0/raw/0.h5` | 2,184,784 bytes | `f296cfb9bfef57ab95a70e6310c09b6f` |
| `extrema_weekly` (weekly bars, W-SAT) | `../data/2026_09_25_1/0/raw/0.h5` | 857,111 bytes | `0f546e9cbc3ec38ca9f54e75ecc87cb1` |

The listed files hold `var10`-`var15` beside `var0`-`var5`; both workflows' gate items read `var12` and `var15` from them. An input without a column a gate item names fails with the missing-column error of the input requirements.
Nothing else is needed: the per-category folders that sit beside each file in `../data/2026_09_25_<n>/0/raw/` hold the single-series files it was merged from, by the upstream pipeline that produced it, and are not read by the pipeline.

**If a listed file is missing, the run that reads it fails:** the trigger task logs one `Task failed` record naming the `FileNotFoundError` (`[Errno 2]` and the missing path) and no `Error in producer` line, no later task or mode runs, and the process returns a non-zero code; rerun with `--quiet-luigi False` for Luigi's own report.

## Setup

| Platform | Setup (from the repo root) | Activate |
|---|---|---|
| Linux | `bash setup/setup.sh` | `source .venv/bin/activate` |
| Windows (PowerShell) | `powershell -ExecutionPolicy Bypass -File setup\setup.ps1` | `. .\.venv\Scripts\Activate.ps1` |

- Both scripts create `.venv` in the repo root, refuse any interpreter that is not Python 3.11, install `setup/requirements.txt` into the venv, and stop with a non-zero return code if any step fails.
- `setup.sh` is for Linux only; do not run it under Git Bash on Windows (it refuses to). If Python 3.11 is not on `PATH`, put it there first. Debian/Ubuntu: install `python3.11-venv` first.
- Linux: x86-64 with glibc ≥ 2.28 (EL8/EL9, Ubuntu 20.04+); older systems have no binary wheels for `h5py` 3.16.
- Windows: if `Activate.ps1` is blocked by the execution policy (the Windows default `Restricted`), run once `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- `setup/requirements.txt` pins the exact versions that produced the reference outputs; Windows-only and Linux-only packages carry platform markers.

The setup scripts also register a notebook kernel automatically (this overwrites an existing user-level kernel of the same name):

- `Python (pythia .venv)`

In VS Code notebooks, pick that item from `Select Kernel`.

## Usage

```bash
python pipeline.py --task all --strategy extrema --profile dev
```

| Flag | Values | Default | Description |
|---|---|---|---|
| `--task` | `trigger`, `signal`, `performance`, `all` | `trigger` | Task(s) to run |
| `--strategy` | name of a strategy YAML in `workflows/` (not a profile: `dev`, `prod`); currently `extrema` (daily bars) and `extrema_weekly` (weekly bars) | `extrema` | Workflow config to load |
| `--profile` | `dev`, `prod` | `prod` | Switch profile merged with the strategy: `dev` writes the plots and printouts of `workflows/dev.yaml`; `prod` writes no plot or printout |
| `--quiet-luigi` | `True` / `False` | `True` | Suppress Luigi scheduler logs |

Tasks can also be run individually in sequence:

```bash
python pipeline.py --task trigger     --strategy extrema --profile dev
python pipeline.py --task signal      --strategy extrema --profile dev
python pipeline.py --task performance --strategy extrema --profile dev
```

## Configuration

Each strategy maps to a YAML file in `workflows/`; it is merged with the profile `--profile` names, `workflows/dev.yaml` or `workflows/prod.yaml` (print/plot/debug switches), which must exist beside it. Key sections:

- `read` — input HDF5 path, date range, series filter, reference series
- `trigger` — the gate on the previous bar (`gate`: its items and the pre-filter), row validity rules, recording mode (`trigger_value`), detection strategy parameters, replay mode, end conditions
- `signal` — correlation threshold, lookback period, signal selection criterion
- `performance` — weight allocation, variation aggregation method, attenuation rate, metric display order

All parameters are validated at startup against Pydantic schemas in `src/utils/common/schema.py`. Unknown parameters raise an error.

### Replay modes

`trigger.replay.mode` selects `ideal` (full-series visibility), `realistic` (incremental, no future rows) or `dual`, which runs every requested task once in `ideal` and once in `realistic` mode. `workflows/extrema.yaml` uses `dual`.

## Output

Outputs are written under `read.output_root` (`./data` in `workflows/extrema.yaml`, inside the folder you run from), in a tree named after the input's date and dataset folders (`<date>/<dataset>/raw/<input file name>`):

```
<output_root>/<date>/<dataset>/<strategy>/<parameters>/<mode>/<task>/<input file name>
```

For the default workflow: `data/2026_09_25_0/0/extrema/var12ge0_83x0-var15ge0_73__w7md5p0_05/{ideal,realistic}/{trigger,signal,performance}/0.h5`; for the weekly workflow: `data/2026_09_25_1/0/extrema/var12ge0_83x0-var15ge0_73__w7md1p0_05/{ideal,realistic}/{trigger,signal,performance}/0.h5` (`<strategy>` is `trigger.strategy.strategy_name`, not the `--strategy` value). The `<parameters>` folder names the gate items, joined by `-` (`var12ge0_83x0` is `var12 >= 0.83` exempting series `0`; `nogate` for none), then the extrema parameters, never `trigger.gate.prefilter`. It also receives copies of the strategy YAML, which records the pre-filter switch, and the run's profile YAML (`<profile>.yaml`) plus `git_info.txt` (commit, branch, remote) of a completed run; the run needs git and a repository with an `origin` remote.

| Task | Output | Contents |
|---|---|---|
| Trigger | `<mode>/trigger/0.h5` | Per-series DataFrames with event flags; trigger tables with begin/end dates, direction, outcome |
| Signal | `<mode>/signal/0.h5` | Signal groups table with correlation scores, group IDs, representative flags |
| Performance | `<mode>/performance/0.h5` | Enriched signals table; cumulative variation curve; per-series metrics |
| Performance plots | `<mode>/performance/plots/{gradient,signal,trigger}/*.png` | Written under the `dev` profile, whose `plot_debug` switches are on; the `prod` profile writes none |

## Tests

**Windows only, except `tests/test_metrics_parity.py`:** `tests/runall.py` and every script that imports `tests/testkit.py` raise `RuntimeError` at import on another platform, Linux included. The tests are plain scripts run from the repo root with the venv active:

```bash
python tests/runall.py
python tests/test_metrics_parity.py
```

- Set `PYTHONUTF8=1` (or run `python -X utf8`): the testkit scripts refuse to start without UTF-8 mode, and on Windows the `✓` symbols of `tests/test_metrics_parity.py` raise `UnicodeEncodeError` under the cp1252 code page when redirected.
- `tests/runall.py` runs every other script under `tests/` as a bounded child, one verdict per script, and returns 1 when any of them fails; each script also runs on its own.
- `tests/replayvalidation.py` replaces the old replay validation suite: ideal pairing parity, no future rows by prefix invariance, rolling against expanding windows and the length boundary, one verdict each.
- `tests/test_metrics_parity.py` keeps its name and its own command (above); `tests/runall.py` runs it too.
- `tests/endtoend.py`, `tests/endmodes.py`, `tests/boundaries.py`, `tests/raisesites.py`, `tests/outputcontract.py`, `tests/consoles.py`, `tests/configcontract.py`, `tests/replotroundtrip.py` and `tests/luigibuild.py` check the hand-derived results of a synthetic input in both modes, the realistic end modes and suppressions, the boundary inputs, every production `raise`, the data contract of a run's outputs, a run under the ANSI code page, the configuration contract, the replot round trip and the `luigi.build` result. `tests/weeklybars.py` reads the weekly input listed under [Input data](#input-data): it checks the weekly workflow, its run on that file against measured figures, and the hand-derived results of a synthetic weekly input in both modes.

## Known limitations

- **Signal task memory.** The signal selection keeps every correlation it computes until the build ends (`src/utils/signal/selection.py`), so its memory grows faster than the series count: the realistic Signal task added about 0.9 GiB at 1k series on 2.7 years and peaked at 1.0-1.4 GiB on 390 series over 10-20 years. If memory becomes the limit, drop a cached correlation once its begin date leaves the period: the cache holds only computed values, so the outputs stay the same.

## Documentation

- `docs/pipeline.md` — full pipeline overview
- `docs/trigger.md` — trigger detection logic
- `docs/signal.md` — signal grouping logic
- `docs/performance.md` — metrics and gradient computation
- `docs/metrics.md` — metric definitions and annualization formulas
