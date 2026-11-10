---
#marp: true
theme: default
paginate: true
size: 16:9
---

# Pipeline

## Brief Summary

**Pythia** is a replay pipeline that detects directional triggers in time series, groups correlated triggers into signals, and measures aggregate performance through gradient analysis. The pipeline reads raw series from HDF5, processes them through three sequential tasks, and outputs enriched results with per-series metrics and a cumulative variation curve.

The pipeline flowchart, Mermaid source [`mmflowcharts/pipeline.mmd`](mmflowcharts/pipeline.mmd).

**Tasks**
1. **Trigger** — Read, clean, and detect begin/end triggers per series using a configurable strategy.
2. **Signal** — Group triggers by time overlap and correlation, select a representative signal per group.
3. **Performance** — Compute per-series metrics, aggregate a gradient curve from signals, and persist final results.

**Execution** — How the configuration is validated, how the tasks run, and how a failure ends the run (see below).

---

## Trigger

The **Trigger** step processes time series data to detect triggers (begin/end) using either a **realistic** mode (incremental, no future rows) or an **ideal** mode (full-series visibility), then extracts triggers with direction (`is_plus_trigger`) and outcome (`is_true`) flags.

**Tasks**
1. **Read** — Load time series from HDF5, with user-defined criteria.
2. **Gate Pre-filter** — Fail the run on a series without a configured gate column; with `trigger.gate.prefilter` set to `true`, skip a non-reference series that one applicable gate item passes on no bar of the read window.
3. **Preprocess** — Remove the rows that fail the `trigger.validity` rules; a row with a NaN in `var0`-`var5` always.
4. **Trigger Detection** — Detect starting triggers (trigger-begins) according to a given strategy and their trigger-ends according to a set of user-specified conditions.
5. **Trigger Extraction** — Pair begin/end dates per direction, read precomputed `trigger_value`, and separate triggers as plus/minus and true/false.
6. **Reference Handling** — Optionally save post-read, pre-preprocessing series data for configured reference series, trimmed by lookback window.

See [trigger.md](trigger.md) for full details.

---

## Signal

The **Signal** step groups triggers produced by the Trigger task into signal-window groups based on time overlap and correlation, then selects a representative signal per group.

**Sub-tasks**
1. **Accumulation** — Collect per-series value columns (`var3`, `trigger_value`) and triggers into a single collector via producer-consumer streaming.
2. **Selection** — Assign group IDs based on time overlap, correlation threshold, and signal_criterion; mark signal rows dynamically.
3. **Persistence** — Write groups table, signal series, and reference series to a new HDF5 file.

See [signal.md](signal.md) for full details.

---

## Performance

The **Performance** task computes per-series metrics for triggers and signals, optionally prints/plots per-series outputs, computes one aggregate gradient curve from signals, and persists the final results to performance HDF5.

**Sub-tasks**
1. **Per-series analysis** — Stream trigger/signal series, compute metrics, and optionally print/plot per series.
2. **Gradient analysis** — Build filtered signal segments, aggregate total variation, print/plot gradient outputs.
3. **Persistence** — Save enriched `/signals`, optional `/strategy_name`, and optional `/total_variation`.

See [performance.md](performance.md) for full details.

---

## Execution

**Purpose**

Validate the configuration, then run each requested task once per replay mode, one Luigi build per task.

**Output**

The pipeline flowchart, Mermaid source [`mmflowcharts/pipeline.mmd`](mmflowcharts/pipeline.mmd).

**Logic**
1. Merge the strategy YAML named by `--strategy` with the profile YAML beside it that `--profile` names (`workflows/dev.yaml` or `workflows/prod.yaml`, default `prod`) and validate the result; the validated schema model is used, so declared defaults apply.
2. With `general.enable_info` set to `false`, INFO records are disabled.
3. Expand `trigger.replay.mode` `"dual"` into an Ideal run, then a Realistic run, of every requested task.
4. Before the first build the run removes the strategy folder's provenance files and the output folder of every (mode, task) pair it will build, `_run_task` removes the task's folder again before its build, a folder that cannot be removed fails the run at once, `output()` never deletes, and a failed task's pair and every later pair lose their folders on the way out while finished tasks keep theirs.
5. Run the task through `luigi.build`; `--quiet-luigi False` prints Luigi's own execution report.
6. After the last task the run copies the strategy YAML, the run's profile YAML (`dev.yaml` or `prod.yaml`) and `git_info.txt` into the strategy folder, which it emptied of them, both profile copies included, when it started.

**Data Conditions**
- A failed item raises at once in every task; a series or item is skipped only by an explicit check of a data condition its task documents, never by catching a failure.
- A failed task logs a `Task failed` record naming the cause, then `_run_task`'s task line (`<Task> task failed; rerun with --quiet-luigi False for Luigi's own report`), which names the task, not the cause, and the run returns a non-zero code; no later task or mode runs.
- An unknown configuration key → validation error at startup.
- `trigger.replay.lookback_window` below `trigger.min_length` → validation error at startup, naming both values.
- A store that cannot be read or created is logged once at ERROR, naming its path, and raised; so is a failed store write, naming its key (`Error storing DataFrame at key '<key>': <error>`); any other exception inside a store's `with` block propagates with its own context and no store line.
- A producer's failure reaches its task through the producer's future and, in the default quiet mode, is reported once, by the `Task failed` record (the task line names no cause, above), with no producer, thread or store record; only a store that cannot be read or created adds its own record (above). A consumer's failure adds one ERROR record naming its series (`Error processing series <id>: <error>`), and a failed store write in a consumer adds two, the write's record (above), then the series record; a consumer's failure or an interrupt stops the producer at its next put, so a task returns only once its producer has ended.
- A failed or killed run leaves no previous output of a task it was to run and did not finish, the failed task's preprocess report and debug plots included, and no `extrema.yaml`, `dev.yaml`, `prod.yaml` or `git_info.txt` copy, so no stale file describes outputs the run did not produce.
- A task's consumer reads the queue until the producer's sentinel, so no item put before the producer ended is left unread, and a producer that ends without the sentinel fails the task with a `RuntimeError` after one wait of `runtime.queue_get_timeout` seconds.
- A failed git call (git missing, a folder outside any repository, no `origin` remote, any other non-zero return code, output that is not UTF-8, an expiry after 60 s) is logged at ERROR naming the command and fails the run after its task outputs are written, with no config copy and no `git_info.txt`.
- Every git call runs in the repository that holds the pipeline source (the folder above `src`), not in the working directory, so `git_info.txt` names the code that ran wherever the run starts.
- No usable temp folder for git's output fails the run with the temp-file error itself (a `FileNotFoundError`), with no config copy and no `git_info.txt`.
- The run reads a task's failure from the result of `luigi.build`, which under the pinned luigi 3.6.0 (`setup/requirements.txt`) is `False` for a task whose `run()` raises, although Luigi's docstring promises `True` "even if tasks may fail", so a change of the pin must keep that result.

**Parameters**
- Command line: `--task`, `--strategy`, `--profile` (`dev` or `prod`, default `prod`: the switch YAML merged beside the strategy YAML; `prod` writes no plot or printout), `--quiet-luigi`.
- Config-based: `general.enable_info`, `trigger.replay.mode` ("ideal"/"realistic"/"dual").
- Runtime: `runtime.queue_maxsize` (the queue's bound, at which a producer's put waits) and `runtime.queue_get_timeout` (the seconds of each wait on an empty queue, also the latency of the ended-producer check).
- Hardcoded: each git call is bounded at 60 s (`GIT_TIMEOUT_S`).
