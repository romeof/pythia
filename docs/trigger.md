# Trigger

## Brief Summary

The **Trigger** step processes time series data to detect triggers (begin/end) using either a **realistic** mode (incremental, no future rows) or an **ideal** mode (full-series visibility), then extracts triggers with direction (`is_plus_trigger`) and outcome (`is_true`) flags. The pipeline reads raw time series from HDF5 files, applies date filtering, cleans invalid data points, detects triggers via a configurable strategy, annotates each row with begin/end markers and end reasons, and pairs begin/end dates to produce per-series triggers. The output is a single HDF5 file containing both the time series (`/series_{id}`) and (`/triggers_{id}`) with the information added during the trigger detection and the extracted triggers information.

**Tasks**
1. **Read** — Load time series from HDF5, with user-defined criteria.
2. **Gate Pre-filter** — Fail the run on a series without a configured gate column; with `trigger.gate.prefilter` set to `true`, skip a non-reference series on which an applicable gate item does not hold on any row of the read window.
3. **Preprocess** — Remove the rows that fail the `trigger.validity` rules; a row with a NaN in `var0`-`var5` always.
4. **Trigger Detection** — Detect starting triggers (trigger-begins) according to a given strategy and their trigger-ends according to a set of user-specified conditions.
5. **Trigger Extraction** — Pair begin/end dates per direction, read precomputed `trigger_value`, and separate triggers as plus/minus and true/false.
6. **Reference Handling** — Optionally save post-read, pre-preprocessing series data for configured reference series, trimmed by lookback window.

### Input

data in `pythia\data\reference\0\raw\0.h5`

### Output

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\trigger\0.h5`

data produced with command (PowerShell, from `pythia\pythia`; it runs the trigger task alone and processes series `0` alone, so the printout in `trigger_printout.txt` follows the execution sequence, and restores `workflows\extrema.yaml` afterwards):

```powershell
cd C:\Users\ziosa\Desktop\work\cld\pythia\pythia
$utf8 = New-Object Text.UTF8Encoding($false)
$cfg  = (Resolve-Path 'workflows\extrema.yaml').Path
$text = [IO.File]::ReadAllText($cfg)
$new  = $text -replace 'skip_series: \[\]', 'skip_series: ["1", "2", "3", "4", "5"]'
if ($new -eq $text) { throw 'skip_series: [] not found in extrema.yaml' }
try {
    [IO.File]::WriteAllText($cfg, $new, $utf8)
    $env:PYTHONUTF8 = '1'
    cmd /c ".venv\Scripts\python.exe -u pipeline.py --task trigger --strategy extrema --profile dev > trigger_printout.txt 2>&1"
} finally {
    [IO.File]::WriteAllText($cfg, $text, $utf8)
}
```

#### What you get

- `trigger_printout.txt`, the printouts of series `0` in execution sequence, the Ideal run, then the Realistic run (`trigger.replay.mode` `"dual"`):
  1. Ideal: `READING HDF5 FILE` → `TRIGGER - READ` → `TRIGGER - PREPROCESS` → `TRIGGER - OUTPUT` → `TRIGGER - OUTCOME`.
  2. Realistic: `READING HDF5 FILE` → `TRIGGER - READ` → `TRIGGER - PREPROCESS` → `TRIGGER - DETECTION` → `TRIGGER - OUTPUT` → `TRIGGER - OUTCOME`.
- Per mode, in `<mode>\trigger\`: `0.h5` holding `/series_0`, `/triggers_0` and `reference_series_0`, series `0` only, and `0_preprocess_report.txt`.
- Realistic mode only, with `trigger.strategy.plot_debug` set to `true` in `workflows\dev.yaml`: the debug plots in `realistic\trigger\plots\0\`, one figure per suppressed candidate (`<row>_s<n>_th.png`) and per ended or still-active trigger (`<row>_e<n>_th.png`), drawn as described under **Realistic Mode Trigger Begin**. Ideal mode has none: it detects in one pass over the series, with no window, detection lag or suppression reason for a figure to explain.
- In the strategy folder: the copies `extrema.yaml`, which records the `skip_series` of the run, `dev.yaml` and `git_info.txt`.

---
## Read

**Purpose**

Stream time series from an HDF5 file one series at a time, decode the datetime index, apply date filtering, and yield `(series_id, DataFrame)` tuples. Daily series and weekly series stamped on Saturdays (W-SAT) are the tested resolutions; every horizon parameter counts rows, so any other spacing of observed dates reads the same way, untested.

**Output**

With `trigger.read.printout_io` set to `true`: a file block (`READING HDF5 FILE: <path>`, the series found, the series to process without `read.skip_series`, their count and the date filter), then per series a `TRIGGER - READ` block with the series id, the shape before the date filter, the date range, the columns, and the first and last 10 rows, followed by the shape after the date filter.

**Logic**
- Read the HDF5 file; iterate through available series keys.
- Decode the datetime index (`epoch_seconds` or `epoch_days` encoding).
- Ignore every other dataset and subgroup stored for a series.
- Keep the value columns `var0`-`var5` and the gate columns `trigger.gate.items` names, in their stored sequence; drop any other stored column.
- Skip / filter out series according to **Data Conditions**.
- Yield `(series_id, DataFrame)` downstream.

**Data Conditions**
- Series in `skip_series` → skipped entirely.
- Date filter preserves all data if `ini_date` or `end_date` is `"none"`.
- Missing `index_encoding` attribute → `ValueError`.
- Unknown encoding format → `ValueError`.

**Parameters**

- Config-based: `read.raw_data`, `read.ini_date`, `read.end_date`, `read.skip_series`, `trigger.gate.items` (the columns its items name).

---

## Gate Pre-filter

**Purpose**

Check each series in the trigger task producer, after reading and before preprocessing:

1. **Missing gate column → the run fails.** A series without a column that `trigger.gate.items` names stops the run, whether `trigger.gate.prefilter` is on or off.
2. **Gate item never met → the series is skipped.** With `trigger.gate.prefilter` set to `true`, a series that is not a reference series is kept only if each gate item that applies to it (one whose `exempt_series` does not name it) is satisfied on at least one row of the read window (`read.ini_date` to `read.end_date`); the rows may differ between items, and a NaN counts as not satisfied. Otherwise the series is skipped.

**Output**

None but one INFO record per skipped series: `Skipping series <id>: gate item <column> <op> <threshold> passes on no row of the read window`.

**Logic**
1. Check that the series holds every column `trigger.gate.items` names, whether or not the pre-filter is on.
2. With `trigger.gate.prefilter` set to `true` and a series outside `read.reference_series`: test the items that apply to the series, those whose `exempt_series` does not name it, in their configured sequence on the date-filtered raw frame; the first item that does not hold on any row skips the series (no preprocessing, no trigger detection, no reference capture).

**Data Conditions**
- A configured gate column absent from a series → `ValueError` naming the column and the series; the run fails with one `Task failed` record, with the pre-filter on or off.
- One applicable item that does not hold on any row of the read window → series skipped; a series on which each item holds on some row, even never all on one row, is kept. A NaN passes no item.
- A reference series is never skipped.
- A skipped series writes no `/series_<id>`, so its dates leave the performance curve index: the switch can change the gradient curve's dates, never a trigger or a reference series.

**Parameters**
- Config-based: `trigger.gate.items` (its keys under **Trigger Detection**), `trigger.gate.prefilter`, `read.reference_series`, `general.enable_info`.

---

## Preprocess

**Purpose**

Clean the time series by removing rows with invalid data that would corrupt trigger detection.

**Output**

With `trigger.preprocess.printout_io` set to `true`, per series a `TRIGGER - PREPROCESS` block: the series id, the shape after cleaning, the issues found (per-series issues and consecutive blocks, or `No issues found - data is clean`), the date range, the columns and the first and last 10 preprocessed rows.

**Logic**
- Build a boolean mask initialized to `True` for all rows.
- Apply data cleaning according to conditions in **Data Conditions**
- Detect blocks of consecutive rows (≥ `trigger.consecutive_rows_with_bad_preprocessed_data`) invalid for one reason, so dates missing between two rows never break a block, and log them to a report file,
  flagging stretches of back-to-back bad data that may indicate a systematic data quality problem rather than isolated outliers.
- Return `df[mask]`.

**Data Conditions**
- Apply the NaN check, then each `trigger.validity` rule that lists a column, in this sequence and each over every row: a failing row gets `False` in the mask and a report line under the reason (`<columns>`: the rule's columns, comma-separated):
  1. **NaN check** (always) — rows with a NaN in `var0`-`var5`; reason `Contains NaN values`. A gate column outside `var0`-`var5`, NaN where a row holds no value, never removes a row.
  2. **`nonnegative`** — rows with a negative value in a listed column; reason `Fails nonnegative (<columns>)`.
  3. **`nonzero`** — rows with a zero in a listed column; reason `Fails nonzero (<columns>)`.
  4. **`not_all_equal`** — rows whose listed columns all hold one value; reason `Fails not_all_equal (<columns>)`. A NaN fails no rule.
- Empty DataFrame or all rows invalid → `df[mask]` returns empty DataFrame (no explicit guard; follows from mask logic).

**Parameters**
- Config-based: `trigger.consecutive_rows_with_bad_preprocessed_data`, `trigger.validity.nonnegative`, `trigger.validity.nonzero`, `trigger.validity.not_all_equal` (each a list of `var0`-`var5`, empty by default; `workflows/extrema.yaml` lists all six, `var5`, and `var0`-`var3`; a `not_all_equal` of one distinct column is refused at startup).

---
## Trigger Detection
**Purpose**

Detect trigger-begins via a given strategy and their trigger-ends according to a set of user-specified conditions. Consider active trigger-begins separately. Process the series incrementally (realistic mode) or in one pass (ideal mode).

**Output**

With `trigger.strategy.printout_debug` set to `true` (Realistic mode), per series a `TRIGGER - DETECTION` block: one row per marked trigger with the columns Type, Detection Begin Date, Trigger Begin Date, Detection End Date, Trigger End Date, Begin Value, End Value, Variation, Duration and End Reason (`ACTIVE` dates and the reason `active` for a trigger still active at the data end), then `Total triggers: N (Plus: p, Minus: m, Active: a)`, and the line `Series '<id>': Realistic mode trigger generation complete` with the count of each begin and end column.

With `trigger.strategy.printout_io` set to `true`, per series a `TRIGGER - OUTPUT` block: the shape after detection, the date range, the columns, the rows that hold a trigger flag or an end reason, and the first and last 10 rows.

**Decision Logic vs Recording**
- **`var3`** drives trigger detection, end conditions and breach checks, and the gate reads the gate columns of the previous row; neither reads `trigger_value`, so strategy behavior is independent of the recording configuration.
- **`trigger_value`** ONLY used to record/store values at trigger begin/end for output and performance measurement. Configurable via `trigger.trigger_value`.

   **Example**
   - End condition: "End if time series moves down 5%" → computed using `var3` at begin and current dates.
   - Output column: `trigger_start_values` → populated with `trigger_value` at trigger activation.

**Sub-tasks**
- Validate data length according to **Data Conditions**.
- Add the `trigger_value` column according to the configured mode (`detection_row` or `following_row`) — for recording only.
- Compute the begin vector of the gate on the previous row, once per series (see **Data Conditions**).
- Dispatch to realistic or ideal mode based on `trigger.replay.mode`.
- Active Flags: Mark `active_trigger_plus_begin` or `active_trigger_minus_begin` at latest begin index if that begin was removed during cleaning.

**Data Conditions**
- Series shorter than `trigger.min_length` → skipped.
- `lookback_window` < `min_length` → rejected at startup by configuration validation, naming both values; no series runs.
- Series shorter than `lookback_window` → skipped.
- A point that `find_peaks` reports on `−var3` is a minimum and gets a plus begin, and one it reports on `var3` is a maximum and gets a minus begin, plateaus included.
- The `find_peaks` prominence floor is `trigger.strategy.prominence_threshold` × the `var3` range of the data the mode sees: the window in Realistic mode, the whole series in Ideal mode.
- The effective start is `df.index[trigger.replay.lookback_window − 1]` of the date-filtered, preprocessed frame; the realistic loop, the extraction filter and the reference trimming use it, and the gradient curve's index starts at the earliest effective start over the processed series.
- The gate on the previous row: a begin on row `i` of the date-filtered, preprocessed frame is allowed only when row `i − 1` passes every item of `trigger.gate.items` that applies to the series, `value op threshold` in the column's dtype; a NaN fails, and row 0, which has no previous row, is never allowed. Ideal mode applies it to the extremum rows, Realistic mode to the activation row; both read one vector computed once per series, whatever `trigger.trigger_value` records.
- `trigger.strategy.window_size` names the output folder and feeds no detection logic.

**Parameters**
- Config-based: `trigger.min_length`, `trigger.replay.lookback_window`, `trigger.replay.mode` ("realistic"/"ideal"), `trigger.gate.items`, `trigger.strategy.prominence_threshold`, `trigger.strategy.window_size`, `trigger.trigger_value` ("detection_row"/"following_row").
- An item of `trigger.gate.items` holds `column` (a positional column name: `var` and digits), `op` (`">="` or `"<="`, quoted in YAML), `threshold` (a finite number) and `exempt_series` (the series ids, digit strings, the item does not apply to; default none); it applies to every other series. An empty list (the default) gates no begin.

### Realistic Mode Trigger Begin

**Logic**
- Discover starting trigger by running the strategy on a sample window.
- Iterates `i` from `lookback_window − 1` to `len(df) − 1`:
1. Window Selection (`_window_bounds`): the rows of the window ending at `i`:
   - Rolling: `i − lookback + 1` through `i`
   - Expanding: `0` through `i`
2. Trigger Generation: the strategy's `_begin_candidate_rows` takes the window's `var3` values and returns the rows `_generate_trigger_begin` marks (plus begins, minus begins, turning points); they give the candidate dates and `turning_points[i]`, so no window frame is built per row.
3. Candidate Detection: per type, the only candidate is the newest date of that type detected so far (`newest_candidates`), taken when the current window reports it and it is in neither `marked_triggers` nor `suppressed_triggers_cache`; select most recent date across both types.
4. Suppression Checks (candidate rejected if any is true, checked in this order):
   - Candidate date < `effective_start_date` (`df.index[lookback − 1]`) → `before_effective_start`.
   - Candidate date ≤ `most_recent_marked_date` → `temporal_constraint`.
   - Row `i − 1` fails an item of `trigger.gate.items` that applies to the series → `gate_blocked`; the candidate stays in `suppressed_triggers_cache`, so it never activates later and, an opposite candidate, never becomes `pending_opposite`.
   - Active trigger exists and is same type → `consecutive_same_type`.
5. Classification: without active trigger → `new_candidate`; with active trigger and opposite type → `pending_opposite`.
6. Activation: if no active trigger and `new_candidate` exists → mark at current iteration index, set `active_trigger`, update `most_recent_marked_date`.

**Output**

With `trigger.strategy.plot_debug` set to `true`, one figure per suppressed, ended or still-active trigger, saved under `plots/<series_id>/` beside the trigger output file: the upper panel draws the window's `var3` with the plus candidates as blue down-pointing and the minus candidates as red up-pointing markers, vertical lines at the detection and trigger dates, and a legend with those dates, their rows, the duration, the begin and end values, the variation and the suppression or end reason; the lower panel, `Zoom on Detection`, repeats the drawing around the detection region.

**Data Conditions**
- Multiple candidates in same iteration → most recent date selected.
- Active trigger state: `(signal_type, signal_date, trigger_begin_value, current_idx, begin_value, begin_row)` or `None`.
- Only the newest candidate of a type seen so far may activate, so an older extremum detected later is never acted on.
- `turning_points[i]` holds the number of turning points in the window ending at row `i`.
- The window frame (`_select_window`, then `_generate_trigger_begin`) is built only when `trigger.strategy.plot_debug` or `trigger.replay.end.different_end_strategy` is `true`: the debug plots and that end mode read it.

**Parameters**
- Config-based: `trigger.replay.lookback_window`, `trigger.replay.window_type` ("rolling"/"expanding"), `trigger.min_length`, `trigger.gate.items`, `trigger.trigger_value` ("detection_row"/"following_row"), `trigger.strategy.plot_debug`, `trigger.replay.end.different_end_strategy`, strategy-specific parameters.
- Hardcoded: Fixed checking sequence for suppression conditions (before effective start → temporal → gate → consecutive same-type).

### Realistic Mode Trigger End

**Logic**
Only when an active trigger exists:

- Minimum duration: no end if `duration < min_duration_periods`. A pending opposite is suppressed within the minimum duration.
- End modes checked in a fixed sequence (first match wins):

| Sequence | Mode | Condition |
|---|---|---|
| 1 | `against_threshold` | Move against the trigger direction ≥ `against_threshold_pct` |
| 2 | `along_threshold` | Move along the trigger direction ≥ `along_threshold_pct` |
| 3 | `opposite_trigger_as_begin_strategy` | Pending opposite exists → end & swap |
| 4 | `different_end_strategy` | Strategy-specific end signal after begin date |
| 5 | `max_duration` | `duration ≥ max_duration_periods` |

- Moves against and along the trigger direction computed from `var3` at begin and current dates (NOT trigger_value).
- On end: mark `trigger_plus_end` or `trigger_minus_end` at current index; set `end_reason`.
- If no end and `pending_opposite` exists → suppress the pending opposite.

**Data Conditions**
- Opposite trigger swap: `active_trigger` remains set (not cleared).
- An opposite extremum seen within the minimum duration stays suppressed and never activates later.
- A gate-blocked opposite candidate never becomes `pending_opposite`, so under `opposite_trigger_as_begin_strategy` it ends no active trigger, which runs on to the next allowed opposite extremum or another end mode; `different_end_strategy` reads the window's own extrema, which the gate does not filter, so a blocked extremum can still end the active trigger there.
- Active triggers at series end: marked at begin date index, not final date.
- **Critical** All end decisions use `var3`; `trigger_value` is for recording only.

**Parameters**
- Config-based: `trigger.replay.min_duration_periods`, `trigger.replay.end.max_duration_periods` (null to disable), `trigger.replay.end.against_threshold_pct` (null to disable), `trigger.replay.end.along_threshold_pct` (null to disable), `trigger.replay.end.opposite_trigger_as_begin_strategy` (true/false), `trigger.replay.end.different_end_strategy` (true/false).
- Hardcoded: Fixed checking sequence for end modes (against_threshold → along_threshold → opposite_trigger_as_begin_strategy → different_end_strategy → max_duration).

### Realistic Mode — flowchart
The realistic loop's flowchart, Mermaid source [`mmflowcharts/trigger_detection_realistic_mode_flowchart.mmd`](mmflowcharts/trigger_detection_realistic_mode_flowchart.mmd).

### Ideal Mode
**Logic**
- Trigger Begin: Generate begin triggers using full series visibility via `_generate_trigger_begin(df)`.
- Gate Suppression: Zero-out the begin flags of every row whose previous row fails an item of `trigger.gate.items` that applies to the series, row 0 included (fully vectorized).
- Trigger End: Generate end triggers using full series visibility via `_generate_trigger_end(df)`: the ends are the opposite begins the gate left, so a blocked begin also removes the opposite trigger's end on its row.
- Duplicate Cleaning: Apply rising-edge pass to remove duplicate triggers; pair begins with ends per signal type, dropping unmatched items.
- Overlap Removal: Remove overlapping intervals across plus/minus pairs (chronological order, first pair wins).

**Data Conditions**
- All triggers detected with full series visibility (future rows visible).
- Duplicate triggers removed via rising-edge detection (0→1 transitions).
- Per signal type, each begin is paired with the first end after it; an end with no unmatched begin before it and a final begin with no end after it are dropped.
- Overlapping intervals across signal types → later interval removed.
- Active trigger flags: marked when pre-cleaned latest begin ≠ post-cleaned latest begin.

**Parameters**
- Config-based: `trigger.gate.items`, `trigger.trigger_value` ("detection_row"/"following_row"), strategy-specific parameters.
- Hardcoded: Rising-edge algorithm for duplicate detection, chronological sorting for overlap removal, `end_reason = "ideal_mode"` for every end row of a final pair and an empty `end_reason` on every other row.

---

## Triggers Extraction

**Purpose**

Extract triggers from trigger columns by pairing begin/end dates per direction (plus/minus), reading precomputed `trigger_value` at those dates, separating each trigger by direction and outcome, and filtering out triggers that begin before the **effective start** date.

**Output**

With `trigger.strategy.printout_io` set to `true`, per series with triggers a `TRIGGER - OUTCOME` block: the shape, the date range from the first begin to the last end, the columns `series_id`, `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger` and `is_true`, and the first and last 10 rows, or every row with `trigger.strategy.full_output` set to `true`.

**Logic**
1. For each direction (plus, minus), collect begin and end date indices from the trigger columns (`trigger_plus_begin`, `trigger_plus_end`, `trigger_minus_begin`, `trigger_minus_end`).
2. Pair each begin with the first strictly subsequent end in the same direction using vectorized `searchsorted` with a cumulative-maximum trick to enforce 1:1 pairing.
3. Combine plus and minus triggers into a single list.
4. Read `trigger_value` at the begin and end row numbers (single source of truth from categorization).
5. Distinguish each trigger:
   - **Plus true** → `trigger_end_values > trigger_start_values`
   - **Minus true** → `trigger_end_values < trigger_start_values`
6. Filter out triggers whose `trigger_begin_dates` < **effective start** (`df.index[trigger.replay.lookback_window − 1]` of the preprocessed frame). This is specifically needed for ideal mode, which detects triggers on the full series with no burn-in awareness. Realistic mode already enforces this via suppression (`before_effective_start`), so the filter is a no-op for that mode.
7. Persist as `/triggers_{series_id}` in the same HDF5 store with columns: `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger`, `is_true`.

**Data Conditions**
- No begins or ends found → returns `None`, series skipped.
- Begins with no subsequent end → dropped.
- Triggers still active at the data end are not extracted, so they are never grouped.
- Ends before any begin → ignored.
- Begin and end at the same timestamp → do NOT pair.
- Triggers with `trigger_begin_dates` before effective start → filtered out.
- All triggers filtered → returns `None`, series skipped.

**Parameters**
- Config-based: `trigger.replay.lookback_window` (used to compute effective start for filtering).

---

## Reference Handling

**Purpose**

Optionally save post-read, pre-preprocessing series data alongside trigger output for configured reference series, enabling fair comparison between the unprocessed and processed results.

**Logic**
1. Load the set of reference series IDs from `read.reference_series`.
2. For each series in that set, capture the DataFrame after reading and the gate column check (the gate pre-filter never skips a reference series) but before preprocessing (NaN cleaning, negative value removal, constant-row suppression have not yet been applied).
3. Trim to the raw rows from the effective start date on, so that the reference starts on the first date on which the strategy loop can emit a trigger.
4. Persist as `reference_series_{series_id}` in the same HDF5 store, independently of whether trigger detection produced any output for that series.

**Data Conditions**
- No reference series configured → reference saving disabled entirely.
- The gate pre-filter never skips a reference series; a reference series without a configured gate column fails the run before its capture.
- A series whose preprocessed frame has fewer than `trigger.replay.lookback_window` rows → no trimming.

**Parameters**
- Config-based: `read.reference_series`, `trigger.replay.lookback_window`.
