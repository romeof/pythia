# Signal

## Brief Summary

The **Signal** step groups triggers produced by the Trigger task into signal-window groups based on time overlap and correlation, then selects a representative signal per group. It reads from the trigger HDF5 file and outputs a new signal HDF5 file containing the groups table, signal series DataFrames (only for `is_signal=True` series), reference series (pass-through), and strategy name.

**Sub-tasks**
1. **Accumulation** — Collect per-series value columns (`var3`, `trigger_value`) and triggers into a single collector via producer-consumer streaming.
2. **Selection** — Assign group IDs based on time overlap, correlation threshold, and signal_criterion; mark signal rows dynamically.
3. **Persistence** — Write groups table, signal series, and reference series to a new HDF5 file.

### Input

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\trigger\0.h5`

### Output

data produced with command `python pipeline.py --task signal --strategy extrema --profile dev`

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\signal\0.h5`

---

## Accumulation

**Purpose**

Stream series and triggers from the trigger HDF5, storing value columns for slope/correlation computation and accumulating per-series triggers for grouping.

**Output**

With `signal.input.printout_io` set to `true`, a `SIGNAL - ACCUMULATED TRIGGERED SERIES` block: the shape, the columns `series_id`, `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger` and `is_true` of the accumulated triggers, and the first and last 10 rows, or every row with `signal.input.full_output` set to `true`.

**Logic**
1. Read the trigger HDF5 file and stream each `(series_id, series_df, triggers_df)` triple via the producer-consumer pattern.
2. For each series:
   - Store `var3` and `trigger_value` columns indexed by date (for slope computation and correlation).
   - Accumulate the triggers DataFrame with base columns (`series_id`, `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger`, `is_true`).
3. Series without triggers are still stored (values may be needed for cross-series correlation).

**Data Conditions**
- Missing or empty series DataFrame → series skipped.
- Non-DatetimeIndex → automatic conversion attempted; a failed conversion raises.
- Missing `var3` column → series skipped.

**Parameters**
- Config-based: `signal.input.printout_io`.

---

## Selection

**Purpose**

Assign each trigger to a group based on time overlap, correlation threshold, and signal_criterion, then mark one trigger per group as the signal.

**Output**

With `signal.selection.printout_io` set to `true`, a `SIGNAL - SELECTION` block: the groups table's shape, its columns (the trigger columns above, then `group_id`, `is_signal`, `slope`, `corr_to_signal`, `series_ids` and `num_series`) and the first and last 10 rows, or every row with `signal.selection.full_output` set to `true`.

**Logic**
1. Concatenate all accumulated triggers, normalize date columns, cast `series_id` to string, sort by `trigger_begin_dates`.
2. Trim each series to its rows from `period_back` rows before its row at or before `earliest_begin` through `latest_end`; the look-back counts rows, as the slope and correlation windows do, so gaps in the dates never shorten a window, and pre-compute two lookup structures: a map from `(series_id, begin_date)` to the corresponding positional index range in the series array, and a list of absolute slopes — one per trigger. Both are derived once before the assignment loop, keeping per-trigger work to O(1) lookups.
3. For each trigger in order:
   - Retrieve the pre-computed absolute slope.
   - Expire active signals whose `signal_end ≤ begin`.
   - Find eligible signals where `signal_begin ≤ begin < signal_end`.
   - Take the eligible signals by ascending `signal_end` (Ideal) or by (`signal_begin`, `group_id`) (Realistic); for each, look up the cached Pearson correlation for this (signal, trigger, begin) pair, or compute it fresh over the lookback window using the pre-computed index bounds and cache the result. Assign to the **first** eligible signal passing `corr_threshold` and stop.
   - If no eligible signal passes → create a new group (current trigger becomes its signal).
   - **Dynamic signal update** according to **Signal Criterion**: if `signal_criterion=earliest_begin_highest_slope` and the joining trigger shares the same begin date with a higher slope, it replaces the current signal.
4. Attach group aggregates: `num_series`, `series_ids` (sorted by descending slope).

**Signal Criterion**

| Criterion | Selection |
|---|---|
| `earliest_begin` | Earliest `trigger_begin_dates` in the group (first trigger, no update) |
| `earliest_begin_highest_slope` | Among triggers at the earliest begin date, highest absolute slope wins (dynamic update during assignment) |

### Example Walkthrough

Consider triggers processed in temporal sequence with `corr_threshold=0.8` and `signal_criterion=earliest_begin_highest_slope` (begin inclusive, end exclusive).

**Triggers:**

```
idx | sid | begin  | end    | slope
----+-----+--------+--------+------
 0  |  A  | 01-01  | 01-07  | 0.40
 1  |  B  | 01-02  | 01-10  | 0.50
 2  |  C  | 01-03  | 01-05  | 0.35
 3  |  D  | 01-07  | 01-12  | 0.60
 4  |  E  | 01-08  | 01-09  | 0.45
 5  |  F  | 01-09  | 01-09  | 0.55
 6  |  G  | 01-10  | 01-15  | 0.70
 7  |  H  | 01-12  | 01-16  | 0.30
 8  |  I  | 01-14  | 01-20  | 0.65
```

**Correlation assumptions** (absolute value):

```
r(A,B)=None   r(A,C)=0.75   r(B,C)=0.85   r(B,D)=None
r(B,E)=0.90   r(B,F)=0.88   r(D,G)=0.82   r(G,H)=0.60   r(H,I)=0.91
```

**Time axis** (`|` = begin inclusive, `]` = end exclusive, `*` = zero-width trigger):

```
        01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
idx 0   |=================]                                          gid=1 — active={} → new signal S1=[01-01,01-07)
idx 1      |=======================]                                 gid=2 — active={S1}; eligible={S1}; r(A,B)=None → new signal S2=[01-02,01-10)
idx 2         |=====]                                                gid=2 — active={S1,S2}; eligible={S1,S2}; r(A,C)=0.75✗ → r(B,C)=0.85✓ → joins gid=2
idx 3                     |==============]                           gid=3 — expire S1 (01-07≤01-07); active={S2}; eligible={S2}; r(B,D)=None → new S3=[01-07,01-12)
idx 4                        |==]                                    gid=2 — active={S2,S3}; eligible={S2,S3}; r(B,E)=0.90✓ → joins gid=2 (first eligible passing)
idx 5                           *                                    gid=2 — active={S2,S3}; eligible={S2,S3}; r(B,F)=0.88✓ → joins gid=2; begin≠signal_begin → no competition
idx 6                              |==============]                  gid=3 — expire S2 (01-10≤01-10); active={S3}; eligible={S3} (01-07≤01-10<01-12); r(D,G)=0.82✓ → joins gid=3
idx 7                                    |===========]               gid=4 — expire S3 (01-12≤01-12); active={} → new signal S4=[01-12,01-16)
idx 8                                          |=================]   gid=4 — active={S4}; eligible={S4} (01-12≤01-14<01-16); r(H,I)=0.91✓ → joins gid=4
```

**Final Groups** (signals marked with `*`):

```
gid=1: A*
gid=2: B*, C, E, F
gid=3: D*, G
gid=4: H*, I
```

**Final result table:**

```
idx | sid | begin  | end    | gid | is_signal | corr_to_signal | note
----+-----+--------+--------+-----+-----------+----------------+---------------------------------------------
 0  |  A  | 01-01  | 01-07  |  1  |   True    |      1.00      | active={} → new signal S1
 1  |  B  | 01-02  | 01-10  |  2  |   True    |      1.00      | eligible={S1}; r(A,B)=None → new signal S2
 2  |  C  | 01-03  | 01-05  |  2  |   False   |      0.85      | eligible={S1,S2}; r(A,C)=0.75✗; r(B,C)=0.85✓ → joins gid=2
 3  |  D  | 01-07  | 01-12  |  3  |   True    |      1.00      | S1 expired at boundary; eligible={S2}; r(B,D)=None → new signal S3
 4  |  E  | 01-08  | 01-09  |  2  |   False   |      0.90      | eligible={S2,S3}; S2 checked first; r(B,E)=0.90✓ → joins gid=2
 5  |  F  | 01-09  | 01-09  |  2  |   False   |      0.88      | eligible={S2,S3}; r(B,F)=0.88✓; begin≠signal_begin → no signal competition
 6  |  G  | 01-10  | 01-15  |  3  |   False   |      0.82      | S2 expired at boundary; eligible={S3} (01-07≤01-10<01-12); r(D,G)=0.82✓ → joins gid=3
 7  |  H  | 01-12  | 01-16  |  4  |   True    |      1.00      | S3 expired at boundary; active={} → new signal S4
 8  |  I  | 01-14  | 01-20  |  4  |   False   |      0.91      | eligible={S4} (01-12≤01-14<01-16); r(H,I)=0.91✓ → joins gid=4
```

**Cases covered:**

| Case | Where |
|---|---|
| No active signals → automatic new signal | idx=0, 7 |
| r=None → cannot join → new signal | idx=1, 3 |
| Multiple eligible signals; first fails threshold, second passes | idx=2 (S1 fails, S2 passes) |
| Signal expires exactly at begin (end-exclusive boundary) | idx=3 (S1.end=D.begin=01-07), idx=6 (S2.end=G.begin=01-10), idx=7 (S3.end=H.begin=01-12) |
| Multiple eligible signals; first passes → break (second never evaluated) | idx=4 (S2 passes, S3 skipped) |
| begin≠signal_begin → `_verify_signal_update` returns immediately, no signal competition | idx=5, 6, 8 |
| All signals expired simultaneously → new signal | idx=7 |

### Dynamic Signal Update

For `signal_criterion=earliest_begin_highest_slope`, the signal within a group can be replaced during assignment — but only by a trigger that shares the **exact same begin date** as the current signal and has a higher slope.

```
        01 02 03 04 05 06 07 08 09 10
idx 0   |=================]             gid=1 — new signal S1 (slope=0.40, signal=A)
idx 1   |==========================]    gid=1 — begin(01-01)=signal_begin(01-01); r(A,B)=0.90✓; slope 0.50>0.40 → B replaces A as signal
idx 2      |=========]                  gid=1 — r(B,C)=0.85✓ → joins gid=1; begin(01-02)≠signal_begin(01-01) → no competition
```

After B joins at idx=1, the signal of gid=1 changes from A to B. Because the signal changed, gid=1 is flagged in `signal_updated_groups`. After the main loop, `_recompute_correlations_for_updated_signals` rewrites `corr_to_signal` for all members of gid=1 using B as the new reference:

```
idx | sid | corr_to_signal (before recompute) | corr_to_signal (after recompute)
----+-----+-----------------------------------+----------------------------------
 0  |  A  |   1.00 (was signal)               |   r(B,A) recomputed vs new signal B
 1  |  B  |   r(A,B)=0.90 (joined)            |   1.00 (B is now signal → itself)
 2  |  C  |   r(B,C)=0.85                     |   r(B,C)=0.85 (unchanged, same signal)
```

### Edge Case: Multiple Eligible Signals

A trigger may satisfy the correlation threshold with respect to more than one active signal simultaneously. Consider a variant of the main walkthrough where:

```
idx 2 (C, begin=01-03, end=01-05): r(A,C)=0.95≥0.8  and  r(B,C)=0.85≥0.8
```

Both correlations clear the threshold, yet C must belong to exactly one group. The implementation resolves the ambiguity by checking eligible signals by ascending `signal_end` (Ideal) or by (`signal_begin`, `group_id`) (Realistic) and assigning to the **first** one that passes, then breaking immediately. Both modes check S1 (`signal_begin`=01-01, `signal_end`=01-07) before S2 (`signal_begin`=01-02, `signal_end`=01-10), so C joins gid=1 and S2 is never evaluated. Note other choices may be possible, e.g. to assign C to the group where it has the strongest correlation with the current seed. 

**Why this situation is rare**

A second signal is created precisely because a trigger inside an existing signal's window *failed* the correlation test against that signal — implying the two signal series were not strongly correlated over the relevant lookback window. For a later trigger C to pass the threshold against *both* signals, the two signals would have to be fairly similar to each other over C's window, which structurally contradicts the earlier failure.

**When it may nevertheless occur**

- **Nonstationarity** — correlations shift over time; the pair (A, B) may have been weakly correlated when B became a signal but strongly co-move by the time C's lookback window is evaluated.
- **Shared drivers** — an external factor causes A and B to co-move specifically during C's window, independently of their earlier relationship.
- **Parameter sensitivity** — short or noisy lookback windows, or a small `corr_threshold`, increase the probability that both correlations exceed the threshold.

**Design rationale**

The current implementation assigns each trigger to exactly one group. Allowing a trigger to belong to multiple groups would double-count its contribution to the total variation curve — a distortion that outweighs the potential benefit. In principle, membership in two groups could offer more opportunities to select a better-performing event; in practice, the case is sufficiently rare and the double counting it would cause sufficiently material that the single-assignment rule is retained.

**Data Conditions**
- No triggers accumulated → returns empty DataFrame.
- Slope computation requires ≥ 2 finite values in the lookback window.
- Correlation requires ≥ 2 overlapping finite values.
- Series without triggers → values still stored (needed for cross-series correlation); only trigger accumulation is skipped.
- The correlation is Pearson's r on `var3` levels over the `signal.period_back` + 1 points that end at the trigger begin, and a trigger joins a signal when `abs(r)` ≥ `signal.corr_threshold`.
- Ideal: eligible signals with equal `signal_end` are taken newest first.
- Realistic: eligible signals are taken by (`signal_begin`, `group_id`), never by `signal_end`, which is not known on the trigger's begin date.

**Parameters**
- Config-based: `signal.corr_threshold`, `signal.period_back`, `signal.signal_criterion`, `signal.selection.printout_io`, `trigger.replay.mode` ("realistic"/"ideal", which selects how eligible signals are taken).

---

## Persistence

**Purpose**

Write the signal HDF5 file containing the groups table, signal series, reference series, and strategy name.

**Logic**
1. Write the signals table (`/signals`) with all group assignments and aggregates.
2. Copy signal series DataFrames from the trigger HDF5 — only for series IDs that appear in `is_signal=True` rows.
3. Copy all reference series from the trigger HDF5 (pass-through for gradient computation).
4. Write strategy name metadata.

**Output HDF5 Structure**
- `/signals` — signals table with columns: `series_id`, `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger`, `is_true`, `group_id`, `is_signal`, `slope`, `corr_to_signal`, `series_ids`, `num_series`.
- `/series_{id}` — the trigger task's processed frame with its 15 columns (`var0` to `var5`, `trigger_value`, `end_reason`, the four begin and end flags, `turning_points` and the two active flags) plus any gate column outside them, only for signal series.
- `/reference_series_{id}` — reference series (pass-through from trigger HDF5).
- `/strategy_name` — strategy name metadata.

