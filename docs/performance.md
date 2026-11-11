# Performance

## Brief Summary

The **Performance** task computes per-series metrics for triggers and signals, optionally prints/plots per-series outputs, computes one aggregate gradient curve from signals, and persists the final results to performance HDF5.

It uses:
- The trigger HDF5 for per-series trigger processing.
- The signal HDF5 for signal metadata, per-series signal processing, reference loading, and gradient computation.

**Sub-tasks**
1. **Per-series analysis** — Stream trigger/signal series, compute metrics, and optionally print/plot per series.
2. **Gradient analysis** — Build filtered signal segments, aggregate total variation (pure and attenuation-adjusted in a single pass), apply optional per-event deduction, print/plot gradient outputs.
3. **Gradient metrics** — Compute aggregate-level metrics (variation rate, dispersion ratios, fall from the running maximum, reference-relative coefficients) on the cumulative variation series, separately for pre-attenuation and post-attenuation curves.
4. **Persistence** — Save enriched `/signals`, optional `/strategy_name`, and gradient plot reconstruction data (see HDF5 schema below).

### Input

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\trigger\0.h5` and `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\signal\0.h5`

### Output

data produced with command `python pipeline.py --task performance --strategy extrema --profile dev`

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\performance\0.h5`

---

## Metric Calculation

Compute per-series metrics (variation, duration, trials-to-true) and direction-based aggregations for trigger and signal rows. See [metrics.md](metrics.md) for full details on metric definitions, aggregation operations, and summary table formatting.

---

### Schema Normalization

**Purpose**

Normalize the rows DataFrame schema for consistent downstream storage and output.

**Logic**
1. Ensure `series_id` is string.
2. Cast boolean columns (`is_true`, `is_plus_trigger`) to `bool`.
3. Cast metric float columns (dynamically loaded from `metrics.json`) to `float32`.

---

## Per-series analysis

**Purpose**

Run per-series processing for triggers and signals via producer-consumer streams, compute metrics, and optionally print/plot each series.

**Output**

With `performance.perseries_trigger.printout_io` set to `true`, per series a `PERFORMANCE - TRIGGER SERIES METRICS` block: the series id, the shape, the columns `series_id`, `trigger_begin_dates`, `trigger_end_dates`, `trigger_start_values`, `trigger_end_values`, `is_plus_trigger`, `is_true` and then the metric columns in the sequence of `performance.metrics.display_order_signals`, and the first and last 10 rows, or every row with `performance.perseries_trigger.full_output` set to `true`; the summary table of metrics.md follows.

With `performance.perseries_trigger.plot_debug` set to `true`, one figure per series: the upper panel draws `var3` in black, each trigger's segment coloured by direction (blue plus, red minus) and styled by outcome (solid true, dotted false) and annotated with its number and variation, stars at the begins still active at the data end, and the legend Plus True, Plus False, Minus True, Minus False; the lower panel holds the summary table.

With `performance.perseries_signal.printout_io` set to `true`, per series a `PERFORMANCE - SIGNAL SERIES METRICS` block of the series' signal rows, laid out as the trigger block above (every row with `performance.perseries_signal.full_output` set to `true`), followed by its summary table.

With `performance.perseries_signal.plot_debug` set to `true`, one figure per series with signals, drawn as the trigger figure above from the series' signal rows.

**Logic**
1. Trigger per-series processing runs only when `performance.perseries_trigger.printout_io` or `performance.perseries_trigger.plot_debug` is `true`.
2. Signal per-series processing always runs because signal enrichment and series accumulation are required for gradient computation.
3. Trigger phase:
   - Stream trigger HDF5 via `stream_trigger_series` as `(series_id, series_df, rows_df)`.
   - For each series, call `process_perseries` to compute metrics, normalize, and optionally print/plot.
4. Signal phase:
   - Batch-load `/signals` and reference series from the signal HDF5.
   - Build a `series_id -> rows` lookup from `/signals`.
   - Stream series from signal HDF5 via `iter_series`.
   - For each series: store full series for gradient, enrich matching signal rows with metrics, and optionally run per-series print/plot on `is_signal=true` rows.
5. Reassemble enriched signals:
   - Concatenate enriched parts.
   - Keep original rows for series that were not present in streamed series.
   - Sort by `[group_id, trigger_begin_dates, series_id]`.

**Data Conditions**
- Empty `rows_df` in per-series processing → series returns success without print/plot.
- Missing or empty `/signals` table in signal HDF5 → warning and early return with empty outputs.
- Missing `var3` in reference DataFrame → reference skipped.

**Parameters**
- Config-based:
  - Trigger phase: `performance.perseries_trigger.printout_io`, `performance.perseries_trigger.plot_debug`, `performance.perseries_trigger.full_output`
  - Signal phase: `performance.perseries_signal.printout_io`, `performance.perseries_signal.plot_debug`, `performance.perseries_signal.full_output`
  - Shared display order: `performance.metrics.display_order_signals`

---

## Signal analysis

**Purpose**

Use the accumulated signal data to compute and visualize the strategy's total variation via its gradient.

---

## Gradient

**Purpose**

Compute one cumulative variation curve from all valid signal segments, enforce weight constraints, optionally apply yearly attenuation and per-event deduction, and generate gradient output. When `performance.gradient.yearly_attenuation_fraction > 0`, both the pure (pre-attenuation) and attenuation-adjusted curves are produced in a single conversion pass — the per-signal pct→per-period rate conversion is shared between the two compounding runs and the second run is skipped entirely when attenuation is disabled.

**Output**

With `performance.gradient.plot_debug` set to `true`, the figure `gradient.png`: the upper panel (`gradient`) draws the cumulative variation curve under the strategy name, the curve after attenuation when `performance.gradient.yearly_attenuation_fraction` is above 0 and one curve per reference series (`reference <id>`), its legend at the upper left; the lower panel (`signal variation (%)`) draws each kept signal's variation segment, coloured by direction and styled by outcome, over a dashed zero line, with the legend Plus True, Plus False, Minus True, Minus False; the summary table of signal and gradient metrics sits below the panels.

### Gradient Input Signals Printout

When `performance.gradient.printout_io` is `true` and `signals_df` is non-empty, input signal rows (`is_signal=true`) are printed by direction (`plus+minus`, `plus`, `minus`). Trials-to-true is recomputed per direction subset so wait times reflect only that subset.

**Output**

With `performance.gradient.printout_io` set to `true`, one `PERFORMANCE - GRADIENT INPUT SIGNALS - <DIRECTION>` block per direction (`PLUS+MINUS`, `PLUS`, `MINUS`) with the input signal rows of that direction, every row with `performance.gradient.full_output` set to `true`.

### Signal Variation Computation

This step computes the values for the middle panel of the full gradient plot.

For each signal row (`is_signal=true`) in `signals_df`, sorted by `[trigger_begin_dates, group_id]`:

1. Look up the series DataFrame from the accumulated `signal_series` dict.
2. Extract the `trigger_value` segment from `[begin, end]` (single source of truth from the precomputed `trigger_value` column).
3. Compute percent-change from the trigger begin value: `pct = variation_pct(values, start_val)` (canonical function from `common.py`).
4. Reindex to the global observed-date index (the curve index) within `[begin, end]` with forward-fill to preserve the trigger timeline.
5. Force signal day (begin) to 0% as baseline — must be after reindex to survive alignment.
6. Determine sign multiplier: `+1` for plus triggers, `−1` for minus triggers.

**Data Conditions**
- Missing series DataFrame or empty segment → signal skipped (returns `None`).
- Missing `trigger_value` column → signal skipped.
- Empty segment after reindex → signal skipped.
- Signal-day percent change is forced to `0%` regardless of the underlying value.

### Weight Constraint Validation

Before aggregation, signals are filtered so the concurrent configured allocation does not exceed `performance.gradient.concurrent_weight_limit`.

**Algorithm**: A sweep-line approach is used — O(S·log S + E) where S = number of signals and E = number of begin/end events — instead of the naive O(D·S) approach.

Each signal generates two set points: a "begin" event at begin and an "end" event at end. Events are sorted and swept left-to-right to track active counts without rescanning all signals per date. On the same date, end events come before begin events, except a zero-length signal's own end, which comes after that day's begin events, so the signal counts on its day.

If `concurrent_count × weight > concurrent_weight_limit`, remove the newest active violator (highest insertion order).

This pre-filter uses the configured weight `w` and is mode-agnostic: exact for multiplicative_fixed (each event allocates `w`); conservative lower-bound for multiplicative_total_with_active_signals/multiplicative_total_before_active_signals (each event allocates `w × V(start_i) ≥ w` when the total value has grown). Computing the exact multiplicative concurrent allocation here would require the full path-dependent compounding, creating a circular dependency — so the configured-weight check is the right approximation.

### Per-Signal Variation Conversion

Each signal's percent-from-start series is converted to signed per-period variation rates:
1. Cumulative definition (from the event begin value): `pct_t = (v_t / v_i - 1) * 100`, where `v_i` is `start_val`.
2. Per-period recovery from cumulative values:
   - `r_t = (1 + pct_t/100) / (1 + pct_{t-1}/100) − 1`.
   - Since `1 + pct_t/100 = v_t / v_i`, the initial value cancels and:
   - `r_t = (v_t / v_i) / (v_{t-1} / v_i) - 1 = v_t / v_{t-1} - 1`.
   - If raw values were used directly at this stage, `r_t = v_t / v_{t-1} - 1` (or the ratio `v_t / v_{t-1}` instead of `1 + r_t`) is algebraically equivalent.
   - Since `pct` is already computed for the gradient plot middle panel, per-period rates are derived directly from `pct_t` here.
3. Reindex each per-period series to the full global observed-date x-axis (the curve index) and assign `0` to non-active dates (outside the valid signal interval), so compounding uses identity `(1 + 0) = 1`.
4. Apply signal direction: `sign=+1` keeps as-is; `sign=−1` negates (`is_minus` contributes positively on negative moves, and negatively on positive moves). Simple negation `−r` is used rather than inverse variation `−r/(1+r)` because, for example, for `−50%`, inverse variation gives `+100%`, which is not the intended behavior.
5. Build each event cumulative path using geometric compounding: `event_cum = cumprod(1 + r_t) - 1`. Weighting is applied after this per-event compounding, according to the selected variation method.

The converted series list is sorted by event start date (ascending), as required by multiplicative aggregation.

### Variation Aggregation Methods

`performance.gradient.variation_method` controls how per-signal variations are aggregated. All modes use geometric per-event compounding (`cumprod(1 + r_i) - 1`) for each event path, then combine those per-event cumulative curves with method-specific weights; they differ only in how each event allocation is set.

#### Multiplicative Fixed (`multiplicative_fixed`)

Each event is allocated `w` (a fixed fraction of baseline `1`). The allocation is locked at event start (no later reallocation). Each event cumulative curve is:

```
event_cum_i(t) = w × (cumprod(1 + r_i)_t − 1)
```

The total variation is the sum of all events' independent cumulative curves:

```
total(t) = Σ event_cum_i(t)
```

`cumsum(w × r_i)` is incorrect because it adds period rates linearly and ignores geometric compounding. Example: `+10%` then `-10%` is `-1%` geometrically (`1.10 × 0.90 = 0.99`) but `0%` with linear cumsum.

On non-active days, variation = 0, so `(1 + 0) = 1` preserves the last active value via cumprod.

#### Multiplicative Total with Active Signals (`multiplicative_total_with_active_signals`)

Each event `i` receives a locked allocation:

```
locked_weight_i = w × V(start_i)
```

where `V(start_i)` includes contributions from all previously processed events as observed at `start_i`, including events still active at `start_i`:

```
V(start_i) = 1 + total_cum(start_i)
```

This means the allocation grows with the total cumulative value, including the increase from still-active events.

#### Multiplicative Total before Active Signals (`multiplicative_total_before_active_signals`)

Same locked allocation formula, but `V(start_i)` includes only previously processed events already ended before `start_i` (`end_date < start_i`). Events still active at `start_i` are excluded:

```
V(start_i) = 1 + Σ (locked_weight_j × event_cum_j(eval_pos))   for j where end_j < start_i
```

After `end_date`, per-period variation is `0` (after global reindex/fill), so `event_cum` stays flat. Reading at `eval_pos` gives the same value as the event's final active value.

#### Comparison

The locked weight differs because `_total_with_active_signals` includes the increase from still-active prior events at `start_i`, while `_total_before_active_signals` includes only the increase from ended prior events.

In both modes, per-event compounding remains geometric and each event is scaled once by `locked_weight_i` (no later reallocation). The modes only govern the base value used to compute the initial allocation.

#### Design Note: Discrete vs Continuous Compounding

All compounding in this module uses **discrete (arithmetic) variations** — not continuous (log-space) compounding.

**Single-series equivalence.** For one series, the two forms are identical in exact arithmetic:

```
exp(Σ log(1 + r_t)) ≡ Π(1 + r_t)
```

**Multi-event aggregation breaks in log-space.** The total variation is built by a linear sum in level-space:

```
total(t) = Σ locked_weight_i × event_cum_i(t)
```

The fundamental incompatibility is the summation across events: the exponential is not linear over addition — `exp(a + b) ≠ exp(a) + exp(b)`. So the moment two or more log-variation terms are summed and then exponentiated, the result diverges from summing the discrete terms directly. This holds **regardless of the weight value** — even with `w = 1` the mismatch appears as soon as multiple events are combined:

```
log(1 + R_total) ≠ Σ w_i × log(1 + r_i)       (for any w, when N > 1)
```

A secondary, smaller effect: with a single event, `w × log(1+r) ≠ log(1 + w×r)` when `w ≠ 1`. But this is minor compared to the multi-event structural issue.

**Numerical example** — two events, `w = 1.0`, after one period:
- Event A: `+20%` (`r_A = 0.20`), Event B: `−10%` (`r_B = −0.10`)
- Discrete: `R = 1.0 × 0.20 + 1.0 × (−0.10) = +10.00%` ✓
- Log-space: `log_R = 1.0 × log(1.20) + 1.0 × log(0.90) = 0.07696` → `exp(0.07696) − 1 = +8.00%` ✗

The ~2 pp gap grows with variation magnitude and number of events. For the ±10–50% range typical of signal segments, the discrepancy is material enough to make the summary table disagree with the plot.

Log-variations are standard for statistical analysis (dispersion modelling, hypothesis testing) because they are additive over time and approximately normally distributed, but for summing event variations the discrete form is the correct choice. Therefore this module uses discrete compounding throughout.

**Parameters**
- Config-based: `performance.gradient.weight`, `performance.gradient.concurrent_weight_limit`, `performance.gradient.variation_method`, `performance.gradient.printout_io`, `performance.gradient.printout_debug`, `performance.gradient.full_output`, `performance.gradient.yearly_attenuation_fraction`, `performance.gradient.attenuation_month`, `performance.gradient.boundary_deduction_pct`, `performance.gradient.mpv`.

### Yearly Attenuation

**Purpose**

Model periodic base-value deductions (e.g. an annual amount proportional to the increase) that reduce the compounding base for subsequent events.

**Logic**

1. Build an **attenuation schedule** from the global observed-date index (the curve index). For each calendar year `Y` spanned by the index:
   - If a subsequent year exists → `attenuation_date` = first index date on or after the 1st of `attenuation_month` in year `Y+1`.
   - If `Y` is the last calendar year → `attenuation_date` = last date in the index.
   - Each item records `attenuation_date`, `year_start` (first observed date of `Y` in the index), and `year_end` (last observed date of `Y` in the index).
2. Attenuation schedule is empty when `performance.gradient.yearly_attenuation_fraction` is `0`.
3. During the event loop, **before** each event is processed, apply all pending attenuations whose `attenuation_date ≤ event start_date`. After all events, apply any remaining attenuations.
4. Each attenuation computes:
   - `v_start = 1 + total_cum[year_start]` — total value at start of the calendar year.
   - `v_end = 1 + total_cum[year_end]` — total value at end of the calendar year.
   - `increase = max(0, v_end − v_start)` — the year's increase, clamped at zero (no attenuation on a year that decreases). The base runs from the year's first to its last observed date, so the interval from a year's last observed date to the next year's first lies in no year's base: one interval between consecutive rows a year, so one week a year on a weekly series.
   - `attenuation_amount = yearly_attenuation_fraction × increase`.
5. Subtract `attenuation_amount` from `total_cum[attenuation_date:]` (all rows from `attenuation_date` onward), reducing the compounding base in-place.
6. Accumulate `attenuation_amount` into a running `cumulative_attenuation` counter for use by `multiplicative_total_before_active_signals` mode.

**Interaction with variation aggregation modes**

The attenuation modifies `total_cum` in-place. The two multiplicative modes obtain attenuation information through different paths:

| Mode | Attenuation source | Mechanism |
|---|---|---|
| `multiplicative_total_with_active_signals` | `total_cum` directly | Reads `total_cum(start_i)` which already includes attenuation deductions → `cumulative_attenuation` is unused |
| `multiplicative_total_before_active_signals` | `cumulative_attenuation` explicitly | Reconstructs value from individual `locked_weight × event_cum` pairs (which do not include attenuations) → subtracts `cumulative_attenuation` to compensate |

Both paths produce the same result — subsequent events compound on a reduced base value.

**Long-horizon behavior**

The attenuation introduces two effects on the post-attenuation gradient curve relative to the pre-attenuation curve:
- **Direct (linear):** Each year's attenuation shifts the curve down by `yearly_attenuation_fraction × max(0, year_increase)`. This gap grows roughly by one attenuation increment per year.
- **Compounding (nonlinear):** Subsequent events see lower `V(start_i)` → lower `locked_weight` → smaller contributions. This missed increase compounds over time.

For small weights, this is a smaller, compounding effect (`≈ w × cumulative_attenuation` per event), barely visible over short horizons — the two curves appear roughly parallel. Over longer horizons (15–20+ years) or with larger weights, the post-attenuation curve visibly diverges downward.

**Edge case — negative values**

An attenuation is sized on `v_end − v_start` (the calendar year's increase) but applied uniformly to all dates from `attenuation_date` onward. If a large year-end increase is followed by a sharp decline in the subsequent period, `total_cum` at some dates after `attenuation_date` may fall below `attenuation_amount`, producing negative values. This is correct behavior: the deduction applies to the prior year's increase regardless of the later values.

**Reference attenuation treatment**

References are passive baselines: they compound freely during each calendar year and only receive a single deduction at the very last date of the index, sized as `yearly_attenuation_fraction × max(0, total_increase)`. There is no intra-year schedule for references. This keeps each reference curve apple-to-apple with the corresponding attenuation-adjusted strategy curve at the end of the horizon, while avoiding the path-dependent interleaving used for the strategy.

**Data Conditions**
- `yearly_attenuation_fraction = 0` → attenuation schedule is empty, no deductions applied.
- Year with a negative change (`v_end < v_start`) → `increase = 0` → `attenuation_amount = 0`, year is skipped.
- Attenuation date falls after the index end (e.g. `attenuation_month` in a future month) → falls back to `idx[-1]`.
- The last calendar year's attenuation is applied at the last index date, the post-attenuation metrics come from that attenuated curve, and the plot's marker shows its attenuated last point.

**Parameters**
- Config-based: `performance.gradient.yearly_attenuation_fraction`, `performance.gradient.attenuation_month`.

### Per-Event Deduction

**Purpose**

Model a per-event deduction that reduces the effective variation at begin and end. The deduction represents any proportional reduction incurred when activating and deactivating an event (e.g. measurement overhead, signal latency, capacity reduction at activation/deactivation).

**Logic**

The deduction is applied per-event after geometric compounding (`event_cum = cumprod(1 + r_t) - 1`) and before weighting. Let `c = boundary_deduction_pct / 100`:

1. **Begin deduction** — scales down the event allocation from event start onward:
   ```
   adjusted_t = (1 + event_cum_t) × (1 - c) - 1
   ```
2. **End deduction** — additionally scales down the end-boundary variation from `end_date` onward:
   ```
   adjusted_t = (1 + adjusted_t) × (1 - c) - 1     for t ≥ end_date
   ```

Net factors:
- During event (before end): `(1 - c)` — reduced allocation from begin.
- After end (flat tail): `(1 - c)²` — the deduction of both boundaries applied.

**Numerical example**

A sensor monitoring system allocates 1000 units of capacity to track a signal source at value 100. Deduction at each event boundary is 0.1% (`boundary_deduction_pct = 0.1`, so `c = 0.001`).

| Step | Description | Calculation |
|---|---|---|
| **Begin (day 0)** | Deduction consumes 0.1% of allocated capacity | Deduction = 1000 × 0.001 = 1.0 unit |
| | Effective capacity deployed | 1000 − 1.0 = 999.0 units |
| | Scaled units tracked at source value 100 | 999.0 / 100 = 9.99 units |
| **Day 5** | Source value rises to 110 (+10%, `event_cum = 0.10`) | Scaled value = 9.99 × 110 = 1098.90 |
| | Variation from baseline 1000 | 1098.90 / 1000 − 1 = **+9.89%** |
| | Formula check | `(1 + 0.10) × (1 − 0.001) − 1 = 1.10 × 0.999 − 1 = +9.89%` ✓ |
| **Day 10 — End** | Source still at 110; end deduction applied | End-scaled value = 1098.90 × (1 − 0.001) = 1097.80 |
| | Final variation from baseline 1000 | 1097.80 / 1000 − 1 = **+9.78%** |
| | Formula check | `(1 + 0.0989) × (1 − 0.001) − 1 = +9.78%` ✓ |

The `(1 + event_cum)` term appears because the begin deduction reduces the number of scaled units tracked. The deduction is fixed (1.0 unit, taken once at begin), but tracking fewer units affects all subsequent values. The end deduction is a separate fixed deduction applied once at event end.

**Data Conditions**
- `boundary_deduction_pct = 0` → no deduction applied (no-op, bit-identical to unmodified path).
- Deduction is applied independently per event, before the event is scaled by `locked_weight`.

**Parameters**
- Config-based: `performance.gradient.boundary_deduction_pct`.

---

## Gradient Metrics

**Purpose**

Compute aggregate-level metrics on the cumulative variation series produced by the gradient. The orchestrator runs once on the pure (pre-attenuation) gradient and once on the attenuation-adjusted gradient (when attenuation is active); only `acr` takes a post-attenuation value, reported as `acr_after_attenuation`.

All metrics operate on **periodic decimal variations** derived from the cumulative percent series. Annualization is empirical:
```
m = (N − 1) / n_years        (periods per year)
n_years = (index[-1] − index[0]).days / 365.25
```
where `N` is the number of points in the series. The shared conversion is:
```
R_t = (1 + C_t/100) / (1 + C_{t-1}/100) − 1
```
which is identical to `v_t / v_{t-1} − 1` because the initial value cancels.

### Minimal Positive Variation Rate (`mpv`)

`performance.gradient.mpv` is a decimal threshold per row of the curve index (e.g. `0.0002` ≈ 0.02% per period on a daily series; a weekly run sets the same annual baseline with about five times that value) acting as a baseline that the strategy must clear before a period counts as positive. It is used by every dispersion-normalized ratio (`snr`, `snr_dw`, `sens_dw`, `exv`, `exv_dw`) so that:
- `snr` / `snr_dw` measure excess over `mpv` rather than excess over zero.
- `snr_dw` and `sens_dw` count any period below `mpv` as a downside breach.
- `exv` / `exv_dw` correct the intercept by `mpv × (1 − sens)` to remain consistent with the dispersion treatment.

Schema-validated range: `[0.0, 0.25]`. Default `0.0` (everything above zero counts as positive).

### Strategy Metrics

| Key | Formula (periodic R = decimal variations) | Notes |
|---|---|---|
| `acr` | `(1 + total_variation)^(1/n_years) − 1`, returned in `%` | Annualized compound rate. Returns `−100%` if `1 + total_variation ≤ 0`. |
| `snr` | `(mean(R) − mpv) / std(R, ddof=1) × √m` | Annualized excess-variation ratio against full dispersion. NaN when `std < 1e-15` or fewer than 2 periods. |
| `snr_dw` | `(mean(R) − mpv) / σ_d × √m`, where `σ_d = √(mean(min(R − mpv, 0)²))` | Annualized excess-variation ratio against downside semi-dispersion. The denominator divides by **N** (all periods), so it reflects both the magnitude *and* the frequency of downside breaches. |
| `max_fall` | `\|min((g_t − running_max_t) / running_max_t)\| × 100` | Maximum fall from the running maximum (positive percent). `g_t = 1 + C_t/100`. |
| `fall_dur` | Walk back from the deepest point to its preceding maximum; walk forward to recovery | Duration in periods (index steps) of the *deepest* fall episode. Truncated at the last index when the deepest event has not recovered by the end of the series. |
| `below_max_dur` | Longest run of consecutive periods below the running maximum (`g_t < running_max_t`) | Longest below-maximum period across all falls from the running maximum. Can correspond to a different event than `max_fall`/`fall_dur`. |

`max_fall`, `fall_dur`, and `below_max_dur` share a single pre-computation of `(gradient, running_max)` to avoid redundant passes. `fall_dur` and `below_max_dur` are JIT-compiled (`numba.njit`) to keep the walks to the deepest point and back tight.

### Reference-Relative Metrics

When at least one reference is available, the **first reference in sorted order** is used as the reference for the relative metrics. The strategy and the reference reference are aligned on the intersection of their indices before computing periodic variations:

| Key | Formula | Notes |
|---|---|---|
| `sens` | `Cov(R_s, R_ref) / Var(R_ref)`  (`ddof=1`) | Sensitivity coefficient against the full reference variation series. |
| `sens_dw` | Same formula, restricted to periods where `R_ref < mpv` | Sensitivity during reference downturns only. NaN if fewer than 2 downside periods. |
| `exv` | `(mean(R_s) − sens × mean(R_ref) − mpv × (1 − sens)) × m × 100` | Annualized excess variation rate over the reference, percent. |
| `exv_dw` | Same formula but with `sens_dw` substituted for `sens` | Annualized excess variation rate using the downside sensitivity. |
| `ref_snr` | `(mean(R_s − R_ref) / std(R_s − R_ref, ddof=1)) × √m` | Annualized excess-over-reference ratio using the full dispersion of the difference series. |
| `ref_snr_dw` | `(mean(R_s − R_ref) / σ_d) × √m`, where `σ_d = √(mean(min(R_s − R_ref, 0)²))` | Annualized downside variant. Like `snr_dw`, divides by **N** to capture both magnitude and frequency of falling below the reference. |

In addition, every reference in the map gets its own `acr` (using each reference's own index for the annualization factor). These appear in the summary table even though the reference-relative coefficients use only the reference reference.

**Pre-attenuation vs post-attenuation orchestration**

The orchestrator is invoked twice (when attenuation is active):
- **Pre-attenuation pass:** pure gradient + pre-attenuation reference map (each reference compounded with no deduction).
- **Post-attenuation pass:** attenuation-adjusted gradient + post-attenuation reference map (each reference with a single deduction at the last date).

When `yearly_attenuation_fraction == 0`, only the pre-attenuation pass runs: the summary table keeps its strategy and reference columns, and its `acr (after attenuation) (%)` row stays blank (`NaN`).

**Why post-attenuation values are not shown for dispersion and relative metrics**

Yearly attenuation amounts create scheduled, discrete step-downs in the cumulative series at known calendar dates (e.g. every June). If the attenuation-adjusted series were used to compute dispersion and relative metrics, these step-downs would be treated as variation events:

- **Variation-to-dispersion ratios (`snr`, `snr_dw`):** the step-down periods count as negative variations, inflating the downside dispersion and producing a lower (worse) ratio that reflects attenuation accounting rather than strategy signal quality.
- **Fall from the running maximum and duration (`max_fall`, `fall_dur`, `below_max_dur`):** each attenuation event appears as a fall episode, inflating the decline magnitude and below-maximum duration beyond what the signal itself generates.
- **Covariance-based relative metrics (`sens`, `sens_dw`, `exv`, `exv_dw`, `ref_snr`, `ref_snr_dw`):** the strategy series has step-downs at attenuation dates while the reference series is kept clean throughout (its attenuation is applied only at `iloc[-1]`). This mismatch distorts covariance-based metrics.

The fairest approach for these metrics is to use the **pre-attenuation series**, which measures only the signal's variation profile. This amounts to measuring *how well the strategy navigates value movements, ignoring scheduled attenuation deductions* — exactly what the pre-attenuation values already show. Showing a parenthesized post-attenuation value would be either redundant (identical to pre-attenuation, since using pre-attenuation in both slots) or misleading (using post-attenuation and penalising for attenuation events). Therefore, for all metrics except `acr`, only the pre-attenuation value is displayed. The `acr` metric is the exception: attenuation directly and permanently reduces compounded variation, so the post-attenuation `acr` is the real post-attenuation annualised cumulative variation and the `pre (post)` pair is meaningful.

**Data Conditions**
- Series shorter than 2 points → all metrics return empty / NaN.
- `n_years < 1e-6` → annualization factor falls back to `(m=1.0, n_years=0.0)` and `acr` returns `0.0`.
- Total variation ≤ −100% → `acr` clamps to `−100.0`.
- No reference available → all reference-relative metric keys are filled with `NaN`.
- Reference with fewer than 2 aligned points → reference-relative metrics skipped (NaN).

**Parameters**
- Config-based: `performance.gradient.mpv`. Indirectly: `read.ini_date`, `read.end_date`, `trigger.replay.lookback_window` (the global observed-date index used for annualization holds the observed dates of the processed series and the references, from the earliest effective start over the trigger store's series to `read.end_date`).

---

## Gradient Plot

**Purpose**

Generate a two-panel figure (shared x-axis) showing total variation and per-signal segments, with reference overlays on the top panel and a summary statistics block (signal metrics + gradient metrics) anchored under the bottom panel.

**Logic**
1. Build the global observed-date index: the sorted union of the observed dates of the processed series and the references from the earliest effective start over the trigger store's series (`df.index[trigger.replay.lookback_window − 1]` of each preprocessed frame) to `read.end_date`, never a weekday calendar.
2. **Top panel**: plot the pure cumulative variation (solid black). When attenuation is active, overlay the attenuation-adjusted curve as a dashed black line on the same axis. Optional reference variation curves are drawn dashed with periodic markers and a unique color per reference. Reference curves are built from per-period variation with geometric compounding (`cumprod(1 + r) - 1`) and aligned to the global observed dates, each 0 on the reference's first row on the index. The y axis is symmetric-log when the curve index spans more than `GRADIENT_LONG_WINDOW_YEARS` calendar years and linear otherwise, each with a bounded tick count.
3. Recalculate top-panel y-limits so they enclose the pure gradient, the attenuation-adjusted gradient (when present), and every reference series, with 10% padding.
4. **Bottom panel**: plot each signal percent-change segment, clipped to `[-50%, +50%]`, styled by plus/minus × true/false, annotated with `(index) [series_id] variation%`. Labels use a 3-level stagger cycle for readability.
5. Emit standalone per-signal plots with the figure, filtered by `performance.gradient.plots_whitelist` and capped by `performance.gradient.max_plots`; the figure and its per-signal plots are drawn only when `performance.gradient.plot_debug` is `true`.
6. Append a summary statistics table below the figure (see *Summary Table* below).

### Summary Table

The summary text appended under the bottom panel has two sections:

1. **Signal Metrics** — transposed layout: metrics are rows, columns are 3 directions (`plus+minus`, `plus`, `minus`) × 3 subdivisions (`true+false`, `true`, `false`). The first row is the signal count per cell; subsequent rows are one per `(metric, operation)` pair from `metrics.json`, in `performance.metrics.display_order_signals`. All aggregation (combined / true / false) is computed upstream by `calculate_direction_metrics`, with trials-to-true recomputed per direction subset.
2. **Gradient Metrics** — a table below the signal block, after an annualization line (`N-1`, years, `m`, `√m`): one column for the strategy and one per reference (`reference <id>`), one row per key of `performance.metrics.display_order_gradient`. Each reference column holds the reference's own standalone metrics and the strategy's relative metrics against it; the strategy column leaves the relative rows blank. Only the `acr (after attenuation) (%)` row (`acr_after_attenuation`) holds post-attenuation values, `NaN` when attenuation is disabled; every other row holds pre-attenuation values.

The vertical figure margin grows with the number of summary lines so the table never overlaps the bottom panel.

**Data Conditions**
- `performance.gradient.plot_debug` `false` → no figure and no per-signal plot; the columns' shared annualization (`N-1`, years) is still checked on every run before persistence, and a column whose pair differs raises `ValueError`.
- Empty `signals_data` or empty filtered signal rows → no figure generated.
- Missing reference data → reference overlay is skipped, and the gradient-metrics section holds only the strategy column, whose reference-relative rows stay blank (`NaN`).
- A standalone signal plot failure raises.
- The signal breakdown counts the signals that build the curve: the `is_signal` rows the weight filter keeps.
- Reference rows before the index start, kept by a reference whose preprocessed frame is shorter than `trigger.replay.lookback_window` → dropped before the base is taken; a reference without a row on the index → no curve.

**Parameters**
- Config-based: `performance.gradient.weight`, `performance.gradient.concurrent_weight_limit`, `performance.gradient.variation_method`, `performance.gradient.yearly_attenuation_fraction`, `performance.gradient.attenuation_month`, `performance.gradient.boundary_deduction_pct`, `performance.gradient.mpv`, `performance.gradient.max_plots`, `performance.gradient.plot_debug`, `performance.gradient.plots_whitelist`, `performance.gradient.printout_io`, `performance.gradient.printout_debug`, `performance.metrics.display_order_signals`, `performance.metrics.display_order_gradient`, `read.ini_date`, `read.end_date`, `trigger.replay.lookback_window`.
- Hardcoded: `GRADIENT_LONG_WINDOW_YEARS` = 5 (`src/utils/common/constants.py`).

---

## Persistence

**Purpose**

Write enriched signals and optional gradient outputs to the performance HDF5 file.

**Logic**
1. Build schema with typed columns:
   - Metrics + trigger values as `float32`
   - Flags as `bool`
   - Group metadata as `int32`
   - Identifiers as `str`
2. Write enriched signals table to `/signals` with `format="table"`.
3. Write strategy name metadata to `/strategy_name` if present.
4. Write gradient plot reconstruction data to HDF5 (see schema below).

**Output HDF5 Structure**

> **Schema break (introduced with gradient plot persistence):** HDF5 files written
> before the gradient plot persistence change used a single `/total_variation` key
> with two columns (the pure curve and its attenuated copy). That
> schema is no longer written. The new replot script (`scripts/gradientreplot.py`)
> requires the new schema; regenerate the performance output with the current
> pipeline to produce a compatible file.

| Key | Format | Description |
|---|---|---|
| `/signals` | `table` | Enriched signals table with metrics, group assignments, and signal flags. |
| `/strategy_name` | `table` | Strategy name metadata. |
| `/total_variation` | `fixed` | Pre-attenuation gradient: columns `date`, `total_variation` (float32). |
| `/total_variation_attenuated` | `fixed` | Attenuation-adjusted gradient: columns `date`, `total_variation_attenuated` (float32). **Only present when `yearly_attenuation_fraction > 0`.** |
| `/gradient_reference_variation` | `table` | Long-form reference curves: `reference_id` (str), `date`, `variation` (float32). |
| `/gradient_signal_segments` | `fixed` | Per-group pct series: `group_id` (int32), `date`, `pct` (float32). |
| `/gradient_filtered_sequence` | `fixed` | Sorted filtered signals: `sequence_index` (int32), `group_id` (int32). |
| `/gradient_terminal_attenuations` | `table` | Terminal attenuation markers: `series_id` (str, `__strategy__` for strategy), `amount` (float32). **Only present when `yearly_attenuation_fraction > 0`.** |
| `/gradient_params` | `table` | Scalar replot parameters (one row): weight, variation_method, ini_date, `effective_start` (the curve's first date), etc. |
| `/gradient_param_sequences` | `table` | List-valued params: `kind` (str), `sequence_index` (int32), `name` (str). |
| `/gradient_metrics_table` | `table` | Tall-form gradient metrics: `metric_key`, `column_header`, `value` (float32). |
| `/signals_breakdown_table` | `table` | Tall-form signals breakdown: `row_key`, `column_header`, `value` (float32). |