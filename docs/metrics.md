# Metrics

## Brief Summary

Two metric families are computed during the Performance task:

- **Signal metrics** (`perseries_/metrics.json`) — Per-series, per-event aggregates applied to trigger and signal rows. Six composite keys are available; any subset is selected via `display_order_signals`.
- **Gradient metrics** (`gradient_/metrics.json`) — Aggregate-level metrics computed on the cumulative variation series. Sixteen keys in two groups (standalone, relative); any subset is selected via `display_order_gradient`.

All metric definitions, labels, and group assignments live in their respective `metrics.json` files. Python code accesses metric keys only through the string constants in `constants.py`.

**Example Output (only selected metrics shown)**

The summary table printed after each per-series block and drawn under the plots: the signal metrics in a transposed layout, the columns the three directions (`plus+minus`, `plus`, `minus`) times the three outcome groups (`true+false`, `true`, `false`), the rows the signal count (`Signals`) and then the keys of `performance.metrics.display_order_signals`; under the gradient plot a second table follows, the gradient metrics with one column for the strategy and one per reference series, its rows the keys of `performance.metrics.display_order_gradient`.

### Input

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\trigger\0.h5` and `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\signal\0.h5`

### Output

data produced with command `python pipeline.py --task performance --strategy extrema --profile dev`

data in `pythia\pythia\data\reference\0\extrema\var12ge0_83x0-var15ge0_73__w7md5p0_05\{ideal,realistic}\performance\0.h5`

---

## Signal Metrics (Per-Series, Per-Event Aggregates)

Six composite keys defined in `perseries_/metrics.json`. None are annualized — these are cross-sectional aggregates over discrete events, not rates over a time span.

| Metric | Purpose | Logic | Notes | Annualized | Annualization | Reason |
|---|---|---|---|---|---|---|
| `variation avg (%)` | Mean per-event % change | `mean(variation)` where variation = `±(end/start − 1) × 100` | Plus events: positive move; minus events: sign reversed | No | — | Cross-sectional average over events; no time span to normalize |
| `variation std (%)` | Dispersion of per-event % changes | `std(variation, ddof=1)` | Measures consistency of event outcomes | No | — | Cross-sectional; no time dimension to annualize |
| `variation std_dw (%)` | Downside semi-dispersion of per-event % changes | `sqrt(mean(min(variation, 0)²))` | RMS of raw negative values; divides by N via mean (ddof=0) | No | — | Cross-sectional; no time dimension to annualize |
| `multiplicative (%)` | Geometric compound of all per-event % changes | `[∏(1 + rᵢ/100) − 1] × 100` via log-space | Returns −100% for a variation of −100% or below; a NaN variation yields NaN; log-space for numerical stability | No | — | Aggregate over events; not a per-year rate |
| `duration avg` | Mean event duration in rows of the series index | `mean(end_pos − begin_pos)` via `DatetimeIndex.get_indexer` | Row numbers from the original series index (its observed dates); not calendar days | No | — | Already a time quantity; annualizing would be meaningless |
| `trials-to-true avg` | Mean wait count between consecutive true events | JIT counter per row, counted within each direction; `avgt` averages over true rows only | `[T,F,T,F,F]→[1,0,2,0,0]`, avgt = (1+2)/2 = 1.5 | No | — | Dimensionless count; no per-year scaling defined |

**Variation formula** — direction-aware: `variation = (end/start − 1) × 100` for plus events, `−(end/start − 1) × 100` for minus events. Computed as contiguous `float32` arrays.

**`following_row` values** — With `trigger.trigger_value` set to `"following_row"`, the values are the next row's `var0`, labelled on the decision day by design.

**Direction-based aggregation** — For each direction (`plus`, `minus`, `plus+minus`), rows are filtered by `is_plus_trigger` and `trials_to_true` is recomputed on the subset, so wait times reflect only same-direction rows. `trials_to_true` counts per direction: the persisted `/signals` column over every row of each direction, the breakdown on each direction subset, ties broken by `trigger_begin_dates`, then `series_id`.

---

## Gradient Metric Keys

Each gradient metric key and its full name:

| Key | Full name |
|---|---|
| `acr` | annualized compound rate |
| `acr_after_attenuation` | annualized compound rate of the attenuation-adjusted curve |
| `std` | annualized dispersion |
| `std_dw` | annualized downside dispersion |
| `snr` | mean-to-dispersion ratio |
| `snr_dw` | mean-to-downside-dispersion ratio |
| `max_fall` | maximum fall from the running maximum |
| `fall_dur` | fall duration |
| `below_max_dur` | below-maximum duration |
| `rho` | correlation with the reference |
| `sens` | sensitivity to the reference |
| `sens_dw` | downside sensitivity to the reference |
| `exv` | excess variation rate |
| `exv_dw` | downside excess variation rate |
| `ref_snr` | excess-over-reference ratio |
| `ref_snr_dw` | downside excess-over-reference ratio |

---

## Gradient Metrics: Standalone

Computed on the cumulative variation series produced by the gradient aggregation. Periodic variations are derived first:

```
R_t = (1 + C_t/100) / (1 + C_{t-1}/100) − 1
```

Annualization factor: `m = (N−1) / n_years` where `N` is the number of data points (the observed dates of the curve index) and `n_years = span_days / 365.25`.

The summary table prints a four-row info block above the metric rows showing, per column: `N-1` (period count), `years`, `m`, and `√m`.

| Metric | Purpose | Logic | Notes | Annualized | Annualization | Reason |
|---|---|---|---|---|---|---|
| `acr (%)` | Annualized compound rate | `(1 + V)^(1/n_years) − 1`, V = final cumulative decimal | Returns 0 if span < 1e-6 yr; −100% if base ≤ 0 | Yes | Geometric | Expresses compounded growth on a per-year basis for cross-series comparability |
| `acr (after attenuation) (%)` | Same as `acr` on attenuation-adjusted series | Identical formula applied to post-attenuation cumulative series | Only computed when attenuation rate > 0 | Yes | Geometric | Same reason as `acr` |
| `std (%)` | Annualized dispersion of periodic variations | `std(R, ddof=1) × √m × 100` | Bessel-corrected; R in decimal | Yes | × √m | √T rule: σ_ann = σ × √m; enables cross-frequency comparability and scale coherence with `acr` |
| `std_dw (%)` | Annualized downside semi-dispersion | `sqrt(mean(min(R − mpv, 0)²)) × √m × 100` | Full-N denominator (not downside-count only); reflects both magnitude and frequency of breaches | Yes | × √m | Same √T rule; downside σ must match `acr`'s annual scale |
| `snr` | Mean-to-dispersion ratio | `(mean(R) − mpv) / std(R, ddof=1) × √m` | NaN if std < 1e-15 or N < 2 | Yes | × √m | mean(R) and std(R) must share per-year scale for the ratio to be meaningful |
| `snr_dw` | Mean-to-downside-dispersion ratio | `(mean(R) − mpv) / sigma_d × √m`, sigma_d as in `std_dw` | NaN if sigma_d < 1e-15 | Yes | × √m | Same as `snr`; downside sigma must match `acr`'s annual scale |
| `max_fall (%)` | Magnitude of the maximum fall from the running maximum | `|min((g_t − max(g₀…t)) / max(g₀…t))| × 100` | g_t = 1 + C_t/100; magnitude only, always positive | No | — | A magnitude, not a rate; no per-year scaling applies |
| `fall_dur` | Fall duration: from the preceding maximum through the deepest point to recovery | Period count from preceding maximum to deepest point to recovery of the deepest decline | Truncated at series end if unrecovered | No | — | Duration of a single event; already in periods |
| `below_max_dur` | Below-maximum duration: the longest below-maximum period | Longest consecutive gap between new-maximum dates | May correspond to a shallower decline than `fall_dur` | No | — | Duration of a single event; already in periods |

---

## Gradient Metrics: Relative (Strategy vs. Reference)

Computed on the intersection of the strategy and reference series (common-index alignment). Periodic variations `R_s` and `R_ref` are derived independently from each series on the shared index.

| Metric | Purpose | Logic | Notes | Annualized | Annualization | Reason |
|---|---|---|---|---|---|---|
| `rho` | Pearson correlation between series and reference periodic variations | `Cov(R_s, R_ref) / sqrt(Var(R_s) × Var(R_ref))`, ddof=1 | NaN if either variance < 1e-30 | No | — | Dimensionless ratio; annualization factors cancel |
| `sens` | Sensitivity of series to reference (full-sample) | `Cov(R_s, R_ref) / Var(R_ref)`, ddof=1 | NaN if Var(R_ref) < 1e-30 | No | — | Ratio of co-variation to reference variation; annualization cancels |
| `sens_dw` | Downside sensitivity during reference declines | Same as `sens` restricted to periods where `R_ref < mpv` | NaN if fewer than 2 downside periods | No | — | Same as `sens`; annualization still cancels on the subset |
| `exv (%)` | Annualized excess variation rate | `(mean(R_s) − sens × mean(R_ref) − mpv × (1−sens)) × m × 100` | Adjusts for mpv threshold; NaN if sens is NaN | Yes | × m | Raw residual is a per-period mean; × m converts to per-year rate |
| `exv_dw (%)` | Annualized downside excess variation rate | Same formula with downside-subset means and `sens_dw`; full-sample `m` | All inputs (means, `sens_dw`) from the `R_ref < mpv` subset; `m` from full history | Yes | × m | Same as `exv`; full-sample `m` preserves comparability between `exv` and `exv_dw` |
| `ref_snr` | Excess-over-reference ratio | `mean(R_s − R_ref) / std(R_s − R_ref, ddof=1) × √m` | NaN if the dispersion of `R_s − R_ref` < 1e-15 | Yes | × √m | Numerator and denominator must share per-year scale; same √T rule as `snr` |
| `ref_snr_dw` | Downside excess-over-reference ratio | `mean(R_s − R_ref) / sigma_d(R_s − R_ref) × √m`, sigma_d over all periods | Full-N denominator (not downside-count only) | Yes | × √m | Same as `ref_snr` |

---

## Why Annualize?

### 1. Comparability Across Sampling Frequencies

The same underlying variation process produces different raw dispersion depending on sampling frequency. The √T rule (`σ_ann = σ_period × √m`) collapses these to the same value:

| Sampling | m | σ_period | σ_ann |
|---|---|---|---|
| Daily | 365 | 1.00% | 1.00% × √365 ≈ **19.1%** |
| Weekly | 52 | 2.65% | 2.65% × √52 ≈ **19.1%** |
| Monthly | 12 | 5.52% | 5.52% × √12 ≈ **19.1%** |

Without annualization, comparing dispersion across series sampled at different frequencies is not meaningful. The empirical factor `m = (N−1)/n_years` adapts to the actual sampling density of each series, making the rule robust to irregular intervals.

### 2. Scale Coherence with ACR

`acr` is defined on a per-year basis. If `std` is reported in raw per-period units, the ratio `acr/std` (i.e. `snr`) has no consistent interpretation:

| Display | acr | std | ratio |
|---|---|---|---|
| Raw (daily) | 12.6% | 1.0% | 12.6 — misleading |
| Annualized | 12.6% | 19.10% | **0.66** |

The annualized ratio (0.66) expresses the per-year variation rate relative to per-year dispersion — a coherent, cross-series-comparable quantity. The raw ratio (12.6) confounds the variation-rate scale (annual) with the dispersion scale (daily), producing a number that cannot be compared across series with different histories or sampling frequencies.

---

## Annualization Summary

| Method | Formula | Applied to | Condition |
|---|---|---|---|
| Geometric | `(1 + V)^(1/n_years) − 1` | `acr`, `acr (after attenuation)` | Total cumulative variation in decimal |
| × √m | `value × √m` | `std`, `std_dw`, `snr`, `snr_dw`, `ref_snr`, `ref_snr_dw` | Dispersion-based or ratio metrics |
| × m | `value × m` | `exv`, `exv_dw` | Per-period mean residual annualized to per-year rate |
| None | — | All signal metrics, `max_fall`, `fall_dur`, `below_max_dur`, `rho`, `sens`, `sens_dw` | Not rates over time, or annualization cancels |

### Design Notes

1. **`exv_dw` and full-sample `m`** — `exv_dw` uses downside-subset means and `sens_dw`, but annualizes with the full-sample `m`. This keeps `exv` and `exv_dw` on the same per-year scale so they can be compared directly: both express an excess variation rate per year, differing only in which subset of periods drives the coefficient and means.

2. **Reference history scope** — Standalone reference metrics (`acr`, `std`, etc.) are computed on the reference's own full history. Relative metrics (`rho`, `sens`, `exv`, `ref_snr`, …) are computed on the intersection of the strategy and reference series. The annualization factor `m` is derived from whichever index is in scope for each metric.
