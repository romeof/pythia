# Audit experiments

Each experiment under `audit/` runs only through `audit/harness.py`, which records the launch, its tree and its
environment. The table names, per experiment, what it checks and the regressions under `tests/` that check the same
findings on every suite run; a finding no regression checks is listed after them, and an experiment that only drives
runs or checks the audit's own records shows none.

| Experiment | Checks | Superseded by |
|---|---|---|
| `exp00run.py` | The driver of the audit's pipeline runs (RUN-1, RUN-2, RUN-T) | none |
| `exp01pairing.py` | Ideal-mode pairing (HYP-01, Q16), `end_reason` on cleaned rows (HYP-02), plateau extrema (HYP-22) | `tests/idealpairing.py`, `tests/idealend.py`, `tests/plateauextrema.py` |
| `exp02recording.py` | Trigger decisions independent of the `trigger_value` recording mode (HYP-03) | `tests/gatemodes.py` (recording independence) |
| `exp03effectivestart.py` | One positional effective start per stage (HYP-04, Q2) | `tests/effectivestart.py` |
| `exp04prefix.py` | No future rows in realistic mode (R-T7), the length guards (R-T4), the gate's previous-bar dependence (HYP-03) | `tests/replayvalidation.py` (no future rows, the length boundary), `tests/gatemodes.py` (the gate's previous-bar dependence) |
| `exp05defaults.py` | Schema defaults, strict validation and merge precedence (HYP-05, Q11, R-X2, R-X6) | `tests/schemadefaults.py`, `tests/configcontract.py`; merge precedence (R-X6): none |
| `exp06returncode.py` | Return code and failure visibility (HYP-08, HYP-24, HYP-27, KNOWN (b)) | `tests/returncode.py`, `tests/configcontract.py`, `tests/zerosignals.py`, `tests/luigibuild.py` |
| `exp07partialwrite.py` | Partial writes, crash consumption and soft write failures (HYP-07, HYP-09, HYP-28, HYP-29, HYP-30) | `tests/atomicwrite.py`, `tests/readfailure.py`, `tests/referenceloader.py`, `tests/persistfailure.py`, `tests/preprocessfailure.py`, `tests/referencefailure.py` |
| `exp08grouping.py` | Signal grouping (R-S1..S3, Q15, Q20, HYP-19, HYP-26, A4(b)) | `tests/pointintimegrouping.py`, `tests/corrcache.py`, `tests/endtoend.py` (R-S2); Q15, HYP-26, A4(b): none |
| `exp09gradient.py` | The gradient recomputed from the documented formulas (R-P2..P6, R-P8, R-P10, HYP-26, HYP-31) | `tests/endtoend.py` (R-P2..P6), `tests/lastyearattenuation.py`; R-P8, R-P10, HYP-26, HYP-31: none |
| `exp10sweepline.py` | The weight constraint's sweep line (Q4, R-P3) | `tests/sweepline.py` |
| `exp11outcome.py` | Trigger extraction edge cases (R-T9) | `tests/outputcontract.py` |
| `exp13compare.py` | A value-by-value comparer of two output trees (CMP-1..3), the tool of the identity checks | none |
| `exp14preprocess.py` | The preprocess drop rules and the consecutive-block report (R-T3, HYP-17) | `tests/badblocks.py`, `tests/boundaries.py` (NaN rows) |
| `exp15triggervalue.py` | `trigger_value` selection (R-T5, R-T16) | `tests/boundaries.py` (R-T16); R-T5: none |
| `exp16candidates.py` | Realistic candidate selection and minimum-duration suppression (Q6, Q7) | `tests/newestcandidate.py`, `tests/endmodes.py`, `tests/minimumduration.py` |
| `exp17replot.py` | The replot against the persisted tables (R-P7, R-P10, plots, HYP-31) | `tests/replotroundtrip.py` |
| `exp18encoding.py` | The preprocess report under the platform's ANSI code page (HYP-12) | `tests/reportencoding.py`, `tests/consoles.py` |
| `exp19replayvariant.py` | The former replay validation suite run unmodified, one planted defect per test (HYP-13, D-1) | `tests/replayvalidation.py`, which replaced that suite |
| `exp20sweeps.py` | AST and text sweeps of production code and of every audit and test script; the pre-commit gate | none |
| `exp21contracts.py` | `Read`'s branches and the persisted data contracts (R-S4, R-P7, R-T1, R-T10, R-T14) | `tests/outputcontract.py` (the contracts); `Read`'s branches (R-T1, R-T10, R-T14): none |
| `exp22hashseed.py` | Run-to-run determinism with two references under differing hash seeds (HYP-15) | `tests/hashseeds.py` |
| `exp23reportcheck.py` | Structural checks of the audit report (acceptance checks A1, A3, A8, A9) | none |
| `exp24perseries.py` | Per-series metrics (R-P1, Q8, HYP-16) | `tests/trialsdirection.py` (Q8); R-P1, HYP-16: none |
| `exp25traces.py` | The six traces of the reference run, value by value (TR-0 to TR-6) | none |
| `exp26structure.py` | Structure readings (elegance and simplicity) over the files of one commit | none |

The other modules serve the experiments: `harness.py` (the launch harness: evidence record, gate, lock, self-test),
`pipelinekit.py` (the in-process helpers of the experiments that run pythia code), `synthetic.py` (synthetic inputs
in the reader's layout) and `perflaunch.py` (timed launches outside the harness, under its lock).
