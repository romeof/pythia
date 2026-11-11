# Design decisions

The documents state each rule below without its reason; this table keeps the reason. Each id names the decision in
the audit records of the pipeline: `Q` a design question the operator decided, `F` a finding whose fix set the rule,
`G` a follow-up group, `E` an inconsistency the audit recorded. One row per id, in id sequence.

| Id | Stated in | Decision and reason |
|---|---|---|
| E-4 | pipeline.md, Execution | The run reads a task's failure from the result of `luigi.build`, which the pinned Luigi returns `False` for a task whose `run()` raises, contrary to its docstring; the run relies on the measured result, so a new pin must keep it. |
| E-5 | pipeline.md, Execution | Git runs in the repository that holds the pipeline source, so `git_info.txt` names the code that ran wherever the run starts; run in the working directory, it named whatever repository the run started from. |
| E-7 | pipeline.md, Execution; trigger.md, Trigger Detection | A lookback window shorter than the minimum length is a configuration error, refused at startup with both values; treated as a data condition, it skipped every series and completed an empty run. |
| F-01 | trigger.md, Trigger Detection | Each turning point takes its type from the `find_peaks` call that reported it, so a point on a plateau keeps its begin; typing a point by a second comparison of its neighbours left tied points without one. |
| F-19 | trigger.md, Ideal Mode | The ideal cleaning leaves `end_reason` only on the end rows of the pairs it keeps, so no printout or plot shows an end reason on a row that is no end. |
| F-20 | trigger.md, Preprocess | A block of consecutive bad data counts rows, not calendar days, so dates missing between two rows never break or hide a block. |
| F-21 | metrics.md, Signal Metrics | The compound of per-event variations keeps a NaN variation as NaN; a compiled shortcut had turned it into −100 %, which reads as a −100 % variation instead of a missing value. |
| G-7 | pipeline.md, Execution | A run removes the outputs and provenance copies it is about to replace before its first build, and a failed run keeps none of them, so no stale file describes outputs the run did not produce. |
| G-8 | pipeline.md, Execution | A failure is reported once: a producer's failure reaches its task through the producer's future and appears in the one `Task failed` record. |
| G-9 | pipeline.md, Execution; trigger.md, Gate Pre-filter; signal.md, Accumulation; performance.md, Summary Table | A failure raises at once; a series or item is skipped only by an explicit check of a documented data condition, never by catching a failure, so malformed data can no longer drop work silently. |
| G-10 | pipeline.md, Execution | A store that cannot be read or created is logged once with its path and raised, and any other exception inside a store's block keeps its own context, so the log names the real cause. |
| G-11 | pipeline.md, Execution | Every git call is bounded and a failed one fails the run after the task outputs are written, so provenance never waits without limit and never goes missing unreported. |
| G-12 | pipeline.md, Execution | A consumer reads until the producer's sentinel, so no item is left unread, and a producer that ends without it fails the task after one queue timeout instead of hanging. |
| Q1 | trigger.md, Trigger Detection | The prominence floor is relative to the data the mode sees, the window in Realistic mode and the whole series in Ideal mode, by design; the two modes are not meant to find the same triggers. |
| Q2 | trigger.md, Trigger Detection | One effective start, counted in rows of the preprocessed frame, serves the realistic loop, the extraction filter, the reference trimming and the gradient index; a second, calendar-based start disagreed with it by three days. |
| Q3 | trigger.md, Trigger Detection | The gate reads the previous row's gate columns in both modes, so the recorded value never changes a decision. |
| Q4 | performance.md, Weight Constraint Validation | On a shared date the sweep takes end events before begin events, so a signal that begins the day another ends is not concurrent with it; a zero-length signal's own end follows that day's begins, so it still counts on its day. |
| Q5 | trigger.md, Gate Pre-filter | Superseded by the gate pre-filter: the total-magnitude filter is removed, and a series is skipped only when one applicable gate item passes on no row of the read window, never a reference series. |
| Q6 | trigger.md, Realistic Mode Trigger End | An opposite extremum seen within the minimum duration stays suppressed for good; activating it once the minimum duration ends would act on a stale extremum. |
| Q7 | trigger.md, Realistic Mode Trigger Begin | Only the newest candidate of a type may activate, so an older extremum detected later is never acted on. |
| Q8 | metrics.md, Signal Metrics | `trials_to_true` is counted within each direction, so a wait count never spans rows of the other direction. |
| Q9 | metrics.md, Signal Metrics | Under `following_row` a value is labelled with its decision day, the row before the `var0` it records, by design. |
| Q10 | signal.md, Edge Case: Multiple Eligible Signals | Group membership compares Pearson's r of `var3` levels over the `signal.period_back` + 1 points ending at the trigger begin, in absolute value, with `signal.corr_threshold`, as designed. |
| Q11 | pipeline.md, Execution | The validated schema model is the configuration the run uses, so a declared default applies when its key is omitted. |
| Q12 | pipeline.md, Execution | The run removes a task's previous output before its build and `output()` never deletes, so Luigi's completeness check never destroys a task's output. |
| Q13 | trigger.md, Trigger Detection | `trigger.strategy.window_size` names the output folder and feeds no detection logic; it stays, so existing output paths stay valid. |
| Q14 | trigger.md, Realistic Mode Trigger Begin | `turning_points` counts the turning points of the window ending at each row, so the realistic column carries information instead of zeros. |
| Q15 | signal.md, Edge Case: Multiple Eligible Signals | In Ideal mode, eligible signals with equal signal end are taken newest first, the coded tie rule kept. |
| Q16 | trigger.md, Ideal Mode | Each begin pairs with the first end after it, and only an end with no begin before it or a final begin with no end after it is dropped, so no complete pair is lost. |
| Q17 | performance.md, Yearly Attenuation | The last calendar year's attenuation applies to the curve at its last date, so the post-attenuation metrics and the plot's marker describe the attenuated curve. |
| Q18 | pipeline.md, Execution | A failed item fails its task at once; no call site logs a failure, finishes the other items and fails later. |
| Q20 | trigger.md, Gate Pre-filter, Triggers Extraction; signal.md, Edge Case: Multiple Eligible Signals | Realistic mode uses only what is known at a trigger's begin: eligible signals are taken by their begin and group, never by their end; a trigger still active at the data end is not extracted; the gate pre-filter reads the whole read window but skips only a series whose every begin the gate would block, so it never changes a trigger. |
| Q21 | performance.md, Summary Table | The signal breakdown counts the signals that build the curve, the signal rows the weight filter keeps, so the table agrees with the plotted curve. |
