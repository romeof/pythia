"""
Centralized constants for the pythia codebase.

Single source of truth for all string literals
and named values used across trigger, signal, and performance modules.

Structure:
1. Data value columns (var3, trigger_value, etc.; the value columns var0-var5, the gate column and exempt id patterns)
2. Trigger detection columns (trigger_plus_begin, etc.)
3. Trigger detection column groups and mappings
4. Trigger outcome columns (trigger_begin_dates, etc.)
5. Signal grouping columns (group_id, is_signal, etc.)
6. End reason values (ideal_mode, max_duration, etc.)
7. Replay mode values (ideal, realistic)
8. Window type values (expanding, rolling)
9. Trigger value mode values (detection_row, following_row)
10. Gate operations (>=, <=: the comparisons of a trigger.gate item) and the output name's gate tokens
11. Signal criterion values (earliest_begin, etc.)
12. Variation method values (multiplicative_fixed, multiplicative_total_with_active_signals, multiplicative_total_before_active_signals)
13. Direction values (plus+minus, plus, minus) and outcome subdivisions (true+false, true, false)
14. Performance summary table labels (section headers, the Signals row)
15. Metric group identifiers (the group field of gradient_/metrics.json)
16. Gradient metric key constants (the keys of gradient_/metrics.json, the only metric-key literals in code)
17. HDF5 key prefixes (series_, triggers_, reference_series_)
18. HDF5 gradient plot keys (/gradient_reference_variation, /gradient_signal_segments, etc.)
19. Pipeline failure record (the prefix of the Luigi FAILURE record)
20. File encodings (the pipeline's own text, never the code page: the preprocess report, git_info.txt and git's captured output, the metric configuration files)
21. Provenance files and git (git_info.txt, the git call bound)
22. Suppression reasons (before_effective_start, gate_blocked, etc.: why a realistic candidate is not taken)
23. Gradient figure (the curve-index span beyond which the top panel is symmetric-log)
24. Config profiles (dev, prod: the switch YAML merged beside the strategy YAML; the default)
25. Signal selection log record (the fallback pairs the summary record names)
26. Plot backend (Agg: the plots packages select it before any module loads pyplot)
27. Literal spellings (the command line's booleans, the aggregation operations, pandas' from_dict orientation)
28. Strategy names (extrema: the strategies schema.strategy_class registers)
"""

# =============================================================================
# DATA VALUE COLUMNS — positional storage names; the per-row value columns
# match the HDF5 contract exactly (var0..var5 per atp_terminology.md).
# =============================================================================

VAR3 = "var3"
VAR4 = "var4"
VAR0 = "var0"
VAR2 = "var2"
VAR1 = "var1"
VAR5 = "var5"
TRIGGER_VALUE = "trigger_value"
# The value columns: the reader's base set, the columns the NaN check and the validity rules read
VALUE_COLUMNS = (VAR0, VAR1, VAR2, VAR3, VAR4, VAR5)
# A positional input column (var0, var12, ...): the column a trigger.gate item names
VAR_COLUMN_PATTERN = r"^var\d+$"
# A series id a trigger.gate item exempts: a digit string, as the input's group names are, so the output name's
# exempt token x<id> never meets an x inside an id
EXEMPT_ID_PATTERN = r"^\d+$"

# =============================================================================
# TRIGGER DETECTION COLUMNS
# =============================================================================

TRIGGER_PLUS_BEGIN = "trigger_plus_begin"
TRIGGER_MINUS_BEGIN = "trigger_minus_begin"
TRIGGER_PLUS_END = "trigger_plus_end"
TRIGGER_MINUS_END = "trigger_minus_end"
ACTIVE_TRIGGER_PLUS_BEGIN = "active_trigger_plus_begin"
ACTIVE_TRIGGER_MINUS_BEGIN = "active_trigger_minus_begin"
TURNING_POINTS = "turning_points"
END_REASON = "end_reason"

# =============================================================================
# TRIGGER DETECTION COLUMN GROUPS AND MAPPINGS
# =============================================================================

BEGIN_COLS = [TRIGGER_PLUS_BEGIN, TRIGGER_MINUS_BEGIN]
END_COLS = [TRIGGER_PLUS_END, TRIGGER_MINUS_END]
ACTIVE_COLS = [ACTIVE_TRIGGER_PLUS_BEGIN, ACTIVE_TRIGGER_MINUS_BEGIN]

BEGIN_TO_END = {
    TRIGGER_PLUS_BEGIN: TRIGGER_PLUS_END,
    TRIGGER_MINUS_BEGIN: TRIGGER_MINUS_END,
}

BEGIN_TO_ACTIVE = {
    TRIGGER_PLUS_BEGIN: ACTIVE_TRIGGER_PLUS_BEGIN,
    TRIGGER_MINUS_BEGIN: ACTIVE_TRIGGER_MINUS_BEGIN,
}

BEGIN_TO_OPPOSITE = {
    TRIGGER_PLUS_BEGIN: TRIGGER_MINUS_BEGIN,
    TRIGGER_MINUS_BEGIN: TRIGGER_PLUS_BEGIN,
}

# =============================================================================
# TRIGGER OUTCOME COLUMNS
# =============================================================================

SERIES_ID = "series_id"
TRIGGER_BEGIN_DATES = "trigger_begin_dates"
TRIGGER_END_DATES = "trigger_end_dates"
TRIGGER_START_VALUES = "trigger_start_values"
TRIGGER_END_VALUES = "trigger_end_values"
IS_PLUS_TRIGGER = "is_plus_trigger"
IS_TRUE = "is_true"
TRIALS_TO_TRUE = "trials_to_true"

# =============================================================================
# SIGNAL GROUPING COLUMNS
# =============================================================================

GROUP_ID = "group_id"
IS_SIGNAL = "is_signal"
SLOPE = "slope"
CORR_TO_SIGNAL = "corr_to_signal"
SERIES_IDS = "series_ids"
NUM_SERIES = "num_series"
# the Signal task's /signals table columns, in order
SIGNAL_TABLE_COLUMNS = [
    SERIES_ID,
    TRIGGER_BEGIN_DATES,
    TRIGGER_END_DATES,
    TRIGGER_START_VALUES,
    TRIGGER_END_VALUES,
    IS_PLUS_TRIGGER,
    IS_TRUE,
    GROUP_ID,
    IS_SIGNAL,
    SLOPE,
    CORR_TO_SIGNAL,
    SERIES_IDS,
    NUM_SERIES,
]

# =============================================================================
# END REASON VALUES
# =============================================================================

END_IDEAL_MODE = "ideal_mode"
END_MAX_DURATION = "max_duration"
END_AGAINST_THRESHOLD = "against_threshold"
END_ALONG_THRESHOLD = "along_threshold"
END_OPPOSITE_TRIGGER_AS_BEGIN = "opposite_trigger_as_begin_strategy"
END_OPPOSITE_TRIGGER_SWAPPED = "opposite_trigger (swapped)"
END_OPPOSITE_TRIGGER = "opposite_trigger"
END_DIFFERENT_END_STRATEGY = "different_end_strategy"

# =============================================================================
# REPLAY MODE VALUES
# =============================================================================

REPLAY_IDEAL = "ideal"
REPLAY_REALISTIC = "realistic"
REPLAY_DUAL = "dual"

# =============================================================================
# WINDOW TYPE VALUES
# =============================================================================

WINDOW_EXPANDING = "expanding"
WINDOW_ROLLING = "rolling"

# =============================================================================
# TRIGGER VALUE MODE VALUES
# =============================================================================

TRIGGER_VALUE_DETECTION_ROW = "detection_row"
TRIGGER_VALUE_FOLLOWING_ROW = "following_row"

# =============================================================================
# GATE OPERATIONS
# =============================================================================
# The comparisons of a trigger.gate item, and their tokens in the output folder name, which never holds > or <
# (Windows forbids both in a folder name).

GATE_OP_GE = ">="
GATE_OP_LE = "<="
GATE_OP_TOKENS = {GATE_OP_GE: "ge", GATE_OP_LE: "le"}
GATE_TOKEN_NONE = "nogate"  # the gate token of an empty item list
GATE_EXEMPT_TOKEN_TEMPLATE = "x{series_id}"  # one per series id an item exempts, after the item's threshold

# =============================================================================
# SIGNAL CRITERION VALUES
# =============================================================================

SIGNAL_EARLIEST_BEGIN = "earliest_begin"
SIGNAL_EARLIEST_BEGIN_HIGHEST_SLOPE = "earliest_begin_highest_slope"

# =============================================================================
# VARIATION METHOD VALUES
# =============================================================================

VARIATION_MULTIPLICATIVE_FIXED = "multiplicative_fixed"
VARIATION_MULTIPLICATIVE_WITH_ACTIVE_SIGNALS = "multiplicative_total_with_active_signals"
VARIATION_MULTIPLICATIVE_BEFORE_ACTIVE_SIGNALS = "multiplicative_total_before_active_signals"

# =============================================================================
# DIRECTION VALUES
# =============================================================================

DIRECTION_PLUS_MINUS = "plus+minus"
DIRECTION_PLUS = "plus"
DIRECTION_MINUS = "minus"
DIRECTIONS = [DIRECTION_PLUS_MINUS, DIRECTION_PLUS, DIRECTION_MINUS]

SUBDIVISION_TRUE_FALSE = "true+false"
SUBDIVISION_TRUE = "true"
SUBDIVISION_FALSE = "false"
SUBDIVISIONS = [SUBDIVISION_TRUE_FALSE, SUBDIVISION_TRUE, SUBDIVISION_FALSE]

# Column names, in sequence, for the 9-cell direction × subdivision grid (3 × 3).
DIRECTION_SUBDIVISION_COLUMNS = [
    f"{d}_{s}" for d in DIRECTIONS for s in SUBDIVISIONS
]

# =============================================================================
# PERFORMANCE SUMMARY TABLE LABELS
# =============================================================================

# Section headers
SIGNAL_METRICS_SECTION = "Signal Metrics"
GRADIENT_METRICS_SECTION = "Gradient Metrics"

# Signals count row label (always emitted as the first signal metrics row)
SIGNALS_COUNT_LABEL = "Signals"

# Gradient metric column headers
GRADIENT_STRATEGY_HEADER = "strategy"
GRADIENT_REFERENCE_HEADER_TEMPLATE = "reference {ref_id}"

# =============================================================================
# METRIC GROUP IDENTIFIERS (match the `group` field in gradient_/metrics.json)
# =============================================================================
METRIC_GROUP_STANDALONE = "standalone"
METRIC_GROUP_RELATIVE = "relative"

# =============================================================================
# GRADIENT METRIC KEY CONSTANTS (match keys in gradient_/metrics.json — SSOT)
# These are the only string literals allowed in Python code for metric keys.
# =============================================================================
METRIC_ACR     = "acr"
METRIC_STD     = "std"
METRIC_STD_DW  = "std_dw"
METRIC_SNR     = "snr"
METRIC_SNR_DW  = "snr_dw"
METRIC_MAX_FALL = "max_fall"
METRIC_FALL_DUR = "fall_dur"
METRIC_BELOW_MAX_DUR = "below_max_dur"
METRIC_RHO     = "rho"
METRIC_SENS    = "sens"
METRIC_SENS_DW = "sens_dw"
METRIC_EXV     = "exv"
METRIC_EXV_DW  = "exv_dw"
METRIC_REF_SNR = "ref_snr"
METRIC_REF_SNR_DW = "ref_snr_dw"

# =============================================================================
# HDF5 KEY PREFIXES
# =============================================================================

HDF5_SERIES_PREFIX = "series_"
HDF5_TRIGGERS_PREFIX = "triggers_"
HDF5_REFERENCE_PREFIX = "reference_series_"
TEMP_STORE_SUFFIX = ".tmp.h5"  # an open_store(..., "w") write goes to <stem>.tmp.h5 until it completes
HDF5_TABLE_INDEX_COLUMN = "index"  # the index column of a format='table' frame, readable alone by HDFStore.select_column

# =============================================================================
# HDF5 GRADIENT PLOT KEYS
# =============================================================================

HDF5_GRADIENT_REFERENCE_VARIATION = "/gradient_reference_variation"
HDF5_GRADIENT_SIGNAL_SEGMENTS = "/gradient_signal_segments"
HDF5_GRADIENT_FILTERED_SEQUENCE = "/gradient_filtered_sequence"
HDF5_GRADIENT_TERMINAL_ATTENUATIONS = "/gradient_terminal_attenuations"
HDF5_GRADIENT_PARAMS = "/gradient_params"
# /gradient_params column: the curve index's first date, the earliest effective start over the processed series
GRADIENT_PARAM_EFFECTIVE_START = "effective_start"
HDF5_GRADIENT_PARAM_SEQUENCES = "/gradient_param_sequences"
HDF5_GRADIENT_METRICS_TABLE = "/gradient_metrics_table"
HDF5_SIGNALS_BREAKDOWN_TABLE = "/signals_breakdown_table"

# Sentinel for the strategy row in /gradient_terminal_attenuations
GRADIENT_STRATEGY_SENTINEL = "__strategy__"

# Kind values for /gradient_param_sequences
GRADIENT_PARAM_SEQ_DISPLAY_SIGNALS = "display_signals"
GRADIENT_PARAM_SEQ_DISPLAY_GRADIENT = "display_gradient"
GRADIENT_PARAM_SEQ_SIGNAL_PLOTS_WHITELIST = "signal_plots_whitelist"

# =============================================================================
# PIPELINE FAILURE RECORD
# =============================================================================

TASK_FAILURE_PREFIX = "Task failed"  # prefix of the pipeline's Luigi FAILURE record

# =============================================================================
# FILE ENCODINGS
# =============================================================================

REPORT_ENCODING = "utf-8"  # the pipeline's own text, never the code page: the preprocess report ("≥", trigger.md:91), git_info.txt and git's captured output and the metric configuration files (metrics.json)

# =============================================================================
# PROVENANCE FILES AND GIT
# =============================================================================

GIT_INFO_FILE = "git_info.txt"  # the git record a completed run writes into its strategy folder
GIT_TIMEOUT_S = 60.0  # bound of each git call; an expiry fails the run (a call takes well under a second)

# =============================================================================
# SUPPRESSION REASONS
# =============================================================================
# Why a realistic-mode candidate is not taken; shown in the debug plots only, never persisted.

SUPPRESSION_BEFORE_EFFECTIVE_START = "before_effective_start"
SUPPRESSION_TEMPORAL_CONSTRAINT = "temporal_constraint"
SUPPRESSION_GATE_BLOCKED = "gate_blocked"
SUPPRESSION_CONSECUTIVE_SAME_TYPE = "consecutive_same_type"
SUPPRESSION_MINIMUM_DURATION = "minimum_duration"
SUPPRESSION_NO_END_CONDITION = "no_end_condition"

# =============================================================================
# GRADIENT FIGURE
# =============================================================================

GRADIENT_LONG_WINDOW_YEARS = 5  # a curve index spanning more than this many calendar years draws the top panel on a symmetric-log axis

# =============================================================================
# CONFIG PROFILES
# =============================================================================
# A profile is the YAML of switches (printouts, plots) beside the strategy YAML,
# <profile>.yaml; load_config merges it, a completed run copies it.

PROFILE_DEV = "dev"
PROFILE_PROD = "prod"
PROFILES = (PROFILE_DEV, PROFILE_PROD)
PROFILE_DEFAULT = PROFILE_PROD  # a run that names no profile

# =============================================================================
# SIGNAL SELECTION LOG RECORD
# =============================================================================

SELECTION_FALLBACK_PAIRS_LOGGED = 5  # the first (signal, join) pairs the date-aligned fallback's summary record names

# =============================================================================
# PLOT BACKEND
# =============================================================================

PLOT_BACKEND = "Agg"  # the plots only save files; each plots package selects it before any pyplot import

# =============================================================================
# LITERAL SPELLINGS
# =============================================================================

CLI_TRUE_VALUES = ("1", "true", "yes", "y")  # the command line's spellings of true (pipeline.py --quiet-luigi)
OPERATION_SUM = "sum"  # the aggregation operations of perseries_/metrics.json
OPERATION_AVG = "avg"
OPERATION_AVGT = "avgt"
OPERATION_MULT = "mult"
OPERATION_STD = "std"
OPERATION_STD_DW = "std_dw"
FROM_DICT_ORIENT = "index"  # pandas DataFrame.from_dict orientation: the dict's keys become the rows

# =============================================================================
# STRATEGY NAMES
# =============================================================================

STRATEGY_EXTREMA = "extrema"  # the extrema (turning points) strategy, trigger.strategy.strategy_name
