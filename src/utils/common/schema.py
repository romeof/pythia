from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    EXEMPT_ID_PATTERN,
    GATE_OP_GE,
    GATE_OP_LE,
    REPLAY_DUAL,
    REPLAY_IDEAL,
    REPLAY_REALISTIC,
    SIGNAL_EARLIEST_BEGIN,
    SIGNAL_EARLIEST_BEGIN_HIGHEST_SLOPE,
    STRATEGY_EXTREMA,
    TRIGGER_VALUE_FOLLOWING_ROW,
    TRIGGER_VALUE_DETECTION_ROW,
    VALUE_COLUMNS,
    VAR_COLUMN_PATTERN,
    VARIATION_MULTIPLICATIVE_BEFORE_ACTIVE_SIGNALS,
    VARIATION_MULTIPLICATIVE_FIXED,
    VARIATION_MULTIPLICATIVE_WITH_ACTIVE_SIGNALS,
    WINDOW_EXPANDING,
    WINDOW_ROLLING,
)

# Canonical gradient metric display order — derived once from metrics.json at import time.
# Adding a metric to metrics.json automatically includes it in the default.
_GRADIENT_METRICS_JSON = Path(__file__).parent.parent / "performance" / "gradient_" / "metrics.json"
_DEFAULT_GRADIENT_SEQUENCE: List[str] = list(json.loads(_GRADIENT_METRICS_JSON.read_text()).keys())

# =============================================================================
# MODEL BASE
# =============================================================================


class ConfigModel(BaseModel):
    """The base of every configuration model: an unknown key is refused (extra "forbid"), an attribute assignment is
    refused (frozen), and the hash is the content's, hash(model_dump_json()): luigi's execution summary puts a task's
    parameter values in a set, and pydantic's own frozen hash fails on a list field. Frozen does not make a section
    immutable: its list fields stay lists, which no code changes in place, since every task of a run shares the
    sections and luigi derived the task id from their content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def __hash__(self) -> int:
        return hash(self.model_dump_json())


# =============================================================================
# READ SECTION (user config)
# =============================================================================


class ReadConfig(ConfigModel):
    raw_data: str
    # Outputs go to <output_root>/<date>/<dataset>/..., with <date> and <dataset> named after raw_data's folders
    output_root: str
    ini_date: str
    end_date: str
    reference_series: List[str] = Field(default_factory=list)
    skip_series: List[str]


# =============================================================================
# TRIGGER SECTION (user + profile merged)
# =============================================================================


class EndConfig(ConfigModel):
    """End configuration: each parameter is self-documenting.
    Set a value to enable, null/false to disable.
    The checking sequence is fixed in code: against_threshold > along_threshold > opposite_trigger > different_end > max_duration
    """

    # Maximum duration in periods before forced end (time bound)
    max_duration_periods: Optional[int] = Field(default=None, ge=1)
    # Relative change against the trigger direction, from the begin value, that ends the trigger
    against_threshold_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Relative change along the trigger direction, from the begin value, that ends the trigger
    along_threshold_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # End on opposite trigger
    opposite_trigger_as_begin_strategy: bool = Field(default=False)
    # Use strategy-specific end logic
    different_end_strategy: bool = Field(default=False)


class ReplayConfig(ConfigModel):
    # Replay mode: "ideal" uses full series (future rows visible), "realistic" uses incremental windows (no future rows), "dual" runs both
    mode: Literal[REPLAY_IDEAL, REPLAY_REALISTIC, REPLAY_DUAL]
    # Burn-in period: number of rows to wait before starting analysis
    lookback_window: int = Field(ge=1)
    # Window type for realistic mode
    window_type: Literal[WINDOW_EXPANDING, WINDOW_ROLLING] = Field(default=WINDOW_EXPANDING)
    # Minimum duration: a trigger cannot end until this many periods have elapsed
    min_duration_periods: int = Field(ge=0, default=1)
    # End configuration
    end: EndConfig


class DebugFlags(ConfigModel):
    """Common debug flags for task/sub-task sections in the profile YAML."""

    printout_io: bool = Field(default=False)


class StrategyConfig(ConfigModel):
    """Strategy parameters (user) merged with debug flags (profile)."""

    # User params
    strategy_name: Literal[STRATEGY_EXTREMA]
    window_size: int = Field(ge=1)
    min_distance: int = Field(ge=1)
    prominence_threshold: float = Field(gt=0)
    width: float = Field(gt=0)
    # Profile params (merged from the profile YAML)
    printout_io: bool = Field(default=False)
    printout_debug: bool = Field(default=False)
    full_output: bool = Field(default=False)
    plot_debug: bool = Field(default=False)


# A value column of the input, the columns a validity rule lists
ValueColumn = Literal[VALUE_COLUMNS]


class ValidityConfig(ConfigModel):
    """Row validity rules of the preprocessing, applied in field sequence after the NaN check that always applies: a
    row that fails one is removed; an empty list applies no rule."""

    # A row with a negative value in a listed column fails
    nonnegative: List[ValueColumn] = Field(default_factory=list)
    # A row with a zero in a listed column fails
    nonzero: List[ValueColumn] = Field(default_factory=list)
    # A row whose listed columns all hold one value fails
    not_all_equal: List[ValueColumn] = Field(default_factory=list)

    @field_validator("not_all_equal")
    @classmethod
    def _two_distinct_columns(cls, v: List[str]) -> List[str]:
        """One distinct column always equals itself, so the rule would remove every row that holds a value."""
        if len(set(v)) == 1:
            raise ValueError(f"not_all_equal lists one distinct column {sorted(set(v))}: give two or more, or none")
        return v


class GateItem(ConfigModel):
    """One item of trigger.gate.items: a begin at row i is allowed only when row i-1 passes every item that applies
    to the series (NaN fails; a first row fails); an item applies to every series its exempt_series does not name."""

    # A positional input column (var0, var12, ...)
    column: str = Field(pattern=VAR_COLUMN_PATTERN)
    op: Literal[GATE_OP_GE, GATE_OP_LE]
    threshold: float = Field(allow_inf_nan=False)
    # The series ids, digit strings as the input's group names are, the item does not apply to
    exempt_series: List[Annotated[str, Field(pattern=EXEMPT_ID_PATTERN)]] = Field(default_factory=list)


class GateConfig(ConfigModel):
    """The gate on the previous row (trigger.gate): its items and the pre-filter switch."""

    items: List[GateItem] = Field(default_factory=list)
    # skip a series one applicable item passes on no row of the read window; never a reference series
    prefilter: bool = Field(default=False)


class TriggerConfig(ConfigModel):
    # The gate on the previous row: the items a begin's previous row must pass, and the pre-filter
    gate: GateConfig = Field(default_factory=GateConfig)
    # Minimum series length required for processing
    min_length: int = Field(ge=1)
    # Number of consecutive rows with bad preprocessed data before series is rejected
    consecutive_rows_with_bad_preprocessed_data: int = Field(ge=1)
    # Selects which value to use for trigger begin/end values
    trigger_value: Literal[TRIGGER_VALUE_DETECTION_ROW, TRIGGER_VALUE_FOLLOWING_ROW]
    # Row validity rules of the preprocessing
    validity: ValidityConfig = Field(default_factory=ValidityConfig)
    # Strategy-specific parameters
    strategy: StrategyConfig
    # Replay configuration
    replay: ReplayConfig
    # Profile sub-sections (merged from the profile YAML)
    read: DebugFlags = Field(default_factory=lambda: DebugFlags())
    preprocess: DebugFlags = Field(default_factory=lambda: DebugFlags())

    @model_validator(mode="after")
    def _lookback_covers_min_length(self) -> "TriggerConfig":
        """A lookback window shorter than min_length would skip every series: a configuration error, rejected at startup (E-7)."""
        if self.replay.lookback_window < self.min_length:
            raise ValueError(
                f"trigger.replay.lookback_window ({self.replay.lookback_window}) < trigger.min_length ({self.min_length})"
            )
        return self


# =============================================================================
# SIGNAL SECTION (user + profile merged)
# =============================================================================


class InputDevConfig(ConfigModel):
    """Dev flags for signal.input."""

    printout_io: bool = Field(default=False)
    full_output: bool = Field(default=False)


class SelectionDevConfig(ConfigModel):
    """Dev flags for signal.selection."""

    printout_io: bool = Field(default=False)
    full_output: bool = Field(default=False)


class SignalConfig(ConfigModel):
    corr_threshold: float
    period_back: int = Field(ge=1)
    signal_criterion: Literal[SIGNAL_EARLIEST_BEGIN, SIGNAL_EARLIEST_BEGIN_HIGHEST_SLOPE]
    # Dev sub-sections
    input: InputDevConfig = Field(default_factory=InputDevConfig)
    selection: SelectionDevConfig = Field(default_factory=SelectionDevConfig)

    @field_validator("corr_threshold")
    @classmethod
    def _corr_range(cls, v: float) -> float:
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("corr_threshold must be in [0.0, 1.0]")
        return float(v)


# =============================================================================
# PERFORMANCE SECTION (user + profile merged)
# =============================================================================


class PerformancePerSeriesTriggerConfig(ConfigModel):
    # Dev params
    printout_io: bool = Field(default=False)
    full_output: bool = Field(default=False)
    plot_debug: bool = Field(default=False)


class PerformancePerSeriesSignalConfig(ConfigModel):
    # Dev params
    printout_io: bool = Field(default=False)
    full_output: bool = Field(default=False)
    plot_debug: bool = Field(default=False)


class PerformanceGradientConfig(ConfigModel):
    """Config for performance.gradient."""

    # User params
    weight: float
    # Weight mechanism at the aggregate level (not per-event compounding, which is always multiplicative/geometric)
    variation_method: Literal[
        VARIATION_MULTIPLICATIVE_FIXED,
        VARIATION_MULTIPLICATIVE_WITH_ACTIVE_SIGNALS,
        VARIATION_MULTIPLICATIVE_BEFORE_ACTIVE_SIGNALS,
    ]
    # Upper bound on the summed weights of concurrently active signals
    concurrent_weight_limit: float
    # Fraction of the yearly increase removed [0.0, 1.0] (0.0 = no attenuation)
    yearly_attenuation_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    # Month (1-12) when the yearly attenuation is applied
    attenuation_month: int = Field(default=1, ge=1, le=12)
    # Deduction as percentage applied at each event boundary, begin and end (0.0 = disabled)
    boundary_deduction_pct: float = Field(default=0.0, ge=0.0)
    # Minimal positive variation rate (decimal). 0.0 = disabled. Max allowed: 0.25
    mpv: float = Field(default=0.0, ge=0.0)

    @field_validator("mpv")
    @classmethod
    def _mpv_range(cls, v: float) -> float:
        if v > 0.25:
            raise ValueError("mpv must be in [0.0, 0.25]; 0.25 is the highest value allowed")
        return v
    # Dev flags
    printout_io: bool = Field(default=False)
    printout_debug: bool = Field(default=False)
    full_output: bool = Field(default=False)
    plot_debug: bool = Field(default=False)
    max_plots: int = Field(ge=0, default=100)
    plots_whitelist: List[str] = Field(default_factory=list)
    # Maximum number of reference series accepted by the pipeline (0 = no references)
    max_references: int = Field(default=3, ge=0)

    @field_validator("weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not (0.0 < float(v) <= 1.0):
            raise ValueError("weight must be in (0.0, 1.0]")
        return float(v)

    @field_validator("concurrent_weight_limit")
    @classmethod
    def _concurrent_weight_limit_range(cls, v: float) -> float:
        if not (0.0 < float(v) <= 1.0):
            raise ValueError("concurrent_weight_limit must be in (0.0, 1.0]")
        return float(v)


class PerformanceMetricsConfig(ConfigModel):
    # Sequence of signal metric composite keys to compute and display
    display_order_signals: List[str]
    # Sequence of gradient metric keys to compute and display.
    # Defaults to all keys in metrics.json in insertion order.
    display_order_gradient: List[str] = Field(default_factory=lambda: list(_DEFAULT_GRADIENT_SEQUENCE))


class PerformanceConfig(ConfigModel):
    perseries_trigger: PerformancePerSeriesTriggerConfig
    perseries_signal: PerformancePerSeriesSignalConfig
    gradient: PerformanceGradientConfig
    metrics: PerformanceMetricsConfig


# =============================================================================
# GENERAL & RUNTIME SECTIONS (profile config)
# =============================================================================


class GeneralConfig(ConfigModel):
    # INFO (20) - General information
    enable_info: bool


class RuntimeConfig(ConfigModel):
    # Start with 32 and increase if needed
    queue_maxsize: int = Field(ge=1)
    queue_get_timeout: int = Field(ge=1)


# =============================================================================
# ROOT CONFIG (merged from strategy.yaml + the profile YAML)
# =============================================================================


class AppConfig(ConfigModel):
    read: ReadConfig
    trigger: TriggerConfig
    signal: SignalConfig
    performance: PerformanceConfig
    general: GeneralConfig
    runtime: RuntimeConfig


def strategy_class(strategy_name: str) -> type:
    """The strategy class registered for trigger.strategy.strategy_name (the one place a new strategy is added,
    beside the Literal of StrategyConfig.strategy_name); a name outside the registry raises a KeyError naming it.
    The class is imported here: a module-level import would complete the cycle schema -> strategies.extrema ->
    schema (extrema.py imports schema, and so do detection.py and realistic.py, which it loads)."""
    from ..trigger.strategies.extrema import ExtremaTrigger

    registry = {STRATEGY_EXTREMA: ExtremaTrigger}
    if strategy_name not in registry:
        raise KeyError(f"Unknown strategy: {strategy_name}; registered: {sorted(registry)}")
    return registry[strategy_name]
