"""
Strategy name generation utilities.

This module provides a centralized way to generate strategy names
based on configuration parameters without instantiating strategy classes.

Structure:
1. Public API (get_strategy_name, get_strategy_parameters)
2. Gate token (_gate_token)
3. Extrema parameter builder (_build_extrema_params)
4. Number token (_number_token)
"""

from typing import Sequence

from .constants import GATE_EXEMPT_TOKEN_TEMPLATE, GATE_OP_TOKENS, GATE_TOKEN_NONE
from .schema import GateItem, StrategyConfig, TriggerConfig

# =============================================================================
# PUBLIC API
# =============================================================================


def get_strategy_name(trigger: TriggerConfig) -> str:
    """Generate a strategy name based on the configuration.

    Args:
        trigger: The trigger section (its gate.items and strategy)

    Returns:
        String representing the strategy name with parameters

    Example:
        For the default workflow (items var12 >= 0.83 exempting series 0 and var15 >= 0.73; extrema strategy with
        window_size=7, min_distance=5, prominence_threshold=0.05):
        Returns: "extrema_var12ge0_83x0-var15ge0_73__w7md5p0_05"
    """
    return f"{trigger.strategy.strategy_name}_{get_strategy_parameters(trigger)}"


def get_strategy_parameters(trigger: TriggerConfig) -> str:
    """Generate the parameters part of the strategy name: the gate token, then the extrema parameters. The name
    identifies the gate list, never trigger.gate.prefilter, which the strategy YAML copy beside the outputs records.

    Args:
        trigger: The trigger section (its gate.items and strategy)

    Returns:
        String representing the parameters (e.g., "var12ge0_83x0-var15ge0_73__w7md5p0_05")
    """
    return f"{_gate_token(trigger.gate.items)}__{_build_extrema_params(trigger.strategy)}"


# =============================================================================
# GATE TOKEN
# =============================================================================


def _gate_token(items: Sequence[GateItem]) -> str:
    """The gate list as configured: the items joined by "-", each {column}{ge|le}{threshold} then x<id> per exempt
    series id; "nogate" for an empty list. An exempt id is a digit string, so x never occurs inside one and two lists
    never share a token; it holds no > or < (Windows forbids both in a folder name)."""
    if not items:
        return GATE_TOKEN_NONE
    return "-".join(
        f"{item.column}{GATE_OP_TOKENS[item.op]}{_number_token(item.threshold)}"
        f"{''.join(GATE_EXEMPT_TOKEN_TEMPLATE.format(series_id=series_id) for series_id in item.exempt_series)}"
        for item in items
    )


# =============================================================================
# EXTREMA PARAMETER BUILDER
# =============================================================================


def _build_extrema_params(strategy: StrategyConfig) -> str:
    """Build the parameter string of the extrema strategy, the only strategy the schema allows."""
    return f"w{strategy.window_size}md{strategy.min_distance}p{_number_token(strategy.prominence_threshold)}"


# =============================================================================
# NUMBER TOKEN
# =============================================================================


def _number_token(value: float) -> str:
    """A number as a folder token: its str with each "." replaced by "_" (0.05 -> 0_05)."""
    return str(value).replace(".", "_")
