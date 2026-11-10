"""
The trigger strategy a config names.

Structure:
1. Strategy creation (create_strategy)
"""

from pathlib import Path

from ..common.schema import TriggerConfig, strategy_class


# =============================================================================
# STRATEGY CREATION
# =============================================================================


def create_strategy(trigger: TriggerConfig, output_file: Path):
    """The trigger strategy trigger.strategy names, built from the class schema.strategy_class registers for it;
    output_file is the trigger task's output."""
    return strategy_class(trigger.strategy.strategy_name)(trigger, output_file)
