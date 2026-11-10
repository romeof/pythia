"""
Configuration loading and validation: the strategy YAML merged with its profile beside it, the strict
validation into the typed sections every task takes, and the cached metric configuration files (metrics.json).

Structure:
1. Configuration loading and validation (load_config, profile_config_file, _deep_merge, validate_config,
   load_json_config)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from .constants import REPORT_ENCODING
from .schema import AppConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION LOADING AND VALIDATION
# =============================================================================


def load_config(config_file: str, profile: str) -> Dict[str, Any]:
    """Load the strategy config and the profile config beside it, then deep-merge into a single dict."""
    with open(config_file, "r") as f:
        user_config = yaml.safe_load(f)

    with open(profile_config_file(config_file, profile), "r") as f:
        profile_config = yaml.safe_load(f)

    return _deep_merge(user_config, profile_config)


def profile_config_file(config_file: str, profile: str) -> Path:
    """The <profile>.yaml beside a strategy YAML: the file load_config merges and copy_config_to_output copies."""
    return Path(config_file).parent / f"{profile}.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values win for scalar conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def validate_config(cfg: Dict[str, Any]) -> AppConfig:
    """Validate the entire configuration strictly (unknown keys rejected) and return the validated model, the
    schema's declared defaults applied: the typed sections every task takes."""
    validated = AppConfig.model_validate(cfg)
    logger.info("Configuration validation passed")
    return validated


_JSON_CONFIG_CACHE: Dict[str, Dict] = {}


def load_json_config(caller_file: str) -> Dict:
    """Load and cache a JSON config file named metrics.json alongside the caller module.

    Args:
        caller_file: __file__ of the calling module (resolves metrics.json path).

    Returns:
        Parsed JSON dictionary, cached after first load.
    """
    config_path = str(Path(caller_file).with_name("metrics.json"))
    cached = _JSON_CONFIG_CACHE.get(config_path)
    if cached is not None:
        return cached
    try:
        with open(config_path, "r", encoding=REPORT_ENCODING) as f:
            _JSON_CONFIG_CACHE[config_path] = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load metric configuration from {config_path}: {e}"
        )
    return _JSON_CONFIG_CACHE[config_path]
