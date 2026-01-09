"""Configuration loading with multi-stage support and environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union
import yaml

# Type alias for configuration dictionaries
ConfigDict = Dict[str, Any]


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def _substitute_env_vars(value: Any, stage: str) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Supports stage-specific env vars (e.g., ${CVAT_USERNAME_DEV}) with fallback
    to generic env vars (e.g., ${CVAT_USERNAME}).

    Args:
        value: Configuration value (string, dict, list, etc.)
        stage: Current stage name (e.g., 'dev', 'prod')

    Returns:
        Value with environment variables substituted

    Raises:
        ConfigError: If an environment variable placeholder is found but the
            variable is not set in the environment
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME} or ${VAR_NAME_STAGE}
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            # Try stage-specific first (e.g., CVAT_USERNAME_DEV)
            stage_var = f"{var_name}_{stage.upper()}"
            if stage_var in os.environ:
                return os.environ[stage_var]
            # Fallback to generic (e.g., CVAT_USERNAME)
            if var_name in os.environ:
                return os.environ[var_name]
            # If not found, raise an error
            raise ConfigError(
                f"Environment variable '{var_name}' not found. "
                f"Tried stage-specific '{stage_var}' and generic '{var_name}'."
            )

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v, stage) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item, stage) for item in value]
    else:
        return value


def _merge_config(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """
    Deep merge two configuration dictionaries. Merging nested dictionaries.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged configuration dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: Path, stage: str) -> ConfigDict:
    """
    Load configuration from YAML file with multi-stage support.

    Configuration structure:
    - Base config: shared settings across stages
    - Stage-specific configs: `stages.<stage>` with overrides

    Environment variable substitution:
    - Stage-specific: ${CVAT_USERNAME_DEV} (for stage='dev')
    - Generic fallback: ${CVAT_USERNAME}

    Args:
        config_path: Path to config YAML file.
        stage: Configuration stage to load

    Returns:
        Merged configuration dictionary for the specified stage

    Raises:
        ConfigError: If config file is missing, invalid, or stage not found
    """
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML configuration: {e}")
    except Exception as e:
        raise ConfigError(f"Failed to read configuration file: {e}")

    if not isinstance(config, dict):
        raise ConfigError("Configuration file must contain a YAML dictionary")

    # Extract base config (everything except 'stages')
    base_config: ConfigDict = {k: v for k, v in config.items() if k != "stages"}

    # Extract stage-specific config
    stages = config.get("stages", {})
    if not isinstance(stages, dict):
        raise ConfigError("'stages' must be a dictionary in configuration file")

    if stage not in stages:
        raise ConfigError(
            f"Stage '{stage}' not found in configuration. "
            f"Available stages: {', '.join(stages.keys())}"
        )

    stage_config = stages[stage]
    if not isinstance(stage_config, dict):
        raise ConfigError(f"Stage '{stage}' configuration must be a dictionary")

    merged_config = _merge_config(base_config, stage_config)
    merged_config = _substitute_env_vars(merged_config, stage)
    merged_config["_stage"] = stage

    return merged_config


def get_config_value(config: ConfigDict, key_path: str) -> Any:
    """
    Get a configuration value using dot-notation path.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'cvat.url', 'datasets.output_dir')

    Returns:
        Configuration value

    Raises:
        ConfigError: If the key path is not found in the configuration
    """
    keys = key_path.split(".")
    value: Any = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ConfigError(f"Configuration key '{key_path}' not found")

    return value
