"""Configuration loading with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union
import yaml
from loguru import logger

# Type alias for configuration dictionaries
ConfigDict = Dict[str, Any]


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


def _substitute_env_vars(value: Any, variables: Dict[str, str]) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Substitutes ${VAR_NAME} placeholders with values from the variables dict.

    Args:
        value: Configuration value (string, dict, list, etc.)
        variables: Dictionary mapping variable names to their values

    Returns:
        Value with environment variables substituted

    Raises:
        ConfigError: If an environment variable placeholder is found but the
            variable is not set in the variables dict
    """
    if isinstance(value, str):
        # Pattern to match ${VAR_NAME}
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return variables[var_name]
            # If not found, raise an error
            raise ConfigError(
                f"Environment variable '{var_name}' not found in variables dict."
            )

        return re.sub(pattern, replace_var, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v, variables) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item, variables) for item in value]
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


def load_config_structure(config_path: Path) -> ConfigDict:
    """
    Load configuration structure from YAML file without variable substitution.

    Loads the YAML file and returns the configuration dictionary as-is.
    Use this when you need the raw config structure (e.g., for loading credentials files).

    Args:
        config_path: Path to config YAML file.

    Returns:
        Configuration dictionary (without substitution)

    Raises:
        ConfigError: If config file is missing or invalid
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

    return config


def resolve_config_variables(
    config: ConfigDict, credentials_path: Path = None
) -> ConfigDict:
    """
    Resolve variable substitutions in a configuration dictionary.

    Loads credentials from file and environment variables, then performs
    substitution of ${VAR_NAME} placeholders in the config.

    Environment variable substitution:
    - Variables use simple names: ${CVAT_USERNAME}
    - Credentials can be loaded from a credentials YAML file (flat format)
    - Environment variables take precedence over credentials file values

    Args:
        config: Configuration dictionary (typically from load_config_structure)
        credentials_path: Optional path to credentials YAML file.
            If None, no credentials file is loaded.

    Returns:
        Configuration dictionary with variables substituted

    Raises:
        ConfigError: If a required variable is not found
    """
    # Load credentials from file if available
    credentials_yaml = {}
    if credentials_path is not None:
        try:
            credentials_yaml = load_config_structure(credentials_path)
            # Ensure values are strings, errors if not
            for key, value in credentials_yaml.items():
                if not isinstance(value, str):
                    raise ConfigError(
                        f"Credentials value for {key} have to be a string"
                    )
        except ConfigError as e:
            # Check if it's a validation error (non-string) or a loading error (missing/invalid file)
            if "have to be a string" in str(e):
                # Re-raise validation errors - these are not optional
                raise
            # Log warning but don't fail - credentials file is optional
            logger.warning(
                f"Failed to load credentials file {credentials_path}: {e}, using only environment variables"
            )
            credentials_yaml = {}

    env_vars = dict(os.environ)
    # Merge: env vars override credentials (env vars take precedence)
    credentials = {**credentials_yaml, **env_vars}
    return _substitute_env_vars(config, credentials)


def load_config(config_path: Path, credentials_path: Path = None) -> ConfigDict:
    """
    Load configuration from YAML file with variable substitution.

    This is a convenience function that combines load_config_structure and
    resolve_config_variables. It loads the config structure, then resolves all
    variable substitutions.

    You can use different config files for different environments. For example:
    - config.dev.yaml with secrets.dev.yaml for dev environment
    - config.prod.yaml with secrets.prod.yaml for prod environment

    Environment variable substitution:
    - Variables use simple names: ${CVAT_USERNAME}
    - Credentials can be loaded from a credentials YAML file (flat format)
    - Environment variables take precedence over credentials file values

    Args:
        config_path: Path to config YAML file.
        credentials_path: Optional path to credentials YAML file.
            If None, no credentials file is loaded.

    Returns:
        Configuration dictionary with variables resolved

    Raises:
        ConfigError: If config file is missing or invalid
    """
    config = load_config_structure(config_path)
    resolved_config = resolve_config_variables(config, credentials_path)
    return resolved_config
