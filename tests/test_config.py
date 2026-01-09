"""Tests for pokedata.config module."""

from pathlib import Path
import pytest

from pokedata.config import (
    ConfigError,
    _substitute_env_vars,
    _merge_config,
    load_config,
)


class TestSubstituteEnvVars:
    """Tests for _substitute_env_vars function."""

    def test_stage_specific_env_var(self, monkeypatch):
        """Test that stage-specific env vars are substituted."""
        monkeypatch.setenv("CVAT_USERNAME_DEV", "dev_user")
        result = _substitute_env_vars("${CVAT_USERNAME}", "dev")
        assert result == "dev_user"

    def test_generic_fallback_env_var(self, monkeypatch):
        """Test that generic env vars are used as fallback."""
        monkeypatch.setenv("CVAT_USERNAME", "generic_user")
        result = _substitute_env_vars("${CVAT_USERNAME}", "prod")
        assert result == "generic_user"

    def test_stage_specific_precedence(self, monkeypatch):
        """Test that stage-specific env vars take precedence over generic."""
        monkeypatch.setenv("CVAT_USERNAME_DEV", "dev_user")
        monkeypatch.setenv("CVAT_USERNAME", "generic_user")
        result = _substitute_env_vars("${CVAT_USERNAME}", "dev")
        assert result == "dev_user"

    def test_error_when_env_var_missing(self):
        """Test that missing env vars raise ConfigError."""
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            _substitute_env_vars("${MISSING_VAR}", "dev")

    def test_multiple_substitutions(self, monkeypatch):
        """Test multiple env var substitutions in one string."""
        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        result = _substitute_env_vars("${VAR1} and ${VAR2}", "dev")
        assert result == "value1 and value2"

    def test_nested_dict_substitution(self, monkeypatch):
        """Test env var substitution in nested dictionaries."""
        monkeypatch.setenv("CVAT_USERNAME_DEV", "dev_user")
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
                "url": "https://example.com",
            }
        }
        result = _substitute_env_vars(config, "dev")
        assert result["cvat"]["username"] == "dev_user"
        assert result["cvat"]["url"] == "https://example.com"

    def test_list_substitution(self, monkeypatch):
        """Test env var substitution in lists."""
        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        config = ["${VAR1}", "${VAR2}", "static"]
        result = _substitute_env_vars(config, "dev")
        assert result == ["value1", "value2", "static"]

    def test_non_string_values_pass_through(self):
        """Test that non-string values pass through unchanged."""
        assert _substitute_env_vars(42, "dev") == 42
        assert _substitute_env_vars(True, "dev") is True
        assert _substitute_env_vars(None, "dev") is None

    def test_empty_string(self):
        """Test that empty strings are handled correctly."""
        result = _substitute_env_vars("", "dev")
        assert result == ""

    def test_no_placeholders(self):
        """Test string without placeholders."""
        result = _substitute_env_vars("plain string", "dev")
        assert result == "plain string"

    def test_error_in_nested_dict(self, monkeypatch):
        """Test that missing env var in nested dict raises ConfigError."""
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
            }
        }
        with pytest.raises(
            ConfigError, match="Environment variable 'CVAT_USERNAME' not found"
        ):
            _substitute_env_vars(config, "dev")

    def test_error_in_list(self):
        """Test that missing env var in list raises ConfigError."""
        config = ["${MISSING_VAR}", "static"]
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            _substitute_env_vars(config, "dev")

    def test_error_message_includes_both_attempts(self):
        """Test that error message shows both stage-specific and generic attempts."""
        with pytest.raises(ConfigError) as exc_info:
            _substitute_env_vars("${VAR_NAME}", "dev")
        error_msg = str(exc_info.value)
        assert "VAR_NAME_DEV" in error_msg
        assert "VAR_NAME" in error_msg


class TestMergeConfig:
    """Tests for _merge_config function."""

    def test_simple_merge(self):
        """Test simple key-value merge."""
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "new_value2", "key3": "value3"}
        result = _merge_config(base, override)
        assert result == {"key1": "value1", "key2": "new_value2", "key3": "value3"}

    def test_deep_merge(self):
        """Test deep merge of nested dictionaries."""
        base = {
            "cvat": {
                "url": "https://base.example.com",
                "username": "base_user",
            },
            "datasets": {"output_dir": "data"},
        }
        override = {
            "cvat": {
                "username": "override_user",
            },
            "dvc": {"repo_path": "."},
        }
        result = _merge_config(base, override)
        assert result["cvat"]["url"] == "https://base.example.com"
        assert result["cvat"]["username"] == "override_user"
        assert result["datasets"]["output_dir"] == "data"
        assert result["dvc"]["repo_path"] == "."

    def test_override_with_non_dict(self):
        """Test that non-dict values override dict values."""
        base = {"key": {"nested": "value"}}
        override = {"key": "simple_value"}
        result = _merge_config(base, override)
        assert result["key"] == "simple_value"

    def test_empty_base(self):
        """Test merging with empty base config."""
        base = {}
        override = {"key": "value"}
        result = _merge_config(base, override)
        assert result == {"key": "value"}

    def test_empty_override(self):
        """Test merging with empty override config."""
        base = {"key": "value"}
        override = {}
        result = _merge_config(base, override)
        assert result == {"key": "value"}

    def test_multiple_levels_nesting(self):
        """Test merge with multiple levels of nesting."""
        base = {
            "level1": {
                "level2": {
                    "level3": "base_value",
                    "other": "base_other",
                }
            }
        }
        override = {
            "level1": {
                "level2": {
                    "level3": "override_value",
                }
            }
        }
        result = _merge_config(base, override)
        assert result["level1"]["level2"]["level3"] == "override_value"
        assert result["level1"]["level2"]["other"] == "base_other"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_successful_load(self, tmp_path):
        """Test successful loading of config with base and stage."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
datasets:
  structure: coco

stages:
  dev:
    cvat:
      url: https://dev.example.com
      username: dev_user
    datasets:
      output_dir: data/dev
  prod:
    cvat:
      url: https://prod.example.com
      username: prod_user
    datasets:
      output_dir: data/prod
"""
        )
        result = load_config(config_file, "dev")
        assert result["datasets"]["structure"] == "coco"
        assert result["cvat"]["url"] == "https://dev.example.com"
        assert result["cvat"]["username"] == "dev_user"
        assert result["datasets"]["output_dir"] == "data/dev"
        assert result["_stage"] == "dev"

    def test_missing_file(self, tmp_path):
        """Test that missing config file raises ConfigError."""
        missing_file = tmp_path / "missing.yaml"
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config(missing_file, "dev")

    def test_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(config_file, "dev")

    def test_missing_stage(self, tmp_path):
        """Test that missing stage raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages:
  dev:
    key: value
"""
        )
        with pytest.raises(ConfigError, match="Stage 'prod' not found"):
            load_config(config_file, "prod")

    def test_config_not_dict(self, tmp_path):
        """Test that non-dict YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")
        with pytest.raises(ConfigError, match="must contain a YAML dictionary"):
            load_config(config_file, "dev")

    def test_stages_not_dict(self, tmp_path):
        """Test that stages not being a dict raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages: not_a_dict
"""
        )
        with pytest.raises(ConfigError, match="'stages' must be a dictionary"):
            load_config(config_file, "dev")

    def test_stage_config_not_dict(self, tmp_path):
        """Test that stage config not being a dict raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages:
  dev: not_a_dict
"""
        )
        with pytest.raises(
            ConfigError, match="Stage 'dev' configuration must be a dictionary"
        ):
            load_config(config_file, "dev")

    def test_env_var_substitution_in_load(self, tmp_path, monkeypatch):
        """Test that env vars are substituted when loading config."""
        monkeypatch.setenv("CVAT_USERNAME_DEV", "env_dev_user")
        monkeypatch.setenv("CVAT_PASSWORD", "env_password")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages:
  dev:
    cvat:
      username: ${CVAT_USERNAME}
      password: ${CVAT_PASSWORD}
"""
        )
        result = load_config(config_file, "dev")
        assert result["cvat"]["username"] == "env_dev_user"
        assert result["cvat"]["password"] == "env_password"

    def test_base_config_merged_with_stage(self, tmp_path):
        """Test that base config is merged with stage config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
datasets:
  structure: coco
  default_format: json

stages:
  dev:
    datasets:
      output_dir: data/dev
"""
        )
        result = load_config(config_file, "dev")
        assert result["datasets"]["structure"] == "coco"
        assert result["datasets"]["default_format"] == "json"
        assert result["datasets"]["output_dir"] == "data/dev"

    def test_stage_parameter_required(self, tmp_path):
        """Test that stage parameter is required."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages:
  dev:
    key: value
"""
        )
        result = load_config(config_file, "dev")
        assert result["_stage"] == "dev"
        assert result["key"] == "value"

    def test_load_config_raises_error_on_missing_env_var(self, tmp_path):
        """Test that load_config raises ConfigError when env var is missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
stages:
  dev:
    cvat:
      username: ${MISSING_VAR}
"""
        )
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            load_config(config_file, "dev")


class TestDirectConfigAccess:
    """Tests for direct dictionary access to configuration."""

    def test_simple_key(self):
        """Test getting a simple top-level key."""
        config = {"key": "value"}
        assert config["key"] == "value"

    def test_nested_key(self):
        """Test getting a nested key using direct dictionary access."""
        config = {
            "cvat": {
                "url": "https://example.com",
                "auth": {
                    "username": "user",
                },
            }
        }
        assert config["cvat"]["url"] == "https://example.com"
        assert config["cvat"]["auth"]["username"] == "user"

    def test_missing_key_raises_keyerror(self):
        """Test that missing key raises KeyError."""
        config = {"key": "value"}
        with pytest.raises(KeyError):
            _ = config["missing"]

    def test_missing_intermediate_key_raises_keyerror(self):
        """Test that missing intermediate key raises KeyError."""
        config = {"cvat": {"url": "https://example.com"}}
        with pytest.raises(KeyError):
            _ = config["cvat"]["missing"]
        with pytest.raises(KeyError):
            _ = config["missing"]["key"]

    def test_deeply_nested_key(self):
        """Test getting a deeply nested key."""
        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "deep_value",
                    }
                }
            }
        }
        assert config["level1"]["level2"]["level3"]["level4"] == "deep_value"

    def test_non_dict_intermediate_value_raises_typeerror(self):
        """Test that accessing non-dict intermediate value raises TypeError."""
        config = {"key": "string_value"}
        with pytest.raises(TypeError):
            _ = config["key"]["nested"]


class TestConfigError:
    """Tests for ConfigError exception."""

    def test_config_error_is_exception(self):
        """Test that ConfigError is an Exception."""
        assert issubclass(ConfigError, Exception)

    def test_config_error_can_be_raised(self):
        """Test that ConfigError can be raised and caught."""
        with pytest.raises(ConfigError):
            raise ConfigError("Test error")
