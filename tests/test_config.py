"""Tests for pokedata.config module."""

from pathlib import Path
import pytest

from pokedata.config import (
    ConfigError,
    _substitute_env_vars,
    _merge_config,
    load_config_structure,
    resolve_config_variables,
    load_config,
)


class TestSubstituteEnvVars:
    """Tests for _substitute_env_vars function."""

    def test_simple_substitution(self):
        """Test that variables are substituted from provided dict."""
        variables = {"CVAT_USERNAME": "test_user"}
        result = _substitute_env_vars("${CVAT_USERNAME}", variables)
        assert result == "test_user"

    def test_error_when_var_missing(self):
        """Test that missing variables raise ConfigError."""
        variables = {}
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            _substitute_env_vars("${MISSING_VAR}", variables)

    def test_multiple_substitutions(self):
        """Test multiple variable substitutions in one string."""
        variables = {"VAR1": "value1", "VAR2": "value2"}
        result = _substitute_env_vars("${VAR1} and ${VAR2}", variables)
        assert result == "value1 and value2"

    def test_nested_dict_substitution(self):
        """Test variable substitution in nested dictionaries."""
        variables = {"CVAT_USERNAME": "test_user"}
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
                "url": "https://example.com",
            }
        }
        result = _substitute_env_vars(config, variables)
        assert result["cvat"]["username"] == "test_user"
        assert result["cvat"]["url"] == "https://example.com"

    def test_list_substitution(self):
        """Test variable substitution in lists."""
        variables = {"VAR1": "value1", "VAR2": "value2"}
        config = ["${VAR1}", "${VAR2}", "static"]
        result = _substitute_env_vars(config, variables)
        assert result == ["value1", "value2", "static"]

    def test_non_string_values_pass_through(self):
        """Test that non-string values pass through unchanged."""
        variables = {}
        assert _substitute_env_vars(42, variables) == 42
        assert _substitute_env_vars(True, variables) is True
        assert _substitute_env_vars(None, variables) is None

    def test_empty_string(self):
        """Test that empty strings are handled correctly."""
        variables = {}
        result = _substitute_env_vars("", variables)
        assert result == ""

    def test_no_placeholders(self):
        """Test string without placeholders."""
        variables = {}
        result = _substitute_env_vars("plain string", variables)
        assert result == "plain string"

    def test_error_in_nested_dict(self):
        """Test that missing variable in nested dict raises ConfigError."""
        variables = {}
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
            }
        }
        with pytest.raises(
            ConfigError, match="Environment variable 'CVAT_USERNAME' not found"
        ):
            _substitute_env_vars(config, variables)

    def test_error_in_list(self):
        """Test that missing variable in list raises ConfigError."""
        variables = {}
        config = ["${MISSING_VAR}", "static"]
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            _substitute_env_vars(config, variables)


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


class TestLoadConfigStructure:
    """Tests for load_config_structure function."""

    def test_successful_load(self, tmp_path):
        """Test successful loading of config structure."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
datasets:
  structure: coco
cvat:
  url: https://example.com
  username: test_user
"""
        )
        result = load_config_structure(config_file)
        assert result["datasets"]["structure"] == "coco"
        assert result["cvat"]["url"] == "https://example.com"
        assert result["cvat"]["username"] == "test_user"

    def test_missing_file(self, tmp_path):
        """Test that missing config file raises ConfigError."""
        missing_file = tmp_path / "missing.yaml"
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config_structure(missing_file)

    def test_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config_structure(config_file)

    def test_config_not_dict(self, tmp_path):
        """Test that non-dict YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")
        with pytest.raises(ConfigError, match="must contain a YAML dictionary"):
            load_config_structure(config_file)


class TestResolveConfigVariables:
    """Tests for resolve_config_variables function."""

    def test_resolves_from_env_vars(self, tmp_path, monkeypatch):
        """Test that variables are resolved from environment variables."""
        monkeypatch.setenv("CVAT_USERNAME", "env_user")
        monkeypatch.setenv("CVAT_PASSWORD", "env_password")
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
                "password": "${CVAT_PASSWORD}",
            }
        }
        result = resolve_config_variables(config)
        assert result["cvat"]["username"] == "env_user"
        assert result["cvat"]["password"] == "env_password"

    def test_resolves_from_credentials_file(self, tmp_path):
        """Test that variables are resolved from credentials file."""
        credentials_file = tmp_path / "credentials.yaml"
        credentials_file.write_text(
            """
CVAT_USERNAME: file_user
CVAT_PASSWORD: file_password
"""
        )
        config = {
            "cvat": {
                "username": "${CVAT_USERNAME}",
                "password": "${CVAT_PASSWORD}",
            }
        }
        result = resolve_config_variables(config, credentials_file)
        assert result["cvat"]["username"] == "file_user"
        assert result["cvat"]["password"] == "file_password"

    def test_env_vars_override_credentials(self, tmp_path, monkeypatch):
        """Test that environment variables override credentials file."""
        monkeypatch.setenv("CVAT_USERNAME", "env_user")
        credentials_file = tmp_path / "credentials.yaml"
        credentials_file.write_text("CVAT_USERNAME: file_user\n")
        config = {"cvat": {"username": "${CVAT_USERNAME}"}}
        result = resolve_config_variables(config, credentials_file)
        assert result["cvat"]["username"] == "env_user"

    def test_missing_credentials_file_is_ok(self, tmp_path, monkeypatch):
        """Test that missing credentials file is handled gracefully."""
        monkeypatch.setenv("CVAT_USERNAME", "env_user")
        missing_credentials = tmp_path / "missing.yaml"
        config = {"cvat": {"username": "${CVAT_USERNAME}"}}
        result = resolve_config_variables(config, missing_credentials)
        assert result["cvat"]["username"] == "env_user"

    def test_error_on_missing_var(self, tmp_path):
        """Test that missing variable raises ConfigError."""
        config = {"cvat": {"username": "${MISSING_VAR}"}}
        with pytest.raises(
            ConfigError, match="Environment variable 'MISSING_VAR' not found"
        ):
            resolve_config_variables(config)

    def test_credentials_non_string_value_raises_error(self, tmp_path):
        """Test that non-string credential values raise ConfigError."""
        credentials_file = tmp_path / "credentials.yaml"
        credentials_file.write_text("CVAT_USERNAME: 12345\n")
        config = {"cvat": {"username": "${CVAT_USERNAME}"}}
        with pytest.raises(
            ConfigError, match="Credentials value for CVAT_USERNAME have to be a string"
        ):
            resolve_config_variables(config, credentials_file)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_successful_load(self, tmp_path):
        """Test successful loading of config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
datasets:
  structure: coco
cvat:
  url: https://example.com
  username: test_user
"""
        )
        result = load_config(config_file)
        assert result["datasets"]["structure"] == "coco"
        assert result["cvat"]["url"] == "https://example.com"
        assert result["cvat"]["username"] == "test_user"

    def test_load_with_credentials(self, tmp_path):
        """Test loading config with credentials file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cvat:\n  username: ${CVAT_USERNAME}\n")
        credentials_file = tmp_path / "credentials.yaml"
        credentials_file.write_text("CVAT_USERNAME: cred_user\n")
        result = load_config(config_file, credentials_file)
        assert result["cvat"]["username"] == "cred_user"

    def test_load_with_env_vars(self, tmp_path, monkeypatch):
        """Test loading config with environment variables."""
        monkeypatch.setenv("CVAT_USERNAME", "env_user")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cvat:\n  username: ${CVAT_USERNAME}\n")
        result = load_config(config_file)
        assert result["cvat"]["username"] == "env_user"

    def test_env_vars_override_credentials(self, tmp_path, monkeypatch):
        """Test that env vars override credentials in load_config."""
        monkeypatch.setenv("CVAT_USERNAME", "env_user")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("cvat:\n  username: ${CVAT_USERNAME}\n")
        credentials_file = tmp_path / "credentials.yaml"
        credentials_file.write_text("CVAT_USERNAME: cred_user\n")
        result = load_config(config_file, credentials_file)
        assert result["cvat"]["username"] == "env_user"

    def test_missing_file(self, tmp_path):
        """Test that missing config file raises ConfigError."""
        missing_file = tmp_path / "missing.yaml"
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config(missing_file)

    def test_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(config_file)

    def test_config_not_dict(self, tmp_path):
        """Test that non-dict YAML raises ConfigError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("- item1\n- item2")
        with pytest.raises(ConfigError, match="must contain a YAML dictionary"):
            load_config(config_file)


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
