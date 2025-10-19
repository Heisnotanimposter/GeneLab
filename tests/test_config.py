"""Tests for configuration management."""

import pytest
from genelab.config.config_loader import ConfigLoader


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def test_default_config(self):
        """Test default configuration."""
        config = ConfigLoader()
        assert config.get("data.base_path") == "./data"
        assert config.get("training.epochs") == 50

    def test_get_nested_key(self):
        """Test getting nested configuration keys."""
        config = ConfigLoader()
        value = config.get("data.base_path")
        assert value == "./data"

    def test_get_missing_key(self):
        """Test getting missing key returns default."""
        config = ConfigLoader()
        value = config.get("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_set_key(self):
        """Test setting configuration key."""
        config = ConfigLoader()
        config.set("test.key", "test_value")
        assert config.get("test.key") == "test_value"

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading configuration."""
        config_path = tmp_path / "test_config.yaml"
        
        # Create config and save
        config = ConfigLoader()
        config.set("test.key", "test_value")
        config.save(str(config_path))
        
        # Load config
        loaded_config = ConfigLoader(str(config_path))
        assert loaded_config.get("test.key") == "test_value"
