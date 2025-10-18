"""Configuration loader for GeneLab."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default to config.yaml in the project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"

        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found at {self.config_path}. Using defaults.")
                self.config = self._get_default_config()
                return

            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f) or {}

            # Create directories if they don't exist
            self._create_directories()

            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()

    def _create_directories(self) -> None:
        """Create necessary directories from configuration."""
        data_config = self.config.get("data", {})
        directories = [
            data_config.get("base_path", "./data"),
            data_config.get("sequences_path", "./data/sequences"),
            data_config.get("models_path", "./models"),
            data_config.get("output_path", "./output"),
            data_config.get("logs_path", "./logs"),
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "data": {
                "base_path": "./data",
                "sequences_path": "./data/sequences",
                "models_path": "./models",
                "output_path": "./output",
                "logs_path": "./logs",
            },
            "entrez": {
                "email": "your.email@example.com",
                "api_key": "",
                "max_retries": 3,
                "delay": 1.0,
            },
            "training": {
                "epochs": 50,
                "batch_size": 32,
                "validation_split": 0.2,
                "test_split": 0.2,
                "random_seed": 42,
            },
            "mutation": {
                "base_rate": 0.01,
                "elitism_rate": 0.1,
                "crossover_rate": 0.7,
                "population_size": 100,
            },
            "rl": {
                "total_timesteps": 10000,
                "learning_rate": 0.0003,
                "gamma": 0.99,
                "batch_size": 64,
                "n_steps": 2048,
            },
            "sequence": {
                "kmer_size": 6,
                "ngram_range": [4, 4],
                "encoding": "one-hot",
            },
            "visualization": {
                "figure_size": [10, 8],
                "dpi": 300,
                "style": "seaborn-v0_8",
                "save_format": "png",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "genelab.log",
            },
            "hardware": {
                "use_gpu": True,
                "gpu_memory_growth": True,
                "num_threads": 4,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data.base_path')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data.base_path')
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save configuration. If None, saves to original path.
        """
        save_path = Path(path) if path else self.config_path

        with open(save_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {save_path}")


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)

    return _config_loader
