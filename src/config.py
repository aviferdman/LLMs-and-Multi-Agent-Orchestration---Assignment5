"""
Configuration management for Context Windows Research.

This module provides centralized configuration loading and validation
from YAML files and environment variables.

Classes:
    Config: Main configuration manager

Example:
    >>> config = Config()
    >>> model = config.get('general.ollama_model')
    >>> output_dir = config.get('general.output_dir')
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


class Config:
    """
    Configuration manager for loading and accessing experiment settings.

    Loads configuration from:
    1. YAML files in config/ directory
    2. Environment variables from .env file
    3. Default values

    Attributes:
        config_dir (Path): Directory containing configuration files
        experiments (dict): Experiment-specific configurations
        models (dict): LLM model configurations
        paths (dict): File path configurations
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses 'config/'
        """
        # Load environment variables
        load_dotenv()

        # Set configuration directory
        if config_dir is None:
            self.config_dir = Path("config")
        else:
            self.config_dir = Path(config_dir)

        # Load configuration files
        self.experiments = self._load_yaml("experiments.yaml")
        self.models = self._load_yaml("models.yaml")
        self.paths = self._load_yaml("paths.yaml")

        # Override with environment variables
        self._apply_env_overrides()

        logger.info("Configuration loaded successfully")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Args:
            filename: Name of YAML file in config directory

        Returns:
            Dictionary containing configuration

        Raises:
            FileNotFoundError: If configuration file not found
            yaml.YAMLError: If YAML parsing fails
        """
        filepath = self.config_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {filepath}")
                return config or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            raise

    def _apply_env_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Override general settings
        if os.getenv("OLLAMA_MODEL"):
            self.experiments["general"]["ollama_model"] = os.getenv("OLLAMA_MODEL")

        if os.getenv("RANDOM_SEED"):
            self.experiments["general"]["random_seed"] = int(os.getenv("RANDOM_SEED"))

        if os.getenv("OUTPUT_DIR"):
            self.experiments["general"]["output_dir"] = os.getenv("OUTPUT_DIR")

        # Override Ollama configuration
        if os.getenv("OLLAMA_BASE_URL"):
            self.models["ollama"]["base_url"] = os.getenv("OLLAMA_BASE_URL")

        logger.debug("Applied environment variable overrides")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value
                     (e.g., 'general.random_seed' or 'experiment_1.num_documents')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> seed = config.get('general.random_seed', 42)
            >>> num_docs = config.get('experiment_1.num_documents', 5)
        """
        keys = key_path.split(".")
        value = self.experiments

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key not found: {key_path}, using default: {default}")
            return default

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific LLM model.

        Args:
            model_name: Name of the model (e.g., 'llama2_13b')

        Returns:
            Dictionary containing model configuration

        Raises:
            ValueError: If model not found in configuration
        """
        if model_name in self.models.get("ollama", {}).get("models", {}):
            return self.models["ollama"]["models"][model_name]
        elif model_name in self.models.get("openai", {}).get("models", {}):
            return self.models["openai"]["models"][model_name]
        else:
            raise ValueError(f"Model configuration not found: {model_name}")

    def get_path(self, path_key: str) -> Path:
        """
        Get file path from configuration.

        Args:
            path_key: Dot-separated path key (e.g., 'data.hebrew_corpus')

        Returns:
            Path object

        Example:
            >>> config = Config()
            >>> corpus_path = config.get_path('data.hebrew_corpus')
        """
        keys = path_key.split(".")
        value = self.paths

        try:
            for key in keys:
                value = value[key]
            return Path(value)
        except (KeyError, TypeError):
            logger.warning(f"Path not found in configuration: {path_key}")
            return Path(".")

    def get_experiment_config(self, experiment_num: int) -> Dict[str, Any]:
        """
        Get configuration for a specific experiment.

        Args:
            experiment_num: Experiment number (1-4)

        Returns:
            Dictionary containing experiment configuration

        Raises:
            ValueError: If experiment number invalid
        """
        if experiment_num not in [1, 2, 3, 4]:
            raise ValueError(f"Invalid experiment number: {experiment_num}")

        experiment_key = f"experiment_{experiment_num}"
        return self.experiments.get(experiment_key, {})

    def validate(self) -> bool:
        """
        Validate configuration completeness and correctness.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required general settings
        required_general = ["random_seed", "num_runs", "ollama_model", "output_dir"]
        for key in required_general:
            if key not in self.experiments.get("general", {}):
                raise ValueError(f"Missing required configuration: general.{key}")

        # Check that output directory exists or can be created
        output_dir = Path(self.experiments["general"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Configuration validation passed")
        return True

    def __repr__(self) -> str:
        """String representation of Config object."""
        return f"Config(config_dir={self.config_dir})"

    def get_general_config(self) -> Dict[str, Any]:
        """
        Get general configuration settings.

        Returns:
            Dictionary containing general configuration
        """
        return self.experiments.get("general", {})


# Alias for backward compatibility
ConfigManager = Config
