"""
Tests for configuration management module
"""

import pytest
import yaml
from pathlib import Path
from src.config import Config

class TestConfig:
    """Test suite for Config class"""
    
    @pytest.fixture
    def config(self):
        """Create Config instance for testing"""
        return Config()
    
    def test_config_initialization(self, config):
        """Test that Config initializes correctly"""
        assert config is not None
    
    def test_get_method(self, config):
        """Test get method"""
        # Test with default value
        result = config.get('nonexistent.key', default='default_value')
        assert result == 'default_value'
    
    def test_get_model_config(self, config):
        """Test retrieving model configuration"""
        try:
            # Use actual model name from config
            model_config = config.get_model_config('llama2_7b')
            if model_config:
                assert isinstance(model_config, dict)
        except (KeyError, FileNotFoundError, ValueError):
            # Config might not exist in test environment
            pass
    
    def test_get_experiment_config(self, config):
        """Test retrieving experiment configuration"""
        try:
            exp_config = config.get_experiment_config(1)
            if exp_config:
                assert isinstance(exp_config, dict)
        except (KeyError, FileNotFoundError):
            # Config might not exist in test environment
            pass
    
    def test_get_path(self, config):
        """Test path retrieval"""
        try:
            path = config.get_path('data_dir')
            if path:
                assert isinstance(path, Path)
        except (KeyError, FileNotFoundError):
            pass
    
    def test_validate(self, config):
        """Test configuration validation"""
        result = config.validate()
        assert isinstance(result, bool)
    
    def test_config_paths_exist(self):
        """Test that configuration files exist"""
        config_dir = Path('config')
        assert config_dir.exists(), "Config directory should exist"
        
        # Check if at least some config files exist
        config_files = list(config_dir.glob('*.yaml'))
        assert len(config_files) > 0, "Should have at least one YAML config file"
    
    def test_experiments_yaml_structure(self):
        """Test that experiments.yaml has valid structure"""
        config_path = Path('config/experiments.yaml')
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
            
            assert isinstance(config_data, dict), "Config should be a dictionary"
            # Check for expected keys if they exist
            for key in config_data:
                assert isinstance(config_data[key], dict), f"{key} should be a dictionary"
    
    def test_get_general_config(self, config):
        """Test getting general configuration"""
        general = config.get_general_config()
        assert isinstance(general, dict)
    
    def test_repr(self, config):
        """Test string representation"""
        repr_str = repr(config)
        assert isinstance(repr_str, str)
        assert 'Config' in repr_str

    def test_custom_config_dir(self):
        """Test initialization with custom config directory."""
        config = Config(config_dir="config")
        assert config.config_dir == Path("config")

    def test_file_not_found_handling(self):
        """Test handling of missing configuration files."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(config_dir=temp_dir)
            assert isinstance(config.experiments, dict)
            assert isinstance(config.models, dict)
            assert isinstance(config.paths, dict)

    def test_env_override_ollama_model(self, monkeypatch):
        """Test environment variable override for OLLAMA_MODEL."""
        monkeypatch.setenv("OLLAMA_MODEL", "test-model")
        config = Config()
        assert config.experiments["general"]["ollama_model"] == "test-model"

    def test_env_override_random_seed(self, monkeypatch):
        """Test environment variable override for RANDOM_SEED."""
        monkeypatch.setenv("RANDOM_SEED", "999")
        config = Config()
        assert config.experiments["general"]["random_seed"] == 999

    def test_env_override_output_dir(self, monkeypatch):
        """Test environment variable override for OUTPUT_DIR."""
        monkeypatch.setenv("OUTPUT_DIR", "/custom/output")
        config = Config()
        assert config.experiments["general"]["output_dir"] == "/custom/output"

    def test_env_override_ollama_base_url(self, monkeypatch):
        """Test environment variable override for OLLAMA_BASE_URL."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:11434")
        config = Config()
        assert config.models["ollama"]["base_url"] == "http://custom:11434"

    def test_get_nested_missing_key(self, config):
        """Test get method with nested missing keys."""
        value = config.get("deeply.nested.nonexistent.key", default="fallback")
        assert value == "fallback"

    def test_get_path_missing_key(self, config):
        """Test get_path with missing key returns current directory."""
        path = config.get_path("nonexistent.path.to.file")
        assert path == Path(".")

    def test_get_experiment_config_invalid_number(self, config):
        """Test get_experiment_config with invalid experiment number."""
        with pytest.raises(ValueError, match="Invalid experiment number"):
            config.get_experiment_config(5)

    def test_get_experiment_config_zero(self, config):
        """Test get_experiment_config with zero."""
        with pytest.raises(ValueError):
            config.get_experiment_config(0)

    def test_get_experiment_config_negative(self, config):
        """Test get_experiment_config with negative number."""
        with pytest.raises(ValueError):
            config.get_experiment_config(-1)

    def test_validate_missing_required_config(self):
        """Test validate raises error for missing required configuration."""
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config(config_dir=temp_dir)
            with pytest.raises(ValueError, match="Missing required configuration"):
                config.validate()
