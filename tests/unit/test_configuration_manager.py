"""
Unit tests for configuration management components.
"""

import pytest
import pytest_asyncio
import os
import yaml
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.contexter.models.config_models import C7DocConfig, ProxyConfig, DownloadConfig, StorageConfig, LoggingConfig
from src.contexter.core.config_manager import ConfigurationManager, ConfigurationError
from src.contexter.core.yaml_parser import YAMLConfigParser
from src.contexter.core.environment_manager import EnvironmentManager


class TestConfigurationModels:
    """Test Pydantic configuration model validation."""
    
    def test_valid_config_creation(self):
        """Test creating a valid configuration."""
        config_data = {
            'proxy': {
                'customer_id': 'test_customer',
                'zone_name': 'test_zone',
                'pool_size': 10
            },
            'download': {
                'max_concurrent': 5,
                'max_contexts': 7,
                'jitter_min': 0.5,
                'jitter_max': 2.0
            }
        }
        
        config = C7DocConfig(**config_data)
        assert config.proxy.customer_id == 'test_customer'
        assert config.download.max_concurrent == 5
        assert config.download.jitter_min < config.download.jitter_max
    
    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        
        # Test negative pool size
        with pytest.raises(ValueError):
            ProxyConfig(pool_size=-1)
        
        # Test invalid zone name
        with pytest.raises(ValueError):
            ProxyConfig(zone_name='invalid zone!')
        
        # Test jitter range validation
        with pytest.raises(ValueError):
            DownloadConfig(jitter_min=5.0, jitter_max=2.0)
        
        # Test invalid logging level
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LoggingConfig(level='INVALID')
    
    def test_cross_section_validation(self):
        """Test validation that spans multiple configuration sections."""
        
        # Configuration where concurrent downloads exceed proxy pool capacity
        config_data = {
            'proxy': {
                'customer_id': 'test',
                'pool_size': 5
            },
            'download': {
                'max_concurrent': 20  # Exceeds proxy pool * 2
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            C7DocConfig(**config_data)
        
        assert 'max_concurrent' in str(exc_info.value)
    
    def test_default_values(self):
        """Test that default values are properly set."""
        config = C7DocConfig()
        
        assert config.proxy.pool_size == 10
        assert config.download.max_concurrent == 10
        assert config.storage.compression_level == 6
        assert config.logging.level == 'INFO'
        assert config.debug_mode == False
        assert config.config_version == '1.0'


class TestYAMLParser:
    """Test YAML parsing with environment variable substitution."""
    
    def setup_method(self):
        """Set up test environment."""
        self.parser = YAMLConfigParser()
    
    def test_environment_variable_substitution(self):
        """Test YAML parsing with environment variable substitution."""
        
        yaml_content = """
        proxy:
          customer_id: ${BRIGHTDATA_CUSTOMER_ID}
          zone_name: test_zone
        download:
          max_concurrent: 10
        """
        
        # Set environment variable
        with patch.dict(os.environ, {'BRIGHTDATA_CUSTOMER_ID': 'test_customer_123'}):
            substituted = self.parser._substitute_environment_variables(yaml_content)
            config_data = yaml.safe_load(substituted)
            
            assert config_data['proxy']['customer_id'] == 'test_customer_123'
            assert config_data['proxy']['zone_name'] == 'test_zone'
    
    def test_environment_variable_with_default(self):
        """Test environment variable substitution with default values."""
        
        yaml_content = """
        proxy:
          zone_name: ${ZONE_NAME:default_zone}
        """
        
        # Test with environment variable not set
        with patch.dict(os.environ, {}, clear=True):
            substituted = self.parser._substitute_environment_variables(yaml_content)
            config_data = yaml.safe_load(substituted)
            
            assert config_data['proxy']['zone_name'] == 'default_zone'
        
        # Test with environment variable set
        with patch.dict(os.environ, {'ZONE_NAME': 'custom_zone'}):
            substituted = self.parser._substitute_environment_variables(yaml_content)
            config_data = yaml.safe_load(substituted)
            
            assert config_data['proxy']['zone_name'] == 'custom_zone'
    
    def test_missing_environment_variable(self):
        """Test error handling for missing environment variables."""
        
        yaml_content = """
        proxy:
          customer_id: ${MISSING_VAR}
        """
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                self.parser._substitute_environment_variables(yaml_content)
            
            assert 'MISSING_VAR' in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_yaml_file_operations(self):
        """Test YAML file loading and saving operations."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            # Test data
            config_data = {
                'proxy': {
                    'customer_id': '${BRIGHTDATA_CUSTOMER_ID}',
                    'zone_name': 'test_zone',
                    'pool_size': 10
                },
                'download': {
                    'max_concurrent': 5
                }
            }
            
            # Save configuration
            await self.parser.save_yaml_config(config_data, config_path)
            assert config_path.exists()
            
            # Load configuration
            with patch.dict(os.environ, {'BRIGHTDATA_CUSTOMER_ID': 'test_customer'}):
                loaded_config = await self.parser.load_yaml_config(config_path)
                
                assert loaded_config['proxy']['customer_id'] == 'test_customer'
                assert loaded_config['proxy']['zone_name'] == 'test_zone'
    
    def test_commented_yaml_generation(self):
        """Test generation of commented YAML configuration."""
        
        config_data = {
            'proxy': {
                'customer_id': 'test_customer',
                'pool_size': 10
            },
            'download': {
                'max_concurrent': 5
            }
        }
        
        yaml_content = self.parser._generate_commented_yaml(config_data)
        
        # Check that comments are included
        assert '# BrightData proxy configuration' in yaml_content
        assert '# Download engine configuration' in yaml_content
        assert '# Number of concurrent proxy connections' in yaml_content
        
        # Check that the YAML is valid
        parsed_data = yaml.safe_load(yaml_content)
        assert parsed_data['proxy']['pool_size'] == 10


class TestEnvironmentManager:
    """Test environment variable management."""
    
    def setup_method(self):
        """Set up test environment."""
        self.env_manager = EnvironmentManager()
    
    def test_required_environment_validation_success(self):
        """Test successful validation of required environment variables."""
        
        with patch.dict(os.environ, {
            'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
            'BRIGHTDATA_PASSWORD': 'test_password'
        }):
            env_vars = self.env_manager.validate_required_environment_vars()
            
            assert env_vars['BRIGHTDATA_CUSTOMER_ID'] == 'test_customer'
            assert env_vars['BRIGHTDATA_PASSWORD'] == 'test_password'
    
    def test_missing_environment_variables(self):
        """Test error handling for missing environment variables."""
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError) as exc_info:
                self.env_manager.validate_required_environment_vars()
            
            error_message = str(exc_info.value)
            assert 'BRIGHTDATA_CUSTOMER_ID' in error_message
            assert 'BRIGHTDATA_PASSWORD' in error_message
            assert 'export' in error_message
    
    def test_brightdata_credentials(self):
        """Test BrightData credential retrieval."""
        
        with patch.dict(os.environ, {
            'BRIGHTDATA_CUSTOMER_ID': 'test_customer_123',
            'BRIGHTDATA_PASSWORD': 'test_password_456'
        }):
            credentials = self.env_manager.get_brightdata_credentials()
            
            assert credentials['customer_id'] == 'test_customer_123'
            assert credentials['password'] == 'test_password_456'
    
    def test_credential_validation(self):
        """Test credential format validation."""
        
        # Test short customer ID
        with patch.dict(os.environ, {
            'BRIGHTDATA_CUSTOMER_ID': 'ab',
            'BRIGHTDATA_PASSWORD': 'valid_password'
        }):
            with pytest.raises(ValueError) as exc_info:
                self.env_manager.get_brightdata_credentials()
            
            assert 'customer_id' in str(exc_info.value).lower()
        
        # Test short password
        with patch.dict(os.environ, {
            'BRIGHTDATA_CUSTOMER_ID': 'valid_customer',
            'BRIGHTDATA_PASSWORD': 'short'
        }):
            with pytest.raises(ValueError) as exc_info:
                self.env_manager.get_brightdata_credentials()
            
            assert 'password' in str(exc_info.value).lower()
    
    def test_optional_config_overrides(self):
        """Test optional configuration overrides from environment."""
        
        with patch.dict(os.environ, {
            'CONTEXTER_CONFIG_PATH': '/custom/path/config.yaml',
            'CONTEXTER_LOG_LEVEL': 'DEBUG',
            'CONTEXTER_DEBUG_MODE': 'true'
        }):
            overrides = self.env_manager.get_optional_config_overrides()
            
            assert overrides['CONTEXTER_CONFIG_PATH'] == '/custom/path/config.yaml'
            assert overrides['CONTEXTER_LOG_LEVEL'] == 'DEBUG'
            assert overrides['CONTEXTER_DEBUG_MODE'] == 'true'


class TestConfigurationManager:
    """Test main configuration manager functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigurationManager()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.config_manager.stop_file_watcher()
    
    @pytest.mark.asyncio
    async def test_default_config_generation(self):
        """Test generation of default configuration."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            # Set required environment variables
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
                'BRIGHTDATA_PASSWORD': 'test_password'
            }):
                await self.config_manager.generate_default_config(config_path)
                
                assert config_path.exists()
                
                # Verify the generated config can be loaded
                config = await self.config_manager.load_config(config_path)
                assert isinstance(config, C7DocConfig)
    
    @pytest.mark.asyncio
    async def test_config_load_and_validation(self):
        """Test configuration loading and validation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
                'BRIGHTDATA_PASSWORD': 'test_password'
            }):
                # Generate and load default config
                await self.config_manager.generate_default_config(config_path)
                config = await self.config_manager.load_config(config_path)
                
                assert config.proxy.customer_id == 'test_customer'
                assert isinstance(config, C7DocConfig)
                
                # Test validation
                validated_config = await self.config_manager.validate_config()
                assert validated_config == config
    
    @pytest.mark.asyncio
    async def test_config_save_and_reload(self):
        """Test configuration saving and reloading."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
                'BRIGHTDATA_PASSWORD': 'test_password'
            }):
                # Create and modify config
                config = C7DocConfig()
                config.download.max_concurrent = 15
                config.storage.compression_level = 9
                
                # Save config
                await self.config_manager.save_config(config, config_path)
                
                # Load and verify changes
                reloaded_config = await self.config_manager.load_config(config_path)
                assert reloaded_config.download.max_concurrent == 15
                assert reloaded_config.storage.compression_level == 9
    
    @pytest.mark.asyncio
    async def test_environment_overrides(self):
        """Test environment variable overrides."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            storage_path = Path(temp_dir) / 'custom_storage'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
                'BRIGHTDATA_PASSWORD': 'test_password',
                'CONTEXTER_STORAGE_PATH': str(storage_path),
                'CONTEXTER_LOG_LEVEL': 'DEBUG',
                'CONTEXTER_DEBUG_MODE': 'true'
            }):
                config = await self.config_manager.load_config(config_path)
                
                assert config.storage.base_path == str(storage_path)
                assert config.logging.level == 'DEBUG'
                assert config.debug_mode == True
    
    @pytest.mark.asyncio
    async def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        
        # Test invalid configuration data
        invalid_config_data = {
            'proxy': {
                'pool_size': -1,  # Invalid
                'zone_name': '',  # Invalid
            },
            'download': {
                'max_concurrent': 100,  # Will exceed proxy pool limit
                'jitter_min': 5.0,
                'jitter_max': 2.0,  # Invalid: max < min
            }
        }
        
        with pytest.raises(ConfigurationError):
            await self.config_manager.validate_config(invalid_config_data)
    
    @pytest.mark.asyncio
    async def test_missing_credentials_handling(self):
        """Test handling of missing credentials."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            # Clear environment variables
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises((ConfigurationError, EnvironmentError)) as exc_info:
                    await self.config_manager.load_config(config_path)
                
                assert 'environment variables' in str(exc_info.value).lower()
    
    def test_default_config_path(self):
        """Test default configuration path resolution."""
        
        # Test default path
        default_path = self.config_manager._get_default_config_path()
        expected_path = Path.home() / '.contexter' / 'config.yaml'
        assert default_path == expected_path
        
        # Test environment override
        with patch.dict(os.environ, {'CONTEXTER_CONFIG_PATH': '/custom/config.yaml'}):
            override_path = self.config_manager._get_default_config_path()
            assert override_path == Path('/custom/config.yaml')
    
    @pytest.mark.asyncio
    async def test_file_watcher_setup(self):
        """Test configuration file watcher setup."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'test_customer',
                'BRIGHTDATA_PASSWORD': 'test_password'
            }):
                # Load config to set path
                await self.config_manager.load_config(config_path)
                
                # Test callback tracking
                callback_calls = []
                
                def test_callback(config):
                    callback_calls.append(config)
                
                # Start file watcher
                self.config_manager.start_file_watcher(test_callback)
                assert self.config_manager.file_watcher is not None
                
                # Stop file watcher
                self.config_manager.stop_file_watcher()
                assert self.config_manager.file_watcher is None