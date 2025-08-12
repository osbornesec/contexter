"""
Integration tests for configuration management workflow.
"""

import pytest
import pytest_asyncio
import os
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from src.contexter.core.config_manager import ConfigurationManager
from src.contexter.models.config_models import C7DocConfig


class TestConfigurationIntegration:
    """Test complete configuration management workflow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigurationManager()
    
    def teardown_method(self):
        """Clean up after tests."""
        self.config_manager.stop_file_watcher()
    
    @pytest.mark.asyncio
    async def test_full_configuration_workflow(self):
        """Test complete configuration management workflow."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            # Set up environment
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'integration_test_customer',
                'BRIGHTDATA_PASSWORD': 'integration_test_password'
            }):
                # Generate default configuration
                await self.config_manager.generate_default_config(config_path)
                assert config_path.exists()
                
                # Load and validate
                config = await self.config_manager.load_config(config_path)
                validated_config = await self.config_manager.validate_config()
                
                assert config == validated_config
                assert config.proxy.customer_id == 'integration_test_customer'
                
                # Modify and save
                config.download.max_concurrent = 15
                config.storage.compression_level = 9
                await self.config_manager.save_config(config, config_path)
                
                # Reload and verify changes
                reloaded_config = await self.config_manager.load_config(config_path)
                assert reloaded_config.download.max_concurrent == 15
                assert reloaded_config.storage.compression_level == 9
                
                # Test that credentials are not stored in file
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                assert 'integration_test_customer' not in config_content
                assert '${BRIGHTDATA_CUSTOMER_ID}' in config_content
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        os.getenv('CI') == 'true', 
        reason="File watching tests are flaky in CI environments"
    )
    async def test_file_watching(self):
        """Test configuration file watching functionality."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'watch_test_customer',
                'BRIGHTDATA_PASSWORD': 'watch_test_password'
            }):
                # Create initial config
                await self.config_manager.generate_default_config(config_path)
                initial_config = await self.config_manager.load_config(config_path)
                
                # Set up file watcher
                config_changes = []
                
                def on_config_change(new_config: C7DocConfig):
                    config_changes.append(new_config)
                
                self.config_manager.start_file_watcher(on_config_change)
                
                try:
                    # Modify configuration file
                    modified_config = initial_config.model_copy(deep=True)
                    modified_config.download.max_concurrent = 8
                    await self.config_manager.save_config(modified_config, config_path)
                    
                    # Wait for file watcher to detect change
                    # Wait longer and check multiple times for the change
                    for i in range(10):  # Check up to 10 times (5 seconds total)
                        await asyncio.sleep(0.5)
                        if len(config_changes) > 0:
                            break
                    
                    # Verify change was detected
                    assert len(config_changes) > 0, f"No config changes detected after file modification. Changes: {config_changes}"
                    assert config_changes[-1].download.max_concurrent == 8
                    
                finally:
                    self.config_manager.stop_file_watcher()
    
    @pytest.mark.asyncio
    async def test_configuration_with_environment_overrides(self):
        """Test configuration loading with various environment overrides."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            custom_storage = Path(temp_dir) / 'custom_storage'
            
            # Test with multiple environment overrides
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'env_test_customer',
                'BRIGHTDATA_PASSWORD': 'env_test_password',
                'CONTEXTER_STORAGE_PATH': str(custom_storage),
                'CONTEXTER_LOG_LEVEL': 'DEBUG',
                'CONTEXTER_DEBUG_MODE': 'true'
            }):
                # Generate config with overrides
                await self.config_manager.generate_default_config(config_path)
                config = await self.config_manager.load_config(config_path)
                
                # Verify overrides were applied
                assert config.storage.base_path == str(custom_storage)
                assert config.logging.level == 'DEBUG'
                assert config.debug_mode == True
                assert custom_storage.exists()  # Should be created during validation
    
    @pytest.mark.asyncio
    async def test_configuration_migration_and_validation(self):
        """Test configuration schema validation and potential migration scenarios."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'migration_test_customer',
                'BRIGHTDATA_PASSWORD': 'migration_test_password'
            }):
                # Create initial configuration
                initial_config = C7DocConfig()
                await self.config_manager.save_config(initial_config, config_path)
                
                # Load and verify initial config
                loaded_config = await self.config_manager.load_config(config_path)
                assert loaded_config.config_version == '1.0'
                
                # Simulate configuration with different settings
                modified_config = loaded_config.model_copy(deep=True)
                modified_config.download.max_concurrent = 20
                modified_config.proxy.pool_size = 15  # Ensure cross-section validation passes
                
                # Save and reload
                await self.config_manager.save_config(modified_config, config_path)
                final_config = await self.config_manager.load_config(config_path)
                
                # Verify changes persisted
                assert final_config.download.max_concurrent == 20
                assert final_config.proxy.pool_size == 15
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_validation(self):
        """Test error recovery scenarios and validation edge cases."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'error_test_customer',
                'BRIGHTDATA_PASSWORD': 'error_test_password'
            }):
                # Test loading non-existent config (should create default)
                config = await self.config_manager.load_config(config_path)
                assert config_path.exists()
                
                # Test with invalid YAML (create corrupted file)
                with open(config_path, 'w') as f:
                    f.write('invalid: yaml: content: [unclosed')
                
                # Should raise error for corrupted YAML
                with pytest.raises(Exception):  # Could be ConfigurationError or ValueError
                    await self.config_manager.load_config(config_path)
                
                # Remove corrupted file and regenerate valid config
                config_path.unlink()  # Delete the corrupted file
                await self.config_manager.generate_default_config(config_path)
                recovered_config = await self.config_manager.load_config(config_path)
                assert isinstance(recovered_config, C7DocConfig)
    
    @pytest.mark.asyncio
    async def test_concurrent_config_operations(self):
        """Test concurrent configuration operations."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'concurrent_test_customer',
                'BRIGHTDATA_PASSWORD': 'concurrent_test_password'
            }):
                # Create initial config
                await self.config_manager.generate_default_config(config_path)
                
                async def load_and_modify_config(modification_value: int):
                    """Helper function to load and modify config concurrently."""
                    config = await self.config_manager.load_config(config_path)
                    config.download.max_concurrent = modification_value
                    await self.config_manager.save_config(config, config_path)
                    return config
                
                # Run concurrent operations
                tasks = [
                    load_and_modify_config(5),
                    load_and_modify_config(8),
                    load_and_modify_config(12)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check that all operations completed successfully
                for result in results:
                    assert isinstance(result, C7DocConfig)
                
                # Verify final state is consistent
                final_config = await self.config_manager.load_config(config_path)
                assert final_config.download.max_concurrent in [5, 8, 12]
    
    @pytest.mark.asyncio
    async def test_storage_path_validation_and_creation(self):
        """Test storage path validation and automatic directory creation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            nested_storage_path = Path(temp_dir) / 'deeply' / 'nested' / 'storage'
            
            with patch.dict(os.environ, {
                'BRIGHTDATA_CUSTOMER_ID': 'storage_test_customer',
                'BRIGHTDATA_PASSWORD': 'storage_test_password',
                'CONTEXTER_STORAGE_PATH': str(nested_storage_path)
            }):
                # Load config with nested storage path
                config = await self.config_manager.load_config(config_path)
                
                # Verify the nested directory was created
                assert nested_storage_path.exists()
                assert nested_storage_path.is_dir()
                
                # Verify config contains the expanded path
                assert config.storage.base_path == str(nested_storage_path)
                
                # Test that the storage directory is writable
                test_file = nested_storage_path / 'test_write.txt'
                test_file.write_text('test content')
                assert test_file.exists()
                assert test_file.read_text() == 'test content'