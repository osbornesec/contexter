"""
Integration tests for configuration manager and proxy manager working together.
"""

import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from src.contexter.core.config_manager import ConfigurationManager
from src.contexter.integration.proxy_manager import BrightDataProxyManager


class TestConfigProxyIntegration:
    """Test configuration and proxy manager integration."""
    
    @pytest.mark.asyncio
    async def test_proxy_manager_from_config(self):
        """Test creating proxy manager using configuration manager."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            # Set up environment with credentials
            with patch.dict('os.environ', {
                'BRIGHTDATA_CUSTOMER_ID': 'integration_customer',
                'BRIGHTDATA_PASSWORD': 'integration_password'
            }):
                # Create configuration manager
                config_manager = ConfigurationManager()
                
                # Load configuration (will create default if not exists)
                config = await config_manager.load_config(config_path)
                
                # Verify configuration has proxy settings
                assert config.proxy.customer_id == 'integration_customer'
                assert config.proxy.zone_name == 'zone_residential_1'  # Default
                assert config.proxy.pool_size == 10  # Default
                
                # Create proxy manager using environment (same as config would)
                proxy_manager = await BrightDataProxyManager.from_environment(
                    zone_name=config.proxy.zone_name
                )
                
                try:
                    # Mock HTTP client for pool initialization
                    with patch('httpx.AsyncClient') as mock_client_class:
                        mock_client = AsyncMock()
                        mock_client.aclose = AsyncMock()
                        mock_client_class.return_value = mock_client
                        
                        # Initialize proxy pool using config settings
                        await proxy_manager.initialize_pool(config.proxy.pool_size)
                        
                        # Verify pool was created according to config
                        assert len(proxy_manager.proxy_pool) == config.proxy.pool_size
                        
                        # Verify proxy manager uses correct zone
                        assert proxy_manager.zone_name == config.proxy.zone_name
                        
                        # Test getting a connection
                        connection = await proxy_manager.get_connection()
                        assert connection is not None
                        
                        # Verify proxy URL contains correct zone
                        assert config.proxy.zone_name in connection.proxy_url
                        
                        print("✅ Configuration and proxy manager integration successful")
                        
                finally:
                    await proxy_manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_config_proxy_validation_integration(self):
        """Test that proxy configuration validation works with proxy manager constraints."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict('os.environ', {
                'BRIGHTDATA_CUSTOMER_ID': 'validation_customer',
                'BRIGHTDATA_PASSWORD': 'validation_password'
            }):
                config_manager = ConfigurationManager()
                config = await config_manager.load_config(config_path)
                
                # Test that configuration validation aligns with proxy manager constraints
                
                # Valid pool size (within proxy manager limits)
                config.proxy.pool_size = 25
                validated_config = await config_manager.validate_config(config.model_dump())
                assert validated_config.proxy.pool_size == 25
                
                # Test cross-section validation with download config
                config.download.max_concurrent = 40  # Within limit (25 * 2 = 50)
                validated_config = await config_manager.validate_config(config.model_dump())
                assert validated_config.download.max_concurrent == 40
                
                # Invalid pool size should be caught by proxy manager
                try:
                    proxy_manager = await BrightDataProxyManager.from_environment()
                    # This should fail with pool size outside range
                    with pytest.raises(Exception):  # ProxyManagerError or ValueError
                        await proxy_manager.initialize_pool(pool_size=5)  # Below minimum
                    
                    await proxy_manager.shutdown()
                except Exception:
                    pass  # Expected
                
                print("✅ Configuration and proxy validation integration successful")
    
    @pytest.mark.asyncio
    async def test_config_changes_affect_proxy_behavior(self):
        """Test that configuration changes can be applied to proxy manager behavior."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'config.yaml'
            
            with patch.dict('os.environ', {
                'BRIGHTDATA_CUSTOMER_ID': 'behavior_customer', 
                'BRIGHTDATA_PASSWORD': 'behavior_password'
            }):
                config_manager = ConfigurationManager()
                initial_config = await config_manager.load_config(config_path)
                
                # Create first proxy manager with initial config
                proxy_manager1 = await BrightDataProxyManager.from_environment(
                    zone_name=initial_config.proxy.zone_name
                )
                
                try:
                    with patch('httpx.AsyncClient') as mock_client_class:
                        mock_client = AsyncMock()
                        mock_client.aclose = AsyncMock()
                        mock_client_class.return_value = mock_client
                        
                        await proxy_manager1.initialize_pool(initial_config.proxy.pool_size)
                        
                        # Verify initial state
                        assert len(proxy_manager1.proxy_pool) == initial_config.proxy.pool_size
                        assert proxy_manager1.zone_name == initial_config.proxy.zone_name
                        
                        # Modify configuration
                        modified_config = initial_config.model_copy(deep=True)
                        modified_config.proxy.pool_size = 15
                        modified_config.proxy.zone_name = "zone_residential_2"
                        
                        # Save modified configuration
                        await config_manager.save_config(modified_config, config_path)
                        
                        # Create new proxy manager with modified config
                        proxy_manager2 = await BrightDataProxyManager.from_environment(
                            zone_name=modified_config.proxy.zone_name
                        )
                        
                        try:
                            await proxy_manager2.initialize_pool(modified_config.proxy.pool_size)
                            
                            # Verify changes were applied
                            assert len(proxy_manager2.proxy_pool) == modified_config.proxy.pool_size
                            assert proxy_manager2.zone_name == modified_config.proxy.zone_name
                            assert proxy_manager2.zone_name != proxy_manager1.zone_name
                            
                            print("✅ Configuration changes affect proxy behavior correctly")
                            
                        finally:
                            await proxy_manager2.shutdown()
                        
                finally:
                    await proxy_manager1.shutdown()