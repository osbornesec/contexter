"""
Tests for DNS resolution functionality in proxy manager.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock
import os

from src.contexter.integration.proxy_manager import BrightDataProxyManager
from src.contexter.models.config_models import ProxyConfig


class TestDNSResolutionFeatures:
    """Test DNS resolution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.customer_id = "brd-customer-hl_b6e19b0e-zone-forcontext"
        self.password = "test_password"
        self.zone_name = "zone_residential_1"
    
    def test_proxy_url_construction_with_dns_local(self):
        """Test proxy URL construction with local DNS resolution."""
        proxy_manager = BrightDataProxyManager(
            customer_id=self.customer_id,
            password=self.password,
            zone_name=self.zone_name,
            dns_resolution="local"
        )
        
        session_id = "test_session_123"
        proxy_url = proxy_manager._build_proxy_url(session_id)
        
        # Should include DNS resolution mode in username
        expected_username = f"{self.customer_id}-session-{session_id}-dns-local"
        expected_url = f"http://{expected_username}:{self.password}@brd.superproxy.io:33335"
        
        assert proxy_url == expected_url
    
    def test_proxy_url_construction_with_dns_remote(self):
        """Test proxy URL construction with remote DNS resolution."""
        proxy_manager = BrightDataProxyManager(
            customer_id=self.customer_id,
            password=self.password,
            zone_name=self.zone_name,
            dns_resolution="remote"
        )
        
        session_id = "test_session_456"
        proxy_url = proxy_manager._build_proxy_url(session_id)
        
        # Should include DNS resolution mode in username
        expected_username = f"{self.customer_id}-session-{session_id}-dns-remote"
        expected_url = f"http://{expected_username}:{self.password}@brd.superproxy.io:33335"
        
        assert proxy_url == expected_url
    
    def test_proxy_config_dns_resolution_validation(self):
        """Test that proxy configuration validates DNS resolution modes."""
        # Valid DNS resolution modes
        valid_modes = ["local", "remote"]
        
        for mode in valid_modes:
            config = ProxyConfig(
                customer_id=self.customer_id,
                zone_name=self.zone_name,
                dns_resolution=mode
            )
            assert config.dns_resolution == mode
    
    @pytest.mark.asyncio
    async def test_connectivity_with_dns_resolution(self):
        """Test proxy connectivity with DNS resolution configured."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            proxy_manager = BrightDataProxyManager(
                customer_id=self.customer_id,
                password=self.password,
                zone_name=self.zone_name,
                dns_resolution="remote"
            )
            
            result = await proxy_manager.test_connectivity()
            
            # Should succeed with proper DNS resolution
            assert result is True
            
            # Verify the correct proxy URL was used
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            proxy_url = call_args[1]['proxy']
            assert f"{self.customer_id}-dns-remote" in proxy_url
    
    @pytest.mark.asyncio
    async def test_environment_variable_integration(self):
        """Test that DNS resolution works with environment variables."""
        with patch.dict(os.environ, {
            'BRIGHTDATA_CUSTOMER_ID': self.customer_id,
            'BRIGHTDATA_PASSWORD': self.password
        }):
            proxy_manager = await BrightDataProxyManager.from_environment(
                zone_name=self.zone_name,
                dns_resolution="local"
            )
            
            # Test that DNS resolution is properly configured
            proxy_url = proxy_manager._build_proxy_url("test_session")
            assert "-dns-local" in proxy_url
    
    def test_default_dns_resolution_mode(self):
        """Test that remote is the default DNS resolution mode."""
        config = ProxyConfig(
            customer_id=self.customer_id,
            zone_name=self.zone_name
            # dns_resolution not specified - should default to "remote"
        )
        
        assert config.dns_resolution == "remote"


class TestDNSResolutionIntegration:
    """Integration tests for DNS resolution functionality."""
    
    def test_config_loading_with_dns_resolution(self):
        """Test that configuration properly loads DNS resolution settings."""
        from src.contexter.models.config_models import C7DocConfig, ProxyConfig
        
        # Create config with DNS resolution
        config_data = {
            "proxy": {
                "customer_id": "test_customer",
                "zone_name": "zone_residential_1",
                "pool_size": 5,
                "health_check_interval": 300,
                "circuit_breaker_threshold": 5,
                "circuit_breaker_timeout": 30,
                "dns_resolution": "local"
            },
            "download": {
                "max_concurrent": 10,
                "max_contexts": 5,
                "jitter_min": 0.5,
                "jitter_max": 2.0,
                "max_retries": 3,
                "request_timeout": 30.0,
                "token_limit": 200000
            },
            "storage": {
                "base_path": "/tmp/test",
                "compression_level": 6,
                "retention_versions": 5,
                "verify_integrity": True,
                "cleanup_threshold_gb": 10.0
            },
            "logging": {
                "level": "INFO",
                "file_path": None,
                "max_file_size_mb": 10,
                "backup_count": 5
            }
        }
        
        config = C7DocConfig(**config_data)
        assert config.proxy.dns_resolution == "local"
    
    def test_status_command_with_dns_resolution(self):
        """Test that the status command works with DNS resolution configured."""
        # This would be an integration test that verifies the status command
        # works properly with the DNS resolution feature enabled
        pass  # Placeholder for full integration test