"""
Tests for Context7 API fixes including correct URL and health check.
"""

import pytest
from unittest.mock import AsyncMock, patch, Mock, MagicMock
import httpx

from src.contexter.integration.context7_client import Context7Client


class TestContext7APIFixes:
    """Test Context7 API fixes and improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.correct_base_url = "https://context7.com/api/v1"
        self.client = Context7Client(base_url=self.correct_base_url)
    
    def test_correct_base_url_usage(self):
        """Test that Context7Client uses the correct base URL by default."""
        default_client = Context7Client()
        assert default_client.base_url == self.correct_base_url
        
        # Test custom base URL
        custom_client = Context7Client(base_url="https://custom.api.com/v1")
        assert custom_client.base_url == "https://custom.api.com/v1"
    
    @pytest.mark.asyncio
    async def test_health_check_with_correct_endpoint(self):
        """Test that health check uses the correct search endpoint."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await self.client.health_check()
            
            # Should succeed
            assert result is True
            
            # Verify the correct endpoint was called
            mock_client.get.assert_called_once()
            call_args = mock_client.get.call_args
            url = call_args[0][0]  # First positional argument (URL)
            
            # Should use the search endpoint for health check
            assert url == f"{self.correct_base_url}/search"
            
            # Should include test query parameter
            params = call_args[1]['params']
            assert params == {'query': 'test'}
    
    @pytest.mark.asyncio
    async def test_health_check_handles_404_as_healthy(self):
        """Test that health check considers 404 responses as healthy (no results)."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 404  # No results found
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await self.client.health_check()
            
            # Should consider 404 as healthy (just means no search results)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_health_check_rejects_server_errors(self):
        """Test that health check properly rejects server errors."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status_code = 500  # Server error
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            result = await self.client.health_check()
            
            # Should consider 500 as unhealthy
            assert result is False
    
    @pytest.mark.asyncio
    async def test_resolve_library_id_with_correct_format(self):
        """Test that resolve_library_id handles the Context7 format correctly."""
        # Mock the internal parsing method to return structured results
        mock_result = [
            type('LibrarySearchResult', (), {
                'id': 'context7/python-3',
                'title': 'Python 3', 
                'description': 'Python 3 documentation'
            })()
        ]
        
        with patch.object(self.client, '_parse_search_response', return_value=mock_result) as mock_parse:
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_client.get.return_value = mock_response
                mock_client.__aenter__.return_value = mock_client
                mock_client.__aexit__.return_value = None
                mock_client_class.return_value = mock_client
                
                result = await self.client.resolve_library_id("python")
                
                # Should return parsed results
                assert len(result) == 1
                assert result[0].id == "context7/python-3"
                assert result[0].title == "Python 3"
    
    @pytest.mark.asyncio
    async def test_dns_resolution_test_with_correct_hostname(self):
        """Test that DNS resolution test uses the correct hostname."""
        # The _test_dns_resolution method should use context7.com as default
        with patch('socket.getaddrinfo') as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [('family', 'type', 'proto', 'canonname', ('76.76.21.21', 443))]
            
            result = await self.client._test_dns_resolution("context7.com")
            
            # Should succeed with the correct hostname
            assert result is True
            
            # Should have called getaddrinfo with correct hostname
            mock_getaddrinfo.assert_called_with("context7.com", 443, 0, 1)
    
    @pytest.mark.asyncio
    async def test_connectivity_test_integration(self):
        """Test the full connectivity test with proper DNS and API checks."""
        with patch.object(self.client, '_test_dns_resolution') as mock_dns_test, \
             patch.object(self.client, 'health_check') as mock_health_check:
            
            # Mock successful DNS resolution
            mock_dns_test.return_value = True
            # Mock successful health check
            mock_health_check.return_value = True
            
            result = await self.client.test_connectivity()
            
            # Should succeed
            assert result is True
            
            # Should test DNS resolution for correct hostname
            mock_dns_test.assert_called_once_with("context7.com")
            
            # Should perform health check with direct client
            mock_health_check.assert_called()
    
    @pytest.mark.asyncio
    async def test_connectivity_fails_on_dns_failure(self):
        """Test that connectivity test fails when DNS resolution fails."""
        with patch.object(self.client, '_test_dns_resolution') as mock_dns_test:
            # Mock DNS resolution failure
            mock_dns_test.return_value = False
            
            result = await self.client.test_connectivity()
            
            # Should fail due to DNS issues
            assert result is False
            
            # Should test DNS resolution for correct hostname
            mock_dns_test.assert_called_once_with("context7.com")


class TestContext7APIIntegration:
    """Integration tests for Context7 API functionality."""
    
    def test_url_building_with_correct_base(self):
        """Test that URLs are built correctly with the new base URL."""
        client = Context7Client()
        
        # Test search URL building
        search_url = client._build_url("search")
        assert search_url == "https://context7.com/api/v1/search"
        
        # Test documentation URL building  
        docs_url = client._build_url("library/docs")
        assert docs_url == "https://context7.com/api/v1/library/docs"
    
    def test_session_headers_configuration(self):
        """Test that session headers are properly configured."""
        client = Context7Client()
        
        # Should have proper session headers
        assert "User-Agent" in client.session_headers
        assert "Accept" in client.session_headers
        assert client.session_headers["Accept"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_error_handling_improvements(self):
        """Test improved error handling for API responses."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()  # Use MagicMock instead of AsyncMock for properties
            mock_response.status_code = 400
            mock_response.text = "Invalid library format. Expected: username/library[/tag]"
            mock_response.json.side_effect = ValueError("Not JSON")  # Force fallback to text
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client
            
            # Should handle API errors gracefully
            with pytest.raises(Exception) as exc_info:
                await client.resolve_library_id("invalid-format")
            
            # Should contain useful error information
            assert "library format" in str(exc_info.value).lower()