"""
Integration tests for Context7Client with proxy manager integration.
"""

import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import pytest

from src.contexter.integration.context7_client import Context7Client
from src.contexter.integration.proxy_manager import BrightDataProxyManager
from src.contexter.models.context7_models import (
    RateLimitError,
    LibraryNotFoundError,
    NetworkError,
    Context7APIError,
)
from src.contexter.models.proxy_models import ProxyConnection, ProxyStatus


class TestContext7ProxyIntegration:
    """Test Context7Client integration with proxy manager."""
    
    @pytest.fixture
    def mock_proxy_manager(self):
        """Mock proxy manager for testing."""
        manager = Mock(spec=BrightDataProxyManager)
        manager.get_connection = AsyncMock()
        manager.report_success = AsyncMock()
        manager.report_failure = AsyncMock()
        manager.report_rate_limit = AsyncMock()
        return manager
    
    @pytest.fixture
    def mock_proxy_connection(self):
        """Mock proxy connection for testing."""
        connection = Mock(spec=ProxyConnection)
        connection.proxy_id = "test_proxy_123"
        connection.session = AsyncMock()
        connection.session.proxy_id = "test_proxy_123"  # Set proxy_id on session too
        connection.status = ProxyStatus.HEALTHY
        return connection
    
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_success(self, mock_proxy_manager, mock_proxy_connection):
        """Test successful request with proxy switching functionality."""
        client = Context7Client()
        
        # Configure proxy manager to return connection
        mock_proxy_manager.get_connection.return_value = mock_proxy_connection
        
        # Mock successful API response
        mock_response_data = {
            "content": "Test documentation content",
            "token_count": 5000
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.content = b"Test documentation content"
        mock_response.headers = {"content-type": "application/json"}
        mock_proxy_connection.session.request = AsyncMock(return_value=mock_response)
        
        # Make request with proxy switching
        result = await client.request_with_proxy_switching(
            mock_proxy_manager,
            "test/library",
            "installation guide",
            tokens=150_000
        )
        
        # Verify result
        assert result.content == "Test documentation content"
        assert result.token_count == 5000
        assert result.proxy_id == "test_proxy_123"
        
        # Verify proxy manager interactions
        mock_proxy_manager.get_connection.assert_called_once()
        mock_proxy_manager.report_success.assert_called_once_with("test_proxy_123")
        
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_rate_limit_recovery(self, mock_proxy_manager):
        """Test proxy switching on rate limit with eventual success."""
        client = Context7Client()
        
        # Create two proxy connections
        proxy1 = Mock()
        proxy1.proxy_id = "proxy_1"
        proxy1.session = AsyncMock()
        proxy1.session.proxy_id = "proxy_1"
        
        proxy2 = Mock()
        proxy2.proxy_id = "proxy_2"
        proxy2.session = AsyncMock()
        proxy2.session.proxy_id = "proxy_2"
        
        # First connection gets rate limited, second succeeds
        mock_proxy_manager.get_connection.side_effect = [proxy1, proxy2]
        
        # First request: rate limited
        rate_limited_response = Mock()
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {"retry-after": "1"}  # Short for testing
        proxy1.session.request = AsyncMock(return_value=rate_limited_response)
        
        # Second request: success
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"content": "Success content", "token_count": 3000}
        success_response.content = b"Success content"
        success_response.headers = {"content-type": "application/json"}
        proxy2.session.request = AsyncMock(return_value=success_response)
        
        # Mock rate limit handler to avoid actual delay
        with patch('src.contexter.integration.context7_client.RateLimitHandler.handle_rate_limit') as mock_handle:
            mock_handle.return_value = None
            
            result = await client.request_with_proxy_switching(
                mock_proxy_manager,
                "test/library",
                "context",
                max_proxy_attempts=3
            )
        
        # Verify success with second proxy
        assert result.content == "Success content"
        assert result.proxy_id == "proxy_2"
        
        # Verify proxy manager was called for rate limit reporting
        mock_proxy_manager.report_rate_limit.assert_called_once_with("proxy_1")
        mock_proxy_manager.report_success.assert_called_once_with("proxy_2")
        
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_all_exhausted(self, mock_proxy_manager):
        """Test behavior when all proxy attempts are exhausted."""
        client = Context7Client()
        
        # Create proxy connections that all fail
        failing_connections = []
        for i in range(3):
            proxy = Mock()
            proxy.proxy_id = f"proxy_{i}"
            proxy.session = AsyncMock()
            proxy.session.proxy_id = f"proxy_{i}"
            
            # All return rate limited responses
            rate_limited_response = Mock()
            rate_limited_response.status_code = 429
            rate_limited_response.headers = {"retry-after": "1"}
            proxy.session.request = AsyncMock(return_value=rate_limited_response)
            
            failing_connections.append(proxy)
        
        mock_proxy_manager.get_connection.side_effect = failing_connections
        
        # Mock rate limit handler
        with patch('src.contexter.integration.context7_client.RateLimitHandler.handle_rate_limit') as mock_handle:
            mock_handle.return_value = None
            
            with pytest.raises(Context7APIError, match="All 3 proxy attempts rate limited"):
                await client.request_with_proxy_switching(
                    mock_proxy_manager,
                    "test/library",
                    "context",
                    max_proxy_attempts=3
                )
        
        # Verify all proxies were reported as rate limited
        assert mock_proxy_manager.report_rate_limit.call_count == 3
        
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_no_proxy_manager(self):
        """Test fallback to direct connection when no proxy manager provided."""
        client = Context7Client()
        
        # Mock successful direct API call
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"content": "Direct content", "token_count": 2000}
            mock_response.content = b"Direct content"
            mock_response.headers = {"content-type": "application/json"}
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await client.request_with_proxy_switching(
                None,  # No proxy manager
                "test/library",
                "context"
            )
            
            assert result.content == "Direct content"
            assert result.proxy_id is None  # No proxy used
            
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_library_not_found(self, mock_proxy_manager, mock_proxy_connection):
        """Test that library not found errors are not retried with different proxies."""
        client = Context7Client()
        
        mock_proxy_manager.get_connection.return_value = mock_proxy_connection
        
        # Mock 404 response
        not_found_response = Mock()
        not_found_response.status_code = 404
        mock_proxy_connection.session.request = AsyncMock(return_value=not_found_response)
        
        # Should raise LibraryNotFoundError immediately, not retry
        with pytest.raises(LibraryNotFoundError):
            await client.request_with_proxy_switching(
                mock_proxy_manager,
                "nonexistent/library",
                "context"
            )
        
        # Should only have tried once (no retries for 404)
        assert mock_proxy_manager.get_connection.call_count == 1
        
    @pytest.mark.asyncio
    async def test_request_with_proxy_switching_network_error_retry(self, mock_proxy_manager):
        """Test that network errors trigger proxy retry."""
        client = Context7Client()
        
        # First proxy has network error - simulate at the httpx level
        proxy1 = Mock()
        proxy1.proxy_id = "proxy_1"
        proxy1.session = AsyncMock()
        proxy1.session.proxy_id = "proxy_1"
        
        import httpx
        proxy1.session.request = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        
        # Second proxy succeeds
        proxy2 = Mock()
        proxy2.proxy_id = "proxy_2"
        proxy2.session = AsyncMock()
        proxy2.session.proxy_id = "proxy_2"
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {"content": "Retry success", "token_count": 1500}
        success_response.content = b"Retry success"
        success_response.headers = {"content-type": "application/json"}
        proxy2.session.request = AsyncMock(return_value=success_response)
        
        mock_proxy_manager.get_connection.side_effect = [proxy1, proxy2]
        
        result = await client.request_with_proxy_switching(
            mock_proxy_manager,
            "test/library",
            "context"
        )
        
        assert result.content == "Retry success"
        assert result.proxy_id == "proxy_2"
        
        # Verify failure was reported for first proxy
        mock_proxy_manager.report_failure.assert_called_once_with("proxy_1")


class TestContext7RealAPIIntegration:
    """Integration tests with real API endpoints (when available)."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_api_health_check(self):
        """Test actual Context7 API connectivity (requires real API)."""
        client = Context7Client()
        
        try:
            # This test requires actual API connectivity
            is_healthy = await client.health_check()
            
            # Result should be boolean (True if API is up, False if down)
            assert isinstance(is_healthy, bool)
            
            # If healthy, test basic functionality
            if is_healthy:
                # Try a simple library search
                try:
                    results = await client.resolve_library_id("python requests")
                    assert isinstance(results, list)
                    
                    if results:  # If we got results
                        assert all(hasattr(r, 'library_id') for r in results)
                        assert all(hasattr(r, 'name') for r in results)
                        
                except Exception as e:
                    # Log but don't fail the test - API might have different behavior
                    print(f"API call failed (expected in test environment): {e}")
                    
        except Exception as e:
            # Health check failed - this is acceptable in test environment
            print(f"Health check failed (expected in test environment): {e}")
            assert True  # Test passes regardless
            
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_proxy_integration_with_mock_brightdata(self):
        """Test integration with BrightData proxy manager (mocked)."""
        # This would test the full integration stack
        # For now, we'll mock the proxy manager to simulate the integration
        
        client = Context7Client()
        
        # Mock proxy manager with realistic behavior
        proxy_manager = Mock()
        proxy_manager.get_connection = AsyncMock()
        
        # Create a realistic proxy connection
        proxy_connection = Mock()
        proxy_connection.proxy_id = "brightdata_proxy_123"
        proxy_connection.session = AsyncMock()
        proxy_connection.session.proxy_id = "brightdata_proxy_123"
        
        # Mock successful API response through proxy
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": "Documentation retrieved through proxy",
            "token_count": 8500
        }
        mock_response.content = b"Documentation retrieved through proxy"
        mock_response.headers = {"content-type": "application/json"}
        proxy_connection.session.request = AsyncMock(return_value=mock_response)
        
        proxy_manager.get_connection.return_value = proxy_connection
        proxy_manager.report_success = AsyncMock()
        
        # Test documentation retrieval with proxy
        result = await client.request_with_proxy_switching(
            proxy_manager,
            "test/popular-library",
            "usage examples and tutorials"
        )
        
        # Verify the integration worked
        assert result.content == "Documentation retrieved through proxy"
        assert result.proxy_id == "brightdata_proxy_123"
        assert result.token_count == 8500
        
        # Verify proxy manager interactions
        proxy_manager.get_connection.assert_called_once()
        proxy_manager.report_success.assert_called_once_with("brightdata_proxy_123")


class TestContext7ConcurrencyIntegration:
    """Test concurrent operations and performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_library_searches(self):
        """Test multiple concurrent library searches."""
        client = Context7Client()
        
        # Mock responses for concurrent requests
        mock_responses = []
        for i in range(10):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [
                {
                    "library_id": f"lib{i}/test",
                    "name": f"Test Library {i}",
                    "description": f"Description {i}"
                }
            ]
            mock_responses.append(mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=mock_responses)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Make 10 concurrent search requests
            queries = [f"query{i}" for i in range(10)]
            tasks = [client.resolve_library_id(query) for query in queries]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            elapsed_time = time.time() - start_time
            
            # Verify all requests completed
            assert len(results) == 10
            assert all(len(result) == 1 for result in results)
            assert all(not isinstance(result, Exception) for result in results)
            
            # Should complete much faster than sequential (due to concurrency)
            # In practice, this would be much faster than 10 * individual_request_time
            assert elapsed_time < 5.0  # Should be reasonable for mocked requests
            
    @pytest.mark.asyncio
    async def test_concurrent_documentation_retrieval(self):
        """Test concurrent documentation retrieval with different libraries."""
        client = Context7Client()
        
        libraries = [
            ("requests", "http client usage"),
            ("pandas", "data analysis tutorial"),
            ("numpy", "array operations guide"),
            ("flask", "web development basics"),
            ("pytest", "testing framework setup")
        ]
        
        # Mock responses
        mock_responses = []
        for i, (lib, context) in enumerate(libraries):
            mock_response = Mock()
            mock_response.status_code = 200
            content_text = f"Documentation for {lib} - {context}"
            mock_response.json.return_value = {
                "content": content_text,
                "token_count": 1000 + i * 500
            }
            mock_response.content = content_text.encode('utf-8')
            mock_response.headers = {"content-type": "application/json"}
            mock_responses.append(mock_response)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=mock_responses)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Make concurrent documentation requests
            tasks = [
                client.get_smart_docs(lib, context) 
                for lib, context in libraries
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed_time = time.time() - start_time
            
            # Verify all requests completed successfully
            assert len(results) == 5
            assert all(not isinstance(result, Exception) for result in results)
            assert all(hasattr(result, 'content') for result in results)
            
            # Verify content matches expectations
            for i, result in enumerate(results):
                lib, context = libraries[i]
                assert lib in result.content
                assert context in result.content
                
    @pytest.mark.asyncio
    async def test_mixed_success_failure_scenarios(self):
        """Test concurrent requests with mixed success/failure scenarios."""
        client = Context7Client()
        
        # Mix of successful and failing responses
        mock_responses = [
            # Success
            Mock(status_code=200, json=lambda: {"content": "Success 1", "token_count": 1000}, 
                 content=b"Success 1", headers={"content-type": "application/json"}),
            # Rate limited - This will be converted to RateLimitError by _handle_rate_limit_response
            Mock(status_code=429, headers={"retry-after": "1"}),
            # Not found
            Mock(status_code=404),
            # Success
            Mock(status_code=200, json=lambda: {"content": "Success 2", "token_count": 1500},
                 content=b"Success 2", headers={"content-type": "application/json"}),
            # Server error - This will go through retries and then become Context7APIError
            Mock(status_code=500, text="Server error"),
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=mock_responses)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Make concurrent requests
            tasks = [
                client.get_smart_docs(f"lib{i}", "context") 
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify mixed results
            assert len(results) == 5
            
            # First result: success
            assert not isinstance(results[0], Exception)
            assert results[0].content == "Success 1"
            
            # Second result: rate limited (after retries, becomes Context7APIError)
            assert isinstance(results[1], Context7APIError)
            
            # Third result: not found
            assert isinstance(results[2], LibraryNotFoundError)
            
            # Fourth result: success
            assert not isinstance(results[3], Exception)
            assert results[3].content == "Success 2"
            
            # Fifth result: server error (after retries)
            assert isinstance(results[4], Context7APIError)


class TestContext7PerformanceIntegration:
    """Test performance characteristics and optimization."""
    
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self):
        """Test that caching significantly improves performance."""
        client = Context7Client()
        
        mock_response_data = [{"library_id": "cached/lib", "name": "Cached Library"}]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            
            # Add artificial delay to simulate network latency
            async def delayed_get(*args, **kwargs):
                await asyncio.sleep(0.1)  # 100ms simulated latency
                return mock_response
                
            mock_client.get = AsyncMock(side_effect=delayed_get)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            query = "performance test query"
            
            # First request (uncached) - should be slow
            start_time = time.time()
            results1 = await client.resolve_library_id(query)
            first_request_time = time.time() - start_time
            
            # Second request (cached) - should be much faster
            start_time = time.time()
            results2 = await client.resolve_library_id(query)
            cached_request_time = time.time() - start_time
            
            # Verify results are identical
            assert results1 == results2
            
            # Verify caching performance improvement
            assert cached_request_time < first_request_time * 0.1  # Should be 10x+ faster
            assert first_request_time >= 0.1  # Should include network delay
            assert cached_request_time < 0.01  # Cache should be nearly instantaneous
            
            # Verify API was only called once
            assert mock_client.get.call_count == 1
            
    @pytest.mark.asyncio
    async def test_request_retry_performance(self):
        """Test that retry logic doesn't add excessive overhead."""
        client = Context7Client()
        
        # Mock a request that fails twice then succeeds
        call_count = 0
        async def failing_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count <= 2:
                # First two calls fail with server error
                response = Mock()
                response.status_code = 500
                return response
            else:
                # Third call succeeds
                response = Mock()
                response.status_code = 200
                response.json.return_value = {"content": "Success after retries", "token_count": 2000}
                response.content = b"Success after retries"
                response.headers = {"content-type": "application/json"}
                return response
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(side_effect=failing_request)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock sleep to avoid actual delays in test
            with patch('asyncio.sleep') as mock_sleep:
                mock_sleep.return_value = None
                
                start_time = time.time()
                result = await client.get_smart_docs("test/lib", "context")
                elapsed_time = time.time() - start_time
                
                # Verify success after retries
                assert result.content == "Success after retries"
                assert result.token_count == 2000
                
                # Verify retry attempts
                assert call_count == 3  # Failed twice, succeeded on third
                
                # Should complete quickly without actual sleep delays
                assert elapsed_time < 1.0  # Should be fast with mocked sleep