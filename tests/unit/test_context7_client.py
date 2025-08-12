"""
Unit tests for Context7Client API integration.
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import httpx

from src.contexter.integration.context7_client import (
    Context7Client,
    RateLimitHandler,
    APIErrorClassifier,
)
from src.contexter.models.context7_models import (
    DocumentationResponse,
    LibrarySearchResult,
    Context7APIError,
    RateLimitError,
    LibraryNotFoundError,
    AuthenticationError,
    NetworkError,
    InvalidResponseError,
    ConfigurationError,
    ErrorCategory,
)


class TestContext7ClientInitialization:
    """Test Context7Client initialization and configuration."""
    
    def test_client_initialization_defaults(self):
        """Test client initialization with default parameters."""
        client = Context7Client()
        
        assert client.base_url == "https://context7.com/api/v1"
        assert client.default_timeout.read == 30.0
        assert client.cache_ttl == 300.0
        assert client.max_cache_size == 100
        assert len(client._search_cache) == 0
        assert client.metrics.request_count == 0
        
    def test_client_initialization_custom(self):
        """Test client initialization with custom parameters."""
        client = Context7Client(
            base_url="https://custom.api.com",
            default_timeout=45.0,
            cache_ttl=600.0,
            max_cache_size=50
        )
        
        assert client.base_url == "https://custom.api.com"
        assert client.default_timeout.read == 45.0
        assert client.cache_ttl == 600.0
        assert client.max_cache_size == 50
        
    def test_build_url(self):
        """Test URL building functionality."""
        client = Context7Client(base_url="https://api.example.com/v1")
        
        assert client._build_url("endpoint") == "https://api.example.com/v1/endpoint"
        assert client._build_url("/endpoint") == "https://api.example.com/v1/endpoint"
        assert client._build_url("path/to/endpoint") == "https://api.example.com/v1/path/to/endpoint"


class TestRateLimitHandler:
    """Test rate limit detection and handling."""
    
    def test_detect_rate_limit_with_header(self):
        """Test rate limit detection with retry-after header."""
        handler = RateLimitHandler()
        
        # Mock response with rate limit
        response = Mock()
        response.status_code = 429
        response.headers = {"retry-after": "120"}
        
        is_rate_limited, retry_after = handler.detect_rate_limit(response)
        
        assert is_rate_limited is True
        assert retry_after == 120
        
    def test_detect_rate_limit_without_header(self):
        """Test rate limit detection without retry-after header."""
        handler = RateLimitHandler()
        
        response = Mock()
        response.status_code = 429
        response.headers = {}
        
        is_rate_limited, retry_after = handler.detect_rate_limit(response)
        
        assert is_rate_limited is True
        assert retry_after == 60  # Default fallback
        
    def test_detect_no_rate_limit(self):
        """Test detection when not rate limited."""
        handler = RateLimitHandler()
        
        response = Mock()
        response.status_code = 200
        response.headers = {}
        
        is_rate_limited, retry_after = handler.detect_rate_limit(response)
        
        assert is_rate_limited is False
        assert retry_after is None
        
    @pytest.mark.asyncio
    async def test_handle_rate_limit(self):
        """Test rate limit handling with delay."""
        handler = RateLimitHandler()
        
        start_time = time.time()
        
        # Use a very short retry time for testing
        with patch('asyncio.sleep') as mock_sleep:
            await handler.handle_rate_limit(1, proxy_id="test_proxy")
            
            # Verify sleep was called with retry + jitter
            mock_sleep.assert_called_once()
            delay_arg = mock_sleep.call_args[0][0]
            assert 2.0 <= delay_arg <= 11.0  # 1 second + 1-10 jitter


class TestAPIErrorClassifier:
    """Test API error classification."""
    
    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        classifier = APIErrorClassifier()
        
        error = RateLimitError("Rate limited")
        category = classifier.classify_error(error)
        
        assert category == ErrorCategory.RATE_LIMITED
        
    def test_classify_library_not_found_error(self):
        """Test classification of library not found errors."""
        classifier = APIErrorClassifier()
        
        error = LibraryNotFoundError("Library not found")
        category = classifier.classify_error(error)
        
        assert category == ErrorCategory.NOT_FOUND
        
    def test_classify_network_error(self):
        """Test classification of network errors."""
        classifier = APIErrorClassifier()
        
        error = httpx.ConnectError("Connection failed")
        category = classifier.classify_error(error)
        
        assert category == ErrorCategory.NETWORK
        
    def test_classify_by_response_status(self):
        """Test classification based on HTTP response status."""
        classifier = APIErrorClassifier()
        
        # Test 401 Unauthorized
        response = Mock()
        response.status_code = 401
        category = classifier.classify_error(Exception(), response)
        assert category == ErrorCategory.AUTHENTICATION
        
        # Test 404 Not Found
        response.status_code = 404
        category = classifier.classify_error(Exception(), response)
        assert category == ErrorCategory.NOT_FOUND
        
        # Test 429 Rate Limited
        response.status_code = 429
        category = classifier.classify_error(Exception(), response)
        assert category == ErrorCategory.RATE_LIMITED
        
        # Test 500 Server Error
        response.status_code = 500
        category = classifier.classify_error(Exception(), response)
        assert category == ErrorCategory.API_ERROR
        
    def test_should_retry_decisions(self):
        """Test retry decision logic."""
        classifier = APIErrorClassifier()
        
        # Should retry rate limited
        should_retry, delay = classifier.should_retry(ErrorCategory.RATE_LIMITED)
        assert should_retry is True
        assert delay == 60.0
        
        # Should not retry not found
        should_retry, delay = classifier.should_retry(ErrorCategory.NOT_FOUND)
        assert should_retry is False
        assert delay == 0.0
        
        # Should retry network errors
        should_retry, delay = classifier.should_retry(ErrorCategory.NETWORK)
        assert should_retry is True
        assert delay == 5.0


class TestContext7ClientHealthCheck:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            is_healthy = await client.health_check()
            
            assert is_healthy is True
            mock_client.get.assert_called_once()
            mock_client.aclose.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_health_check_404_ok(self):
        """Test health check with 404 response (acceptable)."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 404  # No results, but API is up
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            is_healthy = await client.health_check()
            
            assert is_healthy is True
            
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check failure."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            is_healthy = await client.health_check()
            
            assert is_healthy is False
            
    @pytest.mark.asyncio
    async def test_health_check_with_existing_client(self):
        """Test health check using provided client."""
        client = Context7Client()
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)
        
        is_healthy = await client.health_check(client=mock_client)
        
        assert is_healthy is True
        # Should not call aclose() on provided client
        mock_client.aclose.assert_not_called()


class TestLibrarySearch:
    """Test library search functionality."""
    
    @pytest.mark.asyncio
    async def test_resolve_library_id_success(self):
        """Test successful library resolution."""
        client = Context7Client()
        
        mock_response_data = [
            {
                "library_id": "test/library",
                "name": "Test Library", 
                "description": "A test library",
                "trust_score": 8.5,
                "stars": 100
            }
        ]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            results = await client.resolve_library_id("test library")
            
            assert len(results) == 1
            assert results[0].library_id == "test/library"
            assert results[0].name == "Test Library"
            assert results[0].trust_score == 8.5
            assert results[0].star_count == 100
            assert results[0].search_relevance > 0  # Should have some relevance
            
    @pytest.mark.asyncio
    async def test_resolve_library_id_empty_query(self):
        """Test library resolution with empty query."""
        client = Context7Client()
        
        with pytest.raises(ConfigurationError, match="Search query cannot be empty"):
            await client.resolve_library_id("")
            
        with pytest.raises(ConfigurationError, match="Search query cannot be empty"):
            await client.resolve_library_id("   ")
            
    @pytest.mark.asyncio
    async def test_resolve_library_id_no_results(self):
        """Test library resolution with no results."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            results = await client.resolve_library_id("nonexistent")
            
            assert len(results) == 0
            
    @pytest.mark.asyncio
    async def test_resolve_library_id_caching(self):
        """Test library search result caching."""
        client = Context7Client()
        
        mock_response_data = [{"library_id": "test/lib", "name": "Test"}]
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # First call should hit the API
            results1 = await client.resolve_library_id("test query")
            assert mock_client.get.call_count == 1
            
            # Second call should use cache
            results2 = await client.resolve_library_id("test query")
            assert mock_client.get.call_count == 1  # No additional API call
            assert results1 == results2
            
    @pytest.mark.asyncio
    async def test_resolve_library_id_rate_limited(self):
        """Test library resolution with rate limiting."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"retry-after": "60"}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            with pytest.raises(RateLimitError) as exc_info:
                await client.resolve_library_id("test query")
            
            assert exc_info.value.retry_after == 60
            
    @pytest.mark.asyncio
    async def test_parse_search_response_various_formats(self):
        """Test parsing different search response formats."""
        client = Context7Client()
        
        # Test direct list format
        mock_response = Mock()
        mock_response.json.return_value = [
            {"library_id": "test1", "name": "Test 1"}
        ]
        
        results = await client._parse_search_response(mock_response, "test")
        assert len(results) == 1
        assert results[0].library_id == "test1"
        
        # Test structured format with 'results' key
        mock_response.json.return_value = {
            "results": [{"library_id": "test2", "name": "Test 2"}]
        }
        
        results = await client._parse_search_response(mock_response, "test")
        assert len(results) == 1
        assert results[0].library_id == "test2"
        
        # Test malformed data
        mock_response.json.return_value = {"unexpected": "format"}
        results = await client._parse_search_response(mock_response, "test")
        assert len(results) == 0


class TestSmartDocumentationRetrieval:
    """Test smart documentation retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_smart_docs_success(self):
        """Test successful documentation retrieval."""
        client = Context7Client()
        
        mock_response_data = {
            "content": "Test documentation content",
            "token_count": 1000
        }
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data
            mock_response.content = b'{"content": "Test documentation content", "token_count": 1000}'
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = '{"content": "Test documentation content", "token_count": 1000}'
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            result = await client.get_smart_docs("test/library", "test context")
            
            assert result.content == "Test documentation content"
            assert result.token_count == 1000
            assert result.library_id == "test/library"
            assert result.context == "test context"
            assert result.response_time > 0
            
    @pytest.mark.asyncio
    async def test_get_smart_docs_validation(self):
        """Test input validation for get_smart_docs."""
        client = Context7Client()
        
        with pytest.raises(ConfigurationError, match="Library ID cannot be empty"):
            await client.get_smart_docs("", "context")
            
        with pytest.raises(ConfigurationError, match="Context cannot be empty"):
            await client.get_smart_docs("lib", "")
            
    @pytest.mark.asyncio
    async def test_get_smart_docs_token_clamping(self):
        """Test token count clamping to API limits."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"content": "test", "token_count": 1000}
            mock_response.content = b'{"content": "test", "token_count": 1000}'
            mock_response.headers = {"content-type": "application/json"}
            mock_response.text = '{"content": "test", "token_count": 1000}'
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Test token count gets clamped to maximum
            await client.get_smart_docs("test/lib", "context", tokens=300_000)
            
            # Verify API was called with clamped token count
            call_args = mock_client.request.call_args
            assert call_args[1]['params']['tokens'] == 200_000
            
            # Test token count gets clamped to minimum
            await client.get_smart_docs("test/lib", "context", tokens=500)
            call_args = mock_client.request.call_args
            assert call_args[1]['params']['tokens'] == 1000
            
    @pytest.mark.asyncio
    async def test_get_smart_docs_library_not_found(self):
        """Test handling of library not found error."""
        client = Context7Client()
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()
            mock_client_class.return_value = mock_client
            
            with pytest.raises(LibraryNotFoundError) as exc_info:
                await client.get_smart_docs("nonexistent/lib", "context")
            
            assert exc_info.value.library_id == "nonexistent/lib"
            
    @pytest.mark.asyncio
    async def test_parse_documentation_response_formats(self):
        """Test parsing different documentation response formats."""
        client = Context7Client()
        
        # Test direct string response
        mock_response = Mock()
        mock_response.json.return_value = "Direct content string"
        mock_response.headers = {}
        mock_response.content = b'"Direct content string"'
        mock_response.status_code = 200
        
        result = await client._parse_documentation_response(
            mock_response, "test/lib", "context", 1.0, None
        )
        
        assert result.content == "Direct content string"
        assert result.token_count > 0  # Should be estimated
        
        # Test structured JSON response
        mock_response.json.return_value = {
            "content": "Structured content",
            "token_count": 2000,
            "extra_metadata": "value"
        }
        mock_response.content = b'{"content": "Structured content", "token_count": 2000, "extra_metadata": "value"}'
        
        result = await client._parse_documentation_response(
            mock_response, "test/lib", "context", 1.0, None
        )
        
        assert result.content == "Structured content"
        assert result.token_count == 2000
        assert "extra_metadata" in result.metadata
        
        # Test plain text response (non-JSON)
        mock_response.json.side_effect = json.JSONDecodeError("msg", "doc", 0)
        mock_response.text = "Plain text content"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.content = b"Plain text content"
        
        result = await client._parse_documentation_response(
            mock_response, "test/lib", "context", 1.0, None
        )
        
        assert result.content == "Plain text content"
        assert result.metadata["content_type"] == "text/plain"
        
    def test_estimate_token_count(self):
        """Test token count estimation."""
        client = Context7Client()
        
        # Test empty string
        assert client._estimate_token_count("") == 0
        
        # Test simple text
        text = "This is a test with some words"
        tokens = client._estimate_token_count(text)
        assert tokens > 0
        assert tokens >= len(text.split())  # At least word count
        
        # Test longer text
        long_text = "This is a much longer text " * 100
        long_tokens = client._estimate_token_count(long_text)
        assert long_tokens > tokens


class TestStatisticsAndMetrics:
    """Test client statistics and metrics functionality."""
    
    @pytest.mark.asyncio 
    async def test_request_stats_tracking(self):
        """Test request statistics tracking."""
        client = Context7Client()
        
        # Initial state
        stats = client.get_stats()
        assert stats['total_requests'] == 0
        assert stats['success_rate'] == 0
        
        # Update stats
        await client._update_request_stats(success=True, tokens=1000, response_time=2.5)
        await client._update_request_stats(success=False, rate_limited=True)
        await client._update_request_stats(success=False)
        
        stats = client.get_stats()
        assert stats['total_requests'] == 3
        assert stats['successful_requests'] == 1
        assert stats['rate_limited_requests'] == 1
        assert stats['error_requests'] == 1
        assert abs(stats['success_rate'] - 33.33) < 0.01  # 1/3 * 100, allow small floating point diff
        assert stats['average_response_time'] == 2.5
        assert stats['total_tokens_retrieved'] == 1000
        
    @pytest.mark.asyncio
    async def test_cache_management(self):
        """Test search cache management."""
        client = Context7Client(max_cache_size=2)
        
        # Add cache entries
        await client._cache_search_result("key1", [], "query1")
        await client._cache_search_result("key2", [], "query2")
        
        assert len(client._search_cache) == 2
        
        # Adding third entry should evict oldest
        await client._cache_search_result("key3", [], "query3")
        assert len(client._search_cache) == 2
        assert "key1" not in client._search_cache  # Oldest evicted
        assert "key3" in client._search_cache
        
        # Test cache clearing
        await client.clear_cache()
        assert len(client._search_cache) == 0


class TestProxyIntegration:
    """Test proxy integration functionality."""
    
    def test_extract_proxy_id(self):
        """Test proxy ID extraction from client."""
        client = Context7Client()
        
        # Test with proxy ID attribute
        mock_client = Mock()
        mock_client.proxy_id = "test_proxy_123"
        
        proxy_id = client._extract_proxy_id(mock_client)
        assert proxy_id == "test_proxy_123"
        
        # Test without proxy ID attribute
        mock_client_no_proxy = Mock()
        del mock_client_no_proxy.proxy_id  # Ensure no proxy_id
        
        proxy_id = client._extract_proxy_id(mock_client_no_proxy)
        assert proxy_id is None
        
        # Test with None client
        proxy_id = client._extract_proxy_id(None)
        assert proxy_id is None
        
    @pytest.mark.asyncio
    async def test_get_connection_health(self):
        """Test connection health monitoring."""
        client = Context7Client()
        
        with patch.object(client, 'health_check') as mock_health_check:
            mock_health_check.return_value = True
            
            health_info = await client.get_connection_health()
            
            assert health_info['api_reachable'] is True
            assert health_info['status'] == 'healthy'
            assert health_info['response_time'] > 0
            assert 'performance' in health_info
            
        # Test health check failure
        with patch.object(client, 'health_check') as mock_health_check:
            mock_health_check.side_effect = Exception("Connection failed")
            
            health_info = await client.get_connection_health()
            
            assert health_info['api_reachable'] is False
            assert health_info['status'] == 'error'
            assert health_info['error'] == "Connection failed"


class TestBatchOperations:
    """Test batch operation functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_resolve_libraries_success(self):
        """Test successful batch library resolution."""
        client = Context7Client()
        
        with patch.object(client, 'resolve_library_id') as mock_resolve:
            mock_resolve.side_effect = [
                [LibrarySearchResult("lib1", "Library 1", "Description 1")],
                [LibrarySearchResult("lib2", "Library 2", "Description 2")],
            ]
            
            queries = ["query1", "query2"]
            results = await client.batch_resolve_libraries(queries)
            
            assert len(results) == 2
            assert "query1" in results
            assert "query2" in results
            assert len(results["query1"]) == 1
            assert results["query1"][0].library_id == "lib1"
            
    @pytest.mark.asyncio
    async def test_batch_resolve_libraries_with_failures(self):
        """Test batch resolution with some failures."""
        client = Context7Client()
        
        with patch.object(client, 'resolve_library_id') as mock_resolve:
            mock_resolve.side_effect = [
                [LibrarySearchResult("lib1", "Library 1", "Description 1")],
                Context7APIError("API failed"),
            ]
            
            queries = ["query1", "query2"]
            results = await client.batch_resolve_libraries(queries)
            
            assert len(results) == 2
            assert len(results["query1"]) == 1  # Success
            assert len(results["query2"]) == 0  # Failure returns empty list
            
    @pytest.mark.asyncio
    async def test_batch_resolve_libraries_empty(self):
        """Test batch resolution with empty query list."""
        client = Context7Client()
        
        results = await client.batch_resolve_libraries([])
        assert results == {}


# Additional test fixtures and utilities

@pytest.fixture
def sample_library_search_result():
    """Sample library search result for testing."""
    return LibrarySearchResult(
        library_id="test/library",
        name="Test Library",
        description="A test library for unit tests",
        trust_score=8.5,
        star_count=150,
        search_relevance=0.85
    )

@pytest.fixture
def sample_documentation_response():
    """Sample documentation response for testing."""
    return DocumentationResponse(
        content="# Test Library Documentation\n\nThis is test content.",
        token_count=2500,
        library_id="test/library",
        context="getting started tutorial",
        response_time=1.25,
        proxy_id="proxy_123"
    )

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.request = AsyncMock()
    client.aclose = AsyncMock()
    return client