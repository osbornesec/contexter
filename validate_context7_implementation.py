#!/usr/bin/env python3
"""
Validation script for Context7Client implementation.
Tests core functionality without requiring external dependencies.
"""

import asyncio
import sys
from unittest.mock import Mock, AsyncMock
from src.contexter.integration.context7_client import Context7Client
from src.contexter.models.context7_models import (
    LibrarySearchResult, 
    DocumentationResponse,
    RateLimitError,
    LibraryNotFoundError,
    ErrorCategory
)


async def test_library_search():
    """Test library search functionality with mocked responses."""
    print("Testing library search...")
    
    client = Context7Client()
    
    # Mock successful search
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "library_id": "test/library",
            "name": "Test Library",
            "description": "A test library",
            "trust_score": 8.5,
            "stars": 100
        }
    ]
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    
    with mock_client:
        # Mock httpx.AsyncClient creation
        import unittest.mock
        with unittest.mock.patch('httpx.AsyncClient', return_value=mock_client):
            results = await client.resolve_library_id("test library")
            
            assert len(results) == 1
            assert results[0].library_id == "test/library"
            assert results[0].name == "Test Library" 
            assert results[0].trust_score == 8.5
            assert results[0].star_count == 100
            print("✓ Library search successful")


async def test_documentation_retrieval():
    """Test documentation retrieval with mocked responses."""
    print("Testing documentation retrieval...")
    
    client = Context7Client()
    
    # Mock successful documentation response
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "content": "Test documentation content",
        "token_count": 5000
    }
    mock_response.headers = {}
    mock_response.content = b"Test documentation content"  # Add content attribute
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    
    with mock_client:
        import unittest.mock
        with unittest.mock.patch('httpx.AsyncClient', return_value=mock_client):
            result = await client.get_smart_docs("test/library", "test context")
            
            assert result.content == "Test documentation content"
            assert result.token_count == 5000
            assert result.library_id == "test/library"
            assert result.context == "test context"
            print("✓ Documentation retrieval successful")


async def test_rate_limiting():
    """Test rate limit handling."""
    print("Testing rate limit handling...")
    
    client = Context7Client()
    
    # Mock rate limited response
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"retry-after": "60"}
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    
    with mock_client:
        import unittest.mock
        with unittest.mock.patch('httpx.AsyncClient', return_value=mock_client):
            try:
                await client.get_smart_docs("test/library", "test context")
                assert False, "Should have raised RateLimitError"
            except RateLimitError as e:
                assert e.retry_after == 60
                print("✓ Rate limit handling successful")


async def test_error_classification():
    """Test error classification functionality.""" 
    print("Testing error classification...")
    
    client = Context7Client()
    classifier = client.error_classifier
    
    # Test different error types
    rate_limit_error = RateLimitError("Rate limited")
    category = classifier.classify_error(rate_limit_error)
    assert category == ErrorCategory.RATE_LIMITED
    
    not_found_error = LibraryNotFoundError("Not found")
    category = classifier.classify_error(not_found_error)
    assert category == ErrorCategory.NOT_FOUND
    
    # Test retry decisions
    should_retry, delay = classifier.should_retry(ErrorCategory.RATE_LIMITED)
    assert should_retry is True
    assert delay == 60.0
    
    should_retry, delay = classifier.should_retry(ErrorCategory.NOT_FOUND)
    assert should_retry is False
    assert delay == 0.0
    
    print("✓ Error classification successful")


async def test_caching():
    """Test search result caching."""
    print("Testing caching functionality...")
    
    client = Context7Client()
    
    # Test cache operations
    cache_key = client._generate_cache_key("test query")
    assert isinstance(cache_key, str)
    assert len(cache_key) == 32  # MD5 hash length
    
    # Test cache storage and retrieval
    test_results = [LibrarySearchResult("test/lib", "Test", "Description")]
    await client._cache_search_result(cache_key, test_results, "test query")
    
    cached_results = await client._get_cached_search(cache_key)
    assert cached_results is not None
    assert len(cached_results) == 1
    assert cached_results[0].library_id == "test/lib"
    
    print("✓ Caching functionality successful")


async def test_health_check():
    """Test health check functionality."""
    print("Testing health check...")
    
    client = Context7Client()
    
    # Mock healthy response
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.aclose = AsyncMock()
    
    with mock_client:
        import unittest.mock
        with unittest.mock.patch('httpx.AsyncClient', return_value=mock_client):
            is_healthy = await client.health_check()
            assert is_healthy is True
            print("✓ Health check successful")


def test_models():
    """Test data model functionality."""
    print("Testing data models...")
    
    # Test LibrarySearchResult
    result = LibrarySearchResult(
        library_id="test/lib",
        name="Test Library",
        description="Test description",
        trust_score=8.5,
        star_count=150
    )
    
    assert result.is_trusted is True  # trust_score >= 7.0
    assert "⭐ 150" in result.display_name
    
    # Test DocumentationResponse
    doc_response = DocumentationResponse(
        content="Test content",
        token_count=1000,
        library_id="test/lib",
        context="test context",
        response_time=2.5
    )
    
    assert doc_response.tokens_per_second == 400.0  # 1000/2.5
    assert doc_response.content_length > 0
    
    print("✓ Data models successful")


async def test_statistics():
    """Test statistics and metrics functionality."""
    print("Testing statistics...")
    
    client = Context7Client()
    
    # Test initial stats
    stats = client.get_stats()
    assert stats['total_requests'] == 0
    assert stats['success_rate'] == 0
    
    # Test stats after simulated requests
    await client._update_request_stats(success=True, tokens=1000, response_time=2.0)
    await client._update_request_stats(success=False, rate_limited=True)
    
    stats = client.get_stats()
    assert stats['total_requests'] == 2
    assert stats['successful_requests'] == 1
    assert stats['rate_limited_requests'] == 1
    assert stats['success_rate'] == 50.0
    
    print("✓ Statistics successful")


async def main():
    """Run all validation tests."""
    print("=== Context7Client Implementation Validation ===\n")
    
    try:
        # Test models first (non-async)
        test_models()
        
        # Test core functionality (all async)
        await test_statistics()
        await test_library_search()
        await test_documentation_retrieval()
        await test_rate_limiting()
        await test_error_classification()
        await test_caching()
        await test_health_check()
        
        print("\n🎉 All validation tests passed!")
        print("\nContext7Client implementation is ready for integration with:")
        print("- BrightData proxy manager")
        print("- C7DocDownloader download engine")
        print("- Configuration management system")
        print("- Storage and deduplication systems")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))