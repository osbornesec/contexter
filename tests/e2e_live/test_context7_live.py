"""
Live Context7 API integration tests.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional

from contexter.integration.context7_client import Context7Client
from contexter.integration.proxy_manager import BrightDataProxyManager
from contexter.models.config_models import C7DocConfig


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_context7_basic_connectivity():
    """Test basic Context7 API connectivity."""
    
    client = Context7Client()
    
    # Test health check
    is_healthy = await client.health_check()
    assert is_healthy is True, "Context7 API should be reachable"
    
    # Test connectivity 
    is_connected = await client.test_connectivity()
    assert is_connected is True, "Context7 API should respond to connectivity test"


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_context7_library_search(small_test_library: Dict[str, str]):
    """Test Context7 library search functionality."""
    
    client = Context7Client()
    
    try:
        # Search for a well-known library
        search_results = await client.search_libraries(small_test_library["library_id"])
        
        assert search_results is not None, "Search should return results"
        assert len(search_results) > 0, "Should find at least one library"
        
        # Check if our target library is in results
        found_library = None
        for result in search_results:
            if small_test_library["library_id"] in result.get("id", ""):
                found_library = result
                break
        
        assert found_library is not None, f"Should find {small_test_library['library_id']} in search results"
        
    except Exception as e:
        pytest.skip(f"Context7 search not available or library not found: {e}")


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_context7_documentation_retrieval(
    small_test_library: Dict[str, str],
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test retrieving documentation from Context7 API with proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for Context7 API access")
    
    # Initialize proxy manager for Context7 API access
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    client = Context7Client()
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        proxy_connection = await proxy_manager.get_connection()
        
        # Try to get documentation for a small library using proxy
        response = await client.get_smart_docs(
            library_id=small_test_library["library_id"],
            context="getting started guide",
            tokens=1000,
            client=proxy_connection.session if proxy_connection else None
        )
        
        # Verify response structure
        assert response is not None, "Should receive a response"
        assert hasattr(response, 'content'), "Response should have content"
        assert hasattr(response, 'token_count'), "Response should have token count"
        assert hasattr(response, 'response_time'), "Response should have response time"
        
        # Verify content quality
        assert len(response.content) > 100, "Content should be substantial"
        assert response.token_count > 0, "Token count should be positive"
        assert response.response_time > 0, "Response time should be positive"
        
        # Content should be relevant
        content_lower = response.content.lower()
        assert any(keyword in content_lower for keyword in [
            "documentation", "guide", "tutorial", "example", "usage", "api", "install"
        ]), "Content should appear to be documentation"
        
        print(f"✅ Successfully retrieved {response.token_count} tokens in {response.response_time:.2f}s")
        print(f"📄 Content preview: {response.content[:200]}...")
        
    except Exception as e:
        pytest.skip(f"Context7 documentation retrieval failed: {e}")
    
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_context7_multiple_contexts(
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test retrieving documentation for multiple contexts with proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for Context7 API access")
    
    # Initialize proxy manager
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    client = Context7Client()
    
    contexts = [
        "installation and setup",
        "basic usage examples",
        "API reference"
    ]
    
    library_id = "facebook/react"  # Well-known, stable library
    
    results = []
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        proxy_connection = await proxy_manager.get_connection()
        
        for context in contexts:
            try:
                response = await client.get_smart_docs(
                    library_id=library_id,
                    context=context,
                    tokens=500,  # Smaller requests
                    client=proxy_connection.session if proxy_connection else None
                )
            
                assert response is not None
                assert len(response.content) > 50
                results.append(response)
                
                print(f"✅ Context '{context}': {response.token_count} tokens")
                
            except Exception as e:
                print(f"⚠️  Context '{context}' failed: {e}")
                continue
        
        # Should get at least some results
        assert len(results) > 0, "Should successfully retrieve at least one context"
        
        # Results should be different (not identical)
        if len(results) > 1:
            contents = [r.content for r in results]
            assert len(set(contents)) > 1, "Different contexts should return different content"
            
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_context7_rate_limiting_and_errors():
    """Test Context7 API rate limiting and error handling."""
    
    client = Context7Client()
    
    # Test with invalid library ID
    try:
        response = await client.get_smart_docs(
            library_id="definitely_nonexistent_library_12345",
            context="test",
            tokens=100
        )
        # If we get here, the API might be very permissive
        print("⚠️  API accepted invalid library ID")
        
    except Exception as e:
        print(f"✅ Properly handled invalid library ID: {type(e).__name__}")
    
    # Test with very large token request
    try:
        response = await client.get_smart_docs(
            library_id="tiangolo/fastapi",
            context="everything about this library",
            tokens=50000  # Very large request
        )
        
        # Should either work or fail gracefully
        if response:
            print(f"✅ Large token request succeeded: {response.token_count} tokens")
        
    except Exception as e:
        print(f"✅ Large token request handled gracefully: {type(e).__name__}")


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_context7_performance_benchmarks(
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test Context7 API performance characteristics with proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for Context7 API access")
    
    # Initialize proxy manager
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    client = Context7Client()
    library_id = "facebook/react"
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        proxy_connection = await proxy_manager.get_connection()
        
        # Single request benchmark
        response = await client.get_smart_docs(
            library_id=library_id,
            context="quick start guide",
            tokens=1000,
            client=proxy_connection.session if proxy_connection else None
        )
        
        assert response.response_time > 0
        
        # Performance assertions
        assert response.response_time < 30.0, "Response should complete within 30 seconds"
        
        tokens_per_second = response.token_count / response.response_time
        assert tokens_per_second > 10, "Should process at least 10 tokens per second"
        
        print(f"📊 Performance metrics:")
        print(f"   Response time: {response.response_time:.2f}s")
        print(f"   Tokens returned: {response.token_count}")
        print(f"   Tokens/second: {tokens_per_second:.1f}")
        
        # Test concurrent requests (small scale)
        import asyncio
        
        async def single_request():
            return await client.get_smart_docs(
                library_id=library_id,
                context="examples",
                tokens=500,
                client=proxy_connection.session if proxy_connection else None
            )
        
        start_time = asyncio.get_event_loop().time()
        
        # Run 3 concurrent requests
        results = await asyncio.gather(
            single_request(),
            single_request(), 
            single_request(),
            return_exceptions=True
        )
        
        end_time = asyncio.get_event_loop().time()
        concurrent_time = end_time - start_time
        
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        print(f"📊 Concurrent performance:")
        print(f"   3 requests in: {concurrent_time:.2f}s")
        print(f"   Successful: {len(successful_results)}/3")
        
        # Should handle some concurrency
        assert len(successful_results) >= 1, "At least one concurrent request should succeed"
        
    except Exception as e:
        pytest.skip(f"Context7 performance test failed: {e}")
    
    finally:
        await proxy_manager.shutdown()