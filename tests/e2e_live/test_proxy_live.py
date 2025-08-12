"""
Live BrightData proxy integration tests.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional

from contexter.integration.proxy_manager import BrightDataProxyManager
from contexter.models.config_models import C7DocConfig


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_proxy_basic_connectivity(brightdata_credentials: Optional[Dict[str, str]]):
    """Test basic BrightData proxy connectivity."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
        zone_name="zone_residential_1",
        dns_resolution="remote"
    )
    
    try:
        # Test basic connectivity
        is_connected = await proxy_manager.test_connectivity()
        assert is_connected is True, "BrightData proxy should be reachable"
        
        print("✅ BrightData proxy connectivity test passed")
        
    except Exception as e:
        pytest.fail(f"Proxy connectivity test failed: {e}")


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_proxy_pool_initialization(brightdata_credentials: Optional[Dict[str, str]]):
    """Test proxy pool initialization and management."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
        zone_name="zone_residential_1",
        dns_resolution="remote"
    )
    
    try:
        # Initialize minimum proxy pool for testing
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Check pool status
        status = proxy_manager.get_pool_status()
        
        assert status["total_connections"] == 10, "Should have 10 connections in pool"
        assert status["healthy"] > 0, "Should have at least some healthy connections"
        
        print(f"✅ Proxy pool initialized: {status}")
        
        # Test getting a connection
        connection = await proxy_manager.get_connection()
        assert connection is not None, "Should be able to get a connection from pool"
        
        print(f"✅ Successfully acquired connection: {connection.proxy_id}")
        
        # Release connection
        await proxy_manager.release_connection(connection)
        
    except Exception as e:
        pytest.fail(f"Proxy pool initialization failed: {e}")
    
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.proxy_required 
@pytest.mark.asyncio
async def test_proxy_http_requests(brightdata_credentials: Optional[Dict[str, str]]):
    """Test making HTTP requests through BrightData proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
    )
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        
        connection = await proxy_manager.get_connection()
        assert connection is not None
        
        # Test HTTP request through proxy
        response = await connection.session.get("http://httpbin.org/ip")
        
        assert response.status_code == 200, "HTTP request should succeed"
        
        response_data = response.json()
        assert "origin" in response_data, "Response should contain IP information"
        
        print(f"✅ Proxy HTTP request successful")
        print(f"   Origin IP: {response_data.get('origin', 'N/A')}")
        
        # Test HTTPS request
        https_response = await connection.session.get("https://httpbin.org/get")
        assert https_response.status_code == 200, "HTTPS request should succeed"
        
        print("✅ Proxy HTTPS request successful")
        
    except Exception as e:
        pytest.fail(f"Proxy HTTP requests failed: {e}")
    
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_proxy_health_monitoring(brightdata_credentials: Optional[Dict[str, str]]):
    """Test proxy health monitoring functionality."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
    )
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Start health monitoring (short interval for testing)
        await proxy_manager.start_health_monitoring(interval_seconds=30)
        
        # Wait a moment for initial health check
        import asyncio
        await asyncio.sleep(2)
        
        # Check metrics
        metrics = proxy_manager.get_proxy_metrics()
        assert len(metrics) > 0, "Should have proxy metrics available"
        
        print("✅ Health monitoring started successfully")
        print(f"   Monitoring {len(metrics)} proxies")
        
        # Stop monitoring
        await proxy_manager.stop_health_monitoring()
        
        print("✅ Health monitoring stopped successfully")
        
    except Exception as e:
        pytest.fail(f"Proxy health monitoring failed: {e}")
    
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_proxy_performance_benchmarks(brightdata_credentials: Optional[Dict[str, str]]):
    """Test proxy performance characteristics."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
    )
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        
        connection = await proxy_manager.get_connection()
        assert connection is not None
        
        # Performance benchmark
        import time
        start_time = time.time()
        
        response = await connection.session.get("http://httpbin.org/get")
        
        end_time = time.time()
        request_time = end_time - start_time
        
        assert response.status_code == 200
        assert request_time < 30.0, "Request should complete within 30 seconds"
        
        print(f"📊 Proxy performance:")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Status: {response.status_code}")
        
        # Test multiple requests
        request_times = []
        
        for i in range(3):
            start = time.time()
            resp = await connection.session.get("http://httpbin.org/get")
            end = time.time()
            
            assert resp.status_code == 200
            request_times.append(end - start)
        
        avg_time = sum(request_times) / len(request_times)
        print(f"   Average time (3 requests): {avg_time:.2f}s")
        
        # Performance assertions
        assert avg_time < 15.0, "Average request time should be reasonable"
        
    except Exception as e:
        pytest.fail(f"Proxy performance benchmarks failed: {e}")
    
    finally:
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.asyncio 
async def test_proxy_error_handling(brightdata_credentials: Optional[Dict[str, str]]):
    """Test proxy error handling and recovery."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
    )
    
    try:
        await proxy_manager.initialize_pool(pool_size=10)
        
        connection = await proxy_manager.get_connection()
        assert connection is not None
        
        # Test with invalid URL (should handle gracefully)
        try:
            response = await connection.session.get(
                "http://definitely-not-a-real-domain-12345.com",
                timeout=10
            )
        except Exception as e:
            print(f"✅ Properly handled invalid URL: {type(e).__name__}")
        
        # Test with timeout (should handle gracefully)  
        try:
            response = await connection.session.get(
                "http://httpbin.org/delay/60",  # Long delay
                timeout=5  # Short timeout
            )
        except Exception as e:
            print(f"✅ Properly handled timeout: {type(e).__name__}")
        
        # Connection should still be usable for valid requests
        valid_response = await connection.session.get("http://httpbin.org/get")
        assert valid_response.status_code == 200
        
        print("✅ Proxy error handling and recovery working")
        
    except Exception as e:
        pytest.fail(f"Proxy error handling test failed: {e}")
    
    finally:
        await proxy_manager.shutdown()