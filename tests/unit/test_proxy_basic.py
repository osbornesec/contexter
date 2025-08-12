"""
Basic tests for BrightData proxy manager - core functionality only.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, patch
import httpx

from src.contexter.models.proxy_models import ProxyConnection, ProxyStatus, CircuitBreaker
from src.contexter.integration.proxy_manager import BrightDataProxyManager, ProxyManagerError


class TestProxyManagerBasics:
    """Test core proxy manager functionality."""
    
    @pytest.mark.asyncio
    async def test_proxy_manager_full_workflow(self):
        """Test complete proxy manager workflow."""
        
        # Create proxy manager
        with patch.dict('os.environ', {
            'BRIGHTDATA_CUSTOMER_ID': 'test_customer_123',
            'BRIGHTDATA_PASSWORD': 'test_password_456'
        }):
            manager = await BrightDataProxyManager.from_environment("test_zone")
            
            # Mock the HTTP client
            with patch('httpx.AsyncClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.aclose = AsyncMock()
                mock_client_class.return_value = mock_client
                
                try:
                    # Test pool initialization
                    await manager.initialize_pool(pool_size=10)
                    
                    assert len(manager.proxy_pool) == 10
                    assert len(manager.proxy_metrics) == 10
                    
                    # Test connection retrieval
                    connection = await manager.get_connection()
                    assert connection is not None
                    assert connection.status == ProxyStatus.HEALTHY
                    
                    # Test success reporting
                    await manager.report_success(connection, response_time=1.5)
                    assert connection.success_count == 1
                    
                    # Test failure reporting
                    error = httpx.TimeoutException("Test timeout")
                    await manager.report_failure(connection, error)
                    assert connection.failure_count == 1
                    
                    # Test pool status
                    status = manager.get_pool_status()
                    assert status["total_connections"] == 10
                    assert status["healthy"] >= 9  # Most should be healthy
                    
                    # Test metrics
                    metrics = manager.get_proxy_metrics()
                    assert connection.proxy_id in metrics
                    
                    print("✅ All proxy manager core functionality tests passed")
                    
                finally:
                    await manager.shutdown()
    
    def test_circuit_breaker_functionality(self):
        """Test circuit breaker patterns."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=30)
        
        # Test initial state
        assert cb.can_execute() == True
        
        # Test failure accumulation
        for i in range(3):
            cb.record_failure()
        
        # Should be open now
        assert cb.can_execute() == False
        
        # Test recovery with time mock
        with patch('time.time', return_value=cb.last_failure_time + 31):
            assert cb.can_execute() == True  # Half-open
            
            # Record successes to close
            for i in range(3):
                cb.record_success()
            
            assert cb.can_execute() == True
        
        print("✅ Circuit breaker functionality tests passed")
    
    def test_proxy_url_construction(self):
        """Test proxy URL construction."""
        manager = BrightDataProxyManager(
            customer_id="test123",
            password="pass456", 
            zone_name="zone_test"
        )
        
        url = manager._build_proxy_url("session_abc")
        expected = "http://brd-customer-test123-zone-zone_test-session-session_abc-dns-remote:pass456@brd.superproxy.io:33335"
        
        assert url == expected
        
        print("✅ Proxy URL construction test passed")
    
    def test_session_id_generation(self):
        """Test session ID generation."""
        manager = BrightDataProxyManager("test", "test", "test")
        
        id1 = manager._generate_session_id()
        id2 = manager._generate_session_id()
        
        assert id1 != id2
        assert id1.startswith("session_")
        assert id2.startswith("session_")
        
        print("✅ Session ID generation test passed")


if __name__ == "__main__":
    # Run tests directly for quick verification
    import asyncio
    
    test_instance = TestProxyManagerBasics()
    
    # Run sync tests
    test_instance.test_circuit_breaker_functionality()
    test_instance.test_proxy_url_construction()
    test_instance.test_session_id_generation()
    
    # Run async test
    asyncio.run(test_instance.test_proxy_manager_full_workflow())
    
    print("🎉 All basic proxy manager tests completed successfully!")