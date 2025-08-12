"""
Live end-to-end workflow tests with real services.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional

from contexter.core.download_engine import AsyncDownloadEngine
from contexter.core.storage_manager import LocalStorageManager
from contexter.integration.context7_client import Context7Client
from contexter.integration.proxy_manager import BrightDataProxyManager
from contexter.models.download_models import DownloadRequest
from contexter.models.config_models import C7DocConfig


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_live_download_workflow_no_proxy(
    temp_storage_dir: Path,
    small_test_library: Dict[str, str]
):
    """Test complete download workflow without proxy (direct connection)."""
    
    # Setup components
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    context7_client = Context7Client()
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=None,  # No proxy
        context7_client=context7_client,
        max_concurrent=2,
        max_retries=1,
    )
    
    try:
        # Test Context7 connectivity first
        is_connected = await context7_client.test_connectivity()
        if not is_connected:
            pytest.skip("Context7 API not available")
        
        # Create download request
        request = DownloadRequest(
            library_id=small_test_library["library_id"],
            contexts=[
                "getting started guide",
                "basic usage examples"
            ],
            token_limit=1500,  # Reasonable limit
            retry_count=1,
        )
        
        print(f"🚀 Starting live download: {small_test_library['name']}")
        
        # Execute download
        summary = await download_engine.download_library(request)
        
        # Verify download results
        assert summary.library_id == small_test_library["library_id"]
        assert summary.successful_contexts > 0, "Should have at least one successful context"
        assert len(summary.chunks) > 0, "Should have downloaded some chunks"
        assert summary.total_tokens > 0, "Should have some tokens"
        
        print(f"✅ Download completed:")
        print(f"   Successful contexts: {summary.successful_contexts}/{summary.total_contexts_attempted}")
        print(f"   Total tokens: {summary.total_tokens}")
        print(f"   Success rate: {summary.success_rate:.1%}")
        
        # Store results
        storage_result = await storage_manager.store_documentation(
            summary.library_id, summary.chunks
        )
        
        assert storage_result.success is True
        assert storage_result.file_path.exists()
        assert storage_result.compressed_size > 0
        
        print(f"✅ Storage completed:")
        print(f"   File: {storage_result.file_path.name}")
        print(f"   Compressed size: {storage_result.compressed_size} bytes")
        print(f"   Compression ratio: {storage_result.compression_ratio:.1%}")
        
        # Test retrieval
        retrieved_data = await storage_manager.retrieve_documentation(
            summary.library_id, "latest"
        )
        
        assert retrieved_data is not None
        assert retrieved_data["library_id"] == summary.library_id
        assert len(retrieved_data["chunks"]) == len(summary.chunks)
        
        print("✅ Retrieval completed successfully")
        
        # Test integrity
        integrity_ok = await storage_manager.verify_integrity(
            summary.library_id, "latest"
        )
        assert integrity_ok is True
        
        print("✅ Integrity verification passed")
        
    except Exception as e:
        pytest.fail(f"Live download workflow failed: {e}")
    
    finally:
        await download_engine.shutdown()


@pytest.mark.live
@pytest.mark.proxy_required
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_live_download_workflow_with_proxy(
    temp_storage_dir: Path,
    small_test_library: Dict[str, str],
    brightdata_credentials: Optional[Dict[str, str]]
):
    """Test complete download workflow with BrightData proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials not available")
    
    # Setup components
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    context7_client = Context7Client()
    
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"],
        zone_name="zone_residential_1",
        dns_resolution="remote"
    )
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=proxy_manager,
        context7_client=context7_client,
        max_concurrent=2,
        max_retries=1,
    )
    
    try:
        # Initialize proxy pool
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Test connectivity
        proxy_connected = await proxy_manager.test_connectivity()
        context7_connected = await context7_client.test_connectivity()
        
        if not proxy_connected:
            pytest.skip("BrightData proxy not available")
        if not context7_connected:
            pytest.skip("Context7 API not available")
        
        # Create download request
        request = DownloadRequest(
            library_id=small_test_library["library_id"],
            contexts=[
                "installation and setup",
                "API documentation"
            ],
            token_limit=1500,
            retry_count=1,
        )
        
        print(f"🚀 Starting live download with proxy: {small_test_library['name']}")
        
        # Execute download
        summary = await download_engine.download_library(request)
        
        # Verify results
        assert summary.successful_contexts > 0
        assert len(summary.chunks) > 0
        assert summary.total_tokens > 0
        
        print(f"✅ Download with proxy completed:")
        print(f"   Successful contexts: {summary.successful_contexts}/{summary.total_contexts_attempted}")
        print(f"   Total tokens: {summary.total_tokens}")
        
        # Verify proxy was used
        proxy_used = any(chunk.proxy_id != "direct" for chunk in summary.chunks)
        if proxy_used:
            print("✅ Proxy was used for downloads")
        else:
            print("⚠️  Direct connection was used (proxy may have failed over)")
        
        # Store and verify
        storage_result = await storage_manager.store_documentation(
            summary.library_id, summary.chunks
        )
        
        assert storage_result.success is True
        print("✅ Storage with proxy workflow completed")
        
    except Exception as e:
        pytest.fail(f"Live download workflow with proxy failed: {e}")
    
    finally:
        await download_engine.shutdown()


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_live_batch_download(
    temp_storage_dir: Path,
    test_libraries: list[Dict[str, str]],
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test downloading multiple libraries in batch with proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for batch downloads")
    
    # Use subset of libraries for testing
    test_subset = test_libraries[:2]  # Only test 2 libraries
    
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    context7_client = Context7Client()
    
    # Initialize proxy manager for batch downloads
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=proxy_manager,  # Use proxy for reliable batch downloads
        context7_client=context7_client,
        max_concurrent=2,
    )
    
    try:
        # Initialize proxy pool
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Test Context7 connectivity with proxy
        proxy_connection = await proxy_manager.get_connection()
        if not await context7_client.test_connectivity(proxy_connection.session if proxy_connection else None):
            pytest.skip("Context7 API not available")
        
        # Create download requests
        requests = []
        for lib in test_subset:
            request = DownloadRequest(
                library_id=lib["library_id"],
                contexts=["quick start", "examples"],  # Simple contexts
                token_limit=1000,  # Smaller limit for batch
                retry_count=1,
            )
            requests.append(request)
        
        print(f"🚀 Starting batch download of {len(requests)} libraries")
        
        # Execute batch download
        summaries = await download_engine.download_multiple_libraries(
            requests,
            max_concurrent_libraries=2
        )
        
        # Verify results
        assert len(summaries) == len(test_subset)
        
        successful_libs = 0
        total_tokens = 0
        
        for lib_id, summary in summaries.items():
            if summary.successful_contexts > 0:
                successful_libs += 1
                total_tokens += summary.total_tokens
                
                print(f"✅ {lib_id}: {summary.successful_contexts} contexts, {summary.total_tokens} tokens")
                
                # Store each library
                storage_result = await storage_manager.store_documentation(
                    lib_id, summary.chunks
                )
                assert storage_result.success
            else:
                print(f"❌ {lib_id}: No successful contexts")
        
        assert successful_libs > 0, "At least one library should download successfully"
        assert total_tokens > 0, "Should have downloaded some tokens"
        
        print(f"✅ Batch download completed:")
        print(f"   Successful libraries: {successful_libs}/{len(test_subset)}")
        print(f"   Total tokens: {total_tokens}")
        
    except Exception as e:
        pytest.fail(f"Live batch download failed: {e}")
    
    finally:
        await download_engine.shutdown()
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.asyncio
async def test_live_error_recovery(
    temp_storage_dir: Path,
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test error recovery with real API calls using proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for error recovery testing")
    
    context7_client = Context7Client()
    
    # Initialize proxy manager for error recovery testing
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=proxy_manager,  # Use proxy for reliable error recovery
        context7_client=context7_client,
        max_retries=2,  # Allow retries
    )
    
    try:
        # Initialize proxy pool
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Test Context7 API availability with proxy
        proxy_connection = await proxy_manager.get_connection()
        if not await context7_client.test_connectivity(proxy_connection.session if proxy_connection else None):
            pytest.skip("Context7 API not available")
        
        # Test with mix of valid and potentially problematic requests
        request = DownloadRequest(
            library_id="facebook/react",  # Known good library
            contexts=[
                "getting started",  # Should work
                "extremely specific advanced edge case documentation that probably doesn't exist",  # May fail
                "basic usage"  # Should work
            ],
            token_limit=1000,
            retry_count=2,
        )
        
        print("🧪 Testing error recovery with mixed contexts")
        
        summary = await download_engine.download_library(request)
        
        # Should have some success even if some contexts fail
        assert summary.total_contexts_attempted == 3
        
        print(f"📊 Error recovery results:")
        print(f"   Successful: {summary.successful_contexts}")
        print(f"   Failed: {summary.failed_contexts}")
        print(f"   Success rate: {summary.success_rate:.1%}")
        
        # Should have at least some success
        assert summary.successful_contexts > 0, "Should recover from errors and have some success"
        
        if summary.failed_contexts > 0:
            print("✅ Error recovery handled failures gracefully")
        else:
            print("✅ All contexts succeeded (no errors to recover from)")
        
    except Exception as e:
        pytest.fail(f"Live error recovery test failed: {e}")
    
    finally:
        await download_engine.shutdown()
        await proxy_manager.shutdown()


@pytest.mark.live
@pytest.mark.context7_required
@pytest.mark.proxy_required
@pytest.mark.asyncio
async def test_live_performance_realistic_load(
    temp_storage_dir: Path,
    brightdata_credentials: Optional[dict[str, str]]
):
    """Test performance under realistic load conditions with proxy."""
    
    if not brightdata_credentials:
        pytest.skip("BrightData credentials required for performance testing")
    
    context7_client = Context7Client()
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    
    # Initialize proxy manager for performance testing
    proxy_manager = BrightDataProxyManager(
        customer_id=brightdata_credentials["customer_id"],
        password=brightdata_credentials["password"]
    )
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=proxy_manager,  # Use proxy for realistic performance
        context7_client=context7_client,
        max_concurrent=3,  # Realistic concurrency
    )
    
    try:
        # Initialize proxy pool
        await proxy_manager.initialize_pool(pool_size=10)
        
        # Test Context7 API availability with proxy
        proxy_connection = await proxy_manager.get_connection()
        if not await context7_client.test_connectivity(proxy_connection.session if proxy_connection else None):
            pytest.skip("Context7 API not available")
        
        # Realistic workload: medium-sized library with multiple contexts
        request = DownloadRequest(
            library_id="facebook/react",
            contexts=[
                "installation and setup",
                "basic usage guide", 
                "API reference",
                "examples and tutorials",
                "configuration options"
            ],
            token_limit=2000,
            retry_count=1,
        )
        
        print("⚡ Testing realistic performance load")
        
        import time
        start_time = time.time()
        
        summary = await download_engine.download_library(request)
        
        download_time = time.time() - start_time
        
        # Performance assertions
        assert summary.successful_contexts > 0
        assert download_time < 120, "Should complete within 2 minutes"
        
        # Calculate throughput
        if summary.successful_contexts > 0:
            contexts_per_second = summary.successful_contexts / download_time
            tokens_per_second = summary.total_tokens / download_time
            
            print(f"📊 Performance metrics:")
            print(f"   Total time: {download_time:.2f}s")
            print(f"   Contexts/second: {contexts_per_second:.2f}")
            print(f"   Tokens/second: {tokens_per_second:.1f}")
            print(f"   Success rate: {summary.success_rate:.1%}")
            
            # Performance targets
            assert contexts_per_second > 0.05, "Should process at least 0.05 contexts per second"
            assert tokens_per_second > 10, "Should process at least 10 tokens per second"
        
        # Test storage performance
        storage_start = time.time()
        
        storage_result = await storage_manager.store_documentation(
            summary.library_id, summary.chunks
        )
        
        storage_time = time.time() - storage_start
        
        assert storage_result.success
        assert storage_time < 10, "Storage should complete within 10 seconds"
        
        chunks_per_second = len(summary.chunks) / storage_time
        print(f"   Storage chunks/second: {chunks_per_second:.1f}")
        
        print("✅ Realistic performance load test completed")
        
    except Exception as e:
        pytest.fail(f"Live performance test failed: {e}")
    
    finally:
        await download_engine.shutdown()
        await proxy_manager.shutdown()