"""
End-to-end performance and load testing scenarios.
"""

import asyncio
import time
from pathlib import Path
from typing import List
import pytest
from unittest.mock import AsyncMock

from contexter.core.download_engine import AsyncDownloadEngine
from contexter.core.storage_manager import LocalStorageManager
from contexter.models.download_models import DownloadRequest, DocumentationChunk


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_download_performance(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test performance under high concurrency loads."""
    
    # Mock with realistic delays
    async def delayed_response(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate 100ms API response time
        return type('MockResponse', (), {
            'content': f"# Test Content\nGenerated at {time.time()}",
            'token_count': 50,
            'response_time': 0.1,
            'metadata': {'test': True}
        })()
    
    mock_context7_client.get_smart_docs.side_effect = delayed_response
    
    # Test with high concurrency
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_concurrent=15,  # High concurrency
    )
    
    # Create large download request
    contexts = [f"context_{i}" for i in range(50)]  # 50 contexts
    request = DownloadRequest(
        library_id="test/performance-lib",
        contexts=contexts,
        token_limit=2000,
    )
    
    start_time = time.time()
    
    try:
        summary = await download_engine.download_library(request)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance assertions
        assert summary.successful_contexts == 50
        assert total_time < 10.0  # Should complete in under 10 seconds with good concurrency
        
        # Calculate throughput
        contexts_per_second = len(contexts) / total_time
        assert contexts_per_second > 5.0  # Should process at least 5 contexts/second
        
        print(f"Performance metrics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Contexts/second: {contexts_per_second:.2f}")
        print(f"  Average time per context: {total_time/len(contexts):.3f}s")
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_usage_under_load(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test memory usage patterns under sustained load."""
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large content responses
    large_content = "# Large Documentation\n" + "Content line\n" * 1000  # ~13KB content
    
    mock_context7_client.get_smart_docs.return_value = type('MockResponse', (), {
        'content': large_content,
        'token_count': 500,
        'response_time': 0.2,
        'metadata': {}
    })()
    
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_concurrent=10,
    )
    
    try:
        # Process multiple libraries sequentially to test memory cleanup
        memory_readings = [initial_memory]
        
        for i in range(5):  # 5 iterations
            request = DownloadRequest(
                library_id=f"test/memory-lib-{i}",
                contexts=[f"context_{j}" for j in range(20)],  # 20 contexts each
            )
            
            summary = await download_engine.download_library(request)
            
            # Store results
            await storage_manager.store_documentation(
                summary.library_id, summary.chunks
            )
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_readings.append(current_memory)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Memory should not grow indefinitely
        final_memory = memory_readings[-1]
        memory_growth = final_memory - initial_memory
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Growth: {memory_growth:.1f} MB")
        
        # Should not use more than 100MB additional memory
        assert memory_growth < 100, f"Memory growth too high: {memory_growth:.1f} MB"
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_storage_performance_large_datasets(temp_storage_dir: Path):
    """Test storage performance with large datasets."""
    
    storage_manager = LocalStorageManager(
        str(temp_storage_dir),
        compression_level=6,  # Balanced compression
    )
    
    # Generate large dataset
    large_chunks = []
    for i in range(100):  # 100 chunks
        content = f"# Documentation Chunk {i}\n" + "Line of content\n" * 200  # ~2.5KB each
        
        chunk = DocumentationChunk(
            chunk_id=f"perf-chunk-{i}",
            content=content,
            source_context=f"performance context {i}",
            token_count=250,
            content_hash=f"hash{i}",
            proxy_id="perf-proxy",
            download_time=0.5,
            library_id="test/large-lib",
            metadata={"chunk_index": i}
        )
        large_chunks.append(chunk)
    
    # Test storage performance
    start_time = time.time()
    
    storage_result = await storage_manager.store_documentation(
        "test/large-lib", large_chunks
    )
    
    store_time = time.time() - start_time
    
    assert storage_result.success
    assert storage_result.compression_ratio > 0.3  # Should achieve reasonable compression
    
    print(f"Storage performance:")
    print(f"  Storage time: {store_time:.2f}s")
    print(f"  Chunks stored: {len(large_chunks)}")
    print(f"  Compression ratio: {storage_result.compression_ratio:.2%}")
    print(f"  Chunks/second: {len(large_chunks)/store_time:.1f}")
    
    # Test retrieval performance
    start_time = time.time()
    
    retrieved_data = await storage_manager.retrieve_documentation("test/large-lib", "latest")
    
    retrieve_time = time.time() - start_time
    
    assert retrieved_data is not None
    assert len(retrieved_data["chunks"]) == 100
    
    print(f"  Retrieval time: {retrieve_time:.2f}s")
    
    # Performance targets
    assert store_time < 5.0  # Storage should complete in under 5 seconds
    assert retrieve_time < 2.0  # Retrieval should be faster


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_error_recovery_performance(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test performance impact of error recovery mechanisms."""
    
    # Configure flaky responses (50% failure rate)
    call_count = 0
    
    async def flaky_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count % 2 == 0:  # Fail every other call
            raise Exception("Simulated intermittent failure")
        
        return type('MockResponse', (), {
            'content': "# Recovered Content\nThis succeeded after retry",
            'token_count': 40,
            'response_time': 0.3,
            'metadata': {'recovered': True}
        })()
    
    mock_context7_client.get_smart_docs.side_effect = flaky_response
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_retries=3,  # Allow retries
        max_concurrent=5,
    )
    
    request = DownloadRequest(
        library_id="test/flaky-lib",
        contexts=[f"flaky_context_{i}" for i in range(30)],
        retry_count=3,
    )
    
    start_time = time.time()
    
    try:
        summary = await download_engine.download_library(request)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should eventually succeed despite failures
        assert summary.successful_contexts > 0
        
        # Calculate retry overhead
        expected_calls_without_retries = len(request.contexts)
        actual_calls = call_count
        retry_overhead = (actual_calls - expected_calls_without_retries) / expected_calls_without_retries
        
        print(f"Error recovery performance:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Successful contexts: {summary.successful_contexts}/{len(request.contexts)}")
        print(f"  Failed contexts: {summary.failed_contexts}")
        print(f"  Success rate: {summary.success_rate:.1%}")
        print(f"  API calls made: {actual_calls}")
        print(f"  Retry overhead: {retry_overhead:.1%}")
        
        # Should not take excessively long despite retries
        assert total_time < 20.0  # Should complete within reasonable time
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_batch_download_scalability(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test scalability of batch download operations."""
    
    mock_context7_client.get_smart_docs.return_value = type('MockResponse', (), {
        'content': "# Batch Test Content\nStandard documentation content",
        'token_count': 30,
        'response_time': 0.15,
        'metadata': {}
    })()
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_concurrent=8,
    )
    
    # Create multiple library requests
    requests = []
    for i in range(10):  # 10 libraries
        request = DownloadRequest(
            library_id=f"test/batch-lib-{i}",
            contexts=[f"context_{j}" for j in range(5)],  # 5 contexts each
            token_limit=1000,
        )
        requests.append(request)
    
    start_time = time.time()
    
    try:
        summaries = await download_engine.download_multiple_libraries(
            requests,
            max_concurrent_libraries=3,  # Process 3 libraries concurrently
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all libraries processed
        assert len(summaries) == 10
        
        total_contexts = sum(summary.successful_contexts for summary in summaries.values())
        total_expected = sum(len(req.contexts) for req in requests)
        
        assert total_contexts == total_expected
        
        # Calculate batch efficiency
        contexts_per_second = total_contexts / total_time
        
        print(f"Batch download scalability:")
        print(f"  Libraries processed: {len(summaries)}")
        print(f"  Total contexts: {total_contexts}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Contexts/second: {contexts_per_second:.2f}")
        print(f"  Libraries/second: {len(summaries)/total_time:.2f}")
        
        # Should process efficiently
        assert contexts_per_second > 3.0  # At least 3 contexts per second
        assert total_time < 15.0  # Should complete within reasonable time
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_deduplication_performance_large_dataset(temp_storage_dir: Path):
    """Test deduplication performance with large datasets."""
    from contexter.core.deduplication import DeduplicationEngine
    
    dedup_engine = DeduplicationEngine()
    
    # Create large dataset with duplicates and similar content
    chunks = []
    
    # Exact duplicates (20%)
    for i in range(20):
        chunk = DocumentationChunk(
            chunk_id=f"duplicate-{i}",
            content="# Duplicate Content\nThis is exactly the same content",
            source_context="duplicate context",
            token_count=25,
            content_hash="duplicate_hash",
            proxy_id="test-proxy",
            download_time=0.5,
            library_id="test/dedup-perf",
            metadata={}
        )
        chunks.append(chunk)
    
    # Similar content (30%)
    similar_base = "# Similar Documentation\nThis is the base content for similar items"
    for i in range(30):
        content = f"{similar_base}\nUnique variation {i}"
        chunk = DocumentationChunk(
            chunk_id=f"similar-{i}",
            content=content,
            source_context=f"similar context {i}",
            token_count=35 + i,
            content_hash=f"similar_hash_{i}",
            proxy_id="test-proxy",
            download_time=0.5,
            library_id="test/dedup-perf",
            metadata={"variation": i}
        )
        chunks.append(chunk)
    
    # Unique content (50%)
    for i in range(50):
        chunk = DocumentationChunk(
            chunk_id=f"unique-{i}",
            content=f"# Unique Content {i}\nCompletely unique documentation content for item {i}\nWith unique details and information.",
            source_context=f"unique context {i}",
            token_count=45,
            content_hash=f"unique_hash_{i}",
            proxy_id="test-proxy",
            download_time=0.5,
            library_id="test/dedup-perf",
            metadata={"unique_id": i}
        )
        chunks.append(chunk)
    
    print(f"Deduplication performance test:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Exact duplicates: 20")
    print(f"  Similar content: 30") 
    print(f"  Unique content: 50")
    
    start_time = time.time()
    
    # Run deduplication
    unique_chunks, stats = await dedup_engine.deduplicate_chunks(chunks)
    
    dedup_time = time.time() - start_time
    
    print(f"  Processing time: {dedup_time:.2f}s")
    print(f"  Chunks/second: {len(chunks)/dedup_time:.1f}")
    print(f"  Duplicates removed: {stats.duplicates_removed}")
    print(f"  Final unique chunks: {len(unique_chunks)}")
    print(f"  Reduction: {(1 - len(unique_chunks)/len(chunks)):.1%}")
    
    # Performance and correctness assertions
    assert dedup_time < 10.0  # Should process 100 chunks in under 10 seconds
    assert stats.duplicates_removed >= 19  # Should find most exact duplicates
    assert len(unique_chunks) < len(chunks)  # Should reduce dataset size
    assert len(unique_chunks) >= 50  # Should keep all unique content