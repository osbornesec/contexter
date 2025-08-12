"""
End-to-end tests for complete download workflows.
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, AsyncMock

from contexter.core.download_engine import AsyncDownloadEngine
from contexter.core.storage_manager import LocalStorageManager
from contexter.models.download_models import DownloadRequest, DocumentationChunk
from contexter.models.config_models import C7DocConfig


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_library_download_workflow(
    test_config: C7DocConfig,
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
    sample_documentation_chunks: list[DocumentationChunk],
):
    """Test complete end-to-end library download workflow."""
    
    # Setup storage manager
    storage_manager = LocalStorageManager(
        base_path=str(temp_storage_dir),
        compression_level=6,
        retention_limit=3,
    )
    
    # Setup download engine with mocked dependencies
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_concurrent=3,
        max_retries=2,
    )
    
    # Create download request
    request = DownloadRequest(
        library_id="test/example-lib",
        contexts=[
            "getting started guide",
            "API reference documentation", 
            "configuration examples"
        ],
        token_limit=5000,
        retry_count=2,
    )
    
    try:
        # Execute download
        summary = await download_engine.download_library(request)
        
        # Verify download results
        assert summary.library_id == "test/example-lib"
        assert summary.successful_contexts == 3
        assert summary.failed_contexts == 0
        assert len(summary.chunks) == 3
        assert summary.total_tokens > 0
        assert summary.success_rate == 1.0
        
        # Store downloaded chunks
        storage_result = await storage_manager.store_documentation(
            summary.library_id, summary.chunks
        )
        
        # Verify storage results
        assert storage_result.success is True
        assert storage_result.compressed_size > 0
        assert storage_result.compression_ratio > 0.0
        assert storage_result.checksum
        assert storage_result.version_id
        
        # Verify file was created
        assert storage_result.file_path.exists()
        
        # Test retrieval
        retrieved_data = await storage_manager.retrieve_documentation(
            summary.library_id, "latest"
        )
        
        assert retrieved_data is not None
        assert retrieved_data["library_id"] == summary.library_id
        assert len(retrieved_data["chunks"]) == 3
        
        # Test integrity verification
        integrity_ok = await storage_manager.verify_integrity(
            summary.library_id, "latest"
        )
        assert integrity_ok is True
        
    finally:
        # Cleanup
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_library_batch_download(
    test_config: C7DocConfig,
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
    library_test_cases: Dict[str, Dict[str, Any]],
):
    """Test downloading multiple libraries in batch."""
    
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_concurrent=2,
    )
    
    # Create multiple download requests
    requests = []
    for lib_name, test_case in library_test_cases.items():
        request = DownloadRequest(
            library_id=test_case["library_id"],
            contexts=test_case["contexts"],
            token_limit=2000,
        )
        requests.append(request)
    
    try:
        # Execute batch download
        summaries = await download_engine.download_multiple_libraries(
            requests, max_concurrent_libraries=2
        )
        
        # Verify all libraries were processed
        assert len(summaries) == len(library_test_cases)
        
        # Store all results and verify
        for lib_name, summary in summaries.items():
            assert summary.successful_contexts > 0
            assert len(summary.chunks) > 0
            
            # Store each library's documentation
            storage_result = await storage_manager.store_documentation(
                lib_name, summary.chunks
            )
            
            assert storage_result.success is True
            assert storage_result.file_path.exists()
    
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_download_with_failures_and_retries(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test download workflow with simulated failures and retries."""
    
    # Configure mock to fail first two attempts, succeed on third
    call_count = 0
    original_get_docs = mock_context7_client.get_smart_docs
    
    async def failing_get_docs(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count <= 2:
            raise Exception("Simulated network error")
        else:
            return await original_get_docs(*args, **kwargs)
    
    mock_context7_client.get_smart_docs.side_effect = failing_get_docs
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_retries=3,  # Allow enough retries
    )
    
    request = DownloadRequest(
        library_id="test/flaky-lib",
        contexts=["documentation"],
        retry_count=3,
    )
    
    try:
        # This should succeed on the third attempt
        summary = await download_engine.download_library(request)
        
        # Verify it eventually succeeded
        assert summary.successful_contexts == 1
        assert len(summary.chunks) == 1
        
        # Verify retries were attempted
        assert call_count == 3
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e 
@pytest.mark.asyncio
async def test_storage_versioning_and_cleanup(
    temp_storage_dir: Path,
    sample_documentation_chunks: list[DocumentationChunk],
):
    """Test storage versioning and cleanup functionality."""
    
    storage_manager = LocalStorageManager(
        base_path=str(temp_storage_dir),
        retention_limit=2,  # Keep only 2 versions
    )
    
    library_id = "test/versioned-lib"
    
    # Store multiple versions
    versions = []
    for i in range(4):  # Store 4 versions (more than retention limit)
        # Modify chunks slightly for each version
        modified_chunks = []
        for chunk in sample_documentation_chunks:
            modified_chunk = DocumentationChunk(
                chunk_id=f"{chunk.chunk_id}_v{i}",
                content=f"{chunk.content}\nVersion {i} update",
                source_context=chunk.source_context,
                token_count=chunk.token_count + 5,
                content_hash=f"{chunk.content_hash}_v{i}",
                proxy_id=chunk.proxy_id,
                download_time=chunk.download_time,
                library_id=library_id,
                metadata={**chunk.metadata, "version": i}
            )
            modified_chunks.append(modified_chunk)
        
        result = await storage_manager.store_documentation(library_id, modified_chunks)
        assert result.success
        versions.append(result.version_id)
        
        # Small delay to ensure different timestamps
        await asyncio.sleep(0.1)
    
    # Verify only retention_limit versions remain
    stored_versions = await storage_manager.list_versions(library_id)
    assert len(stored_versions) == 2  # retention_limit
    
    # Verify the latest versions are kept
    version_ids = [v.version_id for v in stored_versions]
    assert versions[-1] in version_ids  # Most recent
    assert versions[-2] in version_ids  # Second most recent
    assert versions[0] not in version_ids  # Oldest should be cleaned up
    assert versions[1] not in version_ids  # Second oldest should be cleaned up
    
    # Verify latest version is accessible
    latest_data = await storage_manager.retrieve_documentation(library_id, "latest")
    assert latest_data is not None
    assert "Version 3" in latest_data["chunks"][0]["content"]  # Latest version content


@pytest.mark.e2e
@pytest.mark.asyncio 
async def test_deduplication_workflow(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test end-to-end workflow with deduplication."""
    from contexter.core.deduplication import DeduplicationEngine
    
    # Setup components
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    dedup_engine = DeduplicationEngine()
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
    )
    
    # Mock responses with some duplicate content
    responses = [
        {"content": "# Getting Started\nThis is the getting started guide.", "tokens": 20},
        {"content": "# Getting Started\nThis is the getting started guide.", "tokens": 20},  # Exact duplicate
        {"content": "# API Reference\nDetailed API documentation here.", "tokens": 25},
        {"content": "# Getting Started\nThis is a getting started tutorial.", "tokens": 22},  # Similar content
    ]
    
    mock_context7_client.get_smart_docs.side_effect = [
        type('MockResponse', (), {
            'content': resp['content'],
            'token_count': resp['tokens'],
            'response_time': 0.5,
            'metadata': {}
        })() for resp in responses
    ]
    
    request = DownloadRequest(
        library_id="test/dedup-lib", 
        contexts=[
            "getting started guide",
            "getting started tutorial", 
            "API reference documentation",
            "getting started overview"
        ],
    )
    
    try:
        # Download
        summary = await download_engine.download_library(request)
        assert len(summary.chunks) == 4  # All chunks downloaded
        
        # Apply deduplication
        unique_chunks, dedup_stats = await dedup_engine.deduplicate_chunks(
            summary.chunks
        )
        
        # Verify deduplication worked
        assert len(unique_chunks) < len(summary.chunks)
        assert dedup_stats.duplicates_removed > 0
        assert dedup_stats.unique_content_ratio < 1.0
        
        # Store deduplicated results
        storage_result = await storage_manager.store_documentation(
            summary.library_id, unique_chunks
        )
        
        assert storage_result.success
        
        # Verify stored data contains only unique content
        retrieved = await storage_manager.retrieve_documentation(
            summary.library_id, "latest"
        )
        
        stored_chunks = retrieved["chunks"]
        assert len(stored_chunks) == len(unique_chunks)
        
        # Verify no exact duplicates in stored content
        content_hashes = [chunk["content_hash"] for chunk in stored_chunks]
        assert len(content_hashes) == len(set(content_hashes))
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_system_health_and_monitoring(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test system health monitoring and diagnostics."""
    
    storage_manager = LocalStorageManager(str(temp_storage_dir))
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
    )
    
    try:
        # Test health check
        health_status = await download_engine.health_check()
        
        assert health_status["engine_status"] in ["healthy", "degraded"]
        assert "components" in health_status
        assert "metrics" in health_status
        assert "timestamp" in health_status
        
        # Verify component health
        components = health_status["components"]
        assert "context7_client" in components
        assert "proxy_manager" in components
        assert "concurrent_processor" in components
        
        # Test storage statistics
        stats = await storage_manager.get_storage_statistics()
        
        assert hasattr(stats, 'total_libraries')
        assert hasattr(stats, 'total_versions') 
        assert hasattr(stats, 'total_size_bytes')
        
        # Test with some data
        from contexter.models.download_models import DocumentationChunk
        
        test_chunk = DocumentationChunk(
            chunk_id="health-test",
            content="Health check test content", 
            source_context="health check",
            token_count=10,
            content_hash="health123",
            proxy_id="test-proxy",
            download_time=0.1,
            library_id="test/health-lib",
            metadata={"test": True}
        )
        
        await storage_manager.store_documentation("test/health-lib", [test_chunk])
        
        # Verify updated statistics
        updated_stats = await storage_manager.get_storage_statistics()
        assert updated_stats.total_libraries >= 1
        assert updated_stats.total_size_bytes > 0
        
    finally:
        await download_engine.shutdown()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_recovery_and_resilience(
    temp_storage_dir: Path,
    mock_proxy_manager,
    mock_context7_client,
):
    """Test system resilience under various error conditions."""
    
    download_engine = AsyncDownloadEngine(
        proxy_manager=mock_proxy_manager,
        context7_client=mock_context7_client,
        max_retries=2,
    )
    
    # Test 1: Proxy failure recovery
    mock_proxy_manager.get_connection.side_effect = [
        Exception("Proxy failure"),  # First attempt fails
        mock_proxy_manager.get_connection.return_value  # Second succeeds
    ]
    
    request = DownloadRequest(
        library_id="test/resilient-lib",
        contexts=["documentation"],
    )
    
    try:
        # Should recover from proxy failure
        summary = await download_engine.download_library(request)
        assert summary.successful_contexts == 1
        
        # Test 2: Partial failure tolerance
        mock_context7_client.get_smart_docs.side_effect = [
            Exception("Network timeout"),  # First context fails
            mock_context7_client.get_smart_docs.return_value,  # Second succeeds
        ]
        
        multi_context_request = DownloadRequest(
            library_id="test/partial-fail-lib",
            contexts=["failing context", "working context"],
        )
        
        summary = await download_engine.download_library(multi_context_request)
        
        # Should have 1 success and 1 failure
        assert summary.successful_contexts == 1
        assert summary.failed_contexts == 1
        assert summary.success_rate == 0.5
        
        # Verify error tracking
        assert len(summary.error_summary) > 0
        
    finally:
        await download_engine.shutdown()