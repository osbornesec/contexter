"""
Integration tests for Vector Database system.

Tests end-to-end functionality, performance characteristics,
and integration between all vector components.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
from typing import List

from contexter.vector import (
    QdrantVectorStore, VectorStoreConfig, VectorDocument,
    VectorSearchEngine, SearchQuery, BatchUploader, BatchConfig,
    VectorHealthMonitor, HealthConfig
)


@pytest.mark.asyncio
class TestVectorIntegration:
    """Integration tests for the complete vector database system."""
    
    @pytest_asyncio.fixture
    async def mock_vector_store(self):
        """Create a mock vector store for integration testing."""
        from unittest.mock import Mock, AsyncMock
        
        store = Mock(spec=QdrantVectorStore)
        store.config = VectorStoreConfig(vector_size=128, batch_size=10)
        store._initialized = True
        
        # Mock successful operations
        store.initialize = AsyncMock()
        store.cleanup = AsyncMock()
        store.upsert_vector = AsyncMock(return_value=True)
        async def mock_batch_upload(vectors, **kwargs):
            return {
                "successful_uploads": len(vectors),
                "failed_uploads": 0,
                "total_time": 0.1
            }
        
        store.upsert_vectors_batch = AsyncMock(side_effect=mock_batch_upload)
        
        # Mock search with realistic results
        async def mock_search(query_vector, top_k=10, filters=None, score_threshold=None):
            from contexter.vector.qdrant_vector_store import SearchResult
            results = []
            for i in range(min(top_k, 5)):
                result = SearchResult(
                    id=f"doc_{i}",
                    score=0.95 - (i * 0.05),
                    payload={
                        "content": f"Test document {i}",
                        "doc_type": "api" if i % 2 == 0 else "guide",
                        "library_id": f"lib_{i % 3}",
                        "trust_score": 8.0,
                        "programming_language": "python"
                    }
                )
                results.append(result)
            return results
            
        store.search_vectors = AsyncMock(side_effect=mock_search)
        store.get_vector = AsyncMock(return_value=None)
        store.count_vectors = AsyncMock(return_value=1000)
        store.get_collection_stats = AsyncMock(return_value={
            "collection_name": "test",
            "points_count": 1000,
            "vectors_count": 1000,
            "indexed_vectors_count": 1000,
            "status": "green"
        })
        store.get_health_status = Mock(return_value={
            "healthy": True,
            "initialized": True,
            "metrics": {},
            "config": {}
        })
        
        return store
        
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        documents = []
        for i in range(20):
            doc = VectorDocument(
                id=f"test_doc_{i}",
                vector=np.random.random(128).tolist(),
                payload={
                    "library_id": f"test_lib_{i % 3}",
                    "doc_type": "api" if i % 2 == 0 else "guide",
                    "content": f"This is test document {i} with some content",
                    "trust_score": 7.0 + (i % 3),
                    "programming_language": "python",
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            )
            documents.append(doc)
        return documents
        
    async def test_end_to_end_document_workflow(self, mock_vector_store, sample_documents):
        """Test complete document ingestion and search workflow."""
        # 1. Initialize search engine
        search_engine = VectorSearchEngine(
            vector_store=mock_vector_store,
            enable_caching=True
        )
        
        # 2. Initialize batch uploader
        batch_config = BatchConfig(batch_size=5, max_concurrent_batches=2)
        batch_uploader = BatchUploader(
            vector_store=mock_vector_store,
            config=batch_config
        )
        
        # 3. Upload documents in batches
        upload_result = await batch_uploader.upload_documents(sample_documents[:10])
        
        assert upload_result.status.value == "completed"
        assert upload_result.successful_items == 10
        assert upload_result.failed_items == 0
        
        # 4. Perform search operations
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            top_k=5,
            filters={"doc_type": "api"}
        )
        
        search_results = await search_engine.search(query)
        
        assert len(search_results) > 0
        assert all(r.score > 0 for r in search_results)
        assert all(r.rank > 0 for r in search_results)
        
        # 5. Test similar document search
        similar_docs = await search_engine.search_similar_documents(
            document_id="test_doc_1",
            top_k=3
        )
        
        # Should return empty for non-existent doc (mocked)
        assert len(similar_docs) == 0
        
        # 6. Test metadata search
        metadata_results = await search_engine.search_by_metadata(
            filters={"programming_language": "python"},
            top_k=10
        )
        
        assert isinstance(metadata_results, list)
        
    async def test_batch_processing_performance(self, mock_vector_store, sample_documents):
        """Test batch processing performance characteristics."""
        batch_uploader = BatchUploader(
            vector_store=mock_vector_store,
            config=BatchConfig(batch_size=5)
        )
        
        # Measure batch upload performance
        start_time = time.time()
        result = await batch_uploader.upload_documents(sample_documents)
        processing_time = time.time() - start_time
        
        # Should complete quickly with mocks
        assert processing_time < 1.0
        assert result.total_time < 1.0
        assert result.metadata["throughput_docs_per_sec"] > 0
        
        # Check metrics
        metrics = batch_uploader.get_metrics()
        assert metrics["upload_metrics"]["total_uploads"] == 1
        assert metrics["upload_metrics"]["total_documents"] == len(sample_documents)
        assert metrics["success_rate"] == 1.0
        
    async def test_search_caching_performance(self, mock_vector_store):
        """Test search caching performance."""
        search_engine = VectorSearchEngine(
            vector_store=mock_vector_store,
            enable_caching=True,
            cache_size=100,
            cache_ttl=300
        )
        
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api"}
        )
        
        # First search (cache miss)
        start_time = time.time()
        results1 = await search_engine.search(query)
        first_search_time = time.time() - start_time
        
        # Second search (cache hit)
        start_time = time.time()
        results2 = await search_engine.search(query)
        second_search_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_search_time < first_search_time
        assert len(results1) == len(results2)
        
        # Check cache metrics
        metrics = search_engine.get_search_metrics()
        assert metrics["search_metrics"]["total_searches"] == 2
        assert metrics["search_metrics"]["cache_hits"] == 1
        assert metrics["search_metrics"]["cache_misses"] == 1
        assert metrics["cache_hit_rate"] == 0.5
        
    async def test_health_monitoring_integration(self, mock_vector_store):
        """Test health monitoring system integration."""
        health_config = HealthConfig(
            check_interval_seconds=1,
            enable_performance_monitoring=True,
            enable_alerting=False  # Disable for testing
        )
        
        health_monitor = VectorHealthMonitor(
            vector_store=mock_vector_store,
            config=health_config
        )
        
        # Perform manual health check
        health_report = await health_monitor.perform_health_check()
        
        assert health_report.overall_status.value in ["healthy", "warning", "critical"]
        assert len(health_report.checks) > 0
        assert health_report.summary["total_checks"] > 0
        
        # Check individual health components
        check_names = [check.name for check in health_report.checks]
        expected_checks = [
            "connection_health",
            "collection_health", 
            "performance_health",
            "search_functionality",
            "resource_utilization"
        ]
        
        for expected_check in expected_checks:
            assert expected_check in check_names
            
        # Get health metrics
        health_metrics = health_monitor.get_health_metrics()
        assert "monitoring_status" in health_metrics
        assert "performance_metrics" in health_metrics
        
    async def test_concurrent_operations(self, mock_vector_store, sample_documents):
        """Test concurrent operations across all components."""
        # Initialize components
        search_engine = VectorSearchEngine(mock_vector_store, enable_caching=True)
        batch_uploader = BatchUploader(mock_vector_store)
        health_monitor = VectorHealthMonitor(mock_vector_store)
        
        # Create concurrent tasks
        tasks = []
        
        # Batch upload task
        upload_task = batch_uploader.upload_documents(sample_documents[:5])
        tasks.append(upload_task)
        
        # Multiple search tasks
        for i in range(3):
            query = SearchQuery(
                vector=np.random.random(128).tolist(),
                filters={"doc_type": "api" if i % 2 == 0 else "guide"}
            )
            search_task = search_engine.search(query)
            tasks.append(search_task)
            
        # Health check task
        health_task = health_monitor.perform_health_check()
        tasks.append(health_task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Validate results
        assert len(results) == 5
        
        # Upload result
        upload_result = results[0]
        assert upload_result.successful_items == 5
        
        # Search results
        for i in range(1, 4):
            search_results = results[i]
            assert isinstance(search_results, list)
            assert len(search_results) > 0
            
        # Health check result
        health_result = results[4]
        assert health_result.overall_status.value in ["healthy", "warning", "critical"]
        
    async def test_error_handling_and_recovery(self, mock_vector_store):
        """Test error handling and recovery mechanisms."""
        from unittest.mock import AsyncMock
        
        # Test search engine error handling
        search_engine = VectorSearchEngine(mock_vector_store)
        
        # Mock a search failure
        mock_vector_store.search_vectors = AsyncMock(
            side_effect=Exception("Simulated search failure")
        )
        
        query = SearchQuery(vector=np.random.random(128).tolist())
        
        with pytest.raises(Exception, match="Simulated search failure"):
            await search_engine.search(query)
            
        # Test batch uploader error handling
        batch_uploader = BatchUploader(mock_vector_store)
        
        # Mock batch upload failure
        mock_vector_store.upsert_vectors_batch = AsyncMock(return_value={
            "successful_uploads": 0,
            "failed_uploads": 5,
            "total_time": 0.1
        })
        
        sample_docs = [
            VectorDocument(
                id=f"error_doc_{i}",
                vector=np.random.random(128).tolist(),
                payload={}
            ) for i in range(5)
        ]
        
        result = await batch_uploader.upload_documents(sample_docs)
        
        assert result.status.value == "failed"
        assert result.failed_items == 5
        assert result.successful_items == 0
        
    async def test_memory_efficiency_large_batch(self, mock_vector_store):
        """Test memory efficiency with large document batches."""
        batch_uploader = BatchUploader(
            mock_vector_store,
            config=BatchConfig(batch_size=100, max_concurrent_batches=2)
        )
        
        # Create a large batch of documents
        large_batch = []
        for i in range(500):
            doc = VectorDocument(
                id=f"large_doc_{i}",
                vector=np.random.random(128).tolist(),
                payload={"index": i, "content": f"Document {i}"}
            )
            large_batch.append(doc)
            
        # Process large batch
        result = await batch_uploader.upload_documents(large_batch)
        
        assert result.status.value == "completed"
        assert result.total_items == 500
        assert result.successful_items == 500
        
        # Check that batching was used
        assert mock_vector_store.upsert_vectors_batch.call_count >= 5  # 500/100 = 5 batches
        
    async def test_search_ranking_and_relevance(self, mock_vector_store):
        """Test search ranking and relevance scoring."""
        search_engine = VectorSearchEngine(mock_vector_store)
        
        # Test search with filters that should affect ranking
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            top_k=5,
            filters={"doc_type": "api", "programming_language": "python"}
        )
        
        results = await search_engine.search(query)
        
        # Check ranking
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))
        
        # Check relevance scores
        for result in results:
            assert result.relevance_score >= result.score  # Should be boosted
            assert len(result.match_reasons) > 0
            
        # Check match reasons include expected types
        all_reasons = []
        for result in results:
            all_reasons.extend(result.match_reasons)
            
        reason_text = " ".join(all_reasons).lower()
        assert "similarity" in reason_text
        
    async def test_cache_warm_up_and_optimization(self, mock_vector_store):
        """Test cache warming and search optimization."""
        search_engine = VectorSearchEngine(mock_vector_store, enable_caching=True)
        
        # Create common queries for warming
        common_queries = []
        for i in range(3):
            query = SearchQuery(
                vector=np.random.random(128).tolist(),
                filters={"doc_type": "api" if i % 2 == 0 else "guide"}
            )
            common_queries.append(query)
            
        # Warm cache
        await search_engine.warm_cache(common_queries)
        
        # All queries should now hit cache
        for query in common_queries:
            results = await search_engine.search(query)
            assert len(results) > 0
            
        # Check cache hit rate
        metrics = search_engine.get_search_metrics()
        assert metrics["cache_hit_rate"] > 0.5  # Should have good hit rate
        
        # Test search parameter optimization
        optimized_params = search_engine.optimize_search_parameters(target_latency_ms=25.0)
        assert "ef" in optimized_params
        assert optimized_params["target_latency_ms"] == 25.0


@pytest.mark.asyncio
class TestVectorPerformance:
    """Performance-focused integration tests."""
    
    @pytest_asyncio.fixture
    async def performance_vector_store(self):
        """Create a mock vector store optimized for performance testing."""
        from unittest.mock import Mock, AsyncMock
        
        store = Mock(spec=QdrantVectorStore)
        store.config = VectorStoreConfig(vector_size=2048, batch_size=1000)  # Full size
        
        # Mock with performance characteristics
        async def fast_search(*args, **kwargs):
            # Simulate fast search
            await asyncio.sleep(0.001)  # 1ms
            return []
            
        async def fast_batch_upload(vectors, **kwargs):
            # Simulate fast batch upload
            await asyncio.sleep(0.01)  # 10ms
            return {
                "successful_uploads": len(vectors),
                "failed_uploads": 0,
                "total_time": 0.01
            }
            
        store.search_vectors = AsyncMock(side_effect=fast_search)
        store.upsert_vectors_batch = AsyncMock(side_effect=fast_batch_upload)
        
        return store
        
    async def test_search_latency_performance(self, performance_vector_store):
        """Test search latency meets performance requirements."""
        search_engine = VectorSearchEngine(performance_vector_store)
        
        # Test multiple searches and measure latency
        latencies = []
        
        for _ in range(10):
            query = SearchQuery(vector=np.random.random(2048).tolist())
            
            start_time = time.time()
            await search_engine.search(query)
            latency = time.time() - start_time
            
            latencies.append(latency * 1000)  # Convert to ms
            
        # Calculate performance metrics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        
        # Performance requirements from PRP
        assert avg_latency < 50.0, f"Average latency {avg_latency:.1f}ms exceeds 50ms requirement"
        assert p95_latency < 50.0, f"P95 latency {p95_latency:.1f}ms exceeds 50ms requirement"
        
    async def test_batch_upload_throughput(self, performance_vector_store):
        """Test batch upload throughput meets requirements."""
        batch_uploader = BatchUploader(
            performance_vector_store,
            config=BatchConfig(batch_size=1000, max_concurrent_batches=5)
        )
        
        # Create large batch
        documents = []
        for i in range(5000):  # 5k documents
            doc = VectorDocument(
                id=f"perf_doc_{i}",
                vector=np.random.random(2048).tolist(),
                payload={"index": i}
            )
            documents.append(doc)
            
        # Measure throughput
        start_time = time.time()
        result = await batch_uploader.upload_documents(documents)
        total_time = time.time() - start_time
        
        throughput = len(documents) / total_time
        
        # Should achieve good throughput with mocks
        assert throughput > 1000, f"Throughput {throughput:.0f} docs/sec too low"
        assert result.successful_items == len(documents)
        
    async def test_concurrent_search_performance(self, performance_vector_store):
        """Test concurrent search performance."""
        search_engine = VectorSearchEngine(performance_vector_store)
        
        # Create multiple concurrent searches
        tasks = []
        for i in range(100):  # 100 concurrent searches
            query = SearchQuery(
                vector=np.random.random(2048).tolist(),
                top_k=10
            )
            task = search_engine.search(query)
            tasks.append(task)
            
        # Execute all searches concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All searches should complete successfully
        assert len(results) == 100
        
        # Should handle 100+ concurrent queries efficiently
        queries_per_second = 100 / total_time
        assert queries_per_second > 50, f"Concurrent QPS {queries_per_second:.0f} too low"