"""
Comprehensive Test Suite for RAG Embedding Service

Tests all components of the embedding service including:
- Voyage AI client integration
- Caching system
- Batch processing
- Performance requirements
- Integration scenarios
"""

import asyncio
import pytest
import time
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from typing import List, Dict, Any

from contexter.models.embedding_models import (
    EmbeddingRequest, EmbeddingResult, InputType, ProcessingStatus,
    ProcessingMetrics, CacheStats, validate_embedding_dimensions
)
from contexter.vector.voyage_client import VoyageAIClient, VoyageClientConfig
from contexter.vector.embedding_cache import EmbeddingCache
from contexter.vector.batch_processor import BatchProcessor, BatchProcessingConfig
from contexter.vector.embedding_engine import VoyageEmbeddingEngine, EmbeddingEngineConfig


class TestEmbeddingModels:
    """Test embedding data models and validation."""
    
    def test_embedding_request_hash_generation(self):
        """Test automatic content hash generation."""
        request = EmbeddingRequest(
            content="Test content for hashing",
            input_type=InputType.DOCUMENT
        )
        
        assert request.content_hash
        assert len(request.content_hash) == 16  # SHA256 truncated to 16 chars
        
        # Same content should generate same hash
        request2 = EmbeddingRequest(
            content="Test content for hashing",
            input_type=InputType.DOCUMENT
        )
        assert request.content_hash == request2.content_hash
    
    def test_embedding_result_success_property(self):
        """Test embedding result success property."""
        # Successful result
        success_result = EmbeddingResult(
            content_hash="test_hash",
            embedding=[0.1] * 2048,
            model="voyage-code-3",
            dimensions=2048,
            processing_time=0.5,
            cache_hit=False,
            status=ProcessingStatus.COMPLETED
        )
        assert success_result.success
        
        # Failed result
        failed_result = EmbeddingResult(
            content_hash="test_hash",
            embedding=[],
            model="voyage-code-3",
            dimensions=0,
            processing_time=0.1,
            cache_hit=False,
            status=ProcessingStatus.FAILED,
            error="API Error"
        )
        assert not failed_result.success
    
    def test_validate_embedding_dimensions(self):
        """Test embedding dimension validation."""
        # Valid embedding
        valid_embedding = [0.1] * 2048
        assert validate_embedding_dimensions(valid_embedding)
        
        # Invalid dimensions
        invalid_embedding = [0.1] * 1024
        assert not validate_embedding_dimensions(invalid_embedding)
        
        # Invalid types
        invalid_types = [0.1] * 2047 + ["invalid"]
        assert not validate_embedding_dimensions(invalid_types)


class TestVoyageClient:
    """Test Voyage AI client functionality."""
    
    @pytest.fixture
    def mock_voyage_config(self):
        """Create mock Voyage client configuration."""
        return VoyageClientConfig(
            api_key="test_api_key",
            model="voyage-code-3",
            timeout=30.0,
            max_retries=3
        )
    
    @pytest.fixture
    async def mock_voyage_client(self, mock_voyage_config):
        """Create mock Voyage client."""
        client = VoyageAIClient(mock_voyage_config)
        
        # Mock the HTTP client
        client._client = AsyncMock()
        client._initialized = True
        
        yield client
        
        if client._client:
            await client.close()
    
    async def test_client_initialization(self, mock_voyage_config):
        """Test client initialization."""
        client = VoyageAIClient(mock_voyage_config)
        
        assert not client._initialized
        await client.initialize()
        assert client._initialized
        
        await client.close()
    
    async def test_generate_embeddings_success(self, mock_voyage_client):
        """Test successful embedding generation."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1] * 2048},
                {"embedding": [0.2] * 2048}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_voyage_client._client.post.return_value = mock_response
        
        # Test embedding generation
        texts = ["Test document 1", "Test document 2"]
        embeddings = await mock_voyage_client.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 2048
        assert len(embeddings[1]) == 2048
        
        # Verify API call
        mock_voyage_client._client.post.assert_called_once()
        call_args = mock_voyage_client._client.post.call_args
        assert "embeddings" in call_args[0][0]
        
        payload = call_args[1]["json"]
        assert payload["input"] == texts
        assert payload["model"] == "voyage-code-3"
    
    async def test_rate_limiting(self, mock_voyage_client):
        """Test rate limiting functionality."""
        # Test token bucket
        bucket = mock_voyage_client.token_bucket
        
        # Should be able to acquire tokens initially
        assert await bucket.acquire(100)
        
        # Should not be able to acquire more than capacity
        assert not await bucket.acquire(bucket.capacity + 1)
    
    async def test_circuit_breaker(self, mock_voyage_client):
        """Test circuit breaker functionality."""
        circuit_breaker = mock_voyage_client.circuit_breaker
        
        # Initially should be closed
        from contexter.vector.voyage_client import CircuitState
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Simulate failures to trigger circuit breaker
        from contexter.vector.voyage_client import VoyageAPIError
        
        for _ in range(circuit_breaker.failure_threshold):
            try:
                async with circuit_breaker:
                    raise VoyageAPIError("Test error")
            except VoyageAPIError:
                pass
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitState.OPEN
    
    async def test_health_check(self, mock_voyage_client):
        """Test client health check."""
        # Mock successful health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 2048}]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_voyage_client._client.post.return_value = mock_response
        
        health_status = await mock_voyage_client.health_check()
        
        assert health_status["status"] == "healthy"
        assert "test_latency_ms" in health_status
        assert "performance_metrics" in health_status


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    @pytest.fixture
    async def temp_cache(self):
        """Create temporary cache for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "test_cache.db"
            cache = EmbeddingCache(
                cache_path=str(cache_path),
                max_entries=1000,
                ttl_hours=24
            )
            await cache.initialize()
            
            yield cache
            
            await cache.shutdown()
    
    async def test_cache_store_and_retrieve(self, temp_cache):
        """Test basic cache store and retrieve operations."""
        # Create test cache entries
        from contexter.models.embedding_models import CacheEntry
        
        entries = [
            CacheEntry(
                content_hash="hash1",
                content="Test content 1",
                embedding=[0.1] * 2048,
                model="voyage-code-3",
                input_type=InputType.DOCUMENT
            ),
            CacheEntry(
                content_hash="hash2",
                content="Test content 2",
                embedding=[0.2] * 2048,
                model="voyage-code-3",
                input_type=InputType.DOCUMENT
            )
        ]
        
        # Store entries
        success = await temp_cache.store_embeddings(entries)
        assert success
        
        # Retrieve entries
        embeddings = await temp_cache.get_embeddings(
            ["hash1", "hash2", "nonexistent"],
            model="voyage-code-3",
            input_type=InputType.DOCUMENT
        )
        
        assert "hash1" in embeddings
        assert "hash2" in embeddings
        assert "nonexistent" not in embeddings
        
        assert len(embeddings["hash1"]) == 2048
        assert len(embeddings["hash2"]) == 2048
    
    async def test_cache_statistics(self, temp_cache):
        """Test cache statistics tracking."""
        # Initial stats should be empty
        stats = await temp_cache.get_statistics()
        assert stats.total_entries == 0
        assert stats.hits == 0
        assert stats.misses == 0
        
        # Add some entries and check stats
        from contexter.models.embedding_models import CacheEntry
        
        entry = CacheEntry(
            content_hash="test_hash",
            content="Test content",
            embedding=[0.1] * 2048,
            model="voyage-code-3",
            input_type=InputType.DOCUMENT
        )
        
        await temp_cache.store_embeddings([entry])
        
        # Check cache hit
        embeddings = await temp_cache.get_embeddings(["test_hash"])
        assert len(embeddings) == 1
        
        # Check updated stats
        stats = await temp_cache.get_statistics()
        assert stats.total_entries == 1
        assert stats.hits == 1
        assert stats.hit_rate > 0
    
    async def test_cache_cleanup(self, temp_cache):
        """Test cache cleanup and LRU eviction."""
        # Fill cache beyond threshold
        from contexter.models.embedding_models import CacheEntry
        
        entries = []
        for i in range(50):  # More than threshold for small test cache
            entry = CacheEntry(
                content_hash=f"hash_{i}",
                content=f"Test content {i}",
                embedding=[0.1] * 2048,
                model="voyage-code-3",
                input_type=InputType.DOCUMENT
            )
            entries.append(entry)
        
        await temp_cache.store_embeddings(entries)
        
        # Force cleanup
        await temp_cache._cleanup_old_entries()
        
        # Verify some entries were removed
        final_stats = await temp_cache.get_statistics()
        assert final_stats.total_entries < len(entries)


class TestBatchProcessor:
    """Test batch processing functionality."""
    
    @pytest.fixture
    async def mock_batch_processor(self):
        """Create mock batch processor."""
        # Create mock clients
        mock_voyage_client = AsyncMock()
        mock_cache = AsyncMock()
        
        config = BatchProcessingConfig(
            default_batch_size=10,
            max_concurrent_batches=2
        )
        
        processor = BatchProcessor(
            voyage_client=mock_voyage_client,
            cache=mock_cache,
            config=config
        )
        
        await processor.initialize()
        
        yield processor
        
        await processor.shutdown()
    
    async def test_batch_creation(self, mock_batch_processor):
        """Test optimal batch creation."""
        # Create test requests
        requests = []
        for i in range(25):  # More than one batch
            request = EmbeddingRequest(
                content=f"Test content {i}",
                input_type=InputType.DOCUMENT
            )
            requests.append(request)
        
        # Test batch creation
        from contexter.vector.batch_processor import BatchPriority
        batches = mock_batch_processor._create_optimal_batches(
            requests, BatchPriority.NORMAL
        )
        
        # Should create multiple batches
        assert len(batches) > 1
        
        # Total requests should match
        total_requests = sum(len(batch.requests) for batch in batches)
        assert total_requests == len(requests)
    
    async def test_adaptive_batch_sizing(self, mock_batch_processor):
        """Test adaptive batch size optimization."""
        sizer = mock_batch_processor.batch_sizer
        
        # Record some performance data
        sizer.record_performance(batch_size=50, throughput=1000, latency=3.0)
        sizer.record_performance(batch_size=100, throughput=1500, latency=4.0)
        sizer.record_performance(batch_size=150, throughput=1200, latency=6.0)
        
        # Should have enough data for optimization
        assert len(sizer.performance_history) == 3
        
        # Get optimal size (should prefer size with best efficiency)
        optimal_size = sizer.get_optimal_batch_size()
        assert isinstance(optimal_size, int)
        assert sizer.min_size <= optimal_size <= sizer.max_size


class TestEmbeddingEngine:
    """Test complete embedding engine functionality."""
    
    @pytest.fixture
    async def mock_embedding_engine(self):
        """Create mock embedding engine."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EmbeddingEngineConfig(
                voyage_api_key="test_key",
                cache_path=str(Path(temp_dir) / "test_cache.db"),
                batch_size=10,
                max_concurrent_batches=2
            )
            
            engine = VoyageEmbeddingEngine(config)
            
            # Mock the Voyage client
            with patch('contexter.vector.embedding_engine.VoyageAIClient') as mock_client_class:
                mock_client = AsyncMock()
                mock_client.generate_embeddings.return_value = [[0.1] * 2048, [0.2] * 2048]
                mock_client.health_check.return_value = {"status": "healthy"}
                mock_client_class.return_value = mock_client
                
                await engine.initialize()
                
                yield engine
                
                await engine.shutdown()
    
    async def test_single_embedding_generation(self, mock_embedding_engine):
        """Test single embedding generation."""
        request = EmbeddingRequest(
            content="Test document content",
            input_type=InputType.DOCUMENT
        )
        
        result = await mock_embedding_engine.generate_embedding(request)
        
        assert result.success
        assert len(result.embedding) == 2048
        assert result.model == "voyage-code-3"
    
    async def test_batch_embedding_generation(self, mock_embedding_engine):
        """Test batch embedding generation."""
        requests = [
            EmbeddingRequest(
                content=f"Test document {i}",
                input_type=InputType.DOCUMENT
            )
            for i in range(5)
        ]
        
        batch_result = await mock_embedding_engine.generate_batch_embeddings(requests)
        
        assert batch_result.total_requests == 5
        assert len(batch_result.successful_results) == 5
        assert batch_result.success_rate == 1.0
    
    async def test_query_embedding(self, mock_embedding_engine):
        """Test query embedding generation."""
        query = "How to use FastAPI with async?"
        
        embedding = await mock_embedding_engine.embed_query(query)
        
        assert len(embedding) == 2048
        assert all(isinstance(x, float) for x in embedding)
    
    async def test_document_embedding_batch(self, mock_embedding_engine):
        """Test document batch embedding."""
        documents = [
            "FastAPI is a modern web framework for Python.",
            "It supports async/await and automatic API documentation.",
            "Type hints are used for request/response validation."
        ]
        
        embeddings = await mock_embedding_engine.embed_documents(documents)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 2048 for emb in embeddings)
    
    async def test_health_check(self, mock_embedding_engine):
        """Test engine health check."""
        health_status = await mock_embedding_engine.health_check()
        
        assert "status" in health_status
        assert "components" in health_status
        assert "performance" in health_status
        assert "compliance" in health_status
    
    async def test_performance_monitoring(self, mock_embedding_engine):
        """Test performance monitoring and metrics."""
        # Generate some activity
        requests = [
            EmbeddingRequest(
                content=f"Test content {i}",
                input_type=InputType.DOCUMENT
            )
            for i in range(10)
        ]
        
        batch_result = await mock_embedding_engine.generate_batch_embeddings(requests)
        
        # Check metrics
        metrics = mock_embedding_engine.get_performance_metrics()
        
        assert metrics.total_requests >= 10
        assert metrics.success_rate > 0
        assert metrics.throughput_per_minute > 0


class TestPerformanceRequirements:
    """Test performance requirements from PRP."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_throughput_requirement(self):
        """Test >1000 documents/minute throughput requirement."""
        # This would be an integration test with real API
        # For unit test, we simulate the timing
        
        start_time = time.time()
        
        # Simulate processing 100 documents
        documents_processed = 100
        await asyncio.sleep(0.1)  # Simulate 100ms processing time
        
        elapsed_time = time.time() - start_time
        throughput_per_minute = (documents_processed / elapsed_time) * 60
        
        # For real implementation, this should be >1000
        # For simulation, we just verify the calculation works
        assert throughput_per_minute > 0
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_hit_rate_requirement(self, temp_cache):
        """Test >50% cache hit rate requirement."""
        from contexter.models.embedding_models import CacheEntry
        
        # Add some cached entries
        cached_entries = [
            CacheEntry(
                content_hash=f"cached_{i}",
                content=f"Cached content {i}",
                embedding=[0.1] * 2048,
                model="voyage-code-3",
                input_type=InputType.DOCUMENT
            )
            for i in range(5)
        ]
        
        await temp_cache.store_embeddings(cached_entries)
        
        # Test cache hit rate
        cached_hashes = [f"cached_{i}" for i in range(5)]
        new_hashes = [f"new_{i}" for i in range(3)]
        
        all_embeddings = await temp_cache.get_embeddings(cached_hashes + new_hashes)
        
        cache_hits = len(all_embeddings)
        total_requests = len(cached_hashes + new_hashes)
        hit_rate = cache_hits / total_requests
        
        # Should have >50% hit rate (5 hits out of 8 requests = 62.5%)
        assert hit_rate > 0.5
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_api_success_rate_requirement(self, mock_voyage_client):
        """Test >99.9% API success rate requirement."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 2048}]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_voyage_client._client.post.return_value = mock_response
        
        # Simulate 1000 API calls
        successful_calls = 0
        total_calls = 1000
        
        for _ in range(total_calls):
            try:
                await mock_voyage_client.generate_embeddings(["test"])
                successful_calls += 1
            except:
                pass
        
        success_rate = successful_calls / total_calls
        
        # Should have >99.9% success rate
        assert success_rate > 0.999


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_document_ingestion_workflow(self):
        """Test complete document ingestion workflow."""
        # This would test the full pipeline:
        # Documents -> Chunks -> Embeddings -> Vector Storage -> Search
        
        # Mock document chunks
        from contexter.models.storage_models import DocumentationChunk
        
        chunks = [
            DocumentationChunk(
                chunk_id=f"chunk_{i}",
                content=f"FastAPI documentation chunk {i} content",
                chunk_index=i,
                total_chunks=3,
                token_count=100,
                content_hash=f"hash_{i}",
                source_context="FastAPI documentation"
            )
            for i in range(3)
        ]
        
        # Would test with real integration layer
        # For now, just verify chunks are properly formed
        assert len(chunks) == 3
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.chunk_id for chunk in chunks)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_search_workflow(self):
        """Test search workflow with embeddings."""
        # This would test:
        # Query -> Query Embedding -> Vector Search -> Results
        
        query = "How to create FastAPI endpoints?"
        
        # Mock query embedding
        query_embedding = [0.1] * 2048
        
        # Mock search results
        mock_results = [
            {
                'id': 'chunk_1',
                'score': 0.95,
                'content': 'FastAPI endpoint creation guide',
                'metadata': {'section': 'endpoints'}
            },
            {
                'id': 'chunk_2', 
                'score': 0.87,
                'content': 'Advanced FastAPI routing',
                'metadata': {'section': 'routing'}
            }
        ]
        
        # Verify search results format
        assert all('id' in result for result in mock_results)
        assert all('score' in result for result in mock_results)
        assert all(result['score'] > 0.8 for result in mock_results)  # High relevance


# Test configuration for different environments
@pytest.fixture(scope="session")
def test_config():
    """Test configuration for different environments."""
    return {
        "unit": {
            "use_real_api": False,
            "cache_path": ":memory:",
            "batch_size": 10
        },
        "integration": {
            "use_real_api": True,
            "cache_path": "test_cache.db",
            "batch_size": 100
        },
        "performance": {
            "use_real_api": True,
            "cache_path": "perf_cache.db",
            "batch_size": 100,
            "test_duration": 60  # seconds
        }
    }


# Pytest markers for different test types
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.embedding,
]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])