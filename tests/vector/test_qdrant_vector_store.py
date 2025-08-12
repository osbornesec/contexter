"""
Tests for QdrantVectorStore class.

Validates core vector database operations, performance characteristics,
and integration with Qdrant database system.
"""

import pytest
import pytest_asyncio
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from contexter.vector.qdrant_vector_store import (
    QdrantVectorStore, VectorStoreConfig, VectorDocument, SearchResult
)


@pytest.mark.asyncio
class TestQdrantVectorStore:
    """Test suite for QdrantVectorStore functionality."""
    
    @pytest_asyncio.fixture
    async def mock_qdrant_client(self):
        """Create a mock Qdrant client for testing."""
        with patch('contexter.vector.qdrant_vector_store.AsyncQdrantClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock collections response
            mock_collections = Mock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            
            # Mock collection info
            mock_collection_info = Mock()
            mock_collection_info.points_count = 1000
            mock_collection_info.vectors_count = 1000
            mock_collection_info.indexed_vectors_count = 1000
            mock_collection_info.status = Mock(value="green")
            mock_collection_info.optimizer_status = Mock(value="ok")
            mock_collection_info.config = Mock()
            mock_collection_info.config.params = Mock()
            mock_collection_info.config.params.vectors = Mock()
            mock_collection_info.config.params.vectors.size = 2048
            mock_collection_info.config.params.vectors.distance = Mock(value="Cosine")
            mock_collection_info.config.hnsw_config = Mock()
            mock_collection_info.config.hnsw_config.m = 16
            mock_collection_info.config.hnsw_config.ef_construct = 200
            mock_client.get_collection.return_value = mock_collection_info
            
            # Mock create collection
            mock_client.create_collection.return_value = True
            mock_client.create_payload_index.return_value = True
            
            yield mock_client
            
    @pytest_asyncio.fixture
    async def vector_store(self, mock_qdrant_client):
        """Create a vector store instance for testing."""
        config = VectorStoreConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=128,  # Smaller for testing
            batch_size=10
        )
        
        store = QdrantVectorStore(config)
        await store.initialize()
        return store
        
    @pytest.fixture
    def sample_vector_documents(self):
        """Create sample vector documents for testing."""
        documents = []
        for i in range(50):
            doc = VectorDocument(
                id=f"doc_{i}",
                vector=np.random.random(128).tolist(),
                payload={
                    "library_id": f"test_lib_{i % 5}",
                    "doc_type": "api" if i % 2 == 0 else "guide",
                    "section": f"section_{i % 3}",
                    "content": f"This is test document {i} content",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "trust_score": 7.5,
                    "programming_language": "python"
                }
            )
            documents.append(doc)
        return documents
        
    async def test_vector_store_initialization(self, mock_qdrant_client):
        """Test vector store initializes correctly."""
        config = VectorStoreConfig(collection_name="test_init")
        store = QdrantVectorStore(config)
        
        await store.initialize()
        
        assert store._initialized
        assert store.client is not None
        assert mock_qdrant_client.get_collections.call_count >= 1
        mock_qdrant_client.create_collection.assert_called_once()
        
    async def test_upsert_single_vector(self, vector_store, mock_qdrant_client):
        """Test upserting a single vector document."""
        # Mock successful upsert
        from qdrant_client.models import UpdateStatus
        mock_result = Mock()
        mock_result.status = UpdateStatus.COMPLETED
        mock_qdrant_client.upsert.return_value = mock_result
        
        document = VectorDocument(
            id="test_doc",
            vector=np.random.random(128).tolist(),
            payload={"test": "data"}
        )
        
        result = await vector_store.upsert_vector(document)
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        
    async def test_upsert_vectors_batch(self, vector_store, sample_vector_documents, mock_qdrant_client):
        """Test batch vector upload functionality."""
        # Mock successful batch upsert
        from qdrant_client.models import UpdateStatus
        mock_result = Mock()
        mock_result.status = UpdateStatus.COMPLETED
        mock_qdrant_client.upsert.return_value = mock_result
        
        result = await vector_store.upsert_vectors_batch(sample_vector_documents)
        
        assert result["successful_uploads"] == len(sample_vector_documents)
        assert result["failed_uploads"] == 0
        assert result["total_time"] > 0
        
        # Should be called multiple times for batching
        assert mock_qdrant_client.upsert.call_count >= 1
        
    async def test_search_vectors(self, vector_store, mock_qdrant_client):
        """Test vector similarity search."""
        # Mock search results
        mock_results = []
        for i in range(5):
            mock_point = Mock()
            mock_point.id = f"result_{i}"
            mock_point.score = 0.9 - (i * 0.1)
            mock_point.payload = {"content": f"Result {i}"}
            mock_results.append(mock_point)
            
        mock_qdrant_client.search.return_value = mock_results
        
        query_vector = np.random.random(128).tolist()
        results = await vector_store.search_vectors(
            query_vector=query_vector,
            top_k=5,
            filters={"doc_type": "api"}
        )
        
        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0 for r in results)
        
        mock_qdrant_client.search.assert_called_once()
        
    async def test_search_performance(self, vector_store, mock_qdrant_client):
        """Test search performance meets requirements."""
        # Mock fast search response
        mock_qdrant_client.search.return_value = []
        
        query_vector = np.random.random(128).tolist()
        
        # Measure search latency
        start_time = time.time()
        await vector_store.search_vectors(query_vector)
        search_time = time.time() - start_time
        
        # Should be very fast with mocks (< 10ms)
        assert search_time < 0.01
        
    async def test_get_vector(self, vector_store, mock_qdrant_client):
        """Test retrieving a specific vector by ID."""
        # Mock retrieve response
        mock_point = Mock()
        mock_point.id = "test_id"
        mock_point.vector = [0.1, 0.2, 0.3]
        mock_point.payload = {"test": "data"}
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        result = await vector_store.get_vector("test_id")
        
        assert result is not None
        assert result.id == "test_id"
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.payload == {"test": "data"}
        
    async def test_delete_vector(self, vector_store, mock_qdrant_client):
        """Test deleting a vector by ID."""
        # Mock successful delete
        from qdrant_client.models import UpdateStatus
        mock_result = Mock()
        mock_result.status = UpdateStatus.COMPLETED
        mock_qdrant_client.delete.return_value = mock_result
        
        result = await vector_store.delete_vector("test_id")
        
        assert result is True
        mock_qdrant_client.delete.assert_called_once()
        
    async def test_count_vectors(self, vector_store, mock_qdrant_client):
        """Test counting vectors with and without filters."""
        # Mock count response
        mock_result = Mock()
        mock_result.count = 1500
        mock_qdrant_client.count.return_value = mock_result
        
        count = await vector_store.count_vectors()
        assert count == 1500
        
        # Test with filters
        count_filtered = await vector_store.count_vectors({"doc_type": "api"})
        assert count_filtered == 1500
        
        assert mock_qdrant_client.count.call_count == 2
        
    async def test_collection_optimization(self, vector_store, mock_qdrant_client):
        """Test collection optimization functionality."""
        # Mock collection info with enough vectors
        mock_collection_info = Mock()
        mock_collection_info.points_count = 50000
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        mock_qdrant_client.update_collection.return_value = True
        
        result = await vector_store.optimize_collection()
        
        assert result is True
        mock_qdrant_client.update_collection.assert_called_once()
        
    async def test_get_collection_stats(self, vector_store, mock_qdrant_client):
        """Test getting collection statistics."""
        stats = await vector_store.get_collection_stats()
        
        assert isinstance(stats, dict)
        assert "collection_name" in stats
        assert "points_count" in stats
        assert "vectors_count" in stats
        assert "status" in stats
        assert "config" in stats
        assert "performance_metrics" in stats
        
    async def test_filter_building(self, vector_store):
        """Test filter building for metadata queries."""
        # Test exact match filter
        filters = {"doc_type": "api", "language": "python"}
        qdrant_filter = vector_store._build_filter(filters)
        
        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 2
        
        # Test range filter
        range_filters = {"trust_score": {"gte": 5.0, "lte": 10.0}}
        range_filter = vector_store._build_filter(range_filters)
        
        assert range_filter is not None
        assert len(range_filter.must) == 2
        
        # Test list filter
        list_filters = {"doc_type": ["api", "guide"]}
        list_filter = vector_store._build_filter(list_filters)
        
        assert list_filter is not None
        assert len(list_filter.must) == 2
        
    async def test_concurrent_operations(self, vector_store, sample_vector_documents, mock_qdrant_client):
        """Test concurrent vector operations."""
        # Mock successful operations
        mock_result = Mock()
        mock_result.status.value = "completed"
        mock_qdrant_client.upsert.return_value = mock_result
        mock_qdrant_client.search.return_value = []
        
        # Create concurrent tasks
        tasks = []
        
        # Batch upload task
        upload_task = vector_store.upsert_vectors_batch(sample_vector_documents[:10])
        tasks.append(upload_task)
        
        # Multiple search tasks
        for _ in range(5):
            query_vector = np.random.random(128).tolist()
            search_task = vector_store.search_vectors(query_vector)
            tasks.append(search_task)
            
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 6
        assert results[0]["successful_uploads"] == 10  # Upload result
        assert all(isinstance(r, list) for r in results[1:])  # Search results
        
    async def test_error_handling(self, vector_store, mock_qdrant_client):
        """Test error handling in various scenarios."""
        # Test connection error
        mock_qdrant_client.search.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            query_vector = np.random.random(128).tolist()
            await vector_store.search_vectors(query_vector)
            
        # Test partial batch failure
        mock_qdrant_client.search.side_effect = None  # Reset
        mock_result = Mock()
        mock_result.status.value = "failed"
        mock_qdrant_client.upsert.return_value = mock_result
        
        documents = [VectorDocument(
            id="test",
            vector=np.random.random(128).tolist(),
            payload={}
        )]
        
        result = await vector_store.upsert_vectors_batch(documents)
        assert result["failed_uploads"] == 1
        assert result["successful_uploads"] == 0
        
    async def test_health_status(self, vector_store):
        """Test health status reporting."""
        health = vector_store.get_health_status()
        
        assert isinstance(health, dict)
        assert "healthy" in health
        assert "initialized" in health
        assert "metrics" in health
        assert "config" in health
        
        # Health should be True after successful initialization
        assert health["healthy"] is True
        assert health["initialized"] is True
        
    async def test_memory_efficiency(self, vector_store, mock_qdrant_client):
        """Test memory efficiency with large batches."""
        # Mock successful batch operations
        mock_result = Mock()
        mock_result.status.value = "completed"
        mock_qdrant_client.upsert.return_value = mock_result
        
        # Create a large number of documents
        large_batch = []
        for i in range(1000):
            doc = VectorDocument(
                id=f"large_doc_{i}",
                vector=np.random.random(128).tolist(),
                payload={"index": i}
            )
            large_batch.append(doc)
            
        # Process large batch
        result = await vector_store.upsert_vectors_batch(large_batch, batch_size=100)
        
        assert result["successful_uploads"] == 1000
        assert result["failed_uploads"] == 0
        
        # Should have been processed in multiple batches
        assert mock_qdrant_client.upsert.call_count >= 10
        
    async def test_cleanup(self, vector_store, mock_qdrant_client):
        """Test proper cleanup of resources."""
        assert vector_store._initialized is True
        
        await vector_store.cleanup()
        
        assert vector_store._initialized is False
        mock_qdrant_client.close.assert_called_once()


@pytest.mark.asyncio 
class TestVectorStoreConfig:
    """Test vector store configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VectorStoreConfig()
        
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.vector_size == 2048
        assert config.distance_metric == "Cosine"
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construct == 200
        assert config.batch_size == 1000
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            host="custom_host",
            port=9999,
            vector_size=1024,
            distance_metric="Dot",
            batch_size=500
        )
        
        assert config.host == "custom_host"
        assert config.port == 9999
        assert config.vector_size == 1024
        assert config.distance_metric == "Dot"
        assert config.batch_size == 500
        
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should work
        config = VectorStoreConfig(vector_size=128)
        assert config.vector_size == 128
        
        # Test with various valid values
        config = VectorStoreConfig(
            hnsw_m=32,
            hnsw_ef_construct=400,
            search_ef=256
        )
        assert config.hnsw_m == 32
        assert config.hnsw_ef_construct == 400
        assert config.search_ef == 256


@pytest.mark.asyncio
class TestVectorDocument:
    """Test VectorDocument model."""
    
    def test_vector_document_creation(self):
        """Test creating vector documents."""
        vector = [0.1, 0.2, 0.3, 0.4]
        payload = {"test": "data", "score": 5.0}
        
        doc = VectorDocument(
            id="test_id",
            vector=vector,
            payload=payload
        )
        
        assert doc.id == "test_id"
        assert doc.vector == vector
        assert doc.payload == payload
        
    def test_vector_document_auto_id(self):
        """Test automatic ID generation."""
        doc = VectorDocument(
            vector=[0.1, 0.2],
            payload={}
        )
        
        assert doc.id is not None
        assert len(doc.id) > 0
        
    def test_vector_document_validation(self):
        """Test vector document validation."""
        # Valid document
        doc = VectorDocument(
            vector=[1.0, 2.0, 3.0],
            payload={"valid": True}
        )
        assert len(doc.vector) == 3
        
        # Empty vector should be allowed
        doc = VectorDocument(vector=[], payload={})
        assert len(doc.vector) == 0
        
        # Empty payload should be allowed
        doc = VectorDocument(vector=[1.0], payload={})
        assert doc.payload == {}


@pytest.mark.asyncio
class TestSearchResult:
    """Test SearchResult model."""
    
    def test_search_result_creation(self):
        """Test creating search results."""
        result = SearchResult(
            id="result_1",
            score=0.95,
            payload={"content": "test"},
            vector=[0.1, 0.2]
        )
        
        assert result.id == "result_1"
        assert result.score == 0.95
        assert result.payload == {"content": "test"}
        assert result.vector == [0.1, 0.2]
        
    def test_search_result_without_vector(self):
        """Test search result without vector data."""
        result = SearchResult(
            id="result_2",
            score=0.87,
            payload={"type": "api"}
        )
        
        assert result.id == "result_2"
        assert result.score == 0.87
        assert result.vector is None