"""
Tests for VectorSearchEngine class.

Validates search optimization, caching, ranking algorithms,
and performance characteristics.
"""

import pytest
import pytest_asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from typing import List

from contexter.vector.vector_search_engine import (
    VectorSearchEngine, SearchQuery, RankedSearchResult, SearchCache
)
from contexter.vector.qdrant_vector_store import (
    QdrantVectorStore, VectorStoreConfig, SearchResult
)


@pytest.mark.asyncio
class TestVectorSearchEngine:
    """Test suite for VectorSearchEngine functionality."""
    
    @pytest_asyncio.fixture
    async def mock_vector_store(self):
        """Create a mock vector store for testing."""
        store = Mock(spec=QdrantVectorStore)
        store.config = VectorStoreConfig(vector_size=128)
        
        # Mock search results
        async def mock_search_vectors(query_vector, top_k=10, filters=None, score_threshold=None):
            results = []
            for i in range(min(top_k, 5)):
                result = SearchResult(
                    id=f"result_{i}",
                    score=0.9 - (i * 0.1),
                    payload={
                        "content": f"Test content {i}",
                        "doc_type": "api" if i % 2 == 0 else "guide",
                        "trust_score": 8.0 - i,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "programming_language": "python"
                    }
                )
                results.append(result)
            return results
            
        store.search_vectors = AsyncMock(side_effect=mock_search_vectors)
        
        # Mock get_vector
        async def mock_get_vector(vector_id):
            if vector_id == "existing_doc":
                return Mock(
                    id=vector_id,
                    vector=np.random.random(128).tolist(),
                    payload={"test": "data"}
                )
            return None
            
        store.get_vector = AsyncMock(side_effect=mock_get_vector)
        
        return store
        
    @pytest_asyncio.fixture
    async def search_engine(self, mock_vector_store):
        """Create a search engine instance for testing."""
        return VectorSearchEngine(
            vector_store=mock_vector_store,
            enable_caching=True,
            cache_size=100,
            cache_ttl=300
        )
        
    @pytest.fixture
    def sample_search_query(self):
        """Create a sample search query."""
        return SearchQuery(
            vector=np.random.random(128).tolist(),
            text="test query",
            top_k=10,
            filters={"doc_type": "api"},
            score_threshold=0.5
        )
        
    async def test_search_engine_initialization(self, mock_vector_store):
        """Test search engine initializes correctly."""
        engine = VectorSearchEngine(mock_vector_store)
        
        assert engine.vector_store == mock_vector_store
        assert engine.enable_caching is True
        assert engine._cache is not None
        assert engine._metrics["total_searches"] == 0
        
    async def test_basic_search(self, search_engine, sample_search_query):
        """Test basic search functionality."""
        results = await search_engine.search(sample_search_query)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RankedSearchResult) for r in results)
        assert all(r.rank > 0 for r in results)
        assert all(r.relevance_score > 0 for r in results)
        assert all(len(r.match_reasons) > 0 for r in results)
        
        # Check that metrics were updated
        assert search_engine._metrics["total_searches"] == 1
        
    async def test_search_caching(self, search_engine, sample_search_query):
        """Test search result caching."""
        # First search should miss cache
        results1 = await search_engine.search(sample_search_query)
        assert search_engine._metrics["cache_misses"] == 1
        assert search_engine._metrics["cache_hits"] == 0
        
        # Second identical search should hit cache
        results2 = await search_engine.search(sample_search_query)
        assert search_engine._metrics["cache_hits"] == 1
        assert search_engine._metrics["cache_misses"] == 1
        
        # Results should be identical
        assert len(results1) == len(results2)
        assert results1[0].id == results2[0].id
        
    async def test_search_without_caching(self, mock_vector_store, sample_search_query):
        """Test search engine without caching."""
        engine = VectorSearchEngine(
            vector_store=mock_vector_store,
            enable_caching=False
        )
        
        assert engine._cache is None
        
        results = await engine.search(sample_search_query)
        assert len(results) > 0
        
        # Cache metrics should remain zero
        assert engine._metrics["cache_hits"] == 0
        assert engine._metrics["cache_misses"] == 0
        
    async def test_search_ranking(self, search_engine):
        """Test search result ranking and relevance scoring."""
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api"}
        )
        
        results = await search_engine.search(query)
        
        # Results should be ranked (rank field populated)
        assert all(r.rank == i + 1 for i, r in enumerate(results))
        
        # Relevance scores should be calculated
        assert all(r.relevance_score > 0 for r in results)
        
        # Higher scores should generally come first (vector similarity + boosts)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        
    async def test_search_filters(self, search_engine, mock_vector_store):
        """Test search with various filter combinations."""
        # Test with single filter
        query1 = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api"}
        )
        
        results1 = await search_engine.search(query1)
        assert len(results1) > 0
        
        # Test with multiple filters
        query2 = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api", "programming_language": "python"}
        )
        
        results2 = await search_engine.search(query2)
        assert len(results2) > 0
        
        # Verify filters were passed to vector store
        assert mock_vector_store.search_vectors.call_count >= 2
        
    async def test_search_similar_documents(self, search_engine):
        """Test finding similar documents functionality."""
        # Test with existing document
        results = await search_engine.search_similar_documents(
            document_id="existing_doc",
            top_k=5
        )
        
        assert isinstance(results, list)
        # Should not include the reference document itself
        assert all(r.id != "existing_doc" for r in results)
        
        # Test with non-existent document
        results_empty = await search_engine.search_similar_documents(
            document_id="non_existent",
            top_k=5
        )
        
        assert len(results_empty) == 0
        
    async def test_search_by_metadata(self, search_engine):
        """Test metadata-only search functionality."""
        filters = {"doc_type": "api", "programming_language": "python"}
        
        results = await search_engine.search_by_metadata(filters, top_k=20)
        
        assert isinstance(results, list)
        # Should call vector store with zero vector
        search_engine.vector_store.search_vectors.assert_called()
        
    async def test_search_performance_tracking(self, search_engine, sample_search_query):
        """Test search performance metrics tracking."""
        initial_searches = search_engine._metrics["total_searches"]
        initial_avg_time = search_engine._metrics["avg_search_time"]
        
        # Perform multiple searches
        for _ in range(3):
            await search_engine.search(sample_search_query)
            
        metrics = search_engine.get_search_metrics()
        
        assert metrics["search_metrics"]["total_searches"] == initial_searches + 3
        assert metrics["search_metrics"]["avg_search_time"] > initial_avg_time
        assert metrics["search_metrics"]["last_search_time"] is not None
        
    async def test_cache_warm_up(self, search_engine):
        """Test cache warming functionality."""
        # Create multiple queries for warming
        queries = []
        for i in range(3):
            query = SearchQuery(
                vector=np.random.random(128).tolist(),
                filters={"doc_type": "api" if i % 2 == 0 else "guide"}
            )
            queries.append(query)
            
        await search_engine.warm_cache(queries)
        
        # All queries should now be cached
        for query in queries:
            results = await search_engine.search(query)
            assert len(results) > 0
            
        # Should have cache hits for warmed queries
        assert search_engine._metrics["cache_hits"] >= len(queries)
        
    async def test_clear_cache(self, search_engine, sample_search_query):
        """Test cache clearing functionality."""
        # Populate cache
        await search_engine.search(sample_search_query)
        assert search_engine._metrics["cache_misses"] == 1
        
        # Clear cache
        search_engine.clear_cache()
        
        # Same query should miss cache again
        await search_engine.search(sample_search_query)
        assert search_engine._metrics["cache_misses"] == 2
        
    async def test_search_parameter_optimization(self, search_engine):
        """Test search parameter optimization."""
        # Simulate slow searches
        search_engine._metrics["avg_search_time"] = 0.1  # 100ms
        
        optimized_params = search_engine.optimize_search_parameters(target_latency_ms=50.0)
        
        assert "ef" in optimized_params
        assert "current_latency_ms" in optimized_params
        assert "target_latency_ms" in optimized_params
        assert optimized_params["current_latency_ms"] == 100.0
        
    async def test_match_reasons_generation(self, search_engine):
        """Test generation of match reasons for search results."""
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api", "programming_language": "python"}
        )
        
        results = await search_engine.search(query)
        
        for result in results:
            assert len(result.match_reasons) > 0
            
            # Should include vector similarity reason
            similarity_reasons = [r for r in result.match_reasons if "similarity" in r.lower()]
            assert len(similarity_reasons) > 0
            
            # Should include filter match reasons if applicable
            filter_reasons = [r for r in result.match_reasons if "Matches" in r]
            if result.payload.get("doc_type") == "api":
                assert any("doc_type" in r for r in filter_reasons)
                
    async def test_relevance_score_calculation(self, search_engine):
        """Test relevance score calculation logic."""
        # Create query with filters that should boost certain results
        query = SearchQuery(
            vector=np.random.random(128).tolist(),
            filters={"doc_type": "api"}
        )
        
        results = await search_engine.search(query)
        
        for result in results:
            # Relevance score should be based on vector score + boosts
            assert result.relevance_score >= result.score
            
            # Results with higher trust scores should get boosted
            if result.payload.get("trust_score", 0) > 7.0:
                assert result.relevance_score > result.score
                
    async def test_error_handling(self, search_engine, mock_vector_store):
        """Test error handling in search operations."""
        # Mock vector store to raise exception
        mock_vector_store.search_vectors.side_effect = Exception("Search failed")
        
        query = SearchQuery(vector=np.random.random(128).tolist())
        
        with pytest.raises(Exception, match="Search failed"):
            await search_engine.search(query)
            
        # Reset mock
        mock_vector_store.search_vectors.side_effect = None
        
    async def test_concurrent_searches(self, search_engine):
        """Test concurrent search operations."""
        import asyncio
        
        # Create multiple different queries
        queries = []
        for i in range(5):
            query = SearchQuery(
                vector=np.random.random(128).tolist(),
                filters={"doc_type": "api" if i % 2 == 0 else "guide"},
                top_k=5 + i
            )
            queries.append(query)
            
        # Execute searches concurrently
        tasks = [search_engine.search(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # All searches should complete successfully
        assert len(results) == 5
        assert all(len(result) > 0 for result in results)
        
        # Metrics should reflect all searches
        assert search_engine._metrics["total_searches"] >= 5


@pytest.mark.asyncio
class TestSearchCache:
    """Test search cache functionality."""
    
    @pytest.fixture
    def search_cache(self):
        """Create a search cache for testing."""
        return SearchCache(max_size=5, ttl_seconds=1)
        
    @pytest.fixture
    def sample_query(self):
        """Create a sample search query."""
        return SearchQuery(
            vector=[0.1, 0.2, 0.3],
            filters={"test": "value"}
        )
        
    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            RankedSearchResult(
                id="result_1",
                score=0.9,
                payload={"content": "test"},
                rank=1,
                relevance_score=0.95,
                match_reasons=["High similarity"]
            )
        ]
        
    def test_cache_put_and_get(self, search_cache, sample_query, sample_results):
        """Test basic cache put and get operations."""
        # Cache should be empty initially
        assert search_cache.get(sample_query) is None
        
        # Put results in cache
        search_cache.put(sample_query, sample_results)
        
        # Should be able to retrieve results
        cached_results = search_cache.get(sample_query)
        assert cached_results is not None
        assert len(cached_results) == len(sample_results)
        assert cached_results[0].id == sample_results[0].id
        
    async def test_cache_ttl_expiration(self, search_cache, sample_query, sample_results):
        """Test cache TTL expiration."""
        # Put results in cache
        search_cache.put(sample_query, sample_results)
        
        # Should be retrievable immediately
        assert search_cache.get(sample_query) is not None
        
        # Wait for TTL to expire (1 second + buffer)
        import asyncio
        await asyncio.sleep(1.1)
        
        # Should be expired and return None
        assert search_cache.get(sample_query) is None
        
    def test_cache_size_limit(self, sample_results):
        """Test cache size limit enforcement."""
        cache = SearchCache(max_size=2, ttl_seconds=300)
        
        # Add items up to the limit
        for i in range(3):
            query = SearchQuery(vector=[float(i)])
            cache.put(query, sample_results)
            
        # Cache should only contain the last 2 items
        stats = cache.get_stats()
        assert stats["size"] == 2
        
        # First item should be evicted
        first_query = SearchQuery(vector=[0.0])
        assert cache.get(first_query) is None
        
        # Last two should still be there
        second_query = SearchQuery(vector=[1.0])
        third_query = SearchQuery(vector=[2.0])
        assert cache.get(second_query) is not None
        assert cache.get(third_query) is not None
        
    def test_cache_key_generation(self, search_cache):
        """Test cache key generation for different queries."""
        # Same vector should generate same key
        query1 = SearchQuery(vector=[0.1, 0.2, 0.3])
        query2 = SearchQuery(vector=[0.1, 0.2, 0.3])
        
        key1 = search_cache._make_key(query1)
        key2 = search_cache._make_key(query2)
        assert key1 == key2
        
        # Different vectors should generate different keys
        query3 = SearchQuery(vector=[0.1, 0.2, 0.4])
        key3 = search_cache._make_key(query3)
        assert key1 != key3
        
        # Different filters should generate different keys
        query4 = SearchQuery(vector=[0.1, 0.2, 0.3], filters={"test": "different"})
        key4 = search_cache._make_key(query4)
        assert key1 != key4
        
    def test_cache_clear(self, search_cache, sample_query, sample_results):
        """Test cache clearing functionality."""
        # Populate cache
        search_cache.put(sample_query, sample_results)
        assert search_cache.get(sample_query) is not None
        
        # Clear cache
        search_cache.clear()
        
        # Should be empty
        assert search_cache.get(sample_query) is None
        stats = search_cache.get_stats()
        assert stats["size"] == 0
        
    def test_cache_stats(self, search_cache, sample_query, sample_results):
        """Test cache statistics."""
        initial_stats = search_cache.get_stats()
        assert initial_stats["size"] == 0
        assert initial_stats["max_size"] == 5
        
        # Add some items
        search_cache.put(sample_query, sample_results)
        
        stats = search_cache.get_stats()
        assert stats["size"] == 1
        assert stats["oldest_entry"] is not None


@pytest.mark.asyncio
class TestSearchQuery:
    """Test SearchQuery model validation."""
    
    def test_search_query_creation(self):
        """Test creating search queries with various parameters."""
        # Minimal query
        query = SearchQuery(vector=[0.1, 0.2, 0.3])
        assert len(query.vector) == 3
        assert query.top_k == 10  # default
        assert query.text is None
        
        # Full query
        query = SearchQuery(
            vector=[1.0, 2.0],
            text="search text",
            top_k=20,
            filters={"type": "document"},
            score_threshold=0.7,
            search_params={"ef": 128}
        )
        
        assert query.vector == [1.0, 2.0]
        assert query.text == "search text"
        assert query.top_k == 20
        assert query.filters == {"type": "document"}
        assert query.score_threshold == 0.7
        assert query.search_params == {"ef": 128}
        
    def test_search_query_validation(self):
        """Test search query validation."""
        # Valid query
        query = SearchQuery(vector=[1.0, 2.0, 3.0])
        assert len(query.vector) == 3
        
        # Empty vector should be allowed (for metadata-only search)
        query = SearchQuery(vector=[])
        assert len(query.vector) == 0


@pytest.mark.asyncio
class TestRankedSearchResult:
    """Test RankedSearchResult model."""
    
    def test_ranked_result_creation(self):
        """Test creating ranked search results."""
        result = RankedSearchResult(
            id="test_result",
            score=0.85,
            payload={"content": "test"},
            rank=1,
            relevance_score=0.9,
            match_reasons=["High similarity", "Metadata match"]
        )
        
        assert result.id == "test_result"
        assert result.score == 0.85
        assert result.rank == 1
        assert result.relevance_score == 0.9
        assert len(result.match_reasons) == 2
        
    def test_ranked_result_inheritance(self):
        """Test that RankedSearchResult inherits from SearchResult."""
        result = RankedSearchResult(
            id="test",
            score=0.5,
            payload={},
            rank=2,
            relevance_score=0.6
        )
        
        # Should have SearchResult attributes
        assert hasattr(result, "id")
        assert hasattr(result, "score")
        assert hasattr(result, "payload")
        
        # Should have additional RankedSearchResult attributes
        assert hasattr(result, "rank")
        assert hasattr(result, "relevance_score")
        assert hasattr(result, "match_reasons")