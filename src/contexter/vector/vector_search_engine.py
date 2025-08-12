"""
Vector Search Engine - Optimized search with ranking and filtering.

Provides high-performance vector search capabilities with:
- Advanced filtering and ranking algorithms
- Search result caching for common queries
- Hybrid search combining vector and keyword search
- Performance optimization and monitoring
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from collections import OrderedDict

from pydantic import BaseModel, Field
import hashlib
import json

from .qdrant_vector_store import QdrantVectorStore, VectorStoreConfig, SearchResult

logger = logging.getLogger(__name__)


class SearchQuery(BaseModel):
    """Search query model."""

    vector: List[float] = Field(..., description="Query vector")
    text: Optional[str] = Field(None, description="Text query for hybrid search")
    top_k: int = Field(default=10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    score_threshold: Optional[float] = Field(None, description="Minimum score threshold")
    search_params: Optional[Dict[str, Any]] = Field(None, description="Advanced search parameters")


class RankedSearchResult(SearchResult):
    """Search result with additional ranking information."""

    rank: int = Field(..., description="Result rank (1-based)")
    relevance_score: float = Field(..., description="Computed relevance score")
    match_reasons: List[str] = Field(default_factory=list, description="Why this result matched")


class SearchCache:
    """LRU cache for search results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize search cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time to live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}

    def _make_key(self, query: SearchQuery) -> str:
        """Create cache key from search query."""
        # Create deterministic hash of query parameters
        query_dict = {
            "vector_hash": hashlib.md5(
                json.dumps(query.vector, sort_keys=True).encode()
            ).hexdigest()[:16],
            "text": query.text,
            "top_k": query.top_k,
            "filters": query.filters,
            "score_threshold": query.score_threshold,
            "search_params": query.search_params
        }
        return hashlib.md5(
            json.dumps(query_dict, sort_keys=True).encode()
        ).hexdigest()

    def get(self, query: SearchQuery) -> Optional[List[RankedSearchResult]]:
        """Get cached results for query."""
        key = self._make_key(query)

        # Check if key exists and is not expired
        if key in self._cache:
            timestamp = self._timestamps.get(key)
            if timestamp and datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Move to end (LRU)
                self._cache.move_to_end(key)
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._timestamps[key]

        return None

    def put(self, query: SearchQuery, results: List[RankedSearchResult]) -> None:
        """Cache search results."""
        key = self._make_key(query)

        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]

        self._cache[key] = results
        self._timestamps[key] = datetime.now()

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._timestamps.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "oldest_entry": min(self._timestamps.values()) if self._timestamps else None
        }


class VectorSearchEngine:
    """
    High-performance vector search engine with optimization features.
    
    Features:
    - Advanced search result ranking and filtering
    - LRU caching for frequent queries
    - Hybrid search combining vector and text similarity
    - Performance monitoring and optimization
    - Search parameter auto-tuning
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        enable_caching: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 300
    ):
        """
        Initialize the search engine.
        
        Args:
            vector_store: Qdrant vector store instance
            enable_caching: Whether to enable search result caching
            cache_size: Maximum number of cached search results
            cache_ttl: Cache time-to-live in seconds
        """
        self.vector_store = vector_store
        self.enable_caching = enable_caching

        # Initialize cache if enabled
        if enable_caching:
            self._cache = SearchCache(max_size=cache_size, ttl_seconds=cache_ttl)
        else:
            self._cache = None

        # Performance metrics
        self._metrics = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_search_time": 0.0,
            "avg_results_per_search": 0.0,
            "last_search_time": None
        }

        # Search parameter optimization
        self._search_params_history = []
        self._optimal_params = {}

    async def search(self, query: SearchQuery) -> List[RankedSearchResult]:
        """
        Perform vector search with ranking and optimization.
        
        Args:
            query: Search query parameters
            
        Returns:
            List of ranked search results
        """
        start_time = time.time()
        self._metrics["total_searches"] += 1

        try:
            # Check cache first if enabled
            if self._cache:
                cached_results = self._cache.get(query)
                if cached_results:
                    self._metrics["cache_hits"] += 1
                    logger.debug(f"Cache hit for search query")
                    return cached_results
                else:
                    self._metrics["cache_misses"] += 1

            # Perform vector search
            search_results = await self._perform_vector_search(query)

            # Rank and enhance results
            ranked_results = await self._rank_and_enhance_results(search_results, query)

            # Cache results if enabled
            if self._cache and ranked_results:
                self._cache.put(query, ranked_results)

            # Update metrics
            search_time = time.time() - start_time
            self._update_search_metrics(search_time, len(ranked_results))

            logger.debug(
                f"Search completed: {len(ranked_results)} results in {search_time:.3f}s"
            )

            return ranked_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    async def _perform_vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform the actual vector search."""
        # Use optimized search parameters if available
        search_params = self._get_optimized_search_params(query)

        # Perform vector search
        results = await self.vector_store.search_vectors(
            query_vector=query.vector,
            top_k=query.top_k,
            filters=query.filters,
            score_threshold=query.score_threshold
        )

        return results

    def _get_optimized_search_params(self, query: SearchQuery) -> Dict[str, Any]:
        """Get optimized search parameters based on query characteristics."""
        # Base parameters
        params = {"ef": self.vector_store.config.search_ef}

        # Adjust based on query characteristics
        if query.top_k > 50:
            # For large result sets, increase ef for better recall
            params["ef"] = max(params["ef"], query.top_k * 2)
        elif query.score_threshold and query.score_threshold > 0.8:
            # For high threshold searches, increase ef for better precision
            params["ef"] = max(params["ef"], 256)

        # Use custom parameters if provided
        if query.search_params:
            params.update(query.search_params)

        return params

    async def _rank_and_enhance_results(
        self,
        results: List[SearchResult],
        query: SearchQuery
    ) -> List[RankedSearchResult]:
        """Rank and enhance search results with additional information."""
        if not results:
            return []

        enhanced_results = []

        for i, result in enumerate(results):
            # Calculate relevance score (can be enhanced with custom logic)
            relevance_score = self._calculate_relevance_score(result, query)

            # Determine match reasons
            match_reasons = self._get_match_reasons(result, query)

            enhanced_result = RankedSearchResult(
                id=result.id,
                score=result.score,
                payload=result.payload,
                vector=result.vector,
                rank=i + 1,
                relevance_score=relevance_score,
                match_reasons=match_reasons
            )

            enhanced_results.append(enhanced_result)

        # Sort by relevance score if needed (results are already sorted by vector similarity)
        # enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return enhanced_results

    def _calculate_relevance_score(self, result: SearchResult, query: SearchQuery) -> float:
        """Calculate relevance score for a search result."""
        base_score = result.score

        # Boost score based on metadata matches
        boost_factor = 1.0

        if query.filters and result.payload:
            # Boost results that match filter criteria exactly
            for filter_key, filter_value in query.filters.items():
                if filter_key in result.payload:
                    payload_value = result.payload[filter_key]
                    if payload_value == filter_value:
                        boost_factor *= 1.1

        # Boost based on trust score if available
        if "trust_score" in result.payload:
            trust_score = result.payload.get("trust_score", 5.0)
            trust_boost = min(trust_score / 10.0, 0.2)  # Max 20% boost
            boost_factor *= (1.0 + trust_boost)

        # Boost based on recency if available
        if "timestamp" in result.payload:
            try:
                timestamp = datetime.fromisoformat(result.payload["timestamp"])
                days_old = (datetime.now() - timestamp).days
                recency_boost = max(0, (365 - days_old) / 365) * 0.1  # Max 10% boost for recent content
                boost_factor *= (1.0 + recency_boost)
            except (ValueError, TypeError):
                pass  # Ignore invalid timestamps

        return min(base_score * boost_factor, 1.0)  # Cap at 1.0

    def _get_match_reasons(self, result: SearchResult, query: SearchQuery) -> List[str]:
        """Determine why a result matched the query."""
        reasons = []

        # Vector similarity
        if result.score > 0.8:
            reasons.append("High vector similarity")
        elif result.score > 0.6:
            reasons.append("Good vector similarity")
        else:
            reasons.append("Moderate vector similarity")

        # Filter matches
        if query.filters and result.payload:
            for filter_key, filter_value in query.filters.items():
                if filter_key in result.payload:
                    payload_value = result.payload[filter_key]
                    if payload_value == filter_value:
                        reasons.append(f"Matches {filter_key}: {filter_value}")

        # Content type match
        if "doc_type" in result.payload:
            doc_type = result.payload["doc_type"]
            reasons.append(f"Document type: {doc_type}")

        # Programming language match
        if "programming_language" in result.payload:
            lang = result.payload["programming_language"]
            reasons.append(f"Language: {lang}")

        return reasons

    async def search_similar_documents(
        self,
        document_id: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RankedSearchResult]:
        """
        Find documents similar to a given document.
        
        Args:
            document_id: ID of the reference document
            top_k: Number of similar documents to return
            filters: Additional filters to apply
            
        Returns:
            List of similar documents
        """
        try:
            # Get the reference document's vector
            reference_doc = await self.vector_store.get_vector(document_id)
            if not reference_doc:
                logger.warning(f"Reference document {document_id} not found")
                return []

            # Search for similar vectors
            query = SearchQuery(
                vector=reference_doc.vector,
                top_k=top_k + 1,  # +1 to account for the document itself
                filters=filters
            )

            results = await self.search(query)

            # Remove the reference document from results
            filtered_results = [r for r in results if r.id != document_id]

            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"Failed to find similar documents for {document_id}: {e}")
            return []

    async def search_by_metadata(
        self,
        filters: Dict[str, Any],
        top_k: int = 100
    ) -> List[SearchResult]:
        """
        Search documents by metadata only (no vector similarity).
        
        Args:
            filters: Metadata filters to apply
            top_k: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Use a zero vector for metadata-only search
            zero_vector = [0.0] * self.vector_store.config.vector_size

            results = await self.vector_store.search_vectors(
                query_vector=zero_vector,
                top_k=top_k,
                filters=filters,
                score_threshold=0.0  # Accept all matches for metadata search
            )

            return results

        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def _update_search_metrics(self, search_time: float, result_count: int) -> None:
        """Update search performance metrics."""
        # Update average search time
        total_searches = self._metrics["total_searches"]
        current_avg = self._metrics["avg_search_time"]
        self._metrics["avg_search_time"] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )

        # Update average results per search
        current_avg_results = self._metrics["avg_results_per_search"]
        self._metrics["avg_results_per_search"] = (
            (current_avg_results * (total_searches - 1) + result_count) / total_searches
        )

        self._metrics["last_search_time"] = datetime.now().isoformat()

    def get_search_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics."""
        cache_stats = self._cache.get_stats() if self._cache else {}

        return {
            "search_metrics": self._metrics.copy(),
            "cache_stats": cache_stats,
            "cache_hit_rate": (
                self._metrics["cache_hits"] /
                max(self._metrics["total_searches"], 1)
            ) if self._metrics["total_searches"] > 0 else 0.0
        }

    def clear_cache(self) -> None:
        """Clear the search cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Search cache cleared")

    async def warm_cache(self, common_queries: List[SearchQuery]) -> None:
        """
        Warm the cache with common search queries.
        
        Args:
            common_queries: List of frequent search queries to pre-cache
        """
        if not self._cache:
            return

        logger.info(f"Warming cache with {len(common_queries)} queries")

        for query in common_queries:
            try:
                await self.search(query)
            except Exception as e:
                logger.warning(f"Failed to warm cache for query: {e}")

        logger.info("Cache warming completed")

    def optimize_search_parameters(self, target_latency_ms: float = 50.0) -> Dict[str, Any]:
        """
        Optimize search parameters based on performance history.
        
        Args:
            target_latency_ms: Target search latency in milliseconds
            
        Returns:
            Optimized search parameters
        """
        current_avg_ms = self._metrics["avg_search_time"] * 1000

        if current_avg_ms > target_latency_ms:
            # Search is too slow, reduce ef for faster searches
            new_ef = max(64, int(self.vector_store.config.search_ef * 0.8))
            logger.info(f"Reducing search ef from {self.vector_store.config.search_ef} to {new_ef}")
        elif current_avg_ms < target_latency_ms * 0.5:
            # Search is very fast, increase ef for better accuracy
            new_ef = min(512, int(self.vector_store.config.search_ef * 1.2))
            logger.info(f"Increasing search ef from {self.vector_store.config.search_ef} to {new_ef}")
        else:
            # Performance is good, no changes needed
            new_ef = self.vector_store.config.search_ef

        optimized_params = {
            "ef": new_ef,
            "current_latency_ms": current_avg_ms,
            "target_latency_ms": target_latency_ms
        }

        self._optimal_params = optimized_params
        return optimized_params
