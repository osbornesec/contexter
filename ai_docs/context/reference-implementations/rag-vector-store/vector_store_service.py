"""
Complete RAG Vector Store Service Implementation

This module provides a complete, production-ready vector store service that integrates
Qdrant vector database with Contexter's existing patterns and architecture.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib

import numpy as np
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, HnswConfig,
    PointStruct, Filter, FieldCondition, MatchValue, Range,
    SearchResult, UpdateResult, OptimizersConfig, ScalarQuantization,
    ScalarType
)
from qdrant_client.http import models as rest

# Import Contexter patterns
from contexter.core.config_manager import ConfigManager
from contexter.core.error_classifier import ErrorClassifier
from contexter.core.storage_manager import LocalStorageManager
from contexter.models.storage_models import StorageResult, DocumentationChunk

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Vector document with metadata for storage."""
    
    id: str
    vector: List[float]
    content: str
    library_id: str
    library_name: str
    version: str
    doc_type: str
    section: str
    subsection: Optional[str] = None
    programming_language: Optional[str] = None
    trust_score: float = 0.0
    star_count: int = 0
    token_count: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    embedding_model: str = "voyage-code-3"
    content_hash: str = field(default="")
    indexed_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Generate content hash if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
    
    def to_point_struct(self) -> PointStruct:
        """Convert to Qdrant PointStruct."""
        return PointStruct(
            id=self.id,
            vector=self.vector,
            payload={
                "library_id": self.library_id,
                "library_name": self.library_name,
                "version": self.version,
                "doc_type": self.doc_type,
                "section": self.section,
                "subsection": self.subsection,
                "content": self.content,
                "programming_language": self.programming_language,
                "trust_score": self.trust_score,
                "star_count": self.star_count,
                "token_count": self.token_count,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "embedding_model": self.embedding_model,
                "content_hash": self.content_hash,
                "indexed_at": self.indexed_at.isoformat()
            }
        )


@dataclass
class SearchQuery:
    """Vector search query with filters and options."""
    
    query_vector: List[float]
    limit: int = 10
    score_threshold: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None
    ef: Optional[int] = None
    include_vectors: bool = False
    include_payload: bool = True


@dataclass
class SearchResult:
    """Vector search result."""
    
    id: str
    score: float
    content: str
    library_id: str
    library_name: str
    section: str
    doc_type: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class VectorStoreStats:
    """Vector store performance statistics."""
    
    total_vectors: int = 0
    total_searches: int = 0
    total_inserts: int = 0
    total_errors: int = 0
    avg_search_latency_ms: float = 0.0
    avg_insert_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class VectorStoreService:
    """Complete vector store service with caching, monitoring, and optimization."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        storage_manager: Optional[LocalStorageManager] = None
    ):
        self.config = config_manager.qdrant_config
        self.storage_manager = storage_manager
        
        # Qdrant client
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_initialized = False
        
        # Performance tracking
        self.stats = VectorStoreStats()
        self._search_times: List[float] = []
        self._insert_times: List[float] = []
        
        # Caching for frequently accessed data
        self._search_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._max_cache_size = 1000
        
        # Background maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def initialize(self):
        """Initialize vector store service."""
        
        logger.info("Initializing Vector Store Service")
        
        # Initialize Qdrant client
        await self._initialize_qdrant_client()
        
        # Ensure collection exists
        await self._ensure_collection_exists()
        
        # Start background maintenance
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        logger.info("Vector Store Service initialized successfully")
    
    async def shutdown(self):
        """Shutdown vector store service."""
        
        logger.info("Shutting down Vector Store Service")
        
        # Stop maintenance loop
        self._shutdown_event.set()
        if self._maintenance_task:
            await self._maintenance_task
        
        # Close Qdrant client
        if self.client:
            await self.client.close()
        
        logger.info("Vector Store Service shutdown complete")
    
    async def _initialize_qdrant_client(self):
        """Initialize Qdrant client with optimal configuration."""
        
        try:
            if self.config.api_key:
                # Cloud configuration
                self.client = AsyncQdrantClient(
                    url=f"https://{self.config.host}:{self.config.port}",
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            else:
                # Local configuration
                self.client = AsyncQdrantClient(
                    host=self.config.host,
                    port=self.config.port,
                    grpc_port=self.config.grpc_port,
                    prefer_grpc=self.config.prefer_grpc,
                    timeout=self.config.timeout
                )
            
            # Test connection
            await self.client.get_collections()
            logger.info(f"Qdrant client connected to {self.config.host}:{self.config.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    async def _ensure_collection_exists(self):
        """Ensure collection exists with optimal configuration."""
        
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name not in collection_names:
                await self._create_optimized_collection()
            else:
                await self._validate_collection_config()
            
            self.collection_initialized = True
            
        except Exception as e:
            logger.error(f"Collection initialization failed: {e}")
            raise
    
    async def _create_optimized_collection(self):
        """Create collection with production-optimized configuration."""
        
        logger.info(f"Creating optimized collection: {self.config.collection_name}")
        
        # HNSW configuration optimized for 2048-dimensional vectors
        hnsw_config = HnswConfig(
            m=self.config.hnsw_m,
            ef_construct=self.config.hnsw_ef_construct,
            full_scan_threshold=self.config.hnsw_full_scan_threshold,
            max_indexing_threads=0,  # Use all cores
            on_disk=True  # Store index on disk for memory efficiency
        )
        
        # Optimizers configuration for performance
        optimizers_config = OptimizersConfig(
            deleted_threshold=0.2,  # Clean up when 20% deleted
            vacuum_min_vector_number=1000,
            default_segment_number=16,  # Good for concurrent access
            max_segment_size_kb=200_000,  # 200MB segments
            memmap_threshold_kb=100_000,  # Use memory mapping for large segments
            indexing_threshold_kb=100_000,
            flush_interval_sec=30,
            max_optimization_threads=4
        )
        
        # Vector configuration
        vectors_config = VectorParams(
            size=self.config.vector_size,
            distance=Distance.COSINE,
            hnsw_config=hnsw_config,
            quantization_config=ScalarQuantization(
                type=ScalarType.INT8,
                quantile=0.99,
                always_ram=True  # Keep quantized vectors in RAM
            )
        )
        
        # Create collection
        await self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config,
            optimizers_config=optimizers_config
        )
        
        # Create payload indexes for efficient filtering
        await self._create_payload_indexes()
        
        logger.info(f"Collection {self.config.collection_name} created successfully")
    
    async def _create_payload_indexes(self):
        """Create indexes for efficient payload filtering."""
        
        indexes = [
            ("library_id", rest.PayloadSchemaType.KEYWORD),
            ("doc_type", rest.PayloadSchemaType.KEYWORD),
            ("section", rest.PayloadSchemaType.KEYWORD),
            ("programming_language", rest.PayloadSchemaType.KEYWORD),
            ("trust_score", rest.PayloadSchemaType.FLOAT),
            ("star_count", rest.PayloadSchemaType.INTEGER),
            ("indexed_at", rest.PayloadSchemaType.DATETIME),
            ("content", rest.PayloadSchemaType.TEXT)  # Full-text search
        ]
        
        for field_name, field_type in indexes:
            try:
                await self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_type=field_type
                )
                logger.debug(f"Created index for field: {field_name}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation for {field_name}: {e}")
    
    async def _validate_collection_config(self):
        """Validate existing collection configuration."""
        
        collection_info = await self.client.get_collection(self.config.collection_name)
        
        # Validate vector size
        config_size = collection_info.config.params.vectors.size
        if config_size != self.config.vector_size:
            raise ValueError(
                f"Collection vector size mismatch: expected {self.config.vector_size}, "
                f"got {config_size}"
            )
        
        logger.info("Collection configuration validated")
    
    async def add_documents(
        self,
        documents: List[VectorDocument],
        batch_size: int = 1000,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Add vector documents to the collection."""
        
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        start_time = time.time()
        total_docs = len(documents)
        successful_adds = 0
        errors = []
        
        try:
            # Process documents in batches
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                batch_start = time.time()
                
                try:
                    # Convert to PointStruct objects
                    points = [doc.to_point_struct() for doc in batch]
                    
                    # Upsert batch with wait for consistency
                    result = await self.client.upsert(
                        collection_name=self.config.collection_name,
                        points=points,
                        wait=True
                    )
                    
                    successful_adds += len(batch)
                    
                    # Track batch performance
                    batch_time = (time.time() - batch_start) * 1000
                    self._insert_times.append(batch_time)
                    
                    logger.debug(
                        f"Added batch {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}: "
                        f"{len(batch)} documents in {batch_time:.2f}ms"
                    )
                    
                except Exception as e:
                    error_msg = f"Batch {i//batch_size + 1} failed: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                
                # Progress callback
                if progress_callback:
                    await progress_callback(min(i + batch_size, total_docs), total_docs)
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats.total_inserts += successful_adds
            
            if self._insert_times:
                self.stats.avg_insert_latency_ms = (
                    sum(self._insert_times) / len(self._insert_times)
                )
            
            return {
                "success": True,
                "total_documents": total_docs,
                "successful_adds": successful_adds,
                "failed_adds": total_docs - successful_adds,
                "errors": errors,
                "processing_time_seconds": total_time,
                "throughput_docs_per_second": successful_adds / total_time if total_time > 0 else 0
            }
            
        except Exception as e:
            self.stats.total_errors += 1
            logger.error(f"Document addition failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "total_documents": total_docs,
                "successful_adds": successful_adds
            }
    
    async def search(
        self,
        query: SearchQuery
    ) -> List[SearchResult]:
        """Search for similar vectors with caching."""
        
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        # Check cache first
        cache_key = self._generate_search_cache_key(query)
        cached_result = self._get_cached_search(cache_key)
        
        if cached_result:
            self.stats.cache_hit_rate = (
                (self.stats.cache_hit_rate * self.stats.total_searches + 1) /
                (self.stats.total_searches + 1)
            )
            self.stats.total_searches += 1
            return cached_result
        
        start_time = time.time()
        
        try:
            # Build query filter
            query_filter = self._build_query_filter(query.filters) if query.filters else None
            
            # Configure search parameters
            search_params = rest.SearchParams(
                hnsw_ef=query.ef or self.config.hnsw_ef,
                exact=False  # Use approximate search for speed
            )
            
            # Perform search
            search_results = await self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query.query_vector,
                query_filter=query_filter,
                limit=query.limit,
                score_threshold=query.score_threshold,
                search_params=search_params,
                with_payload=query.include_payload,
                with_vector=query.include_vectors
            )
            
            # Format results
            formatted_results = []
            for hit in search_results:
                result = SearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    content=hit.payload.get("content", "") if hit.payload else "",
                    library_id=hit.payload.get("library_id", "") if hit.payload else "",
                    library_name=hit.payload.get("library_name", "") if hit.payload else "",
                    section=hit.payload.get("section", "") if hit.payload else "",
                    doc_type=hit.payload.get("doc_type", "") if hit.payload else "",
                    metadata=hit.payload if hit.payload else {},
                    vector=hit.vector if query.include_vectors else None
                )
                formatted_results.append(result)
            
            # Update performance statistics
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            self._search_times.append(search_time)
            self.stats.total_searches += 1
            
            if self._search_times:
                self.stats.avg_search_latency_ms = (
                    sum(self._search_times) / len(self._search_times)
                )
            
            # Cache results
            self._cache_search_results(cache_key, formatted_results)
            
            logger.debug(
                f"Search completed in {search_time:.2f}ms, "
                f"found {len(formatted_results)} results"
            )
            
            return formatted_results
            
        except Exception as e:
            self.stats.total_errors += 1
            error_info = ErrorClassifier.classify_error(e)
            
            logger.error(
                f"Vector search failed: {error_info.category} - {e}",
                extra={
                    "query_size": len(query.query_vector),
                    "limit": query.limit,
                    "filters": query.filters
                }
            )
            raise
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        text_query: Optional[str] = None,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector similarity and text search."""
        
        results = []
        
        # Vector similarity search
        vector_query = SearchQuery(
            query_vector=query_vector,
            limit=limit * 2,  # Get more results for reranking
            filters=filters,
            include_payload=True
        )
        
        vector_results = await self.search(vector_query)
        
        # If no text query, return vector results
        if not text_query:
            return vector_results[:limit]
        
        # Text search using payload filter
        text_filters = dict(filters) if filters else {}
        text_filters["content"] = text_query  # This would require text search implementation
        
        # For now, we'll use vector results and rerank based on text similarity
        # In a full implementation, you'd integrate with Qdrant's text search capabilities
        
        # Simple text matching for demonstration
        reranked_results = []
        for result in vector_results:
            text_score = self._calculate_text_similarity(text_query, result.content)
            combined_score = (vector_weight * result.score + text_weight * text_score)
            
            result.score = combined_score
            reranked_results.append(result)
        
        # Sort by combined score and return top results
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:limit]
    
    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """Calculate simple text similarity score."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        return len(intersection) / len(query_words)
    
    def _build_query_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary."""
        
        conditions = []
        
        # Exact match conditions
        exact_match_fields = [
            "library_id", "doc_type", "section", "programming_language", "version"
        ]
        
        for field in exact_match_fields:
            if field in filters:
                if isinstance(filters[field], list):
                    # Multiple values - use MatchAny
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=rest.MatchAny(any=filters[field])
                        )
                    )
                else:
                    # Single value - use MatchValue
                    conditions.append(
                        FieldCondition(
                            key=field,
                            match=MatchValue(value=filters[field])
                        )
                    )
        
        # Range conditions
        if "min_trust_score" in filters:
            conditions.append(
                FieldCondition(
                    key="trust_score",
                    range=Range(gte=filters["min_trust_score"])
                )
            )
        
        if "max_trust_score" in filters:
            conditions.append(
                FieldCondition(
                    key="trust_score",
                    range=Range(lte=filters["max_trust_score"])
                )
            )
        
        if "min_star_count" in filters:
            conditions.append(
                FieldCondition(
                    key="star_count",
                    range=Range(gte=filters["min_star_count"])
                )
            )
        
        # Date range conditions
        if "indexed_after" in filters:
            conditions.append(
                FieldCondition(
                    key="indexed_at",
                    range=Range(gte=filters["indexed_after"])
                )
            )
        
        if "indexed_before" in filters:
            conditions.append(
                FieldCondition(
                    key="indexed_at",
                    range=Range(lte=filters["indexed_before"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    def _generate_search_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for search query."""
        
        # Create deterministic hash from query parameters
        cache_data = {
            "vector_hash": hashlib.sha256(
                json.dumps(query.query_vector, sort_keys=True).encode()
            ).hexdigest()[:16],
            "limit": query.limit,
            "score_threshold": query.score_threshold,
            "filters": sorted(query.filters.items()) if query.filters else None,
            "ef": query.ef,
            "include_vectors": query.include_vectors
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def _get_cached_search(self, cache_key: str) -> Optional[List[SearchResult]]:
        """Retrieve cached search results if not expired."""
        
        if cache_key not in self._search_cache:
            return None
        
        cached_entry = self._search_cache[cache_key]
        
        # Check if cache entry is still valid
        if (datetime.utcnow().timestamp() - cached_entry["timestamp"]) > self._cache_ttl:
            del self._search_cache[cache_key]
            return None
        
        return cached_entry["results"]
    
    def _cache_search_results(self, cache_key: str, results: List[SearchResult]):
        """Cache search results with TTL."""
        
        # Remove old entries if cache is full
        if len(self._search_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = min(
                self._search_cache.keys(),
                key=lambda k: self._search_cache[k]["timestamp"]
            )
            del self._search_cache[oldest_key]
        
        # Add new cache entry
        self._search_cache[cache_key] = {
            "results": results,
            "timestamp": datetime.utcnow().timestamp()
        }
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get comprehensive collection information."""
        
        try:
            collection_info = await self.client.get_collection(self.config.collection_name)
            
            return {
                "name": self.config.collection_name,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "disk_data_size": collection_info.disk_data_size,
                "ram_data_size": collection_info.ram_data_size,
                "config": {
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.name,
                    "hnsw_config": {
                        "m": collection_info.config.params.vectors.hnsw_config.m,
                        "ef_construct": collection_info.config.params.vectors.hnsw_config.ef_construct,
                        "full_scan_threshold": collection_info.config.params.vectors.hnsw_config.full_scan_threshold
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        
        try:
            # Basic connectivity test
            telemetry = await self.client.get_telemetry()
            
            # Collection information
            collection_info = await self.get_collection_info()
            
            # Performance test with dummy search
            dummy_vector = np.random.random(self.config.vector_size).tolist()
            
            start_time = time.time()
            test_query = SearchQuery(query_vector=dummy_vector, limit=1)
            test_results = await self.search(test_query)
            test_latency = (time.time() - start_time) * 1000
            
            # Calculate cache statistics
            cache_stats = {
                "cache_size": len(self._search_cache),
                "cache_hit_rate": self.stats.cache_hit_rate,
                "max_cache_size": self._max_cache_size,
                "cache_ttl_seconds": self._cache_ttl
            }
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "qdrant_version": telemetry.get("app", {}).get("version", "unknown"),
                "collection": collection_info,
                "performance": {
                    "test_search_latency_ms": round(test_latency, 2),
                    "avg_search_latency_ms": round(self.stats.avg_search_latency_ms, 2),
                    "avg_insert_latency_ms": round(self.stats.avg_insert_latency_ms, 2),
                    "total_searches": self.stats.total_searches,
                    "total_inserts": self.stats.total_inserts,
                    "total_errors": self.stats.total_errors
                },
                "cache": cache_stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "performance": {
                    "total_searches": self.stats.total_searches,
                    "total_inserts": self.stats.total_inserts,
                    "total_errors": self.stats.total_errors
                }
            }
    
    async def optimize_collection(self) -> Dict[str, Any]:
        """Optimize collection for better performance."""
        
        logger.info("Starting collection optimization")
        
        try:
            # Update collection with optimized settings
            optimizers_config = OptimizersConfig(
                deleted_threshold=0.1,  # More aggressive cleanup
                vacuum_min_vector_number=1000,
                default_segment_number=8,  # Optimize for current data size
                max_optimization_threads=2
            )
            
            # Apply optimization
            await self.client.update_collection(
                collection_name=self.config.collection_name,
                optimizers_config=optimizers_config
            )
            
            logger.info("Collection optimization completed")
            
            return {
                "success": True,
                "message": "Collection optimization completed",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Collection optimization failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _maintenance_loop(self):
        """Background maintenance tasks."""
        
        logger.info("Starting maintenance loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Run maintenance every 30 minutes
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=1800  # 30 minutes
                )
            except asyncio.TimeoutError:
                # Timeout is expected, continue with maintenance
                pass
            
            if self._shutdown_event.is_set():
                break
            
            try:
                # Cache cleanup
                await self._cleanup_expired_cache()
                
                # Collection health check
                health = await self.health_check()
                
                if health["status"] == "healthy":
                    collection_info = health.get("collection", {})
                    vector_count = collection_info.get("vectors_count", 0)
                    
                    # Auto-optimize if collection is large and segments are fragmented
                    if (vector_count > 100000 and 
                        collection_info.get("segments_count", 0) > 32):
                        
                        logger.info("Auto-optimizing collection due to fragmentation")
                        await self.optimize_collection()
                
                # Update statistics
                self.stats.last_updated = datetime.utcnow()
                self.stats.total_vectors = collection_info.get("vectors_count", 0)
                
                logger.debug("Maintenance tasks completed")
                
            except Exception as e:
                logger.error(f"Maintenance task failed: {e}")
        
        logger.info("Maintenance loop ended")
    
    async def _cleanup_expired_cache(self):
        """Clean up expired cache entries."""
        
        current_time = datetime.utcnow().timestamp()
        expired_keys = []
        
        for key, entry in self._search_cache.items():
            if (current_time - entry["timestamp"]) > self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._search_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> VectorStoreStats:
        """Get current performance statistics."""
        return self.stats


# Usage example and integration with Contexter
async def example_usage():
    """Example of using the VectorStoreService."""
    
    # Initialize with Contexter configuration
    config_manager = ConfigManager("config/contexter_config.yaml")
    storage_manager = LocalStorageManager()
    
    async with VectorStoreService(config_manager, storage_manager) as vector_store:
        
        # Example: Add documents to vector store
        documents = [
            VectorDocument(
                id="doc_1",
                vector=np.random.random(2048).tolist(),
                content="This is example content about FastAPI",
                library_id="fastapi/fastapi",
                library_name="FastAPI",
                version="0.104.1",
                doc_type="guide",
                section="Getting Started",
                programming_language="python",
                trust_score=9.8,
                star_count=84000
            ),
            VectorDocument(
                id="doc_2",
                vector=np.random.random(2048).tolist(),
                content="Advanced FastAPI features and patterns",
                library_id="fastapi/fastapi",
                library_name="FastAPI",
                version="0.104.1",
                doc_type="advanced",
                section="Advanced Features",
                programming_language="python",
                trust_score=9.8,
                star_count=84000
            )
        ]
        
        # Add documents
        add_result = await vector_store.add_documents(documents)
        print(f"Added {add_result['successful_adds']} documents")
        
        # Search for similar content
        query_vector = np.random.random(2048).tolist()
        search_query = SearchQuery(
            query_vector=query_vector,
            limit=10,
            filters={"library_id": "fastapi/fastapi"},
            score_threshold=0.7
        )
        
        results = await vector_store.search(search_query)
        print(f"Found {len(results)} similar documents")
        
        # Hybrid search
        hybrid_results = await vector_store.hybrid_search(
            query_vector=query_vector,
            text_query="FastAPI features",
            limit=5,
            filters={"programming_language": "python"}
        )
        print(f"Hybrid search found {len(hybrid_results)} documents")
        
        # Get collection information
        collection_info = await vector_store.get_collection_info()
        print(f"Collection has {collection_info['vectors_count']} vectors")
        
        # Health check
        health = await vector_store.health_check()
        print(f"Vector store status: {health['status']}")
        
        # Get performance statistics
        stats = vector_store.get_stats()
        print(f"Total searches: {stats.total_searches}")
        print(f"Average search latency: {stats.avg_search_latency_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(example_usage())