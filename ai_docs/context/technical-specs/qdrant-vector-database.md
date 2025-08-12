# Qdrant Vector Database Technical Specifications

## Overview

This document provides comprehensive technical specifications for integrating Qdrant vector database into the Contexter RAG system. It covers architecture patterns, performance optimization, HNSW configuration, and integration with the existing Contexter storage system.

**Key Requirements**:
- Sub-50ms p95 search latency
- Support for 10M+ vectors
- 2048-dimensional vectors (Voyage AI voyage-code-3)
- Batch operations with 1000+ vectors
- 99.9% availability

## Architecture Integration

### Contexter Integration Pattern
```python
# Integration with existing Contexter patterns
from contexter.core.storage_manager import LocalStorageManager
from contexter.models.storage_models import DocumentationChunk

class QdrantVectorStore:
    """Vector store integrated with Contexter storage patterns."""
    
    def __init__(self, storage_manager: LocalStorageManager):
        self.storage_manager = storage_manager
        self.client = None
        self.collection_name = "contexter_documentation"
```

### Collection Schema Design
```yaml
# Optimized for Contexter documentation structure
collection_config:
  name: "contexter_documentation"
  vectors:
    size: 2048
    distance: cosine
  hnsw_config:
    m: 16                    # Balanced performance for 2048-dim vectors
    ef_construct: 200        # Build time vs accuracy trade-off
    ef: 128                  # Query time parameter
    full_scan_threshold: 10000
  optimizers_config:
    deleted_threshold: 0.2   # Cleanup threshold
    vacuum_min_vector_number: 1000
  quantization:
    scalar:
      type: int8
      quantile: 0.99
      always_ram: true       # Keep quantized vectors in RAM
```

### Payload Schema for Contexter Documents
```python
@dataclass
class VectorDocumentPayload:
    """Payload structure for Contexter documentation vectors."""
    
    # Core identification
    library_id: str          # e.g., "fastapi/fastapi"
    library_name: str        # e.g., "FastAPI"
    version: str            # e.g., "0.104.1"
    
    # Document structure
    doc_type: str           # api, guide, tutorial, reference
    section: str            # Main section identifier
    subsection: str         # Subsection identifier
    chunk_index: int        # Position in document
    total_chunks: int       # Total chunks in document
    
    # Content metadata
    content: str            # Original text content
    token_count: int        # Token count for chunking
    programming_language: str # Primary language if applicable
    
    # Quality indicators
    trust_score: float      # 0.0-10.0 from ContextS
    star_count: int         # GitHub stars if applicable
    
    # Technical metadata
    embedding_model: str    # "voyage-code-3"
    indexed_at: datetime
    content_hash: str       # For deduplication
    source_url: str         # Original documentation URL
```

## Performance Configuration

### HNSW Parameter Optimization

```python
# Production-optimized HNSW settings for 2048-dimensional vectors
HNSW_CONFIG = {
    "m": 16,                 # 16 bi-directional links per node
    "ef_construct": 200,     # Construction parameter for accuracy
    "ef": 128,              # Query-time search parameter
    "full_scan_threshold": 10000,  # Use HNSW above this count
    "max_indexing_threads": 0,     # Use all available cores
}

# Dynamic ef adjustment based on query requirements
def get_query_ef(precision_requirement: str) -> int:
    """Adjust ef parameter based on precision needs."""
    precision_map = {
        "fast": 64,      # Lower precision, faster queries
        "balanced": 128,  # Default balanced setting
        "precise": 256,   # Higher precision, slower queries
    }
    return precision_map.get(precision_requirement, 128)
```

### Memory Usage Optimization

```python
class QdrantMemoryManager:
    """Memory usage optimization for large collections."""
    
    def __init__(self):
        self.memory_limits = {
            "vectors_in_ram": True,     # Keep vectors in RAM
            "payload_in_ram": False,    # Keep payload on disk
            "quantization_enabled": True, # Use scalar quantization
        }
    
    def estimate_memory_usage(self, vector_count: int) -> dict:
        """Estimate memory usage for planning."""
        vector_size_bytes = 2048 * 4  # float32
        quantized_size_bytes = 2048 * 1  # int8 quantization
        
        return {
            "full_precision_gb": (vector_count * vector_size_bytes) / (1024**3),
            "quantized_gb": (vector_count * quantized_size_bytes) / (1024**3),
            "hnsw_index_gb": (vector_count * 64) / (1024**3),  # Approximate
        }
```

### Batch Operation Optimization

```python
class BatchVectorUploader:
    """Optimized batch uploading with memory management."""
    
    def __init__(self, client: QdrantClient, batch_size: int = 1000):
        self.client = client
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent batches
    
    async def upsert_vectors_batch(
        self, 
        vectors: List[VectorDocument], 
        progress_callback: Optional[Callable] = None
    ) -> BatchUploadResult:
        """Upload vectors in optimized batches."""
        results = BatchUploadResult()
        
        for i in range(0, len(vectors), self.batch_size):
            batch = vectors[i:i + self.batch_size]
            
            async with self.semaphore:
                try:
                    points = self._create_points_batch(batch)
                    
                    # Use upsert for automatic deduplication
                    operation_info = await self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True  # Wait for indexing
                    )
                    
                    results.successful_uploads += len(batch)
                    if progress_callback:
                        await progress_callback(i + len(batch), len(vectors))
                        
                except Exception as e:
                    results.failed_uploads += len(batch)
                    results.errors.append(f"Batch {i}-{i+len(batch)}: {e}")
                    logger.error(f"Batch upload failed: {e}")
        
        return results
```

## Search Optimization Patterns

### Multi-Modal Search Implementation

```python
class ContexterSearchEngine:
    """Optimized search engine for Contexter documentation."""
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        text_query: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        precision: str = "balanced"
    ) -> List[SearchResult]:
        """Hybrid semantic + keyword search."""
        
        # Build filter conditions
        query_filter = self._build_filter(filters) if filters else None
        
        # Perform vector similarity search
        search_params = {"ef": get_query_ef(precision)}
        
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k * 2,  # Get more results for reranking
            search_params=search_params,
            with_payload=True,
            with_vectors=False  # Don't return vectors to save bandwidth
        )
        
        # Apply reranking if text query provided
        if text_query:
            results = await self._rerank_results(results, text_query)
        
        return self._format_search_results(results[:top_k])

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from search parameters."""
        conditions = []
        
        # Library-specific filtering
        if "library_id" in filters:
            conditions.append(
                FieldCondition(
                    key="library_id",
                    match=MatchValue(value=filters["library_id"])
                )
            )
        
        # Document type filtering
        if "doc_type" in filters:
            conditions.append(
                FieldCondition(
                    key="doc_type",
                    match=MatchAny(any=filters["doc_type"])
                )
            )
        
        # Programming language filtering
        if "programming_language" in filters:
            conditions.append(
                FieldCondition(
                    key="programming_language",
                    match=MatchValue(value=filters["programming_language"])
                )
            )
        
        # Quality score threshold
        if "min_trust_score" in filters:
            conditions.append(
                FieldCondition(
                    key="trust_score",
                    range=Range(gte=filters["min_trust_score"])
                )
            )
        
        # Date range filtering
        if "indexed_after" in filters:
            conditions.append(
                FieldCondition(
                    key="indexed_at",
                    range=Range(gte=filters["indexed_after"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
```

### Caching and Performance Optimization

```python
class QdrantSearchCache:
    """Redis-backed search result caching."""
    
    def __init__(self, redis_client, ttl_seconds: int = 300):
        self.redis = redis_client
        self.ttl = ttl_seconds
    
    async def get_cached_results(
        self, 
        query_hash: str
    ) -> Optional[List[SearchResult]]:
        """Retrieve cached search results."""
        try:
            cached = await self.redis.get(f"search:{query_hash}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    async def cache_results(
        self, 
        query_hash: str, 
        results: List[SearchResult]
    ):
        """Cache search results with TTL."""
        try:
            await self.redis.setex(
                f"search:{query_hash}",
                self.ttl,
                json.dumps([r.dict() for r in results])
            )
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")

    def generate_query_hash(
        self,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> str:
        """Generate deterministic hash for query caching."""
        import hashlib
        
        # Create stable representation
        query_data = {
            "vector": query_vector,
            "filters": sorted(filters.items()) if filters else None,
            "top_k": top_k
        }
        
        query_str = json.dumps(query_data, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]
```

## Health Monitoring and Maintenance

### Collection Health Monitoring

```python
class QdrantHealthMonitor:
    """Health monitoring for Qdrant collections."""
    
    async def check_collection_health(self) -> HealthReport:
        """Comprehensive collection health check."""
        report = HealthReport()
        
        try:
            # Basic connectivity
            cluster_info = await self.client.get_cluster()
            report.connectivity = cluster_info is not None
            
            # Collection statistics
            collection_info = await self.client.get_collection(self.collection_name)
            report.vector_count = collection_info.points_count
            report.index_status = collection_info.status
            
            # Performance metrics
            sample_query_latency = await self._measure_query_latency()
            report.average_latency_ms = sample_query_latency
            
            # Memory usage
            report.memory_usage = await self._get_memory_usage()
            
            # Overall health
            report.is_healthy = (
                report.connectivity and
                report.average_latency_ms < 100 and  # 100ms threshold
                report.memory_usage.get("used_percent", 0) < 90
            )
            
        except Exception as e:
            report.is_healthy = False
            report.error_message = str(e)
            logger.error(f"Health check failed: {e}")
        
        return report
    
    async def _measure_query_latency(self) -> float:
        """Measure average query latency with sample queries."""
        import time
        import numpy as np
        
        sample_vector = np.random.random(2048).tolist()
        latencies = []
        
        for _ in range(5):  # Sample 5 queries
            start_time = time.time()
            
            await self.client.search(
                collection_name=self.collection_name,
                query_vector=sample_vector,
                limit=10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        return sum(latencies) / len(latencies)
```

### Automated Maintenance Procedures

```python
class QdrantMaintenanceManager:
    """Automated maintenance for optimal performance."""
    
    async def optimize_collection(
        self, 
        force: bool = False
    ) -> MaintenanceResult:
        """Optimize collection index for better performance."""
        
        collection_info = await self.client.get_collection(self.collection_name)
        
        # Check if optimization is needed
        if not force and not self._needs_optimization(collection_info):
            return MaintenanceResult(
                action="optimize",
                skipped=True,
                reason="Optimization not needed"
            )
        
        try:
            # Trigger index optimization
            await self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=OptimizersConfig(
                    deleted_threshold=0.1,  # More aggressive cleanup
                    vacuum_min_vector_number=1000,
                    default_segment_number=16
                )
            )
            
            return MaintenanceResult(
                action="optimize",
                success=True,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            return MaintenanceResult(
                action="optimize",
                success=False,
                error=str(e)
            )
    
    def _needs_optimization(self, collection_info) -> bool:
        """Determine if collection needs optimization."""
        # Check various metrics
        deleted_ratio = collection_info.segments_count / collection_info.points_count
        return (
            deleted_ratio > 0.2 or  # High deletion ratio
            collection_info.segments_count > 32 or  # Too many segments
            collection_info.status != "green"  # Health issues
        )
```

## Error Handling and Recovery

### Comprehensive Error Classification

```python
class QdrantErrorHandler:
    """Comprehensive error handling for Qdrant operations."""
    
    async def handle_operation_error(
        self, 
        operation: str, 
        error: Exception
    ) -> ErrorRecoveryAction:
        """Classify error and determine recovery action."""
        
        if isinstance(error, httpx.ConnectError):
            return ErrorRecoveryAction(
                category="connectivity",
                severity="critical",
                action="retry_with_backoff",
                max_retries=3,
                backoff_seconds=5
            )
        
        elif isinstance(error, httpx.TimeoutException):
            return ErrorRecoveryAction(
                category="timeout",
                severity="warning",
                action="retry_with_timeout_increase",
                max_retries=2,
                timeout_multiplier=2.0
            )
        
        elif "collection not found" in str(error).lower():
            return ErrorRecoveryAction(
                category="configuration",
                severity="critical",
                action="recreate_collection",
                auto_recover=True
            )
        
        elif "out of memory" in str(error).lower():
            return ErrorRecoveryAction(
                category="resource",
                severity="critical",
                action="enable_quantization",
                auto_recover=False  # Requires manual intervention
            )
        
        else:
            return ErrorRecoveryAction(
                category="unknown",
                severity="error",
                action="log_and_fail",
                auto_recover=False
            )
```

## Integration with Existing Contexter Patterns

### Storage Integration

```python
class ContexterQdrantIntegration:
    """Integration layer between Contexter storage and Qdrant."""
    
    def __init__(
        self, 
        storage_manager: LocalStorageManager,
        vector_store: QdrantVectorStore
    ):
        self.storage = storage_manager
        self.vector_store = vector_store
    
    async def sync_documentation_to_vectors(
        self, 
        library_id: str, 
        version: str
    ) -> SyncResult:
        """Sync Contexter documentation to vector database."""
        
        # Load documentation from Contexter storage
        doc_chunks = await self.storage.load_documentation_chunks(
            library_id, 
            version
        )
        
        if not doc_chunks:
            return SyncResult(success=False, reason="No documentation found")
        
        # Convert to vector documents
        vector_docs = []
        for chunk in doc_chunks:
            vector_doc = VectorDocument(
                id=self._generate_vector_id(chunk),
                vector=await self._get_or_generate_embedding(chunk.content),
                payload=self._create_payload_from_chunk(chunk, library_id, version)
            )
            vector_docs.append(vector_doc)
        
        # Batch upload to Qdrant
        result = await self.vector_store.upsert_vectors_batch(vector_docs)
        
        return SyncResult(
            success=result.successful_uploads > 0,
            vectors_uploaded=result.successful_uploads,
            errors=result.errors
        )
```

## Performance Benchmarks and SLA Targets

### Target Performance Metrics

```yaml
performance_targets:
  search_latency:
    p50: 25ms
    p95: 50ms
    p99: 100ms
  
  throughput:
    concurrent_searches: 100+
    batch_upload_rate: 10000 vectors/minute
  
  availability:
    uptime: 99.9%
    max_downtime: 8.76 hours/year
  
  scalability:
    max_vectors: 10_000_000
    linear_scaling: true
  
  memory_efficiency:
    ram_usage: <4GB for 10M vectors
    quantization_ratio: 4:1
```

### Monitoring Integration

```python
# Integration with existing Contexter monitoring
class QdrantMetricsCollector:
    """Collect Qdrant metrics for Contexter monitoring."""
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect performance metrics."""
        return {
            "qdrant_collection_size": await self._get_collection_size(),
            "qdrant_query_latency_p95": await self._get_latency_p95(),
            "qdrant_memory_usage_bytes": await self._get_memory_usage(),
            "qdrant_index_efficiency": await self._get_index_efficiency(),
        }
```

This technical specification provides a comprehensive foundation for implementing Qdrant vector database integration that meets all PRP requirements while maintaining consistency with existing Contexter patterns and performance standards.