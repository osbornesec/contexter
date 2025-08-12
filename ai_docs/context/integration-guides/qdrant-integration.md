# Qdrant Vector Database Integration Guide

## Overview

This guide provides step-by-step instructions for integrating Qdrant vector database into the Contexter RAG system. It covers installation, configuration, collection setup, and integration with existing Contexter patterns.

## Prerequisites

- Python 3.9+ environment
- Docker (for local Qdrant instance)
- Existing Contexter project structure
- At least 4GB RAM for development, 16GB+ for production

## Step 1: Qdrant Installation and Setup

### Option A: Docker Installation (Recommended for Development)

```bash
# Pull and run Qdrant container
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant:latest

# Verify installation
curl http://localhost:6333/
```

Expected response:
```json
{
    "title": "qdrant - vector search engine",
    "version": "1.8.0"
}
```

### Option B: Production Docker Setup

Create `docker-compose.yml` in your project root:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.8.0
    restart: unless-stopped
    container_name: contexter-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    expose:
      - 6333
      - 6334
      - 6335
    configs:
      - source: qdrant_config
        target: /qdrant/config/production.yaml
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    command: ["./qdrant", "--config-path", "/qdrant/config/production.yaml"]

configs:
  qdrant_config:
    file: ./config/qdrant-production.yaml
```

### Option C: Cloud Installation (Qdrant Cloud)

```bash
# For Qdrant Cloud integration
pip install qdrant-client[cloud]

# Configure client for cloud
export QDRANT_URL="https://your-cluster.qdrant.tech:6333"
export QDRANT_API_KEY="your-api-key"
```

## Step 2: Python Client Installation

```bash
# Install Qdrant client with async support
pip install qdrant-client[grpc,async] asyncio-compat

# Verify installation
python -c "import qdrant_client; print('Qdrant client installed successfully')"
```

## Step 3: Configuration Integration

### Update Contexter Configuration

Add Qdrant configuration to your existing `config/contexter_config.yaml`:

```yaml
# Existing Contexter configuration
storage:
  base_path: "~/.contexter/downloads"
  retention_limit: 5

# Add Qdrant configuration
vector_database:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  grpc_port: 6334
  prefer_grpc: true
  timeout: 30
  
  # Collection configuration
  collection_name: "contexter_documentation"
  vector_size: 2048
  distance_metric: "cosine"
  
  # HNSW configuration for performance
  hnsw_config:
    m: 16
    ef_construct: 200
    ef: 128
    full_scan_threshold: 10000
  
  # Performance settings
  max_connections: 100
  connection_pool_size: 20
  
  # Authentication (for Qdrant Cloud)
  api_key: null  # Set via environment variable QDRANT_API_KEY
```

### Update Contexter Config Manager

Extend the existing `ConfigManager` to include Qdrant settings:

```python
# In src/contexter/core/config_manager.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    timeout: int = 30
    api_key: Optional[str] = None
    
    collection_name: str = "contexter_documentation"
    vector_size: int = 2048
    distance_metric: str = "cosine"
    
    # HNSW configuration
    hnsw_m: int = 16
    hnsw_ef_construct: int = 200
    hnsw_ef: int = 128
    hnsw_full_scan_threshold: int = 10000
    
    # Connection settings
    max_connections: int = 100
    connection_pool_size: int = 20

class ConfigManager:
    """Extended ConfigManager with Qdrant support."""
    
    def __init__(self, config_path: str):
        # ... existing initialization ...
        
        # Add Qdrant configuration
        vector_db_config = self.config.get("vector_database", {})
        self.qdrant_config = QdrantConfig(**vector_db_config)
    
    # ... rest of existing methods ...
```

## Step 4: Qdrant Client Implementation

Create `src/contexter/integration/qdrant_client.py`:

```python
"""
Qdrant client integration for Contexter RAG system.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import numpy as np
from datetime import datetime

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, CreateCollection, HnswConfig,
    PointStruct, Filter, FieldCondition, MatchValue, Range,
    SearchResult, UpdateResult
)
from qdrant_client.http import models as rest

# Import Contexter patterns
from contexter.core.config_manager import ConfigManager, QdrantConfig
from contexter.core.error_classifier import ErrorClassifier
from contexter.models.storage_models import StorageResult

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Production-ready Qdrant client for Contexter."""
    
    def __init__(self, config: QdrantConfig):
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_initialized = False
        
        # Performance tracking
        self.operation_stats = {
            "searches": 0,
            "inserts": 0,
            "errors": 0,
            "avg_search_time": 0.0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize Qdrant client and collection."""
        
        # Create async client
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
        
        logger.info(f"Qdrant client initialized for {self.config.host}:{self.config.port}")
        
        # Initialize collection
        await self._ensure_collection_exists()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.close()
            logger.info("Qdrant client closed")
    
    async def _ensure_collection_exists(self):
        """Ensure collection exists with proper configuration."""
        
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.config.collection_name}")
                await self._create_collection()
            else:
                logger.info(f"Collection {self.config.collection_name} already exists")
                # Validate collection configuration
                await self._validate_collection_config()
            
            self.collection_initialized = True
            
        except Exception as e:
            error_info = ErrorClassifier.classify_error(e)
            logger.error(f"Collection initialization failed: {error_info.category} - {e}")
            raise
    
    async def _create_collection(self):
        """Create new collection with optimized configuration."""
        
        # Configure HNSW parameters
        hnsw_config = HnswConfig(
            m=self.config.hnsw_m,
            ef_construct=self.config.hnsw_ef_construct,
            full_scan_threshold=self.config.hnsw_full_scan_threshold,
            max_indexing_threads=0  # Use all available cores
        )
        
        # Configure vector parameters
        vectors_config = VectorParams(
            size=self.config.vector_size,
            distance=Distance.COSINE if self.config.distance_metric == "cosine" else Distance.DOT,
            hnsw_config=hnsw_config
        )
        
        # Create collection
        await self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config
        )
        
        # Create payload indexes for common filters
        await self._create_payload_indexes()
        
        logger.info(f"Collection {self.config.collection_name} created successfully")
    
    async def _create_payload_indexes(self):
        """Create indexes for common payload fields."""
        
        index_fields = [
            ("library_id", rest.PayloadSchemaType.KEYWORD),
            ("doc_type", rest.PayloadSchemaType.KEYWORD),
            ("section", rest.PayloadSchemaType.KEYWORD),
            ("programming_language", rest.PayloadSchemaType.KEYWORD),
            ("trust_score", rest.PayloadSchemaType.FLOAT),
            ("indexed_at", rest.PayloadSchemaType.DATETIME)
        ]
        
        for field_name, field_type in index_fields:
            try:
                await self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_type=field_type
                )
                logger.debug(f"Created index for field: {field_name}")
                
            except Exception as e:
                # Index might already exist
                logger.debug(f"Index creation for {field_name} failed (might exist): {e}")
    
    async def _validate_collection_config(self):
        """Validate existing collection configuration."""
        
        collection_info = await self.client.get_collection(self.config.collection_name)
        
        # Check vector size
        if collection_info.config.params.vectors.size != self.config.vector_size:
            raise ValueError(
                f"Collection vector size mismatch: "
                f"expected {self.config.vector_size}, "
                f"got {collection_info.config.params.vectors.size}"
            )
        
        logger.info("Collection configuration validated")
    
    async def upsert_vectors(
        self, 
        vectors: List[Dict[str, Any]], 
        batch_size: int = 1000
    ) -> StorageResult:
        """Upsert vectors in batches."""
        
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        start_time = time.time()
        total_vectors = len(vectors)
        
        try:
            # Process in batches
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                
                # Convert to PointStruct objects
                points = [
                    PointStruct(
                        id=vector["id"],
                        vector=vector["vector"],
                        payload=vector["payload"]
                    )
                    for vector in batch
                ]
                
                # Upsert batch
                operation_result = await self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=points,
                    wait=True
                )
                
                logger.debug(f"Upserted batch {i//batch_size + 1}: {len(points)} vectors")
            
            # Update statistics
            duration = time.time() - start_time
            self.operation_stats["inserts"] += total_vectors
            
            return StorageResult(
                success=True,
                file_path=None,  # Vector storage doesn't use files
                compressed_size=0,
                compression_ratio=0.0,
                checksum="",
                metadata={
                    "vectors_count": total_vectors,
                    "duration": duration,
                    "collection": self.config.collection_name
                }
            )
            
        except Exception as e:
            self.operation_stats["errors"] += 1
            error_info = ErrorClassifier.classify_error(e)
            
            logger.error(f"Vector upsert failed: {error_info.category} - {e}")
            
            return StorageResult(
                success=False,
                file_path=None,
                compressed_size=0,
                compression_ratio=0.0,
                checksum="",
                error_message=str(e),
                metadata={"error_category": error_info.category}
            )
    
    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        
        if not self.collection_initialized:
            raise RuntimeError("Collection not initialized")
        
        start_time = time.time()
        
        try:
            # Build query filter
            query_filter = self._build_filter(filters) if filters else None
            
            # Set search parameters
            search_params = rest.SearchParams(
                hnsw_ef=ef or self.config.hnsw_ef
            )
            
            # Perform search
            search_result = await self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                search_params=search_params,
                with_payload=True,
                with_vector=False  # Don't return vectors to save bandwidth
            )
            
            # Update statistics
            search_time = time.time() - start_time
            self.operation_stats["searches"] += 1
            
            # Update rolling average
            current_avg = self.operation_stats["avg_search_time"]
            search_count = self.operation_stats["searches"]
            self.operation_stats["avg_search_time"] = (
                (current_avg * (search_count - 1) + search_time) / search_count
            )
            
            # Format results
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            self.operation_stats["errors"] += 1
            error_info = ErrorClassifier.classify_error(e)
            
            logger.error(f"Vector search failed: {error_info.category} - {e}")
            raise
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary."""
        
        conditions = []
        
        # String exact match filters
        string_fields = ["library_id", "doc_type", "section", "programming_language"]
        for field in string_fields:
            if field in filters:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=filters[field])
                    )
                )
        
        # Range filters
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
        
        # Date range filters
        if "indexed_after" in filters:
            conditions.append(
                FieldCondition(
                    key="indexed_at",
                    range=Range(gte=filters["indexed_after"])
                )
            )
        
        return Filter(must=conditions) if conditions else None
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and statistics."""
        
        try:
            collection_info = await self.client.get_collection(self.config.collection_name)
            
            return {
                "name": self.config.collection_name,
                "vectors_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        
        try:
            # Check client connection
            telemetry = await self.client.get_telemetry()
            
            # Get collection info
            collection_info = await self.get_collection_info()
            
            # Test search with dummy vector
            dummy_vector = np.random.random(self.config.vector_size).tolist()
            start_time = time.time()
            await self.search_similar(dummy_vector, limit=1)
            search_latency = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "status": "healthy",
                "version": telemetry.get("app", {}).get("version", "unknown"),
                "collection": collection_info,
                "search_latency_ms": round(search_latency, 2),
                "operation_stats": self.operation_stats.copy(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Step 5: Integration with Existing Contexter Services

### Update Document Processing Pipeline

Modify existing document ingestion to include vector generation:

```python
# In src/contexter/core/document_processor.py

from .integration.qdrant_client import QdrantVectorStore
from .integration.voyage_client import VoyageEmbeddingClient

class DocumentProcessor:
    """Extended document processor with vector generation."""
    
    def __init__(
        self, 
        storage_manager: LocalStorageManager,
        vector_store: QdrantVectorStore,
        embedding_client: VoyageEmbeddingClient
    ):
        self.storage_manager = storage_manager
        self.vector_store = vector_store
        self.embedding_client = embedding_client
    
    async def process_library_documentation(
        self,
        library_id: str,
        version: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process documents with vector generation."""
        
        results = {
            "processed_docs": 0,
            "generated_vectors": 0,
            "errors": []
        }
        
        # Step 1: Process documents with existing pipeline
        processed_docs = await self._process_documents_traditional(
            library_id, version, documents
        )
        results["processed_docs"] = len(processed_docs)
        
        # Step 2: Generate embeddings
        texts = [doc["content"] for doc in processed_docs]
        try:
            embeddings = await self.embedding_client.embed_texts(
                texts, 
                model="voyage-code-3",
                input_type="document"
            )
            
            # Step 3: Prepare vectors for storage
            vectors = []
            for i, (doc, embedding) in enumerate(zip(processed_docs, embeddings)):
                vector_data = {
                    "id": f"{library_id}_{version}_{i}",
                    "vector": embedding,
                    "payload": {
                        "library_id": library_id,
                        "library_name": doc.get("library_name", ""),
                        "version": version,
                        "doc_type": doc.get("doc_type", "unknown"),
                        "section": doc.get("section", ""),
                        "content": doc["content"],
                        "token_count": doc.get("token_count", 0),
                        "trust_score": doc.get("trust_score", 0.0),
                        "programming_language": doc.get("programming_language", ""),
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                }
                vectors.append(vector_data)
            
            # Step 4: Store vectors
            storage_result = await self.vector_store.upsert_vectors(vectors)
            
            if storage_result.success:
                results["generated_vectors"] = len(vectors)
            else:
                results["errors"].append(f"Vector storage failed: {storage_result.error_message}")
            
        except Exception as e:
            results["errors"].append(f"Embedding generation failed: {e}")
        
        return results
```

## Step 6: Testing and Validation

### Create Integration Tests

Create `tests/integration/test_qdrant_integration.py`:

```python
import pytest
import asyncio
import numpy as np
from contexter.integration.qdrant_client import QdrantVectorStore
from contexter.core.config_manager import QdrantConfig


@pytest.fixture
async def qdrant_client():
    """Fixture for Qdrant client."""
    config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="test_collection"
    )
    
    client = QdrantVectorStore(config)
    await client.initialize()
    
    yield client
    
    # Cleanup
    try:
        await client.client.delete_collection("test_collection")
    except:
        pass
    await client.cleanup()


@pytest.mark.asyncio
async def test_collection_creation(qdrant_client):
    """Test collection creation."""
    
    collection_info = await qdrant_client.get_collection_info()
    
    assert collection_info["name"] == "test_collection"
    assert collection_info["vector_size"] == 2048
    assert collection_info["status"] == "green"


@pytest.mark.asyncio
async def test_vector_operations(qdrant_client):
    """Test vector upsert and search operations."""
    
    # Generate test vectors
    test_vectors = []
    for i in range(10):
        vector_data = {
            "id": f"test_{i}",
            "vector": np.random.random(2048).tolist(),
            "payload": {
                "library_id": "test_lib",
                "content": f"Test content {i}",
                "doc_type": "api"
            }
        }
        test_vectors.append(vector_data)
    
    # Test upsert
    result = await qdrant_client.upsert_vectors(test_vectors)
    assert result.success
    assert result.metadata["vectors_count"] == 10
    
    # Test search
    query_vector = np.random.random(2048).tolist()
    search_results = await qdrant_client.search_similar(
        query_vector, 
        limit=5
    )
    
    assert len(search_results) <= 5
    assert all("score" in result for result in search_results)
    assert all("payload" in result for result in search_results)


@pytest.mark.asyncio
async def test_health_check(qdrant_client):
    """Test health check functionality."""
    
    health = await qdrant_client.health_check()
    
    assert health["status"] == "healthy"
    assert "search_latency_ms" in health
    assert health["search_latency_ms"] < 1000  # Should be under 1 second
    assert "collection" in health
```

### Performance Testing

Create `tests/performance/test_qdrant_performance.py`:

```python
import pytest
import asyncio
import time
import numpy as np
from contexter.integration.qdrant_client import QdrantVectorStore


@pytest.mark.asyncio
async def test_search_performance(qdrant_client):
    """Test search performance under load."""
    
    # Insert test data
    vectors = []
    for i in range(1000):
        vectors.append({
            "id": f"perf_test_{i}",
            "vector": np.random.random(2048).tolist(),
            "payload": {"content": f"Performance test content {i}"}
        })
    
    await qdrant_client.upsert_vectors(vectors)
    
    # Test concurrent searches
    async def search_task():
        query_vector = np.random.random(2048).tolist()
        start_time = time.time()
        results = await qdrant_client.search_similar(query_vector, limit=10)
        latency = time.time() - start_time
        return latency, len(results)
    
    # Run concurrent searches
    tasks = [search_task() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    
    latencies = [r[0] for r in results]
    result_counts = [r[1] for r in results]
    
    # Performance assertions
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    assert avg_latency < 0.1  # Average under 100ms
    assert p95_latency < 0.2   # P95 under 200ms
    assert all(count <= 10 for count in result_counts)  # All returned correct count
    
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"P95 latency: {p95_latency:.3f}s")
```

## Step 7: Production Deployment

### Production Configuration

Create `config/qdrant-production.yaml`:

```yaml
log_level: INFO

storage:
  # Optimize for SSD storage
  storage_path: /qdrant/storage
  snapshots_path: /qdrant/snapshots
  temp_path: /qdrant/temp
  
  # Memory optimization
  performance:
    max_search_threads: 8
    max_optimization_threads: 4

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  
  # Enable gRPC for better performance
  enable_cors: true
  max_request_size_mb: 32
  max_workers: 8

cluster:
  # Single node configuration
  # For multi-node, configure p2p settings
  enabled: false

telemetry:
  disabled: false
```

### Health Check Script

Create `scripts/check_qdrant_health.py`:

```python
#!/usr/bin/env python3
"""
Health check script for Qdrant in production.
"""

import asyncio
import sys
from contexter.integration.qdrant_client import QdrantVectorStore
from contexter.core.config_manager import ConfigManager


async def main():
    """Run health check."""
    
    try:
        # Load production configuration
        config_manager = ConfigManager("config/contexter_config.yaml")
        
        async with QdrantVectorStore(config_manager.qdrant_config) as client:
            health = await client.health_check()
            
            if health["status"] == "healthy":
                print("✅ Qdrant is healthy")
                print(f"   Version: {health.get('version', 'unknown')}")
                print(f"   Search latency: {health.get('search_latency_ms', 0):.2f}ms")
                print(f"   Vectors: {health.get('collection', {}).get('vectors_count', 0)}")
                sys.exit(0)
            else:
                print("❌ Qdrant is unhealthy")
                print(f"   Error: {health.get('error', 'Unknown error')}")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
```

## Step 8: Monitoring and Observability

### Metrics Collection

Add Qdrant metrics to your existing monitoring:

```python
# In src/contexter/core/metrics_collector.py

class ContexterMetrics:
    """Extended metrics collector with Qdrant metrics."""
    
    def __init__(self):
        # ... existing metrics ...
        
        # Qdrant metrics
        self.qdrant_searches_total = Counter(
            'qdrant_searches_total',
            'Total number of vector searches'
        )
        
        self.qdrant_search_duration = Histogram(
            'qdrant_search_duration_seconds',
            'Time spent on vector searches'
        )
        
        self.qdrant_vectors_total = Counter(
            'qdrant_vectors_total',
            'Total number of vectors stored'
        )
        
        self.qdrant_collection_size = Gauge(
            'qdrant_collection_size',
            'Number of vectors in collection'
        )
    
    async def update_qdrant_metrics(self, vector_store: QdrantVectorStore):
        """Update Qdrant-specific metrics."""
        
        # Update operation counters
        stats = vector_store.operation_stats
        self.qdrant_searches_total._value.set(stats["searches"])
        self.qdrant_vectors_total._value.set(stats["inserts"])
        
        # Update collection size
        collection_info = await vector_store.get_collection_info()
        if "vectors_count" in collection_info:
            self.qdrant_collection_size.set(collection_info["vectors_count"])
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if Qdrant is running
   docker ps | grep qdrant
   
   # Check logs
   docker logs contexter-qdrant
   
   # Test connection
   curl http://localhost:6333/health
   ```

2. **Collection Already Exists Error**
   ```python
   # Delete existing collection if needed
   await client.delete_collection("collection_name")
   ```

3. **Memory Issues**
   ```yaml
   # Reduce HNSW parameters in config
   hnsw_config:
     m: 8  # Reduced from 16
     ef_construct: 100  # Reduced from 200
   ```

4. **Slow Search Performance**
   ```python
   # Increase ef parameter for better accuracy
   search_results = await client.search_similar(
       query_vector,
       ef=256  # Higher than default 128
   )
   ```

### Performance Optimization

1. **Batch Size Tuning**
   - Start with 1000 vectors per batch
   - Reduce if memory issues occur
   - Increase for better throughput on powerful hardware

2. **HNSW Parameter Tuning**
   - `m`: 8-32 (16 is balanced)
   - `ef_construct`: 100-400 (higher = better accuracy, slower indexing)
   - `ef`: 64-512 (higher = better search accuracy, slower search)

3. **Connection Optimization**
   - Use gRPC for better performance
   - Increase connection pool size for high concurrency
   - Use async client for non-blocking operations

This completes the comprehensive Qdrant integration guide. The integration provides production-ready vector search capabilities while maintaining compatibility with existing Contexter patterns and performance requirements.