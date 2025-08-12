"""
Vector database layer for the Contexter RAG system.

This module provides high-performance vector storage and retrieval using Qdrant:
- Vector storage with HNSW indexing
- Batch operations with performance optimization
- Semantic search with metadata filtering
- Collection management and maintenance
- Health monitoring and performance metrics
"""

from .qdrant_vector_store import (
    QdrantVectorStore, VectorStoreConfig, VectorDocument, SearchResult
)
from .vector_search_engine import (
    VectorSearchEngine, SearchQuery, RankedSearchResult, SearchCache
)
from .batch_uploader import (
    BatchUploader, BatchConfig, BatchResult, BatchStatus, BatchProgress
)
from .health_monitor import (
    VectorHealthMonitor, HealthConfig, HealthReport, HealthStatus, HealthCheck
)

__all__ = [
    # Core vector store
    "QdrantVectorStore",
    "VectorStoreConfig",
    "VectorDocument",
    "SearchResult",

    # Search engine
    "VectorSearchEngine",
    "SearchQuery",
    "RankedSearchResult",
    "SearchCache",

    # Batch operations
    "BatchUploader",
    "BatchConfig",
    "BatchResult",
    "BatchStatus",
    "BatchProgress",

    # Health monitoring
    "VectorHealthMonitor",
    "HealthConfig",
    "HealthReport",
    "HealthStatus",
    "HealthCheck",
]
