# RAG Implementation Patterns & Code Standards

## Overview

This document provides specific implementation patterns, code templates, and architectural guidelines for the RAG system development. Each pattern is designed to integrate seamlessly with the existing Contexter codebase while establishing consistent standards for the new RAG functionality.

## File Structure & Organization

### New RAG Module Structure
```
src/contexter/rag/
├── __init__.py
├── vector_db/
│   ├── __init__.py
│   ├── qdrant_store.py          # Vector database operations
│   ├── collection_manager.py    # Collection lifecycle management
│   ├── batch_uploader.py        # Bulk vector operations
│   └── health_monitor.py        # Database health checks
├── storage/
│   ├── __init__.py
│   ├── rag_storage_manager.py   # Enhanced storage for RAG
│   ├── chunk_storage.py         # Document chunk storage
│   ├── metadata_index.py        # Metadata indexing
│   └── cache_storage.py         # Embedding cache management
├── embedding/
│   ├── __init__.py
│   ├── voyage_client.py         # Voyage AI integration
│   ├── embedding_engine.py      # Main embedding interface
│   ├── cache_manager.py         # Persistent embedding cache
│   └── batch_processor.py       # Batch embedding processing
├── ingestion/
│   ├── __init__.py
│   ├── auto_trigger.py          # Ingestion trigger system
│   ├── pipeline.py              # Main ingestion pipeline
│   ├── json_parser.py           # Document parsing
│   ├── chunking_engine.py       # Intelligent document chunking
│   ├── metadata_extractor.py    # Content metadata extraction
│   └── worker_pool.py           # Concurrent processing workers
├── retrieval/
│   ├── __init__.py
│   ├── query_processor.py       # Query analysis and processing
│   ├── semantic_search.py       # Vector similarity search
│   ├── keyword_search.py        # BM25-based keyword search
│   ├── hybrid_search.py         # Combined search orchestration
│   ├── result_fusion.py         # Search result combination
│   └── ranking_engine.py        # Result ranking and filtering
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── search.py            # Search endpoints
│   │   ├── documents.py         # Document management
│   │   └── system.py            # System monitoring
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py              # Authentication middleware
│   │   ├── rate_limit.py        # Rate limiting
│   │   └── monitoring.py        # Request monitoring
│   └── models/
│       ├── __init__.py
│       ├── search_models.py     # Search request/response models
│       └── system_models.py     # System status models
├── testing/
│   ├── __init__.py
│   ├── framework.py             # Test orchestration
│   ├── accuracy_tester.py       # Search accuracy validation
│   ├── performance_tester.py    # Performance benchmarking
│   └── ground_truth.py          # Test data management
└── monitoring/
    ├── __init__.py
    ├── metrics_collector.py     # Prometheus metrics
    ├── tracing.py               # Distributed tracing
    └── dashboards.py            # Grafana dashboard configs
```

### Integration with Existing Structure
Extend existing modules where appropriate:

```python
# src/contexter/models/rag_models.py
from .config_models import BaseConfig
from .storage_models import StorageConfig

class RAGConfig(BaseConfig):
    """RAG system configuration extending existing config patterns"""
    vector_db: VectorDBConfig
    embedding: EmbeddingConfig  
    search: SearchConfig
    ingestion: IngestionConfig
```

## Core Implementation Patterns

### 1. Async Service Pattern
All RAG services follow the existing async context manager pattern:

```python
# Template: src/contexter/rag/base_service.py
from abc import ABC, abstractmethod
from typing import Any, Optional
import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class BaseRAGService(ABC):
    """Base class for all RAG services following existing patterns"""
    
    def __init__(self, config: dict, **kwargs):
        self.config = config
        self.initialized = False
        self._resources = []
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        
    async def initialize(self):
        """Initialize service resources"""
        if self.initialized:
            return
            
        try:
            await self._initialize_resources()
            self.initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            await self.cleanup()
            raise
            
    @abstractmethod
    async def _initialize_resources(self):
        """Implement service-specific initialization"""
        pass
        
    async def cleanup(self):
        """Cleanup service resources"""
        for resource in reversed(self._resources):
            try:
                if hasattr(resource, 'close'):
                    await resource.close()
                elif hasattr(resource, '__aexit__'):
                    await resource.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error cleaning up resource: {e}")
                
        self._resources.clear()
        self.initialized = False
        
    async def health_check(self) -> bool:
        """Service health check"""
        return self.initialized
```

### 2. Error Handling Pattern
Extend existing error classification system:

```python
# src/contexter/rag/errors.py
from ..core.error_classifier import ClassifiedError, ErrorCategory

class RAGError(ClassifiedError):
    """Base RAG system error following existing error patterns"""
    pass

class VectorStoreError(RAGError):
    """Vector database operation errors"""
    category = ErrorCategory.EXTERNAL_SERVICE
    
class EmbeddingError(RAGError):
    """Embedding generation errors"""
    category = ErrorCategory.EXTERNAL_SERVICE
    
class SearchError(RAGError):
    """Search operation errors"""  
    category = ErrorCategory.PROCESSING
    
class IngestionError(RAGError):
    """Document ingestion errors"""
    category = ErrorCategory.PROCESSING

# Usage in services
@error_handler(EmbeddingError, "embedding generation")
async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings with proper error handling"""
    try:
        return await self._generate_embeddings_impl(texts)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise EmbeddingError("Rate limit exceeded", retryable=True)
        elif e.response.status_code >= 500:
            raise EmbeddingError(f"Service error: {e}", retryable=True)
        else:
            raise EmbeddingError(f"Client error: {e}", retryable=False)
```

### 3. Configuration Management Pattern
Extend existing configuration system:

```python
# src/contexter/rag/config.py
from ..core.config_manager import ConfigManager
from ..models.config_models import BaseConfig
from pydantic import BaseModel, Field
from typing import Optional

class VectorDBConfig(BaseModel):
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection_name: str = Field(default="contexter_documentation")
    hnsw_m: int = Field(default=16)
    hnsw_ef_construct: int = Field(default=200)
    hnsw_ef: int = Field(default=100)

class EmbeddingConfig(BaseModel):
    provider: str = Field(default="voyage")
    model: str = Field(default="voyage-code-3")
    api_key: Optional[str] = Field(default=None)
    batch_size: int = Field(default=100)
    cache_ttl_seconds: int = Field(default=604800)  # 7 days
    rate_limit_requests_per_minute: int = Field(default=300)

class SearchConfig(BaseModel):
    semantic_weight: float = Field(default=0.7)
    keyword_weight: float = Field(default=0.3)
    similarity_threshold: float = Field(default=0.1)
    max_results: int = Field(default=100)
    cache_ttl_seconds: int = Field(default=3600)  # 1 hour

class RAGConfig(BaseConfig):
    """RAG system configuration"""
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

class RAGConfigManager(ConfigManager):
    """RAG configuration manager extending existing patterns"""
    
    def __init__(self):
        super().__init__()
        self.rag_config: Optional[RAGConfig] = None
        
    async def load_rag_config(self) -> RAGConfig:
        """Load RAG configuration with environment overrides"""
        if self.rag_config is None:
            config_data = await self.load_config_section("rag")
            
            # Environment variable overrides
            if api_key := self._get_env_var("VOYAGE_API_KEY"):
                config_data.setdefault("embedding", {})["api_key"] = api_key
                
            if db_host := self._get_env_var("QDRANT_HOST"):
                config_data.setdefault("vector_db", {})["host"] = db_host
                
            self.rag_config = RAGConfig(**config_data)
            
        return self.rag_config
```

### 4. Monitoring Integration Pattern
Integrate with existing monitoring systems:

```python
# src/contexter/rag/monitoring/base_monitor.py
from ..core.progress_reporter import ProgressReporter
from prometheus_client import Counter, Histogram, Gauge
from typing import Dict, Any
import time

class RAGMetricsCollector:
    """RAG metrics collection following existing monitoring patterns"""
    
    def __init__(self):
        self.progress_reporter = ProgressReporter()
        
        # Search metrics
        self.search_requests = Counter(
            'rag_search_requests_total',
            'Total search requests',
            ['search_type', 'status']
        )
        
        self.search_latency = Histogram(
            'rag_search_latency_seconds',
            'Search latency in seconds',
            ['search_type'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Ingestion metrics
        self.documents_processed = Counter(
            'rag_documents_processed_total',
            'Total documents processed',
            ['status']
        )
        
        self.ingestion_queue_size = Gauge(
            'rag_ingestion_queue_size',
            'Current ingestion queue size'
        )
        
    async def record_search_request(self, search_type: str, duration: float, success: bool):
        """Record search request metrics"""
        status = 'success' if success else 'error'
        self.search_requests.labels(search_type=search_type, status=status).inc()
        self.search_latency.labels(search_type=search_type).observe(duration)
        
        # Also report to existing progress system
        await self.progress_reporter.report_progress(
            operation_type="search",
            current=1,
            total=1,
            metadata={
                "search_type": search_type,
                "duration_ms": duration * 1000,
                "success": success
            }
        )

# Decorator for automatic monitoring
def monitor_rag_operation(operation_type: str, include_duration: bool = True):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time() if include_duration else None
            
            try:
                result = await func(self, *args, **kwargs)
                
                if hasattr(self, 'metrics_collector'):
                    duration = time.time() - start_time if start_time else 0
                    await self.metrics_collector.record_operation(
                        operation_type, duration, True
                    )
                
                return result
                
            except Exception as e:
                if hasattr(self, 'metrics_collector'):
                    duration = time.time() - start_time if start_time else 0
                    await self.metrics_collector.record_operation(
                        operation_type, duration, False
                    )
                raise
                
        return wrapper
    return decorator
```

### 5. Testing Pattern
Extend existing test infrastructure:

```python
# tests/rag/conftest.py
import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock

from contexter.rag.config import RAGConfig, RAGConfigManager
from contexter.rag.vector_db.qdrant_store import QdrantVectorStore
from contexter.rag.embedding.embedding_engine import EmbeddingEngine

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_storage_path():
    """Temporary storage path for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
async def test_rag_config():
    """Test RAG configuration"""
    return RAGConfig(
        vector_db={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test_collection"
        },
        embedding={
            "provider": "voyage",
            "model": "voyage-code-3",
            "api_key": "test_key",
            "batch_size": 10
        }
    )

@pytest.fixture
async def mock_vector_store():
    """Mock vector store for testing"""
    mock = AsyncMock(spec=QdrantVectorStore)
    mock.search_vectors.return_value = [
        {
            'id': 'test_id_1',
            'score': 0.95,
            'payload': {'content': 'test content 1'}
        }
    ]
    return mock

@pytest.fixture  
async def mock_embedding_engine():
    """Mock embedding engine for testing"""
    mock = AsyncMock(spec=EmbeddingEngine)
    mock.generate_embeddings.return_value = [
        [0.1] * 2048 for _ in range(10)  # Mock 2048-dim embeddings
    ]
    return mock

# Base test class
class BaseRAGTest:
    """Base class for RAG tests with common utilities"""
    
    @pytest.fixture(autouse=True)
    def setup_logging(self, caplog):
        """Setup logging for tests"""
        self.caplog = caplog
        
    def assert_performance_target(self, duration: float, target_ms: int):
        """Assert performance meets target"""
        assert duration < (target_ms / 1000), f"Operation took {duration*1000:.1f}ms, target was {target_ms}ms"
        
    def create_test_documents(self, count: int = 10):
        """Create test documents for testing"""
        return [
            {
                'id': f'test_doc_{i}',
                'content': f'Test document content {i}',
                'metadata': {'type': 'test', 'index': i}
            }
            for i in range(count)
        ]
```

### 6. Data Model Pattern
Consistent data models following existing patterns:

```python
# src/contexter/rag/models/search_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class SearchType(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic" 
    KEYWORD = "keyword"

class DocType(str, Enum):
    API = "api"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"

class SearchFilters(BaseModel):
    """Search filters model"""
    library_ids: Optional[List[str]] = Field(default=None)
    doc_types: Optional[List[DocType]] = Field(default=None)
    programming_languages: Optional[List[str]] = Field(default=None)
    trust_score_min: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    star_count_min: Optional[int] = Field(default=None, ge=0)
    
    @validator('library_ids')
    def validate_library_ids(cls, v):
        if v and len(v) > 50:
            raise ValueError("Too many library IDs (max 50)")
        return v

class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=500)
    filters: Optional[SearchFilters] = Field(default=None)
    top_k: int = Field(default=10, ge=1, le=100)
    search_type: SearchType = Field(default=SearchType.HYBRID)
    similarity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enable_highlighting: bool = Field(default=True)
    
    @validator('semantic_weight', 'keyword_weight')
    def validate_weights(cls, v, values):
        if 'semantic_weight' in values and 'keyword_weight' in values:
            sw = values.get('semantic_weight', 0.7)
            kw = values.get('keyword_weight', 0.3)
            if sw and kw and abs(sw + kw - 1.0) > 0.01:
                raise ValueError("Semantic and keyword weights must sum to 1.0")
        return v

class SearchResult(BaseModel):
    """Individual search result model"""
    result_id: str
    chunk_id: str
    library_id: str
    library_name: str
    version: str
    doc_type: DocType
    content: str
    content_snippet: Optional[str] = Field(default=None)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    semantic_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyword_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)
    created_at: datetime

class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult]
    total_results: int
    query_time_ms: float
    search_metadata: Dict[str, Any] = Field(default_factory=dict)
    filters_applied: Optional[SearchFilters] = Field(default=None)
    pagination: Optional[Dict[str, Any]] = Field(default=None)
```

## Implementation Guidelines

### Development Workflow
1. **Start with Interfaces**: Define service interfaces before implementation
2. **Test-Driven Development**: Write tests alongside implementation
3. **Configuration First**: Ensure all magic numbers are configurable
4. **Error Handling**: Implement comprehensive error handling from the start
5. **Monitoring Integration**: Add metrics collection for all operations
6. **Documentation**: Document all public APIs and complex algorithms

### Code Quality Standards
- **Type Hints**: All functions must have complete type hints
- **Docstrings**: All classes and public methods must have docstrings
- **Error Messages**: All error messages must be actionable
- **Logging**: Use structured logging with appropriate levels
- **Testing**: Minimum 90% test coverage for new code
- **Performance**: All performance targets must be validated with tests

### Integration Points
- **Configuration**: Extend existing `ConfigManager` rather than creating new config systems
- **Storage**: Build on existing `StorageManager` infrastructure
- **Monitoring**: Integrate with existing `ProgressReporter` system
- **Error Handling**: Use existing `ErrorClassifier` patterns
- **Testing**: Follow existing test patterns and fixtures

This pattern guide provides the foundation for consistent, maintainable RAG system implementation that integrates seamlessly with the existing Contexter architecture.