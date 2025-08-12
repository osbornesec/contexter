# RAG Task Execution Guide

## Overview

This guide provides step-by-step execution instructions for each task in the RAG implementation blueprint. Each task includes specific deliverables, implementation steps, validation criteria, and integration points.

## Foundation Layer Tasks (Week 1-2)

### Vector Database Setup (VDB) - 24 hours total

#### VDB-001: Qdrant Client Integration (3 hours)
**Dependencies**: None
**Files to Create/Modify**:
- `src/contexter/rag/vector_db/__init__.py`
- `src/contexter/rag/vector_db/qdrant_store.py`
- `tests/unit/rag/vector_db/test_qdrant_store.py`

**Implementation Steps**:

1. **Install Dependencies** (15 minutes)
```bash
# Add to requirements.txt
echo "qdrant-client[grpc]==1.7.0" >> requirements.txt
echo "numpy>=1.21.0" >> requirements.txt
pip install -r requirements.txt
```

2. **Create Base QdrantVectorStore** (2 hours)
```python
# src/contexter/rag/vector_db/qdrant_store.py
from typing import List, Dict, Any, Optional
import asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, PointStruct
from ..base_service import BaseRAGService
from ..config import VectorDBConfig
from ..errors import VectorStoreError
from ..monitoring.base_monitor import monitor_rag_operation

class QdrantVectorStore(BaseRAGService):
    """Qdrant vector database client with async operations"""
    
    def __init__(self, config: VectorDBConfig):
        super().__init__(config.dict())
        self.config = config
        self.client: Optional[QdrantClient] = None
        self.collection_name = config.collection_name
        self.vector_size = 2048
        
    async def _initialize_resources(self):
        """Initialize Qdrant client connection"""
        try:
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.port + 1 if self.config.port != 6333 else 6334,
                prefer_grpc=True,
                timeout=30.0
            )
            
            # Test connection
            collections = await asyncio.to_thread(self.client.get_collections)
            self._resources.append(self.client)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to connect to Qdrant: {e}")
    
    @monitor_rag_operation("vector_db_health_check")
    async def health_check(self) -> bool:
        """Check Qdrant health"""
        if not self.client:
            return False
            
        try:
            await asyncio.to_thread(self.client.get_collections)
            return True
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            self.client.close()
        await super().cleanup()
```

3. **Create Unit Tests** (45 minutes)
```python
# tests/unit/rag/vector_db/test_qdrant_store.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from contexter.rag.vector_db.qdrant_store import QdrantVectorStore
from contexter.rag.config import VectorDBConfig
from contexter.rag.errors import VectorStoreError

class TestQdrantVectorStore:
    @pytest.fixture
    def config(self):
        return VectorDBConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection"
        )
    
    @pytest.fixture
    def vector_store(self, config):
        return QdrantVectorStore(config)
    
    @patch('contexter.rag.vector_db.qdrant_store.QdrantClient')
    async def test_initialize_success(self, mock_client_class, vector_store):
        mock_client = Mock()
        mock_client.get_collections.return_value = []
        mock_client_class.return_value = mock_client
        
        async with vector_store:
            assert vector_store.client is not None
            assert vector_store.initialized
            
    async def test_health_check_without_client(self, vector_store):
        result = await vector_store.health_check()
        assert result is False
```

**Validation Criteria**:
- [ ] Successfully connects to Qdrant instance
- [ ] Health check returns accurate status
- [ ] Proper error handling for connection failures
- [ ] Unit tests achieve >90% coverage
- [ ] Integration test with real Qdrant instance passes

---

#### VDB-002: Collection Management System (3 hours)
**Dependencies**: VDB-001
**Files to Create/Modify**:
- `src/contexter/rag/vector_db/collection_manager.py`
- Extend `qdrant_store.py`
- `tests/unit/rag/vector_db/test_collection_manager.py`

**Implementation Steps**:

1. **Create Collection Manager** (2 hours)
```python
# src/contexter/rag/vector_db/collection_manager.py
from typing import Optional, Dict, Any
import asyncio
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, CreateCollection
from .qdrant_store import QdrantVectorStore
from ..errors import VectorStoreError
from ..monitoring.base_monitor import monitor_rag_operation

class CollectionManager:
    """Manages Qdrant collection lifecycle operations"""
    
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.config = vector_store.config
    
    @monitor_rag_operation("collection_create")
    async def create_collection(self, recreate: bool = False) -> bool:
        """Create collection with optimized HNSW configuration"""
        if not self.vector_store.client:
            raise VectorStoreError("Vector store client not initialized")
            
        try:
            # Check if collection exists
            collections = await asyncio.to_thread(
                self.vector_store.client.get_collections
            )
            
            collection_names = [c.name for c in collections.collections]
            
            if self.config.collection_name in collection_names:
                if recreate:
                    await asyncio.to_thread(
                        self.vector_store.client.delete_collection,
                        self.config.collection_name
                    )
                else:
                    return True  # Collection already exists
            
            # Create collection with HNSW configuration
            hnsw_config = HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
                full_scan_threshold=10000,
                max_indexing_threads=0  # Use all available cores
            )
            
            vector_params = VectorParams(
                size=self.vector_store.vector_size,
                distance=Distance.COSINE
            )
            
            await asyncio.to_thread(
                self.vector_store.client.create_collection,
                collection_name=self.config.collection_name,
                vectors_config=vector_params,
                hnsw_config=hnsw_config
            )
            
            # Create payload indexes for fast filtering
            await self._create_payload_indexes()
            
            return True
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create collection: {e}")
    
    async def _create_payload_indexes(self):
        """Create indexes for filterable payload fields"""
        indexes = [
            ("library_id", "keyword"),
            ("doc_type", "keyword"),
            ("programming_language", "keyword"),
            ("timestamp", "datetime"),
            ("trust_score", "float")
        ]
        
        for field_name, field_type in indexes:
            try:
                await asyncio.to_thread(
                    self.vector_store.client.create_payload_index,
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_type=field_type
                )
            except Exception as e:
                # Log warning but don't fail collection creation
                print(f"Warning: Failed to create index for {field_name}: {e}")
```

2. **Extend QdrantVectorStore** (45 minutes)
```python
# Add to qdrant_store.py
    async def _initialize_resources(self):
        """Initialize Qdrant client and collection"""
        await super()._initialize_resources()
        
        # Initialize collection manager
        self.collection_manager = CollectionManager(self)
        
        # Ensure collection exists
        await self.collection_manager.create_collection()
```

3. **Create Tests** (45 minutes)
```python
# tests/unit/rag/vector_db/test_collection_manager.py
import pytest
from unittest.mock import Mock, patch, AsyncMock
from contexter.rag.vector_db.collection_manager import CollectionManager
from contexter.rag.vector_db.qdrant_store import QdrantVectorStore

class TestCollectionManager:
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock(spec=QdrantVectorStore)
        mock_store.client = Mock()
        mock_store.config = Mock()
        mock_store.config.collection_name = "test_collection"
        mock_store.config.hnsw_m = 16
        mock_store.config.hnsw_ef_construct = 200
        mock_store.vector_size = 2048
        return mock_store
    
    @pytest.fixture
    def collection_manager(self, mock_vector_store):
        return CollectionManager(mock_vector_store)
    
    async def test_create_collection_new(self, collection_manager, mock_vector_store):
        # Mock no existing collections
        mock_response = Mock()
        mock_response.collections = []
        mock_vector_store.client.get_collections.return_value = mock_response
        
        result = await collection_manager.create_collection()
        
        assert result is True
        mock_vector_store.client.create_collection.assert_called_once()
```

**Validation Criteria**:
- [ ] Successfully creates collection with HNSW configuration
- [ ] Handles existing collection scenarios correctly
- [ ] Creates all required payload indexes
- [ ] Proper error handling for creation failures
- [ ] Collection validation tests pass

---

#### VDB-003: Health Monitoring Integration (2 hours)
**Dependencies**: VDB-002
**Files to Create/Modify**:
- `src/contexter/rag/vector_db/health_monitor.py`
- Extend `qdrant_store.py`
- `tests/unit/rag/vector_db/test_health_monitor.py`

**Implementation Steps**:

1. **Create Health Monitor** (1 hour)
```python
# src/contexter/rag/vector_db/health_monitor.py
from typing import Dict, Any
import asyncio
import time
from dataclasses import dataclass
from .qdrant_store import QdrantVectorStore
from ..monitoring.base_monitor import monitor_rag_operation

@dataclass
class HealthStatus:
    """Health status data structure"""
    healthy: bool
    response_time_ms: float
    collection_exists: bool
    collection_size: int
    last_check: float
    error_message: str = ""

class VectorDBHealthMonitor:
    """Health monitoring for vector database"""
    
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        self.last_status: Optional[HealthStatus] = None
        
    @monitor_rag_operation("vector_db_health_check", include_duration=True)
    async def check_health(self) -> HealthStatus:
        """Comprehensive health check"""
        start_time = time.time()
        
        try:
            if not self.vector_store.client:
                return HealthStatus(
                    healthy=False,
                    response_time_ms=0,
                    collection_exists=False,
                    collection_size=0,
                    last_check=time.time(),
                    error_message="Client not initialized"
                )
            
            # Check basic connectivity
            collections_response = await asyncio.to_thread(
                self.vector_store.client.get_collections
            )
            
            # Check collection existence and get stats
            collection_exists = False
            collection_size = 0
            collection_name = self.vector_store.config.collection_name
            
            for collection in collections_response.collections:
                if collection.name == collection_name:
                    collection_exists = True
                    # Get collection info for size
                    info = await asyncio.to_thread(
                        self.vector_store.client.get_collection,
                        collection_name
                    )
                    collection_size = info.points_count or 0
                    break
            
            response_time_ms = (time.time() - start_time) * 1000
            
            status = HealthStatus(
                healthy=collection_exists,
                response_time_ms=response_time_ms,
                collection_exists=collection_exists,
                collection_size=collection_size,
                last_check=time.time()
            )
            
            self.last_status = status
            return status
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = HealthStatus(
                healthy=False,
                response_time_ms=response_time_ms,
                collection_exists=False,
                collection_size=0,
                last_check=time.time(),
                error_message=str(e)
            )
            
            self.last_status = status
            return status
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get status as dictionary for API responses"""
        if not self.last_status:
            return {"status": "unknown", "last_check": None}
            
        return {
            "status": "healthy" if self.last_status.healthy else "unhealthy",
            "response_time_ms": self.last_status.response_time_ms,
            "collection_exists": self.last_status.collection_exists,
            "collection_size": self.last_status.collection_size,
            "last_check": self.last_status.last_check,
            "error_message": self.last_status.error_message
        }
```

2. **Integration with QdrantVectorStore** (45 minutes)
```python
# Add to qdrant_store.py
    async def _initialize_resources(self):
        """Initialize all resources including health monitor"""
        await super()._initialize_resources()
        
        # Initialize collection manager and health monitor
        self.collection_manager = CollectionManager(self)
        self.health_monitor = VectorDBHealthMonitor(self)
        
        # Ensure collection exists
        await self.collection_manager.create_collection()
        
        # Initial health check
        await self.health_monitor.check_health()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        status = await self.health_monitor.check_health()
        return self.health_monitor.get_status_dict()
```

3. **Create Tests** (15 minutes)
```python
# tests/unit/rag/vector_db/test_health_monitor.py
import pytest
from unittest.mock import Mock, AsyncMock
from contexter.rag.vector_db.health_monitor import VectorDBHealthMonitor, HealthStatus

class TestVectorDBHealthMonitor:
    @pytest.fixture
    def mock_vector_store(self):
        mock = Mock()
        mock.client = Mock()
        mock.config = Mock()
        mock.config.collection_name = "test_collection"
        return mock
    
    @pytest.fixture  
    def health_monitor(self, mock_vector_store):
        return VectorDBHealthMonitor(mock_vector_store)
    
    async def test_health_check_success(self, health_monitor, mock_vector_store):
        # Mock successful responses
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test_collection")]
        mock_vector_store.client.get_collections.return_value = mock_collections
        
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1000
        mock_vector_store.client.get_collection.return_value = mock_collection_info
        
        status = await health_monitor.check_health()
        
        assert status.healthy is True
        assert status.collection_exists is True
        assert status.collection_size == 1000
        assert status.response_time_ms > 0
```

**Validation Criteria**:
- [ ] Health checks complete within 1 second
- [ ] Accurately reports collection status and size
- [ ] Handles connection failures gracefully
- [ ] Status is accessible via dictionary format
- [ ] Integration with monitoring system works

---

### Continue with remaining tasks...

## Task Validation Framework

Each task must pass these validation steps before marking complete:

### 1. Unit Test Validation
```bash
# Run specific task tests
pytest tests/unit/rag/vector_db/test_qdrant_store.py -v --cov=src/contexter/rag/vector_db/qdrant_store --cov-report=term-missing

# Coverage must be >90%
coverage report --include="src/contexter/rag/vector_db/*" --fail-under=90
```

### 2. Integration Test Validation  
```bash
# Run integration tests with real services
pytest tests/integration/rag/vector_db/ -v --integration

# Performance validation
pytest tests/performance/rag/vector_db/ -v --benchmark-only
```

### 3. Configuration Validation
```bash
# Validate configuration loading
python -c "
from contexter.rag.config import RAGConfigManager
import asyncio
async def test():
    config_mgr = RAGConfigManager()
    config = await config_mgr.load_rag_config()
    print(f'Config loaded: {config.vector_db.host}')
asyncio.run(test())
"
```

### 4. Error Handling Validation
```bash
# Test error scenarios
python -c "
from contexter.rag.vector_db.qdrant_store import QdrantVectorStore
from contexter.rag.config import VectorDBConfig
import asyncio

async def test():
    # Test with invalid config
    config = VectorDBConfig(host='invalid_host', port=9999)
    store = QdrantVectorStore(config)
    try:
        async with store:
            pass
    except Exception as e:
        print(f'Expected error: {e}')
        
asyncio.run(test())
"
```

## Continuous Integration Steps

After each task completion:

1. **Run Full Test Suite**:
```bash
pytest tests/ --cov=src/contexter/rag --cov-report=html
```

2. **Code Quality Checks**:
```bash
black src/contexter/rag/
isort src/contexter/rag/
flake8 src/contexter/rag/
mypy src/contexter/rag/
```

3. **Performance Benchmarks**:
```bash
pytest tests/performance/rag/ --benchmark-json=benchmark_results.json
```

4. **Security Scan**:
```bash
bandit -r src/contexter/rag/
safety check
```

This execution guide provides the detailed implementation steps needed for immediate development start. Each task builds incrementally toward the complete RAG system with proper validation at every step.