# Integration Contracts Specification

This document defines comprehensive integration contracts for external services used by the Contexter RAG system, including API specifications, authentication requirements, error handling protocols, and service-level agreements.

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [Voyage AI Integration](#voyage-ai-integration)
3. [Qdrant Integration](#qdrant-integration)
4. [Context7 API Integration](#context7-api-integration)
5. [BrightData Proxy Integration](#brightdata-proxy-integration)
6. [Monitoring Systems Integration](#monitoring-systems-integration)
7. [Authentication and Security](#authentication-and-security)
8. [Error Handling and Recovery](#error-handling-and-recovery)
9. [Performance and SLA Requirements](#performance-and-sla-requirements)
10. [Testing and Validation](#testing-and-validation)

## Integration Architecture

### Integration Patterns

The Contexter RAG system uses several integration patterns to ensure reliability and performance:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx
from datetime import datetime, timedelta

class IntegrationPattern(Enum):
    """Standard integration patterns used across services."""
    DIRECT_API = "direct_api"
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    BATCH_PROCESSING = "batch_processing"
    RATE_LIMITED = "rate_limited"
    CACHED = "cached"

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    base_url: str
    timeout: int
    max_retries: int
    rate_limit: Optional[int] = None
    requires_auth: bool = True
    auth_type: str = "api_key"  # api_key, bearer_token, oauth2

@dataclass
class IntegrationContract:
    """Contract definition for external service integration."""
    service_name: str
    version: str
    endpoint: ServiceEndpoint
    patterns: List[IntegrationPattern]
    sla: Dict[str, Any]
    error_policies: Dict[str, Any]
    monitoring: Dict[str, Any]

class BaseServiceClient(ABC):
    """Abstract base class for all service integrations."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        self.contract = contract
        self.credentials = credentials
        self.client = None
        self.circuit_breaker = None
        self.rate_limiter = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service client."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        pass
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the service."""
        pass
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
```

### Integration Registry

```python
class IntegrationRegistry:
    """Central registry for all service integrations."""
    
    def __init__(self):
        self.contracts = {}
        self.clients = {}
        self.health_status = {}
        
    def register_contract(self, contract: IntegrationContract):
        """Register a service integration contract."""
        self.contracts[contract.service_name] = contract
        
    async def get_client(self, service_name: str) -> BaseServiceClient:
        """Get initialized client for a service."""
        if service_name not in self.clients:
            contract = self.contracts[service_name]
            client = self._create_client(contract)
            await client.initialize()
            self.clients[service_name] = client
            
        return self.clients[service_name]
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all registered services."""
        results = {}
        for service_name, client in self.clients.items():
            try:
                health = await client.health_check()
                results[service_name] = health
            except Exception as e:
                results[service_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        return results
```

## Voyage AI Integration

### Service Contract Definition

```python
VOYAGE_AI_CONTRACT = IntegrationContract(
    service_name="voyage_ai",
    version="v1",
    endpoint=ServiceEndpoint(
        base_url="https://api.voyageai.com/v1",
        timeout=60,
        max_retries=3,
        rate_limit=300,  # requests per minute
        requires_auth=True,
        auth_type="bearer_token"
    ),
    patterns=[
        IntegrationPattern.CIRCUIT_BREAKER,
        IntegrationPattern.RETRY_WITH_BACKOFF,
        IntegrationPattern.BATCH_PROCESSING,
        IntegrationPattern.RATE_LIMITED,
        IntegrationPattern.CACHED
    ],
    sla={
        "availability": 99.9,
        "response_time_p95_ms": 2000,
        "throughput_per_minute": 1000,
        "error_rate_max": 0.1
    },
    error_policies={
        "rate_limit": "exponential_backoff",
        "timeout": "retry_with_circuit_breaker",
        "auth_failure": "refresh_token",
        "server_error": "retry_limited"
    },
    monitoring={
        "health_check_interval": 60,
        "metrics_collection": True,
        "alert_on_failure": True
    }
)
```

### Implementation

```python
class VoyageAIClient(BaseServiceClient):
    """Voyage AI service client implementation."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        super().__init__(contract, credentials)
        self.api_key = credentials.get("api_key")
        self.rate_limiter = TokenBucketRateLimiter(
            rate=contract.endpoint.rate_limit,
            period=60
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        
    async def initialize(self) -> bool:
        """Initialize the Voyage AI client."""
        self.client = httpx.AsyncClient(
            base_url=self.contract.endpoint.base_url,
            timeout=httpx.Timeout(self.contract.endpoint.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        # Verify authentication
        return await self.authenticate()
    
    async def authenticate(self) -> bool:
        """Verify API authentication."""
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid Voyage AI API key")
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "voyage-code-3",
        input_type: str = "document"
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        
        # Rate limiting
        await self.rate_limiter.acquire(len(texts))
        
        # Circuit breaker protection
        return await self.circuit_breaker.call(
            self._generate_embeddings_impl,
            texts, model, input_type
        )
    
    async def _generate_embeddings_impl(
        self,
        texts: List[str],
        model: str,
        input_type: str
    ) -> List[List[float]]:
        """Internal implementation with retry logic."""
        
        payload = {
            "input": texts,
            "model": model,
            "input_type": input_type
        }
        
        for attempt in range(self.contract.endpoint.max_retries):
            try:
                response = await self.client.post("/embeddings", json=payload)
                response.raise_for_status()
                
                data = response.json()
                return [item["embedding"] for item in data["data"]]
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(e.response.headers.get("retry-after", 60))
                    await asyncio.sleep(retry_after + random.uniform(1, 5))
                    continue
                elif e.response.status_code >= 500 and attempt < self.contract.endpoint.max_retries - 1:
                    # Server error - retry with backoff
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                else:
                    raise VoyageAIError(f"API error: {e.response.status_code}")
                    
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < self.contract.endpoint.max_retries - 1:
                    await asyncio.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                else:
                    raise VoyageAIError(f"Connection error: {str(e)}")
        
        raise VoyageAIError("Max retries exceeded")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Voyage AI API."""
        try:
            start_time = datetime.utcnow()
            response = await self.client.get("/models")
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "available_models": response.json().get("data", []),
                "rate_limit_remaining": response.headers.get("x-ratelimit-remaining"),
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

class VoyageAIError(Exception):
    """Voyage AI specific error."""
    pass

class AuthenticationError(VoyageAIError):
    """Authentication failed error."""
    pass
```

### API Specifications

#### Embeddings Endpoint

**Endpoint**: `POST /v1/embeddings`

**Request Schema**:
```json
{
  "input": ["string", "array of strings"],
  "model": "string",
  "input_type": "document|query"
}
```

**Response Schema**:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "voyage-code-3",
  "usage": {
    "total_tokens": 123
  }
}
```

**Error Responses**:
- `400`: Invalid request parameters
- `401`: Invalid API key
- `429`: Rate limit exceeded
- `500`: Internal server error

**Rate Limits**:
- 300 requests per minute
- 1,000,000 tokens per minute
- Batch size limit: 1000 texts per request

## Qdrant Integration

### Service Contract Definition

```python
QDRANT_CONTRACT = IntegrationContract(
    service_name="qdrant",
    version="v1.8",
    endpoint=ServiceEndpoint(
        base_url="http://localhost:6333",
        timeout=30,
        max_retries=3,
        rate_limit=None,  # No rate limiting for local deployment
        requires_auth=False,  # Optional for local deployment
        auth_type="api_key"
    ),
    patterns=[
        IntegrationPattern.CIRCUIT_BREAKER,
        IntegrationPattern.RETRY_WITH_BACKOFF,
        IntegrationPattern.BATCH_PROCESSING
    ],
    sla={
        "availability": 99.95,
        "response_time_p95_ms": 50,
        "throughput_per_second": 1000,
        "error_rate_max": 0.01
    },
    error_policies={
        "connection_error": "retry_with_backoff",
        "timeout": "circuit_breaker",
        "server_error": "retry_limited"
    },
    monitoring={
        "health_check_interval": 30,
        "metrics_collection": True,
        "alert_on_failure": True
    }
)
```

### Implementation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import *
import numpy as np

class QdrantServiceClient(BaseServiceClient):
    """Qdrant vector database client implementation."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        super().__init__(contract, credentials)
        self.api_key = credentials.get("api_key")
        self.collection_name = "contexter_docs"
        
    async def initialize(self) -> bool:
        """Initialize the Qdrant client."""
        
        # Parse URL for host and port
        from urllib.parse import urlparse
        parsed = urlparse(self.contract.endpoint.base_url)
        
        self.client = QdrantClient(
            host=parsed.hostname,
            port=parsed.port or 6333,
            api_key=self.api_key,
            timeout=self.contract.endpoint.timeout,
            prefer_grpc=True
        )
        
        # Initialize collection if it doesn't exist
        await self._ensure_collection_exists()
        
        return True
    
    async def _ensure_collection_exists(self):
        """Ensure the main collection exists with proper configuration."""
        
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=2048,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=200,
                    max_indexing_threads=0
                ),
                optimizers_config=OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=5
                )
            )
            
            # Create payload indexes
            await self._create_payload_indexes()
    
    async def _create_payload_indexes(self):
        """Create indexes on payload fields for efficient filtering."""
        
        index_fields = [
            ("library_id", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("trust_score", PayloadSchemaType.FLOAT),
            ("created_at", PayloadSchemaType.DATETIME)
        ]
        
        for field_name, field_type in index_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Upsert vectors in batches."""
        
        results = {"inserted": 0, "updated": 0, "failed": 0, "errors": []}
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            
            points = []
            for vector_data in batch:
                points.append(PointStruct(
                    id=vector_data["id"],
                    vector=vector_data["vector"],
                    payload=vector_data["payload"]
                ))
            
            try:
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                
                if operation_info.status == UpdateStatus.COMPLETED:
                    results["inserted"] += len(batch)
                else:
                    results["failed"] += len(batch)
                    results["errors"].append(f"Batch {i//batch_size} failed: {operation_info.status}")
                    
            except Exception as e:
                results["failed"] += len(batch)
                results["errors"].append(f"Batch {i//batch_size} error: {str(e)}")
        
        return results
    
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        
        # Build filter conditions
        filter_obj = None
        if filters:
            must_conditions = []
            for field, value in filters.items():
                if isinstance(value, list):
                    # Multiple values - use should (OR)
                    should_conditions = []
                    for v in value:
                        should_conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=v))
                        )
                    must_conditions.append(Filter(should=should_conditions))
                else:
                    # Single value - use must (AND)
                    must_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
            
            if must_conditions:
                filter_obj = Filter(must=must_conditions)
        
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_obj,
                with_payload=True,
                with_vectors=False
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
            
        except Exception as e:
            raise QdrantError(f"Search failed: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant."""
        try:
            start_time = datetime.utcnow()
            
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "collection_status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

class QdrantError(Exception):
    """Qdrant specific error."""
    pass
```

### API Specifications

#### Collection Management

**Create Collection**: `PUT /collections/{collection_name}`
```json
{
  "vectors": {
    "size": 2048,
    "distance": "Cosine"
  },
  "hnsw_config": {
    "m": 16,
    "ef_construct": 200
  }
}
```

#### Vector Operations

**Upsert Points**: `PUT /collections/{collection_name}/points`
```json
{
  "points": [
    {
      "id": "string|integer",
      "vector": [0.1, 0.2, ...],
      "payload": {"key": "value"}
    }
  ]
}
```

**Search**: `POST /collections/{collection_name}/points/search`
```json
{
  "vector": [0.1, 0.2, ...],
  "limit": 10,
  "score_threshold": 0.7,
  "filter": {
    "must": [
      {"key": "library_id", "match": {"value": "fastapi"}}
    ]
  }
}
```

## Context7 API Integration

### Service Contract Definition

```python
CONTEXT7_CONTRACT = IntegrationContract(
    service_name="context7",
    version="v1",
    endpoint=ServiceEndpoint(
        base_url="https://api.context7.dev/v1",
        timeout=300,  # Longer timeout for large documentation
        max_retries=3,
        rate_limit=100,  # requests per hour per IP
        requires_auth=True,
        auth_type="api_key"
    ),
    patterns=[
        IntegrationPattern.CIRCUIT_BREAKER,
        IntegrationPattern.RETRY_WITH_BACKOFF,
        IntegrationPattern.RATE_LIMITED
    ],
    sla={
        "availability": 99.5,
        "response_time_p95_ms": 30000,  # 30 seconds
        "throughput_per_hour": 100,
        "error_rate_max": 0.05
    },
    error_policies={
        "rate_limit": "exponential_backoff",
        "timeout": "retry_with_increased_timeout",
        "token_limit": "split_request"
    },
    monitoring={
        "health_check_interval": 300,
        "metrics_collection": True,
        "alert_on_failure": True
    }
)
```

### Implementation

```python
class Context7Client(BaseServiceClient):
    """Context7 API client implementation."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        super().__init__(contract, credentials)
        self.api_key = credentials.get("api_key")
        self.rate_limiter = TokenBucketRateLimiter(
            rate=contract.endpoint.rate_limit,
            period=3600  # per hour
        )
    
    async def initialize(self) -> bool:
        """Initialize the Context7 client."""
        self.client = httpx.AsyncClient(
            base_url=self.contract.endpoint.base_url,
            timeout=httpx.Timeout(self.contract.endpoint.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        return await self.authenticate()
    
    async def authenticate(self) -> bool:
        """Verify API authentication."""
        try:
            response = await self.client.get("/libraries")
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid Context7 API key")
            raise
    
    async def fetch_documentation(
        self,
        library_id: str,
        contexts: List[str],
        token_limit: int = 200000
    ) -> Dict[str, Any]:
        """Fetch documentation for a library with multiple contexts."""
        
        # Rate limiting
        await self.rate_limiter.acquire(len(contexts))
        
        results = {}
        
        for context in contexts:
            try:
                doc_content = await self._fetch_single_context(
                    library_id, context, token_limit
                )
                results[context] = doc_content
                
            except Context7Error as e:
                results[context] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        return results
    
    async def _fetch_single_context(
        self,
        library_id: str,
        context: str,
        token_limit: int
    ) -> Dict[str, Any]:
        """Fetch documentation for a single context."""
        
        payload = {
            "library_id": library_id,
            "context": context,
            "token_limit": token_limit,
            "format": "structured"
        }
        
        for attempt in range(self.contract.endpoint.max_retries):
            try:
                response = await self.client.post("/fetch", json=payload)
                response.raise_for_status()
                
                return response.json()
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited
                    retry_after = int(e.response.headers.get("retry-after", 3600))
                    await asyncio.sleep(retry_after)
                    continue
                elif e.response.status_code == 413:
                    # Token limit exceeded - reduce limit and retry
                    if token_limit > 50000:
                        token_limit = token_limit // 2
                        payload["token_limit"] = token_limit
                        continue
                    else:
                        raise Context7Error("Token limit too low")
                elif e.response.status_code >= 500 and attempt < self.contract.endpoint.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    raise Context7Error(f"API error: {e.response.status_code}")
                    
            except httpx.TimeoutException:
                if attempt < self.contract.endpoint.max_retries - 1:
                    # Increase timeout for retry
                    self.client.timeout = httpx.Timeout(
                        self.client.timeout.read * 1.5
                    )
                    continue
                else:
                    raise Context7Error("Request timeout")
        
        raise Context7Error("Max retries exceeded")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Context7 API."""
        try:
            start_time = datetime.utcnow()
            response = await self.client.get("/health")
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "rate_limit_remaining": response.headers.get("x-ratelimit-remaining"),
                "timestamp": start_time.isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

class Context7Error(Exception):
    """Context7 specific error."""
    pass
```

## BrightData Proxy Integration

### Service Contract Definition

```python
BRIGHTDATA_CONTRACT = IntegrationContract(
    service_name="brightdata",
    version="v1",
    endpoint=ServiceEndpoint(
        base_url="http://brd.superproxy.io:33335",
        timeout=30,
        max_retries=3,
        rate_limit=None,  # No API rate limit, but connection limits apply
        requires_auth=True,
        auth_type="basic_auth"
    ),
    patterns=[
        IntegrationPattern.CIRCUIT_BREAKER,
        IntegrationPattern.RETRY_WITH_BACKOFF
    ],
    sla={
        "availability": 99.0,
        "response_time_p95_ms": 5000,
        "success_rate": 95.0,
        "error_rate_max": 0.05
    },
    error_policies={
        "connection_error": "rotate_proxy",
        "timeout": "retry_with_different_proxy",
        "auth_failure": "refresh_credentials"
    },
    monitoring={
        "health_check_interval": 120,
        "metrics_collection": True,
        "alert_on_failure": True
    }
)
```

### Implementation

```python
class BrightDataProxyClient(BaseServiceClient):
    """BrightData proxy service client implementation."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        super().__init__(contract, credentials)
        self.username = credentials.get("username")
        self.password = credentials.get("password")
        self.zone = credentials.get("zone", "residential")
        self.proxy_pool = []
        self.current_proxy_index = 0
        
    async def initialize(self) -> bool:
        """Initialize the BrightData proxy client."""
        
        # Create proxy pool with session IDs
        for i in range(10):  # Pool of 10 proxy sessions
            session_id = f"session_{i}_{int(time.time())}"
            proxy_url = f"http://{self.username}-session-{session_id}:{self.password}@{self.contract.endpoint.base_url.replace('http://', '')}"
            
            self.proxy_pool.append({
                "url": proxy_url,
                "session_id": session_id,
                "active": True,
                "failures": 0,
                "last_used": None
            })
        
        return await self.authenticate()
    
    async def authenticate(self) -> bool:
        """Test proxy authentication."""
        try:
            proxy = self._get_next_proxy()
            async with httpx.AsyncClient(proxies={"http://": proxy["url"], "https://": proxy["url"]}) as client:
                response = await client.get("http://httpbin.org/ip", timeout=30)
                response.raise_for_status()
                return True
        except Exception:
            return False
    
    def _get_next_proxy(self) -> Dict[str, Any]:
        """Get next available proxy using round-robin."""
        
        # Filter active proxies
        active_proxies = [p for p in self.proxy_pool if p["active"]]
        
        if not active_proxies:
            # Reactivate all proxies if none are active
            for proxy in self.proxy_pool:
                proxy["active"] = True
                proxy["failures"] = 0
            active_proxies = self.proxy_pool
        
        # Round-robin selection
        proxy = active_proxies[self.current_proxy_index % len(active_proxies)]
        self.current_proxy_index += 1
        
        proxy["last_used"] = datetime.utcnow()
        return proxy
    
    async def get_session(self, timeout: int = 30) -> httpx.AsyncClient:
        """Get HTTP client session with proxy."""
        
        proxy = self._get_next_proxy()
        
        try:
            client = httpx.AsyncClient(
                proxies={
                    "http://": proxy["url"],
                    "https://": proxy["url"]
                },
                timeout=httpx.Timeout(timeout),
                headers={
                    "User-Agent": self._get_random_user_agent()
                }
            )
            
            return client
            
        except Exception as e:
            # Mark proxy as failed
            proxy["failures"] += 1
            if proxy["failures"] >= 3:
                proxy["active"] = False
            
            raise BrightDataError(f"Proxy connection failed: {str(e)}")
    
    def _get_random_user_agent(self) -> str:
        """Get random user agent string."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ]
        import random
        return random.choice(user_agents)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on proxy service."""
        try:
            start_time = datetime.utcnow()
            
            async with await self.get_session() as client:
                response = await client.get("http://httpbin.org/ip")
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                response.raise_for_status()
                ip_info = response.json()
                
                active_proxies = len([p for p in self.proxy_pool if p["active"]])
                
                return {
                    "status": "healthy",
                    "response_time_ms": response_time,
                    "current_ip": ip_info.get("origin"),
                    "active_proxies": active_proxies,
                    "total_proxies": len(self.proxy_pool),
                    "timestamp": start_time.isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "active_proxies": len([p for p in self.proxy_pool if p["active"]]),
                "timestamp": datetime.utcnow().isoformat()
            }

class BrightDataError(Exception):
    """BrightData specific error."""
    pass
```

## Monitoring Systems Integration

### Prometheus Integration

```python
PROMETHEUS_CONTRACT = IntegrationContract(
    service_name="prometheus",
    version="v1",
    endpoint=ServiceEndpoint(
        base_url="http://localhost:9090",
        timeout=10,
        max_retries=2,
        requires_auth=False
    ),
    patterns=[IntegrationPattern.DIRECT_API],
    sla={
        "availability": 99.9,
        "response_time_p95_ms": 1000
    },
    error_policies={
        "connection_error": "log_and_continue"
    },
    monitoring={
        "health_check_interval": 60,
        "metrics_collection": False  # Avoid circular dependency
    }
)

class PrometheusClient(BaseServiceClient):
    """Prometheus metrics client."""
    
    def __init__(self, contract: IntegrationContract, credentials: Dict[str, str]):
        super().__init__(contract, credentials)
        self.metrics_registry = {}
        
    async def initialize(self) -> bool:
        """Initialize Prometheus client."""
        self.client = httpx.AsyncClient(
            base_url=self.contract.endpoint.base_url,
            timeout=httpx.Timeout(self.contract.endpoint.timeout)
        )
        return True
    
    async def push_metrics(self, job_name: str, metrics: Dict[str, Any]) -> bool:
        """Push metrics to Prometheus pushgateway."""
        
        # Convert metrics to Prometheus format
        prometheus_data = self._format_metrics(metrics)
        
        try:
            response = await self.client.post(
                f"/metrics/job/{job_name}",
                content=prometheus_data,
                headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
            )
            response.raise_for_status()
            return True
            
        except Exception as e:
            # Log error but don't fail the application
            print(f"Failed to push metrics: {e}")
            return False
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics in Prometheus exposition format."""
        
        lines = []
        
        for metric_name, metric_data in metrics.items():
            # Add help text
            if "help" in metric_data:
                lines.append(f"# HELP {metric_name} {metric_data['help']}")
            
            # Add type
            if "type" in metric_data:
                lines.append(f"# TYPE {metric_name} {metric_data['type']}")
            
            # Add metric value with labels
            labels = metric_data.get("labels", {})
            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
            if label_str:
                lines.append(f"{metric_name}{{{label_str}}} {metric_data['value']}")
            else:
                lines.append(f"{metric_name} {metric_data['value']}")
        
        return "\n".join(lines) + "\n"
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Prometheus health."""
        try:
            response = await self.client.get("/-/healthy")
            response.raise_for_status()
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Authentication and Security

### Authentication Manager

```python
class AuthenticationManager:
    """Centralized authentication management for all services."""
    
    def __init__(self):
        self.credentials = {}
        self.token_cache = {}
        self.refresh_tokens = {}
        
    def register_credentials(self, service_name: str, auth_data: Dict[str, str]):
        """Register credentials for a service."""
        self.credentials[service_name] = auth_data
    
    async def get_auth_headers(self, service_name: str) -> Dict[str, str]:
        """Get authentication headers for a service."""
        
        if service_name not in self.credentials:
            raise AuthenticationError(f"No credentials registered for {service_name}")
        
        auth_data = self.credentials[service_name]
        auth_type = auth_data.get("type", "api_key")
        
        if auth_type == "api_key":
            return {"Authorization": f"Bearer {auth_data['api_key']}"}
        elif auth_type == "basic":
            import base64
            credentials = f"{auth_data['username']}:{auth_data['password']}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return {"Authorization": f"Basic {encoded}"}
        elif auth_type == "jwt":
            token = await self._get_or_refresh_jwt_token(service_name)
            return {"Authorization": f"Bearer {token}"}
        else:
            raise AuthenticationError(f"Unsupported auth type: {auth_type}")
    
    async def _get_or_refresh_jwt_token(self, service_name: str) -> str:
        """Get or refresh JWT token for a service."""
        
        # Check if we have a valid cached token
        if service_name in self.token_cache:
            token_data = self.token_cache[service_name]
            if token_data["expires_at"] > datetime.utcnow():
                return token_data["token"]
        
        # Refresh token
        auth_data = self.credentials[service_name]
        
        # Implementation depends on service-specific token refresh logic
        # This is a placeholder for JWT token refresh
        new_token = await self._refresh_jwt_token(service_name, auth_data)
        
        # Cache the new token
        self.token_cache[service_name] = {
            "token": new_token,
            "expires_at": datetime.utcnow() + timedelta(hours=1)
        }
        
        return new_token
    
    async def _refresh_jwt_token(self, service_name: str, auth_data: Dict[str, str]) -> str:
        """Refresh JWT token (service-specific implementation)."""
        # Placeholder - implement service-specific token refresh
        raise NotImplementedError("JWT token refresh not implemented")

class SecurityManager:
    """Security utilities for service integrations."""
    
    @staticmethod
    def validate_tls_certificate(hostname: str, cert_data: bytes) -> bool:
        """Validate TLS certificate for a service."""
        # Implementation for certificate validation
        pass
    
    @staticmethod
    def encrypt_credentials(credentials: Dict[str, str], key: str) -> str:
        """Encrypt credentials for secure storage."""
        # Implementation for credential encryption
        pass
    
    @staticmethod
    def decrypt_credentials(encrypted_data: str, key: str) -> Dict[str, str]:
        """Decrypt stored credentials."""
        # Implementation for credential decryption
        pass
```

## Error Handling and Recovery

### Comprehensive Error Handling Framework

```python
from enum import Enum
from typing import Optional, Callable, Any
import asyncio
import random

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"

@dataclass
class ErrorPolicy:
    """Error handling policy definition."""
    error_types: List[type]
    severity: ErrorSeverity
    strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    circuit_breaker_threshold: int = 5
    recovery_timeout: int = 60

class IntegratedErrorHandler:
    """Unified error handling for all service integrations."""
    
    def __init__(self):
        self.error_policies = {}
        self.circuit_breakers = {}
        self.error_stats = {}
        
    def register_policy(self, service_name: str, policy: ErrorPolicy):
        """Register error handling policy for a service."""
        self.error_policies[service_name] = policy
        
        if policy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=policy.circuit_breaker_threshold,
                recovery_timeout=policy.recovery_timeout
            )
    
    async def handle_error(
        self,
        service_name: str,
        error: Exception,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Handle error according to registered policy."""
        
        policy = self.error_policies.get(service_name)
        if not policy:
            raise error  # No policy registered, re-raise
        
        # Check if error type matches policy
        if not any(isinstance(error, error_type) for error_type in policy.error_types):
            raise error  # Error type not covered by policy
        
        # Update error statistics
        self._update_error_stats(service_name, error, policy.severity)
        
        # Apply recovery strategy
        if policy.strategy == RecoveryStrategy.RETRY:
            return await self._retry_with_backoff(
                operation, policy.max_retries, policy.backoff_multiplier, *args, **kwargs
            )
        elif policy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            circuit_breaker = self.circuit_breakers[service_name]
            return await circuit_breaker.call(operation, *args, **kwargs)
        elif policy.strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_fallback(service_name, operation, *args, **kwargs)
        elif policy.strategy == RecoveryStrategy.ESCALATE:
            await self._escalate_error(service_name, error, policy.severity)
            raise error
        elif policy.strategy == RecoveryStrategy.IGNORE:
            await self._log_ignored_error(service_name, error)
            return None
        else:
            raise error
    
    async def _retry_with_backoff(
        self,
        operation: Callable,
        max_retries: int,
        backoff_multiplier: float,
        *args,
        **kwargs
    ) -> Any:
        """Retry operation with exponential backoff."""
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    break
                
                # Exponential backoff with jitter
                delay = (backoff_multiplier ** attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
        
        raise last_exception
    
    async def _execute_fallback(
        self,
        service_name: str,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback operation."""
        # Implementation depends on service-specific fallback logic
        # This could involve using cached data, alternative services, etc.
        pass
    
    async def _escalate_error(
        self,
        service_name: str,
        error: Exception,
        severity: ErrorSeverity
    ):
        """Escalate error to monitoring and alerting systems."""
        
        alert_data = {
            "service": service_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to monitoring system
        # Implementation depends on monitoring setup
        pass
    
    def _update_error_stats(
        self,
        service_name: str,
        error: Exception,
        severity: ErrorSeverity
    ):
        """Update error statistics for monitoring."""
        
        if service_name not in self.error_stats:
            self.error_stats[service_name] = {
                "total_errors": 0,
                "by_type": {},
                "by_severity": {s.value: 0 for s in ErrorSeverity},
                "last_error": None
            }
        
        stats = self.error_stats[service_name]
        stats["total_errors"] += 1
        
        error_type = type(error).__name__
        stats["by_type"][error_type] = stats["by_type"].get(error_type, 0) + 1
        stats["by_severity"][severity.value] += 1
        stats["last_error"] = datetime.utcnow().isoformat()
    
    def get_error_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if service_name:
            return self.error_stats.get(service_name, {})
        return self.error_stats
```

## Performance and SLA Requirements

### Performance Monitoring and SLA Enforcement

```python
class PerformanceMonitor:
    """Monitor and enforce SLA requirements for service integrations."""
    
    def __init__(self):
        self.performance_data = {}
        self.sla_violations = {}
        
    async def record_operation(
        self,
        service_name: str,
        operation: str,
        duration_ms: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record performance data for an operation."""
        
        if service_name not in self.performance_data:
            self.performance_data[service_name] = {}
        
        if operation not in self.performance_data[service_name]:
            self.performance_data[service_name][operation] = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration_ms": 0,
                "min_duration_ms": float('inf'),
                "max_duration_ms": 0,
                "response_times": [],
                "last_call": None
            }
        
        data = self.performance_data[service_name][operation]
        data["total_calls"] += 1
        
        if success:
            data["successful_calls"] += 1
        else:
            data["failed_calls"] += 1
        
        data["total_duration_ms"] += duration_ms
        data["min_duration_ms"] = min(data["min_duration_ms"], duration_ms)
        data["max_duration_ms"] = max(data["max_duration_ms"], duration_ms)
        data["response_times"].append(duration_ms)
        data["last_call"] = datetime.utcnow().isoformat()
        
        # Keep only last 1000 response times for percentile calculation
        if len(data["response_times"]) > 1000:
            data["response_times"] = data["response_times"][-1000:]
        
        # Check SLA compliance
        await self._check_sla_compliance(service_name, operation, data)
    
    async def _check_sla_compliance(
        self,
        service_name: str,
        operation: str,
        performance_data: Dict[str, Any]
    ):
        """Check if performance meets SLA requirements."""
        
        # Get SLA requirements for service
        # This would typically come from the service contract
        sla_requirements = self._get_sla_requirements(service_name)
        
        if not sla_requirements:
            return
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(performance_data)
        
        # Check each SLA requirement
        violations = []
        
        if "response_time_p95_ms" in sla_requirements:
            if current_metrics["p95_ms"] > sla_requirements["response_time_p95_ms"]:
                violations.append({
                    "metric": "response_time_p95_ms",
                    "threshold": sla_requirements["response_time_p95_ms"],
                    "actual": current_metrics["p95_ms"]
                })
        
        if "error_rate_max" in sla_requirements:
            if current_metrics["error_rate"] > sla_requirements["error_rate_max"]:
                violations.append({
                    "metric": "error_rate_max",
                    "threshold": sla_requirements["error_rate_max"],
                    "actual": current_metrics["error_rate"]
                })
        
        if violations:
            await self._handle_sla_violations(service_name, operation, violations)
    
    def _calculate_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from raw data."""
        
        response_times = performance_data["response_times"]
        
        if not response_times:
            return {}
        
        response_times.sort()
        n = len(response_times)
        
        return {
            "avg_ms": sum(response_times) / n,
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "p50_ms": response_times[int(n * 0.5)],
            "p95_ms": response_times[int(n * 0.95)],
            "p99_ms": response_times[int(n * 0.99)],
            "error_rate": performance_data["failed_calls"] / performance_data["total_calls"]
        }
    
    async def _handle_sla_violations(
        self,
        service_name: str,
        operation: str,
        violations: List[Dict[str, Any]]
    ):
        """Handle SLA violations with appropriate actions."""
        
        violation_key = f"{service_name}:{operation}"
        
        if violation_key not in self.sla_violations:
            self.sla_violations[violation_key] = {
                "first_violation": datetime.utcnow(),
                "violation_count": 0,
                "consecutive_violations": 0
            }
        
        violation_data = self.sla_violations[violation_key]
        violation_data["violation_count"] += 1
        violation_data["consecutive_violations"] += 1
        violation_data["last_violation"] = datetime.utcnow()
        violation_data["violations"] = violations
        
        # Trigger alerts based on violation severity
        if violation_data["consecutive_violations"] >= 3:
            await self._trigger_sla_alert(service_name, operation, violation_data)
    
    async def _trigger_sla_alert(
        self,
        service_name: str,
        operation: str,
        violation_data: Dict[str, Any]
    ):
        """Trigger alert for SLA violations."""
        
        alert_data = {
            "service": service_name,
            "operation": operation,
            "violation_count": violation_data["violation_count"],
            "consecutive_violations": violation_data["consecutive_violations"],
            "violations": violation_data["violations"],
            "first_violation": violation_data["first_violation"].isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send alert to monitoring system
        # Implementation depends on alerting setup
        pass
```

## Testing and Validation

### Integration Testing Framework

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

class IntegrationTestSuite:
    """Comprehensive integration testing for all services."""
    
    @pytest.fixture
    async def mock_voyage_client(self):
        """Mock Voyage AI client for testing."""
        client = AsyncMock(spec=VoyageAIClient)
        client.generate_embeddings.return_value = [
            [0.1] * 2048 for _ in range(10)  # Mock embeddings
        ]
        client.health_check.return_value = {
            "status": "healthy",
            "response_time_ms": 150
        }
        return client
    
    @pytest.fixture
    async def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        client = AsyncMock(spec=QdrantServiceClient)
        client.search_vectors.return_value = [
            {
                "id": f"chunk_{i}",
                "score": 0.9 - (i * 0.1),
                "payload": {"content": f"Test content {i}"}
            }
            for i in range(5)
        ]
        client.health_check.return_value = {
            "status": "healthy",
            "vectors_count": 10000
        }
        return client
    
    async def test_voyage_ai_integration(self, mock_voyage_client):
        """Test Voyage AI integration with various scenarios."""
        
        # Test successful embedding generation
        texts = ["Test text 1", "Test text 2"]
        embeddings = await mock_voyage_client.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) == 2048 for emb in embeddings)
        
        # Test health check
        health = await mock_voyage_client.health_check()
        assert health["status"] == "healthy"
    
    async def test_qdrant_integration(self, mock_qdrant_client):
        """Test Qdrant integration with various scenarios."""
        
        # Test vector search
        query_vector = [0.1] * 2048
        results = await mock_qdrant_client.search_vectors(query_vector, limit=5)
        
        assert len(results) == 5
        assert all("score" in result for result in results)
        assert all(result["score"] <= 1.0 for result in results)
        
        # Test health check
        health = await mock_qdrant_client.health_check()
        assert health["status"] == "healthy"
    
    async def test_error_handling(self):
        """Test error handling scenarios."""
        
        # Test rate limiting
        with pytest.raises(VoyageAIError):
            async with httpx.AsyncClient() as client:
                # Simulate rate limit response
                response = httpx.Response(
                    status_code=429,
                    headers={"retry-after": "60"}
                )
                # Test rate limit handling logic
    
    async def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        # Simulate failures to trigger circuit breaker
        for _ in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(self._failing_operation)
        
        # Circuit breaker should be open now
        assert circuit_breaker.state == "OPEN"
        
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(self._failing_operation)
    
    async def _failing_operation(self):
        """Helper method that always fails."""
        raise Exception("Simulated failure")
    
    async def test_performance_monitoring(self):
        """Test performance monitoring and SLA enforcement."""
        
        monitor = PerformanceMonitor()
        
        # Record successful operations
        for i in range(10):
            await monitor.record_operation(
                "test_service",
                "test_operation",
                duration_ms=50 + i,
                success=True
            )
        
        # Record some failures
        for i in range(2):
            await monitor.record_operation(
                "test_service",
                "test_operation",
                duration_ms=100,
                success=False
            )
        
        # Check metrics calculation
        metrics = monitor._calculate_metrics(
            monitor.performance_data["test_service"]["test_operation"]
        )
        
        assert metrics["error_rate"] == 2/12  # 2 failures out of 12 total
        assert metrics["avg_ms"] > 50

# Contract validation tests
class ContractValidationTests:
    """Validate that service implementations meet their contracts."""
    
    def test_voyage_ai_contract_compliance(self):
        """Ensure Voyage AI client meets contract requirements."""
        
        contract = VOYAGE_AI_CONTRACT
        
        # Verify required patterns are supported
        assert IntegrationPattern.RATE_LIMITED in contract.patterns
        assert IntegrationPattern.CIRCUIT_BREAKER in contract.patterns
        assert IntegrationPattern.CACHED in contract.patterns
        
        # Verify SLA requirements are reasonable
        assert contract.sla["response_time_p95_ms"] <= 5000
        assert contract.sla["error_rate_max"] <= 0.1
    
    def test_qdrant_contract_compliance(self):
        """Ensure Qdrant client meets contract requirements."""
        
        contract = QDRANT_CONTRACT
        
        # Verify performance requirements
        assert contract.sla["response_time_p95_ms"] <= 100
        assert contract.sla["availability"] >= 99.9
        
        # Verify error policies are defined
        assert "connection_error" in contract.error_policies
        assert "timeout" in contract.error_policies
```

### Service Validation Framework

```python
class ServiceValidator:
    """Validate service integrations against their contracts."""
    
    def __init__(self, integration_registry: IntegrationRegistry):
        self.registry = integration_registry
        
    async def validate_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Validate all registered services."""
        
        results = {}
        
        for service_name, contract in self.registry.contracts.items():
            try:
                client = await self.registry.get_client(service_name)
                validation_result = await self._validate_service(client, contract)
                results[service_name] = validation_result
            except Exception as e:
                results[service_name] = {
                    "valid": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        return results
    
    async def _validate_service(
        self,
        client: BaseServiceClient,
        contract: IntegrationContract
    ) -> Dict[str, Any]:
        """Validate a single service against its contract."""
        
        validation_result = {
            "valid": True,
            "checks": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Health check validation
        health = await client.health_check()
        validation_result["checks"]["health_check"] = {
            "passed": health.get("status") == "healthy",
            "details": health
        }
        
        if health.get("status") != "healthy":
            validation_result["valid"] = False
        
        # Performance validation
        response_time = health.get("response_time_ms", 0)
        max_response_time = contract.sla.get("response_time_p95_ms", float('inf'))
        
        validation_result["checks"]["performance"] = {
            "passed": response_time <= max_response_time,
            "actual_ms": response_time,
            "threshold_ms": max_response_time
        }
        
        if response_time > max_response_time:
            validation_result["valid"] = False
        
        # Authentication validation
        try:
            auth_result = await client.authenticate()
            validation_result["checks"]["authentication"] = {
                "passed": auth_result,
                "details": "Authentication successful" if auth_result else "Authentication failed"
            }
            
            if not auth_result:
                validation_result["valid"] = False
                
        except Exception as e:
            validation_result["checks"]["authentication"] = {
                "passed": False,
                "details": f"Authentication error: {str(e)}"
            }
            validation_result["valid"] = False
        
        return validation_result
```

This comprehensive integration contracts specification provides:

1. **Standardized Integration Patterns**: Common patterns for all service integrations
2. **Service-Specific Implementations**: Detailed implementations for each external service
3. **Authentication and Security**: Centralized authentication management
4. **Error Handling and Recovery**: Comprehensive error handling with recovery strategies
5. **Performance Monitoring**: SLA enforcement and performance tracking
6. **Testing Framework**: Complete testing and validation infrastructure

These contracts ensure reliable, performant, and secure integration with all external services while providing clear implementation guidance and monitoring capabilities.