# API Integration Strategies

This document defines comprehensive strategies for integrating external APIs and services within the Contexter RAG system, focusing on reliability, performance, and security.

## Table of Contents

1. [API Integration Patterns](#api-integration-patterns)
2. [External Service Integrations](#external-service-integrations)
3. [Authentication Strategies](#authentication-strategies)
4. [Rate Limiting and Throttling](#rate-limiting-and-throttling)
5. [Error Handling and Recovery](#error-handling-and-recovery)
6. [Caching Strategies](#caching-strategies)
7. [API Versioning and Compatibility](#api-versioning-and-compatibility)
8. [Monitoring and Observability](#monitoring-and-observability)

## API Integration Patterns

### 1. Unified API Client Pattern

Standardized approach for all external API integrations with consistent error handling and monitoring.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import httpx
import asyncio
import time
from datetime import datetime, timedelta

class APIMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class APIRequest:
    """Standardized API request structure."""
    method: APIMethod
    endpoint: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    json_data: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retries: int = 3

@dataclass
class APIResponse:
    """Standardized API response structure."""
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    response_time_ms: float
    request_id: str
    cached: bool = False

class APIClient(ABC):
    """Base class for all API integrations."""
    
    def __init__(
        self,
        base_url: str,
        service_name: str,
        auth_handler: Optional['AuthHandler'] = None,
        rate_limiter: Optional['RateLimiter'] = None,
        cache: Optional['APICache'] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.auth_handler = auth_handler
        self.rate_limiter = rate_limiter
        self.cache = cache
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            headers={"User-Agent": "Contexter-RAG/1.0"}
        )
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.last_request_time: Optional[datetime] = None
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """Execute API request with full integration pattern."""
        
        # Generate request ID for tracing
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache and request.method == APIMethod.GET:
                cached_response = await self.cache.get(request)
                if cached_response:
                    return cached_response
            
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            
            # Authentication
            headers = request.headers or {}
            if self.auth_handler:
                auth_headers = await self.auth_handler.get_headers()
                headers.update(auth_headers)
            
            # Execute with retries
            response = await self._execute_with_retry(request, headers, request_id)
            
            # Cache successful GET responses
            if self.cache and request.method == APIMethod.GET and response.status_code == 200:
                await self.cache.set(request, response)
            
            # Update metrics
            self._update_metrics(response.response_time_ms, success=True)
            
            return response
            
        except Exception as e:
            error_time = (time.time() - start_time) * 1000
            self._update_metrics(error_time, success=False)
            raise APIIntegrationError(
                f"API request failed for {self.service_name}",
                service=self.service_name,
                request_id=request_id,
                original_error=e
            )
    
    async def _execute_with_retry(
        self,
        request: APIRequest,
        headers: Dict[str, str],
        request_id: str
    ) -> APIResponse:
        """Execute request with retry logic."""
        
        last_exception = None
        
        for attempt in range(request.retries + 1):
            try:
                start_time = time.time()
                
                # Prepare request
                req_kwargs = {
                    "headers": headers,
                    "timeout": request.timeout
                }
                
                if request.params:
                    req_kwargs["params"] = request.params
                if request.json_data:
                    req_kwargs["json"] = request.json_data
                
                # Execute HTTP request
                response = await self.client.request(
                    method=request.method.value,
                    url=request.endpoint,
                    **req_kwargs
                )
                
                response_time = (time.time() - start_time) * 1000
                
                # Handle HTTP errors
                if response.status_code >= 400:
                    if response.status_code == 429:  # Rate limited
                        if attempt < request.retries:
                            await self._handle_rate_limit(response)
                            continue
                    elif response.status_code >= 500:  # Server error
                        if attempt < request.retries:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    
                    raise HTTPError(
                        f"HTTP {response.status_code}: {response.text}",
                        status_code=response.status_code,
                        response_text=response.text
                    )
                
                # Success
                return APIResponse(
                    status_code=response.status_code,
                    data=response.json() if response.content else {},
                    headers=dict(response.headers),
                    response_time_ms=response_time,
                    request_id=request_id
                )
                
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < request.retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except HTTPError:
                raise
            except Exception as e:
                last_exception = e
                break
        
        raise last_exception or APIIntegrationError("Request failed after retries")
    
    async def _handle_rate_limit(self, response: httpx.Response):
        """Handle rate limiting with retry-after header."""
        retry_after = response.headers.get("retry-after", "60")
        try:
            wait_time = int(retry_after)
        except ValueError:
            wait_time = 60
        
        # Add jitter to prevent thundering herd
        import random
        wait_time += random.uniform(1, 5)
        await asyncio.sleep(wait_time)
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return f"{self.service_name}_{uuid.uuid4().hex[:8]}"
    
    def _update_metrics(self, response_time_ms: float, success: bool):
        """Update performance metrics."""
        self.request_count += 1
        self.total_response_time += response_time_ms
        self.last_request_time = datetime.utcnow()
        
        if not success:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "service_name": self.service_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "avg_response_time_ms": self.total_response_time / max(self.request_count, 1),
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            request = APIRequest(
                method=APIMethod.GET,
                endpoint="/health",
                timeout=10.0,
                retries=1
            )
            
            response = await self.execute_request(request)
            
            return {
                "status": "healthy",
                "response_time_ms": response.response_time_ms,
                "details": response.data
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

class APIIntegrationError(Exception):
    """API integration specific error."""
    
    def __init__(self, message: str, service: str = None, request_id: str = None, original_error: Exception = None):
        super().__init__(message)
        self.service = service
        self.request_id = request_id
        self.original_error = original_error

class HTTPError(APIIntegrationError):
    """HTTP specific error."""
    
    def __init__(self, message: str, status_code: int, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
```

### 2. GraphQL Integration Pattern

For services that provide GraphQL APIs with optimized query building.

```python
from typing import Dict, Any, List, Optional

class GraphQLClient(APIClient):
    """GraphQL-specific API client."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoint = "/graphql"
    
    async def execute_query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        operation_name: Optional[str] = None
    ) -> APIResponse:
        """Execute GraphQL query."""
        
        request = APIRequest(
            method=APIMethod.POST,
            endpoint=self.endpoint,
            json_data={
                "query": query,
                "variables": variables or {},
                "operationName": operation_name
            }
        )
        
        response = await self.execute_request(request)
        
        # Handle GraphQL errors
        if "errors" in response.data:
            raise GraphQLError(
                "GraphQL query failed",
                errors=response.data["errors"]
            )
        
        return response
    
    async def execute_mutation(
        self,
        mutation: str,
        variables: Dict[str, Any]
    ) -> APIResponse:
        """Execute GraphQL mutation."""
        return await self.execute_query(mutation, variables)

class GraphQLError(APIIntegrationError):
    """GraphQL specific error."""
    
    def __init__(self, message: str, errors: List[Dict[str, Any]]):
        super().__init__(message)
        self.errors = errors
```

## External Service Integrations

### 1. Voyage AI Integration

Specialized integration for embedding generation with batch optimization.

```python
import numpy as np
from typing import List, Dict, Any, Optional

class VoyageAIClient(APIClient):
    """Voyage AI API client for embedding generation."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            base_url="https://api.voyageai.com/v1",
            service_name="voyage_ai",
            **kwargs
        )
        self.api_key = api_key
        
        # Voyage AI specific configuration
        self.max_batch_size = 1000
        self.default_model = "voyage-code-3"
        self.input_types = {"document", "query"}
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = None,
        input_type: str = "document",
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for list of texts."""
        
        model = model or self.default_model
        batch_size = batch_size or min(len(texts), self.max_batch_size)
        
        if input_type not in self.input_types:
            raise ValueError(f"Invalid input_type. Must be one of: {self.input_types}")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            request = APIRequest(
                method=APIMethod.POST,
                endpoint="/embeddings",
                json_data={
                    "input": batch_texts,
                    "model": model,
                    "input_type": input_type
                },
                timeout=60.0  # Longer timeout for embedding generation
            )
            
            response = await self.execute_request(request)
            
            # Extract embeddings from response
            embeddings = [
                np.array(item["embedding"], dtype=np.float32)
                for item in response.data["data"]
            ]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """Get available models."""
        request = APIRequest(
            method=APIMethod.GET,
            endpoint="/models"
        )
        
        response = await self.execute_request(request)
        return response.data.get("data", [])
    
    async def health_check(self) -> Dict[str, Any]:
        """Voyage AI specific health check."""
        try:
            # Test with a simple embedding request
            test_embedding = await self.generate_embeddings(
                ["test"],
                input_type="query"
            )
            
            return {
                "status": "healthy",
                "embedding_dimension": len(test_embedding[0]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

# Authentication handler for Voyage AI
class VoyageAIAuthHandler:
    """Authentication handler for Voyage AI."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
```

### 2. Context7 Integration

Integration for documentation fetching with intelligent context management.

```python
class Context7Client(APIClient):
    """Context7 API client for documentation fetching."""
    
    def __init__(self, api_key: str, **kwargs):
        super().__init__(
            base_url="https://api.context7.dev/v1",
            service_name="context7",
            **kwargs
        )
        self.api_key = api_key
        
        # Context7 specific configuration
        self.max_token_limit = 200000
        self.default_format = "structured"
    
    async def fetch_documentation(
        self,
        library_id: str,
        contexts: List[str],
        token_limit: Optional[int] = None,
        format_type: str = None
    ) -> Dict[str, Any]:
        """Fetch documentation for library with multiple contexts."""
        
        token_limit = token_limit or self.max_token_limit
        format_type = format_type or self.default_format
        
        results = {}
        
        for context in contexts:
            try:
                doc_data = await self._fetch_single_context(
                    library_id, context, token_limit, format_type
                )
                results[context] = doc_data
                
            except Exception as e:
                results[context] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        return results
    
    async def _fetch_single_context(
        self,
        library_id: str,
        context: str,
        token_limit: int,
        format_type: str
    ) -> Dict[str, Any]:
        """Fetch documentation for single context."""
        
        request = APIRequest(
            method=APIMethod.POST,
            endpoint="/fetch",
            json_data={
                "library_id": library_id,
                "context": context,
                "token_limit": token_limit,
                "format": format_type
            },
            timeout=300.0  # 5 minutes for large documentation
        )
        
        response = await self.execute_request(request)
        return response.data
    
    async def list_libraries(self) -> List[Dict[str, Any]]:
        """List available libraries."""
        request = APIRequest(
            method=APIMethod.GET,
            endpoint="/libraries"
        )
        
        response = await self.execute_request(request)
        return response.data.get("libraries", [])
    
    async def get_library_info(self, library_id: str) -> Dict[str, Any]:
        """Get information about specific library."""
        request = APIRequest(
            method=APIMethod.GET,
            endpoint=f"/libraries/{library_id}"
        )
        
        response = await self.execute_request(request)
        return response.data

class Context7AuthHandler:
    """Authentication handler for Context7."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
```

### 3. Qdrant Integration

Integration for vector database operations with connection pooling.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np

class QdrantAPIClient(APIClient):
    """Qdrant API client for vector operations."""
    
    def __init__(self, host: str = "localhost", port: int = 6333, **kwargs):
        super().__init__(
            base_url=f"http://{host}:{port}",
            service_name="qdrant",
            **kwargs
        )
        
        # Direct Qdrant client for GRPC operations
        self.qdrant_client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=True
        )
        
        self.collection_name = "contexter_docs"
    
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        
        # Use direct client for performance
        try:
            search_filter = self._build_filter(filters) if filters else None
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
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
            raise APIIntegrationError(f"Vector search failed: {e}", service="qdrant")
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """Upsert vectors in batches."""
        
        results = {"inserted": 0, "failed": 0, "errors": []}
        
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
                operation_info = self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                
                if operation_info.status.name == "COMPLETED":
                    results["inserted"] += len(batch)
                else:
                    results["failed"] += len(batch)
                    results["errors"].append(f"Batch {i//batch_size} failed")
                    
            except Exception as e:
                results["failed"] += len(batch)
                results["errors"].append(f"Batch {i//batch_size} error: {str(e)}")
        
        return results
    
    def _build_filter(self, filters: Dict[str, Any]):
        """Build Qdrant filter from dictionary."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        conditions = []
        for field, value in filters.items():
            if isinstance(value, list):
                # Multiple values - should condition
                should_conditions = []
                for v in value:
                    should_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=v))
                    )
                conditions.append(Filter(should=should_conditions))
            else:
                # Single value - must condition
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def health_check(self) -> Dict[str, Any]:
        """Qdrant specific health check."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            return {
                "status": "healthy",
                "collection_status": collection_info.status.name,
                "vectors_count": collection_info.vectors_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
```

## Authentication Strategies

### 1. JWT Authentication

For services requiring JSON Web Token authentication.

```python
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class JWTAuthHandler:
    """JWT authentication handler."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        expiry_minutes: int = 60
    ):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiry_minutes = expiry_minutes
        self.current_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
    
    async def get_headers(self) -> Dict[str, str]:
        """Get JWT authentication headers."""
        token = await self._get_valid_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def _get_valid_token(self) -> str:
        """Get valid JWT token, refresh if needed."""
        if self._is_token_expired():
            await self._refresh_token()
        
        return self.current_token
    
    def _is_token_expired(self) -> bool:
        """Check if current token is expired."""
        if not self.current_token or not self.token_expiry:
            return True
        
        # Add 5 minute buffer before expiry
        return datetime.utcnow() >= (self.token_expiry - timedelta(minutes=5))
    
    async def _refresh_token(self):
        """Generate new JWT token."""
        payload = {
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.expiry_minutes),
            "service": "contexter_rag"
        }
        
        self.current_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        self.token_expiry = payload["exp"]

class OAuth2AuthHandler:
    """OAuth2 authentication handler."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        scope: Optional[str] = None
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
    
    async def get_headers(self) -> Dict[str, str]:
        """Get OAuth2 authentication headers."""
        token = await self._get_valid_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    async def _get_valid_access_token(self) -> str:
        """Get valid access token, refresh if needed."""
        if self._is_token_expired():
            await self._refresh_access_token()
        
        return self.access_token
    
    def _is_token_expired(self) -> bool:
        """Check if access token is expired."""
        if not self.access_token or not self.token_expiry:
            return True
        
        return datetime.utcnow() >= (self.token_expiry - timedelta(minutes=5))
    
    async def _refresh_access_token(self):
        """Refresh OAuth2 access token."""
        async with httpx.AsyncClient() as client:
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            
            if self.scope:
                data["scope"] = self.scope
            
            response = await client.post(self.token_url, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            
            if "refresh_token" in token_data:
                self.refresh_token = token_data["refresh_token"]
```

## Rate Limiting and Throttling

### 1. Token Bucket Rate Limiter

Advanced rate limiting with burst support and fair queuing.

```python
import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int
    burst_size: Optional[int] = None
    per_key_limit: bool = False
    key_extractor: Optional[callable] = None

class TokenBucketRateLimiter:
    """Token bucket rate limiter with burst support."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.bucket_size = config.burst_size or config.requests_per_minute
        self.refill_rate = config.requests_per_minute / 60.0  # tokens per second
        
        # Global bucket
        self.tokens = self.bucket_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        
        # Per-key buckets if needed
        self.key_buckets: Dict[str, Dict[str, Any]] = {}
    
    async def acquire(self, key: Optional[str] = None, tokens: int = 1) -> bool:
        """Acquire tokens from rate limiter."""
        async with self.lock:
            if self.config.per_key_limit and key:
                return await self._acquire_from_key_bucket(key, tokens)
            else:
                return await self._acquire_from_global_bucket(tokens)
    
    async def _acquire_from_global_bucket(self, tokens: int) -> bool:
        """Acquire tokens from global bucket."""
        self._refill_bucket()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False
    
    async def _acquire_from_key_bucket(self, key: str, tokens: int) -> bool:
        """Acquire tokens from per-key bucket."""
        if key not in self.key_buckets:
            self.key_buckets[key] = {
                "tokens": self.bucket_size,
                "last_refill": time.time()
            }
        
        bucket = self.key_buckets[key]
        self._refill_key_bucket(bucket)
        
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        else:
            return False
    
    def _refill_bucket(self):
        """Refill global token bucket."""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def _refill_key_bucket(self, bucket: Dict[str, Any]):
        """Refill per-key token bucket."""
        now = time.time()
        time_passed = now - bucket["last_refill"]
        tokens_to_add = time_passed * self.refill_rate
        
        bucket["tokens"] = min(self.bucket_size, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
    
    def get_status(self, key: Optional[str] = None) -> Dict[str, Any]:
        """Get rate limiter status."""
        if self.config.per_key_limit and key and key in self.key_buckets:
            bucket = self.key_buckets[key]
            return {
                "available_tokens": bucket["tokens"],
                "bucket_size": self.bucket_size,
                "refill_rate": self.refill_rate,
                "utilization": 1 - (bucket["tokens"] / self.bucket_size)
            }
        else:
            return {
                "available_tokens": self.tokens,
                "bucket_size": self.bucket_size,
                "refill_rate": self.refill_rate,
                "utilization": 1 - (self.tokens / self.bucket_size)
            }

class RateLimitingAPIClient(APIClient):
    """API client with built-in rate limiting."""
    
    def __init__(self, rate_config: RateLimitConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter = TokenBucketRateLimiter(rate_config)
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """Execute request with rate limiting."""
        
        # Extract rate limit key if configured
        rate_key = None
        if self.rate_limiter.config.per_key_limit:
            if self.rate_limiter.config.key_extractor:
                rate_key = self.rate_limiter.config.key_extractor(request)
            else:
                rate_key = "default"
        
        # Acquire rate limit token
        acquired = await self.rate_limiter.acquire(rate_key)
        if not acquired:
            raise RateLimitExceededError(
                f"Rate limit exceeded for {self.service_name}",
                service=self.service_name,
                key=rate_key
            )
        
        return await super().execute_request(request)

class RateLimitExceededError(APIIntegrationError):
    """Rate limit exceeded error."""
    
    def __init__(self, message: str, service: str, key: Optional[str] = None):
        super().__init__(message, service=service)
        self.key = key
```

## Error Handling and Recovery

### 1. Comprehensive Error Classification

```python
from enum import Enum
from typing import Dict, Any, Optional, Type
import traceback

class ErrorCategory(Enum):
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    """Comprehensive error context."""
    category: ErrorCategory
    severity: ErrorSeverity
    service_name: str
    operation: str
    request_id: Optional[str] = None
    user_message: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None
    recovery_suggestions: Optional[List[str]] = None
    retry_after_seconds: Optional[int] = None

class ErrorClassifier:
    """Classifies and enriches API errors."""
    
    ERROR_MAPPINGS = {
        # HTTP status code mappings
        400: (ErrorCategory.CLIENT_ERROR, ErrorSeverity.MEDIUM),
        401: (ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH),
        403: (ErrorCategory.AUTHENTICATION, ErrorSeverity.HIGH),
        404: (ErrorCategory.CLIENT_ERROR, ErrorSeverity.LOW),
        429: (ErrorCategory.RATE_LIMIT, ErrorSeverity.MEDIUM),
        500: (ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH),
        502: (ErrorCategory.SERVER_ERROR, ErrorSeverity.HIGH),
        503: (ErrorCategory.SERVER_ERROR, ErrorSeverity.CRITICAL),
        504: (ErrorCategory.TIMEOUT, ErrorSeverity.HIGH),
        
        # Exception type mappings
        httpx.TimeoutException: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
        httpx.ConnectError: (ErrorCategory.NETWORK, ErrorSeverity.HIGH),
        httpx.ConnectTimeout: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
        httpx.ReadTimeout: (ErrorCategory.TIMEOUT, ErrorSeverity.MEDIUM),
    }
    
    @classmethod
    def classify_error(
        cls,
        error: Exception,
        service_name: str,
        operation: str,
        request_id: Optional[str] = None
    ) -> ErrorContext:
        """Classify error and create context."""
        
        category = ErrorCategory.CLIENT_ERROR
        severity = ErrorSeverity.MEDIUM
        user_message = "An error occurred while processing your request"
        recovery_suggestions = []
        retry_after = None
        
        if isinstance(error, HTTPError):
            # HTTP error classification
            if error.status_code in cls.ERROR_MAPPINGS:
                category, severity = cls.ERROR_MAPPINGS[error.status_code]
            
            if error.status_code == 401:
                user_message = "Authentication failed. Please check your credentials."
                recovery_suggestions = ["Verify API key", "Check token expiration"]
            elif error.status_code == 429:
                user_message = "Rate limit exceeded. Please try again later."
                recovery_suggestions = ["Wait before retrying", "Reduce request frequency"]
                retry_after = 60
            elif error.status_code >= 500:
                user_message = "Service temporarily unavailable. Please try again later."
                recovery_suggestions = ["Retry after delay", "Check service status"]
                retry_after = 30
                
        elif type(error) in cls.ERROR_MAPPINGS:
            # Exception type classification
            category, severity = cls.ERROR_MAPPINGS[type(error)]
            
            if isinstance(error, httpx.TimeoutException):
                user_message = "Request timed out. Please try again."
                recovery_suggestions = ["Retry with longer timeout", "Check network connectivity"]
                retry_after = 5
            elif isinstance(error, httpx.ConnectError):
                user_message = "Unable to connect to service."
                recovery_suggestions = ["Check service availability", "Verify network connectivity"]
                severity = ErrorSeverity.CRITICAL
        
        return ErrorContext(
            category=category,
            severity=severity,
            service_name=service_name,
            operation=operation,
            request_id=request_id,
            user_message=user_message,
            technical_details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            },
            recovery_suggestions=recovery_suggestions,
            retry_after_seconds=retry_after
        )

class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies: Dict[ErrorCategory, callable] = {
            ErrorCategory.RATE_LIMIT: self._handle_rate_limit,
            ErrorCategory.AUTHENTICATION: self._handle_auth_error,
            ErrorCategory.TIMEOUT: self._handle_timeout,
            ErrorCategory.SERVER_ERROR: self._handle_server_error,
            ErrorCategory.NETWORK: self._handle_network_error
        }
    
    async def handle_error(
        self,
        error_context: ErrorContext,
        operation: callable,
        *args,
        **kwargs
    ) -> Any:
        """Handle error with appropriate recovery strategy."""
        
        if error_context.category in self.recovery_strategies:
            return await self.recovery_strategies[error_context.category](
                error_context, operation, *args, **kwargs
            )
        else:
            # No recovery strategy available
            raise APIIntegrationError(
                error_context.user_message,
                service=error_context.service_name,
                request_id=error_context.request_id
            )
    
    async def _handle_rate_limit(self, context: ErrorContext, operation: callable, *args, **kwargs):
        """Handle rate limit errors with backoff."""
        wait_time = context.retry_after_seconds or 60
        await asyncio.sleep(wait_time)
        return await operation(*args, **kwargs)
    
    async def _handle_auth_error(self, context: ErrorContext, operation: callable, *args, **kwargs):
        """Handle authentication errors with token refresh."""
        # This would trigger token refresh in the auth handler
        # Implementation depends on specific auth mechanism
        raise APIIntegrationError(
            "Authentication failed and cannot be automatically recovered",
            service=context.service_name
        )
    
    async def _handle_timeout(self, context: ErrorContext, operation: callable, *args, **kwargs):
        """Handle timeout errors with retry."""
        # Implement exponential backoff retry
        for attempt in range(3):
            try:
                await asyncio.sleep(2 ** attempt)
                return await operation(*args, **kwargs)
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                continue
    
    async def _handle_server_error(self, context: ErrorContext, operation: callable, *args, **kwargs):
        """Handle server errors with circuit breaker."""
        # This would typically trigger circuit breaker logic
        # For now, we'll do a simple retry with backoff
        wait_time = context.retry_after_seconds or 30
        await asyncio.sleep(wait_time)
        return await operation(*args, **kwargs)
    
    async def _handle_network_error(self, context: ErrorContext, operation: callable, *args, **kwargs):
        """Handle network errors with retry."""
        # Similar to timeout handling but with different backoff
        for attempt in range(3):
            try:
                await asyncio.sleep(min(5 * (2 ** attempt), 60))
                return await operation(*args, **kwargs)
            except Exception as e:
                if attempt == 2:
                    raise
                continue
```

## Caching Strategies

### 1. Multi-Level API Caching

```python
import hashlib
import json
import pickle
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import asyncio

class APICache:
    """Multi-level caching for API responses."""
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        memory_ttl_seconds: int = 300,
        redis_client: Optional[Any] = None,
        redis_ttl_seconds: int = 3600
    ):
        self.memory_cache_size = memory_cache_size
        self.memory_ttl_seconds = memory_ttl_seconds
        self.redis_client = redis_client
        self.redis_ttl_seconds = redis_ttl_seconds
        
        # Memory cache (LRU)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []
        self.cache_lock = asyncio.Lock()
    
    async def get(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response for request."""
        cache_key = self._generate_cache_key(request)
        
        # Try memory cache first
        async with self.cache_lock:
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]
                if self._is_cache_valid(cached_item):
                    # Update access order
                    self.access_order.remove(cache_key)
                    self.access_order.append(cache_key)
                    
                    response = cached_item["response"]
                    response.cached = True
                    return response
                else:
                    # Remove expired item
                    del self.memory_cache[cache_key]
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
        
        # Try Redis cache if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    response = pickle.loads(cached_data)
                    response.cached = True
                    
                    # Store in memory cache for faster access
                    await self._store_in_memory(cache_key, response)
                    return response
            except Exception:
                # Redis error, continue without cache
                pass
        
        return None
    
    async def set(self, request: APIRequest, response: APIResponse):
        """Cache response for request."""
        cache_key = self._generate_cache_key(request)
        
        # Store in memory cache
        await self._store_in_memory(cache_key, response)
        
        # Store in Redis cache if available
        if self.redis_client:
            try:
                cached_data = pickle.dumps(response)
                await self.redis_client.setex(
                    cache_key,
                    self.redis_ttl_seconds,
                    cached_data
                )
            except Exception:
                # Redis error, continue without Redis cache
                pass
    
    async def _store_in_memory(self, cache_key: str, response: APIResponse):
        """Store response in memory cache with LRU eviction."""
        async with self.cache_lock:
            # Remove from current position if exists
            if cache_key in self.memory_cache:
                self.access_order.remove(cache_key)
            
            # Add to end (most recent)
            self.access_order.append(cache_key)
            self.memory_cache[cache_key] = {
                "response": response,
                "timestamp": datetime.utcnow()
            }
            
            # Evict oldest items if cache is full
            while len(self.memory_cache) > self.memory_cache_size:
                oldest_key = self.access_order.pop(0)
                del self.memory_cache[oldest_key]
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request."""
        # Create deterministic key based on request parameters
        key_data = {
            "method": request.method.value,
            "endpoint": request.endpoint,
            "params": request.params,
            "json_data": request.json_data
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached item is still valid."""
        timestamp = cached_item["timestamp"]
        expiry = timestamp + timedelta(seconds=self.memory_ttl_seconds)
        return datetime.utcnow() < expiry
    
    async def clear(self, pattern: Optional[str] = None):
        """Clear cache with optional pattern matching."""
        async with self.cache_lock:
            if pattern:
                # Clear entries matching pattern
                keys_to_remove = [
                    key for key in self.memory_cache.keys()
                    if pattern in key
                ]
                for key in keys_to_remove:
                    del self.memory_cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            else:
                # Clear all
                self.memory_cache.clear()
                self.access_order.clear()
        
        # Clear Redis cache if available
        if self.redis_client and pattern:
            try:
                keys = await self.redis_client.keys(f"*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_limit": self.memory_cache_size,
            "memory_utilization": len(self.memory_cache) / self.memory_cache_size,
            "redis_available": self.redis_client is not None
        }
```

## API Versioning and Compatibility

### 1. Version Management

```python
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class APIVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    LATEST = "latest"

@dataclass
class VersionedEndpoint:
    """Versioned API endpoint configuration."""
    path: str
    version: APIVersion
    deprecated: bool = False
    sunset_date: Optional[datetime] = None
    migration_guide: Optional[str] = None

class VersionedAPIClient(APIClient):
    """API client with version management."""
    
    def __init__(
        self,
        base_url: str,
        service_name: str,
        api_version: APIVersion = APIVersion.V1,
        **kwargs
    ):
        super().__init__(base_url, service_name, **kwargs)
        self.api_version = api_version
        self.version_mapping = self._load_version_mapping()
    
    def _load_version_mapping(self) -> Dict[str, VersionedEndpoint]:
        """Load endpoint version mapping."""
        # This would typically be loaded from configuration
        return {
            "embeddings": VersionedEndpoint(
                path="/embeddings",
                version=APIVersion.V1
            ),
            "models": VersionedEndpoint(
                path="/models",
                version=APIVersion.V1
            ),
            "search": VersionedEndpoint(
                path="/search",
                version=APIVersion.V2,
                deprecated=False
            )
        }
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """Execute request with version handling."""
        
        # Resolve versioned endpoint
        versioned_endpoint = self._resolve_endpoint(request.endpoint)
        request.endpoint = versioned_endpoint
        
        response = await super().execute_request(request)
        
        # Handle version-specific response processing
        return self._process_versioned_response(response)
    
    def _resolve_endpoint(self, endpoint: str) -> str:
        """Resolve endpoint with version."""
        endpoint_name = endpoint.lstrip("/")
        
        if endpoint_name in self.version_mapping:
            versioned_endpoint = self.version_mapping[endpoint_name]
            
            # Check for deprecation warnings
            if versioned_endpoint.deprecated:
                import warnings
                warnings.warn(
                    f"Endpoint '{endpoint}' is deprecated. "
                    f"Migration guide: {versioned_endpoint.migration_guide}",
                    DeprecationWarning
                )
            
            # Build versioned path
            version_str = self.api_version.value
            return f"/{version_str}{versioned_endpoint.path}"
        else:
            # Default versioning for unknown endpoints
            version_str = self.api_version.value
            return f"/{version_str}{endpoint}"
    
    def _process_versioned_response(self, response: APIResponse) -> APIResponse:
        """Process response based on version."""
        # Handle version-specific response transformations
        if self.api_version == APIVersion.V1:
            # V1-specific processing
            pass
        elif self.api_version == APIVersion.V2:
            # V2-specific processing
            pass
        
        return response
    
    async def migrate_to_version(self, target_version: APIVersion):
        """Migrate client to new API version."""
        self.api_version = target_version
        # Perform any necessary migration steps
        await self._validate_version_compatibility()
    
    async def _validate_version_compatibility(self):
        """Validate compatibility with target version."""
        # Check if all used endpoints are available in target version
        for endpoint_name, config in self.version_mapping.items():
            if config.version != self.api_version:
                # Check compatibility or availability of migration path
                pass
```

## Monitoring and Observability

### 1. Comprehensive API Monitoring

```python
from typing import Dict, Any, List, Optional
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class APIMetrics:
    """API performance metrics."""
    service_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    max_response_time_ms: float = 0.0
    response_times: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    rate_limit_hits: int = 0
    
    def add_request(
        self,
        response_time_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        cache_hit: bool = False,
        rate_limited: bool = False
    ):
        """Add request metrics."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if error_type:
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.total_response_time_ms += response_time_ms
        self.min_response_time_ms = min(self.min_response_time_ms, response_time_ms)
        self.max_response_time_ms = max(self.max_response_time_ms, response_time_ms)
        
        # Keep last 1000 response times for percentile calculations
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:
            self.response_times.pop(0)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if rate_limited:
            self.rate_limit_hits += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        if self.total_requests == 0:
            return {"service_name": self.service_name, "no_data": True}
        
        # Calculate percentiles
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        percentiles = {}
        if n > 0:
            percentiles = {
                "p50": sorted_times[int(n * 0.5)] if n > 0 else 0,
                "p90": sorted_times[int(n * 0.9)] if n > 0 else 0,
                "p95": sorted_times[int(n * 0.95)] if n > 0 else 0,
                "p99": sorted_times[int(n * 0.99)] if n > 0 else 0
            }
        
        return {
            "service_name": self.service_name,
            "request_metrics": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "success_rate": self.successful_requests / self.total_requests,
                "error_rate": self.failed_requests / self.total_requests
            },
            "performance_metrics": {
                "avg_response_time_ms": self.total_response_time_ms / self.total_requests,
                "min_response_time_ms": self.min_response_time_ms,
                "max_response_time_ms": self.max_response_time_ms,
                **percentiles
            },
            "cache_metrics": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            "error_breakdown": self.error_counts,
            "rate_limit_hits": self.rate_limit_hits
        }

class APIMonitor:
    """Comprehensive API monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, APIMetrics] = {}
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5%
            "response_time_p95": 1000,  # 1 second
            "cache_hit_rate": 0.5  # 50%
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def record_request(
        self,
        service_name: str,
        response_time_ms: float,
        success: bool,
        error_type: Optional[str] = None,
        cache_hit: bool = False,
        rate_limited: bool = False
    ):
        """Record API request metrics."""
        if service_name not in self.metrics:
            self.metrics[service_name] = APIMetrics(service_name)
        
        self.metrics[service_name].add_request(
            response_time_ms, success, error_type, cache_hit, rate_limited
        )
        
        # Check for alert conditions
        self._check_alerts(service_name)
    
    def _check_alerts(self, service_name: str):
        """Check for alert conditions."""
        metrics = self.metrics[service_name]
        summary = metrics.get_summary()
        
        alerts = []
        
        # Check error rate
        error_rate = summary["request_metrics"]["error_rate"]
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "service": service_name,
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "timestamp": datetime.utcnow()
            })
        
        # Check response time
        if "p95" in summary["performance_metrics"]:
            p95_time = summary["performance_metrics"]["p95"]
            if p95_time > self.alert_thresholds["response_time_p95"]:
                alerts.append({
                    "type": "high_response_time",
                    "service": service_name,
                    "value": p95_time,
                    "threshold": self.alert_thresholds["response_time_p95"],
                    "timestamp": datetime.utcnow()
                })
        
        # Check cache hit rate
        cache_hit_rate = summary["cache_metrics"]["cache_hit_rate"]
        if cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alerts.append({
                "type": "low_cache_hit_rate",
                "service": service_name,
                "value": cache_hit_rate,
                "threshold": self.alert_thresholds["cache_hit_rate"],
                "timestamp": datetime.utcnow()
            })
        
        self.alerts.extend(alerts)
    
    def get_service_metrics(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific service."""
        if service_name in self.metrics:
            return self.metrics[service_name].get_summary()
        return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all services."""
        return {
            service_name: metrics.get_summary()
            for service_name, metrics in self.metrics.items()
        }
    
    def get_active_alerts(self, max_age_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get active alerts within time window."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        return [
            alert for alert in self.alerts
            if alert["timestamp"] > cutoff_time
        ]
    
    def clear_alerts(self, service_name: Optional[str] = None):
        """Clear alerts for service or all services."""
        if service_name:
            self.alerts = [
                alert for alert in self.alerts
                if alert["service"] != service_name
            ]
        else:
            self.alerts.clear()

# Global monitor instance
api_monitor = APIMonitor()

class MonitoredAPIClient(APIClient):
    """API client with built-in monitoring."""
    
    async def execute_request(self, request: APIRequest) -> APIResponse:
        """Execute request with monitoring."""
        start_time = time.time()
        success = False
        error_type = None
        cache_hit = False
        rate_limited = False
        
        try:
            response = await super().execute_request(request)
            success = True
            cache_hit = response.cached
            return response
            
        except RateLimitExceededError:
            rate_limited = True
            error_type = "rate_limit"
            raise
        except HTTPError as e:
            error_type = f"http_{e.status_code}"
            raise
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            response_time = (time.time() - start_time) * 1000
            api_monitor.record_request(
                self.service_name,
                response_time,
                success,
                error_type,
                cache_hit,
                rate_limited
            )
```

This comprehensive API integration strategy provides:

1. **Unified Client Pattern**: Consistent approach for all external APIs
2. **Resilient Communication**: Circuit breakers, retries, and timeouts
3. **Advanced Authentication**: JWT, OAuth2, and API key support
4. **Intelligent Rate Limiting**: Token bucket with burst and per-key limits
5. **Multi-Level Caching**: Memory and Redis caching with LRU eviction
6. **Error Classification**: Comprehensive error handling with recovery strategies
7. **Version Management**: API versioning with deprecation handling
8. **Comprehensive Monitoring**: Performance metrics, alerting, and observability

These strategies ensure reliable, performant, and maintainable API integrations for the Contexter RAG system.