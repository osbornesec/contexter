"""
Async Client Patterns for Contexter RAG System

This module provides reusable async client patterns that integrate with existing
Contexter patterns while providing high-performance async operations for RAG components.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import existing Contexter patterns
from contexter.core.error_classifier import ErrorClassifier
from contexter.models.storage_models import StorageResult

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Base configuration for async clients."""
    
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: int = 30
    
    # Authentication
    api_key: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    
    # Rate limiting
    requests_per_second: float = 10.0
    burst_limit: int = 50
    
    # Circuit breaker
    failure_threshold: int = 5
    recovery_timeout: int = 60


@dataclass
class RequestMetrics:
    """Metrics for tracking request performance."""
    
    request_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    retries: int = 0
    
    def complete(self, status_code: int, error: Optional[str] = None):
        """Mark request as complete and calculate duration."""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()
        self.status_code = status_code
        self.error = error


class RateLimiter:
    """Token bucket rate limiter for async operations."""
    
    def __init__(self, rate: float, burst: int):
        """Initialize rate limiter.
        
        Args:
            rate: Requests per second allowed
            burst: Maximum burst capacity
        """
        self.rate = rate
        self.capacity = burst
        self.tokens = burst
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            # Add tokens based on elapsed time
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens: int = 1):
        """Wait until tokens are available."""
        while not await self.acquire(tokens):
            # Calculate wait time
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(min(wait_time, 1.0))


class CircuitBreaker:
    """Circuit breaker pattern for async operations."""
    
    def __init__(
        self, 
        failure_threshold: int = 5, 
        recovery_timeout: int = 60,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            if exc_type is None:
                # Success
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED")
            
            elif isinstance(exc_val, self.expected_exception):
                # Expected failure
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            return False  # Don't suppress exceptions
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout


class AsyncBaseClient(ABC):
    """Base class for async HTTP clients with common patterns."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.metrics: List[RequestMetrics] = []
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rate=config.requests_per_second,
            burst=config.burst_limit
        )
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout
        )
        
        # HTTP client will be initialized in __aenter__
        self._client: Optional[httpx.AsyncClient] = None
        self._session_active = False
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup_client()
    
    async def _initialize_client(self):
        """Initialize HTTP client with optimal settings."""
        
        # Prepare headers
        headers = {
            "User-Agent": "Contexter-RAG/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        if self.config.headers:
            headers.update(self.config.headers)
        
        # Configure limits for connection pooling
        limits = httpx.Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.max_keepalive_connections,
            keepalive_expiry=self.config.keepalive_expiry
        )
        
        # Create client
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.config.timeout),
            limits=limits,
            follow_redirects=True
        )
        
        self._session_active = True
        logger.info(f"Initialized HTTP client for {self.config.base_url}")
    
    async def _cleanup_client(self):
        """Cleanup HTTP client resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._session_active = False
            logger.info("HTTP client closed")
    
    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> httpx.Response:
        """Make HTTP request with retry logic and rate limiting."""
        
        if not self._session_active or not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        metrics = RequestMetrics(
            request_id=request_id,
            start_time=datetime.utcnow()
        )
        
        try:
            # Rate limiting
            await self.rate_limiter.wait_for_tokens()
            
            # Circuit breaker protection
            async with self.circuit_breaker:
                
                # Add request ID to headers
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['X-Request-ID'] = request_id
                
                # Make request
                response = await self._client.request(method, endpoint, **kwargs)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Record success metrics
                metrics.complete(response.status_code)
                self.metrics.append(metrics)
                
                logger.debug(
                    f"Request {request_id} completed",
                    extra={
                        "method": method,
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "duration": metrics.duration
                    }
                )
                
                return response
                
        except Exception as e:
            # Record error metrics
            status_code = getattr(e, 'response', {}).get('status_code', 0)
            metrics.complete(status_code, str(e))
            self.metrics.append(metrics)
            
            # Classify error using Contexter error classifier
            error_info = ErrorClassifier.classify_error(e)
            
            logger.error(
                f"Request {request_id} failed",
                extra={
                    "method": method,
                    "endpoint": endpoint,
                    "error_category": error_info.category,
                    "error": str(e),
                    "duration": metrics.duration
                }
            )
            
            raise
    
    async def get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self._make_request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self._make_request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make PUT request."""
        return await self._make_request("PUT", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make DELETE request."""
        return await self._make_request("DELETE", endpoint, **kwargs)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of request metrics."""
        if not self.metrics:
            return {"total_requests": 0}
        
        successful_requests = [m for m in self.metrics if m.error is None]
        failed_requests = [m for m in self.metrics if m.error is not None]
        durations = [m.duration for m in self.metrics if m.duration is not None]
        
        return {
            "total_requests": len(self.metrics),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(self.metrics) if self.metrics else 0,
            "average_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0
        }
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Implement health check for specific client."""
        pass


class BatchProcessor:
    """Generic batch processing with concurrency control."""
    
    def __init__(
        self,
        max_concurrency: int = 10,
        batch_size: int = 100,
        error_threshold: float = 0.1
    ):
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.error_threshold = error_threshold
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Union[Any, Exception]]:
        """Process items in batches with concurrency control."""
        
        # Split items into batches
        batches = [
            items[i:i + self.batch_size]
            for i in range(0, len(items), self.batch_size)
        ]
        
        results = []
        total_processed = 0
        errors = 0
        
        for batch_index, batch in enumerate(batches):
            # Create tasks for concurrent processing
            batch_tasks = []
            for item in batch:
                task = self._process_single_item(processor_func, item)
                batch_tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Check error rate
            batch_errors = sum(1 for r in batch_results if isinstance(r, Exception))
            errors += batch_errors
            total_processed += len(batch)
            
            current_error_rate = errors / total_processed
            
            if current_error_rate > self.error_threshold:
                logger.error(
                    f"Error rate {current_error_rate:.2%} exceeds threshold {self.error_threshold:.2%}"
                )
                # Could implement circuit breaker logic here
            
            results.extend(batch_results)
            
            # Progress callback
            if progress_callback:
                await progress_callback(total_processed, len(items))
            
            logger.info(
                f"Processed batch {batch_index + 1}/{len(batches)} "
                f"({total_processed}/{len(items)} items, {batch_errors} errors)"
            )
        
        return results
    
    async def _process_single_item(self, processor_func: Callable, item: Any):
        """Process single item with semaphore control."""
        async with self.semaphore:
            try:
                return await processor_func(item)
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                return e


class CacheableAsyncClient(AsyncBaseClient):
    """Async client with built-in caching capabilities."""
    
    def __init__(self, config: ClientConfig, cache_ttl: int = 300):
        super().__init__(config)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = cache_ttl
    
    def _generate_cache_key(self, method: str, endpoint: str, **kwargs) -> str:
        """Generate cache key for request."""
        import hashlib
        import json
        
        cache_data = {
            "method": method,
            "endpoint": endpoint,
            "params": kwargs.get("params", {}),
            "json": kwargs.get("json", {})
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    async def get_cached(self, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired."""
        cache_key = self._generate_cache_key("GET", endpoint, **kwargs)
        
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            
            # Check if cache is still valid
            if datetime.utcnow().timestamp() - cached_data["timestamp"] < self.cache_ttl:
                logger.debug(f"Cache hit for {endpoint}")
                return cached_data["data"]
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
                logger.debug(f"Cache expired for {endpoint}")
        
        return None
    
    async def get_with_cache(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Get with caching support."""
        
        # Check cache first
        cached_response = await self.get_cached(endpoint, **kwargs)
        if cached_response is not None:
            return cached_response
        
        # Make request
        response = await self.get(endpoint, **kwargs)
        response_data = response.json()
        
        # Cache response
        cache_key = self._generate_cache_key("GET", endpoint, **kwargs)
        self.cache[cache_key] = {
            "data": response_data,
            "timestamp": datetime.utcnow().timestamp()
        }
        
        logger.debug(f"Cached response for {endpoint}")
        return response_data
    
    def clear_cache(self):
        """Clear all cached responses."""
        self.cache.clear()
        logger.info("Cache cleared")


@asynccontextmanager
async def async_client_pool(
    configs: List[ClientConfig],
    client_class: type = AsyncBaseClient
) -> AsyncGenerator[List[AsyncBaseClient], None]:
    """Context manager for managing multiple async clients."""
    
    clients = []
    
    try:
        # Initialize all clients
        for config in configs:
            client = client_class(config)
            await client.__aenter__()
            clients.append(client)
        
        yield clients
        
    finally:
        # Cleanup all clients
        cleanup_tasks = []
        for client in clients:
            cleanup_tasks.append(client.__aexit__(None, None, None))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)


# Example implementation for Voyage AI client
class VoyageAsyncClient(AsyncBaseClient):
    """Async client for Voyage AI API."""
    
    async def embed_texts(
        self, 
        texts: List[str], 
        model: str = "voyage-code-3",
        input_type: str = "document"
    ) -> List[List[float]]:
        """Generate embeddings for texts."""
        
        response = await self.post(
            "/v1/embeddings",
            json={
                "input": texts,
                "model": model,
                "input_type": input_type
            }
        )
        
        data = response.json()
        return [item["embedding"] for item in data["data"]]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Voyage AI client."""
        try:
            # Test with a small embedding request
            test_response = await self.embed_texts(["test"], model="voyage-code-3")
            
            return {
                "status": "healthy",
                "service": "voyage-ai",
                "response_time": self.get_metrics_summary().get("average_duration", 0),
                "test_embedding_dims": len(test_response[0]) if test_response else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "voyage-ai",
                "error": str(e)
            }


# Example implementation for Qdrant client
class QdrantAsyncClient(AsyncBaseClient):
    """Async client for Qdrant vector database."""
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        
        search_params = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }
        
        if filters:
            search_params["filter"] = filters
        
        response = await self.post(
            f"/collections/{collection_name}/points/search",
            json=search_params
        )
        
        data = response.json()
        return data.get("result", [])
    
    async def upsert_vectors(
        self,
        collection_name: str,
        points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Upsert vectors into collection."""
        
        response = await self.put(
            f"/collections/{collection_name}/points",
            json={"points": points}
        )
        
        return response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Qdrant client."""
        try:
            response = await self.get("/")
            cluster_info = response.json()
            
            return {
                "status": "healthy",
                "service": "qdrant",
                "version": cluster_info.get("version"),
                "response_time": self.get_metrics_summary().get("average_duration", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "qdrant",
                "error": str(e)
            }


# Usage examples and patterns
async def example_usage():
    """Example of using async client patterns."""
    
    # Single client usage
    voyage_config = ClientConfig(
        base_url="https://api.voyageai.com",
        api_key="your-api-key",
        requests_per_second=5.0,
        max_retries=3
    )
    
    async with VoyageAsyncClient(voyage_config) as voyage_client:
        embeddings = await voyage_client.embed_texts(
            ["Hello world", "This is a test"],
            model="voyage-code-3"
        )
        print(f"Generated {len(embeddings)} embeddings")
    
    # Multiple client pool usage
    configs = [
        ClientConfig(base_url="https://api.voyageai.com", api_key="key1"),
        ClientConfig(base_url="http://localhost:6333")  # Qdrant
    ]
    
    async with async_client_pool(configs, AsyncBaseClient) as clients:
        voyage_client, qdrant_client = clients
        
        # Use clients concurrently
        tasks = [
            voyage_client.health_check(),
            qdrant_client.health_check()
        ]
        
        health_results = await asyncio.gather(*tasks)
        print("Health check results:", health_results)
    
    # Batch processing example
    batch_processor = BatchProcessor(
        max_concurrency=5,
        batch_size=50,
        error_threshold=0.1
    )
    
    items = list(range(200))  # 200 items to process
    
    async def process_item(item):
        # Simulate async processing
        await asyncio.sleep(0.1)
        return f"processed_{item}"
    
    results = await batch_processor.process_batch(
        items,
        process_item,
        progress_callback=lambda processed, total: print(f"Progress: {processed}/{total}")
    )
    
    successful_results = [r for r in results if not isinstance(r, Exception)]
    print(f"Successfully processed {len(successful_results)} items")


if __name__ == "__main__":
    asyncio.run(example_usage())