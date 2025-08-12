"""
Voyage AI HTTP Client

Production-ready HTTP client for Voyage AI's embedding API with comprehensive
rate limiting, circuit breaker patterns, and error handling.
"""

import asyncio
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from ..models.embedding_models import (
    IEmbeddingClient, InputType, RateLimitInfo, ProcessingMetrics
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class VoyageAPIError(Exception):
    """Base exception for Voyage AI API errors."""
    pass


class RateLimitError(VoyageAPIError):
    """Rate limit exceeded error."""
    pass


class CircuitBreakerOpenError(VoyageAPIError):
    """Circuit breaker is open error."""
    pass


@dataclass
class VoyageClientConfig:
    """Configuration for Voyage AI client."""
    
    api_key: str
    model: str = "voyage-code-3"
    base_url: str = "https://api.voyageai.com/v1"
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit_rpm: int = 300  # Requests per minute
    rate_limit_tpm: int = 1000000  # Tokens per minute
    max_concurrent_requests: int = 10
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


class TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed: int) -> bool:
        """
        Try to acquire tokens from bucket.
        
        Args:
            tokens_needed: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            await self._refill()
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            
            return False
    
    async def wait_for_tokens(self, tokens_needed: int, max_wait: float = 60.0):
        """
        Wait for tokens to become available.
        
        Args:
            tokens_needed: Number of tokens needed
            max_wait: Maximum wait time in seconds
        """
        start_time = time.time()
        
        while (time.time() - start_time) < max_wait:
            if await self.acquire(tokens_needed):
                return
            
            # Calculate wait time for next refill
            async with self._lock:
                tokens_deficit = tokens_needed - self.tokens
                wait_time = min(tokens_deficit / self.refill_rate, 1.0)
            
            await asyncio.sleep(wait_time)
        
        raise RateLimitError(f"Could not acquire {tokens_needed} tokens within {max_wait}s")
    
    async def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = VoyageAPIError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to half-open")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open. Last failure: {self.last_failure_time}"
                    )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        async with self._lock:
            if exc_type is None:
                # Success
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to closed state")
            
            elif issubclass(exc_type, self.expected_exception):
                # Expected failure
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.error(
                        f"Circuit breaker opened after {self.failure_count} failures"
                    )
        
        return False  # Don't suppress exceptions
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout


class VoyageAIClient(IEmbeddingClient):
    """Production-ready Voyage AI client with comprehensive error handling."""
    
    def __init__(self, config: VoyageClientConfig):
        self.config = config
        self._client = None
        self._initialized = False
        
        # Rate limiting
        self.rate_limit_info = RateLimitInfo(
            requests_per_minute=config.rate_limit_rpm,
            tokens_per_minute=config.rate_limit_tpm
        )
        
        # Token buckets for dual rate limiting
        self.request_bucket = TokenBucket(
            capacity=config.rate_limit_rpm,
            refill_rate=config.rate_limit_rpm / 60.0  # per second
        )
        
        self.token_bucket = TokenBucket(
            capacity=config.rate_limit_tpm,
            refill_rate=config.rate_limit_tpm / 60.0  # per second
        )
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_failure_threshold,
            recovery_timeout=config.circuit_breaker_recovery_timeout,
            expected_exception=VoyageAPIError
        )
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Metrics
        self.metrics = ProcessingMetrics()
        
        # Performance tracking
        self._request_times = []
        self._last_request_time = None
    
    async def initialize(self):
        """Initialize the HTTP client."""
        if self._initialized:
            return
        
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Contexter-RAG/1.0"
            },
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            )
        )
        
        self._initialized = True
        logger.info("Voyage AI client initialized")
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("Voyage AI client closed")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "voyage-code-3",
        input_type: InputType = InputType.DOCUMENT
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using Voyage AI API.
        
        Args:
            texts: List of texts to embed
            model: Model to use for embedding
            input_type: Type of input (document or query)
            
        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            await self.initialize()
        
        if not texts:
            return []
        
        # Estimate tokens needed
        estimated_tokens = sum(self._estimate_tokens(text) for text in texts)
        
        # Rate limiting
        async with self.semaphore:
            await self.request_bucket.wait_for_tokens(1)  # One request
            await self.token_bucket.wait_for_tokens(estimated_tokens)
            
            # Circuit breaker protection
            async with self.circuit_breaker:
                return await self._make_embedding_request(texts, model, input_type)
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_embedding_request(
        self,
        texts: List[str],
        model: str,
        input_type: InputType
    ) -> List[List[float]]:
        """Make the actual embedding API request with retry logic."""
        start_time = time.time()
        
        payload = {
            "input": texts,
            "model": model,
            "input_type": input_type.value
        }
        
        try:
            response = await self._client.post(
                f"{self.config.base_url}/embeddings",
                json=payload
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("retry-after", 60))
                jitter = random.uniform(1, 5)
                wait_time = retry_after + jitter
                
                logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                raise RateLimitError("Rate limit exceeded")
            
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            # Validate embeddings
            if len(embeddings) != len(texts):
                raise VoyageAPIError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )
            
            for embedding in embeddings:
                if not isinstance(embedding, list) or len(embedding) != 2048:
                    raise VoyageAPIError("Invalid embedding format or dimensions")
            
            # Update metrics
            request_time = time.time() - start_time
            self._update_metrics(len(texts), request_time, True, len(texts) * 1000)  # Rough token estimate
            
            logger.debug(f"Generated {len(embeddings)} embeddings in {request_time:.2f}s")
            
            return embeddings
        
        except httpx.HTTPStatusError as e:
            self._update_metrics(len(texts), time.time() - start_time, False, 0)
            
            if e.response.status_code == 400:
                raise VoyageAPIError(f"Invalid request: {e.response.text}")
            elif e.response.status_code == 401:
                raise VoyageAPIError("Authentication failed - check API key")
            elif e.response.status_code == 403:
                raise VoyageAPIError("Quota exceeded or billing issue")
            elif e.response.status_code >= 500:
                raise VoyageAPIError(f"Server error: {e.response.status_code}")
            else:
                raise VoyageAPIError(f"HTTP error {e.response.status_code}: {e.response.text}")
        
        except httpx.RequestError as e:
            self._update_metrics(len(texts), time.time() - start_time, False, 0)
            raise VoyageAPIError(f"Request failed: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough approximation: 1 token ≈ 0.75 words for technical content
        return int(len(text.split()) * 1.3)
    
    def _update_metrics(self, batch_size: int, latency: float, success: bool, tokens: int):
        """Update client metrics."""
        if success:
            self.metrics.successful_requests += batch_size
            self.metrics.total_tokens_processed += tokens
        else:
            self.metrics.failed_requests += batch_size
        
        self.metrics.total_requests += batch_size
        self.metrics.total_processing_time += latency
        self.metrics.total_api_calls += 1
        
        # Track request times for performance analysis
        self._request_times.append(latency)
        if len(self._request_times) > 1000:  # Keep last 1000 requests
            self._request_times = self._request_times[-1000:]
        
        self._last_request_time = datetime.utcnow()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Voyage AI client.
        
        Returns:
            Health status and metrics
        """
        try:
            # Test with a small embedding request
            test_text = "Health check test"
            start_time = time.time()
            
            embeddings = await self.generate_embeddings(
                [test_text],
                model=self.config.model,
                input_type=InputType.QUERY
            )
            
            latency = time.time() - start_time
            
            return {
                "status": "healthy",
                "api_accessible": True,
                "test_latency_ms": round(latency * 1000, 2),
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "rate_limit_info": {
                    "requests_remaining": self.config.rate_limit_rpm - self.rate_limit_info.current_requests,
                    "tokens_remaining": self.config.rate_limit_tpm - self.rate_limit_info.current_tokens,
                    "window_start": self.rate_limit_info.window_start.isoformat()
                },
                "performance_metrics": {
                    "total_requests": self.metrics.total_requests,
                    "success_rate": self.metrics.success_rate,
                    "average_latency_ms": round(self.metrics.average_processing_time * 1000, 2),
                    "throughput_per_minute": round(self.metrics.throughput_per_minute, 1)
                }
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "last_successful_request": self._last_request_time.isoformat() if self._last_request_time else None
            }
    
    def get_rate_limit_info(self) -> RateLimitInfo:
        """Get current rate limiting information."""
        return self.rate_limit_info
    
    def get_performance_metrics(self) -> ProcessingMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles from recent requests."""
        if not self._request_times:
            return {}
        
        sorted_times = sorted(self._request_times)
        n = len(sorted_times)
        
        return {
            "p50": sorted_times[int(n * 0.5)] * 1000,  # Convert to ms
            "p95": sorted_times[int(n * 0.95)] * 1000,
            "p99": sorted_times[int(n * 0.99)] * 1000,
            "min": min(sorted_times) * 1000,
            "max": max(sorted_times) * 1000
        }


# Factory function for easy client creation
async def create_voyage_client(
    api_key: str,
    model: str = "voyage-code-3",
    **config_overrides
) -> VoyageAIClient:
    """
    Create and initialize a Voyage AI client.
    
    Args:
        api_key: Voyage AI API key
        model: Model to use for embeddings
        **config_overrides: Additional configuration overrides
        
    Returns:
        Initialized VoyageAIClient
    """
    config = VoyageClientConfig(
        api_key=api_key,
        model=model,
        **config_overrides
    )
    
    client = VoyageAIClient(config)
    await client.initialize()
    
    return client