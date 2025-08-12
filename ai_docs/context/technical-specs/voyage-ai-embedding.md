# Voyage AI Embedding Service Technical Specifications

## Overview

This document provides comprehensive technical specifications for integrating Voyage AI's voyage-code-3 embedding model into the Contexter RAG system. It covers client configuration, batch processing, caching strategies, error handling, and performance optimization to achieve >1000 documents/minute throughput.

**Key Requirements**:
- >1000 documents/minute processing throughput
- 99.9% API success rate
- Intelligent caching with >50% hit rate
- Rate limiting compliance (300 requests/minute, 1M tokens/minute)
- Circuit breaker patterns for API resilience

## Voyage AI Client Architecture

### Client Configuration

```python
# Production-optimized Voyage AI client configuration
from voyageai import AsyncClient, Client
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import hashlib
import json

@dataclass
class VoyageClientConfig:
    """Configuration for Voyage AI client integration."""
    
    api_key: str
    model: str = "voyage-code-3"  # Optimized for code documentation
    max_retries: int = 3
    timeout: int = 30
    rate_limit_rpm: int = 300  # Requests per minute
    rate_limit_tpm: int = 1000000  # Tokens per minute (1M)
    batch_size: int = 100  # Optimal batch size for rate limits
    concurrent_batches: int = 3  # Max concurrent batch requests

class VoyageEmbeddingClient:
    """Production-ready Voyage AI embedding client."""
    
    def __init__(self, config: VoyageClientConfig):
        self.config = config
        self.client = AsyncClient(
            api_key=config.api_key,
            max_retries=config.max_retries,
            timeout=config.timeout
        )
        
        # Rate limiting semaphore
        self.request_semaphore = asyncio.Semaphore(config.concurrent_batches)
        self.token_bucket = TokenBucket(
            capacity=config.rate_limit_tpm,
            refill_rate=config.rate_limit_tpm / 60  # Per second
        )
        
        # Circuit breaker for API resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=VoyageAPIError
        )
```

### Model-Specific Optimization

```python
class VoyageModelConfig:
    """Model-specific configuration for voyage-code-3."""
    
    VOYAGE_CODE_3 = {
        "model_name": "voyage-code-3",
        "dimensions": 2048,
        "max_input_tokens": 32000,
        "optimal_batch_size": 100,
        "cost_per_1k_tokens": 0.12,
        "specialization": "code and technical documentation",
        "supported_languages": ["python", "javascript", "typescript", "go", "rust", "java"]
    }
    
    @classmethod
    def get_optimal_chunk_size(cls, text: str) -> int:
        """Calculate optimal chunk size for code documentation."""
        # voyage-code-3 performs best with technical content chunks
        # of 800-1200 tokens with 200 token overlap
        estimated_tokens = len(text.split()) * 1.3  # Rough token estimation
        
        if estimated_tokens <= 1000:
            return len(text)  # Use full text if under limit
        
        # Calculate optimal chunk size
        optimal_tokens = 1000
        chars_per_token = len(text) / estimated_tokens
        return int(optimal_tokens * chars_per_token)
```

## Batch Processing Implementation

### High-Throughput Batch Processor

```python
class VoyageBatchProcessor:
    """High-performance batch processing for embedding generation."""
    
    def __init__(
        self, 
        client: VoyageEmbeddingClient,
        cache: Optional['EmbeddingCache'] = None
    ):
        self.client = client
        self.cache = cache
        self.metrics = ProcessingMetrics()
        
        # Performance tracking
        self.processing_stats = {
            "total_processed": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "failed_requests": 0,
            "average_latency": 0.0
        }
    
    async def process_documents_batch(
        self,
        documents: List[str],
        input_type: str = "document",
        progress_callback: Optional[Callable] = None
    ) -> BatchProcessingResult:
        """Process documents in optimized batches with caching."""
        
        start_time = time.time()
        results = BatchProcessingResult()
        
        # Check cache first
        cached_results, uncached_docs = await self._check_cache(documents)
        results.cache_hits = len(cached_results)
        
        if uncached_docs:
            # Process uncached documents in batches
            api_results = await self._process_api_batches(
                uncached_docs, 
                input_type,
                progress_callback
            )
            
            # Cache new results
            if self.cache:
                await self._cache_results(uncached_docs, api_results.embeddings)
            
            results.api_results = api_results
        
        # Combine cached and new results
        results.embeddings = self._combine_results(
            documents, 
            cached_results, 
            results.api_results.embeddings if results.api_results else []
        )
        
        # Update metrics
        results.total_time = time.time() - start_time
        results.throughput = len(documents) / results.total_time * 60  # per minute
        
        await self._update_metrics(results)
        return results
    
    async def _process_api_batches(
        self,
        documents: List[str],
        input_type: str,
        progress_callback: Optional[Callable] = None
    ) -> APIBatchResult:
        """Process documents through Voyage API in optimized batches."""
        
        batch_size = self.client.config.batch_size
        all_embeddings = []
        errors = []
        
        batches = [
            documents[i:i + batch_size] 
            for i in range(0, len(documents), batch_size)
        ]
        
        # Process batches with concurrency control
        semaphore_tasks = []
        for i, batch in enumerate(batches):
            task = self._process_single_batch(
                batch, input_type, batch_index=i
            )
            semaphore_tasks.append(task)
        
        # Execute with progress tracking
        batch_results = []
        for i, task in enumerate(asyncio.as_completed(semaphore_tasks)):
            try:
                result = await task
                batch_results.append(result)
                
                if progress_callback:
                    processed = (i + 1) * batch_size
                    await progress_callback(
                        min(processed, len(documents)), 
                        len(documents)
                    )
                    
            except Exception as e:
                errors.append(f"Batch {i}: {e}")
                logger.error(f"Batch processing failed: {e}")
        
        # Combine batch results in correct order
        batch_results.sort(key=lambda x: x.batch_index)
        for result in batch_results:
            all_embeddings.extend(result.embeddings)
        
        return APIBatchResult(
            embeddings=all_embeddings,
            errors=errors,
            api_calls=len(batches)
        )
    
    async def _process_single_batch(
        self,
        batch: List[str],
        input_type: str,
        batch_index: int
    ) -> BatchResult:
        """Process a single batch with circuit breaker and rate limiting."""
        
        async with self.client.request_semaphore:
            # Rate limiting with token bucket
            estimated_tokens = sum(len(doc.split()) * 1.3 for doc in batch)
            await self.client.token_bucket.wait_for_tokens(int(estimated_tokens))
            
            # Circuit breaker protection
            async with self.client.circuit_breaker:
                try:
                    # Make API call with retry logic
                    result = await self._embed_with_retry(batch, input_type)
                    
                    return BatchResult(
                        embeddings=result.embeddings,
                        batch_index=batch_index,
                        success=True,
                        tokens_used=estimated_tokens
                    )
                    
                except Exception as e:
                    logger.error(f"Batch {batch_index} failed: {e}")
                    raise VoyageAPIError(f"Batch processing failed: {e}")
    
    @retry(
        wait=wait_exponential(multiplier=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((httpx.RequestError, VoyageAPIError))
    )
    async def _embed_with_retry(
        self, 
        texts: List[str], 
        input_type: str
    ) -> EmbedResult:
        """Embed texts with exponential backoff retry."""
        
        try:
            result = await self.client.client.embed(
                texts=texts,
                model=self.client.config.model,
                input_type=input_type
            )
            
            # Validate result
            if not result.embeddings or len(result.embeddings) != len(texts):
                raise VoyageAPIError("Invalid embedding response")
            
            return result
            
        except Exception as e:
            # Log detailed error information
            logger.error(
                f"Voyage API call failed: {e}", 
                extra={
                    "batch_size": len(texts),
                    "input_type": input_type,
                    "model": self.client.config.model
                }
            )
            raise
```

## Intelligent Caching Implementation

### SQLite-Based Embedding Cache

```python
class EmbeddingCache:
    """High-performance SQLite-based embedding cache with LRU eviction."""
    
    def __init__(
        self, 
        cache_path: str = "~/.contexter/embedding_cache.db",
        max_entries: int = 100000,
        ttl_hours: int = 168  # 7 days
    ):
        self.cache_path = Path(cache_path).expanduser()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_hours * 3600
        
        # Initialize database
        asyncio.create_task(self._init_database())
        
        # Cache statistics
        self.stats = CacheStats()
    
    async def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.cache_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Indexes for performance
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_accessed_at ON embeddings(accessed_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_type ON embeddings(model, input_type)"
            )
            
            await db.commit()
    
    async def get_cached_embeddings(
        self, 
        texts: List[str],
        model: str = "voyage-code-3",
        input_type: str = "document"
    ) -> Tuple[Dict[str, List[float]], List[str]]:
        """Retrieve cached embeddings and identify uncached texts."""
        
        cached_embeddings = {}
        uncached_texts = []
        
        async with aiosqlite.connect(self.cache_path) as db:
            for text in texts:
                content_hash = self._generate_content_hash(text, model, input_type)
                
                cursor = await db.execute("""
                    SELECT embedding, access_count 
                    FROM embeddings 
                    WHERE content_hash = ? 
                    AND model = ? 
                    AND input_type = ?
                    AND datetime(created_at, '+{} seconds') > datetime('now')
                """.format(self.ttl_seconds), (content_hash, model, input_type))
                
                row = await cursor.fetchone()
                
                if row:
                    # Cache hit - update access statistics
                    embedding = pickle.loads(row[0])
                    cached_embeddings[text] = embedding
                    
                    await db.execute("""
                        UPDATE embeddings 
                        SET accessed_at = CURRENT_TIMESTAMP,
                            access_count = access_count + 1
                        WHERE content_hash = ?
                    """, (content_hash,))
                    
                    self.stats.hits += 1
                else:
                    # Cache miss
                    uncached_texts.append(text)
                    self.stats.misses += 1
            
            await db.commit()
        
        return cached_embeddings, uncached_texts
    
    async def cache_embeddings(
        self,
        text_embedding_pairs: List[Tuple[str, List[float]]],
        model: str = "voyage-code-3",
        input_type: str = "document"
    ):
        """Cache new embeddings with automatic cleanup."""
        
        async with aiosqlite.connect(self.cache_path) as db:
            for text, embedding in text_embedding_pairs:
                content_hash = self._generate_content_hash(text, model, input_type)
                embedding_blob = pickle.dumps(embedding)
                
                await db.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (content_hash, content, model, input_type, embedding, 
                     created_at, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 1)
                """, (content_hash, text, model, input_type, embedding_blob))
            
            await db.commit()
        
        # Trigger cleanup if needed
        await self._cleanup_if_needed()
    
    async def _cleanup_if_needed(self):
        """Clean up old entries using LRU eviction."""
        
        async with aiosqlite.connect(self.cache_path) as db:
            # Check if cleanup is needed
            cursor = await db.execute("SELECT COUNT(*) FROM embeddings")
            count = (await cursor.fetchone())[0]
            
            if count > self.max_entries:
                # Remove oldest accessed entries
                entries_to_remove = count - int(self.max_entries * 0.8)  # Remove to 80% capacity
                
                await db.execute("""
                    DELETE FROM embeddings 
                    WHERE content_hash IN (
                        SELECT content_hash 
                        FROM embeddings 
                        ORDER BY accessed_at ASC 
                        LIMIT ?
                    )
                """, (entries_to_remove,))
                
                await db.commit()
                logger.info(f"Cleaned up {entries_to_remove} old cache entries")
    
    def _generate_content_hash(
        self, 
        text: str, 
        model: str, 
        input_type: str
    ) -> str:
        """Generate deterministic hash for cache key."""
        
        cache_key = {
            "text": text,
            "model": model,
            "input_type": input_type
        }
        
        cache_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    async def get_cache_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        
        async with aiosqlite.connect(self.cache_path) as db:
            cursor = await db.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(accessed_at) as newest_access
                FROM embeddings
            """)
            
            row = await cursor.fetchone()
            
            self.stats.total_entries = row[0]
            self.stats.avg_access_count = row[1] or 0
            self.stats.hit_rate = (
                self.stats.hits / (self.stats.hits + self.stats.misses) 
                if (self.stats.hits + self.stats.misses) > 0 else 0
            )
        
        return self.stats
```

## Error Handling and Circuit Breaker

### Comprehensive Error Classification

```python
class VoyageErrorHandler:
    """Advanced error handling for Voyage AI API integration."""
    
    ERROR_CATEGORIES = {
        "rate_limit": {
            "indicators": ["429", "rate limit", "too many requests"],
            "recovery_strategy": "exponential_backoff",
            "max_retries": 5,
            "base_delay": 1.0
        },
        "api_quota": {
            "indicators": ["quota exceeded", "billing", "insufficient credits"],
            "recovery_strategy": "circuit_breaker",
            "max_retries": 0,
            "alert_level": "critical"
        },
        "invalid_input": {
            "indicators": ["invalid input", "malformed", "token limit"],
            "recovery_strategy": "skip_and_log",
            "max_retries": 0,
            "alert_level": "warning"
        },
        "network_error": {
            "indicators": ["connection error", "timeout", "network"],
            "recovery_strategy": "retry_with_backoff",
            "max_retries": 3,
            "base_delay": 2.0
        },
        "api_error": {
            "indicators": ["internal server error", "500", "503"],
            "recovery_strategy": "retry_with_backoff",
            "max_retries": 3,
            "base_delay": 1.0
        }
    }
    
    @classmethod
    def classify_error(cls, error: Exception) -> ErrorClassification:
        """Classify error and determine recovery strategy."""
        
        error_message = str(error).lower()
        
        for category, config in cls.ERROR_CATEGORIES.items():
            if any(indicator in error_message for indicator in config["indicators"]):
                return ErrorClassification(
                    category=category,
                    recovery_strategy=config["recovery_strategy"],
                    max_retries=config["max_retries"],
                    base_delay=config.get("base_delay", 1.0),
                    alert_level=config.get("alert_level", "error"),
                    original_error=error
                )
        
        # Default classification for unknown errors
        return ErrorClassification(
            category="unknown",
            recovery_strategy="retry_with_backoff",
            max_retries=1,
            base_delay=1.0,
            alert_level="error",
            original_error=error
        )

class VoyageCircuitBreaker:
    """Circuit breaker pattern for Voyage API resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Exception = VoyageAPIError
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED  # CLOSED, OPEN, HALF_OPEN
    
    async def __aenter__(self):
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
        if exc_type is None:
            # Success
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker reset to closed state")
        
        elif isinstance(exc_val, self.expected_exception):
            # Expected failure
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
            
            return False  # Don't suppress the exception
        
        # Unexpected exceptions pass through
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
```

## Performance Monitoring and Optimization

### Comprehensive Metrics Collection

```python
class VoyageMetricsCollector:
    """Advanced metrics collection for Voyage AI integration."""
    
    def __init__(self):
        self.metrics = {
            "api_calls": Counter(),
            "cache_hits": Counter(),
            "cache_misses": Counter(),
            "errors": Counter(),
            "latency": Histogram(),
            "throughput": Gauge(),
            "cost_tracking": Counter()
        }
        
        self.performance_targets = {
            "throughput_min": 1000,  # documents/minute
            "cache_hit_rate_min": 0.5,  # 50%
            "error_rate_max": 0.001,  # 0.1%
            "p95_latency_max": 2.0  # seconds
        }
    
    async def record_api_call(
        self,
        batch_size: int,
        latency: float,
        success: bool,
        tokens_used: int
    ):
        """Record API call metrics."""
        
        self.metrics["api_calls"].inc()
        self.metrics["latency"].observe(latency)
        
        if success:
            self.metrics["throughput"].set(batch_size / latency * 60)  # per minute
            
            # Cost tracking (voyage-code-3: $0.12 per 1K tokens)
            cost = (tokens_used / 1000) * 0.12
            self.metrics["cost_tracking"].inc(cost)
        else:
            self.metrics["errors"].inc()
    
    async def record_cache_operation(self, cache_hits: int, cache_misses: int):
        """Record cache performance metrics."""
        
        self.metrics["cache_hits"].inc(cache_hits)
        self.metrics["cache_misses"].inc(cache_misses)
    
    def get_performance_report(self) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        total_calls = self.metrics["api_calls"]._value
        total_errors = self.metrics["errors"]._value
        total_hits = self.metrics["cache_hits"]._value
        total_misses = self.metrics["cache_misses"]._value
        
        error_rate = total_errors / total_calls if total_calls > 0 else 0
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        return PerformanceReport(
            throughput=self.metrics["throughput"]._value,
            error_rate=error_rate,
            cache_hit_rate=hit_rate,
            p95_latency=self._calculate_p95_latency(),
            total_cost=self.metrics["cost_tracking"]._value,
            sla_compliance=self._check_sla_compliance(error_rate, hit_rate)
        )
    
    def _check_sla_compliance(self, error_rate: float, hit_rate: float) -> Dict[str, bool]:
        """Check SLA compliance against performance targets."""
        
        return {
            "throughput": self.metrics["throughput"]._value >= self.performance_targets["throughput_min"],
            "cache_hit_rate": hit_rate >= self.performance_targets["cache_hit_rate_min"],
            "error_rate": error_rate <= self.performance_targets["error_rate_max"],
            "latency": self._calculate_p95_latency() <= self.performance_targets["p95_latency_max"]
        }
```

## Integration with Contexter Patterns

### Contexter-Compatible Implementation

```python
class ContexterVoyageIntegration:
    """Integration layer between Contexter and Voyage AI embedding service."""
    
    def __init__(
        self,
        storage_manager: LocalStorageManager,
        config: VoyageClientConfig
    ):
        self.storage = storage_manager
        self.client = VoyageEmbeddingClient(config)
        self.cache = EmbeddingCache()
        self.processor = VoyageBatchProcessor(self.client, self.cache)
        
        # Integration with Contexter monitoring
        self.metrics = VoyageMetricsCollector()
    
    async def process_documentation_embeddings(
        self,
        library_id: str,
        version: str,
        document_chunks: List[DocumentationChunk]
    ) -> EmbeddingProcessingResult:
        """Process Contexter documentation chunks into embeddings."""
        
        start_time = time.time()
        
        # Prepare texts for embedding
        texts = []
        chunk_metadata = []
        
        for chunk in document_chunks:
            # Optimize text for voyage-code-3 model
            optimized_text = self._optimize_text_for_embedding(chunk.content)
            texts.append(optimized_text)
            
            chunk_metadata.append({
                "chunk_id": chunk.chunk_id,
                "source_context": chunk.source_context,
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash
            })
        
        # Process embeddings in batches
        batch_result = await self.processor.process_documents_batch(
            texts,
            input_type="document"
        )
        
        # Create embedding documents for vector storage
        embedding_documents = []
        for i, (chunk, embedding) in enumerate(zip(document_chunks, batch_result.embeddings)):
            embedding_doc = EmbeddingDocument(
                chunk_id=chunk.chunk_id,
                library_id=library_id,
                version=version,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk_metadata[i],
                model="voyage-code-3",
                created_at=datetime.utcnow()
            )
            embedding_documents.append(embedding_doc)
        
        # Update metrics
        processing_time = time.time() - start_time
        await self.metrics.record_api_call(
            batch_size=len(texts),
            latency=processing_time,
            success=len(batch_result.embeddings) == len(texts),
            tokens_used=sum(chunk.token_count for chunk in document_chunks)
        )
        
        return EmbeddingProcessingResult(
            embedding_documents=embedding_documents,
            processing_time=processing_time,
            cache_hit_rate=batch_result.cache_hits / len(texts),
            api_calls=batch_result.api_results.api_calls if batch_result.api_results else 0,
            success=True
        )
    
    def _optimize_text_for_embedding(self, content: str) -> str:
        """Optimize text content for voyage-code-3 model performance."""
        
        # voyage-code-3 specific optimizations
        optimizations = [
            self._normalize_code_blocks,
            self._enhance_technical_context,
            self._remove_redundant_whitespace,
            self._preserve_code_structure
        ]
        
        optimized_content = content
        for optimization in optimizations:
            optimized_content = optimization(optimized_content)
        
        return optimized_content
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate optimized embedding for search queries."""
        
        # Optimize query for semantic search
        optimized_query = self._optimize_query_for_search(query)
        
        # Use cache for common queries
        cache_key = f"query:{optimized_query}"
        cached_embedding = await self.cache.get_cached_embeddings(
            [optimized_query], 
            input_type="query"
        )
        
        if cached_embedding[0]:
            return list(cached_embedding[0].values())[0]
        
        # Generate new embedding
        result = await self.client.client.embed(
            texts=[optimized_query],
            model=self.client.config.model,
            input_type="query"
        )
        
        embedding = result.embeddings[0]
        
        # Cache query embedding
        await self.cache.cache_embeddings(
            [(optimized_query, embedding)],
            input_type="query"
        )
        
        return embedding
```

## Configuration Templates

### Production Configuration

```yaml
# voyage_ai_config.yaml
voyage_client:
  model: "voyage-code-3"
  api_key_env: "VOYAGE_API_KEY"
  max_retries: 3
  timeout: 30
  
  # Rate limiting
  rate_limits:
    requests_per_minute: 300
    tokens_per_minute: 1000000
  
  # Batch processing
  batch_processing:
    default_batch_size: 100
    max_concurrent_batches: 3
    optimal_chunk_size: 1000
    chunk_overlap: 200
  
  # Circuit breaker
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
  
  # Caching
  cache:
    enabled: true
    max_entries: 100000
    ttl_hours: 168  # 7 days
    cleanup_threshold: 0.8

# Performance targets
performance:
  throughput_target: 1200  # documents/minute
  cache_hit_rate_target: 0.6  # 60%
  max_error_rate: 0.001  # 0.1%
  max_p95_latency: 2.0  # seconds
```

This comprehensive technical specification provides everything needed to implement a production-ready Voyage AI embedding service that integrates seamlessly with the Contexter RAG system while meeting all performance and reliability requirements.