"""
High-Throughput Batch Processor

Advanced batch processing system for embedding generation with intelligent
optimization, priority queues, and comprehensive error handling.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
import heapq

from ..models.embedding_models import (
    EmbeddingRequest, EmbeddingResult, BatchRequest, BatchResult,
    InputType, ProcessingStatus, ProcessingMetrics
)
from .voyage_client import VoyageAIClient, VoyageAPIError
from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Batch processing priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing."""
    
    default_batch_size: int = 100
    max_batch_size: int = 200
    min_batch_size: int = 10
    max_concurrent_batches: int = 5
    batch_timeout_seconds: float = 30.0
    queue_timeout_seconds: float = 300.0
    adaptive_batching: bool = True
    auto_optimize: bool = True
    priority_boost_factor: float = 1.5


@dataclass
class PriorityBatch:
    """Priority queue item for batch processing."""
    
    priority: int
    created_at: float
    batch: BatchRequest
    
    def __lt__(self, other):
        """Compare for priority queue (higher priority first)."""
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


class AdaptiveBatchSizer:
    """Adaptive batch size optimization based on performance metrics."""
    
    def __init__(self, initial_size: int = 100):
        self.current_size = initial_size
        self.performance_history = deque(maxlen=50)
        self.min_size = 10
        self.max_size = 200
        self.optimization_threshold = 10  # Minimum samples before optimization
    
    def record_performance(self, batch_size: int, throughput: float, latency: float):
        """Record batch performance for optimization."""
        efficiency = throughput / latency if latency > 0 else 0
        self.performance_history.append({
            'size': batch_size,
            'throughput': throughput,
            'latency': latency,
            'efficiency': efficiency,
            'timestamp': time.time()
        })
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history."""
        if len(self.performance_history) < self.optimization_threshold:
            return self.current_size
        
        # Analyze recent performance
        recent_samples = list(self.performance_history)[-20:]
        
        # Group by batch size and calculate average efficiency
        size_performance = defaultdict(list)
        for sample in recent_samples:
            size_performance[sample['size']].append(sample['efficiency'])
        
        # Find size with best average efficiency
        best_size = self.current_size
        best_efficiency = 0
        
        for size, efficiencies in size_performance.items():
            if len(efficiencies) >= 3:  # Need minimum samples
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                if avg_efficiency > best_efficiency:
                    best_efficiency = avg_efficiency
                    best_size = size
        
        # Gradual adjustment towards optimal size
        if best_size != self.current_size:
            adjustment = (best_size - self.current_size) * 0.2  # 20% adjustment
            self.current_size = max(
                self.min_size,
                min(self.max_size, int(self.current_size + adjustment))
            )
        
        return self.current_size


class BatchProcessor:
    """
    High-performance batch processor for embedding generation.
    
    Features:
    - Priority-based processing queues
    - Adaptive batch sizing for optimal performance
    - Intelligent caching integration
    - Comprehensive error handling and retry logic
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        voyage_client: VoyageAIClient,
        cache: Optional[EmbeddingCache] = None,
        config: Optional[BatchProcessingConfig] = None
    ):
        self.voyage_client = voyage_client
        self.cache = cache
        self.config = config or BatchProcessingConfig()
        
        # Processing queues by priority
        self.priority_queue = []
        self.queue_lock = asyncio.Lock()
        
        # Concurrency control
        self.processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        # Adaptive batch sizing
        self.batch_sizer = AdaptiveBatchSizer(self.config.default_batch_size)
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.performance_history = deque(maxlen=1000)
        
        # Background processing
        self._processing_tasks = []
        self._shutdown_event = asyncio.Event()
        self._stats_lock = asyncio.Lock()
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.last_errors = deque(maxlen=100)
    
    async def initialize(self):
        """Initialize the batch processor."""
        logger.info("Initializing batch processor")
        
        # Start background processing workers
        for i in range(self.config.max_concurrent_batches):
            task = asyncio.create_task(self._batch_worker(f"worker-{i}"))
            self._processing_tasks.append(task)
        
        logger.info(f"Started {len(self._processing_tasks)} batch processing workers")
    
    async def shutdown(self):
        """Shutdown the batch processor."""
        logger.info("Shutting down batch processor")
        
        self._shutdown_event.set()
        
        # Wait for all workers to complete
        await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        logger.info("Batch processor shutdown complete")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def process_embeddings(
        self,
        requests: List[EmbeddingRequest],
        priority: BatchPriority = BatchPriority.NORMAL,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BatchResult:
        """
        Process embedding requests with batching and optimization.
        
        Args:
            requests: List of embedding requests
            priority: Processing priority
            progress_callback: Optional progress callback
            
        Returns:
            Batch processing result
        """
        if not requests:
            return BatchResult(
                batch_id="empty",
                results=[],
                processing_time=0.0,
                completed_at=datetime.utcnow()
            )
        
        start_time = time.time()
        
        # Step 1: Check cache for existing embeddings
        cached_results, uncached_requests = await self._check_cache_batch(requests)
        
        # Step 2: Process uncached requests if any
        api_results = []
        if uncached_requests:
            api_results = await self._process_uncached_requests(
                uncached_requests,
                priority,
                progress_callback
            )
        
        # Step 3: Combine results in original order
        all_results = await self._combine_results(requests, cached_results, api_results)
        
        # Step 4: Cache new results
        if api_results:
            await self._cache_new_results(uncached_requests, api_results)
        
        # Step 5: Create final batch result
        processing_time = time.time() - start_time
        batch_result = BatchResult(
            batch_id=f"batch_{int(time.time() * 1000)}",
            results=all_results,
            processing_time=processing_time,
            cache_hits=len(cached_results),
            api_calls=len(api_results) // self.batch_sizer.current_size + (1 if len(api_results) % self.batch_sizer.current_size else 0),
            tokens_used=sum(req.metadata.get('token_count', 0) for req in uncached_requests),
            completed_at=datetime.utcnow()
        )
        
        # Update metrics
        await self._update_metrics(batch_result)
        
        logger.info(
            f"Processed {len(requests)} requests: "
            f"{len(cached_results)} cached, {len(api_results)} new, "
            f"in {processing_time:.2f}s"
        )
        
        return batch_result
    
    async def _check_cache_batch(
        self,
        requests: List[EmbeddingRequest]
    ) -> Tuple[Dict[str, EmbeddingResult], List[EmbeddingRequest]]:
        """Check cache for existing embeddings."""
        if not self.cache:
            return {}, requests
        
        # Group requests by model and input type for efficient cache lookup
        grouped_requests = defaultdict(list)
        for req in requests:
            key = (req.metadata.get('model', 'voyage-code-3'), req.input_type)
            grouped_requests[key].append(req)
        
        cached_results = {}
        uncached_requests = []
        
        for (model, input_type), group_requests in grouped_requests.items():
            content_hashes = [req.content_hash for req in group_requests]
            
            # Get cached embeddings
            cached_embeddings = await self.cache.get_embeddings(
                content_hashes, model, input_type
            )
            
            for req in group_requests:
                if req.content_hash in cached_embeddings:
                    # Cache hit
                    cached_results[req.content_hash] = EmbeddingResult(
                        content_hash=req.content_hash,
                        embedding=cached_embeddings[req.content_hash],
                        model=model,
                        dimensions=len(cached_embeddings[req.content_hash]),
                        processing_time=0.001,  # Minimal cache lookup time
                        cache_hit=True,
                        status=ProcessingStatus.CACHED,
                        request_metadata=req.metadata
                    )
                else:
                    # Cache miss
                    uncached_requests.append(req)
        
        return cached_results, uncached_requests
    
    async def _process_uncached_requests(
        self,
        requests: List[EmbeddingRequest],
        priority: BatchPriority,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EmbeddingResult]:
        """Process requests that are not in cache."""
        if not requests:
            return []
        
        # Create batches with optimal sizing
        batches = self._create_optimal_batches(requests, priority)
        
        # Submit batches to processing queue
        batch_futures = []
        for batch in batches:
            future = asyncio.Future()
            await self._enqueue_batch(batch, future)
            batch_futures.append(future)
        
        # Wait for all batches to complete
        results = []
        completed = 0
        
        for future in asyncio.as_completed(batch_futures):
            try:
                batch_result = await future
                results.extend(batch_result.results)
                completed += len(batch_result.results)
                
                if progress_callback:
                    progress_callback(completed, len(requests))
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Create error results for failed batch
                # This would require mapping back to original requests
        
        return results
    
    def _create_optimal_batches(
        self,
        requests: List[EmbeddingRequest],
        priority: BatchPriority
    ) -> List[BatchRequest]:
        """Create optimally sized batches from requests."""
        if self.config.adaptive_batching:
            batch_size = self.batch_sizer.get_optimal_batch_size()
        else:
            batch_size = self.config.default_batch_size
        
        batches = []
        
        # Sort requests by priority within the group
        sorted_requests = sorted(
            requests,
            key=lambda x: (x.priority, x.input_type.value),
            reverse=True
        )
        
        # Create batches
        for i in range(0, len(sorted_requests), batch_size):
            batch_requests = sorted_requests[i:i + batch_size]
            
            batch = BatchRequest(
                requests=batch_requests,
                priority=priority.value,
                timeout=self.config.batch_timeout_seconds
            )
            
            batches.append(batch)
        
        return batches
    
    async def _enqueue_batch(self, batch: BatchRequest, future: asyncio.Future):
        """Add batch to priority processing queue."""
        priority_item = PriorityBatch(
            priority=batch.priority,
            created_at=time.time(),
            batch=batch
        )
        
        async with self.queue_lock:
            heapq.heappush(self.priority_queue, (priority_item, future))
    
    async def _batch_worker(self, worker_id: str):
        """Background worker for processing batches."""
        logger.debug(f"Batch worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next batch from priority queue
                priority_item, future = await self._dequeue_batch()
                
                if priority_item is None:
                    continue
                
                # Process the batch
                async with self.processing_semaphore:
                    result = await self._process_single_batch(
                        priority_item.batch, worker_id
                    )
                    future.set_result(result)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                if 'future' in locals() and not future.done():
                    future.set_exception(e)
                await asyncio.sleep(1)  # Brief pause before retrying
        
        logger.debug(f"Batch worker {worker_id} stopped")
    
    async def _dequeue_batch(self) -> Tuple[Optional[PriorityBatch], Optional[asyncio.Future]]:
        """Get next batch from priority queue."""
        while not self._shutdown_event.is_set():
            async with self.queue_lock:
                if self.priority_queue:
                    return heapq.heappop(self.priority_queue)
            
            # Wait briefly if queue is empty
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.1)
                break  # Shutdown signaled
            except asyncio.TimeoutError:
                continue  # Check queue again
        
        return None, None
    
    async def _process_single_batch(
        self,
        batch: BatchRequest,
        worker_id: str
    ) -> BatchResult:
        """Process a single batch of embedding requests."""
        start_time = time.time()
        
        try:
            # Prepare texts for API call
            texts = [req.content for req in batch.requests]
            model = batch.requests[0].metadata.get('model', 'voyage-code-3')
            input_type = batch.requests[0].input_type
            
            # Make API call
            embeddings = await self.voyage_client.generate_embeddings(
                texts=texts,
                model=model,
                input_type=input_type
            )
            
            # Create results
            results = []
            processing_time = time.time() - start_time
            
            for req, embedding in zip(batch.requests, embeddings):
                result = EmbeddingResult(
                    content_hash=req.content_hash,
                    embedding=embedding,
                    model=model,
                    dimensions=len(embedding),
                    processing_time=processing_time / len(batch.requests),
                    cache_hit=False,
                    status=ProcessingStatus.COMPLETED,
                    request_metadata=req.metadata,
                    tokens_used=req.metadata.get('token_count', 0),
                    created_at=datetime.utcnow()
                )
                results.append(result)
            
            # Record performance for adaptive sizing
            throughput = len(batch.requests) / processing_time * 60  # per minute
            self.batch_sizer.record_performance(
                batch_size=len(batch.requests),
                throughput=throughput,
                latency=processing_time
            )
            
            logger.debug(
                f"Worker {worker_id} processed batch of {len(batch.requests)} "
                f"in {processing_time:.2f}s (throughput: {throughput:.1f}/min)"
            )
            
            return BatchResult(
                batch_id=batch.batch_id,
                results=results,
                processing_time=processing_time,
                cache_hits=0,
                api_calls=1,
                tokens_used=sum(r.tokens_used for r in results),
                completed_at=datetime.utcnow()
            )
            
        except Exception as e:
            # Handle batch processing errors
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Record error
            self.error_counts[type(e).__name__] += 1
            self.last_errors.append({
                'timestamp': datetime.utcnow(),
                'error': error_msg,
                'batch_size': len(batch.requests),
                'worker_id': worker_id
            })
            
            # Create error results
            results = []
            for req in batch.requests:
                result = EmbeddingResult(
                    content_hash=req.content_hash,
                    embedding=[],
                    model=req.metadata.get('model', 'voyage-code-3'),
                    dimensions=0,
                    processing_time=processing_time / len(batch.requests),
                    cache_hit=False,
                    status=ProcessingStatus.FAILED,
                    error=error_msg,
                    request_metadata=req.metadata
                )
                results.append(result)
            
            logger.error(f"Batch processing failed: {error_msg}")
            
            return BatchResult(
                batch_id=batch.batch_id,
                results=results,
                processing_time=processing_time,
                errors=[error_msg],
                completed_at=datetime.utcnow()
            )
    
    async def _combine_results(
        self,
        original_requests: List[EmbeddingRequest],
        cached_results: Dict[str, EmbeddingResult],
        api_results: List[EmbeddingResult]
    ) -> List[EmbeddingResult]:
        """Combine cached and API results in original request order."""
        # Create lookup for API results
        api_results_map = {r.content_hash: r for r in api_results}
        
        # Combine results in original order
        combined_results = []
        for req in original_requests:
            if req.content_hash in cached_results:
                combined_results.append(cached_results[req.content_hash])
            elif req.content_hash in api_results_map:
                combined_results.append(api_results_map[req.content_hash])
            else:
                # Create error result for missing embedding
                error_result = EmbeddingResult(
                    content_hash=req.content_hash,
                    embedding=[],
                    model=req.metadata.get('model', 'voyage-code-3'),
                    dimensions=0,
                    processing_time=0.0,
                    cache_hit=False,
                    status=ProcessingStatus.FAILED,
                    error="Embedding not found in cache or API results",
                    request_metadata=req.metadata
                )
                combined_results.append(error_result)
        
        return combined_results
    
    async def _cache_new_results(
        self,
        requests: List[EmbeddingRequest],
        results: List[EmbeddingResult]
    ):
        """Cache new embedding results."""
        if not self.cache:
            return
        
        # Create cache entries for successful results
        cache_entries = []
        result_map = {r.content_hash: r for r in results if r.success}
        
        for req in requests:
            if req.content_hash in result_map:
                result = result_map[req.content_hash]
                
                from ..models.embedding_models import CacheEntry
                cache_entry = CacheEntry(
                    content_hash=req.content_hash,
                    content=req.content,
                    embedding=result.embedding,
                    model=result.model,
                    input_type=req.input_type,
                    metadata=req.metadata
                )
                cache_entries.append(cache_entry)
        
        if cache_entries:
            await self.cache.store_embeddings(cache_entries)
    
    async def _update_metrics(self, batch_result: BatchResult):
        """Update processing metrics."""
        async with self._stats_lock:
            self.metrics.update_metrics(batch_result)
            
            # Record in performance history
            self.performance_history.append({
                'timestamp': batch_result.completed_at,
                'batch_size': batch_result.total_requests,
                'processing_time': batch_result.processing_time,
                'throughput': batch_result.throughput,
                'cache_hits': batch_result.cache_hits,
                'success_rate': batch_result.success_rate
            })
    
    def get_performance_metrics(self) -> ProcessingMetrics:
        """Get current performance metrics."""
        return self.metrics
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get information about processing queues."""
        return {
            'queue_length': len(self.priority_queue),
            'active_workers': len([t for t in self._processing_tasks if not t.done()]),
            'total_workers': len(self._processing_tasks),
            'current_batch_size': self.batch_sizer.current_size,
            'error_counts': dict(self.error_counts),
            'recent_errors': list(self.last_errors)[-10:]  # Last 10 errors
        }
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get batch size optimization information."""
        return {
            'current_batch_size': self.batch_sizer.current_size,
            'min_batch_size': self.batch_sizer.min_size,
            'max_batch_size': self.batch_sizer.max_size,
            'performance_samples': len(self.batch_sizer.performance_history),
            'adaptive_enabled': self.config.adaptive_batching,
            'auto_optimize_enabled': self.config.auto_optimize
        }