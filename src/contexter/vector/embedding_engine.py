"""
RAG Embedding Engine

Main embedding engine that orchestrates the complete embedding generation system
with Voyage AI integration, caching, batch processing, and monitoring.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from ..models.embedding_models import (
    IEmbeddingEngine, EmbeddingRequest, EmbeddingResult, BatchResult,
    InputType, ProcessingMetrics, CacheStats, PerformanceTargets,
    estimate_token_count, normalize_content_for_hashing
)
from .voyage_client import VoyageAIClient, VoyageClientConfig
from .embedding_cache import EmbeddingCache
from .batch_processor import BatchProcessor, BatchProcessingConfig, BatchPriority

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEngineConfig:
    """Configuration for the embedding engine."""
    
    # Voyage AI configuration
    voyage_api_key: str
    voyage_model: str = "voyage-code-3"
    
    # Cache configuration
    cache_path: str = "~/.contexter/embedding_cache.db"
    cache_max_entries: int = 100000
    cache_ttl_hours: int = 168  # 7 days
    
    # Batch processing configuration
    batch_size: int = 100
    max_concurrent_batches: int = 5
    adaptive_batching: bool = True
    
    # Performance targets
    target_throughput_per_minute: int = 1000
    target_cache_hit_rate: float = 0.5
    max_error_rate: float = 0.001
    
    # Monitoring
    enable_detailed_logging: bool = True
    enable_performance_tracking: bool = True


class VoyageEmbeddingEngine(IEmbeddingEngine):
    """
    Production-ready embedding engine for RAG systems.
    
    Features:
    - Voyage AI integration with rate limiting and circuit breakers
    - Intelligent caching with LRU eviction
    - High-throughput batch processing with adaptive optimization
    - Comprehensive performance monitoring
    - Automatic error recovery and retry logic
    """
    
    def __init__(self, config: EmbeddingEngineConfig):
        self.config = config
        
        # Core components
        self.voyage_client: Optional[VoyageAIClient] = None
        self.cache: Optional[EmbeddingCache] = None
        self.batch_processor: Optional[BatchProcessor] = None
        
        # Performance tracking
        self.metrics = ProcessingMetrics()
        self.performance_targets = PerformanceTargets(
            min_throughput_per_minute=config.target_throughput_per_minute,
            min_cache_hit_rate=config.target_cache_hit_rate,
            max_error_rate=config.max_error_rate
        )
        
        # State management
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        
        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        
        logger.info("Embedding engine configured with Voyage AI")
    
    async def initialize(self):
        """Initialize all engine components."""
        if self._initialized:
            return
        
        logger.info("Initializing embedding engine")
        
        try:
            # Initialize Voyage AI client
            voyage_config = VoyageClientConfig(
                api_key=self.config.voyage_api_key,
                model=self.config.voyage_model,
                max_concurrent_requests=self.config.max_concurrent_batches
            )
            self.voyage_client = VoyageAIClient(voyage_config)
            await self.voyage_client.initialize()
            
            # Initialize cache
            self.cache = EmbeddingCache(
                cache_path=self.config.cache_path,
                max_entries=self.config.cache_max_entries,
                ttl_hours=self.config.cache_ttl_hours
            )
            await self.cache.initialize()
            
            # Initialize batch processor
            batch_config = BatchProcessingConfig(
                default_batch_size=self.config.batch_size,
                max_concurrent_batches=self.config.max_concurrent_batches,
                adaptive_batching=self.config.adaptive_batching
            )
            self.batch_processor = BatchProcessor(
                voyage_client=self.voyage_client,
                cache=self.cache,
                config=batch_config
            )
            await self.batch_processor.initialize()
            
            # Start monitoring if enabled
            if self.config.enable_performance_tracking:
                self._monitoring_task = asyncio.create_task(self._performance_monitor())
            
            self._initialized = True
            logger.info("Embedding engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding engine: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown the embedding engine."""
        logger.info("Shutting down embedding engine")
        
        self._shutdown_event.set()
        
        # Stop monitoring
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown components
        if self.batch_processor:
            await self.batch_processor.shutdown()
        
        if self.cache:
            await self.cache.shutdown()
        
        if self.voyage_client:
            await self.voyage_client.close()
        
        self._initialized = False
        logger.info("Embedding engine shutdown complete")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """
        Generate single embedding with caching.
        
        Args:
            request: Embedding request with content and metadata
            
        Returns:
            Embedding result with vector and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        # Process as single-item batch
        batch_result = await self.generate_batch_embeddings([request])
        
        if batch_result.results:
            return batch_result.results[0]
        else:
            # Return error result
            return EmbeddingResult(
                content_hash=request.content_hash,
                embedding=[],
                model=self.config.voyage_model,
                dimensions=0,
                processing_time=0.0,
                cache_hit=False,
                error="No result generated"
            )
    
    async def generate_batch_embeddings(
        self,
        requests: List[EmbeddingRequest]
    ) -> BatchResult:
        """
        Generate embeddings in batches for optimal throughput.
        
        Args:
            requests: List of embedding requests
            
        Returns:
            Batch result with all embeddings and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        if not requests:
            return BatchResult(
                batch_id="empty",
                results=[],
                processing_time=0.0
            )
        
        start_time = time.time()
        
        try:
            # Validate and prepare requests
            validated_requests = await self._validate_requests(requests)
            
            # Process through batch processor
            result = await self.batch_processor.process_embeddings(
                validated_requests,
                priority=BatchPriority.NORMAL
            )
            
            # Update engine metrics
            await self._update_metrics(result)
            
            logger.debug(
                f"Generated {len(result.results)} embeddings "
                f"({result.cache_hits} cached) in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            
            # Create error results
            error_results = []
            for req in requests:
                error_result = EmbeddingResult(
                    content_hash=req.content_hash,
                    embedding=[],
                    model=self.config.voyage_model,
                    dimensions=0,
                    processing_time=0.0,
                    cache_hit=False,
                    error=str(e)
                )
                error_results.append(error_result)
            
            return BatchResult(
                batch_id="error",
                results=error_results,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Quick embedding generation for search queries.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector for the query
        """
        if not self._initialized:
            await self.initialize()
        
        # Prepare query request
        normalized_query = normalize_content_for_hashing(query)
        
        request = EmbeddingRequest(
            content=normalized_query,
            input_type=InputType.QUERY,
            metadata={
                'model': self.config.voyage_model,
                'token_count': estimate_token_count(normalized_query),
                'query_type': 'search'
            }
        )
        
        # Generate embedding
        result = await self.generate_embedding(request)
        
        if result.success:
            return result.embedding
        else:
            logger.error(f"Query embedding failed: {result.error}")
            raise RuntimeError(f"Failed to generate query embedding: {result.error}")
    
    async def embed_documents(
        self,
        documents: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[List[float]]:
        """
        Embed multiple documents efficiently.
        
        Args:
            documents: List of document texts
            metadata_list: Optional metadata for each document
            
        Returns:
            List of embedding vectors
        """
        if not documents:
            return []
        
        # Prepare requests
        requests = []
        for i, doc in enumerate(documents):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            metadata.update({
                'model': self.config.voyage_model,
                'token_count': estimate_token_count(doc),
                'document_index': i
            })
            
            request = EmbeddingRequest(
                content=doc,
                input_type=InputType.DOCUMENT,
                metadata=metadata
            )
            requests.append(request)
        
        # Process batch
        batch_result = await self.generate_batch_embeddings(requests)
        
        # Extract embeddings in order
        embeddings = []
        for result in batch_result.results:
            if result.success:
                embeddings.append(result.embedding)
            else:
                logger.warning(f"Document embedding failed: {result.error}")
                embeddings.append([])  # Empty embedding for failed documents
        
        return embeddings
    
    async def _validate_requests(
        self,
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingRequest]:
        """Validate and enrich embedding requests."""
        validated_requests = []
        
        for req in requests:
            # Ensure content hash is set
            if not req.content_hash:
                req.content_hash = req._generate_content_hash()
            
            # Add model to metadata if not present
            if 'model' not in req.metadata:
                req.metadata['model'] = self.config.voyage_model
            
            # Add token count estimate
            if 'token_count' not in req.metadata:
                req.metadata['token_count'] = estimate_token_count(req.content)
            
            # Validate content
            if not req.content or not req.content.strip():
                logger.warning("Skipping empty content request")
                continue
            
            # Check content length (rough token limit check)
            estimated_tokens = req.metadata.get('token_count', 0)
            if estimated_tokens > 30000:  # Voyage API limit
                logger.warning(f"Content too long ({estimated_tokens} tokens), truncating")
                # Truncate content to approximate token limit
                words = req.content.split()
                truncated_words = words[:int(30000 / 1.3)]  # Conservative estimate
                req.content = ' '.join(truncated_words)
                req.metadata['token_count'] = estimate_token_count(req.content)
                req.metadata['truncated'] = True
            
            validated_requests.append(req)
        
        return validated_requests
    
    async def _update_metrics(self, batch_result: BatchResult):
        """Update engine performance metrics."""
        self.metrics.update_metrics(batch_result)
        
        if self.config.enable_detailed_logging:
            logger.info(
                f"Batch metrics: {batch_result.total_requests} requests, "
                f"{batch_result.success_rate:.1%} success rate, "
                f"{batch_result.throughput:.1f} docs/min throughput"
            )
    
    async def _performance_monitor(self):
        """Background task for performance monitoring and optimization."""
        logger.info("Starting performance monitoring")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for next monitoring cycle
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=60  # Monitor every minute
                )
            except asyncio.TimeoutError:
                pass  # Expected timeout, continue monitoring
            
            if self._shutdown_event.is_set():
                break
            
            try:
                # Check performance against targets
                compliance = self.performance_targets.check_compliance(self.metrics)
                
                # Log performance status
                if self.config.enable_detailed_logging:
                    logger.info(f"Performance compliance: {compliance}")
                
                # Check for performance issues
                if not compliance.get('throughput', True):
                    logger.warning(
                        f"Throughput below target: {self.metrics.throughput_per_minute:.1f} "
                        f"< {self.performance_targets.min_throughput_per_minute}"
                    )
                
                if not compliance.get('cache_hit_rate', True):
                    logger.warning(
                        f"Cache hit rate below target: {self.metrics.cache_hit_rate:.1%} "
                        f"< {self.performance_targets.min_cache_hit_rate:.1%}"
                    )
                
                if not compliance.get('error_rate', True):
                    error_rate = 1 - self.metrics.success_rate
                    logger.warning(
                        f"Error rate above target: {error_rate:.1%} "
                        f"> {self.performance_targets.max_error_rate:.1%}"
                    )
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
        
        logger.info("Performance monitoring stopped")
    
    def get_cache_statistics(self) -> CacheStats:
        """Get cache performance statistics."""
        if not self.cache:
            return CacheStats()
        
        # This would need to be made synchronous or we need an async version
        # For now, return empty stats if cache not available
        return CacheStats()
    
    def get_performance_metrics(self) -> ProcessingMetrics:
        """Get processing performance metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on embedding service.
        
        Returns:
            Health status with detailed component information
        """
        health_status = {
            "status": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
            "performance": {},
            "compliance": {}
        }
        
        try:
            if not self._initialized:
                health_status["status"] = "not_initialized"
                return health_status
            
            # Check Voyage AI client
            if self.voyage_client:
                client_health = await self.voyage_client.health_check()
                health_status["components"]["voyage_client"] = client_health
            
            # Check cache
            if self.cache:
                cache_stats = await self.cache.get_statistics()
                health_status["components"]["cache"] = {
                    "status": "healthy",
                    "total_entries": cache_stats.total_entries,
                    "hit_rate": cache_stats.hit_rate,
                    "size_mb": round(cache_stats.total_size_bytes / (1024 * 1024), 2)
                }
            
            # Check batch processor
            if self.batch_processor:
                queue_info = self.batch_processor.get_queue_info()
                health_status["components"]["batch_processor"] = {
                    "status": "healthy",
                    "queue_length": queue_info["queue_length"],
                    "active_workers": queue_info["active_workers"],
                    "current_batch_size": queue_info["current_batch_size"]
                }
            
            # Performance metrics
            health_status["performance"] = {
                "total_requests": self.metrics.total_requests,
                "success_rate": round(self.metrics.success_rate, 3),
                "cache_hit_rate": round(self.metrics.cache_hit_rate, 3),
                "throughput_per_minute": round(self.metrics.throughput_per_minute, 1),
                "average_latency_ms": round(self.metrics.average_processing_time * 1000, 2)
            }
            
            # Performance compliance
            compliance = self.performance_targets.check_compliance(self.metrics)
            health_status["compliance"] = compliance
            
            # Overall status
            all_components_healthy = all(
                comp.get("status") == "healthy" 
                for comp in health_status["components"].values()
            )
            
            all_targets_met = all(compliance.values())
            
            if all_components_healthy and all_targets_met:
                health_status["status"] = "healthy"
            elif all_components_healthy:
                health_status["status"] = "degraded"  # Components OK but performance issues
            else:
                health_status["status"] = "unhealthy"
            
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive status information for monitoring."""
        status = {
            "engine": {
                "initialized": self._initialized,
                "config": {
                    "model": self.config.voyage_model,
                    "batch_size": self.config.batch_size,
                    "cache_ttl_hours": self.config.cache_ttl_hours
                }
            },
            "health": await self.health_check(),
            "metrics": {
                "processing": {
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "success_rate": self.metrics.success_rate,
                    "throughput_per_minute": self.metrics.throughput_per_minute
                },
                "caching": {
                    "cache_hits": self.metrics.cache_hits,
                    "cache_misses": self.metrics.cache_misses,
                    "hit_rate": self.metrics.cache_hit_rate
                },
                "cost": {
                    "total_api_calls": self.metrics.total_api_calls,
                    "total_tokens": self.metrics.total_tokens_processed,
                    "estimated_cost": self.metrics.total_cost
                }
            }
        }
        
        # Add component-specific details
        if self.batch_processor:
            status["batch_processor"] = self.batch_processor.get_queue_info()
            status["optimization"] = self.batch_processor.get_optimization_info()
        
        if self.voyage_client:
            status["rate_limiting"] = {
                "requests_per_minute": self.voyage_client.config.rate_limit_rpm,
                "tokens_per_minute": self.voyage_client.config.rate_limit_tpm,
                "current_usage": self.voyage_client.get_rate_limit_info().__dict__
            }
            status["latency_percentiles"] = self.voyage_client.get_latency_percentiles()
        
        return status


# Factory function for easy engine creation
async def create_embedding_engine(
    voyage_api_key: str,
    cache_path: Optional[str] = None,
    **config_overrides
) -> VoyageEmbeddingEngine:
    """
    Create and initialize an embedding engine.
    
    Args:
        voyage_api_key: Voyage AI API key
        cache_path: Optional cache database path
        **config_overrides: Additional configuration overrides
        
    Returns:
        Initialized embedding engine
    """
    config_dict = {
        "voyage_api_key": voyage_api_key
    }
    
    if cache_path:
        config_dict["cache_path"] = cache_path
    
    config_dict.update(config_overrides)
    
    config = EmbeddingEngineConfig(**config_dict)
    engine = VoyageEmbeddingEngine(config)
    await engine.initialize()
    
    return engine