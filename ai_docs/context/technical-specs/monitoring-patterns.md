# RAG Monitoring and Observability Technical Specifications

## Overview
- **Purpose**: Comprehensive monitoring and observability for RAG system performance, health, and business metrics
- **Version**: Prometheus 2.45+ with Python Client 0.19+
- **Last Updated**: 2025-01-12

## Key Concepts

### Monitoring Architecture Layers
- **Metrics Collection**: Prometheus-based time series metrics with custom business KPIs
- **Distributed Tracing**: OpenTelemetry-based request flow visualization  
- **Structured Logging**: Centralized logging with correlation IDs and context
- **Alerting**: Intelligent alerting with correlation and noise reduction
- **Dashboards**: Real-time operational and business intelligence visualization

### Core Metrics Categories
- **Search Performance**: Latency percentiles, throughput, error rates
- **System Health**: Resource utilization, component availability, dependency status
- **Business Intelligence**: Query patterns, user engagement, library usage statistics
- **Data Pipeline**: Ingestion rates, processing times, data quality metrics

## Implementation Patterns

### Pattern: RAG Metrics Collection Framework
```python
from prometheus_client import (
    Counter, Histogram, Gauge, Enum, Info, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
import time
import asyncio
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from functools import wraps
import logging
from dataclasses import dataclass

@dataclass
class MetricsConfig:
    """Configuration for RAG metrics collection."""
    namespace: str = "rag"
    subsystem: str = ""
    registry: Optional[CollectorRegistry] = None
    enable_exemplars: bool = True
    histogram_buckets: List[float] = None
    
    def __post_init__(self):
        if self.registry is None:
            self.registry = CollectorRegistry()
        if self.histogram_buckets is None:
            self.histogram_buckets = [
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            ]

class RAGMetricsCollector:
    """Centralized metrics collector for RAG system components."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = config.registry
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all RAG system metrics."""
        
        # Search Engine Metrics
        self.search_requests_total = Counter(
            f'{self.config.namespace}_search_requests_total',
            'Total number of search requests processed',
            ['search_type', 'status', 'library_id'],
            registry=self.registry
        )
        
        self.search_latency_seconds = Histogram(
            f'{self.config.namespace}_search_latency_seconds',
            'Search request latency distribution in seconds',
            ['search_type', 'complexity'],
            buckets=self.config.histogram_buckets,
            registry=self.registry
        )
        
        self.search_results_count = Histogram(
            f'{self.config.namespace}_search_results_count',
            'Number of results returned per search',
            ['search_type'],
            buckets=[0, 1, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        # Document Ingestion Metrics
        self.documents_processed_total = Counter(
            f'{self.config.namespace}_documents_processed_total',
            'Total number of documents processed',
            ['status', 'document_type', 'library_id'],
            registry=self.registry
        )
        
        self.document_processing_duration_seconds = Histogram(
            f'{self.config.namespace}_document_processing_duration_seconds',
            'Time spent processing individual documents',
            ['document_type', 'processing_stage'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.ingestion_throughput_docs_per_minute = Gauge(
            f'{self.config.namespace}_ingestion_throughput_docs_per_minute',
            'Current document ingestion throughput',
            ['library_id'],
            registry=self.registry
        )
        
        self.ingestion_queue_size = Gauge(
            f'{self.config.namespace}_ingestion_queue_size',
            'Number of documents waiting to be processed',
            registry=self.registry
        )
        
        # Vector Storage Metrics
        self.vector_operations_total = Counter(
            f'{self.config.namespace}_vector_operations_total',
            'Total vector database operations',
            ['operation_type', 'status', 'collection_name'],
            registry=self.registry
        )
        
        self.vector_operation_duration_seconds = Histogram(
            f'{self.config.namespace}_vector_operation_duration_seconds',
            'Vector operation latency distribution',
            ['operation_type', 'collection_name'],
            buckets=self.config.histogram_buckets,
            registry=self.registry
        )
        
        self.vector_storage_size_bytes = Gauge(
            f'{self.config.namespace}_vector_storage_size_bytes',
            'Total size of vector storage in bytes',
            ['collection_name'],
            registry=self.registry
        )
        
        self.vector_count_total = Gauge(
            f'{self.config.namespace}_vector_count_total',
            'Total number of vectors stored',
            ['collection_name'],
            registry=self.registry
        )
        
        # Embedding Service Metrics
        self.embedding_requests_total = Counter(
            f'{self.config.namespace}_embedding_requests_total',
            'Total embedding generation requests',
            ['status', 'model', 'batch_size_bucket'],
            registry=self.registry
        )
        
        self.embedding_generation_duration_seconds = Histogram(
            f'{self.config.namespace}_embedding_generation_duration_seconds',
            'Embedding generation latency distribution',
            ['model', 'batch_size_bucket'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.embedding_cache_hit_rate = Gauge(
            f'{self.config.namespace}_embedding_cache_hit_rate',
            'Embedding cache hit rate percentage (0-100)',
            registry=self.registry
        )
        
        self.embedding_tokens_processed_total = Counter(
            f'{self.config.namespace}_embedding_tokens_processed_total',
            'Total tokens processed for embedding generation',
            ['model'],
            registry=self.registry
        )
        
        # System Health Metrics
        self.component_health_status = Enum(
            f'{self.config.namespace}_component_health_status',
            'Health status of RAG system components',
            ['component_name'],
            states=['healthy', 'degraded', 'unhealthy', 'unknown'],
            registry=self.registry
        )
        
        self.system_memory_usage_bytes = Gauge(
            f'{self.config.namespace}_system_memory_usage_bytes',
            'Memory usage by component in bytes',
            ['component_name', 'memory_type'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            f'{self.config.namespace}_active_connections',
            'Number of active connections',
            ['connection_type', 'target_service'],
            registry=self.registry
        )
        
        # Business Intelligence Metrics
        self.user_queries_total = Counter(
            f'{self.config.namespace}_user_queries_total',
            'Total user queries processed',
            ['user_type', 'query_intent', 'library_category'],
            registry=self.registry
        )
        
        self.library_popularity = Counter(
            f'{self.config.namespace}_library_popularity_total',
            'Library access frequency',
            ['library_id', 'library_category', 'access_type'],
            registry=self.registry
        )
        
        self.user_engagement_duration_seconds = Histogram(
            f'{self.config.namespace}_user_engagement_duration_seconds',
            'User session duration distribution',
            ['user_type'],
            buckets=[1, 5, 15, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        # Error Tracking
        self.errors_total = Counter(
            f'{self.config.namespace}_errors_total',
            'Total errors by component and type',
            ['component_name', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # System Information
        self.build_info = Info(
            f'{self.config.namespace}_build_info',
            'RAG system build information',
            registry=self.registry
        )
    
    def get_metrics_handler(self):
        """Return WSGI/ASGI compatible metrics handler."""
        def metrics_app(environ, start_response):
            """WSGI application for serving metrics."""
            data = generate_latest(self.registry)
            status = '200 OK'
            response_headers = [
                ('Content-type', CONTENT_TYPE_LATEST),
                ('Content-Length', str(len(data)))
            ]
            start_response(status, response_headers)
            return [data]
        
        return metrics_app

# Global metrics instance
_metrics_collector = None

def get_metrics_collector(config: Optional[MetricsConfig] = None) -> RAGMetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        if config is None:
            config = MetricsConfig()
        _metrics_collector = RAGMetricsCollector(config)
    return _metrics_collector
```

### Pattern: Performance Monitoring Decorators
```python
def monitor_search_performance(search_type: str = "hybrid", complexity: str = "medium"):
    """Decorator for monitoring search operation performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            
            # Extract additional context from arguments if available
            library_id = kwargs.get('library_id', 'unknown')
            trace_id = kwargs.get('trace_id', None)
            
            start_time = time.time()
            status = 'success'
            result_count = 0
            
            try:
                result = await func(*args, **kwargs)
                
                # Extract result metrics
                if hasattr(result, '__len__'):
                    result_count = len(result)
                elif isinstance(result, dict) and 'results' in result:
                    result_count = len(result['results'])
                
                return result
                
            except Exception as e:
                status = 'error'
                # Record error details
                metrics.errors_total.labels(
                    component_name='search_engine',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                
                # Record search request metrics
                exemplar = {'trace_id': trace_id} if trace_id else None
                
                metrics.search_requests_total.labels(
                    search_type=search_type,
                    status=status,
                    library_id=library_id
                ).inc(exemplar=exemplar)
                
                metrics.search_latency_seconds.labels(
                    search_type=search_type,
                    complexity=complexity
                ).observe(duration, exemplar=exemplar)
                
                if status == 'success':
                    metrics.search_results_count.labels(
                        search_type=search_type
                    ).observe(result_count)
        
        return wrapper
    return decorator

def monitor_document_processing(document_type: str, processing_stage: str = "full"):
    """Decorator for monitoring document processing performance."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            
            library_id = kwargs.get('library_id', 'unknown')
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                status = 'error'
                metrics.errors_total.labels(
                    component_name='document_processor',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                
                metrics.documents_processed_total.labels(
                    status=status,
                    document_type=document_type,
                    library_id=library_id
                ).inc()
                
                if status == 'success':
                    metrics.document_processing_duration_seconds.labels(
                        document_type=document_type,
                        processing_stage=processing_stage
                    ).observe(duration)
        
        return wrapper
    return decorator

def monitor_vector_operations(operation_type: str, collection_name: str = "default"):
    """Decorator for monitoring vector database operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                status = 'error'
                metrics.errors_total.labels(
                    component_name='vector_store',
                    error_type=type(e).__name__,
                    severity='error'
                ).inc()
                raise
                
            finally:
                duration = time.time() - start_time
                
                metrics.vector_operations_total.labels(
                    operation_type=operation_type,
                    status=status,
                    collection_name=collection_name
                ).inc()
                
                if status == 'success':
                    metrics.vector_operation_duration_seconds.labels(
                        operation_type=operation_type,
                        collection_name=collection_name
                    ).observe(duration)
        
        return wrapper
    return decorator
```

### Pattern: Health Check Implementation
```python
from enum import Enum
from typing import NamedTuple, List

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthCheck(NamedTuple):
    """Individual health check result."""
    component_name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    check_duration: float

class RAGHealthMonitor:
    """Comprehensive health monitoring for RAG system components."""
    
    def __init__(self, metrics_collector: RAGMetricsCollector):
        self.metrics = metrics_collector
        self.checks = {}
        self._register_default_checks()
        
    def _register_default_checks(self):
        """Register default health checks for core components."""
        self.register_check("vector_store", self._check_vector_store)
        self.register_check("embedding_service", self._check_embedding_service)
        self.register_check("search_engine", self._check_search_engine)
        self.register_check("document_processor", self._check_document_processor)
        self.register_check("memory_usage", self._check_memory_usage)
        
    def register_check(self, name: str, check_func):
        """Register a custom health check."""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> List[HealthCheck]:
        """Run all registered health checks."""
        results = []
        
        for check_name, check_func in self.checks.items():
            start_time = time.time()
            
            try:
                status, message, details = await check_func()
                check_duration = time.time() - start_time
                
                result = HealthCheck(
                    component_name=check_name,
                    status=status,
                    message=message,
                    details=details,
                    check_duration=check_duration
                )
                
                # Update metrics
                self.metrics.component_health_status.labels(
                    component_name=check_name
                ).state(status.value)
                
            except Exception as e:
                check_duration = time.time() - start_time
                result = HealthCheck(
                    component_name=check_name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}",
                    details={'error_type': type(e).__name__},
                    check_duration=check_duration
                )
                
                self.metrics.component_health_status.labels(
                    component_name=check_name
                ).state(HealthStatus.UNKNOWN.value)
            
            results.append(result)
        
        return results
    
    async def _check_vector_store(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check vector store health and performance."""
        try:
            # Test basic connectivity
            start_time = time.time()
            # await self.vector_store.health_check()
            connection_time = time.time() - start_time
            
            # Check collection info
            # collections_info = await self.vector_store.get_collections_info()
            
            details = {
                'connection_latency_ms': round(connection_time * 1000, 2),
                # 'collections_count': len(collections_info),
                # 'total_vectors': sum(info['vectors_count'] for info in collections_info.values())
            }
            
            # Determine status based on latency and availability
            if connection_time > 1.0:  # > 1 second is concerning
                return HealthStatus.DEGRADED, "High connection latency", details
            elif connection_time > 0.1:  # > 100ms is sub-optimal
                return HealthStatus.DEGRADED, "Elevated connection latency", details
            else:
                return HealthStatus.HEALTHY, "Vector store operating normally", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Vector store connectivity failed: {str(e)}", {}
    
    async def _check_embedding_service(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check embedding service health and performance."""
        try:
            # Test embedding generation with small sample
            start_time = time.time()
            # test_result = await self.embedding_service.generate_embedding("health check test")
            generation_time = time.time() - start_time
            
            # Check cache hit rate
            current_hit_rate = self.metrics.embedding_cache_hit_rate._value.get() or 0
            
            details = {
                'generation_latency_ms': round(generation_time * 1000, 2),
                'cache_hit_rate_percent': round(current_hit_rate, 1)
            }
            
            if generation_time > 2.0:  # > 2 seconds is too slow
                return HealthStatus.DEGRADED, "Slow embedding generation", details
            elif current_hit_rate < 70:  # < 70% cache hit rate
                return HealthStatus.DEGRADED, "Low cache hit rate", details
            else:
                return HealthStatus.HEALTHY, "Embedding service operating normally", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Embedding service failed: {str(e)}", {}
    
    async def _check_search_engine(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check search engine health and performance."""
        try:
            # Perform test search
            start_time = time.time()
            # results = await self.search_engine.search("test query", top_k=1)
            search_time = time.time() - start_time
            
            details = {
                'search_latency_ms': round(search_time * 1000, 2),
                # 'test_results_count': len(results)
            }
            
            if search_time > 0.5:  # > 500ms is concerning
                return HealthStatus.DEGRADED, "High search latency", details
            # elif len(results) == 0:
            #     return HealthStatus.DEGRADED, "No search results returned", details
            else:
                return HealthStatus.HEALTHY, "Search engine operating normally", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Search engine failed: {str(e)}", {}
    
    async def _check_document_processor(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check document processing pipeline health."""
        try:
            # Check ingestion queue size
            queue_size = self.metrics.ingestion_queue_size._value.get() or 0
            
            # Check recent processing throughput
            # This would typically look at recent throughput metrics
            # current_throughput = self.metrics.ingestion_throughput_docs_per_minute._value.get() or 0
            
            details = {
                'queue_size': int(queue_size),
                # 'throughput_docs_per_min': round(current_throughput, 1)
            }
            
            if queue_size > 10000:  # Large queue backlog
                return HealthStatus.DEGRADED, "Large processing queue backlog", details
            # elif current_throughput == 0 and queue_size > 0:
            #     return HealthStatus.UNHEALTHY, "Processing pipeline stalled", details
            else:
                return HealthStatus.HEALTHY, "Document processor operating normally", details
                
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Document processor check failed: {str(e)}", {}
    
    async def _check_memory_usage(self) -> tuple[HealthStatus, str, Dict[str, Any]]:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            details = {
                'system_memory_used_percent': round(memory.percent, 1),
                'system_memory_available_mb': round(memory.available / 1024 / 1024, 1),
                'process_memory_rss_mb': round(process_memory.rss / 1024 / 1024, 1),
                'process_memory_vms_mb': round(process_memory.vms / 1024 / 1024, 1)
            }
            
            # Update memory metrics
            self.metrics.system_memory_usage_bytes.labels(
                component_name='system',
                memory_type='used'
            ).set(memory.used)
            
            self.metrics.system_memory_usage_bytes.labels(
                component_name='process',
                memory_type='rss'
            ).set(process_memory.rss)
            
            if memory.percent > 95:  # > 95% memory usage is critical
                return HealthStatus.UNHEALTHY, "Critical memory usage", details
            elif memory.percent > 85:  # > 85% memory usage is concerning
                return HealthStatus.DEGRADED, "High memory usage", details
            else:
                return HealthStatus.HEALTHY, "Memory usage normal", details
                
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Memory check failed: {str(e)}", {}
```

## Common Gotchas

### Gotcha: Metrics Cardinality Explosion
- **Problem**: Too many label combinations create excessive memory usage and slow queries
- **Solution**: Limit label cardinality and use label dropping for high-cardinality data
- **Example**:
```python
# BAD: Unlimited cardinality
user_requests = Counter('user_requests_total', 'Requests by user', ['user_id'])  # Could be millions of users

# GOOD: Limited cardinality with bucketing
user_type_requests = Counter(
    'user_requests_total', 
    'Requests by user type', 
    ['user_type']  # Limited to: anonymous, registered, premium
)

# Helper function to bucket high-cardinality labels
def get_user_type_bucket(user_id: str) -> str:
    """Convert user ID to bounded user type bucket."""
    if user_id.startswith('anon_'):
        return 'anonymous'
    elif user_id.startswith('prem_'):
        return 'premium'
    else:
        return 'registered'
```

### Gotcha: Blocking Metrics Collection
- **Problem**: Synchronous metrics collection blocks async request processing
- **Solution**: Use async-compatible metrics collection and batching
- **Example**:
```python
import asyncio
from collections import defaultdict
from typing import Dict, List

class AsyncMetricsBatcher:
    """Batches metrics updates to avoid blocking async operations."""
    
    def __init__(self, flush_interval: float = 1.0, max_batch_size: int = 1000):
        self.flush_interval = flush_interval
        self.max_batch_size = max_batch_size
        self.batch_data = defaultdict(list)
        self.batch_lock = asyncio.Lock()
        self.flush_task = None
        
    async def start(self):
        """Start background flushing task."""
        self.flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """Stop and flush remaining data."""
        if self.flush_task:
            self.flush_task.cancel()
            await self._flush_batch()
    
    async def record_metric(self, metric_name: str, value: float, labels: Dict[str, str]):
        """Record metric data asynchronously."""
        async with self.batch_lock:
            self.batch_data[metric_name].append({
                'value': value,
                'labels': labels,
                'timestamp': time.time()
            })
            
            # Force flush if batch is full
            if len(self.batch_data[metric_name]) >= self.max_batch_size:
                await self._flush_batch()
    
    async def _flush_loop(self):
        """Background task to flush metrics periodically."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Metrics flush error: {e}")
    
    async def _flush_batch(self):
        """Flush current batch to Prometheus metrics."""
        async with self.batch_lock:
            if not self.batch_data:
                return
                
            # Process all batched metrics
            for metric_name, data_points in self.batch_data.items():
                for point in data_points:
                    # Apply to actual Prometheus metric
                    # This would need to be connected to your actual metrics
                    pass
            
            self.batch_data.clear()

# Usage in async context
async def process_request_with_batched_metrics(request):
    """Process request with non-blocking metrics collection."""
    start_time = time.time()
    
    try:
        # Process request
        result = await handle_request(request)
        
        # Record metrics asynchronously (non-blocking)
        await metrics_batcher.record_metric(
            'request_duration_seconds',
            time.time() - start_time,
            {'status': 'success', 'endpoint': request.path}
        )
        
        return result
        
    except Exception as e:
        await metrics_batcher.record_metric(
            'request_duration_seconds',
            time.time() - start_time,
            {'status': 'error', 'endpoint': request.path}
        )
        raise
```

### Gotcha: Inconsistent Metric Labels
- **Problem**: Inconsistent label naming and values across components cause fragmented metrics
- **Solution**: Implement centralized label standardization
- **Example**:
```python
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class StandardLabels:
    """Standardized labels for consistent metrics."""
    
    # Component identification
    component_name: str
    component_version: str = "unknown"
    
    # Request context
    endpoint: str = ""
    method: str = ""
    status: str = ""
    
    # Business context
    library_id: str = "unknown"
    user_type: str = "unknown"
    
    # Error context
    error_type: str = ""
    severity: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary, excluding empty values."""
        return {
            k: str(v) for k, v in self.__dict__.items() 
            if v and v != "unknown" and v != ""
        }
    
    @classmethod
    def for_search(cls, search_type: str, library_id: str = "", status: str = "success") -> 'StandardLabels':
        """Create standard labels for search operations."""
        return cls(
            component_name="search_engine",
            endpoint=f"/search/{search_type}",
            method="POST",
            status=status,
            library_id=library_id or "unknown"
        )
    
    @classmethod
    def for_ingestion(cls, document_type: str, library_id: str = "", status: str = "success") -> 'StandardLabels':
        """Create standard labels for document ingestion."""
        return cls(
            component_name="document_processor",
            endpoint="/documents/ingest",
            method="POST", 
            status=status,
            library_id=library_id or "unknown"
        )
    
    @classmethod
    def for_error(cls, component: str, error_type: str, severity: str = "error") -> 'StandardLabels':
        """Create standard labels for error tracking."""
        return cls(
            component_name=component,
            error_type=error_type,
            severity=severity,
            status="error"
        )

# Usage with standardized labels
def record_search_metrics(search_type: str, library_id: str, duration: float, success: bool):
    """Record search metrics with standardized labels."""
    status = "success" if success else "error"
    labels = StandardLabels.for_search(search_type, library_id, status)
    
    metrics.search_requests_total.labels(**labels.to_dict()).inc()
    
    if success:
        metrics.search_latency_seconds.labels(
            **{k: v for k, v in labels.to_dict().items() if k != 'status'}
        ).observe(duration)
```

## Best Practices

### Metrics Naming Convention
```python
# Follow Prometheus naming conventions
METRIC_NAMING_STANDARDS = {
    'counters': '{namespace}_{component}_{action}_total',
    'histograms': '{namespace}_{component}_{measurement}_{unit}',
    'gauges': '{namespace}_{component}_{state}_{unit}',
    'enums': '{namespace}_{component}_{property}_status'
}

# Examples:
# rag_search_requests_total (counter)
# rag_search_latency_seconds (histogram)  
# rag_vector_storage_size_bytes (gauge)
# rag_component_health_status (enum)
```

### Dashboard Configuration Templates
```python
# Grafana dashboard configuration
GRAFANA_DASHBOARD_CONFIG = {
    "dashboard": {
        "title": "RAG System Performance Dashboard",
        "tags": ["rag", "search", "performance"],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "30s",
        "panels": [
            {
                "title": "Search Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(rag_search_requests_total[5m])",
                        "legendFormat": "{{search_type}} requests/sec"
                    }
                ],
                "yAxes": [{"unit": "reqps"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "title": "Search Latency Percentiles", 
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(rag_search_latency_seconds_bucket[5m]))",
                        "legendFormat": "95th percentile"
                    },
                    {
                        "expr": "histogram_quantile(0.99, rate(rag_search_latency_seconds_bucket[5m]))",
                        "legendFormat": "99th percentile"
                    }
                ],
                "yAxes": [{"unit": "s", "max": 1}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "title": "Error Rate",
                "type": "singlestat",
                "targets": [
                    {
                        "expr": "rate(rag_search_requests_total{status=\"error\"}[5m]) / rate(rag_search_requests_total[5m]) * 100"
                    }
                ],
                "format": "percent",
                "thresholds": [1, 5],
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
            }
        ]
    }
}
```

### Alert Rules Configuration
```yaml
# prometheus_alerts.yml
groups:
  - name: rag_system_alerts
    rules:
      - alert: RAGHighSearchLatency
        expr: histogram_quantile(0.95, rate(rag_search_latency_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
          component: search_engine
        annotations:
          summary: "High search latency detected"
          description: "95th percentile search latency is {{ $value }}s, above threshold of 0.5s"
      
      - alert: RAGHighErrorRate
        expr: rate(rag_search_requests_total{status="error"}[5m]) / rate(rag_search_requests_total[5m]) > 0.05
        for: 1m
        labels:
          severity: critical
          component: search_engine
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}, above threshold of 5%"
      
      - alert: RAGComponentUnhealthy
        expr: rag_component_health_status != 0  # 0 = healthy
        for: 30s
        labels:
          severity: critical
          component: "{{ $labels.component_name }}"
        annotations:
          summary: "RAG component unhealthy"
          description: "Component {{ $labels.component_name }} is in unhealthy state"
      
      - alert: RAGIngestionQueueBacklog
        expr: rag_ingestion_queue_size > 10000
        for: 5m
        labels:
          severity: warning
          component: document_processor
        annotations:
          summary: "Large ingestion queue backlog"
          description: "Document ingestion queue has {{ $value }} items, indicating processing delays"
      
      - alert: RAGLowCacheHitRate
        expr: rag_embedding_cache_hit_rate < 70
        for: 10m
        labels:
          severity: warning
          component: embedding_service
        annotations:
          summary: "Low embedding cache hit rate"
          description: "Cache hit rate is {{ $value }}%, below optimal threshold of 70%"
```

## Integration Points

### FastAPI Integration
- **Metrics Endpoint**: Expose `/metrics` endpoint for Prometheus scraping
- **Middleware**: Automatic request/response metrics collection
- **Health Checks**: `/health` endpoint with component status

### Vector Database Integration
- **Qdrant Metrics**: Collection size, operation latencies, index performance
- **Connection Monitoring**: Pool health, timeout detection
- **Storage Metrics**: Disk usage, memory consumption

### Embedding Service Integration
- **Voyage AI Metrics**: API latencies, token usage, rate limiting
- **Cache Performance**: Hit rates, eviction rates, cache size
- **Batch Processing**: Throughput, queue sizes, error rates

## References

- [Prometheus Python Client Documentation](https://prometheus.github.io/client_python/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Grafana Dashboard Best Practices](https://grafana.com/docs/grafana/latest/dashboards/manage-dashboards/)
- [Alert Manager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)

## Related Contexts

- `technical-specs/testing-patterns.md` - Monitoring integration in tests
- `technical-specs/qdrant-vector-database.md` - Vector storage metrics
- `technical-specs/voyage-ai-embedding.md` - Embedding service metrics
- `technical-specs/fastapi-integration.md` - API endpoint monitoring
- `integration-guides/monitoring-integration.md` - Step-by-step setup