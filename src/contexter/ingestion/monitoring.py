"""
Ingestion Pipeline Monitoring - Performance tracking and optimization.

Comprehensive monitoring system for the ingestion pipeline with
performance tracking, alerting, and optimization recommendations.
"""

import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    
    # Processing metrics
    jobs_per_minute: float
    avg_processing_time: float
    queue_utilization: float
    worker_utilization: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_usage_percent: float
    
    # Quality metrics
    avg_quality_score: float
    success_rate: float
    
    # Component metrics
    embedding_latency: float
    vector_storage_latency: float
    parsing_success_rate: float
    chunking_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'processing': {
                'jobs_per_minute': self.jobs_per_minute,
                'avg_processing_time': self.avg_processing_time,
                'queue_utilization': self.queue_utilization,
                'worker_utilization': self.worker_utilization
            },
            'resources': {
                'cpu_usage_percent': self.cpu_usage_percent,
                'memory_usage_mb': self.memory_usage_mb,
                'memory_usage_percent': self.memory_usage_percent
            },
            'quality': {
                'avg_quality_score': self.avg_quality_score,
                'success_rate': self.success_rate
            },
            'components': {
                'embedding_latency': self.embedding_latency,
                'vector_storage_latency': self.vector_storage_latency,
                'parsing_success_rate': self.parsing_success_rate,
                'chunking_efficiency': self.chunking_efficiency
            }
        }


@dataclass
class Alert:
    """Performance alert."""
    id: str
    severity: str  # 'info', 'warning', 'critical'
    title: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'id': self.id,
            'severity': self.severity,
            'title': self.title,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'created_at': self.created_at.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class PerformanceMonitor:
    """
    Performance monitoring and optimization system.
    
    Tracks pipeline performance, generates alerts, and provides
    optimization recommendations based on real-time metrics.
    """
    
    def __init__(
        self,
        pipeline: 'IngestionPipeline',
        monitoring_interval: float = 30.0,
        metric_history_size: int = 1000
    ):
        """
        Initialize performance monitor.
        
        Args:
            pipeline: Ingestion pipeline to monitor
            monitoring_interval: Monitoring frequency in seconds
            metric_history_size: Number of metrics to keep in history
        """
        self.pipeline = pipeline
        self.monitoring_interval = monitoring_interval
        self.metric_history_size = metric_history_size
        
        # Metric storage
        self.metrics_history: deque = deque(maxlen=metric_history_size)
        self.alerts: List[Alert] = []
        self.max_alerts = 100
        
        # Monitoring state
        self.monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance thresholds
        self.thresholds = {
            'jobs_per_minute_min': 10.0,
            'avg_processing_time_max': 60.0,
            'queue_utilization_max': 0.8,
            'worker_utilization_min': 0.3,
            'cpu_usage_max': 80.0,
            'memory_usage_max': 2048.0,  # MB
            'memory_usage_percent_max': 80.0,
            'success_rate_min': 0.95,
            'avg_quality_score_min': 0.7,
            'embedding_latency_max': 10.0,
            'vector_storage_latency_max': 5.0
        }
        
        # Optimization recommendations
        self.optimization_rules = [
            self._check_worker_scaling,
            self._check_queue_capacity,
            self._check_memory_usage,
            self._check_chunking_settings,
            self._check_batch_sizes
        ]
        
        logger.info("Performance monitor initialized")
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Performance monitoring already running")
            return
        
        self._shutdown_event.clear()
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        logger.info("Stopping performance monitoring")
        
        self._shutdown_event.set()
        
        if self.monitoring_task:
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Monitoring task shutdown timeout, cancelling")
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting performance monitoring loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(metrics)
                
                # Log key metrics periodically
                if len(self.metrics_history) % 10 == 0:  # Every 10 intervals
                    logger.info(
                        f"Performance: {metrics.jobs_per_minute:.1f} jobs/min, "
                        f"{metrics.avg_processing_time:.1f}s avg, "
                        f"{metrics.success_rate:.1%} success rate"
                    )
                
                # Wait for next monitoring cycle
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.monitoring_interval
                    )
                except asyncio.TimeoutError:
                    pass  # Expected timeout, continue monitoring
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retrying
        
        logger.info("Performance monitoring loop stopped")
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        try:
            # Get pipeline statistics
            pipeline_stats = self.pipeline.get_statistics()
            
            # Get component statistics
            queue_stats = self.pipeline.job_queue.get_statistics()
            worker_stats = self.pipeline.worker_pool.get_statistics()
            
            # Get embedding engine metrics
            embedding_health = await self.pipeline.embedding_engine.health_check()
            embedding_perf = embedding_health.get('performance', {})
            
            # Get vector storage metrics
            vector_health = self.pipeline.vector_storage.get_health_status()
            vector_metrics = vector_health.get('metrics', {})
            
            # Get system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate derived metrics
            jobs_per_minute = queue_stats.get('total_processing_time', 0)
            if jobs_per_minute > 0:
                jobs_per_minute = 60.0 / (jobs_per_minute / max(1, queue_stats.get('jobs_completed', 1)))
            
            # Create metrics snapshot
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                jobs_per_minute=jobs_per_minute,
                avg_processing_time=worker_stats.get('avg_processing_time', 0.0),
                queue_utilization=queue_stats.get('utilization', 0.0),
                worker_utilization=worker_stats.get('worker_utilization', 0.0),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=process_memory,
                memory_usage_percent=memory_info.percent,
                avg_quality_score=pipeline_stats.get('quality', {}).get('avg_quality_score', 0.0),
                success_rate=queue_stats.get('success_rate', 0.0),
                embedding_latency=embedding_perf.get('average_latency_ms', 0.0) / 1000.0,
                vector_storage_latency=vector_metrics.get('search_latency_p95', 0.0),
                parsing_success_rate=1.0,  # Would need to track from parser
                chunking_efficiency=1.0    # Would need to track chunk quality
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            # Return empty metrics on error
            return PerformanceMetrics(
                timestamp=datetime.now(),
                jobs_per_minute=0.0,
                avg_processing_time=0.0,
                queue_utilization=0.0,
                worker_utilization=0.0,
                cpu_usage_percent=0.0,
                memory_usage_mb=0.0,
                memory_usage_percent=0.0,
                avg_quality_score=0.0,
                success_rate=0.0,
                embedding_latency=0.0,
                vector_storage_latency=0.0,
                parsing_success_rate=0.0,
                chunking_efficiency=0.0
            )
    
    async def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and generate alerts."""
        
        # Jobs per minute too low
        if metrics.jobs_per_minute < self.thresholds['jobs_per_minute_min']:
            await self._create_alert(
                'warning',
                'Low Processing Throughput',
                f'Processing rate ({metrics.jobs_per_minute:.1f} jobs/min) is below minimum threshold',
                'jobs_per_minute',
                metrics.jobs_per_minute,
                self.thresholds['jobs_per_minute_min']
            )
        
        # Processing time too high
        if metrics.avg_processing_time > self.thresholds['avg_processing_time_max']:
            await self._create_alert(
                'warning',
                'High Processing Time',
                f'Average processing time ({metrics.avg_processing_time:.1f}s) exceeds threshold',
                'avg_processing_time',
                metrics.avg_processing_time,
                self.thresholds['avg_processing_time_max']
            )
        
        # Queue utilization too high
        if metrics.queue_utilization > self.thresholds['queue_utilization_max']:
            await self._create_alert(
                'critical',
                'High Queue Utilization',
                f'Queue utilization ({metrics.queue_utilization:.1%}) is critically high',
                'queue_utilization',
                metrics.queue_utilization,
                self.thresholds['queue_utilization_max']
            )
        
        # Worker utilization too low
        if metrics.worker_utilization < self.thresholds['worker_utilization_min']:
            await self._create_alert(
                'info',
                'Low Worker Utilization',
                f'Worker utilization ({metrics.worker_utilization:.1%}) suggests underutilization',
                'worker_utilization',
                metrics.worker_utilization,
                self.thresholds['worker_utilization_min']
            )
        
        # CPU usage too high
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage_max']:
            await self._create_alert(
                'warning',
                'High CPU Usage',
                f'CPU usage ({metrics.cpu_usage_percent:.1f}%) is high',
                'cpu_usage_percent',
                metrics.cpu_usage_percent,
                self.thresholds['cpu_usage_max']
            )
        
        # Memory usage too high
        if metrics.memory_usage_mb > self.thresholds['memory_usage_max']:
            await self._create_alert(
                'critical',
                'High Memory Usage',
                f'Memory usage ({metrics.memory_usage_mb:.1f}MB) exceeds limit',
                'memory_usage_mb',
                metrics.memory_usage_mb,
                self.thresholds['memory_usage_max']
            )
        
        # Success rate too low
        if metrics.success_rate < self.thresholds['success_rate_min']:
            await self._create_alert(
                'critical',
                'Low Success Rate',
                f'Success rate ({metrics.success_rate:.1%}) is below acceptable threshold',
                'success_rate',
                metrics.success_rate,
                self.thresholds['success_rate_min']
            )
        
        # Quality score too low
        if metrics.avg_quality_score < self.thresholds['avg_quality_score_min']:
            await self._create_alert(
                'warning',
                'Low Quality Score',
                f'Average quality score ({metrics.avg_quality_score:.2f}) is below threshold',
                'avg_quality_score',
                metrics.avg_quality_score,
                self.thresholds['avg_quality_score_min']
            )
    
    async def _create_alert(
        self,
        severity: str,
        title: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ):
        """Create and store an alert."""
        
        # Check if similar alert already exists (avoid spam)
        recent_alerts = [
            alert for alert in self.alerts[-10:]
            if alert.metric_name == metric_name and not alert.resolved
        ]
        
        if recent_alerts:
            # Update existing alert instead of creating new one
            recent_alerts[-1].metric_value = metric_value
            recent_alerts[-1].created_at = datetime.now()
            return
        
        alert = Alert(
            id=f"{metric_name}_{int(time.time())}",
            severity=severity,
            title=title,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            created_at=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Log alert
        log_level = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.error
        }.get(severity, logger.info)
        
        log_level(f"Performance Alert ({severity}): {title} - {message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No metrics collected yet'}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calculate trends (last 10 measurements)
        recent_metrics = list(self.metrics_history)[-10:]
        
        trends = {}
        if len(recent_metrics) >= 2:
            first = recent_metrics[0]
            last = recent_metrics[-1]
            
            trends = {
                'jobs_per_minute': self._calculate_trend(
                    [m.jobs_per_minute for m in recent_metrics]
                ),
                'avg_processing_time': self._calculate_trend(
                    [m.avg_processing_time for m in recent_metrics]
                ),
                'success_rate': self._calculate_trend(
                    [m.success_rate for m in recent_metrics]
                ),
                'memory_usage_mb': self._calculate_trend(
                    [m.memory_usage_mb for m in recent_metrics]
                )
            }
        
        # Get active alerts
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        
        # Generate optimization recommendations
        recommendations = await self._generate_recommendations()
        
        return {
            'status': 'healthy' if not active_alerts else 'degraded',
            'latest_metrics': latest_metrics.to_dict(),
            'trends': trends,
            'active_alerts': [alert.to_dict() for alert in active_alerts],
            'alert_summary': {
                'total': len(active_alerts),
                'critical': len([a for a in active_alerts if a.severity == 'critical']),
                'warning': len([a for a in active_alerts if a.severity == 'warning']),
                'info': len([a for a in active_alerts if a.severity == 'info'])
            },
            'recommendations': recommendations,
            'monitoring_info': {
                'monitoring_interval': self.monitoring_interval,
                'metrics_collected': len(self.metrics_history),
                'thresholds': self.thresholds
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Calculate linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    async def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        # Run optimization rules
        for rule in self.optimization_rules:
            try:
                rule_recommendations = await rule()
                recommendations.extend(rule_recommendations)
            except Exception as e:
                logger.error(f"Error in optimization rule: {e}")
        
        return recommendations
    
    async def _check_worker_scaling(self) -> List[Dict[str, Any]]:
        """Check if worker scaling is needed."""
        recommendations = []
        
        if len(self.metrics_history) < 5:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-5:]
        avg_utilization = sum(m.worker_utilization for m in recent_metrics) / len(recent_metrics)
        avg_queue_utilization = sum(m.queue_utilization for m in recent_metrics) / len(recent_metrics)
        
        if avg_utilization > 0.8 and avg_queue_utilization > 0.5:
            recommendations.append({
                'type': 'scaling',
                'priority': 'high',
                'title': 'Scale Up Workers',
                'description': 'High worker and queue utilization suggests need for more workers',
                'action': f'Consider increasing workers from {self.pipeline.max_workers} to {self.pipeline.max_workers + 2}',
                'metrics': {
                    'worker_utilization': avg_utilization,
                    'queue_utilization': avg_queue_utilization
                }
            })
        elif avg_utilization < 0.3 and avg_queue_utilization < 0.1:
            recommendations.append({
                'type': 'scaling',
                'priority': 'low',
                'title': 'Scale Down Workers',
                'description': 'Low utilization suggests over-provisioning',
                'action': f'Consider reducing workers from {self.pipeline.max_workers} to {max(1, self.pipeline.max_workers - 1)}',
                'metrics': {
                    'worker_utilization': avg_utilization,
                    'queue_utilization': avg_queue_utilization
                }
            })
        
        return recommendations
    
    async def _check_queue_capacity(self) -> List[Dict[str, Any]]:
        """Check queue capacity optimization."""
        recommendations = []
        
        if len(self.metrics_history) < 3:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-3:]
        max_queue_utilization = max(m.queue_utilization for m in recent_metrics)
        
        if max_queue_utilization > 0.9:
            recommendations.append({
                'type': 'capacity',
                'priority': 'medium',
                'title': 'Increase Queue Capacity',
                'description': 'Queue utilization is near capacity limit',
                'action': 'Consider increasing queue size or improving processing speed',
                'metrics': {
                    'max_queue_utilization': max_queue_utilization
                }
            })
        
        return recommendations
    
    async def _check_memory_usage(self) -> List[Dict[str, Any]]:
        """Check memory usage optimization."""
        recommendations = []
        
        if len(self.metrics_history) < 3:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-3:]
        avg_memory_mb = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        if avg_memory_mb > 1500:  # 1.5GB
            recommendations.append({
                'type': 'memory',
                'priority': 'medium',
                'title': 'Optimize Memory Usage',
                'description': 'Memory usage is approaching limits',
                'action': 'Consider reducing chunk size or batch sizes to lower memory usage',
                'metrics': {
                    'avg_memory_mb': avg_memory_mb
                }
            })
        
        return recommendations
    
    async def _check_chunking_settings(self) -> List[Dict[str, Any]]:
        """Check chunking optimization."""
        recommendations = []
        
        # This would analyze chunk quality and processing efficiency
        # For now, provide basic recommendations
        
        chunk_size = self.pipeline.chunking_engine.chunk_size
        chunk_overlap = self.pipeline.chunking_engine.chunk_overlap
        
        if chunk_overlap / chunk_size > 0.3:
            recommendations.append({
                'type': 'chunking',
                'priority': 'low',
                'title': 'Optimize Chunk Overlap',
                'description': 'Chunk overlap ratio is high, may impact processing efficiency',
                'action': f'Consider reducing overlap from {chunk_overlap} to {int(chunk_size * 0.2)} tokens',
                'metrics': {
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'overlap_ratio': chunk_overlap / chunk_size
                }
            })
        
        return recommendations
    
    async def _check_batch_sizes(self) -> List[Dict[str, Any]]:
        """Check batch size optimization."""
        recommendations = []
        
        # This would analyze embedding generation and vector storage batch efficiency
        # For now, provide basic recommendations based on performance
        
        if len(self.metrics_history) >= 5:
            recent_metrics = list(self.metrics_history)[-5:]
            avg_embedding_latency = sum(m.embedding_latency for m in recent_metrics) / len(recent_metrics)
            
            if avg_embedding_latency > 5.0:
                recommendations.append({
                    'type': 'batching',
                    'priority': 'medium',
                    'title': 'Optimize Embedding Batch Size',
                    'description': 'High embedding latency suggests batch size optimization needed',
                    'action': 'Consider adjusting embedding batch size for better throughput',
                    'metrics': {
                        'avg_embedding_latency': avg_embedding_latency
                    }
                })
        
        return recommendations
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False