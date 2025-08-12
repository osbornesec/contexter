"""
Vector Health Monitor - Comprehensive health checking and monitoring.

Provides health monitoring capabilities for vector database systems:
- Connection health and performance monitoring
- Collection health and integrity verification
- Performance metrics collection and analysis
- Alerting and notification capabilities
- Health history tracking and reporting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .qdrant_vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: datetime
    duration_ms: float


class HealthConfig(BaseModel):
    """Configuration for health monitoring."""

    check_interval_seconds: int = Field(default=30, description="Health check interval")
    connection_timeout_seconds: float = Field(default=5.0, description="Connection timeout")
    search_latency_threshold_ms: float = Field(default=50.0, description="Search latency threshold")
    collection_size_threshold: int = Field(default=1000000, description="Collection size warning threshold")
    error_rate_threshold: float = Field(default=0.05, description="Error rate threshold (5%)")
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")
    enable_alerting: bool = Field(default=True, description="Enable health alerting")
    history_retention_hours: int = Field(default=24, description="Health history retention")


class HealthReport(BaseModel):
    """Comprehensive health report."""

    overall_status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheck]
    summary: Dict[str, Any]
    recommendations: List[str]


class VectorHealthMonitor:
    """
    Comprehensive health monitor for vector database systems.
    
    Features:
    - Connection health monitoring with timeout handling
    - Collection health and integrity verification
    - Performance metrics monitoring and alerting
    - Search latency and throughput monitoring
    - Health history tracking and trend analysis
    - Configurable alerting and notifications
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        config: Optional[HealthConfig] = None
    ):
        """
        Initialize the health monitor.
        
        Args:
            vector_store: Qdrant vector store instance
            config: Health monitoring configuration
        """
        self.vector_store = vector_store
        self.config = config or HealthConfig()

        # Health tracking
        self._health_history: List[HealthReport] = []
        self._last_health_check = None
        self._monitoring_task = None
        self._is_monitoring = False

        # Performance tracking
        self._performance_metrics = {
            "search_latencies": [],
            "connection_failures": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "last_error": None,
            "uptime_start": datetime.now()
        }

        # Alert callbacks
        self._alert_callbacks: List[Callable[[HealthReport], None]] = []

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring is already running")
            return

        logger.info("Starting vector database health monitoring")
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self._is_monitoring:
            return

        logger.info("Stopping vector database health monitoring")
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                # Perform health check
                report = await self.perform_health_check()

                # Store in history
                self._add_to_history(report)

                # Trigger alerts if needed
                if self.config.enable_alerting:
                    await self._check_and_send_alerts(report)

                # Wait for next check
                await asyncio.sleep(self.config.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.check_interval_seconds)

    async def perform_health_check(self) -> HealthReport:
        """
        Perform comprehensive health check.
        
        Returns:
            HealthReport with all check results
        """
        logger.debug("Performing comprehensive health check")
        start_time = time.time()

        checks = []

        # Connection health check
        checks.append(await self._check_connection_health())

        # Collection health check
        checks.append(await self._check_collection_health())

        # Performance health check
        if self.config.enable_performance_monitoring:
            checks.append(await self._check_performance_health())

        # Search functionality check
        checks.append(await self._check_search_functionality())

        # Resource utilization check
        checks.append(await self._check_resource_utilization())

        # Determine overall status
        overall_status = self._determine_overall_status(checks)

        # Generate summary and recommendations
        summary = self._generate_summary(checks)
        recommendations = self._generate_recommendations(checks)

        total_time = time.time() - start_time

        report = HealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(),
            checks=checks,
            summary={
                **summary,
                "total_check_time_ms": total_time * 1000,
                "checks_performed": len(checks)
            },
            recommendations=recommendations
        )

        self._last_health_check = report
        logger.debug(f"Health check completed in {total_time:.3f}s - Status: {overall_status.value}")

        return report

    async def _check_connection_health(self) -> HealthCheck:
        """Check connection health and responsiveness."""
        start_time = time.time()

        try:
            # Test basic connection
            collection_stats = await asyncio.wait_for(
                self.vector_store.get_collection_stats(),
                timeout=self.config.connection_timeout_seconds
            )

            duration_ms = (time.time() - start_time) * 1000

            if collection_stats:
                self._performance_metrics["successful_operations"] += 1

                return HealthCheck(
                    name="connection_health",
                    status=HealthStatus.HEALTHY,
                    message="Connection is responsive and healthy",
                    metrics={
                        "response_time_ms": duration_ms,
                        "connected": True,
                        "collection_accessible": True
                    },
                    timestamp=datetime.now(),
                    duration_ms=duration_ms
                )
            else:
                self._performance_metrics["failed_operations"] += 1

                return HealthCheck(
                    name="connection_health",
                    status=HealthStatus.WARNING,
                    message="Connection established but collection stats unavailable",
                    metrics={
                        "response_time_ms": duration_ms,
                        "connected": True,
                        "collection_accessible": False
                    },
                    timestamp=datetime.now(),
                    duration_ms=duration_ms
                )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            self._performance_metrics["connection_failures"] += 1
            self._performance_metrics["failed_operations"] += 1

            return HealthCheck(
                name="connection_health",
                status=HealthStatus.CRITICAL,
                message=f"Connection timeout after {self.config.connection_timeout_seconds}s",
                metrics={
                    "response_time_ms": duration_ms,
                    "connected": False,
                    "timeout": True
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._performance_metrics["connection_failures"] += 1
            self._performance_metrics["failed_operations"] += 1
            self._performance_metrics["last_error"] = str(e)

            return HealthCheck(
                name="connection_health",
                status=HealthStatus.CRITICAL,
                message=f"Connection failed: {str(e)}",
                metrics={
                    "response_time_ms": duration_ms,
                    "connected": False,
                    "error": str(e)
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

    async def _check_collection_health(self) -> HealthCheck:
        """Check collection health and integrity."""
        start_time = time.time()

        try:
            collection_stats = await self.vector_store.get_collection_stats()
            duration_ms = (time.time() - start_time) * 1000

            if not collection_stats:
                return HealthCheck(
                    name="collection_health",
                    status=HealthStatus.CRITICAL,
                    message="Collection statistics unavailable",
                    metrics={},
                    timestamp=datetime.now(),
                    duration_ms=duration_ms
                )

            points_count = collection_stats.get("points_count", 0)
            vectors_count = collection_stats.get("vectors_count", 0)
            indexed_count = collection_stats.get("indexed_vectors_count", 0)
            collection_status = collection_stats.get("status", "unknown")

            # Determine status based on metrics
            if collection_status == "red":
                status = HealthStatus.CRITICAL
                message = "Collection is in critical state"
            elif collection_status == "yellow":
                status = HealthStatus.WARNING
                message = "Collection has warnings"
            elif points_count > self.config.collection_size_threshold:
                status = HealthStatus.WARNING
                message = f"Collection is large ({points_count:,} points) - consider optimization"
            elif vectors_count != indexed_count and indexed_count > 0:
                indexing_ratio = indexed_count / vectors_count if vectors_count > 0 else 0
                if indexing_ratio < 0.9:
                    status = HealthStatus.WARNING
                    message = f"Indexing incomplete ({indexing_ratio:.1%} indexed)"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Collection is healthy and well-indexed"
            else:
                status = HealthStatus.HEALTHY
                message = "Collection is healthy"

            return HealthCheck(
                name="collection_health",
                status=status,
                message=message,
                metrics={
                    "points_count": points_count,
                    "vectors_count": vectors_count,
                    "indexed_vectors_count": indexed_count,
                    "collection_status": collection_status,
                    "indexing_ratio": indexed_count / vectors_count if vectors_count > 0 else 0
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheck(
                name="collection_health",
                status=HealthStatus.CRITICAL,
                message=f"Collection health check failed: {str(e)}",
                metrics={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

    async def _check_performance_health(self) -> HealthCheck:
        """Check performance metrics and trends."""
        start_time = time.time()

        try:
            # Calculate error rate
            total_ops = (
                self._performance_metrics["successful_operations"] +
                self._performance_metrics["failed_operations"]
            )
            error_rate = (
                self._performance_metrics["failed_operations"] / total_ops
                if total_ops > 0 else 0
            )

            # Calculate average search latency
            latencies = self._performance_metrics["search_latencies"]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            # Calculate uptime
            uptime_duration = datetime.now() - self._performance_metrics["uptime_start"]
            uptime_hours = uptime_duration.total_seconds() / 3600

            duration_ms = (time.time() - start_time) * 1000

            # Determine status
            if error_rate > self.config.error_rate_threshold:
                status = HealthStatus.CRITICAL
                message = f"High error rate: {error_rate:.1%}"
            elif avg_latency > self.config.search_latency_threshold_ms:
                status = HealthStatus.WARNING
                message = f"High search latency: {avg_latency:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance metrics are healthy"

            return HealthCheck(
                name="performance_health",
                status=status,
                message=message,
                metrics={
                    "error_rate": error_rate,
                    "avg_search_latency_ms": avg_latency,
                    "successful_operations": self._performance_metrics["successful_operations"],
                    "failed_operations": self._performance_metrics["failed_operations"],
                    "connection_failures": self._performance_metrics["connection_failures"],
                    "uptime_hours": uptime_hours,
                    "last_error": self._performance_metrics["last_error"]
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheck(
                name="performance_health",
                status=HealthStatus.UNKNOWN,
                message=f"Performance check failed: {str(e)}",
                metrics={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

    async def _check_search_functionality(self) -> HealthCheck:
        """Check search functionality with a test query."""
        start_time = time.time()

        try:
            # Create a test vector (zero vector)
            test_vector = [0.0] * self.vector_store.config.vector_size

            # Perform test search
            search_start = time.time()
            results = await self.vector_store.search_vectors(
                query_vector=test_vector,
                top_k=5,
                score_threshold=0.0
            )
            search_duration = (time.time() - search_start) * 1000

            # Record search latency
            self._performance_metrics["search_latencies"].append(search_duration)

            # Keep only recent latencies (last 100)
            if len(self._performance_metrics["search_latencies"]) > 100:
                self._performance_metrics["search_latencies"] = self._performance_metrics["search_latencies"][-100:]

            duration_ms = (time.time() - start_time) * 1000

            # Determine status based on latency
            if search_duration > self.config.search_latency_threshold_ms * 2:
                status = HealthStatus.WARNING
                message = f"Search latency is high: {search_duration:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = "Search functionality is working properly"

            self._performance_metrics["successful_operations"] += 1

            return HealthCheck(
                name="search_functionality",
                status=status,
                message=message,
                metrics={
                    "search_latency_ms": search_duration,
                    "results_count": len(results),
                    "search_successful": True
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._performance_metrics["failed_operations"] += 1
            self._performance_metrics["last_error"] = str(e)

            return HealthCheck(
                name="search_functionality",
                status=HealthStatus.CRITICAL,
                message=f"Search functionality failed: {str(e)}",
                metrics={
                    "search_successful": False,
                    "error": str(e)
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

    async def _check_resource_utilization(self) -> HealthCheck:
        """Check resource utilization and collection statistics."""
        start_time = time.time()

        try:
            vector_store_health = self.vector_store.get_health_status()
            duration_ms = (time.time() - start_time) * 1000

            if vector_store_health.get("healthy", False):
                status = HealthStatus.HEALTHY
                message = "Resource utilization is normal"
            else:
                status = HealthStatus.WARNING
                message = "Vector store reports health issues"

            return HealthCheck(
                name="resource_utilization",
                status=status,
                message=message,
                metrics={
                    "vector_store_healthy": vector_store_health.get("healthy", False),
                    "vector_store_metrics": vector_store_health.get("metrics", {}),
                    "last_check": vector_store_health.get("last_check", 0)
                },
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheck(
                name="resource_utilization",
                status=HealthStatus.UNKNOWN,
                message=f"Resource check failed: {str(e)}",
                metrics={"error": str(e)},
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )

    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Determine overall health status from individual checks."""
        if any(check.status == HealthStatus.CRITICAL for check in checks):
            return HealthStatus.CRITICAL
        elif any(check.status == HealthStatus.WARNING for check in checks):
            return HealthStatus.WARNING
        elif any(check.status == HealthStatus.UNKNOWN for check in checks):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def _generate_summary(self, checks: List[HealthCheck]) -> Dict[str, Any]:
        """Generate health check summary."""
        status_counts = {}
        total_duration = 0

        for check in checks:
            status = check.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_duration += check.duration_ms

        return {
            "status_breakdown": status_counts,
            "total_checks": len(checks),
            "average_check_duration_ms": total_duration / len(checks) if checks else 0,
            "checks_passed": status_counts.get("healthy", 0),
            "checks_with_warnings": status_counts.get("warning", 0),
            "checks_failed": status_counts.get("critical", 0)
        }

    def _generate_recommendations(self, checks: List[HealthCheck]) -> List[str]:
        """Generate recommendations based on health check results."""
        recommendations = []

        for check in checks:
            if check.status == HealthStatus.CRITICAL:
                if check.name == "connection_health":
                    recommendations.append("Check network connectivity and Qdrant server status")
                elif check.name == "collection_health":
                    recommendations.append("Investigate collection integrity and consider rebuilding index")
                elif check.name == "search_functionality":
                    recommendations.append("Investigate search functionality and collection configuration")

            elif check.status == HealthStatus.WARNING:
                if check.name == "collection_health":
                    if "large" in check.message.lower():
                        recommendations.append("Consider collection optimization or archival of old data")
                    elif "indexing" in check.message.lower():
                        recommendations.append("Wait for indexing to complete or trigger manual optimization")
                elif check.name == "performance_health":
                    if "latency" in check.message.lower():
                        recommendations.append("Optimize search parameters or scale vector database")
                    elif "error rate" in check.message.lower():
                        recommendations.append("Investigate recent errors and connection stability")

        # Add general recommendations
        if not recommendations:
            recommendations.append("System is healthy - continue regular monitoring")
        else:
            recommendations.append("Monitor system closely and consider maintenance during low-traffic periods")

        return recommendations

    def _add_to_history(self, report: HealthReport) -> None:
        """Add health report to history with retention management."""
        self._health_history.append(report)

        # Clean up old history
        cutoff_time = datetime.now() - timedelta(hours=self.config.history_retention_hours)
        self._health_history = [
            r for r in self._health_history
            if r.timestamp > cutoff_time
        ]

    async def _check_and_send_alerts(self, report: HealthReport) -> None:
        """Check if alerts should be sent and trigger callbacks."""
        if report.overall_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
            for callback in self._alert_callbacks:
                try:
                    callback(report)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable[[HealthReport], None]) -> None:
        """Add an alert callback function."""
        self._alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[HealthReport], None]) -> None:
        """Remove an alert callback function."""
        if callback in self._alert_callbacks:
            self._alert_callbacks.remove(callback)

    def get_current_health(self) -> Optional[HealthReport]:
        """Get the most recent health report."""
        return self._last_health_check

    def get_health_history(self, hours: int = 24) -> List[HealthReport]:
        """Get health history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            report for report in self._health_history
            if report.timestamp > cutoff_time
        ]

    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics."""
        return {
            "monitoring_status": {
                "is_monitoring": self._is_monitoring,
                "check_interval_seconds": self.config.check_interval_seconds,
                "last_check": self._last_health_check.timestamp.isoformat() if self._last_health_check else None
            },
            "performance_metrics": self._performance_metrics.copy(),
            "history_stats": {
                "total_reports": len(self._health_history),
                "retention_hours": self.config.history_retention_hours,
                "oldest_report": self._health_history[0].timestamp.isoformat() if self._health_history else None
            },
            "alert_configuration": {
                "alerting_enabled": self.config.enable_alerting,
                "alert_callbacks_count": len(self._alert_callbacks),
                "thresholds": {
                    "search_latency_ms": self.config.search_latency_threshold_ms,
                    "error_rate": self.config.error_rate_threshold,
                    "collection_size": self.config.collection_size_threshold
                }
            }
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._performance_metrics = {
            "search_latencies": [],
            "connection_failures": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "last_error": None,
            "uptime_start": datetime.now()
        }
        logger.info("Health monitor metrics reset")
