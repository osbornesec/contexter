"""
Progress reporting and metrics collection system for download operations.
"""

import asyncio
import logging
import time
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ..models.download_models import (
    DocumentationChunk,
    DownloadSummary,
    ProgressMetrics,
)

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""

    DOWNLOAD_STARTED = "download_started"
    CONTEXT_STARTED = "context_started"
    CONTEXT_COMPLETED = "context_completed"
    CONTEXT_FAILED = "context_failed"
    CONTEXT_RETRYING = "context_retrying"
    DOWNLOAD_COMPLETED = "download_completed"
    DOWNLOAD_FAILED = "download_failed"
    BATCH_STARTED = "batch_started"
    BATCH_COMPLETED = "batch_completed"


@dataclass
class ProgressEvent:
    """Single progress event with metadata."""

    event_type: ProgressEventType
    library_id: str
    timestamp: datetime
    context: Optional[str] = None
    task_id: Optional[str] = None
    tokens: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        """Get age of event in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


class ProgressCallback(Protocol):
    """Protocol for progress callback functions."""

    def __call__(self, event: ProgressEvent, metrics: ProgressMetrics) -> None:
        """Handle progress event with current metrics."""
        pass


@dataclass
class RealtimeStats:
    """Real-time statistics for download operations."""

    current_concurrent_downloads: int = 0
    total_tokens_per_second: float = 0.0
    average_context_duration: float = 0.0
    success_rate_trend: List[float] = field(default_factory=list)
    recent_errors: deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    peak_concurrency: int = 0

    def update_concurrency(self, current: int) -> None:
        """Update concurrency tracking."""
        self.current_concurrent_downloads = current
        self.peak_concurrency = max(self.peak_concurrency, current)

    def add_success_rate(self, rate: float) -> None:
        """Add success rate data point."""
        self.success_rate_trend.append(rate)
        # Keep only recent trend data (last 20 points)
        if len(self.success_rate_trend) > 20:
            self.success_rate_trend.pop(0)

    def add_error(self, error: str) -> None:
        """Add recent error."""
        self.recent_errors.append({"error": error, "timestamp": datetime.now()})


class ProgressReporter:
    """
    Comprehensive progress reporting system with real-time metrics.

    Tracks download progress, collects performance metrics, and provides
    real-time updates through callback mechanisms.
    """

    def __init__(
        self,
        library_id: str,
        total_contexts: int,
        enable_realtime_stats: bool = True,
        event_buffer_size: int = 100,
    ):
        """
        Initialize progress reporter.

        Args:
            library_id: Library being downloaded
            total_contexts: Total number of contexts to process
            enable_realtime_stats: Enable detailed real-time statistics
            event_buffer_size: Size of event history buffer
        """
        self.library_id = library_id
        self.total_contexts = total_contexts
        self.enable_realtime_stats = enable_realtime_stats

        # Progress tracking
        self.metrics = ProgressMetrics(total_contexts=total_contexts)
        self.context_progress: Dict[str, Dict[str, Any]] = {}

        # Event tracking
        self.events: deque[ProgressEvent] = deque(maxlen=event_buffer_size)
        self.callbacks: List[ProgressCallback] = []

        # Real-time statistics
        self.realtime_stats = RealtimeStats() if enable_realtime_stats else None

        # Timing tracking
        self.context_timers: Dict[str, float] = {}
        self.download_start_time = time.time()

        logger.info(
            f"Initialized progress reporter for {library_id}: {total_contexts} contexts"
        )

    def add_callback(self, callback: ProgressCallback) -> None:
        """Add progress callback function."""
        self.callbacks.append(callback)
        logger.debug(f"Added progress callback: {callback}")

    def remove_callback(self, callback: ProgressCallback) -> None:
        """Remove progress callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Removed progress callback: {callback}")

    async def report_download_started(self, contexts: List[str]) -> None:
        """Report download operation started."""
        event = ProgressEvent(
            event_type=ProgressEventType.DOWNLOAD_STARTED,
            library_id=self.library_id,
            timestamp=datetime.now(),
            metadata={"total_contexts": len(contexts), "contexts": contexts},
        )

        await self._emit_event(event)

        # Initialize context progress tracking
        for context in contexts:
            self.context_progress[context] = {
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "tokens": 0,
                "error": None,
            }

        logger.info(f"Download started for {self.library_id}: {len(contexts)} contexts")

    async def report_context_started(self, context: str, task_id: str) -> None:
        """Report context download started."""
        start_time = time.time()

        event = ProgressEvent(
            event_type=ProgressEventType.CONTEXT_STARTED,
            library_id=self.library_id,
            timestamp=datetime.now(),
            context=context,
            task_id=task_id,
        )

        # Update progress tracking
        if context in self.context_progress:
            self.context_progress[context].update(
                {"status": "in_progress", "start_time": start_time, "task_id": task_id}
            )

        # Track timing
        self.context_timers[context] = start_time

        # Update real-time stats
        if self.realtime_stats:
            active_contexts = sum(
                1
                for p in self.context_progress.values()
                if p["status"] == "in_progress"
            )
            self.realtime_stats.update_concurrency(active_contexts)

        await self._emit_event(event)
        logger.debug(f"Context started: {context} (task: {task_id})")

    async def report_context_completed(
        self, context: str, chunk: DocumentationChunk, task_id: Optional[str] = None
    ) -> None:
        """Report context download completed successfully."""
        end_time = time.time()
        duration = end_time - self.context_timers.get(context, end_time)

        event = ProgressEvent(
            event_type=ProgressEventType.CONTEXT_COMPLETED,
            library_id=self.library_id,
            timestamp=datetime.now(),
            context=context,
            task_id=task_id,
            tokens=chunk.token_count,
            metadata={
                "duration": duration,
                "chunk_id": chunk.chunk_id,
                "content_hash": chunk.content_hash,
                "proxy_id": chunk.proxy_id,
            },
        )

        # Update progress tracking
        if context in self.context_progress:
            self.context_progress[context].update(
                {
                    "status": "completed",
                    "end_time": end_time,
                    "tokens": chunk.token_count,
                    "duration": duration,
                }
            )

        # Update metrics
        self.metrics.update_completion(chunk)

        # Update real-time stats
        if self.realtime_stats:
            self._update_realtime_stats_on_completion(duration, chunk.token_count)

        await self._emit_event(event)
        logger.debug(
            f"Context completed: {context} ({chunk.token_count} tokens in {duration:.2f}s)"
        )

    async def report_context_failed(
        self,
        context: str,
        error: Exception,
        task_id: Optional[str] = None,
        is_retry: bool = False,
    ) -> None:
        """Report context download failed."""
        end_time = time.time()
        duration = end_time - self.context_timers.get(context, end_time)
        error_message = str(error)

        event_type = (
            ProgressEventType.CONTEXT_RETRYING
            if is_retry
            else ProgressEventType.CONTEXT_FAILED
        )

        event = ProgressEvent(
            event_type=event_type,
            library_id=self.library_id,
            timestamp=datetime.now(),
            context=context,
            task_id=task_id,
            error_message=error_message,
            metadata={
                "duration": duration,
                "error_type": type(error).__name__,
                "is_retry": is_retry,
            },
        )

        # Update progress tracking
        if context in self.context_progress:
            status = "retrying" if is_retry else "failed"
            self.context_progress[context].update(
                {
                    "status": status,
                    "end_time": end_time,
                    "error": error_message,
                    "duration": duration,
                }
            )

        # Update metrics if not a retry
        if not is_retry:
            self.metrics.update_failure()

        # Update real-time stats
        if self.realtime_stats:
            self.realtime_stats.add_error(error_message)
            if not is_retry:
                self._update_success_rate_trend()

        await self._emit_event(event)

        log_level = logging.WARNING if not is_retry else logging.DEBUG
        logger.log(
            log_level,
            f"Context {'retrying' if is_retry else 'failed'}: {context} - {error_message}",
        )

    async def report_download_completed(self, summary: DownloadSummary) -> None:
        """Report download operation completed."""
        event = ProgressEvent(
            event_type=ProgressEventType.DOWNLOAD_COMPLETED,
            library_id=self.library_id,
            timestamp=datetime.now(),
            tokens=summary.total_tokens,
            metadata={
                "success_rate": summary.success_rate,
                "total_duration": summary.duration_seconds,
                "efficiency_score": summary.efficiency_score,
                "chunks_count": len(summary.chunks),
            },
        )

        await self._emit_event(event)

        logger.info(
            f"Download completed for {self.library_id}: "
            f"{summary.successful_contexts}/{summary.total_contexts_attempted} successful, "
            f"{summary.total_tokens} tokens, {summary.success_rate:.1f}% success rate"
        )

    async def report_download_failed(self, error: Exception) -> None:
        """Report download operation failed completely."""
        event = ProgressEvent(
            event_type=ProgressEventType.DOWNLOAD_FAILED,
            library_id=self.library_id,
            timestamp=datetime.now(),
            error_message=str(error),
            metadata={
                "error_type": type(error).__name__,
                "contexts_attempted": len(self.context_progress),
                "contexts_completed": sum(
                    1
                    for p in self.context_progress.values()
                    if p["status"] == "completed"
                ),
            },
        )

        await self._emit_event(event)
        logger.error(f"Download failed for {self.library_id}: {error}")

    def _update_realtime_stats_on_completion(
        self, duration: float, tokens: int
    ) -> None:
        """Update real-time statistics on context completion."""
        if not self.realtime_stats:
            return

        # Update active concurrency
        active_contexts = sum(
            1 for p in self.context_progress.values() if p["status"] == "in_progress"
        )
        self.realtime_stats.update_concurrency(active_contexts)

        # Update average duration
        completed_durations = [
            p["duration"]
            for p in self.context_progress.values()
            if p["status"] == "completed" and "duration" in p
        ]

        if completed_durations:
            self.realtime_stats.average_context_duration = sum(
                completed_durations
            ) / len(completed_durations)

        # Update tokens per second
        total_duration = time.time() - self.download_start_time
        if total_duration > 0:
            self.realtime_stats.total_tokens_per_second = (
                self.metrics.total_tokens_retrieved / total_duration
            )

        # Update success rate trend
        self._update_success_rate_trend()

    def _update_success_rate_trend(self) -> None:
        """Update success rate trend data."""
        if not self.realtime_stats:
            return

        current_success_rate = self.metrics.success_rate
        self.realtime_stats.add_success_rate(current_success_rate)

    async def _emit_event(self, event: ProgressEvent) -> None:
        """Emit progress event to all registered callbacks."""
        # Add to event history
        self.events.append(event)

        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, self.metrics)
                else:
                    callback(event, self.metrics)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get comprehensive current status."""
        elapsed_time = time.time() - self.download_start_time

        status = {
            "library_id": self.library_id,
            "elapsed_time": elapsed_time,
            "metrics": {
                "total_contexts": self.metrics.total_contexts,
                "completed": self.metrics.completed_contexts,
                "failed": self.metrics.failed_contexts,
                "in_progress": self.metrics.in_progress_contexts,
                "completion_rate": self.metrics.completion_rate,
                "success_rate": self.metrics.success_rate,
                "total_tokens": self.metrics.total_tokens_retrieved,
                "estimated_remaining_time": self.metrics.estimated_remaining_time,
            },
            "context_details": {},
        }

        # Add context details
        for context, progress in self.context_progress.items():
            context_details = status["context_details"]
            assert isinstance(context_details, dict)
            context_details[context] = {
                "status": progress["status"],
                "tokens": progress.get("tokens", 0),
                "duration": progress.get("duration"),
                "error": progress.get("error"),
            }

        # Add real-time stats if available
        if self.realtime_stats:
            status["realtime_stats"] = {
                "current_concurrent": self.realtime_stats.current_concurrent_downloads,
                "peak_concurrent": self.realtime_stats.peak_concurrency,
                "tokens_per_second": self.realtime_stats.total_tokens_per_second,
                "avg_context_duration": self.realtime_stats.average_context_duration,
                "success_rate_trend": self.realtime_stats.success_rate_trend[
                    -5:
                ],  # Last 5 points
                "recent_errors": list(self.realtime_stats.recent_errors),
            }

        return status

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of context progress."""
        completed = []
        failed = []
        in_progress = []

        for context, progress in self.context_progress.items():
            summary = {
                "context": context,
                "status": progress["status"],
                "tokens": progress.get("tokens", 0),
                "duration": progress.get("duration"),
            }

            if progress["status"] == "completed":
                completed.append(summary)
            elif progress["status"] == "failed":
                summary["error"] = progress.get("error")
                failed.append(summary)
            elif progress["status"] == "in_progress":
                in_progress.append(summary)

        return {
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "summary": {
                "completed_count": len(completed),
                "failed_count": len(failed),
                "in_progress_count": len(in_progress),
                "total_tokens": sum(c["tokens"] for c in completed),
            },
        }

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent progress events."""
        recent_events = list(self.events)[-limit:] if self.events else []

        return [
            {
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "context": event.context,
                "task_id": event.task_id,
                "tokens": event.tokens,
                "error": event.error_message,
                "age_seconds": event.age_seconds,
                "metadata": event.metadata,
            }
            for event in recent_events
        ]


class BatchProgressReporter:
    """
    Progress reporter for batch download operations across multiple libraries.
    """

    def __init__(self, library_ids: List[str]):
        """
        Initialize batch progress reporter.

        Args:
            library_ids: List of library IDs being downloaded
        """
        self.library_ids = library_ids
        self.library_reporters: Dict[str, ProgressReporter] = {}
        self.batch_start_time = time.time()
        self.callbacks: List[
            Union[
                Callable[[ProgressEvent, Dict[str, Any]], None],
                Callable[[ProgressEvent, Dict[str, Any]], Awaitable[None]],
            ]
        ] = []
        self.events: deque[ProgressEvent] = deque(
            maxlen=200
        )  # Larger buffer for batch operations

        logger.info(
            f"Initialized batch progress reporter for {len(library_ids)} libraries"
        )

    def add_library_reporter(self, library_id: str, reporter: ProgressReporter) -> None:
        """Add individual library reporter."""
        self.library_reporters[library_id] = reporter

        # Forward library events to batch events
        reporter.add_callback(self._forward_library_event)  # type: ignore

    async def _forward_library_event(
        self, event: ProgressEvent, metrics: ProgressMetrics
    ) -> None:
        """Forward library-level events to batch-level tracking."""
        self.events.append(event)

        # Call batch-level callbacks
        batch_summary = self.get_batch_summary()
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, batch_summary)
                else:
                    callback(event, batch_summary)
            except Exception as e:
                logger.warning(f"Batch progress callback failed: {e}")

    def add_callback(
        self,
        callback: Union[
            Callable[[ProgressEvent, Dict[str, Any]], None],
            Callable[[ProgressEvent, Dict[str, Any]], Awaitable[None]],
        ],
    ) -> None:
        """Add batch-level progress callback."""
        self.callbacks.append(callback)

    def get_batch_summary(self) -> Dict[str, Any]:
        """Get comprehensive batch progress summary."""
        total_libraries = len(self.library_ids)
        completed_libraries = 0
        failed_libraries = 0
        in_progress_libraries = 0
        total_contexts = 0
        completed_contexts = 0
        failed_contexts = 0
        total_tokens = 0

        library_statuses = {}

        for library_id in self.library_ids:
            if library_id in self.library_reporters:
                reporter = self.library_reporters[library_id]
                metrics = reporter.metrics

                # Determine library status
                if metrics.completion_rate >= 1.0:
                    if metrics.success_rate > 0:
                        completed_libraries += 1
                        status = "completed"
                    else:
                        failed_libraries += 1
                        status = "failed"
                else:
                    in_progress_libraries += 1
                    status = "in_progress"

                # Aggregate metrics
                total_contexts += metrics.total_contexts
                completed_contexts += metrics.completed_contexts
                failed_contexts += metrics.failed_contexts
                total_tokens += metrics.total_tokens_retrieved

                library_statuses[library_id] = {
                    "status": status,
                    "completion_rate": metrics.completion_rate,
                    "success_rate": metrics.success_rate,
                    "tokens": metrics.total_tokens_retrieved,
                    "contexts": {
                        "total": metrics.total_contexts,
                        "completed": metrics.completed_contexts,
                        "failed": metrics.failed_contexts,
                    },
                }
            else:
                library_statuses[library_id] = {
                    "status": "pending",
                    "completion_rate": 0.0,
                    "success_rate": 0.0,
                    "tokens": 0,
                    "contexts": {"total": 0, "completed": 0, "failed": 0},
                }

        # Calculate batch-level metrics
        elapsed_time = time.time() - self.batch_start_time
        batch_completion_rate = (
            (completed_libraries + failed_libraries) / total_libraries
            if total_libraries > 0
            else 0
        )
        overall_success_rate = (
            (completed_contexts / (completed_contexts + failed_contexts) * 100)
            if (completed_contexts + failed_contexts) > 0
            else 0
        )

        return {
            "batch_metrics": {
                "total_libraries": total_libraries,
                "completed_libraries": completed_libraries,
                "failed_libraries": failed_libraries,
                "in_progress_libraries": in_progress_libraries,
                "batch_completion_rate": batch_completion_rate,
                "elapsed_time": elapsed_time,
            },
            "aggregate_metrics": {
                "total_contexts": total_contexts,
                "completed_contexts": completed_contexts,
                "failed_contexts": failed_contexts,
                "total_tokens": total_tokens,
                "overall_success_rate": overall_success_rate,
                "tokens_per_second": (
                    total_tokens / elapsed_time if elapsed_time > 0 else 0
                ),
            },
            "library_statuses": library_statuses,
        }
