"""
Concurrent processing engine with semaphore-based rate limiting and intelligent jitter.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from ..models.download_models import (
    ConcurrentProcessingError,
    TaskTimeoutError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ProcessorState(Enum):
    """Concurrent processor state."""

    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ProcessingStats:
    """Statistics for concurrent processing operations."""

    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    total_processing_time: float = 0.0
    max_concurrent_reached: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def active_tasks(self) -> int:
        """Calculate currently active tasks."""
        return (
            self.total_tasks
            - self.completed_tasks
            - self.failed_tasks
            - self.cancelled_tasks
        )

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        attempted_tasks = self.completed_tasks + self.failed_tasks
        if attempted_tasks == 0:
            return 0.0
        return (self.completed_tasks / attempted_tasks) * 100.0

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per task."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_processing_time / self.completed_tasks

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate total processing duration."""
        if not self.start_time:
            return None

        end_time = self.end_time or datetime.now()
        return (end_time - self.start_time).total_seconds()

    @property
    def throughput_tasks_per_second(self) -> float:
        """Calculate throughput in tasks per second."""
        duration = self.duration_seconds
        if not duration or duration <= 0:
            return 0.0

        return self.completed_tasks / duration


@dataclass
class JitterConfig:
    """Configuration for intelligent jitter timing."""

    min_delay: float = 0.5  # Minimum delay in seconds
    max_delay: float = 2.0  # Maximum delay in seconds
    progressive_factor: float = 1.2  # Factor for progressive delay increases
    adaptive_enabled: bool = True  # Enable adaptive jitter based on performance
    burst_protection: bool = True  # Enable burst protection

    def __post_init__(self) -> None:
        """Validate jitter configuration."""
        if self.min_delay < 0 or self.max_delay < self.min_delay:
            raise ValueError("Invalid jitter delay configuration")

        if self.progressive_factor < 1.0:
            raise ValueError("Progressive factor must be >= 1.0")


class TaskScheduler(Generic[T]):
    """
    Task scheduler with priority queuing and intelligent ordering.
    """

    def __init__(self, enable_priority_scheduling: bool = True):
        """
        Initialize task scheduler.

        Args:
            enable_priority_scheduling: Enable priority-based task ordering
        """
        self.enable_priority_scheduling = enable_priority_scheduling
        self._task_queue: List[T] = []
        self._scheduling_stats = {"reorders": 0, "priority_boosts": 0}

    def schedule_tasks(self, tasks: List[T]) -> List[T]:
        """
        Schedule tasks with intelligent ordering.

        Args:
            tasks: List of tasks to schedule

        Returns:
            Optimally ordered list of tasks
        """
        if not self.enable_priority_scheduling:
            # Simple randomization to avoid predictable patterns
            shuffled = tasks.copy()
            random.shuffle(shuffled)
            return shuffled

        # Sort by priority if available
        try:
            # Attempt to sort by priority_score if available
            scheduled = sorted(
                tasks, key=lambda t: getattr(t, "priority_score", 0), reverse=True
            )
            self._scheduling_stats["reorders"] += 1

            logger.debug(f"Scheduled {len(tasks)} tasks with priority ordering")
            return scheduled

        except (AttributeError, TypeError):
            # Fallback to original order if priority not available
            logger.debug(
                f"Priority scheduling not available, using original order for {len(tasks)} tasks"
            )
            return tasks

    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get scheduling statistics."""
        return self._scheduling_stats.copy()


class JitterManager:
    """
    Manages intelligent jitter timing to prevent thundering herd effects.
    """

    def __init__(self, config: Optional[JitterConfig] = None):
        """
        Initialize jitter manager.

        Args:
            config: Jitter configuration (uses defaults if None)
        """
        self.config = config or JitterConfig()
        self._performance_history: List[float] = []
        self._recent_delays: List[float] = []
        self._burst_tracker = {"last_burst_time": 0, "burst_count": 0}

        logger.debug(f"Initialized jitter manager: {self.config}")

    def calculate_jitter(self, task_index: int, total_tasks: int) -> float:
        """
        Calculate intelligent jitter delay for a task.

        Args:
            task_index: Index of the task (0-based)
            total_tasks: Total number of tasks

        Returns:
            Jitter delay in seconds
        """
        # Base jitter calculation
        if task_index == 0:
            # First task starts immediately
            delay = 0.0
        else:
            # Random jitter within configured range
            delay = random.uniform(self.config.min_delay, self.config.max_delay)

            # Apply progressive delay for later tasks to spread load
            if self.config.progressive_factor > 1.0 and task_index > 5:
                progress_multiplier = 1 + ((task_index - 5) * 0.1)
                progress_multiplier = min(
                    progress_multiplier, self.config.progressive_factor
                )
                delay *= progress_multiplier

        # Apply adaptive adjustments if enabled
        if self.config.adaptive_enabled:
            delay = self._apply_adaptive_adjustments(delay, task_index)

        # Apply burst protection if enabled
        if self.config.burst_protection:
            delay = self._apply_burst_protection(delay)

        # Clamp to reasonable bounds
        delay = max(0.0, min(delay, self.config.max_delay * 3))

        # Track delay for performance analysis
        self._recent_delays.append(delay)
        if len(self._recent_delays) > 20:
            self._recent_delays.pop(0)  # Keep only recent delays

        logger.debug(
            f"Calculated jitter delay: {delay:.3f}s for task {task_index + 1}/{total_tasks}"
        )
        return delay

    def _apply_adaptive_adjustments(self, base_delay: float, task_index: int) -> float:
        """Apply adaptive adjustments based on recent performance."""
        if not self._performance_history:
            return base_delay

        # Calculate recent average performance
        recent_performance = (
            self._performance_history[-5:]
            if len(self._performance_history) >= 5
            else self._performance_history
        )
        avg_performance = sum(recent_performance) / len(recent_performance)

        # Adjust delay based on performance
        # Good performance (fast) = reduce delays
        # Poor performance (slow) = increase delays
        if avg_performance < 2.0:  # Fast performance
            adjustment = 0.8
        elif avg_performance > 10.0:  # Slow performance
            adjustment = 1.3
        else:
            adjustment = 1.0

        adjusted_delay = base_delay * adjustment
        logger.debug(
            f"Adaptive adjustment: {adjustment:.2f} (avg_perf: {avg_performance:.2f}s)"
        )

        return adjusted_delay

    def _apply_burst_protection(self, base_delay: float) -> float:
        """Apply burst protection to prevent overwhelming downstream services."""
        current_time = time.time()

        # Check for recent burst activity
        if (
            current_time - self._burst_tracker["last_burst_time"] < 10
        ):  # Within 10 seconds
            self._burst_tracker["burst_count"] += 1

            # Apply increasing delays for bursts
            if self._burst_tracker["burst_count"] > 5:
                burst_multiplier = min(
                    1.5, 1 + (self._burst_tracker["burst_count"] - 5) * 0.1
                )
                base_delay *= burst_multiplier
                logger.debug(
                    f"Burst protection applied: {burst_multiplier:.2f}x multiplier"
                )
        else:
            # Reset burst tracking
            self._burst_tracker["burst_count"] = 0

        self._burst_tracker["last_burst_time"] = int(current_time)
        return base_delay

    def record_performance(self, processing_time: float) -> None:
        """Record performance data for adaptive adjustments."""
        self._performance_history.append(processing_time)

        # Keep only recent performance data
        if len(self._performance_history) > 50:
            self._performance_history.pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get jitter performance statistics."""
        recent_delays = self._recent_delays[-10:] if self._recent_delays else []
        recent_performance = (
            self._performance_history[-10:] if self._performance_history else []
        )

        return {
            "avg_recent_delay": (
                sum(recent_delays) / len(recent_delays) if recent_delays else 0.0
            ),
            "max_recent_delay": max(recent_delays) if recent_delays else 0.0,
            "avg_recent_performance": (
                sum(recent_performance) / len(recent_performance)
                if recent_performance
                else 0.0
            ),
            "burst_count": self._burst_tracker["burst_count"],
            "performance_samples": len(self._performance_history),
        }


class ConcurrentProcessor(Generic[T, R]):
    """
    High-performance concurrent processor with semaphore-based rate limiting,
    intelligent jitter, and comprehensive error handling.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        jitter_config: Optional[JitterConfig] = None,
        enable_priority_scheduling: bool = True,
        task_timeout: Optional[float] = None,
    ):
        """
        Initialize concurrent processor.

        Args:
            max_concurrent: Maximum concurrent tasks (1-50)
            jitter_config: Jitter configuration for timing
            enable_priority_scheduling: Enable intelligent task scheduling
            task_timeout: Per-task timeout in seconds (None = no timeout)
        """
        # Validate parameters
        if not (1 <= max_concurrent <= 50):
            raise ValueError(
                f"max_concurrent must be between 1 and 50, got {max_concurrent}"
            )

        self.max_concurrent = max_concurrent
        self.task_timeout = task_timeout

        # Initialize components
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.jitter_manager = JitterManager(jitter_config)
        self.task_scheduler = TaskScheduler[T](enable_priority_scheduling)

        # State tracking
        self.state = ProcessorState.IDLE
        self.stats = ProcessingStats()
        self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"Initialized concurrent processor: max_concurrent={max_concurrent}, timeout={task_timeout}"
        )

    async def process_with_concurrency(
        self,
        tasks: List[T],
        processor_func: Callable[[T], R],
        progress_callback: Optional[Callable[[ProcessingStats], None]] = None,
    ) -> List[R]:
        """
        Process tasks concurrently with rate limiting and error handling.

        Args:
            tasks: List of tasks to process
            processor_func: Async function to process each task
            progress_callback: Optional callback for progress updates

        Returns:
            List of results (may contain exceptions for failed tasks)

        Raises:
            ConcurrentProcessingError: If processing configuration is invalid
            TaskTimeoutError: If tasks consistently timeout
        """
        if not tasks:
            logger.info("No tasks to process")
            return []

        if self.state != ProcessorState.IDLE:
            raise ConcurrentProcessingError(
                f"Processor not idle (current state: {self.state.value})"
            )

        logger.info(f"Starting concurrent processing of {len(tasks)} tasks")

        try:
            # Initialize processing
            self.state = ProcessorState.PROCESSING
            self.stats = ProcessingStats(
                total_tasks=len(tasks), start_time=datetime.now()
            )

            # Schedule tasks for optimal processing order
            scheduled_tasks = self.task_scheduler.schedule_tasks(tasks)

            # Create coroutines with jitter delays
            coroutines = []
            for i, task in enumerate(scheduled_tasks):
                delay = self.jitter_manager.calculate_jitter(i, len(scheduled_tasks))
                coroutine = self._process_single_task_with_delay(
                    task, processor_func, delay, f"task_{i}"
                )
                coroutines.append(coroutine)

            # Execute with progress tracking
            results = await self._execute_with_progress_tracking(
                coroutines, progress_callback
            )

            # Finalize statistics
            self.stats.end_time = datetime.now()
            self.state = ProcessorState.IDLE

            # Log completion summary
            self._log_completion_summary()

            return results

        except Exception as e:
            self.state = ProcessorState.IDLE
            self.stats.end_time = datetime.now()
            logger.error(f"Concurrent processing failed: {e}")
            raise ConcurrentProcessingError(f"Processing failed: {e}") from e

    async def _process_single_task_with_delay(
        self, task: T, processor_func: Callable[[T], R], delay: float, task_name: str
    ) -> R:
        """
        Process single task with delay, semaphore control, and error handling.

        Args:
            task: Task to process
            processor_func: Processing function
            delay: Initial delay before processing
            task_name: Name for tracking/logging

        Returns:
            Processing result or exception
        """
        # Apply initial delay
        if delay > 0:
            await asyncio.sleep(delay)

        # Check for shutdown request
        if self.state == ProcessorState.SHUTTING_DOWN:
            logger.info(f"Skipping {task_name} due to shutdown request")
            self.stats.cancelled_tasks += 1
            return None  # type: ignore

        start_time = time.time()

        try:
            # Acquire semaphore for rate limiting
            async with self.semaphore:
                # Update concurrent task tracking
                self._update_max_concurrent()

                logger.debug(f"Processing {task_name} (acquired semaphore)")

                # Process with timeout if configured
                if self.task_timeout:
                    try:
                        processor_result = processor_func(task)
                        result: R = await asyncio.wait_for(
                            processor_result,  # type: ignore[arg-type]
                            timeout=self.task_timeout,
                        )
                    except asyncio.TimeoutError:
                        error_msg = (
                            f"Task {task_name} timed out after {self.task_timeout}s"
                        )
                        logger.warning(error_msg)
                        self.stats.failed_tasks += 1
                        raise TaskTimeoutError(
                            error_msg,
                            task_id=task_name,
                            timeout_seconds=self.task_timeout,
                        ) from None
                else:
                    # No timeout
                    result = await processor_func(task)  # type: ignore

                # Record successful completion
                processing_time = time.time() - start_time
                self.stats.completed_tasks += 1
                self.stats.total_processing_time += processing_time

                self.jitter_manager.record_performance(processing_time)

                logger.debug(f"Completed {task_name} in {processing_time:.3f}s")
                return result

        except Exception as e:
            # Record failure
            processing_time = time.time() - start_time
            self.stats.failed_tasks += 1

            # Log error appropriately
            if isinstance(e, TaskTimeoutError):
                logger.warning(f"Task {task_name} timed out: {e}")
            else:
                logger.error(
                    f"Task {task_name} failed after {processing_time:.3f}s: {e}"
                )

            # Return the exception for the caller to handle
            return e  # type: ignore

    def _update_max_concurrent(self) -> None:
        """Update maximum concurrent tasks reached statistic."""
        current_active = len(self._active_tasks)
        self.stats.max_concurrent_reached = max(
            self.stats.max_concurrent_reached,
            current_active + 1,  # +1 for current task
        )

    async def _execute_with_progress_tracking(
        self,
        coroutines: List[Any],
        progress_callback: Optional[Callable[[ProcessingStats], None]],
    ) -> List[R]:
        """Execute coroutines with progress tracking and callback updates."""

        # Execute all coroutines concurrently
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Call progress callback if provided
        if progress_callback:
            try:
                progress_callback(self.stats)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

        return results  # type: ignore

    def _log_completion_summary(self) -> None:
        """Log comprehensive completion summary."""
        duration = self.stats.duration_seconds or 0

        logger.info(
            f"Concurrent processing completed: "
            f"{self.stats.completed_tasks}/{self.stats.total_tasks} successful "
            f"({self.stats.completion_rate:.1f}% completion, "
            f"{self.stats.success_rate:.1f}% success rate) "
            f"in {duration:.2f}s"
        )

        if self.stats.failed_tasks > 0:
            logger.warning(f"Failed tasks: {self.stats.failed_tasks}")

        if self.stats.cancelled_tasks > 0:
            logger.warning(f"Cancelled tasks: {self.stats.cancelled_tasks}")

        logger.debug(
            f"Performance metrics: "
            f"avg_time={self.stats.average_processing_time:.3f}s, "
            f"throughput={self.stats.throughput_tasks_per_second:.1f} tasks/sec, "
            f"max_concurrent={self.stats.max_concurrent_reached}"
        )

    async def shutdown(self, timeout: float = 30.0) -> None:
        """
        Gracefully shutdown the processor.

        Args:
            timeout: Maximum time to wait for active tasks to complete
        """
        logger.info("Initiating concurrent processor shutdown")
        self.state = ProcessorState.SHUTTING_DOWN

        # Signal shutdown to running tasks
        self._shutdown_event.set()

        # Wait for active tasks to complete
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._active_tasks.values(), return_exceptions=True
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Shutdown timeout reached, cancelling {len(self._active_tasks)} active tasks"
                )
                for task in self._active_tasks.values():
                    task.cancel()

        self.state = ProcessorState.SHUTDOWN
        logger.info("Concurrent processor shutdown completed")

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        base_stats = {
            "processing": {
                "total_tasks": self.stats.total_tasks,
                "completed_tasks": self.stats.completed_tasks,
                "failed_tasks": self.stats.failed_tasks,
                "cancelled_tasks": self.stats.cancelled_tasks,
                "completion_rate": self.stats.completion_rate,
                "success_rate": self.stats.success_rate,
                "average_processing_time": self.stats.average_processing_time,
                "throughput_tasks_per_second": self.stats.throughput_tasks_per_second,
                "max_concurrent_reached": self.stats.max_concurrent_reached,
                "duration_seconds": self.stats.duration_seconds,
            },
            "configuration": {
                "max_concurrent": self.max_concurrent,
                "task_timeout": self.task_timeout,
                "current_state": self.state.value,
            },
        }

        # Add jitter manager stats
        base_stats["jitter"] = self.jitter_manager.get_performance_stats()

        # Add task scheduler stats
        base_stats["scheduling"] = self.task_scheduler.get_scheduling_stats()

        return base_stats

    @property
    def is_active(self) -> bool:
        """Check if processor is currently active."""
        return self.state == ProcessorState.PROCESSING

    @property
    def can_process(self) -> bool:
        """Check if processor can accept new tasks."""
        return self.state == ProcessorState.IDLE
