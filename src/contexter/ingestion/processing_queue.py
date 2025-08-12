"""
Processing Queue Management - Priority-based job scheduling system.

Provides efficient queue management for concurrent document processing
with priority-based scheduling, worker pool management, and comprehensive
error handling with retry logic.
"""

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
import json

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Processing job status states."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10
    CRITICAL = 15


@dataclass
class IngestionJob:
    """
    Ingestion job with priority-based scheduling.
    
    Represents a single document processing job with all necessary
    metadata for processing and tracking.
    """
    # Core job information
    job_id: str
    library_id: str
    version: str
    doc_path: Path
    priority: int
    metadata: Dict[str, Any]
    
    # Scheduling and timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status tracking
    status: JobStatus = JobStatus.QUEUED
    processing_attempts: int = 0
    max_retries: int = 3
    
    # Error handling
    error_message: Optional[str] = None
    last_error_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    
    # Processing results
    processing_time: Optional[float] = None
    chunks_processed: Optional[int] = None
    vectors_created: Optional[int] = None
    
    def __lt__(self, other: 'IngestionJob') -> bool:
        """
        Priority comparison for heap queue.
        Higher priority values come first (max-heap behavior).
        """
        if self.priority != other.priority:
            return self.priority > other.priority
        
        # If same priority, older jobs first
        return self.created_at < other.created_at
    
    def start_processing(self, worker_id: str) -> None:
        """Mark job as started."""
        self.status = JobStatus.PROCESSING
        self.started_at = datetime.now()
        self.worker_id = worker_id
        self.processing_attempts += 1
    
    def complete_successfully(self, chunks_processed: int, vectors_created: int) -> None:
        """Mark job as completed successfully."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now()
        self.chunks_processed = chunks_processed
        self.vectors_created = vectors_created
        
        if self.started_at:
            self.processing_time = (self.completed_at - self.started_at).total_seconds()
    
    def fail_with_error(self, error_message: str) -> None:
        """Mark job as failed with error."""
        self.error_message = error_message
        self.last_error_at = datetime.now()
        
        if self.processing_attempts >= self.max_retries:
            self.status = JobStatus.FAILED
        else:
            self.status = JobStatus.RETRYING
    
    def reset_for_retry(self) -> None:
        """Reset job state for retry."""
        self.status = JobStatus.QUEUED
        self.started_at = None
        self.worker_id = None
        # Keep error information for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'library_id': self.library_id,
            'version': self.version,
            'doc_path': str(self.doc_path),
            'priority': self.priority,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status.value,
            'processing_attempts': self.processing_attempts,
            'max_retries': self.max_retries,
            'error_message': self.error_message,
            'last_error_at': self.last_error_at.isoformat() if self.last_error_at else None,
            'worker_id': self.worker_id,
            'processing_time': self.processing_time,
            'chunks_processed': self.chunks_processed,
            'vectors_created': self.vectors_created
        }


class QueueFullError(Exception):
    """Raised when the processing queue is full."""
    pass


class IngestionQueue:
    """
    Priority-based ingestion queue with async support.
    
    Features:
    - Priority-based scheduling with heap queue
    - Async-safe operations with proper locking
    - Queue capacity management and monitoring
    - Job persistence and recovery capabilities
    - Comprehensive metrics and statistics
    """
    
    def __init__(
        self, 
        max_size: int = 10000,
        persistence_path: Optional[Path] = None
    ):
        """
        Initialize the ingestion queue.
        
        Args:
            max_size: Maximum queue capacity
            persistence_path: Optional path for job persistence
        """
        self.max_size = max_size
        self.persistence_path = persistence_path
        
        # Queue implementation
        self.queue: List[IngestionJob] = []
        self.job_registry: Dict[str, IngestionJob] = {}
        
        # Async synchronization
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition()
        self.not_full = asyncio.Condition()
        
        # Statistics
        self.stats = {
            'jobs_queued': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'queue_wait_times': [],
            'priority_distribution': {p.value: 0 for p in JobPriority},
            'last_activity': None
        }
        
        logger.info(f"Ingestion queue initialized (capacity: {max_size})")
    
    async def put(
        self, 
        library_id: str,
        version: str,
        doc_path: Path,
        priority: int,
        metadata: Dict[str, Any],
        max_retries: int = 3
    ) -> str:
        """
        Add a job to the priority queue.
        
        Args:
            library_id: Library identifier
            version: Version string
            doc_path: Path to documentation
            priority: Job priority (0-15, higher = more urgent)
            metadata: Job metadata
            max_retries: Maximum retry attempts
            
        Returns:
            Job ID for tracking
            
        Raises:
            QueueFullError: If queue is at capacity
        """
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            library_id=library_id,
            version=version,
            doc_path=doc_path,
            priority=priority,
            metadata=metadata,
            max_retries=max_retries
        )
        
        async with self.not_full:
            while len(self.queue) >= self.max_size:
                if self.not_full._waiters:
                    await self.not_full.wait()
                else:
                    raise QueueFullError(f"Queue is full (size: {len(self.queue)})")
            
            async with self.lock:
                # Add to heap queue
                heapq.heappush(self.queue, job)
                self.job_registry[job.job_id] = job
                
                # Update statistics
                self.stats['jobs_queued'] += 1
                self.stats['priority_distribution'][min(priority, 15)] += 1
                self.stats['last_activity'] = datetime.now()
                
                # Persist if enabled
                if self.persistence_path:
                    await self._persist_job(job)
            
            # Notify waiting consumers
            async with self.not_empty:
                self.not_empty.notify()
        
        logger.debug(f"Queued job {job.job_id} for {library_id}:{version} (priority: {priority})")
        return job.job_id
    
    async def get(self, timeout: Optional[float] = None) -> Optional[IngestionJob]:
        """
        Get the highest priority job from the queue.
        
        Args:
            timeout: Maximum wait time (None = wait indefinitely)
            
        Returns:
            Highest priority job or None if timeout
        """
        async with self.not_empty:
            while not self.queue:
                if timeout is not None:
                    try:
                        await asyncio.wait_for(self.not_empty.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self.not_empty.wait()
            
            async with self.lock:
                job = heapq.heappop(self.queue)
                
                # Calculate queue wait time
                wait_time = (datetime.now() - job.created_at).total_seconds()
                self.stats['queue_wait_times'].append(wait_time)
                
                # Keep only recent wait times for statistics
                if len(self.stats['queue_wait_times']) > 1000:
                    self.stats['queue_wait_times'] = self.stats['queue_wait_times'][-500:]
            
            # Notify waiting producers
            async with self.not_full:
                self.not_full.notify()
        
        logger.debug(f"Dequeued job {job.job_id} after {wait_time:.1f}s wait")
        return job
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        async with self.lock:
            job = self.job_registry.get(job_id)
            if job:
                return job.to_dict()
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled, False if not found or already processing
        """
        async with self.lock:
            job = self.job_registry.get(job_id)
            if not job:
                return False
            
            if job.status == JobStatus.PROCESSING:
                logger.warning(f"Cannot cancel job {job_id} - already processing")
                return False
            
            # Remove from queue if still queued
            if job.status == JobStatus.QUEUED:
                try:
                    self.queue.remove(job)
                    heapq.heapify(self.queue)  # Restore heap property
                except ValueError:
                    pass  # Job not in queue anymore
            
            job.status = JobStatus.CANCELLED
            logger.info(f"Cancelled job {job_id}")
            return True
    
    async def retry_failed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Retry recently failed jobs.
        
        Args:
            max_age_hours: Maximum age of failed jobs to retry
            
        Returns:
            Number of jobs requeued for retry
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        retry_count = 0
        
        async with self.lock:
            failed_jobs = [
                job for job in self.job_registry.values()
                if job.status == JobStatus.FAILED and 
                job.last_error_at and job.last_error_at > cutoff_time and
                job.processing_attempts < job.max_retries
            ]
            
            for job in failed_jobs:
                if len(self.queue) < self.max_size:
                    job.reset_for_retry()
                    heapq.heappush(self.queue, job)
                    retry_count += 1
        
        if retry_count > 0:
            # Notify waiting consumers
            async with self.not_empty:
                self.not_empty.notify_all()
            
            logger.info(f"Requeued {retry_count} failed jobs for retry")
        
        return retry_count
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: JobStatus,
        **kwargs
    ) -> bool:
        """
        Update job status and metadata.
        
        Args:
            job_id: Job identifier
            status: New status
            **kwargs: Additional fields to update
            
        Returns:
            True if updated successfully
        """
        async with self.lock:
            job = self.job_registry.get(job_id)
            if not job:
                return False
            
            job.status = status
            
            # Update specific fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            # Update statistics
            if status == JobStatus.COMPLETED:
                self.stats['jobs_completed'] += 1
                if job.processing_time:
                    self.stats['total_processing_time'] += job.processing_time
            elif status == JobStatus.FAILED:
                self.stats['jobs_failed'] += 1
            
            # Persist if enabled
            if self.persistence_path:
                await self._persist_job(job)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        # Calculate derived metrics
        wait_times = self.stats['queue_wait_times']
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0.0
        max_wait_time = max(wait_times) if wait_times else 0.0
        
        avg_processing_time = 0.0
        if self.stats['jobs_completed'] > 0:
            avg_processing_time = self.stats['total_processing_time'] / self.stats['jobs_completed']
        
        # Count jobs by status
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(
                1 for job in self.job_registry.values() 
                if job.status == status
            )
        
        return {
            'queue_size': len(self.queue),
            'max_capacity': self.max_size,
            'utilization': len(self.queue) / self.max_size,
            'total_jobs': len(self.job_registry),
            'jobs_queued': self.stats['jobs_queued'],
            'jobs_completed': self.stats['jobs_completed'],
            'jobs_failed': self.stats['jobs_failed'],
            'success_rate': (
                self.stats['jobs_completed'] / max(1, self.stats['jobs_completed'] + self.stats['jobs_failed'])
            ),
            'avg_wait_time': avg_wait_time,
            'max_wait_time': max_wait_time,
            'avg_processing_time': avg_processing_time,
            'total_processing_time': self.stats['total_processing_time'],
            'status_distribution': status_counts,
            'priority_distribution': self.stats['priority_distribution'],
            'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None
        }
    
    async def cleanup_completed_jobs(self, max_age_hours: int = 168) -> int:
        """
        Clean up old completed jobs to free memory.
        
        Args:
            max_age_hours: Maximum age of completed jobs to keep (default: 7 days)
            
        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleanup_count = 0
        
        async with self.lock:
            jobs_to_remove = []
            
            for job_id, job in self.job_registry.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.job_registry[job_id]
                cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} old completed jobs")
        
        return cleanup_count
    
    async def _persist_job(self, job: IngestionJob) -> None:
        """Persist job state for recovery."""
        if not self.persistence_path:
            return
        
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            job_file = self.persistence_path / f"{job.job_id}.json"
            
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist job {job.job_id}: {e}")


class WorkerPool:
    """
    Concurrent worker pool for processing ingestion jobs.
    
    Features:
    - Configurable worker count with dynamic scaling
    - Automatic error recovery and retry logic
    - Worker health monitoring and replacement
    - Load balancing across available workers
    - Comprehensive performance metrics
    """
    
    def __init__(
        self,
        max_workers: int,
        job_processor: Callable[[IngestionJob], Awaitable[bool]],
        queue: IngestionQueue,
        worker_timeout: float = 300.0
    ):
        """
        Initialize the worker pool.
        
        Args:
            max_workers: Maximum number of concurrent workers
            job_processor: Async function to process jobs
            queue: Job queue to pull from
            worker_timeout: Maximum processing time per job
        """
        self.max_workers = max_workers
        self.job_processor = job_processor
        self.queue = queue
        self.worker_timeout = worker_timeout
        
        # Worker management
        self.workers: Dict[str, asyncio.Task] = {}
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.stats = {
            'workers_started': 0,
            'workers_stopped': 0,
            'jobs_processed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'worker_restarts': 0,
            'last_activity': None
        }
        
        logger.info(f"Worker pool initialized with {max_workers} workers")
    
    async def start(self) -> None:
        """Start the worker pool."""
        if self.workers:
            logger.warning("Worker pool already started")
            return
        
        self._shutdown_event.clear()
        
        # Start workers
        for i in range(self.max_workers):
            await self._start_worker(f"worker-{i}")
        
        logger.info(f"Worker pool started with {len(self.workers)} workers")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool gracefully."""
        logger.info("Stopping worker pool")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for workers to finish
        if self.workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.workers.values(), return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning("Worker shutdown timeout, cancelling remaining workers")
                
                # Force cancel remaining workers
                for worker_id, task in self.workers.items():
                    if not task.done():
                        task.cancel()
                        logger.debug(f"Cancelled worker {worker_id}")
                
                # Wait for cancellations
                try:
                    await asyncio.gather(*self.workers.values(), return_exceptions=True)
                except Exception:
                    pass  # Ignore cancellation exceptions
        
        self.workers.clear()
        self.worker_stats.clear()
        
        logger.info("Worker pool stopped")
    
    async def scale_workers(self, new_worker_count: int) -> None:
        """
        Scale the worker pool to a new size.
        
        Args:
            new_worker_count: Target number of workers
        """
        current_count = len(self.workers)
        
        if new_worker_count == current_count:
            return
        
        if new_worker_count > current_count:
            # Scale up - add workers
            for i in range(current_count, new_worker_count):
                await self._start_worker(f"worker-{i}")
            logger.info(f"Scaled up to {new_worker_count} workers")
            
        else:
            # Scale down - stop workers
            workers_to_stop = current_count - new_worker_count
            worker_ids = list(self.workers.keys())
            
            for i in range(workers_to_stop):
                worker_id = worker_ids[i]
                await self._stop_worker(worker_id)
            
            logger.info(f"Scaled down to {new_worker_count} workers")
        
        self.max_workers = new_worker_count
    
    async def _start_worker(self, worker_id: str) -> None:
        """Start a new worker."""
        if worker_id in self.workers:
            logger.warning(f"Worker {worker_id} already exists")
            return
        
        # Create worker task
        task = asyncio.create_task(self._worker_loop(worker_id))
        self.workers[worker_id] = task
        
        # Initialize worker stats
        self.worker_stats[worker_id] = {
            'started_at': datetime.now(),
            'jobs_processed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'last_job_at': None,
            'current_job': None,
            'health_status': 'healthy'
        }
        
        self.stats['workers_started'] += 1
        logger.debug(f"Started worker {worker_id}")
    
    async def _stop_worker(self, worker_id: str) -> None:
        """Stop a specific worker."""
        if worker_id not in self.workers:
            return
        
        task = self.workers[worker_id]
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        del self.workers[worker_id]
        del self.worker_stats[worker_id]
        
        self.stats['workers_stopped'] += 1
        logger.debug(f"Stopped worker {worker_id}")
    
    async def _worker_loop(self, worker_id: str) -> None:
        """Main worker processing loop."""
        logger.debug(f"Worker {worker_id} started")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get job from queue
                    job = await self.queue.get(timeout=5.0)
                    if not job:
                        continue  # Timeout, check shutdown event
                    
                    # Process the job
                    await self._process_job(worker_id, job)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    await asyncio.sleep(1.0)  # Brief pause before retrying
                    
        except asyncio.CancelledError:
            logger.debug(f"Worker {worker_id} cancelled")
            raise
        except Exception as e:
            logger.error(f"Worker {worker_id} crashed: {e}")
            
            # Restart worker if not shutting down
            if not self._shutdown_event.is_set():
                self.stats['worker_restarts'] += 1
                await asyncio.sleep(5.0)  # Brief pause before restart
                await self._start_worker(worker_id)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_job(self, worker_id: str, job: IngestionJob) -> None:
        """Process a single job with error handling and metrics."""
        start_time = time.time()
        
        try:
            # Update job and worker status
            job.start_processing(worker_id)
            await self.queue.update_job_status(job.job_id, JobStatus.PROCESSING)
            
            self.worker_stats[worker_id]['current_job'] = job.job_id
            
            # Process job with timeout
            success = await asyncio.wait_for(
                self.job_processor(job),
                timeout=self.worker_timeout
            )
            
            processing_time = time.time() - start_time
            
            if success:
                # Job completed successfully
                job.complete_successfully(
                    chunks_processed=job.chunks_processed or 0,
                    vectors_created=job.vectors_created or 0
                )
                await self.queue.update_job_status(
                    job.job_id, 
                    JobStatus.COMPLETED,
                    processing_time=processing_time
                )
                
                # Update statistics
                self.stats['jobs_processed'] += 1
                self.stats['total_processing_time'] += processing_time
                
                self.worker_stats[worker_id]['jobs_processed'] += 1
                self.worker_stats[worker_id]['total_processing_time'] += processing_time
                
                logger.info(
                    f"Worker {worker_id} completed job {job.job_id} "
                    f"for {job.library_id}:{job.version} in {processing_time:.2f}s"
                )
                
            else:
                # Job failed
                error_msg = "Job processor returned False"
                job.fail_with_error(error_msg)
                
                if job.status == JobStatus.FAILED:
                    await self.queue.update_job_status(job.job_id, JobStatus.FAILED)
                    self.stats['jobs_failed'] += 1
                else:
                    await self.queue.update_job_status(job.job_id, JobStatus.RETRYING)
                
                self.worker_stats[worker_id]['jobs_failed'] += 1
                
                logger.warning(f"Worker {worker_id} failed job {job.job_id}: {error_msg}")
            
        except asyncio.TimeoutError:
            # Job processing timeout
            error_msg = f"Job processing timeout ({self.worker_timeout}s)"
            job.fail_with_error(error_msg)
            
            if job.status == JobStatus.FAILED:
                await self.queue.update_job_status(job.job_id, JobStatus.FAILED)
                self.stats['jobs_failed'] += 1
            else:
                await self.queue.update_job_status(job.job_id, JobStatus.RETRYING)
            
            self.worker_stats[worker_id]['jobs_failed'] += 1
            
            logger.error(f"Worker {worker_id} timeout processing job {job.job_id}")
            
        except Exception as e:
            # Unexpected error
            error_msg = f"Unexpected error: {str(e)}"
            job.fail_with_error(error_msg)
            
            if job.status == JobStatus.FAILED:
                await self.queue.update_job_status(job.job_id, JobStatus.FAILED)
                self.stats['jobs_failed'] += 1
            else:
                await self.queue.update_job_status(job.job_id, JobStatus.RETRYING)
            
            self.worker_stats[worker_id]['jobs_failed'] += 1
            
            logger.error(f"Worker {worker_id} error processing job {job.job_id}: {e}")
            
        finally:
            # Update worker status
            self.worker_stats[worker_id]['current_job'] = None
            self.worker_stats[worker_id]['last_job_at'] = datetime.now()
            self.stats['last_activity'] = datetime.now()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive worker pool statistics."""
        active_workers = len(self.workers)
        busy_workers = sum(
            1 for stats in self.worker_stats.values()
            if stats['current_job'] is not None
        )
        
        # Calculate worker utilization
        worker_utilization = busy_workers / max(1, active_workers)
        
        # Calculate success rate
        total_jobs = self.stats['jobs_processed'] + self.stats['jobs_failed']
        success_rate = self.stats['jobs_processed'] / max(1, total_jobs)
        
        # Calculate average processing time
        avg_processing_time = 0.0
        if self.stats['jobs_processed'] > 0:
            avg_processing_time = self.stats['total_processing_time'] / self.stats['jobs_processed']
        
        return {
            'max_workers': self.max_workers,
            'active_workers': active_workers,
            'busy_workers': busy_workers,
            'worker_utilization': worker_utilization,
            'jobs_processed': self.stats['jobs_processed'],
            'jobs_failed': self.stats['jobs_failed'],
            'success_rate': success_rate,
            'total_processing_time': self.stats['total_processing_time'],
            'avg_processing_time': avg_processing_time,
            'workers_started': self.stats['workers_started'],
            'workers_stopped': self.stats['workers_stopped'],
            'worker_restarts': self.stats['worker_restarts'],
            'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None,
            'worker_details': {
                worker_id: {
                    **stats,
                    'started_at': stats['started_at'].isoformat(),
                    'last_job_at': stats['last_job_at'].isoformat() if stats['last_job_at'] else None
                }
                for worker_id, stats in self.worker_stats.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get worker pool health status."""
        active_workers = len(self.workers)
        healthy_workers = sum(
            1 for stats in self.worker_stats.values()
            if stats['health_status'] == 'healthy'
        )
        
        # Determine overall health
        if active_workers == 0:
            health_status = 'stopped'
        elif healthy_workers == active_workers:
            health_status = 'healthy'
        elif healthy_workers > active_workers * 0.5:
            health_status = 'degraded'
        else:
            health_status = 'unhealthy'
        
        return {
            'status': health_status,
            'active_workers': active_workers,
            'healthy_workers': healthy_workers,
            'max_workers': self.max_workers,
            'worker_health': {
                worker_id: stats['health_status']
                for worker_id, stats in self.worker_stats.items()
            }
        }