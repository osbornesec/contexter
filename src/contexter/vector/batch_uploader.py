"""
Batch Uploader - High-performance batch vector operations.

Provides optimized batch upload capabilities with:
- Parallel processing with semaphore control
- Error handling and retry logic
- Progress tracking and detailed reporting
- Memory-efficient processing for large batches
- Conflict resolution and deduplication
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, AsyncGenerator, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .qdrant_vector_store import QdrantVectorStore, VectorDocument

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    batch_id: str
    status: BatchStatus
    total_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    errors: List[str]
    metadata: Dict[str, Any]


class BatchConfig(BaseModel):
    """Configuration for batch operations."""

    batch_size: int = Field(default=1000, description="Number of items per batch")
    max_concurrent_batches: int = Field(default=5, description="Max concurrent batch operations")
    retry_attempts: int = Field(default=3, description="Number of retry attempts for failed items")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    memory_limit_mb: int = Field(default=512, description="Memory limit for batch processing")
    enable_deduplication: bool = Field(default=True, description="Enable duplicate detection")
    conflict_resolution: str = Field(default="overwrite", description="How to handle conflicts")
    progress_callback_interval: int = Field(default=100, description="Progress callback frequency")


class BatchProgress(BaseModel):
    """Batch processing progress information."""

    batch_id: str
    total_batches: int
    completed_batches: int
    current_batch_items: int
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_operation: str = ""


class BatchUploader:
    """
    High-performance batch uploader for vector documents.
    
    Features:
    - Memory-efficient processing of large document collections
    - Parallel batch processing with semaphore control
    - Comprehensive error handling and retry logic
    - Progress tracking with customizable callbacks
    - Duplicate detection and conflict resolution
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        config: Optional[BatchConfig] = None
    ):
        """
        Initialize the batch uploader.
        
        Args:
            vector_store: Qdrant vector store instance
            config: Batch processing configuration
        """
        self.vector_store = vector_store
        self.config = config or BatchConfig()

        # Semaphore to control concurrent batch operations
        self._batch_semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

        # Progress tracking
        self._active_uploads: Dict[str, BatchProgress] = {}
        self._upload_history: List[BatchResult] = []

        # Performance metrics
        self._metrics = {
            "total_uploads": 0,
            "total_documents": 0,
            "total_successful": 0,
            "total_failed": 0,
            "avg_batch_time": 0.0,
            "avg_throughput_docs_per_sec": 0.0,
            "last_upload_time": None
        }

    async def upload_documents(
        self,
        documents: List[VectorDocument],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        batch_id: Optional[str] = None
    ) -> BatchResult:
        """
        Upload a collection of documents in optimized batches.
        
        Args:
            documents: List of vector documents to upload
            progress_callback: Optional callback for progress updates
            batch_id: Optional custom batch ID
            
        Returns:
            BatchResult with processing summary
        """
        if not documents:
            return BatchResult(
                batch_id="empty",
                status=BatchStatus.COMPLETED,
                total_items=0,
                successful_items=0,
                failed_items=0,
                processing_time=0.0,
                errors=[],
                metadata={}
            )

        upload_id = batch_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting batch upload {upload_id} with {len(documents)} documents")

        try:
            # Initialize progress tracking
            progress = BatchProgress(
                batch_id=upload_id,
                total_batches=0,
                completed_batches=0,
                current_batch_items=0,
                total_items=len(documents),
                processed_items=0,
                successful_items=0,
                failed_items=0,
                start_time=datetime.now(),
                current_operation="Initializing"
            )

            self._active_uploads[upload_id] = progress

            # Deduplicate documents if enabled
            if self.config.enable_deduplication:
                progress.current_operation = "Deduplicating documents"
                if progress_callback:
                    progress_callback(progress)

                documents = await self._deduplicate_documents(documents)
                progress.total_items = len(documents)

            # Split into batches
            batches = self._split_into_batches(documents)
            progress.total_batches = len(batches)

            # Process batches
            progress.current_operation = "Processing batches"
            results = await self._process_batches(batches, progress, progress_callback)

            # Aggregate results
            total_successful = sum(r.successful_items for r in results)
            total_failed = sum(r.failed_items for r in results)
            all_errors = [error for r in results for error in r.errors]

            processing_time = time.time() - start_time

            # Determine final status
            if total_failed == 0:
                status = BatchStatus.COMPLETED
            elif total_successful == 0:
                status = BatchStatus.FAILED
            else:
                status = BatchStatus.PARTIAL

            # Create final result
            final_result = BatchResult(
                batch_id=upload_id,
                status=status,
                total_items=len(documents),
                successful_items=total_successful,
                failed_items=total_failed,
                processing_time=processing_time,
                errors=all_errors,
                metadata={
                    "batches_processed": len(batches),
                    "avg_batch_size": len(documents) / len(batches) if batches else 0,
                    "throughput_docs_per_sec": len(documents) / processing_time if processing_time > 0 else 0
                }
            )

            # Update metrics
            self._update_metrics(final_result)

            # Cleanup
            del self._active_uploads[upload_id]
            self._upload_history.append(final_result)

            logger.info(
                f"Batch upload {upload_id} completed: {total_successful} successful, "
                f"{total_failed} failed in {processing_time:.2f}s"
            )

            return final_result

        except Exception as e:
            logger.error(f"Batch upload {upload_id} failed: {e}")

            # Create error result
            error_result = BatchResult(
                batch_id=upload_id,
                status=BatchStatus.FAILED,
                total_items=len(documents),
                successful_items=0,
                failed_items=len(documents),
                processing_time=time.time() - start_time,
                errors=[str(e)],
                metadata={}
            )

            # Cleanup
            if upload_id in self._active_uploads:
                del self._active_uploads[upload_id]
            self._upload_history.append(error_result)

            raise

    async def upload_documents_stream(
        self,
        document_stream: AsyncGenerator[VectorDocument, None],
        progress_callback: Optional[Callable[[BatchProgress], None]] = None,
        batch_id: Optional[str] = None
    ) -> BatchResult:
        """
        Upload documents from an async stream for memory efficiency.
        
        Args:
            document_stream: Async generator of vector documents
            progress_callback: Optional callback for progress updates
            batch_id: Optional custom batch ID
            
        Returns:
            BatchResult with processing summary
        """
        upload_id = batch_id or str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Starting streaming batch upload {upload_id}")

        try:
            # Initialize progress tracking
            progress = BatchProgress(
                batch_id=upload_id,
                total_batches=0,
                completed_batches=0,
                current_batch_items=0,
                total_items=0,  # Unknown for streams
                processed_items=0,
                successful_items=0,
                failed_items=0,
                start_time=datetime.now(),
                current_operation="Streaming upload"
            )

            self._active_uploads[upload_id] = progress

            # Process stream in batches
            batch_results = []
            current_batch = []
            batch_number = 0

            async for document in document_stream:
                current_batch.append(document)
                progress.total_items += 1

                # Process batch when full
                if len(current_batch) >= self.config.batch_size:
                    batch_number += 1
                    progress.total_batches = batch_number
                    progress.current_operation = f"Processing batch {batch_number}"

                    if progress_callback:
                        progress_callback(progress)

                    # Process the batch
                    batch_result = await self._process_single_batch(
                        current_batch, f"{upload_id}_batch_{batch_number}"
                    )
                    batch_results.append(batch_result)

                    # Update progress
                    progress.processed_items += len(current_batch)
                    progress.successful_items += batch_result.successful_items
                    progress.failed_items += batch_result.failed_items
                    progress.completed_batches += 1

                    # Clear batch
                    current_batch = []

            # Process remaining documents
            if current_batch:
                batch_number += 1
                progress.total_batches = batch_number
                progress.current_operation = f"Processing final batch {batch_number}"

                if progress_callback:
                    progress_callback(progress)

                batch_result = await self._process_single_batch(
                    current_batch, f"{upload_id}_batch_{batch_number}"
                )
                batch_results.append(batch_result)

                progress.processed_items += len(current_batch)
                progress.successful_items += batch_result.successful_items
                progress.failed_items += batch_result.failed_items
                progress.completed_batches += 1

            # Aggregate results
            total_successful = sum(r.successful_items for r in batch_results)
            total_failed = sum(r.failed_items for r in batch_results)
            all_errors = [error for r in batch_results for error in r.errors]

            processing_time = time.time() - start_time

            # Determine final status
            if total_failed == 0:
                status = BatchStatus.COMPLETED
            elif total_successful == 0:
                status = BatchStatus.FAILED
            else:
                status = BatchStatus.PARTIAL

            # Create final result
            final_result = BatchResult(
                batch_id=upload_id,
                status=status,
                total_items=progress.total_items,
                successful_items=total_successful,
                failed_items=total_failed,
                processing_time=processing_time,
                errors=all_errors,
                metadata={
                    "batches_processed": len(batch_results),
                    "stream_processing": True,
                    "throughput_docs_per_sec": progress.total_items / processing_time if processing_time > 0 else 0
                }
            )

            # Update metrics and cleanup
            self._update_metrics(final_result)
            del self._active_uploads[upload_id]
            self._upload_history.append(final_result)

            logger.info(
                f"Streaming upload {upload_id} completed: {total_successful} successful, "
                f"{total_failed} failed in {processing_time:.2f}s"
            )

            return final_result

        except Exception as e:
            logger.error(f"Streaming upload {upload_id} failed: {e}")

            # Cleanup and create error result
            if upload_id in self._active_uploads:
                progress = self._active_uploads[upload_id]
                del self._active_uploads[upload_id]
            else:
                progress = BatchProgress(
                    batch_id=upload_id,
                    total_batches=0,
                    completed_batches=0,
                    current_batch_items=0,
                    total_items=0,
                    processed_items=0,
                    successful_items=0,
                    failed_items=0,
                    start_time=datetime.now()
                )

            error_result = BatchResult(
                batch_id=upload_id,
                status=BatchStatus.FAILED,
                total_items=progress.total_items,
                successful_items=progress.successful_items,
                failed_items=progress.total_items - progress.successful_items,
                processing_time=time.time() - start_time,
                errors=[str(e)],
                metadata={"stream_processing": True}
            )

            self._upload_history.append(error_result)
            raise

    def _split_into_batches(self, documents: List[VectorDocument]) -> List[List[VectorDocument]]:
        """Split documents into optimally-sized batches."""
        batches = []

        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]
            batches.append(batch)

        return batches

    async def _process_batches(
        self,
        batches: List[List[VectorDocument]],
        progress: BatchProgress,
        progress_callback: Optional[Callable[[BatchProgress], None]]
    ) -> List[BatchResult]:
        """Process multiple batches with concurrency control."""
        batch_tasks = []

        for i, batch in enumerate(batches):
            batch_id = f"{progress.batch_id}_batch_{i+1}"
            task = self._process_single_batch_with_progress(
                batch, batch_id, progress, progress_callback
            )
            batch_tasks.append(task)

        # Execute batches with controlled concurrency
        results = []
        for task in asyncio.as_completed(batch_tasks):
            result = await task
            results.append(result)

        return results

    async def _process_single_batch_with_progress(
        self,
        batch: List[VectorDocument],
        batch_id: str,
        overall_progress: BatchProgress,
        progress_callback: Optional[Callable[[BatchProgress], None]]
    ) -> BatchResult:
        """Process a single batch and update overall progress."""
        async with self._batch_semaphore:
            result = await self._process_single_batch(batch, batch_id)

            # Update overall progress
            overall_progress.completed_batches += 1
            overall_progress.processed_items += result.total_items
            overall_progress.successful_items += result.successful_items
            overall_progress.failed_items += result.failed_items
            overall_progress.current_operation = f"Completed batch {overall_progress.completed_batches}/{overall_progress.total_batches}"

            # Call progress callback if provided
            if progress_callback and overall_progress.completed_batches % self.config.progress_callback_interval == 0:
                progress_callback(overall_progress)

            return result

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _process_single_batch(
        self,
        batch: List[VectorDocument],
        batch_id: str
    ) -> BatchResult:
        """Process a single batch with retry logic."""
        start_time = time.time()

        try:
            # Upload batch to vector store
            upload_result = await self.vector_store.upsert_vectors_batch(batch)

            processing_time = time.time() - start_time

            # Create result
            result = BatchResult(
                batch_id=batch_id,
                status=BatchStatus.COMPLETED if upload_result["failed_uploads"] == 0 else BatchStatus.PARTIAL,
                total_items=len(batch),
                successful_items=upload_result["successful_uploads"],
                failed_items=upload_result["failed_uploads"],
                processing_time=processing_time,
                errors=[],
                metadata={
                    "upload_result": upload_result,
                    "documents_per_second": len(batch) / processing_time if processing_time > 0 else 0
                }
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Batch {batch_id} failed: {e}")

            return BatchResult(
                batch_id=batch_id,
                status=BatchStatus.FAILED,
                total_items=len(batch),
                successful_items=0,
                failed_items=len(batch),
                processing_time=processing_time,
                errors=[str(e)],
                metadata={}
            )

    async def _deduplicate_documents(self, documents: List[VectorDocument]) -> List[VectorDocument]:
        """Remove duplicate documents based on ID."""
        seen_ids = set()
        deduplicated = []

        for doc in documents:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                deduplicated.append(doc)

        if len(deduplicated) < len(documents):
            logger.info(f"Deduplicated {len(documents) - len(deduplicated)} documents")

        return deduplicated

    def _update_metrics(self, result: BatchResult) -> None:
        """Update performance metrics with batch result."""
        self._metrics["total_uploads"] += 1
        self._metrics["total_documents"] += result.total_items
        self._metrics["total_successful"] += result.successful_items
        self._metrics["total_failed"] += result.failed_items

        # Update average batch time
        total_uploads = self._metrics["total_uploads"]
        current_avg = self._metrics["avg_batch_time"]
        self._metrics["avg_batch_time"] = (
            (current_avg * (total_uploads - 1) + result.processing_time) / total_uploads
        )

        # Update average throughput
        if result.processing_time > 0:
            throughput = result.total_items / result.processing_time
            current_avg_throughput = self._metrics["avg_throughput_docs_per_sec"]
            self._metrics["avg_throughput_docs_per_sec"] = (
                (current_avg_throughput * (total_uploads - 1) + throughput) / total_uploads
            )

        self._metrics["last_upload_time"] = datetime.now().isoformat()

    def get_upload_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get current progress for an active upload."""
        return self._active_uploads.get(batch_id)

    def get_upload_history(self, limit: int = 10) -> List[BatchResult]:
        """Get recent upload history."""
        return self._upload_history[-limit:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get upload performance metrics."""
        success_rate = (
            self._metrics["total_successful"] /
            max(self._metrics["total_documents"], 1)
        ) if self._metrics["total_documents"] > 0 else 0.0

        return {
            "upload_metrics": self._metrics.copy(),
            "success_rate": success_rate,
            "active_uploads": len(self._active_uploads),
            "config": {
                "batch_size": self.config.batch_size,
                "max_concurrent_batches": self.config.max_concurrent_batches,
                "retry_attempts": self.config.retry_attempts
            }
        }

    def clear_history(self) -> None:
        """Clear upload history to free memory."""
        self._upload_history.clear()
        logger.info("Upload history cleared")
