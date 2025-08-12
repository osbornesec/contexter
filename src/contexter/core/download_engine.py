"""
Async download engine with multi-context strategy and comprehensive error handling.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from ..integration.context7_client import Context7Client
from ..integration.proxy_manager import BrightDataProxyManager
from ..models.context7_models import (
    DocumentationResponse,
)
from ..models.download_models import (
    ConcurrentProcessingError,
    ContextGenerationError,
    DocumentationChunk,
    DownloadError,
    DownloadRequest,
    DownloadSummary,
    DownloadTask,
    ProgressMetrics,
)
from ..models.proxy_models import ProxyConnection
from .concurrent_processor import ConcurrentProcessor, JitterConfig
from .context_generator import ContextGenerator
from .content_parser import ContentParser, EntryDeduplicator, DocumentationEntry

logger = logging.getLogger(__name__)


class DownloadEngineError(Exception):
    """Base exception for download engine operations."""

    pass


class AsyncDownloadEngine:
    """
    Main download orchestration engine with multi-context strategy.

    Coordinates context generation, concurrent processing, proxy management,
    and error recovery to achieve comprehensive documentation retrieval.
    """

    def __init__(
        self,
        proxy_manager: Optional[BrightDataProxyManager] = None,
        context7_client: Optional[Context7Client] = None,
        max_concurrent: int = 10,
        max_retries: int = 3,
        enable_progress_tracking: bool = True,
    ):
        """
        Initialize async download engine.

        Args:
            proxy_manager: BrightData proxy manager for connection handling
            context7_client: Context7 API client for documentation retrieval
            max_concurrent: Maximum concurrent downloads (1-20)
            max_retries: Maximum retries per context (0-5)
            enable_progress_tracking: Enable detailed progress tracking
        """
        # Validate parameters
        if not (1 <= max_concurrent <= 20):
            raise ValueError(
                f"max_concurrent must be between 1 and 20, got {max_concurrent}"
            )

        if not (0 <= max_retries <= 5):
            raise ValueError(f"max_retries must be between 0 and 5, got {max_retries}")

        # Initialize components
        self.proxy_manager = proxy_manager
        self.context7_client = context7_client or Context7Client()
        self.max_retries = max_retries
        self.enable_progress_tracking = enable_progress_tracking

        # Initialize sub-components
        self.context_generator = ContextGenerator()
        self.content_parser = ContentParser()
        self.entry_deduplicator = EntryDeduplicator(similarity_threshold=0.8)

        # Configure concurrent processor with intelligent jitter
        jitter_config = JitterConfig(
            min_delay=0.5, max_delay=2.0, adaptive_enabled=True, burst_protection=True
        )

        self.concurrent_processor = ConcurrentProcessor[
            DownloadTask, DocumentationChunk
        ](
            max_concurrent=max_concurrent,
            jitter_config=jitter_config,
            enable_priority_scheduling=True,
            task_timeout=60.0,  # 60 second timeout per task
        )

        # State tracking
        self._active_downloads: Dict[str, asyncio.Task[Any]] = {}
        self._download_metrics: Dict[str, ProgressMetrics] = {}

        logger.info(
            f"Initialized AsyncDownloadEngine: max_concurrent={max_concurrent}, max_retries={max_retries}"
        )

    async def download_library(
        self,
        request: DownloadRequest,
        progress_callback: Optional[Callable[[ProgressMetrics], None]] = None,
    ) -> DownloadSummary:
        """
        Download complete documentation using multi-context strategy.

        Args:
            request: Download request with library ID and parameters
            progress_callback: Optional callback for progress updates

        Returns:
            Download summary with results and metrics

        Raises:
            DownloadEngineError: If download fails completely
            ContextGenerationError: If context generation fails
            ConcurrentProcessingError: If concurrent processing fails
        """
        start_time = datetime.now()
        library_id = request.library_id

        logger.info(f"Starting multi-context download for library: {library_id}")

        try:
            # Check library size first - for small libraries, bypass context generation
            library_info = await self._get_library_info(library_id)
            total_tokens = library_info.get('totalTokens', 0) if library_info else 0
            
            if total_tokens > 0 and total_tokens < 200_000:
                logger.info(f"Library {library_id} has {total_tokens:,} tokens - fetching all content in single request")
                # For small libraries, use a single comprehensive context
                request.contexts = ["comprehensive documentation and API reference"]
                # Set token limit to library size to get everything
                if not request.token_limit or request.token_limit > total_tokens:
                    request.token_limit = total_tokens
            else:
                # For large libraries, use exhaustive coverage strategy
                if total_tokens >= 200_000:
                    logger.info(f"Library {library_id} has {total_tokens:,} tokens - using exhaustive coverage strategy")
                    request = await self._apply_exhaustive_coverage_strategy(request, library_info)
                else:
                    # Generate contexts if not provided (for medium/unknown size libraries)
                    if not request.contexts:
                        logger.info(f"Generating contexts for {library_id} (size: {total_tokens:,} tokens)")
                        request.contexts = (
                            await self.context_generator.generate_contexts_with_validation(
                                library_id
                            )
                        )

            logger.info(f"Using {len(request.contexts)} contexts for {library_id}")

            # Initialize progress tracking
            if self.enable_progress_tracking:
                progress_metrics = ProgressMetrics(total_contexts=len(request.contexts))
                self._download_metrics[library_id] = progress_metrics
            else:
                progress_metrics = None

            # Create download tasks
            download_tasks = self._create_download_tasks(request)

            # Process tasks concurrently
            task_progress_callback = (
                self._create_task_progress_callback(progress_metrics, progress_callback)
                if progress_metrics
                else None
            )

            # Cast the results to the expected type
            raw_results = await self.concurrent_processor.process_with_concurrency(
                download_tasks,
                self._download_context_task,  # type: ignore
                task_progress_callback,
            )
            results: List[Union[DocumentationChunk, Exception]] = raw_results  # type: ignore

            # Process results and create summary
            summary = self._create_download_summary(
                library_id, download_tasks, results, start_time, datetime.now(), library_info
            )

            # Clean up tracking
            if library_id in self._download_metrics:
                del self._download_metrics[library_id]

            logger.info(
                f"Download completed for {library_id}: "
                f"{summary.successful_contexts}/{summary.total_contexts_attempted} successful, "
                f"{summary.total_tokens} tokens, "
                f"{summary.success_rate:.1f}% success rate"
            )

            return summary

        except Exception as e:
            logger.error(f"Download failed for {library_id}: {e}")

            # Clean up on failure
            if library_id in self._download_metrics:
                del self._download_metrics[library_id]

            # Create failure summary (currently commented for future use)
            # failure_summary = DownloadSummary(
            #     library_id=library_id,
            #     total_contexts_attempted=len(request.contexts) if request.contexts else 0,
            #     successful_contexts=0,
            #     failed_contexts=len(request.contexts) if request.contexts else 0,
            #     chunks=[],
            #     total_tokens=0,
            #     total_download_time=0.0,
            #     start_time=start_time,
            #     end_time=datetime.now(),
            #     error_summary={type(e).__name__: 1}
            # )

            # Re-raise with context
            if isinstance(e, (ContextGenerationError, ConcurrentProcessingError)):
                raise
            else:
                raise DownloadEngineError(
                    f"Download engine failed for {library_id}: {e}"
                ) from e

    def _create_download_tasks(self, request: DownloadRequest) -> List[DownloadTask]:
        """Create download tasks from request."""
        tasks = []

        for i, context in enumerate(request.contexts):
            task = DownloadTask(
                task_id=f"{request.library_id}_ctx_{i}_{int(time.time())}",
                library_id=request.library_id,
                context=context,
                token_limit=request.token_limit,
                priority=i,  # Earlier contexts have higher priority
                max_retries=request.retry_count,
            )

            tasks.append(task)

        logger.debug(f"Created {len(tasks)} download tasks for {request.library_id}")
        return tasks

    def _create_task_progress_callback(
        self,
        progress_metrics: ProgressMetrics,
        user_callback: Optional[Callable[[ProgressMetrics], None]],
    ) -> Callable[[Any], None]:
        """Create callback for task progress updates."""

        def task_progress_callback(processing_stats: Any) -> None:
            """Update progress metrics and call user callback."""
            try:
                # Update progress metrics based on processing stats
                progress_metrics.completed_contexts = processing_stats.completed_tasks
                progress_metrics.failed_contexts = processing_stats.failed_tasks
                progress_metrics.cancelled_contexts = processing_stats.cancelled_tasks

                # Call user callback if provided
                if user_callback:
                    user_callback(progress_metrics)

            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        return task_progress_callback

    async def _download_context_task(self, task: DownloadTask) -> DocumentationChunk:
        """
        Download documentation for single context with comprehensive error handling.

        Args:
            task: Download task to execute

        Returns:
            Documentation chunk with content and metadata

        Raises:
            Various exceptions for different failure modes
        """
        attempt = 0
        last_exception = None
        task.start()

        logger.debug(f"Starting download task: {task.task_id}")

        while attempt < task.max_retries:
            attempt += 1
            connection = None

            try:
                # Get proxy connection if proxy manager available
                if self.proxy_manager:
                    connection = await self._acquire_proxy_connection(task.priority)
                    if not connection:
                        logger.warning(
                            f"No proxy connection available for {task.task_id}"
                        )
                        # Continue with direct connection

                # Record proxy ID for tracking
                if connection:
                    task.proxy_id = connection.proxy_id
                    logger.debug(
                        f"Using proxy {connection.proxy_id} for {task.task_id}"
                    )

                # Make API request
                start_time = time.time()

                client_session = connection.session if connection else None
                response = await self.context7_client.get_smart_docs(
                    library_id=task.library_id,
                    context=task.context,
                    tokens=task.token_limit,
                    client=client_session,
                )

                download_time = time.time() - start_time

                # Report success to proxy manager
                if self.proxy_manager and connection:
                    await self._report_proxy_success(connection, download_time)

                # Create documentation chunk
                chunk = self._create_documentation_chunk(task, response, download_time)

                # Mark task as completed
                task.complete(chunk)

                logger.debug(
                    f"Successfully downloaded {task.task_id}: "
                    f"{chunk.token_count} tokens in {download_time:.2f}s"
                )

                return chunk

            except Exception as e:
                last_exception = e

                # Report failure to proxy manager
                if self.proxy_manager and connection:
                    await self._report_proxy_failure(connection, e)

                # Handle specific error types
                retry_info = await self._handle_download_error(e, task, attempt)

                if not retry_info.should_retry or attempt >= task.max_retries:
                    break

                # Wait before retry
                if retry_info.delay > 0:
                    logger.debug(f"Waiting {retry_info.delay:.1f}s before retry")
                    await asyncio.sleep(retry_info.delay)

        # All retries exhausted - mark task as failed
        error_msg = (
            f"Failed to download context after {attempt} attempts: {last_exception}"
        )
        task.fail(error_msg)

        logger.error(f"Task {task.task_id} failed: {error_msg}")
        raise DownloadError(error_msg, task_id=task.task_id)

    async def _acquire_proxy_connection(
        self, priority: int = 0
    ) -> Optional[ProxyConnection]:
        """Acquire proxy connection with error handling."""
        try:
            if not self.proxy_manager:
                return None

            connection = await self.proxy_manager.get_connection(priority=priority)
            if not connection:
                logger.warning("No healthy proxy connections available")
                return None

            return connection

        except Exception as e:
            logger.error(f"Failed to acquire proxy connection: {e}")
            return None

    async def _report_proxy_success(
        self, connection: ProxyConnection, response_time: float
    ) -> None:
        """Report successful proxy usage."""
        try:
            if self.proxy_manager and hasattr(self.proxy_manager, "report_success"):
                await self.proxy_manager.report_success(connection, response_time)
        except Exception as e:
            logger.warning(f"Failed to report proxy success: {e}")

    async def _report_proxy_failure(
        self, connection: ProxyConnection, error: Exception
    ) -> None:
        """Report proxy failure."""
        try:
            if self.proxy_manager and hasattr(self.proxy_manager, "report_failure"):
                await self.proxy_manager.report_failure(connection, error)
        except Exception as e:
            logger.warning(f"Failed to report proxy failure: {e}")

    def _create_documentation_chunk(
        self, task: DownloadTask, response: DocumentationResponse, download_time: float
    ) -> DocumentationChunk:
        """Create documentation chunk from API response with entry extraction."""
        
        # Parse content to extract individual entries
        try:
            entries = self.content_parser.parse_content(
                response.content, task.library_id, task.context
            )
            logger.debug(f"Extracted {len(entries)} entries from chunk")
            
            # Serialize entries for storage
            entries_data = [
                {
                    "entry_id": entry.entry_id,
                    "entry_type": entry.entry_type.value,
                    "title": entry.title,
                    "description": entry.description,
                    "content": entry.content,
                    "source_url": entry.source_url,
                    "language": entry.language,
                    "tags": list(entry.tags),
                    "content_hash": entry.content_hash,
                    "metadata": entry.metadata
                } for entry in entries
            ]
        except Exception as e:
            logger.warning(f"Failed to parse entries from chunk: {e}")
            entries_data = []

        chunk = DocumentationChunk(
            chunk_id=f"{task.library_id}_{self._hash_context(task.context)}_{int(time.time())}",
            content=response.content,
            source_context=task.context,
            token_count=response.token_count,
            content_hash=self._calculate_content_hash(response.content),
            proxy_id=task.proxy_id or "direct",
            download_time=download_time,
            library_id=task.library_id,
            metadata={
                "task_id": task.task_id,
                "attempt_count": task.retry_count + 1,
                "response_time": response.response_time,
                "api_metadata": response.metadata,
                "extracted_entries": entries_data,
                "entry_count": len(entries_data),
            },
        )

        return chunk

    def _hash_context(self, context: str) -> str:
        """Create hash of context for chunk ID generation."""
        hasher = hashlib.md5()
        hasher.update(context.encode("utf-8"))
        return hasher.hexdigest()[:8]

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate content hash for deduplication."""
        try:
            import xxhash

            hasher: Any = xxhash.xxh64()
        except ImportError:
            # Fallback to hashlib if xxhash not available
            import hashlib

            hasher = hashlib.md5()

        if hasattr(hasher, "update"):
            hasher.update(content.encode("utf-8"))
            return str(hasher.hexdigest())
        else:
            # For hashlib objects
            return str(hashlib.md5(content.encode("utf-8")).hexdigest())

    async def _handle_download_error(
        self, error: Exception, task: DownloadTask, attempt: int
    ) -> Any:
        """Handle download error and determine retry strategy."""
        from .error_classifier import ErrorClassifier

        # Use error classifier to determine retry strategy
        classifier = ErrorClassifier()
        retry_decision = await classifier.classify_and_decide_retry(
            error, task, attempt, self.max_retries
        )

        return retry_decision

    def _create_download_summary(
        self,
        library_id: str,
        tasks: List[DownloadTask],
        results: List[Union[DocumentationChunk, Exception]],
        start_time: datetime,
        end_time: datetime,
        library_info: Optional[Dict[str, Any]] = None,
    ) -> DownloadSummary:
        """Create comprehensive download summary from results."""

        # Separate successful and failed results
        successful_chunks = []
        failed_tasks = []
        error_counts: Dict[str, int] = {}

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_tasks.append(tasks[i])
                error_type = type(result).__name__
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            elif result is not None:
                successful_chunks.append(result)

        # Calculate metrics
        total_tokens = sum(chunk.token_count for chunk in successful_chunks)
        total_download_time = sum(chunk.download_time for chunk in successful_chunks)

        # Calculate coverage metrics
        expected_tokens = 0
        coverage_percentage = 0.0
        if library_info:
            expected_tokens = library_info.get('totalTokens', 0)
            if expected_tokens > 0:
                coverage_percentage = min((total_tokens / expected_tokens) * 100, 100.0)
                logger.info(f"Coverage: {total_tokens:,}/{expected_tokens:,} tokens ({coverage_percentage:.1f}%)")

        # Collect and deduplicate all entries across chunks
        all_entries_data = []
        for chunk in successful_chunks:
            chunk_entries = chunk.metadata.get('extracted_entries', [])
            all_entries_data.extend(chunk_entries)
        
        # Create entry objects for deduplication
        all_entries = []
        for entry_data in all_entries_data:
            try:
                from .content_parser import EntryType
                entry = DocumentationEntry(
                    entry_id=entry_data['entry_id'],
                    entry_type=EntryType(entry_data['entry_type']),
                    title=entry_data['title'],
                    description=entry_data['description'],
                    content=entry_data['content'],
                    source_url=entry_data.get('source_url'),
                    language=entry_data.get('language'),
                    tags=set(entry_data.get('tags', [])),
                    metadata=entry_data.get('metadata', {}),
                    content_hash=entry_data.get('content_hash', '')
                )
                all_entries.append(entry)
            except Exception as e:
                logger.warning(f"Failed to recreate entry from data: {e}")
        
        # Deduplicate entries
        if all_entries:
            unique_entries = self.entry_deduplicator.deduplicate_entries(all_entries)
            logger.info(f"Deduplicated entries: {len(all_entries)} -> {len(unique_entries)}")
        else:
            unique_entries = []

        summary = DownloadSummary(
            library_id=library_id,
            total_contexts_attempted=len(tasks),
            successful_contexts=len(successful_chunks),
            failed_contexts=len(failed_tasks),
            chunks=successful_chunks,
            total_tokens=total_tokens,
            total_download_time=total_download_time,
            start_time=start_time,
            end_time=end_time,
            error_summary=error_counts,
            expected_total_tokens=expected_tokens,
            coverage_percentage=coverage_percentage,
            unique_entries_count=len(unique_entries),
            total_entries_extracted=len(all_entries_data),
        )

        return summary

    async def _get_library_info(self, library_id: str) -> Optional[Dict[str, Any]]:
        """
        Get library information including token count to optimize download strategy.
        
        Args:
            library_id: Library identifier to get info for
            
        Returns:
            Dictionary with library info including totalTokens, or None if not found
        """
        try:
            # Search for the library to get its metadata
            search_results = await self.context7_client.resolve_library_id(library_id, limit=1)
            
            if search_results and len(search_results) > 0:
                result = search_results[0]
                # Extract token count from metadata
                api_response = result.metadata.get('api_response', {})
                total_tokens = api_response.get('totalTokens', 0)
                
                logger.debug(f"Library {library_id} has {total_tokens:,} total tokens")
                
                return {
                    'totalTokens': total_tokens,
                    'totalSnippets': api_response.get('totalSnippets', 0),
                    'totalPages': api_response.get('totalPages', 0),
                    'library_info': result
                }
            else:
                logger.warning(f"No library info found for {library_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to get library info for {library_id}: {e}")
            return None

    async def _apply_exhaustive_coverage_strategy(
        self, request: DownloadRequest, library_info: Optional[Dict[str, Any]]
    ) -> DownloadRequest:
        """
        Apply exhaustive coverage strategy for large libraries (>200k tokens).
        
        This strategy ensures comprehensive coverage by using multiple approaches:
        1. Base comprehensive contexts
        2. Structural/architectural contexts  
        3. Functional area contexts
        4. Advanced topic contexts
        
        Args:
            request: Original download request
            library_info: Library metadata including total tokens
            
        Returns:
            Modified request with exhaustive context list
        """
        library_id = request.library_id
        total_tokens = library_info.get('totalTokens', 0) if library_info else 0
        
        logger.info(f"Applying exhaustive coverage for {library_id} ({total_tokens:,} tokens)")
        
        # Generate comprehensive context set
        exhaustive_contexts = []
        
        # 1. Base comprehensive contexts (essential coverage)
        base_contexts = await self._generate_base_comprehensive_contexts(library_id)
        exhaustive_contexts.extend(base_contexts)
        
        # 2. Structural/architectural contexts
        structural_contexts = await self._generate_structural_contexts(library_id)
        exhaustive_contexts.extend(structural_contexts)
        
        # 3. Functional area contexts (based on library analysis)
        functional_contexts = await self._generate_functional_area_contexts(library_id, library_info)
        exhaustive_contexts.extend(functional_contexts)
        
        # 4. Advanced/specialized contexts
        advanced_contexts = await self._generate_advanced_contexts(library_id)
        exhaustive_contexts.extend(advanced_contexts)
        
        # 5. Remove duplicates while preserving order
        seen = set()
        unique_contexts = []
        for context in exhaustive_contexts:
            if context not in seen:
                seen.add(context)
                unique_contexts.append(context)
        
        # Set higher token limit per request for large libraries
        request.token_limit = min(50_000, total_tokens // len(unique_contexts)) if unique_contexts else 50_000
        request.contexts = unique_contexts
        
        logger.info(f"Generated {len(unique_contexts)} exhaustive contexts with {request.token_limit:,} tokens per context")
        
        return request
    
    async def _generate_base_comprehensive_contexts(self, library_id: str) -> List[str]:
        """Generate base comprehensive contexts for complete coverage."""
        lib_name = library_id.split("/")[-1]
        
        return [
            f"{lib_name} complete API documentation reference all methods functions classes",
            f"{lib_name} getting started tutorial installation setup configuration guide", 
            f"{lib_name} comprehensive examples code samples usage patterns",
            f"{lib_name} troubleshooting error handling debugging FAQ common issues",
            f"{lib_name} advanced configuration options parameters settings customization",
            f"{lib_name} best practices coding patterns design guidelines",
        ]
    
    async def _generate_structural_contexts(self, library_id: str) -> List[str]:
        """Generate contexts focused on library structure and architecture."""
        lib_name = library_id.split("/")[-1]
        
        return [
            f"{lib_name} architecture design patterns internal structure",
            f"{lib_name} modules packages organization codebase structure",
            f"{lib_name} core classes objects interfaces inheritance hierarchy", 
            f"{lib_name} data models schemas types validation serialization",
            f"{lib_name} configuration management settings environment variables",
            f"{lib_name} plugin system extensions hooks callbacks",
        ]
    
    async def _generate_functional_area_contexts(
        self, library_id: str, library_info: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate contexts for specific functional areas based on library analysis."""
        lib_name = library_id.split("/")[-1]
        
        # Get library metadata for smarter context generation
        total_snippets = library_info.get('totalSnippets', 0) if library_info else 0
        total_pages = library_info.get('totalPages', 0) if library_info else 0
        
        contexts = []
        
        # Base functional areas that most libraries have
        contexts.extend([
            f"{lib_name} authentication authorization security implementation",
            f"{lib_name} performance optimization benchmarks tuning scaling",
            f"{lib_name} testing unit tests integration tests mocking fixtures",
            f"{lib_name} deployment production setup monitoring logging", 
            f"{lib_name} migration upgrade compatibility changelog breaking changes",
        ])
        
        # Add more contexts for very large libraries (lots of content)
        if total_snippets > 500:
            contexts.extend([
                f"{lib_name} networking HTTP requests clients protocols",
                f"{lib_name} data processing transformation validation parsing",
                f"{lib_name} database integration ORM queries transactions", 
                f"{lib_name} async asynchronous programming concurrency threading",
                f"{lib_name} caching memory management optimization strategies",
            ])
        
        # Add even more for massive libraries 
        if total_snippets > 1000:
            contexts.extend([
                f"{lib_name} command line interface CLI tools scripts",
                f"{lib_name} web framework routing templates middleware",
                f"{lib_name} file system operations directory management",
                f"{lib_name} serialization JSON XML formats encoding decoding", 
                f"{lib_name} third party integrations external services APIs",
            ])
            
        return contexts
    
    async def _generate_advanced_contexts(self, library_id: str) -> List[str]:
        """Generate advanced/specialized contexts for complete coverage.""" 
        lib_name = library_id.split("/")[-1]
        
        return [
            f"{lib_name} internals implementation details source code analysis",
            f"{lib_name} edge cases corner cases error scenarios failure handling",
            f"{lib_name} community resources plugins third party extensions",
            f"{lib_name} comparison alternatives similar libraries differences",
            f"{lib_name} migration guide upgrade path version differences",
            f"{lib_name} contributors development setup building from source",
        ]

    async def download_multiple_libraries(
        self,
        requests: List[DownloadRequest],
        max_concurrent_libraries: int = 3,
        progress_callback: Optional[Callable[[str, ProgressMetrics], None]] = None,
    ) -> Dict[str, DownloadSummary]:
        """
        Download multiple libraries concurrently.

        Args:
            requests: List of download requests
            max_concurrent_libraries: Maximum concurrent library downloads
            progress_callback: Optional callback for progress updates (library_id, metrics)

        Returns:
            Dictionary mapping library IDs to download summaries
        """
        if not requests:
            return {}

        logger.info(f"Starting batch download of {len(requests)} libraries")

        # Create semaphore for library-level concurrency
        library_semaphore = asyncio.Semaphore(max_concurrent_libraries)

        async def download_with_semaphore(
            request: DownloadRequest,
        ) -> tuple[str, DownloadSummary]:
            """Download single library with semaphore control."""
            async with library_semaphore:
                # Create per-library progress callback
                library_progress_callback = None
                if progress_callback:

                    def library_progress_callback(
                        metrics: ProgressMetrics, lib_id: str = request.library_id
                    ) -> Any:
                        return progress_callback(lib_id, metrics)

                summary = await self.download_library(
                    request, library_progress_callback
                )
                return request.library_id, summary

        # Execute downloads concurrently
        try:
            results = await asyncio.gather(
                *[download_with_semaphore(request) for request in requests],
                return_exceptions=True,
            )

            # Process results
            summaries = {}
            failed_count = 0

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    library_id = requests[i].library_id
                    logger.error(f"Batch download failed for {library_id}: {result}")

                    # Create failure summary
                    summaries[library_id] = DownloadSummary(
                        library_id=library_id,
                        total_contexts_attempted=0,
                        successful_contexts=0,
                        failed_contexts=1,
                        chunks=[],
                        total_tokens=0,
                        total_download_time=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_summary={type(result).__name__: 1},
                    )
                else:
                    # result should be tuple[str, DownloadSummary]
                    if isinstance(result, tuple) and len(result) == 2:
                        library_id, summary = result
                        summaries[library_id] = summary
                    else:
                        logger.error(f"Unexpected result type: {type(result)}")
                        failed_count += 1

            successful_count = len(summaries) - failed_count
            logger.info(
                f"Batch download completed: {successful_count}/{len(requests)} libraries successful"
            )

            return summaries

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            raise DownloadEngineError(f"Batch download failed: {e}") from e

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of download engine.

        Returns:
            Dictionary with health status and component information
        """
        health_info: Dict[str, Any] = {
            "engine_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "metrics": {},
        }
        components = health_info["components"]
        metrics = health_info["metrics"]

        try:
            # Check Context7 client
            context7_health = await self.context7_client.health_check()
            components["context7_client"] = {
                "status": "healthy" if context7_health else "unhealthy",
                "reachable": context7_health,
            }

            # Check proxy manager if available
            if self.proxy_manager:
                # Simple proxy pool check
                proxy_pool = getattr(self.proxy_manager, "proxy_pool", [])
                proxy_status = len(proxy_pool) > 0
                components["proxy_manager"] = {
                    "status": "healthy" if proxy_status else "degraded",
                    "proxy_pool_size": len(proxy_pool),
                }
            else:
                components["proxy_manager"] = {
                    "status": "not_configured",
                    "proxy_pool_size": 0,
                }

            # Check concurrent processor
            components["concurrent_processor"] = {
                "status": (
                    "healthy" if self.concurrent_processor.can_process else "busy"
                ),
                "state": self.concurrent_processor.state.value,
                "max_concurrent": self.concurrent_processor.max_concurrent,
            }

            # Add current metrics
            metrics.update(
                {
                    "active_downloads": len(self._active_downloads),
                    "tracked_libraries": len(self._download_metrics),
                }
            )

            # Overall health assessment
            component_statuses = [comp["status"] for comp in components.values()]

            if "unhealthy" in component_statuses:
                health_info["engine_status"] = "degraded"
            elif "degraded" in component_statuses:
                health_info["engine_status"] = "degraded"

        except Exception as e:
            health_info["engine_status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"Health check failed: {e}")

        return health_info

    async def shutdown(self, timeout: float = 60.0) -> None:
        """
        Gracefully shutdown the download engine.

        Args:
            timeout: Maximum time to wait for active downloads to complete
        """
        logger.info("Initiating download engine shutdown")

        try:
            # Cancel active downloads
            if self._active_downloads:
                logger.info(
                    f"Cancelling {len(self._active_downloads)} active downloads"
                )
                for _, task in self._active_downloads.items():
                    task.cancel()

                # Wait for cancellation to complete
                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *self._active_downloads.values(), return_exceptions=True
                        ),
                        timeout=min(timeout, 10.0),
                    )
                except asyncio.TimeoutError:
                    logger.warning("Download cancellation timed out")

            # Shutdown concurrent processor
            await self.concurrent_processor.shutdown(timeout=timeout - 10.0)

            # Clean up tracking
            self._active_downloads.clear()
            self._download_metrics.clear()

            logger.info("Download engine shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
