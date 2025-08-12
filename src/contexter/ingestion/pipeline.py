"""
Ingestion Pipeline - Unified document processing orchestrator.

Main pipeline class that orchestrates the complete document ingestion
process from trigger to vector storage with comprehensive monitoring
and error handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

from .trigger_system import AutoIngestionTrigger, IngestionTriggerEvent
from .processing_queue import IngestionQueue, IngestionJob, WorkerPool, JobStatus
from .json_parser import JSONDocumentParser, ParsedSection
from .chunking_engine import IntelligentChunkingEngine, DocumentChunk
from .metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single document."""
    job_id: str
    library_id: str
    version: str
    success: bool
    
    # Processing metrics
    processing_time: float
    sections_parsed: int
    chunks_created: int
    vectors_generated: int
    
    # Quality metrics
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    
    # Error information
    error_message: Optional[str] = None
    failed_stage: Optional[str] = None
    
    # Detailed results
    parsed_sections: List[ParsedSection] = field(default_factory=list)
    document_chunks: List[DocumentChunk] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'job_id': self.job_id,
            'library_id': self.library_id,
            'version': self.version,
            'success': self.success,
            'processing_time': self.processing_time,
            'sections_parsed': self.sections_parsed,
            'chunks_created': self.chunks_created,
            'vectors_generated': self.vectors_generated,
            'avg_quality_score': self.avg_quality_score,
            'min_quality_score': self.min_quality_score,
            'max_quality_score': self.max_quality_score,
            'error_message': self.error_message,
            'failed_stage': self.failed_stage
        }


@dataclass
class IngestionStatistics:
    """Comprehensive ingestion pipeline statistics."""
    
    # Overall statistics
    total_jobs_processed: int = 0
    total_jobs_successful: int = 0
    total_jobs_failed: int = 0
    total_processing_time: float = 0.0
    
    # Document statistics
    total_documents_parsed: int = 0
    total_sections_extracted: int = 0
    total_chunks_created: int = 0
    total_vectors_generated: int = 0
    
    # Quality statistics
    avg_quality_score: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    
    # Performance statistics
    avg_processing_time_per_job: float = 0.0
    avg_processing_time_per_chunk: float = 0.0
    throughput_per_minute: float = 0.0
    
    # Error statistics
    error_counts_by_stage: Dict[str, int] = field(default_factory=dict)
    error_counts_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Time-based statistics
    start_time: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def update_from_result(self, result: ProcessingResult):
        """Update statistics from a processing result."""
        self.total_jobs_processed += 1
        self.total_processing_time += result.processing_time
        self.last_activity = datetime.now()
        
        if result.success:
            self.total_jobs_successful += 1
            self.total_documents_parsed += 1
            self.total_sections_extracted += result.sections_parsed
            self.total_chunks_created += result.chunks_created
            self.total_vectors_generated += result.vectors_generated
            
            # Update quality statistics
            if result.avg_quality_score > 0:
                self.quality_scores.append(result.avg_quality_score)
                self.avg_quality_score = sum(self.quality_scores) / len(self.quality_scores)
        else:
            self.total_jobs_failed += 1
            
            # Update error statistics
            if result.failed_stage:
                self.error_counts_by_stage[result.failed_stage] = (
                    self.error_counts_by_stage.get(result.failed_stage, 0) + 1
                )
            
            if result.error_message:
                error_type = result.error_message.split(':')[0] if ':' in result.error_message else 'unknown'
                self.error_counts_by_type[error_type] = (
                    self.error_counts_by_type.get(error_type, 0) + 1
                )
        
        # Update derived statistics
        if self.total_jobs_processed > 0:
            self.avg_processing_time_per_job = self.total_processing_time / self.total_jobs_processed
        
        if self.total_chunks_created > 0:
            self.avg_processing_time_per_chunk = self.total_processing_time / self.total_chunks_created
        
        # Calculate throughput
        if self.start_time:
            elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
            if elapsed_minutes > 0:
                self.throughput_per_minute = self.total_jobs_processed / elapsed_minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'overall': {
                'total_jobs_processed': self.total_jobs_processed,
                'total_jobs_successful': self.total_jobs_successful,
                'total_jobs_failed': self.total_jobs_failed,
                'success_rate': (
                    self.total_jobs_successful / max(1, self.total_jobs_processed)
                ),
                'total_processing_time': self.total_processing_time
            },
            'documents': {
                'total_documents_parsed': self.total_documents_parsed,
                'total_sections_extracted': self.total_sections_extracted,
                'total_chunks_created': self.total_chunks_created,
                'total_vectors_generated': self.total_vectors_generated,
                'avg_sections_per_document': (
                    self.total_sections_extracted / max(1, self.total_documents_parsed)
                ),
                'avg_chunks_per_document': (
                    self.total_chunks_created / max(1, self.total_documents_parsed)
                )
            },
            'quality': {
                'avg_quality_score': self.avg_quality_score,
                'quality_distribution': {
                    'excellent': sum(1 for s in self.quality_scores if s >= 0.9),
                    'good': sum(1 for s in self.quality_scores if 0.7 <= s < 0.9),
                    'fair': sum(1 for s in self.quality_scores if 0.5 <= s < 0.7),
                    'poor': sum(1 for s in self.quality_scores if s < 0.5)
                }
            },
            'performance': {
                'avg_processing_time_per_job': self.avg_processing_time_per_job,
                'avg_processing_time_per_chunk': self.avg_processing_time_per_chunk,
                'throughput_per_minute': self.throughput_per_minute
            },
            'errors': {
                'error_counts_by_stage': self.error_counts_by_stage,
                'error_counts_by_type': self.error_counts_by_type
            },
            'timeline': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'last_activity': self.last_activity.isoformat() if self.last_activity else None
            }
        }


class IngestionPipeline:
    """
    Unified document ingestion pipeline.
    
    Orchestrates the complete document ingestion process with:
    - Auto-trigger system integration
    - Priority-based job processing
    - Multi-stage document processing
    - Vector generation and storage
    - Comprehensive monitoring and statistics
    """
    
    def __init__(
        self,
        storage_manager: 'StorageManager',
        embedding_engine: 'EmbeddingEngine',
        vector_storage: 'VectorStorage',
        max_workers: int = 5,
        quality_threshold: float = 0.7,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            storage_manager: Storage system interface
            embedding_engine: Embedding generation service
            vector_storage: Vector database interface
            max_workers: Maximum concurrent workers
            quality_threshold: Minimum quality score for processing
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap size for context preservation
        """
        self.storage_manager = storage_manager
        self.embedding_engine = embedding_engine
        self.vector_storage = vector_storage
        self.max_workers = max_workers
        self.quality_threshold = quality_threshold
        
        # Initialize pipeline components
        self.trigger_system = AutoIngestionTrigger(
            storage_manager=storage_manager,
            ingestion_pipeline=self,
            quality_threshold=quality_threshold
        )
        
        self.job_queue = IngestionQueue(max_size=10000)
        
        self.worker_pool = WorkerPool(
            max_workers=max_workers,
            job_processor=self._process_job,
            queue=self.job_queue
        )
        
        # Processing components
        self.json_parser = JSONDocumentParser()
        self.chunking_engine = IntelligentChunkingEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.metadata_extractor = MetadataExtractor()
        
        # Statistics and monitoring
        self.statistics = IngestionStatistics()
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        
        # Processing results cache
        self.recent_results: List[ProcessingResult] = []
        self.max_cached_results = 100
        
        logger.info(f"Ingestion pipeline initialized with {max_workers} workers")
    
    async def initialize(self):
        """Initialize the ingestion pipeline."""
        if self._initialized:
            return
        
        logger.info("Initializing ingestion pipeline")
        
        try:
            # Initialize external dependencies
            await self.storage_manager.initialize()
            await self.embedding_engine.initialize()
            await self.vector_storage.initialize()
            
            # Start pipeline components
            await self.trigger_system.start_processing()
            await self.worker_pool.start()
            
            # Initialize statistics
            self.statistics.start_time = datetime.now()
            
            self._initialized = True
            logger.info("Ingestion pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ingestion pipeline: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown the ingestion pipeline gracefully."""
        logger.info("Shutting down ingestion pipeline")
        
        self._shutdown_event.set()
        
        # Stop pipeline components
        if hasattr(self, 'worker_pool'):
            await self.worker_pool.stop()
        
        if hasattr(self, 'trigger_system'):
            await self.trigger_system.stop_processing()
        
        # Cleanup external dependencies
        if hasattr(self, 'storage_manager'):
            await self.storage_manager.cleanup()
        
        if hasattr(self, 'embedding_engine'):
            await self.embedding_engine.shutdown()
        
        if hasattr(self, 'vector_storage'):
            await self.vector_storage.cleanup()
        
        self._initialized = False
        logger.info("Ingestion pipeline shutdown complete")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
    
    async def queue_document(
        self,
        library_id: str,
        version: str,
        doc_path: Path,
        priority: int,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Queue a document for ingestion processing.
        
        Args:
            library_id: Library identifier
            version: Version string
            doc_path: Path to documentation file
            priority: Processing priority (0-15)
            metadata: Document metadata
            
        Returns:
            Job ID for tracking
        """
        if not self._initialized:
            await self.initialize()
        
        job_id = await self.job_queue.put(
            library_id=library_id,
            version=version,
            doc_path=doc_path,
            priority=priority,
            metadata=metadata
        )
        
        logger.info(f"Queued document {library_id}:{version} for ingestion (job: {job_id})")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific ingestion job."""
        return await self.job_queue.get_job_status(job_id)
    
    async def wait_for_job_completion(
        self, 
        job_id: str, 
        timeout: float = 300.0
    ) -> Optional[ProcessingResult]:
        """
        Wait for a job to complete and return results.
        
        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            Processing result or None if timeout/not found
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            job_status = await self.get_job_status(job_id)
            
            if not job_status:
                return None
            
            if job_status['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                # Look for result in recent results cache
                for result in self.recent_results:
                    if result.job_id == job_id:
                        return result
                
                # If not in cache, create result from job status
                return ProcessingResult(
                    job_id=job_id,
                    library_id=job_status['library_id'],
                    version=job_status['version'],
                    success=job_status['status'] == JobStatus.COMPLETED.value,
                    processing_time=job_status.get('processing_time', 0.0),
                    sections_parsed=0,  # Not available from job status
                    chunks_created=job_status.get('chunks_processed', 0),
                    vectors_generated=job_status.get('vectors_created', 0),
                    avg_quality_score=0.0,
                    min_quality_score=0.0,
                    max_quality_score=0.0,
                    error_message=job_status.get('error_message')
                )
            
            await asyncio.sleep(1.0)  # Check every second
        
        return None  # Timeout
    
    async def _process_job(self, job: IngestionJob) -> bool:
        """Process a single ingestion job through the complete pipeline."""
        start_time = time.time()
        result = ProcessingResult(
            job_id=job.job_id,
            library_id=job.library_id,
            version=job.version,
            success=False,
            processing_time=0.0,
            sections_parsed=0,
            chunks_created=0,
            vectors_generated=0,
            avg_quality_score=0.0,
            min_quality_score=0.0,
            max_quality_score=0.0
        )
        
        try:
            logger.info(f"Processing job {job.job_id} for {job.library_id}:{job.version}")
            
            # Stage 1: Parse JSON document
            logger.debug(f"Stage 1: Parsing document {job.doc_path}")
            sections = await self.json_parser.parse_document(job.doc_path)
            result.sections_parsed = len(sections)
            result.parsed_sections = sections
            
            if not sections:
                result.failed_stage = "json_parsing"
                result.error_message = "No sections parsed from document"
                return False
            
            # Stage 2: Chunk document sections
            logger.debug(f"Stage 2: Chunking {len(sections)} sections")
            chunks = await self.chunking_engine.chunk_document_sections(sections)
            result.chunks_created = len(chunks)
            result.document_chunks = chunks
            
            if not chunks:
                result.failed_stage = "chunking"
                result.error_message = "No chunks created from sections"
                return False
            
            # Stage 3: Extract and enrich metadata
            logger.debug(f"Stage 3: Enriching {len(chunks)} chunks")
            enriched_chunks = await self.metadata_extractor.enrich_chunks(chunks)
            
            # Calculate quality scores
            quality_scores = [
                chunk.metadata.get('quality_score', 0.5) 
                for chunk in enriched_chunks
            ]
            
            if quality_scores:
                result.avg_quality_score = sum(quality_scores) / len(quality_scores)
                result.min_quality_score = min(quality_scores)
                result.max_quality_score = max(quality_scores)
            
            # Stage 4: Generate embeddings
            logger.debug(f"Stage 4: Generating embeddings for {len(enriched_chunks)} chunks")
            embedding_requests = []
            
            # Import here to avoid circular imports
            from ..models.embedding_models import EmbeddingRequest, InputType
            
            for chunk in enriched_chunks:
                request = EmbeddingRequest(
                    content=chunk.content,
                    input_type=InputType.DOCUMENT,
                    metadata={
                        'chunk_id': chunk.chunk_id,
                        'library_id': chunk.library_id,
                        'version': chunk.version,
                        'chunk_type': chunk.chunk_type,
                        'programming_language': chunk.programming_language,
                        'token_count': chunk.token_count
                    }
                )
                embedding_requests.append(request)
            
            embedding_results = await self.embedding_engine.generate_batch_embeddings(
                embedding_requests
            )
            
            if not embedding_results.results:
                result.failed_stage = "embedding_generation"
                result.error_message = "No embeddings generated"
                return False
            
            # Stage 5: Store vectors
            logger.debug(f"Stage 5: Storing {len(embedding_results.results)} vectors")
            vector_documents = []
            
            # Import here to avoid circular imports
            from ..vector.qdrant_vector_store import VectorDocument
            
            for i, (chunk, embedding_result) in enumerate(zip(enriched_chunks, embedding_results.results)):
                if embedding_result.success and embedding_result.embedding:
                    vector_doc = VectorDocument(
                        id=chunk.chunk_id,
                        vector=embedding_result.embedding,
                        payload={
                            'library_id': chunk.library_id,
                            'version': chunk.version,
                            'chunk_index': chunk.chunk_index,
                            'total_chunks': chunk.total_chunks,
                            'chunk_type': chunk.chunk_type,
                            'programming_language': chunk.programming_language,
                            'semantic_boundary': chunk.semantic_boundary,
                            'token_count': chunk.token_count,
                            'quality_score': chunk.metadata.get('quality_score', 0.5),
                            'content_preview': chunk.content[:200],  # First 200 chars for preview
                            'tags': chunk.metadata.get('content_analysis', {}).get('tags', []),
                            'created_at': chunk.created_at.isoformat()
                        }
                    )
                    vector_documents.append(vector_doc)
            
            if vector_documents:
                storage_result = await self.vector_storage.upsert_vectors_batch(vector_documents)
                result.vectors_generated = storage_result.get('successful_uploads', 0)
            
            # Stage 6: Update job with results
            job.complete_successfully(
                chunks_processed=result.chunks_created,
                vectors_created=result.vectors_generated
            )
            
            result.success = True
            result.processing_time = time.time() - start_time
            
            logger.info(
                f"Successfully processed job {job.job_id}: "
                f"{result.sections_parsed} sections, {result.chunks_created} chunks, "
                f"{result.vectors_generated} vectors in {result.processing_time:.2f}s"
            )
            
            return True
            
        except Exception as e:
            result.processing_time = time.time() - start_time
            result.error_message = str(e)
            result.failed_stage = result.failed_stage or "unknown"
            
            logger.error(f"Failed to process job {job.job_id}: {e}")
            return False
            
        finally:
            # Update statistics and cache result
            self.statistics.update_from_result(result)
            self._cache_result(result)
    
    def _cache_result(self, result: ProcessingResult):
        """Cache processing result for retrieval."""
        self.recent_results.append(result)
        
        # Keep only recent results
        if len(self.recent_results) > self.max_cached_results:
            self.recent_results = self.recent_results[-self.max_cached_results:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        pipeline_stats = self.statistics.to_dict()
        
        # Add component statistics
        pipeline_stats['components'] = {
            'trigger_system': self.trigger_system.get_statistics(),
            'job_queue': self.job_queue.get_statistics(),
            'worker_pool': self.worker_pool.get_statistics(),
            'json_parser': self.json_parser.get_parsing_statistics(),
            'chunking_engine': self.chunking_engine.get_chunking_statistics(),
            'metadata_extractor': self.metadata_extractor.get_extraction_statistics()
        }
        
        return pipeline_stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check on the pipeline."""
        health_status = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'metrics': {}
        }
        
        try:
            # Check pipeline initialization
            if not self._initialized:
                health_status['status'] = 'not_initialized'
                return health_status
            
            # Check trigger system
            trigger_health = await self.trigger_system.health_check()
            health_status['components']['trigger_system'] = trigger_health
            
            # Check worker pool
            worker_health = self.worker_pool.get_health_status()
            health_status['components']['worker_pool'] = worker_health
            
            # Check queue
            queue_stats = self.job_queue.get_statistics()
            health_status['components']['job_queue'] = {
                'status': 'healthy' if queue_stats['utilization'] < 0.9 else 'warning',
                'utilization': queue_stats['utilization'],
                'queue_size': queue_stats['queue_size']
            }
            
            # Check external dependencies
            embedding_health = await self.embedding_engine.health_check()
            health_status['components']['embedding_engine'] = embedding_health
            
            vector_health = self.vector_storage.get_health_status()
            health_status['components']['vector_storage'] = vector_health
            
            # Overall metrics
            health_status['metrics'] = {
                'success_rate': (
                    self.statistics.total_jobs_successful / 
                    max(1, self.statistics.total_jobs_processed)
                ),
                'throughput_per_minute': self.statistics.throughput_per_minute,
                'avg_processing_time': self.statistics.avg_processing_time_per_job,
                'total_processed': self.statistics.total_jobs_processed
            }
            
            # Determine overall status
            component_statuses = [
                comp.get('status', 'unknown') 
                for comp in health_status['components'].values()
            ]
            
            if 'critical' in component_statuses or 'error' in component_statuses:
                health_status['status'] = 'critical'
            elif 'warning' in component_statuses or 'degraded' in component_statuses:
                health_status['status'] = 'degraded'
            elif all(status in ['healthy', 'unknown'] for status in component_statuses):
                health_status['status'] = 'healthy'
            else:
                health_status['status'] = 'partial'
                
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status
    
    async def get_processing_summary(self) -> Dict[str, Any]:
        """Get detailed processing summary with recent activity."""
        summary = {
            'pipeline_info': {
                'initialized': self._initialized,
                'max_workers': self.max_workers,
                'quality_threshold': self.quality_threshold,
                'chunk_size': self.chunking_engine.chunk_size,
                'chunk_overlap': self.chunking_engine.chunk_overlap
            },
            'statistics': self.get_statistics(),
            'health': await self.health_check(),
            'recent_results': [
                result.to_dict() for result in self.recent_results[-10:]
            ]
        }
        
        return summary