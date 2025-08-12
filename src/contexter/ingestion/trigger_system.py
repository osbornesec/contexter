"""
Auto-Ingestion Trigger System

Event-driven trigger system that automatically starts document ingestion
when downloads complete, with comprehensive quality validation and 
priority-based queue management.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import json
from enum import Enum

logger = logging.getLogger(__name__)


class TriggerEventType(Enum):
    """Types of ingestion trigger events."""
    DOWNLOAD_COMPLETE = "download_complete"
    MANUAL_TRIGGER = "manual_trigger"
    SCHEDULED_REPROCESSING = "scheduled_reprocessing"
    QUALITY_REVALIDATION = "quality_revalidation"


@dataclass
class IngestionTriggerEvent:
    """
    Event data for ingestion triggers.
    
    Contains all information needed to process a document
    through the ingestion pipeline.
    """
    library_id: str
    version: str
    doc_path: Path
    metadata: Dict[str, Any]
    trigger_timestamp: datetime
    event_type: TriggerEventType = TriggerEventType.DOWNLOAD_COMPLETE
    priority_boost: int = 0  # Additional priority points
    retry_count: int = 0
    previous_quality_score: Optional[float] = None


class AutoIngestionTrigger:
    """
    Automatic ingestion trigger with quality validation.
    
    Features:
    - Event-driven architecture with async processing
    - Quality validation with configurable thresholds
    - Priority calculation based on library metadata
    - Error handling with retry logic and monitoring
    - Status tracking and progress reporting
    """
    
    def __init__(
        self, 
        storage_manager: 'StorageManager',
        ingestion_pipeline: 'IngestionPipeline',
        quality_threshold: float = 0.7,
        max_queue_size: int = 10000,
        processing_timeout: float = 300.0
    ):
        """
        Initialize the auto-ingestion trigger system.
        
        Args:
            storage_manager: Storage system interface
            ingestion_pipeline: Target pipeline for processing
            quality_threshold: Minimum quality score (0.0-1.0)
            max_queue_size: Maximum trigger event queue size
            processing_timeout: Maximum processing time per event
        """
        self.storage_manager = storage_manager
        self.ingestion_pipeline = ingestion_pipeline
        self.quality_threshold = quality_threshold
        self.max_queue_size = max_queue_size
        self.processing_timeout = processing_timeout
        
        # Event processing
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.quality_validator = None  # Will be initialized in start_processing
        self.processing_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics and monitoring
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_rejected': 0,
            'events_failed': 0,
            'quality_scores': [],
            'processing_times': [],
            'last_activity': None,
            'start_time': None
        }
        
        # Event handlers
        self._event_handlers: Dict[TriggerEventType, Callable] = {
            TriggerEventType.DOWNLOAD_COMPLETE: self._handle_download_complete,
            TriggerEventType.MANUAL_TRIGGER: self._handle_manual_trigger,
            TriggerEventType.SCHEDULED_REPROCESSING: self._handle_scheduled_reprocessing,
            TriggerEventType.QUALITY_REVALIDATION: self._handle_quality_revalidation
        }
        
        logger.info(f"Auto-ingestion trigger initialized (threshold: {quality_threshold})")
    
    async def start_processing(self):
        """Start the background event processing task."""
        if self.processing_task and not self.processing_task.done():
            logger.warning("Processing task already running")
            return
        
        # Import here to avoid circular imports
        from .quality_validator import QualityValidator
        self.quality_validator = QualityValidator()
        
        self.stats['start_time'] = datetime.now()
        self._shutdown_event.clear()
        
        # Start background processing
        self.processing_task = asyncio.create_task(self._process_trigger_events())
        logger.info("Auto-ingestion trigger processing started")
    
    async def stop_processing(self):
        """Stop the background event processing task."""
        logger.info("Stopping auto-ingestion trigger processing")
        
        self._shutdown_event.set()
        
        if self.processing_task:
            try:
                await asyncio.wait_for(self.processing_task, timeout=30.0)
            except asyncio.TimeoutError:
                logger.warning("Processing task shutdown timeout, cancelling")
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Auto-ingestion trigger processing stopped")
    
    async def on_download_complete(
        self, 
        library_id: str, 
        version: str, 
        doc_path: Path, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Handle download completion event and trigger ingestion.
        
        Args:
            library_id: Unique identifier for the library
            version: Version string
            doc_path: Path to downloaded documentation
            metadata: Library metadata from download
            
        Returns:
            True if event was queued successfully
        """
        trigger_event = IngestionTriggerEvent(
            library_id=library_id,
            version=version,
            doc_path=doc_path,
            metadata=metadata,
            trigger_timestamp=datetime.now(),
            event_type=TriggerEventType.DOWNLOAD_COMPLETE
        )
        
        return await self._queue_event(trigger_event)
    
    async def trigger_manual_ingestion(
        self,
        library_id: str,
        version: str,
        doc_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        priority_boost: int = 5
    ) -> bool:
        """
        Manually trigger ingestion for a specific document.
        
        Args:
            library_id: Library identifier
            version: Version string
            doc_path: Path to documentation
            metadata: Optional metadata override
            priority_boost: Additional priority points
            
        Returns:
            True if event was queued successfully
        """
        if metadata is None:
            # Try to load metadata from storage
            doc_data = await self.storage_manager.retrieve_documentation(library_id, version)
            metadata = doc_data.get('metadata', {}) if doc_data else {}
        
        trigger_event = IngestionTriggerEvent(
            library_id=library_id,
            version=version,
            doc_path=doc_path,
            metadata=metadata,
            trigger_timestamp=datetime.now(),
            event_type=TriggerEventType.MANUAL_TRIGGER,
            priority_boost=priority_boost
        )
        
        return await self._queue_event(trigger_event)
    
    async def trigger_reprocessing(
        self,
        library_id: str,
        version: Optional[str] = None,
        reason: str = "scheduled_maintenance"
    ) -> int:
        """
        Trigger reprocessing for library versions.
        
        Args:
            library_id: Library to reprocess
            version: Specific version (all versions if None)
            reason: Reason for reprocessing
            
        Returns:
            Number of reprocessing events queued
        """
        versions_to_process = []
        
        if version:
            versions_to_process = [version]
        else:
            # Get all versions for the library
            version_list = await self.storage_manager.get_library_versions(library_id)
            versions_to_process = [v['version'] for v in version_list]
        
        queued_count = 0
        
        for v in versions_to_process:
            # Get documentation path and metadata
            doc_data = await self.storage_manager.retrieve_documentation(library_id, v)
            if not doc_data:
                logger.warning(f"No documentation found for {library_id} v{v}")
                continue
            
            # Construct document path (this would need to be adapted based on storage structure)
            doc_path = Path(f"~/.contexter/documentation/{library_id}/{v}.json")
            
            trigger_event = IngestionTriggerEvent(
                library_id=library_id,
                version=v,
                doc_path=doc_path,
                metadata={**doc_data.get('metadata', {}), 'reprocessing_reason': reason},
                trigger_timestamp=datetime.now(),
                event_type=TriggerEventType.SCHEDULED_REPROCESSING,
                priority_boost=2
            )
            
            if await self._queue_event(trigger_event):
                queued_count += 1
        
        logger.info(f"Queued {queued_count} reprocessing events for {library_id}")
        return queued_count
    
    async def _queue_event(self, event: IngestionTriggerEvent) -> bool:
        """Queue an ingestion trigger event for processing."""
        try:
            if self.event_queue.full():
                logger.error("Trigger event queue is full, dropping event")
                return False
            
            await self.event_queue.put(event)
            self.stats['events_received'] += 1
            self.stats['last_activity'] = datetime.now()
            
            logger.debug(f"Queued {event.event_type.value} event for {event.library_id}:{event.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue trigger event: {e}")
            return False
    
    async def _process_trigger_events(self):
        """Background processor for ingestion trigger events."""
        logger.info("Starting trigger event processing loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for event with timeout
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(), 
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue  # Check shutdown event and continue
                
                # Process the event
                await self._process_trigger_event(event)
                
            except Exception as e:
                logger.error(f"Error in trigger event processing loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying
        
        logger.info("Trigger event processing loop stopped")
    
    async def _process_trigger_event(self, event: IngestionTriggerEvent):
        """Process individual trigger event with quality validation."""
        start_time = time.time()
        
        try:
            logger.debug(f"Processing {event.event_type.value} event for {event.library_id}:{event.version}")
            
            # Get appropriate handler for event type
            handler = self._event_handlers.get(event.event_type)
            if not handler:
                logger.error(f"No handler for event type: {event.event_type}")
                self.stats['events_failed'] += 1
                return
            
            # Execute the handler
            success = await asyncio.wait_for(
                handler(event),
                timeout=self.processing_timeout
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
            if success:
                self.stats['events_processed'] += 1
                logger.info(f"Successfully processed {event.library_id}:{event.version} in {processing_time:.2f}s")
            else:
                self.stats['events_failed'] += 1
                logger.warning(f"Failed to process {event.library_id}:{event.version}")
            
        except asyncio.TimeoutError:
            logger.error(f"Trigger event processing timeout for {event.library_id}:{event.version}")
            self.stats['events_failed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process trigger event for {event.library_id}:{event.version}: {e}")
            self.stats['events_failed'] += 1
    
    async def _handle_download_complete(self, event: IngestionTriggerEvent) -> bool:
        """Handle download completion events."""
        try:
            # Validate document quality
            quality_score = await self.quality_validator.assess_document_quality(
                event.doc_path, event.metadata
            )
            
            self.stats['quality_scores'].append(quality_score)
            
            if quality_score >= self.quality_threshold:
                # Calculate priority based on metadata and quality
                priority = self._calculate_priority(
                    event.metadata, 
                    quality_score, 
                    event.priority_boost
                )
                
                # Queue for ingestion
                success = await self.ingestion_pipeline.queue_document(
                    library_id=event.library_id,
                    version=event.version,
                    doc_path=event.doc_path,
                    priority=priority,
                    metadata={
                        **event.metadata,
                        'quality_score': quality_score,
                        'trigger_timestamp': event.trigger_timestamp.isoformat(),
                        'trigger_type': event.event_type.value
                    }
                )
                
                if success:
                    logger.info(
                        f"Queued {event.library_id}:{event.version} for ingestion "
                        f"(quality: {quality_score:.2f}, priority: {priority})"
                    )
                    return True
                else:
                    logger.error(f"Failed to queue {event.library_id}:{event.version} for ingestion")
                    return False
            else:
                logger.warning(
                    f"Rejected {event.library_id}:{event.version} due to low quality: {quality_score:.2f} "
                    f"< {self.quality_threshold}"
                )
                self.stats['events_rejected'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to handle download complete event: {e}")
            return False
    
    async def _handle_manual_trigger(self, event: IngestionTriggerEvent) -> bool:
        """Handle manual trigger events (bypass quality check)."""
        try:
            # For manual triggers, we bypass quality validation but still assess
            quality_score = 1.0  # Manual triggers get maximum quality score
            if self.quality_validator:
                try:
                    quality_score = await self.quality_validator.assess_document_quality(
                        event.doc_path, event.metadata
                    )
                except Exception as e:
                    logger.warning(f"Quality assessment failed for manual trigger: {e}")
            
            priority = self._calculate_priority(
                event.metadata, 
                quality_score, 
                event.priority_boost + 5  # Manual triggers get extra priority
            )
            
            success = await self.ingestion_pipeline.queue_document(
                library_id=event.library_id,
                version=event.version,
                doc_path=event.doc_path,
                priority=priority,
                metadata={
                    **event.metadata,
                    'quality_score': quality_score,
                    'trigger_timestamp': event.trigger_timestamp.isoformat(),
                    'trigger_type': event.event_type.value,
                    'manual_trigger': True
                }
            )
            
            if success:
                logger.info(f"Manually queued {event.library_id}:{event.version} for ingestion")
                return True
            else:
                logger.error(f"Failed to manually queue {event.library_id}:{event.version}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to handle manual trigger event: {e}")
            return False
    
    async def _handle_scheduled_reprocessing(self, event: IngestionTriggerEvent) -> bool:
        """Handle scheduled reprocessing events."""
        # Similar to manual trigger but with lower priority
        return await self._handle_manual_trigger(event)
    
    async def _handle_quality_revalidation(self, event: IngestionTriggerEvent) -> bool:
        """Handle quality revalidation events."""
        # Revalidate quality and compare to previous score
        try:
            new_quality_score = await self.quality_validator.assess_document_quality(
                event.doc_path, event.metadata
            )
            
            previous_score = event.previous_quality_score or 0.0
            
            # Only process if quality improved significantly
            if new_quality_score >= self.quality_threshold and new_quality_score > previous_score + 0.1:
                priority = self._calculate_priority(event.metadata, new_quality_score, event.priority_boost)
                
                success = await self.ingestion_pipeline.queue_document(
                    library_id=event.library_id,
                    version=event.version,
                    doc_path=event.doc_path,
                    priority=priority,
                    metadata={
                        **event.metadata,
                        'quality_score': new_quality_score,
                        'previous_quality_score': previous_score,
                        'trigger_timestamp': event.trigger_timestamp.isoformat(),
                        'trigger_type': event.event_type.value,
                        'revalidation': True
                    }
                )
                
                if success:
                    logger.info(
                        f"Revalidated and queued {event.library_id}:{event.version} "
                        f"(quality improved: {previous_score:.2f} -> {new_quality_score:.2f})"
                    )
                    return True
            else:
                logger.debug(
                    f"Quality revalidation for {event.library_id}:{event.version} "
                    f"did not meet improvement threshold: {new_quality_score:.2f}"
                )
                return True  # Not an error, just no improvement
                
        except Exception as e:
            logger.error(f"Failed to handle quality revalidation event: {e}")
            return False
    
    def _calculate_priority(
        self, 
        metadata: Dict[str, Any], 
        quality_score: float, 
        priority_boost: int = 0
    ) -> int:
        """
        Calculate ingestion priority based on metadata and quality.
        
        Priority factors:
        - Star count (0-3 points)
        - Trust score (0-2 points) 
        - Quality score (0-3 points)
        - Category importance (0-2 points)
        - Recency (0-1 points)
        - Manual boost (variable)
        
        Returns priority score 0-15 (higher = more urgent)
        """
        priority = 0
        
        # Star count factor (0-3 points)
        star_count = metadata.get('star_count', 0)
        if star_count >= 10000:
            priority += 3
        elif star_count >= 1000:
            priority += 2
        elif star_count >= 100:
            priority += 1
        
        # Trust score factor (0-2 points)
        trust_score = metadata.get('trust_score', 0.0)
        if trust_score >= 0.8:
            priority += 2
        elif trust_score >= 0.6:
            priority += 1
        
        # Quality score factor (0-3 points)
        if quality_score >= 0.9:
            priority += 3
        elif quality_score >= 0.8:
            priority += 2
        elif quality_score >= 0.7:
            priority += 1
        
        # Category importance (0-2 points)
        category = metadata.get('category', '').lower()
        important_categories = {
            'web-framework', 'database', 'machine-learning', 
            'testing', 'security', 'api'
        }
        if category in important_categories:
            priority += 2
        elif category in ['utility', 'cli', 'development']:
            priority += 1
        
        # Recency factor (0-1 points) - recent downloads get slight boost
        download_timestamp = metadata.get('download_timestamp')
        if download_timestamp:
            try:
                download_time = datetime.fromisoformat(download_timestamp.replace('Z', '+00:00'))
                age_hours = (datetime.now().astimezone() - download_time).total_seconds() / 3600
                if age_hours < 24:  # Downloaded in last 24 hours
                    priority += 1
            except Exception:
                pass  # Ignore timestamp parsing errors
        
        # Apply manual boost
        priority += priority_boost
        
        # Clamp to valid range
        return max(0, min(15, priority))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and status information."""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['max_processing_time'] = 0.0
        
        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['min_quality_score'] = min(stats['quality_scores'])
        else:
            stats['avg_quality_score'] = 0.0
            stats['min_quality_score'] = 0.0
        
        # Add queue status
        stats['queue_size'] = self.event_queue.qsize()
        stats['queue_capacity'] = self.max_queue_size
        stats['queue_utilization'] = stats['queue_size'] / self.max_queue_size
        
        # Add processing status
        stats['is_processing'] = (
            self.processing_task is not None and 
            not self.processing_task.done()
        )
        
        # Calculate success rates
        total_events = stats['events_processed'] + stats['events_failed'] + stats['events_rejected']
        if total_events > 0:
            stats['success_rate'] = stats['events_processed'] / total_events
            stats['rejection_rate'] = stats['events_rejected'] / total_events
            stats['failure_rate'] = stats['events_failed'] / total_events
        else:
            stats['success_rate'] = 0.0
            stats['rejection_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'metrics': self.get_statistics()
        }
        
        try:
            # Check processing task health
            if self.processing_task and not self.processing_task.done():
                health_status['checks']['processing_task'] = 'healthy'
            else:
                health_status['checks']['processing_task'] = 'stopped'
            
            # Check queue capacity
            queue_utilization = health_status['metrics']['queue_utilization']
            if queue_utilization < 0.8:
                health_status['checks']['queue_capacity'] = 'healthy'
            elif queue_utilization < 0.95:
                health_status['checks']['queue_capacity'] = 'warning'
            else:
                health_status['checks']['queue_capacity'] = 'critical'
            
            # Check processing performance
            success_rate = health_status['metrics']['success_rate']
            if success_rate >= 0.95:
                health_status['checks']['success_rate'] = 'healthy'
            elif success_rate >= 0.85:
                health_status['checks']['success_rate'] = 'warning'
            else:
                health_status['checks']['success_rate'] = 'critical'
            
            # Check components
            if self.quality_validator:
                health_status['checks']['quality_validator'] = 'healthy'
            else:
                health_status['checks']['quality_validator'] = 'not_initialized'
            
            # Overall status
            check_statuses = list(health_status['checks'].values())
            if 'critical' in check_statuses:
                health_status['status'] = 'critical'
            elif 'warning' in check_statuses:
                health_status['status'] = 'warning'
            elif all(status in ['healthy', 'not_initialized'] for status in check_statuses):
                health_status['status'] = 'healthy'
            else:
                health_status['status'] = 'degraded'
                
        except Exception as e:
            health_status['status'] = 'error'
            health_status['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_status