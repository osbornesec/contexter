"""
RAG Document Ingestion Pipeline

Automated document processing pipeline for transforming C7DocDownloader JSON output
into searchable vector embeddings with comprehensive metadata enrichment.

Key Components:
- Auto-ingestion trigger system with quality validation
- Priority-based processing queue with worker pool management  
- Intelligent document chunking with semantic boundary preservation
- Metadata extraction and enrichment with content analysis
- Seamless integration with embedding engine and vector storage

Performance Targets:
- >1000 documents/minute processing throughput
- <10 seconds trigger latency after download completion
- 99%+ parsing success rate for valid JSON documentation
- <2GB memory usage during peak processing operations
"""

from .trigger_system import AutoIngestionTrigger, IngestionTriggerEvent
from .processing_queue import IngestionQueue, IngestionJob, WorkerPool
from .json_parser import JSONDocumentParser, DocumentParsingError
from .chunking_engine import IntelligentChunkingEngine, ChunkingStrategy
from .metadata_extractor import MetadataExtractor, ContentAnalyzer
from .pipeline import IngestionPipeline, IngestionStatistics
from .quality_validator import QualityValidator, QualityAssessment

__all__ = [
    # Core Pipeline
    'IngestionPipeline',
    'IngestionStatistics',
    
    # Trigger System
    'AutoIngestionTrigger',
    'IngestionTriggerEvent',
    
    # Processing Queue
    'IngestionQueue',
    'IngestionJob', 
    'WorkerPool',
    
    # Document Processing
    'JSONDocumentParser',
    'DocumentParsingError',
    'IntelligentChunkingEngine',
    'ChunkingStrategy',
    
    # Metadata & Quality
    'MetadataExtractor',
    'ContentAnalyzer',
    'QualityValidator',
    'QualityAssessment',
]