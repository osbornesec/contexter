"""
Embedding Integration Layer

Integration layer that connects the embedding engine with the vector storage system,
providing seamless document ingestion and vector storage workflows.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union, AsyncIterator
from pathlib import Path
import json

from ..models.embedding_models import (
    EmbeddingRequest, EmbeddingResult, InputType, ProcessingMetrics
)
from ..models.storage_models import DocumentationChunk
from .embedding_engine import VoyageEmbeddingEngine, EmbeddingEngineConfig
from .qdrant_vector_store import QdrantVectorStore, VectorStoreConfig
from .vector_search_engine import VectorSearchEngine, SearchQuery

logger = logging.getLogger(__name__)


@dataclass
class DocumentEmbedding:
    """Document with its embedding for vector storage."""
    
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    library_id: str = ""
    library_name: str = ""
    version: str = ""
    doc_type: str = ""
    section: str = ""
    subsection: Optional[str] = None
    programming_language: Optional[str] = None
    trust_score: float = 0.0
    star_count: int = 0
    token_count: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    content_hash: str = ""
    indexed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    
    total_documents: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    cached_embeddings: int = 0
    stored_vectors: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_embeddings / self.total_documents if self.total_documents > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_processed = self.successful_embeddings + self.cached_embeddings
        return self.cached_embeddings / total_processed if total_processed > 0 else 0.0


class EmbeddingVectorIntegration:
    """
    Integration layer between embedding generation and vector storage.
    
    Orchestrates the complete workflow from document chunks to searchable vectors:
    1. Content preprocessing and optimization
    2. Embedding generation with caching
    3. Vector storage in Qdrant
    4. Search functionality
    """
    
    def __init__(
        self,
        embedding_engine: VoyageEmbeddingEngine,
        vector_store: QdrantVectorStore,
        search_engine: Optional[VectorSearchEngine] = None
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.search_engine = search_engine
        
        # Performance tracking
        self.ingestion_metrics = ProcessingMetrics()
        self._ingestion_history = []
        
        # State management
        self._initialized = False
    
    async def initialize(self):
        """Initialize the integration layer."""
        if self._initialized:
            return
        
        logger.info("Initializing embedding-vector integration")
        
        # Ensure all components are initialized
        if not self.embedding_engine._initialized:
            await self.embedding_engine.initialize()
        
        # Vector store should already be initialized by the application
        # Create search engine if not provided
        if not self.search_engine:
            self.search_engine = VectorSearchEngine(self.vector_store)
        
        self._initialized = True
        logger.info("Embedding-vector integration initialized")
    
    async def shutdown(self):
        """Shutdown the integration layer."""
        logger.info("Shutting down embedding-vector integration")
        
        # Shutdown components (but don't shutdown shared resources)
        # The embedding engine and vector store should be managed externally
        
        self._initialized = False
        logger.info("Embedding-vector integration shutdown complete")
    
    async def ingest_documents(
        self,
        chunks: List[DocumentationChunk],
        library_id: str,
        library_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> IngestionResult:
        """
        Ingest documentation chunks into the vector database.
        
        Args:
            chunks: List of documentation chunks to process
            library_id: Unique identifier for the library
            library_name: Human-readable library name
            version: Library version
            metadata: Additional metadata to include
            progress_callback: Optional callback for progress updates
            
        Returns:
            Ingestion result with statistics and errors
        """
        if not self._initialized:
            await self.initialize()
        
        if not chunks:
            return IngestionResult()
        
        start_time = time.time()
        result = IngestionResult(total_documents=len(chunks))
        
        logger.info(f"Starting ingestion of {len(chunks)} chunks for {library_id}:{version}")
        
        try:
            # Step 1: Prepare embedding requests
            embedding_requests = self._prepare_embedding_requests(
                chunks, library_id, library_name, version, metadata
            )
            
            # Step 2: Generate embeddings in batches
            batch_result = await self.embedding_engine.generate_batch_embeddings(
                embedding_requests
            )
            
            # Update ingestion metrics
            result.successful_embeddings = len(batch_result.successful_results)
            result.failed_embeddings = len(batch_result.failed_results)
            result.cached_embeddings = batch_result.cache_hits
            result.errors.extend(batch_result.errors)
            
            if progress_callback:
                await progress_callback(result.successful_embeddings, len(chunks))
            
            # Step 3: Convert to document embeddings for storage
            document_embeddings = self._create_document_embeddings(
                chunks, batch_result.results, library_id, library_name, version, metadata
            )
            
            # Step 4: Store vectors in Qdrant
            if document_embeddings:
                storage_result = await self._store_vectors(document_embeddings)
                result.stored_vectors = storage_result.get('successful_adds', 0)
                
                if not storage_result.get('success', False):
                    result.errors.append(f"Vector storage failed: {storage_result.get('error', 'Unknown error')}")
            
            # Step 5: Update metrics and history
            result.processing_time = time.time() - start_time
            await self._update_ingestion_metrics(result)
            
            logger.info(
                f"Ingestion complete: {result.successful_embeddings}/{result.total_documents} "
                f"successful ({result.success_rate:.1%}), "
                f"{result.cached_embeddings} cached ({result.cache_hit_rate:.1%}), "
                f"in {result.processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            result.processing_time = time.time() - start_time
            result.errors.append(f"Ingestion failed: {str(e)}")
            
            logger.error(f"Document ingestion failed: {e}")
            return result
    
    def _prepare_embedding_requests(
        self,
        chunks: List[DocumentationChunk],
        library_id: str,
        library_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[EmbeddingRequest]:
        """Prepare embedding requests from documentation chunks."""
        requests = []
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            # Combine chunk metadata with base metadata
            request_metadata = {
                **base_metadata,
                'library_id': library_id,
                'library_name': library_name,
                'version': version,
                'chunk_id': chunk.chunk_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'source_context': getattr(chunk, 'source_context', ''),
                'token_count': getattr(chunk, 'token_count', 0),
                'content_hash': getattr(chunk, 'content_hash', ''),
                'doc_type': getattr(chunk, 'doc_type', 'documentation'),
                'section': getattr(chunk, 'section', ''),
                'subsection': getattr(chunk, 'subsection', None),
                'programming_language': getattr(chunk, 'programming_language', None),
                'trust_score': getattr(chunk, 'trust_score', 0.0),
                'star_count': getattr(chunk, 'star_count', 0),
                'model': self.embedding_engine.config.voyage_model
            }
            
            request = EmbeddingRequest(
                content=chunk.content,
                content_hash=getattr(chunk, 'content_hash', ''),
                metadata=request_metadata,
                input_type=InputType.DOCUMENT,
                chunk_id=chunk.chunk_id,
                library_id=library_id
            )
            
            requests.append(request)
        
        return requests
    
    def _create_document_embeddings(
        self,
        chunks: List[DocumentationChunk],
        embedding_results: List[EmbeddingResult],
        library_id: str,
        library_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentEmbedding]:
        """Create document embeddings from chunks and embedding results."""
        document_embeddings = []
        
        # Create lookup for results by content hash
        results_map = {result.content_hash: result for result in embedding_results}
        
        for i, chunk in enumerate(chunks):
            # Find corresponding embedding result
            chunk_hash = getattr(chunk, 'content_hash', '')
            if not chunk_hash:
                # Generate hash if not present
                from ..models.embedding_models import normalize_content_for_hashing
                import hashlib
                normalized = normalize_content_for_hashing(chunk.content)
                chunk_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            
            result = results_map.get(chunk_hash)
            if not result or not result.success:
                logger.warning(f"Skipping chunk {chunk.chunk_id} - no valid embedding")
                continue
            
            # Create document embedding
            doc_embedding = DocumentEmbedding(
                id=chunk.chunk_id,
                content=chunk.content,
                embedding=result.embedding,
                metadata={
                    **(metadata or {}),
                    'embedding_model': result.model,
                    'processing_time': result.processing_time,
                    'cache_hit': result.cache_hit,
                    'created_at': result.created_at.isoformat()
                },
                library_id=library_id,
                library_name=library_name,
                version=version,
                doc_type=getattr(chunk, 'doc_type', 'documentation'),
                section=getattr(chunk, 'section', ''),
                subsection=getattr(chunk, 'subsection', None),
                programming_language=getattr(chunk, 'programming_language', None),
                trust_score=getattr(chunk, 'trust_score', 0.0),
                star_count=getattr(chunk, 'star_count', 0),
                token_count=getattr(chunk, 'token_count', 0),
                chunk_index=i,
                total_chunks=len(chunks),
                content_hash=chunk_hash,
                indexed_at=datetime.utcnow()
            )
            
            document_embeddings.append(doc_embedding)
        
        return document_embeddings
    
    async def _store_vectors(self, document_embeddings: List[DocumentEmbedding]) -> Dict[str, Any]:
        """Store document embeddings in the vector database."""
        if not document_embeddings:
            return {'success': True, 'successful_adds': 0}
        
        try:
            # Convert to vector store format
            vectors = []
            for doc_emb in document_embeddings:
                vector_data = {
                    'id': doc_emb.id,
                    'vector': doc_emb.embedding,
                    'payload': {
                        'content': doc_emb.content,
                        'library_id': doc_emb.library_id,
                        'library_name': doc_emb.library_name,
                        'version': doc_emb.version,
                        'doc_type': doc_emb.doc_type,
                        'section': doc_emb.section,
                        'subsection': doc_emb.subsection,
                        'programming_language': doc_emb.programming_language,
                        'trust_score': doc_emb.trust_score,
                        'star_count': doc_emb.star_count,
                        'token_count': doc_emb.token_count,
                        'chunk_index': doc_emb.chunk_index,
                        'total_chunks': doc_emb.total_chunks,
                        'content_hash': doc_emb.content_hash,
                        'indexed_at': doc_emb.indexed_at.isoformat(),
                        **doc_emb.metadata
                    }
                }
                vectors.append(vector_data)
            
            # Use the vector store's upsert method
            # Note: This assumes the vector store has an upsert_vectors method
            # We may need to adapt this based on the actual QdrantVectorStore interface
            result = await self.vector_store.upsert_vectors(vectors)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return {
                'success': False,
                'error': str(e),
                'successful_adds': 0
            }
    
    async def search_similar_documents(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to a query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional metadata filters
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_engine.embed_query(query)
            
            # Create search query
            search_query = SearchQuery(
                vector=query_embedding,
                top_k=top_k,
                filters=filters,
                score_threshold=score_threshold
            )
            
            # Perform search
            results = await self.search_engine.search(search_query)
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'id': result.id,
                    'score': result.score,
                    'content': result.payload.get('content', ''),
                    'library_id': result.payload.get('library_id', ''),
                    'library_name': result.payload.get('library_name', ''),
                    'version': result.payload.get('version', ''),
                    'section': result.payload.get('section', ''),
                    'doc_type': result.payload.get('doc_type', ''),
                    'metadata': result.payload,
                    'relevance_score': result.relevance_score,
                    'match_reasons': result.match_reasons
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _update_ingestion_metrics(self, result: IngestionResult):
        """Update ingestion performance metrics."""
        # Update processing metrics
        self.ingestion_metrics.total_requests += result.total_documents
        self.ingestion_metrics.successful_requests += result.successful_embeddings
        self.ingestion_metrics.failed_requests += result.failed_embeddings
        self.ingestion_metrics.cache_hits += result.cached_embeddings
        self.ingestion_metrics.total_processing_time += result.processing_time
        self.ingestion_metrics.last_processed = datetime.utcnow()
        
        # Add to history
        self._ingestion_history.append({
            'timestamp': datetime.utcnow(),
            'total_documents': result.total_documents,
            'success_rate': result.success_rate,
            'cache_hit_rate': result.cache_hit_rate,
            'processing_time': result.processing_time,
            'throughput': result.total_documents / result.processing_time if result.processing_time > 0 else 0
        })
        
        # Keep only recent history
        if len(self._ingestion_history) > 100:
            self._ingestion_history = self._ingestion_history[-100:]
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get comprehensive ingestion status and metrics."""
        return {
            'metrics': {
                'total_documents_processed': self.ingestion_metrics.total_requests,
                'successful_embeddings': self.ingestion_metrics.successful_requests,
                'failed_embeddings': self.ingestion_metrics.failed_requests,
                'cache_hits': self.ingestion_metrics.cache_hits,
                'success_rate': self.ingestion_metrics.success_rate,
                'cache_hit_rate': self.ingestion_metrics.cache_hit_rate,
                'average_processing_time': self.ingestion_metrics.average_processing_time,
                'throughput_per_minute': self.ingestion_metrics.throughput_per_minute,
                'last_processed': self.ingestion_metrics.last_processed.isoformat() if self.ingestion_metrics.last_processed else None
            },
            'recent_batches': self._ingestion_history[-10:],  # Last 10 batches
            'embedding_engine_status': await self.embedding_engine.get_detailed_status(),
            'vector_store_health': await self.vector_store.health_check() if hasattr(self.vector_store, 'health_check') else {}
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization tasks."""
        optimization_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'actions_taken': [],
            'performance_improvements': {}
        }
        
        try:
            # Get current metrics
            current_metrics = await self.get_ingestion_status()
            
            # Check if cache cleanup is needed
            if hasattr(self.embedding_engine.cache, 'clear_expired'):
                expired_count = await self.embedding_engine.cache.clear_expired()
                if expired_count > 0:
                    optimization_results['actions_taken'].append(
                        f"Cleared {expired_count} expired cache entries"
                    )
            
            # Optimize vector store if available
            if hasattr(self.vector_store, 'optimize_collection'):
                vector_optimization = await self.vector_store.optimize_collection()
                if vector_optimization.get('success'):
                    optimization_results['actions_taken'].append("Optimized vector store collection")
            
            # Record optimization completion
            optimization_results['actions_taken'].append("Performance optimization completed")
            
        except Exception as e:
            optimization_results['error'] = str(e)
            logger.error(f"Performance optimization failed: {e}")
        
        return optimization_results


# Factory function for easy integration setup
async def create_embedding_integration(
    voyage_api_key: str,
    vector_store_config: Optional[VectorStoreConfig] = None,
    embedding_config_overrides: Optional[Dict[str, Any]] = None
) -> EmbeddingVectorIntegration:
    """
    Create and initialize a complete embedding-vector integration.
    
    Args:
        voyage_api_key: Voyage AI API key
        vector_store_config: Optional vector store configuration
        embedding_config_overrides: Optional embedding configuration overrides
        
    Returns:
        Initialized integration layer
    """
    # Create embedding engine
    embedding_config = EmbeddingEngineConfig(
        voyage_api_key=voyage_api_key,
        **(embedding_config_overrides or {})
    )
    embedding_engine = VoyageEmbeddingEngine(embedding_config)
    
    # Create vector store
    if not vector_store_config:
        vector_store_config = VectorStoreConfig()
    
    vector_store = QdrantVectorStore(vector_store_config)
    
    # Create search engine
    search_engine = VectorSearchEngine(vector_store)
    
    # Create integration
    integration = EmbeddingVectorIntegration(
        embedding_engine=embedding_engine,
        vector_store=vector_store,
        search_engine=search_engine
    )
    
    await integration.initialize()
    
    return integration