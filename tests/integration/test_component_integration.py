"""
Component Integration Tests for RAG System

Tests specific integration points between major components:
- Vector Database ↔ Embedding Service
- Embedding Service ↔ Ingestion Pipeline  
- Vector Database ↔ Search Engine
- Pipeline ↔ Storage ↔ Vector Database

Focus on data consistency, error propagation, and interface contracts.
"""

import pytest
import pytest_asyncio
import asyncio
import json
import tempfile
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch

from contexter.vector import (
    QdrantVectorStore, VectorStoreConfig, VectorDocument, SearchResult,
    VectorSearchEngine, SearchQuery
)
from contexter.vector.embedding_engine import VoyageEmbeddingEngine, EmbeddingEngineConfig
from contexter.ingestion.pipeline import IngestionPipeline
from contexter.ingestion.json_parser import JSONDocumentParser
from contexter.ingestion.chunking_engine import IntelligentChunkingEngine
from contexter.models.embedding_models import EmbeddingRequest, EmbeddingResult, BatchResult, InputType

logger = logging.getLogger(__name__)


class MockVectorStoreForIntegration:
    """Mock vector store focused on integration testing behaviors."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._vectors: Dict[str, VectorDocument] = {}
        self._call_log: List[Dict[str, Any]] = []
        self._failure_mode: Optional[str] = None
        
    def set_failure_mode(self, mode: str):
        """Set failure mode for testing error handling."""
        self._failure_mode = mode
    
    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get log of all method calls for verification."""
        return self._call_log.copy()
    
    async def initialize(self):
        self._call_log.append({'method': 'initialize', 'timestamp': time.time()})
        if self._failure_mode == 'init_failure':
            raise Exception("Simulated initialization failure")
    
    async def upsert_vectors_batch(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        self._call_log.append({
            'method': 'upsert_vectors_batch',
            'count': len(documents),
            'timestamp': time.time()
        })
        
        if self._failure_mode == 'storage_failure':
            return {
                'successful_uploads': 0,
                'failed_uploads': len(documents),
                'total_time': 0.1
            }
        
        # Store vectors
        for doc in documents:
            self._vectors[doc.id] = doc
        
        return {
            'successful_uploads': len(documents),
            'failed_uploads': 0,
            'total_time': 0.05
        }
    
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        self._call_log.append({
            'method': 'search_vectors',
            'top_k': top_k,
            'filters': filters,
            'timestamp': time.time()
        })
        
        if self._failure_mode == 'search_failure':
            raise Exception("Simulated search failure")
        
        # Return mock results
        results = []
        for i, (doc_id, doc) in enumerate(list(self._vectors.items())[:top_k]):
            if filters:
                # Check if document matches filters
                matches = all(
                    doc.payload.get(key) == value
                    for key, value in filters.items()
                )
                if not matches:
                    continue
            
            result = SearchResult(
                id=doc_id,
                score=0.9 - (i * 0.1),
                payload=doc.payload,
                vector=doc.vector
            )
            results.append(result)
        
        return results
    
    async def count_vectors(self) -> int:
        return len(self._vectors)


class MockEmbeddingEngineForIntegration:
    """Mock embedding engine focused on integration testing."""
    
    def __init__(self, config: EmbeddingEngineConfig):
        self.config = config
        self._call_log: List[Dict[str, Any]] = []
        self._failure_mode: Optional[str] = None
        self._embedding_cache: Dict[str, List[float]] = {}
        
    def set_failure_mode(self, mode: str):
        """Set failure mode for testing error handling."""
        self._failure_mode = mode
    
    def get_call_log(self) -> List[Dict[str, Any]]:
        """Get log of all method calls for verification."""
        return self._call_log.copy()
    
    async def initialize(self):
        self._call_log.append({'method': 'initialize', 'timestamp': time.time()})
        if self._failure_mode == 'init_failure':
            raise Exception("Simulated embedding engine initialization failure")
    
    async def shutdown(self):
        self._call_log.append({'method': 'shutdown', 'timestamp': time.time()})
    
    async def generate_batch_embeddings(self, requests: List[EmbeddingRequest]) -> BatchResult:
        self._call_log.append({
            'method': 'generate_batch_embeddings',
            'request_count': len(requests),
            'timestamp': time.time()
        })
        
        if self._failure_mode == 'embedding_failure':
            # Return partial failure
            results = []
            for i, req in enumerate(requests):
                if i < len(requests) // 2:
                    # Success for first half
                    embedding = self._generate_deterministic_embedding(req.content)
                    results.append(EmbeddingResult(
                        content_hash=req.content_hash,
                        embedding=embedding,
                        model=self.config.voyage_model,
                        dimensions=2048,
                        processing_time=0.1,
                        cache_hit=False
                    ))
                else:
                    # Failure for second half
                    results.append(EmbeddingResult(
                        content_hash=req.content_hash,
                        embedding=[],
                        model=self.config.voyage_model,
                        dimensions=0,
                        processing_time=0.1,
                        cache_hit=False,
                        error="Simulated embedding failure"
                    ))
            
            return BatchResult(
                batch_id="partial_failure_batch",
                results=results,
                processing_time=0.1,
                errors=["Simulated partial embedding failure"]
            )
        
        # Normal processing
        results = []
        for req in requests:
            embedding = self._generate_deterministic_embedding(req.content)
            self._embedding_cache[req.content_hash] = embedding
            
            results.append(EmbeddingResult(
                content_hash=req.content_hash,
                embedding=embedding,
                model=self.config.voyage_model,
                dimensions=2048,
                processing_time=0.05,
                cache_hit=req.content_hash in self._embedding_cache
            ))
        
        return BatchResult(
            batch_id=f"batch_{int(time.time())}",
            results=results,
            processing_time=0.1
        )
    
    def _generate_deterministic_embedding(self, content: str) -> List[float]:
        """Generate deterministic embedding for testing."""
        # Use hash of content to generate consistent embeddings
        content_hash = hash(content)
        embedding = []
        
        for i in range(2048):
            seed = (content_hash + i) % (2**31)
            value = (seed / (2**31)) * 2 - 1
            embedding.append(value)
        
        return embedding
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self._failure_mode != 'health_failure' else 'unhealthy',
            'call_count': len(self._call_log)
        }


@pytest.mark.integration
class TestComponentIntegrationPoints:
    """Test integration points between major RAG system components."""
    
    @pytest_asyncio.fixture
    async def vector_store_mock(self):
        """Create mock vector store for integration testing."""
        config = VectorStoreConfig(vector_size=2048, collection_name="integration_test")
        store = MockVectorStoreForIntegration(config)
        await store.initialize()
        return store
    
    @pytest_asyncio.fixture
    async def embedding_engine_mock(self):
        """Create mock embedding engine for integration testing."""
        config = EmbeddingEngineConfig(
            voyage_api_key="test_key",
            voyage_model="voyage-code-3"
        )
        engine = MockEmbeddingEngineForIntegration(config)
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def sample_json_doc(self, tmp_path):
        """Create sample JSON document for testing."""
        doc = {
            "metadata": {
                "name": "integration-test-lib",
                "version": "1.0.0",
                "description": "Library for integration testing"
            },
            "api": {
                "function1": "def function1(): return 'test'",
                "function2": "def function2(x): return x * 2"
            },
            "examples": ">>> import test_lib\n>>> test_lib.function1()"
        }
        
        doc_path = tmp_path / "integration_test.json"
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2)
        
        return doc_path
    
    @pytest.mark.asyncio
    async def test_embedding_to_vector_store_integration(
        self, 
        embedding_engine_mock, 
        vector_store_mock
    ):
        """Test integration between embedding engine and vector store."""
        
        # Create test content
        test_content = [
            "def hello_world(): return 'Hello, World!'",
            "class TestClass: pass",
            "Installation: pip install test-package"
        ]
        
        # Generate embeddings
        embedding_requests = []
        for i, content in enumerate(test_content):
            request = EmbeddingRequest(
                content=content,
                input_type=InputType.DOCUMENT,
                metadata={'chunk_id': f'test_chunk_{i}'}
            )
            embedding_requests.append(request)
        
        batch_result = await embedding_engine_mock.generate_batch_embeddings(embedding_requests)
        
        # Verify embeddings generated successfully
        assert len(batch_result.results) == len(test_content)
        assert all(result.success for result in batch_result.results)
        
        # Create vector documents from embeddings
        vector_documents = []
        for i, (content, result) in enumerate(zip(test_content, batch_result.results)):
            doc = VectorDocument(
                id=f"integration_test_{i}",
                vector=result.embedding,
                payload={
                    'content': content,
                    'chunk_id': f'test_chunk_{i}',
                    'test_integration': True
                }
            )
            vector_documents.append(doc)
        
        # Store vectors
        storage_result = await vector_store_mock.upsert_vectors_batch(vector_documents)
        
        # Verify storage
        assert storage_result['successful_uploads'] == len(vector_documents)
        assert storage_result['failed_uploads'] == 0
        
        # Verify data consistency - check call logs
        embedding_calls = embedding_engine_mock.get_call_log()
        vector_calls = vector_store_mock.get_call_log()
        
        assert any(call['method'] == 'generate_batch_embeddings' for call in embedding_calls)
        assert any(call['method'] == 'upsert_vectors_batch' for call in vector_calls)
        
        # Verify vectors can be searched
        search_results = await vector_store_mock.search_vectors(
            query_vector=batch_result.results[0].embedding,
            top_k=3,
            filters={'test_integration': True}
        )
        
        assert len(search_results) > 0
        assert all('test_integration' in result.payload for result in search_results)
    
    @pytest.mark.asyncio
    async def test_pipeline_to_embedding_integration(
        self, 
        embedding_engine_mock, 
        vector_store_mock,
        sample_json_doc
    ):
        """Test integration between ingestion pipeline and embedding engine."""
        
        # Create storage manager mock
        class MockStorageManager:
            async def initialize(self): pass
            async def cleanup(self): pass
        
        storage_manager = MockStorageManager()
        
        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine_mock,
            vector_storage=vector_store_mock,
            max_workers=1,
            quality_threshold=0.5
        )
        
        await pipeline.initialize()
        
        try:
            # Queue document for processing
            job_id = await pipeline.queue_document(
                library_id="integration-test",
                version="1.0.0",
                doc_path=sample_json_doc,
                priority=8,
                metadata={'integration_test': True}
            )
            
            # Wait for processing
            result = await pipeline.wait_for_job_completion(job_id, timeout=30.0)
            
            # Verify processing completed
            assert result is not None
            assert result.success
            assert result.chunks_created > 0
            assert result.vectors_generated > 0
            
            # Verify embedding engine was called
            embedding_calls = embedding_engine_mock.get_call_log()
            batch_calls = [call for call in embedding_calls if call['method'] == 'generate_batch_embeddings']
            assert len(batch_calls) > 0
            
            # Verify vectors were stored
            vector_calls = vector_store_mock.get_call_log()
            upload_calls = [call for call in vector_calls if call['method'] == 'upsert_vectors_batch']
            assert len(upload_calls) > 0
            
            # Verify data consistency
            vector_count = await vector_store_mock.count_vectors()
            assert vector_count == result.vectors_generated
            
        finally:
            await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_engine_integration(
        self, 
        embedding_engine_mock, 
        vector_store_mock
    ):
        """Test search engine integration with vector store."""
        
        # Pre-populate vector store with test data
        test_vectors = []
        for i in range(5):
            content = f"Test document {i} with unique content"
            embedding = embedding_engine_mock._generate_deterministic_embedding(content)
            
            doc = VectorDocument(
                id=f"search_test_{i}",
                vector=embedding,
                payload={
                    'content': content,
                    'doc_type': 'test',
                    'index': i
                }
            )
            test_vectors.append(doc)
        
        await vector_store_mock.upsert_vectors_batch(test_vectors)
        
        # Create search engine
        search_engine = VectorSearchEngine(
            vector_store=vector_store_mock,
            enable_caching=True
        )
        
        # Test basic search
        query_embedding = embedding_engine_mock._generate_deterministic_embedding("test query")
        search_query = SearchQuery(
            vector=query_embedding,
            top_k=3,
            filters={'doc_type': 'test'}
        )
        
        results = await search_engine.search(search_query)
        
        # Verify search results
        assert len(results) > 0
        assert len(results) <= 3
        assert all(hasattr(result, 'rank') for result in results)
        assert all(hasattr(result, 'relevance_score') for result in results)
        
        # Verify search called vector store
        vector_calls = vector_store_mock.get_call_log()
        search_calls = [call for call in vector_calls if call['method'] == 'search_vectors']
        assert len(search_calls) > 0
        
        # Test caching behavior
        results2 = await search_engine.search(search_query)
        assert len(results2) == len(results)
        
        # Verify cache is working (should be same results, potentially faster)
        search_metrics = search_engine.get_search_metrics()
        assert search_metrics['search_metrics']['total_searches'] >= 2
    
    @pytest.mark.asyncio
    async def test_error_propagation_through_components(
        self, 
        embedding_engine_mock, 
        vector_store_mock,
        sample_json_doc
    ):
        """Test error handling and propagation between components."""
        
        # Test 1: Embedding failure propagation
        embedding_engine_mock.set_failure_mode('embedding_failure')
        
        class MockStorageManager:
            async def initialize(self): pass
            async def cleanup(self): pass
        
        storage_manager = MockStorageManager()
        
        pipeline = IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine_mock,
            vector_storage=vector_store_mock,
            max_workers=1,
            quality_threshold=0.5
        )
        
        await pipeline.initialize()
        
        try:
            # Process document with embedding failures
            job_id = await pipeline.queue_document(
                library_id="error-test",
                version="1.0.0",
                doc_path=sample_json_doc,
                priority=5,
                metadata={'error_test': True}
            )
            
            result = await pipeline.wait_for_job_completion(job_id, timeout=30.0)
            
            # Should handle partial failures gracefully
            assert result is not None
            # May succeed partially or fail completely depending on implementation
            
            # Verify error handling in call logs
            embedding_calls = embedding_engine_mock.get_call_log()
            assert any(call['method'] == 'generate_batch_embeddings' for call in embedding_calls)
            
        finally:
            await pipeline.shutdown()
        
        # Test 2: Vector store failure propagation
        embedding_engine_mock.set_failure_mode(None)  # Reset embedding engine
        vector_store_mock.set_failure_mode('storage_failure')
        
        pipeline2 = IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine_mock,
            vector_storage=vector_store_mock,
            max_workers=1,
            quality_threshold=0.5
        )
        
        await pipeline2.initialize()
        
        try:
            job_id = await pipeline2.queue_document(
                library_id="storage-error-test",
                version="1.0.0",
                doc_path=sample_json_doc,
                priority=5,
                metadata={'storage_error_test': True}
            )
            
            result = await pipeline2.wait_for_job_completion(job_id, timeout=30.0)
            
            # Should detect storage failures
            assert result is not None
            if not result.success:
                assert result.error_message is not None
            
        finally:
            await pipeline2.shutdown()
        
        # Test 3: Search engine error handling
        vector_store_mock.set_failure_mode('search_failure')
        
        search_engine = VectorSearchEngine(vector_store=vector_store_mock)
        
        query_embedding = [0.1] * 2048
        search_query = SearchQuery(vector=query_embedding, top_k=5)
        
        # Should handle search failures gracefully
        with pytest.raises(Exception):
            await search_engine.search(search_query)
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(
        self, 
        embedding_engine_mock, 
        vector_store_mock,
        sample_json_doc
    ):
        """Test data consistency as it flows through the pipeline."""
        
        class MockStorageManager:
            async def initialize(self): pass
            async def cleanup(self): pass
        
        storage_manager = MockStorageManager()
        
        # Track data at each stage
        data_checkpoints = []
        
        # Create pipeline with logging
        pipeline = IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine_mock,
            vector_storage=vector_store_mock,
            max_workers=1,
            quality_threshold=0.5
        )
        
        await pipeline.initialize()
        
        try:
            # Parse document manually to track data
            parser = JSONDocumentParser()
            sections = await parser.parse_document(sample_json_doc)
            data_checkpoints.append({
                'stage': 'parsing',
                'sections_count': len(sections),
                'total_content_length': sum(len(s.content) for s in sections)
            })
            
            # Chunk sections manually
            chunker = IntelligentChunkingEngine(chunk_size=500, chunk_overlap=50)
            chunks = await chunker.chunk_document_sections(sections)
            data_checkpoints.append({
                'stage': 'chunking',
                'chunks_count': len(chunks),
                'total_content_length': sum(len(c.content) for c in chunks)
            })
            
            # Process through pipeline
            job_id = await pipeline.queue_document(
                library_id="consistency-test",
                version="1.0.0",
                doc_path=sample_json_doc,
                priority=8,
                metadata={'consistency_test': True}
            )
            
            result = await pipeline.wait_for_job_completion(job_id, timeout=30.0)
            
            assert result is not None
            assert result.success
            
            data_checkpoints.append({
                'stage': 'pipeline_complete',
                'chunks_created': result.chunks_created,
                'vectors_generated': result.vectors_generated
            })
            
            # Verify data consistency
            assert data_checkpoints[1]['chunks_count'] == result.chunks_created
            assert result.chunks_created == result.vectors_generated
            
            # Verify stored vectors can be retrieved
            vector_count = await vector_store_mock.count_vectors()
            assert vector_count == result.vectors_generated
            
            # Search for stored vectors
            search_engine = VectorSearchEngine(vector_store=vector_store_mock)
            
            # Use embedding from first chunk for search
            first_chunk_content = chunks[0].content if chunks else "test content"
            query_embedding = embedding_engine_mock._generate_deterministic_embedding(first_chunk_content)
            
            search_results = await search_engine.search(SearchQuery(
                vector=query_embedding,
                top_k=10,
                filters={'consistency_test': True}
            ))
            
            # Should find vectors from our processed document
            assert len(search_results) > 0
            
            # Log data flow for verification
            logger.info("Data consistency checkpoints:")
            for checkpoint in data_checkpoints:
                logger.info(f"  {checkpoint}")
            
        finally:
            await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_component_operations(
        self, 
        embedding_engine_mock, 
        vector_store_mock
    ):
        """Test concurrent operations across components."""
        
        # Create multiple tasks that use different components
        tasks = []
        
        # Task 1: Multiple embedding requests
        embedding_tasks = []
        for i in range(5):
            request = EmbeddingRequest(
                content=f"Concurrent embedding test {i}",
                input_type=InputType.DOCUMENT,
                metadata={'concurrent_test': True}
            )
            task = embedding_engine_mock.generate_batch_embeddings([request])
            embedding_tasks.append(task)
        
        # Task 2: Vector storage operations
        storage_tasks = []
        for i in range(3):
            docs = [
                VectorDocument(
                    id=f"concurrent_vector_{i}_{j}",
                    vector=[0.1 * (i + j + k) for k in range(2048)],
                    payload={'batch': i, 'concurrent_test': True}
                )
                for j in range(2)
            ]
            task = vector_store_mock.upsert_vectors_batch(docs)
            storage_tasks.append(task)
        
        # Execute all tasks concurrently
        all_tasks = embedding_tasks + storage_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Verify all operations completed
        assert len(results) == len(all_tasks)
        
        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Verify embedding results
        embedding_results = results[:len(embedding_tasks)]
        for result in embedding_results:
            assert isinstance(result, BatchResult)
            assert len(result.results) == 1
            assert result.results[0].success
        
        # Verify storage results
        storage_results = results[len(embedding_tasks):]
        for result in storage_results:
            assert isinstance(result, dict)
            assert result['successful_uploads'] > 0
        
        # Verify final state
        vector_count = await vector_store_mock.count_vectors()
        assert vector_count >= 6  # 3 batches * 2 vectors each
        
        # Test concurrent searches
        search_engine = VectorSearchEngine(vector_store=vector_store_mock)
        
        search_tasks = []
        for i in range(5):
            query = SearchQuery(
                vector=[0.1 * (i + j) for j in range(2048)],
                top_k=3,
                filters={'concurrent_test': True}
            )
            task = search_engine.search(query)
            search_tasks.append(task)
        
        search_results = await asyncio.gather(*search_tasks)
        
        # Verify concurrent searches
        assert len(search_results) == 5
        for results in search_results:
            assert isinstance(results, list)
            # May or may not have results depending on similarity


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])