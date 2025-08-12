"""
End-to-end integration tests for the ingestion pipeline.

Tests complete workflows from document ingestion to vector storage
with real components and realistic data scenarios.
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from contexter.ingestion.pipeline import IngestionPipeline
from contexter.ingestion.json_parser import JSONDocumentParser
from contexter.ingestion.chunking_engine import IntelligentChunkingEngine
from contexter.ingestion.metadata_extractor import MetadataExtractor
from contexter.ingestion.monitoring import PerformanceMonitor


class MockStorageManager:
    """Mock storage manager for integration tests."""
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass


class MockEmbeddingEngine:
    """Mock embedding engine that simulates realistic behavior."""
    
    def __init__(self):
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    async def generate_batch_embeddings(self, requests):
        """Generate mock embeddings with realistic latency."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        from contexter.models.embedding_models import EmbeddingResult, BatchResult
        
        results = []
        for req in requests:
            # Generate mock embedding vector
            embedding = [0.1] * 2048  # 2048-dimensional vector
            
            result = EmbeddingResult(
                content_hash=req.content_hash,
                embedding=embedding,
                model="mock-model",
                dimensions=2048,
                processing_time=0.1,
                cache_hit=False
            )
            results.append(result)
        
        return BatchResult(
            batch_id="mock_batch",
            results=results,
            processing_time=0.1
        )
    
    async def health_check(self):
        return {
            'status': 'healthy',
            'performance': {
                'average_latency_ms': 100
            }
        }


class MockVectorStorage:
    """Mock vector storage that simulates database operations."""
    
    def __init__(self):
        self.vectors = {}
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def upsert_vectors_batch(self, documents):
        """Mock vector storage with realistic latency."""
        await asyncio.sleep(0.05)  # Simulate storage time
        
        for doc in documents:
            self.vectors[doc.id] = doc
        
        return {
            'successful_uploads': len(documents),
            'failed_uploads': 0,
            'total_time': 0.05
        }
    
    def get_health_status(self):
        return {
            'status': 'healthy',
            'metrics': {
                'search_latency_p95': 0.02
            }
        }


class TestEndToEndIngestion:
    """Test complete end-to-end ingestion workflows."""
    
    @pytest.fixture
    def test_documents(self, tmp_path):
        """Create realistic test documents."""
        documents = {}
        
        # Python library documentation
        python_doc = {
            "metadata": {
                "name": "requests",
                "version": "2.28.0",
                "description": "HTTP library for Python",
                "category": "http-client",
                "star_count": 50000,
                "trust_score": 0.95
            },
            "installation": "pip install requests",
            "getting_started": """
Getting Started with Requests

The requests library is the de facto standard for making HTTP requests in Python.
It abstracts the complexities of making requests behind a beautiful, simple API.

Basic usage:
>>> import requests
>>> r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
>>> r.status_code
200
            """,
            "api_reference": {
                "get": """
requests.get(url, params=None, **kwargs)

Sends a GET request.

Parameters:
    url: URL for the new Request object
    params: Dictionary or bytes to send in query string
    **kwargs: Optional arguments that request takes

Returns:
    Response object

Example:
>>> r = requests.get('https://httpbin.org/get')
>>> r.json()
{'args': {}, 'headers': {...}, 'origin': '...', 'url': '...'}
                """,
                "post": """
requests.post(url, data=None, json=None, **kwargs)

Sends a POST request.

Parameters:
    url: URL for the new Request object
    data: Dictionary, list of tuples, bytes, or file-like object
    json: JSON serializable object
    **kwargs: Optional arguments that request takes

Returns:
    Response object
                """
            },
            "examples": {
                "basic": """
import requests

# GET request
response = requests.get('https://api.github.com/users/octocat')
print(response.json())

# POST request with JSON
data = {'key': 'value'}
response = requests.post('https://httpbin.org/post', json=data)
print(response.status_code)
                """,
                "authentication": """
import requests
from requests.auth import HTTPBasicAuth

# Basic authentication
response = requests.get(
    'https://api.github.com/user',
    auth=HTTPBasicAuth('username', 'password')
)

# Token authentication
headers = {'Authorization': 'token YOUR_TOKEN'}
response = requests.get('https://api.github.com/user', headers=headers)
                """
            }
        }
        
        python_doc_path = tmp_path / "requests_doc.json"
        with open(python_doc_path, 'w') as f:
            json.dump(python_doc, f, indent=2)
        documents['python_lib'] = python_doc_path
        
        # JavaScript framework documentation
        js_doc = {
            "library_info": {
                "library_id": "express",
                "name": "Express.js",
                "version": "4.18.0",
                "category": "web-framework",
                "star_count": 60000,
                "trust_score": 0.98
            },
            "contexts": [
                {
                    "content": """
Express.js is a minimal and flexible Node.js web application framework
that provides a robust set of features for web and mobile applications.

Installation:
npm install express

Basic Example:
const express = require('express')
const app = express()
const port = 3000

app.get('/', (req, res) => {
  res.send('Hello World!')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
                    """,
                    "source": "README.md",
                    "token_count": 120
                },
                {
                    "content": """
Routing

Routing refers to how an application responds to client requests.
You define routing using methods of the Express app object.

Basic routes:
app.get('/', handler)      // GET method route
app.post('/', handler)     // POST method route
app.put('/', handler)      // PUT method route
app.delete('/', handler)   // DELETE method route

Route parameters:
app.get('/users/:userId/books/:bookId', (req, res) => {
  res.send(req.params)
})
                    """,
                    "source": "routing.md",
                    "token_count": 95
                }
            ],
            "total_tokens": 215
        }
        
        js_doc_path = tmp_path / "express_doc.json"
        with open(js_doc_path, 'w') as f:
            json.dump(js_doc, f, indent=2)
        documents['js_framework'] = js_doc_path
        
        return documents
    
    @pytest.fixture
    async def ingestion_pipeline(self):
        """Create real ingestion pipeline with mock dependencies."""
        storage_manager = MockStorageManager()
        embedding_engine = MockEmbeddingEngine()
        vector_storage = MockVectorStorage()
        
        pipeline = IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine,
            vector_storage=vector_storage,
            max_workers=2,
            quality_threshold=0.6
        )
        
        await pipeline.initialize()
        yield pipeline
        await pipeline.shutdown()
    
    @pytest.mark.asyncio
    async def test_complete_ingestion_workflow(self, ingestion_pipeline, test_documents):
        """Test complete ingestion workflow from JSON to vectors."""
        
        # Process Python library document
        python_doc_path = test_documents['python_lib']
        
        job_id = await ingestion_pipeline.queue_document(
            library_id="requests",
            version="2.28.0",
            doc_path=python_doc_path,
            priority=8,
            metadata={
                "library_name": "requests",
                "category": "http-client",
                "star_count": 50000,
                "trust_score": 0.95
            }
        )
        
        # Wait for processing to complete
        result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=30.0)
        
        assert result is not None
        assert result.success
        assert result.sections_parsed > 0
        assert result.chunks_created > 0
        assert result.vectors_generated > 0
        assert result.avg_quality_score > 0.5
        
        # Verify vectors were stored
        vector_storage = ingestion_pipeline.vector_storage
        assert len(vector_storage.vectors) == result.vectors_generated
        
        # Check processing time is reasonable
        assert result.processing_time < 10.0
    
    @pytest.mark.asyncio
    async def test_multiple_document_processing(self, ingestion_pipeline, test_documents):
        """Test processing multiple documents concurrently."""
        
        job_ids = []
        
        # Queue both documents
        for doc_name, doc_path in test_documents.items():
            job_id = await ingestion_pipeline.queue_document(
                library_id=doc_name,
                version="1.0.0",
                doc_path=doc_path,
                priority=5,
                metadata={"test": True}
            )
            job_ids.append(job_id)
        
        # Wait for all jobs to complete
        results = []
        for job_id in job_ids:
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=30.0)
            assert result is not None
            results.append(result)
        
        # Verify all processed successfully
        assert all(result.success for result in results)
        assert sum(result.vectors_generated for result in results) > 0
        
        # Check total vectors stored
        total_vectors = len(ingestion_pipeline.vector_storage.vectors)
        expected_vectors = sum(result.vectors_generated for result in results)
        assert total_vectors == expected_vectors
    
    @pytest.mark.asyncio
    async def test_error_handling_malformed_json(self, ingestion_pipeline, tmp_path):
        """Test error handling with malformed JSON."""
        
        # Create malformed JSON file
        malformed_path = tmp_path / "malformed.json"
        with open(malformed_path, 'w') as f:
            f.write('{"incomplete": json')
        
        job_id = await ingestion_pipeline.queue_document(
            library_id="malformed",
            version="1.0.0",
            doc_path=malformed_path,
            priority=1,
            metadata={}
        )
        
        result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=10.0)
        
        assert result is not None
        assert not result.success
        assert result.error_message is not None
        assert "json" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, ingestion_pipeline, test_documents):
        """Test performance monitoring during processing."""
        
        # Start performance monitoring
        monitor = PerformanceMonitor(ingestion_pipeline, monitoring_interval=1.0)
        await monitor.start_monitoring()
        
        try:
            # Process a document
            python_doc_path = test_documents['python_lib']
            
            job_id = await ingestion_pipeline.queue_document(
                library_id="perf_test",
                version="1.0.0",
                doc_path=python_doc_path,
                priority=5,
                metadata={}
            )
            
            # Wait for processing
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=15.0)
            assert result.success
            
            # Wait for monitoring to collect metrics
            await asyncio.sleep(2.0)
            
            # Check monitoring data
            summary = monitor.get_performance_summary()
            assert summary['status'] in ['healthy', 'degraded']
            assert 'latest_metrics' in summary
            assert len(monitor.metrics_history) > 0
            
        finally:
            await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_quality_threshold_filtering(self, ingestion_pipeline, tmp_path):
        """Test quality threshold filtering."""
        
        # Create very low quality document
        low_quality_doc = {
            "x": "y"  # Minimal content
        }
        
        low_quality_path = tmp_path / "low_quality.json"
        with open(low_quality_path, 'w') as f:
            json.dump(low_quality_doc, f)
        
        # Use high quality threshold
        ingestion_pipeline.quality_threshold = 0.8
        
        # Manually trigger quality validation
        from contexter.ingestion.quality_validator import QualityValidator
        validator = QualityValidator()
        
        quality_score = await validator.assess_document_quality(low_quality_path)
        
        # Quality should be below threshold
        assert quality_score < 0.8
        
        # Document should still process (pipeline doesn't reject, just logs warning)
        job_id = await ingestion_pipeline.queue_document(
            library_id="low_quality",
            version="1.0.0",
            doc_path=low_quality_path,
            priority=1,
            metadata={}
        )
        
        result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=10.0)
        
        # Should complete but with low quality scores
        assert result.success
        assert result.avg_quality_score < 0.8


class TestComponentIntegration:
    """Test integration between pipeline components."""
    
    @pytest.mark.asyncio
    async def test_parser_to_chunker_integration(self, tmp_path):
        """Test integration between JSON parser and chunking engine."""
        
        # Create test document
        test_doc = {
            "introduction": "This is a comprehensive introduction to the library.",
            "installation": "Install using: pip install library",
            "api": {
                "function1": "def function1(): pass",
                "function2": "def function2(): pass"
            },
            "examples": ">>> import library\n>>> library.function1()"
        }
        
        doc_path = tmp_path / "integration_test.json"
        with open(doc_path, 'w') as f:
            json.dump(test_doc, f)
        
        # Parse document
        parser = JSONDocumentParser()
        sections = await parser.parse_document(doc_path)
        
        assert len(sections) >= 3
        
        # Chunk sections
        chunker = IntelligentChunkingEngine(chunk_size=200, chunk_overlap=50)
        chunks = await chunker.chunk_document_sections(sections)
        
        assert len(chunks) >= len(sections)
        assert all(chunk.token_count > 0 for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_chunker_to_extractor_integration(self):
        """Test integration between chunking engine and metadata extractor."""
        
        # Create sample chunks
        from contexter.ingestion.chunking_engine import DocumentChunk
        
        chunks = [
            DocumentChunk(
                chunk_id="test_1",
                library_id="test-lib",
                version="1.0.0",
                chunk_index=0,
                total_chunks=2,
                content="""
def calculate_sum(a, b):
    '''Calculate sum of two numbers.'''
    return a + b

# Example usage
result = calculate_sum(5, 3)
print(result)  # Output: 8
                """,
                content_hash="hash1",
                token_count=50,
                char_count=200,
                chunk_type="code",
                programming_language="python",
                semantic_boundary=True,
                metadata={'section': 'examples'}
            ),
            DocumentChunk(
                chunk_id="test_2",
                library_id="test-lib",
                version="1.0.0",
                chunk_index=1,
                total_chunks=2,
                content="""
Installation Guide

To install this library, run the following command:

pip install test-lib

Requirements:
- Python 3.7+
- numpy
- requests
                """,
                content_hash="hash2",
                token_count=40,
                char_count=150,
                chunk_type="text",
                programming_language=None,
                semantic_boundary=True,
                metadata={'section': 'installation'}
            )
        ]
        
        # Extract metadata
        extractor = MetadataExtractor()
        enriched_chunks = await extractor.enrich_chunks(chunks)
        
        assert len(enriched_chunks) == len(chunks)
        
        for chunk in enriched_chunks:
            assert 'content_analysis' in chunk.metadata
            assert 'quality_score' in chunk.metadata
            
            analysis = chunk.metadata['content_analysis']
            assert 'language_detection' in analysis
            assert 'content_classification' in analysis
            assert 'features' in analysis
    
    @pytest.mark.asyncio
    async def test_full_component_chain(self, tmp_path):
        """Test full component chain from parsing to metadata extraction."""
        
        # Create comprehensive test document
        test_doc = {
            "metadata": {
                "name": "comprehensive-lib",
                "version": "2.0.0",
                "description": "A comprehensive library for testing"
            },
            "getting_started": """
# Getting Started

Welcome to comprehensive-lib! This guide will help you get up and running.

## Installation

Install the library using pip:

```bash
pip install comprehensive-lib
```

## Quick Start

Here's a simple example to get you started:

```python
import comprehensive_lib as cl

# Create a new instance
instance = cl.ComprehensiveClass()

# Use the instance
result = instance.process_data({'key': 'value'})
print(result)
```
            """,
            "api_reference": {
                "ComprehensiveClass": """
class ComprehensiveClass:
    '''Main class for comprehensive processing.'''
    
    def __init__(self, config=None):
        '''Initialize the class.
        
        Args:
            config: Optional configuration dictionary
        '''
        self.config = config or {}
    
    def process_data(self, data):
        '''Process input data.
        
        Args:
            data: Dictionary containing data to process
            
        Returns:
            Processed data dictionary
        '''
        # Implementation details...
        return {'processed': data}
                """,
                "utility_functions": """
def validate_input(data):
    '''Validate input data format.'''
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    return True

def format_output(result):
    '''Format processing results.'''
    return {
        'success': True,
        'data': result,
        'timestamp': time.time()
    }
                """
            }
        }
        
        doc_path = tmp_path / "comprehensive_test.json"
        with open(doc_path, 'w') as f:
            json.dump(test_doc, f, indent=2)
        
        # Run through complete chain
        
        # 1. Parse
        parser = JSONDocumentParser()
        sections = await parser.parse_document(doc_path)
        
        assert len(sections) >= 3
        
        # 2. Chunk
        chunker = IntelligentChunkingEngine(chunk_size=300, chunk_overlap=50)
        chunks = await chunker.chunk_document_sections(sections)
        
        assert len(chunks) >= len(sections)
        
        # 3. Extract metadata
        extractor = MetadataExtractor()
        enriched_chunks = await extractor.enrich_chunks(chunks)
        
        assert len(enriched_chunks) == len(chunks)
        
        # Verify end-to-end data flow
        for chunk in enriched_chunks:
            # Should have original section metadata
            assert 'library_id' in chunk.metadata or 'name' in chunk.metadata
            
            # Should have content analysis
            assert 'content_analysis' in chunk.metadata
            analysis = chunk.metadata['content_analysis']
            
            # Should detect Python language in code chunks
            if chunk.chunk_type == 'code':
                assert analysis['language_detection']['primary_language'] == 'python'
            
            # Should detect features
            features = analysis['features']
            if 'pip install' in chunk.content:
                assert features['has_installation_instructions']
            if 'class ' in chunk.content or 'def ' in chunk.content:
                assert features['has_code_examples']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])