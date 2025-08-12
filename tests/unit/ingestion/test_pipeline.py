"""
Unit tests for the ingestion pipeline components.

Comprehensive test suite covering all ingestion pipeline functionality
with mocks for external dependencies and performance validation.
"""

import pytest
import asyncio
import tempfile
import json
import gzip
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Import ingestion components
from contexter.ingestion.pipeline import IngestionPipeline, ProcessingResult, IngestionStatistics
from contexter.ingestion.trigger_system import AutoIngestionTrigger, IngestionTriggerEvent, TriggerEventType
from contexter.ingestion.processing_queue import IngestionQueue, IngestionJob, WorkerPool, JobStatus
from contexter.ingestion.json_parser import JSONDocumentParser, ParsedSection, SchemaType
from contexter.ingestion.chunking_engine import IntelligentChunkingEngine, DocumentChunk, ChunkingStrategy
from contexter.ingestion.metadata_extractor import MetadataExtractor, ContentAnalysis
from contexter.ingestion.quality_validator import QualityValidator, QualityAssessment
from contexter.ingestion.monitoring import PerformanceMonitor, PerformanceMetrics


class TestJSONDocumentParser:
    """Test JSON document parsing functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create JSON parser instance."""
        return JSONDocumentParser()
    
    @pytest.fixture
    def sample_json_doc(self, tmp_path):
        """Create sample JSON document."""
        doc_data = {
            "metadata": {
                "name": "test-library",
                "version": "1.0.0",
                "description": "Test library for parsing"
            },
            "installation": "pip install test-library",
            "api_reference": {
                "functions": "def test_function(): pass",
                "classes": "class TestClass: pass"
            },
            "examples": ">>> import test_library\n>>> test_library.test_function()"
        }
        
        doc_path = tmp_path / "test_doc.json"
        with open(doc_path, 'w') as f:
            json.dump(doc_data, f)
        
        return doc_path
    
    @pytest.fixture
    def sample_context7_doc(self, tmp_path):
        """Create sample Context7 format document."""
        doc_data = {
            "library_info": {
                "library_id": "test-lib",
                "name": "Test Library",
                "version": "2.0.0",
                "category": "web-framework"
            },
            "contexts": [
                {
                    "content": "This is a comprehensive guide to using Test Library.",
                    "source": "README.md",
                    "token_count": 150
                },
                {
                    "content": "def example_function():\n    return 'Hello World'",
                    "source": "examples.py",
                    "token_count": 50
                }
            ],
            "total_tokens": 200
        }
        
        doc_path = tmp_path / "context7_doc.json"
        with open(doc_path, 'w') as f:
            json.dump(doc_data, f)
        
        return doc_path
    
    @pytest.fixture
    def compressed_doc(self, tmp_path):
        """Create compressed JSON document."""
        doc_data = {
            "name": "compressed-lib",
            "content": "This is compressed documentation content."
        }
        
        doc_path = tmp_path / "compressed_doc.json.gz"
        with gzip.open(doc_path, 'wt') as f:
            json.dump(doc_data, f)
        
        return doc_path
    
    @pytest.mark.asyncio
    async def test_parse_standard_schema(self, parser, sample_json_doc):
        """Test parsing standard library schema."""
        sections = await parser.parse_document(sample_json_doc)
        
        assert len(sections) >= 3
        assert any(section.section_id.endswith('installation') for section in sections)
        assert any(section.section_type == 'code' for section in sections)
        
        # Check metadata preservation
        for section in sections:
            assert section.metadata['library_id'] == 'test-library'
            assert section.metadata['version'] == '1.0.0'
    
    @pytest.mark.asyncio
    async def test_parse_context7_schema(self, parser, sample_context7_doc):
        """Test parsing Context7 output schema."""
        sections = await parser.parse_document(sample_context7_doc)
        
        assert len(sections) == 2
        assert sections[0].metadata['library_id'] == 'test-lib'
        assert sections[0].metadata['context_index'] == 0
        assert sections[1].section_type == 'code'
    
    @pytest.mark.asyncio
    async def test_parse_compressed_document(self, parser, compressed_doc):
        """Test parsing compressed JSON documents."""
        sections = await parser.parse_document(compressed_doc)
        
        assert len(sections) >= 1
        assert 'compressed documentation content' in sections[0].content
    
    @pytest.mark.asyncio
    async def test_schema_detection(self, parser):
        """Test automatic schema detection."""
        # Context7 schema
        context7_content = {
            "library_info": {"name": "test"},
            "contexts": [],
            "total_tokens": 100
        }
        schema_type = parser._detect_schema(context7_content)
        assert schema_type == SchemaType.CONTEXT7_OUTPUT
        
        # Standard schema
        standard_content = {
            "name": "test-lib",
            "version": "1.0.0",
            "installation": "pip install",
            "api_reference": "Functions and classes"
        }
        schema_type = parser._detect_schema(standard_content)
        assert schema_type == SchemaType.STANDARD_LIBRARY
    
    @pytest.mark.asyncio
    async def test_content_type_detection(self, parser):
        """Test content type detection."""
        # Code content
        code_content = "def example():\n    return True"
        content_type = parser._determine_content_type(code_content)
        assert content_type == 'code'
        
        # API content
        api_content = "GET /api/v1/users\nReturns list of users"
        content_type = parser._determine_content_type(api_content)
        assert content_type == 'api'
        
        # Text content
        text_content = "This is a comprehensive guide to using the library."
        content_type = parser._determine_content_type(text_content)
        assert content_type == 'text'


class TestIntelligentChunkingEngine:
    """Test intelligent chunking functionality."""
    
    @pytest.fixture
    def chunking_engine(self):
        """Create chunking engine instance."""
        return IntelligentChunkingEngine(
            chunk_size=500,  # Smaller for testing
            chunk_overlap=100,
            max_chunks_per_doc=50
        )
    
    @pytest.fixture
    def sample_sections(self):
        """Create sample parsed sections."""
        return [
            ParsedSection(
                content="This is a short text section for testing chunking.",
                metadata={'library_id': 'test-lib', 'version': '1.0.0'},
                section_id="section_1",
                section_type="text"
            ),
            ParsedSection(
                content="""
def example_function(param1, param2):
    '''
    Example function that demonstrates code chunking.
    
    Args:
        param1: First parameter
        param2: Second parameter
        
    Returns:
        Combined result
    '''
    result = param1 + param2
    return result

class ExampleClass:
    '''Example class for testing.'''
    
    def __init__(self):
        self.value = 0
    
    def method(self):
        return self.value * 2
                """,
                metadata={'library_id': 'test-lib', 'version': '1.0.0', 'detected_language': 'python'},
                section_id="section_2",
                section_type="code"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_chunk_document_sections(self, chunking_engine, sample_sections):
        """Test chunking complete document sections."""
        chunks = await chunking_engine.chunk_document_sections(sample_sections)
        
        assert len(chunks) >= 2
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)
        assert all(chunk.content_hash for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_code_aware_chunking(self, chunking_engine, sample_sections):
        """Test code-aware chunking strategy."""
        code_section = sample_sections[1]  # Code section
        strategy = chunking_engine._select_chunking_strategy(code_section)
        
        assert strategy == ChunkingStrategy.CODE_AWARE
        
        chunks = await chunking_engine._chunk_code_aware_content(code_section)
        assert len(chunks) >= 1
        assert chunks[0].chunking_strategy == ChunkingStrategy.CODE_AWARE.value
    
    @pytest.mark.asyncio
    async def test_narrative_chunking(self, chunking_engine):
        """Test narrative text chunking."""
        long_text = "This is a test. " * 200  # Create long text
        section = ParsedSection(
            content=long_text,
            metadata={'library_id': 'test-lib', 'version': '1.0.0'},
            section_id="long_text",
            section_type="text"
        )
        
        chunks = await chunking_engine._chunk_narrative_content(section)
        
        # Should create multiple chunks for long text
        assert len(chunks) >= 2
        assert all(chunk.chunk_type == 'text' for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_boundaries(self, chunking_engine):
        """Test semantic boundary detection."""
        code_text = """
def function_one():
    return 1

def function_two():
    return 2
        """
        
        section = ParsedSection(
            content=code_text,
            metadata={'detected_language': 'python'},
            section_id="functions",
            section_type="code"
        )
        
        boundary_pos = chunking_engine._find_code_boundary(code_text, section)
        assert boundary_pos is not None
        assert boundary_pos > 0
    
    @pytest.mark.asyncio
    async def test_chunk_overlap(self, chunking_engine):
        """Test chunk overlap functionality."""
        long_text = "Word " * 1000  # Create text that will need chunking
        section = ParsedSection(
            content=long_text,
            metadata={'library_id': 'test-lib'},
            section_id="overlap_test",
            section_type="text"
        )
        
        chunks = await chunking_engine._chunk_narrative_content(section)
        
        if len(chunks) > 1:
            # Check that overlapping chunks share some content
            assert chunks[1].has_overlap
            assert chunks[1].overlap_start_tokens > 0
    
    def test_token_counting(self, chunking_engine):
        """Test accurate token counting."""
        test_text = "This is a test sentence for token counting."
        tokens = chunking_engine.tokenizer.encode(test_text)
        
        assert len(tokens) > 0
        assert len(tokens) < len(test_text.split()) * 2  # Reasonable token ratio


class TestMetadataExtractor:
    """Test metadata extraction and enrichment."""
    
    @pytest.fixture
    def metadata_extractor(self):
        """Create metadata extractor instance."""
        return MetadataExtractor()
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks."""
        return [
            DocumentChunk(
                chunk_id="test_chunk_1",
                library_id="test-lib",
                version="1.0.0",
                chunk_index=0,
                total_chunks=2,
                content="This is a Python tutorial showing how to use pip install.",
                content_hash="hash1",
                token_count=50,
                char_count=200,
                chunk_type="text",
                programming_language="python",
                semantic_boundary=True,
                metadata={'section': 'tutorial'}
            ),
            DocumentChunk(
                chunk_id="test_chunk_2",
                library_id="test-lib",
                version="1.0.0",
                chunk_index=1,
                total_chunks=2,
                content="""
def example_api_function(param: str) -> dict:
    '''
    API function example.
    
    GET /api/example
    
    Returns response data.
    '''
    return {"result": param}
                """,
                content_hash="hash2",
                token_count=75,
                char_count=300,
                chunk_type="code",
                programming_language="python",
                semantic_boundary=True,
                metadata={'section': 'api_reference'}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_enrich_chunks(self, metadata_extractor, sample_chunks):
        """Test comprehensive chunk enrichment."""
        enriched_chunks = await metadata_extractor.enrich_chunks(sample_chunks)
        
        assert len(enriched_chunks) == len(sample_chunks)
        
        for chunk in enriched_chunks:
            assert 'content_analysis' in chunk.metadata
            assert 'quality_score' in chunk.metadata
            assert 'enrichment_timestamp' in chunk.metadata
    
    @pytest.mark.asyncio
    async def test_content_analysis(self, metadata_extractor):
        """Test content analysis functionality."""
        test_content = """
This is a Python tutorial showing how to install and use the library.

```python
import example_lib

def use_library():
    return example_lib.process_data()
```

The library provides REST API endpoints:
- GET /api/data
- POST /api/process
        """
        
        analysis = metadata_extractor.content_analyzer.analyze_content(test_content)
        
        assert analysis.primary_language == 'python'
        assert analysis.has_code_examples
        assert analysis.has_api_references
        assert analysis.readability_score > 0
        assert len(analysis.generated_tags) > 0
    
    def test_language_detection(self, metadata_extractor):
        """Test programming language detection."""
        python_code = "import numpy as np\ndef function(): pass"
        analysis = metadata_extractor.content_analyzer.analyze_content(python_code)
        assert analysis.primary_language == 'python'
        
        js_code = "function example() { return true; }"
        analysis = metadata_extractor.content_analyzer.analyze_content(js_code)
        assert analysis.primary_language == 'javascript'
    
    def test_content_type_classification(self, metadata_extractor):
        """Test content type classification."""
        api_content = "GET /api/users\nReturns list of users\nResponse: 200 OK"
        analysis = metadata_extractor.content_analyzer.analyze_content(api_content)
        assert analysis.content_type == 'api_reference'
        
        tutorial_content = "Step 1: Install the package\nStep 2: Import the module"
        analysis = metadata_extractor.content_analyzer.analyze_content(tutorial_content)
        assert analysis.content_type == 'tutorial'


class TestQualityValidator:
    """Test document quality validation."""
    
    @pytest.fixture
    def quality_validator(self):
        """Create quality validator instance."""
        return QualityValidator()
    
    @pytest.fixture
    def high_quality_doc(self, tmp_path):
        """Create high-quality document."""
        doc_data = {
            "metadata": {
                "name": "comprehensive-lib",
                "version": "2.1.0",
                "description": "A comprehensive library with excellent documentation"
            },
            "installation": "pip install comprehensive-lib",
            "getting_started": "Follow this tutorial to get started with the library.",
            "api_reference": {
                "functions": "def process_data(data): pass",
                "examples": ">>> process_data({'key': 'value'})"
            },
            "examples": "Complete working examples with explanations.",
            "troubleshooting": "Common issues and solutions."
        }
        
        doc_path = tmp_path / "high_quality.json"
        with open(doc_path, 'w') as f:
            json.dump(doc_data, f)
        
        return doc_path
    
    @pytest.fixture
    def low_quality_doc(self, tmp_path):
        """Create low-quality document."""
        doc_data = {
            "content": "Brief content."
        }
        
        doc_path = tmp_path / "low_quality.json"
        with open(doc_path, 'w') as f:
            json.dump(doc_data, f)
        
        return doc_path
    
    @pytest.mark.asyncio
    async def test_high_quality_assessment(self, quality_validator, high_quality_doc):
        """Test assessment of high-quality document."""
        assessment = await quality_validator.assess_document_comprehensive(high_quality_doc)
        
        assert assessment.overall_score > 0.7
        assert assessment.completeness_score > 0.6
        assert assessment.structure_score > 0.5
        assert assessment.has_installation_guide
        assert len(assessment.issues) == 0
    
    @pytest.mark.asyncio
    async def test_low_quality_assessment(self, quality_validator, low_quality_doc):
        """Test assessment of low-quality document."""
        assessment = await quality_validator.assess_document_comprehensive(low_quality_doc)
        
        assert assessment.overall_score < 0.5
        assert assessment.completeness_score < 0.5
        assert len(assessment.issues) > 0
        assert len(assessment.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_quality_score_only(self, quality_validator, high_quality_doc):
        """Test quick quality score assessment."""
        score = await quality_validator.assess_document_quality(high_quality_doc)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent quality


class TestIngestionQueue:
    """Test ingestion queue functionality."""
    
    @pytest.fixture
    def ingestion_queue(self):
        """Create ingestion queue instance."""
        return IngestionQueue(max_size=100)
    
    @pytest.mark.asyncio
    async def test_queue_operations(self, ingestion_queue):
        """Test basic queue operations."""
        # Test putting job
        job_id = await ingestion_queue.put(
            library_id="test-lib",
            version="1.0.0",
            doc_path=Path("/test/path"),
            priority=5,
            metadata={"test": True}
        )
        
        assert job_id
        
        # Test getting job
        job = await ingestion_queue.get(timeout=1.0)
        assert job is not None
        assert job.library_id == "test-lib"
        assert job.priority == 5
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, ingestion_queue):
        """Test priority-based job ordering."""
        # Add jobs with different priorities
        low_priority_id = await ingestion_queue.put(
            "lib1", "1.0", Path("/test1"), 1, {}
        )
        high_priority_id = await ingestion_queue.put(
            "lib2", "1.0", Path("/test2"), 10, {}
        )
        medium_priority_id = await ingestion_queue.put(
            "lib3", "1.0", Path("/test3"), 5, {}
        )
        
        # Should get high priority first
        job1 = await ingestion_queue.get(timeout=1.0)
        assert job1.priority == 10
        
        job2 = await ingestion_queue.get(timeout=1.0)
        assert job2.priority == 5
        
        job3 = await ingestion_queue.get(timeout=1.0)
        assert job3.priority == 1
    
    @pytest.mark.asyncio
    async def test_job_status_tracking(self, ingestion_queue):
        """Test job status tracking and updates."""
        job_id = await ingestion_queue.put("test", "1.0", Path("/test"), 5, {})
        
        # Check initial status
        status = await ingestion_queue.get_job_status(job_id)
        assert status['status'] == JobStatus.QUEUED.value
        
        # Update status
        success = await ingestion_queue.update_job_status(
            job_id, JobStatus.PROCESSING, processing_time=1.5
        )
        assert success
        
        # Check updated status
        status = await ingestion_queue.get_job_status(job_id)
        assert status['status'] == JobStatus.PROCESSING.value


class TestAutoIngestionTrigger:
    """Test auto-ingestion trigger system."""
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Create mock storage manager."""
        storage_manager = Mock()
        storage_manager.retrieve_documentation = AsyncMock(return_value={
            'metadata': {'name': 'test-lib', 'version': '1.0.0'}
        })
        return storage_manager
    
    @pytest.fixture
    def mock_ingestion_pipeline(self):
        """Create mock ingestion pipeline."""
        pipeline = Mock()
        pipeline.queue_document = AsyncMock(return_value="job_123")
        return pipeline
    
    @pytest.fixture
    def trigger_system(self, mock_storage_manager, mock_ingestion_pipeline):
        """Create trigger system instance."""
        return AutoIngestionTrigger(
            storage_manager=mock_storage_manager,
            ingestion_pipeline=mock_ingestion_pipeline,
            quality_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_download_completion_trigger(self, trigger_system):
        """Test download completion trigger."""
        doc_path = Path("/test/doc.json")
        metadata = {"library_id": "test-lib", "star_count": 1000}
        
        # Mock quality validator
        with patch('contexter.ingestion.trigger_system.QualityValidator') as mock_validator_class:
            mock_validator = Mock()
            mock_validator.assess_document_quality = AsyncMock(return_value=0.8)
            trigger_system.quality_validator = mock_validator
            
            success = await trigger_system.on_download_complete(
                "test-lib", "1.0.0", doc_path, metadata
            )
            
            assert success
    
    @pytest.mark.asyncio
    async def test_manual_trigger(self, trigger_system):
        """Test manual ingestion trigger."""
        doc_path = Path("/test/manual_doc.json")
        
        success = await trigger_system.trigger_manual_ingestion(
            "manual-lib", "2.0.0", doc_path, {"manual": True}, priority_boost=5
        )
        
        assert success
    
    def test_priority_calculation(self, trigger_system):
        """Test priority calculation logic."""
        # High priority metadata
        high_priority_metadata = {
            'star_count': 10000,
            'trust_score': 0.9,
            'category': 'web-framework'
        }
        priority = trigger_system._calculate_priority(high_priority_metadata, 0.9, 0)
        assert priority >= 8
        
        # Low priority metadata
        low_priority_metadata = {
            'star_count': 10,
            'trust_score': 0.3,
            'category': 'utility'
        }
        priority = trigger_system._calculate_priority(low_priority_metadata, 0.5, 0)
        assert priority <= 5


class TestIngestionPipeline:
    """Test complete ingestion pipeline integration."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for pipeline."""
        storage_manager = Mock()
        storage_manager.initialize = AsyncMock()
        storage_manager.cleanup = AsyncMock()
        
        embedding_engine = Mock()
        embedding_engine.initialize = AsyncMock()
        embedding_engine.shutdown = AsyncMock()
        embedding_engine.generate_batch_embeddings = AsyncMock()
        
        vector_storage = Mock()
        vector_storage.initialize = AsyncMock()
        vector_storage.cleanup = AsyncMock()
        vector_storage.upsert_vectors_batch = AsyncMock(return_value={
            'successful_uploads': 5
        })
        
        return storage_manager, embedding_engine, vector_storage
    
    @pytest.fixture
    def ingestion_pipeline(self, mock_dependencies):
        """Create ingestion pipeline instance."""
        storage_manager, embedding_engine, vector_storage = mock_dependencies
        
        return IngestionPipeline(
            storage_manager=storage_manager,
            embedding_engine=embedding_engine,
            vector_storage=vector_storage,
            max_workers=2,
            quality_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, ingestion_pipeline):
        """Test pipeline initialization."""
        await ingestion_pipeline.initialize()
        
        assert ingestion_pipeline._initialized
        assert ingestion_pipeline.statistics.start_time is not None
    
    @pytest.mark.asyncio
    async def test_document_queuing(self, ingestion_pipeline):
        """Test document queuing functionality."""
        await ingestion_pipeline.initialize()
        
        job_id = await ingestion_pipeline.queue_document(
            library_id="test-lib",
            version="1.0.0",
            doc_path=Path("/test/doc.json"),
            priority=5,
            metadata={"test": True}
        )
        
        assert job_id
        
        # Check job status
        status = await ingestion_pipeline.get_job_status(job_id)
        assert status is not None
        assert status['library_id'] == "test-lib"
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, ingestion_pipeline):
        """Test statistics tracking."""
        result = ProcessingResult(
            job_id="test_job",
            library_id="test-lib",
            version="1.0.0",
            success=True,
            processing_time=2.5,
            sections_parsed=3,
            chunks_created=10,
            vectors_generated=10,
            avg_quality_score=0.8,
            min_quality_score=0.7,
            max_quality_score=0.9
        )
        
        ingestion_pipeline.statistics.update_from_result(result)
        
        assert ingestion_pipeline.statistics.total_jobs_successful == 1
        assert ingestion_pipeline.statistics.total_chunks_created == 10
        assert ingestion_pipeline.statistics.avg_quality_score == 0.8
    
    @pytest.mark.asyncio
    async def test_health_check(self, ingestion_pipeline):
        """Test pipeline health check."""
        await ingestion_pipeline.initialize()
        
        health = await ingestion_pipeline.health_check()
        
        assert 'status' in health
        assert 'components' in health
        assert 'metrics' in health


# Performance and load testing
class TestPerformance:
    """Test performance characteristics and load handling."""
    
    @pytest.mark.asyncio
    async def test_chunking_performance(self):
        """Test chunking engine performance."""
        chunking_engine = IntelligentChunkingEngine(chunk_size=1000)
        
        # Create large document
        large_content = "This is a test sentence. " * 1000
        section = ParsedSection(
            content=large_content,
            metadata={'library_id': 'perf-test'},
            section_id="large_section",
            section_type="text"
        )
        
        start_time = asyncio.get_event_loop().time()
        chunks = await chunking_engine.chunk_document_sections([section])
        end_time = asyncio.get_event_loop().time()
        
        processing_time = end_time - start_time
        
        assert processing_time < 5.0  # Should process within 5 seconds
        assert len(chunks) > 0
        assert all(chunk.token_count <= 1200 for chunk in chunks)  # Respect size limits
    
    @pytest.mark.asyncio
    async def test_queue_throughput(self):
        """Test queue throughput under load."""
        queue = IngestionQueue(max_size=1000)
        
        # Add many jobs quickly
        start_time = asyncio.get_event_loop().time()
        
        job_ids = []
        for i in range(100):
            job_id = await queue.put(
                f"lib_{i}", "1.0.0", Path(f"/test_{i}"), i % 10, {}
            )
            job_ids.append(job_id)
        
        end_time = asyncio.get_event_loop().time()
        
        queuing_time = end_time - start_time
        assert queuing_time < 2.0  # Should queue 100 jobs within 2 seconds
        
        # Verify all jobs queued
        stats = queue.get_statistics()
        assert stats['total_jobs'] == 100
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage remains reasonable."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many chunks
        chunking_engine = IntelligentChunkingEngine()
        large_sections = []
        
        for i in range(50):
            section = ParsedSection(
                content="Test content " * 500,
                metadata={'library_id': f'memory_test_{i}'},
                section_id=f"section_{i}",
                section_type="text"
            )
            large_sections.append(section)
        
        all_chunks = await chunking_engine.chunk_document_sections(large_sections)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        del all_chunks
        del large_sections
        gc.collect()
        
        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])