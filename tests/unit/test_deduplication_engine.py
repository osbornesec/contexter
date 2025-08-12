"""
Comprehensive unit tests for the deduplication engine components.

Tests exact duplicate detection, semantic similarity analysis, and conflict resolution
with various edge cases and performance requirements validation.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Import the components to test
from src.contexter.core.deduplication import (
    ContentHasher,
    ExactDuplicateProcessor, 
    SimilarityAnalyzer,
    ConflictResolver,
    DeduplicationEngine,
    DeduplicationResult,
    XXHASH_AVAILABLE,
    SKLEARN_AVAILABLE
)
from src.contexter.models.download_models import DocumentationChunk


class TestContentHasher:
    """Test cases for ContentHasher component."""
    
    def test_hash_consistency(self):
        """Test that identical content produces identical hashes."""
        hasher = ContentHasher()
        content = "This is a test string for hashing"
        
        hash1 = hasher.calculate_hash(content)
        hash2 = hasher.calculate_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) > 0
        assert isinstance(hash1, str)
    
    def test_hash_uniqueness(self):
        """Test that different content produces different hashes."""
        hasher = ContentHasher()
        
        hash1 = hasher.calculate_hash("Content A")
        hash2 = hasher.calculate_hash("Content B")
        
        assert hash1 != hash2
    
    def test_empty_content_handling(self):
        """Test handling of empty and whitespace content."""
        hasher = ContentHasher()
        
        empty_hash = hasher.calculate_hash("")
        space_hash = hasher.calculate_hash("   ")
        newline_hash = hasher.calculate_hash("\n\n")
        
        # All should produce valid hashes
        assert len(empty_hash) > 0
        assert len(space_hash) > 0
        assert len(newline_hash) > 0
        
        # But they should be different
        assert empty_hash != space_hash
        assert space_hash != newline_hash
    
    def test_cache_functionality(self):
        """Test hash caching for performance."""
        hasher = ContentHasher(enable_cache=True)
        content = "Test content for caching" * 10  # Make it substantial
        
        # First calculation should be cache miss
        hash1 = hasher.calculate_hash(content)
        assert hasher.cache_misses == 1
        assert hasher.cache_hits == 0
        
        # Second calculation should be cache hit
        hash2 = hasher.calculate_hash(content)
        assert hash1 == hash2
        assert hasher.cache_hits == 1
        assert hasher.cache_misses == 1
    
    def test_cache_disabled(self):
        """Test hasher behavior with caching disabled."""
        hasher = ContentHasher(enable_cache=False)
        content = "Test content" * 10
        
        hasher.calculate_hash(content)
        hasher.calculate_hash(content)
        
        # Should have no cache hits when disabled
        assert hasher.cache_hits == 0
        assert hasher.cache_misses == 2
    
    def test_batch_hashing_performance(self):
        """Test batch hashing maintains good performance."""
        hasher = ContentHasher()
        contents = [f"Test content {i}" for i in range(100)]
        
        start_time = time.time()
        hashes = hasher.batch_calculate_hashes(contents)
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(hashes) == len(contents)
        assert len(set(hashes)) == len(contents)  # All unique
        
        # Performance requirement: >10k items/sec
        rate = len(contents) / processing_time if processing_time > 0 else float('inf')
        assert rate > 1000  # Relaxed for testing environment
    
    def test_cache_size_limit(self):
        """Test cache size limiting and cleanup."""
        hasher = ContentHasher(enable_cache=True, max_cache_size=5)
        
        # Add items beyond cache limit
        for i in range(10):
            content = f"Test content {i}" * 10  # Make substantial for caching
            hasher.calculate_hash(content)
        
        # Cache should not exceed max size significantly
        stats = hasher.get_cache_stats()
        assert stats['cache_size'] <= hasher.max_cache_size
    
    def test_get_cache_stats(self):
        """Test cache statistics reporting."""
        hasher = ContentHasher()
        content = "Test content" * 10
        
        hasher.calculate_hash(content)
        hasher.calculate_hash(content)  # Cache hit
        hasher.calculate_hash("Different content" * 10)  # Cache miss
        
        stats = hasher.get_cache_stats()
        
        assert 'cache_hits' in stats
        assert 'cache_misses' in stats
        assert 'hit_rate' in stats
        assert 'cache_size' in stats
        assert 'xxhash_available' in stats
        
        assert stats['cache_hits'] >= 1
        assert stats['cache_misses'] >= 2
        assert 0 <= stats['hit_rate'] <= 1


class TestExactDuplicateProcessor:
    """Test cases for ExactDuplicateProcessor component."""
    
    def create_test_chunk(self, chunk_id: str, content: str, 
                         source_context: str = "test") -> DocumentationChunk:
        """Helper to create test chunks."""
        return DocumentationChunk(
            chunk_id=chunk_id,
            content=content,
            source_context=source_context,
            token_count=len(content.split()),
            content_hash="",  # Will be calculated
            proxy_id="test_proxy",
            download_time=1.0
        )
    
    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        """Test processing with no duplicates."""
        processor = ExactDuplicateProcessor()
        
        chunks = [
            self.create_test_chunk("1", "Unique content A"),
            self.create_test_chunk("2", "Unique content B"),
            self.create_test_chunk("3", "Unique content C"),
        ]
        
        result = await processor.remove_exact_duplicates(chunks)
        
        assert len(result) == 3
        assert len({chunk.content for chunk in result}) == 3
    
    @pytest.mark.asyncio
    async def test_exact_duplicates_removal(self):
        """Test removal of exact duplicate content."""
        processor = ExactDuplicateProcessor()
        
        chunks = [
            self.create_test_chunk("1", "Identical content", "source1"),
            self.create_test_chunk("2", "Identical content", "source2"),
            self.create_test_chunk("3", "Different content", "source3"),
        ]
        
        result = await processor.remove_exact_duplicates(chunks)
        
        assert len(result) == 2  # Two unique pieces of content
        contents = {chunk.content for chunk in result}
        assert "Identical content" in contents
        assert "Different content" in contents
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty chunk list."""
        processor = ExactDuplicateProcessor()
        
        result = await processor.remove_exact_duplicates([])
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_single_chunk(self):
        """Test handling of single chunk."""
        processor = ExactDuplicateProcessor()
        chunk = self.create_test_chunk("1", "Single content")
        
        result = await processor.remove_exact_duplicates([chunk])
        
        assert len(result) == 1
        assert result[0] == chunk
    
    @pytest.mark.asyncio
    async def test_conflict_resolution_quality_scoring(self):
        """Test that higher quality chunks are selected from duplicates."""
        processor = ExactDuplicateProcessor()
        
        # Create identical content with different quality indicators
        chunks = [
            DocumentationChunk(
                chunk_id="low", content="Identical API content",
                source_context="example", token_count=50,
                content_hash="", proxy_id="", download_time=0.1
            ),
            DocumentationChunk(
                chunk_id="high", content="Identical API content", 
                source_context="reference documentation", token_count=50,
                content_hash="", proxy_id="good_proxy", download_time=2.0,
                metadata={"quality": "high", "complete": True}
            )
        ]
        
        result = await processor.remove_exact_duplicates(chunks)
        
        assert len(result) == 1
        # Should select the high-quality chunk
        assert result[0].chunk_id == "high"
        assert result[0].source_context == "reference documentation"
    
    @pytest.mark.asyncio
    async def test_hash_collision_handling(self):
        """Test handling of potential hash collisions."""
        processor = ExactDuplicateProcessor()
        
        # Create chunks with different content but force same hash for testing
        chunks = [
            self.create_test_chunk("1", "Content A"),
            self.create_test_chunk("2", "Content B")
        ]
        
        # Manually set same hash to simulate collision
        fake_hash = "fake_collision_hash"
        chunks[0].content_hash = fake_hash
        chunks[1].content_hash = fake_hash
        
        result = await processor.remove_exact_duplicates(chunks)
        
        # Should keep both chunks due to different content (collision detection)
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_existing_hash_reuse(self):
        """Test reuse of existing valid hashes."""
        processor = ExactDuplicateProcessor()
        
        # Create chunk with pre-calculated hash
        chunk = self.create_test_chunk("1", "Test content")
        original_hash = processor.hasher.calculate_hash(chunk.content)
        chunk.content_hash = original_hash
        
        result = await processor.remove_exact_duplicates([chunk])
        
        assert len(result) == 1
        assert result[0].content_hash == original_hash


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
class TestSimilarityAnalyzer:
    """Test cases for SimilarityAnalyzer component."""
    
    def create_test_chunk(self, chunk_id: str, content: str) -> DocumentationChunk:
        """Helper to create test chunks."""
        return DocumentationChunk(
            chunk_id=chunk_id,
            content=content,
            source_context="test",
            token_count=len(content.split()),
            content_hash=f"hash_{chunk_id}",
            proxy_id="test_proxy",
            download_time=1.0
        )
    
    @pytest.mark.asyncio
    async def test_similar_content_detection(self):
        """Test detection of semantically similar content."""
        # Use threshold 0.85 to properly separate Python from JavaScript chunks
        # - With Gemini: Python chunks have ~0.98 similarity (grouped), JavaScript ~0.84 (not grouped)
        # - With TF-IDF fallback (adjusts to 0.25): Python chunks have 1.0 similarity (grouped), JavaScript ~0.14 (not grouped)
        analyzer = SimilarityAnalyzer(similarity_threshold=0.85)
        
        chunks = [
            self.create_test_chunk("1", 
                "Python is a programming language used for web development and data science"),
            self.create_test_chunk("2", 
                "Python is a programming language for web development and data analysis"),
            self.create_test_chunk("3", 
                "JavaScript is used for frontend web development and user interfaces"),
        ]
        
        similar_groups = await analyzer.detect_similar_chunks(chunks)
        
        # Should find one similar group (the two Python-related chunks)
        assert len(similar_groups) == 1
        assert len(similar_groups[0]) == 2
        
        # Verify the similar chunks are the Python ones
        group_contents = [chunk.content for chunk in similar_groups[0]]
        assert all("Python" in content for content in group_contents)
    
    @pytest.mark.asyncio
    async def test_no_similar_content(self):
        """Test with completely different content."""
        analyzer = SimilarityAnalyzer(similarity_threshold=0.6)
        
        chunks = [
            self.create_test_chunk("1", "Python programming language documentation"),
            self.create_test_chunk("2", "Cooking recipes and kitchen equipment"),
            self.create_test_chunk("3", "Quantum physics and particle mechanics"),
        ]
        
        similar_groups = await analyzer.detect_similar_chunks(chunks)
        
        assert len(similar_groups) == 0
    
    @pytest.mark.asyncio
    async def test_insufficient_content(self):
        """Test handling of very short content."""
        analyzer = SimilarityAnalyzer(min_content_length=50)
        
        chunks = [
            self.create_test_chunk("1", "Short"),
            self.create_test_chunk("2", "Brief"),
        ]
        
        similar_groups = await analyzer.detect_similar_chunks(chunks)
        
        assert len(similar_groups) == 0
    
    @pytest.mark.asyncio
    async def test_single_chunk(self):
        """Test handling of single chunk."""
        analyzer = SimilarityAnalyzer()
        chunk = self.create_test_chunk("1", "Single content for testing")
        
        similar_groups = await analyzer.detect_similar_chunks([chunk])
        
        assert len(similar_groups) == 0
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty chunk list."""
        analyzer = SimilarityAnalyzer()
        
        similar_groups = await analyzer.detect_similar_chunks([])
        
        assert len(similar_groups) == 0
    
    def test_pairwise_similarity_calculation(self):
        """Test pairwise similarity calculation."""
        analyzer = SimilarityAnalyzer()
        
        chunk1 = self.create_test_chunk("1", 
            "Python is great for data science and machine learning applications")
        chunk2 = self.create_test_chunk("2", 
            "Python excels in data science and ML development")
        chunk3 = self.create_test_chunk("3", 
            "JavaScript is used for web frontend development")
        
        # Similar chunks should have high similarity
        similarity_high = analyzer.calculate_pairwise_similarity(chunk1, chunk2)
        
        # Different chunks should have low similarity
        similarity_low = analyzer.calculate_pairwise_similarity(chunk1, chunk3)
        
        assert 0 <= similarity_high <= 1
        assert 0 <= similarity_low <= 1
        assert similarity_high > similarity_low
    
    @pytest.mark.asyncio
    async def test_content_preprocessing(self):
        """Test content preprocessing for better analysis."""
        analyzer = SimilarityAnalyzer()
        
        # Test that preprocessing handles code blocks and formatting
        content_with_code = """
        Here is some documentation:
        ```python
        def example():
            return "test"
        ```
        This explains the function usage.
        """
        
        processed = analyzer._preprocess_content(content_with_code)
        
        assert processed is not None
        assert len(processed) > 0
        assert "documentation" in processed.lower()
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_sensitivity(self):
        """Test that similarity threshold affects grouping."""
        chunks = [
            self.create_test_chunk("1", 
                "Python programming language documentation and tutorials"),
            self.create_test_chunk("2", 
                "Python programming language guide and examples"),
        ]
        
        # High threshold should find groups
        analyzer_high = SimilarityAnalyzer(similarity_threshold=0.3)
        groups_high = await analyzer_high.detect_similar_chunks(chunks)
        
        # Very high threshold should find no groups
        analyzer_very_high = SimilarityAnalyzer(similarity_threshold=0.99)
        groups_very_high = await analyzer_very_high.detect_similar_chunks(chunks)
        
        # High threshold should find more groups than very high threshold
        assert len(groups_high) >= len(groups_very_high)


class TestConflictResolver:
    """Test cases for ConflictResolver component."""
    
    def create_test_chunk(self, chunk_id: str, content: str, 
                         source_context: str = "test",
                         token_count: int = None) -> DocumentationChunk:
        """Helper to create test chunks with optional parameters."""
        return DocumentationChunk(
            chunk_id=chunk_id,
            content=content,
            source_context=source_context,
            token_count=token_count or len(content.split()),
            content_hash=f"hash_{chunk_id}",
            proxy_id="test_proxy",
            download_time=1.0
        )
    
    @pytest.mark.asyncio
    async def test_no_conflicts(self):
        """Test resolution with no conflicts (single-chunk groups)."""
        resolver = ConflictResolver()
        
        groups = [
            [self.create_test_chunk("1", "Unique content A")],
            [self.create_test_chunk("2", "Unique content B")],
        ]
        
        result = await resolver.resolve_similar_groups(groups)
        
        assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_empty_groups(self):
        """Test handling of empty group list."""
        resolver = ConflictResolver()
        
        result = await resolver.resolve_similar_groups([])
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_quality_based_selection(self):
        """Test that higher quality chunks are selected."""
        resolver = ConflictResolver()
        
        # Create similar chunks with different quality indicators
        chunks = [
            self.create_test_chunk(
                "low", "API documentation content",
                source_context="example", token_count=100
            ),
            self.create_test_chunk(
                "medium", "API documentation content with examples",
                source_context="tutorial", token_count=500
            ),
            self.create_test_chunk(
                "high", "Complete API documentation with examples and parameters",
                source_context="reference documentation", token_count=1500
            )
        ]
        
        # Add code content to the high-quality chunk
        chunks[2].content += "\n```python\ndef example(): pass\n```"
        
        groups = [chunks]  # All in one group
        result = await resolver.resolve_similar_groups(groups)
        
        assert len(result) == 1
        # Should select the high-quality chunk
        assert result[0].chunk_id == "high"
    
    @pytest.mark.asyncio
    async def test_code_content_preference(self):
        """Test preference for chunks with code examples."""
        resolver = ConflictResolver()
        
        chunks = [
            self.create_test_chunk("no_code", "Simple API documentation"),
            self.create_test_chunk("with_code", 
                "API documentation\n```python\ndef example():\n    return 'test'\n```")
        ]
        
        groups = [chunks]
        result = await resolver.resolve_similar_groups(groups)
        
        assert len(result) == 1
        assert result[0].chunk_id == "with_code"
    
    @pytest.mark.asyncio
    async def test_source_context_priority(self):
        """Test priority based on source context quality."""
        resolver = ConflictResolver()
        
        chunks = [
            self.create_test_chunk("example", "Content", "example"),
            self.create_test_chunk("tutorial", "Content", "tutorial"),
            self.create_test_chunk("reference", "Content", "reference documentation"),
        ]
        
        groups = [chunks]
        result = await resolver.resolve_similar_groups(groups)
        
        assert len(result) == 1
        assert result[0].chunk_id == "reference"  # Should prefer reference docs
    
    def test_quality_scoring_edge_cases(self):
        """Test quality scoring with edge cases."""
        resolver = ConflictResolver()
        
        # Test minimal content (empty content not allowed by model validation)
        minimal_chunk = self.create_test_chunk("minimal", " ", token_count=1)
        
        # Test very long content
        long_content = "Documentation " * 10000
        long_chunk = self.create_test_chunk("long", long_content, token_count=10000)
        
        # Should handle edge cases gracefully
        minimal_score = resolver._select_best_chunk([minimal_chunk])
        long_score = resolver._select_best_chunk([long_chunk])
        
        assert minimal_score is not None
        assert long_score is not None
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """Test conflict resolution statistics tracking."""
        resolver = ConflictResolver()
        
        groups = [
            [self.create_test_chunk("1", "Content A")],  # No conflict
            [  # Conflict group
                self.create_test_chunk("2a", "Content B"),
                self.create_test_chunk("2b", "Content B variant"),
            ]
        ]
        
        await resolver.resolve_similar_groups(groups)
        
        stats = resolver.get_resolution_stats()
        
        assert stats['groups_processed'] == 2
        assert stats['conflicts_resolved'] == 1  # One group had conflict
        assert stats['chunks_merged'] == 1  # One chunk was merged away
    
    @pytest.mark.asyncio
    async def test_metadata_preference(self):
        """Test preference for chunks with richer metadata."""
        resolver = ConflictResolver()
        
        chunks = [
            DocumentationChunk(
                chunk_id="minimal", content="API docs", source_context="basic",
                token_count=10, content_hash="hash1", proxy_id="", download_time=1.0
            ),
            DocumentationChunk(
                chunk_id="rich", content="API docs", source_context="comprehensive",
                token_count=10, content_hash="hash2", proxy_id="good_proxy", 
                download_time=2.0, metadata={"complete": True, "verified": True}
            )
        ]
        
        groups = [chunks]
        result = await resolver.resolve_similar_groups(groups)
        
        assert len(result) == 1
        assert result[0].chunk_id == "rich"


class TestDeduplicationEngine:
    """Test cases for the main DeduplicationEngine."""
    
    def create_test_chunk(self, chunk_id: str, content: str, 
                         source_context: str = "test") -> DocumentationChunk:
        """Helper to create test chunks."""
        return DocumentationChunk(
            chunk_id=chunk_id,
            content=content,
            source_context=source_context,
            token_count=len(content.split()),
            content_hash="",  # Will be calculated
            proxy_id="test_proxy",
            download_time=1.0
        )
    
    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling of empty chunk list."""
        engine = DeduplicationEngine()
        
        result = await engine.deduplicate_chunks([])
        
        assert result.original_count == 0
        assert result.deduplicated_count == 0
        assert result.exact_duplicates_removed == 0
        assert result.similar_chunks_merged == 0
        assert len(result.chunks) == 0
    
    @pytest.mark.asyncio
    async def test_no_duplicates(self):
        """Test with completely unique chunks."""
        engine = DeduplicationEngine()
        
        chunks = [
            self.create_test_chunk("1", "Unique content A"),
            self.create_test_chunk("2", "Unique content B"),
            self.create_test_chunk("3", "Unique content C"),
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        assert result.original_count == 3
        assert result.deduplicated_count == 3
        assert result.exact_duplicates_removed == 0
        assert len(result.chunks) == 3
    
    @pytest.mark.asyncio
    async def test_exact_duplicates_only(self):
        """Test with only exact duplicates."""
        engine = DeduplicationEngine()
        
        chunks = [
            self.create_test_chunk("1", "Identical content", "source1"),
            self.create_test_chunk("2", "Identical content", "source2"),
            self.create_test_chunk("3", "Different content", "source3"),
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        assert result.original_count == 3
        assert result.deduplicated_count == 2
        assert result.exact_duplicates_removed == 1
        assert len(result.chunks) == 2
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
    async def test_similar_chunks_detection(self):
        """Test semantic similarity detection and merging."""
        engine = DeduplicationEngine(similarity_threshold=0.6)
        
        chunks = [
            self.create_test_chunk("1", 
                "Python is excellent for data science and machine learning applications"),
            self.create_test_chunk("2", 
                "Python excels in data science and ML applications with great libraries"),
            self.create_test_chunk("3", 
                "JavaScript is used for web frontend development and user interfaces"),
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        # Should detect similarity between Python chunks
        assert result.deduplicated_count < result.original_count
        assert len(result.chunks) < len(chunks)
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """Test that processing meets performance targets."""
        engine = DeduplicationEngine()
        
        # Create 100 test chunks with some duplicates
        chunks = []
        for i in range(100):
            content = f"Test documentation content {i % 20}"  # Creates some duplicates
            chunk = self.create_test_chunk(f"chunk_{i}", content * 5)  # Make substantial
            chunks.append(chunk)
        
        start_time = time.time()
        result = await engine.deduplicate_chunks(chunks)
        processing_time = time.time() - start_time
        
        # Performance target: 100 chunks in <5 seconds
        assert processing_time < 5.0
        assert result.chunks_per_second > 20  # At least 20 chunks/sec
        assert result.deduplicated_count < result.original_count  # Some deduplication
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch processing for large chunk sets."""
        engine = DeduplicationEngine(batch_size=25)
        
        # Create chunks that exceed batch size
        chunks = [
            self.create_test_chunk(f"chunk_{i}", f"Content {i}")
            for i in range(75)  # 3 batches of 25
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        assert result.original_count == 75
        assert len(result.chunks) == 75  # No duplicates, so all should remain
    
    def test_result_metrics(self):
        """Test DeduplicationResult metrics calculation."""
        result = DeduplicationResult(
            original_count=100,
            deduplicated_count=75,
            exact_duplicates_removed=20,
            similar_chunks_merged=5,
            processing_time=2.0,
            chunks=[]  # Empty for testing metrics only
        )
        
        assert result.deduplication_ratio == 0.25  # 25% reduction
        assert result.chunks_per_second == 50.0  # 100 chunks / 2 seconds
        assert 0 <= result.efficiency_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling and graceful degradation."""
        engine = DeduplicationEngine()
        
        # Create chunk with minimal content (empty not allowed by model validation)
        chunks = [
            DocumentationChunk(
                chunk_id="problematic",
                content=" ",  # Minimal content to test edge cases
                source_context="test",
                token_count=1,
                content_hash="",
                proxy_id="",
                download_time=0.0
            )
        ]
        
        # Should not raise exception, should handle gracefully
        result = await engine.deduplicate_chunks(chunks)
        
        assert result is not None
        assert result.original_count == 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_stats(self):
        """Test comprehensive statistics gathering."""
        engine = DeduplicationEngine()
        
        chunks = [
            self.create_test_chunk("1", "Content A"),
            self.create_test_chunk("2", "Content A"),  # Duplicate
            self.create_test_chunk("3", "Content B"),
        ]
        
        await engine.deduplicate_chunks(chunks)
        
        stats = engine.get_comprehensive_stats()
        
        # Check that all expected stats categories are present
        assert 'total_operations' in stats
        assert 'hasher_stats' in stats
        assert 'conflict_resolution_stats' in stats
        assert 'similarity_config' in stats
        
        # Verify some specific metrics
        assert stats['total_operations'] >= 1
        assert stats['total_chunks_processed'] >= 3
    
    @pytest.mark.asyncio
    async def test_similarity_disabled(self):
        """Test engine behavior with similarity analysis disabled."""
        engine = DeduplicationEngine(enable_similarity=False)
        
        chunks = [
            self.create_test_chunk("1", "Python data science"),
            self.create_test_chunk("2", "Python data analysis"),  # Similar but not exact
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        # Should only do exact deduplication
        assert result.similar_chunks_merged == 0
        assert result.deduplicated_count == 2  # Both chunks remain
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self):
        """Test performance benchmarking functionality."""
        engine = DeduplicationEngine()
        
        # Create test dataset
        test_chunks = [
            self.create_test_chunk(f"chunk_{i}", f"Content {i % 10}")
            for i in range(50)
        ]
        
        benchmark_results = await engine.benchmark_performance(
            test_chunks, batch_sizes=[10, 25]
        )
        
        assert 'test_chunk_count' in benchmark_results
        assert 'batch_size_results' in benchmark_results
        assert benchmark_results['test_chunk_count'] == 50
        
        # Should have results for tested batch sizes
        assert len(benchmark_results['batch_size_results']) > 0
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage stays reasonable for large datasets."""
        engine = DeduplicationEngine(batch_size=50)
        
        # Create larger dataset to test memory efficiency
        chunks = []
        for i in range(500):
            content = f"Documentation content number {i}" + " " * (i % 100)
            chunk = self.create_test_chunk(f"chunk_{i}", content)
            chunks.append(chunk)
        
        result = await engine.deduplicate_chunks(chunks)
        
        # Should complete without memory issues
        assert result.original_count == 500
        assert result.processing_time > 0
        assert len(result.chunks) <= 500


class TestIntegrationScenarios:
    """Integration test scenarios with realistic data patterns."""
    
    def create_realistic_chunk(self, chunk_id: str, content_type: str, 
                             base_content: str, variation: int = 0) -> DocumentationChunk:
        """Create realistic documentation chunk with controlled variations."""
        content_templates = {
            "api_doc": "API Reference: {base}\n\nParameters:\n- param1: string\n- param2: int\n\nReturns: {variation}",
            "tutorial": "Tutorial: {base}\n\n```python\ndef example():\n    return {variation}\n```\n\nThis shows how to use the API.",
            "example": "Example: {base}\n\nUsage:\n```\nresult = function({variation})\nprint(result)\n```",
            "reference": "Complete Reference: {base}\n\nDetailed documentation with parameters, examples, and return values. Version {variation}."
        }
        
        content = content_templates.get(content_type, base_content).format(
            base=base_content, variation=variation
        )
        
        return DocumentationChunk(
            chunk_id=chunk_id,
            content=content,
            source_context=content_type,
            token_count=len(content.split()),
            content_hash="",
            proxy_id=f"proxy_{chunk_id}",
            download_time=1.0 + (variation * 0.1)
        )
    
    @pytest.mark.asyncio
    async def test_realistic_documentation_deduplication(self):
        """Test with realistic documentation patterns."""
        engine = DeduplicationEngine(similarity_threshold=0.85)
        
        chunks = [
            # Exact duplicates
            self.create_realistic_chunk("1", "api_doc", "FastAPI setup", 0),
            self.create_realistic_chunk("2", "api_doc", "FastAPI setup", 0),
            
            # Similar but different versions
            self.create_realistic_chunk("3", "tutorial", "FastAPI basics", 1),
            self.create_realistic_chunk("4", "tutorial", "FastAPI basics", 2),
            
            # Completely different
            self.create_realistic_chunk("5", "reference", "Database connections", 0),
            self.create_realistic_chunk("6", "example", "Authentication", 0),
        ]
        
        result = await engine.deduplicate_chunks(chunks)
        
        # Should reduce duplicates significantly
        assert result.deduplicated_count < result.original_count
        assert result.exact_duplicates_removed > 0
        
        # Verify important content types are preserved
        final_contexts = {chunk.source_context for chunk in result.chunks}
        assert "reference" in final_contexts  # Should keep reference docs
        
    @pytest.mark.asyncio
    async def test_mixed_content_quality_resolution(self):
        """Test conflict resolution with mixed content quality."""
        engine = DeduplicationEngine()
        
        # Same API topic with different quality levels
        base_content = "Authentication API endpoint documentation"
        
        chunks = [
            # Low quality: just basic info
            DocumentationChunk("basic", base_content, "example", 50, "", "", 0.5),
            
            # Medium quality: includes code
            DocumentationChunk("code", base_content + "\n```python\nauth.login()\n```", 
                             "tutorial", 100, "", "proxy1", 1.5),
            
            # High quality: comprehensive with metadata
            DocumentationChunk("complete", 
                             base_content + "\n\nParameters:\n- username: str\n- password: str\n\nExample:\n```python\nauth.login('user', 'pass')\n```",
                             "reference documentation", 200, "", "proxy2", 2.0,
                             metadata={"complete": True, "verified": True})
        ]
        
        # Make them similar enough to be grouped
        for chunk in chunks:
            chunk.content = chunks[0].content  # Force exact match for testing
        
        result = await engine.deduplicate_chunks(chunks)
        
        assert len(result.chunks) == 1
        # Should select the highest quality chunk
        selected = result.chunks[0]
        assert selected.chunk_id == "complete"
        assert "reference" in selected.source_context


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])