"""
Unit tests for context generation engine.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.contexter.core.context_generator import (
    ContextGenerator,
    LibraryProfile,
    ContextDiversityOptimizer,
    ContextGenerationError
)


class TestLibraryProfile:
    """Test LibraryProfile analysis."""
    
    def test_web_library_analysis(self):
        """Test analysis of web-related library."""
        profile = LibraryProfile.from_library_id("encode/httpx")
        
        assert profile.name == "httpx"
        assert profile.domain == "web"
        assert "httpx" in profile.keywords
        assert "HTTP requests" in profile.context_modifiers
        assert "authentication" in profile.priority_topics
    
    def test_data_library_analysis(self):
        """Test analysis of data-related library."""
        profile = LibraryProfile.from_library_id("pandas-dev/pandas")
        
        assert profile.name == "pandas"
        assert profile.domain == "data"
        assert "pandas" in profile.keywords
        assert "data processing" in profile.context_modifiers
        assert "performance" in profile.priority_topics
    
    def test_testing_library_analysis(self):
        """Test analysis of testing library."""
        profile = LibraryProfile.from_library_id("pytest-dev/pytest")
        
        assert profile.name == "pytest"
        assert profile.domain == "testing"
        assert "pytest" in profile.keywords
        assert "unit testing" in profile.context_modifiers
        assert "test fixtures" in profile.priority_topics
    
    def test_async_library_analysis(self):
        """Test analysis of async library."""
        profile = LibraryProfile.from_library_id("trio-util/async-generator")
        
        assert profile.name == "async-generator"
        assert profile.domain == "async"
        assert "async-generator" in profile.keywords
        assert "asynchronous programming" in profile.context_modifiers
        assert "performance" in profile.priority_topics
    
    def test_general_library_analysis(self):
        """Test analysis of general library."""
        profile = LibraryProfile.from_library_id("some-org/random-lib")
        
        assert profile.name == "random-lib"
        assert profile.domain == "general"
        assert "random-lib" in profile.keywords


class TestContextDiversityOptimizer:
    """Test context diversity optimization."""
    
    def test_basic_optimization(self):
        """Test basic context optimization."""
        optimizer = ContextDiversityOptimizer(similarity_threshold=0.5)
        
        contexts = [
            "httpx complete API documentation reference guide",
            "httpx getting started tutorial installation examples",
            "httpx advanced configuration options parameters",
            "httpx API documentation complete reference guide",  # Similar to first
            "httpx troubleshooting error handling debugging guide"
        ]
        
        optimized = optimizer.optimize_contexts(contexts)
        
        # Should remove similar context
        assert len(optimized) < len(contexts)
        # Contexts are sorted by length descending, so longer context should be first
        # Similar context should be removed
        assert contexts[3] not in optimized
        # Should have fewer contexts due to similarity removal
        assert len(optimized) >= 3  # Should keep at least 3 diverse contexts
    
    def test_keyword_extraction(self):
        """Test keyword extraction from contexts."""
        optimizer = ContextDiversityOptimizer()
        
        context = "httpx complete API documentation reference guide"
        keywords = optimizer._extract_keywords(context)
        
        assert "httpx" in keywords
        assert "complete" in keywords
        assert "api" in keywords
        assert "documentation" in keywords
        assert "reference" in keywords
        assert "guide" in keywords
        # Stop words should be filtered out
        assert "the" not in keywords
        assert "and" not in keywords
    
    def test_similarity_calculation(self):
        """Test similarity calculation between keyword sets."""
        optimizer = ContextDiversityOptimizer()
        
        keywords1 = {"httpx", "api", "documentation", "reference"}
        keywords2 = {"httpx", "api", "complete", "guide"}
        keywords3 = {"fastapi", "web", "framework", "async"}
        
        # Some similarity
        similarity12 = optimizer._calculate_similarity(keywords1, keywords2)
        assert 0.0 < similarity12 < 1.0
        
        # No similarity
        similarity13 = optimizer._calculate_similarity(keywords1, keywords3)
        assert similarity13 == 0.0
        
        # Perfect similarity
        similarity_same = optimizer._calculate_similarity(keywords1, keywords1)
        assert similarity_same == 1.0
    
    def test_empty_contexts_handling(self):
        """Test handling of empty context list."""
        optimizer = ContextDiversityOptimizer()
        
        assert optimizer.optimize_contexts([]) == []
        assert optimizer.optimize_contexts(["single context"]) == ["single context"]


class TestContextGenerator:
    """Test main context generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create context generator instance."""
        return ContextGenerator(max_contexts=7, min_contexts=3)
    
    @pytest.mark.asyncio
    async def test_basic_context_generation(self, generator):
        """Test basic context generation."""
        contexts = await generator.generate_contexts("encode/httpx")
        
        assert len(contexts) >= 3
        assert len(contexts) <= 7
        
        # All contexts should contain the library name
        for context in contexts:
            assert "httpx" in context.lower()
        
        # Should have diverse contexts
        unique_contexts = set(contexts)
        assert len(unique_contexts) == len(contexts)  # No duplicates
    
    @pytest.mark.asyncio
    async def test_context_generation_with_domain_specific_terms(self, generator):
        """Test that domain-specific terms are included."""
        contexts = await generator.generate_contexts("encode/httpx")
        
        # Should include web-related terms for httpx
        context_text = " ".join(contexts).lower()
        assert any(term in context_text for term in ["http", "api", "request", "client"])
    
    @pytest.mark.asyncio
    async def test_context_length_constraints(self, generator):
        """Test context length constraints."""
        # Test with custom limits
        generator_small = ContextGenerator(max_contexts=3, min_contexts=2)
        contexts = await generator_small.generate_contexts("test/lib")
        
        assert len(contexts) >= 2
        assert len(contexts) <= 3
    
    @pytest.mark.asyncio
    async def test_fallback_context_generation(self, generator):
        """Test fallback context generation on errors."""
        # Mock the profile creation to raise an exception
        with patch.object(LibraryProfile, 'from_library_id', side_effect=Exception("Profile error")):
            contexts = await generator.generate_contexts("test/lib")
            
            # Should still return contexts using fallback
            assert len(contexts) >= generator.min_contexts
            assert all("lib" in context for context in contexts)
    
    @pytest.mark.asyncio
    async def test_context_prioritization(self, generator):
        """Test that contexts are properly prioritized."""
        contexts = await generator.generate_contexts("encode/httpx")
        
        # First few contexts should include essential topics
        first_context = contexts[0].lower()
        assert any(keyword in first_context 
                  for keyword in ["documentation", "api", "reference", "guide"])
    
    @pytest.mark.asyncio
    async def test_generate_contexts_with_validation(self, generator):
        """Test context generation with validation."""
        contexts = await generator.generate_contexts_with_validation("encode/httpx")
        
        # Should pass all validation checks
        assert len(contexts) >= generator.min_contexts
        assert len(contexts) <= generator.max_contexts
        assert all("httpx" in context.lower() for context in contexts)
        assert len(set(contexts)) == len(contexts)  # No duplicates
    
    @pytest.mark.asyncio
    async def test_validation_failure_cases(self, generator):
        """Test validation failure scenarios."""
        # Mock generator to return invalid contexts
        with patch.object(generator, 'generate_contexts', return_value=[]):
            with pytest.raises(ContextGenerationError, match="No contexts generated"):
                await generator.generate_contexts_with_validation("test/lib")
        
        # Mock generator to return insufficient contexts
        with patch.object(generator, 'generate_contexts', return_value=["ctx1", "ctx2"]):
            with pytest.raises(ContextGenerationError, match="Insufficient contexts"):
                await generator.generate_contexts_with_validation("test/lib")
        
        # Mock generator to return contexts without library name
        with patch.object(generator, 'generate_contexts', 
                         return_value=["generic context 1", "generic context 2", "generic context 3"]):
            with pytest.raises(ContextGenerationError, match="doesn't contain library name"):
                await generator.generate_contexts_with_validation("test/lib")
    
    @pytest.mark.asyncio
    async def test_duplicate_context_detection(self, generator):
        """Test duplicate context detection in validation."""
        # Mock generator to return duplicate contexts
        duplicate_contexts = [
            "httpx documentation guide",
            "httpx api reference",
            "httpx documentation guide",  # Duplicate
            "httpx tutorial examples"
        ]
        
        with patch.object(generator, 'generate_contexts', return_value=duplicate_contexts):
            with pytest.raises(ContextGenerationError, match="duplicate contexts"):
                await generator.generate_contexts_with_validation("encode/httpx")
    
    def test_fallback_context_generation_direct(self, generator):
        """Test direct fallback context generation."""
        fallback_contexts = generator._generate_fallback_contexts("testlib")
        
        assert len(fallback_contexts) >= 3
        assert all("testlib" in context for context in fallback_contexts)
        assert "testlib documentation" in fallback_contexts
        assert "testlib API reference guide" in fallback_contexts
    
    def test_context_prioritization_scoring(self, generator):
        """Test context prioritization scoring."""
        profile = LibraryProfile.from_library_id("encode/httpx")
        
        contexts = [
            "httpx advanced configuration options",
            "httpx complete API documentation reference guide",  # Should score higher
            "httpx troubleshooting error handling",
            "httpx getting started tutorial examples"  # Should score high
        ]
        
        prioritized = generator._prioritize_contexts(contexts, profile)
        
        # API documentation and getting started should be prioritized (among top contexts)
        # Check that high-priority contexts are in the top 3
        first_three = prioritized[:3]
        assert any("API documentation" in context for context in first_three)
        assert any("getting started" in context for context in first_three)


class TestContextGenerationIntegration:
    """Integration tests for context generation components."""
    
    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self):
        """Test complete context generation pipeline."""
        generator = ContextGenerator(max_contexts=5, min_contexts=3)
        
        # Test with various library types
        test_libraries = [
            "encode/httpx",
            "pandas-dev/pandas", 
            "pytest-dev/pytest",
            "fastapi/fastapi",
            "python/cpython"
        ]
        
        for library_id in test_libraries:
            contexts = await generator.generate_contexts_with_validation(library_id)
            
            library_name = library_id.split('/')[-1].lower()
            
            # Basic validation
            assert 3 <= len(contexts) <= 5
            assert all(library_name in context.lower() for context in contexts)
            assert len(set(contexts)) == len(contexts)
            
            # Quality checks
            assert all(len(context.strip()) >= 10 for context in contexts)
            assert all(context.strip() for context in contexts)
    
    @pytest.mark.asyncio
    async def test_context_diversity_across_domains(self):
        """Test that different domains generate diverse contexts."""
        generator = ContextGenerator()
        
        # Generate contexts for different domain types
        web_contexts = await generator.generate_contexts("requests/requests")
        data_contexts = await generator.generate_contexts("pandas-dev/pandas")
        test_contexts = await generator.generate_contexts("pytest-dev/pytest")
        
        # Convert to text for analysis
        web_text = " ".join(web_contexts).lower()
        data_text = " ".join(data_contexts).lower()
        test_text = " ".join(test_contexts).lower()
        
        # Web contexts should contain web-specific terms
        assert any(term in web_text for term in ["http", "request", "api", "client"])
        
        # Data contexts should contain data-specific terms
        assert any(term in data_text for term in ["data", "analysis", "processing", "dataframe"])
        
        # Test contexts should contain testing-specific terms  
        assert any(term in test_text for term in ["test", "fixture", "assert", "mock"])
        
        # Should have different focuses
        assert web_text != data_text
        assert data_text != test_text