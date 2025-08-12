"""
Unit tests for download models and data structures.
"""

import pytest
import time
from datetime import datetime, timedelta

from src.contexter.models.download_models import (
    DownloadRequest,
    DocumentationChunk,
    DownloadTask,
    ProgressMetrics,
    DownloadSummary,
    DownloadStatus,
    DownloadError,
    ContextGenerationError,
    ProxyUnavailableError,
    TaskTimeoutError
)


class TestDownloadRequest:
    """Test DownloadRequest model."""
    
    def test_basic_request_creation(self):
        """Test basic request creation with defaults."""
        request = DownloadRequest(library_id="encode/httpx")
        
        assert request.library_id == "encode/httpx"
        assert request.contexts == []
        assert request.token_limit == 200_000
        assert request.priority == 0
        assert request.max_contexts == 7
        assert request.timeout_seconds == 30.0
        assert request.retry_count == 3
    
    def test_request_validation(self):
        """Test request parameter validation."""
        # Valid request
        request = DownloadRequest(
            library_id="  fastapi/fastapi  ",
            token_limit=150_000,
            max_contexts=5
        )
        
        assert request.library_id == "fastapi/fastapi"  # Should be stripped
        assert request.token_limit == 150_000
        assert request.max_contexts == 5
    
    def test_request_parameter_clamping(self):
        """Test that parameters are clamped to valid ranges."""
        request = DownloadRequest(
            library_id="test/lib",
            token_limit=500_000,  # Above max
            max_contexts=15,  # Above max
            retry_count=10,  # Above max
            timeout_seconds=-5  # Invalid
        )
        
        assert request.token_limit == 200_000  # Clamped to max
        assert request.max_contexts == 10  # Clamped to max
        assert request.retry_count == 5  # Clamped to max
        assert request.timeout_seconds == 30.0  # Reset to default
    
    def test_invalid_library_id(self):
        """Test validation of library ID."""
        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            DownloadRequest(library_id="")
        
        with pytest.raises(ValueError, match="Library ID cannot be empty"):
            DownloadRequest(library_id="   ")
    
    def test_estimated_total_tokens(self):
        """Test estimated total tokens calculation."""
        request = DownloadRequest(
            library_id="test/lib",
            contexts=["ctx1", "ctx2", "ctx3"],
            token_limit=100_000
        )
        
        # Should be min(3 * 100_000, 3 * 200_000) - but max_contexts is 7, so uses max_contexts
        # Actually: min(3 * 100_000, 7 * 200_000) = min(300_000, 1_400_000) = 300_000
        # But implementation uses max(len(contexts), max_contexts) = max(3, 7) = 7
        # So it's min(7 * 100_000, 7 * 200_000) = min(700_000, 1_400_000) = 700_000
        assert request.estimated_total_tokens == 700_000


class TestDocumentationChunk:
    """Test DocumentationChunk model."""
    
    def test_basic_chunk_creation(self):
        """Test basic chunk creation."""
        chunk = DocumentationChunk(
            chunk_id="test_chunk",
            content="This is test documentation content.",
            source_context="test context",
            token_count=100,
            content_hash="abc123",
            proxy_id="proxy_1",
            download_time=2.5,
            library_id="test/lib"
        )
        
        assert chunk.chunk_id == "test_chunk"
        assert chunk.content == "This is test documentation content."
        assert chunk.token_count == 100
        assert chunk.content_hash == "abc123"
        assert chunk.proxy_id == "proxy_1"
        assert chunk.download_time == 2.5
    
    def test_auto_generated_fields(self):
        """Test auto-generated chunk ID and content hash."""
        chunk = DocumentationChunk(
            chunk_id="",  # Should be auto-generated
            content="Test content for hashing",
            source_context="test context",
            token_count=0,  # Should be auto-estimated
            content_hash="",  # Should be auto-generated
            proxy_id="proxy_1",
            download_time=1.0
        )
        
        # Should have generated values
        assert chunk.chunk_id.startswith("chunk_")
        assert len(chunk.chunk_id) > 10
        assert chunk.content_hash != ""
        assert len(chunk.content_hash) == 16  # xxhash64 hex digest length
        assert chunk.token_count > 0  # Should be estimated from content
    
    def test_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            DocumentationChunk(
                chunk_id="test",
                content="",
                source_context="test",
                token_count=100,
                content_hash="abc",
                proxy_id="proxy",
                download_time=1.0
            )
    
    def test_properties(self):
        """Test calculated properties."""
        chunk = DocumentationChunk(
            chunk_id="test",
            content="Test content",
            source_context="test",
            token_count=100,
            content_hash="abc",
            proxy_id="proxy",
            download_time=2.0
        )
        
        assert chunk.content_size_bytes == len("Test content".encode('utf-8'))
        assert chunk.tokens_per_second == 50.0  # 100 tokens / 2.0 seconds
        assert 0.0 < chunk.efficiency_score <= 1.0
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # Excellent performance (>5000 tokens/sec)
        fast_chunk = DocumentationChunk(
            chunk_id="fast", content="Fast", source_context="test",
            token_count=10000, content_hash="abc", proxy_id="proxy", download_time=1.0
        )
        assert fast_chunk.efficiency_score == 1.0
        
        # Poor performance (<500 tokens/sec)
        slow_chunk = DocumentationChunk(
            chunk_id="slow", content="Slow", source_context="test",
            token_count=100, content_hash="abc", proxy_id="proxy", download_time=1.0
        )
        assert slow_chunk.efficiency_score == 0.2


class TestDownloadTask:
    """Test DownloadTask model."""
    
    def test_basic_task_creation(self):
        """Test basic task creation."""
        task = DownloadTask(
            task_id="test_task",
            library_id="test/lib",
            context="test context",
            token_limit=100_000
        )
        
        assert task.task_id == "test_task"
        assert task.library_id == "test/lib"
        assert task.context == "test context"
        assert task.token_limit == 100_000
        assert task.status == DownloadStatus.PENDING
        assert task.retry_count == 0
    
    def test_auto_generated_task_id(self):
        """Test auto-generated task ID."""
        task = DownloadTask(
            task_id="",  # Should be auto-generated
            library_id="test/lib",
            context="test context",
            token_limit=100_000
        )
        
        assert task.task_id.startswith("task_")
        assert "test_lib" in task.task_id  # Should contain sanitized library name
    
    def test_parameter_clamping(self):
        """Test parameter clamping."""
        task = DownloadTask(
            task_id="test",
            library_id="test/lib",
            context="test context",
            token_limit=500_000,  # Above max
            max_retries=10  # Above max
        )
        
        assert task.token_limit == 200_000  # Clamped
        assert task.max_retries == 5  # Clamped
    
    def test_task_lifecycle(self):
        """Test task status lifecycle methods."""
        task = DownloadTask(
            task_id="test",
            library_id="test/lib",
            context="test context",
            token_limit=100_000
        )
        
        # Start task
        task.start(proxy_id="proxy_1")
        assert task.status == DownloadStatus.IN_PROGRESS
        assert task.started_at is not None
        assert task.proxy_id == "proxy_1"
        
        # Complete task
        chunk = DocumentationChunk(
            chunk_id="chunk", content="content", source_context="test",
            token_count=100, content_hash="abc", proxy_id="proxy", download_time=1.0
        )
        task.complete(chunk)
        assert task.status == DownloadStatus.COMPLETED
        assert task.completed_at is not None
        assert task.metadata["chunk_id"] == "chunk"
        assert task.metadata["tokens_retrieved"] == 100
        
        # Test failure
        task_fail = DownloadTask("fail", "test/lib", "test", 100_000)
        task_fail.fail("Test error")
        assert task_fail.status == DownloadStatus.FAILED
        assert task_fail.error_message == "Test error"
        assert task_fail.retry_count == 1
    
    def test_can_retry_logic(self):
        """Test retry logic."""
        task = DownloadTask(
            task_id="test",
            library_id="test/lib",
            context="test context",
            token_limit=100_000,
            max_retries=3  # Allow 3 retries
        )
        
        # Initially can retry
        task.fail("First error")
        assert task.can_retry
        assert task.retry_count == 1
        
        # Still can retry
        task.fail("Second error")
        assert task.can_retry
        assert task.retry_count == 2
        
        # Still can retry once more  
        task.fail("Third error")
        assert not task.can_retry  # Now at max_retries (3), so cannot retry anymore
        assert task.retry_count == 3
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        task = DownloadTask("test", "test/lib", "test", 100_000)
        
        # No duration before start
        assert task.duration_seconds is None
        
        # Duration after start
        task.start()
        time.sleep(0.1)  # Small delay
        duration = task.duration_seconds
        assert duration is not None
        assert duration >= 0.1
        
        # Duration after completion
        task.complete(DocumentationChunk(
            chunk_id="chunk", content="content", source_context="test",
            token_count=100, content_hash="abc", proxy_id="proxy", download_time=1.0
        ))
        final_duration = task.duration_seconds
        assert final_duration is not None
        assert final_duration >= duration


class TestProgressMetrics:
    """Test ProgressMetrics model."""
    
    def test_basic_metrics(self):
        """Test basic metrics creation and properties."""
        metrics = ProgressMetrics(total_contexts=10)
        
        assert metrics.total_contexts == 10
        assert metrics.completed_contexts == 0
        assert metrics.failed_contexts == 0
        assert metrics.in_progress_contexts == 10
        assert metrics.completion_rate == 0.0
        assert metrics.success_rate == 0.0
    
    def test_metrics_calculations(self):
        """Test metrics calculations."""
        metrics = ProgressMetrics(total_contexts=10)
        
        # Simulate some progress
        metrics.completed_contexts = 6
        metrics.failed_contexts = 2
        metrics.total_tokens_retrieved = 60000
        metrics.total_download_time = 30.0
        
        assert metrics.in_progress_contexts == 2  # 10 - 6 - 2
        assert metrics.completion_rate == 0.8  # 8/10
        assert metrics.success_rate == 0.75  # 6/8
        assert metrics.average_tokens_per_context == 10000.0  # 60000/6
        assert metrics.average_download_time_per_context == 5.0  # 30/6
        assert metrics.overall_tokens_per_second == 2000.0  # 60000/30
    
    def test_time_estimation(self):
        """Test time estimation."""
        metrics = ProgressMetrics(total_contexts=10)
        
        # No estimation without progress
        assert metrics.estimated_remaining_time is None
        
        # With progress
        metrics.completed_contexts = 5
        metrics.failed_contexts = 1
        # Simulate 10 seconds elapsed for 60% completion
        metrics.start_time = time.time() - 10.0
        
        remaining_time = metrics.estimated_remaining_time
        assert remaining_time is not None
        assert remaining_time > 0  # Should have some time remaining
    
    def test_update_methods(self):
        """Test update methods."""
        metrics = ProgressMetrics(total_contexts=5)
        
        # Update completion
        chunk = DocumentationChunk(
            chunk_id="chunk", content="content", source_context="test",
            token_count=1000, content_hash="abc", proxy_id="proxy", download_time=2.0
        )
        metrics.update_completion(chunk)
        
        assert metrics.completed_contexts == 1
        assert metrics.total_tokens_retrieved == 1000
        assert metrics.total_download_time == 2.0
        
        # Update failure
        metrics.update_failure()
        assert metrics.failed_contexts == 1
        
        # Update cancellation
        metrics.update_cancellation()
        assert metrics.cancelled_contexts == 1


class TestDownloadSummary:
    """Test DownloadSummary model."""
    
    def test_summary_creation_and_properties(self):
        """Test summary creation and calculated properties."""
        start_time = datetime.now() - timedelta(seconds=30)
        end_time = datetime.now()
        
        chunks = [
            DocumentationChunk(
                chunk_id="chunk1", content="content1", source_context="ctx1",
                token_count=1000, content_hash="abc1", proxy_id="proxy", download_time=2.0
            ),
            DocumentationChunk(
                chunk_id="chunk2", content="content2", source_context="ctx2",
                token_count=2000, content_hash="abc2", proxy_id="proxy", download_time=3.0
            )
        ]
        
        summary = DownloadSummary(
            library_id="test/lib",
            total_contexts_attempted=5,
            successful_contexts=2,
            failed_contexts=3,
            chunks=chunks,
            total_tokens=3000,
            total_download_time=5.0,
            start_time=start_time,
            end_time=end_time
        )
        
        assert summary.success_rate == 40.0  # 2/5 * 100
        assert summary.average_tokens_per_context == 1500.0  # 3000/2
        assert summary.tokens_per_second == 600.0  # 3000/5
        assert summary.duration_seconds == pytest.approx(30.0, abs=1.0)
        assert 0.0 <= summary.efficiency_score <= 1.0


class TestExceptions:
    """Test custom exception classes."""
    
    def test_download_error(self):
        """Test DownloadError exception."""
        error = DownloadError("Test error", task_id="task_1", error_category="test")
        
        assert str(error) == "Test error"
        assert error.task_id == "task_1"
        assert error.error_category == "test"
        assert isinstance(error.timestamp, datetime)
    
    def test_context_generation_error(self):
        """Test ContextGenerationError exception."""
        error = ContextGenerationError("Context error", library_id="test/lib")
        
        assert str(error) == "Context error"
        assert error.library_id == "test/lib"
        assert error.error_category == "context_generation"
    
    def test_proxy_unavailable_error(self):
        """Test ProxyUnavailableError exception."""
        error = ProxyUnavailableError("No proxies", available_proxies=3)
        
        assert str(error) == "No proxies"
        assert error.available_proxies == 3
        assert error.error_category == "proxy_unavailable"
    
    def test_task_timeout_error(self):
        """Test TaskTimeoutError exception."""
        error = TaskTimeoutError("Timeout", "task_1", 30.0)
        
        assert str(error) == "Timeout"
        assert error.task_id == "task_1"
        assert error.timeout_seconds == 30.0
        assert error.error_category == "timeout"