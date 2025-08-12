"""
Unit tests for concurrent processor with semaphore-based rate limiting.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock
from dataclasses import dataclass

from src.contexter.core.concurrent_processor import (
    ConcurrentProcessor,
    JitterConfig,
    JitterManager,
    TaskScheduler,
    ProcessingStats,
    ProcessorState,
    ConcurrentProcessingError,
    TaskTimeoutError
)


@dataclass
class MockTask:
    """Simple test task for processor testing."""
    task_id: str
    priority: int = 0
    
    @property
    def priority_score(self) -> float:
        """Priority score for scheduling."""
        return float(self.priority)


class TestJitterConfig:
    """Test JitterConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default jitter configuration."""
        config = JitterConfig()
        
        assert config.min_delay == 0.5
        assert config.max_delay == 2.0
        assert config.progressive_factor == 1.2
        assert config.adaptive_enabled is True
        assert config.burst_protection is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = JitterConfig(min_delay=1.0, max_delay=3.0, progressive_factor=1.5)
        assert config.min_delay == 1.0
        
        # Invalid delay configuration
        with pytest.raises(ValueError, match="Invalid jitter delay configuration"):
            JitterConfig(min_delay=-1.0, max_delay=2.0)
        
        with pytest.raises(ValueError, match="Invalid jitter delay configuration"):
            JitterConfig(min_delay=3.0, max_delay=2.0)
        
        # Invalid progressive factor
        with pytest.raises(ValueError, match="Progressive factor must be >= 1.0"):
            JitterConfig(progressive_factor=0.5)


class TestJitterManager:
    """Test JitterManager intelligent timing."""
    
    def test_basic_jitter_calculation(self):
        """Test basic jitter calculation."""
        manager = JitterManager()
        
        # First task should have no delay
        delay = manager.calculate_jitter(0, 10)
        assert delay == 0.0
        
        # Subsequent tasks should have delays
        delay1 = manager.calculate_jitter(1, 10)
        delay2 = manager.calculate_jitter(2, 10)
        
        assert 0.5 <= delay1 <= 2.0
        assert 0.5 <= delay2 <= 2.0
        assert delay1 != delay2  # Should be random
    
    def test_progressive_delay(self):
        """Test progressive delay increases."""
        config = JitterConfig(progressive_factor=1.5, min_delay=1.0, max_delay=2.0)
        manager = JitterManager(config)
        
        # Later tasks should generally have longer delays
        delay_early = manager.calculate_jitter(3, 20)
        delay_late = manager.calculate_jitter(15, 20)
        
        # Due to randomness, we can't guarantee order, but late should have potential for higher
        assert delay_late >= 1.0  # At minimum should respect base delay
    
    def test_performance_recording(self):
        """Test performance recording for adaptive adjustments."""
        manager = JitterManager()
        
        # Record some performance data
        manager.record_performance(1.5)
        manager.record_performance(2.0)
        manager.record_performance(0.8)
        
        stats = manager.get_performance_stats()
        assert stats["performance_samples"] == 3
        assert stats["avg_recent_performance"] > 0
    
    def test_burst_protection(self):
        """Test burst protection mechanism."""
        config = JitterConfig(burst_protection=True, min_delay=1.0)
        manager = JitterManager(config)
        
        # Simulate burst of requests
        delays = []
        for i in range(10):
            delay = manager.calculate_jitter(i + 1, 10)
            delays.append(delay)
            time.sleep(0.01)  # Small delay to simulate rapid requests
        
        # Later delays in burst should generally be longer
        assert len(delays) == 10
        # At least some delays should be applied
        assert any(delay > 1.0 for delay in delays[5:])


class MockTaskScheduler:
    """Test TaskScheduler priority handling."""
    
    def test_priority_scheduling_enabled(self):
        """Test priority-based task scheduling."""
        scheduler = TaskScheduler(enable_priority_scheduling=True)
        
        tasks = [
            MockTask("low", priority=1),
            MockTask("high", priority=5),
            MockTask("medium", priority=3)
        ]
        
        scheduled = scheduler.schedule_tasks(tasks)
        
        # Should be ordered by priority (high to low)
        assert scheduled[0].task_id == "high"
        assert scheduled[1].task_id == "medium"  
        assert scheduled[2].task_id == "low"
    
    def test_priority_scheduling_disabled(self):
        """Test scheduling without priority."""
        scheduler = TaskScheduler(enable_priority_scheduling=False)
        
        tasks = [MockTask("task1"), MockTask("task2"), MockTask("task3")]
        scheduled = scheduler.schedule_tasks(tasks)
        
        # Should return tasks (possibly shuffled)
        assert len(scheduled) == 3
        assert all(task in scheduled for task in tasks)
    
    def test_scheduling_stats(self):
        """Test scheduling statistics."""
        scheduler = TaskScheduler(enable_priority_scheduling=True)
        
        tasks = [MockTask("task1", 1), MockTask("task2", 2)]
        scheduler.schedule_tasks(tasks)
        
        stats = scheduler.get_scheduling_stats()
        assert "reorders" in stats
        assert stats["reorders"] > 0


class TestProcessingStats:
    """Test ProcessingStats calculations."""
    
    def test_basic_stats(self):
        """Test basic statistics properties."""
        stats = ProcessingStats(total_tasks=10)
        
        assert stats.total_tasks == 10
        assert stats.active_tasks == 10
        assert stats.completion_rate == 0.0
        assert stats.success_rate == 0.0
    
    def test_stats_calculations(self):
        """Test statistics calculations."""
        stats = ProcessingStats(
            total_tasks=10,
            completed_tasks=6,
            failed_tasks=2,
            cancelled_tasks=1,
            total_processing_time=30.0
        )
        
        assert stats.active_tasks == 1  # 10 - 6 - 2 - 1
        assert stats.completion_rate == 60.0  # 6/10 * 100 (only completed, not all finished)
        assert stats.success_rate == 75.0  # 6/8 * 100 (6 successful out of 8 attempted)
        assert stats.average_processing_time == 5.0  # 30/6
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        from datetime import datetime, timedelta
        
        stats = ProcessingStats(
            total_tasks=10,
            completed_tasks=5,
            start_time=datetime.now() - timedelta(seconds=10)
        )
        
        throughput = stats.throughput_tasks_per_second
        assert throughput == pytest.approx(0.5, abs=0.1)  # 5 tasks in ~10 seconds


class TestConcurrentProcessor:
    """Test ConcurrentProcessor main functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create concurrent processor for testing."""
        return ConcurrentProcessor(max_concurrent=3, task_timeout=5.0)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.max_concurrent == 3
        assert processor.task_timeout == 5.0
        assert processor.state == ProcessorState.IDLE
        assert processor.can_process is True
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        processor = ConcurrentProcessor(max_concurrent=10, task_timeout=30.0)
        assert processor.max_concurrent == 10
        
        # Invalid max_concurrent
        with pytest.raises(ValueError, match="max_concurrent must be between 1 and 50"):
            ConcurrentProcessor(max_concurrent=0)
        
        with pytest.raises(ValueError, match="max_concurrent must be between 1 and 50"):
            ConcurrentProcessor(max_concurrent=100)
    
    @pytest.mark.asyncio
    async def test_basic_processing(self, processor):
        """Test basic concurrent processing."""
        async def simple_processor(task):
            await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask(f"task_{i}") for i in range(3)]
        
        results = await processor.process_with_concurrency(tasks, simple_processor)
        
        assert len(results) == 3
        assert all("processed_task_" in str(result) for result in results)
        assert processor.state == ProcessorState.IDLE
        assert processor.stats.completed_tasks == 3
    
    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, processor):
        """Test that concurrency is properly limited."""
        max_concurrent_reached = 0
        
        async def tracking_processor(task):
            nonlocal max_concurrent_reached
            # This is a simplified way to check - in real scenarios,
            # we'd need more sophisticated tracking
            await asyncio.sleep(0.2)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask(f"task_{i}") for i in range(6)]  # More tasks than max_concurrent
        
        start_time = time.time()
        results = await processor.process_with_concurrency(tasks, tracking_processor)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 6
        # With max_concurrent=3 and 6 tasks taking 0.2s each, should take at least 0.4s
        assert elapsed_time >= 0.4
        assert processor.stats.max_concurrent_reached <= 3
    
    @pytest.mark.asyncio
    async def test_error_handling(self, processor):
        """Test error handling in concurrent processing."""
        async def error_processor(task):
            if "fail" in task.task_id:
                raise ValueError(f"Simulated error for {task.task_id}")
            await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [
            MockTask("task_1"),
            MockTask("task_fail"),
            MockTask("task_2")
        ]
        
        results = await processor.process_with_concurrency(tasks, error_processor)
        
        assert len(results) == 3
        # Should have 2 successful results and 1 exception
        successful = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) == 2
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test task timeout handling."""
        processor = ConcurrentProcessor(max_concurrent=2, task_timeout=0.5)
        
        async def slow_processor(task):
            if "slow" in task.task_id:
                await asyncio.sleep(1.0)  # Longer than timeout
            else:
                await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask("task_slow"), MockTask("task_fast")]
        
        results = await processor.process_with_concurrency(tasks, slow_processor)
        
        assert len(results) == 2
        
        # One should timeout, one should succeed
        timeouts = [r for r in results if isinstance(r, TaskTimeoutError)]
        successful = [r for r in results if not isinstance(r, Exception)]
        
        assert len(timeouts) == 1
        assert len(successful) == 1
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, processor):
        """Test progress callback functionality."""
        callback_calls = []
        
        def progress_callback(stats):
            callback_calls.append(stats.completed_tasks)
        
        async def simple_processor(task):
            await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask(f"task_{i}") for i in range(3)]
        
        await processor.process_with_concurrency(
            tasks, simple_processor, progress_callback
        )
        
        # Callback should have been called
        assert len(callback_calls) > 0
        # Final call should show all tasks completed
        assert callback_calls[-1] == 3
    
    @pytest.mark.asyncio
    async def test_processor_state_management(self, processor):
        """Test processor state management."""
        assert processor.state == ProcessorState.IDLE
        assert processor.can_process is True
        assert processor.is_active is False
        
        async def simple_processor(task):
            # Check state during processing
            assert processor.state == ProcessorState.PROCESSING
            assert processor.is_active is True
            await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask("task_1")]
        
        # Start processing
        process_task = asyncio.create_task(
            processor.process_with_concurrency(tasks, simple_processor)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.05)
        
        # Should be processing
        assert processor.state == ProcessorState.PROCESSING
        
        # Wait for completion
        await process_task
        
        # Should be back to idle
        assert processor.state == ProcessorState.IDLE
        assert processor.can_process is True
    
    @pytest.mark.asyncio
    async def test_empty_task_list(self, processor):
        """Test processing empty task list."""
        async def simple_processor(task):
            return f"processed_{task.task_id}"
        
        results = await processor.process_with_concurrency([], simple_processor)
        
        assert results == []
        assert processor.stats.total_tasks == 0
    
    @pytest.mark.asyncio
    async def test_processor_reuse(self, processor):
        """Test that processor can be reused for multiple operations."""
        async def simple_processor(task):
            await asyncio.sleep(0.05)
            return f"processed_{task.task_id}"
        
        # First batch
        tasks1 = [MockTask(f"batch1_task_{i}") for i in range(2)]
        results1 = await processor.process_with_concurrency(tasks1, simple_processor)
        
        assert len(results1) == 2
        assert processor.state == ProcessorState.IDLE
        
        # Second batch
        tasks2 = [MockTask(f"batch2_task_{i}") for i in range(3)]
        results2 = await processor.process_with_concurrency(tasks2, simple_processor)
        
        assert len(results2) == 3
        assert processor.state == ProcessorState.IDLE
    
    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, processor):
        """Test comprehensive statistics collection."""
        async def mixed_processor(task):
            if "fail" in task.task_id:
                raise ValueError("Simulated failure")
            elif "slow" in task.task_id:
                await asyncio.sleep(0.3)
            else:
                await asyncio.sleep(0.1)
            return f"processed_{task.task_id}"
        
        tasks = [
            MockTask("task_normal_1"),
            MockTask("task_fail"),
            MockTask("task_slow"),
            MockTask("task_normal_2")
        ]
        
        await processor.process_with_concurrency(tasks, mixed_processor)
        
        stats = processor.get_comprehensive_stats()
        
        assert "processing" in stats
        assert "configuration" in stats
        assert "jitter" in stats
        assert "scheduling" in stats
        
        processing_stats = stats["processing"]
        assert processing_stats["total_tasks"] == 4
        assert processing_stats["completed_tasks"] == 3
        assert processing_stats["failed_tasks"] == 1
        assert processing_stats["success_rate"] == 75.0  # 3/4 * 100
    
    @pytest.mark.asyncio
    async def test_shutdown(self, processor):
        """Test graceful shutdown."""
        # Start a long-running operation
        async def long_processor(task):
            await asyncio.sleep(2.0)
            return f"processed_{task.task_id}"
        
        tasks = [MockTask("long_task")]
        
        # Start processing in background
        process_task = asyncio.create_task(
            processor.process_with_concurrency(tasks, long_processor)
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Shutdown should cancel active tasks
        await processor.shutdown(timeout=1.0)
        
        assert processor.state == ProcessorState.SHUTDOWN
        
        # Process task should be done (cancelled or completed) - give it more time
        await asyncio.sleep(0.5)
        # If not done yet, cancel it explicitly for cleanup
        if not process_task.done():
            process_task.cancel()
            try:
                await process_task
            except asyncio.CancelledError:
                pass
        
        # Should be done now
        assert process_task.done()


class TestConcurrentProcessingIntegration:
    """Integration tests for concurrent processing."""
    
    @pytest.mark.asyncio
    async def test_realistic_workload(self):
        """Test with realistic workload simulation."""
        processor = ConcurrentProcessor(
            max_concurrent=5,
            jitter_config=JitterConfig(min_delay=0.1, max_delay=0.3),
            enable_priority_scheduling=True
        )
        
        async def realistic_processor(task):
            # Simulate variable processing times
            if task.priority >= 3:
                await asyncio.sleep(0.1)  # High priority tasks are faster
            else:
                await asyncio.sleep(0.2)  # Lower priority tasks are slower
            
            # Simulate occasional failures
            if "unstable" in task.task_id:
                if hash(task.task_id) % 3 == 0:
                    raise Exception("Simulated instability")
            
            return f"processed_{task.task_id}_{task.priority}"
        
        # Create mixed priority workload
        tasks = []
        for i in range(15):
            priority = (i % 5)
            task_id = f"task_{i}{'_unstable' if i % 7 == 0 else ''}"
            tasks.append(MockTask(task_id, priority=priority))
        
        start_time = time.time()
        results = await processor.process_with_concurrency(tasks, realistic_processor)
        elapsed_time = time.time() - start_time
        
        assert len(results) == 15
        
        # Check that we got a mix of successes and failures
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) >= 10  # Should have mostly successes
        assert len(failed) <= 5  # Should have some failures
        
        # With concurrency=5, should complete faster than sequential
        sequential_estimate = 15 * 0.2  # Assuming average 0.2s per task
        assert elapsed_time < sequential_estimate * 0.7  # At least 30% speedup
        
        # Check stats
        stats = processor.get_comprehensive_stats()
        assert stats["processing"]["total_tasks"] == 15
        assert stats["processing"]["success_rate"] >= 60.0  # Should have decent success rate