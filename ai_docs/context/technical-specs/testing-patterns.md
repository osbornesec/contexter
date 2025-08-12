# RAG Testing Framework Technical Specifications

## Overview
- **Purpose**: Comprehensive testing framework for RAG system accuracy, performance, and integration
- **Version**: pytest 8.4+ with async support
- **Last Updated**: 2025-01-12

## Key Concepts

### Testing Categories
- **Accuracy Testing**: Search relevance, recall@K, precision, NDCG validation  
- **Performance Testing**: Latency benchmarking, throughput validation, load testing
- **Integration Testing**: End-to-end pipeline validation, component interaction
- **Regression Testing**: Performance degradation detection, accuracy drift monitoring

### Core Testing Architecture
```python
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
import pytest
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field
import logging
from pathlib import Path
from contextlib import asynccontextmanager

@dataclass
class TestResult:
    """Standardized test result container for all RAG tests."""
    test_name: str
    passed: bool
    score: Optional[float]
    execution_time: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'score': self.score,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'error_message': self.error_message
        }

@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for RAG evaluation."""
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_10: float
    ndcg_at_10: float
    mrr: float  # Mean Reciprocal Rank
    map_score: float  # Mean Average Precision
    
    @property
    def summary_score(self) -> float:
        """Weighted average of key metrics for overall quality assessment."""
        return (
            0.3 * self.recall_at_10 +
            0.2 * self.precision_at_10 +
            0.3 * self.ndcg_at_10 +
            0.2 * self.mrr
        )
```

## Implementation Patterns

### Pattern: Async Test Framework Setup
```python
class RAGTestFramework:
    """Main orchestrator for RAG system testing."""
    
    def __init__(self, config_path: Path):
        self.config = self._load_config(config_path)
        self.test_results: List[TestResult] = []
        self.ground_truth_data: Optional[Dict[str, Any]] = None
        self.search_engine = None  # Injected during setup
        self.embedding_service = None
        self.vector_store = None
        
    async def setup_test_environment(self):
        """Initialize all test dependencies and connections."""
        # Setup test database connections
        await self._setup_test_vector_store()
        
        # Initialize embedding service with test configuration
        await self._setup_test_embedding_service()
        
        # Load ground truth datasets
        self.ground_truth_data = await self._load_ground_truth()
        
        # Setup search engine with test indices
        await self._setup_test_search_engine()
        
        logging.info("RAG test environment initialized successfully")
    
    async def teardown_test_environment(self):
        """Clean up test resources and connections."""
        if self.search_engine:
            await self.search_engine.close()
        if self.embedding_service:
            await self.embedding_service.close()
        if self.vector_store:
            await self.vector_store.close()
    
    @asynccontextmanager
    async def test_session(self):
        """Context manager for complete test session lifecycle."""
        await self.setup_test_environment()
        try:
            yield self
        finally:
            await self.teardown_test_environment()
```

### Pattern: Accuracy Testing Implementation
```python
class AccuracyTester:
    """Specialized tester for RAG accuracy validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_engine = None
        self.ground_truth = None
        
    @pytest.fixture(autouse=True)
    async def setup_accuracy_test(self, rag_test_framework):
        """Auto-setup fixture for accuracy tests."""
        self.search_engine = rag_test_framework.search_engine
        self.ground_truth = rag_test_framework.ground_truth_data
        
    async def test_search_recall_at_k(self) -> TestResult:
        """Test recall@K metrics for search relevance."""
        start_time = time.time()
        
        try:
            recall_scores = {'recall@1': [], 'recall@5': [], 'recall@10': []}
            
            for query_data in self.ground_truth['test_queries']:
                query = query_data['query']
                relevant_docs = set(query_data['relevant_document_ids'])
                
                # Perform hybrid search
                search_results = await self.search_engine.search(
                    query=query,
                    top_k=10,
                    search_type='hybrid',
                    include_metadata=True
                )
                
                retrieved_docs = [r['result_id'] for r in search_results]
                
                # Calculate recall@K for different K values
                for k in [1, 5, 10]:
                    retrieved_k = set(retrieved_docs[:k])
                    recall_k = len(retrieved_k & relevant_docs) / len(relevant_docs) if relevant_docs else 0.0
                    recall_scores[f'recall@{k}'].append(recall_k)
            
            # Calculate average recall scores
            avg_recalls = {
                metric: np.mean(scores) 
                for metric, scores in recall_scores.items()
            }
            
            # Determine pass/fail based on thresholds
            passed = avg_recalls['recall@10'] >= self.config['accuracy_thresholds']['recall_at_10']
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name='search_recall_at_k',
                passed=passed,
                score=avg_recalls['recall@10'],
                execution_time=execution_time,
                metadata={
                    'detailed_recalls': avg_recalls,
                    'query_count': len(self.ground_truth['test_queries']),
                    'threshold': self.config['accuracy_thresholds']['recall_at_10']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='search_recall_at_k',
                passed=False,
                score=0.0,
                execution_time=execution_time,
                metadata={'error_type': type(e).__name__},
                error_message=str(e)
            )
    
    async def test_ranking_quality_ndcg(self) -> TestResult:
        """Test ranking quality using NDCG@10."""
        start_time = time.time()
        
        try:
            ndcg_scores = []
            
            for query_data in self.ground_truth['test_queries']:
                if 'document_relevance_scores' not in query_data:
                    continue
                    
                query = query_data['query']
                relevance_scores = query_data['document_relevance_scores']
                
                search_results = await self.search_engine.search(
                    query=query,
                    top_k=10,
                    search_type='hybrid'
                )
                
                # Calculate NDCG@10
                ndcg = self._calculate_ndcg(search_results, relevance_scores, k=10)
                ndcg_scores.append(ndcg)
            
            avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
            passed = avg_ndcg >= self.config['accuracy_thresholds']['ndcg_at_10']
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name='ranking_quality_ndcg',
                passed=passed,
                score=avg_ndcg,
                execution_time=execution_time,
                metadata={
                    'ndcg_distribution': {
                        'mean': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
                        'std': float(np.std(ndcg_scores)) if ndcg_scores else 0.0,
                        'min': float(np.min(ndcg_scores)) if ndcg_scores else 0.0,
                        'max': float(np.max(ndcg_scores)) if ndcg_scores else 0.0
                    },
                    'query_count': len(ndcg_scores),
                    'threshold': self.config['accuracy_thresholds']['ndcg_at_10']
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='ranking_quality_ndcg',
                passed=False,
                score=0.0,
                execution_time=execution_time,
                metadata={'error_type': type(e).__name__},
                error_message=str(e)
            )
    
    def _calculate_ndcg(
        self, 
        search_results: List[Dict], 
        relevance_scores: Dict[str, float], 
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        # DCG calculation
        dcg = 0.0
        for i, result in enumerate(search_results[:k]):
            doc_id = result['result_id']
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG calculation (perfect ranking)
        sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(
            rel / np.log2(i + 2) 
            for i, rel in enumerate(sorted_relevances)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
```

### Pattern: Performance Testing Implementation
```python
class PerformanceTester:
    """Specialized tester for RAG performance validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.search_engine = None
        self.ingestion_pipeline = None
        
    async def test_search_latency_percentiles(self) -> TestResult:
        """Test search latency percentiles under normal load."""
        start_time = time.time()
        
        try:
            latencies = []
            test_queries = self._generate_performance_test_queries()
            
            # Warm up the system
            for _ in range(10):
                await self.search_engine.search(
                    query="warm up query",
                    top_k=5
                )
            
            # Measure latencies
            for query in test_queries:
                query_start = time.time()
                
                await self.search_engine.search(
                    query=query,
                    top_k=10,
                    search_type='hybrid'
                )
                
                query_latency = time.time() - query_start
                latencies.append(query_latency)
            
            # Calculate percentiles
            latencies_ms = [l * 1000 for l in latencies]  # Convert to milliseconds
            percentiles = {
                'p50': np.percentile(latencies_ms, 50),
                'p95': np.percentile(latencies_ms, 95),
                'p99': np.percentile(latencies_ms, 99),
                'mean': np.mean(latencies_ms),
                'max': np.max(latencies_ms)
            }
            
            # Check against thresholds
            thresholds = self.config['performance_thresholds']['search_latency_ms']
            passed = (
                percentiles['p95'] <= thresholds['p95'] and
                percentiles['p99'] <= thresholds['p99']
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name='search_latency_percentiles',
                passed=passed,
                score=percentiles['p95'],  # Use p95 as primary score
                execution_time=execution_time,
                metadata={
                    'latency_percentiles_ms': percentiles,
                    'thresholds_ms': thresholds,
                    'query_count': len(test_queries),
                    'total_test_time': execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='search_latency_percentiles',
                passed=False,
                score=float('inf'),
                execution_time=execution_time,
                metadata={'error_type': type(e).__name__},
                error_message=str(e)
            )
    
    async def test_concurrent_search_throughput(self) -> TestResult:
        """Test search throughput under concurrent load."""
        start_time = time.time()
        
        try:
            concurrent_users = self.config['performance_test']['concurrent_users']
            queries_per_user = self.config['performance_test']['queries_per_user']
            
            async def user_session(user_id: int) -> List[float]:
                """Simulate a user session with multiple queries."""
                session_latencies = []
                test_queries = self._generate_performance_test_queries(queries_per_user)
                
                for query in test_queries:
                    query_start = time.time()
                    
                    await self.search_engine.search(
                        query=query,
                        top_k=10,
                        search_type='hybrid'
                    )
                    
                    query_latency = time.time() - query_start
                    session_latencies.append(query_latency)
                
                return session_latencies
            
            # Run concurrent user sessions
            load_test_start = time.time()
            
            tasks = [
                user_session(user_id) 
                for user_id in range(concurrent_users)
            ]
            
            session_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            load_test_duration = time.time() - load_test_start
            
            # Process results
            all_latencies = []
            successful_sessions = 0
            
            for result in session_results:
                if isinstance(result, list):
                    all_latencies.extend(result)
                    successful_sessions += 1
            
            total_queries = len(all_latencies)
            throughput_qps = total_queries / load_test_duration if load_test_duration > 0 else 0
            
            # Performance metrics
            latencies_ms = [l * 1000 for l in all_latencies]
            performance_metrics = {
                'throughput_qps': throughput_qps,
                'concurrent_users': concurrent_users,
                'successful_sessions': successful_sessions,
                'total_queries': total_queries,
                'avg_latency_ms': np.mean(latencies_ms) if latencies_ms else 0,
                'p95_latency_ms': np.percentile(latencies_ms, 95) if latencies_ms else 0,
                'load_test_duration': load_test_duration
            }
            
            # Check against thresholds
            throughput_threshold = self.config['performance_thresholds']['min_throughput_qps']
            latency_threshold = self.config['performance_thresholds']['search_latency_ms']['p95']
            
            passed = (
                throughput_qps >= throughput_threshold and
                performance_metrics['p95_latency_ms'] <= latency_threshold and
                successful_sessions >= concurrent_users * 0.95  # 95% success rate
            )
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name='concurrent_search_throughput',
                passed=passed,
                score=throughput_qps,
                execution_time=execution_time,
                metadata=performance_metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name='concurrent_search_throughput',
                passed=False,
                score=0.0,
                execution_time=execution_time,
                metadata={'error_type': type(e).__name__},
                error_message=str(e)
            )
    
    def _generate_performance_test_queries(self, count: int = 100) -> List[str]:
        """Generate realistic test queries for performance testing."""
        # This would typically load from a curated set of realistic queries
        base_queries = [
            "how to implement async patterns in Python",
            "FastAPI authentication best practices", 
            "vector database performance optimization",
            "machine learning model deployment strategies",
            "microservices architecture patterns",
            "database indexing optimization techniques",
            "REST API security implementation",
            "cloud infrastructure scaling patterns"
        ]
        
        # Expand with variations
        queries = []
        for i in range(count):
            base_query = base_queries[i % len(base_queries)]
            # Add slight variations to avoid caching
            variation = f"{base_query} example {i % 20}"
            queries.append(variation)
        
        return queries
```

## Common Gotchas

### Gotcha: Async Test Isolation
- **Problem**: Async tests can share event loops and cause resource leakage
- **Solution**: Use pytest-asyncio with proper fixture scoping
- **Example**:
```python
@pytest.mark.asyncio
@pytest.fixture(scope="function")  # Ensure fresh fixture per test
async def isolated_rag_system():
    """Provides isolated RAG system instance per test."""
    system = RAGSystem(test_config)
    await system.initialize()
    yield system
    await system.cleanup()  # Critical: cleanup resources

# Proper test isolation
@pytest.mark.asyncio
async def test_search_accuracy(isolated_rag_system):
    results = await isolated_rag_system.search("test query")
    assert len(results) > 0
    # Cleanup is automatic via fixture teardown
```

### Gotcha: Ground Truth Data Management
- **Problem**: Ground truth datasets become stale or don't match production data patterns
- **Solution**: Implement automated ground truth validation and refresh
- **Example**:
```python
class GroundTruthManager:
    """Manages ground truth datasets with validation."""
    
    async def validate_ground_truth_freshness(self) -> bool:
        """Ensure ground truth data is current."""
        dataset_age = time.time() - self.ground_truth['created_timestamp']
        max_age = self.config['ground_truth_max_age_days'] * 24 * 3600
        
        if dataset_age > max_age:
            logging.warning(f"Ground truth dataset is {dataset_age/86400:.1f} days old")
            return False
        
        # Validate query patterns match current usage
        current_queries = await self._sample_recent_queries()
        similarity = self._calculate_query_distribution_similarity(
            self.ground_truth['queries'], 
            current_queries
        )
        
        return similarity > 0.7  # 70% similarity threshold

    async def refresh_ground_truth_if_needed(self):
        """Auto-refresh ground truth when it becomes stale."""
        if not await self.validate_ground_truth_freshness():
            logging.info("Refreshing ground truth dataset")
            await self._generate_updated_ground_truth()
```

### Gotcha: Performance Test Environment Consistency
- **Problem**: Performance tests give inconsistent results due to environment variations
- **Solution**: Implement environment normalization and resource controls
- **Example**:
```python
@pytest.fixture(scope="session")
async def performance_test_environment():
    """Ensures consistent performance test environment."""
    # Set resource limits
    resource.setrlimit(resource.RLIMIT_NPROC, (1000, 1000))
    
    # Clear system caches
    if platform.system() == 'Linux':
        os.system('sync && echo 3 > /proc/sys/vm/drop_caches')
    
    # Warm up services
    await warm_up_all_services()
    
    # Wait for system to stabilize
    await asyncio.sleep(5)
    
    yield
    
    # Cleanup
    await cleanup_test_environment()

@pytest.mark.asyncio
async def test_search_performance(performance_test_environment, rag_system):
    # Performance test runs in controlled environment
    start_time = time.time()
    results = await rag_system.search("performance test query")
    latency = time.time() - start_time
    
    assert latency < 0.05  # 50ms threshold
```

### Gotcha: Test Data Cleanup
- **Problem**: Test data accumulates and affects subsequent test runs
- **Solution**: Implement comprehensive cleanup with verification
- **Example**:
```python
@pytest.fixture(autouse=True)
async def ensure_test_cleanup():
    """Ensures complete test cleanup even if tests fail."""
    test_session_id = str(uuid.uuid4())
    
    yield test_session_id
    
    # Cleanup test data with verification
    cleanup_successful = await cleanup_test_data(test_session_id)
    
    if not cleanup_successful:
        logging.error(f"Failed to cleanup test session: {test_session_id}")
        # Force cleanup or fail the test session
        await force_cleanup_test_data(test_session_id)
    
    # Verify cleanup
    remaining_data = await check_remaining_test_data(test_session_id)
    if remaining_data:
        pytest.fail(f"Test cleanup incomplete: {remaining_data}")
```

## Best Practices

### Testing Configuration Management
```python
# test_config.yaml
accuracy_thresholds:
  recall_at_1: 0.8
  recall_at_5: 0.9
  recall_at_10: 0.95
  precision_at_10: 0.85
  ndcg_at_10: 0.8

performance_thresholds:
  search_latency_ms:
    p50: 20
    p95: 50
    p99: 100
  min_throughput_qps: 100
  max_memory_mb: 512

test_data:
  ground_truth_path: "test_data/ground_truth.json"
  synthetic_queries_count: 1000
  performance_test_duration: 300  # seconds

integration_test:
  vector_store_url: "http://localhost:6333"
  embedding_service_url: "http://localhost:8001"
  test_database: "rag_test"
```

### Comprehensive Test Suite Organization
```python
# tests/conftest.py
@pytest.fixture(scope="session")
async def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent / "test_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session") 
async def rag_test_framework(test_config):
    """Session-wide RAG test framework."""
    framework = RAGTestFramework(test_config)
    async with framework.test_session():
        yield framework

# tests/test_accuracy.py
@pytest.mark.asyncio
@pytest.mark.accuracy
class TestRAGAccuracy:
    async def test_search_recall(self, rag_test_framework):
        tester = AccuracyTester(rag_test_framework.config)
        result = await tester.test_search_recall_at_k()
        assert result.passed, f"Recall test failed: {result.error_message}"
    
    async def test_ranking_quality(self, rag_test_framework):
        tester = AccuracyTester(rag_test_framework.config)
        result = await tester.test_ranking_quality_ndcg()
        assert result.passed, f"NDCG test failed: {result.error_message}"

# tests/test_performance.py
@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.slow
class TestRAGPerformance:
    async def test_latency_percentiles(self, rag_test_framework):
        tester = PerformanceTester(rag_test_framework.config)
        result = await tester.test_search_latency_percentiles()
        assert result.passed, f"Latency test failed: {result.error_message}"
```

### CI/CD Integration Pattern
```python
# scripts/run_tests.py
async def run_test_suite():
    """Run complete RAG test suite with proper reporting."""
    
    # Setup test environment
    await setup_ci_test_environment()
    
    try:
        # Run test categories in order
        accuracy_results = await run_accuracy_tests()
        performance_results = await run_performance_tests()
        integration_results = await run_integration_tests()
        
        # Generate comprehensive report
        test_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'accuracy': accuracy_results,
            'performance': performance_results,
            'integration': integration_results,
            'summary': {
                'total_tests': len(accuracy_results) + len(performance_results) + len(integration_results),
                'passed': sum(r.passed for r in [*accuracy_results, *performance_results, *integration_results]),
                'overall_success': all(r.passed for r in [*accuracy_results, *performance_results, *integration_results])
            }
        }
        
        # Save report for CI
        with open('test_results.json', 'w') as f:
            json.dump(test_report, f, indent=2)
        
        return test_report['summary']['overall_success']
        
    finally:
        await cleanup_ci_test_environment()

if __name__ == "__main__":
    import sys
    success = asyncio.run(run_test_suite())
    sys.exit(0 if success else 1)
```

## Integration Points

### Vector Database Testing
- **Qdrant Integration**: Test vector operations, indexing, and search performance
- **Data Consistency**: Validate vector-document mapping integrity
- **Index Optimization**: Test HNSW parameter tuning effects

### Embedding Service Testing  
- **Voyage AI Integration**: Test embedding generation accuracy and consistency
- **Caching Validation**: Verify embedding cache hit rates and consistency
- **Batch Processing**: Test throughput and error handling for batch operations

### Search Engine Testing
- **Hybrid Search**: Validate semantic + keyword search combination
- **Result Ranking**: Test ranking algorithm effectiveness
- **Query Processing**: Validate query understanding and expansion

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Plugin](https://pytest-asyncio.readthedocs.io/)
- [Information Retrieval Evaluation Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [NDCG Calculation Methods](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [Load Testing Best Practices](https://docs.python.org/3/library/asyncio.html)

## Related Contexts

- `technical-specs/qdrant-vector-database.md` - Vector storage testing
- `technical-specs/voyage-ai-embedding.md` - Embedding service testing  
- `technical-specs/fastapi-integration.md` - API endpoint testing
- `code-patterns/async-client-patterns.py` - Async testing utilities
- `integration-guides/monitoring-integration.md` - Test metrics collection