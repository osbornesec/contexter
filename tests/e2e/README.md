# End-to-End (E2E) Tests for C7DocDownloader

This directory contains comprehensive end-to-end tests that validate the complete C7DocDownloader system functionality from CLI commands down to storage operations.

## Test Structure

### 📁 Test Files

- **`test_full_download_workflow.py`** - Complete download workflows and system integration
- **`test_cli_integration.py`** - CLI command integration and user interface testing  
- **`test_performance_scenarios.py`** - Performance, load testing, and scalability scenarios
- **`conftest.py`** - Shared fixtures and test configuration

### 🏷️ Test Categories

Tests are marked with pytest markers for easy filtering:

- `@pytest.mark.e2e` - All end-to-end tests
- `@pytest.mark.slow` - Long-running performance tests
- `@pytest.mark.asyncio` - Async test functions

## Running E2E Tests

### Quick Start

```bash
# Run all E2E tests (recommended)
python scripts/run_e2e_tests.py

# Run only quick tests (skip performance)  
python scripts/run_e2e_tests.py --quick

# Run only performance tests
python scripts/run_e2e_tests.py --performance-only

# Run with verbose output
python scripts/run_e2e_tests.py --verbose
```

### Manual pytest Commands

```bash
# Run all E2E tests
pytest tests/e2e/ -m "e2e"

# Run quick tests only
pytest tests/e2e/ -m "e2e and not slow"

# Run specific test file
pytest tests/e2e/test_full_download_workflow.py -v

# Run with coverage
pytest tests/e2e/ -m "e2e" --cov=src/contexter --cov-report=html
```

## Test Scenarios

### 🔄 Core Workflow Tests (`test_full_download_workflow.py`)

**Complete Library Download Workflow**
- End-to-end download → storage → retrieval cycle
- Proxy management integration
- Context7 API interaction
- Storage compression and integrity verification

**Multi-Library Batch Processing**
- Concurrent library downloads
- Resource management under load
- Error isolation between libraries

**Failure Recovery & Resilience**
- Network timeout handling
- Partial failure tolerance
- Retry mechanism validation
- Graceful degradation

**Storage Versioning**
- Multiple version storage
- Retention policy enforcement
- Version cleanup automation
- Latest version resolution

**Deduplication Integration**
- Hash-based exact duplicate removal
- Semantic similarity detection
- Content merging workflows
- Performance optimization

**System Health Monitoring**
- Component health checks
- Performance metrics collection
- Diagnostic information gathering

### 🖥️ CLI Integration Tests (`test_cli_integration.py`)

**Command Integration**
- `contexter status` - System health reporting
- `contexter download` - Library download operations
- `contexter config` - Configuration management

**Error Handling**
- Invalid command arguments
- Missing configuration files
- Network connectivity issues
- Graceful error recovery

**Output Formatting**
- Rich console formatting
- Progress indicators
- Error message clarity
- Help system completeness

**Configuration Management**
- Config file discovery
- Environment variable precedence
- Default value handling
- Validation workflows

### ⚡ Performance Tests (`test_performance_scenarios.py`)

**Concurrency & Throughput**
- High-concurrency download testing (15+ concurrent operations)
- Context processing throughput measurement
- Resource utilization optimization
- Bottleneck identification

**Memory Management**
- Memory usage under sustained load
- Memory leak detection
- Garbage collection efficiency
- Large dataset handling

**Storage Performance**
- Large dataset storage/retrieval timing
- Compression performance analysis
- I/O optimization validation
- Scalability testing

**Error Recovery Performance**
- Retry overhead measurement
- Circuit breaker efficiency
- Failure rate tolerance
- Recovery time optimization

**Batch Processing Scalability**
- Multi-library processing efficiency
- Resource scheduling optimization
- Concurrent library limits
- Throughput scaling characteristics

## Test Environment & Mocking

### 🎭 Mocked Components

E2E tests use comprehensive mocking to ensure:
- **Isolated testing** - No external API dependencies
- **Predictable responses** - Controlled test scenarios
- **Performance testing** - Realistic timing simulation
- **Error simulation** - Controlled failure injection

**Mocked Services:**
- BrightData Proxy Manager
- Context7 API Client  
- Network connections
- File system operations

### 📋 Test Fixtures

**Configuration Fixtures:**
- `test_config` - Complete test configuration
- `temp_storage_dir` - Isolated storage directory
- `mock_environment_vars` - Environment variable setup

**Component Fixtures:**
- `mock_proxy_manager` - Proxy management simulation
- `mock_context7_client` - API client simulation
- `sample_documentation_chunks` - Test data generation

**Scenario Fixtures:**
- `library_test_cases` - Multiple library test scenarios
- Performance benchmarking data

## Performance Targets

### 📊 Benchmarks

**Download Performance:**
- **Throughput:** > 5 contexts/second under normal load
- **Concurrency:** 15+ concurrent downloads without degradation
- **Memory:** < 100MB additional memory growth under load
- **Latency:** < 10 seconds for 50 context downloads

**Storage Performance:**
- **Write Speed:** > 20 chunks/second for storage operations
- **Compression:** > 30% size reduction ratio
- **Retrieval:** < 2 seconds for large dataset retrieval

**System Performance:**
- **Startup:** < 5 seconds for system initialization
- **Batch Processing:** > 3 contexts/second across multiple libraries
- **Error Recovery:** < 20 seconds for failure recovery cycles

### 🎯 Success Criteria

**Functionality:**
- ✅ All core workflows complete successfully
- ✅ Error conditions handled gracefully  
- ✅ Data integrity maintained throughout operations
- ✅ CLI commands provide expected user experience

**Performance:**
- ✅ Throughput targets met under normal load
- ✅ Memory usage remains stable under sustained operations
- ✅ Response times within acceptable ranges
- ✅ Scalability demonstrated across test scenarios

**Quality:**
- ✅ No data loss or corruption
- ✅ Proper cleanup of resources
- ✅ Consistent behavior across test runs
- ✅ Clear error messages and diagnostics

## Troubleshooting E2E Tests

### Common Issues

**Test Timeouts:**
```bash
# Increase timeout for slow systems
pytest tests/e2e/ --timeout=300
```

**Memory Issues:**
```bash
# Run tests with memory profiling
pytest tests/e2e/ --memray
```

**Async Issues:**
```bash
# Run with async debugging
pytest tests/e2e/ --asyncio-mode=debug
```

**Flaky Tests:**
```bash
# Re-run failed tests
pytest tests/e2e/ --lf --verbose
```

### Debug Mode

Enable comprehensive debugging:

```bash
export CONTEXTER_LOG_LEVEL=DEBUG
export PYTHONPATH=/home/michael/dev/contexter/src
pytest tests/e2e/ -v -s --tb=long
```

## Contributing

When adding new E2E tests:

1. **Follow naming patterns:** `test_<scenario>_<aspect>.py`
2. **Use appropriate markers:** `@pytest.mark.e2e`, `@pytest.mark.slow`
3. **Mock external dependencies:** Keep tests isolated and fast
4. **Include performance assertions:** Validate not just functionality but performance
5. **Document test scenarios:** Clear docstrings explaining test purpose
6. **Clean up resources:** Proper async cleanup in test teardown

## Integration with CI/CD

E2E tests are designed for integration with continuous integration:

```yaml
# Example CI configuration
test_e2e:
  script:
    - python scripts/run_e2e_tests.py --quick  # Quick tests for PR validation
    - python scripts/run_e2e_tests.py  # Full tests for main branch

test_performance:
  script:
    - python scripts/run_e2e_tests.py --performance-only  # Nightly performance tests
```

The E2E test suite provides comprehensive validation of the C7DocDownloader system, ensuring reliability, performance, and user experience across all components.