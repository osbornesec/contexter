# Automated Testing Framework

## Document Information
- **Version**: 1.0.0
- **Created**: 2025-01-15
- **System**: Contexter RAG Implementation
- **Scope**: Automated testing infrastructure, tools, and processes

## Overview

This document defines the comprehensive automated testing framework for the RAG system, including test infrastructure, automation tools, CI/CD integration, and continuous testing processes that ensure high-quality software delivery.

## Testing Framework Architecture

```mermaid
graph TB
    subgraph "Test Execution Engine"
        TestOrchestrator[Test Orchestrator<br/>Coordinates all test execution]
        TestScheduler[Test Scheduler<br/>Manages test timing and triggers]
        ResultCollector[Result Collector<br/>Aggregates test outcomes]
    end
    
    subgraph "Test Categories"
        UnitTests[Unit Tests<br/>Component isolation testing]
        IntegrationTests[Integration Tests<br/>Service interaction testing]
        E2ETests[End-to-End Tests<br/>Complete workflow testing]
        PerformanceTests[Performance Tests<br/>Load and stress testing]
        SecurityTests[Security Tests<br/>Vulnerability and compliance]
    end
    
    subgraph "Test Infrastructure"
        TestEnvironments[Test Environments<br/>Isolated testing infrastructure]
        TestData[Test Data Management<br/>Data generation and lifecycle]
        MockServices[Mock Services<br/>External dependency simulation]
        Monitoring[Test Monitoring<br/>Real-time test observability]
    end
    
    subgraph "Reporting & Analytics"
        TestReports[Test Reports<br/>Detailed test outcomes]
        QualityMetrics[Quality Metrics<br/>Trend analysis and KPIs]
        AlertingSystem[Alerting System<br/>Failure notifications]
        Dashboard[Quality Dashboard<br/>Real-time quality overview]
    end
    
    TestOrchestrator --> UnitTests
    TestOrchestrator --> IntegrationTests
    TestOrchestrator --> E2ETests
    TestOrchestrator --> PerformanceTests
    TestOrchestrator --> SecurityTests
    
    UnitTests --> TestEnvironments
    IntegrationTests --> TestData
    E2ETests --> MockServices
    PerformanceTests --> Monitoring
    SecurityTests --> TestReports
    
    ResultCollector --> QualityMetrics
    QualityMetrics --> Dashboard
    QualityMetrics --> AlertingSystem
    
    style TestOrchestrator fill:#e8f5e8
    style Dashboard fill:#fff3e0
```

## Test Infrastructure

### Test Environment Management

**Environment Configuration**:
```yaml
# config/test_environments.yaml
test_environments:
  unit:
    type: "local"
    isolation_level: "process"
    resources:
      memory: "512MB"
      cpu: "1 core"
    services:
      database: "sqlite_memory"
      cache: "local_memory"
      vector_store: "mock"
    
  integration:
    type: "docker_compose"
    isolation_level: "container"
    resources:
      memory: "2GB"
      cpu: "2 cores"
    services:
      database: "postgresql_test"
      cache: "redis_test"
      vector_store: "qdrant_test"
      embedding_service: "mock_voyage"
    
  e2e:
    type: "kubernetes"
    isolation_level: "namespace"
    resources:
      memory: "4GB"
      cpu: "4 cores"
    services:
      database: "postgresql"
      cache: "redis"
      vector_store: "qdrant"
      embedding_service: "voyage_sandbox"
    
  performance:
    type: "cloud_infrastructure"
    isolation_level: "vpc"
    resources:
      memory: "16GB"
      cpu: "8 cores"
    services:
      database: "postgresql_performance"
      cache: "redis_cluster"
      vector_store: "qdrant_cluster"
      embedding_service: "voyage_production"
```

**Environment Provisioning**:
```python
# infrastructure/test_environment_manager.py
class TestEnvironmentManager:
    def __init__(self, config: TestConfig):
        self.config = config
        self.environments = {}
        self.docker_client = docker.from_env()
        
    async def provision_environment(self, env_type: str) -> TestEnvironment:
        """Provision a test environment based on type."""
        env_config = self.config.environments[env_type]
        
        if env_type == "unit":
            return await self._provision_unit_environment(env_config)
        elif env_type == "integration":
            return await self._provision_integration_environment(env_config)
        elif env_type == "e2e":
            return await self._provision_e2e_environment(env_config)
        elif env_type == "performance":
            return await self._provision_performance_environment(env_config)
    
    async def _provision_integration_environment(self, config: Dict) -> TestEnvironment:
        """Provision Docker-based integration environment."""
        compose_config = {
            "version": "3.8",
            "services": {
                "qdrant": {
                    "image": "qdrant/qdrant:v1.7.0",
                    "ports": ["6333:6333"],
                    "environment": {
                        "QDRANT__SERVICE__HTTP_PORT": "6333"
                    },
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:6333/health"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "ports": ["6379:6379"],
                    "healthcheck": {
                        "test": ["CMD", "redis-cli", "ping"],
                        "interval": "10s",
                        "timeout": "3s",
                        "retries": 5
                    }
                },
                "postgres": {
                    "image": "postgres:15-alpine",
                    "environment": {
                        "POSTGRES_DB": "test_db",
                        "POSTGRES_USER": "test_user",
                        "POSTGRES_PASSWORD": "test_password"
                    },
                    "ports": ["5432:5432"],
                    "healthcheck": {
                        "test": ["CMD-SHELL", "pg_isready -U test_user -d test_db"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                }
            }
        }
        
        # Start Docker Compose environment
        compose_file = "docker-compose.test.yml"
        with open(compose_file, "w") as f:
            yaml.dump(compose_config, f)
        
        subprocess.run(["docker-compose", "-f", compose_file, "up", "-d"])
        
        # Wait for services to be healthy
        await self._wait_for_services_health(compose_config["services"])
        
        return TestEnvironment(
            env_type="integration",
            services={
                "qdrant": "http://localhost:6333",
                "redis": "redis://localhost:6379",
                "postgres": "postgresql://test_user:test_password@localhost:5432/test_db"
            },
            cleanup_callback=lambda: subprocess.run(["docker-compose", "-f", compose_file, "down"])
        )
    
    async def _wait_for_services_health(self, services: Dict[str, Any]):
        """Wait for all services to be healthy."""
        max_wait_time = 120  # 2 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            all_healthy = True
            
            for service_name in services.keys():
                try:
                    result = subprocess.run(
                        ["docker-compose", "ps", "-q", service_name],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0 or not result.stdout.strip():
                        all_healthy = False
                        break
                        
                    # Check health status
                    health_result = subprocess.run(
                        ["docker", "inspect", "--format='{{.State.Health.Status}}'", result.stdout.strip()],
                        capture_output=True,
                        text=True
                    )
                    
                    if "healthy" not in health_result.stdout:
                        all_healthy = False
                        break
                        
                except Exception as e:
                    logger.warning(f"Health check failed for {service_name}: {e}")
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("All services are healthy")
                return
                
            await asyncio.sleep(5)
        
        raise EnvironmentProvisioningError("Services failed to become healthy within timeout")
```

### Test Data Management

**Test Data Framework**:
```python
# testing/test_data_manager.py
class TestDataManager:
    def __init__(self, config: TestDataConfig):
        self.config = config
        self.data_generators = {}
        self.test_datasets = {}
        
    def register_generator(self, data_type: str, generator: Callable):
        """Register a data generator for a specific type."""
        self.data_generators[data_type] = generator
    
    async def generate_test_data(self, data_type: str, size: int, **kwargs) -> List[Any]:
        """Generate test data of specified type and size."""
        if data_type not in self.data_generators:
            raise ValueError(f"No generator registered for data type: {data_type}")
        
        generator = self.data_generators[data_type]
        return await generator(size, **kwargs)
    
    async def load_ground_truth_dataset(self, dataset_name: str) -> GroundTruthDataset:
        """Load curated ground truth dataset for accuracy testing."""
        dataset_path = self.config.ground_truth_datasets.get(dataset_name)
        if not dataset_path or not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Ground truth dataset not found: {dataset_name}")
        
        with open(dataset_path) as f:
            dataset_data = json.load(f)
        
        return GroundTruthDataset(
            name=dataset_name,
            queries=dataset_data["queries"],
            documents=dataset_data["documents"],
            relevance_judgments=dataset_data["relevance_judgments"],
            metadata=dataset_data.get("metadata", {})
        )

# Test data generators
@test_data_generator("technical_documents")
async def generate_technical_documents(size: int, **kwargs) -> List[Document]:
    """Generate realistic technical documentation for testing."""
    documents = []
    topics = ["python", "machine_learning", "web_development", "data_science", "api_design"]
    
    for i in range(size):
        topic = random.choice(topics)
        
        # Generate realistic technical content
        content = await generate_realistic_technical_content(topic)
        
        document = Document(
            document_id=f"test-doc-{i:04d}",
            library_id=f"test-{topic}",
            content=content,
            metadata={
                "topic": topic,
                "language": "python" if topic in ["python", "machine_learning", "data_science"] else "general",
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "generated": True,
                "test_category": kwargs.get("category", "general")
            }
        )
        
        documents.append(document)
    
    return documents

@test_data_generator("search_queries")
async def generate_search_queries(size: int, **kwargs) -> List[SearchQuery]:
    """Generate realistic search queries for testing."""
    query_templates = [
        "how to {action} {object} in {language}",
        "{concept} {application} best practices",
        "{problem} troubleshooting guide",
        "{tool} vs {alternative_tool} comparison",
        "{framework} {feature} implementation example"
    ]
    
    query_params = {
        "action": ["implement", "create", "debug", "optimize", "configure"],
        "object": ["functions", "classes", "APIs", "databases", "algorithms"],
        "language": ["Python", "JavaScript", "Java", "Go", "Rust"],
        "concept": ["async programming", "machine learning", "data structures", "design patterns"],
        "application": ["development", "implementation", "deployment", "testing"],
        "problem": ["memory leaks", "performance issues", "connection errors", "import errors"],
        "tool": ["Docker", "Kubernetes", "FastAPI", "SQLAlchemy", "pytest"],
        "alternative_tool": ["Podman", "Docker Swarm", "Flask", "Django ORM", "unittest"],
        "framework": ["FastAPI", "Django", "Flask", "PyTorch", "TensorFlow"],
        "feature": ["authentication", "routing", "middleware", "serialization", "validation"]
    }
    
    queries = []
    for i in range(size):
        template = random.choice(query_templates)
        
        # Fill template with random parameters
        query_text = template
        for param_type, values in query_params.items():
            if f"{{{param_type}}}" in query_text:
                query_text = query_text.replace(f"{{{param_type}}}", random.choice(values))
        
        query = SearchQuery(
            query_id=f"test-query-{i:04d}",
            text=query_text,
            intent=classify_query_intent(query_text),
            metadata={
                "generated": True,
                "complexity": calculate_query_complexity(query_text),
                "category": kwargs.get("category", "general")
            }
        )
        
        queries.append(query)
    
    return queries

async def generate_realistic_technical_content(topic: str) -> str:
    """Generate realistic technical content for a given topic."""
    content_templates = {
        "python": """
# {title}

## Overview
{overview}

## Implementation

```python
{code_example}
```

## Usage

{usage_explanation}

## Best Practices

{best_practices}

## Common Pitfalls

{common_pitfalls}
        """,
        "machine_learning": """
# {title}

## Introduction
{introduction}

## Algorithm Overview
{algorithm_overview}

## Implementation
```python
{ml_code_example}
```

## Model Training
{training_explanation}

## Evaluation Metrics
{evaluation_metrics}

## Hyperparameter Tuning
{hyperparameter_tuning}
        """,
        # Additional templates for other topics...
    }
    
    template = content_templates.get(topic, content_templates["python"])
    
    # Generate content for each placeholder
    content_values = await generate_content_values(topic)
    
    return template.format(**content_values)
```

### Mock Services Framework

**Service Mocking Infrastructure**:
```python
# testing/mock_services.py
class MockServiceFramework:
    def __init__(self):
        self.mock_servers = {}
        self.response_patterns = {}
        
    async def start_mock_service(self, service_name: str, config: MockServiceConfig) -> MockService:
        """Start a mock service with specified behavior."""
        
        if service_name == "voyage_ai":
            return await self._start_voyage_ai_mock(config)
        elif service_name == "qdrant":
            return await self._start_qdrant_mock(config)
        elif service_name == "redis":
            return await self._start_redis_mock(config)
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    async def _start_voyage_ai_mock(self, config: MockServiceConfig) -> MockService:
        """Start mock Voyage AI service."""
        
        class VoyageAIMock:
            def __init__(self):
                self.request_count = 0
                self.rate_limit_enabled = config.enable_rate_limiting
                self.failure_rate = config.failure_rate
                
            async def handle_embedding_request(self, request: dict) -> dict:
                """Handle embedding API request."""
                self.request_count += 1
                
                # Simulate rate limiting
                if self.rate_limit_enabled and self.request_count > 100:
                    if self.request_count % 101 == 0:  # Every 101st request
                        return {"error": "Rate limit exceeded", "retry_after": 1}, 429
                
                # Simulate failures
                if random.random() < self.failure_rate:
                    return {"error": "Service temporarily unavailable"}, 503
                
                # Generate mock embeddings
                input_texts = request.get("input", [])
                embeddings = []
                
                for text in input_texts:
                    # Generate deterministic embeddings based on text hash
                    text_hash = hashlib.md5(text.encode()).hexdigest()
                    embedding = [
                        (int(text_hash[i:i+2], 16) - 128) / 128.0
                        for i in range(0, min(len(text_hash), 32), 2)
                    ]
                    # Pad or truncate to 2048 dimensions
                    embedding = (embedding * (2048 // len(embedding) + 1))[:2048]
                    embeddings.append(embedding)
                
                return {
                    "data": [{"embedding": emb} for emb in embeddings],
                    "usage": {
                        "total_tokens": sum(len(text.split()) for text in input_texts)
                    }
                }, 200
        
        # Start HTTP server for mock service
        mock_service = VoyageAIMock()
        
        app = FastAPI()
        
        @app.post("/v1/embeddings")
        async def embeddings_endpoint(request: dict):
            response_data, status_code = await mock_service.handle_embedding_request(request)
            return JSONResponse(content=response_data, status_code=status_code)
        
        # Start server in background
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=config.port, log_level="error")
        )
        
        server_task = asyncio.create_task(server.serve())
        
        # Wait for server to start
        await asyncio.sleep(1)
        
        return MockService(
            name="voyage_ai",
            url=f"http://127.0.0.1:{config.port}",
            mock_instance=mock_service,
            server_task=server_task,
            cleanup_callback=lambda: server_task.cancel()
        )

class MockServiceBehavior:
    """Define behavior patterns for mock services."""
    
    @staticmethod
    def create_latency_pattern(base_latency: float, variance: float) -> Callable:
        """Create latency simulation function."""
        async def simulate_latency():
            latency = base_latency + random.uniform(-variance, variance)
            await asyncio.sleep(max(0, latency))
        return simulate_latency
    
    @staticmethod
    def create_failure_pattern(failure_rate: float, failure_types: List[str]) -> Callable:
        """Create failure simulation function."""
        def simulate_failure() -> Optional[Exception]:
            if random.random() < failure_rate:
                failure_type = random.choice(failure_types)
                if failure_type == "timeout":
                    return TimeoutError("Request timeout")
                elif failure_type == "connection":
                    return ConnectionError("Connection failed")
                elif failure_type == "rate_limit":
                    return HTTPError("Rate limit exceeded")
                elif failure_type == "server_error":
                    return HTTPError("Internal server error")
            return None
        return simulate_failure
```

## Test Automation Framework

### Test Orchestration

**Test Orchestrator Implementation**:
```python
# testing/test_orchestrator.py
class TestOrchestrator:
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_suites = {}
        self.environment_manager = TestEnvironmentManager(config)
        self.data_manager = TestDataManager(config.data)
        self.result_collector = TestResultCollector(config.reporting)
        
    async def register_test_suite(self, suite_name: str, suite: TestSuite):
        """Register a test suite with the orchestrator."""
        self.test_suites[suite_name] = suite
    
    async def run_test_pipeline(self, pipeline_config: TestPipelineConfig) -> TestPipelineResult:
        """Run complete test pipeline."""
        pipeline_start_time = time.time()
        pipeline_results = {}
        
        logger.info(f"Starting test pipeline: {pipeline_config.name}")
        
        try:
            # Phase 1: Environment Setup
            environments = {}
            for env_type in pipeline_config.required_environments:
                logger.info(f"Provisioning {env_type} environment")
                environments[env_type] = await self.environment_manager.provision_environment(env_type)
            
            # Phase 2: Test Data Preparation
            test_data = {}
            for data_type, data_config in pipeline_config.test_data.items():
                logger.info(f"Generating {data_type} test data")
                test_data[data_type] = await self.data_manager.generate_test_data(
                    data_type, data_config.size, **data_config.parameters
                )
            
            # Phase 3: Execute Test Suites
            for phase in pipeline_config.phases:
                phase_start_time = time.time()
                phase_results = {}
                
                logger.info(f"Executing phase: {phase.name}")
                
                # Run test suites in parallel if configured
                if phase.parallel_execution:
                    tasks = []
                    for suite_name in phase.test_suites:
                        suite = self.test_suites[suite_name]
                        task = asyncio.create_task(
                            self._run_test_suite(
                                suite, environments, test_data, phase.timeout
                            )
                        )
                        tasks.append((suite_name, task))
                    
                    # Wait for all suites to complete
                    for suite_name, task in tasks:
                        try:
                            phase_results[suite_name] = await task
                        except Exception as e:
                            phase_results[suite_name] = TestSuiteResult(
                                suite_name=suite_name,
                                passed=False,
                                error=str(e),
                                execution_time=time.time() - phase_start_time
                            )
                else:
                    # Run test suites sequentially
                    for suite_name in phase.test_suites:
                        suite = self.test_suites[suite_name]
                        phase_results[suite_name] = await self._run_test_suite(
                            suite, environments, test_data, phase.timeout
                        )
                        
                        # Stop on failure if configured
                        if phase.stop_on_failure and not phase_results[suite_name].passed:
                            logger.error(f"Stopping phase due to failure in {suite_name}")
                            break
                
                phase_execution_time = time.time() - phase_start_time
                pipeline_results[phase.name] = TestPhaseResult(
                    phase_name=phase.name,
                    suite_results=phase_results,
                    execution_time=phase_execution_time,
                    passed=all(result.passed for result in phase_results.values())
                )
                
                # Stop pipeline on phase failure if configured
                if pipeline_config.stop_on_phase_failure and not pipeline_results[phase.name].passed:
                    logger.error(f"Stopping pipeline due to phase failure: {phase.name}")
                    break
            
            # Phase 4: Results Collection and Analysis
            overall_passed = all(phase.passed for phase in pipeline_results.values())
            pipeline_execution_time = time.time() - pipeline_start_time
            
            pipeline_result = TestPipelineResult(
                pipeline_name=pipeline_config.name,
                phase_results=pipeline_results,
                overall_passed=overall_passed,
                execution_time=pipeline_execution_time,
                timestamp=datetime.utcnow()
            )
            
            # Collect and store results
            await self.result_collector.store_pipeline_result(pipeline_result)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
        finally:
            # Cleanup environments
            for env in environments.values():
                try:
                    await env.cleanup()
                except Exception as e:
                    logger.warning(f"Environment cleanup failed: {e}")
    
    async def _run_test_suite(
        self, 
        suite: TestSuite, 
        environments: Dict[str, TestEnvironment],
        test_data: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> TestSuiteResult:
        """Run individual test suite."""
        suite_start_time = time.time()
        test_results = []
        
        try:
            # Initialize test suite with environment and data
            await suite.setup(environments, test_data)
            
            # Execute all tests in the suite
            for test_case in suite.test_cases:
                test_start_time = time.time()
                
                try:
                    # Run test with timeout
                    if timeout:
                        test_result = await asyncio.wait_for(
                            test_case.execute(),
                            timeout=timeout
                        )
                    else:
                        test_result = await test_case.execute()
                    
                    test_execution_time = time.time() - test_start_time
                    test_result.execution_time = test_execution_time
                    test_results.append(test_result)
                    
                except asyncio.TimeoutError:
                    test_results.append(TestResult(
                        test_name=test_case.name,
                        passed=False,
                        error="Test timed out",
                        execution_time=timeout
                    ))
                except Exception as e:
                    test_results.append(TestResult(
                        test_name=test_case.name,
                        passed=False,
                        error=str(e),
                        execution_time=time.time() - test_start_time
                    ))
            
            # Calculate suite results
            passed_tests = sum(1 for result in test_results if result.passed)
            suite_passed = passed_tests == len(test_results)
            suite_execution_time = time.time() - suite_start_time
            
            return TestSuiteResult(
                suite_name=suite.name,
                test_results=test_results,
                passed=suite_passed,
                total_tests=len(test_results),
                passed_tests=passed_tests,
                execution_time=suite_execution_time
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name=suite.name,
                passed=False,
                error=str(e),
                execution_time=time.time() - suite_start_time
            )
        finally:
            try:
                await suite.cleanup()
            except Exception as e:
                logger.warning(f"Test suite cleanup failed: {e}")
```

### CI/CD Integration

**GitHub Actions Test Automation**:
```yaml
# .github/workflows/automated_testing.yml
name: Automated Testing Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

env:
  PYTHON_VERSION: "3.11"
  POETRY_VERSION: "1.6.1"

jobs:
  test-matrix:
    name: Generate Test Matrix
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.matrix.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
      - name: Generate test matrix
        id: matrix
        run: |
          # Generate dynamic test matrix based on changed files
          python scripts/generate_test_matrix.py > test_matrix.json
          echo "matrix=$(cat test_matrix.json)" >> $GITHUB_OUTPUT

  level-1-syntax:
    name: "Level 1: Syntax & Quality"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Run syntax validation
        run: make validate-syntax
      
      - name: Upload syntax reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: syntax-reports
          path: |
            ruff-report.json
            mypy-report/
            security-report.json

  level-2-unit:
    name: "Level 2: Unit Tests"
    needs: level-1-syntax
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.test-matrix.outputs.matrix) }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Run unit tests
        run: |
          pytest tests/unit/${{ matrix.component }} \
            --cov=src/contexter/rag/${{ matrix.component }} \
            --cov-report=xml \
            --cov-report=term-missing \
            --junit-xml=unit-test-results-${{ matrix.component }}.xml \
            --cov-fail-under=95
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: coverage.xml
          flags: unit-tests,${{ matrix.component }}
          name: codecov-${{ matrix.component }}
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: unit-test-results
          path: unit-test-results-*.xml

  level-3-integration:
    name: "Level 3: Integration Tests"
    needs: level-2-unit
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:v1.7.0
        ports:
          - 6333:6333
        options: >-
          --health-cmd "curl -f http://localhost:6333/health"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 3s
          --health-retries 5
      
      postgres:
        image: postgres:15-alpine
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_password
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Wait for services
        run: |
          for i in {1..30}; do
            if curl -f http://localhost:6333/health && redis-cli -h localhost ping; then
              echo "Services are ready"
              break
            fi
            echo "Waiting for services..."
            sleep 5
          done
      
      - name: Run integration tests
        env:
          QDRANT_URL: http://localhost:6333
          REDIS_URL: redis://localhost:6379
          POSTGRES_URL: postgresql://test_user:test_password@localhost:5432/test_db
        run: |
          pytest tests/integration/ \
            -v \
            --tb=short \
            --junit-xml=integration-test-results.xml \
            --timeout=300
      
      - name: Upload integration test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: integration-test-results
          path: integration-test-results.xml

  level-4-domain:
    name: "Level 4: Domain Validation"
    needs: level-3-integration
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.event_name == 'pull_request'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Setup test data
        run: |
          python scripts/setup_domain_test_data.py
      
      - name: Run RAG accuracy tests
        env:
          VOYAGE_API_KEY: ${{ secrets.VOYAGE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          pytest tests/domain/test_rag_accuracy.py \
            -v \
            --tb=short \
            --junit-xml=domain-accuracy-results.xml
      
      - name: Run performance validation
        run: |
          pytest tests/domain/test_performance_validation.py \
            -v \
            --tb=short \
            --junit-xml=domain-performance-results.xml
      
      - name: Generate domain validation report
        run: |
          python scripts/generate_domain_validation_report.py \
            --accuracy-results domain-accuracy-results.xml \
            --performance-results domain-performance-results.xml \
            --output domain-validation-report.html
      
      - name: Upload domain validation results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: domain-validation-results
          path: |
            domain-*-results.xml
            domain-validation-report.html

  performance-baseline:
    name: "Performance Baseline"
    needs: level-4-domain
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
          pip install locust
      
      - name: Run performance baseline tests
        run: |
          python scripts/run_performance_baseline.py \
            --duration 300 \
            --users 50 \
            --spawn-rate 5 \
            --output performance-baseline-report.json
      
      - name: Store performance baseline
        run: |
          python scripts/store_performance_baseline.py \
            --report performance-baseline-report.json \
            --commit ${{ github.sha }}
      
      - name: Upload performance baseline
        uses: actions/upload-artifact@v3
        with:
          name: performance-baseline
          path: performance-baseline-report.json

  test-report:
    name: "Generate Test Report"
    needs: [level-1-syntax, level-2-unit, level-3-integration, level-4-domain]
    runs-on: ubuntu-latest
    if: always()
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download all artifacts
        uses: actions/download-artifact@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements-dev.txt
      
      - name: Generate comprehensive test report
        run: |
          python scripts/generate_comprehensive_test_report.py \
            --artifacts-dir . \
            --output comprehensive-test-report.html \
            --format html,json
      
      - name: Upload comprehensive test report
        uses: actions/upload-artifact@v3
        with:
          name: comprehensive-test-report
          path: |
            comprehensive-test-report.html
            comprehensive-test-report.json
      
      - name: Comment PR with test results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(fs.readFileSync('comprehensive-test-report.json', 'utf8'));
            
            const comment = `
            ## 🧪 Test Results Summary
            
            | Level | Status | Tests | Coverage |
            |-------|--------|-------|----------|
            | Level 1 (Syntax) | ${report.level1.passed ? '✅' : '❌'} | ${report.level1.tests} | N/A |
            | Level 2 (Unit) | ${report.level2.passed ? '✅' : '❌'} | ${report.level2.tests} | ${report.level2.coverage}% |
            | Level 3 (Integration) | ${report.level3.passed ? '✅' : '❌'} | ${report.level3.tests} | N/A |
            | Level 4 (Domain) | ${report.level4.passed ? '✅' : '❌'} | ${report.level4.tests} | N/A |
            
            **Overall Status**: ${report.overall.passed ? '✅ PASSED' : '❌ FAILED'}
            
            [View detailed report](${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID})
            `;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

This comprehensive automated testing framework provides robust test infrastructure, automation tools, and CI/CD integration to ensure continuous quality validation throughout the development lifecycle.