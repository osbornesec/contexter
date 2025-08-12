# Test Scenarios and Test Cases

## Document Information
- **Version**: 1.0.0
- **Created**: 2025-01-15
- **System**: Contexter RAG Implementation
- **Scope**: Comprehensive test scenarios for all validation levels

## Overview

This document provides detailed test scenarios and test cases for validating the RAG system across all four validation levels. Each scenario includes setup instructions, test data, expected outcomes, and execution commands.

## Level 1: Syntax & Code Quality Test Scenarios

### Scenario S1-001: Python Code Quality Validation

**Purpose**: Validate Python code meets quality standards
**Execution Time**: <10 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S1-001-001: Linting Validation**
```bash
# Command
ruff check src/ tests/ --output-format=json

# Expected Result
{
  "success": true,
  "violations": [],
  "total_files_checked": 47,
  "total_violations": 0
}

# Validation Criteria
- Zero linting violations
- All files processed without errors
- No E501 (line too long) violations
- No F401 (unused import) violations
```

**TC-S1-001-002: Type Checking Validation**
```bash
# Command
mypy src/ --strict --json-report mypy-report

# Expected Result
{
  "success": true,
  "errors": [],
  "files_checked": 47,
  "total_errors": 0
}

# Validation Criteria
- Zero type checking errors
- All function signatures properly typed
- All return types specified
- No 'Any' types without justification
```

**TC-S1-001-003: Code Formatting Validation**
```bash
# Command
black src/ tests/ --check --diff

# Expected Result
"All done! ✨ 🍰 ✨
47 files would be left unchanged."

# Validation Criteria
- No formatting changes required
- All files comply with Black formatting
- Line length adherence to 88 characters
```

**Test Data**:
```python
# tests/data/code_quality_samples.py

# Valid code sample
async def process_document(document: Document) -> ProcessingResult:
    """Process a document through the RAG pipeline."""
    try:
        chunks = await chunking_service.chunk_document(document)
        embeddings = await embedding_service.generate_embeddings(chunks)
        result = await vector_store.upsert_vectors(embeddings)
        return ProcessingResult(success=True, chunks_processed=len(chunks))
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return ProcessingResult(success=False, error=str(e))

# Invalid code samples for negative testing
def bad_function():  # Missing type hints
    x=1+2  # Poor formatting
    return x  # Missing return type
```

### Scenario S1-002: Configuration Validation

**Purpose**: Validate all configuration files are syntactically correct
**Execution Time**: <5 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S1-002-001: YAML Configuration Validation**
```bash
# Command
python -c "
import yaml
import sys
try:
    with open('config/rag_config.yaml') as f:
        yaml.safe_load(f)
    print('YAML validation: PASSED')
except Exception as e:
    print(f'YAML validation: FAILED - {e}')
    sys.exit(1)
"

# Expected Result
"YAML validation: PASSED"

# Validation Criteria
- YAML syntax is valid
- All required configuration keys present
- Value types match expected types
- No duplicate keys
```

**Test Data**:
```yaml
# tests/data/valid_rag_config.yaml
rag:
  vector_db:
    host: "localhost"
    port: 6333
    collection_name: "test_collection"
    hnsw:
      m: 16
      ef_construct: 200
  embedding:
    provider: "voyage"
    model: "voyage-code-3"
    batch_size: 100

# tests/data/invalid_rag_config.yaml (for negative testing)
rag:
  vector_db:
    host: "localhost"
    port: "invalid_port"  # Should be integer
    - invalid_yaml_syntax
```

### Scenario S1-003: Security Scanning

**Purpose**: Identify security vulnerabilities in code
**Execution Time**: <15 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S1-003-001: Security Vulnerability Scanning**
```bash
# Command
bandit -r src/ -f json -o security-report.json

# Expected Result
{
  "metrics": {
    "total_lines_of_code": 2847,
    "total_lines_skipped": 0
  },
  "results": [],
  "errors": []
}

# Validation Criteria
- Zero high or medium severity issues
- No hardcoded secrets detected
- No insecure random number generation
- No SQL injection vulnerabilities
```

**Test Data**:
```python
# tests/data/security_test_samples.py

# Secure code sample
import secrets
import hashlib
from cryptography.fernet import Fernet

def generate_secure_token() -> str:
    """Generate cryptographically secure token."""
    return secrets.token_urlsafe(32)

def hash_password(password: str, salt: str) -> str:
    """Securely hash password with salt."""
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)

# Insecure code samples for negative testing
def bad_token_generation():
    import random
    return str(random.randint(1000, 9999))  # Insecure random

API_KEY = "hardcoded-api-key-123"  # Hardcoded secret
```

## Level 2: Unit Testing Scenarios

### Scenario S2-001: Vector Database Component Testing

**Purpose**: Validate vector database operations in isolation
**Execution Time**: <30 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S2-001-001: Vector Insertion with Mocked Backend**
```python
# tests/unit/rag/vector_db/test_qdrant_store.py

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = AsyncMock()
    client.upsert.return_value = UpdateResult(
        operation_id=0,
        status=UpdateStatus.COMPLETED
    )
    return client

@pytest.fixture
def qdrant_store(mock_qdrant_client):
    """QdrantStore with mocked client."""
    store = QdrantStore(config=test_config)
    store.client = mock_qdrant_client
    return store

async def test_vector_insertion_success(qdrant_store, mock_qdrant_client):
    """Test successful vector insertion."""
    # Test Data
    vectors = [
        {
            "id": "test-doc-1",
            "vector": [0.1, 0.2, 0.3] * 682 + [0.1, 0.2],  # 2048 dimensions
            "payload": {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "content": "test content",
                "metadata": {"language": "python"}
            }
        }
    ]
    
    # Execute
    result = await qdrant_store.upsert_vectors(vectors)
    
    # Validate
    assert result.success is True
    assert result.inserted_count == 1
    assert result.failed_count == 0
    
    # Verify API call
    mock_qdrant_client.upsert.assert_called_once()
    call_args = mock_qdrant_client.upsert.call_args
    assert call_args[1]["collection_name"] == test_config.collection_name
    assert len(call_args[1]["points"]) == 1

# Command to run
pytest tests/unit/rag/vector_db/test_qdrant_store.py::test_vector_insertion_success -v

# Expected Result
tests/unit/rag/vector_db/test_qdrant_store.py::test_vector_insertion_success PASSED [100%]
```

**TC-S2-001-002: Search Operation with Filters**
```python
async def test_search_with_metadata_filters(qdrant_store, mock_qdrant_client):
    """Test vector search with metadata filtering."""
    # Mock search results
    mock_results = [
        ScoredPoint(
            id="test-1",
            version=1,
            score=0.95,
            payload={
                "document_id": "doc-1",
                "content": "Python machine learning",
                "metadata": {"language": "python", "topic": "ml"}
            },
            vector=None
        ),
        ScoredPoint(
            id="test-2", 
            version=1,
            score=0.87,
            payload={
                "document_id": "doc-2",
                "content": "JavaScript frameworks",
                "metadata": {"language": "javascript", "topic": "web"}
            },
            vector=None
        )
    ]
    mock_qdrant_client.search.return_value = mock_results
    
    # Test Data
    query_vector = [0.1] * 2048
    filters = {
        "language": "python",
        "topic": "ml"
    }
    
    # Execute
    results = await qdrant_store.search(
        query_vector=query_vector,
        filters=filters,
        limit=10,
        score_threshold=0.7
    )
    
    # Validate
    assert len(results) == 2
    assert results[0].score == 0.95
    assert results[0].payload["language"] == "python"
    assert all(r.score >= 0.7 for r in results)
    
    # Verify API call with correct filters
    mock_qdrant_client.search.assert_called_once()
    call_args = mock_qdrant_client.search.call_args[1]
    assert call_args["limit"] == 10
    assert call_args["score_threshold"] == 0.7
```

**TC-S2-001-003: Error Handling for Connection Failures**
```python
async def test_connection_error_handling(qdrant_store, mock_qdrant_client):
    """Test handling of connection errors."""
    # Mock connection failure
    mock_qdrant_client.upsert.side_effect = ConnectionError("Connection failed")
    
    vectors = [{"id": "test", "vector": [0.1] * 2048, "payload": {}}]
    
    # Execute and expect graceful error handling
    result = await qdrant_store.upsert_vectors(vectors)
    
    # Validate error handling
    assert result.success is False
    assert result.inserted_count == 0
    assert result.failed_count == 1
    assert "Connection failed" in result.error_message
    assert result.retry_recommended is True
```

### Scenario S2-002: Embedding Service Component Testing

**Purpose**: Validate embedding generation with various inputs
**Execution Time**: <20 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S2-002-001: Batch Embedding Generation**
```python
# tests/unit/rag/embedding/test_voyage_client.py

@pytest.fixture
def mock_httpx_client():
    """Mock HTTP client for API calls."""
    client = AsyncMock()
    return client

@pytest.fixture
def voyage_client(mock_httpx_client):
    """VoyageClient with mocked HTTP client."""
    client = VoyageClient(
        api_key="test-key",
        model="voyage-code-3",
        base_url="https://api.voyageai.com"
    )
    client.http_client = mock_httpx_client
    return client

@pytest.mark.parametrize("batch_size,expected_api_calls", [
    (1, 3),  # 3 texts, batch_size=1 -> 3 API calls
    (2, 2),  # 3 texts, batch_size=2 -> 2 API calls  
    (5, 1),  # 3 texts, batch_size=5 -> 1 API call
])
async def test_batch_embedding_generation(
    voyage_client, mock_httpx_client, batch_size, expected_api_calls
):
    """Test batch embedding with different batch sizes."""
    # Test Data
    texts = [
        "def process_data(data): return data.clean()",
        "class DataProcessor: def __init__(self): pass",
        "import pandas as pd; df = pd.DataFrame()"
    ]
    
    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2] * 1024} for _ in range(min(batch_size, len(texts)))],
        "usage": {"total_tokens": 50}
    }
    mock_response.status_code = 200
    mock_httpx_client.post.return_value = mock_response
    
    # Execute
    embeddings = await voyage_client.embed_batch(texts, batch_size=batch_size)
    
    # Validate
    assert len(embeddings) == len(texts)
    assert all(len(emb) == 2048 for emb in embeddings)
    assert mock_httpx_client.post.call_count == expected_api_calls
    
    # Validate API call structure
    for call in mock_httpx_client.post.call_args_list:
        request_data = call[1]["json"]
        assert "input" in request_data
        assert "model" in request_data
        assert request_data["model"] == "voyage-code-3"
        assert len(request_data["input"]) <= batch_size

# Command to run
pytest tests/unit/rag/embedding/test_voyage_client.py::test_batch_embedding_generation -v
```

**TC-S2-002-002: Caching Behavior Validation**
```python
async def test_embedding_cache_hit_miss(voyage_client):
    """Test embedding cache hit and miss behavior."""
    # Test Data
    text = "def example_function(): pass"
    expected_embedding = [0.1, 0.2] * 1024
    
    # First call - cache miss
    with patch.object(voyage_client, '_make_api_request') as mock_api:
        mock_api.return_value = {"data": [{"embedding": expected_embedding}]}
        
        embedding1 = await voyage_client.embed_single(text)
        assert mock_api.call_count == 1
    
    # Second call - cache hit (no API call)
    with patch.object(voyage_client, '_make_api_request') as mock_api:
        embedding2 = await voyage_client.embed_single(text)
        assert mock_api.call_count == 0  # No API call made
        assert embedding1 == embedding2
    
    # Verify cache statistics
    cache_stats = voyage_client.get_cache_stats()
    assert cache_stats["hits"] == 1
    assert cache_stats["misses"] == 1
    assert cache_stats["hit_rate"] == 0.5
```

**TC-S2-002-003: Rate Limiting and Retry Logic**
```python
async def test_rate_limiting_retry_logic(voyage_client, mock_httpx_client):
    """Test rate limiting handling and retry logic."""
    # Mock rate limit response followed by success
    rate_limit_response = Mock()
    rate_limit_response.status_code = 429
    rate_limit_response.json.return_value = {"error": "Rate limit exceeded"}
    rate_limit_response.headers = {"Retry-After": "1"}
    
    success_response = Mock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "data": [{"embedding": [0.1] * 2048}],
        "usage": {"total_tokens": 10}
    }
    
    mock_httpx_client.post.side_effect = [rate_limit_response, success_response]
    
    # Execute with timing
    start_time = time.time()
    embedding = await voyage_client.embed_single("test text")
    elapsed_time = time.time() - start_time
    
    # Validate
    assert len(embedding) == 2048
    assert mock_httpx_client.post.call_count == 2  # Retry happened
    assert elapsed_time >= 1.0  # Waited for retry delay
    
    # Verify rate limit metrics
    rate_limit_stats = voyage_client.get_rate_limit_stats()
    assert rate_limit_stats["rate_limits_hit"] == 1
    assert rate_limit_stats["total_retries"] == 1
```

### Scenario S2-003: Document Processing Component Testing

**Purpose**: Validate document chunking and processing logic
**Execution Time**: <25 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S2-003-001: Intelligent Chunking Algorithm**
```python
# tests/unit/rag/ingestion/test_chunking_engine.py

@pytest.fixture
def chunking_engine():
    """ChunkingEngine with test configuration."""
    config = ChunkingConfig(
        chunk_size=1000,
        overlap_size=200,
        respect_boundaries=True,
        min_chunk_size=100
    )
    return ChunkingEngine(config)

async def test_code_aware_chunking(chunking_engine):
    """Test chunking that respects code structure."""
    # Test Data - Python code with classes and functions
    code_content = '''
class DataProcessor:
    """Processes data for machine learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = None
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data."""
        # Remove null values
        data = data.dropna()
        
        # Normalize numerical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        
        return data
        
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the machine learning model."""
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X, y)
        
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from file."""
    return pd.read_csv(file_path)

def main():
    """Main execution function."""
    processor = DataProcessor({"normalize": True})
    data = load_data("data.csv")
    processed_data = processor.preprocess_data(data)
    print(f"Processed {len(processed_data)} rows")
'''
    
    # Execute chunking
    chunks = await chunking_engine.chunk_text(
        content=code_content,
        content_type="python",
        metadata={"file_type": "py", "language": "python"}
    )
    
    # Validate chunking results
    assert len(chunks) >= 3  # Should create multiple chunks
    assert all(len(chunk.content) <= 1000 for chunk in chunks)  # Respect size limit
    assert all(len(chunk.content) >= 100 for chunk in chunks)   # Respect min size
    
    # Validate code structure preservation
    class_chunks = [c for c in chunks if "class DataProcessor" in c.content]
    assert len(class_chunks) >= 1  # Class should be in at least one chunk
    
    function_chunks = [c for c in chunks if "def " in c.content]
    assert len(function_chunks) >= 2  # Multiple functions should be preserved
    
    # Validate overlap behavior
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i + 1]
        
        # Check for semantic overlap (shared tokens)
        current_tokens = set(current_chunk.content.split())
        next_tokens = set(next_chunk.content.split())
        overlap_tokens = current_tokens & next_tokens
        
        # Should have some overlap but not too much
        overlap_ratio = len(overlap_tokens) / min(len(current_tokens), len(next_tokens))
        assert 0.1 <= overlap_ratio <= 0.5  # 10-50% overlap

# Command to run
pytest tests/unit/rag/ingestion/test_chunking_engine.py::test_code_aware_chunking -v
```

**TC-S2-003-002: Metadata Extraction and Enrichment**
```python
async def test_metadata_extraction_enrichment(chunking_engine):
    """Test metadata extraction from document content."""
    # Test Data - Markdown documentation
    markdown_content = '''
# API Documentation

## Overview
This document describes the REST API for the document processing service.

### Authentication
All API requests require authentication using API keys.

```python
import requests

headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

response = requests.get("https://api.example.com/documents", headers=headers)
```

### Endpoints

#### GET /documents
Retrieve a list of all documents.

**Parameters:**
- `limit` (integer): Maximum number of results (default: 20)
- `offset` (integer): Pagination offset (default: 0)

**Response:**
```json
{
  "documents": [
    {
      "id": "doc-123",
      "title": "Sample Document",
      "created_at": "2025-01-15T10:00:00Z"
    }
  ]
}
```
'''
    
    # Execute chunking with metadata extraction
    chunks = await chunking_engine.chunk_text(
        content=markdown_content,
        content_type="markdown",
        metadata={
            "file_name": "api_documentation.md",
            "section": "API Reference",
            "version": "1.0"
        }
    )
    
    # Validate metadata enrichment
    assert len(chunks) >= 2  # Should create multiple logical chunks
    
    # Check for heading-based metadata
    title_chunks = [c for c in chunks if c.metadata.get("heading_level") == 1]
    assert len(title_chunks) >= 1  # Should identify main title
    assert any("API Documentation" in c.content for c in title_chunks)
    
    # Check for code block detection
    code_chunks = [c for c in chunks if c.metadata.get("contains_code") is True]
    assert len(code_chunks) >= 1  # Should detect code blocks
    assert any("import requests" in c.content for c in code_chunks)
    
    # Check for API endpoint detection
    endpoint_chunks = [c for c in chunks if c.metadata.get("contains_api_endpoint") is True]
    assert len(endpoint_chunks) >= 1  # Should detect API endpoints
    assert any("GET /documents" in c.content for c in endpoint_chunks)
    
    # Validate preserved file metadata
    for chunk in chunks:
        assert chunk.metadata["file_name"] == "api_documentation.md"
        assert chunk.metadata["section"] == "API Reference"
        assert chunk.metadata["version"] == "1.0"
        assert chunk.metadata["content_type"] == "markdown"
```

## Level 3: Integration Testing Scenarios

### Scenario S3-001: End-to-End Document Processing Pipeline

**Purpose**: Validate complete document ingestion to search pipeline
**Execution Time**: <2 minutes
**Automation Level**: 100%

**Test Cases**:

**TC-S3-001-001: Complete Pipeline with Real Components**
```python
# tests/integration/test_document_pipeline.py

@pytest.fixture
async def rag_system():
    """Fully configured RAG system for integration testing."""
    config = load_integration_test_config()
    
    # Initialize all components
    vector_store = QdrantStore(config.vector_db)
    embedding_service = VoyageClient(config.embedding)
    storage_manager = RAGStorageManager(config.storage)
    ingestion_pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        storage_manager=storage_manager,
        config=config.ingestion
    )
    
    system = RAGSystem(
        vector_store=vector_store,
        embedding_service=embedding_service,
        storage_manager=storage_manager,
        ingestion_pipeline=ingestion_pipeline,
        config=config
    )
    
    await system.initialize()
    yield system
    await system.cleanup()

async def test_document_ingestion_to_search_pipeline(rag_system):
    """Test complete document processing and retrieval."""
    # Test Data - Realistic technical document
    test_document = {
        "document_id": "test-python-guide",
        "library_id": "python-docs",
        "content": '''
# Python Machine Learning Guide

## Data Preprocessing

Data preprocessing is a crucial step in machine learning workflows. Here's how to handle missing values:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocess dataset for machine learning."""
    # Handle missing values
    df = df.dropna(subset=['target'])
    df = df.fillna(df.mean())
    
    # Scale features
    scaler = StandardScaler()
    features = df.select_dtypes(include=[np.number])
    df[features.columns] = scaler.fit_transform(features)
    
    return df
```

## Model Training

Use scikit-learn for model training:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(X, y):
    """Train random forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    return model, accuracy
```
        ''',
        "metadata": {
            "language": "python",
            "topic": "machine_learning",
            "difficulty": "intermediate",
            "file_type": "markdown",
            "version": "1.0"
        }
    }
    
    # Step 1: Ingest document
    start_time = time.time()
    ingestion_result = await rag_system.ingest_document(test_document)
    ingestion_time = time.time() - start_time
    
    # Validate ingestion
    assert ingestion_result.success is True, f"Ingestion failed: {ingestion_result.error}"
    assert ingestion_result.chunks_processed > 0, "No chunks were processed"
    assert ingestion_result.vectors_stored > 0, "No vectors were stored"
    assert ingestion_time < 30.0, f"Ingestion took too long: {ingestion_time:.2f}s"
    
    # Wait for asynchronous processing to complete
    await asyncio.sleep(2)
    
    # Step 2: Test semantic search
    semantic_query = "how to preprocess data for machine learning"
    semantic_results = await rag_system.search(
        query=semantic_query,
        search_type="semantic",
        limit=5
    )
    
    # Validate semantic search results
    assert len(semantic_results) > 0, "Semantic search returned no results"
    assert semantic_results[0].score > 0.7, f"Low relevance score: {semantic_results[0].score}"
    assert any("preprocess" in result.content.lower() for result in semantic_results), \
        "No results contain preprocessing content"
    
    # Step 3: Test keyword search
    keyword_query = "RandomForestClassifier scikit-learn"
    keyword_results = await rag_system.search(
        query=keyword_query,
        search_type="keyword", 
        limit=5
    )
    
    # Validate keyword search results
    assert len(keyword_results) > 0, "Keyword search returned no results"
    assert any("RandomForestClassifier" in result.content for result in keyword_results), \
        "No results contain exact keyword match"
    
    # Step 4: Test hybrid search
    hybrid_query = "train machine learning model with random forest"
    hybrid_results = await rag_system.search(
        query=hybrid_query,
        search_type="hybrid",
        limit=10
    )
    
    # Validate hybrid search results
    assert len(hybrid_results) > 0, "Hybrid search returned no results"
    assert len(hybrid_results) >= len(semantic_results), "Hybrid should return more comprehensive results"
    
    # Validate result quality
    top_result = hybrid_results[0]
    assert top_result.score > 0.6, f"Top result score too low: {top_result.score}"
    assert top_result.document_id == "test-python-guide", "Wrong document returned"
    
    # Step 5: Test metadata filtering
    filtered_results = await rag_system.search(
        query="machine learning",
        search_type="hybrid",
        filters={"language": "python", "topic": "machine_learning"},
        limit=5
    )
    
    # Validate filtered results
    assert len(filtered_results) > 0, "Filtered search returned no results"
    for result in filtered_results:
        assert result.metadata.get("language") == "python", "Filter not applied correctly"
        assert result.metadata.get("topic") == "machine_learning", "Topic filter not applied"

# Command to run
pytest tests/integration/test_document_pipeline.py::test_document_ingestion_to_search_pipeline -v
```

**TC-S3-001-002: Error Recovery and Resilience**
```python
async def test_pipeline_error_recovery(rag_system):
    """Test pipeline error handling and recovery mechanisms."""
    # Test Data - Document with potential processing issues
    problematic_document = {
        "document_id": "problematic-doc",
        "library_id": "test-lib",
        "content": "a" * 50000,  # Very large content
        "metadata": {"test": True}
    }
    
    # Test embedding service failure simulation
    with patch.object(rag_system.embedding_service, 'embed_batch') as mock_embed:
        # First call fails, second succeeds
        mock_embed.side_effect = [
            EmbeddingServiceError("Service temporarily unavailable"),
            [[0.1] * 2048, [0.2] * 2048]  # Successful embeddings
        ]
        
        # Execute with retry mechanism
        result = await rag_system.ingest_document(problematic_document)
        
        # Validate recovery
        assert result.success is True, "Failed to recover from embedding service error"
        assert mock_embed.call_count == 2, "Retry mechanism not triggered"
        assert result.retry_count == 1, "Retry count not tracked correctly"
    
    # Test vector store failure simulation
    with patch.object(rag_system.vector_store, 'upsert_vectors') as mock_upsert:
        mock_upsert.side_effect = VectorStoreError("Connection timeout")
        
        failed_document = {"document_id": "fail-doc", "content": "test content"}
        result = await rag_system.ingest_document(failed_document)
        
        # Validate graceful failure handling
        assert result.success is False, "Should have failed gracefully"
        assert "Connection timeout" in result.error_message
        assert result.retry_recommended is True
```

### Scenario S3-002: API Integration Testing

**Purpose**: Validate API endpoints with backend integration
**Execution Time**: <90 seconds
**Automation Level**: 100%

**Test Cases**:

**TC-S3-002-001: Search API with Authentication**
```python
# tests/integration/test_api_integration.py

@pytest.fixture
async def authenticated_client():
    """Test client with valid authentication."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Create test API key
        api_key = await create_test_api_key()
        client.headers.update({"Authorization": f"Bearer {api_key}"})
        yield client

async def test_search_api_with_real_backend(authenticated_client):
    """Test search API with integrated backend services."""
    # Setup: Ensure test data exists
    await seed_test_data([
        {
            "document_id": "api-test-doc-1",
            "content": "FastAPI is a modern web framework for building APIs with Python",
            "metadata": {"framework": "fastapi", "language": "python"}
        },
        {
            "document_id": "api-test-doc-2", 
            "content": "Django provides a comprehensive web development framework",
            "metadata": {"framework": "django", "language": "python"}
        }
    ])
    
    # Test 1: Basic search request
    search_request = {
        "query": "web framework Python",
        "limit": 10,
        "search_type": "hybrid"
    }
    
    response = await authenticated_client.post("/api/v1/search", json=search_request)
    
    # Validate response structure
    assert response.status_code == 200
    data = response.json()
    
    assert "results" in data
    assert "metadata" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) > 0
    
    # Validate result structure
    first_result = data["results"][0]
    required_fields = ["document_id", "content", "score", "metadata"]
    for field in required_fields:
        assert field in first_result, f"Missing required field: {field}"
    
    # Validate metadata
    metadata = data["metadata"]
    assert "search_time" in metadata
    assert "total_results" in metadata
    assert metadata["search_time"] < 0.5  # Under 500ms
    assert metadata["total_results"] >= len(data["results"])
    
    # Test 2: Search with filters
    filtered_request = {
        "query": "framework",
        "limit": 5,
        "search_type": "hybrid",
        "filters": {"framework": "fastapi"}
    }
    
    filtered_response = await authenticated_client.post("/api/v1/search", json=filtered_request)
    assert filtered_response.status_code == 200
    
    filtered_data = filtered_response.json()
    assert len(filtered_data["results"]) > 0
    
    # Verify filter application
    for result in filtered_data["results"]:
        assert result["metadata"].get("framework") == "fastapi"

# Command to run
pytest tests/integration/test_api_integration.py::test_search_api_with_real_backend -v
```

**TC-S3-002-002: Rate Limiting and Security**
```python
async def test_api_rate_limiting_enforcement():
    """Test API rate limiting with real Redis backend."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        api_key = await create_test_api_key(rate_limit=5)  # 5 requests per minute
        client.headers.update({"Authorization": f"Bearer {api_key}"})
        
        search_request = {"query": "test", "limit": 1}
        
        # Make requests within limit
        for i in range(5):
            response = await client.post("/api/v1/search", json=search_request)
            assert response.status_code == 200, f"Request {i+1} should succeed"
        
        # Exceed rate limit
        response = await client.post("/api/v1/search", json=search_request)
        assert response.status_code == 429, "Should return rate limit error"
        
        rate_limit_data = response.json()
        assert "error" in rate_limit_data
        assert "rate limit" in rate_limit_data["error"].lower()
        assert "retry_after" in rate_limit_data

async def test_api_authentication_security():
    """Test API authentication and authorization."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        search_request = {"query": "test", "limit": 1}
        
        # Test without authentication
        response = await client.post("/api/v1/search", json=search_request)
        assert response.status_code == 401, "Should require authentication"
        
        # Test with invalid API key
        client.headers.update({"Authorization": "Bearer invalid-key"})
        response = await client.post("/api/v1/search", json=search_request)
        assert response.status_code == 401, "Should reject invalid API key"
        
        # Test with expired API key
        expired_key = await create_expired_api_key()
        client.headers.update({"Authorization": f"Bearer {expired_key}"})
        response = await client.post("/api/v1/search", json=search_request)
        assert response.status_code == 401, "Should reject expired API key"
```

## Level 4: Domain-Specific Validation Scenarios

### Scenario S4-001: RAG Accuracy Validation

**Purpose**: Validate RAG system meets business accuracy requirements
**Execution Time**: <5 minutes
**Automation Level**: 80% (manual review of edge cases)

**Test Cases**:

**TC-S4-001-001: Search Recall and Precision Validation**
```python
# tests/domain/test_rag_accuracy.py

@pytest.fixture
def ground_truth_dataset():
    """Load curated ground truth dataset for accuracy testing."""
    return [
        {
            "query_id": "q001",
            "query": "how to implement async functions in Python",
            "relevant_document_ids": ["doc-python-async-1", "doc-python-async-2", "doc-concurrency-1"],
            "highly_relevant_ids": ["doc-python-async-1"],
            "document_relevance_scores": {
                "doc-python-async-1": 1.0,
                "doc-python-async-2": 0.8,
                "doc-concurrency-1": 0.6,
                "doc-threading-1": 0.3,
                "doc-unrelated-1": 0.0
            }
        },
        {
            "query_id": "q002", 
            "query": "machine learning data preprocessing techniques",
            "relevant_document_ids": ["doc-ml-preprocessing-1", "doc-data-cleaning-1", "doc-feature-engineering-1"],
            "highly_relevant_ids": ["doc-ml-preprocessing-1", "doc-data-cleaning-1"],
            "document_relevance_scores": {
                "doc-ml-preprocessing-1": 1.0,
                "doc-data-cleaning-1": 0.9,
                "doc-feature-engineering-1": 0.7,
                "doc-ml-training-1": 0.4,
                "doc-statistics-1": 0.2
            }
        },
        # Additional queries for comprehensive testing...
    ]

async def test_search_recall_accuracy_validation(rag_system, ground_truth_dataset):
    """Validate search recall meets business requirements (>95% recall@10)."""
    recall_scores = {'recall@1': [], 'recall@5': [], 'recall@10': []}
    precision_scores = {'precision@1': [], 'precision@5': [], 'precision@10': []}
    
    for query_data in ground_truth_dataset:
        query = query_data['query']
        relevant_docs = set(query_data['relevant_document_ids'])
        
        # Perform search
        search_results = await rag_system.search(
            query=query,
            limit=10,
            search_type='hybrid'
        )
        
        retrieved_docs = [r.document_id for r in search_results]
        
        # Calculate recall and precision at different K values
        for k in [1, 5, 10]:
            retrieved_k = set(retrieved_docs[:k])
            
            # Recall@K
            if len(relevant_docs) > 0:
                recall_k = len(retrieved_k & relevant_docs) / len(relevant_docs)
                recall_scores[f'recall@{k}'].append(recall_k)
            
            # Precision@K
            if len(retrieved_k) > 0:
                precision_k = len(retrieved_k & relevant_docs) / len(retrieved_k)
                precision_scores[f'precision@{k}'].append(precision_k)
    
    # Calculate average metrics
    avg_recalls = {metric: np.mean(scores) for metric, scores in recall_scores.items()}
    avg_precisions = {metric: np.mean(scores) for metric, scores in precision_scores.items()}
    
    # Log detailed results
    logger.info(f"Recall metrics: {avg_recalls}")
    logger.info(f"Precision metrics: {avg_precisions}")
    
    # Validate business requirements
    assert avg_recalls['recall@1'] >= 0.80, f"Recall@1 {avg_recalls['recall@1']:.3f} below requirement 0.80"
    assert avg_recalls['recall@5'] >= 0.90, f"Recall@5 {avg_recalls['recall@5']:.3f} below requirement 0.90"
    assert avg_recalls['recall@10'] >= 0.95, f"Recall@10 {avg_recalls['recall@10']:.3f} below requirement 0.95"
    
    assert avg_precisions['precision@1'] >= 0.85, f"Precision@1 {avg_precisions['precision@1']:.3f} below requirement 0.85"
    assert avg_precisions['precision@5'] >= 0.70, f"Precision@5 {avg_precisions['precision@5']:.3f} below requirement 0.70"

# Command to run
pytest tests/domain/test_rag_accuracy.py::test_search_recall_accuracy_validation -v
```

**TC-S4-001-002: NDCG Quality Score Validation**
```python
async def test_ranking_quality_ndcg_validation(rag_system, ground_truth_dataset):
    """Validate result ranking quality using NDCG (Normalized Discounted Cumulative Gain)."""
    ndcg_scores = []
    
    for query_data in ground_truth_dataset:
        query = query_data['query']
        relevance_scores = query_data['document_relevance_scores']
        
        search_results = await rag_system.search(
            query=query,
            limit=10,
            search_type='hybrid'
        )
        
        # Calculate NDCG@10
        ndcg = calculate_ndcg(search_results, relevance_scores, k=10)
        ndcg_scores.append(ndcg)
        
        logger.debug(f"Query: {query[:50]}... | NDCG@10: {ndcg:.3f}")
    
    # Analyze NDCG distribution
    avg_ndcg = np.mean(ndcg_scores)
    min_ndcg = np.min(ndcg_scores)
    p90_ndcg = np.percentile(ndcg_scores, 10)  # 10th percentile (worst 10%)
    
    logger.info(f"NDCG@10 - Average: {avg_ndcg:.3f}, Min: {min_ndcg:.3f}, P10: {p90_ndcg:.3f}")
    
    # Validate ranking quality requirements
    assert avg_ndcg >= 0.80, f"Average NDCG@10 {avg_ndcg:.3f} below requirement 0.80"
    assert p90_ndcg >= 0.60, f"P10 NDCG@10 {p90_ndcg:.3f} below requirement 0.60"
    assert min_ndcg >= 0.40, f"Minimum NDCG@10 {min_ndcg:.3f} below requirement 0.40"

def calculate_ndcg(search_results: List[SearchResult], relevance_scores: Dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    # DCG calculation
    dcg = 0.0
    for i, result in enumerate(search_results[:k]):
        doc_id = result.document_id
        relevance = relevance_scores.get(doc_id, 0.0)
        dcg += relevance / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # IDCG calculation (perfect ranking)
    sorted_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted_relevances))
    
    return dcg / idcg if idcg > 0 else 0.0
```

This comprehensive test scenarios document provides detailed, executable test cases across all four validation levels, ensuring thorough quality assurance for the RAG system implementation.