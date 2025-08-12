"""
Comprehensive End-to-End Integration Tests for RAG System

Tests the complete RAG pipeline from document ingestion to vector search,
validating integration between all three major components:
1. Vector Database Layer (Qdrant)
2. Embedding Service (Voyage AI)
3. Document Ingestion Pipeline

Performance targets and success criteria validation included.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import json
import time
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

# Import RAG system components
from contexter.vector import (
    QdrantVectorStore, VectorStoreConfig, VectorDocument, SearchResult,
    VectorSearchEngine, SearchQuery, BatchUploader, BatchConfig
)
from contexter.vector.embedding_engine import VoyageEmbeddingEngine, EmbeddingEngineConfig
from contexter.ingestion.pipeline import IngestionPipeline
from contexter.ingestion.json_parser import JSONDocumentParser
from contexter.ingestion.chunking_engine import IntelligentChunkingEngine
from contexter.ingestion.metadata_extractor import MetadataExtractor
from contexter.models.embedding_models import EmbeddingRequest, EmbeddingResult, BatchResult, InputType


logger = logging.getLogger(__name__)


@dataclass 
class RAGSystemMetrics:
    """Comprehensive metrics for RAG system performance tracking."""
    
    # Ingestion metrics
    total_documents_processed: int = 0
    total_chunks_created: int = 0
    total_vectors_stored: int = 0
    avg_ingestion_time_per_doc: float = 0.0
    
    # Search metrics
    total_searches_performed: int = 0
    avg_search_latency_ms: float = 0.0
    search_accuracy_score: float = 0.0
    
    # Quality metrics
    avg_chunk_quality_score: float = 0.0
    embedding_success_rate: float = 0.0
    vector_storage_success_rate: float = 0.0
    
    # Performance compliance
    meets_ingestion_target: bool = False
    meets_search_latency_target: bool = False
    meets_memory_target: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            'ingestion': {
                'documents_processed': self.total_documents_processed,
                'chunks_created': self.total_chunks_created,
                'vectors_stored': self.total_vectors_stored,
                'avg_time_per_doc': self.avg_ingestion_time_per_doc
            },
            'search': {
                'searches_performed': self.total_searches_performed,
                'avg_latency_ms': self.avg_search_latency_ms,
                'accuracy_score': self.search_accuracy_score
            },
            'quality': {
                'avg_chunk_quality': self.avg_chunk_quality_score,
                'embedding_success_rate': self.embedding_success_rate,
                'storage_success_rate': self.vector_storage_success_rate
            },
            'compliance': {
                'ingestion_target': self.meets_ingestion_target,
                'search_latency_target': self.meets_search_latency_target,
                'memory_target': self.meets_memory_target
            }
        }


class MockQdrantVectorStore:
    """Production-like mock for Qdrant vector store with realistic behavior."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self._vectors: Dict[str, VectorDocument] = {}
        self._initialized = False
        self._performance_metrics = {
            'search_count': 0,
            'upload_count': 0,
            'avg_search_time': 0.0
        }
    
    async def initialize(self):
        """Initialize the mock vector store."""
        await asyncio.sleep(0.01)  # Simulate initialization time
        self._initialized = True
    
    async def cleanup(self):
        """Cleanup the mock vector store."""
        self._vectors.clear()
        self._initialized = False
    
    async def upsert_vector(self, document: VectorDocument) -> bool:
        """Store a single vector document."""
        await asyncio.sleep(0.001)  # Simulate storage latency
        self._vectors[document.id] = document
        self._performance_metrics['upload_count'] += 1
        return True
    
    async def upsert_vectors_batch(self, documents: List[VectorDocument]) -> Dict[str, Any]:
        """Store multiple vector documents in batch."""
        start_time = time.time()
        
        # Simulate realistic batch upload time
        await asyncio.sleep(len(documents) * 0.001)
        
        successful_uploads = 0
        for doc in documents:
            self._vectors[doc.id] = doc
            successful_uploads += 1
        
        processing_time = time.time() - start_time
        self._performance_metrics['upload_count'] += successful_uploads
        
        return {
            'successful_uploads': successful_uploads,
            'failed_uploads': 0,
            'total_time': processing_time
        }
    
    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar vectors with realistic simulation."""
        start_time = time.time()
        
        # Simulate search latency based on collection size
        base_latency = 0.005 + (len(self._vectors) * 0.00001)
        await asyncio.sleep(base_latency)
        
        results = []
        sorted_docs = list(self._vectors.values())[:top_k]
        
        for i, doc in enumerate(sorted_docs):
            # Apply filters if specified
            if filters:
                matches_filter = True
                for key, value in filters.items():
                    if key not in doc.payload or doc.payload[key] != value:
                        matches_filter = False
                        break
                if not matches_filter:
                    continue
            
            # Generate realistic similarity scores
            score = 0.95 - (i * 0.05) + (hash(doc.id) % 10) / 100
            
            if score_threshold and score < score_threshold:
                continue
            
            result = SearchResult(
                id=doc.id,
                score=score,
                payload=doc.payload,
                vector=doc.vector
            )
            results.append(result)
        
        # Update performance metrics
        search_time = time.time() - start_time
        self._performance_metrics['search_count'] += 1
        current_avg = self._performance_metrics['avg_search_time']
        count = self._performance_metrics['search_count']
        self._performance_metrics['avg_search_time'] = (
            (current_avg * (count - 1) + search_time) / count
        )
        
        return results[:top_k]
    
    async def get_vector(self, vector_id: str) -> Optional[VectorDocument]:
        """Retrieve a specific vector by ID."""
        await asyncio.sleep(0.001)
        return self._vectors.get(vector_id)
    
    async def count_vectors(self) -> int:
        """Get total vector count."""
        return len(self._vectors)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector store."""
        return {
            'status': 'healthy' if self._initialized else 'not_initialized',
            'vector_count': len(self._vectors),
            'performance': self._performance_metrics
        }


class MockVoyageEmbeddingEngine:
    """Production-like mock for Voyage AI embedding engine."""
    
    def __init__(self, config: EmbeddingEngineConfig):
        self.config = config
        self._initialized = False
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'cache_hits': 0,
            'processing_time': 0.0
        }
        self._cache = {}  # Simple content hash -> embedding cache
    
    async def initialize(self):
        """Initialize the embedding engine."""
        await asyncio.sleep(0.05)  # Simulate initialization
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the embedding engine."""
        self._initialized = False
        self._cache.clear()
    
    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate a single embedding with caching simulation."""
        start_time = time.time()
        
        # Check cache first
        cache_key = request.content_hash
        if cache_key in self._cache:
            self._metrics['cache_hits'] += 1
            return EmbeddingResult(
                content_hash=cache_key,
                embedding=self._cache[cache_key],
                model=self.config.voyage_model,
                dimensions=2048,
                processing_time=0.001,  # Fast cache hit
                cache_hit=True
            )
        
        # Simulate API call latency
        await asyncio.sleep(0.1 + len(request.content) * 0.00001)
        
        # Generate deterministic mock embedding based on content
        embedding = self._generate_mock_embedding(request.content)
        self._cache[cache_key] = embedding
        
        processing_time = time.time() - start_time
        self._metrics['total_requests'] += 1
        self._metrics['successful_requests'] += 1
        self._metrics['processing_time'] += processing_time
        
        return EmbeddingResult(
            content_hash=cache_key,
            embedding=embedding,
            model=self.config.voyage_model,
            dimensions=2048,
            processing_time=processing_time,
            cache_hit=False
        )
    
    async def generate_batch_embeddings(self, requests: List[EmbeddingRequest]) -> BatchResult:
        """Generate embeddings in batch with realistic performance."""
        start_time = time.time()
        results = []
        
        # Process in batches for realism
        batch_size = min(len(requests), self.config.batch_size)
        
        for request in requests:
            result = await self.generate_embedding(request)
            results.append(result)
        
        processing_time = time.time() - start_time
        
        return BatchResult(
            batch_id=f"batch_{int(time.time())}",
            results=results,
            processing_time=processing_time,
            cache_hits=sum(1 for r in results if r.cache_hit),
            total_requests=len(requests)
        )
    
    def _generate_mock_embedding(self, content: str) -> List[float]:
        """Generate deterministic mock embedding based on content."""
        # Use content hash to generate consistent but varied embeddings
        content_hash = hash(content)
        embedding = []
        
        for i in range(2048):
            # Create pseudo-random but deterministic values
            seed = (content_hash + i) % (2**31)
            value = (seed / (2**31)) * 2 - 1  # Range: -1 to 1
            embedding.append(value)
        
        return embedding
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of the embedding engine."""
        avg_processing_time = (
            self._metrics['processing_time'] / max(1, self._metrics['total_requests'])
        )
        
        return {
            'status': 'healthy' if self._initialized else 'not_initialized',
            'performance': {
                'total_requests': self._metrics['total_requests'],
                'success_rate': (
                    self._metrics['successful_requests'] / 
                    max(1, self._metrics['total_requests'])
                ),
                'cache_hit_rate': (
                    self._metrics['cache_hits'] / 
                    max(1, self._metrics['total_requests'])
                ),
                'avg_processing_time': avg_processing_time
            }
        }


class MockStorageManager:
    """Simple mock storage manager."""
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass


@pytest.fixture
def sample_documentation_files(tmp_path):
    """Create realistic documentation files for testing."""
    docs = {}
    
    # Python library documentation
    python_doc = {
        "metadata": {
            "name": "advanced-ml-lib",
            "version": "3.2.1",
            "description": "Advanced machine learning library with GPU acceleration",
            "category": "machine-learning",
            "star_count": 15000,
            "trust_score": 0.92
        },
        "installation": {
            "pip": "pip install advanced-ml-lib[gpu]",
            "conda": "conda install -c conda-forge advanced-ml-lib",
            "requirements": [
                "numpy>=1.21.0",
                "scipy>=1.7.0",
                "torch>=1.12.0",
                "transformers>=4.20.0"
            ]
        },
        "quick_start": """
# Advanced ML Library Quick Start

## Installation
```bash
pip install advanced-ml-lib[gpu]
```

## Basic Usage
```python
import advanced_ml as aml

# Create a neural network model
model = aml.NeuralNetwork(
    layers=[128, 64, 32, 1],
    activation='relu',
    optimizer='adam'
)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
```
        """,
        "api_reference": {
            "NeuralNetwork": """
class NeuralNetwork:
    '''High-performance neural network implementation with GPU support.'''
    
    def __init__(self, layers, activation='relu', optimizer='adam', device='auto'):
        '''Initialize neural network.
        
        Args:
            layers: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            optimizer: Optimization algorithm ('adam', 'sgd', 'rmsprop')
            device: Compute device ('cpu', 'cuda', 'auto')
        '''
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.device = self._select_device(device)
        self._build_network()
    
    def fit(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        '''Train the neural network.
        
        Args:
            X: Training features (numpy array or tensor)
            y: Training targets (numpy array or tensor)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            
        Returns:
            Training history dictionary
        '''
        # Implementation details...
        return self._training_loop(X, y, epochs, batch_size, validation_split)
    
    def predict(self, X):
        '''Make predictions on new data.
        
        Args:
            X: Input features (numpy array or tensor)
            
        Returns:
            Predictions (numpy array)
        '''
        return self._forward_pass(X)
            """,
            "Transformer": """
class Transformer:
    '''Transformer model for sequence-to-sequence tasks.'''
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        '''Initialize transformer model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        '''
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self._build_transformer()
    
    def encode(self, input_ids, attention_mask=None):
        '''Encode input sequence.'''
        return self._encoder_forward(input_ids, attention_mask)
    
    def decode(self, encoder_output, target_ids):
        '''Decode target sequence.'''
        return self._decoder_forward(encoder_output, target_ids)
            """
        },
        "examples": {
            "image_classification": """
# Image Classification Example

import advanced_ml as aml
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Normalize features
X = X / 16.0

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = aml.NeuralNetwork(
    layers=[64, 128, 64, 10],
    activation='relu',
    optimizer='adam'
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")
            """,
            "text_generation": """
# Text Generation with Transformer

import advanced_ml as aml

# Load pre-trained model
model = aml.Transformer.from_pretrained('gpt-small')

# Generate text
prompt = "The future of artificial intelligence"
generated = model.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    do_sample=True
)

print(generated)
            """
        }
    }
    
    python_doc_path = tmp_path / "advanced_ml_lib.json"
    with open(python_doc_path, 'w') as f:
        json.dump(python_doc, f, indent=2)
    docs['python_ml'] = python_doc_path
    
    # JavaScript framework documentation
    js_doc = {
        "library_info": {
            "library_id": "react-advanced-ui",
            "name": "React Advanced UI",
            "version": "5.8.2",
            "category": "ui-framework",
            "star_count": 25000,
            "trust_score": 0.89
        },
        "contexts": [
            {
                "content": """
# React Advanced UI Framework

A comprehensive UI component library for React applications with TypeScript support.

## Installation

```bash
npm install react-advanced-ui
# or
yarn add react-advanced-ui
```

## Basic Setup

```tsx
import React from 'react';
import { ThemeProvider, Button, Card } from 'react-advanced-ui';

function App() {
  return (
    <ThemeProvider theme="modern">
      <Card padding="lg">
        <Button variant="primary" size="lg">
          Get Started
        </Button>
      </Card>
    </ThemeProvider>
  );
}

export default App;
```
                """,
                "source": "README.md",
                "token_count": 150
            },
            {
                "content": """
# Component API Reference

## Button Component

```tsx
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  disabled?: boolean;
  loading?: boolean;
  onClick?: (event: MouseEvent) => void;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  children,
  ...props
}) => {
  return (
    <button
      className={clsx(
        'btn',
        `btn--${variant}`,
        `btn--${size}`,
        { 'btn--disabled': disabled, 'btn--loading': loading }
      )}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading ? <Spinner size="sm" /> : children}
    </button>
  );
};
```

## Card Component

```tsx
interface CardProps {
  padding?: 'none' | 'sm' | 'md' | 'lg' | 'xl';
  shadow?: boolean;
  rounded?: boolean;
  children: React.ReactNode;
}

const Card: React.FC<CardProps> = ({
  padding = 'md',
  shadow = true,
  rounded = true,
  children
}) => {
  return (
    <div
      className={clsx(
        'card',
        `card--padding-${padding}`,
        { 'card--shadow': shadow, 'card--rounded': rounded }
      )}
    >
      {children}
    </div>
  );
};
```
                """,
                "source": "components.md",
                "token_count": 280
            }
        ],
        "total_tokens": 430
    }
    
    js_doc_path = tmp_path / "react_advanced_ui.json"
    with open(js_doc_path, 'w') as f:
        json.dump(js_doc, f, indent=2)
    docs['js_ui'] = js_doc_path
    
    return docs


@pytest_asyncio.fixture
async def rag_system_components():
    """Create integrated RAG system components with realistic mocks."""
    # Mock configurations
    vector_config = VectorStoreConfig(
        vector_size=2048,
        collection_name="test_docs",
        batch_size=100
    )
    
    embedding_config = EmbeddingEngineConfig(
        voyage_api_key="test_key",
        voyage_model="voyage-code-3",
        batch_size=50,
        cache_max_entries=10000
    )
    
    # Create mock components
    vector_store = MockQdrantVectorStore(vector_config)
    embedding_engine = MockVoyageEmbeddingEngine(embedding_config)
    storage_manager = MockStorageManager()
    
    # Initialize components
    await vector_store.initialize()
    await embedding_engine.initialize()
    await storage_manager.initialize()
    
    # Create integrated systems
    search_engine = VectorSearchEngine(
        vector_store=vector_store,
        enable_caching=True,
        cache_size=1000
    )
    
    ingestion_pipeline = IngestionPipeline(
        storage_manager=storage_manager,
        embedding_engine=embedding_engine,
        vector_storage=vector_store,
        max_workers=3,
        quality_threshold=0.6
    )
    
    await ingestion_pipeline.initialize()
    
    components = {
        'vector_store': vector_store,
        'embedding_engine': embedding_engine,
        'storage_manager': storage_manager,
        'search_engine': search_engine,
        'ingestion_pipeline': ingestion_pipeline
    }
    
    yield components
    
    # Cleanup
    await ingestion_pipeline.shutdown()
    await embedding_engine.shutdown()
    await vector_store.cleanup()


@pytest.mark.integration
class TestRAGSystemEndToEnd:
    """Comprehensive end-to-end integration tests for the RAG system."""
    
    @pytest.mark.asyncio
    async def test_complete_document_ingestion_to_search_workflow(
        self, 
        rag_system_components, 
        sample_documentation_files
    ):
        """Test complete workflow from document ingestion to successful search."""
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        search_engine = components['search_engine']
        vector_store = components['vector_store']
        
        metrics = RAGSystemMetrics()
        
        # Phase 1: Ingest documents
        logger.info("Phase 1: Document ingestion")
        ingestion_start = time.time()
        
        job_ids = []
        for doc_name, doc_path in sample_documentation_files.items():
            job_id = await ingestion_pipeline.queue_document(
                library_id=doc_name,
                version="1.0.0",
                doc_path=doc_path,
                priority=8,
                metadata={
                    "test_run": True,
                    "doc_type": "integration_test"
                }
            )
            job_ids.append(job_id)
        
        # Wait for all ingestion jobs to complete
        ingestion_results = []
        for job_id in job_ids:
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=60.0)
            assert result is not None, f"Job {job_id} failed to complete"
            assert result.success, f"Job {job_id} failed: {result.error_message}"
            ingestion_results.append(result)
        
        ingestion_time = time.time() - ingestion_start
        
        # Update metrics
        metrics.total_documents_processed = len(ingestion_results)
        metrics.total_chunks_created = sum(r.chunks_created for r in ingestion_results)
        metrics.total_vectors_stored = sum(r.vectors_generated for r in ingestion_results)
        metrics.avg_ingestion_time_per_doc = ingestion_time / len(ingestion_results)
        metrics.avg_chunk_quality_score = sum(r.avg_quality_score for r in ingestion_results) / len(ingestion_results)
        
        # Validate ingestion results
        assert metrics.total_documents_processed >= 2
        assert metrics.total_chunks_created >= 10, f"Expected at least 10 chunks, got {metrics.total_chunks_created}"
        assert metrics.total_vectors_stored == metrics.total_chunks_created
        assert metrics.avg_chunk_quality_score > 0.5
        
        # Verify vectors are stored
        vector_count = await vector_store.count_vectors()
        assert vector_count == metrics.total_vectors_stored
        
        logger.info(f"Ingestion completed: {metrics.total_documents_processed} docs, "
                   f"{metrics.total_chunks_created} chunks, {ingestion_time:.2f}s")
        
        # Phase 2: Perform searches
        logger.info("Phase 2: Vector search validation")
        
        search_queries = [
            {
                "text": "neural network training with GPU acceleration",
                "expected_matches": ["advanced-ml-lib", "NeuralNetwork", "gpu"]
            },
            {
                "text": "React component Button interface TypeScript",
                "expected_matches": ["react-advanced-ui", "Button", "typescript"]
            },
            {
                "text": "machine learning transformer model implementation",
                "expected_matches": ["Transformer", "model", "machine"]
            },
            {
                "text": "JavaScript framework UI components installation",
                "expected_matches": ["react", "component", "installation"]
            }
        ]
        
        search_latencies = []
        search_success_count = 0
        
        for query_data in search_queries:
            # Generate embedding for search query
            search_start = time.time()
            
            # Mock query embedding (in real system, would use embedding engine)
            query_embedding = components['embedding_engine']._generate_mock_embedding(query_data["text"])
            
            # Perform vector search
            search_query = SearchQuery(
                vector=query_embedding,
                top_k=5,
                filters={"test_run": True}
            )
            
            search_results = await search_engine.search(search_query)
            search_latency = (time.time() - search_start) * 1000  # Convert to ms
            search_latencies.append(search_latency)
            
            # Validate search results
            assert len(search_results) > 0, f"No results for query: {query_data['text']}"
            
            # Check relevance (simplified - check payload content)
            result_content = " ".join([
                str(result.payload.get('content_preview', '')) 
                for result in search_results
            ]).lower()
            
            relevance_matches = sum(
                1 for expected in query_data["expected_matches"]
                if expected.lower() in result_content
            )
            
            if relevance_matches > 0:
                search_success_count += 1
            
            logger.info(f"Search query '{query_data['text'][:50]}...' returned "
                       f"{len(search_results)} results in {search_latency:.1f}ms")
        
        # Update search metrics
        metrics.total_searches_performed = len(search_queries)
        metrics.avg_search_latency_ms = sum(search_latencies) / len(search_latencies)
        metrics.search_accuracy_score = search_success_count / len(search_queries)
        
        # Phase 3: Performance validation
        logger.info("Phase 3: Performance validation")
        
        # Check performance targets
        INGESTION_TARGET_SECONDS = 30.0  # 30 seconds per document max
        SEARCH_LATENCY_TARGET_MS = 100.0  # 100ms max per search
        
        metrics.meets_ingestion_target = metrics.avg_ingestion_time_per_doc <= INGESTION_TARGET_SECONDS
        metrics.meets_search_latency_target = metrics.avg_search_latency_ms <= SEARCH_LATENCY_TARGET_MS
        metrics.meets_memory_target = True  # Simplified for mock testing
        
        # Embedding and storage success rates
        metrics.embedding_success_rate = 1.0  # All mocked operations succeed
        metrics.vector_storage_success_rate = 1.0
        
        # Phase 4: System health validation
        logger.info("Phase 4: System health validation")
        
        # Check component health
        vector_health = vector_store.get_health_status()
        embedding_health = await components['embedding_engine'].health_check()
        pipeline_health = await ingestion_pipeline.health_check()
        
        assert vector_health['status'] == 'healthy'
        assert embedding_health['status'] == 'healthy'
        assert pipeline_health['status'] == 'healthy'
        
        # Final assertions
        assert metrics.total_documents_processed >= 2, "Insufficient documents processed"
        assert metrics.total_chunks_created >= 10, "Insufficient chunks created"
        assert metrics.search_accuracy_score >= 0.5, f"Search accuracy too low: {metrics.search_accuracy_score}"
        assert metrics.avg_chunk_quality_score >= 0.5, "Chunk quality too low"
        assert metrics.meets_search_latency_target, f"Search latency too high: {metrics.avg_search_latency_ms}ms"
        
        logger.info("End-to-end RAG system test completed successfully")
        logger.info(f"Final metrics: {metrics.to_dict()}")
        
        return metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_ingestion_and_search(
        self, 
        rag_system_components, 
        sample_documentation_files
    ):
        """Test concurrent ingestion and search operations."""
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        search_engine = components['search_engine']
        
        # Start ingestion tasks
        ingestion_tasks = []
        for doc_name, doc_path in sample_documentation_files.items():
            task = ingestion_pipeline.queue_document(
                library_id=f"concurrent_{doc_name}",
                version="1.0.0",
                doc_path=doc_path,
                priority=5,
                metadata={"concurrent_test": True}
            )
            ingestion_tasks.append(task)
        
        # Wait for jobs to start
        job_ids = await asyncio.gather(*ingestion_tasks)
        
        # Perform concurrent searches while ingestion is happening
        search_tasks = []
        for i in range(5):
            query_text = f"test query {i}"
            query_embedding = components['embedding_engine']._generate_mock_embedding(query_text)
            
            search_query = SearchQuery(
                vector=query_embedding,
                top_k=3
            )
            
            search_task = search_engine.search(search_query)
            search_tasks.append(search_task)
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks)
        
        # Wait for ingestion to complete
        ingestion_results = []
        for job_id in job_ids:
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=60.0)
            assert result is not None
            ingestion_results.append(result)
        
        # Validate concurrent operations
        assert len(search_results) == 5
        assert all(isinstance(result, list) for result in search_results)
        assert all(result.success for result in ingestion_results)
        
        logger.info("Concurrent ingestion and search test completed successfully")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, 
        rag_system_components, 
        tmp_path
    ):
        """Test error handling and recovery across system components."""
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        search_engine = components['search_engine']
        
        # Test 1: Malformed document handling
        malformed_doc = {"incomplete": "json document without proper structure"}
        malformed_path = tmp_path / "malformed.json"
        with open(malformed_path, 'w') as f:
            f.write('{"incomplete": json')  # Invalid JSON
        
        job_id = await ingestion_pipeline.queue_document(
            library_id="error_test",
            version="1.0.0",
            doc_path=malformed_path,
            priority=1,
            metadata={}
        )
        
        result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=30.0)
        assert result is not None
        assert not result.success
        assert "json" in result.error_message.lower()
        
        # Test 2: Empty document handling
        empty_doc = {}
        empty_path = tmp_path / "empty.json"
        with open(empty_path, 'w') as f:
            json.dump(empty_doc, f)
        
        job_id = await ingestion_pipeline.queue_document(
            library_id="empty_test",
            version="1.0.0",
            doc_path=empty_path,
            priority=1,
            metadata={}
        )
        
        result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=30.0)
        # Should handle gracefully, possibly with low quality score
        assert result is not None
        
        # Test 3: Search with invalid query
        invalid_query = SearchQuery(
            vector=[],  # Empty vector
            top_k=5
        )
        
        # Should handle gracefully
        try:
            await search_engine.search(invalid_query)
        except Exception as e:
            # Expect some form of validation error
            assert "vector" in str(e).lower()
        
        logger.info("Error handling and recovery test completed")
    
    @pytest.mark.asyncio 
    async def test_scalability_with_large_dataset(
        self, 
        rag_system_components, 
        tmp_path
    ):
        """Test system scalability with larger document sets."""
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        search_engine = components['search_engine']
        
        # Create multiple large documents
        large_docs = []
        for i in range(10):
            doc = {
                "metadata": {
                    "name": f"large_lib_{i}",
                    "version": "1.0.0",
                    "description": f"Large library {i} for scalability testing"
                },
                "sections": {}
            }
            
            # Add multiple sections to each document
            for j in range(20):
                section_content = f"""
                Section {j} of library {i}
                
                This is a comprehensive section that contains detailed information
                about feature {j} of the library. It includes code examples,
                API documentation, and usage guidelines.
                
                ```python
                def feature_{j}_function():
                    '''Function for feature {j}'''
                    return f"Feature {j} implementation"
                
                class Feature{j}Class:
                    '''Class for feature {j}'''
                    
                    def __init__(self):
                        self.feature_id = {j}
                        self.library_id = {i}
                    
                    def process(self, data):
                        return f"Processing data with feature {j}"
                ```
                
                Additional documentation and examples would go here.
                This content is designed to create realistic chunk sizes
                and test the system's ability to handle larger datasets.
                """
                
                doc["sections"][f"feature_{j}"] = section_content
            
            doc_path = tmp_path / f"large_doc_{i}.json"
            with open(doc_path, 'w') as f:
                json.dump(doc, f, indent=2)
            large_docs.append(doc_path)
        
        # Measure ingestion performance
        start_time = time.time()
        
        # Queue all documents
        job_ids = []
        for i, doc_path in enumerate(large_docs):
            job_id = await ingestion_pipeline.queue_document(
                library_id=f"large_lib_{i}",
                version="1.0.0",
                doc_path=doc_path,
                priority=5,
                metadata={"scalability_test": True}
            )
            job_ids.append(job_id)
        
        # Wait for all to complete
        results = []
        for job_id in job_ids:
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=120.0)
            assert result is not None
            results.append(result)
        
        ingestion_time = time.time() - start_time
        
        # Validate scalability metrics
        total_docs = len(results)
        successful_docs = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results if r.success)
        total_vectors = sum(r.vectors_generated for r in results if r.success)
        
        success_rate = successful_docs / total_docs
        throughput = total_docs / ingestion_time
        
        # Scalability assertions
        assert success_rate >= 0.9, f"Success rate too low: {success_rate}"
        assert total_chunks >= 100, f"Expected more chunks: {total_chunks}"
        assert total_vectors == total_chunks, "Vector count mismatch"
        assert throughput > 0.1, f"Throughput too low: {throughput} docs/sec"
        
        # Test search performance with larger dataset
        search_latencies = []
        for i in range(10):
            query_embedding = components['embedding_engine']._generate_mock_embedding(f"test query {i}")
            
            search_start = time.time()
            search_query = SearchQuery(
                vector=query_embedding,
                top_k=10,
                filters={"scalability_test": True}
            )
            
            search_results = await search_engine.search(search_query)
            search_latency = (time.time() - search_start) * 1000
            search_latencies.append(search_latency)
            
            assert len(search_results) > 0
        
        avg_search_latency = sum(search_latencies) / len(search_latencies)
        assert avg_search_latency < 200.0, f"Search latency too high: {avg_search_latency}ms"
        
        logger.info(f"Scalability test completed: {total_docs} docs, {total_chunks} chunks, "
                   f"{ingestion_time:.1f}s ingestion, {avg_search_latency:.1f}ms avg search")
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(
        self, 
        rag_system_components, 
        sample_documentation_files
    ):
        """Test memory usage during RAG operations."""
        import psutil
        import os
        
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process documents and monitor memory
        memory_samples = [initial_memory]
        
        for doc_name, doc_path in sample_documentation_files.items():
            job_id = await ingestion_pipeline.queue_document(
                library_id=doc_name,
                version="1.0.0", 
                doc_path=doc_path,
                priority=5,
                metadata={"memory_test": True}
            )
            
            # Wait for completion
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=60.0)
            assert result is not None
            assert result.success
            
            # Sample memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
        
        peak_memory = max(memory_samples)
        memory_growth = peak_memory - initial_memory
        
        # Memory usage assertions (adjust based on realistic expectations)
        MAX_MEMORY_GROWTH_MB = 100  # Allow 100MB growth for test documents
        assert memory_growth < MAX_MEMORY_GROWTH_MB, f"Memory growth too high: {memory_growth:.1f}MB"
        
        logger.info(f"Memory usage: initial {initial_memory:.1f}MB, "
                   f"peak {peak_memory:.1f}MB, growth {memory_growth:.1f}MB")
    
    @pytest.mark.asyncio
    async def test_configuration_and_environment_setup(
        self, 
        tmp_path
    ):
        """Test system configuration and environment setup."""
        # Test configuration validation
        vector_config = VectorStoreConfig(
            vector_size=2048,
            collection_name="test_config",
            batch_size=50
        )
        
        embedding_config = EmbeddingEngineConfig(
            voyage_api_key="test_key",
            voyage_model="voyage-code-3",
            batch_size=25
        )
        
        # Validate configurations
        assert vector_config.vector_size == 2048
        assert embedding_config.batch_size == 25
        assert embedding_config.voyage_model == "voyage-code-3"
        
        # Test component initialization
        vector_store = MockQdrantVectorStore(vector_config)
        embedding_engine = MockVoyageEmbeddingEngine(embedding_config)
        
        await vector_store.initialize()
        await embedding_engine.initialize()
        
        # Test health checks
        vector_health = vector_store.get_health_status()
        embedding_health = await embedding_engine.health_check()
        
        assert vector_health['status'] == 'healthy'
        assert embedding_health['status'] == 'healthy'
        
        # Cleanup
        await embedding_engine.shutdown()
        await vector_store.cleanup()
        
        logger.info("Configuration and environment setup test completed")


@pytest.mark.performance
class TestRAGSystemPerformance:
    """Performance-focused tests for the RAG system."""
    
    @pytest.mark.asyncio
    async def test_ingestion_throughput_performance(
        self, 
        rag_system_components, 
        tmp_path
    ):
        """Test ingestion throughput performance targets."""
        components = rag_system_components
        ingestion_pipeline = components['ingestion_pipeline']
        
        # Create test documents for throughput testing
        docs = []
        for i in range(5):  # Smaller set for focused performance testing
            doc = {
                "metadata": {"name": f"perf_lib_{i}", "version": "1.0.0"},
                "content": f"Performance test content for library {i}. " * 100  # Substantial content
            }
            doc_path = tmp_path / f"perf_doc_{i}.json"
            with open(doc_path, 'w') as f:
                json.dump(doc, f)
            docs.append(doc_path)
        
        # Measure throughput
        start_time = time.time()
        
        job_ids = []
        for i, doc_path in enumerate(docs):
            job_id = await ingestion_pipeline.queue_document(
                library_id=f"perf_lib_{i}",
                version="1.0.0",
                doc_path=doc_path,
                priority=8,
                metadata={"performance_test": True}
            )
            job_ids.append(job_id)
        
        # Wait for completion
        results = []
        for job_id in job_ids:
            result = await ingestion_pipeline.wait_for_job_completion(job_id, timeout=60.0)
            assert result is not None
            assert result.success
            results.append(result)
        
        total_time = time.time() - start_time
        throughput = len(docs) / total_time
        
        # Performance targets
        MIN_THROUGHPUT_DOCS_PER_SEC = 0.2  # Minimum 1 doc per 5 seconds
        assert throughput >= MIN_THROUGHPUT_DOCS_PER_SEC, f"Throughput too low: {throughput:.3f} docs/sec"
        
        logger.info(f"Ingestion throughput: {throughput:.3f} docs/sec")
    
    @pytest.mark.asyncio
    async def test_search_latency_performance(
        self, 
        rag_system_components
    ):
        """Test search latency performance targets."""
        components = rag_system_components
        search_engine = components['search_engine']
        
        # Pre-populate with some vectors
        vector_store = components['vector_store']
        for i in range(50):
            doc = VectorDocument(
                id=f"perf_vector_{i}",
                vector=[0.1 * j for j in range(2048)],  # Simple vector
                payload={"content": f"Performance test vector {i}"}
            )
            await vector_store.upsert_vector(doc)
        
        # Measure search latencies
        latencies = []
        for i in range(20):  # Multiple search samples
            query_embedding = [0.1 * j for j in range(2048)]
            search_query = SearchQuery(vector=query_embedding, top_k=10)
            
            start_time = time.time()
            results = await search_engine.search(search_query)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            assert len(results) > 0
        
        # Performance analysis
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        
        # Performance targets
        MAX_AVG_LATENCY_MS = 100.0
        MAX_P95_LATENCY_MS = 150.0
        
        assert avg_latency <= MAX_AVG_LATENCY_MS, f"Average latency too high: {avg_latency:.1f}ms"
        assert p95_latency <= MAX_P95_LATENCY_MS, f"P95 latency too high: {p95_latency:.1f}ms"
        
        logger.info(f"Search performance: avg {avg_latency:.1f}ms, p95 {p95_latency:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])