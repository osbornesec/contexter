# Contexter RAG System Implementation Plan
## Voyage AI Embeddings + Qdrant Vector Database

### Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technical Specifications](#technical-specifications)
4. [Implementation Phases](#implementation-phases)
5. [Detailed Component Design](#detailed-component-design)
6. [Testing Strategy](#testing-strategy)
7. [Performance Optimization](#performance-optimization)
8. [Timeline and Milestones](#timeline-and-milestones)

---

## Executive Summary

### Project Objectives
- Process and vectorize Contexter library documentation extracts from JSON files
- Generate high-quality code-optimized embeddings using Voyage AI code-3 model (2048 dimensions)
- Store embeddings in locally-hosted Qdrant vector database optimized for RAG
- Enable fast semantic search for code documentation retrieval
- Support hybrid search combining semantic similarity with metadata filtering

### Key Deliverables
1. **Embedding Pipeline**: Automated system for processing JSON extracts into vector embeddings
2. **Vector Storage System**: Optimized Qdrant setup with proper indexing and metadata
3. **Search Interface**: Semantic and hybrid search capabilities with <50ms latency
4. **RAG Integration**: Context retrieval system for augmented generation workflows

### Success Metrics
- Query latency: p95 < 50ms, p99 < 100ms
- Embedding generation: > 1000 documents/minute
- Search accuracy: Recall@10 > 95%
- Storage efficiency: < 100GB for 10M vectors
- System availability: > 99.9% uptime

---

## System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Input Layer                              │
│  JSON Extracts → Document Parser → Chunking Engine           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Embedding Layer                            │
│  Voyage AI Client → Batch Processor → Embedding Cache        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│  Qdrant Collections → HNSW Index → Metadata Payloads         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Search Layer                              │
│  Semantic Search → Hybrid Filters → Result Reranking         │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions
1. **Data Flow**:
   - JSON files → Parser → Chunks → Embeddings → Vectors → Index
   - Query → Embedding → Search → Ranking → Results

2. **Control Flow**:
   - Orchestrator manages pipeline execution
   - Error handler manages retries and fallbacks
   - Monitor tracks performance metrics

3. **Storage Flow**:
   - Embeddings cached in SQLite for reuse
   - Vectors stored in Qdrant with metadata
   - Indexes optimized for similarity search

---

## Technical Specifications

### Core Technologies
```yaml
embedding_model:
  provider: Voyage AI
  model: voyage-code-3
  dimensions: 2048
  context_length: 16000
  api_endpoint: https://api.voyageai.com/v1/embeddings
  batch_size: 100
  rate_limit: 300 requests/minute

vector_database:
  system: Qdrant
  version: 1.8.0+
  deployment: Docker local
  port: 6333
  grpc_port: 6334
  
indexing:
  algorithm: HNSW
  parameters:
    m: 16                    # Number of bi-directional links
    ef_construct: 200        # Size of dynamic candidate list
    ef: 100                  # Size of dynamic candidate list for search
    full_scan_threshold: 10000
    
storage:
  vector_dtype: float32      # Full precision initially
  payload_indexing: true
  wal_capacity_mb: 32
  optimizers_config:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 5
```

### Data Specifications
```yaml
document_processing:
  chunk_size: 1000           # tokens
  chunk_overlap: 200         # tokens
  tokenizer: cl100k_base     # OpenAI tokenizer
  max_chunks_per_doc: 100
  
metadata_schema:
  library_id: string         # Required
  library_name: string       # Required
  version: string           
  doc_type: enum            # [api, guide, tutorial, reference]
  section: string
  subsection: string
  token_count: integer
  char_count: integer
  timestamp: datetime
  source_file: string
  chunk_index: integer
  total_chunks: integer
  language: string          # Programming language
  tags: array[string]
  trust_score: float
  star_count: integer
```

### System Requirements
```yaml
hardware:
  cpu: 8+ cores
  ram: 32GB minimum, 64GB recommended
  storage: 500GB SSD
  network: 100Mbps+
  
software:
  os: Ubuntu 22.04 / macOS 13+
  python: 3.9+
  docker: 24.0+
  docker-compose: 2.20+
  
python_dependencies:
  voyageai: "^0.2.0"
  qdrant-client: "^1.8.0"
  httpx: "^0.25.0"
  pydantic: "^2.0.0"
  tiktoken: "^0.5.0"
  numpy: "^1.24.0"
  tqdm: "^4.66.0"
  rich: "^13.0.0"
  pytest: "^7.4.0"
  pytest-asyncio: "^0.21.0"
```

---

## Implementation Phases

### Phase 1: Environment Setup (Days 1-3)

#### 1.1 Qdrant Deployment
```bash
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.8.0
    container_name: contexter_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_config:/qdrant/config
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
      - QDRANT__STORAGE__WAL__WAL_CAPACITY_MB=32
      - QDRANT__STORAGE__PERFORMANCE__OPTIMIZERS__DELETED_THRESHOLD=0.2
    restart: unless-stopped
```

#### 1.2 Project Structure Creation
```
contexter_rag/
├── src/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── voyage_client.py      # Voyage AI API wrapper
│   │   ├── embedding_generator.py # Batch processing logic
│   │   └── cache_manager.py      # SQLite embedding cache
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── qdrant_manager.py     # Qdrant operations
│   │   ├── collection_schema.py  # Collection definitions
│   │   └── index_optimizer.py    # Index tuning utilities
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── json_parser.py        # Parse contexter JSON
│   │   ├── document_chunker.py   # Smart chunking logic
│   │   └── metadata_extractor.py # Extract metadata
│   ├── search/
│   │   ├── __init__.py
│   │   ├── semantic_search.py    # Vector similarity search
│   │   ├── hybrid_search.py      # Combined search strategies
│   │   ├── reranker.py          # Result reranking
│   │   └── query_processor.py    # Query optimization
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── logger.py            # Structured logging
│   │   ├── metrics.py           # Performance tracking
│   │   └── validators.py        # Data validation
│   └── cli/
│       ├── __init__.py
│       └── commands.py          # CLI interface
├── config/
│   ├── settings.yaml            # Main configuration
│   └── logging.yaml             # Logging configuration
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── scripts/
│   ├── setup_qdrant.py         # Initialize Qdrant
│   ├── migrate_data.py         # Data migration utilities
│   └── benchmark.py            # Performance testing
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

#### 1.3 Configuration Setup
```yaml
# config/settings.yaml
voyage_ai:
  api_key: ${VOYAGE_API_KEY}
  model: voyage-code-3
  dimensions: 2048
  batch_size: 100
  max_retries: 3
  timeout: 30
  rate_limit:
    requests_per_minute: 300
    tokens_per_minute: 1000000

qdrant:
  host: localhost
  port: 6333
  grpc_port: 6334
  collection_name: contexter_docs
  timeout: 30
  prefer_grpc: true
  
processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_chunks_per_doc: 100
  tokenizer: cl100k_base
  batch_size: 1000
  
cache:
  embedding_cache_path: ./cache/embeddings.db
  ttl_hours: 168  # 1 week
  max_size_gb: 10
  
performance:
  max_concurrent_embeddings: 10
  max_concurrent_uploads: 5
  query_cache_size: 1000
  query_cache_ttl: 3600
```

### Phase 2: Data Ingestion Pipeline (Days 4-7)

#### 2.1 JSON Parser Implementation
```python
# src/ingestion/json_parser.py
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    library_id: str
    library_name: str
    version: Optional[str] = None
    doc_type: str = "reference"
    trust_score: float = 0.0
    star_count: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
    source_file: str
    tags: List[str] = Field(default_factory=list)

class ParsedDocument(BaseModel):
    content: str
    metadata: DocumentMetadata
    chunks: Optional[List[Dict[str, Any]]] = None

class ContexterJSONParser:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        
    def parse_library_file(self, file_path: Path) -> List[ParsedDocument]:
        """Parse a single library JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        library_metadata = self._extract_library_metadata(data)
        
        # Parse different sections
        for section_name, section_content in data.items():
            if section_name in ['metadata', '_meta']:
                continue
                
            doc = self._parse_section(
                section_name=section_name,
                section_content=section_content,
                library_metadata=library_metadata,
                source_file=str(file_path)
            )
            documents.extend(doc)
        
        return documents
    
    def _extract_library_metadata(self, data: Dict) -> Dict:
        """Extract library-level metadata"""
        meta = data.get('metadata', {})
        return {
            'library_id': meta.get('library_id', 'unknown'),
            'library_name': meta.get('name', 'unknown'),
            'version': meta.get('version'),
            'trust_score': meta.get('trust_score', 0.0),
            'star_count': meta.get('star_count', 0),
            'tags': meta.get('tags', [])
        }
    
    def _parse_section(
        self,
        section_name: str,
        section_content: Any,
        library_metadata: Dict,
        source_file: str
    ) -> List[ParsedDocument]:
        """Parse a documentation section"""
        documents = []
        
        if isinstance(section_content, str):
            # Simple text content
            metadata = DocumentMetadata(
                **library_metadata,
                doc_type=self._infer_doc_type(section_name),
                section=section_name,
                source_file=source_file
            )
            documents.append(ParsedDocument(
                content=section_content,
                metadata=metadata
            ))
        elif isinstance(section_content, dict):
            # Nested structure
            for key, value in section_content.items():
                subdocs = self._parse_section(
                    section_name=f"{section_name}.{key}",
                    section_content=value,
                    library_metadata=library_metadata,
                    source_file=source_file
                )
                documents.extend(subdocs)
        elif isinstance(section_content, list):
            # List of items
            for i, item in enumerate(section_content):
                subdocs = self._parse_section(
                    section_name=f"{section_name}[{i}]",
                    section_content=item,
                    library_metadata=library_metadata,
                    source_file=source_file
                )
                documents.extend(subdocs)
        
        return documents
    
    def _infer_doc_type(self, section_name: str) -> str:
        """Infer document type from section name"""
        section_lower = section_name.lower()
        if 'api' in section_lower:
            return 'api'
        elif 'guide' in section_lower or 'tutorial' in section_lower:
            return 'guide'
        elif 'example' in section_lower:
            return 'example'
        else:
            return 'reference'
```

#### 2.2 Document Chunking Strategy
```python
# src/ingestion/document_chunker.py
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any]
    token_count: int
    char_count: int
    chunk_index: int
    total_chunks: int
    overlap_start: bool = False
    overlap_end: bool = False

class SmartChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        tokenizer_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        
    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Smart chunking with semantic boundaries"""
        
        # Tokenize the entire document
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            # Document fits in single chunk
            return [Chunk(
                text=text,
                metadata=metadata,
                token_count=len(tokens),
                char_count=len(text),
                chunk_index=0,
                total_chunks=1
            )]
        
        # Find semantic boundaries (paragraphs, code blocks, etc.)
        boundaries = self._find_semantic_boundaries(text, tokens)
        
        # Create chunks respecting boundaries
        chunks = self._create_chunks_with_boundaries(
            text=text,
            tokens=tokens,
            boundaries=boundaries,
            metadata=metadata
        )
        
        return chunks
    
    def _find_semantic_boundaries(
        self,
        text: str,
        tokens: List[int]
    ) -> List[int]:
        """Find natural breaking points in text"""
        boundaries = [0]
        
        # Common boundary patterns
        patterns = [
            '\n\n',      # Paragraph breaks
            '\n```',     # Code block boundaries
            '\n##',      # Markdown headers
            '\n###',
            '\ndef ',    # Function definitions
            '\nclass ',  # Class definitions
        ]
        
        for pattern in patterns:
            pos = 0
            while True:
                pos = text.find(pattern, pos)
                if pos == -1:
                    break
                # Convert character position to token position
                token_pos = len(self.tokenizer.encode(text[:pos]))
                boundaries.append(token_pos)
                pos += len(pattern)
        
        boundaries.append(len(tokens))
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _create_chunks_with_boundaries(
        self,
        text: str,
        tokens: List[int],
        boundaries: List[int],
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create chunks respecting semantic boundaries"""
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(tokens):
            # Find the best endpoint for this chunk
            target_end = current_pos + self.chunk_size
            
            # Find nearest boundary before target
            best_boundary = target_end
            for boundary in boundaries:
                if current_pos < boundary <= target_end:
                    best_boundary = boundary
                    break
                elif boundary > target_end:
                    # If no boundary found within chunk size,
                    # use the target or next boundary (whichever is closer)
                    if boundary - target_end < self.chunk_size * 0.2:
                        best_boundary = boundary
                    break
            
            # Extract chunk with overlap
            chunk_start = max(0, current_pos - (self.chunk_overlap if chunk_index > 0 else 0))
            chunk_end = min(len(tokens), best_boundary)
            
            # Convert tokens back to text
            chunk_tokens = tokens[chunk_start:chunk_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunk_start_token': chunk_start,
                    'chunk_end_token': chunk_end,
                },
                token_count=len(chunk_tokens),
                char_count=len(chunk_text),
                chunk_index=chunk_index,
                total_chunks=-1,  # Will be set after all chunks created
                overlap_start=chunk_index > 0,
                overlap_end=False  # Will be set for last chunk
            ))
            
            current_pos = chunk_end - self.chunk_overlap
            chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        if chunks:
            chunks[-1].overlap_end = False
        
        return chunks
```

### Phase 3: Embedding Generation (Days 8-11)

#### 3.1 Voyage AI Client Implementation
```python
# src/embeddings/voyage_client.py
import httpx
import asyncio
from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
import backoff
from datetime import datetime, timedelta

@dataclass
class EmbeddingResult:
    embedding: List[float]
    model: str
    usage: Dict[str, int]
    timestamp: datetime

class VoyageAIClient:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-3",
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = "https://api.voyageai.com/v1"
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            requests_per_minute=300,
            tokens_per_minute=1000000
        )
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            ),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPStatusError, httpx.RequestError),
        max_tries=3,
        max_time=60
    )
    async def embed_batch(
        self,
        texts: List[str],
        input_type: str = "document"
    ) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of texts"""
        
        # Wait for rate limit clearance
        await self.rate_limiter.acquire(len(texts))
        
        # Prepare request
        payload = {
            "model": self.model,
            "input": texts,
            "input_type": input_type
        }
        
        # Make request
        response = await self.client.post(
            f"{self.base_url}/embeddings",
            json=payload
        )
        
        # Handle response
        if response.status_code == 429:
            # Rate limited - extract retry after
            retry_after = int(response.headers.get("Retry-After", 60))
            await asyncio.sleep(retry_after)
            return await self.embed_batch(texts, input_type)
        
        response.raise_for_status()
        data = response.json()
        
        # Parse embeddings
        results = []
        for embedding_data in data["data"]:
            results.append(EmbeddingResult(
                embedding=embedding_data["embedding"],
                model=data["model"],
                usage=data.get("usage", {}),
                timestamp=datetime.now()
            ))
        
        return results
    
    async def embed_single(
        self,
        text: str,
        input_type: str = "document"
    ) -> EmbeddingResult:
        """Generate embedding for single text"""
        results = await self.embed_batch([text], input_type)
        return results[0]
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times: List[datetime] = []
        self.token_usage: List[Tuple[datetime, int]] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self, batch_size: int):
        """Wait until rate limit allows request"""
        async with self.lock:
            now = datetime.now()
            
            # Clean old entries
            cutoff = now - timedelta(minutes=1)
            self.request_times = [
                t for t in self.request_times if t > cutoff
            ]
            self.token_usage = [
                (t, tokens) for t, tokens in self.token_usage if t > cutoff
            ]
            
            # Check request rate
            while len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    now = datetime.now()
                    cutoff = now - timedelta(minutes=1)
                    self.request_times = [
                        t for t in self.request_times if t > cutoff
                    ]
            
            # Record request
            self.request_times.append(now)
```

#### 3.2 Batch Embedding Processor
```python
# src/embeddings/embedding_generator.py
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm.asyncio import tqdm
import pickle
from datetime import datetime

from .voyage_client import VoyageAIClient, EmbeddingResult
from .cache_manager import EmbeddingCache
from ..ingestion.document_chunker import Chunk

class EmbeddingGenerator:
    def __init__(
        self,
        voyage_client: VoyageAIClient,
        cache_manager: Optional[EmbeddingCache] = None,
        max_concurrent: int = 10,
        batch_size: int = 100
    ):
        self.voyage_client = voyage_client
        self.cache_manager = cache_manager
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'errors': 0,
            'total_tokens': 0
        }
    
    async def generate_embeddings(
        self,
        chunks: List[Chunk],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[Tuple[Chunk, np.ndarray]]:
        """Generate embeddings for list of chunks"""
        
        results = []
        
        # Check cache first
        to_embed = []
        if use_cache and self.cache_manager:
            for chunk in chunks:
                cache_key = self._generate_cache_key(chunk)
                cached_embedding = await self.cache_manager.get(cache_key)
                
                if cached_embedding is not None:
                    results.append((chunk, cached_embedding))
                    self.stats['cache_hits'] += 1
                else:
                    to_embed.append(chunk)
        else:
            to_embed = chunks
        
        # Process chunks in batches
        if to_embed:
            batches = self._create_batches(to_embed)
            
            # Progress bar
            if show_progress:
                pbar = tqdm(
                    total=len(to_embed),
                    desc="Generating embeddings"
                )
            
            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = self._process_batch(batch)
                tasks.append(task)
            
            # Gather results
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for batch_result in batch_results:
                if isinstance(batch_result, Exception):
                    self.stats['errors'] += 1
                    print(f"Error processing batch: {batch_result}")
                    continue
                
                for chunk, embedding in batch_result:
                    results.append((chunk, embedding))
                    
                    # Cache the embedding
                    if use_cache and self.cache_manager:
                        cache_key = self._generate_cache_key(chunk)
                        await self.cache_manager.set(
                            cache_key,
                            embedding,
                            metadata={
                                'model': self.voyage_client.model,
                                'timestamp': datetime.now().isoformat()
                            }
                        )
                    
                    if show_progress:
                        pbar.update(1)
            
            if show_progress:
                pbar.close()
        
        self.stats['total_processed'] += len(chunks)
        
        return results
    
    async def _process_batch(
        self,
        batch: List[Chunk]
    ) -> List[Tuple[Chunk, np.ndarray]]:
        """Process a single batch of chunks"""
        
        async with self.semaphore:
            # Extract texts
            texts = [chunk.text for chunk in batch]
            
            # Generate embeddings
            embedding_results = await self.voyage_client.embed_batch(
                texts=texts,
                input_type="document"
            )
            
            self.stats['api_calls'] += 1
            
            # Pair chunks with embeddings
            results = []
            for chunk, embedding_result in zip(batch, embedding_results):
                embedding_array = np.array(
                    embedding_result.embedding,
                    dtype=np.float32
                )
                results.append((chunk, embedding_array))
                
                # Update token usage
                if embedding_result.usage:
                    self.stats['total_tokens'] += embedding_result.usage.get(
                        'total_tokens', 0
                    )
            
            return results
    
    def _create_batches(
        self,
        chunks: List[Chunk]
    ) -> List[List[Chunk]]:
        """Create batches of chunks"""
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _generate_cache_key(self, chunk: Chunk) -> str:
        """Generate unique cache key for chunk"""
        # Combine metadata for unique key
        key_parts = [
            chunk.metadata.get('library_id', 'unknown'),
            chunk.metadata.get('section', 'unknown'),
            str(chunk.chunk_index),
            str(hash(chunk.text[:100]))  # First 100 chars hash
        ]
        return ":".join(key_parts)
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n=== Embedding Generation Statistics ===")
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"API calls: {self.stats['api_calls']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Total tokens: {self.stats['total_tokens']:,}")
        
        if self.stats['total_processed'] > 0:
            cache_rate = (self.stats['cache_hits'] / self.stats['total_processed']) * 100
            print(f"Cache hit rate: {cache_rate:.1f}%")
```

### Phase 4: Vector Storage Implementation (Days 12-15)

#### 4.1 Qdrant Collection Manager
```python
# src/storage/qdrant_manager.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, UpdateStatus,
    OptimizersConfigDiff, HnswConfigDiff,
    PayloadSchemaType, CollectionInfo
)
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import asyncio
from datetime import datetime

class QdrantManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        collection_name: str = "contexter_docs",
        vector_size: int = 2048,
        prefer_grpc: bool = True
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port if prefer_grpc else None,
            prefer_grpc=prefer_grpc
        )
        
        # Collection configuration
        self.vector_config = VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
        
        # HNSW index configuration for optimal performance
        self.hnsw_config = HnswConfigDiff(
            m=16,                    # Number of edges per node
            ef_construct=200,        # Beam size during index building
            full_scan_threshold=10000,
            max_indexing_threads=0   # Use all available threads
        )
        
        # Optimizer configuration
        self.optimizer_config = OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            default_segment_number=5,
            max_segment_size=200000,
            memmap_threshold=50000,
            indexing_threshold=20000,
            flush_interval_sec=5,
            max_optimization_threads=0
        )
    
    async def initialize_collection(self, recreate: bool = False):
        """Initialize or recreate the collection"""
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and recreate:
            # Delete existing collection
            self.client.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.vector_config,
                hnsw_config=self.hnsw_config,
                optimizers_config=self.optimizer_config,
                shard_number=2,
                replication_factor=1,
                write_consistency_factor=1
            )
            
            # Create payload indexes for efficient filtering
            await self._create_payload_indexes()
            
            print(f"Collection '{self.collection_name}' created successfully")
        else:
            print(f"Collection '{self.collection_name}' already exists")
        
        # Get collection info
        info = self.client.get_collection(self.collection_name)
        print(f"Collection info: {info.vectors_count} vectors, {info.points_count} points")
    
    async def _create_payload_indexes(self):
        """Create indexes on frequently queried fields"""
        
        index_fields = [
            ("library_id", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("language", PayloadSchemaType.KEYWORD),
            ("timestamp", PayloadSchemaType.DATETIME),
            ("token_count", PayloadSchemaType.INTEGER),
            ("trust_score", PayloadSchemaType.FLOAT),
        ]
        
        for field_name, field_type in index_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )
    
    async def upsert_vectors(
        self,
        vectors: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> List[str]:
        """Insert or update vectors with payloads"""
        
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]
        
        # Ensure all inputs have same length
        assert len(vectors) == len(payloads) == len(ids), \
            "Vectors, payloads, and ids must have same length"
        
        # Process in batches
        inserted_ids = []
        for i in range(0, len(vectors), batch_size):
            batch_end = min(i + batch_size, len(vectors))
            
            # Prepare batch
            points = []
            for j in range(i, batch_end):
                # Ensure vector is correct shape and type
                vector = vectors[j]
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                
                # Add timestamp if not present
                if 'indexed_at' not in payloads[j]:
                    payloads[j]['indexed_at'] = datetime.now().isoformat()
                
                point = PointStruct(
                    id=ids[j],
                    vector=vector,
                    payload=payloads[j]
                )
                points.append(point)
            
            # Upsert batch
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                inserted_ids.extend(ids[i:batch_end])
            else:
                print(f"Warning: Batch {i//batch_size} failed to insert")
        
        return inserted_ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        with_payload: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        
        # Ensure vector is correct type
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Build filter conditions
        filter_obj = None
        if filters:
            filter_obj = self._build_filter(filters)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=filter_obj,
            with_payload=with_payload,
            with_vectors=False  # Don't return vectors to save bandwidth
        )
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'payload': result.payload if with_payload else None
            })
        
        return formatted_results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary"""
        
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use OR
                for v in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=v)
                        )
                    )
            else:
                # Single value
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        
        info = self.client.get_collection(self.collection_name)
        
        return {
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'segments_count': info.segments_count,
            'status': info.status,
            'optimizer_status': info.optimizer_status,
            'indexed_vectors_count': info.indexed_vectors_count
        }
    
    async def optimize_collection(self):
        """Trigger collection optimization"""
        
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=self.optimizer_config,
            hnsw_config=self.hnsw_config
        )
        
        print(f"Optimization triggered for collection '{self.collection_name}'")
```

### Phase 5: Search Implementation (Days 16-19)

#### 5.1 Semantic Search Engine
```python
# src/search/semantic_search.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

from ..embeddings.voyage_client import VoyageAIClient
from ..storage.qdrant_manager import QdrantManager

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: Optional[List[str]] = None

class SemanticSearchEngine:
    def __init__(
        self,
        voyage_client: VoyageAIClient,
        qdrant_manager: QdrantManager,
        cache_queries: bool = True,
        cache_ttl: int = 3600
    ):
        self.voyage_client = voyage_client
        self.qdrant_manager = qdrant_manager
        self.cache_queries = cache_queries
        self.cache_ttl = cache_ttl
        
        # Query cache
        self.query_cache: Dict[str, Tuple[np.ndarray, float]] = {}
        
        # Search statistics
        self.stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_latency_ms': 0
        }
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[SearchResult]:
        """Perform semantic search"""
        
        import time
        start_time = time.time()
        
        self.stats['total_searches'] += 1
        
        # Get query embedding
        query_embedding = await self._get_query_embedding(query)
        
        # Search in Qdrant
        raw_results = await self.qdrant_manager.search(
            query_vector=query_embedding,
            limit=limit * 2 if rerank else limit,  # Get more for reranking
            score_threshold=score_threshold,
            filters=filters,
            with_payload=True
        )
        
        # Convert to SearchResult objects
        search_results = []
        for result in raw_results:
            search_results.append(SearchResult(
                id=result['id'],
                content=result['payload'].get('content', ''),
                score=result['score'],
                metadata=result['payload']
            ))
        
        # Rerank if requested
        if rerank and len(search_results) > 0:
            search_results = await self._rerank_results(
                query=query,
                results=search_results,
                limit=limit
            )
        
        # Update statistics
        latency_ms = (time.time() - start_time) * 1000
        self.stats['avg_latency_ms'] = (
            (self.stats['avg_latency_ms'] * (self.stats['total_searches'] - 1) + latency_ms)
            / self.stats['total_searches']
        )
        
        return search_results[:limit]
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get or generate query embedding"""
        
        # Check cache
        if self.cache_queries and query in self.query_cache:
            embedding, timestamp = self.query_cache[query]
            import time
            if time.time() - timestamp < self.cache_ttl:
                self.stats['cache_hits'] += 1
                return embedding
        
        # Generate new embedding
        result = await self.voyage_client.embed_single(
            text=query,
            input_type="query"  # Use query mode for queries
        )
        
        embedding = np.array(result.embedding, dtype=np.float32)
        
        # Cache it
        if self.cache_queries:
            import time
            self.query_cache[query] = (embedding, time.time())
            
            # Clean old cache entries
            current_time = time.time()
            self.query_cache = {
                k: v for k, v in self.query_cache.items()
                if current_time - v[1] < self.cache_ttl
            }
        
        return embedding
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        limit: int
    ) -> List[SearchResult]:
        """Rerank results using additional signals"""
        
        # Simple reranking based on metadata signals
        for result in results:
            # Boost score based on metadata
            boost = 1.0
            
            # Boost by trust score
            trust_score = result.metadata.get('trust_score', 0)
            if trust_score > 8:
                boost *= 1.2
            elif trust_score > 6:
                boost *= 1.1
            
            # Boost by doc type relevance
            doc_type = result.metadata.get('doc_type', '')
            if 'api' in query.lower() and doc_type == 'api':
                boost *= 1.3
            elif 'guide' in query.lower() and doc_type == 'guide':
                boost *= 1.3
            
            # Apply boost
            result.score *= boost
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def print_stats(self):
        """Print search statistics"""
        print("\n=== Search Statistics ===")
        print(f"Total searches: {self.stats['total_searches']}")
        print(f"Cache hits: {self.stats['cache_hits']}")
        print(f"Average latency: {self.stats['avg_latency_ms']:.1f}ms")
        
        if self.stats['total_searches'] > 0:
            cache_rate = (self.stats['cache_hits'] / self.stats['total_searches']) * 100
            print(f"Cache hit rate: {cache_rate:.1f}%")
```

#### 5.2 Hybrid Search Implementation
```python
# src/search/hybrid_search.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .semantic_search import SemanticSearchEngine, SearchResult

@dataclass
class HybridSearchConfig:
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    metadata_boost: Dict[str, float] = None

class HybridSearchEngine:
    def __init__(
        self,
        semantic_engine: SemanticSearchEngine,
        config: Optional[HybridSearchConfig] = None
    ):
        self.semantic_engine = semantic_engine
        self.config = config or HybridSearchConfig()
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        keyword_fields: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword matching"""
        
        # Semantic search
        semantic_results = await self.semantic_engine.search(
            query=query,
            limit=limit * 3,  # Get more for merging
            filters=filters,
            rerank=False  # We'll do our own ranking
        )
        
        # Keyword matching (if fields specified)
        if keyword_fields:
            keyword_scores = self._calculate_keyword_scores(
                query=query,
                results=semantic_results,
                fields=keyword_fields
            )
        else:
            keyword_scores = {r.id: 0.0 for r in semantic_results}
        
        # Combine scores
        combined_results = []
        for result in semantic_results:
            # Calculate combined score
            semantic_score = result.score * self.config.semantic_weight
            keyword_score = keyword_scores.get(result.id, 0) * self.config.keyword_weight
            
            # Apply metadata boosts
            metadata_boost = self._calculate_metadata_boost(result.metadata)
            
            # Final score
            final_score = (semantic_score + keyword_score) * metadata_boost
            
            result.score = final_score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:limit]
    
    def _calculate_keyword_scores(
        self,
        query: str,
        results: List[SearchResult],
        fields: List[str]
    ) -> Dict[str, float]:
        """Calculate keyword matching scores"""
        
        # Simple keyword matching (can be enhanced with TF-IDF)
        query_terms = set(query.lower().split())
        scores = {}
        
        for result in results:
            score = 0.0
            
            for field in fields:
                field_value = result.metadata.get(field, '')
                if isinstance(field_value, str):
                    field_terms = set(field_value.lower().split())
                    
                    # Calculate Jaccard similarity
                    intersection = query_terms & field_terms
                    union = query_terms | field_terms
                    
                    if union:
                        similarity = len(intersection) / len(union)
                        score += similarity
            
            scores[result.id] = score / len(fields) if fields else 0
        
        return scores
    
    def _calculate_metadata_boost(
        self,
        metadata: Dict[str, Any]
    ) -> float:
        """Calculate boost factor based on metadata"""
        
        boost = 1.0
        
        if self.config.metadata_boost:
            for field, weight in self.config.metadata_boost.items():
                value = metadata.get(field)
                
                if field == 'trust_score' and value:
                    # Scale trust score boost
                    boost *= (1 + (value / 10) * weight)
                elif field == 'star_count' and value:
                    # Logarithmic scaling for star count
                    import math
                    boost *= (1 + math.log10(max(1, value)) * weight)
                elif value:
                    # Binary boost for other fields
                    boost *= (1 + weight)
        
        return boost
```

### Phase 6: Integration & Testing (Days 20-23)

#### 6.1 Main Pipeline Orchestrator
```python
# src/pipeline.py
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from .ingestion.json_parser import ContexterJSONParser
from .ingestion.document_chunker import SmartChunker
from .embeddings.voyage_client import VoyageAIClient
from .embeddings.embedding_generator import EmbeddingGenerator
from .embeddings.cache_manager import EmbeddingCache
from .storage.qdrant_manager import QdrantManager
from .search.semantic_search import SemanticSearchEngine
from .search.hybrid_search import HybridSearchEngine, HybridSearchConfig
from .utils.config import load_config
from .utils.logger import setup_logger

class RAGPipeline:
    def __init__(self, config_path: str = "config/settings.yaml"):
        # Load configuration
        self.config = load_config(config_path)
        self.logger = setup_logger("rag_pipeline")
        
        # Initialize components
        self.voyage_client = None
        self.qdrant_manager = None
        self.embedding_generator = None
        self.search_engine = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all pipeline components"""
        
        self.logger.info("Initializing RAG pipeline...")
        
        # Initialize Voyage AI client
        self.voyage_client = VoyageAIClient(
            api_key=self.config['voyage_ai']['api_key'],
            model=self.config['voyage_ai']['model'],
            batch_size=self.config['voyage_ai']['batch_size']
        )
        
        # Initialize Qdrant manager
        self.qdrant_manager = QdrantManager(
            host=self.config['qdrant']['host'],
            port=self.config['qdrant']['port'],
            collection_name=self.config['qdrant']['collection_name'],
            vector_size=self.config['voyage_ai']['dimensions']
        )
        
        # Initialize collection
        await self.qdrant_manager.initialize_collection(recreate=False)
        
        # Initialize embedding cache
        cache = EmbeddingCache(
            cache_path=self.config['cache']['embedding_cache_path'],
            ttl_hours=self.config['cache']['ttl_hours']
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            voyage_client=self.voyage_client,
            cache_manager=cache,
            max_concurrent=self.config['performance']['max_concurrent_embeddings']
        )
        
        # Initialize search engine
        semantic_engine = SemanticSearchEngine(
            voyage_client=self.voyage_client,
            qdrant_manager=self.qdrant_manager
        )
        
        self.search_engine = HybridSearchEngine(
            semantic_engine=semantic_engine,
            config=HybridSearchConfig(
                semantic_weight=0.7,
                keyword_weight=0.3,
                metadata_boost={
                    'trust_score': 0.1,
                    'star_count': 0.05
                }
            )
        )
        
        self.initialized = True
        self.logger.info("RAG pipeline initialized successfully")
    
    async def process_json_files(
        self,
        input_dir: str,
        pattern: str = "*.json"
    ) -> Dict[str, Any]:
        """Process JSON files and generate embeddings"""
        
        if not self.initialized:
            await self.initialize()
        
        input_path = Path(input_dir)
        json_files = list(input_path.glob(pattern))
        
        self.logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Statistics
        stats = {
            'files_processed': 0,
            'documents_parsed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'vectors_stored': 0,
            'errors': [],
            'start_time': datetime.now()
        }
        
        # Process each file
        for json_file in json_files:
            try:
                self.logger.info(f"Processing {json_file}")
                
                # Parse JSON
                parser = ContexterJSONParser(input_path)
                documents = parser.parse_library_file(json_file)
                stats['documents_parsed'] += len(documents)
                
                # Chunk documents
                chunker = SmartChunker(
                    chunk_size=self.config['processing']['chunk_size'],
                    chunk_overlap=self.config['processing']['chunk_overlap']
                )
                
                all_chunks = []
                for doc in documents:
                    chunks = chunker.chunk_document(
                        text=doc.content,
                        metadata=doc.metadata.dict()
                    )
                    all_chunks.extend(chunks)
                
                stats['chunks_created'] += len(all_chunks)
                
                # Generate embeddings
                embeddings_with_chunks = await self.embedding_generator.generate_embeddings(
                    chunks=all_chunks,
                    use_cache=True,
                    show_progress=True
                )
                
                stats['embeddings_generated'] += len(embeddings_with_chunks)
                
                # Store in Qdrant
                vectors = [emb for _, emb in embeddings_with_chunks]
                payloads = [
                    {
                        **chunk.metadata,
                        'content': chunk.text,
                        'token_count': chunk.token_count,
                        'char_count': chunk.char_count,
                        'chunk_index': chunk.chunk_index,
                        'total_chunks': chunk.total_chunks
                    }
                    for chunk, _ in embeddings_with_chunks
                ]
                
                inserted_ids = await self.qdrant_manager.upsert_vectors(
                    vectors=vectors,
                    payloads=payloads
                )
                
                stats['vectors_stored'] += len(inserted_ids)
                stats['files_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing {json_file}: {e}")
                stats['errors'].append({
                    'file': str(json_file),
                    'error': str(e)
                })
        
        # Calculate duration
        stats['end_time'] = datetime.now()
        stats['duration_seconds'] = (
            stats['end_time'] - stats['start_time']
        ).total_seconds()
        
        # Print statistics
        self.embedding_generator.print_stats()
        
        return stats
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        
        if not self.initialized:
            await self.initialize()
        
        results = await self.search_engine.search(
            query=query,
            limit=limit,
            filters=filters,
            keyword_fields=['section', 'doc_type']
        )
        
        # Format results for output
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.id,
                'score': result.score,
                'content': result.content[:500] + '...' if len(result.content) > 500 else result.content,
                'metadata': {
                    'library_id': result.metadata.get('library_id'),
                    'section': result.metadata.get('section'),
                    'doc_type': result.metadata.get('doc_type'),
                    'chunk_index': result.metadata.get('chunk_index'),
                    'total_chunks': result.metadata.get('total_chunks')
                }
            })
        
        return formatted_results
    
    async def cleanup(self):
        """Clean up resources"""
        
        if self.voyage_client:
            await self.voyage_client.close()
        
        self.logger.info("Pipeline cleanup completed")
```

#### 6.2 CLI Interface
```python
# src/cli/commands.py
import click
import asyncio
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ..pipeline import RAGPipeline

console = Console()

@click.group()
def cli():
    """Contexter RAG System CLI"""
    pass

@cli.command()
@click.option('--input-dir', '-i', required=True, help='Directory containing JSON files')
@click.option('--pattern', '-p', default='*.json', help='File pattern to match')
@click.option('--config', '-c', default='config/settings.yaml', help='Configuration file')
def ingest(input_dir: str, pattern: str, config: str):
    """Ingest and process JSON files"""
    
    async def run():
        pipeline = RAGPipeline(config_path=config)
        await pipeline.initialize()
        
        stats = await pipeline.process_json_files(
            input_dir=input_dir,
            pattern=pattern
        )
        
        # Display results
        console.print("\n[bold green]Ingestion Complete![/bold green]")
        console.print(f"Files processed: {stats['files_processed']}")
        console.print(f"Documents parsed: {stats['documents_parsed']}")
        console.print(f"Chunks created: {stats['chunks_created']}")
        console.print(f"Embeddings generated: {stats['embeddings_generated']}")
        console.print(f"Vectors stored: {stats['vectors_stored']}")
        console.print(f"Duration: {stats['duration_seconds']:.1f} seconds")
        
        if stats['errors']:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in stats['errors']:
                console.print(f"  - {error['file']}: {error['error']}")
        
        await pipeline.cleanup()
    
    asyncio.run(run())

@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Number of results')
@click.option('--filter-library', '-f', help='Filter by library ID')
@click.option('--config', '-c', default='config/settings.yaml', help='Configuration file')
def search(query: str, limit: int, filter_library: str, config: str):
    """Search for relevant documents"""
    
    async def run():
        pipeline = RAGPipeline(config_path=config)
        await pipeline.initialize()
        
        # Build filters
        filters = {}
        if filter_library:
            filters['library_id'] = filter_library
        
        # Perform search
        with console.status("[cyan]Searching...[/cyan]"):
            results = await pipeline.search(
                query=query,
                limit=limit,
                filters=filters if filters else None
            )
        
        # Display results
        if results:
            table = Table(title=f"Search Results for: '{query}'")
            table.add_column("Score", style="cyan", width=8)
            table.add_column("Library", style="green", width=20)
            table.add_column("Section", style="yellow", width=20)
            table.add_column("Content", style="white", width=60)
            
            for result in results:
                table.add_row(
                    f"{result['score']:.3f}",
                    result['metadata'].get('library_id', 'N/A'),
                    result['metadata'].get('section', 'N/A'),
                    result['content']
                )
            
            console.print(table)
        else:
            console.print("[yellow]No results found[/yellow]")
        
        await pipeline.cleanup()
    
    asyncio.run(run())

@cli.command()
@click.option('--config', '-c', default='config/settings.yaml', help='Configuration file')
def stats(config: str):
    """Show collection statistics"""
    
    async def run():
        pipeline = RAGPipeline(config_path=config)
        await pipeline.initialize()
        
        stats = await pipeline.qdrant_manager.get_collection_stats()
        
        console.print("\n[bold blue]Collection Statistics[/bold blue]")
        console.print(f"Total vectors: {stats['vectors_count']:,}")
        console.print(f"Total points: {stats['points_count']:,}")
        console.print(f"Indexed vectors: {stats['indexed_vectors_count']:,}")
        console.print(f"Segments: {stats['segments_count']}")
        console.print(f"Status: {stats['status']}")
        
        await pipeline.cleanup()
    
    asyncio.run(run())

if __name__ == '__main__':
    cli()
```

---

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_chunker.py
import pytest
from src.ingestion.document_chunker import SmartChunker

def test_chunker_single_chunk():
    chunker = SmartChunker(chunk_size=1000, chunk_overlap=200)
    text = "This is a short text."
    chunks = chunker.chunk_document(text, {})
    assert len(chunks) == 1
    assert chunks[0].text == text

def test_chunker_multiple_chunks():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=20)
    text = " ".join(["word"] * 500)  # Long text
    chunks = chunker.chunk_document(text, {})
    assert len(chunks) > 1
    # Check overlap
    for i in range(len(chunks) - 1):
        assert chunks[i].overlap_end or i == len(chunks) - 1
```

### Integration Tests
```python
# tests/integration/test_pipeline.py
import pytest
import asyncio
from src.pipeline import RAGPipeline

@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    pipeline = RAGPipeline(config_path="tests/config/test_settings.yaml")
    await pipeline.initialize()
    
    # Test search
    results = await pipeline.search("test query", limit=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    
    await pipeline.cleanup()
```

### Performance Benchmarks
```python
# tests/benchmarks/test_performance.py
import time
import asyncio
from src.pipeline import RAGPipeline

async def benchmark_search_latency():
    pipeline = RAGPipeline()
    await pipeline.initialize()
    
    queries = ["query1", "query2", "query3"]
    latencies = []
    
    for query in queries:
        start = time.time()
        await pipeline.search(query, limit=10)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f"Average latency: {avg_latency:.1f}ms")
    print(f"P95 latency: {p95_latency:.1f}ms")
    
    assert avg_latency < 100  # Should be under 100ms
    assert p95_latency < 200  # P95 should be under 200ms
```

---

## Performance Optimization

### 1. Query Optimization
- **Embedding Cache**: Cache frequently used query embeddings
- **Result Cache**: Cache complete search results with TTL
- **Batch Processing**: Process multiple queries concurrently

### 2. Storage Optimization
- **Index Tuning**: Optimize HNSW parameters for dataset size
- **Payload Indexing**: Index only necessary fields
- **Segment Optimization**: Regular vacuum and merge operations

### 3. Network Optimization
- **Connection Pooling**: Reuse HTTP connections
- **gRPC Usage**: Use gRPC for Qdrant communication
- **Batch Operations**: Minimize round trips

### 4. Resource Optimization
- **Memory Management**: Use memory mapping for large datasets
- **CPU Utilization**: Parallel processing where possible
- **Disk I/O**: SSD storage for Qdrant data

---

## Timeline and Milestones

### Week 1: Foundation (Days 1-7)
- ✅ Day 1-2: Environment setup and Qdrant deployment
- ✅ Day 3: Project structure and configuration
- ✅ Day 4-5: JSON parser implementation
- ✅ Day 6-7: Document chunking strategy

### Week 2: Core Implementation (Days 8-14)
- ✅ Day 8-9: Voyage AI client integration
- ✅ Day 10-11: Embedding generation pipeline
- ✅ Day 12-13: Qdrant storage implementation
- ✅ Day 14: Collection optimization

### Week 3: Search & Integration (Days 15-21)
- ✅ Day 15-16: Semantic search engine
- ✅ Day 17-18: Hybrid search implementation
- ✅ Day 19: Result reranking
- ✅ Day 20-21: Pipeline orchestration

### Week 4: Testing & Optimization (Days 22-28)
- ✅ Day 22-23: Unit and integration tests
- ✅ Day 24-25: Performance benchmarking
- ✅ Day 26-27: Optimization and tuning
- ✅ Day 28: Documentation and deployment

### Deliverables
1. **Working RAG System**: Complete pipeline from JSON to search
2. **Performance Metrics**: <50ms p95 latency, >95% recall@10
3. **Documentation**: API docs, deployment guide, usage examples
4. **Test Suite**: Comprehensive unit, integration, and performance tests
5. **CLI Tools**: User-friendly interface for ingestion and search

### Success Criteria
- ✅ Process 10,000+ documents without errors
- ✅ Generate embeddings at >1000 docs/minute
- ✅ Search latency <50ms for p95
- ✅ Storage efficiency <100GB for 10M vectors
- ✅ Recall@10 >95% on test queries
- ✅ System uptime >99.9%

---

## Appendix: Configuration Files

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.8.0
    container_name: contexter_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
      - ./qdrant_config:/qdrant/config
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Requirements
```txt
# requirements.txt
voyageai>=0.2.0
qdrant-client>=1.8.0
httpx>=0.25.0
pydantic>=2.0.0
tiktoken>=0.5.0
numpy>=1.24.0
tqdm>=4.66.0
rich>=13.0.0
click>=8.1.0
pyyaml>=6.0
python-dotenv>=1.0.0
backoff>=2.2.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

### Environment Variables
```bash
# .env
VOYAGE_API_KEY=your_voyage_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
LOG_LEVEL=INFO
```