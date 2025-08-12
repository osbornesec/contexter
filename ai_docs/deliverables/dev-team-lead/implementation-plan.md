# Implementation Plan - Contexter Documentation Platform

## 1. Executive Summary

### Project Overview
The Contexter Documentation Platform is an integrated system combining the C7DocDownloader for comprehensive documentation acquisition and the RAG system for intelligent semantic search. This implementation plan provides a detailed roadmap for building a production-ready platform that delivers sub-50ms search latency, processes 1000+ documents per minute, and maintains 99.9% availability.

### Key Technical Decisions
- **Async-First Architecture**: Python 3.9+ with asyncio for optimal I/O-bound performance
- **Proxy Management**: BrightData residential proxy integration with circuit breaker patterns
- **Vector Database**: Qdrant v1.8.0+ with HNSW indexing for high-performance semantic search
- **Embeddings**: Voyage AI voyage-code-3 model optimized for code documentation
- **Integration Pattern**: Auto-ingestion pipeline triggering RAG processing post-download
- **Development Approach**: Sprint-based development with 2-week iterations

### Success Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Download Speed | 90% complete within 30s | Performance monitoring |
| Search Latency p95 | < 50ms | Application APM |
| Search Latency p99 | < 100ms | Application APM |
| Embedding Generation | > 1000 docs/minute | Pipeline metrics |
| Concurrent Users | 100 without degradation | Load testing |
| System Availability | > 99.9% | Infrastructure monitoring |
| Memory Usage | < 8GB during operation | Resource monitoring |

## 2. Technical Stack & Dependencies

### Core Technology Stack
| Component | Technology | Version | Justification |
|-----------|------------|---------|---------------|
| **Runtime** | Python | 3.9+ | Modern asyncio features, typing support |
| **HTTP Client** | httpx | 0.24+ | Async HTTP client for Voyage AI integration |
| **Async HTTP** | aiohttp | 3.8+ | Proxy support, connection pooling, performance |
| **CLI Framework** | Click | 8.1+ | Mature, extensive features, rich integration |
| **Vector Database** | Qdrant | 1.8.0+ | Local deployment, HNSW indexing, performance |
| **Embeddings** | Voyage AI | voyage-code-3 | Code-optimized, 2048-dimensional vectors |
| **Text Processing** | tiktoken | 0.5+ | Accurate tokenization for chunking |
| **JSON Processing** | orjson | 3.9+ | Fastest JSON parsing performance |
| **Progress Display** | rich | 13.0+ | Beautiful CLI progress bars |
| **Configuration** | PyYAML | 6.0+ | Human-readable config management |
| **Hashing** | xxhash | 3.3+ | Fast deduplication hashing |
| **Compression** | gzip | Built-in | Standard compression for storage |

### External Services
- **BrightData**: Residential proxy service for rate limit bypass
- **Context7 API**: Documentation aggregation service
- **Voyage AI**: Code-optimized embedding generation service

### Infrastructure Requirements
- **Minimum Hardware**: 8 CPU cores, 32GB RAM, 500GB SSD
- **Recommended Hardware**: 16 CPU cores, 64GB RAM, 1TB NVMe SSD
- **Network**: 100Mbps+ bandwidth for concurrent downloads
- **Operating System**: Ubuntu 22.04 LTS or macOS 13+

## 3. Component Breakdown

### 3.1 Proxy Manager Module
**Task ID**: PROXY-001  
**Estimated Time**: 12 hours  
**Dependencies**: None  
**Priority**: Critical Path

**Acceptance Criteria**:
- [ ] Rotating proxy pool with health checks
- [ ] Circuit breaker pattern for failed proxies
- [ ] Connection pooling with configurable limits
- [ ] Automatic failover within 5 seconds
- [ ] Rate limiting compliance with BrightData

**Implementation Pattern**:
```python
import asyncio
import aiohttp
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from enum import Enum

class ProxyStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"

class AsyncProxyManager:
    def __init__(self, proxy_configs: List[Dict], health_check_interval: int = 300):
        self.proxy_pool = ProxyPool(proxy_configs)
        self.circuit_breaker = CircuitBreaker()
        self.health_monitor = HealthMonitor(health_check_interval)
        self._lock = asyncio.Lock()
        
    async def get_session(self, timeout: int = 30) -> aiohttp.ClientSession:
        """Get HTTP session with healthy proxy."""
        async with self._lock:
            proxy = await self.proxy_pool.get_healthy_proxy()
            if not proxy:
                raise ProxyUnavailableError("No healthy proxies available")
            
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            return aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=timeout),
                headers={"User-Agent": self._get_rotating_user_agent()},
                connector_limit=10,
                trust_env=True
            )

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError()
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

class ProxyPool:
    def __init__(self, proxy_configs: List[Dict]):
        self.proxies = [ProxyNode(config) for config in proxy_configs]
        self.current_index = 0
        
    async def get_healthy_proxy(self) -> Optional[ProxyNode]:
        """Get next healthy proxy with round-robin selection."""
        attempts = len(self.proxies)
        
        while attempts > 0:
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            
            if proxy.is_healthy():
                return proxy
                
            attempts -= 1
            
        return None
```

**Risk Mitigation**:
- Implement exponential backoff for failed connections
- Monitor proxy costs and usage limits
- Fallback to direct connections if all proxies fail

### 3.2 Download Engine Module
**Task ID**: DOWNLOAD-001  
**Estimated Time**: 16 hours  
**Dependencies**: PROXY-001  
**Priority**: Critical Path

**Acceptance Criteria**:
- [ ] Async request processing with semaphore-based rate limiting
- [ ] Multi-context query generation (5+ contexts per library)
- [ ] Intelligent retry logic with exponential backoff
- [ ] Memory-efficient streaming for large responses
- [ ] Integration with auto-ingestion pipeline

**Implementation Pattern**:
```python
import asyncio
from typing import List, Dict, Any, AsyncGenerator
import aiohttp
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class DownloadContext:
    library_id: str
    query_context: str
    priority: int
    metadata: Dict[str, Any]

class AsyncDownloadEngine:
    def __init__(
        self,
        proxy_manager: AsyncProxyManager,
        context7_client: Context7Client,
        max_concurrent: int = 10,
        request_timeout: int = 30
    ):
        self.proxy_manager = proxy_manager
        self.context7_client = context7_client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_timeout = request_timeout
        self.ingestion_pipeline = None  # Set during initialization
        
    async def download_library(
        self,
        library_id: str,
        contexts: List[str] = None,
        auto_ingest: bool = True
    ) -> Dict[str, Any]:
        """Download library documentation with multiple contexts."""
        
        # Generate contexts if not provided
        if not contexts:
            contexts = await self._generate_smart_contexts(library_id)
        
        # Create download contexts
        download_contexts = [
            DownloadContext(
                library_id=library_id,
                query_context=context,
                priority=i,
                metadata={"context_type": self._classify_context(context)}
            )
            for i, context in enumerate(contexts)
        ]
        
        # Process contexts concurrently
        results = []
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._download_context(ctx))
                for ctx in download_contexts
            ]
        
        # Collect results
        for task in tasks:
            if not task.exception():
                results.append(task.result())
        
        # Deduplicate and merge
        merged_content = await self._deduplicate_and_merge(results)
        
        # Store documentation
        doc_path = await self.storage_manager.store_documentation(
            library_id=library_id,
            content=merged_content,
            metadata={
                "contexts_used": len(contexts),
                "total_chunks": len(results),
                "download_timestamp": datetime.now().isoformat()
            }
        )
        
        # Trigger auto-ingestion if enabled
        if auto_ingest and self.ingestion_pipeline:
            await self.ingestion_pipeline.queue_document(library_id, doc_path)
        
        return {
            "library_id": library_id,
            "document_path": doc_path,
            "contexts_processed": len(contexts),
            "chunks_merged": len(results),
            "auto_ingestion_triggered": auto_ingest
        }
    
    async def _download_context(self, context: DownloadContext) -> Dict[str, Any]:
        """Download single context with retry logic."""
        
        async with self.semaphore:
            for attempt in range(3):
                try:
                    async with self.proxy_manager.get_session() as session:
                        response = await self.context7_client.fetch_documentation(
                            session=session,
                            library_id=context.library_id,
                            query_context=context.query_context,
                            timeout=self.request_timeout
                        )
                        
                        return {
                            "context": context.query_context,
                            "content": response,
                            "metadata": context.metadata,
                            "attempt": attempt + 1
                        }
                        
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt == 2:  # Last attempt
                        raise DownloadError(f"Failed to download {context.library_id} context {context.query_context}: {e}")
                    
                    # Exponential backoff with jitter
                    backoff = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(backoff)

    @asynccontextmanager
    async def batch_download(self, batch_size: int = 50):
        """Context manager for efficient batch downloads."""
        batch_semaphore = asyncio.Semaphore(batch_size)
        download_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "start_time": datetime.now()
        }
        
        async def process_batch_item(library_id: str, contexts: List[str]):
            async with batch_semaphore:
                try:
                    result = await self.download_library(library_id, contexts)
                    download_stats["successful"] += 1
                    return result
                except Exception as e:
                    download_stats["failed"] += 1
                    logger.error(f"Batch download failed for {library_id}: {e}")
                    return None
                finally:
                    download_stats["total_processed"] += 1
        
        try:
            yield process_batch_item
        finally:
            download_stats["end_time"] = datetime.now()
            duration = (download_stats["end_time"] - download_stats["start_time"]).total_seconds()
            
            logger.info(f"Batch download completed in {duration:.1f}s: "
                       f"{download_stats['successful']} successful, "
                       f"{download_stats['failed']} failed")
```

### 3.3 Deduplication Engine Module
**Task ID**: DEDUPE-001  
**Estimated Time**: 12 hours  
**Dependencies**: DOWNLOAD-001  
**Priority**: High

**Acceptance Criteria**:
- [ ] Hash-based content deduplication (>99% accuracy)
- [ ] Semantic similarity detection for near-duplicates
- [ ] Intelligent content merging with section preservation
- [ ] Performance target: 100 chunks processed in <5 seconds
- [ ] Memory-efficient processing for large document sets

**Implementation Pattern**:
```python
import xxhash
import asyncio
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import numpy as np

@dataclass
class ContentChunk:
    content: str
    content_hash: str
    source_context: str
    metadata: Dict[str, Any]
    similarity_hash: str = None

class AsyncDeduplicationEngine:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        hash_algorithm: str = "xxhash64"
    ):
        self.similarity_threshold = similarity_threshold
        self.hash_algorithm = hash_algorithm
        self.seen_hashes: Set[str] = set()
        self.content_index: Dict[str, ContentChunk] = {}
        
    async def deduplicate_and_merge(
        self,
        content_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Deduplicate content and merge intelligently."""
        
        # Convert to ContentChunk objects
        chunks = [
            ContentChunk(
                content=chunk["content"],
                content_hash=self._generate_hash(chunk["content"]),
                source_context=chunk.get("context", "unknown"),
                metadata=chunk.get("metadata", {}),
                similarity_hash=self._generate_similarity_hash(chunk["content"])
            )
            for chunk in content_chunks
        ]
        
        # Phase 1: Exact duplicate removal
        exact_deduped = await self._remove_exact_duplicates(chunks)
        
        # Phase 2: Near-duplicate detection and merging
        merged_chunks = await self._merge_similar_content(exact_deduped)
        
        # Phase 3: Intelligent section organization
        organized_content = await self._organize_content_structure(merged_chunks)
        
        return {
            "merged_content": organized_content,
            "deduplication_stats": {
                "original_chunks": len(chunks),
                "after_exact_dedup": len(exact_deduped),
                "after_similarity_merge": len(merged_chunks),
                "final_sections": len(organized_content.get("sections", {})),
                "deduplication_ratio": 1 - (len(merged_chunks) / len(chunks))
            }
        }
    
    def _generate_hash(self, content: str) -> str:
        """Generate fast hash for content."""
        if self.hash_algorithm == "xxhash64":
            return xxhash.xxh64(content.encode()).hexdigest()
        else:
            return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_similarity_hash(self, content: str) -> str:
        """Generate hash for similarity detection (normalized content)."""
        # Normalize content for similarity detection
        normalized = " ".join(content.lower().split())
        # Remove common programming artifacts that don't affect meaning
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return xxhash.xxh64(normalized.encode()).hexdigest()
    
    async def _remove_exact_duplicates(
        self,
        chunks: List[ContentChunk]
    ) -> List[ContentChunk]:
        """Remove exact content duplicates."""
        
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            if chunk.content_hash not in seen_hashes:
                seen_hashes.add(chunk.content_hash)
                unique_chunks.append(chunk)
                
        return unique_chunks
    
    async def _merge_similar_content(
        self,
        chunks: List[ContentChunk]
    ) -> List[ContentChunk]:
        """Merge similar content chunks."""
        
        merged_chunks = []
        similarity_groups: Dict[str, List[ContentChunk]] = {}
        
        # Group by similarity hash
        for chunk in chunks:
            if chunk.similarity_hash not in similarity_groups:
                similarity_groups[chunk.similarity_hash] = []
            similarity_groups[chunk.similarity_hash].append(chunk)
        
        # Merge groups with similar content
        for group in similarity_groups.values():
            if len(group) == 1:
                merged_chunks.append(group[0])
            else:
                # Merge multiple similar chunks
                merged_chunk = await self._merge_chunk_group(group)
                merged_chunks.append(merged_chunk)
        
        return merged_chunks
    
    async def _merge_chunk_group(
        self,
        chunks: List[ContentChunk]
    ) -> ContentChunk:
        """Merge a group of similar chunks into one."""
        
        # Find the most comprehensive chunk as base
        base_chunk = max(chunks, key=lambda c: len(c.content))
        
        # Collect additional information from other chunks
        additional_contexts = []
        combined_metadata = base_chunk.metadata.copy()
        
        for chunk in chunks:
            if chunk != base_chunk:
                additional_contexts.append(chunk.source_context)
                # Merge metadata
                for key, value in chunk.metadata.items():
                    if key not in combined_metadata:
                        combined_metadata[key] = value
                    elif isinstance(value, list):
                        if isinstance(combined_metadata[key], list):
                            combined_metadata[key].extend(value)
                        else:
                            combined_metadata[key] = [combined_metadata[key]] + value
        
        return ContentChunk(
            content=base_chunk.content,
            content_hash=base_chunk.content_hash,
            source_context=base_chunk.source_context,
            metadata={
                **combined_metadata,
                "merged_from_contexts": additional_contexts,
                "merge_count": len(chunks)
            },
            similarity_hash=base_chunk.similarity_hash
        )
```

### 3.4 RAG Ingestion Pipeline Module
**Task ID**: RAG-INGEST-001  
**Estimated Time**: 14 hours  
**Dependencies**: DEDUPE-001  
**Priority**: Critical Path

**Acceptance Criteria**:
- [ ] Automatic processing trigger after download completion
- [ ] JSON document parsing and intelligent chunking (1000 tokens, 200 overlap)
- [ ] Metadata extraction and enrichment
- [ ] Queue management for batch processing
- [ ] Error recovery and retry mechanisms

**Implementation Pattern**:
```python
import asyncio
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from datetime import datetime
import tiktoken
import json

@dataclass
class IngestionJob:
    library_id: str
    document_path: Path
    priority: int
    created_at: datetime
    metadata: Dict[str, Any]
    processing_attempts: int = 0

class AutoIngestionPipeline:
    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_concurrent: int = 5
    ):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_concurrent = max_concurrent
        
        # Queue management
        self.ingestion_queue = asyncio.Queue()
        self.processing_semaphore = asyncio.Semaphore(max_concurrent)
        self.worker_tasks = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_chunks": 0,
            "total_embeddings": 0
        }
        
        # Tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    async def start_workers(self, num_workers: int = 3):
        """Start background worker tasks."""
        self.is_running = True
        
        for i in range(num_workers):
            worker_task = asyncio.create_task(
                self._worker(worker_id=i),
                name=f"ingestion-worker-{i}"
            )
            self.worker_tasks.append(worker_task)
        
        logger.info(f"Started {num_workers} ingestion workers")
    
    async def queue_document(
        self,
        library_id: str,
        document_path: Path,
        priority: int = 0,
        metadata: Dict[str, Any] = None
    ):
        """Queue document for RAG ingestion."""
        
        job = IngestionJob(
            library_id=library_id,
            document_path=document_path,
            priority=priority,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        await self.ingestion_queue.put(job)
        self.stats["total_jobs"] += 1
        
        logger.info(f"Queued document for ingestion: {library_id} from {document_path}")
    
    async def _worker(self, worker_id: int):
        """Background worker for processing ingestion jobs."""
        
        logger.info(f"Ingestion worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                job = await asyncio.wait_for(
                    self.ingestion_queue.get(),
                    timeout=5.0
                )
                
                async with self.processing_semaphore:
                    await self._process_ingestion_job(job, worker_id)
                    
            except asyncio.TimeoutError:
                # No jobs available, continue polling
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
    
    async def _process_ingestion_job(self, job: IngestionJob, worker_id: int):
        """Process single ingestion job."""
        
        logger.info(f"Worker {worker_id} processing: {job.library_id}")
        
        try:
            # Parse JSON document
            documents = await self._parse_json_document(job.document_path)
            
            # Chunk documents
            all_chunks = []
            for doc in documents:
                chunks = await self._chunk_document(
                    content=doc["content"],
                    metadata={**doc.get("metadata", {}), **job.metadata}
                )
                all_chunks.extend(chunks)
            
            self.stats["total_chunks"] += len(all_chunks)
            
            # Generate embeddings in batches
            embedding_batches = self._create_embedding_batches(all_chunks, batch_size=100)
            
            for batch in embedding_batches:
                embeddings = await self.embedding_engine.generate_batch_embeddings(
                    texts=[chunk["content"] for chunk in batch],
                    metadata=[chunk["metadata"] for chunk in batch]
                )
                
                # Store in vector database
                await self.vector_store.upsert_embeddings(embeddings)
                
                self.stats["total_embeddings"] += len(embeddings)
            
            # Mark job as completed
            self.stats["completed_jobs"] += 1
            logger.info(f"Completed ingestion: {job.library_id} ({len(all_chunks)} chunks)")
            
        except Exception as e:
            job.processing_attempts += 1
            
            if job.processing_attempts < 3:
                # Retry job
                await asyncio.sleep(2 ** job.processing_attempts)  # Exponential backoff
                await self.ingestion_queue.put(job)
                logger.warning(f"Retrying ingestion job {job.library_id} (attempt {job.processing_attempts})")
            else:
                # Mark as failed
                self.stats["failed_jobs"] += 1
                logger.error(f"Failed ingestion job {job.library_id} after 3 attempts: {e}")
    
    async def _parse_json_document(self, document_path: Path) -> List[Dict[str, Any]]:
        """Parse JSON document and extract content sections."""
        
        with open(document_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Extract library metadata
        library_metadata = data.get("metadata", {})
        
        # Parse content sections
        for section_name, section_content in data.items():
            if section_name == "metadata":
                continue
            
            if isinstance(section_content, str):
                documents.append({
                    "content": section_content,
                    "metadata": {
                        **library_metadata,
                        "section": section_name,
                        "doc_type": self._infer_doc_type(section_name)
                    }
                })
            elif isinstance(section_content, dict):
                for subsection_name, subsection_content in section_content.items():
                    if isinstance(subsection_content, str):
                        documents.append({
                            "content": subsection_content,
                            "metadata": {
                                **library_metadata,
                                "section": section_name,
                                "subsection": subsection_name,
                                "doc_type": self._infer_doc_type(f"{section_name}.{subsection_name}")
                            }
                        })
        
        return documents
    
    async def _chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Intelligently chunk document content."""
        
        # Tokenize content
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= self.chunk_size:
            # Single chunk
            return [{
                "content": content,
                "metadata": {**metadata, "chunk_index": 0, "total_chunks": 1},
                "token_count": len(tokens)
            }]
        
        # Multi-chunk processing with semantic boundaries
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(tokens):
            # Determine chunk end position
            chunk_end = min(current_pos + self.chunk_size, len(tokens))
            
            # Find semantic boundary (avoid splitting mid-sentence)
            if chunk_end < len(tokens):
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                # Look for sentence or paragraph boundaries
                boundary_markers = ['. ', '.\n', '\n\n', '```\n']
                best_boundary = -1
                
                for marker in boundary_markers:
                    boundary = chunk_text.rfind(marker)
                    if boundary > len(chunk_text) * 0.7:  # At least 70% of chunk
                        best_boundary = boundary + len(marker)
                        break
                
                if best_boundary > 0:
                    chunk_text = chunk_text[:best_boundary]
                    chunk_tokens = self.tokenizer.encode(chunk_text)
                    chunk_end = current_pos + len(chunk_tokens)
            else:
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Add overlap for continuity
            overlap_start = max(0, current_pos - self.chunk_overlap) if chunk_index > 0 else current_pos
            overlap_tokens = tokens[overlap_start:chunk_end]
            overlap_text = self.tokenizer.decode(overlap_tokens)
            
            chunks.append({
                "content": overlap_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_index,
                    "total_chunks": -1,  # Will be updated after all chunks created
                    "chunk_start_token": overlap_start,
                    "chunk_end_token": chunk_end,
                    "has_overlap": chunk_index > 0
                },
                "token_count": len(overlap_tokens)
            })
            
            current_pos = chunk_end - self.chunk_overlap
            chunk_index += 1
        
        # Update total chunks count
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
```

### 3.5 Vector Storage Module (Qdrant Integration)
**Task ID**: VECTOR-001  
**Estimated Time**: 10 hours  
**Dependencies**: RAG-INGEST-001  
**Priority**: Critical Path

**Acceptance Criteria**:
- [ ] Qdrant collection initialization with HNSW indexing
- [ ] Batch vector upload with error handling
- [ ] Payload indexing for efficient filtering
- [ ] Vector search with sub-50ms latency
- [ ] Collection optimization and maintenance

**Implementation Pattern**:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest,
    OptimizersConfigDiff, HnswConfigDiff,
    PayloadSchemaType
)
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import asyncio
from datetime import datetime

class QdrantVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "contexter_docs",
        vector_size: int = 2048,
        prefer_grpc: bool = True
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize client with optimized settings
        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=6334 if prefer_grpc else None,
            prefer_grpc=prefer_grpc,
            timeout=30
        )
        
        # HNSW configuration for optimal performance
        self.hnsw_config = HnswConfigDiff(
            m=16,                    # Number of bi-directional links
            ef_construct=200,        # Size of dynamic candidate list during construction
            full_scan_threshold=10000,
            max_indexing_threads=0   # Use all available threads
        )
        
        # Optimizer configuration for maintenance
        self.optimizer_config = OptimizersConfigDiff(
            deleted_threshold=0.2,
            vacuum_min_vector_number=1000,
            default_segment_number=5,
            max_segment_size=200000,
            memmap_threshold=50000,
            indexing_threshold=20000,
            flush_interval_sec=5
        )
    
    async def initialize_collection(self, recreate: bool = False):
        """Initialize or recreate vector collection."""
        
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            # Create collection with optimized configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                hnsw_config=self.hnsw_config,
                optimizers_config=self.optimizer_config,
                shard_number=2,
                replication_factor=1
            )
            
            # Create payload indexes for efficient filtering
            await self._create_payload_indexes()
            
            logger.info(f"Created collection '{self.collection_name}' with HNSW indexing")
    
    async def _create_payload_indexes(self):
        """Create indexes on frequently queried payload fields."""
        
        index_fields = [
            ("library_id", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("chunk_index", PayloadSchemaType.INTEGER),
            ("timestamp", PayloadSchemaType.DATETIME),
            ("trust_score", PayloadSchemaType.FLOAT)
        ]
        
        for field_name, field_type in index_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            logger.info(f"Created payload index for field: {field_name}")
    
    async def upsert_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> List[str]:
        """Upsert embeddings in batches with error handling."""
        
        inserted_ids = []
        
        # Process in batches for optimal performance
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            points = []
            for embedding_data in batch:
                # Generate UUID if not provided
                point_id = embedding_data.get("id", str(uuid4()))
                
                # Ensure vector is correct format
                vector = embedding_data["vector"]
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                
                # Add timestamp to payload
                payload = embedding_data.get("payload", {})
                payload["indexed_at"] = datetime.now().isoformat()
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # Upsert batch with wait for completion
            try:
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                
                if operation_info.status.name == "COMPLETED":
                    batch_ids = [p.id for p in points]
                    inserted_ids.extend(batch_ids)
                    logger.debug(f"Upserted batch of {len(batch)} vectors")
                else:
                    logger.error(f"Batch upsert failed with status: {operation_info.status}")
                    
            except Exception as e:
                logger.error(f"Error upserting batch: {e}")
                # Continue with other batches rather than failing completely
                continue
        
        logger.info(f"Successfully upserted {len(inserted_ids)} vectors")
        return inserted_ids
    
    async def search_vectors(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search with filtering."""
        
        # Convert numpy array to list
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Build filter conditions
        filter_obj = None
        if filters:
            filter_obj = self._build_filter_conditions(filters)
        
        # Set search parameters
        params = search_params or {}
        ef = params.get("ef", 100)  # Search precision parameter
        
        try:
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_obj,
                with_payload=True,
                with_vectors=False,  # Don't return vectors to save bandwidth
                search_params={"ef": ef}
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary conditions."""
        
        must_conditions = []
        should_conditions = []
        
        for field, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use should (OR)
                for v in value:
                    should_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=v))
                    )
            elif isinstance(value, dict) and "range" in value:
                # Range filter
                range_filter = value["range"]
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        range=range_filter
                    )
                )
            else:
                # Single value - use must (AND)
                must_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )
        
        filter_conditions = {}
        if must_conditions:
            filter_conditions["must"] = must_conditions
        if should_conditions:
            filter_conditions["should"] = should_conditions
        
        return Filter(**filter_conditions) if filter_conditions else None
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection statistics."""
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "status": collection_info.status,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "payload_schema": collection_info.payload_schema,
                "optimizer_status": collection_info.optimizer_status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    async def optimize_collection(self):
        """Trigger collection optimization for better performance."""
        
        try:
            # Update collection configuration to trigger optimization
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizers_config=self.optimizer_config
            )
            
            logger.info(f"Triggered optimization for collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
```

### 3.6 Hybrid Search Engine Module
**Task ID**: SEARCH-001  
**Estimated Time**: 10 hours  
**Dependencies**: VECTOR-001  
**Priority**: High

**Acceptance Criteria**:
- [ ] Semantic vector search with Voyage AI query embeddings
- [ ] Keyword-based search with BM25-like scoring
- [ ] Hybrid fusion algorithm (70% semantic, 30% keyword)
- [ ] Result reranking with quality signals
- [ ] Sub-50ms p95 search latency

**Implementation Pattern**:
```python
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
import re
from collections import Counter

@dataclass
class SearchResult:
    id: str
    content: str
    score: float
    semantic_score: float
    keyword_score: float
    metadata: Dict[str, Any]
    highlights: List[str] = None

class HybridSearchEngine:
    def __init__(
        self,
        embedding_client: VoyageAIClient,
        vector_store: QdrantVectorStore,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank_enabled: bool = True
    ):
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rerank_enabled = rerank_enabled
        
        # Query cache for performance
        self.query_embedding_cache: Dict[str, Tuple[np.ndarray, datetime]] = {}
        self.cache_ttl_seconds = 3600  # 1 hour
        
        # Search statistics
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_semantic_latency_ms": 0,
            "avg_total_latency_ms": 0
        }
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid",  # "hybrid", "semantic", "keyword"
        score_threshold: float = 0.1,
        highlight_terms: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword approaches."""
        
        start_time = datetime.now()
        self.search_stats["total_searches"] += 1
        
        try:
            if search_type == "semantic":
                results = await self._semantic_search(query, limit, filters, score_threshold)
            elif search_type == "keyword":
                results = await self._keyword_search(query, limit, filters)
            else:  # hybrid
                results = await self._hybrid_search(query, limit, filters, score_threshold)
            
            # Apply result reranking
            if self.rerank_enabled and results:
                results = await self._rerank_results(query, results)
            
            # Add highlights if requested
            if highlight_terms and results:
                results = await self._add_highlights(query, results)
            
            # Update statistics
            total_latency = (datetime.now() - start_time).total_seconds() * 1000
            self._update_search_stats(total_latency)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []
    
    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        score_threshold: float
    ) -> List[SearchResult]:
        """Combine semantic and keyword search results."""
        
        # Get more results than needed for better hybrid ranking
        expanded_limit = min(limit * 3, 100)
        
        # Perform semantic search
        semantic_results = await self._semantic_search(
            query, expanded_limit, filters, score_threshold * 0.5
        )
        
        # Perform keyword search on semantic results
        keyword_scores = await self._calculate_keyword_scores(query, semantic_results)
        
        # Combine scores using weighted average
        hybrid_results = []
        for result in semantic_results:
            keyword_score = keyword_scores.get(result.id, 0.0)
            
            # Calculate hybrid score
            hybrid_score = (
                result.semantic_score * self.semantic_weight +
                keyword_score * self.keyword_weight
            )
            
            hybrid_result = SearchResult(
                id=result.id,
                content=result.content,
                score=hybrid_score,
                semantic_score=result.semantic_score,
                keyword_score=keyword_score,
                metadata=result.metadata
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        return hybrid_results
    
    async def _semantic_search(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        score_threshold: float
    ) -> List[SearchResult]:
        """Perform semantic vector similarity search."""
        
        semantic_start = datetime.now()
        
        # Get query embedding (with caching)
        query_embedding = await self._get_query_embedding(query)
        
        # Search in vector store
        search_results = await self.vector_store.search_vectors(
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters,
            search_params={"ef": 128}  # Higher precision for better results
        )
        
        # Convert to SearchResult objects
        semantic_results = []
        for result in search_results:
            semantic_results.append(SearchResult(
                id=result["id"],
                content=result["payload"].get("content", ""),
                score=result["score"],
                semantic_score=result["score"],
                keyword_score=0.0,
                metadata=result["payload"]
            ))
        
        # Update semantic search latency
        semantic_latency = (datetime.now() - semantic_start).total_seconds() * 1000
        self._update_semantic_latency(semantic_latency)
        
        return semantic_results
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with caching."""
        
        # Check cache first
        if query in self.query_embedding_cache:
            embedding, timestamp = self.query_embedding_cache[query]
            age_seconds = (datetime.now() - timestamp).total_seconds()
            
            if age_seconds < self.cache_ttl_seconds:
                self.search_stats["cache_hits"] += 1
                return embedding
        
        # Generate new embedding
        embedding_result = await self.embedding_client.embed_single(
            text=query,
            input_type="query"  # Use query mode for search queries
        )
        
        embedding = np.array(embedding_result.embedding, dtype=np.float32)
        
        # Cache the embedding
        self.query_embedding_cache[query] = (embedding, datetime.now())
        
        # Clean old cache entries periodically
        if len(self.query_embedding_cache) > 1000:
            await self._cleanup_embedding_cache()
        
        return embedding
    
    async def _calculate_keyword_scores(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Dict[str, float]:
        """Calculate keyword-based scores for results."""
        
        # Tokenize query
        query_terms = self._tokenize_text(query.lower())
        query_term_counts = Counter(query_terms)
        
        keyword_scores = {}
        
        for result in results:
            # Get searchable text (content + metadata)
            searchable_text = result.content
            if result.metadata.get("section"):
                searchable_text += " " + result.metadata["section"]
            if result.metadata.get("doc_type"):
                searchable_text += " " + result.metadata["doc_type"]
            
            # Tokenize document text
            doc_terms = self._tokenize_text(searchable_text.lower())
            doc_term_counts = Counter(doc_terms)
            
            # Calculate BM25-like score
            score = self._calculate_bm25_score(
                query_term_counts,
                doc_term_counts,
                len(doc_terms)
            )
            
            keyword_scores[result.id] = score
        
        return keyword_scores
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for keyword matching."""
        # Remove special characters and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if len(token) > 2]
    
    def _calculate_bm25_score(
        self,
        query_terms: Counter,
        doc_terms: Counter,
        doc_length: int,
        k1: float = 1.2,
        b: float = 0.75,
        avg_doc_length: float = 1000
    ) -> float:
        """Calculate BM25-like score for keyword matching."""
        
        score = 0.0
        
        for term, query_freq in query_terms.items():
            if term in doc_terms:
                term_freq = doc_terms[term]
                
                # BM25 formula components
                idf = math.log((1000 + 1) / (1 + 1))  # Simplified IDF
                
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                term_score = idf * (numerator / denominator)
                score += term_score
        
        return score
    
    async def _rerank_results(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Apply result reranking based on quality signals."""
        
        for result in results:
            # Apply quality-based boosts
            quality_boost = self._calculate_quality_boost(result.metadata, query)
            result.score *= quality_boost
        
        # Resort by adjusted scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _calculate_quality_boost(
        self,
        metadata: Dict[str, Any],
        query: str
    ) -> float:
        """Calculate quality boost factor based on metadata signals."""
        
        boost = 1.0
        
        # Trust score boost
        trust_score = metadata.get("trust_score", 0)
        if trust_score > 8:
            boost *= 1.3
        elif trust_score > 6:
            boost *= 1.15
        elif trust_score > 4:
            boost *= 1.05
        
        # Document type relevance
        doc_type = metadata.get("doc_type", "").lower()
        query_lower = query.lower()
        
        if "api" in query_lower and doc_type == "api":
            boost *= 1.2
        elif "guide" in query_lower and doc_type == "guide":
            boost *= 1.2
        elif "tutorial" in query_lower and doc_type == "tutorial":
            boost *= 1.2
        
        # Section relevance
        section = metadata.get("section", "").lower()
        if any(term in section for term in query_lower.split()):
            boost *= 1.1
        
        # Recency boost (newer content slightly preferred)
        timestamp = metadata.get("timestamp")
        if timestamp:
            try:
                doc_date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_days = (datetime.now() - doc_date.replace(tzinfo=None)).days
                if age_days < 30:
                    boost *= 1.05
            except:
                pass
        
        return boost
    
    async def _add_highlights(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """Add highlighted snippets to search results."""
        
        query_terms = set(self._tokenize_text(query.lower()))
        
        for result in results:
            highlights = []
            content_lower = result.content.lower()
            
            # Find sentences containing query terms
            sentences = result.content.split('. ')
            for sentence in sentences[:3]:  # Max 3 highlights
                sentence_terms = set(self._tokenize_text(sentence.lower()))
                if query_terms & sentence_terms:  # Intersection
                    # Highlight matching terms
                    highlighted = sentence
                    for term in query_terms:
                        if term in sentence.lower():
                            highlighted = re.sub(
                                f'\\b{re.escape(term)}\\b',
                                f'**{term}**',
                                highlighted,
                                flags=re.IGNORECASE
                            )
                    highlights.append(highlighted)
            
            result.highlights = highlights
        
        return results
    
    def _update_search_stats(self, total_latency_ms: float):
        """Update search performance statistics."""
        current_count = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_total_latency_ms"]
        
        # Calculate new running average
        self.search_stats["avg_total_latency_ms"] = (
            (current_avg * (current_count - 1) + total_latency_ms) / current_count
        )
    
    def _update_semantic_latency(self, semantic_latency_ms: float):
        """Update semantic search latency statistics."""
        current_count = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_semantic_latency_ms"]
        
        self.search_stats["avg_semantic_latency_ms"] = (
            (current_avg * (current_count - 1) + semantic_latency_ms) / current_count
        )
    
    async def _cleanup_embedding_cache(self):
        """Remove old entries from embedding cache."""
        cutoff_time = datetime.now() - timedelta(seconds=self.cache_ttl_seconds)
        
        expired_keys = [
            key for key, (_, timestamp) in self.query_embedding_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self.query_embedding_cache[key]
        
        logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get current search performance statistics."""
        total_searches = self.search_stats["total_searches"]
        cache_rate = (self.search_stats["cache_hits"] / total_searches * 100) if total_searches > 0 else 0
        
        return {
            **self.search_stats,
            "cache_hit_rate_percent": cache_rate,
            "cached_queries": len(self.query_embedding_cache)
        }
```

## 4. Integration Points

### 4.1 C7DocDownloader to RAG Pipeline Integration
The auto-ingestion pattern creates seamless flow from download completion to searchable vectors:

1. **Download Completion Trigger**: When `AsyncDownloadEngine.download_library()` completes successfully, it triggers `AutoIngestionPipeline.queue_document()`
2. **Quality Validation**: Documents undergo completeness validation before RAG processing
3. **Automatic Processing**: Background workers process the ingestion queue continuously
4. **Error Recovery**: Failed ingestion jobs retry with exponential backoff
5. **Status Tracking**: Both systems maintain shared status tracking for end-to-end visibility

### 4.2 Unified Configuration Management
Single YAML configuration file manages both C7DocDownloader and RAG system settings:

```yaml
# Shared proxy configuration
proxy:
  provider: brightdata
  rotation_enabled: true
  health_check_interval: 300

# Download engine settings  
downloader:
  max_concurrent_requests: 10
  request_timeout: 30
  auto_ingest: true

# RAG system settings
rag:
  auto_process: true
  embedding_cache_enabled: true
  search_hybrid_enabled: true
```

### 4.3 Error Handling Integration
Comprehensive error classification covers the entire pipeline:

```python
class ErrorCategory(Enum):
    # Download errors
    NETWORK = "network"
    PROXY = "proxy"
    API = "api"
    
    # RAG errors
    INGESTION = "ingestion"
    EMBEDDING = "embedding"
    SEARCH = "search"
    VECTOR_DATABASE = "vector_database"

class IntegratedErrorHandler:
    async def handle_pipeline_error(self, error: Exception, stage: str, context: Dict):
        error_category = self.classify_error(error, stage)
        recovery_strategy = self.get_recovery_strategy(error_category)
        
        if recovery_strategy.is_recoverable:
            await self.attempt_recovery(error, recovery_strategy, context)
        else:
            await self.escalate_error(error, context)
```

## 5. Performance Optimization

### 5.1 Async Concurrency Patterns
- **Semaphore-Based Rate Limiting**: Control concurrent operations to prevent resource exhaustion
- **Circuit Breaker Pattern**: Automatic failure detection and recovery for external services
- **Connection Pooling**: Reuse HTTP connections for better performance
- **Batch Processing**: Group operations for optimal throughput

### 5.2 Memory Management
- **Streaming Processing**: Handle large documents without loading everything into memory
- **Chunked Operations**: Process data in manageable chunks to prevent memory spikes
- **Cache Management**: LRU eviction policies for embedding and query caches
- **Resource Cleanup**: Proper async context manager usage for resource cleanup

### 5.3 Storage Optimization
- **HNSW Indexing**: Optimal vector index configuration for sub-50ms search
- **Payload Indexing**: Index frequently queried metadata fields
- **Compression**: gzip compression for document storage (60%+ size reduction)
- **Batch Upserts**: Minimize database round trips with batched operations

## 6. Risk Mitigation

### 6.1 Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|---------|------------|
| Voyage AI Rate Limiting | High | Medium | Intelligent batching, embedding cache, exponential backoff |
| Proxy Service Disruption | Medium | High | Multi-provider support, circuit breakers, health monitoring |
| Vector Database Performance | Medium | High | HNSW optimization, collection monitoring, horizontal scaling |
| Memory Pressure | Medium | Medium | Streaming processing, memory monitoring, resource limits |

### 6.2 Integration Risks
- **Complex State Management**: Use clear state machines and status tracking
- **Data Consistency**: Implement transaction-like patterns where possible
- **Error Propagation**: Comprehensive error handling with context preservation
- **Testing Complexity**: Extensive integration test suite with mocking

## 7. Implementation Timeline

### Sprint 1 (Weeks 1-2): Foundation Infrastructure
**Week 1: Core Components**
- PROXY-001: Proxy Manager Implementation (3 days)
- CONFIG-001: Configuration Manager Setup (1 day) 
- DOWNLOAD-001: Download Engine Foundation (4 days)

**Week 2: Storage & Integration**
- DEDUPE-001: Deduplication Engine (3 days)
- STORAGE-001: Storage Manager Implementation (2 days)
- Integration testing and debugging (2 days)

**Deliverables**: Working download pipeline with proxy rotation and basic deduplication

### Sprint 2 (Weeks 3-4): RAG System Implementation
**Week 3: RAG Pipeline**
- RAG-INGEST-001: Auto-ingestion Pipeline (3 days)
- VECTOR-001: Qdrant Vector Storage (2 days)
- EMBED-001: Voyage AI Integration (2 days)

**Week 4: Search & CLI**
- SEARCH-001: Hybrid Search Engine (3 days)
- CLI-001: Command Line Interface (2 days)
- End-to-end integration testing (2 days)

**Deliverables**: Complete RAG system with semantic search capabilities

### Sprint 3 (Weeks 5-6): Optimization & Production Readiness
**Week 5: Performance Optimization**
- Performance testing and optimization (3 days)
- Memory usage optimization (2 days)
- Search latency improvements (2 days)

**Week 6: Production Features**
- Monitoring and metrics integration (2 days)
- Error handling improvements (2 days)
- Documentation and deployment guides (3 days)

**Deliverables**: Production-ready system meeting all performance targets

## 8. Testing Strategy

### 8.1 Unit Testing
- **Coverage Target**: >90% code coverage
- **Framework**: pytest with async support
- **Mock Strategy**: Mock external services (Voyage AI, BrightData, Context7)
- **Performance Tests**: Memory usage and timing assertions

### 8.2 Integration Testing
- **End-to-End Workflows**: Complete download-to-search pipelines
- **Error Scenarios**: Network failures, service outages, rate limiting
- **Data Integrity**: Validate data consistency throughout pipeline
- **Performance Benchmarks**: Latency and throughput testing

### 8.3 Load Testing
- **Concurrent Users**: 100 simultaneous search queries
- **Bulk Ingestion**: 10,000 documents processed without degradation
- **Memory Stability**: Extended operation without memory leaks
- **Failure Recovery**: System recovery after component failures

## 9. Success Criteria

### 9.1 Functional Requirements
- [ ] Download 95% of requested libraries successfully
- [ ] Process >1000 documents per minute during ingestion
- [ ] Search accuracy >95% recall@10 for test queries
- [ ] Auto-ingestion triggers within 10 seconds of download completion
- [ ] CLI interface supports all major operations

### 9.2 Performance Requirements
- [ ] Search latency p95 <50ms, p99 <100ms
- [ ] Download completion 90% within 30 seconds
- [ ] Memory usage <8GB during normal operations
- [ ] System handles 100 concurrent users without degradation
- [ ] Vector database supports 10M+ vectors with sub-second queries

### 9.3 Quality Requirements
- [ ] 99.9% system availability during business hours
- [ ] Comprehensive error handling with actionable messages
- [ ] Complete API documentation with examples
- [ ] Monitoring and alerting for all critical components
- [ ] Automated deployment and rollback capabilities

---

**Implementation Plan Version**: 2.0 (Integrated Platform)  
**Last Updated**: 2025-01-12  
**Prepared By**: Development Team Lead  
**Review Date**: Sprint Planning Session  
**Approval Required**: System Architect, Product Owner

*This implementation plan provides a comprehensive roadmap for building the Contexter Documentation Platform, ensuring successful integration of both download and search capabilities while meeting all performance and quality requirements.*