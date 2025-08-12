# RAG System Gotchas and Mitigations

## Overview

This document identifies critical implementation challenges, common mistakes, and edge cases that could derail the RAG system execution based on analysis of the PRPs and technology-specific research. Each gotcha includes symptoms, root causes, mitigation strategies, and prevention measures.

## Critical Gotchas (Must Address)

### Vector Database (Qdrant) Gotchas

#### Gotcha: HNSW Memory Explosion with Large Collections
- **Symptoms**: RAM usage growing exponentially beyond expected limits, OOM kills, performance degradation
- **Root Cause**: HNSW index memory usage scales non-linearly with vector count, especially with high `m` parameter values
- **Affected Versions**: All Qdrant versions
- **Impact**: High - Can crash entire system
- **Mitigation**:
  ```python
  # Conservative HNSW configuration for large collections
  hnsw_config = HnswConfigDiff(
      m=8,  # Lower than default 16 for memory efficiency
      ef_construct=100,  # Reduced from 200
      full_scan_threshold=5000  # Reduced threshold
  )
  
  # Monitor memory usage and implement alerts
  def monitor_memory_usage():
      memory_mb = get_collection_memory_usage()
      if memory_mb > MEMORY_THRESHOLD:
          trigger_alert("HNSW memory usage exceeding threshold")
  ```
- **Prevention**: Start with conservative HNSW parameters, implement memory monitoring, use memory-efficient vector storage
- **Detection**: Monitor process memory usage, set alerts at 80% of available RAM

#### Gotcha: Concurrent Write Conflicts and Index Corruption
- **Symptoms**: Points randomly disappearing, inconsistent search results, index rebuild failures
- **Root Cause**: Multiple workers writing to collection simultaneously without proper coordination
- **Affected Versions**: All versions under high concurrency
- **Impact**: High - Data integrity issues
- **Mitigation**:
  ```python
  # Implement write coordination with semaphore
  write_semaphore = asyncio.Semaphore(1)  # Single writer at a time
  
  async def safe_upsert_batch(vectors):
      async with write_semaphore:
          # Use optimistic locking with retries
          for attempt in range(3):
              try:
                  result = await qdrant_client.upsert(
                      collection_name=collection_name,
                      points=vectors,
                      wait=True  # Ensure operation completes
                  )
                  return result
              except Exception as e:
                  if attempt == 2:
                      raise
                  await asyncio.sleep(2 ** attempt)
  ```
- **Prevention**: Single writer pattern, use WAL mode, implement proper error handling
- **Detection**: Monitor failed upsert operations, verify data consistency with periodic checks

#### Gotcha: Search Performance Degradation Under Load
- **Symptoms**: Search latency increasing from 50ms to >500ms under concurrent load
- **Root Cause**: Default `ef` parameter too low for concurrent queries, connection pool exhaustion
- **Affected Versions**: All versions
- **Impact**: Medium - User experience degradation
- **Mitigation**:
  ```python
  # Optimize search parameters for concurrent load
  search_params = {
      "ef": 256,  # Higher than default for better recall under load
      "timeout": 30  # Explicit timeout
  }
  
  # Implement connection pooling
  client = QdrantClient(
      host=host,
      port=port,
      grpc_port=6334,
      prefer_grpc=True,
      https=False,
      timeout=30
  )
  ```
- **Prevention**: Load test with realistic concurrency, tune `ef` parameter, implement result caching
- **Detection**: Monitor p95/p99 search latency, alert on degradation

### Embedding Service (Voyage AI) Gotchas

#### Gotcha: Rate Limit Race Conditions
- **Symptoms**: Sudden API errors after working fine, inconsistent rate limit enforcement
- **Root Cause**: Multiple workers hitting rate limits simultaneously, client-side token counting mismatches
- **Affected Versions**: All API versions
- **Impact**: High - Processing pipeline stalls
- **Mitigation**:
  ```python
  # Implement distributed rate limiting with Redis
  import aioredis
  
  class DistributedRateLimiter:
      def __init__(self, redis_client):
          self.redis = redis_client
          
      async def acquire_tokens(self, num_tokens: int, window_minutes: int = 1):
          key = f"voyage_tokens:{int(time.time() // (window_minutes * 60))}"
          
          # Use Redis pipeline for atomic operations
          async with self.redis.pipeline() as pipe:
              current_tokens = await pipe.get(key) or 0
              if int(current_tokens) + num_tokens > RATE_LIMIT_TPM:
                  raise RateLimitExceededError("Token limit exceeded")
              
              await pipe.incrby(key, num_tokens)
              await pipe.expire(key, window_minutes * 60)
              await pipe.execute()
  ```
- **Prevention**: Use server-side rate limiting, implement circuit breaker pattern, add jitter to requests
- **Detection**: Monitor API response codes, track 429 errors with alerts

#### Gotcha: Token Counting Mismatches Between Client and Server
- **Symptoms**: Unexpected billing charges, rate limit errors with seemingly valid requests
- **Root Cause**: Different tokenizers used client-side vs server-side, especially with newer models
- **Affected Versions**: Models released after June 2024
- **Impact**: Medium - Cost overruns and processing failures
- **Mitigation**:
  ```python
  # Always specify model parameter and use server-side token counting
  def count_tokens_safely(text: str, model: str = "voyage-code-3"):
      try:
          # Use Voyage's tokenization endpoint when available
          response = voyage_client.count_tokens(text=text, model=model)
          return response.total_tokens
      except Exception:
          # Fallback to conservative estimation
          return len(text) // 4  # Conservative estimate
  
  # Add safety margin for batch requests
  def calculate_batch_size(texts: List[str], model: str):
      total_tokens = sum(count_tokens_safely(text, model) for text in texts)
      # Add 10% safety margin
      return int(total_tokens * 1.1)
  ```
- **Prevention**: Always specify model parameter, use conservative token estimates, implement server-side validation
- **Detection**: Monitor token usage vs estimates, alert on significant discrepancies

#### Gotcha: Embedding Cache Invalidation Issues
- **Symptoms**: Inconsistent search results, cache hits returning wrong embeddings
- **Root Cause**: Hash collisions, cache corruption, model version changes not invalidating cache
- **Affected Versions**: All versions
- **Impact**: High - Incorrect search results
- **Mitigation**:
  ```python
  # Robust cache key generation with model version
  def generate_cache_key(content: str, model: str, input_type: str) -> str:
      content_hash = hashlib.sha256(content.encode()).hexdigest()
      # Include model version and input type in cache key
      cache_key = f"{model}:{input_type}:{content_hash}"
      
      # Add collision detection
      if cache_key in cache:
          cached_content = cache.get_original_content(cache_key)
          if cached_content != content:
              # Hash collision detected
              cache_key = f"{cache_key}:{uuid.uuid4().hex[:8]}"
      
      return cache_key
  
  # Implement cache validation
  async def validate_cache_entry(cache_key: str, original_content: str):
      cached_data = await cache.get(cache_key)
      if cached_data and cached_data.get('content_hash') != hashlib.sha256(original_content.encode()).hexdigest():
          await cache.delete(cache_key)
          raise CacheCorruptionError("Cache corruption detected")
  ```
- **Prevention**: Include model version in cache keys, implement cache validation, use checksums
- **Detection**: Periodic cache integrity checks, monitor cache hit accuracy

### Document Processing Gotchas

#### Gotcha: Memory Explosion with Large Document Processing
- **Symptoms**: Memory usage spiking to >8GB during ingestion, OOM kills
- **Root Cause**: Loading entire documents into memory for processing, inefficient chunking algorithms
- **Affected Versions**: All versions
- **Impact**: High - System crashes
- **Mitigation**:
  ```python
  # Implement streaming document processing
  async def process_document_streaming(doc_path: Path, max_memory_mb: int = 512):
      current_memory = 0
      chunks = []
      
      async with aiofiles.open(doc_path, 'r') as f:
          async for line in f:
              chunk_size = len(line.encode('utf-8'))
              
              if current_memory + chunk_size > max_memory_mb * 1024 * 1024:
                  # Process current batch and clear memory
                  await process_chunks_batch(chunks)
                  chunks.clear()
                  current_memory = 0
                  gc.collect()  # Force garbage collection
              
              chunks.append(line)
              current_memory += chunk_size
      
      # Process remaining chunks
      if chunks:
          await process_chunks_batch(chunks)
  ```
- **Prevention**: Stream processing, set memory limits, implement backpressure
- **Detection**: Monitor memory usage during ingestion, set memory alerts

#### Gotcha: JSON Schema Variability Breaking Parser
- **Symptoms**: Parser failing on certain libraries, inconsistent data extraction
- **Root Cause**: Assuming uniform JSON schema across all documentation sources
- **Affected Versions**: All versions
- **Impact**: Medium - Processing failures for specific libraries
- **Mitigation**:
  ```python
  # Robust adaptive JSON parsing
  class AdaptiveJSONParser:
      def __init__(self):
          self.schema_patterns = {
              'context7': self._parse_context7_schema,
              'standard': self._parse_standard_schema,
              'nested': self._parse_nested_schema
          }
      
      async def parse_document(self, doc_path: Path) -> List[Dict[str, Any]]:
          try:
              content = await self._load_json(doc_path)
              schema_type = self._detect_schema(content)
              
              parser = self.schema_patterns.get(schema_type, self._parse_fallback)
              return await parser(content)
              
          except json.JSONDecodeError as e:
              # Attempt JSON repair
              return await self._repair_and_parse(doc_path, e)
          except Exception as e:
              logger.error(f"Failed to parse {doc_path}: {e}")
              return []  # Graceful degradation
      
      def _detect_schema(self, content: Dict) -> str:
          # Schema detection logic based on key patterns
          if 'context7_metadata' in content:
              return 'context7'
          elif isinstance(content.get('documentation'), dict):
              return 'nested'
          return 'standard'
  ```
- **Prevention**: Test with diverse document formats, implement schema detection, graceful degradation
- **Detection**: Monitor parsing success rates by library, alert on failures

### Storage Layer Gotchas

#### Gotcha: SQLite WAL Mode Corruption Under High Concurrency
- **Symptoms**: Database corruption, "database is locked" errors, transaction failures
- **Root Cause**: Multiple processes accessing SQLite without proper WAL configuration
- **Affected Versions**: All SQLite versions under high load
- **Impact**: High - Data loss potential
- **Mitigation**:
  ```python
  # Proper SQLite configuration for production
  async def configure_sqlite_for_production(db_path: str):
      async with aiosqlite.connect(db_path) as db:
          # Enable WAL mode for better concurrency
          await db.execute("PRAGMA journal_mode=WAL")
          await db.execute("PRAGMA synchronous=NORMAL")
          await db.execute("PRAGMA cache_size=10000")
          await db.execute("PRAGMA temp_store=memory")
          
          # Set timeouts for busy database
          await db.execute("PRAGMA busy_timeout=30000")  # 30 seconds
          
          # Enable foreign key constraints
          await db.execute("PRAGMA foreign_keys=ON")
          
          await db.commit()
  
  # Implement connection pooling
  class SQLiteConnectionPool:
      def __init__(self, db_path: str, max_connections: int = 10):
          self.db_path = db_path
          self.pool = asyncio.Queue(maxsize=max_connections)
          self.total_connections = 0
          
      async def get_connection(self):
          if self.pool.empty() and self.total_connections < self.max_connections:
              conn = await aiosqlite.connect(self.db_path)
              await configure_sqlite_for_production(conn)
              self.total_connections += 1
              return conn
          return await self.pool.get()
  ```
- **Prevention**: Use WAL mode, implement connection pooling, set proper timeouts
- **Detection**: Monitor database lock errors, implement integrity checks

#### Gotcha: Compression Ratio Degradation with Mixed Content
- **Symptoms**: Compression ratios dropping from 60% to 20% for certain libraries
- **Root Cause**: Binary data or pre-compressed content being treated as text
- **Affected Versions**: All versions
- **Impact**: Low - Storage efficiency reduction
- **Mitigation**:
  ```python
  # Content-aware compression
  def choose_compression_strategy(content: Dict[str, Any]) -> str:
      # Detect content type
      content_str = json.dumps(content)
      
      # Check for binary data patterns
      if self._has_binary_patterns(content_str):
          return 'store'  # Don't compress binary data
      
      # Check existing compression
      if self._is_already_compressed(content_str):
          return 'store'  # Don't double-compress
      
      # Test compression ratio
      test_compressed = gzip.compress(content_str.encode(), compresslevel=1)
      ratio = len(test_compressed) / len(content_str.encode())
      
      if ratio > 0.8:  # Poor compression
          return 'store'
      else:
          return 'compress'
  ```
- **Prevention**: Content type detection, compression ratio testing, adaptive strategies
- **Detection**: Monitor compression ratios by library type, investigate outliers

### API Integration Gotchas

#### Gotcha: FastAPI Memory Leaks with Long-Running Requests
- **Symptoms**: Memory usage growing continuously, eventual OOM kills
- **Root Cause**: Unclosed database connections, global variable accumulation, asyncio task leaks
- **Affected Versions**: All FastAPI versions
- **Impact**: High - Service degradation
- **Mitigation**:
  ```python
  # Proper dependency management with cleanup
  async def get_db_session():
      session = None
      try:
          session = SessionLocal()
          yield session
      finally:
          if session:
              await session.close()
  
  # Request-scoped cleanup middleware
  @app.middleware("http")
  async def cleanup_middleware(request: Request, call_next):
      response = await call_next(request)
      
      # Force garbage collection after each request
      if hasattr(request.state, 'cleanup_tasks'):
          for task in request.state.cleanup_tasks:
              if not task.done():
                  task.cancel()
      
      gc.collect()
      return response
  
  # Monitor memory usage
  @app.get("/health/memory")
  async def memory_health():
      memory_info = psutil.Process().memory_info()
      if memory_info.rss > MEMORY_THRESHOLD:
          raise HTTPException(500, "Memory usage too high")
      return {"memory_mb": memory_info.rss / 1024 / 1024}
  ```
- **Prevention**: Use dependency injection with cleanup, implement request timeouts, monitor memory
- **Detection**: Memory monitoring, garbage collection tracking

#### Gotcha: Async/Await Mixing Blocking Operations
- **Symptoms**: API response times jumping from 50ms to 5+ seconds sporadically
- **Root Cause**: Blocking operations in async handlers, database connection blocking
- **Affected Versions**: All FastAPI versions
- **Impact**: High - User experience degradation
- **Mitigation**:
  ```python
  # Identify and fix blocking operations
  import asyncio
  from concurrent.futures import ThreadPoolExecutor
  
  # Bad - blocks event loop
  def bad_handler():
      time.sleep(1)  # Blocking!
      return {"status": "ok"}
  
  # Good - non-blocking
  async def good_handler():
      await asyncio.sleep(1)  # Non-blocking
      return {"status": "ok"}
  
  # For unavoidable blocking operations
  executor = ThreadPoolExecutor(max_workers=10)
  
  async def handle_blocking_operation():
      loop = asyncio.get_event_loop()
      result = await loop.run_in_executor(
          executor, blocking_function
      )
      return result
  ```
- **Prevention**: Use async/await consistently, offload blocking operations to thread pool
- **Detection**: Monitor event loop lag, profile endpoint response times

### Search Engine Gotchas

#### Gotcha: Hybrid Search Score Imbalance
- **Symptoms**: Search results heavily biased toward semantic or keyword results
- **Root Cause**: Score ranges from different search methods not normalized properly
- **Affected Versions**: All versions
- **Impact**: Medium - Search quality degradation
- **Mitigation**:
  ```python
  # Proper score normalization for hybrid search
  class HybridScoreNormalizer:
      def __init__(self):
          self.semantic_score_range = (0.0, 1.0)
          self.keyword_score_range = (0.0, 100.0)  # BM25 scores can be much higher
      
      def normalize_scores(self, semantic_results: List, keyword_results: List):
          # Normalize semantic scores (already 0-1)
          normalized_semantic = [(r, r['score']) for r in semantic_results]
          
          # Normalize keyword scores to 0-1 range
          if keyword_results:
              max_keyword_score = max(r['score'] for r in keyword_results)
              normalized_keyword = [
                  (r, r['score'] / max_keyword_score if max_keyword_score > 0 else 0)
                  for r in keyword_results
              ]
          else:
              normalized_keyword = []
          
          return normalized_semantic, normalized_keyword
      
      def fuse_scores(self, semantic_results, keyword_results, 
                     semantic_weight=0.7, keyword_weight=0.3):
          # Create unified result set with balanced scoring
          result_map = {}
          
          # Add semantic results
          for result, norm_score in semantic_results:
              result_id = result['id']
              result_map[result_id] = {
                  **result,
                  'semantic_score': norm_score,
                  'keyword_score': 0.0
              }
          
          # Add keyword results
          for result, norm_score in keyword_results:
              result_id = result['id']
              if result_id in result_map:
                  result_map[result_id]['keyword_score'] = norm_score
              else:
                  result_map[result_id] = {
                      **result,
                      'semantic_score': 0.0,
                      'keyword_score': norm_score
                  }
          
          # Calculate final scores
          final_results = []
          for result in result_map.values():
              final_score = (
                  result['semantic_score'] * semantic_weight +
                  result['keyword_score'] * keyword_weight
              )
              result['final_score'] = final_score
              final_results.append(result)
          
          return sorted(final_results, key=lambda x: x['final_score'], reverse=True)
  ```
- **Prevention**: Implement proper score normalization, A/B test fusion weights
- **Detection**: Monitor search result quality metrics, user engagement patterns

### Monitoring and Observability Gotchas

#### Gotcha: High-Cardinality Metrics Causing Memory Issues
- **Symptoms**: Prometheus memory usage growing unbounded, query timeouts
- **Root Cause**: Creating metrics with too many label combinations
- **Affected Versions**: All Prometheus versions
- **Impact**: Medium - Monitoring system failure
- **Mitigation**:
  ```python
  # Limit metric cardinality
  class SafeMetricsCollector:
      def __init__(self, max_labels_per_metric: int = 1000):
          self.max_labels_per_metric = max_labels_per_metric
          self.label_counts = defaultdict(int)
      
      def inc_counter_safe(self, metric_name: str, labels: Dict[str, str]):
          # Hash labels to create unique key
          label_key = tuple(sorted(labels.items()))
          metric_key = f"{metric_name}:{hash(label_key)}"
          
          if self.label_counts[metric_name] >= self.max_labels_per_metric:
              # Use generic label to avoid cardinality explosion
              labels = {"status": "other"}
          
          self.label_counts[metric_name] += 1
          
          # Record metric with safe labels
          getattr(metrics, metric_name).labels(**labels).inc()
  ```
- **Prevention**: Limit label cardinality, use sampling for high-volume metrics
- **Detection**: Monitor Prometheus memory usage, query performance

## Common Pitfalls

### Pitfall: Ignoring Async Context in Database Operations
- **Scenario**: Using synchronous database operations in async endpoints
- **Example of Wrong Approach**:
  ```python
  # Wrong - blocks event loop
  @app.get("/search")
  async def search_endpoint(query: str):
      results = db.execute("SELECT * FROM docs WHERE content LIKE %s", query)
      return results
  ```
- **Correct Approach**:
  ```python
  # Correct - uses async database operations
  @app.get("/search")
  async def search_endpoint(query: str):
      async with get_db_session() as db:
          results = await db.execute("SELECT * FROM docs WHERE content LIKE %s", query)
          return results
  ```

### Pitfall: Not Handling Embedding API Failures Gracefully
- **Scenario**: Embedding service failures causing entire pipeline to halt
- **Example of Wrong Approach**:
  ```python
  # Wrong - no error handling
  async def process_documents(docs):
      for doc in docs:
          embedding = await embedding_service.generate(doc.content)
          await store_embedding(doc.id, embedding)
  ```
- **Correct Approach**:
  ```python
  # Correct - graceful error handling with retries
  async def process_documents(docs):
      for doc in docs:
          try:
              embedding = await embedding_service.generate(doc.content)
              await store_embedding(doc.id, embedding)
          except EmbeddingServiceError as e:
              logger.warning(f"Failed to process doc {doc.id}: {e}")
              await add_to_retry_queue(doc)
          except Exception as e:
              logger.error(f"Unexpected error processing doc {doc.id}: {e}")
              await add_to_failed_queue(doc)
  ```

## Edge Cases

### Edge Case: Empty or Very Small Document Collections
- **Description**: System behavior when collections have <100 documents
- **Test Data**: Collections with 0, 1, 5, 50 documents
- **Handling Strategy**: Implement minimum collection size checks, fallback to keyword-only search
- **Code**:
  ```python
  async def search_with_fallback(query: str, collection_id: str):
      collection_stats = await get_collection_stats(collection_id)
      
      if collection_stats.document_count < MIN_SEMANTIC_SEARCH_THRESHOLD:
          # Fall back to keyword search only
          return await keyword_search_only(query, collection_id)
      
      return await hybrid_search(query, collection_id)
  ```

### Edge Case: Unicode and Special Character Handling
- **Description**: Documents containing unusual Unicode characters, emojis, or special formatting
- **Test Data**: Documents with CJK characters, right-to-left text, mathematical symbols
- **Handling Strategy**: Normalize Unicode, implement charset detection
- **Code**:
  ```python
  import unicodedata
  
  def normalize_content(content: str) -> str:
      # Normalize Unicode to NFC form
      normalized = unicodedata.normalize('NFC', content)
      
      # Remove control characters but preserve line breaks
      cleaned = ''.join(
          char for char in normalized 
          if unicodedata.category(char)[0] != 'C' or char in '\n\r\t'
      )
      
      return cleaned
  ```

### Edge Case: Vector Dimension Mismatches
- **Description**: Embeddings with unexpected dimensions due to model changes
- **Test Data**: Vectors with 1536, 2048, 4096 dimensions
- **Handling Strategy**: Validate dimensions, implement dimension padding/truncation
- **Code**:
  ```python
  def validate_and_fix_dimensions(embedding: List[float], expected_dim: int) -> List[float]:
      if len(embedding) == expected_dim:
          return embedding
      elif len(embedding) > expected_dim:
          # Truncate to expected dimension
          logger.warning(f"Truncating embedding from {len(embedding)} to {expected_dim}")
          return embedding[:expected_dim]
      else:
          # Pad with zeros
          logger.warning(f"Padding embedding from {len(embedding)} to {expected_dim}")
          return embedding + [0.0] * (expected_dim - len(embedding))
  ```

## Version-Specific Issues

### Qdrant v1.8.0+
- **Known Issue**: Changed default HNSW parameters affecting memory usage
- **Workaround**: Explicitly set HNSW config in collection creation
- **Fixed In**: v1.9.0 (improved defaults)

### Voyage AI Models (Post June 2024)
- **Known Issue**: New tokenizer causing token count mismatches
- **Workaround**: Always specify model parameter in requests
- **Monitoring**: Track token usage vs estimates, alert on >10% variance

### FastAPI v0.100.0+
- **Known Issue**: Dependency injection scope changes affecting database connections
- **Workaround**: Use explicit dependency scope management
- **Fixed In**: v0.104.0 (improved documentation and defaults)

## Troubleshooting Guide

### Problem: Search Results Empty Despite Having Data
**Diagnostic Steps**:
1. Check collection status: `GET /collections/{name}`
2. Verify vector count: `collection.vectors_count > 0`
3. Test with basic similarity search
4. Check embedding dimensions match

**Solutions** (try in order):
1. Verify collection has been optimized: trigger optimization if needed
2. Check similarity threshold: lower threshold to 0.0 for testing
3. Validate query embedding dimensions
4. Rebuild collection index if corrupted

### Problem: High Memory Usage During Ingestion
**Diagnostic Steps**:
1. Monitor memory usage: `ps aux | grep qdrant`
2. Check batch sizes in ingestion pipeline
3. Review document sizes being processed
4. Monitor garbage collection frequency

**Solutions** (try in order):
1. Reduce batch size for vector upserts
2. Implement streaming document processing
3. Force garbage collection between batches
4. Increase system memory or implement disk-based processing

### Problem: API Response Times Degrading Over Time
**Diagnostic Steps**:
1. Check database connection pool status
2. Monitor embedding cache hit rates
3. Review async task queue lengths
4. Check for memory leaks in FastAPI

**Solutions** (try in order):
1. Restart application to clear memory leaks
2. Increase database connection pool size
3. Clear and rebuild embedding cache
4. Scale horizontally with additional workers

## Performance Gotchas

### Issue: HNSW Index Build Time Scaling Non-Linearly
- **Cause**: ef_construct parameter too high for large collections
- **Solution**: Use progressive index building with lower ef_construct initially
- **Code**:
  ```python
  async def progressive_index_build(collection_name: str, vector_count: int):
      if vector_count > 1_000_000:
          # Use lower ef_construct for initial build
          await update_collection_config(
              collection_name,
              hnsw_config=HnswConfigDiff(ef_construct=50)
          )
          
          # Optimize after initial build
          await optimize_collection(collection_name)
          
          # Increase ef_construct for better quality
          await update_collection_config(
              collection_name,
              hnsw_config=HnswConfigDiff(ef_construct=200)
          )
  ```

### Issue: Embedding Cache Growing Unbounded
- **Cause**: No LRU eviction policy implemented
- **Solution**: Implement proper cache management with size limits
- **Code**:
  ```python
  class BoundedEmbeddingCache:
      def __init__(self, max_size_gb: float = 10.0):
          self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
          self.current_size = 0
          
      async def evict_lru_entries(self):
          while self.current_size > self.max_size_bytes:
              # Remove least recently used entries
              lru_entries = await self.get_lru_entries(limit=100)
              for entry in lru_entries:
                  await self.delete_entry(entry.cache_key)
                  self.current_size -= entry.size_bytes
  ```

## Security Gotchas

### Vulnerability: API Key Exposure in Logs
- **Risk**: Voyage AI API keys logged in error messages
- **Mitigation**: Implement proper secret redaction in logging
- **Code**:
  ```python
  def sanitize_log_message(message: str) -> str:
      # Redact API keys, tokens, and other secrets
      patterns = [
          r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9-_]+)',
          r'bearer\s+([a-zA-Z0-9-_\.]+)',
          r'token["\s]*[:=]["\s]*([a-zA-Z0-9-_]+)'
      ]
      
      for pattern in patterns:
          message = re.sub(pattern, r'\1***REDACTED***', message, flags=re.IGNORECASE)
      
      return message
  ```

### Vulnerability: Injection Through Search Queries
- **Risk**: SQL injection through search query parameters
- **Mitigation**: Use parameterized queries and input validation
- **Code**:
  ```python
  from pydantic import BaseModel, validator
  
  class SearchRequest(BaseModel):
      query: str
      filters: Optional[Dict[str, Any]] = None
      
      @validator('query')
      def validate_query(cls, v):
          if len(v) > 1000:
              raise ValueError('Query too long')
          
          # Remove potentially dangerous characters
          dangerous_chars = ['<', '>', '"', "'", ';', '--']
          for char in dangerous_chars:
              if char in v:
                  raise ValueError(f'Invalid character: {char}')
          
          return v
  ```

## Prevention Checklist

### Pre-Development
- [ ] Review all technology-specific gotchas and known issues
- [ ] Design error handling and recovery strategies
- [ ] Plan monitoring and alerting for each component
- [ ] Establish performance baselines and SLAs

### During Development
- [ ] Implement comprehensive error handling for all external API calls
- [ ] Add proper async/await usage validation
- [ ] Test with realistic data volumes and concurrency
- [ ] Validate memory usage patterns under load

### Pre-Production
- [ ] Load test with 10x expected traffic
- [ ] Validate monitoring and alerting systems
- [ ] Test disaster recovery procedures
- [ ] Verify security controls and secret management

### Operations
- [ ] Monitor key performance metrics continuously
- [ ] Implement automated scaling policies
- [ ] Maintain runbooks for common issues
- [ ] Conduct regular security audits

This comprehensive gotcha documentation should help the development team anticipate and avoid the most common and critical issues when implementing the RAG system.