-- ===============================================================================
-- Contexter RAG System Database Schemas
-- ===============================================================================
-- 
-- This file contains comprehensive database schema definitions for the
-- Contexter RAG system, including operational metadata, caching structures,
-- and monitoring data schemas.
--
-- Database Systems:
-- - SQLite: Embedding cache, operational metadata, configuration
-- - Qdrant: Vector storage (collection definitions)
-- - Redis: Query cache, session storage, rate limiting
--
-- Version: 2.0.0
-- Last Updated: 2025-01-15
-- ===============================================================================

-- ===============================================================================
-- SECTION 1: OPERATIONAL METADATA (SQLite)
-- ===============================================================================

-- Create the main operational database
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;

-- Document metadata and processing status
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    library_id TEXT NOT NULL,
    version TEXT,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    format TEXT NOT NULL DEFAULT 'json',
    compression TEXT DEFAULT 'gzip',
    compressed_size INTEGER,
    
    -- Processing status
    status TEXT NOT NULL DEFAULT 'queued' 
        CHECK (status IN ('queued', 'parsing', 'chunking', 'embedding', 'storing', 'completed', 'failed', 'archived')),
    priority INTEGER DEFAULT 0,
    
    -- Metadata
    metadata_json TEXT, -- JSON blob for flexible metadata
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME,
    expires_at DATETIME,
    
    -- Constraints
    UNIQUE(library_id, version),
    CHECK (file_size > 0),
    CHECK (priority >= 0 AND priority <= 10),
    CHECK (compressed_size IS NULL OR compressed_size <= file_size)
);

-- Index for efficient document queries
CREATE INDEX IF NOT EXISTS idx_documents_library_id ON documents(library_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_priority ON documents(priority DESC);
CREATE INDEX IF NOT EXISTS idx_documents_expires_at ON documents(expires_at) WHERE expires_at IS NOT NULL;

-- Document processing statistics
CREATE TABLE IF NOT EXISTS processing_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    
    -- Token and chunk statistics
    total_tokens INTEGER NOT NULL DEFAULT 0,
    chunks_created INTEGER NOT NULL DEFAULT 0,
    embeddings_generated INTEGER NOT NULL DEFAULT 0,
    vectors_stored INTEGER NOT NULL DEFAULT 0,
    
    -- Timing statistics
    processing_time_seconds REAL DEFAULT 0,
    parsing_time_seconds REAL DEFAULT 0,
    chunking_time_seconds REAL DEFAULT 0,
    embedding_time_seconds REAL DEFAULT 0,
    storage_time_seconds REAL DEFAULT 0,
    
    -- Resource usage
    peak_memory_mb REAL DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,
    cache_hits INTEGER DEFAULT 0,
    
    -- Error tracking
    error_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (total_tokens >= 0),
    CHECK (chunks_created >= 0),
    CHECK (embeddings_generated >= 0),
    CHECK (vectors_stored >= 0 AND vectors_stored <= embeddings_generated),
    CHECK (processing_time_seconds >= 0),
    CHECK (peak_memory_mb >= 0),
    CHECK (error_count >= 0)
);

-- Index for processing stats queries
CREATE INDEX IF NOT EXISTS idx_processing_stats_document_id ON processing_stats(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_stats_created_at ON processing_stats(created_at);

-- Document chunks metadata
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    
    -- Chunk properties
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    character_count INTEGER NOT NULL,
    
    -- Position information
    start_token INTEGER NOT NULL,
    end_token INTEGER NOT NULL,
    has_overlap BOOLEAN DEFAULT FALSE,
    overlap_tokens INTEGER DEFAULT 0,
    
    -- Content classification
    chunk_type TEXT DEFAULT 'text' 
        CHECK (chunk_type IN ('text', 'code', 'mixed', 'table', 'list')),
    boundary_type TEXT
        CHECK (boundary_type IN ('sentence', 'paragraph', 'section', 'code_block', 'token_limit') OR boundary_type IS NULL),
    language TEXT,
    
    -- Metadata
    section TEXT,
    subsection TEXT,
    heading_context_json TEXT, -- JSON array of hierarchical headings
    metadata_json TEXT, -- Additional chunk-specific metadata
    
    -- Embedding status
    embedding_status TEXT DEFAULT 'pending'
        CHECK (embedding_status IN ('pending', 'processing', 'completed', 'failed')),
    embedding_id TEXT, -- Reference to vector storage
    
    -- Timestamps
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(document_id, chunk_index),
    CHECK (chunk_index >= 0),
    CHECK (total_chunks > 0),
    CHECK (chunk_index < total_chunks),
    CHECK (token_count > 0),
    CHECK (character_count > 0),
    CHECK (start_token >= 0),
    CHECK (end_token > start_token),
    CHECK (overlap_tokens >= 0)
);

-- Indexes for chunk queries
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_status ON document_chunks(embedding_status);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON document_chunks(language) WHERE language IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON document_chunks(chunk_type);

-- Processing errors log
CREATE TABLE IF NOT EXISTS processing_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_id TEXT REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Error details
    error_code TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_stage TEXT NOT NULL
        CHECK (error_stage IN ('parsing', 'chunking', 'embedding', 'storage')),
    
    -- Error context
    context_json TEXT, -- JSON with error context details
    retry_count INTEGER DEFAULT 0,
    is_recoverable BOOLEAN DEFAULT TRUE,
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolution_notes TEXT,
    resolved_at DATETIME,
    
    -- Timestamps
    occurred_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (retry_count >= 0),
    CHECK ((resolved = FALSE AND resolved_at IS NULL) OR (resolved = TRUE AND resolved_at IS NOT NULL))
);

-- Index for error tracking
CREATE INDEX IF NOT EXISTS idx_errors_document_id ON processing_errors(document_id);
CREATE INDEX IF NOT EXISTS idx_errors_error_code ON processing_errors(error_code);
CREATE INDEX IF NOT EXISTS idx_errors_error_stage ON processing_errors(error_stage);
CREATE INDEX IF NOT EXISTS idx_errors_occurred_at ON processing_errors(occurred_at);
CREATE INDEX IF NOT EXISTS idx_errors_resolved ON processing_errors(resolved);

-- ===============================================================================
-- SECTION 2: EMBEDDING CACHE (SQLite)
-- ===============================================================================

-- Embedding cache for performance optimization
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash TEXT PRIMARY KEY,
    content_preview TEXT NOT NULL, -- First 200 chars for debugging
    
    -- Embedding data
    embedding_blob BLOB NOT NULL, -- Compressed numpy array
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    input_type TEXT NOT NULL 
        CHECK (input_type IN ('document', 'query')),
    dimensions INTEGER NOT NULL DEFAULT 2048,
    
    -- Generation metadata
    generation_time_ms REAL NOT NULL,
    api_call_id TEXT,
    batch_id TEXT,
    cost_usd REAL,
    
    -- Cache management
    access_count INTEGER DEFAULT 1,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME,
    
    -- Quality metrics
    norm REAL, -- Vector norm for validation
    zero_dimensions INTEGER DEFAULT 0,
    
    -- Constraints
    CHECK (dimensions > 0),
    CHECK (generation_time_ms >= 0),
    CHECK (access_count > 0),
    CHECK (cost_usd IS NULL OR cost_usd >= 0),
    CHECK (norm IS NULL OR norm >= 0),
    CHECK (zero_dimensions >= 0 AND zero_dimensions <= dimensions)
);

-- Indexes for cache operations
CREATE INDEX IF NOT EXISTS idx_cache_model ON embedding_cache(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_cache_input_type ON embedding_cache(input_type);
CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON embedding_cache(last_accessed);
CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON embedding_cache(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_cache_created_at ON embedding_cache(created_at);

-- Cache statistics for monitoring
CREATE TABLE IF NOT EXISTS cache_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Statistics snapshot
    total_entries INTEGER NOT NULL,
    cache_size_mb REAL NOT NULL,
    hit_rate REAL NOT NULL,
    avg_access_count REAL NOT NULL,
    
    -- Model breakdown
    model_distribution_json TEXT, -- JSON with model usage stats
    
    -- Performance metrics
    avg_generation_time_ms REAL,
    total_cost_usd REAL,
    
    -- Cleanup statistics
    expired_entries INTEGER DEFAULT 0,
    evicted_entries INTEGER DEFAULT 0,
    
    -- Timestamp
    snapshot_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (total_entries >= 0),
    CHECK (cache_size_mb >= 0),
    CHECK (hit_rate >= 0 AND hit_rate <= 1),
    CHECK (avg_access_count >= 1),
    CHECK (avg_generation_time_ms IS NULL OR avg_generation_time_ms >= 0),
    CHECK (total_cost_usd IS NULL OR total_cost_usd >= 0)
);

-- Index for stats queries
CREATE INDEX IF NOT EXISTS idx_cache_stats_snapshot_at ON cache_stats(snapshot_at);

-- ===============================================================================
-- SECTION 3: SEARCH AND QUERY METADATA (SQLite)
-- ===============================================================================

-- Search query log for analytics and optimization
CREATE TABLE IF NOT EXISTS search_queries (
    query_id TEXT PRIMARY KEY,
    
    -- Query details
    query_text TEXT NOT NULL,
    normalized_query TEXT,
    query_hash TEXT NOT NULL,
    
    -- Search parameters
    search_type TEXT NOT NULL 
        CHECK (search_type IN ('semantic', 'keyword', 'hybrid')),
    limit_requested INTEGER NOT NULL,
    threshold REAL NOT NULL,
    filters_json TEXT, -- JSON representation of filters
    
    -- Query analysis
    intent TEXT 
        CHECK (intent IN ('informational', 'procedural', 'api_reference', 'troubleshooting', 'example', 'comparison') OR intent IS NULL),
    complexity TEXT 
        CHECK (complexity IN ('simple', 'moderate', 'complex') OR complexity IS NULL),
    language TEXT,
    
    -- Execution metadata
    user_id TEXT,
    session_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    
    -- Results metadata
    results_count INTEGER NOT NULL DEFAULT 0,
    execution_time_ms REAL NOT NULL,
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (limit_requested > 0 AND limit_requested <= 1000),
    CHECK (threshold >= 0 AND threshold <= 1),
    CHECK (results_count >= 0),
    CHECK (execution_time_ms >= 0)
);

-- Indexes for query analysis
CREATE INDEX IF NOT EXISTS idx_queries_query_hash ON search_queries(query_hash);
CREATE INDEX IF NOT EXISTS idx_queries_search_type ON search_queries(search_type);
CREATE INDEX IF NOT EXISTS idx_queries_executed_at ON search_queries(executed_at);
CREATE INDEX IF NOT EXISTS idx_queries_user_id ON search_queries(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_queries_session_id ON search_queries(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_queries_execution_time ON search_queries(execution_time_ms);

-- Search results for relevance tracking
CREATE TABLE IF NOT EXISTS search_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL REFERENCES search_queries(query_id) ON DELETE CASCADE,
    
    -- Result identification
    chunk_id TEXT NOT NULL,
    result_rank INTEGER NOT NULL,
    
    -- Scoring details
    final_score REAL NOT NULL,
    semantic_score REAL,
    keyword_score REAL,
    quality_boost REAL DEFAULT 1.0,
    
    -- User interaction
    clicked BOOLEAN DEFAULT FALSE,
    click_position INTEGER,
    dwell_time_seconds REAL,
    
    -- Result metadata
    metadata_json TEXT,
    highlights_json TEXT, -- JSON array of highlights
    
    -- Timestamps
    served_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    clicked_at DATETIME,
    
    -- Constraints
    UNIQUE(query_id, chunk_id),
    CHECK (result_rank > 0),
    CHECK (final_score >= 0 AND final_score <= 1),
    CHECK (semantic_score IS NULL OR (semantic_score >= 0 AND semantic_score <= 1)),
    CHECK (keyword_score IS NULL OR (keyword_score >= 0 AND keyword_score <= 1)),
    CHECK (quality_boost >= 0),
    CHECK (click_position IS NULL OR click_position > 0),
    CHECK (dwell_time_seconds IS NULL OR dwell_time_seconds >= 0),
    CHECK ((clicked = FALSE AND clicked_at IS NULL) OR (clicked = TRUE AND clicked_at IS NOT NULL))
);

-- Indexes for result analysis
CREATE INDEX IF NOT EXISTS idx_results_query_id ON search_results(query_id);
CREATE INDEX IF NOT EXISTS idx_results_chunk_id ON search_results(chunk_id);
CREATE INDEX IF NOT EXISTS idx_results_result_rank ON search_results(result_rank);
CREATE INDEX IF NOT EXISTS idx_results_clicked ON search_results(clicked);
CREATE INDEX IF NOT EXISTS idx_results_served_at ON search_results(served_at);

-- Query suggestions and autocomplete data
CREATE TABLE IF NOT EXISTS query_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Suggestion data
    suggestion_text TEXT NOT NULL,
    suggestion_type TEXT NOT NULL 
        CHECK (suggestion_type IN ('completion', 'correction', 'related')),
    source_query TEXT NOT NULL,
    
    -- Usage statistics
    usage_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0,
    avg_results_count REAL DEFAULT 0,
    
    -- Quality metrics
    relevance_score REAL NOT NULL,
    confidence REAL DEFAULT 0,
    
    -- Metadata
    context_json TEXT, -- JSON with contextual information
    
    -- Management
    enabled BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(suggestion_text, source_query),
    CHECK (usage_count >= 0),
    CHECK (success_rate >= 0 AND success_rate <= 1),
    CHECK (avg_results_count >= 0),
    CHECK (relevance_score >= 0 AND relevance_score <= 1),
    CHECK (confidence >= 0 AND confidence <= 1)
);

-- Indexes for suggestion retrieval
CREATE INDEX IF NOT EXISTS idx_suggestions_source_query ON query_suggestions(source_query);
CREATE INDEX IF NOT EXISTS idx_suggestions_suggestion_text ON query_suggestions(suggestion_text);
CREATE INDEX IF NOT EXISTS idx_suggestions_relevance_score ON query_suggestions(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_suggestions_enabled ON query_suggestions(enabled);

-- ===============================================================================
-- SECTION 4: SYSTEM METRICS AND MONITORING (SQLite)
-- ===============================================================================

-- System performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Metric identification
    metric_name TEXT NOT NULL,
    metric_type TEXT NOT NULL 
        CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'summary')),
    component TEXT NOT NULL,
    
    -- Metric value and metadata
    value REAL NOT NULL,
    labels_json TEXT, -- JSON representation of metric labels
    description TEXT,
    
    -- Aggregation data (for histograms/summaries)
    count INTEGER,
    sum REAL,
    min_value REAL,
    max_value REAL,
    p50 REAL,
    p95 REAL,
    p99 REAL,
    
    -- Timestamp
    collected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (count IS NULL OR count >= 0),
    CHECK (sum IS NULL OR sum >= 0),
    CHECK (min_value IS NULL OR max_value IS NULL OR min_value <= max_value)
);

-- Indexes for metrics queries
CREATE INDEX IF NOT EXISTS idx_metrics_name_component ON performance_metrics(metric_name, component);
CREATE INDEX IF NOT EXISTS idx_metrics_collected_at ON performance_metrics(collected_at);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_type ON performance_metrics(metric_type);

-- Health check results
CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Component identification
    component_name TEXT NOT NULL,
    check_type TEXT NOT NULL,
    
    -- Health status
    status TEXT NOT NULL 
        CHECK (status IN ('healthy', 'degraded', 'unhealthy')),
    response_time_ms REAL NOT NULL,
    
    -- Details
    message TEXT,
    details_json TEXT, -- JSON with detailed health information
    error_message TEXT,
    
    -- Timestamps
    checked_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (response_time_ms >= 0)
);

-- Indexes for health monitoring
CREATE INDEX IF NOT EXISTS idx_health_component ON health_checks(component_name);
CREATE INDEX IF NOT EXISTS idx_health_status ON health_checks(status);
CREATE INDEX IF NOT EXISTS idx_health_checked_at ON health_checks(checked_at);

-- System alerts and notifications
CREATE TABLE IF NOT EXISTS system_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Alert identification
    alert_id TEXT UNIQUE NOT NULL,
    component TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    
    -- Severity and status
    severity TEXT NOT NULL 
        CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    status TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'acknowledged', 'resolved')),
    
    -- Alert content
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT,
    
    -- Timestamps
    triggered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at DATETIME,
    resolved_at DATETIME,
    
    -- Resolution
    resolution_notes TEXT,
    resolved_by TEXT,
    
    -- Constraints
    CHECK ((status = 'active' AND acknowledged_at IS NULL) OR 
           (status = 'acknowledged' AND acknowledged_at IS NOT NULL) OR
           (status = 'resolved' AND resolved_at IS NOT NULL))
);

-- Indexes for alert management
CREATE INDEX IF NOT EXISTS idx_alerts_alert_id ON system_alerts(alert_id);
CREATE INDEX IF NOT EXISTS idx_alerts_component ON system_alerts(component);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON system_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON system_alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_triggered_at ON system_alerts(triggered_at);

-- ===============================================================================
-- SECTION 5: CONFIGURATION AND FEATURE FLAGS (SQLite)
-- ===============================================================================

-- Dynamic configuration storage
CREATE TABLE IF NOT EXISTS system_configuration (
    key TEXT PRIMARY KEY,
    
    -- Configuration value
    value TEXT NOT NULL,
    value_type TEXT NOT NULL 
        CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    
    -- Metadata
    description TEXT,
    category TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    validation_pattern TEXT, -- Regex pattern for validation
    
    -- Versioning
    version INTEGER DEFAULT 1,
    previous_value TEXT,
    
    -- Management
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_by TEXT,
    
    -- Constraints
    CHECK (version > 0)
);

-- Indexes for configuration queries
CREATE INDEX IF NOT EXISTS idx_config_category ON system_configuration(category);
CREATE INDEX IF NOT EXISTS idx_config_updated_at ON system_configuration(updated_at);

-- Feature flags with context-based rules
CREATE TABLE IF NOT EXISTS feature_flags (
    flag_name TEXT PRIMARY KEY,
    
    -- Flag configuration
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    description TEXT NOT NULL,
    category TEXT,
    
    -- Context-based rules
    rules_json TEXT, -- JSON with complex flag rules
    rollout_percentage INTEGER DEFAULT 0,
    
    -- User/environment targeting
    user_whitelist_json TEXT, -- JSON array of whitelisted users
    user_blacklist_json TEXT, -- JSON array of blacklisted users
    environment_rules_json TEXT, -- JSON with environment-specific rules
    
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    updated_by TEXT,
    
    -- Constraints
    CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100)
);

-- Index for feature flag queries
CREATE INDEX IF NOT EXISTS idx_flags_category ON feature_flags(category);
CREATE INDEX IF NOT EXISTS idx_flags_enabled ON feature_flags(enabled);

-- Feature flag evaluation log
CREATE TABLE IF NOT EXISTS flag_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Evaluation context
    flag_name TEXT NOT NULL REFERENCES feature_flags(flag_name),
    user_id TEXT,
    session_id TEXT,
    environment TEXT,
    
    -- Evaluation result
    result BOOLEAN NOT NULL,
    rule_matched TEXT, -- Which rule determined the result
    
    -- Context
    context_json TEXT, -- JSON with evaluation context
    
    -- Timestamp
    evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Index for flag evaluation analysis
CREATE INDEX IF NOT EXISTS idx_flag_evals_flag_name ON flag_evaluations(flag_name);
CREATE INDEX IF NOT EXISTS idx_flag_evals_evaluated_at ON flag_evaluations(evaluated_at);
CREATE INDEX IF NOT EXISTS idx_flag_evals_result ON flag_evaluations(result);

-- ===============================================================================
-- SECTION 6: QDRANT VECTOR DATABASE COLLECTIONS
-- ===============================================================================

-- Note: Qdrant collections are defined programmatically, but here are the
-- specifications for documentation purposes

/*
Qdrant Collection: contexter_docs
- Vector size: 2048 (Voyage AI voyage-code-3 dimensions)
- Distance metric: Cosine
- HNSW configuration:
  - m: 16 (bi-directional links)
  - ef_construct: 200 (dynamic candidate list size)
  - max_indexing_threads: 0 (use all available threads)
  - full_scan_threshold: 10000

Payload Schema:
{
  "chunk_id": "string",          // Unique chunk identifier
  "document_id": "string",       // Parent document identifier
  "library_id": "string",        // Library identifier
  "library_name": "string",      // Human-readable library name
  "version": "string",           // Library version
  "content": "string",           // Full text content of the chunk
  "section": "string",           // Document section name
  "subsection": "string",        // Document subsection name
  "doc_type": "string",          // Document type (api, guide, tutorial, etc.)
  "chunk_type": "string",        // Content type (text, code, mixed, etc.)
  "language": "string",          // Programming language
  "chunk_index": "integer",      // Chunk position in document
  "total_chunks": "integer",     // Total chunks in document
  "token_count": "integer",      // Number of tokens in chunk
  "heading_context": ["string"], // Hierarchical heading context
  "trust_score": "float",        // Quality/trust score
  "star_count": "integer",       // GitHub stars
  "created_at": "datetime",      // Creation timestamp
  "embedding_model": "string",   // Model used for embedding
  "embedding_version": "string"  // Model version
}

Payload Indexes (for efficient filtering):
- library_id (keyword)
- doc_type (keyword)
- language (keyword)
- section (keyword)
- chunk_type (keyword)
- trust_score (float range)
- star_count (integer range)
- created_at (datetime range)
- chunk_index (integer)
- token_count (integer range)
*/

-- ===============================================================================
-- SECTION 7: REDIS CACHE STRUCTURES
-- ===============================================================================

-- Note: Redis structures are defined by key patterns, documented here for reference

/*
Redis Key Patterns and Structures:

1. Query Result Cache:
   Key: "search:cache:{query_hash}"
   Type: Hash
   TTL: 3600 seconds (1 hour)
   Fields:
   - query: Original query text
   - results: JSON-serialized search results
   - total_count: Total number of results
   - execution_time_ms: Query execution time
   - cached_at: Cache creation timestamp

2. Rate Limiting:
   Key: "rate_limit:{user_id}:{endpoint}"
   Type: String (counter)
   TTL: 60 seconds
   Value: Number of requests in current window

3. Session Storage:
   Key: "session:{session_id}"
   Type: Hash
   TTL: 86400 seconds (24 hours)
   Fields:
   - user_id: User identifier
   - created_at: Session creation time
   - last_activity: Last activity timestamp
   - preferences: JSON-serialized user preferences

4. Query Suggestions Cache:
   Key: "suggestions:{partial_query_hash}"
   Type: List
   TTL: 1800 seconds (30 minutes)
   Value: JSON-serialized suggestion objects

5. Health Check Cache:
   Key: "health:{component_name}"
   Type: Hash
   TTL: 300 seconds (5 minutes)
   Fields:
   - status: healthy/degraded/unhealthy
   - response_time_ms: Response time
   - last_check: Timestamp of last check
   - details: JSON-serialized health details

6. Feature Flag Cache:
   Key: "flags:{flag_name}:{context_hash}"
   Type: String
   TTL: 300 seconds (5 minutes)
   Value: "true" or "false"

7. Configuration Cache:
   Key: "config:{config_key}"
   Type: String
   TTL: 600 seconds (10 minutes)
   Value: Configuration value

8. Metrics Aggregation:
   Key: "metrics:{metric_name}:{time_bucket}"
   Type: Hash
   TTL: 3600 seconds (1 hour)
   Fields:
   - count: Number of observations
   - sum: Sum of values
   - min: Minimum value
   - max: Maximum value
   - last_update: Last update timestamp
*/

-- ===============================================================================
-- SECTION 8: DATABASE MAINTENANCE AND OPTIMIZATION
-- ===============================================================================

-- Create views for common queries

-- Document processing overview
CREATE VIEW IF NOT EXISTS v_document_processing_overview AS
SELECT 
    d.document_id,
    d.library_id,
    d.version,
    d.status,
    d.created_at,
    d.processed_at,
    ps.total_tokens,
    ps.chunks_created,
    ps.embeddings_generated,
    ps.vectors_stored,
    ps.processing_time_seconds,
    ps.peak_memory_mb,
    ps.error_count,
    (ps.vectors_stored * 1.0 / NULLIF(ps.chunks_created, 0)) as vector_completion_rate
FROM documents d
LEFT JOIN processing_stats ps ON d.document_id = ps.document_id;

-- Search performance analytics
CREATE VIEW IF NOT EXISTS v_search_analytics AS
SELECT 
    DATE(executed_at) as search_date,
    search_type,
    COUNT(*) as query_count,
    AVG(execution_time_ms) as avg_execution_time_ms,
    AVG(results_count) as avg_results_count,
    COUNT(CASE WHEN cache_hit THEN 1 END) * 1.0 / COUNT(*) as cache_hit_rate,
    AVG(threshold) as avg_threshold
FROM search_queries
GROUP BY DATE(executed_at), search_type;

-- System health summary
CREATE VIEW IF NOT EXISTS v_system_health_summary AS
SELECT 
    component_name,
    status,
    COUNT(*) as check_count,
    AVG(response_time_ms) as avg_response_time_ms,
    MAX(checked_at) as last_check,
    MIN(checked_at) as first_check
FROM health_checks
WHERE checked_at >= datetime('now', '-24 hours')
GROUP BY component_name, status;

-- Cache performance summary
CREATE VIEW IF NOT EXISTS v_cache_performance AS
SELECT 
    model_name,
    COUNT(*) as total_entries,
    SUM(LENGTH(embedding_blob)) / 1024.0 / 1024.0 as size_mb,
    AVG(access_count) as avg_access_count,
    AVG(generation_time_ms) as avg_generation_time_ms,
    COUNT(CASE WHEN last_accessed >= datetime('now', '-1 day') THEN 1 END) as recent_access_count
FROM embedding_cache
GROUP BY model_name;

-- ===============================================================================
-- SECTION 9: CLEANUP AND MAINTENANCE PROCEDURES
-- ===============================================================================

-- Cleanup old search queries (keep last 30 days)
CREATE TRIGGER IF NOT EXISTS cleanup_old_search_queries
AFTER INSERT ON search_queries
WHEN (SELECT COUNT(*) FROM search_queries) > 100000
BEGIN
    DELETE FROM search_queries 
    WHERE executed_at < datetime('now', '-30 days');
END;

-- Cleanup old health checks (keep last 7 days)
CREATE TRIGGER IF NOT EXISTS cleanup_old_health_checks
AFTER INSERT ON health_checks
WHEN (SELECT COUNT(*) FROM health_checks) > 50000
BEGIN
    DELETE FROM health_checks 
    WHERE checked_at < datetime('now', '-7 days');
END;

-- Update document updated_at timestamp on status change
CREATE TRIGGER IF NOT EXISTS update_document_timestamp
AFTER UPDATE ON documents
FOR EACH ROW
WHEN NEW.status != OLD.status
BEGIN
    UPDATE documents 
    SET updated_at = CURRENT_TIMESTAMP 
    WHERE document_id = NEW.document_id;
END;

-- Update embedding cache access statistics
CREATE TRIGGER IF NOT EXISTS update_cache_access
AFTER UPDATE ON embedding_cache
FOR EACH ROW
WHEN NEW.last_accessed != OLD.last_accessed
BEGIN
    UPDATE embedding_cache 
    SET access_count = access_count + 1
    WHERE content_hash = NEW.content_hash;
END;

-- ===============================================================================
-- SECTION 10: BACKUP AND RECOVERY SCHEMAS
-- ===============================================================================

-- Backup metadata table
CREATE TABLE IF NOT EXISTS backup_metadata (
    backup_id TEXT PRIMARY KEY,
    backup_type TEXT NOT NULL 
        CHECK (backup_type IN ('full', 'incremental', 'differential')),
    
    -- Backup scope
    tables_included_json TEXT NOT NULL, -- JSON array of table names
    records_count INTEGER NOT NULL,
    backup_size_bytes INTEGER NOT NULL,
    
    -- Compression and encryption
    compression_algorithm TEXT DEFAULT 'gzip',
    compression_ratio REAL,
    encrypted BOOLEAN DEFAULT FALSE,
    encryption_algorithm TEXT,
    
    -- Storage location
    storage_path TEXT NOT NULL,
    storage_type TEXT DEFAULT 'local'
        CHECK (storage_type IN ('local', 's3', 'gcs', 'azure')),
    
    -- Verification
    checksum TEXT NOT NULL,
    verified BOOLEAN DEFAULT FALSE,
    verification_date DATETIME,
    
    -- Timestamps
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    duration_seconds REAL,
    
    -- Status
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'aborted')),
    error_message TEXT,
    
    -- Constraints
    CHECK (records_count >= 0),
    CHECK (backup_size_bytes >= 0),
    CHECK (compression_ratio IS NULL OR (compression_ratio >= 0 AND compression_ratio <= 1)),
    CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
    CHECK ((status = 'completed' AND completed_at IS NOT NULL) OR 
           (status != 'completed' AND completed_at IS NULL))
);

-- Index for backup queries
CREATE INDEX IF NOT EXISTS idx_backup_started_at ON backup_metadata(started_at);
CREATE INDEX IF NOT EXISTS idx_backup_status ON backup_metadata(status);
CREATE INDEX IF NOT EXISTS idx_backup_type ON backup_metadata(backup_type);

-- ===============================================================================
-- SECTION 11: VACUUM AND MAINTENANCE SCHEDULE
-- ===============================================================================

-- Note: These are maintenance commands to be run periodically

/*
Daily Maintenance:
- VACUUM; (Reclaim space from deleted records)
- ANALYZE; (Update query planner statistics)
- DELETE FROM search_queries WHERE executed_at < datetime('now', '-30 days');
- DELETE FROM health_checks WHERE checked_at < datetime('now', '-7 days');

Weekly Maintenance:
- PRAGMA integrity_check; (Verify database integrity)
- PRAGMA foreign_key_check; (Verify foreign key constraints)
- UPDATE embedding_cache SET last_accessed = datetime('now') WHERE expires_at < datetime('now');

Monthly Maintenance:
- PRAGMA optimize; (Optimize database structure)
- Backup database files
- Archive old processing_errors records
- Clean up expired embedding cache entries
*/

-- ===============================================================================
-- END OF SCHEMA DEFINITION
-- ===============================================================================

-- Final PRAGMA settings for optimal performance
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA incremental_vacuum;
PRAGMA optimize;