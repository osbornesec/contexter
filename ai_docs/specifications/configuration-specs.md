# Configuration Specifications

This document provides comprehensive configuration specifications for the Contexter RAG system, including schemas, validation rules, default values, environment variables, and deployment configurations.

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Feature Flags](#feature-flags)
5. [Performance Tuning](#performance-tuning)
6. [Security Settings](#security-settings)
7. [Validation Rules](#validation-rules)
8. [Configuration Management](#configuration-management)

## Configuration Architecture

### Configuration Hierarchy

The Contexter RAG system uses a multi-layered configuration approach with the following precedence (highest to lowest):

1. **Command Line Arguments** - Override all other settings
2. **Environment Variables** - Runtime environment configuration
3. **Configuration Files** - YAML/JSON configuration files
4. **Feature Flags** - Dynamic runtime toggles
5. **Default Values** - Built-in system defaults

### Configuration Schema Structure

```yaml
# Master configuration schema
version: "2.0.0"
environment: "production"  # development, staging, production

# Core system configuration
system:
  log_level: "INFO"
  debug_mode: false
  max_workers: 10
  timezone: "UTC"
  
# Search engine configuration
search:
  enabled: true
  hybrid_weights:
    semantic: 0.7
    keyword: 0.3
  default_threshold: 0.7
  max_results: 100
  cache_enabled: true
  cache_ttl_seconds: 3600
  
# Document ingestion configuration
ingestion:
  enabled: true
  auto_process: true
  chunk_size: 1000
  chunk_overlap: 200
  max_chunks_per_document: 1000
  max_concurrent_jobs: 5
  quality_threshold: 0.5
  
# Embedding service configuration
embedding:
  provider: "voyage_ai"
  model: "voyage-code-3"
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 100
  cache_enabled: true
  cache_ttl_days: 7
  rate_limit_per_minute: 300
  
# Vector database configuration
vector_store:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  collection_name: "contexter_docs"
  vector_size: 2048
  distance_metric: "cosine"
  
# Storage configuration
storage:
  base_path: "${CONTEXTER_STORAGE_PATH:/app/data}"
  compression: "gzip"
  format: "json"
  enable_versioning: true
  max_versions: 5
  
# Monitoring and observability
monitoring:
  enabled: true
  metrics_port: 8080
  health_check_interval: 300
  prometheus_enabled: true
  jaeger_enabled: false
  
# Security configuration
security:
  api_key_required: true
  jwt_secret: "${JWT_SECRET}"
  rate_limiting_enabled: true
  cors_enabled: true
  allowed_origins: ["*"]
```

## Environment Variables

### Required Environment Variables

These environment variables must be set for the system to function:

```bash
# Embedding Service Configuration
export VOYAGE_API_KEY="your-voyage-ai-api-key"

# Security Configuration
export JWT_SECRET="your-jwt-secret-key-here"
export API_KEY="your-api-key-for-authentication"

# Database Credentials (if using external services)
export QDRANT_API_KEY="your-qdrant-api-key"  # Optional for cloud Qdrant
```

### Optional Environment Variables

These variables provide additional configuration options:

```bash
# System Configuration
export CONTEXTER_ENV="production"              # Environment name
export CONTEXTER_LOG_LEVEL="INFO"             # Logging level
export CONTEXTER_DEBUG="false"                # Debug mode
export CONTEXTER_MAX_WORKERS="10"             # Worker threads

# Storage Configuration
export CONTEXTER_STORAGE_PATH="/app/data"     # Base storage path
export CONTEXTER_BACKUP_PATH="/app/backups"   # Backup storage path
export CONTEXTER_CACHE_PATH="/app/cache"      # Cache storage path

# Network Configuration
export CONTEXTER_HOST="0.0.0.0"              # Bind host
export CONTEXTER_PORT="8000"                 # Application port
export CONTEXTER_METRICS_PORT="8080"         # Metrics port

# Vector Database Configuration
export QDRANT_HOST="localhost"               # Qdrant host
export QDRANT_PORT="6333"                    # Qdrant port
export QDRANT_GRPC_PORT="6334"               # Qdrant gRPC port
export QDRANT_COLLECTION="contexter_docs"    # Collection name

# Performance Tuning
export CONTEXTER_SEARCH_CACHE_SIZE="1000"    # Search cache size
export CONTEXTER_EMBEDDING_CACHE_SIZE="10000" # Embedding cache size
export CONTEXTER_MAX_CONCURRENT_REQUESTS="100" # Max concurrent requests
export CONTEXTER_REQUEST_TIMEOUT="30"        # Request timeout (seconds)

# Feature Flags
export CONTEXTER_HYBRID_SEARCH_ENABLED="true"  # Enable hybrid search
export CONTEXTER_AUTO_INGESTION_ENABLED="true" # Enable auto ingestion
export CONTEXTER_MONITORING_ENABLED="true"     # Enable monitoring
export CONTEXTER_RATE_LIMITING_ENABLED="true"  # Enable rate limiting

# Monitoring Configuration
export PROMETHEUS_PUSHGATEWAY_URL=""          # Prometheus pushgateway
export JAEGER_AGENT_HOST=""                   # Jaeger tracing host
export JAEGER_AGENT_PORT="6831"               # Jaeger agent port

# Development Configuration
export CONTEXTER_DEV_MODE="false"            # Development mode
export CONTEXTER_HOT_RELOAD="false"          # Enable hot reload
export CONTEXTER_PROFILING_ENABLED="false"   # Enable profiling
```

### Environment Variable Validation

```python
from typing import Dict, Any, Optional
import os
import re

class EnvironmentValidator:
    """Validate and parse environment variables."""
    
    REQUIRED_VARS = {
        "VOYAGE_API_KEY": {
            "pattern": r"^[a-zA-Z0-9_-]+$",
            "description": "Voyage AI API key for embedding generation"
        },
        "JWT_SECRET": {
            "min_length": 32,
            "description": "JWT secret for token signing"
        }
    }
    
    OPTIONAL_VARS = {
        "CONTEXTER_LOG_LEVEL": {
            "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "INFO"
        },
        "CONTEXTER_PORT": {
            "type": "int",
            "min_value": 1024,
            "max_value": 65535,
            "default": 8000
        },
        "CONTEXTER_MAX_WORKERS": {
            "type": "int",
            "min_value": 1,
            "max_value": 100,
            "default": 10
        }
    }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """Validate all environment variables and return parsed config."""
        config = {}
        errors = []
        
        # Validate required variables
        for var_name, rules in cls.REQUIRED_VARS.items():
            value = os.getenv(var_name)
            if not value:
                errors.append(f"Required environment variable {var_name} is not set")
                continue
                
            validation_result = cls._validate_variable(var_name, value, rules)
            if validation_result["valid"]:
                config[var_name] = validation_result["value"]
            else:
                errors.append(f"Invalid {var_name}: {validation_result['error']}")
        
        # Validate optional variables
        for var_name, rules in cls.OPTIONAL_VARS.items():
            value = os.getenv(var_name)
            if value is None:
                config[var_name] = rules.get("default")
                continue
                
            validation_result = cls._validate_variable(var_name, value, rules)
            if validation_result["valid"]:
                config[var_name] = validation_result["value"]
            else:
                errors.append(f"Invalid {var_name}: {validation_result['error']}")
        
        if errors:
            raise EnvironmentValidationError(f"Environment validation failed: {'; '.join(errors)}")
        
        return config
    
    @classmethod
    def _validate_variable(cls, name: str, value: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single environment variable."""
        try:
            # Type conversion
            if rules.get("type") == "int":
                parsed_value = int(value)
                if "min_value" in rules and parsed_value < rules["min_value"]:
                    return {"valid": False, "error": f"Value too small (min: {rules['min_value']})"}
                if "max_value" in rules and parsed_value > rules["max_value"]:
                    return {"valid": False, "error": f"Value too large (max: {rules['max_value']})"}
                return {"valid": True, "value": parsed_value}
            
            elif rules.get("type") == "bool":
                parsed_value = value.lower() in ("true", "1", "yes", "on")
                return {"valid": True, "value": parsed_value}
            
            # String validation
            if "min_length" in rules and len(value) < rules["min_length"]:
                return {"valid": False, "error": f"Too short (min: {rules['min_length']} chars)"}
            
            if "pattern" in rules and not re.match(rules["pattern"], value):
                return {"valid": False, "error": f"Does not match required pattern"}
            
            if "choices" in rules and value not in rules["choices"]:
                return {"valid": False, "error": f"Must be one of: {', '.join(rules['choices'])}"}
            
            return {"valid": True, "value": value}
            
        except ValueError as e:
            return {"valid": False, "error": f"Type conversion failed: {e}"}

class EnvironmentValidationError(Exception):
    """Environment variable validation error."""
    pass
```

## Configuration Files

### Main Configuration File

The primary configuration file `config.yaml` contains all system settings:

```yaml
# config.yaml - Complete Contexter RAG Configuration
version: "2.0.0"
environment: "${CONTEXTER_ENV:production}"

# System-wide settings
system:
  name: "contexter-rag"
  version: "2.0.0"
  description: "Contexter RAG System for Documentation Search"
  
  # Logging configuration
  logging:
    level: "${CONTEXTER_LOG_LEVEL:INFO}"
    format: "json"
    file_path: "${CONTEXTER_LOG_PATH:/app/logs/contexter.log}"
    max_size_mb: 100
    backup_count: 5
    enable_correlation_id: true
    
  # Runtime configuration
  runtime:
    debug_mode: "${CONTEXTER_DEBUG:false}"
    max_workers: "${CONTEXTER_MAX_WORKERS:10}"
    worker_timeout_seconds: 300
    graceful_shutdown_timeout: 30
    timezone: "UTC"
    
  # Memory management
  memory:
    max_heap_size_mb: 8192
    gc_threshold: 0.8
    enable_memory_profiling: false

# API server configuration
api:
  host: "${CONTEXTER_HOST:0.0.0.0}"
  port: "${CONTEXTER_PORT:8000}"
  workers: "${CONTEXTER_API_WORKERS:4}"
  timeout: "${CONTEXTER_REQUEST_TIMEOUT:30}"
  max_request_size_mb: 100
  
  # CORS configuration
  cors:
    enabled: true
    allowed_origins: 
      - "http://localhost:3000"
      - "https://app.contexter.dev"
    allowed_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: ["*"]
    allow_credentials: true
    
  # Rate limiting
  rate_limiting:
    enabled: "${CONTEXTER_RATE_LIMITING_ENABLED:true}"
    requests_per_minute: 1000
    burst_size: 100
    redis_url: "${REDIS_URL:redis://localhost:6379/0}"

# Search engine configuration
search:
  enabled: true
  engine_type: "hybrid"  # semantic, keyword, hybrid
  
  # Hybrid search weights
  hybrid_weights:
    semantic: 0.7
    keyword: 0.3
  
  # Search parameters
  default_threshold: 0.7
  max_results: 100
  max_query_length: 1000
  enable_query_expansion: true
  enable_spell_correction: false
  
  # Caching
  cache:
    enabled: true
    ttl_seconds: 3600
    max_size: 10000
    backend: "redis"  # memory, redis
    
  # Result reranking
  reranking:
    enabled: true
    algorithm: "cross_encoder"
    max_candidates: 100
    quality_boost_factor: 1.2
    
  # Performance
  performance:
    target_latency_p95_ms: 50
    target_latency_p99_ms: 100
    circuit_breaker_enabled: true
    circuit_breaker_threshold: 10

# Document ingestion configuration
ingestion:
  enabled: true
  auto_process: true
  
  # Processing parameters
  chunking:
    strategy: "token_based"  # token_based, sentence_based, paragraph_based
    chunk_size: 1000
    overlap_size: 200
    max_chunks_per_document: 1000
    preserve_code_blocks: true
    preserve_tables: true
    
  # Quality control
  quality:
    min_chunk_size: 50
    max_chunk_size: 4000
    min_quality_score: 0.5
    enable_content_validation: true
    
  # Processing queue
  queue:
    max_concurrent_jobs: 5
    job_timeout_seconds: 3600
    retry_attempts: 3
    retry_backoff_multiplier: 2
    
  # Performance targets
  performance:
    target_throughput_docs_per_minute: 1000
    max_memory_per_job_mb: 512

# Embedding service configuration
embedding:
  provider: "voyage_ai"
  model: "voyage-code-3"
  
  # API configuration
  api:
    key: "${VOYAGE_API_KEY}"
    base_url: "https://api.voyageai.com/v1"
    timeout_seconds: 60
    max_retries: 3
    backoff_factor: 2
    
  # Batch processing
  batch:
    size: 100
    max_concurrent: 5
    optimize_batch_size: true
    
  # Caching
  cache:
    enabled: true
    ttl_days: 7
    max_size_gb: 10
    compression_enabled: true
    backend: "sqlite"  # sqlite, redis
    
  # Rate limiting
  rate_limit:
    requests_per_minute: 300
    tokens_per_minute: 1000000
    adaptive_throttling: true
    
  # Performance monitoring
  performance:
    target_latency_p95_ms: 500
    target_throughput_per_minute: 1000
    cost_tracking_enabled: true

# Vector database configuration
vector_store:
  provider: "qdrant"
  
  # Connection settings
  connection:
    host: "${QDRANT_HOST:localhost}"
    port: "${QDRANT_PORT:6333}"
    grpc_port: "${QDRANT_GRPC_PORT:6334}"
    prefer_grpc: true
    timeout_seconds: 30
    api_key: "${QDRANT_API_KEY:}"
    
  # Collection configuration
  collection:
    name: "${QDRANT_COLLECTION:contexter_docs}"
    vector_size: 2048
    distance_metric: "cosine"
    shard_number: 1
    replication_factor: 1
    
  # HNSW index configuration
  hnsw:
    m: 16
    ef_construct: 200
    max_indexing_threads: 0
    full_scan_threshold: 10000
    
  # Optimization settings
  optimization:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    max_segment_size: 200000
    indexing_threshold: 20000
    flush_interval_seconds: 5
    
  # Performance targets
  performance:
    target_search_latency_p95_ms: 50
    target_upsert_throughput_per_second: 1000
    max_memory_usage_gb: 8

# Storage configuration
storage:
  # Base storage settings
  base_path: "${CONTEXTER_STORAGE_PATH:/app/data}"
  compression: "gzip"  # none, gzip, brotli
  format: "json"  # json, pickle, parquet
  
  # Versioning
  versioning:
    enabled: true
    max_versions: 5
    compression_threshold_mb: 10
    
  # Backup configuration
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention_days: 30
    backup_path: "${CONTEXTER_BACKUP_PATH:/app/backups}"
    compression: "gzip"
    
  # Cleanup configuration
  cleanup:
    enabled: true
    schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
    max_age_days: 90
    min_free_space_gb: 10

# Monitoring and observability
monitoring:
  enabled: "${CONTEXTER_MONITORING_ENABLED:true}"
  
  # Metrics collection
  metrics:
    enabled: true
    port: "${CONTEXTER_METRICS_PORT:8080}"
    path: "/metrics"
    collection_interval_seconds: 15
    
    # Prometheus configuration
    prometheus:
      enabled: true
      pushgateway_url: "${PROMETHEUS_PUSHGATEWAY_URL:}"
      job_name: "contexter-rag"
      push_interval_seconds: 60
      
  # Health checks
  health:
    enabled: true
    endpoint: "/health"
    check_interval_seconds: 30
    timeout_seconds: 10
    
    # Component health checks
    components:
      vector_store: true
      embedding_service: true
      storage: true
      cache: true
      
  # Distributed tracing
  tracing:
    enabled: "${CONTEXTER_TRACING_ENABLED:false}"
    jaeger:
      agent_host: "${JAEGER_AGENT_HOST:localhost}"
      agent_port: "${JAEGER_AGENT_PORT:6831}"
      service_name: "contexter-rag"
      sampling_rate: 0.1
      
  # Alerting
  alerting:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL:}"
    alert_cooldown_minutes: 15
    
    # Alert rules
    rules:
      high_error_rate:
        threshold: 0.05
        window_minutes: 5
      high_latency:
        threshold_p95_ms: 100
        window_minutes: 5
      low_throughput:
        threshold_per_minute: 100
        window_minutes: 10

# Security configuration
security:
  # Authentication
  authentication:
    enabled: true
    api_key_required: true
    jwt_enabled: true
    jwt_secret: "${JWT_SECRET}"
    jwt_expiry_hours: 24
    
  # Authorization
  authorization:
    enabled: true
    default_role: "user"
    admin_api_key: "${ADMIN_API_KEY:}"
    
  # TLS/SSL
  tls:
    enabled: false
    cert_file: "${TLS_CERT_FILE:}"
    key_file: "${TLS_KEY_FILE:}"
    ca_file: "${TLS_CA_FILE:}"
    
  # Data protection
  data_protection:
    encrypt_at_rest: false
    encryption_key: "${ENCRYPTION_KEY:}"
    hash_algorithm: "sha256"
    
  # Request validation
  validation:
    max_request_size_mb: 100
    enable_input_sanitization: true
    enable_sql_injection_protection: true
    enable_xss_protection: true

# Feature flags
features:
  # Search features
  hybrid_search: "${CONTEXTER_HYBRID_SEARCH_ENABLED:true}"
  semantic_search: true
  keyword_search: true
  query_suggestions: true
  similar_documents: true
  
  # Ingestion features
  auto_ingestion: "${CONTEXTER_AUTO_INGESTION_ENABLED:true}"
  batch_processing: true
  priority_queue: true
  quality_validation: true
  
  # Performance features
  caching: true
  compression: true
  async_processing: true
  circuit_breakers: true
  
  # Monitoring features
  metrics_collection: "${CONTEXTER_MONITORING_ENABLED:true}"
  health_checks: true
  performance_tracking: true
  cost_tracking: true
  
  # Experimental features
  experimental_reranking: false
  experimental_embedding_models: false
  experimental_vector_quantization: false

# Development configuration
development:
  enabled: "${CONTEXTER_DEV_MODE:false}"
  hot_reload: "${CONTEXTER_HOT_RELOAD:false}"
  debug_endpoints: true
  profiling: "${CONTEXTER_PROFILING_ENABLED:false}"
  mock_external_services: false
  test_data_enabled: false
  
  # Development overrides
  overrides:
    log_level: "DEBUG"
    cache_ttl_seconds: 60
    rate_limit_requests_per_minute: 10000
    embedding_batch_size: 10
```

### Environment-Specific Configuration Files

#### Development Configuration (`config.development.yaml`)

```yaml
# Development environment overrides
environment: "development"

system:
  logging:
    level: "DEBUG"
    format: "text"
    
api:
  cors:
    allowed_origins: ["*"]
    
search:
  cache:
    ttl_seconds: 60
    
embedding:
  batch:
    size: 10
  cache:
    ttl_days: 1
    
monitoring:
  metrics:
    collection_interval_seconds: 5
    
development:
  enabled: true
  hot_reload: true
  debug_endpoints: true
  mock_external_services: true
```

#### Production Configuration (`config.production.yaml`)

```yaml
# Production environment overrides
environment: "production"

system:
  logging:
    level: "INFO"
    format: "json"
    
api:
  cors:
    allowed_origins: 
      - "https://app.contexter.dev"
      - "https://docs.contexter.dev"
      
security:
  authentication:
    enabled: true
    api_key_required: true
  tls:
    enabled: true
    
monitoring:
  tracing:
    enabled: true
    sampling_rate: 0.01
    
features:
  experimental_reranking: false
  experimental_embedding_models: false
```

## Feature Flags

### Dynamic Feature Toggle System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime, timedelta

class FeatureFlagProvider(ABC):
    """Abstract base class for feature flag providers."""
    
    @abstractmethod
    async def get_flag(self, flag_name: str, default: Any = False) -> Any:
        """Get feature flag value."""
        pass
    
    @abstractmethod
    async def set_flag(self, flag_name: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set feature flag value."""
        pass
    
    @abstractmethod
    async def list_flags(self) -> Dict[str, Any]:
        """List all feature flags."""
        pass

class FeatureFlagManager:
    """Centralized feature flag management."""
    
    def __init__(self, provider: FeatureFlagProvider, cache_ttl: int = 300):
        self.provider = provider
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
    
    async def is_enabled(self, flag_name: str, default: bool = False, context: Optional[Dict] = None) -> bool:
        """Check if a feature flag is enabled."""
        
        # Check cache first
        if self._is_cached(flag_name):
            return self.cache[flag_name]
        
        # Get from provider
        flag_value = await self.provider.get_flag(flag_name, default)
        
        # Apply context-based rules if provided
        if context and isinstance(flag_value, dict):
            flag_value = self._evaluate_context_rules(flag_value, context)
        
        # Cache the result
        self.cache[flag_name] = flag_value
        self.cache_timestamps[flag_name] = datetime.utcnow()
        
        return flag_value
    
    async def get_flag_value(self, flag_name: str, default: Any = None) -> Any:
        """Get feature flag value (not just boolean)."""
        
        if self._is_cached(flag_name):
            return self.cache[flag_name]
        
        flag_value = await self.provider.get_flag(flag_name, default)
        
        self.cache[flag_name] = flag_value
        self.cache_timestamps[flag_name] = datetime.utcnow()
        
        return flag_value
    
    def _is_cached(self, flag_name: str) -> bool:
        """Check if flag is cached and not expired."""
        if flag_name not in self.cache:
            return False
        
        cached_at = self.cache_timestamps.get(flag_name)
        if not cached_at:
            return False
        
        age = (datetime.utcnow() - cached_at).total_seconds()
        return age < self.cache_ttl
    
    def _evaluate_context_rules(self, flag_config: Dict, context: Dict) -> bool:
        """Evaluate context-based feature flag rules."""
        
        # Default value
        default_enabled = flag_config.get("enabled", False)
        
        # User-based rules
        if "user_rules" in flag_config:
            user_id = context.get("user_id")
            if user_id in flag_config["user_rules"].get("whitelist", []):
                return True
            if user_id in flag_config["user_rules"].get("blacklist", []):
                return False
        
        # Percentage rollout
        if "rollout_percentage" in flag_config:
            user_id = context.get("user_id", "")
            rollout_percentage = flag_config["rollout_percentage"]
            user_hash = hash(user_id) % 100
            if user_hash < rollout_percentage:
                return True
        
        # Environment-based rules
        if "environment_rules" in flag_config:
            environment = context.get("environment")
            env_rules = flag_config["environment_rules"]
            if environment in env_rules:
                return env_rules[environment]
        
        return default_enabled

# Feature flag definitions
FEATURE_FLAGS = {
    # Search features
    "search.hybrid_enabled": {
        "description": "Enable hybrid search combining semantic and keyword",
        "default": True,
        "category": "search"
    },
    "search.query_suggestions": {
        "description": "Enable query suggestions and autocomplete",
        "default": True,
        "category": "search"
    },
    "search.result_reranking": {
        "description": "Enable advanced result reranking",
        "default": True,
        "category": "search"
    },
    "search.semantic_caching": {
        "description": "Enable semantic similarity caching",
        "default": True,
        "category": "search"
    },
    
    # Ingestion features
    "ingestion.auto_processing": {
        "description": "Enable automatic document processing",
        "default": True,
        "category": "ingestion"
    },
    "ingestion.quality_validation": {
        "description": "Enable document quality validation",
        "default": True,
        "category": "ingestion"
    },
    "ingestion.batch_optimization": {
        "description": "Enable batch processing optimization",
        "default": True,
        "category": "ingestion"
    },
    
    # Performance features
    "performance.circuit_breakers": {
        "description": "Enable circuit breaker pattern for resilience",
        "default": True,
        "category": "performance"
    },
    "performance.adaptive_batching": {
        "description": "Enable adaptive batch size optimization",
        "default": False,
        "category": "performance"
    },
    "performance.memory_optimization": {
        "description": "Enable advanced memory optimization",
        "default": False,
        "category": "performance"
    },
    
    # Experimental features
    "experimental.new_embedding_model": {
        "description": "Test new embedding model (voyage-code-4)",
        "default": False,
        "category": "experimental",
        "rollout_percentage": 5
    },
    "experimental.vector_quantization": {
        "description": "Enable vector quantization for storage savings",
        "default": False,
        "category": "experimental"
    },
    "experimental.multi_language_support": {
        "description": "Enable multi-language document processing",
        "default": False,
        "category": "experimental"
    }
}
```

## Performance Tuning

### Performance Configuration Schema

```yaml
# Performance tuning configuration
performance:
  # Target performance metrics
  targets:
    search_latency_p95_ms: 50
    search_latency_p99_ms: 100
    ingestion_throughput_docs_per_minute: 1000
    api_response_time_p95_ms: 100
    system_availability_percent: 99.9
  
  # Memory configuration
  memory:
    max_heap_size_gb: 8
    gc_threshold: 0.8
    enable_jemalloc: true
    
    # Component memory limits
    limits:
      vector_store_gb: 4
      embedding_cache_gb: 2
      search_cache_mb: 512
      ingestion_buffer_mb: 1024
  
  # CPU configuration
  cpu:
    max_worker_threads: 16
    io_thread_pool_size: 32
    embedding_worker_threads: 4
    search_worker_threads: 8
    
  # I/O optimization
  io:
    async_enabled: true
    connection_pool_size: 100
    connection_timeout_seconds: 30
    read_timeout_seconds: 60
    write_timeout_seconds: 30
    
  # Caching strategies
  caching:
    # Search result cache
    search_cache:
      enabled: true
      max_size: 10000
      ttl_seconds: 3600
      eviction_policy: "lru"
      
    # Embedding cache
    embedding_cache:
      enabled: true
      max_size_gb: 10
      ttl_days: 7
      compression_enabled: true
      
    # Query vector cache
    query_cache:
      enabled: true
      max_size: 1000
      ttl_seconds: 1800
      
  # Batch processing optimization
  batching:
    # Embedding generation
    embedding:
      batch_size: 100
      max_concurrent_batches: 5
      adaptive_sizing: true
      
    # Vector storage
    vector_storage:
      batch_size: 1000
      flush_interval_seconds: 5
      
    # Document processing
    document_processing:
      chunk_batch_size: 50
      processing_timeout_seconds: 300
      
  # Connection pooling
  connection_pools:
    qdrant:
      min_connections: 5
      max_connections: 50
      acquire_timeout_seconds: 30
      
    redis:
      min_connections: 2
      max_connections: 20
      
    http:
      min_connections: 10
      max_connections: 100
      
  # Circuit breaker configuration
  circuit_breakers:
    vector_store:
      failure_threshold: 10
      recovery_timeout_seconds: 60
      success_threshold: 5
      
    embedding_service:
      failure_threshold: 5
      recovery_timeout_seconds: 30
      success_threshold: 3
```

## Security Settings

### Security Configuration Schema

```yaml
# Security configuration
security:
  # General security settings
  general:
    security_headers_enabled: true
    content_security_policy: "default-src 'self'"
    x_frame_options: "DENY"
    x_content_type_options: "nosniff"
    
  # Authentication configuration
  authentication:
    # API key authentication
    api_key:
      enabled: true
      header_name: "X-API-Key"
      query_param_name: "api_key"
      min_length: 32
      max_age_days: 365
      
    # JWT authentication
    jwt:
      enabled: true
      secret: "${JWT_SECRET}"
      algorithm: "HS256"
      expiry_hours: 24
      refresh_enabled: true
      refresh_expiry_days: 7
      issuer: "contexter-rag"
      audience: "contexter-api"
      
    # OAuth2 configuration
    oauth2:
      enabled: false
      providers:
        google:
          client_id: "${GOOGLE_CLIENT_ID:}"
          client_secret: "${GOOGLE_CLIENT_SECRET:}"
        github:
          client_id: "${GITHUB_CLIENT_ID:}"
          client_secret: "${GITHUB_CLIENT_SECRET:}"
  
  # Authorization configuration
  authorization:
    enabled: true
    default_role: "user"
    
    # Role-based access control
    roles:
      admin:
        permissions:
          - "documents:*"
          - "search:*"
          - "config:*"
          - "metrics:*"
          - "admin:*"
      user:
        permissions:
          - "documents:read"
          - "search:query"
          - "search:suggest"
      readonly:
        permissions:
          - "search:query"
          
    # API endpoint permissions
    endpoints:
      "/api/v1/search":
        required_permissions: ["search:query"]
      "/api/v1/documents":
        GET: ["documents:read"]
        POST: ["documents:write"]
        DELETE: ["documents:delete"]
      "/api/v1/admin/*":
        required_permissions: ["admin:*"]
        
  # Rate limiting configuration
  rate_limiting:
    enabled: true
    
    # Global rate limits
    global:
      requests_per_minute: 1000
      burst_size: 100
      
    # Per-user rate limits
    per_user:
      requests_per_minute: 100
      burst_size: 20
      
    # Per-endpoint rate limits
    endpoints:
      "/api/v1/search":
        requests_per_minute: 60
        burst_size: 10
      "/api/v1/documents":
        requests_per_minute: 20
        burst_size: 5
        
  # Input validation and sanitization
  validation:
    enabled: true
    
    # Request size limits
    max_request_size_mb: 100
    max_json_depth: 10
    max_array_length: 1000
    max_string_length: 10000
    
    # Content filtering
    enable_sql_injection_protection: true
    enable_xss_protection: true
    enable_path_traversal_protection: true
    allowed_file_types: ["json", "txt", "md"]
    
    # Parameter validation
    strict_parameter_validation: true
    require_content_type_header: true
    
  # Data encryption
  encryption:
    # Encryption at rest
    at_rest:
      enabled: false
      algorithm: "AES-256-GCM"
      key: "${ENCRYPTION_KEY:}"
      key_rotation_days: 90
      
    # Encryption in transit
    in_transit:
      tls_enabled: false
      tls_version: "1.3"
      cert_file: "${TLS_CERT_FILE:}"
      key_file: "${TLS_KEY_FILE:}"
      ca_file: "${TLS_CA_FILE:}"
      
  # Audit logging
  audit:
    enabled: true
    log_file: "/app/logs/audit.log"
    log_format: "json"
    
    # Events to audit
    events:
      authentication: true
      authorization: true
      data_access: true
      configuration_changes: true
      admin_actions: true
      
  # Security monitoring
  monitoring:
    enabled: true
    
    # Intrusion detection
    intrusion_detection:
      enabled: true
      max_failed_attempts: 5
      lockout_duration_minutes: 15
      
    # Anomaly detection
    anomaly_detection:
      enabled: true
      baseline_window_hours: 24
      deviation_threshold: 3.0
      
    # Security alerts
    alerts:
      enabled: true
      webhook_url: "${SECURITY_WEBHOOK_URL:}"
      alert_types:
        - "failed_authentication"
        - "rate_limit_exceeded"
        - "suspicious_activity"
        - "data_breach_attempt"
```

## Validation Rules

### Configuration Validation Framework

```python
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class ValidationRule:
    """Define a configuration validation rule."""
    field_path: str
    rule_type: str
    parameters: Dict[str, Any]
    error_message: str
    severity: str = "error"  # error, warning, info

class ConfigurationValidator:
    """Comprehensive configuration validation."""
    
    VALIDATION_RULES = [
        # System configuration rules
        ValidationRule(
            "system.logging.level",
            "choice",
            {"choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
            "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        ),
        ValidationRule(
            "system.runtime.max_workers",
            "range",
            {"min": 1, "max": 100},
            "Max workers must be between 1 and 100"
        ),
        
        # API configuration rules
        ValidationRule(
            "api.port",
            "range",
            {"min": 1024, "max": 65535},
            "API port must be between 1024 and 65535"
        ),
        ValidationRule(
            "api.timeout",
            "range",
            {"min": 1, "max": 300},
            "API timeout must be between 1 and 300 seconds"
        ),
        
        # Search configuration rules
        ValidationRule(
            "search.hybrid_weights.semantic",
            "range",
            {"min": 0.0, "max": 1.0},
            "Semantic weight must be between 0.0 and 1.0"
        ),
        ValidationRule(
            "search.hybrid_weights.keyword",
            "range",
            {"min": 0.0, "max": 1.0},
            "Keyword weight must be between 0.0 and 1.0"
        ),
        ValidationRule(
            "search.max_results",
            "range",
            {"min": 1, "max": 1000},
            "Max results must be between 1 and 1000"
        ),
        
        # Embedding configuration rules
        ValidationRule(
            "embedding.model",
            "choice",
            {"choices": ["voyage-code-3"]},
            "Embedding model must be 'voyage-code-3'"
        ),
        ValidationRule(
            "embedding.batch.size",
            "range",
            {"min": 1, "max": 1000},
            "Embedding batch size must be between 1 and 1000"
        ),
        
        # Vector store configuration rules
        ValidationRule(
            "vector_store.collection.vector_size",
            "choice",
            {"choices": [2048]},
            "Vector size must be 2048 for Voyage AI compatibility"
        ),
        ValidationRule(
            "vector_store.collection.distance_metric",
            "choice",
            {"choices": ["cosine", "euclidean", "dot"]},
            "Distance metric must be cosine, euclidean, or dot"
        ),
        
        # Storage configuration rules
        ValidationRule(
            "storage.compression",
            "choice",
            {"choices": ["none", "gzip", "brotli"]},
            "Compression must be none, gzip, or brotli"
        ),
        ValidationRule(
            "storage.versioning.max_versions",
            "range",
            {"min": 1, "max": 100},
            "Max versions must be between 1 and 100"
        ),
        
        # Performance validation rules
        ValidationRule(
            "performance.targets.search_latency_p95_ms",
            "range",
            {"min": 1, "max": 5000},
            "Search latency target must be between 1ms and 5000ms"
        ),
        ValidationRule(
            "performance.memory.max_heap_size_gb",
            "range",
            {"min": 1, "max": 64},
            "Max heap size must be between 1GB and 64GB"
        ),
        
        # Security validation rules
        ValidationRule(
            "security.authentication.jwt.expiry_hours",
            "range",
            {"min": 1, "max": 8760},  # 1 hour to 1 year
            "JWT expiry must be between 1 hour and 1 year"
        ),
        ValidationRule(
            "security.rate_limiting.global.requests_per_minute",
            "range",
            {"min": 1, "max": 100000},
            "Global rate limit must be between 1 and 100000 requests per minute"
        )
    ]
    
    @classmethod
    def validate_configuration(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete configuration and return validation result."""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        for rule in cls.VALIDATION_RULES:
            result = cls._validate_field(config, rule)
            
            if not result["valid"]:
                if rule.severity == "error":
                    validation_result["errors"].append({
                        "field": rule.field_path,
                        "message": rule.error_message,
                        "value": result.get("value")
                    })
                    validation_result["valid"] = False
                elif rule.severity == "warning":
                    validation_result["warnings"].append({
                        "field": rule.field_path,
                        "message": rule.error_message,
                        "value": result.get("value")
                    })
                else:  # info
                    validation_result["info"].append({
                        "field": rule.field_path,
                        "message": rule.error_message,
                        "value": result.get("value")
                    })
        
        # Additional cross-field validations
        cls._validate_cross_field_rules(config, validation_result)
        
        return validation_result
    
    @classmethod
    def _validate_field(cls, config: Dict[str, Any], rule: ValidationRule) -> Dict[str, Any]:
        """Validate a single configuration field."""
        
        # Get field value using dot notation
        value = cls._get_nested_value(config, rule.field_path)
        
        if value is None:
            return {"valid": True}  # Optional field
        
        result = {"valid": True, "value": value}
        
        if rule.rule_type == "range":
            min_val = rule.parameters.get("min")
            max_val = rule.parameters.get("max")
            
            if isinstance(value, (int, float)):
                if min_val is not None and value < min_val:
                    result["valid"] = False
                if max_val is not None and value > max_val:
                    result["valid"] = False
            else:
                result["valid"] = False
                
        elif rule.rule_type == "choice":
            choices = rule.parameters.get("choices", [])
            if value not in choices:
                result["valid"] = False
                
        elif rule.rule_type == "regex":
            pattern = rule.parameters.get("pattern")
            if not re.match(pattern, str(value)):
                result["valid"] = False
                
        elif rule.rule_type == "path":
            path = Path(value)
            if rule.parameters.get("must_exist", False) and not path.exists():
                result["valid"] = False
            if rule.parameters.get("must_be_directory", False) and not path.is_dir():
                result["valid"] = False
                
        return result
    
    @classmethod
    def _get_nested_value(cls, config: Dict[str, Any], field_path: str) -> Any:
        """Get nested configuration value using dot notation."""
        keys = field_path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
                
        return value
    
    @classmethod
    def _validate_cross_field_rules(cls, config: Dict[str, Any], validation_result: Dict[str, Any]):
        """Validate rules that depend on multiple fields."""
        
        # Validate hybrid search weights sum to 1
        semantic_weight = cls._get_nested_value(config, "search.hybrid_weights.semantic")
        keyword_weight = cls._get_nested_value(config, "search.hybrid_weights.keyword")
        
        if semantic_weight is not None and keyword_weight is not None:
            if abs((semantic_weight + keyword_weight) - 1.0) > 0.001:
                validation_result["errors"].append({
                    "field": "search.hybrid_weights",
                    "message": "Semantic and keyword weights must sum to 1.0",
                    "value": {"semantic": semantic_weight, "keyword": keyword_weight}
                })
                validation_result["valid"] = False
        
        # Validate memory allocation doesn't exceed total
        max_heap = cls._get_nested_value(config, "performance.memory.max_heap_size_gb")
        vector_store_mem = cls._get_nested_value(config, "performance.memory.limits.vector_store_gb")
        embedding_cache_mem = cls._get_nested_value(config, "performance.memory.limits.embedding_cache_gb")
        
        if all(x is not None for x in [max_heap, vector_store_mem, embedding_cache_mem]):
            total_allocated = vector_store_mem + embedding_cache_mem
            if total_allocated > max_heap * 0.8:  # Leave 20% for other components
                validation_result["warnings"].append({
                    "field": "performance.memory.limits",
                    "message": f"Memory allocation ({total_allocated}GB) exceeds 80% of heap size ({max_heap}GB)",
                    "value": {"allocated": total_allocated, "heap": max_heap}
                })
        
        # Validate performance targets are realistic
        search_latency = cls._get_nested_value(config, "performance.targets.search_latency_p95_ms")
        api_response_time = cls._get_nested_value(config, "performance.targets.api_response_time_p95_ms")
        
        if search_latency and api_response_time and search_latency >= api_response_time:
            validation_result["warnings"].append({
                "field": "performance.targets",
                "message": "Search latency target should be less than API response time target",
                "value": {"search": search_latency, "api": api_response_time}
            })

class ConfigurationValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, validation_result: Dict[str, Any]):
        self.validation_result = validation_result
        error_messages = [error["message"] for error in validation_result["errors"]]
        super().__init__(f"Configuration validation failed: {'; '.join(error_messages)}")
```

## Configuration Management

### Configuration Loading and Management System

```python
import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class ConfigurationSource:
    """Configuration source metadata."""
    source_type: str  # file, environment, default
    source_path: Optional[str]
    priority: int
    last_loaded: datetime
    checksum: Optional[str] = None

class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, config_dir: Path, environment: str = "production"):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.config_cache = {}
        self.config_sources = []
        self.watchers = []
        self.logger = logging.getLogger(__name__)
        
    async def load_configuration(self) -> Dict[str, Any]:
        """Load configuration from all sources with proper precedence."""
        
        config = {}
        
        # 1. Load default configuration
        default_config = self._load_default_config()
        config.update(default_config)
        
        # 2. Load base configuration file
        base_config_path = self.config_dir / "config.yaml"
        if base_config_path.exists():
            base_config = self._load_yaml_file(base_config_path)
            config = self._deep_merge(config, base_config)
            
        # 3. Load environment-specific configuration
        env_config_path = self.config_dir / f"config.{self.environment}.yaml"
        if env_config_path.exists():
            env_config = self._load_yaml_file(env_config_path)
            config = self._deep_merge(config, env_config)
            
        # 4. Apply environment variable overrides
        env_overrides = self._extract_env_overrides()
        config = self._deep_merge(config, env_overrides)
        
        # 5. Validate configuration
        validation_result = ConfigurationValidator.validate_configuration(config)
        if not validation_result["valid"]:
            raise ConfigurationValidationError(validation_result)
        
        # 6. Process configuration (variable substitution, etc.)
        config = self._process_configuration(config)
        
        # 7. Cache configuration
        self.config_cache = config
        
        self.logger.info(f"Configuration loaded successfully for environment: {self.environment}")
        
        return config
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        
        return {
            "version": "2.0.0",
            "system": {
                "name": "contexter-rag",
                "logging": {
                    "level": "INFO",
                    "format": "json"
                },
                "runtime": {
                    "debug_mode": False,
                    "max_workers": 10,
                    "timezone": "UTC"
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "timeout": 30
            },
            "search": {
                "enabled": True,
                "hybrid_weights": {
                    "semantic": 0.7,
                    "keyword": 0.3
                },
                "default_threshold": 0.7,
                "max_results": 100
            },
            "ingestion": {
                "enabled": True,
                "auto_process": True,
                "chunking": {
                    "chunk_size": 1000,
                    "overlap_size": 200
                }
            },
            "embedding": {
                "provider": "voyage_ai",
                "model": "voyage-code-3",
                "batch": {
                    "size": 100,
                    "max_concurrent": 5
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "connection": {
                    "host": "localhost",
                    "port": 6333
                },
                "collection": {
                    "name": "contexter_docs",
                    "vector_size": 2048,
                    "distance_metric": "cosine"
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "port": 8080
                }
            }
        }
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                
            self.logger.info(f"Loaded configuration from: {file_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {file_path}: {e}")
            return {}
    
    def _extract_env_overrides(self) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables."""
        
        overrides = {}
        
        # Define environment variable to config path mappings
        env_mappings = {
            "CONTEXTER_LOG_LEVEL": "system.logging.level",
            "CONTEXTER_DEBUG": "system.runtime.debug_mode",
            "CONTEXTER_MAX_WORKERS": "system.runtime.max_workers",
            "CONTEXTER_HOST": "api.host",
            "CONTEXTER_PORT": "api.port",
            "CONTEXTER_SEARCH_THRESHOLD": "search.default_threshold",
            "CONTEXTER_CHUNK_SIZE": "ingestion.chunking.chunk_size",
            "CONTEXTER_EMBEDDING_BATCH_SIZE": "embedding.batch.size",
            "QDRANT_HOST": "vector_store.connection.host",
            "QDRANT_PORT": "vector_store.connection.port",
            "VOYAGE_API_KEY": "embedding.api.key",
            "JWT_SECRET": "security.authentication.jwt.secret"
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                converted_value = self._convert_env_value(value)
                self._set_nested_value(overrides, config_path, converted_value)
        
        return overrides
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        
        # Boolean conversion
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        elif value.lower() in ("false", "0", "no", "off"):
            return False
        
        # Numeric conversion
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set nested configuration value using dot notation."""
        
        keys = path.split(".")
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _process_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration with variable substitution and transformations."""
        
        # Variable substitution
        config = self._substitute_variables(config)
        
        # Path resolution
        config = self._resolve_paths(config)
        
        return config
    
    def _substitute_variables(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration."""
        
        if isinstance(config, dict):
            return {key: self._substitute_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_variables(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_string_variables(config)
        else:
            return config
    
    def _substitute_string_variables(self, value: str) -> str:
        """Substitute environment variables in string values."""
        
        # Pattern: ${VAR_NAME:default_value}
        import re
        
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replace_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_var, value)
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths in configuration."""
        
        path_fields = [
            "system.logging.file_path",
            "storage.base_path",
            "storage.backup.backup_path",
            "security.tls.cert_file",
            "security.tls.key_file"
        ]
        
        for field_path in path_fields:
            value = self._get_nested_value(config, field_path)
            if value and isinstance(value, str):
                resolved_path = Path(value).expanduser().resolve()
                self._set_nested_value(config, field_path, str(resolved_path))
        
        return config
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested configuration value using dot notation."""
        
        keys = path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    async def reload_configuration(self) -> Dict[str, Any]:
        """Reload configuration from all sources."""
        
        self.logger.info("Reloading configuration...")
        return await self.load_configuration()
    
    def get_config(self, path: Optional[str] = None) -> Any:
        """Get configuration value by path."""
        
        if not self.config_cache:
            raise RuntimeError("Configuration not loaded. Call load_configuration() first.")
        
        if path is None:
            return self.config_cache
        
        return self._get_nested_value(self.config_cache, path)
    
    async def update_config(self, path: str, value: Any, persist: bool = False) -> bool:
        """Update configuration value at runtime."""
        
        if not self.config_cache:
            raise RuntimeError("Configuration not loaded.")
        
        # Validate the update
        temp_config = self.config_cache.copy()
        self._set_nested_value(temp_config, path, value)
        
        validation_result = ConfigurationValidator.validate_configuration(temp_config)
        if not validation_result["valid"]:
            raise ConfigurationValidationError(validation_result)
        
        # Apply the update
        self._set_nested_value(self.config_cache, path, value)
        
        # Persist to file if requested
        if persist:
            await self._persist_config_change(path, value)
        
        self.logger.info(f"Configuration updated: {path} = {value}")
        
        # Notify watchers
        await self._notify_config_watchers(path, value)
        
        return True
    
    async def _persist_config_change(self, path: str, value: Any):
        """Persist configuration change to file."""
        
        # This would typically update the appropriate config file
        # Implementation depends on specific requirements
        pass
    
    async def _notify_config_watchers(self, path: str, value: Any):
        """Notify registered configuration watchers of changes."""
        
        for watcher in self.watchers:
            try:
                await watcher(path, value)
            except Exception as e:
                self.logger.error(f"Config watcher error: {e}")
    
    def register_watcher(self, callback):
        """Register a configuration change watcher."""
        
        self.watchers.append(callback)
    
    def export_config(self, format: str = "yaml", include_secrets: bool = False) -> str:
        """Export current configuration as string."""
        
        config = self.config_cache.copy()
        
        if not include_secrets:
            config = self._redact_secrets(config)
        
        if format == "yaml":
            return yaml.dump(config, default_flow_style=False, indent=2)
        elif format == "json":
            return json.dumps(config, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _redact_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from configuration."""
        
        secret_patterns = [
            "key", "secret", "password", "token", "credential"
        ]
        
        def redact_recursive(obj):
            if isinstance(obj, dict):
                return {
                    key: "***REDACTED***" if any(pattern in key.lower() for pattern in secret_patterns)
                    else redact_recursive(value)
                    for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            else:
                return obj
        
        return redact_recursive(config)
```

This comprehensive configuration specification provides:

1. **Hierarchical Configuration**: Multi-layered configuration with proper precedence
2. **Environment Variables**: Complete environment variable definitions with validation
3. **Configuration Files**: YAML-based configuration with environment-specific overrides
4. **Feature Flags**: Dynamic feature toggle system with context-based rules
5. **Performance Tuning**: Detailed performance configuration parameters
6. **Security Settings**: Comprehensive security configuration options
7. **Validation Framework**: Extensive validation rules and cross-field validation
8. **Configuration Management**: Runtime configuration loading, updating, and persistence

The specification ensures that all aspects of the RAG system can be configured safely and efficiently, with proper validation and environment-specific customization capabilities.