"""
Embedding Service Configuration Management

Centralized configuration management for the RAG embedding service with
validation, environment variable support, and configuration templates.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator, ValidationError

logger = logging.getLogger(__name__)


class VoyageAIConfig(BaseModel):
    """Configuration for Voyage AI integration."""
    
    api_key: str = Field(..., description="Voyage AI API key")
    model: str = Field(default="voyage-code-3", description="Embedding model to use")
    base_url: str = Field(default="https://api.voyageai.com/v1", description="API base URL")
    timeout: float = Field(default=30.0, ge=1.0, le=300.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    
    # Rate limiting
    rate_limit_rpm: int = Field(default=300, ge=1, description="Requests per minute limit")
    rate_limit_tpm: int = Field(default=1000000, ge=1000, description="Tokens per minute limit")
    max_concurrent_requests: int = Field(default=10, ge=1, le=50, description="Max concurrent requests")
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1, description="Failures before circuit opens")
    circuit_breaker_recovery_timeout: int = Field(default=60, ge=10, description="Recovery timeout in seconds")
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v.strip()
    
    @validator('model')
    def validate_model(cls, v):
        supported_models = ["voyage-code-3", "voyage-2", "voyage-large-2"]
        if v not in supported_models:
            logger.warning(f"Model {v} may not be supported. Supported models: {supported_models}")
        return v


class CacheConfig(BaseModel):
    """Configuration for embedding caching."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    cache_path: str = Field(default="~/.contexter/embedding_cache.db", description="Cache database path")
    max_entries: int = Field(default=100000, ge=1000, description="Maximum cache entries")
    ttl_hours: int = Field(default=168, ge=1, description="Time-to-live in hours")
    cleanup_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Cleanup threshold (0.0-1.0)")
    enable_wal_mode: bool = Field(default=True, description="Enable SQLite WAL mode")
    
    @validator('cache_path')
    def expand_cache_path(cls, v):
        return str(Path(v).expanduser().resolve())


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing."""
    
    default_batch_size: int = Field(default=100, ge=1, le=500, description="Default batch size")
    max_batch_size: int = Field(default=200, ge=10, le=500, description="Maximum batch size")
    min_batch_size: int = Field(default=10, ge=1, le=100, description="Minimum batch size")
    max_concurrent_batches: int = Field(default=5, ge=1, le=20, description="Max concurrent batches")
    batch_timeout_seconds: float = Field(default=30.0, ge=1.0, description="Batch timeout")
    queue_timeout_seconds: float = Field(default=300.0, ge=10.0, description="Queue timeout")
    adaptive_batching: bool = Field(default=True, description="Enable adaptive batch sizing")
    auto_optimize: bool = Field(default=True, description="Enable automatic optimization")
    priority_boost_factor: float = Field(default=1.5, ge=1.0, description="Priority boost factor")


class PerformanceConfig(BaseModel):
    """Configuration for performance targets and monitoring."""
    
    target_throughput_per_minute: int = Field(default=1000, ge=100, description="Target throughput")
    target_cache_hit_rate: float = Field(default=0.5, ge=0.0, le=1.0, description="Target cache hit rate")
    max_error_rate: float = Field(default=0.001, ge=0.0, le=1.0, description="Maximum error rate")
    max_p95_latency_seconds: float = Field(default=2.0, ge=0.1, description="Max 95th percentile latency")
    max_cost_per_document: float = Field(default=0.001, ge=0.0, description="Max cost per document")
    
    # Monitoring
    enable_detailed_logging: bool = Field(default=True, description="Enable detailed logging")
    enable_performance_tracking: bool = Field(default=True, description="Enable performance tracking")
    enable_cost_tracking: bool = Field(default=True, description="Enable cost tracking")
    monitoring_interval_seconds: int = Field(default=60, ge=10, description="Monitoring interval")


class EmbeddingServiceConfig(BaseModel):
    """Complete configuration for the embedding service."""
    
    voyage_ai: VoyageAIConfig
    cache: CacheConfig = Field(default_factory=CacheConfig)
    batch_processing: BatchProcessingConfig = Field(default_factory=BatchProcessingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Global settings
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()


class ConfigManager:
    """Configuration manager with environment variable support and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._config: Optional[EmbeddingServiceConfig] = None
    
    def load_config(
        self,
        config_path: Optional[str] = None,
        from_env: bool = True,
        validate: bool = True
    ) -> EmbeddingServiceConfig:
        """
        Load configuration from file and environment variables.
        
        Args:
            config_path: Path to configuration file
            from_env: Whether to override with environment variables
            validate: Whether to validate configuration
            
        Returns:
            Validated configuration object
        """
        config_data = {}
        
        # Load from file if provided
        if config_path or self.config_path:
            file_path = config_path or self.config_path
            config_data = self._load_from_file(file_path)
        
        # Override with environment variables
        if from_env:
            env_overrides = self._load_from_environment()
            config_data = self._merge_configs(config_data, env_overrides)
        
        # Apply defaults if needed
        if not config_data.get('voyage_ai', {}).get('api_key'):
            api_key = os.getenv('VOYAGE_API_KEY')
            if api_key:
                if 'voyage_ai' not in config_data:
                    config_data['voyage_ai'] = {}
                config_data['voyage_ai']['api_key'] = api_key
        
        # Validate and create config object
        try:
            self._config = EmbeddingServiceConfig(**config_data)
            
            if validate:
                self._validate_config()
            
            logger.info("Configuration loaded successfully")
            return self._config
            
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return {}
            
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {file_path}")
            return config_data or {}
            
        except Exception as e:
            logger.error(f"Failed to load configuration file {file_path}: {e}")
            return {}
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Environment variable mappings
        env_mappings = {
            # Voyage AI
            'VOYAGE_API_KEY': 'voyage_ai.api_key',
            'VOYAGE_MODEL': 'voyage_ai.model',
            'VOYAGE_TIMEOUT': 'voyage_ai.timeout',
            'VOYAGE_MAX_RETRIES': 'voyage_ai.max_retries',
            'VOYAGE_RATE_LIMIT_RPM': 'voyage_ai.rate_limit_rpm',
            'VOYAGE_RATE_LIMIT_TPM': 'voyage_ai.rate_limit_tpm',
            
            # Cache
            'EMBEDDING_CACHE_PATH': 'cache.cache_path',
            'EMBEDDING_CACHE_MAX_ENTRIES': 'cache.max_entries',
            'EMBEDDING_CACHE_TTL_HOURS': 'cache.ttl_hours',
            'EMBEDDING_CACHE_ENABLED': 'cache.enabled',
            
            # Batch processing
            'EMBEDDING_BATCH_SIZE': 'batch_processing.default_batch_size',
            'EMBEDDING_MAX_CONCURRENT_BATCHES': 'batch_processing.max_concurrent_batches',
            'EMBEDDING_ADAPTIVE_BATCHING': 'batch_processing.adaptive_batching',
            
            # Performance
            'EMBEDDING_TARGET_THROUGHPUT': 'performance.target_throughput_per_minute',
            'EMBEDDING_TARGET_CACHE_HIT_RATE': 'performance.target_cache_hit_rate',
            'EMBEDDING_MAX_ERROR_RATE': 'performance.max_error_rate',
            
            # Global
            'LOG_LEVEL': 'log_level',
            'ENVIRONMENT': 'environment'
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                value = self._convert_env_value(value)
                self._set_nested_value(env_config, config_path, value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self):
        """Perform additional configuration validation."""
        if not self._config:
            raise ValueError("No configuration loaded")
        
        # Check API key
        if not self._config.voyage_ai.api_key:
            raise ValueError(
                "Voyage AI API key is required. Set VOYAGE_API_KEY environment variable "
                "or include in configuration file."
            )
        
        # Check cache path permissions
        cache_path = Path(self._config.cache.cache_path)
        cache_dir = cache_path.parent
        
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ValueError(f"Cannot create cache directory: {cache_dir}")
        
        # Validate performance targets
        perf = self._config.performance
        if perf.target_cache_hit_rate > 1.0:
            raise ValueError("Target cache hit rate cannot exceed 100%")
        
        if perf.max_error_rate > 1.0:
            raise ValueError("Max error rate cannot exceed 100%")
        
        logger.info("Configuration validation passed")
    
    def get_config(self) -> EmbeddingServiceConfig:
        """Get the current configuration."""
        if not self._config:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    def save_config(self, file_path: str):
        """Save current configuration to file."""
        if not self._config:
            raise ValueError("No configuration to save")
        
        config_dict = self._config.dict()
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def create_template_config(self, file_path: str, environment: str = "development"):
        """Create a template configuration file."""
        template_config = {
            'voyage_ai': {
                'api_key': '${VOYAGE_API_KEY}',
                'model': 'voyage-code-3',
                'timeout': 30.0,
                'max_retries': 3,
                'rate_limit_rpm': 300,
                'rate_limit_tpm': 1000000,
                'max_concurrent_requests': 10
            },
            'cache': {
                'enabled': True,
                'cache_path': '~/.contexter/embedding_cache.db',
                'max_entries': 100000,
                'ttl_hours': 168,
                'cleanup_threshold': 0.8
            },
            'batch_processing': {
                'default_batch_size': 100,
                'max_concurrent_batches': 5,
                'adaptive_batching': True,
                'auto_optimize': True
            },
            'performance': {
                'target_throughput_per_minute': 1000,
                'target_cache_hit_rate': 0.5,
                'max_error_rate': 0.001,
                'enable_detailed_logging': environment == "development",
                'enable_performance_tracking': True
            },
            'log_level': 'DEBUG' if environment == "development" else 'INFO',
            'environment': environment
        }
        
        try:
            with open(file_path, 'w') as f:
                f.write("# Contexter RAG Embedding Service Configuration\n")
                f.write(f"# Environment: {environment}\n")
                f.write("# \n")
                f.write("# Set VOYAGE_API_KEY environment variable or replace ${VOYAGE_API_KEY}\n")
                f.write("# with your actual API key\n\n")
                
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Template configuration created at {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to create template configuration: {e}")
            raise


# Utility functions for easy configuration management

def load_config_from_env() -> EmbeddingServiceConfig:
    """Load configuration primarily from environment variables."""
    manager = ConfigManager()
    return manager.load_config(from_env=True)


def load_config_from_file(file_path: str) -> EmbeddingServiceConfig:
    """Load configuration from file with environment variable overrides."""
    manager = ConfigManager(file_path)
    return manager.load_config()


def create_development_config() -> EmbeddingServiceConfig:
    """Create a development configuration with sensible defaults."""
    config_data = {
        'voyage_ai': {
            'api_key': os.getenv('VOYAGE_API_KEY', ''),
            'model': 'voyage-code-3'
        },
        'environment': 'development',
        'log_level': 'DEBUG',
        'performance': {
            'enable_detailed_logging': True,
            'target_throughput_per_minute': 500  # Lower for development
        }
    }
    
    return EmbeddingServiceConfig(**config_data)


def create_production_config() -> EmbeddingServiceConfig:
    """Create a production configuration with optimized settings."""
    config_data = {
        'voyage_ai': {
            'api_key': os.getenv('VOYAGE_API_KEY', ''),
            'model': 'voyage-code-3',
            'max_concurrent_requests': 20  # Higher for production
        },
        'batch_processing': {
            'default_batch_size': 200,  # Larger batches
            'max_concurrent_batches': 10
        },
        'cache': {
            'max_entries': 500000,  # Larger cache
            'ttl_hours': 336  # 2 weeks
        },
        'environment': 'production',
        'log_level': 'INFO',
        'performance': {
            'enable_detailed_logging': False,
            'target_throughput_per_minute': 2000  # Higher target
        }
    }
    
    return EmbeddingServiceConfig(**config_data)