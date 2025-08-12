"""
RAG Embedding Service Usage Examples

Comprehensive examples demonstrating how to use the RAG embedding service
for various use cases including document ingestion, search, and monitoring.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

# Import the embedding service components
from contexter.vector.embedding_engine import VoyageEmbeddingEngine, create_embedding_engine
from contexter.vector.embedding_integration import EmbeddingVectorIntegration, create_embedding_integration
from contexter.vector.embedding_config import (
    ConfigManager, EmbeddingServiceConfig, create_development_config, create_production_config
)
from contexter.models.embedding_models import EmbeddingRequest, InputType
from contexter.models.storage_models import DocumentationChunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """Basic usage example - single embeddings and queries."""
    
    print("=== Basic Embedding Service Usage ===")
    
    # Create embedding engine with default configuration
    engine = await create_embedding_engine(
        voyage_api_key="your-voyage-api-key-here",  # Replace with actual key
        cache_path="./examples/cache.db",
        batch_size=50
    )
    
    try:
        # Generate single document embedding
        print("1. Generating single document embedding...")
        
        request = EmbeddingRequest(
            content="FastAPI is a modern, fast web framework for building APIs with Python 3.7+",
            input_type=InputType.DOCUMENT,
            metadata={"source": "FastAPI documentation", "section": "introduction"}
        )
        
        result = await engine.generate_embedding(request)
        
        if result.success:
            print(f"✓ Generated embedding with {result.dimensions} dimensions")
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  Cache hit: {result.cache_hit}")
        else:
            print(f"✗ Failed: {result.error}")
        
        # Generate query embedding for search
        print("\n2. Generating query embedding...")
        
        query = "How to create REST API endpoints?"
        query_embedding = await engine.embed_query(query)
        
        print(f"✓ Query embedding generated: {len(query_embedding)} dimensions")
        
        # Batch document embedding
        print("\n3. Batch document embedding...")
        
        documents = [
            "FastAPI supports automatic API documentation with Swagger UI",
            "Type hints are used for request/response validation in FastAPI",
            "AsyncIO is natively supported for high-performance async operations",
            "Dependency injection system makes code modular and testable"
        ]
        
        embeddings = await engine.embed_documents(
            documents,
            metadata_list=[
                {"section": "documentation"},
                {"section": "validation"}, 
                {"section": "async"},
                {"section": "dependencies"}
            ]
        )
        
        successful_embeddings = [e for e in embeddings if len(e) > 0]
        print(f"✓ Generated {len(successful_embeddings)}/{len(documents)} embeddings")
        
        # Check performance metrics
        metrics = engine.get_performance_metrics()
        print(f"\n📊 Performance Metrics:")
        print(f"  Total requests: {metrics.total_requests}")
        print(f"  Success rate: {metrics.success_rate:.1%}")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.1%}")
        print(f"  Throughput: {metrics.throughput_per_minute:.1f} docs/min")
        
    finally:
        await engine.shutdown()


async def example_configuration_management():
    """Example of configuration management and customization."""
    
    print("\n=== Configuration Management ===")
    
    # 1. Create development configuration
    print("1. Creating development configuration...")
    
    dev_config = create_development_config()
    print(f"✓ Development config created for environment: {dev_config.environment}")
    print(f"  Log level: {dev_config.log_level}")
    print(f"  Batch size: {dev_config.batch_processing.default_batch_size}")
    
    # 2. Create production configuration  
    print("\n2. Creating production configuration...")
    
    prod_config = create_production_config()
    print(f"✓ Production config created for environment: {prod_config.environment}")
    print(f"  Cache entries: {prod_config.cache.max_entries:,}")
    print(f"  Concurrent batches: {prod_config.batch_processing.max_concurrent_batches}")
    print(f"  Target throughput: {prod_config.performance.target_throughput_per_minute:,}/min")
    
    # 3. Load from configuration file
    print("\n3. Creating configuration template...")
    
    config_manager = ConfigManager()
    template_path = "./examples/embedding_config.yaml"
    
    # Create template
    config_manager.create_template_config(template_path, environment="development")
    print(f"✓ Template created at {template_path}")
    
    # Load from file
    loaded_config = config_manager.load_config(template_path, from_env=True)
    print(f"✓ Configuration loaded from file")
    print(f"  Model: {loaded_config.voyage_ai.model}")
    print(f"  Cache TTL: {loaded_config.cache.ttl_hours} hours")


async def example_document_ingestion_workflow():
    """Complete document ingestion workflow example."""
    
    print("\n=== Document Ingestion Workflow ===")
    
    # Mock some documentation chunks (in real usage, these come from document processing)
    chunks = [
        DocumentationChunk(
            chunk_id="fastapi_intro_1",
            content="FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            chunk_index=0,
            total_chunks=5,
            token_count=25,
            content_hash="hash_intro_1",
            source_context="FastAPI Introduction",
            doc_type="guide",
            section="introduction",
            programming_language="python"
        ),
        DocumentationChunk(
            chunk_id="fastapi_features_1",
            content="Key features include: Fast: Very high performance, on par with NodeJS and Go. Fast to code: Increase the speed to develop features by about 200% to 300%.",
            chunk_index=1,
            total_chunks=5,
            token_count=30,
            content_hash="hash_features_1",
            source_context="FastAPI Features",
            doc_type="guide",
            section="features",
            programming_language="python"
        ),
        DocumentationChunk(
            chunk_id="fastapi_validation_1",
            content="Automatic request validation using Pydantic models. Type hints provide automatic documentation and validation for request/response data.",
            chunk_index=2,
            total_chunks=5,
            token_count=22,
            content_hash="hash_validation_1",
            source_context="FastAPI Validation",
            doc_type="guide",
            section="validation",
            programming_language="python"
        )
    ]
    
    print(f"Ingesting {len(chunks)} documentation chunks...")
    
    # Create integration layer (in real usage, this would be initialized once)
    integration = await create_embedding_integration(
        voyage_api_key="your-voyage-api-key-here",  # Replace with actual key
        embedding_config_overrides={
            "cache_path": "./examples/integration_cache.db",
            "batch_size": 10
        }
    )
    
    try:
        # Progress callback for monitoring
        def progress_callback(current: int, total: int):
            progress = (current / total) * 100
            print(f"  Progress: {current}/{total} ({progress:.1f}%)")
        
        # Ingest documents
        result = await integration.ingest_documents(
            chunks=chunks,
            library_id="fastapi/fastapi",
            library_name="FastAPI",
            version="0.104.1",
            metadata={
                "trust_score": 9.8,
                "star_count": 84000,
                "language": "python"
            },
            progress_callback=progress_callback
        )
        
        print(f"\n📈 Ingestion Results:")
        print(f"  Total documents: {result.total_documents}")
        print(f"  Successful embeddings: {result.successful_embeddings}")
        print(f"  Cached embeddings: {result.cached_embeddings}")
        print(f"  Stored vectors: {result.stored_vectors}")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Success rate: {result.success_rate:.1%}")
        print(f"  Cache hit rate: {result.cache_hit_rate:.1%}")
        
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"    - {error}")
        
        # Test search functionality
        print(f"\n🔍 Testing search functionality...")
        
        search_results = await integration.search_similar_documents(
            query="How does FastAPI provide high performance?",
            top_k=3,
            filters={"programming_language": "python"}
        )
        
        print(f"Found {len(search_results)} similar documents:")
        for i, result in enumerate(search_results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Section: {result['section']}")
            print(f"     Content: {result['content'][:100]}...")
            print(f"     Reasons: {', '.join(result['match_reasons'])}")
        
    finally:
        await integration.shutdown()


async def example_performance_monitoring():
    """Example of performance monitoring and optimization."""
    
    print("\n=== Performance Monitoring ===")
    
    engine = await create_embedding_engine(
        voyage_api_key="your-voyage-api-key-here",  # Replace with actual key
        cache_path="./examples/perf_cache.db",
        enable_performance_tracking=True
    )
    
    try:
        # Generate some load for monitoring
        print("1. Generating load for performance monitoring...")
        
        # Create test requests
        test_requests = []
        for i in range(50):
            request = EmbeddingRequest(
                content=f"Test document {i} with various content for performance testing. "
                       f"This document contains information about topic {i % 10}.",
                input_type=InputType.DOCUMENT,
                metadata={"test_batch": True, "document_id": i}
            )
            test_requests.append(request)
        
        # Process in batches
        start_time = time.time()
        batch_result = await engine.generate_batch_embeddings(test_requests)
        processing_time = time.time() - start_time
        
        print(f"✓ Processed {len(test_requests)} requests in {processing_time:.2f}s")
        print(f"  Throughput: {(len(test_requests) / processing_time) * 60:.1f} docs/min")
        
        # Get comprehensive health check
        print("\n2. Health check results:")
        
        health_status = await engine.health_check()
        print(f"  Overall status: {health_status['status']}")
        
        if 'components' in health_status:
            for component, status in health_status['components'].items():
                comp_status = status.get('status', 'unknown')
                print(f"  {component}: {comp_status}")
        
        # Performance metrics
        if 'performance' in health_status:
            perf = health_status['performance']
            print(f"\n📊 Performance Metrics:")
            print(f"  Total requests: {perf.get('total_requests', 0):,}")
            print(f"  Success rate: {perf.get('success_rate', 0):.1%}")
            print(f"  Cache hit rate: {perf.get('cache_hit_rate', 0):.1%}")
            print(f"  Throughput: {perf.get('throughput_per_minute', 0):.1f} docs/min")
            print(f"  Avg latency: {perf.get('average_latency_ms', 0):.1f}ms")
        
        # Performance compliance
        if 'compliance' in health_status:
            compliance = health_status['compliance']
            print(f"\n✅ SLA Compliance:")
            for metric, compliant in compliance.items():
                status = "✓" if compliant else "✗"
                print(f"  {metric}: {status}")
        
        # Detailed status
        print("\n3. Detailed system status:")
        
        detailed_status = await engine.get_detailed_status()
        
        # Engine status
        engine_info = detailed_status.get('engine', {})
        print(f"  Engine initialized: {engine_info.get('initialized', False)}")
        print(f"  Model: {engine_info.get('config', {}).get('model', 'unknown')}")
        
        # Cache performance
        if 'metrics' in detailed_status:
            cache_metrics = detailed_status['metrics'].get('caching', {})
            cache_hits = cache_metrics.get('cache_hits', 0)
            cache_misses = cache_metrics.get('cache_misses', 0)
            total_cache_requests = cache_hits + cache_misses
            
            if total_cache_requests > 0:
                hit_rate = cache_hits / total_cache_requests
                print(f"  Cache efficiency: {hit_rate:.1%} ({cache_hits}/{total_cache_requests})")
        
        # Cost tracking
        cost_metrics = detailed_status.get('metrics', {}).get('cost', {})
        total_cost = cost_metrics.get('estimated_cost', 0)
        total_tokens = cost_metrics.get('total_tokens', 0)
        
        if total_tokens > 0:
            cost_per_1k = (total_cost / total_tokens) * 1000
            print(f"  Cost efficiency: ${cost_per_1k:.4f} per 1K tokens")
        
    finally:
        await engine.shutdown()


async def example_error_handling_and_recovery():
    """Example of error handling and recovery scenarios."""
    
    print("\n=== Error Handling and Recovery ===")
    
    # Create engine with invalid API key to demonstrate error handling
    try:
        engine = await create_embedding_engine(
            voyage_api_key="invalid-api-key",  # Intentionally invalid
            cache_path="./examples/error_cache.db"
        )
        
        print("1. Testing error handling with invalid API key...")
        
        # This should fail gracefully
        request = EmbeddingRequest(
            content="Test content for error handling",
            input_type=InputType.DOCUMENT
        )
        
        result = await engine.generate_embedding(request)
        
        if not result.success:
            print(f"✓ Error handled gracefully: {result.error}")
        else:
            print("✗ Expected error but got success")
        
        # Check if circuit breaker is working
        health_status = await engine.health_check()
        print(f"  Health status after error: {health_status['status']}")
        
        await engine.shutdown()
        
    except Exception as e:
        print(f"✓ Exception properly caught: {e}")
    
    print("\n2. Testing cache resilience...")
    
    # Test with valid configuration but demonstrate cache handling
    engine = await create_embedding_engine(
        voyage_api_key="your-voyage-api-key-here",  # Replace with actual key
        cache_path="./examples/resilience_cache.db"
    )
    
    try:
        # Generate some cached content
        request = EmbeddingRequest(
            content="Content that will be cached for resilience testing",
            input_type=InputType.DOCUMENT
        )
        
        # First request (will be cached)
        result1 = await engine.generate_embedding(request)
        print(f"  First request cache hit: {result1.cache_hit}")
        
        # Second request (should hit cache)
        result2 = await engine.generate_embedding(request)
        print(f"  Second request cache hit: {result2.cache_hit}")
        
        # Verify cache is working
        if not result1.cache_hit and result2.cache_hit:
            print("✓ Cache resilience verified")
        else:
            print("? Cache behavior may need investigation")
        
    finally:
        await engine.shutdown()


async def example_production_deployment():
    """Example of production deployment configuration and monitoring."""
    
    print("\n=== Production Deployment Example ===")
    
    # Production configuration
    config_manager = ConfigManager()
    
    # Create production config template
    prod_config_path = "./examples/production_config.yaml"
    config_manager.create_template_config(prod_config_path, environment="production")
    
    print(f"1. Created production config template: {prod_config_path}")
    
    # Load production configuration
    prod_config = create_production_config()
    
    print("2. Production configuration settings:")
    print(f"   Environment: {prod_config.environment}")
    print(f"   Log level: {prod_config.log_level}")
    print(f"   Cache entries: {prod_config.cache.max_entries:,}")
    print(f"   Batch size: {prod_config.batch_processing.default_batch_size}")
    print(f"   Concurrent batches: {prod_config.batch_processing.max_concurrent_batches}")
    print(f"   Target throughput: {prod_config.performance.target_throughput_per_minute:,}/min")
    print(f"   Target cache hit rate: {prod_config.performance.target_cache_hit_rate:.1%}")
    
    print("\n3. Production monitoring checklist:")
    monitoring_items = [
        "✓ Health check endpoint configured",
        "✓ Performance metrics collection enabled", 
        "✓ Error rate monitoring with alerting",
        "✓ Cache hit rate tracking",
        "✓ Cost monitoring and budget alerts",
        "✓ Circuit breaker status monitoring",
        "✓ Rate limiting compliance tracking",
        "✓ Log aggregation configured",
        "✓ Resource usage monitoring (CPU, memory)",
        "✓ Backup and recovery procedures"
    ]
    
    for item in monitoring_items:
        print(f"   {item}")
    
    print("\n4. Recommended deployment workflow:")
    deployment_steps = [
        "1. Deploy to staging environment",
        "2. Run comprehensive integration tests",
        "3. Perform load testing with expected traffic",
        "4. Validate performance targets are met", 
        "5. Test failover and recovery scenarios",
        "6. Deploy to production with blue-green strategy",
        "7. Monitor metrics for 24 hours post-deployment",
        "8. Scale resources based on actual usage patterns"
    ]
    
    for step in deployment_steps:
        print(f"   {step}")


async def main():
    """Run all examples."""
    
    print("🚀 RAG Embedding Service Examples")
    print("=" * 50)
    
    # Note: These examples require a valid Voyage AI API key
    # Replace "your-voyage-api-key-here" with your actual API key
    
    try:
        # Run examples (comment out sections if API key not available)
        await example_basic_usage()
        await example_configuration_management()
        # await example_document_ingestion_workflow()  # Requires valid API key
        # await example_performance_monitoring()       # Requires valid API key
        await example_error_handling_and_recovery()
        await example_production_deployment()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Example execution failed")


if __name__ == "__main__":
    # Create examples directory
    Path("./examples").mkdir(exist_ok=True)
    
    # Run examples
    asyncio.run(main())