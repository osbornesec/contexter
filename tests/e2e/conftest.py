"""
E2E test configuration and fixtures.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from contexter.core.config_manager import ConfigurationManager
from contexter.models.config_models import (
    C7DocConfig, 
    ProxyConfig, 
    DownloadConfig, 
    StorageConfig, 
    LoggingConfig
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(temp_storage_dir: Path) -> C7DocConfig:
    """Create a test configuration."""
    return C7DocConfig(
        proxy=ProxyConfig(
            customer_id="test-customer-123",
            zone_name="zone_residential_1",
            dns_resolution="remote",
            pool_size=5,
            health_check_interval=60,
        ),
        download=DownloadConfig(
            default_token_limit=10000,
            max_concurrent=5,
            retry_count=2,
            timeout_seconds=30.0,
            jitter_max=2.0,
        ),
        storage=StorageConfig(
            base_path=str(temp_storage_dir),
            compression_level=6,
            retention_limit=3,
        ),
        logging=LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_path=str(temp_storage_dir / "c7doc.log"),
        ),
    )


@pytest.fixture
def mock_proxy_manager():
    """Mock BrightData proxy manager."""
    mock_manager = AsyncMock()
    mock_manager.test_connectivity.return_value = True
    mock_manager.initialize_pool = AsyncMock()
    mock_manager.get_connection = AsyncMock()
    mock_manager.report_success = AsyncMock()
    mock_manager.report_failure = AsyncMock()
    mock_manager.shutdown = AsyncMock()
    
    # Mock connection
    mock_connection = MagicMock()
    mock_connection.proxy_id = "test-proxy-1"
    mock_connection.session = AsyncMock()
    mock_manager.get_connection.return_value = mock_connection
    
    return mock_manager


@pytest.fixture
def mock_context7_client():
    """Mock Context7 client."""
    mock_client = AsyncMock()
    mock_client.test_connectivity.return_value = True
    mock_client.health_check.return_value = True
    
    # Mock response data
    mock_response = MagicMock()
    mock_response.content = "# Test Documentation\n\nThis is test documentation content."
    mock_response.token_count = 25
    mock_response.response_time = 0.5
    mock_response.metadata = {"test": True}
    
    mock_client.get_smart_docs.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_environment_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        "BRIGHTDATA_CUSTOMER_ID": "test-customer-123",
        "BRIGHTDATA_PASSWORD": "test-password-456",
        "CONTEXTER_CONFIG_PATH": "~/.contexter/config.yaml",
        "CONTEXTER_STORAGE_PATH": "~/.contexter/downloads",
    }):
        yield


@pytest.fixture
def sample_documentation_chunks():
    """Sample documentation chunks for testing."""
    from contexter.models.download_models import DocumentationChunk
    
    return [
        DocumentationChunk(
            chunk_id="test-chunk-1",
            content="# Introduction\n\nThis is the introduction to the library.",
            source_context="Getting started with the library",
            token_count=20,
            content_hash="abc123",
            proxy_id="test-proxy-1",
            download_time=0.5,
            library_id="test-lib",
            metadata={"section": "intro"}
        ),
        DocumentationChunk(
            chunk_id="test-chunk-2", 
            content="# API Reference\n\nDetailed API documentation.",
            source_context="API reference and methods",
            token_count=18,
            content_hash="def456",
            proxy_id="test-proxy-1",
            download_time=0.6,
            library_id="test-lib",
            metadata={"section": "api"}
        ),
        DocumentationChunk(
            chunk_id="test-chunk-3",
            content="# Examples\n\nCode examples and tutorials.",
            source_context="Examples and tutorials",
            token_count=22,
            content_hash="ghi789",
            proxy_id="test-proxy-2", 
            download_time=0.4,
            library_id="test-lib",
            metadata={"section": "examples"}
        ),
    ]


@pytest.fixture
def library_test_cases() -> Dict[str, Dict[str, Any]]:
    """Test cases for different library scenarios."""
    return {
        "simple_library": {
            "library_id": "test/simple-lib",
            "contexts": ["basic usage", "installation guide"],
            "expected_chunks": 2,
            "expected_tokens_min": 40,
        },
        "complex_library": {
            "library_id": "test/complex-lib", 
            "contexts": [
                "getting started",
                "API reference", 
                "advanced features",
                "configuration options",
                "troubleshooting"
            ],
            "expected_chunks": 5,
            "expected_tokens_min": 100,
        },
        "minimal_library": {
            "library_id": "test/minimal-lib",
            "contexts": ["quick start"],
            "expected_chunks": 1,
            "expected_tokens_min": 15,
        }
    }


# Add E2E marker
pytestmark = pytest.mark.e2e