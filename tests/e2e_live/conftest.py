"""
Live E2E test configuration and fixtures.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional
import pytest

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
def live_test_config(temp_storage_dir: Path) -> C7DocConfig:
    """Create a live test configuration."""
    return C7DocConfig(
        proxy=ProxyConfig(
            customer_id=os.getenv("BRIGHTDATA_CUSTOMER_ID", "test-customer-123"),
            zone_name="zone_residential_1",
            dns_resolution="remote",
            pool_size=10,  # Minimum allowed pool size
            health_check_interval=60,
        ),
        download=DownloadConfig(
            default_token_limit=2000,  # Smaller limit for tests
            max_concurrent=3,  # Reduced concurrency
            retry_count=1,  # Fewer retries
            timeout_seconds=30.0,
            jitter_max=1.0,
        ),
        storage=StorageConfig(
            base_path=str(temp_storage_dir),
            compression_level=6,
            retention_limit=2,  # Keep fewer versions
        ),
        logging=LoggingConfig(
            level="INFO",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            file_path=str(temp_storage_dir / "c7doc.log"),
        ),
    )


@pytest.fixture
def brightdata_credentials() -> Optional[dict[str, str]]:
    """Get BrightData credentials from environment."""
    customer_id = os.getenv("BRIGHTDATA_CUSTOMER_ID")
    password = os.getenv("BRIGHTDATA_PASSWORD")
    
    if not customer_id or not password:
        return None
    
    return {
        "customer_id": customer_id,
        "password": password
    }


@pytest.fixture
def test_libraries() -> list[dict[str, str]]:
    """Test libraries known to work well with Context7."""
    return [
        {
            "library_id": "facebook/react",
            "name": "React",
            "expected_contexts": 3
        },
        {
            "library_id": "microsoft/typescript", 
            "name": "TypeScript",
            "expected_contexts": 3
        },
        {
            "library_id": "nodejs/node",
            "name": "Node.js", 
            "expected_contexts": 3
        }
    ]


@pytest.fixture
def small_test_library() -> dict[str, str]:
    """A small, reliable test library."""
    return {
        "library_id": "facebook/react",
        "name": "React",
        "expected_contexts": 2
    }


def pytest_configure(config):
    """Configure pytest for live tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "live: marks tests as live tests requiring real services"
    )
    config.addinivalue_line(
        "markers", "proxy_required: marks tests requiring BrightData proxy"
    )
    config.addinivalue_line(
        "markers", "context7_required: marks tests requiring Context7 API"
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests if environment not configured."""
    
    # Check for required environment variables
    has_brightdata = bool(os.getenv("BRIGHTDATA_CUSTOMER_ID") and os.getenv("BRIGHTDATA_PASSWORD"))
    has_context7 = True  # Context7 doesn't require special auth
    
    skip_proxy = pytest.mark.skip(reason="BrightData credentials not configured")
    skip_context7 = pytest.mark.skip(reason="Context7 API not available")
    
    for item in items:
        if "proxy_required" in item.keywords and not has_brightdata:
            item.add_marker(skip_proxy)
        if "context7_required" in item.keywords and not has_context7:
            item.add_marker(skip_context7)


# Add live marker to all tests in this directory
pytestmark = pytest.mark.live