"""
C7DocDownloader - Context7 Documentation Downloader

A high-performance Python CLI application for comprehensive documentation retrieval
using intelligent proxy rotation and advanced deduplication.
"""

__version__ = "1.0.0"
__author__ = "C7DocDownloader Team"
__email__ = "team@c7doc.dev"

# Import main components for easy access
from .core.config_manager import ConfigurationManager
from .integration.proxy_manager import BrightDataProxyManager
from .models.config_models import C7DocConfig

# Other components will be imported as they are implemented
# from .core.download_engine import DownloadEngine
# from .core.storage import StorageManager
# from .integration.context7_client import Context7Client

__all__ = [
    "ConfigurationManager",
    "C7DocConfig",
    "BrightDataProxyManager",
    # "DownloadEngine",
    # "StorageManager",
    # "Context7Client",
]
