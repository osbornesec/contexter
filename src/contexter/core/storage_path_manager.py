"""
Directory structure management and path utilities for storage operations.
"""

import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.storage_models import VersionManagementError

logger = logging.getLogger(__name__)


class StoragePathManager:
    """Manages directory structure and path sanitization for storage operations."""

    def __init__(self, base_path: str = "~/.contexter/downloads"):
        """Initialize with base storage path."""
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Path sanitization patterns
        self._invalid_chars_pattern = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
        self._multiple_dots_pattern = re.compile(r"\.{2,}")
        self._multiple_underscores_pattern = re.compile(r"_{2,}")

        logger.debug(f"Initialized StoragePathManager with base path: {self.base_path}")

    def sanitize_library_id(self, library_id: str) -> str:
        """
        Sanitize library ID for filesystem usage.

        Args:
            library_id: Raw library identifier

        Returns:
            Sanitized library ID safe for filesystem paths
        """
        if not library_id or not library_id.strip():
            return "unknown_library"

        # Start with the original ID
        sanitized = library_id.strip()

        # Replace invalid filesystem characters with underscores
        sanitized = self._invalid_chars_pattern.sub("_", sanitized)

        # Replace path separators
        sanitized = sanitized.replace("/", "_").replace("\\", "_")

        # Replace other problematic characters
        replacements = {
            " ": "_",  # Spaces
            "\t": "_",  # Tabs
            "\n": "_",  # Newlines
            "\r": "_",  # Carriage returns
            "~": "_",  # Tilde (can be problematic in some contexts)
            "`": "_",  # Backticks
        }

        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

        # Collapse multiple dots and underscores
        sanitized = self._multiple_dots_pattern.sub(".", sanitized)
        sanitized = self._multiple_underscores_pattern.sub("_", sanitized)

        # Remove leading/trailing dots and underscores
        sanitized = sanitized.strip("._")

        # Ensure it's not empty after sanitization
        if not sanitized:
            sanitized = "sanitized_library"

        # Prevent reserved names on Windows
        windows_reserved = {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "COM1",
            "COM2",
            "COM3",
            "COM4",
            "COM5",
            "COM6",
            "COM7",
            "COM8",
            "COM9",
            "LPT1",
            "LPT2",
            "LPT3",
            "LPT4",
            "LPT5",
            "LPT6",
            "LPT7",
            "LPT8",
            "LPT9",
        }

        if sanitized.upper() in windows_reserved:
            sanitized = f"{sanitized}_lib"

        # Limit length to prevent filesystem issues
        if len(sanitized) > 100:
            # Keep first 80 chars and add hash of full name
            import hashlib

            name_hash = hashlib.md5(library_id.encode()).hexdigest()[:8]
            sanitized = f"{sanitized[:80]}_{name_hash}"

        logger.debug(f"Sanitized library ID: '{library_id}' -> '{sanitized}'")
        return sanitized

    def get_library_directory(self, library_id: str) -> Path:
        """Get directory path for a library."""
        sanitized_id = self.sanitize_library_id(library_id)
        return self.base_path / sanitized_id

    def get_version_directory(self, library_id: str, version_id: str) -> Path:
        """Get directory path for a specific version."""
        return self.get_library_directory(library_id) / version_id

    def generate_version_id(self, timestamp: Optional[float] = None) -> str:
        """
        Generate timestamp-based version ID.

        Args:
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            Version ID in format v_{unix_timestamp}_{microseconds}
        """
        if timestamp is None:
            timestamp = time.time()

        # Use timestamp with microseconds for uniqueness
        timestamp_int = int(timestamp)
        microseconds = int((timestamp - timestamp_int) * 1000000)

        version_id = f"v_{timestamp_int}_{microseconds:06d}"

        logger.debug(f"Generated version ID: {version_id}")
        return version_id

    def parse_version_id(self, version_id: str) -> Dict[str, Any]:
        """
        Parse version ID to extract timestamp information.

        Args:
            version_id: Version ID to parse

        Returns:
            Dictionary with parsed information
        """
        try:
            # Expected format: v_{timestamp}_{microseconds}
            parts = version_id.split("_")

            if len(parts) >= 2 and parts[0] == "v":
                timestamp_int = int(parts[1])
                microseconds = int(parts[2]) if len(parts) > 2 else 0

                full_timestamp = timestamp_int + (microseconds / 1000000)

                return {
                    "is_valid": True,
                    "timestamp": full_timestamp,
                    "timestamp_int": timestamp_int,
                    "microseconds": microseconds,
                    "version_id": version_id,
                }
            else:
                return {"is_valid": False, "version_id": version_id}

        except ValueError:
            return {"is_valid": False, "version_id": version_id}

    def get_documentation_file_path(self, library_id: str, version_id: str) -> Path:
        """Get path for documentation file."""
        return (
            self.get_version_directory(library_id, version_id) / "documentation.json.gz"
        )

    def get_metadata_file_path(self, library_id: str, version_id: str) -> Path:
        """Get path for metadata file."""
        return self.get_version_directory(library_id, version_id) / "metadata.json"

    def get_library_metadata_path(self, library_id: str) -> Path:
        """Get path for library-level metadata."""
        return self.get_library_directory(library_id) / "library_metadata.json"

    def get_latest_symlink_path(self, library_id: str) -> Path:
        """Get path for latest version symlink."""
        return self.get_library_directory(library_id) / "latest"

    def get_global_index_path(self) -> Path:
        """Get path for global storage index."""
        return self.base_path / "index.json"

    async def create_directory_structure(
        self, library_id: str, version_id: str
    ) -> None:
        """Create complete directory structure for a library version."""
        try:
            version_dir = self.get_version_directory(library_id, version_id)
            version_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Created directory structure: {version_dir}")

        except Exception as e:
            raise VersionManagementError(
                f"Failed to create directory structure: {e}",
                library_id=library_id,
                version_id=version_id,
                operation="create_directory",
            ) from e

    async def update_latest_symlink(self, library_id: str, version_id: str) -> None:
        """Update or create 'latest' symlink to current version."""
        latest_link = self.get_latest_symlink_path(library_id)

        try:
            # Remove existing symlink if present
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()

            # Create new symlink (relative path for portability)
            latest_link.symlink_to(version_id)
            logger.debug(f"Updated latest symlink for {library_id} -> {version_id}")

        except OSError as e:
            # Symlinks might not be supported on all filesystems
            logger.warning(f"Failed to create symlink for {library_id}: {e}")

            # Create a file with the version ID as fallback
            fallback_file = latest_link.with_suffix(".txt")
            try:
                fallback_file.write_text(version_id)
                logger.info(f"Created fallback latest file for {library_id}")
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback latest file: {fallback_error}")

        except Exception as e:
            logger.error(f"Failed to update latest symlink for {library_id}: {e}")
            # Don't fail the entire operation for symlink issues

    async def resolve_latest_version(self, library_id: str) -> Optional[str]:
        """Resolve 'latest' to actual version ID."""
        latest_link = self.get_latest_symlink_path(library_id)

        try:
            if latest_link.is_symlink():
                # Read symlink target
                target = latest_link.readlink()
                return str(target)

            elif latest_link.with_suffix(".txt").exists():
                # Read fallback file
                version_id = latest_link.with_suffix(".txt").read_text().strip()
                return version_id

            else:
                # Fallback: find the newest version directory
                versions = await self.list_library_versions(library_id)
                if versions:
                    return versions[0]  # List is sorted newest first

        except Exception as e:
            logger.error(f"Failed to resolve latest version for {library_id}: {e}")

        return None

    async def list_library_versions(self, library_id: str) -> List[str]:
        """List all versions for a library, sorted by timestamp (newest first)."""
        library_dir = self.get_library_directory(library_id)

        if not library_dir.exists():
            return []

        versions = []

        try:
            for item in library_dir.iterdir():
                if (
                    item.is_dir()
                    and item.name.startswith("v_")
                    and item.name not in ["latest", "latest.txt"]
                ):
                    versions.append(item.name)

            # Sort by timestamp (newest first)
            def sort_key(version_id: str) -> float:
                parsed = self.parse_version_id(version_id)
                return parsed.get("timestamp", 0) if parsed["is_valid"] else 0

            versions.sort(key=sort_key, reverse=True)

            logger.debug(f"Found {len(versions)} versions for {library_id}")

        except Exception as e:
            logger.error(f"Failed to list versions for {library_id}: {e}")

        return versions

    async def list_all_libraries(self) -> List[str]:
        """List all libraries in storage."""
        libraries: List[str] = []

        try:
            if not self.base_path.exists():
                return libraries

            for item in self.base_path.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    # Check if it has version directories
                    has_versions = any(
                        child.is_dir() and child.name.startswith("v_")
                        for child in item.iterdir()
                    )

                    if has_versions:
                        libraries.append(item.name)

            libraries.sort()
            logger.debug(f"Found {len(libraries)} libraries in storage")

        except Exception as e:
            logger.error(f"Failed to list libraries: {e}")

        return libraries

    def validate_version_id(self, version_id: str) -> bool:
        """Validate that a version ID has the correct format."""
        parsed = self.parse_version_id(version_id)
        return bool(parsed["is_valid"])

    def calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of all files in a directory recursively."""
        total_size = 0

        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        except Exception as e:
            logger.warning(f"Failed to calculate directory size for {directory}: {e}")

        return total_size

    async def cleanup_empty_directories(self) -> int:
        """Clean up empty library directories."""
        cleanup_count = 0

        try:
            for library_dir in self.base_path.iterdir():
                if library_dir.is_dir() and not library_dir.name.startswith("."):
                    try:
                        # Check if directory is empty or has no version directories
                        version_dirs = [
                            child
                            for child in library_dir.iterdir()
                            if child.is_dir() and child.name.startswith("v_")
                        ]

                        if not version_dirs:
                            # Directory has no versions, safe to remove
                            import shutil

                            shutil.rmtree(library_dir)
                            cleanup_count += 1
                            logger.debug(
                                f"Removed empty library directory: {library_dir.name}"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup directory {library_dir}: {e}"
                        )

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} empty library directories")

        except Exception as e:
            logger.error(f"Failed to cleanup empty directories: {e}")

        return cleanup_count

    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary information about storage structure."""
        try:
            total_size = self.calculate_directory_size(self.base_path)

            summary = {
                "base_path": str(self.base_path),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "exists": self.base_path.exists(),
                "is_writable": (
                    os.access(self.base_path, os.W_OK)
                    if self.base_path.exists()
                    else False
                ),
                "free_space_bytes": (
                    shutil.disk_usage(self.base_path).free
                    if self.base_path.exists()
                    else 0
                ),
            }

            if self.base_path.exists():
                free_space_bytes = summary["free_space_bytes"]
                assert isinstance(free_space_bytes, (int, float))
                summary["free_space_mb"] = free_space_bytes / (1024 * 1024)
                free_space_mb = summary["free_space_mb"]
                assert isinstance(free_space_mb, (int, float))
                summary["free_space_gb"] = free_space_mb / 1024

            return summary

        except Exception as e:
            logger.error(f"Failed to generate storage summary: {e}")
            return {"base_path": str(self.base_path), "error": str(e)}

    def __str__(self) -> str:
        """String representation of path manager."""
        return f"StoragePathManager(base_path='{self.base_path}')"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"StoragePathManager(base_path=Path('{self.base_path}'))"
