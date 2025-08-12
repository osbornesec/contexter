"""
Atomic file operations with integrity verification and compression support.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Union

from ..models.storage_models import (
    AtomicOperationError,
    CompressionError,
    IntegrityError,
)

logger = logging.getLogger(__name__)


class CompressedFileWriter:
    """High-performance compressed file writer with integrity checking."""

    def __init__(self, file_path: Path, compression_level: int = 6):
        self.file_path = file_path
        self.compression_level = compression_level
        self.original_size = 0
        self.compressed_size = 0
        self.start_time = time.time()
        self.hasher = hashlib.sha256()
        self._file_handle: Any = None
        self._data_written = False

    async def __aenter__(self) -> "CompressedFileWriter":
        """Async context manager entry."""
        self._file_handle = open(self.file_path, "wb")
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with proper cleanup."""
        if self._file_handle:
            try:
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file handle: {e}")
            finally:
                self._file_handle = None

    async def write_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write JSON data with compression and integrity tracking."""
        if self._data_written:
            raise AtomicOperationError("Data already written to this writer")

        try:
            # Serialize to JSON with consistent formatting
            json_content = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ": "),  # Consistent separators
            )
            json_bytes = json_content.encode("utf-8")
            self.original_size = len(json_bytes)

            # Update hash with original data (before compression)
            self.hasher.update(json_bytes)

            # Compress data in executor to avoid blocking
            compressed_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gzip.compress(json_bytes, compresslevel=self.compression_level),
            )
            self.compressed_size = len(compressed_data)

            # Write compressed data
            self._file_handle.write(compressed_data)
            self._file_handle.flush()

            # Ensure data is written to disk
            os.fsync(self._file_handle.fileno())

            self._data_written = True
            logger.debug(
                f"Wrote {self.original_size} bytes compressed to {self.compressed_size} bytes"
            )

            return await self._get_compression_metadata()

        except Exception as e:
            raise CompressionError(
                f"Failed to write JSON data: {e}", operation="write_json"
            ) from e

    async def write_text(self, text: str) -> Dict[str, Any]:
        """Write text data with compression and integrity tracking."""
        if self._data_written:
            raise AtomicOperationError("Data already written to this writer")

        try:
            text_bytes = text.encode("utf-8")
            self.original_size = len(text_bytes)

            # Update hash with original data
            self.hasher.update(text_bytes)

            # Compress data
            compressed_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: gzip.compress(text_bytes, compresslevel=self.compression_level),
            )
            self.compressed_size = len(compressed_data)

            # Write compressed data
            self._file_handle.write(compressed_data)
            self._file_handle.flush()

            # Ensure data is written to disk
            os.fsync(self._file_handle.fileno())

            self._data_written = True
            return await self._get_compression_metadata()

        except Exception as e:
            raise CompressionError(
                f"Failed to write text data: {e}", operation="write_text"
            ) from e

    async def finalize(self) -> Dict[str, Any]:
        """Finalize writing and get final statistics."""
        if not self._data_written:
            logger.warning("Finalizing writer without data written")

        if self._file_handle:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None
            except Exception as e:
                logger.error(f"Error finalizing file: {e}")

        # Get actual file size (may differ from calculated compressed size)
        try:
            actual_size = self.file_path.stat().st_size
            self.compressed_size = actual_size
        except Exception as e:
            logger.warning(f"Could not get actual file size: {e}")

        return await self._get_compression_metadata()

    async def _get_compression_metadata(self) -> Dict[str, Any]:
        """Get compression statistics and metadata."""
        compression_ratio = (
            (self.original_size - self.compressed_size) / self.original_size
            if self.original_size > 0
            else 0
        )
        compression_time = time.time() - self.start_time
        checksum = self.hasher.hexdigest()

        return {
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": compression_ratio,
            "compression_level": self.compression_level,
            "compression_time": compression_time,
            "checksum": checksum,
            "algorithm": "gzip",
            "efficiency_percent": compression_ratio * 100,
            "speed_mbps": (
                (self.original_size / (1024 * 1024)) / compression_time
                if compression_time > 0
                else 0
            ),
        }


class RegularFileWriter:
    """Non-compressed file writer with integrity tracking."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_size = 0
        self.start_time = time.time()
        self.hasher = hashlib.sha256()
        self._file_handle: Any = None
        self._data_written = False

    async def __aenter__(self) -> "RegularFileWriter":
        """Async context manager entry."""
        self._file_handle = open(self.file_path, "w", encoding="utf-8")
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception as e:
                logger.warning(f"Error closing file handle: {e}")
            finally:
                self._file_handle = None

    async def write_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Write JSON data without compression."""
        if self._data_written:
            raise AtomicOperationError("Data already written to this writer")

        try:
            json_content = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ": "),
            )

            if self._file_handle is not None:
                self._file_handle.write(json_content)
                self._file_handle.flush()

            # Update hash and size
            content_bytes = json_content.encode("utf-8")
            self.hasher.update(content_bytes)
            self.file_size = len(content_bytes)
            self._data_written = True

            return await self._get_metadata()

        except Exception as e:
            raise AtomicOperationError(
                f"Failed to write JSON data: {e}", operation="write_json"
            ) from e

    async def write_text(self, text: str) -> Dict[str, Any]:
        """Write plain text data."""
        if self._data_written:
            raise AtomicOperationError("Data already written to this writer")

        try:
            if self._file_handle is not None:
                self._file_handle.write(text)
                self._file_handle.flush()

            # Update hash and size
            content_bytes = text.encode("utf-8")
            self.hasher.update(content_bytes)
            self.file_size = len(content_bytes)
            self._data_written = True

            return await self._get_metadata()

        except Exception as e:
            raise AtomicOperationError(
                f"Failed to write text data: {e}", operation="write_text"
            ) from e

    async def finalize(self) -> Dict[str, Any]:
        """Finalize writing and return metadata."""
        if not self._data_written:
            logger.warning("Finalizing writer without data written")

        if self._file_handle is not None:
            try:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None
            except Exception as e:
                logger.error(f"Error finalizing file: {e}")

        try:
            actual_size = self.file_path.stat().st_size
            self.file_size = actual_size
        except Exception as e:
            logger.warning(f"Could not get actual file size: {e}")

        return await self._get_metadata()

    async def _get_metadata(self) -> Dict[str, Any]:
        """Get file metadata."""
        write_time = time.time() - self.start_time
        return {
            "file_size": self.file_size,
            "checksum": self.hasher.hexdigest(),
            "compressed": False,
            "write_time": write_time,
            "speed_mbps": (
                (self.file_size / (1024 * 1024)) / write_time if write_time > 0 else 0
            ),
        }


class AtomicFileManager:
    """Atomic file operations with integrity verification."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized AtomicFileManager with base path: {self.base_path}")

    @asynccontextmanager
    async def atomic_write(
        self, target_path: Path, compressed: bool = True
    ) -> AsyncGenerator[Union[CompressedFileWriter, RegularFileWriter], None]:
        """Context manager for atomic file writes."""
        # Resolve target path relative to base_path
        if not target_path.is_absolute():
            target_path = self.base_path / target_path

        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file in same directory as target
        temp_fd: Optional[int] = None
        temp_path: Optional[Path] = None
        writer: Optional[Union[CompressedFileWriter, RegularFileWriter]] = None

        try:
            temp_fd, temp_path_str = tempfile.mkstemp(
                dir=target_path.parent, prefix=f".{target_path.name}.", suffix=".tmp"
            )
            temp_path = Path(temp_path_str)

            logger.debug(f"Created temporary file: {temp_path}")

            # Create appropriate writer
            if compressed:
                writer = CompressedFileWriter(temp_path)
            else:
                writer = RegularFileWriter(temp_path)

            # Use the writer
            if writer is not None:
                async with writer:
                    yield writer

                # Finalize the writer to ensure all data is written
                await writer.finalize()

            # Verify file was written successfully
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise AtomicOperationError(
                    "Temporary file is empty or missing after write"
                )

            # Atomic move to final location
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: shutil.move(str(temp_path), str(target_path))
            )

            logger.info(f"Atomically wrote file: {target_path}")

        except Exception as e:
            # Clean up temp file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(
                        f"Failed to clean up temporary file {temp_path}: {cleanup_error}"
                    )

            logger.error(f"Atomic write failed for {target_path}: {e}")
            raise AtomicOperationError(
                f"Atomic write failed: {e}", operation="atomic_write"
            ) from e

        finally:
            # Ensure temp file descriptor is closed
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass

    async def atomic_read(self, file_path: Path, verify_checksum: bool = True) -> bytes:
        """Read file with optional integrity verification."""
        # Resolve path relative to base_path
        if not file_path.is_absolute():
            full_path = self.base_path / file_path
        else:
            full_path = file_path

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            # Read file content
            with open(full_path, "rb") as f:
                content = f.read()

            # Verify integrity if requested
            if verify_checksum:
                await self._verify_file_integrity(file_path, content)

            logger.debug(f"Successfully read {len(content)} bytes from {file_path}")
            return content

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise AtomicOperationError(
                f"File read failed: {e}", operation="atomic_read"
            ) from e

    async def _verify_file_integrity(self, file_path: Path, content: bytes) -> None:
        """Verify file integrity using stored checksum."""
        # Look for metadata file with checksum
        metadata_path = file_path.parent / "metadata.json"
        metadata_full_path = (
            self.base_path / metadata_path
            if not metadata_path.is_absolute()
            else metadata_path
        )

        if not metadata_full_path.exists():
            logger.warning(
                f"No metadata file found for integrity verification: {metadata_path}"
            )
            return

        try:
            with open(metadata_full_path) as f:
                metadata = json.load(f)

            stored_checksum = metadata.get("checksum")
            if not stored_checksum:
                logger.warning(f"No checksum found in metadata: {metadata_path}")
                return

            # Decompress if needed and calculate checksum of original content
            try:
                # Check if file is compressed by attempting decompression
                decompressed_content = gzip.decompress(content)
                current_checksum = hashlib.sha256(decompressed_content).hexdigest()
            except gzip.BadGzipFile:
                # File is not compressed
                current_checksum = hashlib.sha256(content).hexdigest()

            if current_checksum != stored_checksum:
                raise IntegrityError(
                    f"Checksum mismatch for {file_path}",
                    file_path=file_path,
                    expected_checksum=stored_checksum,
                    actual_checksum=current_checksum,
                )

            logger.debug(f"Integrity verification passed for {file_path}")

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to read metadata for integrity check: {e}")
            # Don't fail the read operation for metadata issues
        except IntegrityError:
            # Re-raise integrity errors
            raise
        except Exception as e:
            logger.warning(f"Integrity verification failed with unexpected error: {e}")

    async def cleanup_temp_files(self, max_age_hours: float = 24.0) -> int:
        """Clean up old temporary files that may have been left behind."""
        cleanup_count = 0

        try:
            current_time = time.time()
            age_threshold = max_age_hours * 3600  # Convert to seconds

            # Recursively find all .tmp files
            for temp_file in self.base_path.rglob("*.tmp"):
                try:
                    if temp_file.is_file():
                        file_age = current_time - temp_file.stat().st_mtime

                        if file_age > age_threshold:
                            temp_file.unlink()
                            cleanup_count += 1
                            logger.debug(f"Cleaned up old temp file: {temp_file}")

                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_file}: {e}")

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} old temporary files")

        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")

        return cleanup_count
