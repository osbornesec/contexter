"""
Main storage management interface with atomic operations and integrity verification.
"""

import gzip
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.storage_models import (
    DocumentationChunk,
    IntegrityError,
    StorageResult,
    StorageStatistics,
    VersionInfo,
)
from .atomic_file_manager import AtomicFileManager
from .compression_engine import CompressionEngine
from .storage_path_manager import StoragePathManager

logger = logging.getLogger(__name__)


class LocalStorageManager:
    """Main storage management interface with atomic operations and comprehensive features."""

    def __init__(
        self,
        base_path: str = "~/.contexter/downloads",
        retention_limit: int = 5,
    ):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for storage
            retention_limit: Number of versions to keep per library
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.path_manager = StoragePathManager(str(self.base_path))
        self.file_manager = AtomicFileManager(self.base_path)
        self.retention_limit = retention_limit

        # Performance tracking
        self.operation_stats = {
            "stores": 0,
            "retrievals": 0,
            "verifications": 0,
            "cleanups": 0,
            "total_bytes_stored": 0,
            "total_compression_time": 0.0,
        }

        logger.info(
            f"Initialized LocalStorageManager: {self.base_path} "
            f"(retention={retention_limit})"
        )

    async def store_documentation(
        self, library_id: str, chunks: List[DocumentationChunk]
    ) -> StorageResult:
        """
        Store documentation with full integrity protection and version management.

        Args:
            library_id: Identifier for the library
            chunks: List of documentation chunks to store

        Returns:
            StorageResult with operation details
        """
        operation_start = datetime.now()

        try:
            if not chunks:
                return StorageResult(
                    success=False,
                    file_path=Path(),
                    compressed_size=0,
                    compression_ratio=0.0,
                    checksum="",
                    error_message="No chunks provided for storage",
                )

            # Generate version ID and create directory structure
            version_id = self.path_manager.generate_version_id()
            await self.path_manager.create_directory_structure(library_id, version_id)

            logger.info(
                f"Storing documentation for {library_id} version {version_id} "
                f"({len(chunks)} chunks)"
            )

            # Prepare documentation data structure
            doc_data = await self._prepare_documentation_data(
                library_id, version_id, chunks
            )

            # Store documentation file atomically
            doc_file_path = (
                Path(self.path_manager.sanitize_library_id(library_id))
                / version_id
                / "documentation.json.gz"
            )

            compression_metadata = {}
            async with self.file_manager.atomic_write(
                doc_file_path, compressed=True
            ) as writer:
                compression_metadata = await writer.write_json(doc_data)

            # Create comprehensive metadata
            metadata = await self._create_storage_metadata(
                library_id, version_id, chunks, compression_metadata, operation_start
            )

            # Store metadata file
            metadata_file_path = (
                Path(self.path_manager.sanitize_library_id(library_id))
                / version_id
                / "metadata.json"
            )
            async with self.file_manager.atomic_write(
                metadata_file_path, compressed=False
            ) as writer:
                await writer.write_json(metadata)

            # Update latest symlink
            await self.path_manager.update_latest_symlink(library_id, version_id)

            # Update library-level metadata
            await self._update_library_metadata(library_id, version_id, chunks)

            # Cleanup old versions
            cleaned_count = await self.cleanup_old_versions(library_id)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old versions for {library_id}")

            # Create successful result
            result = StorageResult(
                success=True,
                file_path=self.base_path / doc_file_path,
                compressed_size=compression_metadata["compressed_size"],
                compression_ratio=compression_metadata["compression_ratio"],
                checksum=compression_metadata["checksum"],
                version_id=version_id,
                original_size=compression_metadata["original_size"],
                compression_time=compression_metadata["compression_time"],
                metadata={
                    "chunk_count": len(chunks),
                    "library_id": library_id,
                    "storage_time": (datetime.now() - operation_start).total_seconds(),
                    "efficiency_percent": compression_metadata["compression_ratio"]
                    * 100,
                },
            )

            # Update operation statistics
            self.operation_stats["stores"] += 1
            self.operation_stats["total_bytes_stored"] += compression_metadata[
                "original_size"
            ]
            self.operation_stats["total_compression_time"] += compression_metadata[
                "compression_time"
            ]

            logger.info(
                f"Successfully stored {library_id} v{version_id}: "
                f"{compression_metadata['compressed_size']} bytes "
                f"({compression_metadata['compression_ratio']:.1%} compression)"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to store documentation for {library_id}: {e}"
            logger.error(error_msg, exc_info=True)

            return StorageResult(
                success=False,
                file_path=Path(),
                compressed_size=0,
                compression_ratio=0.0,
                checksum="",
                error_message=error_msg,
                metadata={"error_type": type(e).__name__},
            )

    async def retrieve_documentation(
        self, library_id: str, version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored documentation by library and version.

        Args:
            library_id: Library identifier
            version: Version to retrieve ('latest' or specific version ID)

        Returns:
            Documentation data or None if not found
        """
        try:
            # Resolve version if 'latest'
            resolved_version = version
            if version == "latest":
                latest_version = await self.path_manager.resolve_latest_version(
                    library_id
                )
                if not latest_version:
                    logger.warning(f"No versions found for library {library_id}")
                    return None
                resolved_version = latest_version

            # Verify integrity before reading
            if not await self.verify_integrity(library_id, resolved_version):
                logger.error(
                    f"Integrity verification failed for {library_id} v{resolved_version}"
                )
                return None

            # Read and decompress documentation
            doc_file_path = (
                Path(self.path_manager.sanitize_library_id(library_id))
                / resolved_version
                / "documentation.json.gz"
            )
            compressed_data = await self.file_manager.atomic_read(
                doc_file_path, verify_checksum=False
            )

            # Decompress and parse
            doc_data = await self.compression_engine.decompress_json(compressed_data)

            # Update operation statistics
            self.operation_stats["retrievals"] += 1

            logger.debug(f"Successfully retrieved {library_id} v{resolved_version}")
            return doc_data

        except Exception as e:
            logger.error(f"Failed to retrieve {library_id} v{version}: {e}")
            return None

    async def list_versions(self, library_id: str) -> List[VersionInfo]:
        """List all versions for a library with metadata."""
        versions = []

        try:
            version_names = await self.path_manager.list_library_versions(library_id)

            for version_id in version_names:
                try:
                    # Load version metadata
                    metadata_path = self.path_manager.get_metadata_file_path(
                        library_id, version_id
                    )

                    if not metadata_path.exists():
                        logger.warning(
                            f"Missing metadata for {library_id} v{version_id}"
                        )
                        continue

                    with open(metadata_path) as f:
                        metadata = json.load(f)

                    # Get file size
                    doc_path = self.path_manager.get_documentation_file_path(
                        library_id, version_id
                    )
                    file_size = doc_path.stat().st_size if doc_path.exists() else 0

                    version_info = VersionInfo(
                        version_id=version_id,
                        library_id=library_id,
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                        file_size=file_size,
                        checksum=metadata.get("checksum", ""),
                        metadata=metadata,
                    )

                    versions.append(version_info)

                except Exception as e:
                    logger.error(
                        f"Failed to load version info for {library_id} v{version_id}: {e}"
                    )
                    continue

            # Sort by creation time (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)

        except Exception as e:
            logger.error(f"Failed to list versions for {library_id}: {e}")

        return versions

    async def verify_integrity(self, library_id: str, version: str = "latest") -> bool:
        """
        Verify integrity of stored documentation.

        Args:
            library_id: Library identifier
            version: Version to verify

        Returns:
            True if integrity check passes
        """
        try:
            # Resolve version if 'latest'
            resolved_version = version
            if version == "latest":
                latest_version = await self.path_manager.resolve_latest_version(
                    library_id
                )
                if not latest_version:
                    return False
                resolved_version = latest_version

            # Get file paths
            doc_file_path = self.path_manager.get_documentation_file_path(
                library_id, resolved_version
            )
            metadata_file_path = self.path_manager.get_metadata_file_path(
                library_id, resolved_version
            )

            if not doc_file_path.exists() or not metadata_file_path.exists():
                logger.error(f"Missing files for {library_id} v{resolved_version}")
                return False

            # Load stored metadata
            with open(metadata_file_path) as f:
                metadata = json.load(f)

            stored_checksum = metadata.get("checksum")
            if not stored_checksum:
                logger.error(
                    f"No checksum in metadata for {library_id} v{resolved_version}"
                )
                return False

            # Read compressed file and verify checksum of original content
            with open(doc_file_path, "rb") as f:
                compressed_content = f.read()

            try:
                # Decompress and calculate checksum of original data
                original_content = gzip.decompress(compressed_content)
                current_checksum = hashlib.sha256(original_content).hexdigest()
            except gzip.BadGzipFile:
                # File might not be compressed
                current_checksum = hashlib.sha256(compressed_content).hexdigest()

            # Verify checksums match
            if current_checksum != stored_checksum:
                raise IntegrityError(
                    f"Checksum mismatch for {library_id} v{resolved_version}",
                    file_path=doc_file_path,
                    expected_checksum=stored_checksum,
                    actual_checksum=current_checksum,
                )

            # Verify file can be parsed as JSON
            try:
                if compressed_content.startswith(b"\x1f\x8b"):  # gzip magic bytes
                    decompressed = gzip.decompress(compressed_content).decode("utf-8")
                else:
                    decompressed = compressed_content.decode("utf-8")
                json.loads(decompressed)
            except Exception as e:
                logger.error(
                    f"File corruption detected for {library_id} v{resolved_version}: {e}"
                )
                return False

            # Update operation statistics
            self.operation_stats["verifications"] += 1

            logger.debug(
                f"Integrity verification passed for {library_id} v{resolved_version}"
            )
            return True

        except IntegrityError:
            logger.error(f"Integrity verification failed for {library_id} v{version}")
            return False
        except Exception as e:
            logger.error(
                f"Integrity verification failed for {library_id} v{version}: {e}"
            )
            return False

    async def cleanup_old_versions(self, library_id: str) -> int:
        """Clean up old versions beyond retention limit."""
        if self.retention_limit <= 0:
            return 0

        cleaned_count = 0

        try:
            versions = await self.list_versions(library_id)

            if len(versions) <= self.retention_limit:
                return 0  # Nothing to clean up

            # Remove oldest versions beyond retention limit
            versions_to_remove = versions[self.retention_limit :]

            for version_info in versions_to_remove:
                try:
                    version_dir = self.path_manager.get_version_directory(
                        library_id, version_info.version_id
                    )

                    if version_dir.exists():
                        # Remove entire version directory
                        shutil.rmtree(version_dir)
                        cleaned_count += 1
                        logger.debug(
                            f"Removed old version: {library_id} v{version_info.version_id}"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to remove version {version_info.version_id}: {e}"
                    )
                    continue

            # Update operation statistics
            self.operation_stats["cleanups"] += cleaned_count

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old versions for {library_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup versions for {library_id}: {e}")

        return cleaned_count

    async def get_storage_statistics(self) -> StorageStatistics:
        """Get comprehensive storage statistics."""
        stats = StorageStatistics()

        try:
            # Get list of all libraries
            libraries = await self.path_manager.list_all_libraries()
            stats.total_libraries = len(libraries)

            library_stats = []
            total_chunks = 0
            largest_size = 0
            oldest_age = 0.0
            corrupted_count = 0
            missing_metadata_count = 0

            for library_id in libraries:
                try:
                    versions = await self.list_versions(library_id)
                    stats.total_versions += len(versions)

                    if not versions:
                        continue

                    library_total_size = 0
                    library_compressed_size = 0
                    library_chunks = 0

                    for version in versions:
                        stats.total_size_bytes += version.original_size
                        stats.total_compressed_size += version.file_size
                        library_total_size += version.original_size
                        library_compressed_size += version.file_size
                        library_chunks += version.chunk_count

                        if version.file_size > largest_size:
                            largest_size = version.file_size

                        if version.age_hours > oldest_age:
                            oldest_age = version.age_hours

                        if version.is_corrupted:
                            corrupted_count += 1

                        if not version.metadata:
                            missing_metadata_count += 1

                    total_chunks += library_chunks

                    library_info = {
                        "library_id": library_id,
                        "version_count": len(versions),
                        "latest_version": versions[0].version_id if versions else None,
                        "total_size": library_total_size,
                        "compressed_size": library_compressed_size,
                        "chunk_count": library_chunks,
                        "compression_ratio": (
                            (library_total_size - library_compressed_size)
                            / library_total_size
                            if library_total_size > 0
                            else 0
                        ),
                        "latest_update": (
                            versions[0].created_at.isoformat() if versions else None
                        ),
                    }

                    library_stats.append(library_info)

                except Exception as e:
                    logger.error(f"Failed to get statistics for {library_id}: {e}")
                    continue

            # Calculate derived statistics
            stats.total_chunks = total_chunks
            stats.largest_library_size = largest_size
            stats.oldest_version_age_hours = oldest_age
            stats.corrupted_files = corrupted_count
            stats.missing_metadata = missing_metadata_count
            stats.libraries = library_stats

            if stats.total_size_bytes > 0:
                stats.average_compression_ratio = (
                    stats.total_size_bytes - stats.total_compressed_size
                ) / stats.total_size_bytes

            # Add operation statistics
            stats.libraries.append({"operation_stats": self.operation_stats.copy()})

        except Exception as e:
            logger.error(f"Failed to calculate storage statistics: {e}")

        return stats

    async def _prepare_documentation_data(
        self, library_id: str, version_id: str, chunks: List[DocumentationChunk]
    ) -> Dict[str, Any]:
        """Prepare documentation data structure for storage."""
        return {
            "library_id": library_id,
            "version_id": version_id,
            "downloaded_at": datetime.now().isoformat(),
            "storage_format_version": "1.0",
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source_context": chunk.source_context,
                    "token_count": chunk.token_count,
                    "content_hash": chunk.content_hash,
                    "proxy_id": chunk.proxy_id,
                    "download_time": chunk.download_time,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            "summary": {
                "total_chunks": len(chunks),
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "total_content_length": sum(len(chunk.content) for chunk in chunks),
                "unique_contexts": len({chunk.source_context for chunk in chunks}),
                "proxy_usage": list(
                    {chunk.proxy_id for chunk in chunks if chunk.proxy_id}
                ),
            },
        }

    async def _create_storage_metadata(
        self,
        library_id: str,
        version_id: str,
        chunks: List[DocumentationChunk],
        compression_metadata: Dict[str, Any],
        operation_start: datetime,
    ) -> Dict[str, Any]:
        """Create comprehensive storage metadata."""
        return {
            "library_id": library_id,
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "operation_duration": (datetime.now() - operation_start).total_seconds(),
            "chunk_count": len(chunks),
            "integrity_verified": True,
            "storage_manager_version": "1.0",
            **compression_metadata,
            "storage_info": {
                "compression_algorithm": "gzip",
                "format": "json",
                "encoding": "utf-8",
                "retention_policy": f"keep_last_{self.retention_limit}",
            },
            "content_summary": {
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "average_chunk_size": (
                    sum(len(chunk.content) for chunk in chunks) / len(chunks)
                    if chunks
                    else 0
                ),
                "unique_contexts": len({chunk.source_context for chunk in chunks}),
                "content_types": list(
                    {chunk.metadata.get("content_type", "unknown") for chunk in chunks}
                ),
            },
        }

    async def _update_library_metadata(
        self, library_id: str, version_id: str, chunks: List[DocumentationChunk]
    ) -> None:
        """Update library-level metadata."""
        try:
            library_metadata_path = self.path_manager.get_library_metadata_path(
                library_id
            )

            # Load existing metadata if it exists
            existing_metadata = {}
            if library_metadata_path.exists():
                with open(library_metadata_path) as f:
                    existing_metadata = json.load(f)

            # Update metadata
            updated_metadata = {
                "library_id": library_id,
                "last_updated": datetime.now().isoformat(),
                "latest_version": version_id,
                "total_versions": len(
                    await self.path_manager.list_library_versions(library_id)
                ),
                "last_chunk_count": len(chunks),
                "update_history": existing_metadata.get("update_history", []),
            }

            # Add to update history
            updated_metadata["update_history"].append(
                {
                    "version_id": version_id,
                    "timestamp": datetime.now().isoformat(),
                    "chunk_count": len(chunks),
                }
            )

            # Keep only last 10 history entries
            updated_metadata["update_history"] = updated_metadata["update_history"][
                -10:
            ]

            # Write updated metadata
            async with self.file_manager.atomic_write(
                library_metadata_path.relative_to(self.base_path), compressed=False
            ) as writer:
                await writer.write_json(updated_metadata)

        except Exception as e:
            logger.warning(f"Failed to update library metadata for {library_id}: {e}")

    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics for this manager instance."""
        stats = self.operation_stats.copy()

        # Add derived metrics
        if stats["stores"] > 0:
            stats["average_compression_time"] = (
                stats["total_compression_time"] / stats["stores"]
            )
            stats["average_bytes_per_store"] = (
                stats["total_bytes_stored"] / stats["stores"]
            )

        compression_stats = self.compression_engine.get_compression_statistics()
        stats.update(compression_stats)

        return stats

    def reset_operation_statistics(self) -> None:
        """Reset operation statistics."""
        self.operation_stats = {
            "stores": 0,
            "retrievals": 0,
            "verifications": 0,
            "cleanups": 0,
            "total_bytes_stored": 0,
            "total_compression_time": 0.0,
        }
        self.compression_engine.reset_statistics()
        logger.info("Operation statistics reset")
