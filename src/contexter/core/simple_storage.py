"""
Simplified storage manager without compression for documentation chunks.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.storage_models import (
    DocumentationChunk,
    StorageResult,
)

logger = logging.getLogger(__name__)


class SimpleStorageManager:
    """Simplified storage manager that saves plain JSON files without compression."""

    def __init__(
        self,
        base_path: str = "~/.contexter/downloads",
        retention_limit: int = 5,
    ):
        """
        Initialize simple storage manager.

        Args:
            base_path: Base directory for storage
            retention_limit: Number of versions to keep per library
        """
        self.base_path = Path(base_path).expanduser().resolve()
        self.retention_limit = retention_limit
        
        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized SimpleStorageManager: {self.base_path}")

    async def store_documentation(
        self, library_id: str, chunks: List[DocumentationChunk]
    ) -> StorageResult:
        """
        Store documentation chunks as plain JSON.
        
        Args:
            library_id: Library identifier
            chunks: List of documentation chunks
            
        Returns:
            Storage result with file information
        """
        try:
            operation_start = datetime.now()
            
            # Create version ID
            version_id = f"v_{int(operation_start.timestamp())}"
            
            # Sanitize library ID for filesystem
            safe_library_id = library_id.replace("/", "_").replace("\\", "_")
            
            # Create directory structure
            library_dir = self.base_path / safe_library_id
            version_dir = library_dir / version_id
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare documentation data
            doc_data = {
                "library_id": library_id,
                "version_id": version_id,
                "stored_at": operation_start.isoformat(),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_context": chunk.source_context,
                        "token_count": chunk.token_count,
                        "content_hash": chunk.content_hash,
                        "proxy_id": chunk.proxy_id,
                        "download_time": chunk.download_time,
                        "library_id": chunk.library_id,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ]
            }
            
            # Write documentation file
            doc_file_path = version_dir / "documentation.json"
            with open(doc_file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            # Calculate file stats
            file_size = doc_file_path.stat().st_size
            
            # Calculate checksum
            with open(doc_file_path, 'rb') as f:
                file_content = f.read()
                checksum = hashlib.sha256(file_content).hexdigest()
            
            # Create metadata
            total_tokens = sum(chunk.token_count for chunk in chunks)
            metadata = {
                "library_id": library_id,
                "version_id": version_id,
                "created_at": operation_start.isoformat(),
                "chunk_count": len(chunks),
                "file_size": file_size,
                "checksum": checksum,
                "total_tokens": total_tokens,
                "storage_info": {
                    "format": "json",
                    "encoding": "utf-8",
                    "compression": "none"
                }
            }
            
            # Write metadata file
            metadata_file_path = version_dir / "metadata.json"
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Update latest symlink
            latest_link = library_dir / "latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()
            latest_link.symlink_to(version_id)
            
            # Update library metadata
            await self._update_library_metadata(library_id, version_id, len(chunks))
            
            # Clean up old versions
            await self._cleanup_old_versions(library_id)
            
            operation_time = (datetime.now() - operation_start).total_seconds()
            
            logger.info(
                f"Stored documentation for {library_id}: "
                f"{len(chunks)} chunks, {file_size:,} bytes, "
                f"{total_tokens:,} tokens in {operation_time:.3f}s"
            )
            
            return StorageResult(
                success=True,
                file_path=doc_file_path,
                compressed_size=file_size,  # No compression, so same as original
                compression_ratio=0.0,     # No compression
                checksum=checksum,
                version_id=version_id,
                original_size=file_size,
                compression_time=0.0,      # No compression time
                metadata={
                    "chunk_count": len(chunks),
                    "library_id": library_id,
                    "total_tokens": total_tokens,
                    "storage_time": operation_time,
                },
            )
            
        except Exception as e:
            logger.error(f"Failed to store documentation for {library_id}: {e}")
            return StorageResult(
                success=False,
                error=f"Storage failed: {e}",
                file_path=None,
                compressed_size=0,
                compression_ratio=0.0,
                checksum="",
                version_id="",
                original_size=0,
                compression_time=0.0,
                metadata={},
            )

    async def retrieve_documentation(
        self, library_id: str, version: str = "latest"
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored documentation.
        
        Args:
            library_id: Library identifier
            version: Version to retrieve ("latest" or specific version ID)
            
        Returns:
            Documentation data or None if not found
        """
        try:
            safe_library_id = library_id.replace("/", "_").replace("\\", "_")
            library_dir = self.base_path / safe_library_id
            
            if version == "latest":
                latest_link = library_dir / "latest"
                if not latest_link.exists():
                    return None
                version_dir = latest_link.resolve()
            else:
                version_dir = library_dir / version
                
            if not version_dir.exists():
                return None
                
            doc_file_path = version_dir / "documentation.json"
            if not doc_file_path.exists():
                return None
                
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to retrieve documentation for {library_id}: {e}")
            return None

    async def _update_library_metadata(
        self, library_id: str, version_id: str, chunk_count: int
    ) -> None:
        """Update library-level metadata."""
        try:
            safe_library_id = library_id.replace("/", "_").replace("\\", "_")
            library_dir = self.base_path / safe_library_id
            metadata_file = library_dir / "library_metadata.json"
            
            # Load existing metadata or create new
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {
                    "library_id": library_id,
                    "total_versions": 0,
                    "update_history": []
                }
            
            # Update metadata
            metadata["latest_version"] = version_id
            metadata["last_updated"] = datetime.now().isoformat()
            metadata["last_chunk_count"] = chunk_count
            metadata["total_versions"] = metadata.get("total_versions", 0) + 1
            
            # Add to history
            metadata["update_history"].append({
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "chunk_count": chunk_count
            })
            
            # Keep only last 10 history entries
            metadata["update_history"] = metadata["update_history"][-10:]
            
            # Write updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to update library metadata: {e}")

    async def _cleanup_old_versions(self, library_id: str) -> int:
        """Clean up old versions beyond retention limit."""
        try:
            safe_library_id = library_id.replace("/", "_").replace("\\", "_")
            library_dir = self.base_path / safe_library_id
            
            if not library_dir.exists():
                return 0
            
            # Get all version directories
            version_dirs = [
                d for d in library_dir.iterdir() 
                if d.is_dir() and d.name.startswith("v_")
            ]
            
            # Sort by creation time (newest first)
            version_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            
            # Remove old versions beyond retention limit
            removed_count = 0
            for old_version in version_dirs[self.retention_limit:]:
                try:
                    import shutil
                    shutil.rmtree(old_version)
                    removed_count += 1
                    logger.debug(f"Removed old version: {old_version.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove old version {old_version.name}: {e}")
            
            return removed_count
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old versions: {e}")
            return 0