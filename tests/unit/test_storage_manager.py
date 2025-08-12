"""
Comprehensive unit tests for the storage management system.
"""

import pytest
import asyncio
import tempfile
import hashlib
import gzip
import json
import time
import random
from pathlib import Path
from datetime import datetime

from src.contexter.models.storage_models import (
    DocumentationChunk,
    StorageResult,
    VersionInfo,
    IntegrityError,
    StorageError,
    AtomicOperationError
)
from src.contexter.core.storage_manager import LocalStorageManager
from src.contexter.core.atomic_file_manager import AtomicFileManager, CompressedFileWriter, RegularFileWriter
from src.contexter.core.compression_engine import CompressionEngine, CompressionValidator
from src.contexter.core.storage_path_manager import StoragePathManager


class TestStorageModels:
    """Test storage data models."""
    
    def test_documentation_chunk_creation(self):
        """Test DocumentationChunk creation and hash calculation."""
        chunk = DocumentationChunk(
            chunk_id="test_1",
            content="Test content for chunk",
            source_context="test context",
            token_count=100,
            content_hash=""  # Will be auto-calculated
        )
        
        assert chunk.chunk_id == "test_1"
        assert chunk.content == "Test content for chunk"
        assert chunk.content_hash  # Should be auto-calculated
        assert len(chunk.content_hash) == 64  # SHA-256 hex length
    
    def test_storage_result_compression_ratio(self):
        """Test StorageResult compression ratio calculation."""
        result = StorageResult(
            success=True,
            file_path=Path("/test/path"),
            compressed_size=400,
            compression_ratio=0.0,  # Will be calculated
            checksum="test_checksum",
            original_size=1000
        )
        
        assert result.compression_ratio == 0.6  # (1000-400)/1000
        assert result.metadata['compression_efficiency'] == 60.0
        assert result.metadata['size_reduction_bytes'] == 600
    
    def test_version_info_age_calculation(self):
        """Test VersionInfo age calculation."""
        past_time = datetime.now().replace(hour=datetime.now().hour - 2)  # 2 hours ago
        
        version = VersionInfo(
            version_id="v_1234567890_000000",
            library_id="test/lib",
            created_at=past_time,
            file_size=1000,
            checksum="test_checksum"
        )
        
        assert version.age_hours >= 1.9  # Should be close to 2 hours
        assert version.size_mb == 1000 / (1024 * 1024)


class TestAtomicFileManager:
    """Test atomic file operations."""
    
    @pytest.mark.asyncio
    async def test_atomic_write_success(self):
        """Test successful atomic file write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = AtomicFileManager(Path(temp_dir))
            test_file = Path("test_dir/test_file.json.gz")
            
            test_data = {"test": "data", "numbers": [1, 2, 3], "nested": {"key": "value"}}
            
            # Write file atomically
            async with file_manager.atomic_write(test_file, compressed=True) as writer:
                metadata = await writer.write_json(test_data)
            
            # Verify file exists and has content
            full_path = Path(temp_dir) / test_file
            assert full_path.exists()
            assert full_path.stat().st_size > 0
            
            # Verify metadata
            assert metadata['checksum']
            assert metadata['compression_ratio'] > 0
            assert metadata['original_size'] > metadata['compressed_size']
    
    @pytest.mark.asyncio
    async def test_atomic_write_failure_cleanup(self):
        """Test cleanup of temporary files on write failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = AtomicFileManager(Path(temp_dir))
            test_file = Path("test_file.json")
            
            # Create a scenario that will cause write failure
            with pytest.raises(AtomicOperationError):
                async with file_manager.atomic_write(test_file, compressed=False) as writer:
                    await writer.write_json({"test": "data"})
                    # Force an error by writing again (should fail)
                    await writer.write_json({"another": "write"})
            
            # Verify no temporary files are left behind
            temp_files = list(Path(temp_dir).rglob("*.tmp"))
            assert len(temp_files) == 0
    
    @pytest.mark.asyncio
    async def test_atomic_read_with_verification(self):
        """Test atomic read with integrity verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_manager = AtomicFileManager(Path(temp_dir))
            test_file = Path("test_file.json.gz")
            
            test_data = {"content": "test data for verification"}
            
            # Write file with metadata
            async with file_manager.atomic_write(test_file, compressed=True) as writer:
                metadata = await writer.write_json(test_data)
            
            # Create metadata file
            metadata_file = test_file.parent / "metadata.json"
            async with file_manager.atomic_write(metadata_file, compressed=False) as writer:
                await writer.write_json({"checksum": metadata['checksum']})
            
            # Read file with verification
            content = await file_manager.atomic_read(test_file, verify_checksum=True)
            
            # Verify content can be decompressed and matches
            decompressed = gzip.decompress(content).decode('utf-8')
            loaded_data = json.loads(decompressed)
            assert loaded_data == test_data
    
    @pytest.mark.asyncio
    async def test_compressed_file_writer_efficiency(self):
        """Test compression efficiency of CompressedFileWriter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_compression.json.gz"
            
            # Create highly repetitive data that should compress well
            test_data = {
                "repeated_content": "This is repeated content. " * 1000,
                "array": [{"same": "structure"} for _ in range(100)]
            }
            
            async with CompressedFileWriter(test_file, compression_level=6) as writer:
                metadata = await writer.write_json(test_data)
            
            # Should achieve good compression ratio
            assert metadata['compression_ratio'] > 0.8  # Should compress to less than 20% of original
            assert metadata['original_size'] > 0
            assert metadata['compressed_size'] > 0
            assert metadata['checksum']
            assert metadata['algorithm'] == 'gzip'


class TestCompressionEngine:
    """Test compression engine functionality."""
    
    @pytest.mark.asyncio
    async def test_json_compression_and_decompression(self):
        """Test JSON compression and decompression roundtrip."""
        engine = CompressionEngine(default_level=6)
        
        test_data = {
            "library": "test-library",
            "chunks": [
                {"id": i, "content": f"Content for chunk {i} " * 50}
                for i in range(10)
            ],
            "metadata": {"version": "1.0", "timestamp": "2024-01-01T00:00:00"}
        }
        
        # Compress data
        compressed_bytes, metadata = await engine.compress_json(test_data)
        
        # Verify compression metadata
        assert metadata['compression_ratio'] > 0
        assert metadata['original_size'] > metadata['compressed_size']
        assert metadata['checksum']
        assert metadata['efficiency_percent'] > 0
        
        # Decompress data
        decompressed_data = await engine.decompress_json(compressed_bytes)
        
        # Verify roundtrip integrity
        assert decompressed_data == test_data
    
    @pytest.mark.asyncio
    async def test_compression_level_benchmarking(self):
        """Test compression level benchmarking."""
        engine = CompressionEngine()
        
        # Create test data with good compression potential
        test_data = {
            "repetitive": "ABC" * 1000,
            "structured": [{"key": "value", "number": i} for i in range(100)]
        }
        
        # Benchmark compression levels
        results = await engine.benchmark_compression_levels(test_data)
        
        # Verify results for each level
        for level in range(1, 10):
            assert level in results
            if 'compression_ratio' in results[level]:
                assert results[level]['compression_ratio'] > 0
                assert results[level]['compression_time'] >= 0
        
        # Should recommend a level
        assert 'recommended_level' in results
        assert 1 <= results['recommended_level'] <= 9
    
    @pytest.mark.asyncio
    async def test_compression_statistics_tracking(self):
        """Test compression statistics tracking."""
        engine = CompressionEngine()
        
        # Perform multiple compression operations
        for i in range(5):
            test_data = {"iteration": i, "content": f"Data for iteration {i}" * 100}
            await engine.compress_json(test_data)
        
        # Get statistics
        stats = engine.get_compression_statistics()
        
        assert stats['total_operations'] == 5
        assert stats['total_original_bytes'] > 0
        assert stats['total_compressed_bytes'] > 0
        assert stats['average_ratio'] > 0
        assert stats['overall_throughput_mbps'] >= 0
    
    @pytest.mark.asyncio
    async def test_compression_validator(self):
        """Test compression validation."""
        original_data = b"Test data for compression validation" * 100
        
        # Compress data manually
        compressed_data = gzip.compress(original_data, compresslevel=6)
        expected_checksum = hashlib.sha256(original_data).hexdigest()
        
        # Validate compressed data
        result = await CompressionValidator.validate_compressed_data(
            original_data, compressed_data, expected_checksum
        )
        
        assert result['is_valid']
        assert result['decompression_successful']
        assert result['content_match']
        assert result['checksum_match']
        assert len(result['errors']) == 0


class TestStoragePathManager:
    """Test storage path management."""
    
    def test_library_id_sanitization(self):
        """Test library ID sanitization for filesystem safety."""
        path_manager = StoragePathManager()
        
        test_cases = [
            ("normal/library", "normal_library"),
            ("lib:with<>chars", "lib_with_chars"),
            ("lib with spaces", "lib_with_spaces"),
            ("lib..with...dots", "lib.with.dots"),
            ("lib___multiple___underscores", "lib_multiple_underscores"),
            ("", "unknown_library"),
            ("   ", "unknown_library"),
            ("CON", "CON_lib"),  # Windows reserved name
            ("x" * 150, "x" * 80 + "_" + "a1b2c3d4")  # Length limit with hash (approximate)
        ]
        
        for original, expected_pattern in test_cases:
            sanitized = path_manager.sanitize_library_id(original)
            
            if expected_pattern.endswith("a1b2c3d4"):  # Hash case
                assert len(sanitized) <= 89  # 80 + 1 + 8
                assert "_" in sanitized
            else:
                assert sanitized == expected_pattern
    
    def test_version_id_generation_and_parsing(self):
        """Test version ID generation and parsing."""
        path_manager = StoragePathManager()
        
        # Generate version ID
        timestamp = time.time()
        version_id = path_manager.generate_version_id(timestamp)
        
        assert version_id.startswith("v_")
        assert len(version_id.split("_")) >= 3  # v_timestamp_microseconds
        
        # Parse version ID
        parsed = path_manager.parse_version_id(version_id)
        assert parsed['is_valid']
        assert abs(parsed['timestamp'] - timestamp) < 0.001  # Should be very close
        assert parsed['version_id'] == version_id
    
    def test_directory_structure_paths(self):
        """Test directory structure path generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = StoragePathManager(temp_dir)
            
            library_id = "test/library"
            version_id = "v_1234567890_000000"
            
            # Test path generation
            lib_dir = path_manager.get_library_directory(library_id)
            version_dir = path_manager.get_version_directory(library_id, version_id)
            doc_file = path_manager.get_documentation_file_path(library_id, version_id)
            metadata_file = path_manager.get_metadata_file_path(library_id, version_id)
            latest_link = path_manager.get_latest_symlink_path(library_id)
            
            # Verify path structure
            expected_lib = Path(temp_dir) / "test_library"
            assert lib_dir == expected_lib
            assert version_dir == expected_lib / version_id
            assert doc_file == expected_lib / version_id / "documentation.json.gz"
            assert metadata_file == expected_lib / version_id / "metadata.json"
            assert latest_link == expected_lib / "latest"
    
    @pytest.mark.asyncio
    async def test_version_listing_and_sorting(self):
        """Test version listing and timestamp-based sorting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path_manager = StoragePathManager(temp_dir)
            library_id = "test/lib"
            
            # Create multiple version directories with different timestamps
            timestamps = [1000000000, 1000000100, 1000000050, 1000000200]
            version_ids = []
            
            for ts in timestamps:
                version_id = path_manager.generate_version_id(ts)
                version_ids.append(version_id)
                await path_manager.create_directory_structure(library_id, version_id)
            
            # List versions
            listed_versions = await path_manager.list_library_versions(library_id)
            
            # Should be sorted newest first
            assert len(listed_versions) == 4
            
            # Parse timestamps and verify ordering
            parsed_timestamps = []
            for version_id in listed_versions:
                parsed = path_manager.parse_version_id(version_id)
                parsed_timestamps.append(parsed['timestamp'])
            
            # Should be in descending order
            assert parsed_timestamps == sorted(parsed_timestamps, reverse=True)


class TestLocalStorageManager:
    """Test main storage manager functionality."""
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_documentation(self):
        """Test complete store and retrieve workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageManager(temp_dir, compression_level=6, retention_limit=3)
            
            # Create test chunks
            chunks = []
            for i in range(5):
                chunk = DocumentationChunk(
                    chunk_id=f"test_chunk_{i}",
                    content=f"Test content for chunk {i}. " * 50,
                    source_context=f"test_context_{i}",
                    token_count=random.randint(500, 1500),
                    content_hash=""  # Auto-calculated
                )
                chunks.append(chunk)
            
            # Store documentation
            result = await storage.store_documentation("test/library", chunks)
            
            # Verify storage success
            assert result.success
            assert result.compression_ratio > 0.5  # Should achieve good compression
            assert result.checksum
            assert result.version_id
            assert result.file_path.exists()
            
            # Retrieve documentation
            retrieved_doc = await storage.retrieve_documentation("test/library")
            
            # Verify retrieved data
            assert retrieved_doc is not None
            assert retrieved_doc['library_id'] == "test/library"
            assert len(retrieved_doc['chunks']) == 5
            assert retrieved_doc['summary']['total_chunks'] == 5
            
            # Verify chunk integrity
            for i, chunk_data in enumerate(retrieved_doc['chunks']):
                original_chunk = chunks[i]
                assert chunk_data['chunk_id'] == original_chunk.chunk_id
                assert chunk_data['content'] == original_chunk.content
                assert chunk_data['source_context'] == original_chunk.source_context
    
    @pytest.mark.asyncio
    async def test_version_management_and_cleanup(self):
        """Test version management with automatic cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageManager(temp_dir, retention_limit=3)
            library_id = "test/versioned"
            
            # Create multiple versions
            version_results = []
            for i in range(6):  # More than retention limit
                chunks = [DocumentationChunk(
                    chunk_id=f"v{i}_chunk",
                    content=f"Version {i} content " * 20,
                    source_context=f"version_{i}_context",
                    token_count=100,
                    content_hash=""
                )]
                
                result = await storage.store_documentation(library_id, chunks)
                assert result.success
                version_results.append(result)
                
                # Small delay to ensure different timestamps
                await asyncio.sleep(0.01)
            
            # List versions (should only have 3 due to retention)
            versions = await storage.list_versions(library_id)
            assert len(versions) == 3
            
            # Verify these are the newest versions
            version_timestamps = [v.created_at for v in versions]
            assert version_timestamps == sorted(version_timestamps, reverse=True)
            
            # Verify latest version can be retrieved
            latest_doc = await storage.retrieve_documentation(library_id, "latest")
            assert latest_doc is not None
            assert "Version 5" in latest_doc['chunks'][0]['content']  # Should be the last version created
    
    @pytest.mark.asyncio
    async def test_integrity_verification(self):
        """Test integrity verification and corruption detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageManager(temp_dir)
            library_id = "test/integrity"
            
            chunks = [DocumentationChunk(
                chunk_id="integrity_test",
                content="Content for integrity testing",
                source_context="integrity_context",
                token_count=50,
                content_hash=""
            )]
            
            # Store documentation
            result = await storage.store_documentation(library_id, chunks)
            assert result.success
            
            # Verify integrity passes initially
            assert await storage.verify_integrity(library_id, "latest")
            
            # Get the stored file path and corrupt it
            versions = await storage.list_versions(library_id)
            version_id = versions[0].version_id
            doc_path = storage.path_manager.get_documentation_file_path(library_id, version_id)
            
            # Corrupt the file by appending bytes
            with open(doc_path, 'ab') as f:
                f.write(b'CORRUPTION_DATA')
            
            # Verify integrity now fails
            assert not await storage.verify_integrity(library_id, "latest")
            
            # Retrieval should also fail due to integrity check
            corrupted_doc = await storage.retrieve_documentation(library_id, "latest")
            assert corrupted_doc is None
    
    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage operations on different libraries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageManager(temp_dir)
            
            async def store_library(library_id: str, chunk_count: int):
                chunks = [
                    DocumentationChunk(
                        chunk_id=f"{library_id}_chunk_{i}",
                        content=f"Content for {library_id} chunk {i} " * 30,
                        source_context=f"{library_id}_context_{i}",
                        token_count=random.randint(800, 1200),
                        content_hash=""
                    )
                    for i in range(chunk_count)
                ]
                
                return await storage.store_documentation(library_id, chunks)
            
            # Store multiple libraries concurrently
            library_tasks = [
                store_library(f"concurrent/lib_{i}", random.randint(3, 8))
                for i in range(15)  # 15 concurrent operations
            ]
            
            results = await asyncio.gather(*library_tasks)
            
            # Verify all operations succeeded
            assert all(result.success for result in results)
            
            # Verify all libraries can be retrieved
            for i, result in enumerate(results):
                library_id = f"concurrent/lib_{i}"
                doc = await storage.retrieve_documentation(library_id)
                assert doc is not None
                assert doc['library_id'] == library_id
                assert len(doc['chunks']) >= 3
    
    @pytest.mark.asyncio
    async def test_storage_statistics_calculation(self):
        """Test comprehensive storage statistics calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageManager(temp_dir)
            
            # Store documentation for multiple libraries
            libraries_data = [
                ("stats/lib1", 3),
                ("stats/lib2", 5),
                ("stats/lib3", 2)
            ]
            
            for library_id, chunk_count in libraries_data:
                chunks = [
                    DocumentationChunk(
                        chunk_id=f"chunk_{i}",
                        content=f"Statistics test content {i}" * 100,
                        source_context=f"context_{i}",
                        token_count=1000,
                        content_hash=""
                    )
                    for i in range(chunk_count)
                ]
                
                result = await storage.store_documentation(library_id, chunks)
                assert result.success
            
            # Get storage statistics
            stats = await storage.get_storage_statistics()
            
            # Verify basic statistics
            assert stats.total_libraries == 3
            assert stats.total_versions >= 3  # At least one version per library
            assert stats.total_chunks == 3 + 5 + 2  # Sum of all chunks
            assert stats.total_size_bytes > 0
            assert stats.total_compressed_size > 0
            assert stats.average_compression_ratio > 0
            assert len(stats.libraries) >= 3
            
            # Verify operation statistics
            op_stats = storage.get_operation_statistics()
            assert op_stats['stores'] == 3
            assert op_stats['total_bytes_stored'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with read-only directory to simulate path issues
            readonly_path = Path(temp_dir) / "readonly"
            readonly_path.mkdir()
            readonly_path.chmod(0o444)  # Read-only
            invalid_storage = LocalStorageManager(str(readonly_path))
            
            chunks = [DocumentationChunk(
                chunk_id="error_test",
                content="Test content",
                source_context="error_context",
                token_count=50,
                content_hash=""
            )]
            
            # Should handle path creation failure gracefully
            result = await invalid_storage.store_documentation("test/error", chunks)
            assert not result.success
            assert result.error_message
            
            # Test with empty chunks list
            valid_storage = LocalStorageManager(temp_dir)
            result = await valid_storage.store_documentation("test/empty", [])
            assert not result.success
            assert "No chunks provided" in result.error_message
            
            # Test retrieval of non-existent library
            doc = await valid_storage.retrieve_documentation("non/existent")
            assert doc is None
            
            # Test version listing for non-existent library
            versions = await valid_storage.list_versions("non/existent")
            assert len(versions) == 0


if __name__ == "__main__":
    pytest.main([__file__])