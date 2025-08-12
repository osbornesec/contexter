"""
Compression utilities and performance tracking for storage operations.
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..models.storage_models import CompressionError

logger = logging.getLogger(__name__)


class CompressionEngine:
    """High-performance compression utilities with comprehensive metrics."""

    def __init__(self, default_level: int = 6):
        """Initialize with default compression level."""
        self.default_level = default_level
        self.compression_stats = {
            "total_operations": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "total_compression_time": 0.0,
            "average_ratio": 0.0,
            "best_ratio": 0.0,
            "worst_ratio": 1.0,
        }

    async def compress_json(
        self, data: Dict[str, Any], compression_level: Optional[int] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress JSON data and return compressed bytes with metadata."""
        level = compression_level or self.default_level
        start_time = time.time()

        try:
            # Serialize to JSON with consistent formatting
            json_content = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ": "),
            )
            original_bytes = json_content.encode("utf-8")
            original_size = len(original_bytes)

            # Calculate checksum of original data
            checksum = hashlib.sha256(original_bytes).hexdigest()

            # Compress in executor to avoid blocking
            compressed_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gzip.compress(original_bytes, compresslevel=level)
            )

            compressed_size = len(compressed_bytes)
            compression_time = time.time() - start_time

            # Calculate metrics
            compression_ratio = (
                (original_size - compressed_size) / original_size
                if original_size > 0
                else 0
            )

            metadata = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_level": level,
                "compression_time": compression_time,
                "checksum": checksum,
                "algorithm": "gzip",
                "efficiency_percent": compression_ratio * 100,
                "speed_mbps": (
                    (original_size / (1024 * 1024)) / compression_time
                    if compression_time > 0
                    else 0
                ),
            }

            # Update global stats
            self._update_stats(
                original_size, compressed_size, compression_time, compression_ratio
            )

            logger.debug(
                f"Compressed {original_size} -> {compressed_size} bytes "
                f"({compression_ratio:.1%} ratio, {compression_time:.3f}s)"
            )

            return compressed_bytes, metadata

        except Exception as e:
            raise CompressionError(
                f"JSON compression failed: {e}", operation="compress_json"
            ) from e

    async def decompress_json(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress JSON data and parse."""
        try:
            # Decompress in executor to avoid blocking
            decompressed_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gzip.decompress(compressed_data)
            )

            # Parse JSON
            json_content = decompressed_bytes.decode("utf-8")
            data = json.loads(json_content)

            logger.debug(
                f"Decompressed {len(compressed_data)} -> {len(decompressed_bytes)} bytes"
            )

            return data  # type: ignore

        except gzip.BadGzipFile as e:
            raise CompressionError(
                f"Invalid gzip data: {e}", operation="decompress_json"
            ) from e
        except json.JSONDecodeError as e:
            raise CompressionError(
                f"Invalid JSON data after decompression: {e}",
                operation="decompress_json",
            ) from e
        except Exception as e:
            raise CompressionError(
                f"JSON decompression failed: {e}", operation="decompress_json"
            ) from e

    async def compress_text(
        self, text: str, compression_level: Optional[int] = None
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress plain text and return compressed bytes with metadata."""
        level = compression_level or self.default_level
        start_time = time.time()

        try:
            text_bytes = text.encode("utf-8")
            original_size = len(text_bytes)

            # Calculate checksum
            checksum = hashlib.sha256(text_bytes).hexdigest()

            # Compress in executor
            compressed_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gzip.compress(text_bytes, compresslevel=level)
            )

            compressed_size = len(compressed_bytes)
            compression_time = time.time() - start_time
            compression_ratio = (
                (original_size - compressed_size) / original_size
                if original_size > 0
                else 0
            )

            metadata = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "compression_level": level,
                "compression_time": compression_time,
                "checksum": checksum,
                "algorithm": "gzip",
                "efficiency_percent": compression_ratio * 100,
                "speed_mbps": (
                    (original_size / (1024 * 1024)) / compression_time
                    if compression_time > 0
                    else 0
                ),
            }

            self._update_stats(
                original_size, compressed_size, compression_time, compression_ratio
            )

            return compressed_bytes, metadata

        except Exception as e:
            raise CompressionError(
                f"Text compression failed: {e}", operation="compress_text"
            ) from e

    async def decompress_text(self, compressed_data: bytes) -> str:
        """Decompress text data."""
        try:
            decompressed_bytes = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gzip.decompress(compressed_data)
            )

            return decompressed_bytes.decode("utf-8")

        except gzip.BadGzipFile as e:
            raise CompressionError(
                f"Invalid gzip data: {e}", operation="decompress_text"
            ) from e
        except UnicodeDecodeError as e:
            raise CompressionError(
                f"Invalid UTF-8 data after decompression: {e}",
                operation="decompress_text",
            ) from e
        except Exception as e:
            raise CompressionError(
                f"Text decompression failed: {e}", operation="decompress_text"
            ) from e

    @staticmethod
    async def decompress_json_file(file_path: Path) -> Dict[str, Any]:
        """Decompress and parse JSON file with fallback to regular file."""
        try:
            # Try to read as compressed file
            compressed_data = file_path.read_bytes()

            try:
                # Attempt gzip decompression
                decompressed_data = gzip.decompress(compressed_data)
                json_content = decompressed_data.decode("utf-8")

                logger.debug(f"Successfully decompressed {file_path}")

            except gzip.BadGzipFile:
                # File is not compressed, read as regular JSON
                json_content = compressed_data.decode("utf-8")
                logger.debug(f"Read uncompressed JSON file: {file_path}")

            return json.loads(json_content)  # type: ignore

        except json.JSONDecodeError as e:
            raise CompressionError(
                f"Invalid JSON in {file_path}: {e}", operation="decompress_json_file"
            ) from e
        except Exception as e:
            raise CompressionError(
                f"Failed to read {file_path}: {e}", operation="decompress_json_file"
            ) from e

    @staticmethod
    async def is_compressed(file_path: Path) -> bool:
        """Check if a file is gzip compressed."""
        try:
            with open(file_path, "rb") as f:
                # Read first 3 bytes to check gzip magic number
                magic = f.read(3)
                return magic[:2] == b"\x1f\x8b"
        except Exception:
            return False

    async def benchmark_compression_levels(
        self, data: Dict[str, Any]
    ) -> Dict[int, Dict[str, Any]]:
        """Benchmark different compression levels for given data."""
        results = {}

        logger.info("Benchmarking compression levels 1-9...")

        for level in range(1, 10):
            try:
                compressed_bytes, metadata = await self.compress_json(data, level)

                results[level] = {
                    "compression_ratio": metadata["compression_ratio"],
                    "compression_time": metadata["compression_time"],
                    "compressed_size": metadata["compressed_size"],
                    "efficiency_percent": metadata["efficiency_percent"],
                    "speed_mbps": metadata["speed_mbps"],
                }

                logger.debug(
                    f"Level {level}: {metadata['efficiency_percent']:.1f}% "
                    f"compression in {metadata['compression_time']:.3f}s"
                )

            except Exception as e:
                logger.error(f"Benchmark failed for level {level}: {e}")
                results[level] = {"error": str(e)}

        # Find optimal level (best ratio/time trade-off)
        if results:
            optimal_level = self._find_optimal_compression_level(results)
            results["recommended_level"] = optimal_level  # type: ignore

            logger.info(
                f"Compression benchmark complete. Recommended level: {optimal_level}"
            )

        return results

    def _find_optimal_compression_level(
        self, benchmark_results: Dict[int, Dict[str, Any]]
    ) -> int:
        """Find optimal compression level based on ratio/speed trade-off."""
        best_score = 0
        best_level = 6  # Default

        for level, metrics in benchmark_results.items():
            if (
                isinstance(level, int)
                and "compression_ratio" in metrics
                and "compression_time" in metrics
            ):
                # Score = compression_ratio / time (higher is better)
                ratio = metrics["compression_ratio"]
                time_factor = 1 / (
                    metrics["compression_time"] + 0.001
                )  # Add small value to avoid division by zero
                score = ratio * time_factor

                if score > best_score:
                    best_score = score
                    best_level = level

        return best_level

    def _update_stats(
        self,
        original_size: int,
        compressed_size: int,
        compression_time: float,
        compression_ratio: float,
    ) -> None:
        """Update global compression statistics."""
        self.compression_stats["total_operations"] += 1
        self.compression_stats["total_original_bytes"] += original_size
        self.compression_stats["total_compressed_bytes"] += compressed_size
        self.compression_stats["total_compression_time"] += compression_time

        # Update ratio statistics
        if compression_ratio > self.compression_stats["best_ratio"]:
            self.compression_stats["best_ratio"] = compression_ratio

        if compression_ratio < self.compression_stats["worst_ratio"]:
            self.compression_stats["worst_ratio"] = compression_ratio

        # Calculate running average ratio
        total_original = self.compression_stats["total_original_bytes"]
        total_compressed = self.compression_stats["total_compressed_bytes"]
        if total_original > 0:
            self.compression_stats["average_ratio"] = (
                total_original - total_compressed
            ) / total_original

    @staticmethod
    def calculate_compression_efficiency(
        original_size: int, compressed_size: int
    ) -> Dict[str, float]:
        """Calculate comprehensive compression metrics."""
        if original_size == 0:
            return {
                "ratio": 0.0,
                "reduction": 0.0,
                "efficiency": 0.0,
                "space_saved": 0.0,
            }

        ratio = compressed_size / original_size
        reduction = (original_size - compressed_size) / original_size
        efficiency = reduction * 100  # Percentage reduction
        space_saved = original_size - compressed_size  # Bytes saved

        return {
            "ratio": ratio,
            "reduction": reduction,
            "efficiency": efficiency,
            "space_saved": space_saved,
            "compression_factor": 1 / ratio if ratio > 0 else float("inf"),
        }

    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        stats = self.compression_stats.copy()

        # Add derived metrics
        if stats["total_operations"] > 0:
            stats["average_compression_time"] = (
                stats["total_compression_time"] / stats["total_operations"]
            )

            stats["average_original_size"] = (
                stats["total_original_bytes"] / stats["total_operations"]
            )

            stats["average_compressed_size"] = (
                stats["total_compressed_bytes"] / stats["total_operations"]
            )

            total_time = stats["total_compression_time"]
            if total_time > 0:
                stats["overall_throughput_mbps"] = (
                    stats["total_original_bytes"] / (1024 * 1024)
                ) / total_time
            else:
                stats["overall_throughput_mbps"] = 0.0

        return stats

    def reset_statistics(self) -> None:
        """Reset compression statistics."""
        self.compression_stats = {
            "total_operations": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "total_compression_time": 0.0,
            "average_ratio": 0.0,
            "best_ratio": 0.0,
            "worst_ratio": 1.0,
        }
        logger.info("Compression statistics reset")


class CompressionValidator:
    """Validate compression results and detect corruption."""

    @staticmethod
    async def validate_compressed_data(
        original_data: bytes,
        compressed_data: bytes,
        expected_checksum: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate compressed data integrity."""
        validation_result = {
            "is_valid": False,
            "original_size": len(original_data),
            "compressed_size": len(compressed_data),
            "checksum_match": False,
            "decompression_successful": False,
            "content_match": False,
            "errors": [],
        }

        try:
            # Check if data can be decompressed
            decompressed_data = await asyncio.get_event_loop().run_in_executor(
                None, lambda: gzip.decompress(compressed_data)
            )
            validation_result["decompression_successful"] = True

            # Check if content matches
            if decompressed_data == original_data:
                validation_result["content_match"] = True
            else:
                errors = validation_result["errors"]
                assert isinstance(errors, list)
                errors.append("Decompressed content does not match original")

            # Validate checksum if provided
            if expected_checksum:
                actual_checksum = hashlib.sha256(original_data).hexdigest()
                if actual_checksum == expected_checksum:
                    validation_result["checksum_match"] = True
                else:
                    errors = validation_result["errors"]
                    assert isinstance(errors, list)
                    errors.append(
                        f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                    )
            else:
                validation_result["checksum_match"] = True  # No checksum to validate

            # Overall validation
            validation_result["is_valid"] = (
                validation_result["decompression_successful"]
                and validation_result["content_match"]
                and validation_result["checksum_match"]
            )

        except gzip.BadGzipFile:
            errors = validation_result["errors"]
            assert isinstance(errors, list)
            errors.append("Invalid gzip format")
        except Exception as e:
            errors = validation_result["errors"]
            assert isinstance(errors, list)
            errors.append(f"Validation error: {e}")

        return validation_result
