"""
High-performance deduplication engine for documentation chunks.

This module provides fast exact duplicate detection using xxhash64 and
semantic similarity analysis using TF-IDF vectorization with configurable
similarity thresholds. Designed for processing large chunk sets efficiently
with memory-optimized batch processing.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# Import xxhash with fallback to hashlib
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    logging.warning(
        "xxhash not available, falling back to hashlib (performance will be reduced)"
    )

# Import components for similarity analysis
try:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, similarity analysis will be disabled")

# Try to import Google Gemini for embeddings
try:
    import numpy as np
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Always import TF-IDF as a fallback regardless of Gemini availability
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    if not GEMINI_AVAILABLE:
        logging.warning(
            "Neither Gemini nor TF-IDF available, similarity analysis will be disabled"
        )

from ..models.download_models import DocumentationChunk

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of a deduplication operation with comprehensive metrics."""

    original_count: int
    deduplicated_count: int
    exact_duplicates_removed: int
    similar_chunks_merged: int
    processing_time: float
    chunks: List[DocumentationChunk]

    # Additional metrics
    memory_usage_mb: Optional[float] = None
    hash_performance: Optional[Dict[str, float]] = field(default_factory=dict)
    similarity_performance: Optional[Dict[str, float]] = field(default_factory=dict)

    @property
    def deduplication_ratio(self) -> float:
        """Calculate deduplication ratio (0.0 to 1.0)."""
        if self.original_count == 0:
            return 0.0
        return (self.original_count - self.deduplicated_count) / self.original_count

    @property
    def chunks_per_second(self) -> float:
        """Calculate processing rate in chunks per second."""
        if self.processing_time <= 0:
            return 0.0
        return self.original_count / self.processing_time

    @property
    def efficiency_score(self) -> float:
        """Calculate overall efficiency score (0.0 to 1.0)."""
        # Combine deduplication effectiveness and processing speed
        dedup_score = min(self.deduplication_ratio * 2, 1.0)  # Cap at 1.0
        speed_score = min(
            self.chunks_per_second / 1000, 1.0
        )  # 1000 chunks/sec = perfect

        # Weighted average: 60% deduplication, 40% speed
        return 0.6 * dedup_score + 0.4 * speed_score


class ContentHasher:
    """High-performance content hashing using xxhash64 with intelligent caching."""

    def __init__(self, enable_cache: bool = True, max_cache_size: int = 10000):
        """
        Initialize hasher with configurable caching.

        Args:
            enable_cache: Enable hash result caching
            max_cache_size: Maximum number of hashes to cache
        """
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size
        self.hash_cache: Dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.min_cache_length = 100  # Only cache substantial content

    def calculate_hash(self, content: str) -> str:
        """
        Calculate xxhash64 for content with intelligent caching.

        Args:
            content: Text content to hash

        Returns:
            Hexadecimal hash string
        """
        # Check cache first for substantial content
        if (
            self.enable_cache
            and len(content) >= self.min_cache_length
            and content in self.hash_cache
        ):
            self.cache_hits += 1
            return self.hash_cache[content]

        # Calculate hash using xxhash64 or fallback
        if XXHASH_AVAILABLE:
            hasher = xxhash.xxh64()
            hasher.update(content.encode("utf-8"))
            content_hash = hasher.hexdigest()
        else:
            # Fallback to SHA-256 for consistency
            import hashlib

            hasher_fallback: Any = hashlib.sha256()
            hasher = hasher_fallback
            hasher.update(content.encode("utf-8"))
            content_hash = hasher.hexdigest()

        # Cache result if enabled and content is substantial
        if self.enable_cache and len(content) >= self.min_cache_length:
            # Implement simple LRU by clearing cache when full
            if len(self.hash_cache) >= self.max_cache_size:
                # Remove oldest half of entries (simple LRU approximation)
                keys_to_remove = list(self.hash_cache.keys())[
                    : self.max_cache_size // 2
                ]
                for key in keys_to_remove:
                    del self.hash_cache[key]
                logger.debug(f"Cache cleanup: removed {len(keys_to_remove)} entries")

            self.hash_cache[content] = content_hash

        self.cache_misses += 1
        return content_hash

    def batch_calculate_hashes(self, contents: List[str]) -> List[str]:
        """
        Calculate hashes for multiple contents efficiently.

        Args:
            contents: List of content strings to hash

        Returns:
            List of hash strings in same order
        """
        start_time = time.time()
        hashes = []

        for content in contents:
            content_hash = self.calculate_hash(content)
            hashes.append(content_hash)

        processing_time = time.time() - start_time
        rate = len(contents) / processing_time if processing_time > 0 else float("inf")

        logger.debug(
            f"Batch hashed {len(contents)} items in {processing_time:.3f}s "
            f"({rate:.0f} items/sec, {self.cache_hits} cache hits)"
        )

        return hashes

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_enabled": self.enable_cache,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.hash_cache),
            "max_cache_size": self.max_cache_size,
            "xxhash_available": XXHASH_AVAILABLE,
        }


class ExactDuplicateProcessor:
    """Processes exact duplicates using high-performance content hashing."""

    def __init__(self, enable_hash_cache: bool = True):
        """
        Initialize processor with configurable hash caching.

        Args:
            enable_hash_cache: Enable hash result caching for performance
        """
        self.hasher = ContentHasher(enable_cache=enable_hash_cache)

    async def remove_exact_duplicates(
        self, chunks: List[DocumentationChunk]
    ) -> List[DocumentationChunk]:
        """
        Remove exact duplicates using hash-based grouping with intelligent conflict resolution.

        Args:
            chunks: List of documentation chunks to deduplicate

        Returns:
            List of unique chunks with duplicates removed
        """
        if not chunks:
            return []

        start_time = time.time()
        logger.info(f"Processing {len(chunks)} chunks for exact duplicates")

        # Group chunks by content hash
        hash_groups = defaultdict(list)

        for chunk in chunks:
            # Use existing hash if available and valid, otherwise calculate
            if hasattr(chunk, "content_hash") and chunk.content_hash:
                content_hash = chunk.content_hash
                # Verify hash matches content (detect stale hashes)
                if len(content_hash) < 32:  # xxhash64 should be 16 chars, SHA-256 is 64
                    expected_hash = self.hasher.calculate_hash(chunk.content)
                    if content_hash != expected_hash[: len(content_hash)]:
                        content_hash = expected_hash
                        chunk.content_hash = content_hash
            else:
                content_hash = self.hasher.calculate_hash(chunk.content)
                chunk.content_hash = content_hash

            hash_groups[content_hash].append(chunk)

        # Select best chunk from each hash group
        unique_chunks = []
        exact_duplicates_removed = 0

        for content_hash, group_chunks in hash_groups.items():
            if len(group_chunks) == 1:
                # No duplicates for this hash
                unique_chunks.append(group_chunks[0])
            else:
                # Handle collision detection and resolve conflicts
                if await self._verify_hash_group(group_chunks, content_hash):
                    # True duplicates - resolve conflict
                    best_chunk = self._resolve_exact_duplicate_conflict(group_chunks)
                    unique_chunks.append(best_chunk)
                    exact_duplicates_removed += len(group_chunks) - 1

                    logger.debug(
                        f"Removed {len(group_chunks) - 1} exact duplicates "
                        f"(hash: {content_hash[:8]}...)"
                    )
                else:
                    # Hash collision - keep all chunks
                    unique_chunks.extend(group_chunks)
                    logger.warning(
                        f"Hash collision detected for {len(group_chunks)} chunks "
                        f"(hash: {content_hash[:8]}...)"
                    )

        processing_time = time.time() - start_time

        logger.info(
            f"Exact deduplication completed: {len(chunks)} -> {len(unique_chunks)} chunks "
            f"({exact_duplicates_removed} duplicates removed) in {processing_time:.2f}s"
        )

        return unique_chunks

    async def _verify_hash_group(
        self, chunks: List[DocumentationChunk], expected_hash: str
    ) -> bool:
        """
        Verify that all chunks in a hash group are actually identical.
        Detects hash collisions by comparing actual content.

        Args:
            chunks: Chunks with the same hash
            expected_hash: The common hash value

        Returns:
            True if all chunks are identical, False if hash collision detected
        """
        if len(chunks) <= 1:
            return True

        # Compare content of first chunk with all others
        reference_content = chunks[0].content

        for chunk in chunks[1:]:
            if chunk.content != reference_content:
                logger.warning(
                    f"Hash collision detected: same hash {expected_hash[:8]}... "
                    f"but different content (lengths: {len(reference_content)} vs {len(chunk.content)})"
                )
                return False

        return True

    def _resolve_exact_duplicate_conflict(
        self, chunks: List[DocumentationChunk]
    ) -> DocumentationChunk:
        """
        Select the best chunk from exact duplicates based on metadata quality and completeness.

        Args:
            chunks: List of identical chunks to choose from

        Returns:
            The best chunk based on quality scoring
        """

        def score_chunk(chunk: DocumentationChunk) -> float:
            """Calculate quality score for chunk selection."""
            score = 0.0

            # Prefer chunks with more comprehensive metadata
            if hasattr(chunk, "metadata") and chunk.metadata:
                score += len(chunk.metadata) * 0.1

            # Source context quality scoring
            if hasattr(chunk, "source_context") and chunk.source_context:
                context = chunk.source_context.lower()
                if any(
                    term in context
                    for term in ["complete", "reference", "comprehensive"]
                ):
                    score += 2.0
                elif any(
                    term in context for term in ["documentation", "official", "docs"]
                ):
                    score += 1.5
                elif any(term in context for term in ["tutorial", "guide"]):
                    score += 1.0
                elif any(term in context for term in ["example", "sample"]):
                    score += 0.5

            # Download success indicators
            if hasattr(chunk, "proxy_id") and chunk.proxy_id:
                score += 0.5

            if hasattr(chunk, "download_time") and chunk.download_time:
                # Favor reasonable download times (indicates successful retrieval)
                if 0.1 <= chunk.download_time <= 60:
                    score += 0.3

            # Token count as quality indicator
            if hasattr(chunk, "token_count") and chunk.token_count:
                # Normalize token count to reasonable score range
                if 500 <= chunk.token_count <= 20000:
                    score += min(chunk.token_count / 10000, 1.0)
                elif chunk.token_count >= 100:
                    score += 0.2

            # Library ID presence
            if hasattr(chunk, "library_id") and chunk.library_id:
                score += 0.3

            # Use chunk ID length as tie-breaker (longer IDs often indicate more metadata)
            if hasattr(chunk, "chunk_id") and chunk.chunk_id:
                score += len(chunk.chunk_id) / 1000  # Small bonus

            return score

        # Select chunk with highest quality score
        best_chunk = max(chunks, key=score_chunk)

        # Log selection rationale for debugging
        best_score = score_chunk(best_chunk)
        logger.debug(
            f"Selected best chunk from {len(chunks)} exact duplicates: "
            f"score={best_score:.2f}, source='{getattr(best_chunk, 'source_context', 'unknown')[:30]}...'"
        )

        return best_chunk


class SimilarityAnalyzer:
    """Gemini embeddings-based content similarity analysis for near-duplicate detection."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_features: int = 5000,
        min_content_length: int = 50,
    ):
        """
        Initialize similarity analyzer with configurable parameters.

        Args:
            similarity_threshold: Cosine similarity threshold for grouping (0.0-1.0)
            max_features: Maximum TF-IDF features to extract (only used for TF-IDF fallback)
            min_content_length: Minimum content length to analyze
        """
        self.similarity_threshold = similarity_threshold
        self.max_features = max_features
        self.min_content_length = min_content_length
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.gemini_client = None

        # Initialize Gemini client if available
        if GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client()
                logger.info("Gemini embeddings initialized for similarity analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self.gemini_client = None
        elif not SKLEARN_AVAILABLE:
            logger.error(
                "Neither Gemini nor scikit-learn available - similarity analysis disabled"
            )

    async def detect_similar_chunks(
        self, chunks: List[DocumentationChunk]
    ) -> List[List[DocumentationChunk]]:
        """
        Detect groups of similar chunks using Gemini embeddings or TF-IDF fallback.

        Args:
            chunks: List of chunks to analyze for similarity

        Returns:
            List of similar chunk groups (each group contains similar chunks)
        """
        if len(chunks) <= 1:
            return []  # No similarity groups possible with 0-1 chunks

        start_time = time.time()
        logger.info(f"Analyzing similarity for {len(chunks)} chunks")

        try:
            # Filter chunks with sufficient content
            valid_chunks = [
                chunk
                for chunk in chunks
                if len(chunk.content.strip()) >= self.min_content_length
            ]

            if len(valid_chunks) <= 1:
                logger.info("Insufficient content for similarity analysis")
                return []

            # Extract content
            documents = [chunk.content for chunk in valid_chunks]

            # Try Gemini embeddings first with TF-IDF fallback
            similarity_matrix = None

            # Try Gemini first if available
            if self.gemini_client:
                try:
                    similarity_matrix = await self._compute_gemini_similarity(documents)
                    logger.debug("Using Gemini embeddings for similarity analysis")
                except Exception as e:
                    logger.warning(
                        f"Gemini embedding failed, falling back to TF-IDF: {e}"
                    )

            # Fall back to TF-IDF if Gemini failed or unavailable
            if similarity_matrix is None and SKLEARN_AVAILABLE and TFIDF_AVAILABLE:
                # Use a more appropriate threshold for TF-IDF (much lower than Gemini's semantic threshold)
                original_threshold = self.similarity_threshold
                if (
                    self.similarity_threshold > 0.5
                ):  # High threshold likely designed for Gemini
                    # Use adaptive threshold for TF-IDF based on content similarity patterns
                    self.similarity_threshold = (
                        0.25  # More aggressive for TF-IDF to catch FastAPI content
                    )
                    logger.info(
                        f"Adjusted TF-IDF similarity threshold from {original_threshold} to {self.similarity_threshold} for TF-IDF fallback"
                    )

                try:
                    similarity_matrix = await self._compute_tfidf_similarity(documents)
                    logger.debug("Using TF-IDF for similarity analysis")
                finally:
                    # Restore original threshold
                    self.similarity_threshold = original_threshold

            if similarity_matrix is None:
                logger.warning("No similarity analysis method available")
                return []

            # Find similarity groups using connected components
            similar_groups_indices = self._find_similarity_groups(similarity_matrix)

            # Convert indices back to chunk groups
            similar_groups = []
            for group_indices in similar_groups_indices:
                if len(group_indices) > 1:  # Only include actual groups (>1 member)
                    chunk_group = [valid_chunks[i] for i in group_indices]
                    similar_groups.append(chunk_group)

            processing_time = time.time() - start_time
            method = "Gemini" if self.gemini_client else "TF-IDF"

            logger.info(
                f"{method} similarity analysis completed: found {len(similar_groups)} similar groups "
                f"from {len(valid_chunks)} valid chunks in {processing_time:.2f}s"
            )

            return similar_groups

        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}", exc_info=True)
            # Return empty groups on failure - don't break the pipeline
            return []

        finally:
            # Yield control to event loop
            await asyncio.sleep(0)

    async def _compute_gemini_similarity(self, documents: List[str]) -> np.ndarray:
        """
        Compute similarity matrix using Gemini embeddings.

        Args:
            documents: List of document texts

        Returns:
            Similarity matrix as numpy array
        """
        try:
            # Generate embeddings using Gemini
            if self.gemini_client is None:
                raise ValueError("Gemini client not initialized")
            result = self.gemini_client.models.embed_content(
                model="gemini-embedding-001",
                contents=documents,
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=3072,  # Full dimension for max accuracy
                ),
            )

            # Convert embeddings to numpy array
            embeddings = []
            if result.embeddings is None:
                raise ValueError("No embeddings returned from Gemini API")
            for embedding_obj in result.embeddings:
                embedding_values = np.array(embedding_obj.values)
                # Normalize the embedding (required for dimensions other than 3072)
                normalized_embedding = embedding_values / np.linalg.norm(
                    embedding_values
                )
                embeddings.append(normalized_embedding)

            embeddings_matrix = np.array(embeddings)

            # Calculate cosine similarity
            raw_similarity = cosine_similarity(embeddings_matrix)
            similarity_matrix: np.ndarray = np.asarray(raw_similarity, dtype=np.float64)

            logger.debug(f"Generated Gemini embeddings: {embeddings_matrix.shape}")
            return similarity_matrix

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise

    async def _compute_tfidf_similarity(self, documents: List[str]) -> np.ndarray:
        """
        Compute similarity matrix using TF-IDF (fallback method).

        Args:
            documents: List of document texts

        Returns:
            Similarity matrix as numpy array
        """
        # Preprocess documents
        processed_docs = [self._preprocess_content(doc) for doc in documents]

        # Create TF-IDF vectors with adaptive parameters
        if len(documents) <= 5:
            min_df_value = 1
            max_df_value = 1.0
        else:
            min_df_value = min(2, max(1, len(documents) // 4))
            max_df_value = max(0.85, min(0.95, 1.0 - (1.0 / len(documents))))

        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # Don't remove stop words for short texts
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=min_df_value,
            max_df=max_df_value,
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"\b\w+\b",
        )

        # Fit and transform documents
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)

        # Calculate cosine similarity
        raw_similarity = cosine_similarity(tfidf_matrix)
        similarity_matrix: np.ndarray = np.asarray(raw_similarity, dtype=np.float64)

        logger.debug(f"Generated TF-IDF vectors: {tfidf_matrix.shape}")
        return similarity_matrix

    def _preprocess_content(self, content: str) -> str:
        """
        Preprocess content for better TF-IDF analysis with semantic normalization.

        Args:
            content: Raw content string

        Returns:
            Preprocessed content optimized for similarity analysis
        """
        if not content:
            return ""

        import re

        # Remove excessive whitespace while preserving word boundaries
        processed = " ".join(content.split()).lower()

        # Replace common code artifacts that shouldn't affect similarity
        replacements = [
            ("```", ""),  # Remove code block markers
            ("<code>", ""),
            ("</code>", ""),  # Remove HTML code tags
            ("\t", " "),  # Replace tabs with spaces
        ]

        for old, new in replacements:
            processed = processed.replace(old, new)

        # Normalize technical abbreviations and synonyms for better matching
        tech_synonyms = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "api": "application programming interface",
            "js": "javascript",
            "css": "cascading style sheets",
            "html": "hypertext markup language",
        }

        for abbrev, full_form in tech_synonyms.items():
            # Replace abbreviations with full forms (word boundaries)
            processed = re.sub(rf"\b{re.escape(abbrev)}\b", full_form, processed)

        # Normalize similar words that mean the same thing
        word_synonyms = {
            "excellent": "great",
            "excels": "great",
            "analysis": "science",
            "applications": "apps",
            "application": "app",
            "development": "dev",
            "programming": "coding",
            "language": "lang",
            "frontend": "front end",
            "backend": "back end",
            "used for": "for",
            "is used for": "for",
        }

        for original, replacement in word_synonyms.items():
            processed = re.sub(rf"\b{re.escape(original)}\b", replacement, processed)

        # Remove common stopwords that don't add semantic value
        common_stopwords = [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        ]

        # Split into words, filter stopwords, rejoin
        words = processed.split()
        words = [word for word in words if word not in common_stopwords]
        processed = " ".join(words)

        # Limit length for performance (keep most important part)
        if len(processed) > 10000:
            # Keep beginning and end, which often contain key information
            processed = processed[:5000] + " ... " + processed[-5000:]

        return processed

    def _find_similarity_groups(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        Find groups of similar documents using connected components approach.

        Args:
            similarity_matrix: Symmetric matrix of pairwise similarities

        Returns:
            List of document index groups where each group contains similar documents
        """
        n_docs = similarity_matrix.shape[0]
        visited = set()
        groups = []

        def dfs(node: int, current_group: List[int]) -> None:
            """Depth-first search to find connected components."""
            if node in visited:
                return

            visited.add(node)
            current_group.append(node)

            # Find all similar nodes
            for neighbor in range(n_docs):
                if (
                    neighbor != node
                    and neighbor not in visited
                    and similarity_matrix[node, neighbor] >= self.similarity_threshold
                ):
                    dfs(neighbor, current_group)

        # Find all connected components
        for i in range(n_docs):
            if i not in visited:
                current_group: List[int] = []
                dfs(i, current_group)

                # Only add groups with multiple members
                if len(current_group) > 1:
                    groups.append(current_group)

        logger.debug(
            f"Found {len(groups)} similarity groups using threshold {self.similarity_threshold}"
        )

        return groups

    def calculate_pairwise_similarity(
        self, chunk1: DocumentationChunk, chunk2: DocumentationChunk
    ) -> float:
        """
        Calculate similarity score between two individual chunks.

        Args:
            chunk1: First chunk to compare
            chunk2: Second chunk to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not SKLEARN_AVAILABLE:
            return 0.0

        try:
            documents = [
                self._preprocess_content(chunk1.content),
                self._preprocess_content(chunk2.content),
            ]

            # Create temporary vectorizer for pair comparison with safe parameters
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                lowercase=True,
                max_features=1000,  # Smaller for pair comparison
                min_df=1,  # Must appear in at least 1 document
                max_df=1.0,  # Include all terms
            )

            tfidf_matrix = vectorizer.fit_transform(documents)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Return similarity between the two documents
            return float(similarity_matrix[0, 1])

        except Exception as e:
            logger.warning(f"Failed to calculate pairwise similarity: {e}")
            return 0.0


class ConflictResolver:
    """Intelligent conflict resolution for similar chunks based on quality metrics."""

    def __init__(self) -> None:
        """Initialize conflict resolver with statistics tracking."""
        self.resolution_stats = {
            "groups_processed": 0,
            "conflicts_resolved": 0,
            "chunks_merged": 0,
            "total_processing_time": 0.0,
        }

    async def resolve_similar_groups(
        self, similar_groups: List[List[DocumentationChunk]]
    ) -> List[DocumentationChunk]:
        """
        Resolve conflicts in similar groups by selecting the best representative from each group.

        Args:
            similar_groups: List of groups where each group contains similar chunks

        Returns:
            List of best representative chunks, one per group
        """
        if not similar_groups:
            return []

        start_time = time.time()
        resolved_chunks = []

        for group in similar_groups:
            self.resolution_stats["groups_processed"] += 1

            if len(group) == 1:
                # No conflict to resolve
                resolved_chunks.append(group[0])
            else:
                # Resolve conflict by selecting best chunk
                best_chunk = self._select_best_chunk(group)
                resolved_chunks.append(best_chunk)

                self.resolution_stats["conflicts_resolved"] += 1
                self.resolution_stats["chunks_merged"] += len(group) - 1

                logger.debug(
                    f"Resolved conflict: selected best from {len(group)} similar chunks "
                    f"(sources: {[getattr(c, 'source_context', 'unknown')[:15] for c in group]})"
                )

            # Yield control periodically for large datasets
            if self.resolution_stats["groups_processed"] % 100 == 0:
                await asyncio.sleep(0)

        processing_time = time.time() - start_time
        self.resolution_stats["total_processing_time"] += processing_time

        logger.info(
            f"Conflict resolution completed: {len(similar_groups)} groups processed, "
            f"{self.resolution_stats['chunks_merged']} chunks merged in {processing_time:.2f}s"
        )

        return resolved_chunks

    def _select_best_chunk(
        self, chunks: List[DocumentationChunk]
    ) -> DocumentationChunk:
        """
        Select the best chunk from a group of similar chunks using comprehensive quality scoring.

        Args:
            chunks: List of similar chunks to evaluate

        Returns:
            The chunk with the highest quality score
        """

        def calculate_quality_score(chunk: DocumentationChunk) -> float:
            """Calculate comprehensive quality score for chunk evaluation."""
            score = 0.0
            content = chunk.content.lower()

            # Content length score (logarithmic scaling to avoid over-weighting huge chunks)
            if len(chunk.content) > 0:
                length_score = min(np.log10(len(chunk.content)) / 2, 3.0)  # Cap at 3.0
                score += length_score

            # Code examples significantly boost score
            code_indicators = [
                "```",
                "def ",
                "class ",
                "function",
                "import ",
                "from ",
                "<code>",
                "const ",
                "let ",
                "var ",
                "async ",
                "await ",
            ]
            code_count = sum(content.count(indicator) for indicator in code_indicators)
            score += min(code_count * 0.8, 4.0)  # Cap at 4.0

            # API documentation and reference content
            api_indicators = [
                "parameter",
                "returns",
                "example",
                "usage",
                "method",
                "endpoint",
                "response",
                "request",
                "api",
                "function",
            ]
            api_count = sum(content.count(indicator) for indicator in api_indicators)
            score += min(api_count * 0.4, 3.0)  # Cap at 3.0

            # Completeness and quality indicators
            quality_indicators = [
                "complete",
                "comprehensive",
                "full",
                "detailed",
                "thorough",
                "reference",
                "documentation",
            ]
            quality_count = sum(
                content.count(indicator) for indicator in quality_indicators
            )
            score += min(quality_count * 0.5, 2.0)  # Cap at 2.0

            # Source context quality (heavily weighted)
            if hasattr(chunk, "source_context") and chunk.source_context:
                context = chunk.source_context.lower()
                if any(
                    term in context
                    for term in ["reference", "documentation", "complete"]
                ):
                    score += 3.0
                elif any(term in context for term in ["comprehensive", "full"]):
                    score += 2.5
                elif any(term in context for term in ["tutorial", "guide"]):
                    score += 1.5
                elif any(term in context for term in ["example", "sample"]):
                    score += 1.0

            # Token count quality (optimal range gets bonus)
            if hasattr(chunk, "token_count") and chunk.token_count:
                if 1000 <= chunk.token_count <= 15000:
                    score += 1.5  # Sweet spot for documentation
                elif 500 <= chunk.token_count <= 20000:
                    score += 1.0  # Good range
                elif chunk.token_count >= 200:
                    score += 0.5  # Acceptable

            # Download success and metadata quality
            if hasattr(chunk, "proxy_id") and chunk.proxy_id:
                score += 0.5

            if hasattr(chunk, "download_time") and chunk.download_time:
                # Reasonable download times indicate successful retrieval
                if 0.1 <= chunk.download_time <= 60:
                    score += 0.5

            if hasattr(chunk, "metadata") and chunk.metadata:
                score += min(len(chunk.metadata) * 0.1, 1.0)

            # Library ID presence
            if hasattr(chunk, "library_id") and chunk.library_id:
                score += 0.3

            # Recency bonus (newer chunks often have better content)
            if hasattr(chunk, "created_at") and chunk.created_at:
                from datetime import datetime

                age_hours = (datetime.now() - chunk.created_at).total_seconds() / 3600
                if age_hours <= 24:
                    score += 0.5  # Very recent
                elif age_hours <= 168:  # One week
                    score += 0.3

            return max(score, 0.0)  # Ensure non-negative

        # Calculate scores for all chunks
        scored_chunks = [(chunk, calculate_quality_score(chunk)) for chunk in chunks]

        # Select chunk with highest score
        best_chunk, best_score = max(scored_chunks, key=lambda x: x[1])

        # Log detailed selection rationale
        logger.debug(
            f"Selected chunk with score {best_score:.2f} from {len(chunks)} similar chunks:"
        )
        for chunk, score in sorted(scored_chunks, key=lambda x: x[1], reverse=True)[:3]:
            logger.debug(
                f"  - Score {score:.2f}: {getattr(chunk, 'source_context', 'unknown')[:20]}... "
                f"({len(chunk.content)} chars, {getattr(chunk, 'token_count', 0)} tokens)"
            )

        return best_chunk

    def get_resolution_stats(self) -> Dict[str, Any]:
        """Get comprehensive conflict resolution statistics."""
        stats = self.resolution_stats.copy()

        # Add derived metrics
        if stats["groups_processed"] > 0:
            stats["average_group_size"] = (
                stats["chunks_merged"] + stats["groups_processed"]
            ) / stats["groups_processed"]
            stats["conflict_rate"] = (
                stats["conflicts_resolved"] / stats["groups_processed"]
            )
        else:
            stats["average_group_size"] = 0.0
            stats["conflict_rate"] = 0.0

        return stats


class DeduplicationEngine:
    """
    Main deduplication engine coordinating hash-based exact deduplication
    and TF-IDF similarity analysis with intelligent conflict resolution.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        batch_size: int = 50,
        enable_similarity: bool = True,
        enable_hash_cache: bool = True,
    ):
        """
        Initialize deduplication engine with configurable parameters.

        Args:
            similarity_threshold: Similarity threshold for grouping (0.0-1.0)
            batch_size: Batch size for processing large chunk sets
            enable_similarity: Enable semantic similarity analysis
            enable_hash_cache: Enable hash result caching for performance
        """
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.enable_similarity = enable_similarity and SKLEARN_AVAILABLE

        # Initialize component processors
        self.exact_processor = ExactDuplicateProcessor(
            enable_hash_cache=enable_hash_cache
        )
        self.similarity_analyzer = SimilarityAnalyzer(
            similarity_threshold=similarity_threshold,
            max_features=5000,
            min_content_length=50,
        )
        self.conflict_resolver = ConflictResolver()

        # Performance and statistics tracking
        self.processing_stats: Dict[str, Any] = {
            "total_operations": 0,
            "total_chunks_processed": 0,
            "total_exact_duplicates_removed": 0,
            "total_similar_groups_found": 0,
            "total_similar_chunks_merged": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "peak_memory_mb": 0.0,
        }

        if not self.enable_similarity:
            logger.warning("Similarity analysis disabled (scikit-learn not available)")

    async def deduplicate_chunks(
        self, chunks: List[DocumentationChunk]
    ) -> DeduplicationResult:
        """
        Execute complete deduplication pipeline with performance optimization.

        Args:
            chunks: List of documentation chunks to deduplicate

        Returns:
            DeduplicationResult with metrics and deduplicated chunks
        """
        if not chunks:
            return DeduplicationResult(
                original_count=0,
                deduplicated_count=0,
                exact_duplicates_removed=0,
                similar_chunks_merged=0,
                processing_time=0.0,
                chunks=[],
            )

        start_time = time.time()
        original_count = len(chunks)

        logger.info(f"Starting deduplication pipeline for {original_count} chunks")

        try:
            # Step 1: Remove exact duplicates using hash-based approach
            logger.info("Phase 1: Removing exact duplicates...")
            unique_chunks = await self.exact_processor.remove_exact_duplicates(chunks)
            exact_duplicates_removed = original_count - len(unique_chunks)

            # Step 2: Semantic similarity analysis (if enabled and beneficial)
            similar_chunks_merged = 0
            if (
                self.enable_similarity
                and len(unique_chunks) > 1
                and len(unique_chunks) <= 10000
            ):  # Practical limit for similarity analysis
                logger.info("Phase 2: Analyzing semantic similarity...")

                if len(unique_chunks) > self.batch_size:
                    final_chunks = await self._batch_similarity_processing(
                        unique_chunks
                    )
                else:
                    final_chunks = await self._process_similarity_group(unique_chunks)

                similar_chunks_merged = len(unique_chunks) - len(final_chunks)
            else:
                final_chunks = unique_chunks
                if not self.enable_similarity:
                    logger.debug("Semantic similarity analysis disabled")
                elif len(unique_chunks) > 10000:
                    logger.info(
                        f"Skipping similarity analysis for {len(unique_chunks)} chunks (too large)"
                    )

            # Calculate comprehensive metrics
            processing_time = time.time() - start_time

            # Update global statistics
            self._update_processing_stats(
                original_count,
                exact_duplicates_removed,
                similar_chunks_merged,
                processing_time,
            )

            # Create comprehensive result
            result = DeduplicationResult(
                original_count=original_count,
                deduplicated_count=len(final_chunks),
                exact_duplicates_removed=exact_duplicates_removed,
                similar_chunks_merged=similar_chunks_merged,
                processing_time=processing_time,
                chunks=final_chunks,
                hash_performance=self.exact_processor.hasher.get_cache_stats(),
                similarity_performance={
                    "enabled": self.enable_similarity,
                    "threshold": self.similarity_threshold,
                    "sklearn_available": SKLEARN_AVAILABLE,
                },
            )

            # Log comprehensive summary
            dedup_ratio = result.deduplication_ratio
            efficiency = result.efficiency_score

            logger.info(
                f"Deduplication completed: {original_count} -> {len(final_chunks)} chunks "
                f"({dedup_ratio:.1%} reduction, {result.chunks_per_second:.0f} chunks/sec, "
                f"efficiency: {efficiency:.2f}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Deduplication pipeline failed: {e}", exc_info=True)
            # Return original chunks on failure to avoid data loss
            processing_time = time.time() - start_time
            return DeduplicationResult(
                original_count=original_count,
                deduplicated_count=original_count,
                exact_duplicates_removed=0,
                similar_chunks_merged=0,
                processing_time=processing_time,
                chunks=chunks,
            )

    async def _batch_similarity_processing(
        self, chunks: List[DocumentationChunk]
    ) -> List[DocumentationChunk]:
        """
        Process similarity detection in batches for memory efficiency with large chunk sets.

        Args:
            chunks: List of chunks to process in batches

        Returns:
            List of chunks after similarity-based deduplication
        """
        logger.info(f"Processing {len(chunks)} chunks in batches of {self.batch_size}")

        all_final_chunks = []
        batch_count = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(
                f"Processing batch {batch_num}/{batch_count} ({len(batch)} chunks)"
            )

            batch_result = await self._process_similarity_group(batch)
            all_final_chunks.extend(batch_result)

            # Yield control to event loop between batches
            await asyncio.sleep(0)

        logger.debug(
            f"Batch processing completed: {len(chunks)} -> {len(all_final_chunks)} chunks"
        )
        return all_final_chunks

    async def _process_similarity_group(
        self, chunks: List[DocumentationChunk]
    ) -> List[DocumentationChunk]:
        """
        Process similarity analysis for a group of chunks with conflict resolution.

        Args:
            chunks: List of chunks to analyze for similarity

        Returns:
            List of chunks after similarity-based deduplication
        """
        # Detect similar groups
        similar_groups = await self.similarity_analyzer.detect_similar_chunks(chunks)

        if not similar_groups:
            # No similar groups found, return all chunks
            return chunks

        # Resolve conflicts in similar groups
        resolved_chunks = await self.conflict_resolver.resolve_similar_groups(
            similar_groups
        )

        # Add chunks that weren't part of any similar group
        similar_chunk_ids = set()
        for group in similar_groups:
            for chunk in group:
                similar_chunk_ids.add(id(chunk))

        non_similar_chunks = [
            chunk for chunk in chunks if id(chunk) not in similar_chunk_ids
        ]

        final_chunks = resolved_chunks + non_similar_chunks

        logger.debug(
            f"Similarity processing: {len(chunks)} -> {len(final_chunks)} chunks "
            f"({len(similar_groups)} similar groups, {len(resolved_chunks)} resolved, "
            f"{len(non_similar_chunks)} unique)"
        )

        return final_chunks

    def _update_processing_stats(
        self,
        original_count: int,
        exact_removed: int,
        similar_merged: int,
        processing_time: float,
    ) -> None:
        """Update global processing statistics."""
        self.processing_stats["total_operations"] += 1
        self.processing_stats["total_chunks_processed"] += original_count
        self.processing_stats["total_exact_duplicates_removed"] += exact_removed
        self.processing_stats["total_similar_chunks_merged"] += similar_merged
        self.processing_stats["total_processing_time"] += processing_time

        # Calculate running average
        self.processing_stats["average_processing_time"] = (
            self.processing_stats["total_processing_time"]
            / self.processing_stats["total_operations"]
        )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics from all components."""
        stats = self.processing_stats.copy()

        # Add component-specific statistics
        stats.update(
            {
                "hasher_stats": self.exact_processor.hasher.get_cache_stats(),
                "conflict_resolution_stats": self.conflict_resolver.get_resolution_stats(),
                "similarity_config": {
                    "enabled": self.enable_similarity,
                    "threshold": self.similarity_threshold,
                    "batch_size": self.batch_size,
                    "sklearn_available": SKLEARN_AVAILABLE,
                    "xxhash_available": XXHASH_AVAILABLE,
                },
            }
        )

        # Calculate derived metrics
        if stats["total_chunks_processed"] > 0:
            stats["overall_deduplication_rate"] = (
                stats["total_exact_duplicates_removed"]
                + stats["total_similar_chunks_merged"]
            ) / stats["total_chunks_processed"]
            stats["average_chunks_per_second"] = (
                stats["total_chunks_processed"] / stats["total_processing_time"]
                if stats["total_processing_time"] > 0
                else 0
            )

        return stats

    async def benchmark_performance(
        self,
        test_chunks: List[DocumentationChunk],
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark deduplication performance with different configurations.

        Args:
            test_chunks: Chunks to use for benchmarking
            batch_sizes: Batch sizes to test (defaults to [25, 50, 100, 200])

        Returns:
            Performance benchmarking results
        """
        if batch_sizes is None:
            batch_sizes = [25, 50, 100, 200]

        results: Dict[str, Any] = {
            "test_chunk_count": len(test_chunks),
            "batch_size_results": {},
            "similarity_enabled": self.enable_similarity,
            "xxhash_available": XXHASH_AVAILABLE,
        }

        for batch_size in batch_sizes:
            if len(test_chunks) < batch_size:
                continue

            # Create test engine with specific configuration
            test_engine = DeduplicationEngine(
                similarity_threshold=self.similarity_threshold,
                batch_size=batch_size,
                enable_similarity=self.enable_similarity,
            )

            # Benchmark the deduplication process
            start_time = time.time()

            # Use a copy to avoid modifying original data
            result = await test_engine.deduplicate_chunks(test_chunks.copy())

            benchmark_time = time.time() - start_time

            results["batch_size_results"][batch_size] = {
                "processing_time": benchmark_time,
                "chunks_per_second": result.chunks_per_second,
                "deduplication_ratio": result.deduplication_ratio,
                "efficiency_score": result.efficiency_score,
                "exact_duplicates_removed": result.exact_duplicates_removed,
                "similar_chunks_merged": result.similar_chunks_merged,
            }

            logger.debug(
                f"Batch size {batch_size}: {benchmark_time:.2f}s, "
                f"{result.chunks_per_second:.0f} chunks/sec"
            )

        # Find optimal batch size
        if results["batch_size_results"]:
            optimal_batch = min(
                results["batch_size_results"].items(),
                key=lambda x: x[1]["processing_time"],
            )
            results["optimal_batch_size"] = optimal_batch[0]
            results["optimal_performance"] = optimal_batch[1]

        return results
