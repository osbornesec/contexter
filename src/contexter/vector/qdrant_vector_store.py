"""
Qdrant Vector Store - High-performance vector database client.

Provides comprehensive vector storage and retrieval capabilities with:
- Async-first design with connection pooling
- HNSW indexing with optimized parameters
- Batch operations with error handling
- Collection management and optimization
- Health monitoring and performance metrics
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
        CollectionInfo, FieldCondition, Filter, FilterSelector,
        PointStruct, ScoredPoint, SearchRequest, UpdateStatus
    )
    from qdrant_client.http.exceptions import ResponseHandlingException
except ImportError as e:
    raise ImportError(
        "Qdrant client not installed. Install with: pip install qdrant-client[grpc]"
    ) from e

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    """Configuration for Qdrant vector store."""

    host: str = Field(default="localhost", description="Qdrant server host")
    port: int = Field(default=6333, description="Qdrant HTTP port")
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    prefer_grpc: bool = Field(default=True, description="Use gRPC when available")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    collection_name: str = Field(default="contexter_documentation", description="Collection name")
    vector_size: int = Field(default=2048, description="Vector dimension size")
    distance_metric: str = Field(default="Cosine", description="Distance metric")

    # HNSW Configuration
    hnsw_m: int = Field(default=16, description="HNSW M parameter")
    hnsw_ef_construct: int = Field(default=200, description="HNSW ef_construct parameter")
    hnsw_full_scan_threshold: int = Field(default=10000, description="Full scan threshold")
    hnsw_max_indexing_threads: int = Field(default=0, description="Max indexing threads (0=auto)")

    # Performance tuning
    max_concurrent_operations: int = Field(default=10, description="Max concurrent operations")
    batch_size: int = Field(default=1000, description="Default batch size")
    search_ef: int = Field(default=128, description="Search ef parameter")

    # Health and monitoring
    health_check_interval: float = Field(default=30.0, description="Health check interval")
    enable_performance_monitoring: bool = Field(default=True, description="Enable metrics")


class VectorDocument(BaseModel):
    """Vector document model for storage."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float] = Field(..., description="Vector embeddings")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Metadata payload")


class SearchResult(BaseModel):
    """Search result model."""

    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class QdrantVectorStore:
    """
    High-performance Qdrant vector store with async support.
    
    Features:
    - Async-first design with connection pooling
    - HNSW indexing optimization for 2048-dimensional vectors
    - Batch operations with error handling and retries
    - Collection management with automated optimization
    - Health monitoring and performance metrics
    """

    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.client: Optional[AsyncQdrantClient] = None
        self._initialized = False
        self._connection_semaphore = asyncio.Semaphore(config.max_concurrent_operations)
        self._last_health_check = 0.0
        self._health_status = False

        # Performance metrics
        self._metrics = {
            "operations_count": 0,
            "search_latency_p95": 0.0,
            "batch_upload_time": 0.0,
            "error_count": 0,
            "last_optimization": None
        }

    async def initialize(self) -> None:
        """Initialize the vector store and create collection if needed."""
        if self._initialized:
            return

        logger.info(f"Initializing Qdrant vector store on {self.config.host}:{self.config.port}")

        try:
            # Initialize async client
            self.client = AsyncQdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port if self.config.prefer_grpc else None,
                prefer_grpc=self.config.prefer_grpc,
                timeout=self.config.timeout
            )

            # Verify connection
            await self._verify_connection()

            # Create collection if it doesn't exist
            await self._ensure_collection_exists()

            # Perform initial health check
            await self._health_check()

            self._initialized = True
            logger.info("Qdrant vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant vector store: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources and close connections."""
        if self.client:
            await self.client.close()
            self.client = None
        self._initialized = False
        logger.info("Qdrant vector store cleanup completed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _verify_connection(self) -> None:
        """Verify connection to Qdrant server."""
        try:
            # Test connection with a simple operation
            collections = await self.client.get_collections()
            logger.debug(f"Connected to Qdrant. Found {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise

    async def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.config.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.config.collection_name}")
                await self._create_collection()
            else:
                # Verify collection configuration
                collection_info = await self.client.get_collection(self.config.collection_name)
                await self._verify_collection_config(collection_info)

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    async def _create_collection(self) -> None:
        """Create a new collection with optimized configuration."""
        try:
            # HNSW configuration optimized for 2048-dimensional vectors
            hnsw_config = HnswConfigDiff(
                m=self.config.hnsw_m,
                ef_construct=self.config.hnsw_ef_construct,
                full_scan_threshold=self.config.hnsw_full_scan_threshold,
                max_indexing_threads=self.config.hnsw_max_indexing_threads
            )

            # Optimizer configuration for performance
            optimizer_config = OptimizersConfigDiff(
                indexing_threshold=20000,  # Start indexing after 20k vectors
                flush_interval_sec=5,      # Flush every 5 seconds
                max_optimization_threads=0 # Use all available cores
            )

            # Vector configuration
            vectors_config = VectorParams(
                size=self.config.vector_size,
                distance=getattr(Distance, self.config.distance_metric.upper())
            )

            # Create collection
            result = await self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config
            )

            if result:
                logger.info(f"Collection '{self.config.collection_name}' created successfully")
                await self._create_payload_indexes()
            else:
                raise Exception("Collection creation returned false")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def _create_payload_indexes(self) -> None:
        """Create indexes on payload fields for efficient filtering."""
        indexed_fields = [
            "library_id",
            "doc_type",
            "section",
            "timestamp",
            "programming_language",
            "trust_score"
        ]

        for field in indexed_fields:
            try:
                await self.client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field,
                    field_schema="keyword" if field in ["library_id", "doc_type", "section", "programming_language"] else "integer"
                )
                logger.debug(f"Created payload index for field: {field}")
            except Exception as e:
                logger.warning(f"Failed to create index for {field}: {e}")

    async def _verify_collection_config(self, collection_info: CollectionInfo) -> None:
        """Verify collection configuration matches requirements."""
        config = collection_info.config

        # Verify vector size
        if config.params.vectors.size != self.config.vector_size:
            logger.warning(
                f"Vector size mismatch: expected {self.config.vector_size}, "
                f"got {config.params.vectors.size}"
            )

        # Verify distance metric
        expected_distance = getattr(Distance, self.config.distance_metric.upper())
        if config.params.vectors.distance != expected_distance:
            logger.warning(
                f"Distance metric mismatch: expected {expected_distance}, "
                f"got {config.params.vectors.distance}"
            )

    async def _health_check(self) -> bool:
        """Perform health check on the vector store."""
        try:
            # Check if enough time has passed since last health check
            current_time = time.time()
            if current_time - self._last_health_check < self.config.health_check_interval:
                return self._health_status

            # Perform health check
            collection_info = await self.client.get_collection(self.config.collection_name)

            # Update metrics
            self._health_status = True
            self._last_health_check = current_time

            if self.config.enable_performance_monitoring:
                await self._update_metrics(collection_info)

            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self._health_status = False
            return False

    async def _update_metrics(self, collection_info: CollectionInfo) -> None:
        """Update performance metrics."""
        try:
            points_count = collection_info.points_count or 0
            vectors_count = collection_info.vectors_count or 0

            self._metrics.update({
                "points_count": points_count,
                "vectors_count": vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                "collection_status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status.value if collection_info.optimizer_status else "unknown"
            })

        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")

    async def upsert_vector(self, document: VectorDocument) -> bool:
        """
        Insert or update a single vector document.
        
        Args:
            document: Vector document to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._connection_semaphore:
                point = PointStruct(
                    id=document.id,
                    vector=document.vector,
                    payload=document.payload
                )

                result = await self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=[point]
                )

                self._metrics["operations_count"] += 1
                return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to upsert vector {document.id}: {e}")
            self._metrics["error_count"] += 1
            return False

    async def upsert_vectors_batch(
        self,
        documents: List[VectorDocument],
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in batches.
        
        Args:
            documents: List of vector documents to store
            batch_size: Size of each batch (defaults to config.batch_size)
            
        Returns:
            Results summary with success/failure counts
        """
        if not documents:
            return {"successful_uploads": 0, "failed_uploads": 0, "total_time": 0.0}

        batch_size = batch_size or self.config.batch_size
        start_time = time.time()

        successful_uploads = 0
        failed_uploads = 0

        try:
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]

                points = [
                    PointStruct(
                        id=doc.id,
                        vector=doc.vector,
                        payload=doc.payload
                    )
                    for doc in batch
                ]

                try:
                    async with self._connection_semaphore:
                        result = await self.client.upsert(
                            collection_name=self.config.collection_name,
                            points=points
                        )

                        if result.status == UpdateStatus.COMPLETED:
                            successful_uploads += len(batch)
                        else:
                            failed_uploads += len(batch)
                            logger.warning(f"Batch upload failed with status: {result.status}")

                except Exception as e:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
                    failed_uploads += len(batch)

            total_time = time.time() - start_time
            self._metrics["batch_upload_time"] = total_time
            self._metrics["operations_count"] += len(documents)

            logger.info(
                f"Batch upload completed: {successful_uploads} successful, "
                f"{failed_uploads} failed, {total_time:.2f}s"
            )

            return {
                "successful_uploads": successful_uploads,
                "failed_uploads": failed_uploads,
                "total_time": total_time
            }

        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            self._metrics["error_count"] += 1
            raise

    async def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector for similarity search
            top_k: Number of results to return
            filters: Metadata filters to apply
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores and metadata
        """
        start_time = time.time()

        try:
            async with self._connection_semaphore:
                # Build query filter if provided
                query_filter = self._build_filter(filters) if filters else None

                # Search parameters optimized for accuracy
                search_params = {"ef": self.config.search_ef}

                results = await self.client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=query_filter,
                    search_params=search_params,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False  # Don't return vectors unless needed
                )

                # Update performance metrics
                search_time = time.time() - start_time
                self._metrics["search_latency_p95"] = max(
                    self._metrics["search_latency_p95"], search_time
                )
                self._metrics["operations_count"] += 1

                # Format results
                search_results = [
                    SearchResult(
                        id=str(result.id),
                        score=result.score,
                        payload=result.payload or {}
                    )
                    for result in results
                ]

                logger.debug(
                    f"Search completed: {len(search_results)} results in {search_time:.3f}s"
                )

                return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            self._metrics["error_count"] += 1
            raise

    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary."""
        conditions = []

        for field, value in filters.items():
            if isinstance(value, list):
                # IN filter for list values
                for v in value:
                    conditions.append(
                        FieldCondition(key=field, match={"value": v})
                    )
            elif isinstance(value, dict):
                # Range filter for dict values with 'gte', 'lte', etc.
                for op, val in value.items():
                    if op == "gte":
                        conditions.append(
                            FieldCondition(key=field, range={"gte": val})
                        )
                    elif op == "lte":
                        conditions.append(
                            FieldCondition(key=field, range={"lte": val})
                        )
                    elif op == "gt":
                        conditions.append(
                            FieldCondition(key=field, range={"gt": val})
                        )
                    elif op == "lt":
                        conditions.append(
                            FieldCondition(key=field, range={"lt": val})
                        )
            else:
                # Exact match filter
                conditions.append(
                    FieldCondition(key=field, match={"value": value})
                )

        return Filter(must=conditions) if conditions else None

    async def get_vector(self, vector_id: str) -> Optional[VectorDocument]:
        """
        Retrieve a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            Vector document if found, None otherwise
        """
        try:
            async with self._connection_semaphore:
                results = await self.client.retrieve(
                    collection_name=self.config.collection_name,
                    ids=[vector_id],
                    with_payload=True,
                    with_vectors=True
                )

                if results and len(results) > 0:
                    point = results[0]
                    return VectorDocument(
                        id=str(point.id),
                        vector=point.vector,
                        payload=point.payload or {}
                    )

                return None

        except Exception as e:
            logger.error(f"Failed to retrieve vector {vector_id}: {e}")
            return None

    async def delete_vector(self, vector_id: str) -> bool:
        """
        Delete a specific vector by ID.
        
        Args:
            vector_id: ID of the vector to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self._connection_semaphore:
                result = await self.client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=[vector_id]
                )

                return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def delete_vectors_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Delete vectors matching the given filters.
        
        Args:
            filters: Filters to identify vectors to delete
            
        Returns:
            Number of vectors deleted
        """
        try:
            async with self._connection_semaphore:
                query_filter = self._build_filter(filters)

                result = await self.client.delete(
                    collection_name=self.config.collection_name,
                    points_selector=FilterSelector(filter=query_filter)
                )

                if result.status == UpdateStatus.COMPLETED:
                    # Note: Qdrant doesn't return exact count, so we estimate
                    logger.info("Bulk delete completed successfully")
                    return 1  # Return success indicator

                return 0

        except Exception as e:
            logger.error(f"Failed to delete vectors by filter: {e}")
            return 0

    async def count_vectors(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count vectors in the collection.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of vectors matching the criteria
        """
        try:
            async with self._connection_semaphore:
                query_filter = self._build_filter(filters) if filters else None

                result = await self.client.count(
                    collection_name=self.config.collection_name,
                    count_filter=query_filter
                )

                return result.count

        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0

    async def optimize_collection(self) -> bool:
        """
        Optimize the collection for better search performance.
        
        Returns:
            True if optimization started successfully
        """
        try:
            async with self._connection_semaphore:
                # Check current collection status
                collection_info = await self.client.get_collection(self.config.collection_name)

                # Only optimize if there are enough vectors
                if collection_info.points_count and collection_info.points_count > 10000:
                    logger.info("Starting collection optimization...")

                    result = await self.client.update_collection(
                        collection_name=self.config.collection_name,
                        optimizer_config=OptimizersConfigDiff(
                            indexing_threshold=max(20000, collection_info.points_count // 10)
                        )
                    )

                    if result:
                        self._metrics["last_optimization"] = datetime.now().isoformat()
                        logger.info("Collection optimization started successfully")
                        return True

                return False

        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            async with self._connection_semaphore:
                collection_info = await self.client.get_collection(self.config.collection_name)

                stats = {
                    "collection_name": self.config.collection_name,
                    "points_count": collection_info.points_count or 0,
                    "vectors_count": collection_info.vectors_count or 0,
                    "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                    "status": collection_info.status.value,
                    "optimizer_status": collection_info.optimizer_status.value if collection_info.optimizer_status else "unknown",
                    "config": {
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.value,
                        "hnsw_m": collection_info.config.hnsw_config.m if collection_info.config.hnsw_config else None,
                        "hnsw_ef_construct": collection_info.config.hnsw_config.ef_construct if collection_info.config.hnsw_config else None
                    },
                    "performance_metrics": self._metrics.copy()
                }

                return stats

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def vacuum_collection(self) -> bool:
        """
        Vacuum the collection to reclaim deleted vector space.
        
        Returns:
            True if vacuum completed successfully
        """
        try:
            async with self._connection_semaphore:
                # Note: In Qdrant, vacuum is typically handled automatically
                # But we can trigger optimization which includes cleanup
                logger.info("Starting collection vacuum...")

                result = await self.optimize_collection()

                if result:
                    logger.info("Collection vacuum completed")

                return result

        except Exception as e:
            logger.error(f"Failed to vacuum collection: {e}")
            return False

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get current health status and metrics.
        
        Returns:
            Health status information
        """
        return {
            "healthy": self._health_status,
            "last_check": self._last_health_check,
            "initialized": self._initialized,
            "metrics": self._metrics.copy(),
            "config": {
                "collection_name": self.config.collection_name,
                "vector_size": self.config.vector_size,
                "distance_metric": self.config.distance_metric,
                "batch_size": self.config.batch_size,
                "search_ef": self.config.search_ef
            }
        }
