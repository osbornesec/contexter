"""
Embedding Cache System

High-performance SQLite-based embedding cache with LRU eviction,
intelligent cleanup, and comprehensive performance monitoring.
"""

import asyncio
import json
import logging
import pickle
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiosqlite

from ..models.embedding_models import (
    IEmbeddingCache, CacheEntry, CacheStats, InputType
)

logger = logging.getLogger(__name__)


class EmbeddingCache(IEmbeddingCache):
    """
    High-performance SQLite-based embedding cache with advanced features.
    
    Features:
    - LRU eviction with configurable thresholds
    - Automatic cleanup of expired entries
    - Comprehensive statistics and monitoring
    - Atomic operations for data integrity
    - Connection pooling for concurrent access
    """
    
    def __init__(
        self,
        cache_path: str = "~/.contexter/embedding_cache.db",
        max_entries: int = 100000,
        ttl_hours: int = 168,  # 7 days
        cleanup_threshold: float = 0.8,  # Clean when 80% full
        enable_wal_mode: bool = True
    ):
        """
        Initialize embedding cache.
        
        Args:
            cache_path: Path to SQLite database file
            max_entries: Maximum number of cached entries
            ttl_hours: Time-to-live for cache entries in hours
            cleanup_threshold: Threshold for triggering cleanup (0.0-1.0)
            enable_wal_mode: Enable WAL mode for better concurrency
        """
        self.cache_path = Path(cache_path).expanduser()
        self.max_entries = max_entries
        self.ttl_hours = ttl_hours
        self.ttl_seconds = ttl_hours * 3600
        self.cleanup_threshold = cleanup_threshold
        self.enable_wal_mode = enable_wal_mode
        
        # Statistics tracking
        self.stats = CacheStats()
        self._stats_lock = asyncio.Lock()
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Performance tracking
        self._operation_times = []
        self._last_cleanup = datetime.utcnow()
        
        # Connection pool for concurrent access
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_lock = asyncio.Lock()
        self._max_connections = 10
        
        # Initialize flag
        self._initialized = False
    
    async def initialize(self):
        """Initialize the cache database and start background tasks."""
        if self._initialized:
            return
        
        try:
            # Create directory if needed
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database schema
            await self._init_database()
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._background_cleanup())
            
            # Load initial statistics
            await self._update_statistics()
            
            self._initialized = True
            logger.info(f"Embedding cache initialized at {self.cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding cache: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the cache and cleanup resources."""
        logger.info("Shutting down embedding cache")
        
        # Stop background tasks
        self._shutdown_event.set()
        if self._cleanup_task:
            await self._cleanup_task
        
        # Close all connections
        async with self._pool_lock:
            for conn in self._connection_pool:
                await conn.close()
            self._connection_pool.clear()
        
        self._initialized = False
        logger.info("Embedding cache shutdown complete")
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    async def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        async with aiosqlite.connect(self.cache_path) as db:
            # Enable WAL mode for better concurrency
            if self.enable_wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")
            
            # Performance optimizations
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            await db.execute("PRAGMA temp_store=MEMORY")
            
            # Create main table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Indexes for performance
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_accessed_at 
                ON embeddings(accessed_at)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_type 
                ON embeddings(model, input_type)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON embeddings(created_at)
            """)
            
            # Statistics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache_statistics (
                    id INTEGER PRIMARY KEY,
                    total_entries INTEGER DEFAULT 0,
                    total_hits INTEGER DEFAULT 0,
                    total_misses INTEGER DEFAULT 0,
                    total_evictions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Initialize statistics if not exists
            await db.execute("""
                INSERT OR IGNORE INTO cache_statistics (id) VALUES (1)
            """)
            
            await db.commit()
    
    async def _get_connection(self) -> aiosqlite.Connection:
        """Get a connection from the pool or create a new one."""
        async with self._pool_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            
            # Create new connection if pool is empty
            if len(self._connection_pool) < self._max_connections:
                conn = await aiosqlite.connect(self.cache_path)
                
                # Configure connection
                if self.enable_wal_mode:
                    await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA synchronous=NORMAL")
                
                return conn
            
            # If we can't create more connections, wait and retry
            await asyncio.sleep(0.01)
            return await self._get_connection()
    
    async def _return_connection(self, conn: aiosqlite.Connection):
        """Return a connection to the pool."""
        async with self._pool_lock:
            if len(self._connection_pool) < self._max_connections:
                self._connection_pool.append(conn)
            else:
                await conn.close()
    
    async def get_embeddings(
        self,
        content_hashes: List[str],
        model: str = "voyage-code-3",
        input_type: InputType = InputType.DOCUMENT
    ) -> Dict[str, List[float]]:
        """
        Retrieve cached embeddings by content hash.
        
        Args:
            content_hashes: List of content hashes to look up
            model: Model name for filtering
            input_type: Input type for filtering
            
        Returns:
            Dictionary mapping content hash to embedding vector
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        cached_embeddings = {}
        
        conn = await self._get_connection()
        try:
            # Query in batches for better performance
            batch_size = 100
            for i in range(0, len(content_hashes), batch_size):
                batch = content_hashes[i:i + batch_size]
                placeholders = ",".join("?" * len(batch))
                
                query = f"""
                    SELECT content_hash, embedding, dimensions, access_count
                    FROM embeddings 
                    WHERE content_hash IN ({placeholders})
                    AND model = ? 
                    AND input_type = ?
                    AND datetime(created_at, '+{self.ttl_seconds} seconds') > datetime('now')
                """
                
                params = batch + [model, input_type.value]
                
                async with conn.execute(query, params) as cursor:
                    async for row in cursor:
                        content_hash, embedding_blob, dimensions, access_count = row
                        
                        try:
                            # Deserialize embedding
                            embedding = pickle.loads(embedding_blob)
                            
                            # Validate dimensions
                            if len(embedding) != dimensions:
                                logger.warning(f"Dimension mismatch for {content_hash}")
                                continue
                            
                            cached_embeddings[content_hash] = embedding
                            
                            # Update access statistics (async)
                            asyncio.create_task(self._update_access_stats(conn, content_hash))
                            
                        except (pickle.UnpicklingError, TypeError) as e:
                            logger.warning(f"Failed to deserialize embedding for {content_hash}: {e}")
                            continue
            
            # Update cache statistics
            hits = len(cached_embeddings)
            misses = len(content_hashes) - hits
            
            async with self._stats_lock:
                self.stats.hits += hits
                self.stats.misses += misses
                self.stats.update_hit_rate()
            
            # Update database statistics
            if hits > 0 or misses > 0:
                await conn.execute("""
                    UPDATE cache_statistics 
                    SET total_hits = total_hits + ?,
                        total_misses = total_misses + ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (hits, misses))
                await conn.commit()
            
            # Track operation time
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)
            if len(self._operation_times) > 1000:
                self._operation_times = self._operation_times[-1000:]
            
            logger.debug(
                f"Cache lookup: {hits} hits, {misses} misses in {operation_time:.3f}s"
            )
            
            return cached_embeddings
        
        finally:
            await self._return_connection(conn)
    
    async def _update_access_stats(self, conn: aiosqlite.Connection, content_hash: str):
        """Update access statistics for a cache entry."""
        try:
            await conn.execute("""
                UPDATE embeddings 
                SET accessed_at = CURRENT_TIMESTAMP,
                    access_count = access_count + 1
                WHERE content_hash = ?
            """, (content_hash,))
            await conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update access stats for {content_hash}: {e}")
    
    async def store_embeddings(self, entries: List[CacheEntry]) -> bool:
        """
        Store embeddings in cache.
        
        Args:
            entries: List of cache entries to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        if not entries:
            return True
        
        start_time = time.time()
        
        conn = await self._get_connection()
        try:
            # Prepare data for bulk insert
            insert_data = []
            for entry in entries:
                embedding_blob = pickle.dumps(entry.embedding)
                metadata_json = json.dumps(entry.metadata)
                
                insert_data.append((
                    entry.content_hash,
                    entry.content,
                    entry.model,
                    entry.input_type.value,
                    embedding_blob,
                    len(entry.embedding),
                    metadata_json
                ))
            
            # Bulk insert with conflict resolution
            await conn.executemany("""
                INSERT OR REPLACE INTO embeddings 
                (content_hash, content, model, input_type, embedding, dimensions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, insert_data)
            
            await conn.commit()
            
            # Update statistics
            async with self._stats_lock:
                self.stats.total_entries += len(entries)
            
            # Check if cleanup is needed
            if await self._should_cleanup():
                asyncio.create_task(self._cleanup_old_entries())
            
            operation_time = time.time() - start_time
            logger.debug(f"Stored {len(entries)} embeddings in {operation_time:.3f}s")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to store embeddings: {e}")
            return False
        
        finally:
            await self._return_connection(conn)
    
    async def _should_cleanup(self) -> bool:
        """Check if cache cleanup is needed."""
        # Check entry count threshold
        current_count = await self._get_entry_count()
        if current_count >= (self.max_entries * self.cleanup_threshold):
            return True
        
        # Check time-based cleanup (daily)
        if (datetime.utcnow() - self._last_cleanup).total_seconds() >= 86400:
            return True
        
        return False
    
    async def _get_entry_count(self) -> int:
        """Get current number of entries in cache."""
        conn = await self._get_connection()
        try:
            async with conn.execute("SELECT COUNT(*) FROM embeddings") as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0
        finally:
            await self._return_connection(conn)
    
    async def clear_expired(self, ttl_hours: int = 168) -> int:
        """
        Clear expired cache entries.
        
        Args:
            ttl_hours: Time-to-live in hours
            
        Returns:
            Number of entries removed
        """
        if not self._initialized:
            await self.initialize()
        
        conn = await self._get_connection()
        try:
            # Delete expired entries
            cutoff_time = datetime.utcnow() - timedelta(hours=ttl_hours)
            
            result = await conn.execute("""
                DELETE FROM embeddings 
                WHERE created_at < ?
            """, (cutoff_time.isoformat(),))
            
            deleted_count = result.rowcount
            await conn.commit()
            
            # Update statistics
            async with self._stats_lock:
                self.stats.evictions += deleted_count
                self.stats.total_entries -= deleted_count
            
            logger.info(f"Cleared {deleted_count} expired cache entries")
            
            return deleted_count
        
        finally:
            await self._return_connection(conn)
    
    async def _cleanup_old_entries(self):
        """Clean up old entries using LRU eviction."""
        start_time = time.time()
        
        conn = await self._get_connection()
        try:
            # Get current count
            current_count = await self._get_entry_count()
            
            if current_count <= self.max_entries:
                return
            
            # Calculate how many entries to remove
            target_count = int(self.max_entries * 0.8)  # Remove to 80% capacity
            entries_to_remove = current_count - target_count
            
            # Remove least recently accessed entries
            result = await conn.execute("""
                DELETE FROM embeddings 
                WHERE content_hash IN (
                    SELECT content_hash 
                    FROM embeddings 
                    ORDER BY accessed_at ASC, access_count ASC
                    LIMIT ?
                )
            """, (entries_to_remove,))
            
            removed_count = result.rowcount
            await conn.commit()
            
            # Update statistics
            async with self._stats_lock:
                self.stats.evictions += removed_count
                self.stats.total_entries -= removed_count
            
            await conn.execute("""
                UPDATE cache_statistics 
                SET total_evictions = total_evictions + ?,
                    total_entries = total_entries - ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (removed_count, removed_count))
            await conn.commit()
            
            cleanup_time = time.time() - start_time
            logger.info(
                f"LRU cleanup: removed {removed_count} entries in {cleanup_time:.2f}s"
            )
            
            self._last_cleanup = datetime.utcnow()
        
        finally:
            await self._return_connection(conn)
    
    async def _background_cleanup(self):
        """Background task for periodic cache maintenance."""
        logger.info("Starting background cache cleanup task")
        
        while not self._shutdown_event.is_set():
            try:
                # Wait for next cleanup cycle (1 hour)
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=3600  # 1 hour
                )
            except asyncio.TimeoutError:
                # Timeout is expected, continue with cleanup
                pass
            
            if self._shutdown_event.is_set():
                break
            
            try:
                # Perform maintenance tasks
                await self.clear_expired(self.ttl_hours)
                await self._cleanup_old_entries()
                await self._update_statistics()
                await self._optimize_database()
                
            except Exception as e:
                logger.error(f"Background cleanup failed: {e}")
        
        logger.info("Background cache cleanup task stopped")
    
    async def _optimize_database(self):
        """Optimize database performance."""
        conn = await self._get_connection()
        try:
            # Analyze tables for query optimization
            await conn.execute("ANALYZE")
            
            # Vacuum if needed (WAL mode doesn't benefit from VACUUM as much)
            if not self.enable_wal_mode:
                await conn.execute("VACUUM")
            
            await conn.commit()
            logger.debug("Database optimization completed")
        
        except Exception as e:
            logger.warning(f"Database optimization failed: {e}")
        
        finally:
            await self._return_connection(conn)
    
    async def _update_statistics(self):
        """Update comprehensive cache statistics."""
        conn = await self._get_connection()
        try:
            # Get detailed statistics
            async with conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(accessed_at) as newest_access,
                    SUM(LENGTH(embedding)) as total_size_bytes
                FROM embeddings
            """) as cursor:
                row = await cursor.fetchone()
                
                if row and row[0] > 0:
                    (total_entries, avg_access_count, oldest_entry, 
                     newest_access, total_size_bytes) = row
                    
                    async with self._stats_lock:
                        self.stats.total_entries = total_entries
                        self.stats.avg_access_count = avg_access_count or 0
                        self.stats.total_size_bytes = total_size_bytes or 0
                        
                        if oldest_entry:
                            self.stats.oldest_entry = datetime.fromisoformat(oldest_entry)
                        if newest_access:
                            self.stats.newest_entry = datetime.fromisoformat(newest_access)
            
            # Get hit/miss statistics from database
            async with conn.execute("""
                SELECT total_hits, total_misses, total_evictions
                FROM cache_statistics WHERE id = 1
            """) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    db_hits, db_misses, db_evictions = row
                    
                    async with self._stats_lock:
                        # Sync with database stats (they may be more persistent)
                        self.stats.hits = max(self.stats.hits, db_hits or 0)
                        self.stats.misses = max(self.stats.misses, db_misses or 0)
                        self.stats.evictions = max(self.stats.evictions, db_evictions or 0)
                        self.stats.update_hit_rate()
        
        finally:
            await self._return_connection(conn)
    
    async def get_statistics(self) -> CacheStats:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Current cache statistics
        """
        if not self._initialized:
            await self.initialize()
        
        await self._update_statistics()
        
        async with self._stats_lock:
            return CacheStats(
                total_entries=self.stats.total_entries,
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                total_size_bytes=self.stats.total_size_bytes,
                hit_rate=self.stats.hit_rate,
                avg_access_count=self.stats.avg_access_count,
                oldest_entry=self.stats.oldest_entry,
                newest_entry=self.stats.newest_entry
            )
    
    async def get_performance_info(self) -> Dict[str, Any]:
        """Get cache performance information."""
        stats = await self.get_statistics()
        
        # Calculate operation time percentiles
        percentiles = {}
        if self._operation_times:
            sorted_times = sorted(self._operation_times)
            n = len(sorted_times)
            percentiles = {
                "p50_ms": round(sorted_times[int(n * 0.5)] * 1000, 2),
                "p95_ms": round(sorted_times[int(n * 0.95)] * 1000, 2),
                "p99_ms": round(sorted_times[int(n * 0.99)] * 1000, 2)
            }
        
        return {
            "statistics": {
                "total_entries": stats.total_entries,
                "hit_rate": round(stats.hit_rate, 3),
                "total_size_mb": round(stats.total_size_bytes / (1024 * 1024), 2),
                "avg_access_count": round(stats.avg_access_count, 1)
            },
            "performance": percentiles,
            "configuration": {
                "max_entries": self.max_entries,
                "ttl_hours": self.ttl_hours,
                "cleanup_threshold": self.cleanup_threshold,
                "wal_mode": self.enable_wal_mode
            },
            "health": {
                "last_cleanup": self._last_cleanup.isoformat(),
                "connection_pool_size": len(self._connection_pool)
            }
        }