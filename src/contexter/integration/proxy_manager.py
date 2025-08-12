"""
BrightData proxy management with health monitoring and circuit breaker patterns.
"""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from ..core.environment_manager import EnvironmentManager
from ..models.proxy_models import (
    CircuitBreaker,
    ProxyConnection,
    ProxyMetrics,
    ProxyStatus,
)

logger = logging.getLogger(__name__)


class ProxyManagerError(Exception):
    """Base exception for proxy management errors."""

    pass


class BrightDataProxyManager:
    """Main proxy management interface for BrightData integration."""

    def __init__(
        self,
        customer_id: str,
        password: str,
        zone_name: str = "zone_residential_1",
        dns_resolution: str = "remote",
    ):
        """
        Initialize BrightData proxy manager.

        Args:
            customer_id: BrightData customer ID
            password: BrightData password
            zone_name: BrightData zone name (default: zone_residential_1)
            dns_resolution: DNS resolution mode - 'local' (resolved by Super Proxy) or 'remote' (resolved by target proxy)
        """
        self.customer_id = customer_id
        self.password = password
        self.zone_name = zone_name
        self.dns_resolution = dns_resolution
        self.proxy_pool: List[ProxyConnection] = []
        self.proxy_metrics: dict[str, ProxyMetrics] = {}
        self._session_counter = 0
        self._current_index = 0
        self._lock = asyncio.Lock()
        self.monitoring_active = False
        self.health_check_task: Optional[asyncio.Task[None]] = None
        self.ssl_cert_path = self._find_ssl_certificate()

        logger.info(
            f"Initialized BrightData proxy manager for customer {customer_id[:8]}..."
        )
        if self.ssl_cert_path:
            logger.info(f"Found SSL certificate: {self.ssl_cert_path.name}")

    def _find_ssl_certificate(self) -> Optional[Path]:
        """
        Find BrightData SSL certificate in standard locations.

        Returns:
            Path to certificate file if found, None otherwise
        """
        search_paths = [
            Path.home() / ".contexter",
            Path.cwd(),
            Path(__file__).parent.parent.parent,  # Project root
        ]

        cert_patterns = [
            "*BrightData*certificate*.crt",
            "*brightdata*cert*.crt",
            "brightdata*.pem",
            "brightdata*.crt",
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for pattern in cert_patterns:
                cert_files = list(search_path.glob(pattern))
                if cert_files:
                    cert_file = cert_files[0]  # Use first match
                    logger.debug(f"Found SSL certificate: {cert_file}")
                    return cert_file

        logger.warning("No BrightData SSL certificate found")
        return None

    @classmethod
    async def from_environment(
        cls, zone_name: str = "zone_residential_1", dns_resolution: str = "remote"
    ) -> "BrightDataProxyManager":
        """
        Create proxy manager using environment variables.

        Args:
            zone_name: BrightData zone name
            dns_resolution: DNS resolution mode ('local' or 'remote')

        Returns:
            BrightDataProxyManager instance

        Raises:
            ProxyManagerError: If environment variables are not set
        """
        try:
            env_manager = EnvironmentManager()
            credentials = env_manager.get_brightdata_credentials()

            return cls(
                customer_id=credentials["customer_id"],
                password=credentials["password"],
                zone_name=zone_name,
                dns_resolution=dns_resolution,
            )
        except Exception as e:
            raise ProxyManagerError(
                f"Failed to create proxy manager from environment: {e}"
            ) from e

    def _build_proxy_url(self, session_id: str) -> str:
        """
        Build BrightData proxy URL with session-based rotation.

        Args:
            session_id: Unique session ID for IP rotation

        Returns:
            Complete proxy URL with authentication
        """
        # Build username with DNS resolution mode
        # Check if customer_id already includes 'brd-customer-' prefix
        if self.customer_id.startswith("brd-customer-"):
            username = (
                f"{self.customer_id}-session-{session_id}-dns-{self.dns_resolution}"
            )
        else:
            username = f"brd-customer-{self.customer_id}-zone-{self.zone_name}-session-{session_id}-dns-{self.dns_resolution}"
        return f"http://{username}:{self.password}@brd.superproxy.io:33335"

    def _generate_session_id(self) -> str:
        """
        Generate unique session ID for IP rotation.

        Returns:
            Unique session ID string
        """
        self._session_counter += 1
        timestamp = int(time.time())
        return f"session_{self._session_counter}_{timestamp}"

    async def initialize_pool(self, pool_size: int = 10) -> None:
        """
        Initialize proxy connection pool with health monitoring.

        Args:
            pool_size: Number of proxy connections to create (10-50)

        Raises:
            ProxyManagerError: If pool initialization fails
        """
        if not (10 <= pool_size <= 50):
            raise ProxyManagerError(
                f"Pool size must be between 10 and 50, got {pool_size}"
            )

        logger.info(f"Initializing proxy pool with {pool_size} connections")

        try:
            # Clear existing pool
            await self._cleanup_existing_pool()

            # Create new connections
            for i in range(pool_size):
                session_id = self._generate_session_id()
                proxy_url = self._build_proxy_url(session_id)

                # Create HTTP client with proxy configuration
                # Configure SSL verification
                # BrightData uses self-signed certificates for HTTPS interception
                # Disable SSL verification if no custom certificate is available
                verify_ssl: Any = False
                if self.ssl_cert_path and self.ssl_cert_path.exists():
                    verify_ssl = str(self.ssl_cert_path)

                client = httpx.AsyncClient(
                    proxy=proxy_url,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(
                        max_keepalive_connections=5, max_connections=10
                    ),
                    verify=verify_ssl,
                )

                # Create proxy connection object
                connection = ProxyConnection(
                    proxy_id=f"brightdata_{i}_{int(time.time())}",
                    proxy_url=proxy_url,
                    session=client,
                    status=ProxyStatus.HEALTHY,
                    circuit_breaker=CircuitBreaker(),
                )

                self.proxy_pool.append(connection)

                # Initialize metrics
                self.proxy_metrics[connection.proxy_id] = ProxyMetrics(
                    proxy_id=connection.proxy_id
                )

                logger.debug(f"Created proxy connection: {connection.proxy_id}")

            logger.info(
                f"Successfully initialized proxy pool with {len(self.proxy_pool)} connections"
            )

        except Exception as e:
            logger.error(f"Failed to initialize proxy pool: {e}")
            raise ProxyManagerError(f"Proxy pool initialization failed: {e}") from e

    async def _cleanup_existing_pool(self) -> None:
        """Clean up existing proxy connections."""
        if self.proxy_pool:
            logger.debug(f"Cleaning up {len(self.proxy_pool)} existing connections")
            for connection in self.proxy_pool:
                try:
                    await connection.session.aclose()
                except Exception as e:
                    logger.warning(
                        f"Error closing connection {connection.proxy_id}: {e}"
                    )

            self.proxy_pool.clear()
            self.proxy_metrics.clear()

    async def get_connection(self, priority: int = 0) -> Optional[ProxyConnection]:
        """
        Get next healthy proxy connection with round-robin rotation.

        Args:
            priority: Connection priority (higher = more preferred)

        Returns:
            Available proxy connection or None if no healthy connections
        """
        async with self._lock:
            if not self.proxy_pool:
                logger.error("No proxy connections in pool")
                return None

            attempts = len(self.proxy_pool)

            while attempts > 0:
                connection = self.proxy_pool[self._current_index]
                self._current_index = (self._current_index + 1) % len(self.proxy_pool)

                # Check if connection is usable
                if (
                    connection.status != ProxyStatus.FAILED
                    and connection.circuit_breaker
                    and connection.circuit_breaker.can_execute()
                ):
                    connection.last_used = datetime.now()

                    logger.debug(f"Selected proxy connection: {connection.proxy_id}")
                    return connection

                attempts -= 1

            logger.warning("No healthy proxy connections available")
            return None

    async def release_connection(self, connection: ProxyConnection) -> None:
        """
        Release a proxy connection back to the pool.

        Args:
            connection: The proxy connection to release
        """
        # Connection is always available in the pool, no action needed
        # This method is provided for API compatibility
        logger.debug(f"Released connection: {connection.proxy_id}")

    async def report_success(
        self, connection: ProxyConnection, response_time: float
    ) -> None:
        """
        Report successful request through proxy connection.

        Args:
            connection: The proxy connection used
            response_time: Response time in seconds
        """
        try:
            # Update connection metrics
            connection.success_count += 1
            if connection.circuit_breaker:
                connection.circuit_breaker.record_success()

            # Update proxy metrics
            metrics = self.proxy_metrics.get(connection.proxy_id)
            if metrics:
                metrics.successful_requests += 1
                metrics.total_requests += 1
                metrics.last_used = datetime.now()

                # Update average response time (exponential moving average)
                if metrics.average_response_time == 0:
                    metrics.average_response_time = response_time
                else:
                    metrics.average_response_time = (
                        metrics.average_response_time * 0.8 + response_time * 0.2
                    )

            # Update health score based on success
            self._update_health_score_on_success(connection, response_time)

            logger.debug(
                f"Recorded success for {connection.proxy_id}: {response_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error reporting success for {connection.proxy_id}: {e}")

    async def report_failure(
        self, connection: ProxyConnection, error: Exception
    ) -> None:
        """
        Report failed request through proxy connection.

        Args:
            connection: The proxy connection used
            error: The error that occurred
        """
        try:
            # Update connection metrics
            connection.failure_count += 1
            if connection.circuit_breaker:
                connection.circuit_breaker.record_failure()

            # Update proxy metrics
            metrics = self.proxy_metrics.get(connection.proxy_id)
            if metrics:
                metrics.failed_requests += 1
                metrics.total_requests += 1
                metrics.last_used = datetime.now()

            # Update health score and status based on failure
            self._update_health_score_on_failure(connection, error)

            logger.warning(
                f"Recorded failure for {connection.proxy_id}: {type(error).__name__}: {error}"
            )

            # Check if proxy should be replaced
            if connection.health_score < 0.3 or connection.failure_count >= 5:
                logger.info(
                    f"Proxy {connection.proxy_id} marked for replacement due to poor health"
                )
                await self._replace_proxy(connection)

        except Exception as e:
            logger.error(f"Error reporting failure for {connection.proxy_id}: {e}")

    def _update_health_score_on_success(
        self, connection: ProxyConnection, response_time: float
    ) -> None:
        """Update health score based on successful request."""
        # Latency component (40%): better score for faster responses
        latency_score = max(0.0, min(1.0, 1 - (response_time / 10.0)))

        # Success rate component (40%)
        total_requests = connection.success_count + connection.failure_count
        if total_requests > 0:
            success_rate = connection.success_count / total_requests
        else:
            success_rate = 1.0

        # Uptime component (20%): based on status
        uptime_score = 1.0 if connection.status == ProxyStatus.HEALTHY else 0.5

        # Calculate weighted health score
        new_score = latency_score * 0.4 + success_rate * 0.4 + uptime_score * 0.2

        # Use exponential moving average to smooth updates
        connection.health_score = connection.health_score * 0.7 + new_score * 0.3

        # Update status based on health score
        if connection.health_score > 0.7:
            connection.status = ProxyStatus.HEALTHY
        elif connection.health_score > 0.3:
            connection.status = ProxyStatus.DEGRADED

    def _update_health_score_on_failure(
        self, connection: ProxyConnection, error: Exception
    ) -> None:
        """Update health score based on failed request."""
        # Reduce health score based on failure severity
        if isinstance(error, (httpx.TimeoutException, httpx.ConnectTimeout)):
            penalty = 0.3  # High penalty for timeouts
        elif isinstance(error, (httpx.HTTPStatusError,)):
            penalty = 0.2  # Medium penalty for HTTP errors
        else:
            penalty = 0.25  # Standard penalty for other errors

        connection.health_score = max(0.0, connection.health_score - penalty)

        # Update status based on health score
        if connection.health_score < 0.3:
            connection.status = ProxyStatus.FAILED
        elif connection.health_score < 0.7:
            connection.status = ProxyStatus.DEGRADED

    async def start_health_monitoring(self, interval_seconds: int = 300) -> None:
        """
        Start background health monitoring for all proxies.

        Args:
            interval_seconds: Health check interval in seconds (default: 300 = 5 minutes)
        """
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return

        self.monitoring_active = True
        logger.info(f"Starting health monitoring with {interval_seconds}s interval")

        async def health_check_worker() -> None:
            """Background worker for health checks."""
            while self.monitoring_active:
                try:
                    await self._run_health_checks()
                except Exception as e:
                    logger.error(f"Error in health check worker: {e}")

                await asyncio.sleep(interval_seconds)

            logger.info("Health monitoring stopped")

        self.health_check_task = asyncio.create_task(health_check_worker())

    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            logger.info("Stopped health monitoring")

    async def test_connectivity(self) -> bool:
        """
        Test basic proxy connectivity without full initialization.

        Returns:
            True if basic connectivity can be established
        """
        try:
            # Test proxy connectivity with DNS resolution mode
            # Check if customer_id already includes 'brd-customer-' prefix
            if self.customer_id.startswith("brd-customer-"):
                base_username = f"{self.customer_id}-dns-{self.dns_resolution}"
            else:
                base_username = f"brd-customer-{self.customer_id}-zone-{self.zone_name}-dns-{self.dns_resolution}"
            proxy_url = (
                f"http://{base_username}:{self.password}@brd.superproxy.io:33335"
            )

            # Configure SSL verification if certificate is available
            # BrightData uses self-signed certificates for HTTPS interception
            # Disable SSL verification if no custom certificate is available
            verify_ssl: Any = False
            if self.ssl_cert_path and self.ssl_cert_path.exists():
                verify_ssl = str(self.ssl_cert_path)
                logger.debug(f"Using SSL certificate: {self.ssl_cert_path}")

            logger.debug(f"Testing proxy connectivity with: {proxy_url.split('@')[1]}")

            async with httpx.AsyncClient(
                proxy=proxy_url,
                timeout=httpx.Timeout(10.0, connect=5.0),
                verify=verify_ssl,
            ) as client:
                # Simple connectivity test
                response = await client.get("http://httpbin.org/ip")
                is_connected = response.status_code == 200

                if is_connected:
                    logger.info(
                        f"Proxy connectivity successful with: {proxy_url.split('@')[1]}"
                    )
                    return True
                else:
                    logger.warning(
                        f"Proxy connectivity failed: HTTP {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Proxy connectivity test failed: {e}")
            return False

    async def _run_health_checks(self) -> None:
        """Run health checks on all proxies."""
        if not self.proxy_pool:
            logger.debug("No proxies to health check")
            return

        logger.debug(f"Running health checks on {len(self.proxy_pool)} proxies")

        # Create health check tasks for non-failed proxies
        tasks = []
        for connection in self.proxy_pool:
            if connection.status != ProxyStatus.FAILED:
                task = asyncio.create_task(
                    self._check_proxy_health(connection),
                    name=f"health_check_{connection.proxy_id}",
                )
                tasks.append(task)

        if tasks:
            # Wait for all health checks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Health check task {i} failed: {result}")

    async def _check_proxy_health(self, connection: ProxyConnection) -> None:
        """
        Check individual proxy health and update score.

        Args:
            connection: The proxy connection to check
        """
        try:
            start_time = time.time()

            # Use a simple HTTP endpoint for health checking
            response = await connection.session.get(
                "http://httpbin.org/ip", timeout=10.0
            )

            response_time = time.time() - start_time

            if response.status_code == 200:
                # Health check successful
                logger.debug(
                    f"Health check passed for {connection.proxy_id}: {response_time:.2f}s"
                )

                # Update health metrics as if it was a successful request
                self._update_health_score_on_success(connection, response_time)

                # Reset some failure count on successful health check
                if connection.failure_count > 0:
                    connection.failure_count = max(0, connection.failure_count - 1)
            else:
                # Health check failed due to bad status code
                error = httpx.HTTPStatusError(
                    f"Health check failed with status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                await self._handle_health_check_failure(connection, error)

        except Exception as error:
            # Health check failed due to exception
            await self._handle_health_check_failure(connection, error)

    async def _handle_health_check_failure(
        self, connection: ProxyConnection, error: Exception
    ) -> None:
        """
        Handle proxy health check failure.

        Args:
            connection: The proxy connection that failed
            error: The error that occurred
        """
        logger.warning(
            f"Health check failed for {connection.proxy_id}: {type(error).__name__}: {error}"
        )

        # Update health score and failure tracking
        self._update_health_score_on_failure(connection, error)

        # Check if proxy should be replaced
        if connection.health_score < 0.3 or connection.failure_count >= 3:
            logger.info(
                f"Replacing proxy {connection.proxy_id} due to failed health check"
            )
            await self._replace_proxy(connection)

    async def _replace_proxy(self, old_connection: ProxyConnection) -> None:
        """
        Replace a failed proxy with a new one.

        Args:
            old_connection: The proxy connection to replace
        """
        try:
            async with self._lock:
                # Find the connection in the pool
                try:
                    index = self.proxy_pool.index(old_connection)
                except ValueError:
                    logger.error(
                        f"Connection {old_connection.proxy_id} not found in pool"
                    )
                    return

                # Close the old connection
                try:
                    await old_connection.session.aclose()
                except Exception as e:
                    logger.warning(
                        f"Error closing old connection {old_connection.proxy_id}: {e}"
                    )

                # Create new connection
                session_id = self._generate_session_id()
                proxy_url = self._build_proxy_url(session_id)

                # Configure SSL verification
                # BrightData uses self-signed certificates for HTTPS interception
                # Disable SSL verification if no custom certificate is available
                verify_ssl: Any = False
                if self.ssl_cert_path and self.ssl_cert_path.exists():
                    verify_ssl = str(self.ssl_cert_path)

                client = httpx.AsyncClient(
                    proxy=proxy_url,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(
                        max_keepalive_connections=5, max_connections=10
                    ),
                    verify=verify_ssl,
                )

                new_connection = ProxyConnection(
                    proxy_id=f"brightdata_{index}_{int(time.time())}",
                    proxy_url=proxy_url,
                    session=client,
                    status=ProxyStatus.HEALTHY,
                    circuit_breaker=CircuitBreaker(),
                )

                # Replace in pool
                self.proxy_pool[index] = new_connection

                # Update metrics
                del self.proxy_metrics[old_connection.proxy_id]
                self.proxy_metrics[new_connection.proxy_id] = ProxyMetrics(
                    proxy_id=new_connection.proxy_id
                )

                logger.info(
                    f"Replaced proxy {old_connection.proxy_id} with {new_connection.proxy_id}"
                )

        except Exception as e:
            logger.error(f"Error replacing proxy {old_connection.proxy_id}: {e}")

    async def shutdown(self) -> None:
        """Shutdown proxy manager and clean up resources."""
        logger.info("Shutting down BrightData proxy manager")

        # Stop health monitoring
        await self.stop_health_monitoring()

        # Close all connections
        await self._cleanup_existing_pool()

        logger.info("BrightData proxy manager shutdown complete")

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get current status of the proxy pool.

        Returns:
            Dictionary with pool status information
        """
        if not self.proxy_pool:
            return {
                "total_connections": 0,
                "healthy": 0,
                "degraded": 0,
                "failed": 0,
                "circuit_open": 0,
            }

        status_counts = {}
        for status in ProxyStatus:
            status_counts[status.value] = 0

        for connection in self.proxy_pool:
            status_counts[connection.status.value] += 1

        return {
            "total_connections": len(self.proxy_pool),
            "healthy": status_counts[ProxyStatus.HEALTHY.value],
            "degraded": status_counts[ProxyStatus.DEGRADED.value],
            "failed": status_counts[ProxyStatus.FAILED.value],
            "circuit_open": status_counts[ProxyStatus.CIRCUIT_OPEN.value],
            "monitoring_active": self.monitoring_active,
        }

    def get_proxy_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for all proxies.

        Returns:
            Dictionary with proxy performance metrics
        """
        return {
            proxy_id: {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.success_rate,
                "failure_rate": metrics.failure_rate,
                "average_response_time": metrics.average_response_time,
                "last_used": (
                    metrics.last_used.isoformat() if metrics.last_used else None
                ),
                "uptime_percentage": metrics.uptime_percentage,
            }
            for proxy_id, metrics in self.proxy_metrics.items()
        }

    async def __aenter__(self) -> "BrightDataProxyManager":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        await self.shutdown()
