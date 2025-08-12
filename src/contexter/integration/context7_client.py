"""
Context7 API client implementation with proxy integration and rate limiting.
"""

import asyncio
import hashlib
import json
import logging
import random
import socket
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx

if TYPE_CHECKING:
    from .proxy_manager import BrightDataProxyManager

from ..models.context7_models import (
    AuthenticationError,
    CachedSearchResult,
    ConfigurationError,
    Context7APIError,
    DocumentationResponse,
    ErrorCategory,
    InvalidResponseError,
    LibraryNotFoundError,
    LibrarySearchResult,
    NetworkError,
    RateLimitError,
    RequestMetrics,
)

logger = logging.getLogger(__name__)


class RateLimitHandler:
    """Handles rate limiting detection and response strategies."""

    def detect_rate_limit(self, response: httpx.Response) -> Tuple[bool, Optional[int]]:
        """
        Detect rate limiting and extract retry-after value.

        Args:
            response: HTTP response to check

        Returns:
            Tuple of (is_rate_limited, retry_after_seconds)
        """
        is_rate_limited = response.status_code == 429
        retry_after = None

        if is_rate_limited:
            # Check retry-after header
            retry_header = response.headers.get("retry-after")
            if retry_header:
                try:
                    retry_after = int(retry_header)
                except ValueError:
                    # Handle HTTP-date format if needed (simplified)
                    retry_after = 60  # Default fallback
            else:
                # Default retry time if no header provided
                retry_after = 60

        return is_rate_limited, retry_after

    async def handle_rate_limit(
        self, retry_after: int, proxy_id: Optional[str] = None
    ) -> None:
        """
        Handle rate limit with appropriate delay and logging.

        Args:
            retry_after: Seconds to wait before retry
            proxy_id: ID of proxy that hit rate limit
        """
        # Add jitter to prevent thundering herd
        jitter = random.uniform(1, 10)
        total_delay = retry_after + jitter

        logger.warning(
            f"Rate limited (proxy: {proxy_id or 'unknown'}), "
            f"waiting {total_delay:.1f}s (retry-after: {retry_after}s)"
        )

        await asyncio.sleep(total_delay)


class APIErrorClassifier:
    """Classify API errors for appropriate handling strategies."""

    def classify_error(
        self, error: Exception, response: Optional[httpx.Response] = None
    ) -> ErrorCategory:
        """
        Classify error into appropriate category.

        Args:
            error: Exception that occurred
            response: HTTP response if available

        Returns:
            ErrorCategory for the error
        """
        # Check explicit error types first
        if isinstance(error, RateLimitError):
            return ErrorCategory.RATE_LIMITED
        elif isinstance(error, LibraryNotFoundError):
            return ErrorCategory.NOT_FOUND
        elif isinstance(error, AuthenticationError):
            return ErrorCategory.AUTHENTICATION
        elif isinstance(error, NetworkError):
            return ErrorCategory.NETWORK

        # Check response status codes
        if response:
            if response.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif response.status_code == 429:
                return ErrorCategory.RATE_LIMITED
            elif response.status_code == 404:
                return ErrorCategory.NOT_FOUND
            elif 400 <= response.status_code < 500:
                return ErrorCategory.CLIENT_ERROR
            elif 500 <= response.status_code < 600:
                return ErrorCategory.API_ERROR

        # Check network-related errors
        if isinstance(error, (httpx.ConnectError, httpx.TimeoutException)):
            return ErrorCategory.NETWORK

        return ErrorCategory.CLIENT_ERROR

    def should_retry(self, error_category: ErrorCategory) -> Tuple[bool, float]:
        """
        Determine if error should be retried and suggested delay.

        Args:
            error_category: Category of error

        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        retry_strategies = {
            ErrorCategory.RATE_LIMITED: (True, 60.0),
            ErrorCategory.NOT_FOUND: (False, 0.0),
            ErrorCategory.AUTHENTICATION: (False, 0.0),
            ErrorCategory.NETWORK: (True, 5.0),
            ErrorCategory.API_ERROR: (True, 15.0),
            ErrorCategory.CLIENT_ERROR: (False, 0.0),
        }

        return retry_strategies.get(error_category, (False, 0.0))


class Context7Client:
    """
    Context7 API client with proxy integration and rate limiting.

    Provides methods for library search and smart documentation retrieval
    with comprehensive error handling and performance optimization.
    """

    def __init__(
        self,
        base_url: str = "https://context7.com/api/v1",
        default_timeout: float = 30.0,
        cache_ttl: float = 300.0,  # 5 minutes
        max_cache_size: int = 100,
    ):
        """
        Initialize Context7 API client.

        Args:
            base_url: Context7 API base URL
            default_timeout: Default request timeout in seconds
            cache_ttl: Cache TTL for search results in seconds
            max_cache_size: Maximum number of cached search results
        """
        self.base_url = base_url.rstrip("/")
        
        # Validate that base_url matches Context7 expected format
        if not base_url.startswith(("https://context7.com/api", "http://localhost")):
            logger.warning(f"Base URL may not be compatible with Context7 API: {base_url}")
        self.default_timeout = httpx.Timeout(default_timeout, connect=10.0)
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size

        # HTTP headers for all requests (Context7 compatible)
        self.session_headers = {
            "User-Agent": "C7DocDownloader/1.0.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-Context7-Source": "mcp-server",  # Required by Context7 API
        }

        # Component initialization
        self.rate_limit_handler = RateLimitHandler()
        self.error_classifier = APIErrorClassifier()

        # Search result cache
        self._search_cache: Dict[str, CachedSearchResult] = {}
        self._cache_lock = asyncio.Lock()

        # Request metrics
        self.metrics = RequestMetrics()
        self._stats_lock = asyncio.Lock()

        logger.info(f"Context7Client initialized with base URL: {self.base_url}")

    def _build_url(self, endpoint: str) -> str:
        """
        Build full API URL for endpoint with Context7 API format.

        Args:
            endpoint: API endpoint path

        Returns:
            Complete URL for the endpoint
        """
        # Context7 API uses /v1/ prefix for endpoints
        if endpoint in ["search", "smart_docs"]:
            if not self.base_url.endswith("/v1"):
                base = f"{self.base_url}/v1"
            else:
                base = self.base_url
            return urljoin(f"{base}/", endpoint.lstrip("/"))
        else:
            return urljoin(f"{self.base_url}/", endpoint.lstrip("/"))

    async def _update_request_stats(
        self,
        success: bool,
        rate_limited: bool = False,
        tokens: int = 0,
        response_time: float = 0.0,
    ) -> None:
        """
        Update internal request statistics.

        Args:
            success: Whether request was successful
            rate_limited: Whether request was rate limited
            tokens: Number of tokens retrieved
            response_time: Response time in seconds
        """
        async with self._stats_lock:
            self.metrics.request_count += 1

            if success:
                self.metrics.success_count += 1
                self.metrics.total_tokens_retrieved += tokens
                self.metrics.total_response_time += response_time
            elif rate_limited:
                self.metrics.rate_limit_count += 1
            else:
                self.metrics.error_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive API usage statistics.

        Returns:
            Dictionary containing detailed usage metrics
        """
        return {
            "total_requests": self.metrics.request_count,
            "successful_requests": self.metrics.success_count,
            "rate_limited_requests": self.metrics.rate_limit_count,
            "error_requests": self.metrics.error_count,
            "success_rate": self.metrics.success_rate,
            "rate_limit_rate": self.metrics.rate_limit_rate,
            "average_response_time": self.metrics.average_response_time,
            "average_tokens_per_request": self.metrics.average_tokens_per_request,
            "total_tokens_retrieved": self.metrics.total_tokens_retrieved,
            "cache_size": len(self._search_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (simplified implementation)."""
        # This would be more sophisticated with proper hit/miss tracking
        return 0.0  # Placeholder for now

    async def health_check(self, client: Optional[httpx.AsyncClient] = None) -> bool:
        """
        Check Context7 API health and accessibility.

        Args:
            client: Optional HTTP client to use (for proxy testing)

        Returns:
            True if API is accessible and healthy
        """
        logger.debug("Performing Context7 API health check")

        use_internal_client = client is None

        try:
            if use_internal_client:
                client = httpx.AsyncClient(
                    timeout=httpx.Timeout(10.0, connect=5.0),
                    headers=self.session_headers,
                )

            # Use a simple library search as health check
            if client is not None:
                response = await client.get(
                    self._build_url("search"), params={"query": "test"}, timeout=5.0
                )
            else:
                return False

            # Consider 200, 404, and 429 as healthy 
            # 200 = success, 404 = no results, 429 = rate limited but API is responding
            is_healthy = response.status_code in [200, 404, 429]

            logger.debug(
                f"Health check result: {'healthy' if is_healthy else 'unhealthy'} "
                f"(status: {response.status_code})"
            )

            return is_healthy

        except Exception as e:
            logger.error(f"Context7 health check failed: {e}")
            return False

        finally:
            if use_internal_client and client:
                await client.aclose()

    async def test_connectivity(
        self, client: Optional[httpx.AsyncClient] = None
    ) -> bool:
        """
        Test connectivity to Context7 API with fallback mechanisms.

        Args:
            client: Optional HTTP client to use (for proxy testing)

        Returns:
            True if API is accessible and healthy
        """
        # First test DNS resolution
        hostname = (
            self.base_url.replace("https://", "").replace("http://", "").split("/")[0]
        )
        dns_ok = await self._test_dns_resolution(hostname)
        if not dns_ok:
            logger.warning("DNS resolution failed for Context7 API")
            return False

        # First try direct connection (without proxy)
        try:
            logger.debug("Testing Context7 API with direct connection")
            direct_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0), headers=self.session_headers
            )

            result = await self.health_check(direct_client)
            await direct_client.aclose()

            if result:
                logger.info("Context7 API accessible via direct connection")
                return True
            else:
                logger.debug("Direct connection failed, trying with proxy if available")

        except Exception as e:
            logger.debug(f"Direct connection test failed: {e}")

        # If direct connection fails and proxy client is provided, try with proxy
        if client:
            try:
                logger.debug("Testing Context7 API through proxy")
                return await self.health_check(client)
            except Exception as e:
                logger.debug(f"Proxy connection test failed: {e}")

        return False

    async def _test_dns_resolution(self, hostname: str = "context7.com") -> bool:
        """
        Test DNS resolution for Context7 API using multiple DNS servers.

        Args:
            hostname: The hostname to resolve

        Returns:
            True if DNS resolution succeeds
        """
        dns_servers = [
            "8.8.8.8",  # Google DNS
            "1.1.1.1",  # Cloudflare DNS
            "208.67.222.222",  # OpenDNS
            "9.9.9.9",  # Quad9 DNS
        ]

        for dns_server in dns_servers:
            try:
                logger.debug(
                    f"Testing DNS resolution for {hostname} using {dns_server}"
                )

                # Create a custom resolver
                resolver = socket.getaddrinfo(
                    hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM
                )
                if resolver:
                    logger.debug(f"DNS resolution successful using {dns_server}")
                    return True

            except Exception as e:
                logger.debug(f"DNS resolution failed with {dns_server}: {e}")
                continue

        logger.warning(f"All DNS servers failed to resolve {hostname}")
        return False

    def _generate_cache_key(self, query: str) -> str:
        """
        Generate cache key for search query.

        Args:
            query: Search query string

        Returns:
            MD5 hash of normalized query
        """
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _validate_library_id(self, library_id: str) -> str:
        """
        Validate and normalize library ID format for Context7 API.
        
        Context7 expects library IDs in format: username/library[/tag]
        Examples: facebook/react, vercel/next.js, mongodb/docs
        
        Args:
            library_id: Library identifier to validate
            
        Returns:
            Normalized library ID
            
        Raises:
            ConfigurationError: If library ID format is invalid
        """
        if not library_id or not library_id.strip():
            raise ConfigurationError("Library ID cannot be empty")
        
        library_id = library_id.strip()
        
        # Remove leading slash if present (Context7 handles this)
        if library_id.startswith("/"):
            library_id = library_id[1:]
        
        # Validate format: should contain at least one slash for username/library
        if "/" not in library_id:
            raise ConfigurationError(
                f"Invalid library format. Expected: username/library[/tag], got: {library_id}. "
                f"Examples: facebook/react, vercel/next.js, mongodb/docs"
            )
        
        parts = library_id.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ConfigurationError(
                f"Invalid library format. Username and library name required: {library_id}"
            )
        
        return library_id

    async def _get_cached_search(
        self, cache_key: str
    ) -> Optional[List[LibrarySearchResult]]:
        """
        Get cached search result if still valid.

        Args:
            cache_key: Cache key for the search

        Returns:
            Cached results if valid, None otherwise
        """
        async with self._cache_lock:
            cached = self._search_cache.get(cache_key)

            if cached and not cached.is_expired:
                logger.debug(f"Cache hit for query (age: {cached.age_seconds:.1f}s)")
                return cached.results
            elif cached:
                # Remove expired entry
                del self._search_cache[cache_key]
                logger.debug("Removed expired cache entry")

        return None

    async def _cache_search_result(
        self, cache_key: str, results: List[LibrarySearchResult], query: str
    ) -> None:
        """
        Cache search results with TTL management.

        Args:
            cache_key: Cache key for the search
            results: Search results to cache
            query: Original search query
        """
        async with self._cache_lock:
            # Cache cleanup if needed
            if len(self._search_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(
                    self._search_cache.keys(),
                    key=lambda k: self._search_cache[k].timestamp,
                )
                del self._search_cache[oldest_key]
                logger.debug("Removed oldest cache entry due to size limit")

            # Store new cached result
            self._search_cache[cache_key] = CachedSearchResult(
                results=results, timestamp=time.time(), query=query, ttl=self.cache_ttl
            )

            logger.debug(
                f"Cached {len(results)} search results for query: '{query[:30]}...'"
            )

    async def clear_cache(self) -> None:
        """Clear all cached search results."""
        async with self._cache_lock:
            cache_size = len(self._search_cache)
            self._search_cache.clear()
            logger.info(f"Cleared {cache_size} cached search results")

    def _extract_proxy_id(self, client: Optional[httpx.AsyncClient]) -> Optional[str]:
        """
        Extract proxy ID from client if available.

        Args:
            client: HTTP client that may have proxy information

        Returns:
            Proxy ID if available, None otherwise
        """
        if client:
            # Try to get proxy ID from client attributes
            return getattr(client, "proxy_id", None)
        return None

    async def resolve_library_id(
        self, query: str, client: Optional[httpx.AsyncClient] = None, limit: int = 20
    ) -> List[LibrarySearchResult]:
        """
        Search for libraries using natural language queries.

        Args:
            query: Natural language search query
            client: Optional HTTP client (for proxy integration)
            limit: Maximum number of results to return

        Returns:
            List of matching library search results

        Raises:
            Context7APIError: If API request fails
            ConfigurationError: If query is invalid
        """
        # Validate input
        if not query or not query.strip():
            raise ConfigurationError("Search query cannot be empty")

        query = query.strip()
        cache_key = self._generate_cache_key(query)

        # Check cache first
        cached_results = await self._get_cached_search(cache_key)
        if cached_results is not None:
            logger.debug(
                f"Returning {len(cached_results)} cached results for query: '{query}'"
            )
            return cached_results

        start_time = time.time()
        use_internal_client = client is None
        proxy_id = self._extract_proxy_id(client)

        try:
            if use_internal_client:
                client = httpx.AsyncClient(
                    timeout=self.default_timeout, headers=self.session_headers
                )

            # Build request
            url = self._build_url("search")
            params: Dict[str, Any] = {
                "query": query,
                "limit": min(max(limit, 1), 50),  # Clamp between 1 and 50
            }

            logger.info(
                f"Searching for libraries with query: '{query}' (limit: {params['limit']})"
            )

            # Make API request
            if client is not None:
                response = await client.get(url, params=params)
            else:
                raise Context7APIError("HTTP client is None")
            response_time = time.time() - start_time

            # Handle response based on status code
            if response.status_code == 200:
                results = await self._parse_search_response(response, query)
                await self._update_request_stats(
                    success=True, response_time=response_time
                )

                # Cache successful results
                await self._cache_search_result(cache_key, results, query)

                logger.info(
                    f"Found {len(results)} libraries for query '{query}' "
                    f"in {response_time:.2f}s (proxy: {proxy_id or 'direct'})"
                )

                return results

            elif response.status_code == 404:
                # No results found - this is okay
                logger.info(f"No libraries found for query: '{query}'")
                await self._update_request_stats(
                    success=True, response_time=response_time
                )

                # Cache empty results too
                empty_results: List[LibrarySearchResult] = []
                await self._cache_search_result(cache_key, empty_results, query)

                return empty_results

            elif response.status_code == 429:
                # Handle rate limiting
                await self._handle_rate_limit_response(response, proxy_id)

            else:
                # Handle other errors
                await self._handle_api_error_response(response, "library search")

        except RateLimitError:
            # Re-raise rate limit errors
            raise

        except Exception as e:
            await self._update_request_stats(success=False)

            # Classify and handle error (for future use)
            # error_category = self.error_classifier.classify_error(e)

            if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
                raise NetworkError(
                    f"Network error during library search: {e}", e
                ) from e
            else:
                raise Context7APIError(
                    f"Library search failed for query '{query}': {e}"
                ) from e

        finally:
            if use_internal_client and client:
                await client.aclose()

        # Should not reach here
        return []

    async def _parse_search_response(
        self, response: httpx.Response, original_query: str
    ) -> List[LibrarySearchResult]:
        """
        Parse library search response into structured results.

        Args:
            response: HTTP response from search API
            original_query: Original search query for metadata

        Returns:
            List of parsed library search results

        Raises:
            InvalidResponseError: If response format is invalid
        """
        try:
            data = response.json()

            # Handle different response formats that Context7 might return
            results_data = []

            if isinstance(data, list):
                # Direct list of results
                results_data = data
            elif isinstance(data, dict):
                # Structured response with results in various keys
                for key in ["results", "libraries", "data", "items"]:
                    if key in data:
                        results_data = data[key]
                        break
                else:
                    # If no standard key found, treat the whole dict as single result
                    if any(k in data for k in ["library_id", "id", "name"]):
                        results_data = [data]
                    else:
                        results_data = []

            parsed_results = []

            for i, item in enumerate(results_data):
                try:
                    if not isinstance(item, dict):
                        logger.warning(
                            f"Skipping non-dict result item at index {i}: {item}"
                        )
                        continue

                    # Extract required fields with fallbacks
                    library_id = item.get("library_id") or item.get("id")
                    if not library_id:
                        logger.warning(f"Skipping result without library_id: {item}")
                        continue

                    name = item.get("name", item.get("title", "Unknown Library"))
                    description = item.get("description", item.get("summary", ""))

                    # Parse numeric fields safely
                    trust_score = 0.0
                    try:
                        trust_score = float(item.get("trust_score", 0.0))
                    except (ValueError, TypeError):
                        pass

                    star_count = -1
                    try:
                        star_value = item.get("star_count", item.get("stars", -1))
                        if star_value is not None:
                            star_count = int(star_value)
                    except (ValueError, TypeError):
                        pass

                    # Parse token and content information (from Context7 API)
                    total_tokens = -1
                    try:
                        token_value = item.get("totalTokens", item.get("total_tokens", -1))
                        if token_value is not None:
                            total_tokens = int(token_value)
                    except (ValueError, TypeError):
                        pass

                    total_snippets = -1
                    try:
                        snippets_value = item.get("totalSnippets", item.get("total_snippets", -1))
                        if snippets_value is not None:
                            total_snippets = int(snippets_value)
                    except (ValueError, TypeError):
                        pass

                    total_pages = -1
                    try:
                        pages_value = item.get("totalPages", item.get("total_pages", -1))
                        if pages_value is not None:
                            total_pages = int(pages_value)
                    except (ValueError, TypeError):
                        pass

                    # Calculate search relevance (simple heuristic)
                    search_relevance = 0.0
                    original_lower = original_query.lower()
                    if name and original_lower in name.lower():
                        search_relevance += 0.5
                    if description and original_lower in description.lower():
                        search_relevance += 0.3

                    result = LibrarySearchResult(
                        library_id=str(library_id),
                        name=str(name),
                        description=str(description),
                        trust_score=trust_score,
                        star_count=star_count,
                        search_relevance=search_relevance,
                        total_tokens=total_tokens,
                        total_snippets=total_snippets,
                        total_pages=total_pages,
                        metadata={
                            "original_query": original_query,
                            "result_index": i,
                            "api_response": item,
                        },
                    )

                    parsed_results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to parse search result item {i}: {e}")
                    continue

            # Sort by relevance and trust score
            parsed_results.sort(
                key=lambda x: (x.search_relevance, x.trust_score), reverse=True
            )

            logger.debug(f"Successfully parsed {len(parsed_results)} search results")
            return parsed_results

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise InvalidResponseError(
                f"Invalid JSON in search response: {e}", response.text[:500]
            ) from e

        except Exception as e:
            logger.error(f"Error parsing search response: {e}")
            raise InvalidResponseError(f"Failed to parse search response: {e}") from e

    async def batch_resolve_libraries(
        self, queries: List[str], client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, List[LibrarySearchResult]]:
        """
        Resolve multiple library queries in batch.

        Args:
            queries: List of search queries
            client: Optional HTTP client for proxy integration

        Returns:
            Dictionary mapping queries to their results
        """
        if not queries:
            return {}

        logger.info(f"Batch resolving {len(queries)} library queries")

        # For now, implement as concurrent individual requests
        # Could be optimized with actual batch API if Context7 supports it
        tasks = [self.resolve_library_id(query, client) for query in queries]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_results: Dict[str, List[LibrarySearchResult]] = {}
            for query, result in zip(queries, results):
                if isinstance(result, Exception):
                    logger.error(f"Batch query failed for '{query}': {result}")
                    batch_results[query] = []
                else:
                    batch_results[query] = result if isinstance(result, list) else []

            successful_count = sum(1 for r in results if not isinstance(r, Exception))
            logger.info(
                f"Batch resolve completed: {successful_count}/{len(queries)} successful"
            )

            return batch_results

        except Exception as e:
            logger.error(f"Batch library resolution failed: {e}")
            # Return partial results where possible
            return {query: [] for query in queries}

    async def get_smart_docs(
        self,
        library_id: str,
        context: str,
        tokens: int = 200_000,
        client: Optional[httpx.AsyncClient] = None,
        version: Optional[str] = None,
        extra_libraries: Optional[List[str]] = None,
    ) -> DocumentationResponse:
        """
        Fetch smart documentation for a library with contextual query.

        Args:
            library_id: Library identifier to fetch documentation for
            context: Contextual search terms for targeted documentation
            tokens: Maximum tokens to retrieve (default: 200K for full coverage)
            client: Optional HTTP client (for proxy integration)
            version: Optional library version specification
            extra_libraries: Optional additional libraries for context

        Returns:
            Documentation response with content and metadata

        Raises:
            Context7APIError: If API request fails
            LibraryNotFoundError: If library doesn't exist
            RateLimitError: If rate limited
            ConfigurationError: If parameters are invalid
        """
        # Validate inputs
        if not library_id or not library_id.strip():
            raise ConfigurationError("Library ID cannot be empty")

        # Context is optional for Context7 API (becomes 'topic' parameter)
        library_id = library_id.strip()
        if context:
            context = context.strip()

        # Ensure token count is within API limits
        tokens = min(max(tokens, 1000), 200_000)  # Clamp between 1K and 200K

        start_time = time.time()
        use_internal_client = client is None
        proxy_id = self._extract_proxy_id(client)

        try:
            if use_internal_client:
                client = httpx.AsyncClient(
                    timeout=self.default_timeout, headers=self.session_headers
                )

            # Validate and normalize library ID
            validated_library_id = self._validate_library_id(library_id)
            
            # Build request parameters for Context7 API format
            # Context7 uses the library ID as part of the URL path
            url = self._build_url(validated_library_id)
            params = {
                "tokens": tokens,
                "type": "txt",  # Context7 uses 'type' not 'format'
            }
            
            # Add topic parameter if context provided
            if context:
                params["topic"] = context

            # Add optional parameters
            if version:
                params["version"] = version.strip()

            if extra_libraries:
                # Convert list to comma-separated string or handle as API expects
                params["extra_libraries"] = ",".join(extra_libraries)

            context_info = f"with topic '{context[:50]}...'" if context else "(no specific topic)"
            logger.info(
                f"Fetching documentation for {validated_library_id} "
                f"{context_info} "
                f"({tokens} tokens, proxy: {proxy_id or 'direct'})"
            )

            # Make API request with retry logic for temporary failures
            if client is not None:
                response = await self._make_request_with_retry(
                    client, "GET", url, params=params
                )
            else:
                raise Context7APIError("HTTP client is None")

            response_time = time.time() - start_time

            # Handle successful response
            if response.status_code == 200:
                doc_response = await self._parse_documentation_response(
                    response, validated_library_id, context or "", response_time, proxy_id
                )

                await self._update_request_stats(
                    success=True,
                    tokens=doc_response.token_count,
                    response_time=response_time,
                )

                logger.info(
                    f"Successfully retrieved documentation for {validated_library_id}: "
                    f"{doc_response.token_count} tokens in {response_time:.2f}s "
                    f"({doc_response.tokens_per_second:.1f} tokens/sec)"
                )

                return doc_response

            elif response.status_code == 404:
                raise LibraryNotFoundError(
                    f"Library not found: {validated_library_id}", library_id=validated_library_id
                )

            elif response.status_code == 429:
                await self._handle_rate_limit_response(response, proxy_id)

            else:
                await self._handle_api_error_response(
                    response, "documentation retrieval"
                )

        except (RateLimitError, LibraryNotFoundError):
            # Re-raise specific errors
            raise

        except Exception as e:
            await self._update_request_stats(success=False)

            # Classify and handle error
            if isinstance(e, (httpx.ConnectError, httpx.TimeoutException)):
                raise NetworkError(
                    f"Network error during documentation retrieval for {library_id}: {e}",
                    e,
                ) from e
            else:
                raise Context7APIError(
                    f"Documentation retrieval failed for {library_id}: {e}"
                ) from e

        finally:
            if use_internal_client and client:
                await client.aclose()

        # Should not reach here due to error handling above
        raise Context7APIError("Unexpected end of get_smart_docs method")

    async def _make_request_with_retry(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic for transient failures.

        Args:
            client: HTTP client to use
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            max_retries: Maximum number of retry attempts
            **kwargs: Additional request arguments

        Returns:
            HTTP response

        Raises:
            Various HTTP and network exceptions after retries exhausted
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)

                # Don't retry client errors (4xx) except rate limiting
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return response

                # Don't retry successful responses
                if response.status_code < 400 or response.status_code == 404:
                    return response

                # For server errors (5xx) and rate limiting, consider retry
                if attempt < max_retries:
                    retry_delay = (2**attempt) + random.uniform(
                        1, 3
                    )  # Exponential backoff with jitter
                    logger.warning(
                        f"Request failed with status {response.status_code}, "
                        f"retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # Return the response even if it's an error (for final error handling)
                return response

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exception = e

                if attempt < max_retries:
                    retry_delay = (2**attempt) + random.uniform(1, 3)
                    logger.warning(
                        f"Network error: {e}, retrying in {retry_delay:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                # Re-raise the last network exception
                raise

        # Should not reach here, but raise last exception if we do
        if last_exception:
            raise last_exception
        else:
            raise Context7APIError("Unexpected end of retry logic")

    async def _parse_documentation_response(
        self,
        response: httpx.Response,
        library_id: str,
        context: str,
        response_time: float,
        proxy_id: Optional[str],
    ) -> DocumentationResponse:
        """
        Parse documentation response into structured format.

        Args:
            response: HTTP response from documentation API
            library_id: Library ID from request
            context: Context from request
            response_time: Time taken for the request
            proxy_id: Proxy ID if used

        Returns:
            Parsed documentation response

        Raises:
            InvalidResponseError: If response format is invalid
        """
        try:
            # First, try to parse as JSON
            try:
                data = response.json()

                # Handle different response formats
                content = ""
                token_count = 0
                metadata = {}

                if isinstance(data, str):
                    # Direct content response
                    content = data
                    token_count = self._estimate_token_count(content)

                elif isinstance(data, dict):
                    # Structured response - check various possible keys
                    content = (
                        data.get("content")
                        or data.get("documentation")
                        or data.get("text")
                        or data.get("body")
                        or ""
                    )

                    # Get token count from API or estimate
                    token_count = (
                        data.get("token_count")
                        or data.get("tokens")
                        or self._estimate_token_count(content)
                    )

                    # Extract additional metadata
                    metadata.update(
                        {
                            key: value
                            for key, value in data.items()
                            if key
                            not in [
                                "content",
                                "documentation",
                                "text",
                                "body",
                                "token_count",
                                "tokens",
                            ]
                        }
                    )

                else:
                    # Fallback: convert to string
                    content = str(data)
                    token_count = self._estimate_token_count(content)

            except json.JSONDecodeError:
                # Handle non-JSON responses (plain text, markdown, etc.)
                content = response.text
                token_count = self._estimate_token_count(content)
                metadata = {
                    "content_type": response.headers.get("content-type", "unknown")
                }

            if not content:
                raise InvalidResponseError("Empty content received from Context7 API")

            return DocumentationResponse(
                content=content,
                token_count=int(token_count),
                library_id=library_id,
                context=context,
                response_time=response_time,
                proxy_id=proxy_id,
                metadata={
                    **metadata,
                    "api_response_size": len(response.content),
                    "response_headers": dict(response.headers),
                    "status_code": response.status_code,
                    "estimated_tokens": token_count
                    != int(token_count),  # Flag if estimated
                },
            )

        except Exception as e:
            logger.error(f"Error parsing documentation response: {e}")
            raise InvalidResponseError(
                f"Failed to parse documentation response: {e}"
            ) from e

    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text content.

        Args:
            text: Text content to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Simple estimation: ~4 characters per token on average
        # This is a rough approximation for English text
        char_count = len(text)
        estimated_tokens = char_count // 4

        # Account for whitespace and punctuation
        word_count = len(text.split())
        token_estimate = max(word_count, estimated_tokens)

        return int(token_estimate)

    async def _handle_rate_limit_response(
        self, response: httpx.Response, proxy_id: Optional[str] = None
    ) -> None:
        """
        Handle rate limit response with proxy switching consideration.

        Args:
            response: HTTP response with 429 status code
            proxy_id: ID of proxy that hit rate limit

        Raises:
            RateLimitError: Always raises with retry information
        """
        is_rate_limited, retry_after = self.rate_limit_handler.detect_rate_limit(
            response
        )

        if is_rate_limited:
            await self._update_request_stats(success=False, rate_limited=True)

            logger.warning(
                f"Rate limited by Context7 API (proxy: {proxy_id or 'direct'}), "
                f"retry after {retry_after}s"
            )

            # Extract additional error information from response
            error_details = ""
            try:
                error_data = response.json()
                if isinstance(error_data, dict) and "message" in error_data:
                    error_details = f": {error_data['message']}"
            except Exception:
                pass

            raise RateLimitError(
                f"Context7 API rate limit exceeded{error_details}",
                retry_after=retry_after,
                proxy_id=proxy_id,
            )

    async def _handle_api_error_response(
        self, response: httpx.Response, operation: str
    ) -> None:
        """
        Handle API error responses with appropriate error classification.

        Args:
            response: HTTP error response
            operation: Operation that failed (for error messages)

        Raises:
            Appropriate Context7APIError subclass based on response
        """
        error_msg = f"Context7 {operation} failed with status {response.status_code}"

        # Try to extract error details from response
        error_details = ""
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                error_details = str(
                    error_data.get("error", error_data.get("message", ""))
                )
        except Exception:
            # Fallback to text content
            error_details = response.text[:200] if response.text else ""

        if error_details:
            error_msg += f": {error_details}"

        await self._update_request_stats(success=False)

        # Classify error and raise appropriate exception
        if response.status_code == 401:
            raise AuthenticationError(error_msg)
        elif response.status_code == 404:
            raise LibraryNotFoundError(error_msg)
        elif response.status_code == 429:
            # Should have been handled by _handle_rate_limit_response
            await self._handle_rate_limit_response(response)
        elif 400 <= response.status_code < 500:
            raise Context7APIError(
                error_msg, ErrorCategory.CLIENT_ERROR, response.status_code
            )
        elif 500 <= response.status_code < 600:
            raise Context7APIError(
                error_msg, ErrorCategory.API_ERROR, response.status_code
            )
        else:
            raise Context7APIError(
                error_msg, ErrorCategory.CLIENT_ERROR, response.status_code
            )

    async def request_with_proxy_switching(
        self,
        proxy_manager: Optional["BrightDataProxyManager"],
        library_id: str,
        context: str,
        tokens: int = 200_000,
        max_proxy_attempts: int = 3,
    ) -> DocumentationResponse:
        """
        Make documentation request with automatic proxy switching on rate limits.

        Args:
            proxy_manager: Proxy manager for connection handling
            library_id: Library to fetch documentation for
            context: Context for documentation retrieval
            tokens: Maximum tokens to retrieve
            max_proxy_attempts: Maximum proxy switching attempts

        Returns:
            Documentation response

        Raises:
            Context7APIError: If all proxy attempts fail
            LibraryNotFoundError: If library doesn't exist
        """
        if not proxy_manager:
            # No proxy manager - use direct connection
            return await self.get_smart_docs(library_id, context, tokens)

        last_exception = None

        for attempt in range(max_proxy_attempts):
            try:
                # Get proxy connection
                proxy_connection = await proxy_manager.get_connection()

                if not proxy_connection or not proxy_connection.session:
                    logger.warning(
                        f"No proxy connection available (attempt {attempt + 1})"
                    )
                    if attempt == max_proxy_attempts - 1:
                        # Last attempt - try direct connection
                        return await self.get_smart_docs(library_id, context, tokens)
                    continue

                logger.debug(
                    f"Attempting documentation retrieval with proxy {proxy_connection.proxy_id} "
                    f"(attempt {attempt + 1}/{max_proxy_attempts})"
                )

                # Make request with proxy
                result = await self.get_smart_docs(
                    library_id, context, tokens, client=proxy_connection.session
                )

                # Report success to proxy manager
                if hasattr(proxy_manager, "report_success"):
                    await proxy_manager.report_success(
                        proxy_connection, result.response_time
                    )

                logger.info(
                    f"Documentation retrieval successful with proxy {proxy_connection.proxy_id}"
                )

                return result

            except RateLimitError as e:
                last_exception = e
                logger.warning(
                    f"Rate limited with proxy {e.proxy_id}, "
                    f"switching proxy (attempt {attempt + 1}/{max_proxy_attempts})"
                )

                # Report rate limit to proxy manager
                if hasattr(proxy_manager, "report_rate_limit") and e.proxy_id:
                    await proxy_manager.report_rate_limit(e.proxy_id)

                # Wait for rate limit cooldown with jitter
                await self.rate_limit_handler.handle_rate_limit(
                    e.retry_after, e.proxy_id
                )

                # Continue to next proxy attempt
                continue

            except LibraryNotFoundError:
                # Library not found - no point in retrying with different proxy
                raise

            except Exception as e:
                last_exception = e  # type: ignore

                # Report failure to proxy manager
                if hasattr(proxy_manager, "report_failure") and proxy_connection:
                    await proxy_manager.report_failure(proxy_connection, e)

                proxy_id = proxy_connection.proxy_id if proxy_connection else "unknown"
                logger.warning(
                    f"Request failed with proxy {proxy_id}: {e} "
                    f"(attempt {attempt + 1}/{max_proxy_attempts})"
                )

                # For network errors, try next proxy
                if isinstance(e, NetworkError) and attempt < max_proxy_attempts - 1:
                    continue

                # For other errors, re-raise immediately
                raise

        # All proxy attempts exhausted
        if isinstance(last_exception, RateLimitError):
            raise Context7APIError(
                f"All {max_proxy_attempts} proxy attempts rate limited for {library_id}"
            ) from last_exception
        else:
            raise Context7APIError(
                f"Documentation retrieval failed after {max_proxy_attempts} proxy attempts"
            ) from last_exception

    async def get_connection_health(
        self, client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive health information for Context7 connection.

        Args:
            client: Optional HTTP client to test (for proxy health check)

        Returns:
            Dictionary with health status and performance metrics
        """
        start_time = time.time()

        health_info = {
            "timestamp": start_time,
            "api_reachable": False,
            "response_time": 0.0,
            "proxy_id": self._extract_proxy_id(client),
            "error": None,
        }

        try:
            # Perform health check
            is_healthy = await self.health_check(client)
            response_time = time.time() - start_time

            health_info.update(
                {
                    "api_reachable": is_healthy,
                    "response_time": response_time,
                    "status": "healthy" if is_healthy else "unhealthy",
                }
            )

            # Add performance assessment
            if response_time > 10.0:
                health_info["performance"] = "slow"
            elif response_time > 5.0:
                health_info["performance"] = "moderate"
            else:
                health_info["performance"] = "fast"

        except Exception as e:
            health_info.update(
                {
                    "api_reachable": False,
                    "response_time": time.time() - start_time,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )

        return health_info
