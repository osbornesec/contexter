# FastAPI Integration Technical Specifications

## Overview

This document provides comprehensive technical specifications for implementing FastAPI-based API endpoints for the Contexter RAG system. It covers async endpoint patterns, dependency injection, authentication, error handling, middleware, and integration with the existing Contexter architecture.

**Key Requirements**:
- <100ms API response time for search queries
- JWT and API key authentication
- Rate limiting with Redis backend
- OpenAPI 3.0 specification with documentation
- Comprehensive error handling and validation
- Integration with Contexter monitoring patterns

## FastAPI Application Architecture

### Application Structure and Configuration

```python
# Core FastAPI application setup integrated with Contexter patterns
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import asyncio
from typing import Optional, List, Dict, Any
from contexter.core.config_manager import ConfigManager
from contexter.core.error_classifier import ErrorClassifier

class ContexterFastAPIApp:
    """Production-ready FastAPI application for Contexter RAG system."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.app = self._create_app()
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        
        app = FastAPI(
            title="Contexter RAG API",
            description="High-performance RAG system for documentation search and retrieval",
            version="1.0.0",
            docs_url="/docs" if self.config.enable_docs else None,
            redoc_url="/redoc" if self.config.enable_docs else None,
            openapi_url="/openapi.json" if self.config.enable_docs else None,
            
            # Performance optimizations
            generate_unique_id_function=self._generate_unique_operation_id,
            separate_input_output_schemas=False  # Reduce OpenAPI spec size
        )
        
        return app
    
    def _setup_middleware(self):
        """Configure middleware for security, CORS, and monitoring."""
        
        # CORS middleware for cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Response-Time"]
        )
        
        # Trusted host middleware for security
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.trusted_hosts
        )
        
        # Custom middleware for request tracking and performance monitoring
        self.app.add_middleware(ContexterTrackingMiddleware)
        self.app.add_middleware(PerformanceMonitoringMiddleware)
        self.app.add_middleware(RateLimitingMiddleware)
```

### Async Endpoint Patterns

```python
class RAGEndpoints:
    """Async endpoint implementations for RAG operations."""
    
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embedding_service: VoyageEmbeddingClient,
        search_engine: ContexterSearchEngine,
        auth_service: AuthenticationService
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.search_engine = search_engine
        self.auth_service = auth_service
    
    async def semantic_search(
        self,
        query: SearchRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
        background_tasks: BackgroundTasks = BackgroundTasks()
    ) -> SearchResponse:
        """Perform semantic search with authentication and rate limiting."""
        
        # Validate request
        await self._validate_search_request(query)
        
        # Track usage for rate limiting
        await self._track_user_usage(current_user.user_id, "search")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(
                query.query_text
            )
            
            # Perform vector similarity search
            search_results = await self.search_engine.hybrid_search(
                query_vector=query_embedding,
                text_query=query.query_text,
                filters=query.filters,
                top_k=query.limit,
                precision=query.precision
            )
            
            # Format response
            response = SearchResponse(
                query=query.query_text,
                results=[
                    SearchResult(
                        content=result.payload["content"],
                        score=result.score,
                        library_id=result.payload["library_id"],
                        section=result.payload["section"],
                        doc_type=result.payload["doc_type"],
                        metadata=result.payload
                    )
                    for result in search_results
                ],
                total_results=len(search_results),
                query_time=time.time() - start_time
            )
            
            # Log search analytics in background
            background_tasks.add_task(
                self._log_search_analytics,
                current_user.user_id,
                query,
                response
            )
            
            return response
            
        except Exception as e:
            # Comprehensive error handling
            error_info = ErrorClassifier.classify_error(e)
            
            # Log error with context
            logger.error(
                f"Search failed for user {current_user.user_id}",
                extra={
                    "error_category": error_info.category,
                    "query": query.query_text,
                    "user_id": current_user.user_id,
                    "elapsed_time": time.time() - start_time
                }
            )
            
            # Return appropriate HTTP error
            raise HTTPException(
                status_code=error_info.http_status_code,
                detail=error_info.user_message
            )
    
    async def library_search(
        self,
        library_id: str,
        query: str,
        limit: int = 10,
        current_user: AuthenticatedUser = Depends(get_current_user)
    ) -> LibrarySearchResponse:
        """Search within a specific library context."""
        
        # Library-specific search with enhanced filtering
        filters = {
            "library_id": library_id,
            "min_trust_score": 7.0  # Higher quality threshold
        }
        
        search_request = SearchRequest(
            query_text=query,
            filters=filters,
            limit=limit,
            precision="balanced"
        )
        
        return await self.semantic_search(search_request, current_user)
    
    async def multi_library_search(
        self,
        libraries: List[str],
        query: str,
        limit: int = 20,
        current_user: AuthenticatedUser = Depends(get_current_user)
    ) -> MultiLibrarySearchResponse:
        """Search across multiple libraries with result aggregation."""
        
        # Create concurrent search tasks for each library
        search_tasks = []
        for library_id in libraries:
            task = self.library_search(library_id, query, limit // len(libraries), current_user)
            search_tasks.append(task)
        
        # Execute searches concurrently
        library_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Aggregate and rank results
        aggregated_results = []
        for i, result in enumerate(library_results):
            if isinstance(result, Exception):
                logger.warning(f"Search failed for library {libraries[i]}: {result}")
                continue
            
            for search_result in result.results:
                aggregated_results.append(search_result)
        
        # Re-rank combined results
        ranked_results = await self._rerank_multi_library_results(
            aggregated_results, 
            query, 
            limit
        )
        
        return MultiLibrarySearchResponse(
            query=query,
            libraries=libraries,
            results=ranked_results,
            total_results=len(ranked_results)
        )
```

## Authentication and Authorization

### JWT-based Authentication System

```python
class AuthenticationService:
    """Comprehensive authentication service for Contexter API."""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.jwt_secret = config.jwt_secret_key
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        # API key management
        self.api_key_store = RedisAPIKeyStore(config.redis_client)
        
        # User session management
        self.session_store = UserSessionStore(config.redis_client)
    
    async def authenticate_jwt_token(
        self, 
        credentials: HTTPAuthorizationCredentials
    ) -> AuthenticatedUser:
        """Authenticate JWT bearer token."""
        
        try:
            token = credentials.credentials
            
            # Decode and validate JWT
            payload = jwt.decode(
                token, 
                self.jwt_secret, 
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": True}
            )
            
            user_id = payload.get("user_id")
            if not user_id:
                raise AuthenticationError("Invalid token: missing user_id")
            
            # Check if user session is still valid
            session_valid = await self.session_store.validate_session(
                user_id, 
                payload.get("session_id")
            )
            
            if not session_valid:
                raise AuthenticationError("Session expired or invalid")
            
            # Create authenticated user object
            return AuthenticatedUser(
                user_id=user_id,
                username=payload.get("username"),
                permissions=payload.get("permissions", []),
                rate_limit_tier=payload.get("rate_limit_tier", "basic"),
                authenticated_at=datetime.utcnow()
            )
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    async def authenticate_api_key(self, api_key: str) -> AuthenticatedUser:
        """Authenticate API key."""
        
        # Validate API key format
        if not self._validate_api_key_format(api_key):
            raise AuthenticationError("Invalid API key format")
        
        # Lookup API key in store
        api_key_info = await self.api_key_store.get_api_key_info(api_key)
        
        if not api_key_info:
            raise AuthenticationError("Invalid API key")
        
        if api_key_info.is_revoked:
            raise AuthenticationError("API key revoked")
        
        if api_key_info.expires_at and api_key_info.expires_at < datetime.utcnow():
            raise AuthenticationError("API key expired")
        
        # Update last used timestamp
        await self.api_key_store.update_last_used(api_key)
        
        return AuthenticatedUser(
            user_id=api_key_info.user_id,
            username=api_key_info.username,
            permissions=api_key_info.permissions,
            rate_limit_tier=api_key_info.rate_limit_tier,
            api_key_id=api_key_info.key_id,
            authenticated_at=datetime.utcnow()
        )

# Dependency injection for authentication
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> AuthenticatedUser:
    """Multi-method authentication dependency."""
    
    # Try API key authentication first
    if api_key:
        try:
            return await auth_service.authenticate_api_key(api_key)
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # Try JWT authentication
    if credentials:
        try:
            return await auth_service.authenticate_jwt_token(credentials)
        except AuthenticationError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
    
    # No authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"}
    )
```

## Rate Limiting and Performance

### Redis-based Rate Limiting

```python
class RateLimitingMiddleware:
    """Advanced rate limiting middleware with Redis backend."""
    
    def __init__(self, redis_client, rate_limits: Dict[str, RateLimit]):
        self.redis = redis_client
        self.rate_limits = rate_limits
        
        # Rate limit configurations by tier
        self.tier_limits = {
            "free": RateLimit(requests_per_minute=100, requests_per_hour=1000),
            "basic": RateLimit(requests_per_minute=300, requests_per_hour=5000),
            "premium": RateLimit(requests_per_minute=1000, requests_per_hour=20000),
            "enterprise": RateLimit(requests_per_minute=5000, requests_per_hour=100000)
        }
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Extract user information from request
        user_info = await self._extract_user_info(request)
        
        if user_info:
            # Check rate limits
            rate_limit_result = await self._check_rate_limits(
                user_info.user_id,
                user_info.rate_limit_tier,
                request.url.path
            )
            
            if rate_limit_result.exceeded:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": rate_limit_result.retry_after,
                        "limit": rate_limit_result.limit,
                        "reset_time": rate_limit_result.reset_time.isoformat()
                    },
                    headers={
                        "Retry-After": str(rate_limit_result.retry_after),
                        "X-RateLimit-Limit": str(rate_limit_result.limit),
                        "X-RateLimit-Remaining": str(rate_limit_result.remaining),
                        "X-RateLimit-Reset": str(int(rate_limit_result.reset_time.timestamp()))
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if user_info and hasattr(response, 'headers'):
            rate_limit_info = await self._get_rate_limit_info(
                user_info.user_id,
                user_info.rate_limit_tier
            )
            
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info.limit)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.remaining)
            response.headers["X-RateLimit-Reset"] = str(int(rate_limit_info.reset_time.timestamp()))
        
        return response
    
    async def _check_rate_limits(
        self, 
        user_id: str, 
        tier: str, 
        endpoint: str
    ) -> RateLimitResult:
        """Check if user has exceeded rate limits."""
        
        limits = self.tier_limits.get(tier, self.tier_limits["free"])
        current_time = datetime.utcnow()
        
        # Check minute and hour limits
        minute_key = f"rate_limit:{user_id}:minute:{current_time.strftime('%Y%m%d%H%M')}"
        hour_key = f"rate_limit:{user_id}:hour:{current_time.strftime('%Y%m%d%H')}"
        
        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)
        
        results = await pipe.execute()
        
        minute_count = results[0]
        hour_count = results[2]
        
        # Check if limits exceeded
        if minute_count > limits.requests_per_minute:
            return RateLimitResult(
                exceeded=True,
                limit=limits.requests_per_minute,
                remaining=0,
                retry_after=60 - current_time.second,
                reset_time=current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
            )
        
        if hour_count > limits.requests_per_hour:
            return RateLimitResult(
                exceeded=True,
                limit=limits.requests_per_hour,
                remaining=0,
                retry_after=3600 - (current_time.minute * 60 + current_time.second),
                reset_time=current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            )
        
        return RateLimitResult(
            exceeded=False,
            limit=limits.requests_per_minute,
            remaining=limits.requests_per_minute - minute_count,
            reset_time=current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        )
```

## Request/Response Models and Validation

### Pydantic Models for Type Safety

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
import re

class DocumentType(str, Enum):
    """Supported document types."""
    API = "api"
    GUIDE = "guide"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"

class SearchPrecision(str, Enum):
    """Search precision levels."""
    FAST = "fast"
    BALANCED = "balanced"
    PRECISE = "precise"

class SearchRequest(BaseModel):
    """Request model for semantic search."""
    
    query_text: str = Field(
        ..., 
        min_length=1, 
        max_length=1000,
        description="Search query text"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Search filters for library, document type, etc."
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    
    precision: SearchPrecision = Field(
        default=SearchPrecision.BALANCED,
        description="Search precision vs speed trade-off"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in search results"
    )
    
    @validator('query_text')
    def validate_query_text(cls, v):
        """Validate query text format."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        
        # Check for potentially malicious patterns
        if re.search(r'[<>"\']|script|javascript', v, re.IGNORECASE):
            raise ValueError("Query contains invalid characters")
        
        return v.strip()
    
    @validator('filters')
    def validate_filters(cls, v):
        """Validate filter parameters."""
        if v is None:
            return v
        
        # Validate known filter keys
        valid_keys = {
            'library_id', 'doc_type', 'programming_language', 
            'min_trust_score', 'indexed_after', 'section'
        }
        
        for key in v.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid filter key: {key}")
        
        # Validate specific filter values
        if 'doc_type' in v and v['doc_type'] not in [dt.value for dt in DocumentType]:
            raise ValueError("Invalid document type")
        
        if 'min_trust_score' in v and not (0 <= v['min_trust_score'] <= 10):
            raise ValueError("Trust score must be between 0 and 10")
        
        return v

class SearchResult(BaseModel):
    """Individual search result."""
    
    content: str = Field(..., description="Content text")
    score: float = Field(..., ge=0, le=1, description="Relevance score")
    library_id: str = Field(..., description="Source library identifier")
    library_name: str = Field(..., description="Human-readable library name")
    section: str = Field(..., description="Document section")
    doc_type: DocumentType = Field(..., description="Document type")
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    
    query: str = Field(..., description="Original query text")
    
    results: List[SearchResult] = Field(
        default_factory=list,
        description="Search results"
    )
    
    total_results: int = Field(
        ..., 
        ge=0,
        description="Total number of results found"
    )
    
    query_time: float = Field(
        ..., 
        ge=0,
        description="Query execution time in seconds"
    )
    
    filters_applied: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters that were applied"
    )

class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracking")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
```

## Error Handling and Monitoring

### Comprehensive Error Handling

```python
class ContexterAPIErrorHandler:
    """Centralized error handling for FastAPI endpoints."""
    
    @staticmethod
    def setup_exception_handlers(app: FastAPI):
        """Setup global exception handlers."""
        
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with consistent format."""
            
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTP_ERROR",
                    message=exc.detail,
                    request_id=request.headers.get("X-Request-ID"),
                    details={"status_code": exc.status_code}
                ).dict()
            )
        
        @app.exception_handler(ValidationError)
        async def validation_exception_handler(request: Request, exc: ValidationError):
            """Handle Pydantic validation errors."""
            
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(
                    error="VALIDATION_ERROR",
                    message="Request validation failed",
                    request_id=request.headers.get("X-Request-ID"),
                    details={"validation_errors": exc.errors()}
                ).dict()
            )
        
        @app.exception_handler(AuthenticationError)
        async def auth_exception_handler(request: Request, exc: AuthenticationError):
            """Handle authentication errors."""
            
            return JSONResponse(
                status_code=401,
                content=ErrorResponse(
                    error="AUTHENTICATION_ERROR",
                    message=str(exc),
                    request_id=request.headers.get("X-Request-ID")
                ).dict(),
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        @app.exception_handler(RateLimitExceededError)
        async def rate_limit_exception_handler(request: Request, exc: RateLimitExceededError):
            """Handle rate limit exceeded errors."""
            
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error="RATE_LIMIT_EXCEEDED",
                    message="Too many requests",
                    request_id=request.headers.get("X-Request-ID"),
                    details={
                        "retry_after": exc.retry_after,
                        "limit": exc.limit
                    }
                ).dict(),
                headers={"Retry-After": str(exc.retry_after)}
            )
        
        @app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle unexpected errors."""
            
            # Log error with full context
            logger.error(
                "Unhandled exception in API",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "exception": str(exc),
                    "exception_type": type(exc).__name__
                },
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred",
                    request_id=request.headers.get("X-Request-ID")
                ).dict()
            )
```

### Performance Monitoring Middleware

```python
class PerformanceMonitoringMiddleware:
    """Middleware for tracking API performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    async def __call__(self, request: Request, call_next):
        """Track request performance and metrics."""
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to headers
        request.headers["X-Request-ID"] = request_id
        
        # Track request start
        self.metrics.http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Track performance metrics
            self.metrics.http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).observe(response_time)
            
            # Track response status
            self.metrics.http_responses_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-ID"] = request_id
            
            # Log slow requests
            if response_time > 1.0:  # Log requests taking more than 1 second
                logger.warning(
                    f"Slow request detected",
                    extra={
                        "request_id": request_id,
                        "path": request.url.path,
                        "method": request.method,
                        "response_time": response_time,
                        "status_code": response.status_code
                    }
                )
            
            return response
            
        except Exception as e:
            # Track error metrics
            response_time = time.time() - start_time
            
            self.metrics.http_request_duration_seconds.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).observe(response_time)
            
            self.metrics.http_errors_total.labels(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            ).inc()
            
            # Re-raise exception for error handlers
            raise
```

## API Documentation and OpenAPI

### Enhanced OpenAPI Configuration

```python
def setup_openapi_documentation(app: FastAPI) -> None:
    """Configure comprehensive OpenAPI documentation."""
    
    # Custom OpenAPI schema with enhanced security definitions
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Contexter RAG API",
            version="1.0.0",
            description="""
# Contexter RAG API

High-performance Retrieval-Augmented Generation API for technical documentation search and retrieval.

## Features

- **Semantic Search**: Advanced vector-based similarity search
- **Multi-Library Support**: Search across multiple documentation libraries
- **Real-time Results**: Sub-100ms response times
- **Authentication**: JWT and API key support
- **Rate Limiting**: Tiered usage limits
- **OpenAPI 3.0**: Complete API specification

## Authentication

This API supports two authentication methods:

### JWT Bearer Token
Include JWT token in the Authorization header:
```
Authorization: Bearer <jwt_token>
```

### API Key
Include API key in the X-API-Key header:
```
X-API-Key: <api_key>
```

## Rate Limits

Rate limits are enforced based on your subscription tier:

| Tier | Requests/Minute | Requests/Hour |
|------|----------------|---------------|
| Free | 100 | 1,000 |
| Basic | 300 | 5,000 |
| Premium | 1,000 | 20,000 |
| Enterprise | 5,000 | 100,000 |

## Error Handling

All errors follow a consistent format:

```json
{
    "error": "ERROR_TYPE",
    "message": "Human-readable message",
    "details": {},
    "request_id": "uuid",
    "timestamp": "2024-01-01T00:00:00Z"
}
```
            """,
            routes=app.routes,
        )
        
        # Enhanced security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "JWTBearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token for user authentication"
            },
            "APIKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for programmatic access"
            }
        }
        
        # Apply security to all endpoints
        for path in openapi_schema["paths"]:
            for method in openapi_schema["paths"][path]:
                if method != "options":
                    openapi_schema["paths"][path][method]["security"] = [
                        {"JWTBearer": []},
                        {"APIKey": []}
                    ]
        
        # Add custom response examples
        _add_response_examples(openapi_schema)
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi

def _add_response_examples(openapi_schema: Dict[str, Any]) -> None:
    """Add comprehensive response examples to OpenAPI schema."""
    
    # Example responses for search endpoints
    search_examples = {
        "200": {
            "description": "Successful search response",
            "content": {
                "application/json": {
                    "example": {
                        "query": "How to create FastAPI endpoints",
                        "results": [
                            {
                                "content": "FastAPI makes it easy to create API endpoints...",
                                "score": 0.92,
                                "library_id": "fastapi/fastapi",
                                "library_name": "FastAPI",
                                "section": "Tutorial",
                                "doc_type": "guide",
                                "metadata": {
                                    "programming_language": "python",
                                    "trust_score": 9.8
                                }
                            }
                        ],
                        "total_results": 15,
                        "query_time": 0.045,
                        "filters_applied": {
                            "doc_type": "guide"
                        }
                    }
                }
            }
        },
        "400": {
            "description": "Bad request - invalid parameters",
            "content": {
                "application/json": {
                    "example": {
                        "error": "VALIDATION_ERROR",
                        "message": "Request validation failed",
                        "details": {
                            "validation_errors": [
                                {
                                    "loc": ["query_text"],
                                    "msg": "ensure this value has at least 1 characters",
                                    "type": "value_error.any_str.min_length"
                                }
                            ]
                        },
                        "request_id": "550e8400-e29b-41d4-a716-446655440000",
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }
            }
        }
    }
    
    # Apply examples to search endpoints
    for path in openapi_schema["paths"]:
        if "search" in path:
            for method in openapi_schema["paths"][path]:
                if "responses" in openapi_schema["paths"][path][method]:
                    openapi_schema["paths"][path][method]["responses"].update(search_examples)
```

## Integration with Contexter Architecture

### Service Integration Layer

```python
class ContexterAPIIntegration:
    """Integration layer between FastAPI and Contexter core services."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Initialize core Contexter services
        self.storage_manager = LocalStorageManager(
            base_path=config.storage_base_path,
            retention_limit=config.storage_retention_limit
        )
        
        self.vector_store = QdrantVectorStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.qdrant_collection
        )
        
        self.embedding_service = VoyageEmbeddingClient(
            VoyageClientConfig(
                api_key=config.voyage_api_key,
                model=config.embedding_model
            )
        )
        
        self.search_engine = ContexterSearchEngine(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service
        )
        
        # Initialize FastAPI app
        self.app = ContexterFastAPIApp(config).app
        
        # Setup dependency injection
        self._setup_dependencies()
        
        # Setup routes
        self._setup_api_routes()
    
    def _setup_dependencies(self):
        """Configure dependency injection for FastAPI."""
        
        async def get_vector_store() -> QdrantVectorStore:
            return self.vector_store
        
        async def get_embedding_service() -> VoyageEmbeddingClient:
            return self.embedding_service
        
        async def get_search_engine() -> ContexterSearchEngine:
            return self.search_engine
        
        async def get_auth_service() -> AuthenticationService:
            return AuthenticationService(self.config.auth_config)
        
        # Override FastAPI dependencies
        self.app.dependency_overrides.update({
            get_vector_store: lambda: self.vector_store,
            get_embedding_service: lambda: self.embedding_service,
            get_search_engine: lambda: self.search_engine,
            get_auth_service: lambda: AuthenticationService(self.config.auth_config)
        })
    
    def _setup_api_routes(self):
        """Setup API routes with proper integration."""
        
        # RAG endpoints
        rag_endpoints = RAGEndpoints(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            search_engine=self.search_engine,
            auth_service=AuthenticationService(self.config.auth_config)
        )
        
        # Health check endpoint
        @self.app.get("/health", tags=["Health"])
        async def health_check():
            """Health check endpoint for monitoring."""
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "services": {}
            }
            
            # Check vector store health
            try:
                vector_health = await self.vector_store.health_check()
                health_status["services"]["vector_store"] = {
                    "status": "healthy" if vector_health.is_healthy else "unhealthy",
                    "details": vector_health.dict()
                }
            except Exception as e:
                health_status["services"]["vector_store"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check embedding service health
            try:
                embedding_health = await self.embedding_service.health_check()
                health_status["services"]["embedding_service"] = {
                    "status": "healthy" if embedding_health.is_healthy else "unhealthy",
                    "details": embedding_health.dict()
                }
            except Exception as e:
                health_status["services"]["embedding_service"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Determine overall health
            service_statuses = [
                service["status"] for service in health_status["services"].values()
            ]
            
            if all(status == "healthy" for status in service_statuses):
                health_status["status"] = "healthy"
                return health_status
            else:
                health_status["status"] = "degraded"
                return JSONResponse(
                    status_code=503,
                    content=health_status
                )
        
        # Register search endpoints
        self.app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])(
            rag_endpoints.semantic_search
        )
        
        self.app.get("/api/v1/libraries/{library_id}/search", response_model=SearchResponse, tags=["Search"])(
            rag_endpoints.library_search
        )
        
        self.app.post("/api/v1/search/multi-library", response_model=MultiLibrarySearchResponse, tags=["Search"])(
            rag_endpoints.multi_library_search
        )
```

This comprehensive FastAPI integration specification provides all the necessary components for implementing a production-ready API layer for the Contexter RAG system, with full authentication, rate limiting, error handling, and monitoring capabilities that integrate seamlessly with existing Contexter patterns.