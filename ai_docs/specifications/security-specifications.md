# Security Specifications

## Document Information
- **Version**: 1.0.0
- **Last Updated**: 2025-01-15
- **Specification Type**: Security Requirements
- **System**: Contexter RAG System

## Overview

This document defines comprehensive security requirements, threat models, authentication mechanisms, and security controls for the Contexter RAG system. All security measures follow industry best practices and compliance standards.

## 1. Authentication and Authorization

### 1.1 Authentication Mechanisms

```yaml
authentication_methods:
  api_key_authentication:
    format: "Bearer {api_key}"
    key_length: 64
    character_set: "alphanumeric + special chars"
    expiration: "90 days maximum"
    rotation_requirement: "mandatory"
    
  jwt_tokens:
    algorithm: "RS256"
    key_length: 2048
    expiration: "1 hour"
    refresh_token_expiration: "30 days"
    issuer: "contexter-auth-service"
    
  oauth2_integration:
    supported_providers: ["Google", "GitHub", "Microsoft"]
    scopes: ["read:docs", "write:docs", "admin:system"]
    pkce_required: true
```

### 1.2 Authorization Framework

```python
# Role-Based Access Control (RBAC) implementation
from enum import Enum
from typing import Set, Dict, Any

class Permission(Enum):
    READ_DOCUMENTS = "read:documents"
    WRITE_DOCUMENTS = "write:documents"
    DELETE_DOCUMENTS = "delete:documents"
    SEARCH_DOCUMENTS = "search:documents"
    ADMIN_SYSTEM = "admin:system"
    VIEW_METRICS = "view:metrics"
    MANAGE_USERS = "manage:users"

class Role(Enum):
    GUEST = "guest"
    USER = "user"
    PREMIUM_USER = "premium_user"
    ADMIN = "admin"
    SYSTEM = "system"

# Role permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.GUEST: {Permission.SEARCH_DOCUMENTS},
    Role.USER: {
        Permission.SEARCH_DOCUMENTS,
        Permission.READ_DOCUMENTS,
    },
    Role.PREMIUM_USER: {
        Permission.SEARCH_DOCUMENTS,
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.VIEW_METRICS,
    },
    Role.ADMIN: {
        Permission.SEARCH_DOCUMENTS,
        Permission.READ_DOCUMENTS,
        Permission.WRITE_DOCUMENTS,
        Permission.DELETE_DOCUMENTS,
        Permission.ADMIN_SYSTEM,
        Permission.VIEW_METRICS,
        Permission.MANAGE_USERS,
    },
    Role.SYSTEM: {
        # System role has all permissions
        perm for perm in Permission
    }
}

class SecurityContext:
    def __init__(self, user_id: str, role: Role, permissions: Set[Permission]):
        self.user_id = user_id
        self.role = role
        self.permissions = permissions
        self.session_id = generate_session_id()
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions
    
    def require_permission(self, permission: Permission) -> None:
        if not self.has_permission(permission):
            raise AuthorizationError(f"Permission {permission.value} required")
```

### 1.3 Session Management

```python
# Secure session management
class SessionManager:
    def __init__(self, redis_client, session_timeout: int = 3600):
        self.redis = redis_client
        self.session_timeout = session_timeout
        self.max_concurrent_sessions = 5
    
    async def create_session(self, user_id: str, context: SecurityContext) -> str:
        """Create secure session with timeout and concurrency limits."""
        # Check concurrent session limit
        existing_sessions = await self.get_user_sessions(user_id)
        if len(existing_sessions) >= self.max_concurrent_sessions:
            # Remove oldest session
            oldest_session = min(existing_sessions, key=lambda s: s['created_at'])
            await self.revoke_session(oldest_session['session_id'])
        
        # Generate secure session ID
        session_id = secrets.token_urlsafe(32)
        
        # Store session data
        session_data = {
            "user_id": user_id,
            "role": context.role.value,
            "permissions": [p.value for p in context.permissions],
            "created_at": context.created_at.isoformat(),
            "last_activity": context.last_activity.isoformat(),
            "ip_address": context.ip_address,
            "user_agent": context.user_agent,
        }
        
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(session_data)
        )
        
        return session_id
    
    async def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate session and update last activity."""
        session_data = await self.redis.get(f"session:{session_id}")
        if not session_data:
            return None
        
        data = json.loads(session_data)
        
        # Update last activity
        data['last_activity'] = datetime.utcnow().isoformat()
        await self.redis.setex(
            f"session:{session_id}",
            self.session_timeout,
            json.dumps(data)
        )
        
        return SecurityContext.from_session_data(data)
    
    async def revoke_session(self, session_id: str) -> None:
        """Revoke session immediately."""
        await self.redis.delete(f"session:{session_id}")
```

## 2. Data Protection

### 2.1 Encryption Standards

```yaml
encryption_requirements:
  data_at_rest:
    algorithm: "AES-256-GCM"
    key_management: "AWS KMS / HashiCorp Vault"
    key_rotation: "90 days"
    
    databases:
      sqlite: "SQLCipher encryption"
      qdrant: "TLS encryption + encrypted storage"
      redis: "TLS encryption + AUTH"
    
    file_storage:
      algorithm: "AES-256-CBC"
      key_derivation: "PBKDF2 (100k iterations)"
      compression_before_encryption: true
  
  data_in_transit:
    minimum_tls_version: "TLS 1.2"
    preferred_tls_version: "TLS 1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      - "TLS_AES_128_GCM_SHA256"
    
    certificate_requirements:
      key_size: 2048
      signature_algorithm: "SHA-256"
      validity_period: "365 days maximum"
      san_required: true
```

### 2.2 Data Classification and Handling

```python
# Data classification system
from enum import Enum
from typing import Dict, List, Any

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class PIIType(Enum):
    EMAIL = "email"
    IP_ADDRESS = "ip_address"
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    API_KEY = "api_key"

# Data handling policies
DATA_POLICIES: Dict[DataClassification, Dict[str, Any]] = {
    DataClassification.PUBLIC: {
        "encryption_required": False,
        "access_logging": False,
        "retention_days": 365,
        "anonymization_required": False,
    },
    DataClassification.INTERNAL: {
        "encryption_required": True,
        "access_logging": True,
        "retention_days": 1095,  # 3 years
        "anonymization_required": False,
    },
    DataClassification.CONFIDENTIAL: {
        "encryption_required": True,
        "access_logging": True,
        "retention_days": 2555,  # 7 years
        "anonymization_required": True,
        "access_approval_required": True,
    },
    DataClassification.RESTRICTED: {
        "encryption_required": True,
        "access_logging": True,
        "retention_days": 90,
        "anonymization_required": True,
        "access_approval_required": True,
        "multi_party_approval": True,
    }
}

class DataProtectionService:
    def __init__(self, encryption_service, audit_service):
        self.encryption_service = encryption_service
        self.audit_service = audit_service
    
    async def store_data(
        self, 
        data: Any, 
        classification: DataClassification,
        context: SecurityContext
    ) -> str:
        """Store data according to classification policy."""
        policy = DATA_POLICIES[classification]
        
        # Check access permissions
        if policy.get("access_approval_required"):
            if not context.has_permission(Permission.ADMIN_SYSTEM):
                raise AuthorizationError("Admin approval required")
        
        # Encrypt if required
        if policy["encryption_required"]:
            data = await self.encryption_service.encrypt(data)
        
        # Anonymize PII if required
        if policy["anonymization_required"]:
            data = self.anonymize_pii(data)
        
        # Log access if required
        if policy["access_logging"]:
            await self.audit_service.log_data_access(
                user_id=context.user_id,
                data_classification=classification.value,
                action="store",
                timestamp=datetime.utcnow()
            )
        
        # Store with TTL
        storage_id = await self.store_with_ttl(data, policy["retention_days"])
        
        return storage_id
    
    def anonymize_pii(self, data: str) -> str:
        """Anonymize personally identifiable information."""
        import re
        
        # Email addresses
        data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '[EMAIL_REDACTED]', data)
        
        # IP addresses
        data = re.sub(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', 
                     '[IP_REDACTED]', data)
        
        # API keys (common patterns)
        data = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[API_KEY_REDACTED]', data)
        
        return data
```

### 2.3 Sensitive Data Detection

```python
# Sensitive data detection and handling
import re
from typing import List, Tuple

class SensitiveDataDetector:
    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b(?:\d{3}-?\d{2}-?\d{4})\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'api_key': r'\b[A-Za-z0-9]{32,128}\b',
            'jwt': r'\beyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'private_key': r'-----BEGIN [A-Z ]+PRIVATE KEY-----',
        }
    
    def scan_content(self, content: str) -> List[Tuple[str, str, int, int]]:
        """Scan content for sensitive data patterns."""
        findings = []
        
        for pattern_name, pattern in self.patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                findings.append((
                    pattern_name,
                    match.group(),
                    match.start(),
                    match.end()
                ))
        
        return findings
    
    def redact_content(self, content: str) -> Tuple[str, List[dict]]:
        """Redact sensitive data from content."""
        redacted_content = content
        redaction_log = []
        
        findings = self.scan_content(content)
        
        # Sort by position (reverse order to maintain positions)
        findings.sort(key=lambda x: x[2], reverse=True)
        
        for pattern_name, matched_text, start_pos, end_pos in findings:
            # Replace with redaction marker
            redaction_marker = f"[{pattern_name.upper()}_REDACTED]"
            redacted_content = (
                redacted_content[:start_pos] + 
                redaction_marker + 
                redacted_content[end_pos:]
            )
            
            # Log redaction
            redaction_log.append({
                "type": pattern_name,
                "position": start_pos,
                "length": len(matched_text),
                "redacted_at": datetime.utcnow().isoformat(),
            })
        
        return redacted_content, redaction_log
```

## 3. Input Validation and Sanitization

### 3.1 Input Validation Framework

```python
# Comprehensive input validation
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict, Any
import bleach
import html

class InputValidator:
    """Centralized input validation and sanitization."""
    
    @staticmethod
    def sanitize_html(content: str) -> str:
        """Sanitize HTML content to prevent XSS."""
        allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'code', 'pre']
        allowed_attributes = {}
        
        return bleach.clean(
            content,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
    
    @staticmethod
    def sanitize_sql(query: str) -> str:
        """Basic SQL injection prevention."""
        # Remove dangerous SQL keywords
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
            'EXEC', 'EXECUTE', 'UNION', 'SCRIPT', '--', '/*', '*/'
        ]
        
        sanitized = query
        for keyword in dangerous_keywords:
            sanitized = re.sub(
                f'\\b{keyword}\\b', 
                '', 
                sanitized, 
                flags=re.IGNORECASE
            )
        
        return sanitized.strip()
    
    @staticmethod
    def validate_file_upload(file_content: bytes, filename: str) -> bool:
        """Validate uploaded file for security."""
        # Check file size (max 100MB)
        if len(file_content) > 100 * 1024 * 1024:
            raise ValidationError("File too large")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.txt', '.md', '.json', '.csv'}
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValidationError(f"File type {file_ext} not allowed")
        
        # Check for malicious content
        if b'<script' in file_content.lower():
            raise ValidationError("Potentially malicious content detected")
        
        return True

# Pydantic models with validation
class SearchQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=20, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def sanitize_query(cls, v):
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';]', '', v)
        # Limit length and strip whitespace
        return sanitized.strip()[:1000]
    
    @validator('filters')
    def validate_filters(cls, v):
        if v is None:
            return v
        
        # Limit filter complexity
        if len(v) > 10:
            raise ValueError("Too many filters")
        
        # Validate filter values
        for key, value in v.items():
            if not isinstance(key, str) or len(key) > 50:
                raise ValueError(f"Invalid filter key: {key}")
            
            if isinstance(value, str) and len(value) > 200:
                raise ValueError(f"Filter value too long for key: {key}")
        
        return v

class DocumentUploadRequest(BaseModel):
    library_id: str = Field(..., min_length=1, max_length=100)
    version: Optional[str] = Field(None, max_length=50)
    file_content: bytes
    filename: str = Field(..., min_length=1, max_length=255)
    
    @validator('library_id')
    def validate_library_id(cls, v):
        # Allow only alphanumeric, dash, underscore
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Invalid library_id format")
        return v
    
    @validator('filename')
    def validate_filename(cls, v):
        # Sanitize filename
        sanitized = re.sub(r'[<>:"/\\|?*]', '', v)
        return sanitized[:255]
```

### 3.2 Rate Limiting and DDoS Protection

```python
# Rate limiting implementation
from collections import defaultdict, deque
import time
from typing import Dict, Deque

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.local_cache: Dict[str, Deque[float]] = defaultdict(deque)
    
    async def is_allowed(
        self, 
        identifier: str, 
        limit: int, 
        window_seconds: int,
        burst_limit: Optional[int] = None
    ) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        redis_key = f"rate_limit:{identifier}:{window_seconds}"
        
        try:
            # Use Redis for distributed rate limiting
            async with self.redis.pipeline() as pipe:
                pipe.multi()
                pipe.zremrangebyscore(redis_key, 0, current_time - window_seconds)
                pipe.zcard(redis_key)
                pipe.zadd(redis_key, {str(current_time): current_time})
                pipe.expire(redis_key, window_seconds)
                results = await pipe.execute()
                
                current_requests = results[1]
                
                # Check burst limit first
                if burst_limit and current_requests > burst_limit:
                    await self._log_rate_limit_violation(
                        identifier, "burst_limit", current_requests, burst_limit
                    )
                    return False
                
                # Check regular limit
                if current_requests >= limit:
                    await self._log_rate_limit_violation(
                        identifier, "rate_limit", current_requests, limit
                    )
                    return False
                
                return True
                
        except Exception as e:
            # Fallback to local rate limiting
            logger.warning(f"Redis rate limiting failed, using local: {e}")
            return self._local_rate_limit(identifier, limit, window_seconds)
    
    def _local_rate_limit(self, identifier: str, limit: int, window_seconds: int) -> bool:
        """Fallback local rate limiting."""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Clean old entries
        requests = self.local_cache[identifier]
        while requests and requests[0] < window_start:
            requests.popleft()
        
        # Check limit
        if len(requests) >= limit:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    async def _log_rate_limit_violation(
        self, 
        identifier: str, 
        violation_type: str, 
        current_requests: int, 
        limit: int
    ):
        """Log rate limit violations for security monitoring."""
        violation_data = {
            "identifier": identifier,
            "violation_type": violation_type,
            "current_requests": current_requests,
            "limit": limit,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Store violation for analysis
        await self.redis.lpush(
            "security:rate_limit_violations",
            json.dumps(violation_data)
        )
        
        # Keep only last 1000 violations
        await self.redis.ltrim("security:rate_limit_violations", 0, 999)

# Rate limiting configuration
RATE_LIMITS = {
    "search_queries": {"limit": 60, "window": 60, "burst": 10},  # 60/min, burst 10
    "document_upload": {"limit": 10, "window": 60, "burst": 3},   # 10/min, burst 3
    "api_general": {"limit": 1000, "window": 3600, "burst": 100}, # 1000/hour, burst 100
    "authentication": {"limit": 5, "window": 300, "burst": 2},    # 5/5min, burst 2
}
```

## 4. Security Monitoring and Auditing

### 4.1 Audit Logging

```python
# Comprehensive audit logging system
from enum import Enum
from typing import Optional, Dict, Any

class AuditEventType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ADMIN = "system_admin"

class AuditSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditLogger:
    def __init__(self, storage_service, encryption_service):
        self.storage = storage_service
        self.encryption = encryption_service
    
    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        resource: Optional[str],
        action: str,
        result: str,  # success, failure, error
        details: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Log security audit event."""
        
        audit_event = {
            "event_id": generate_uuid(),
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "severity": severity.value,
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "resource": resource,
            "action": action,
            "result": result,
            "details": details or {},
            "error_message": error_message,
        }
        
        # Encrypt sensitive audit data
        encrypted_event = await self.encryption.encrypt(json.dumps(audit_event))
        
        # Store in tamper-evident log
        event_id = await self.storage.store_audit_event(encrypted_event)
        
        # Send critical events to SIEM
        if severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
            await self._send_to_siem(audit_event)
        
        return event_id
    
    async def _send_to_siem(self, audit_event: Dict[str, Any]):
        """Send high-priority events to SIEM system."""
        # Implementation would integrate with SIEM system
        pass
```

### 4.2 Intrusion Detection

```python
# Intrusion detection system
class SecurityMonitor:
    def __init__(self, redis_client, alert_service):
        self.redis = redis_client
        self.alert_service = alert_service
        self.suspicious_patterns = {
            'brute_force': {
                'failed_logins': {'count': 5, 'window': 300},  # 5 failures in 5 minutes
                'action': 'temporary_ban',
            },
            'sql_injection': {
                'patterns': [
                    r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                    r"\b(drop|delete|update)\b.*\btable\b",
                    r"['\"]\s*;\s*(drop|delete|update|insert)",
                ],
                'action': 'immediate_ban',
            },
            'xss_attempt': {
                'patterns': [
                    r"<script[^>]*>.*?</script>",
                    r"javascript:",
                    r"on\w+\s*=",
                ],
                'action': 'block_request',
            },
            'data_exfiltration': {
                'large_requests': {'size': 1000, 'count': 10, 'window': 60},
                'action': 'rate_limit',
            },
        }
    
    async def analyze_request(
        self, 
        user_id: str, 
        ip_address: str, 
        request_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Analyze incoming request for security threats."""
        threats_detected = []
        
        # Check for brute force attempts
        if await self._detect_brute_force(user_id, ip_address):
            threats_detected.append("brute_force")
        
        # Check for injection attempts
        request_content = json.dumps(request_data)
        if await self._detect_injection_attempts(request_content):
            threats_detected.append("injection_attempt")
        
        # Check for unusual access patterns
        if await self._detect_unusual_patterns(user_id, request_data):
            threats_detected.append("unusual_access")
        
        # Take action if threats detected
        if threats_detected:
            await self._handle_security_threats(
                user_id, ip_address, threats_detected
            )
        
        # Allow request if no critical threats
        critical_threats = {'sql_injection', 'brute_force'}
        is_allowed = not any(threat in critical_threats for threat in threats_detected)
        
        return is_allowed, threats_detected
    
    async def _detect_brute_force(self, user_id: str, ip_address: str) -> bool:
        """Detect brute force authentication attempts."""
        keys_to_check = [
            f"failed_login:{user_id}",
            f"failed_login:ip:{ip_address}",
        ]
        
        for key in keys_to_check:
            failed_attempts = await self.redis.get(key)
            if failed_attempts and int(failed_attempts) >= 5:
                return True
        
        return False
    
    async def _detect_injection_attempts(self, content: str) -> bool:
        """Detect SQL injection and XSS attempts."""
        injection_patterns = (
            self.suspicious_patterns['sql_injection']['patterns'] +
            self.suspicious_patterns['xss_attempt']['patterns']
        )
        
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    async def _detect_unusual_patterns(self, user_id: str, request_data: Dict[str, Any]) -> bool:
        """Detect unusual access patterns."""
        # Check for unusual request sizes
        request_size = len(json.dumps(request_data))
        if request_size > 1024 * 1024:  # 1MB
            return True
        
        # Check request frequency
        current_minute = int(time.time() / 60)
        request_key = f"requests:{user_id}:{current_minute}"
        request_count = await self.redis.incr(request_key)
        await self.redis.expire(request_key, 60)
        
        if request_count > 100:  # More than 100 requests per minute
            return True
        
        return False
    
    async def _handle_security_threats(
        self, 
        user_id: str, 
        ip_address: str, 
        threats: List[str]
    ):
        """Handle detected security threats."""
        for threat in threats:
            if threat == "brute_force":
                # Temporary ban
                await self.redis.setex(f"banned:{ip_address}", 3600, "brute_force")
                
            elif threat == "injection_attempt":
                # Immediate ban
                await self.redis.setex(f"banned:{ip_address}", 86400, "injection")
                
            elif threat == "unusual_access":
                # Rate limit
                await self.redis.setex(f"rate_limited:{user_id}", 300, "unusual")
        
        # Send alert
        await self.alert_service.send_security_alert(
            severity="HIGH",
            threats=threats,
            user_id=user_id,
            ip_address=ip_address,
            timestamp=datetime.utcnow()
        )
```

## 5. Vulnerability Management

### 5.1 Security Headers and Hardening

```python
# Security headers middleware
class SecurityHeadersMiddleware:
    def __init__(self):
        self.security_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Prevent framing
            "X-Frame-Options": "DENY",
            
            # Strict transport security
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            ),
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Feature policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=()"
            ),
        }
    
    async def process_response(self, response):
        """Add security headers to response."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Remove server information
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)
        
        return response
```

### 5.2 Dependency Security Scanning

```yaml
# Security scanning configuration
security_scanning:
  dependency_scanning:
    tools:
      - "pip-audit"  # Python dependencies
      - "safety"     # Known vulnerabilities
      - "bandit"     # Code security analysis
    
    schedule: "daily"
    failure_threshold: "high"
    auto_remediation: false
    
  container_scanning:
    tools:
      - "trivy"      # Container vulnerability scanner
      - "grype"      # Container security scanner
    
    base_images:
      - "python:3.11-slim"
      - "redis:7-alpine"
      - "ubuntu:22.04"
    
    compliance_profiles:
      - "CIS Docker Benchmark"
      - "NIST Cybersecurity Framework"
  
  code_scanning:
    static_analysis:
      - "semgrep"    # Static analysis
      - "CodeQL"     # Security vulnerabilities
      - "sonarqube"  # Code quality and security
    
    dynamic_analysis:
      - "OWASP ZAP"  # Web application security
      - "nuclei"     # Vulnerability scanner
```

## 6. Incident Response

### 6.1 Security Incident Classification

```yaml
incident_classification:
  severity_levels:
    critical:
      description: "Active security breach, data compromise, or system compromise"
      response_time: "15 minutes"
      escalation: "C-level executives"
      examples:
        - "Unauthorized access to production data"
        - "Data exfiltration detected"
        - "Ransomware attack"
        - "Root access compromise"
    
    high:
      description: "Significant security threat or vulnerability exploitation"
      response_time: "1 hour"
      escalation: "Security team lead"
      examples:
        - "Successful brute force attack"
        - "SQL injection exploitation"
        - "Privilege escalation"
        - "Sensitive data exposure"
    
    medium:
      description: "Security policy violation or suspicious activity"
      response_time: "4 hours"
      escalation: "Security team"
      examples:
        - "Multiple failed login attempts"
        - "Unusual access patterns"
        - "Policy violations"
        - "Configuration weaknesses"
    
    low:
      description: "Security event requiring investigation"
      response_time: "24 hours"
      escalation: "Security analyst"
      examples:
        - "Failed authentication events"
        - "Rate limiting triggered"
        - "Information gathering attempts"
```

### 6.2 Incident Response Procedures

```python
# Incident response automation
class IncidentResponseManager:
    def __init__(self, alert_service, audit_service, communication_service):
        self.alert_service = alert_service
        self.audit_service = audit_service
        self.communication_service = communication_service
        self.incident_counter = 0
    
    async def handle_security_incident(
        self,
        severity: str,
        incident_type: str,
        affected_systems: List[str],
        evidence: Dict[str, Any],
        detected_by: str,
    ) -> str:
        """Handle security incident according to severity."""
        
        self.incident_counter += 1
        incident_id = f"SEC-{datetime.now().strftime('%Y%m%d')}-{self.incident_counter:04d}"
        
        # Create incident record
        incident = {
            "incident_id": incident_id,
            "severity": severity,
            "incident_type": incident_type,
            "affected_systems": affected_systems,
            "evidence": evidence,
            "detected_by": detected_by,
            "detected_at": datetime.utcnow().isoformat(),
            "status": "new",
        }
        
        # Log incident
        await self.audit_service.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            action="incident_created",
            result="success",
            details=incident
        )
        
        # Execute severity-specific response
        if severity == "critical":
            await self._handle_critical_incident(incident)
        elif severity == "high":
            await self._handle_high_incident(incident)
        elif severity == "medium":
            await self._handle_medium_incident(incident)
        else:
            await self._handle_low_incident(incident)
        
        return incident_id
    
    async def _handle_critical_incident(self, incident: Dict[str, Any]):
        """Handle critical security incidents."""
        # Immediate containment
        await self._isolate_affected_systems(incident["affected_systems"])
        
        # Emergency notifications
        await self.communication_service.send_emergency_alert(
            recipients=["security-team@company.com", "cto@company.com"],
            incident_id=incident["incident_id"],
            severity="CRITICAL",
            details=incident
        )
        
        # Preserve evidence
        await self._preserve_forensic_evidence(incident)
        
        # Update incident status
        incident["status"] = "containment"
        incident["response_started_at"] = datetime.utcnow().isoformat()
    
    async def _isolate_affected_systems(self, systems: List[str]):
        """Isolate affected systems to prevent spread."""
        for system in systems:
            # Implementation would depend on infrastructure
            # Could involve disabling services, blocking traffic, etc.
            logger.critical(f"Isolating system: {system}")
    
    async def _preserve_forensic_evidence(self, incident: Dict[str, Any]):
        """Preserve evidence for forensic analysis."""
        evidence_snapshot = {
            "incident_id": incident["incident_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "system_logs": await self._collect_system_logs(),
            "network_traffic": await self._collect_network_logs(),
            "audit_trail": await self._collect_audit_trail(),
            "memory_dumps": await self._collect_memory_dumps(),
        }
        
        # Store evidence securely
        await self._store_forensic_evidence(evidence_snapshot)
```

## 7. Compliance and Standards

### 7.1 Compliance Framework

```yaml
compliance_standards:
  gdpr:
    requirements:
      - "Data subject consent management"
      - "Right to be forgotten implementation"
      - "Data portability features"
      - "Privacy by design principles"
      - "Data protection impact assessments"
    
    implementation:
      consent_management: true
      data_deletion: true
      data_export: true
      privacy_controls: true
    
  ccpa:
    requirements:
      - "Consumer right to know"
      - "Consumer right to delete"
      - "Consumer right to opt-out"
      - "Non-discrimination provisions"
    
    implementation:
      disclosure_requirements: true
      deletion_rights: true
      opt_out_mechanisms: true
  
  sox:
    requirements:
      - "Internal controls over financial reporting"
      - "Audit trail maintenance"
      - "Change management processes"
      - "Data integrity controls"
    
    implementation:
      audit_logging: true
      change_controls: true
      data_validation: true
  
  iso_27001:
    requirements:
      - "Information security management system"
      - "Risk assessment and treatment"
      - "Security controls implementation"
      - "Continual improvement processes"
    
    implementation:
      isms_framework: true
      risk_management: true
      security_controls: true
      monitoring_review: true
```

### 7.2 Data Retention and Privacy

```python
# Privacy and data retention management
class PrivacyManager:
    def __init__(self, storage_service, audit_service):
        self.storage = storage_service
        self.audit = audit_service
        self.retention_policies = {
            "user_data": {"retention_days": 2555, "anonymize": True},     # 7 years
            "session_data": {"retention_days": 30, "anonymize": False},    # 30 days
            "audit_logs": {"retention_days": 2555, "anonymize": False},    # 7 years
            "search_queries": {"retention_days": 365, "anonymize": True},  # 1 year
            "error_logs": {"retention_days": 90, "anonymize": True},       # 90 days
        }
    
    async def handle_data_subject_request(
        self,
        request_type: str,  # "access", "deletion", "portability"
        user_id: str,
        verification_token: str,
    ) -> Dict[str, Any]:
        """Handle GDPR data subject requests."""
        
        # Verify request authenticity
        if not await self._verify_data_subject_request(user_id, verification_token):
            raise AuthenticationError("Request verification failed")
        
        # Log the request
        await self.audit.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            action=f"data_subject_request_{request_type}",
            result="initiated"
        )
        
        if request_type == "access":
            return await self._handle_data_access_request(user_id)
        elif request_type == "deletion":
            return await self._handle_data_deletion_request(user_id)
        elif request_type == "portability":
            return await self._handle_data_portability_request(user_id)
        else:
            raise ValueError(f"Invalid request type: {request_type}")
    
    async def _handle_data_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """Handle right to be forgotten request."""
        deleted_data = {
            "user_sessions": 0,
            "search_queries": 0,
            "uploaded_documents": 0,
            "preferences": 0,
        }
        
        # Delete user sessions
        deleted_data["user_sessions"] = await self.storage.delete_user_sessions(user_id)
        
        # Anonymize search queries
        deleted_data["search_queries"] = await self.storage.anonymize_user_queries(user_id)
        
        # Delete uploaded documents (if user owns them)
        deleted_data["uploaded_documents"] = await self.storage.delete_user_documents(user_id)
        
        # Delete user preferences
        deleted_data["preferences"] = await self.storage.delete_user_preferences(user_id)
        
        # Log completion
        await self.audit.log_event(
            event_type=AuditEventType.DATA_MODIFICATION,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            action="data_deletion_completed",
            result="success",
            details=deleted_data
        )
        
        return {
            "request_id": generate_uuid(),
            "status": "completed",
            "deleted_items": deleted_data,
            "completed_at": datetime.utcnow().isoformat(),
        }
```

## 8. Security Testing

### 8.1 Security Test Framework

```python
# Security testing framework
import pytest
from unittest.mock import Mock, patch
from security.authentication import AuthenticationService
from security.authorization import AuthorizationService

class SecurityTestSuite:
    """Comprehensive security testing suite."""
    
    @pytest.fixture
    def auth_service(self):
        return AuthenticationService()
    
    @pytest.fixture
    def authz_service(self):
        return AuthorizationService()
    
    # Authentication tests
    async def test_password_strength_requirements(self, auth_service):
        """Test password strength validation."""
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "abc123",
            "password123",
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValidationError):
                await auth_service.validate_password(password)
    
    async def test_brute_force_protection(self, auth_service):
        """Test brute force attack protection."""
        user_id = "test_user"
        
        # Simulate multiple failed login attempts
        for i in range(5):
            result = await auth_service.authenticate(user_id, "wrong_password")
            assert not result.success
        
        # Sixth attempt should be blocked
        result = await auth_service.authenticate(user_id, "wrong_password")
        assert result.blocked_by_rate_limit
    
    # Authorization tests
    async def test_privilege_escalation_prevention(self, authz_service):
        """Test prevention of privilege escalation."""
        user_context = SecurityContext(
            user_id="regular_user",
            role=Role.USER,
            permissions={Permission.READ_DOCUMENTS}
        )
        
        # Should not be able to access admin functions
        with pytest.raises(AuthorizationError):
            authz_service.require_permission(user_context, Permission.ADMIN_SYSTEM)
    
    # Input validation tests
    async def test_sql_injection_prevention(self):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE documents; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM users",
            "'; UPDATE users SET role='admin' WHERE id=1; --",
        ]
        
        validator = InputValidator()
        
        for malicious_input in malicious_inputs:
            sanitized = validator.sanitize_sql(malicious_input)
            # Should not contain dangerous SQL keywords
            dangerous_keywords = ['DROP', 'UPDATE', 'UNION', '--']
            for keyword in dangerous_keywords:
                assert keyword not in sanitized.upper()
    
    async def test_xss_prevention(self):
        """Test XSS attack prevention."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
        ]
        
        validator = InputValidator()
        
        for payload in xss_payloads:
            sanitized = validator.sanitize_html(payload)
            # Should not contain script tags or javascript
            assert '<script' not in sanitized.lower()
            assert 'javascript:' not in sanitized.lower()
            assert 'onerror=' not in sanitized.lower()
    
    # Encryption tests
    async def test_data_encryption_integrity(self):
        """Test data encryption and decryption integrity."""
        encryption_service = EncryptionService()
        
        test_data = "Sensitive information that needs encryption"
        
        # Encrypt data
        encrypted = await encryption_service.encrypt(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt data
        decrypted = await encryption_service.decrypt(encrypted)
        assert decrypted == test_data
    
    # Rate limiting tests
    async def test_rate_limiting_enforcement(self):
        """Test rate limiting enforcement."""
        rate_limiter = RateLimiter(Mock())
        identifier = "test_user"
        
        # Should allow requests within limit
        for i in range(10):
            allowed = await rate_limiter.is_allowed(identifier, 10, 60)
            assert allowed
        
        # Should block requests exceeding limit
        blocked = await rate_limiter.is_allowed(identifier, 10, 60)
        assert not blocked
    
    # Session management tests
    async def test_session_security(self):
        """Test session security features."""
        session_manager = SessionManager(Mock())
        
        # Create session
        context = SecurityContext("user123", Role.USER, {Permission.READ_DOCUMENTS})
        session_id = await session_manager.create_session("user123", context)
        
        # Session ID should be cryptographically secure
        assert len(session_id) >= 32
        assert session_id.isalnum() or '_' in session_id or '-' in session_id
        
        # Should validate session correctly
        retrieved_context = await session_manager.validate_session(session_id)
        assert retrieved_context.user_id == "user123"
        
        # Should invalidate revoked sessions
        await session_manager.revoke_session(session_id)
        revoked_context = await session_manager.validate_session(session_id)
        assert revoked_context is None
```

### 8.2 Penetration Testing Checklist

```yaml
penetration_testing:
  scope:
    - "Web application endpoints"
    - "API authentication mechanisms"
    - "Database security"
    - "File upload functionality"
    - "Session management"
    - "Input validation"
  
  methodology:
    - "OWASP Testing Guide"
    - "NIST SP 800-115"
    - "PTES (Penetration Testing Execution Standard)"
  
  test_categories:
    authentication:
      - "Brute force attacks"
      - "Password policy bypass"
      - "Session fixation"
      - "Authentication bypass"
      - "Multi-factor authentication bypass"
    
    authorization:
      - "Privilege escalation"
      - "Horizontal privilege bypass"
      - "Role-based access control bypass"
      - "Resource-based authorization bypass"
    
    input_validation:
      - "SQL injection"
      - "NoSQL injection"
      - "Cross-site scripting (XSS)"
      - "XML external entity (XXE)"
      - "Server-side request forgery (SSRF)"
      - "Template injection"
    
    session_management:
      - "Session hijacking"
      - "Session fixation"
      - "Session timeout bypass"
      - "Concurrent session limits"
    
    business_logic:
      - "Rate limiting bypass"
      - "Workflow bypass"
      - "Data validation bypass"
      - "Financial calculation manipulation"
  
  tools:
    automated:
      - "OWASP ZAP"
      - "Burp Suite Professional"
      - "Nuclei"
      - "Nessus"
      - "OpenVAS"
    
    manual:
      - "Burp Suite Professional"
      - "Custom Python scripts"
      - "cURL"
      - "Postman"
  
  reporting:
    severity_classification: "CVSS 3.1"
    remediation_timeline:
      critical: "24 hours"
      high: "72 hours"
      medium: "30 days"
      low: "90 days"
```

---

**Document Control:**
- **Created**: 2025-01-15
- **Version**: 1.0.0
- **Next Review**: 2025-02-15
- **Owner**: Contexter Security Team
- **Stakeholders**: Development Team, DevOps, Compliance Team