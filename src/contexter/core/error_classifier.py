"""
Error classification and recovery strategies for download operations.
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..models.context7_models import (
    AuthenticationError,
    Context7APIError,
    InvalidResponseError,
    LibraryNotFoundError,
    NetworkError,
    RateLimitError,
)
from ..models.download_models import DownloadTask

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Detailed error categories for classification."""

    RATE_LIMITED = "rate_limited"
    PROXY_ERROR = "proxy_error"
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    AUTHENTICATION = "authentication"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategy types."""

    NO_RETRY = "no_retry"
    IMMEDIATE = "immediate"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    PROXY_SWITCH = "proxy_switch"
    RATE_LIMIT_WAIT = "rate_limit_wait"


@dataclass
class RetryDecision:
    """Decision about whether and how to retry a failed operation."""

    should_retry: bool
    strategy: RetryStrategy
    delay: float
    max_additional_attempts: int
    switch_proxy: bool = False
    error_category: Optional[ErrorCategory] = None
    severity: Optional[ErrorSeverity] = None
    reason: str = ""
    actionable_advice: str = ""


@dataclass
class ErrorPattern:
    """Pattern for matching and classifying errors."""

    error_types: Tuple[type, ...]
    keywords: List[str]
    category: ErrorCategory
    severity: ErrorSeverity
    default_strategy: RetryStrategy
    max_retries: int
    base_delay: float

    def matches(self, error: Exception) -> bool:
        """Check if error matches this pattern."""
        # Check error type
        if self.error_types and isinstance(error, self.error_types):
            return True

        # Check keywords in error message
        error_message = str(error).lower()
        return any(keyword.lower() in error_message for keyword in self.keywords)


class ErrorClassifier:
    """
    Classifies errors and determines appropriate retry strategies.

    Provides intelligent error handling with context-aware retry decisions
    and actionable recovery advice.
    """

    def __init__(self) -> None:
        """Initialize error classifier with predefined patterns."""
        self.error_patterns = self._create_error_patterns()
        self.error_statistics = {
            "total_errors": 0,
            "by_category": {},
            "by_severity": {},
            "retry_success_rate": {},
        }

        logger.info("Initialized error classifier with comprehensive patterns")

    def _create_error_patterns(self) -> List[ErrorPattern]:
        """Create comprehensive error classification patterns."""

        patterns = [
            # Rate limiting errors
            ErrorPattern(
                error_types=(RateLimitError,),
                keywords=["rate limit", "too many requests", "429"],
                category=ErrorCategory.RATE_LIMITED,
                severity=ErrorSeverity.MEDIUM,
                default_strategy=RetryStrategy.RATE_LIMIT_WAIT,
                max_retries=3,
                base_delay=60.0,
            ),
            # Proxy-specific errors
            ErrorPattern(
                error_types=(),
                keywords=[
                    "proxy",
                    "tunnel",
                    "socks",
                    "connection refused",
                    "proxy authentication",
                ],
                category=ErrorCategory.PROXY_ERROR,
                severity=ErrorSeverity.MEDIUM,
                default_strategy=RetryStrategy.PROXY_SWITCH,
                max_retries=2,
                base_delay=1.0,
            ),
            # Network connectivity errors
            ErrorPattern(
                error_types=(NetworkError, httpx.ConnectError, httpx.TimeoutException),
                keywords=["connection", "timeout", "dns", "network", "unreachable"],
                category=ErrorCategory.NETWORK_ERROR,
                severity=ErrorSeverity.MEDIUM,
                default_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=3,
                base_delay=5.0,
            ),
            # Authentication errors
            ErrorPattern(
                error_types=(AuthenticationError,),
                keywords=[
                    "unauthorized",
                    "authentication",
                    "401",
                    "forbidden",
                    "403",
                    "invalid token",
                ],
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                default_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                base_delay=0.0,
            ),
            # Not found errors
            ErrorPattern(
                error_types=(LibraryNotFoundError,),
                keywords=["not found", "404", "does not exist", "invalid library"],
                category=ErrorCategory.NOT_FOUND,
                severity=ErrorSeverity.HIGH,
                default_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                base_delay=0.0,
            ),
            # Timeout errors
            ErrorPattern(
                error_types=(asyncio.TimeoutError,),
                keywords=["timeout", "timed out", "deadline exceeded"],
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                default_strategy=RetryStrategy.LINEAR_BACKOFF,
                max_retries=2,
                base_delay=10.0,
            ),
            # API server errors
            ErrorPattern(
                error_types=(Context7APIError, InvalidResponseError),
                keywords=[
                    "500",
                    "502",
                    "503",
                    "504",
                    "internal server error",
                    "bad gateway",
                ],
                category=ErrorCategory.API_ERROR,
                severity=ErrorSeverity.HIGH,
                default_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_retries=2,
                base_delay=15.0,
            ),
            # Configuration errors
            ErrorPattern(
                error_types=(ValueError, TypeError),
                keywords=["invalid", "missing", "configuration", "parameter"],
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                default_strategy=RetryStrategy.NO_RETRY,
                max_retries=0,
                base_delay=0.0,
            ),
        ]

        return patterns

    async def classify_and_decide_retry(
        self,
        error: Exception,
        task: DownloadTask,
        current_attempt: int,
        max_total_attempts: int,
    ) -> RetryDecision:
        """
        Classify error and make intelligent retry decision.

        Args:
            error: Exception that occurred
            task: Download task that failed
            current_attempt: Current attempt number (1-based)
            max_total_attempts: Maximum total attempts allowed

        Returns:
            Retry decision with strategy and timing
        """
        # Update statistics
        total_errors = self.error_statistics["total_errors"]
        assert isinstance(total_errors, int)
        self.error_statistics["total_errors"] = total_errors + 1

        # Find matching error pattern
        pattern = self._find_matching_pattern(error)

        if pattern:
            category = pattern.category
            severity = pattern.severity

            # Update category statistics
            by_category = self.error_statistics["by_category"]
            assert isinstance(by_category, dict)
            by_category[category.value] = by_category.get(category.value, 0) + 1

            # Update severity statistics
            by_severity = self.error_statistics["by_severity"]
            assert isinstance(by_severity, dict)
            by_severity[severity.value] = by_severity.get(severity.value, 0) + 1
        else:
            # Unknown error pattern
            category = ErrorCategory.UNKNOWN
            severity = ErrorSeverity.MEDIUM
            pattern = self._create_default_pattern()

        # Make retry decision based on pattern and context
        retry_decision = await self._make_retry_decision(
            error, task, pattern, current_attempt, max_total_attempts
        )

        # Add classification metadata
        retry_decision.error_category = category
        retry_decision.severity = severity

        # Generate actionable advice
        retry_decision.actionable_advice = self._generate_actionable_advice(
            error, pattern, retry_decision
        )

        logger.debug(
            f"Error classified: category={category.value}, severity={severity.value}, "
            f"retry={retry_decision.should_retry}, strategy={retry_decision.strategy.value}"
        )

        return retry_decision

    def _find_matching_pattern(self, error: Exception) -> Optional[ErrorPattern]:
        """Find the first matching error pattern."""
        for pattern in self.error_patterns:
            if pattern.matches(error):
                return pattern
        return None

    def _create_default_pattern(self) -> ErrorPattern:
        """Create default pattern for unknown errors."""
        return ErrorPattern(
            error_types=(),
            keywords=[],
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            default_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            max_retries=2,
            base_delay=10.0,
        )

    async def _make_retry_decision(
        self,
        error: Exception,
        task: DownloadTask,
        pattern: ErrorPattern,
        current_attempt: int,
        max_total_attempts: int,
    ) -> RetryDecision:
        """Make intelligent retry decision based on error pattern and context."""

        # Check if we've exhausted retry attempts
        if current_attempt >= max_total_attempts:
            return RetryDecision(
                should_retry=False,
                strategy=RetryStrategy.NO_RETRY,
                delay=0.0,
                max_additional_attempts=0,
                reason=f"Maximum attempts ({max_total_attempts}) reached",
            )

        # Check if error pattern allows retries
        if pattern.default_strategy == RetryStrategy.NO_RETRY:
            return RetryDecision(
                should_retry=False,
                strategy=RetryStrategy.NO_RETRY,
                delay=0.0,
                max_additional_attempts=0,
                reason=f"Error category {pattern.category.value} is not retryable",
            )

        # Check pattern-specific retry limits
        if current_attempt > pattern.max_retries:
            return RetryDecision(
                should_retry=False,
                strategy=RetryStrategy.NO_RETRY,
                delay=0.0,
                max_additional_attempts=0,
                reason=f"Pattern max retries ({pattern.max_retries}) exceeded",
            )

        # Calculate retry parameters based on strategy
        remaining_attempts = min(
            max_total_attempts - current_attempt,
            pattern.max_retries - (current_attempt - 1),
        )

        delay = self._calculate_retry_delay(pattern, current_attempt, error)

        # Special handling for specific error types
        should_switch_proxy = self._should_switch_proxy(pattern, error)

        return RetryDecision(
            should_retry=True,
            strategy=pattern.default_strategy,
            delay=delay,
            max_additional_attempts=remaining_attempts,
            switch_proxy=should_switch_proxy,
            reason=f"Retry attempt {current_attempt} using {pattern.default_strategy.value}",
        )

    def _calculate_retry_delay(
        self, pattern: ErrorPattern, attempt: int, error: Exception
    ) -> float:
        """Calculate appropriate retry delay based on strategy."""

        base_delay = pattern.base_delay

        if pattern.default_strategy == RetryStrategy.IMMEDIATE:
            return 0.0

        elif pattern.default_strategy == RetryStrategy.LINEAR_BACKOFF:
            # Linear increase: base_delay * attempt
            delay = base_delay * attempt

        elif pattern.default_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            # Exponential increase: base_delay * (2 ^ (attempt - 1))
            delay = base_delay * (2 ** (attempt - 1))

        elif pattern.default_strategy == RetryStrategy.PROXY_SWITCH:
            # Quick retry after proxy switch
            delay = random.uniform(1.0, 3.0)

        elif pattern.default_strategy == RetryStrategy.RATE_LIMIT_WAIT:
            # Handle rate limiting with jitter
            if isinstance(error, RateLimitError):
                delay = error.retry_delay_with_jitter
            else:
                delay = base_delay + random.uniform(5.0, 15.0)

        else:
            # Default exponential backoff
            delay = base_delay * (2 ** (attempt - 1))

        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5) * delay
        delay += jitter

        # Cap maximum delay
        max_delay = min(300.0, base_delay * 10)  # Max 5 minutes or 10x base
        delay = min(delay, max_delay)

        return delay

    def _should_switch_proxy(self, pattern: ErrorPattern, error: Exception) -> bool:
        """Determine if proxy should be switched for this error."""

        # Always switch for proxy-specific errors
        if pattern.category == ErrorCategory.PROXY_ERROR:
            return True

        # Switch for rate limiting (might be IP-based)
        if pattern.category == ErrorCategory.RATE_LIMITED:
            return True

        # Switch for certain network errors
        if pattern.category == ErrorCategory.NETWORK_ERROR:
            error_message = str(error).lower()
            proxy_indicators = ["connection refused", "tunnel", "proxy"]
            return any(indicator in error_message for indicator in proxy_indicators)

        return False

    def _generate_actionable_advice(
        self, error: Exception, pattern: ErrorPattern, decision: RetryDecision
    ) -> str:
        """Generate actionable advice for error resolution."""

        advice_map = {
            ErrorCategory.RATE_LIMITED: (
                "Rate limit encountered. Consider reducing request frequency "
                "or using proxy rotation to distribute load."
            ),
            ErrorCategory.PROXY_ERROR: (
                "Proxy connection failed. Check proxy configuration and credentials. "
                "If using BrightData, verify zone settings and account status."
            ),
            ErrorCategory.NETWORK_ERROR: (
                "Network connectivity issue. Check internet connection and "
                "DNS resolution. Consider increasing timeout values."
            ),
            ErrorCategory.AUTHENTICATION: (
                "Authentication failed. Verify API credentials and tokens. "
                "Check if credentials have expired or been revoked."
            ),
            ErrorCategory.NOT_FOUND: (
                "Library not found. Verify library ID format and existence. "
                "Check if library name or repository has changed."
            ),
            ErrorCategory.TIMEOUT: (
                "Request timed out. Consider increasing timeout values or "
                "reducing token limits to speed up processing."
            ),
            ErrorCategory.API_ERROR: (
                "API server error. This is usually temporary. "
                "Check Context7 service status and try again later."
            ),
            ErrorCategory.CONFIGURATION: (
                "Configuration error. Review parameters and settings. "
                "Ensure all required values are provided and valid."
            ),
            ErrorCategory.UNKNOWN: (
                "Unknown error occurred. Review error details and logs. "
                "Consider reporting if issue persists."
            ),
        }

        base_advice = advice_map.get(pattern.category, "Review error details and logs.")

        # Add retry-specific advice
        if decision.should_retry:
            retry_advice = f" Automatic retry in {decision.delay:.1f} seconds."
            if decision.switch_proxy:
                retry_advice += " Will switch to different proxy."
        else:
            retry_advice = " No automatic retry will be attempted."

        return base_advice + retry_advice

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        stats = self.error_statistics.copy()

        # Calculate percentages
        total_errors = stats["total_errors"]
        assert isinstance(total_errors, int)
        total = total_errors
        if total > 0:
            by_category = stats["by_category"]
            by_severity = stats["by_severity"]
            assert isinstance(by_category, dict)
            assert isinstance(by_severity, dict)

            stats["category_percentages"] = {
                category: (count / total) * 100
                for category, count in by_category.items()
            }

            stats["severity_percentages"] = {
                severity: (count / total) * 100
                for severity, count in by_severity.items()
            }

        return stats

    def get_top_error_categories(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top error categories by frequency."""
        by_category = self.error_statistics["by_category"]
        assert isinstance(by_category, dict)
        categories = by_category.items()
        return sorted(categories, key=lambda x: x[1], reverse=True)[:limit]

    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_statistics = {
            "total_errors": 0,
            "by_category": {},
            "by_severity": {},
            "retry_success_rate": {},
        }
        logger.info("Error statistics reset")


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and adaptive improvements.
    """

    def __init__(self, classifier: Optional[ErrorClassifier] = None):
        """
        Initialize error recovery manager.

        Args:
            classifier: Error classifier to use (creates default if None)
        """
        self.classifier = classifier or ErrorClassifier()
        self.recovery_history: Dict[str, Dict[str, Any]] = {}
        self.adaptive_strategies = {
            "proxy_switching_threshold": 0.3,  # Switch proxy if >30% proxy errors
            "rate_limit_backoff_multiplier": 1.5,  # Increase delays if frequent rate limits
            "max_concurrent_reduction": 0.5,  # Reduce concurrency if high error rates
        }

        logger.info("Initialized error recovery manager")

    async def handle_error_with_recovery(
        self,
        error: Exception,
        task: DownloadTask,
        attempt: int,
        max_attempts: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> RetryDecision:
        """
        Handle error with adaptive recovery strategies.

        Args:
            error: Exception that occurred
            task: Failed download task
            attempt: Current attempt number
            max_attempts: Maximum attempts allowed
            context: Additional context for adaptive decisions

        Returns:
            Enhanced retry decision with adaptive adjustments
        """
        # Get base retry decision from classifier
        decision = await self.classifier.classify_and_decide_retry(
            error, task, attempt, max_attempts
        )

        # Apply adaptive adjustments based on history
        if context:
            decision = self._apply_adaptive_adjustments(decision, error, context)

        # Record recovery attempt for learning
        self._record_recovery_attempt(error, decision, task.library_id)

        return decision

    def _apply_adaptive_adjustments(
        self, decision: RetryDecision, error: Exception, context: Dict[str, Any]
    ) -> RetryDecision:
        """Apply adaptive adjustments based on current context."""

        # Get current error rates from context
        error_rate = context.get("error_rate", 0.0)
        proxy_error_rate = context.get("proxy_error_rate", 0.0)
        rate_limit_frequency = context.get("rate_limit_frequency", 0.0)

        # Adjust proxy switching behavior
        if proxy_error_rate > self.adaptive_strategies["proxy_switching_threshold"]:
            if decision.error_category in [
                ErrorCategory.NETWORK_ERROR,
                ErrorCategory.TIMEOUT,
            ]:
                decision.switch_proxy = True
                logger.debug(
                    f"Adaptive proxy switching enabled due to high proxy error rate: {proxy_error_rate:.2f}"
                )

        # Adjust rate limit backoff
        if (
            rate_limit_frequency > 0.2
            and decision.error_category == ErrorCategory.RATE_LIMITED
        ):
            multiplier = self.adaptive_strategies["rate_limit_backoff_multiplier"]
            decision.delay *= multiplier
            logger.debug(
                f"Adaptive rate limit backoff applied: delay increased by {multiplier}x"
            )

        # Suggest concurrency reduction for high error rates
        if error_rate > 0.4:
            decision.actionable_advice += f" Consider reducing concurrency due to high error rate ({error_rate:.1%})."

        return decision

    def _record_recovery_attempt(
        self, error: Exception, decision: RetryDecision, library_id: str
    ) -> None:
        """Record recovery attempt for adaptive learning."""

        if library_id not in self.recovery_history:
            self.recovery_history[library_id] = {
                "attempts": 0,
                "successes": 0,
                "error_types": {},
                "recovery_strategies": {},
            }

        history = self.recovery_history[library_id]
        history["attempts"] += 1

        # Track error types
        error_type = type(error).__name__
        history["error_types"][error_type] = (
            history["error_types"].get(error_type, 0) + 1
        )

        # Track recovery strategies
        strategy = decision.strategy.value
        history["recovery_strategies"][strategy] = (
            history["recovery_strategies"].get(strategy, 0) + 1
        )

    def record_recovery_success(self, library_id: str, attempt_number: int) -> None:
        """Record successful recovery for adaptive learning."""
        if library_id in self.recovery_history:
            self.recovery_history[library_id]["successes"] += 1

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        total_libraries = len(self.recovery_history)
        total_attempts = sum(h["attempts"] for h in self.recovery_history.values())
        total_successes = sum(h["successes"] for h in self.recovery_history.values())

        recovery_rate = (
            (total_successes / total_attempts * 100) if total_attempts > 0 else 0
        )

        return {
            "total_libraries_with_errors": total_libraries,
            "total_recovery_attempts": total_attempts,
            "total_recoveries": total_successes,
            "overall_recovery_rate": recovery_rate,
            "adaptive_strategies": self.adaptive_strategies.copy(),
            "classifier_stats": self.classifier.get_error_statistics(),
        }

    def suggest_configuration_improvements(self) -> List[str]:
        """Suggest configuration improvements based on error patterns."""
        suggestions = []

        stats = self.classifier.get_error_statistics()
        top_categories = self.classifier.get_top_error_categories()

        for category, count in top_categories:
            percentage = (
                (count / stats["total_errors"]) * 100
                if stats["total_errors"] > 0
                else 0
            )

            if category == "rate_limited" and percentage > 20:
                suggestions.append(
                    f"High rate limiting ({percentage:.1f}%): Consider reducing concurrency "
                    "or implementing more aggressive jitter"
                )

            elif category == "proxy_error" and percentage > 15:
                suggestions.append(
                    f"High proxy errors ({percentage:.1f}%): Review proxy configuration "
                    "and consider expanding proxy pool"
                )

            elif category == "timeout" and percentage > 10:
                suggestions.append(
                    f"High timeout rate ({percentage:.1f}%): Consider increasing timeout values "
                    "or reducing token limits per request"
                )

            elif category == "network_error" and percentage > 15:
                suggestions.append(
                    f"High network errors ({percentage:.1f}%): Check network stability "
                    "and DNS configuration"
                )

        return suggestions
