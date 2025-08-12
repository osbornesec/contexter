"""
Context generation engine for intelligent documentation retrieval.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Set

from ..models.download_models import ContextGenerationError

logger = logging.getLogger(__name__)


@dataclass
class LibraryProfile:
    """Profile containing library-specific context customization data."""

    name: str
    domain: str
    keywords: Set[str]
    context_modifiers: List[str]
    priority_topics: List[str]

    @classmethod
    def from_library_id(cls, library_id: str) -> "LibraryProfile":
        """Create library profile from library ID analysis."""
        # Extract library name (e.g., 'encode/httpx' -> 'httpx')
        name = library_id.split("/")[-1].lower()

        # Determine domain and keywords based on name patterns
        domain, keywords, modifiers, priorities = cls._analyze_library_name(name)

        return cls(
            name=name,
            domain=domain,
            keywords=keywords,
            context_modifiers=modifiers,
            priority_topics=priorities,
        )

    @classmethod
    def _analyze_library_name(
        cls, name: str
    ) -> tuple[str, Set[str], List[str], List[str]]:
        """Analyze library name to determine domain and characteristics."""

        # Domain classification patterns
        web_patterns = ["http", "web", "request", "client", "api", "rest", "scrape"]
        data_patterns = ["pandas", "numpy", "data", "csv", "json", "xml", "parse"]
        test_patterns = ["test", "pytest", "unittest", "mock", "fixture"]
        ml_patterns = ["ml", "torch", "tensor", "scikit", "learn", "model"]
        async_patterns = ["async", "await", "trio", "asyncio", "aio"]
        db_patterns = ["sql", "mongo", "redis", "db", "database", "orm"]
        ui_patterns = ["gui", "tkinter", "qt", "web", "flask", "django", "fastapi"]
        crypto_patterns = ["crypto", "hash", "encrypt", "ssl", "tls", "jwt"]

        # Initialize defaults
        domain = "general"
        keywords = {name}
        modifiers = []
        priorities = []

        # Check patterns and classify
        if any(pattern in name for pattern in web_patterns):
            domain = "web"
            modifiers.extend(["HTTP requests", "API integration", "web scraping"])
            priorities.extend(["authentication", "error handling", "rate limiting"])

        elif any(pattern in name for pattern in data_patterns):
            domain = "data"
            modifiers.extend(["data processing", "analysis", "manipulation"])
            priorities.extend(["performance", "memory usage", "large datasets"])

        elif any(pattern in name for pattern in test_patterns):
            domain = "testing"
            modifiers.extend(["unit testing", "integration testing", "mocking"])
            priorities.extend(["test fixtures", "assertions", "coverage"])

        elif any(pattern in name for pattern in ml_patterns):
            domain = "machine_learning"
            modifiers.extend(["machine learning", "model training", "predictions"])
            priorities.extend(["optimization", "hyperparameters", "evaluation"])

        elif any(pattern in name for pattern in async_patterns):
            domain = "async"
            modifiers.extend(["asynchronous programming", "concurrency", "event loops"])
            priorities.extend(["performance", "debugging", "best practices"])

        elif any(pattern in name for pattern in db_patterns):
            domain = "database"
            modifiers.extend(["database operations", "queries", "connections"])
            priorities.extend(["performance", "transactions", "migrations"])

        elif any(pattern in name for pattern in ui_patterns):
            domain = "ui"
            modifiers.extend(["user interface", "web development", "frontend"])
            priorities.extend(["routing", "templates", "deployment"])

        elif any(pattern in name for pattern in crypto_patterns):
            domain = "security"
            modifiers.extend(["cryptography", "security", "encryption"])
            priorities.extend(["best practices", "vulnerabilities", "implementation"])

        # Add common technical keywords
        keywords.update([name, f"{name} library", f"{name} python"])

        return domain, keywords, modifiers, priorities


class ContextDiversityOptimizer:
    """Optimizes context diversity to minimize overlap and maximize coverage."""

    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize optimizer with similarity threshold.

        Args:
            similarity_threshold: Maximum allowed similarity between contexts (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold

        # Common stop words to ignore in similarity calculations
        self.stop_words = {
            "the",
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
            "from",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "how",
            "what",
            "when",
            "where",
            "why",
        }

    def optimize_contexts(self, contexts: List[str]) -> List[str]:
        """
        Optimize context list for maximum diversity.

        Args:
            contexts: List of context strings to optimize

        Returns:
            Optimized list of diverse contexts
        """
        if len(contexts) <= 1:
            return contexts

        optimized: list[str] = []
        used_keywords: set[str] = set()

        # Sort contexts by estimated information content (longer = more specific)
        sorted_contexts = sorted(contexts, key=len, reverse=True)

        for context in sorted_contexts:
            if self._should_include_context(context, optimized, used_keywords):
                optimized.append(context)
                used_keywords.update(self._extract_keywords(context))

        logger.debug(
            f"Optimized {len(contexts)} contexts to {len(optimized)} diverse contexts"
        )
        return optimized

    def _should_include_context(
        self, context: str, existing: List[str], used_keywords: Set[str]
    ) -> bool:
        """Check if context should be included based on diversity criteria."""
        if not existing:
            return True  # Always include first context

        context_keywords = self._extract_keywords(context)

        # Calculate similarity to existing contexts
        for existing_context in existing:
            existing_keywords = self._extract_keywords(existing_context)
            similarity = self._calculate_similarity(context_keywords, existing_keywords)

            if similarity > self.similarity_threshold:
                logger.debug(
                    f"Rejecting similar context (similarity: {similarity:.2f}): {context[:50]}..."
                )
                return False

        # Check if context adds meaningful new keywords
        new_keywords = context_keywords - used_keywords
        if len(new_keywords) < 2:  # Require at least 2 new meaningful keywords
            logger.debug(
                f"Rejecting context with insufficient new keywords: {context[:50]}..."
            )
            return False

        return True

    def _extract_keywords(self, context: str) -> Set[str]:
        """Extract meaningful keywords from context string."""
        # Convert to lowercase and split into words
        words = re.findall(r"\b\w+\b", context.lower())

        # Filter out stop words and short words
        keywords = {
            word for word in words if len(word) >= 3 and word not in self.stop_words
        }

        return keywords

    def _calculate_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Calculate Jaccard similarity between two keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0

        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)

        return intersection / union if union > 0 else 0.0


class ContextGenerator:
    """
    Generates intelligent search contexts for comprehensive documentation coverage.

    Uses library-specific analysis to create diverse, targeted contexts that maximize
    documentation retrieval while minimizing overlap and redundancy.
    """

    def __init__(self, max_contexts: int = 7, min_contexts: int = 3):
        """
        Initialize context generator.

        Args:
            max_contexts: Maximum number of contexts to generate
            min_contexts: Minimum number of contexts to generate
        """
        self.max_contexts = min(max(max_contexts, 3), 10)  # Clamp to 3-10 range
        self.min_contexts = min(max(min_contexts, 3), self.max_contexts)

        self.diversity_optimizer = ContextDiversityOptimizer()

        # Base context templates for comprehensive coverage
        self.base_templates = [
            "{lib} complete API documentation reference guide",
            "{lib} getting started tutorial installation examples",
            "{lib} advanced configuration options parameters settings",
            "{lib} troubleshooting error handling debugging guide",
            "{lib} best practices code examples patterns",
            "{lib} performance optimization benchmarks tuning",
            "{lib} integration testing deployment production setup",
            "{lib} migration upgrade changelog breaking changes",
            "{lib} security considerations vulnerabilities best practices",
            "{lib} community resources plugins extensions",
        ]

        logger.info(
            f"Initialized context generator with {self.max_contexts} max contexts"
        )

    async def generate_contexts(self, library_id: str) -> List[str]:
        """
        Generate intelligent search contexts for comprehensive coverage.

        Args:
            library_id: Library identifier (e.g., 'encode/httpx')

        Returns:
            List of optimized search contexts

        Raises:
            ContextGenerationError: If context generation fails
        """
        try:
            logger.info(f"Generating contexts for library: {library_id}")

            # Create library profile for customization
            profile = LibraryProfile.from_library_id(library_id)
            logger.debug(
                f"Library profile: domain={profile.domain}, name={profile.name}"
            )

            # Generate base contexts using templates
            base_contexts = self._generate_base_contexts(profile)

            # Add domain-specific contexts
            domain_contexts = self._generate_domain_specific_contexts(profile)

            # Combine and optimize for diversity
            all_contexts = base_contexts + domain_contexts
            optimized_contexts = self.diversity_optimizer.optimize_contexts(
                all_contexts
            )

            # Ensure we have the right number of contexts
            final_contexts = self._finalize_context_list(optimized_contexts, profile)

            logger.info(
                f"Generated {len(final_contexts)} optimized contexts for {library_id}"
            )

            # Log contexts for debugging
            for i, context in enumerate(final_contexts, 1):
                logger.debug(f"Context {i}: {context}")

            return final_contexts

        except Exception as e:
            logger.error(f"Context generation failed for {library_id}: {e}")

            # Fallback to minimal contexts
            fallback_contexts = self._generate_fallback_contexts(library_id)
            logger.warning(f"Using {len(fallback_contexts)} fallback contexts")

            return fallback_contexts

    def _generate_base_contexts(self, profile: LibraryProfile) -> List[str]:
        """Generate base contexts using templates."""
        contexts = []

        for template in self.base_templates[: self.max_contexts]:
            context = template.format(lib=profile.name)

            # Add domain-specific modifiers
            if profile.context_modifiers:
                # Select relevant modifier for this template
                if "getting started" in template:
                    modifier = (
                        profile.context_modifiers[0]
                        if profile.context_modifiers
                        else ""
                    )
                elif "advanced" in template or "configuration" in template:
                    modifier = (
                        profile.context_modifiers[-1]
                        if profile.context_modifiers
                        else ""
                    )
                else:
                    # Use first modifier for other templates
                    modifier = (
                        profile.context_modifiers[0]
                        if profile.context_modifiers
                        else ""
                    )

                if modifier:
                    context += f" {modifier}"

            contexts.append(context)

        return contexts

    def _generate_domain_specific_contexts(self, profile: LibraryProfile) -> List[str]:
        """Generate domain-specific contexts based on library profile."""
        domain_templates = {
            "web": [
                "{lib} HTTP client authentication headers cookies sessions",
                "{lib} API rate limiting retry logic error handling",
                "{lib} async requests concurrent connections performance",
            ],
            "data": [
                "{lib} data manipulation filtering grouping aggregation",
                "{lib} performance optimization memory usage large datasets",
                "{lib} data validation cleaning preprocessing",
            ],
            "testing": [
                "{lib} test fixtures setup teardown mocking",
                "{lib} assertions custom matchers test organization",
                "{lib} coverage reporting integration continuous testing",
            ],
            "machine_learning": [
                "{lib} model training hyperparameter optimization",
                "{lib} data preprocessing feature engineering",
                "{lib} model evaluation metrics validation",
            ],
            "async": [
                "{lib} async await concurrency event loops",
                "{lib} error handling cancellation timeouts",
                "{lib} performance optimization memory management",
            ],
            "database": [
                "{lib} connection pooling transaction management",
                "{lib} query optimization indexing performance",
                "{lib} migrations schema changes database design",
            ],
            "ui": [
                "{lib} routing templates static files",
                "{lib} forms validation user input handling",
                "{lib} deployment configuration production setup",
            ],
            "security": [
                "{lib} encryption decryption key management",
                "{lib} authentication authorization security best practices",
                "{lib} vulnerabilities common pitfalls secure implementation",
            ],
            "general": [
                "{lib} architecture design patterns",
                "{lib} logging debugging troubleshooting",
                "{lib} compatibility version differences",
            ],
        }

        templates = domain_templates.get(profile.domain, domain_templates["general"])
        contexts = []

        for template in templates:
            context = template.format(lib=profile.name)
            contexts.append(context)

        return contexts

    def _finalize_context_list(
        self, contexts: List[str], profile: LibraryProfile
    ) -> List[str]:
        """Finalize context list with proper length and prioritization."""

        # Ensure minimum contexts with fallbacks if needed
        if len(contexts) < self.min_contexts:
            fallback_contexts = self._generate_fallback_contexts(profile.name)
            contexts.extend(fallback_contexts[len(contexts) : self.min_contexts])

        # Limit to maximum contexts
        if len(contexts) > self.max_contexts:
            # Prioritize contexts based on importance
            contexts = self._prioritize_contexts(contexts, profile)[: self.max_contexts]

        return contexts

    def _prioritize_contexts(
        self, contexts: List[str], profile: LibraryProfile
    ) -> List[str]:
        """Prioritize contexts based on library profile and importance."""

        def context_priority_score(context: str) -> float:
            """Calculate priority score for context."""
            score = 0.0

            # Base priority for essential topics
            if any(
                keyword in context.lower()
                for keyword in ["api documentation", "getting started"]
            ):
                score += 10.0
            elif any(
                keyword in context.lower() for keyword in ["examples", "tutorial"]
            ):
                score += 8.0
            elif any(
                keyword in context.lower() for keyword in ["troubleshooting", "error"]
            ):
                score += 7.0
            elif any(
                keyword in context.lower()
                for keyword in ["best practices", "performance"]
            ):
                score += 6.0
            else:
                score += 5.0

            # Boost for priority topics
            for priority_topic in profile.priority_topics:
                if priority_topic.lower() in context.lower():
                    score += 3.0

            # Boost for domain relevance
            domain_keywords = {
                "web": ["http", "api", "request", "client"],
                "data": ["data", "analysis", "processing"],
                "testing": ["test", "mock", "fixture"],
                "async": ["async", "concurrent", "performance"],
                "database": ["query", "connection", "transaction"],
                "security": ["security", "encryption", "auth"],
            }

            relevant_keywords = domain_keywords.get(profile.domain, [])
            for keyword in relevant_keywords:
                if keyword in context.lower():
                    score += 2.0

            return score

        # Sort by priority score descending
        prioritized = sorted(contexts, key=context_priority_score, reverse=True)
        return prioritized

    def _generate_fallback_contexts(self, library_name: str) -> List[str]:
        """Generate minimal fallback contexts when main generation fails."""
        fallback_templates = [
            f"{library_name} documentation",
            f"{library_name} API reference guide",
            f"{library_name} tutorial examples",
            f"{library_name} configuration setup",
            f"how to use {library_name} python",
        ]

        logger.warning(f"Using fallback contexts for {library_name}")
        return fallback_templates

    async def generate_contexts_with_validation(self, library_id: str) -> List[str]:
        """
        Generate contexts with additional validation and quality checks.

        Args:
            library_id: Library identifier

        Returns:
            List of validated, high-quality contexts

        Raises:
            ContextGenerationError: If validation fails
        """
        try:
            # Generate contexts
            contexts = await self.generate_contexts(library_id)

            # Validate context quality
            self._validate_context_quality(contexts, library_id)

            # Additional diversity check
            self._validate_context_diversity(contexts)

            logger.info(
                f"Context validation passed for {library_id}: {len(contexts)} contexts"
            )
            return contexts

        except Exception as e:
            raise ContextGenerationError(
                f"Context generation and validation failed for {library_id}: {e}",
                library_id=library_id,
            ) from e

    def _validate_context_quality(self, contexts: List[str], library_id: str) -> None:
        """Validate that contexts meet quality criteria."""
        if not contexts:
            raise ContextGenerationError(f"No contexts generated for {library_id}")

        if len(contexts) < self.min_contexts:
            raise ContextGenerationError(
                f"Insufficient contexts generated: {len(contexts)} < {self.min_contexts}"
            )

        # Check for empty or too-short contexts
        library_name = library_id.split("/")[-1].lower()
        for i, context in enumerate(contexts):
            if not context or len(context.strip()) < 10:
                raise ContextGenerationError(
                    f"Context {i + 1} is too short or empty: '{context}'"
                )

            if library_name not in context.lower():
                raise ContextGenerationError(
                    f"Context {i + 1} doesn't contain library name '{library_name}': '{context}'"
                )

    def _validate_context_diversity(self, contexts: List[str]) -> None:
        """Validate that contexts have sufficient diversity."""
        if len(contexts) <= 1:
            return  # Cannot check diversity with single context

        # Check for duplicate contexts
        unique_contexts = set(contexts)
        if len(unique_contexts) != len(contexts):
            duplicates = len(contexts) - len(unique_contexts)
            raise ContextGenerationError(f"Found {duplicates} duplicate contexts")

        # Check overall diversity using keywords
        all_keywords = set()
        context_keywords = []

        for context in contexts:
            keywords = self.diversity_optimizer._extract_keywords(context)
            context_keywords.append(keywords)
            all_keywords.update(keywords)

        # Calculate average unique keywords per context
        if all_keywords:
            avg_unique_per_context = len(all_keywords) / len(contexts)
            if (
                avg_unique_per_context < 3
            ):  # Less than 3 unique keywords per context on average
                raise ContextGenerationError(
                    f"Insufficient context diversity: {avg_unique_per_context:.1f} avg unique keywords per context"
                )

        logger.debug(
            f"Context diversity validation passed: {len(all_keywords)} unique keywords across {len(contexts)} contexts"
        )
