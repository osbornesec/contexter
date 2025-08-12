"""
Metadata Extraction and Enrichment - Content analysis and tagging system.

Comprehensive metadata extraction system that analyzes document chunks
to extract and enrich metadata for enhanced search and retrieval.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import hashlib

from .chunking_engine import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysis:
    """
    Comprehensive content analysis results.
    
    Contains detailed analysis of chunk content including
    programming languages, content types, and quality metrics.
    """
    # Language detection
    primary_language: Optional[str]
    language_confidence: float
    detected_languages: Dict[str, float]
    
    # Content classification
    content_type: str  # 'documentation', 'tutorial', 'reference', 'example'
    doc_type: str     # 'api', 'guide', 'tutorial', 'reference'
    
    # Technical analysis
    complexity_score: float  # 0.0-1.0, higher = more complex
    technical_depth: str     # 'beginner', 'intermediate', 'advanced'
    
    # Content features
    has_code_examples: bool
    has_api_references: bool
    has_installation_instructions: bool
    has_configuration_details: bool
    
    # Quality metrics
    readability_score: float  # 0.0-1.0, higher = more readable
    completeness_score: float # 0.0-1.0, higher = more complete
    usefulness_score: float   # 0.0-1.0, higher = more useful
    
    # Extracted entities
    api_endpoints: List[str]
    code_snippets: List[str]
    configuration_keys: List[str]
    mentioned_technologies: List[str]
    
    # Generated tags
    generated_tags: List[str]
    confidence_scores: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            'language_detection': {
                'primary_language': self.primary_language,
                'confidence': self.language_confidence,
                'all_detected': self.detected_languages
            },
            'content_classification': {
                'content_type': self.content_type,
                'doc_type': self.doc_type,
                'complexity_score': self.complexity_score,
                'technical_depth': self.technical_depth
            },
            'features': {
                'has_code_examples': self.has_code_examples,
                'has_api_references': self.has_api_references,
                'has_installation_instructions': self.has_installation_instructions,
                'has_configuration_details': self.has_configuration_details
            },
            'quality_metrics': {
                'readability_score': self.readability_score,
                'completeness_score': self.completeness_score,
                'usefulness_score': self.usefulness_score
            },
            'extracted_entities': {
                'api_endpoints': self.api_endpoints,
                'code_snippets': self.code_snippets[:5],  # Limit for storage
                'configuration_keys': self.configuration_keys,
                'mentioned_technologies': self.mentioned_technologies
            },
            'tags': self.generated_tags,
            'confidence_scores': self.confidence_scores
        }


class ContentAnalyzer:
    """
    Advanced content analyzer for technical documentation.
    
    Performs deep analysis of chunk content to extract technical
    metadata, classify content types, and generate descriptive tags.
    """
    
    def __init__(self):
        """Initialize the content analyzer."""
        
        # Language detection patterns with confidence weights
        self.language_patterns = {
            'python': {
                'import \\w+': 2.0,
                'def \\w+\\(': 2.0,
                'pip install': 1.5,
                'from \\w+ import': 2.0,
                'class \\w+\\(': 1.5,
                '__init__': 2.0,
                'if __name__ == "__main__"': 3.0,
                '\\.py\\b': 1.0,
                'python': 1.0
            },
            'javascript': {
                'function\\s+\\w+': 2.0,
                'const\\s+\\w+\\s*=': 1.5,
                'let\\s+\\w+\\s*=': 1.5,
                'require\\(': 2.0,
                'npm install': 1.5,
                'import.*from': 1.5,
                'export\\s+': 1.5,
                '\\.js\\b': 1.0,
                'node\\.js': 1.5,
                'javascript': 1.0
            },
            'typescript': {
                'interface\\s+\\w+': 2.0,
                'type\\s+\\w+\\s*=': 2.0,
                '\\.ts\\b': 1.5,
                'typescript': 1.5,
                'declare\\s+': 1.5
            },
            'java': {
                'public class': 2.0,
                'import java\\.': 2.0,
                'public static void main': 3.0,
                '@Override': 1.5,
                '\\.java\\b': 1.0,
                'maven': 1.0,
                'gradle': 1.0
            },
            'go': {
                'package \\w+': 2.0,
                'func \\w+': 2.0,
                'import "': 1.5,
                'go get': 1.5,
                '\\.go\\b': 1.0,
                'golang': 1.5
            },
            'rust': {
                'fn \\w+': 2.0,
                'use \\w+::': 2.0,
                'cargo': 1.5,
                'impl\\s+': 1.5,
                '\\.rs\\b': 1.0,
                'rust': 1.0
            },
            'ruby': {
                'def \\w+': 2.0,
                'require ': 1.5,
                'gem install': 1.5,
                'class \\w+': 1.5,
                '\\.rb\\b': 1.0,
                'ruby': 1.0
            },
            'php': {
                '<\\?php': 3.0,
                'namespace ': 2.0,
                'use ': 1.5,
                'composer': 1.5,
                '\\.php\\b': 1.0,
                'php': 1.0
            },
            'c_cpp': {
                '#include': 2.0,
                'int main\\(': 3.0,
                'printf\\(': 1.5,
                '\\.cpp\\b': 1.5,
                '\\.c\\b': 1.5,
                '\\.h\\b': 1.0
            },
            'shell': {
                '#!/bin/bash': 3.0,
                '\\$ ': 1.5,
                'curl ': 1.0,
                'wget ': 1.0,
                'sudo ': 1.0,
                '\\.sh\\b': 1.5
            }
        }
        
        # Content type classification patterns
        self.content_type_patterns = {
            'api_reference': {
                'GET /': 2.0,
                'POST /': 2.0,
                'PUT /': 2.0,
                'DELETE /': 2.0,
                'PATCH /': 2.0,
                'endpoint': 1.5,
                'parameters?:': 1.5,
                'response:': 1.5,
                'status code': 1.5,
                'curl ': 1.0
            },
            'tutorial': {
                'step \\d+': 2.0,
                'tutorial': 2.0,
                'getting started': 2.0,
                "let's": 1.5,
                'first, ': 1.0,
                'next, ': 1.0,
                'finally, ': 1.0,
                'now ': 0.5
            },
            'installation_guide': {
                'install': 2.0,
                'pip install': 2.0,
                'npm install': 2.0,
                'gem install': 2.0,
                'requirements': 1.5,
                'dependencies': 1.5,
                'setup': 1.5,
                'configuration': 1.0
            },
            'configuration': {
                'config': 2.0,
                'settings': 1.5,
                'environment': 1.5,
                '\\.env': 1.5,
                'configuration': 2.0,
                'options': 1.0,
                'parameters': 1.0
            },
            'examples': {
                'example': 2.0,
                'demo': 1.5,
                'sample': 1.5,
                '>>>': 2.0,
                'output:': 1.5,
                'result:': 1.5
            },
            'troubleshooting': {
                'error': 2.0,
                'troubleshoot': 2.0,
                'problem': 1.5,
                'issue': 1.5,
                'fix': 1.5,
                'solution': 1.5,
                'common problems': 2.0
            }
        }
        
        # Technology detection patterns
        self.technology_patterns = {
            # Frameworks
            'django': r'\bdjango\b',
            'flask': r'\bflask\b',
            'fastapi': r'\bfastapi\b',
            'react': r'\breact\b',
            'vue': r'\bvue\b',
            'angular': r'\bangular\b',
            'express': r'\bexpress\b',
            'spring': r'\bspring\b',
            'rails': r'\brails\b',
            
            # Databases
            'postgresql': r'\bpostgres(?:ql)?\b',
            'mysql': r'\bmysql\b',
            'mongodb': r'\bmongodb?\b',
            'redis': r'\bredis\b',
            'sqlite': r'\bsqlite\b',
            'elasticsearch': r'\belasticsearch\b',
            
            # Cloud services
            'aws': r'\baws\b|\bamazon web services\b',
            'gcp': r'\bgcp\b|\bgoogle cloud\b',
            'azure': r'\bazure\b',
            'docker': r'\bdocker\b',
            'kubernetes': r'\bkubernetes\b|\bk8s\b',
            
            # Tools
            'git': r'\bgit\b',
            'nginx': r'\bnginx\b',
            'apache': r'\bapache\b',
            'jenkins': r'\bjenkins\b',
            'terraform': r'\bterraform\b'
        }
        
        # API endpoint patterns
        self.api_patterns = [
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]*)',
            r'/api/v?\d+/[^\s]*',
            r'/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(?:/[a-zA-Z0-9_-]+)*'
        ]
        
        # Configuration key patterns
        self.config_patterns = [
            r'([A-Z_][A-Z0-9_]*)\s*=',  # Environment variables
            r'"([a-zA-Z_][a-zA-Z0-9_.]*)"\s*:',  # JSON config keys
            r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*:',  # YAML config keys
            r'--([a-zA-Z-]+)',  # Command line flags
        ]
        
        logger.info("Content analyzer initialized")
    
    def analyze_content(self, content: str) -> ContentAnalysis:
        """
        Perform comprehensive content analysis.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Detailed content analysis results
        """
        # Language detection
        language_scores = self._detect_languages(content)
        primary_language = max(language_scores, key=language_scores.get) if language_scores else None
        language_confidence = language_scores.get(primary_language, 0.0) if primary_language else 0.0
        
        # Content type classification
        content_type, doc_type = self._classify_content_type(content)
        
        # Technical analysis
        complexity_score = self._calculate_complexity_score(content)
        technical_depth = self._determine_technical_depth(content, complexity_score)
        
        # Feature detection
        features = self._detect_content_features(content)
        
        # Quality metrics
        readability_score = self._calculate_readability_score(content)
        completeness_score = self._calculate_completeness_score(content)
        usefulness_score = self._calculate_usefulness_score(content, features)
        
        # Entity extraction
        api_endpoints = self._extract_api_endpoints(content)
        code_snippets = self._extract_code_snippets(content)
        config_keys = self._extract_configuration_keys(content)
        technologies = self._extract_technologies(content)
        
        # Tag generation
        generated_tags, confidence_scores = self._generate_tags(
            content, primary_language, content_type, features, technologies
        )
        
        return ContentAnalysis(
            primary_language=primary_language,
            language_confidence=language_confidence,
            detected_languages=language_scores,
            content_type=content_type,
            doc_type=doc_type,
            complexity_score=complexity_score,
            technical_depth=technical_depth,
            has_code_examples=features['has_code_examples'],
            has_api_references=features['has_api_references'],
            has_installation_instructions=features['has_installation_instructions'],
            has_configuration_details=features['has_configuration_details'],
            readability_score=readability_score,
            completeness_score=completeness_score,
            usefulness_score=usefulness_score,
            api_endpoints=api_endpoints,
            code_snippets=code_snippets,
            configuration_keys=config_keys,
            mentioned_technologies=technologies,
            generated_tags=generated_tags,
            confidence_scores=confidence_scores
        )
    
    def _detect_languages(self, content: str) -> Dict[str, float]:
        """Detect programming languages with confidence scores."""
        language_scores = {}
        content_lower = content.lower()
        
        for language, patterns in self.language_patterns.items():
            total_score = 0.0
            
            for pattern, weight in patterns.items():
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                total_score += matches * weight
            
            if total_score > 0:
                # Normalize by content length
                normalized_score = total_score / (len(content) / 1000 + 1)
                language_scores[language] = min(1.0, normalized_score)
        
        return language_scores
    
    def _classify_content_type(self, content: str) -> Tuple[str, str]:
        """Classify content type and documentation type."""
        content_lower = content.lower()
        type_scores = {}
        
        for content_type, patterns in self.content_type_patterns.items():
            score = 0.0
            
            for pattern, weight in patterns.items():
                matches = len(re.findall(pattern, content_lower, re.IGNORECASE))
                score += matches * weight
            
            if score > 0:
                type_scores[content_type] = score
        
        # Determine primary content type
        if type_scores:
            primary_type = max(type_scores, key=type_scores.get)
        else:
            primary_type = 'documentation'
        
        # Map to doc type
        doc_type_mapping = {
            'api_reference': 'api',
            'tutorial': 'guide',
            'installation_guide': 'guide',
            'configuration': 'reference',
            'examples': 'tutorial',
            'troubleshooting': 'reference',
            'documentation': 'reference'
        }
        
        doc_type = doc_type_mapping.get(primary_type, 'reference')
        
        return primary_type, doc_type
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate technical complexity score."""
        score = 0.0
        
        # Technical vocabulary density
        technical_words = [
            'algorithm', 'implementation', 'architecture', 'framework',
            'database', 'deployment', 'scalability', 'performance',
            'security', 'authentication', 'authorization', 'middleware',
            'asynchronous', 'synchronous', 'concurrency', 'parallelism'
        ]
        
        word_count = len(content.split())
        if word_count > 0:
            technical_density = sum(
                content.lower().count(word) for word in technical_words
            ) / word_count
            score += min(0.3, technical_density * 10)
        
        # Code complexity indicators
        code_patterns = [
            r'class\s+\w+',      # Classes
            r'def\s+\w+\(',      # Functions
            r'import\s+\w+',     # Imports
            r'if\s+.*:',         # Conditionals
            r'for\s+.*:',        # Loops
            r'try\s*:',          # Exception handling
        ]
        
        code_complexity = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in code_patterns
        )
        
        if code_complexity > 0:
            score += min(0.4, code_complexity / 20)
        
        # API complexity
        api_methods = len(re.findall(r'(?:GET|POST|PUT|DELETE|PATCH)', content))
        if api_methods > 0:
            score += min(0.3, api_methods / 10)
        
        return min(1.0, score)
    
    def _determine_technical_depth(self, content: str, complexity_score: float) -> str:
        """Determine technical depth level."""
        if complexity_score < 0.3:
            return 'beginner'
        elif complexity_score < 0.7:
            return 'intermediate'
        else:
            return 'advanced'
    
    def _detect_content_features(self, content: str) -> Dict[str, bool]:
        """Detect presence of various content features."""
        content_lower = content.lower()
        
        return {
            'has_code_examples': bool(
                re.search(r'```|def \w+|function \w+|class \w+', content)
            ),
            'has_api_references': bool(
                re.search(r'(?:GET|POST|PUT|DELETE|PATCH)\s+/', content)
            ),
            'has_installation_instructions': bool(
                re.search(r'pip install|npm install|gem install|install', content_lower)
            ),
            'has_configuration_details': bool(
                re.search(r'config|settings|environment|\.env', content_lower)
            )
        }
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score."""
        if not content.strip():
            return 0.0
        
        # Simple readability metrics
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        words = content.split()
        avg_sentence_length = len(words) / len(sentences)
        
        # Prefer moderate sentence lengths
        if 5 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif avg_sentence_length < 5:
            length_score = 0.6
        else:
            length_score = max(0.2, 1.0 - (avg_sentence_length - 20) / 50)
        
        # Check for good structure
        structure_score = 0.0
        if re.search(r'\n\n', content):  # Has paragraphs
            structure_score += 0.3
        if re.search(r'^#+ ', content, re.MULTILINE):  # Has headers
            structure_score += 0.3
        if re.search(r'^\* |^\d+\. ', content, re.MULTILINE):  # Has lists
            structure_score += 0.2
        
        # Check for clarity indicators
        clarity_score = 0.0
        clarity_words = ['example', 'for instance', 'such as', 'note that', 'important']
        for word in clarity_words:
            if word in content.lower():
                clarity_score += 0.1
        
        clarity_score = min(0.2, clarity_score)
        
        return min(1.0, (length_score * 0.5) + structure_score + clarity_score)
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate content completeness score."""
        score = 0.0
        
        # Length indicates completeness
        char_count = len(content)
        if char_count > 100:
            score += 0.2
        if char_count > 500:
            score += 0.2
        if char_count > 1000:
            score += 0.1
        
        # Structural completeness
        if re.search(r'introduction|overview', content, re.IGNORECASE):
            score += 0.1
        if re.search(r'example|demo', content, re.IGNORECASE):
            score += 0.1
        if re.search(r'usage|how to', content, re.IGNORECASE):
            score += 0.1
        if re.search(r'parameters?|arguments?', content, re.IGNORECASE):
            score += 0.1
        if re.search(r'returns?|response', content, re.IGNORECASE):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_usefulness_score(self, content: str, features: Dict[str, bool]) -> float:
        """Calculate content usefulness score."""
        score = 0.0
        
        # Features add usefulness
        if features['has_code_examples']:
            score += 0.3
        if features['has_api_references']:
            score += 0.2
        if features['has_installation_instructions']:
            score += 0.2
        if features['has_configuration_details']:
            score += 0.1
        
        # Practical information
        practical_indicators = [
            'step', 'tutorial', 'guide', 'how to', 'example',
            'demo', 'sample', 'usage', 'getting started'
        ]
        
        practical_score = sum(
            0.05 for indicator in practical_indicators
            if indicator in content.lower()
        )
        score += min(0.2, practical_score)
        
        return min(1.0, score)
    
    def _extract_api_endpoints(self, content: str) -> List[str]:
        """Extract API endpoints from content."""
        endpoints = set()
        
        for pattern in self.api_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    endpoints.add(match[0] if match else '')
                else:
                    endpoints.add(match)
        
        # Clean and filter endpoints
        clean_endpoints = []
        for endpoint in endpoints:
            if endpoint and len(endpoint) > 1 and '/' in endpoint:
                clean_endpoints.append(endpoint.strip())
        
        return list(clean_endpoints)[:10]  # Limit to first 10
    
    def _extract_code_snippets(self, content: str) -> List[str]:
        """Extract code snippets from content."""
        snippets = []
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
        snippets.extend([block.strip() for block in code_blocks if block.strip()])
        
        # Extract inline code
        inline_code = re.findall(r'`([^`]+)`', content)
        snippets.extend([code.strip() for code in inline_code if len(code) > 5])
        
        # Limit and clean
        return [snippet for snippet in snippets[:5] if len(snippet) < 500]
    
    def _extract_configuration_keys(self, content: str) -> List[str]:
        """Extract configuration keys and environment variables."""
        config_keys = set()
        
        for pattern in self.config_patterns:
            matches = re.findall(pattern, content)
            config_keys.update(matches)
        
        # Filter out common non-config words
        exclude_words = {'true', 'false', 'null', 'undefined', 'string', 'number'}
        
        return [
            key for key in config_keys 
            if key.lower() not in exclude_words and len(key) > 2
        ][:20]  # Limit to 20
    
    def _extract_technologies(self, content: str) -> List[str]:
        """Extract mentioned technologies and tools."""
        technologies = set()
        content_lower = content.lower()
        
        for tech, pattern in self.technology_patterns.items():
            if re.search(pattern, content_lower):
                technologies.add(tech)
        
        return list(technologies)
    
    def _generate_tags(
        self,
        content: str,
        primary_language: Optional[str],
        content_type: str,
        features: Dict[str, bool],
        technologies: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Generate descriptive tags with confidence scores."""
        tags = []
        confidence_scores = {}
        
        # Language tags
        if primary_language:
            tags.append(primary_language)
            confidence_scores[primary_language] = 0.9
        
        # Content type tags
        if content_type != 'documentation':
            tags.append(content_type.replace('_', '-'))
            confidence_scores[content_type.replace('_', '-')] = 0.8
        
        # Feature tags
        if features['has_code_examples']:
            tags.append('code-examples')
            confidence_scores['code-examples'] = 0.7
        
        if features['has_api_references']:
            tags.append('api-reference')
            confidence_scores['api-reference'] = 0.8
        
        if features['has_installation_instructions']:
            tags.append('installation')
            confidence_scores['installation'] = 0.7
        
        if features['has_configuration_details']:
            tags.append('configuration')
            confidence_scores['configuration'] = 0.6
        
        # Technology tags
        for tech in technologies:
            tags.append(tech)
            confidence_scores[tech] = 0.6
        
        # Content-based tags
        content_lower = content.lower()
        
        if 'beginner' in content_lower or 'getting started' in content_lower:
            tags.append('beginner-friendly')
            confidence_scores['beginner-friendly'] = 0.5
        
        if 'advanced' in content_lower or 'complex' in content_lower:
            tags.append('advanced')
            confidence_scores['advanced'] = 0.5
        
        if 'tutorial' in content_lower:
            tags.append('tutorial')
            confidence_scores['tutorial'] = 0.6
        
        # Remove duplicates and sort by confidence
        unique_tags = list(dict.fromkeys(tags))  # Preserves order
        
        return unique_tags[:15], confidence_scores  # Limit to 15 tags


class MetadataExtractor:
    """
    Comprehensive metadata extraction and enrichment system.
    
    Orchestrates content analysis and metadata enrichment for
    document chunks to enhance search and retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.content_analyzer = ContentAnalyzer()
        
        # Quality scoring weights
        self.quality_weights = {
            'completeness': 0.25,
            'readability': 0.25,
            'usefulness': 0.25,
            'technical_accuracy': 0.25
        }
        
        logger.info("Metadata extractor initialized")
    
    async def enrich_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Enrich document chunks with comprehensive metadata.
        
        Args:
            chunks: List of document chunks to enrich
            
        Returns:
            List of enriched chunks with enhanced metadata
        """
        enriched_chunks = []
        
        for chunk in chunks:
            try:
                enriched_chunk = await self._enrich_single_chunk(chunk)
                enriched_chunks.append(enriched_chunk)
            except Exception as e:
                logger.error(f"Failed to enrich chunk {chunk.chunk_id}: {e}")
                # Return original chunk if enrichment fails
                enriched_chunks.append(chunk)
        
        # Perform collection-level enrichment
        await self._enrich_chunk_relationships(enriched_chunks)
        
        logger.info(f"Enriched {len(enriched_chunks)} chunks with metadata")
        return enriched_chunks
    
    async def _enrich_single_chunk(self, chunk: DocumentChunk) -> DocumentChunk:
        """Enrich a single chunk with content analysis."""
        
        # Perform content analysis
        analysis = self.content_analyzer.analyze_content(chunk.content)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(analysis, chunk)
        
        # Update chunk metadata
        chunk.metadata.update({
            'content_analysis': analysis.to_dict(),
            'quality_score': quality_score,
            'enrichment_timestamp': datetime.now().isoformat(),
            'enrichment_version': '1.0'
        })
        
        # Update chunk properties from analysis
        if analysis.primary_language:
            chunk.programming_language = analysis.primary_language
        
        # Update chunk type if analysis provides better classification
        if analysis.doc_type in ['api', 'guide', 'tutorial', 'reference']:
            chunk.chunk_type = analysis.doc_type
        
        # Update quality scores
        chunk.completeness_score = max(chunk.completeness_score, analysis.completeness_score)
        
        return chunk
    
    def _calculate_overall_quality_score(
        self, 
        analysis: ContentAnalysis, 
        chunk: DocumentChunk
    ) -> float:
        """Calculate overall quality score from analysis."""
        
        scores = {
            'completeness': analysis.completeness_score,
            'readability': analysis.readability_score,
            'usefulness': analysis.usefulness_score,
            'technical_accuracy': self._assess_technical_accuracy(analysis, chunk)
        }
        
        # Weighted average
        overall_score = sum(
            score * self.quality_weights[metric]
            for metric, score in scores.items()
        )
        
        return min(1.0, overall_score)
    
    def _assess_technical_accuracy(
        self, 
        analysis: ContentAnalysis, 
        chunk: DocumentChunk
    ) -> float:
        """Assess technical accuracy based on content consistency."""
        score = 0.5  # Base score
        
        # Language consistency
        if analysis.primary_language and chunk.programming_language:
            if analysis.primary_language == chunk.programming_language:
                score += 0.2
        
        # Content type consistency
        if analysis.content_type and chunk.chunk_type:
            type_mapping = {
                'api_reference': 'api',
                'tutorial': 'example',
                'installation_guide': 'text',
                'configuration': 'text'
            }
            expected_type = type_mapping.get(analysis.content_type, 'text')
            if expected_type == chunk.chunk_type:
                score += 0.1
        
        # Code quality indicators
        if analysis.has_code_examples:
            # Check for complete code blocks
            if '```' in chunk.content and chunk.content.count('```') % 2 == 0:
                score += 0.1
            
            # Check for proper indentation
            lines = chunk.content.split('\n')
            indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
            if indented_lines > 0:
                score += 0.1
        
        return min(1.0, score)
    
    async def _enrich_chunk_relationships(self, chunks: List[DocumentChunk]) -> None:
        """Enrich chunks with relationship metadata."""
        
        # Group chunks by library and version
        library_chunks = {}
        for chunk in chunks:
            key = f"{chunk.library_id}_{chunk.version}"
            if key not in library_chunks:
                library_chunks[key] = []
            library_chunks[key].append(chunk)
        
        # Add collection-level metadata
        for library_key, lib_chunks in library_chunks.items():
            await self._add_collection_metadata(lib_chunks)
    
    async def _add_collection_metadata(self, chunks: List[DocumentChunk]) -> None:
        """Add collection-level metadata to chunks."""
        
        if not chunks:
            return
        
        # Calculate collection statistics
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_quality = sum(
            chunk.metadata.get('quality_score', 0.5) for chunk in chunks
        ) / len(chunks)
        
        # Analyze content distribution
        languages = [chunk.programming_language for chunk in chunks if chunk.programming_language]
        language_distribution = dict(Counter(languages))
        
        chunk_types = [chunk.chunk_type for chunk in chunks]
        type_distribution = dict(Counter(chunk_types))
        
        # Extract common technologies across chunks
        all_technologies = []
        for chunk in chunks:
            analysis = chunk.metadata.get('content_analysis', {})
            technologies = analysis.get('extracted_entities', {}).get('mentioned_technologies', [])
            all_technologies.extend(technologies)
        
        common_technologies = [tech for tech, count in Counter(all_technologies).items() if count > 1]
        
        # Add collection metadata to each chunk
        collection_metadata = {
            'collection_stats': {
                'total_chunks': len(chunks),
                'total_tokens': total_tokens,
                'avg_quality_score': avg_quality,
                'language_distribution': language_distribution,
                'type_distribution': type_distribution,
                'common_technologies': common_technologies
            }
        }
        
        for chunk in chunks:
            chunk.metadata.update(collection_metadata)
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get metadata extraction statistics."""
        return {
            'extractor_version': '1.0',
            'content_analyzer_version': '1.0',
            'supported_languages': list(self.content_analyzer.language_patterns.keys()),
            'supported_content_types': list(self.content_analyzer.content_type_patterns.keys()),
            'supported_technologies': list(self.content_analyzer.technology_patterns.keys()),
            'quality_weights': self.quality_weights
        }