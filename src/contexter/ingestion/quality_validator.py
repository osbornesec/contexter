"""
Quality Validator - Document quality assessment system.

Evaluates documentation quality using multiple metrics to determine
if documents meet ingestion thresholds and priority scoring.
"""

import json
import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class QualityAssessment:
    """
    Comprehensive quality assessment results.
    
    Contains detailed scoring across multiple quality dimensions
    with specific feedback for improvement opportunities.
    """
    overall_score: float  # 0.0-1.0 overall quality score
    
    # Dimension scores (0.0-1.0 each)
    completeness_score: float
    structure_score: float
    content_quality_score: float
    metadata_quality_score: float
    format_consistency_score: float
    
    # Detailed metrics
    total_sections: int
    populated_sections: int
    total_content_length: int
    code_examples_count: int
    api_references_count: int
    
    # Quality indicators
    has_installation_guide: bool
    has_api_documentation: bool
    has_examples: bool
    has_version_info: bool
    language_detected: Optional[str]
    
    # Issues and recommendations
    issues: List[str]
    recommendations: List[str]
    
    # Processing metadata
    assessment_time: datetime
    processing_duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary for storage/serialization."""
        return {
            'overall_score': self.overall_score,
            'scores': {
                'completeness': self.completeness_score,
                'structure': self.structure_score,
                'content_quality': self.content_quality_score,
                'metadata_quality': self.metadata_quality_score,
                'format_consistency': self.format_consistency_score
            },
            'metrics': {
                'total_sections': self.total_sections,
                'populated_sections': self.populated_sections,
                'total_content_length': self.total_content_length,
                'code_examples_count': self.code_examples_count,
                'api_references_count': self.api_references_count
            },
            'indicators': {
                'has_installation_guide': self.has_installation_guide,
                'has_api_documentation': self.has_api_documentation,
                'has_examples': self.has_examples,
                'has_version_info': self.has_version_info,
                'language_detected': self.language_detected
            },
            'issues': self.issues,
            'recommendations': self.recommendations,
            'metadata': {
                'assessment_time': self.assessment_time.isoformat(),
                'processing_duration': self.processing_duration
            }
        }


class QualityValidator:
    """
    Document quality validator with comprehensive assessment.
    
    Evaluates documentation across multiple dimensions:
    - Completeness: How much content is present
    - Structure: Organization and formatting quality
    - Content Quality: Usefulness and clarity of content
    - Metadata Quality: Completeness of library metadata
    - Format Consistency: Adherence to expected schemas
    """
    
    def __init__(self):
        """Initialize the quality validator."""
        
        # Quality scoring weights (must sum to 1.0)
        self.weights = {
            'completeness': 0.25,      # 25% - How complete is the documentation
            'structure': 0.20,         # 20% - How well organized
            'content_quality': 0.25,   # 25% - Quality of actual content
            'metadata_quality': 0.15,  # 15% - Library metadata completeness
            'format_consistency': 0.15 # 15% - Schema adherence
        }
        
        # Expected sections for different doc types
        self.expected_sections = {
            'standard': [
                'installation', 'getting_started', 'api_reference', 
                'examples', 'documentation', 'readme'
            ],
            'api_library': [
                'api_reference', 'authentication', 'endpoints',
                'examples', 'quickstart', 'installation'
            ],
            'framework': [
                'installation', 'tutorial', 'guides', 'api',
                'examples', 'configuration', 'deployment'
            ]
        }
        
        # Content quality patterns
        self.quality_patterns = {
            'code_examples': [
                r'```[\w]*\n.*?\n```',
                r'<code>.*?</code>',
                r'`[^`]+`',
                r'>>> ',
                r'$ '
            ],
            'api_references': [
                r'def \w+\(',
                r'function \w+\(',
                r'class \w+',
                r'@[\w.]+',
                r'GET|POST|PUT|DELETE|PATCH',
                r'/api/v?\d+/',
                r'\.endpoint\(',
                r'\.route\('
            ],
            'installation_indicators': [
                r'pip install',
                r'npm install',
                r'yarn add',
                r'gem install',
                r'cargo install',
                r'go get',
                r'composer require'
            ],
            'version_indicators': [
                r'v?\d+\.\d+\.\d+',
                r'version',
                r'release',
                r'changelog'
            ]
        }
        
        # Language detection patterns
        self.language_patterns = {
            'python': [r'import \w+', r'from \w+ import', r'def \w+\(', r'\.py\b', r'pip install'],
            'javascript': [r'require\(', r'import.*from', r'function\s+\w+', r'\.js\b', r'npm install'],
            'java': [r'import java\.', r'public class', r'\.java\b', r'maven', r'gradle'],
            'go': [r'package \w+', r'import "', r'func \w+', r'\.go\b', r'go get'],
            'rust': [r'use \w+::', r'fn \w+', r'\.rs\b', r'cargo', r'crates\.io'],
            'ruby': [r'require ', r'class \w+', r'def \w+', r'\.rb\b', r'gem install'],
            'php': [r'<\?php', r'namespace ', r'use ', r'\.php\b', r'composer'],
            'c_cpp': [r'#include', r'int main\(', r'\.h\b', r'\.cpp\b', r'\.c\b']
        }
        
        logger.info("Quality validator initialized")
    
    async def assess_document_quality(
        self, 
        doc_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Assess document quality and return overall score.
        
        Args:
            doc_path: Path to documentation file
            metadata: Optional metadata for enhanced assessment
            
        Returns:
            Overall quality score (0.0-1.0)
        """
        assessment = await self.assess_document_comprehensive(doc_path, metadata)
        return assessment.overall_score
    
    async def assess_document_comprehensive(
        self, 
        doc_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """
        Perform comprehensive document quality assessment.
        
        Args:
            doc_path: Path to documentation file
            metadata: Optional metadata for enhanced assessment
            
        Returns:
            Detailed quality assessment
        """
        start_time = asyncio.get_event_loop().time()
        assessment_timestamp = datetime.now()
        
        try:
            # Load document content
            doc_content = await self._load_document(doc_path)
            
            # Perform quality assessments
            completeness_score = await self._assess_completeness(doc_content, metadata)
            structure_score = await self._assess_structure(doc_content)
            content_quality_score = await self._assess_content_quality(doc_content)
            metadata_quality_score = await self._assess_metadata_quality(doc_content, metadata)
            format_consistency_score = await self._assess_format_consistency(doc_content)
            
            # Calculate overall score
            overall_score = (
                completeness_score * self.weights['completeness'] +
                structure_score * self.weights['structure'] +
                content_quality_score * self.weights['content_quality'] +
                metadata_quality_score * self.weights['metadata_quality'] +
                format_consistency_score * self.weights['format_consistency']
            )
            
            # Extract detailed metrics
            metrics = await self._extract_metrics(doc_content)
            quality_indicators = await self._extract_quality_indicators(doc_content)
            issues, recommendations = await self._identify_issues_and_recommendations(
                doc_content, 
                {
                    'completeness': completeness_score,
                    'structure': structure_score,
                    'content_quality': content_quality_score,
                    'metadata_quality': metadata_quality_score,
                    'format_consistency': format_consistency_score
                }
            )
            
            # Create assessment
            assessment = QualityAssessment(
                overall_score=overall_score,
                completeness_score=completeness_score,
                structure_score=structure_score,
                content_quality_score=content_quality_score,
                metadata_quality_score=metadata_quality_score,
                format_consistency_score=format_consistency_score,
                **metrics,
                **quality_indicators,
                issues=issues,
                recommendations=recommendations,
                assessment_time=assessment_timestamp,
                processing_duration=asyncio.get_event_loop().time() - start_time
            )
            
            logger.debug(
                f"Quality assessment complete: {assessment.overall_score:.2f} "
                f"({assessment.processing_duration:.2f}s)"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Quality assessment failed for {doc_path}: {e}")
            
            # Return minimal assessment on error
            return QualityAssessment(
                overall_score=0.0,
                completeness_score=0.0,
                structure_score=0.0,
                content_quality_score=0.0,
                metadata_quality_score=0.0,
                format_consistency_score=0.0,
                total_sections=0,
                populated_sections=0,
                total_content_length=0,
                code_examples_count=0,
                api_references_count=0,
                has_installation_guide=False,
                has_api_documentation=False,
                has_examples=False,
                has_version_info=False,
                language_detected=None,
                issues=[f"Assessment failed: {str(e)}"],
                recommendations=["Fix document loading or format issues"],
                assessment_time=assessment_timestamp,
                processing_duration=asyncio.get_event_loop().time() - start_time
            )
    
    async def _load_document(self, doc_path: Path) -> Dict[str, Any]:
        """Load and parse document content."""
        try:
            if not doc_path.exists():
                raise FileNotFoundError(f"Document not found: {doc_path}")
            
            # Handle compressed files
            if doc_path.suffix == '.gz':
                with gzip.open(doc_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Parse JSON
            if content.strip().startswith('{'):
                return json.loads(content)
            else:
                # Handle plain text by wrapping in a basic structure
                return {
                    'content': content,
                    'metadata': {},
                    'format': 'text'
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {doc_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load document {doc_path}: {e}")
            raise
    
    async def _assess_completeness(
        self, 
        doc_content: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Assess documentation completeness."""
        score = 0.0
        
        # Count populated sections
        total_sections = 0
        populated_sections = 0
        
        # Check common section patterns
        for section_name in self.expected_sections['standard']:
            total_sections += 1
            
            # Look for section in various forms
            section_found = False
            for key, value in doc_content.items():
                if (section_name.lower() in key.lower() and 
                    value and 
                    (isinstance(value, str) and len(value.strip()) > 10 or
                     isinstance(value, dict) and value)):
                    section_found = True
                    break
            
            if section_found:
                populated_sections += 1
        
        # Base completeness score
        if total_sections > 0:
            score = populated_sections / total_sections
        
        # Bonus for additional content types
        if self._has_code_examples(doc_content):
            score += 0.1
        
        if self._has_api_documentation(doc_content):
            score += 0.1
        
        if self._has_installation_guide(doc_content):
            score += 0.05
        
        return min(1.0, score)
    
    async def _assess_structure(self, doc_content: Dict[str, Any]) -> float:
        """Assess documentation structure and organization."""
        score = 0.0
        
        # Check for hierarchical structure
        if isinstance(doc_content, dict):
            score += 0.3  # Basic dict structure
            
            # Look for nested organization
            nested_sections = 0
            for key, value in doc_content.items():
                if isinstance(value, dict) and len(value) > 1:
                    nested_sections += 1
            
            if nested_sections > 0:
                score += 0.2  # Has nested sections
            
            # Check for consistent key naming
            keys = list(doc_content.keys())
            if len(keys) > 2:
                # Look for consistent naming patterns
                snake_case = sum(1 for k in keys if '_' in k and k.islower())
                camel_case = sum(1 for k in keys if any(c.isupper() for c in k[1:]))
                
                consistency_ratio = max(snake_case, camel_case) / len(keys)
                score += consistency_ratio * 0.2
        
        # Check for logical content ordering
        content_text = json.dumps(doc_content).lower()
        
        # Installation before usage
        if 'install' in content_text and 'usage' in content_text:
            install_pos = content_text.find('install')
            usage_pos = content_text.find('usage')
            if install_pos < usage_pos:
                score += 0.1
        
        # API reference structure
        if 'api' in content_text or 'reference' in content_text:
            score += 0.1
        
        # Examples after explanations
        if 'example' in content_text:
            score += 0.1
        
        return min(1.0, score)
    
    async def _assess_content_quality(self, doc_content: Dict[str, Any]) -> float:
        """Assess quality of actual content."""
        score = 0.0
        content_text = json.dumps(doc_content).lower()
        
        # Content length assessment
        content_length = len(content_text)
        if content_length > 1000:
            score += 0.2
        if content_length > 5000:
            score += 0.1
        if content_length > 10000:
            score += 0.1
        
        # Code examples quality
        code_examples = self._count_code_examples(content_text)
        if code_examples > 0:
            score += 0.2
        if code_examples > 5:
            score += 0.1
        
        # API references
        api_refs = self._count_api_references(content_text)
        if api_refs > 0:
            score += 0.2
        if api_refs > 10:
            score += 0.1
        
        # Language-specific indicators
        language_detected = self._detect_programming_language(content_text)
        if language_detected:
            score += 0.1
        
        return min(1.0, score)
    
    async def _assess_metadata_quality(
        self, 
        doc_content: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Assess metadata completeness and quality."""
        score = 0.0
        
        # Check document metadata
        doc_metadata = doc_content.get('metadata', {})
        
        required_fields = ['name', 'version', 'description']
        for field in required_fields:
            if field in doc_metadata and doc_metadata[field]:
                score += 0.2
        
        # Check external metadata
        if metadata:
            external_fields = ['library_id', 'star_count', 'trust_score', 'category']
            for field in external_fields:
                if field in metadata and metadata[field] is not None:
                    score += 0.1
        
        return min(1.0, score)
    
    async def _assess_format_consistency(self, doc_content: Dict[str, Any]) -> float:
        """Assess format consistency and schema adherence."""
        score = 0.5  # Base score for valid JSON
        
        # Check for consistent section naming
        if isinstance(doc_content, dict):
            keys = list(doc_content.keys())
            
            # Consistent key casing
            if len(keys) > 1:
                lower_keys = sum(1 for k in keys if k.islower())
                consistency = lower_keys / len(keys)
                score += consistency * 0.2
            
            # Standard sections present
            standard_sections = ['metadata', 'content', 'documentation']
            found_standard = sum(1 for section in standard_sections if section in keys)
            score += (found_standard / len(standard_sections)) * 0.3
        
        return min(1.0, score)
    
    async def _extract_metrics(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed content metrics."""
        content_text = json.dumps(doc_content)
        
        return {
            'total_sections': len(doc_content) if isinstance(doc_content, dict) else 1,
            'populated_sections': sum(
                1 for v in doc_content.values() 
                if v and (isinstance(v, str) and len(v.strip()) > 10 or isinstance(v, dict) and v)
            ) if isinstance(doc_content, dict) else 1,
            'total_content_length': len(content_text),
            'code_examples_count': self._count_code_examples(content_text),
            'api_references_count': self._count_api_references(content_text)
        }
    
    async def _extract_quality_indicators(self, doc_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality indicators."""
        content_text = json.dumps(doc_content).lower()
        
        return {
            'has_installation_guide': self._has_installation_guide(doc_content),
            'has_api_documentation': self._has_api_documentation(doc_content),
            'has_examples': self._has_code_examples(doc_content),
            'has_version_info': self._has_version_info(content_text),
            'language_detected': self._detect_programming_language(content_text)
        }
    
    async def _identify_issues_and_recommendations(
        self, 
        doc_content: Dict[str, Any], 
        scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """Identify issues and generate recommendations."""
        issues = []
        recommendations = []
        
        # Completeness issues
        if scores['completeness'] < 0.5:
            issues.append("Documentation appears incomplete")
            recommendations.append("Add missing sections: installation, examples, API reference")
        
        # Structure issues
        if scores['structure'] < 0.6:
            issues.append("Poor organization and structure")
            recommendations.append("Improve section organization and naming consistency")
        
        # Content quality issues
        if scores['content_quality'] < 0.5:
            issues.append("Limited code examples and API references")
            recommendations.append("Add more code examples and detailed API documentation")
        
        # Format issues
        if scores['format_consistency'] < 0.7:
            issues.append("Inconsistent formatting or schema")
            recommendations.append("Standardize section naming and structure")
        
        # Specific content checks
        content_text = json.dumps(doc_content).lower()
        
        if not self._has_installation_guide(doc_content):
            issues.append("Missing installation instructions")
            recommendations.append("Add clear installation guide")
        
        if self._count_code_examples(content_text) < 3:
            issues.append("Insufficient code examples")
            recommendations.append("Include more practical code examples")
        
        return issues, recommendations
    
    def _has_installation_guide(self, doc_content: Dict[str, Any]) -> bool:
        """Check if document has installation guide."""
        content_text = json.dumps(doc_content).lower()
        return any(
            re.search(pattern, content_text) 
            for pattern in self.quality_patterns['installation_indicators']
        )
    
    def _has_api_documentation(self, doc_content: Dict[str, Any]) -> bool:
        """Check if document has API documentation."""
        content_text = json.dumps(doc_content).lower()
        return ('api' in content_text and len(content_text) > 1000) or \
               self._count_api_references(content_text) > 5
    
    def _has_code_examples(self, doc_content: Dict[str, Any]) -> bool:
        """Check if document has code examples."""
        content_text = json.dumps(doc_content)
        return self._count_code_examples(content_text) > 0
    
    def _has_version_info(self, content_text: str) -> bool:
        """Check if document has version information."""
        return any(
            re.search(pattern, content_text, re.IGNORECASE) 
            for pattern in self.quality_patterns['version_indicators']
        )
    
    def _count_code_examples(self, content_text: str) -> int:
        """Count code examples in content."""
        count = 0
        for pattern in self.quality_patterns['code_examples']:
            matches = re.findall(pattern, content_text, re.DOTALL | re.MULTILINE)
            count += len(matches)
        return count
    
    def _count_api_references(self, content_text: str) -> int:
        """Count API references in content."""
        count = 0
        for pattern in self.quality_patterns['api_references']:
            matches = re.findall(pattern, content_text, re.IGNORECASE)
            count += len(matches)
        return count
    
    def _detect_programming_language(self, content_text: str) -> Optional[str]:
        """Detect primary programming language from content."""
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_text, re.IGNORECASE)
                score += len(matches)
            
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return None