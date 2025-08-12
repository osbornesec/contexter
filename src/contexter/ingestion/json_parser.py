"""
JSON Document Parser - Adaptive parsing system for documentation.

Robust JSON parser that handles various document schemas from C7DocDownloader
and other sources with comprehensive error recovery and content extraction.
"""

import json
import gzip
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentParsingError(Exception):
    """Raised when document parsing fails."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class SchemaType(Enum):
    """Detected document schema types."""
    STANDARD_LIBRARY = "standard_library"
    CONTEXT7_OUTPUT = "context7_output"
    GENERIC_DOCUMENTATION = "generic_documentation"
    RAW_TEXT = "raw_text"
    UNKNOWN = "unknown"


@dataclass
class ParsedSection:
    """
    Parsed document section with extracted content and metadata.
    
    Represents a logical section of documentation that can be
    processed independently through the ingestion pipeline.
    """
    content: str
    metadata: Dict[str, Any]
    section_id: str
    section_type: str  # 'text', 'code', 'api', 'example'
    hierarchy_level: int = 0
    parent_section: Optional[str] = None
    content_hash: Optional[str] = None
    
    def __post_init__(self):
        """Generate content hash if not provided."""
        if not self.content_hash:
            import hashlib
            self.content_hash = hashlib.sha256(
                self.content.encode('utf-8')
            ).hexdigest()[:16]


class JSONDocumentParser:
    """
    Adaptive JSON document parser with schema detection.
    
    Features:
    - Automatic schema detection and adaptive parsing
    - Support for compressed (gzip) and uncompressed JSON
    - Hierarchical content extraction with metadata preservation
    - Error recovery for malformed JSON with detailed reporting
    - Content type detection (text, code, API, examples)
    - Programming language detection and categorization
    """
    
    def __init__(self):
        """Initialize the JSON document parser."""
        
        # Schema detection patterns
        self.schema_patterns = {
            SchemaType.CONTEXT7_OUTPUT: [
                'contexts',
                'library_info',
                'documentation_sections',
                'total_tokens'
            ],
            SchemaType.STANDARD_LIBRARY: [
                'name',
                'version',
                'description',
                'installation',
                'api_reference'
            ]
        }
        
        # Content type detection patterns
        self.content_type_patterns = {
            'code': [
                r'```[\w]*\n.*?\n```',
                r'<code>.*?</code>',
                r'^\s*(?:def|class|function|var|let|const|import|from)\s',
                r'^\s*[a-zA-Z_]\w*\s*\(',
                r'^\s*[{}]\s*$'
            ],
            'api': [
                r'(?:GET|POST|PUT|DELETE|PATCH)\s+/',
                r'/api/v?\d+/',
                r'\.endpoint\(',
                r'\.route\(',
                r'@app\.',
                r'@[a-zA-Z_]\w*\(',
                r'def\s+\w+\([^)]*\)\s*:'
            ],
            'example': [
                r'(?:example|demo|sample)',
                r'>>> ',
                r'$ ',
                r'console\.log',
                r'print\(',
                r'Example:',
                r'Usage:'
            ]
        }
        
        # Section hierarchy keywords
        self.hierarchy_keywords = {
            0: ['installation', 'getting_started', 'introduction', 'overview'],
            1: ['usage', 'examples', 'tutorial', 'guide'],
            2: ['api', 'reference', 'documentation'],
            3: ['advanced', 'configuration', 'deployment'],
            4: ['troubleshooting', 'faq', 'changelog']
        }
        
        # Language detection patterns
        self.language_patterns = {
            'python': [r'import \w+', r'def \w+\(', r'pip install', r'\.py\b'],
            'javascript': [r'require\(', r'function\s+\w+', r'npm install', r'\.js\b'],
            'java': [r'import java\.', r'public class', r'maven', r'\.java\b'],
            'go': [r'package \w+', r'func \w+', r'go get', r'\.go\b'],
            'rust': [r'use \w+::', r'fn \w+', r'cargo', r'\.rs\b'],
            'ruby': [r'require ', r'def \w+', r'gem install', r'\.rb\b'],
            'php': [r'<\?php', r'namespace ', r'composer', r'\.php\b'],
            'typescript': [r'interface \w+', r'type \w+', r'\.ts\b']
        }
        
        logger.info("JSON document parser initialized")
    
    async def parse_document(self, doc_path: Path) -> List[ParsedSection]:
        """
        Parse JSON document and extract sections.
        
        Args:
            doc_path: Path to documentation file
            
        Returns:
            List of parsed document sections
            
        Raises:
            DocumentParsingError: If parsing fails
        """
        try:
            # Load and parse JSON content
            content = await self._load_json_content(doc_path)
            
            # Detect schema type
            schema_type = self._detect_schema(content)
            
            # Parse based on detected schema
            if schema_type == SchemaType.CONTEXT7_OUTPUT:
                sections = await self._parse_context7_schema(content)
            elif schema_type == SchemaType.STANDARD_LIBRARY:
                sections = await self._parse_standard_schema(content)
            elif schema_type == SchemaType.GENERIC_DOCUMENTATION:
                sections = await self._parse_generic_schema(content)
            else:
                sections = await self._parse_unknown_schema(content)
            
            # Enrich sections with additional metadata
            enriched_sections = await self._enrich_sections(sections, doc_path)
            
            logger.info(
                f"Parsed {len(enriched_sections)} sections from {doc_path} "
                f"(schema: {schema_type.value})"
            )
            
            return enriched_sections
            
        except DocumentParsingError:
            raise
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to parse document {doc_path}: {str(e)}", 
                original_error=e
            )
    
    async def _load_json_content(self, doc_path: Path) -> Dict[str, Any]:
        """Load and parse JSON content with compression support."""
        try:
            if not doc_path.exists():
                raise DocumentParsingError(f"Document not found: {doc_path}")
            
            # Handle compressed files
            if doc_path.suffix == '.gz' or doc_path.name.endswith('.json.gz'):
                with gzip.open(doc_path, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Validate content
            if not content.strip():
                raise DocumentParsingError("Empty document")
            
            # Parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                # Try to recover from common JSON issues
                recovered_content = self._attempt_json_recovery(content)
                if recovered_content:
                    return json.loads(recovered_content)
                else:
                    raise DocumentParsingError(
                        f"Invalid JSON format: {str(e)}", 
                        original_error=e
                    )
                    
        except DocumentParsingError:
            raise
        except Exception as e:
            raise DocumentParsingError(
                f"Failed to load document: {str(e)}", 
                original_error=e
            )
    
    def _attempt_json_recovery(self, content: str) -> Optional[str]:
        """Attempt to recover from common JSON formatting issues."""
        try:
            # Remove common trailing issues
            content = content.strip()
            
            # Remove trailing commas
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            # Fix unescaped quotes in strings (basic attempt)
            content = re.sub(r'(?<!\\)"(?=.*".*:)', r'\\"', content)
            
            # Validate recovered JSON
            json.loads(content)
            return content
            
        except Exception:
            return None
    
    def _detect_schema(self, content: Dict[str, Any]) -> SchemaType:
        """Detect document schema type based on structure and keys."""
        if not isinstance(content, dict):
            return SchemaType.RAW_TEXT
        
        content_keys = set(content.keys())
        
        # Check for Context7 output schema
        context7_indicators = set(self.schema_patterns[SchemaType.CONTEXT7_OUTPUT])
        if len(content_keys.intersection(context7_indicators)) >= 2:
            return SchemaType.CONTEXT7_OUTPUT
        
        # Check for standard library schema
        standard_indicators = set(self.schema_patterns[SchemaType.STANDARD_LIBRARY])
        if len(content_keys.intersection(standard_indicators)) >= 3:
            return SchemaType.STANDARD_LIBRARY
        
        # Check for generic documentation structure
        common_doc_keys = {
            'readme', 'documentation', 'guide', 'examples', 
            'tutorial', 'reference', 'api', 'install'
        }
        if any(key.lower() in common_doc_keys for key in content_keys):
            return SchemaType.GENERIC_DOCUMENTATION
        
        return SchemaType.UNKNOWN
    
    async def _parse_context7_schema(self, content: Dict[str, Any]) -> List[ParsedSection]:
        """Parse Context7 output schema format."""
        sections = []
        
        # Extract library metadata
        library_info = content.get('library_info', {})
        base_metadata = {
            'library_id': library_info.get('library_id', ''),
            'name': library_info.get('name', ''),
            'version': library_info.get('version', ''),
            'category': library_info.get('category', ''),
            'trust_score': library_info.get('trust_score', 0.0),
            'schema_type': SchemaType.CONTEXT7_OUTPUT.value
        }
        
        # Parse contexts sections
        contexts = content.get('contexts', [])
        for i, context in enumerate(contexts):
            if isinstance(context, dict) and 'content' in context:
                section_metadata = {
                    **base_metadata,
                    'context_index': i,
                    'source': context.get('source', ''),
                    'url': context.get('url', ''),
                    'token_count': context.get('token_count', 0)
                }
                
                # Determine section type and hierarchy
                section_type = self._determine_content_type(context['content'])
                hierarchy_level = self._determine_hierarchy_level(
                    context.get('source', ''), context['content']
                )
                
                section = ParsedSection(
                    content=context['content'],
                    metadata=section_metadata,
                    section_id=f"context_{i}",
                    section_type=section_type,
                    hierarchy_level=hierarchy_level
                )
                
                sections.append(section)
        
        # Parse documentation sections if present
        doc_sections = content.get('documentation_sections', {})
        for section_name, section_content in doc_sections.items():
            if isinstance(section_content, str) and section_content.strip():
                section_metadata = {
                    **base_metadata,
                    'section_name': section_name,
                    'section_source': 'documentation_sections'
                }
                
                section_type = self._determine_content_type(section_content)
                hierarchy_level = self._determine_hierarchy_level(section_name, section_content)
                
                section = ParsedSection(
                    content=section_content,
                    metadata=section_metadata,
                    section_id=f"doc_{section_name}",
                    section_type=section_type,
                    hierarchy_level=hierarchy_level
                )
                
                sections.append(section)
        
        return sections
    
    async def _parse_standard_schema(self, content: Dict[str, Any]) -> List[ParsedSection]:
        """Parse standard library documentation schema."""
        sections = []
        
        # Extract base metadata
        metadata_section = content.get('metadata', {})
        base_metadata = {
            'library_id': metadata_section.get('name', ''),
            'name': metadata_section.get('name', ''),
            'version': metadata_section.get('version', ''),
            'description': metadata_section.get('description', ''),
            'schema_type': SchemaType.STANDARD_LIBRARY.value
        }
        
        # Process each top-level section
        for section_name, section_content in content.items():
            if section_name == 'metadata':
                continue
            
            if isinstance(section_content, str) and section_content.strip():
                # Simple string content
                section = self._create_section_from_content(
                    section_content, section_name, base_metadata
                )
                sections.append(section)
                
            elif isinstance(section_content, dict):
                # Nested section content
                parent_section_id = f"section_{section_name}"
                
                for subsection_name, subsection_content in section_content.items():
                    if isinstance(subsection_content, str) and subsection_content.strip():
                        subsection_metadata = {
                            **base_metadata,
                            'section': section_name,
                            'subsection': subsection_name
                        }
                        
                        section = self._create_section_from_content(
                            subsection_content,
                            f"{section_name}.{subsection_name}",
                            subsection_metadata,
                            parent_section=parent_section_id
                        )
                        sections.append(section)
        
        return sections
    
    async def _parse_generic_schema(self, content: Dict[str, Any]) -> List[ParsedSection]:
        """Parse generic documentation format."""
        sections = []
        
        # Extract any available metadata
        base_metadata = {
            'schema_type': SchemaType.GENERIC_DOCUMENTATION.value
        }
        
        # Look for common metadata fields
        for key in ['name', 'title', 'version', 'description']:
            if key in content and isinstance(content[key], str):
                base_metadata[key] = content[key]
        
        # Process all content sections
        for section_name, section_content in content.items():
            if self._is_metadata_field(section_name):
                continue
            
            if isinstance(section_content, str) and section_content.strip():
                section = self._create_section_from_content(
                    section_content, section_name, base_metadata
                )
                sections.append(section)
                
            elif isinstance(section_content, dict):
                # Handle nested content
                for subsection_name, subsection_content in section_content.items():
                    if isinstance(subsection_content, str) and subsection_content.strip():
                        subsection_metadata = {
                            **base_metadata,
                            'section': section_name,
                            'subsection': subsection_name
                        }
                        
                        section = self._create_section_from_content(
                            subsection_content,
                            f"{section_name}.{subsection_name}",
                            subsection_metadata
                        )
                        sections.append(section)
            
            elif isinstance(section_content, list):
                # Handle list content
                for i, item in enumerate(section_content):
                    if isinstance(item, str) and item.strip():
                        item_metadata = {
                            **base_metadata,
                            'section': section_name,
                            'item_index': i
                        }
                        
                        section = self._create_section_from_content(
                            item,
                            f"{section_name}[{i}]",
                            item_metadata
                        )
                        sections.append(section)
        
        return sections
    
    async def _parse_unknown_schema(self, content: Dict[str, Any]) -> List[ParsedSection]:
        """Parse unknown or non-standard schema with best effort."""
        sections = []
        
        base_metadata = {
            'schema_type': SchemaType.UNKNOWN.value
        }
        
        # Try to extract any textual content
        text_content = self._extract_text_content(content)
        
        if text_content:
            for i, (key, text) in enumerate(text_content):
                section_metadata = {
                    **base_metadata,
                    'extracted_from': key,
                    'extraction_method': 'text_search'
                }
                
                section = self._create_section_from_content(
                    text, f"extracted_{i}", section_metadata
                )
                sections.append(section)
        
        # If no text content found, create a single section from JSON dump
        if not sections:
            json_content = json.dumps(content, indent=2)
            
            section = ParsedSection(
                content=json_content,
                metadata={
                    **base_metadata,
                    'content_type': 'json_dump'
                },
                section_id="full_content",
                section_type="text",
                hierarchy_level=0
            )
            sections.append(section)
        
        return sections
    
    def _create_section_from_content(
        self,
        content: str,
        section_name: str,
        base_metadata: Dict[str, Any],
        parent_section: Optional[str] = None
    ) -> ParsedSection:
        """Create a ParsedSection from content with metadata enrichment."""
        
        # Determine content type
        section_type = self._determine_content_type(content)
        
        # Determine hierarchy level
        hierarchy_level = self._determine_hierarchy_level(section_name, content)
        
        # Create enriched metadata
        section_metadata = {
            **base_metadata,
            'section_name': section_name,
            'content_length': len(content),
            'word_count': len(content.split()),
            'line_count': len(content.splitlines()),
            'detected_language': self._detect_programming_language(content)
        }
        
        # Add content analysis
        if section_type == 'code':
            section_metadata['code_blocks'] = len(re.findall(r'```.*?```', content, re.DOTALL))
        elif section_type == 'api':
            section_metadata['api_endpoints'] = len(re.findall(
                r'(?:GET|POST|PUT|DELETE|PATCH)\s+/', content, re.IGNORECASE
            ))
        
        return ParsedSection(
            content=content,
            metadata=section_metadata,
            section_id=f"section_{section_name.replace('.', '_')}",
            section_type=section_type,
            hierarchy_level=hierarchy_level,
            parent_section=parent_section
        )
    
    def _determine_content_type(self, content: str) -> str:
        """Determine content type based on patterns."""
        content_lower = content.lower()
        
        # Check each content type pattern
        for content_type, patterns in self.content_type_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    matches += 1
            
            # If multiple patterns match, classify as that type
            if matches >= 2 or (content_type == 'code' and matches >= 1):
                return content_type
        
        return 'text'  # Default to text
    
    def _determine_hierarchy_level(self, section_name: str, content: str) -> int:
        """Determine hierarchical level of section."""
        section_name_lower = section_name.lower()
        
        # Check hierarchy keywords
        for level, keywords in self.hierarchy_keywords.items():
            if any(keyword in section_name_lower for keyword in keywords):
                return level
        
        # Check content for hierarchy indicators
        if re.search(r'^#\s', content, re.MULTILINE):
            return 0  # Top-level header
        elif re.search(r'^##\s', content, re.MULTILINE):
            return 1  # Second-level header
        elif re.search(r'^###\s', content, re.MULTILINE):
            return 2  # Third-level header
        
        return 2  # Default level
    
    def _detect_programming_language(self, content: str) -> Optional[str]:
        """Detect programming language from content patterns."""
        content_lower = content.lower()
        
        language_scores = {}
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            # Return language with highest score
            return max(language_scores, key=language_scores.get)
        
        return None
    
    def _is_metadata_field(self, field_name: str) -> bool:
        """Check if field is metadata rather than content."""
        metadata_fields = {
            'id', 'name', 'version', 'title', 'description',
            'author', 'license', 'url', 'created_at', 'updated_at',
            'metadata', 'config', 'settings'
        }
        return field_name.lower() in metadata_fields
    
    def _extract_text_content(self, obj: Any) -> List[Tuple[str, str]]:
        """Recursively extract text content from arbitrary object structure."""
        text_content = []
        
        def extract_recursive(obj: Any, path: str = ""):
            if isinstance(obj, str) and len(obj.strip()) > 50:
                # Only include substantial text content
                text_content.append((path, obj.strip()))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    extract_recursive(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    extract_recursive(item, new_path)
        
        extract_recursive(obj)
        return text_content
    
    async def _enrich_sections(
        self, 
        sections: List[ParsedSection], 
        doc_path: Path
    ) -> List[ParsedSection]:
        """Enrich sections with additional metadata and processing info."""
        
        # Add document-level metadata
        doc_metadata = {
            'source_file': str(doc_path),
            'parsed_at': datetime.now().isoformat(),
            'total_sections': len(sections),
            'parser_version': '1.0'
        }
        
        # Calculate section statistics
        total_content_length = sum(len(section.content) for section in sections)
        
        for i, section in enumerate(sections):
            # Add position and relationship metadata
            section.metadata.update({
                **doc_metadata,
                'section_index': i,
                'relative_size': len(section.content) / total_content_length if total_content_length > 0 else 0.0
            })
            
            # Add next/previous section relationships
            if i > 0:
                section.metadata['previous_section'] = sections[i-1].section_id
            if i < len(sections) - 1:
                section.metadata['next_section'] = sections[i+1].section_id
        
        return sections
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """Get parser performance and usage statistics."""
        # This would be implemented with instance variables tracking
        # parsing operations, success rates, etc.
        return {
            'parser_version': '1.0',
            'supported_schemas': [schema.value for schema in SchemaType],
            'supported_formats': ['json', 'json.gz'],
            'content_types_detected': list(self.content_type_patterns.keys()),
            'languages_detected': list(self.language_patterns.keys())
        }