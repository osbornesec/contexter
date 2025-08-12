"""
Content parsing and entry extraction for Context7 documentation.

This module parses raw documentation content from Context7 and extracts
individual entries (code snippets, Q&A, documentation blocks) into
structured data for complete coverage and systematic processing.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from enum import Enum

logger = logging.getLogger(__name__)


class EntryType(Enum):
    """Types of documentation entries."""
    CODE_SNIPPET = "code_snippet"
    QA_SECTION = "qa_section"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    INSTALLATION = "installation"
    EXAMPLE = "example"
    TUTORIAL = "tutorial"
    UNKNOWN = "unknown"


@dataclass
class DocumentationEntry:
    """Individual documentation entry with structured data."""
    
    entry_id: str
    entry_type: EntryType
    title: str
    description: str
    content: str
    source_url: Optional[str] = None
    language: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, str] = field(default_factory=dict)
    content_hash: str = ""
    
    def __post_init__(self):
        """Generate content hash and normalize data."""
        import hashlib
        # Generate content hash for deduplication
        content_str = f"{self.title}|{self.description}|{self.content}"
        self.content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        # Normalize language
        if self.language:
            self.language = self.language.lower().strip()
            
        # Extract tags from title and description
        self._extract_tags()
    
    def _extract_tags(self):
        """Extract relevant tags from title and description."""
        text = f"{self.title} {self.description}".lower()
        
        # Common programming patterns and keywords
        tag_patterns = {
            'api', 'rest', 'authentication', 'serialization', 'validation',
            'testing', 'tutorial', 'configuration', 'installation', 'setup',
            'example', 'debug', 'error', 'fix', 'migration', 'upgrade',
            'deployment', 'production', 'development', 'django', 'python',
            'javascript', 'typescript', 'shell', 'bash', 'html', 'css',
            'json', 'xml', 'database', 'model', 'view', 'serializer',
            'permission', 'middleware', 'routing', 'url', 'form', 'field'
        }
        
        for pattern in tag_patterns:
            if pattern in text:
                self.tags.add(pattern)


class ContentParser:
    """
    Parses Context7 documentation content and extracts individual entries.
    
    Handles different content types and formats, ensuring complete coverage
    by systematically extracting every code snippet, Q&A, and documentation block.
    """
    
    def __init__(self):
        """Initialize the content parser."""
        self.section_patterns = {
            'code_snippets': re.compile(r'={20,}\s*CODE SNIPPETS\s*={20,}', re.IGNORECASE),
            'qa_sections': re.compile(r'={20,}\s*Q[&A\s]*A?\s*={20,}', re.IGNORECASE),
            'documentation': re.compile(r'={20,}\s*DOCUMENTATION\s*={20,}', re.IGNORECASE),
            'examples': re.compile(r'={20,}\s*EXAMPLES\s*={20,}', re.IGNORECASE),
        }
        
        # Entry separator pattern
        self.entry_separator = re.compile(r'^-{30,}$', re.MULTILINE)
        
        # Field patterns for structured entries
        self.field_patterns = {
            'title': re.compile(r'^TITLE:\s*(.+)$', re.MULTILINE),
            'description': re.compile(r'^DESCRIPTION:\s*(.+?)(?=^[A-Z]+:|^-{30,}|^={20,}|\Z)', re.MULTILINE | re.DOTALL),
            'source': re.compile(r'^SOURCE:\s*(.+)$', re.MULTILINE),
            'language': re.compile(r'^LANGUAGE:\s*(.+)$', re.MULTILINE),
            'code': re.compile(r'^CODE:\s*\n```(?:[a-zA-Z]*\n)?(.*?)\n```', re.MULTILINE | re.DOTALL),
        }
    
    def parse_content(self, raw_content: str, library_id: str, context: str) -> List[DocumentationEntry]:
        """
        Parse raw content and extract all individual entries.
        
        Args:
            raw_content: Raw documentation content from Context7
            library_id: Library identifier for metadata
            context: Search context that generated this content
            
        Returns:
            List of structured documentation entries
        """
        logger.info(f"Parsing content for {library_id} (context: {context[:50]}...)")
        
        entries = []
        
        # Split content by major sections
        sections = self._split_into_sections(raw_content)
        
        for section_type, section_content in sections.items():
            logger.debug(f"Processing {section_type} section ({len(section_content)} chars)")
            
            # Extract individual entries from each section
            section_entries = self._extract_entries_from_section(
                section_content, section_type, library_id, context
            )
            entries.extend(section_entries)
        
        # Handle content without clear section headers (fallback parsing)
        if not sections:
            logger.debug("No clear sections found, using fallback parsing")
            fallback_entries = self._extract_entries_fallback(raw_content, library_id, context)
            entries.extend(fallback_entries)
        
        logger.info(f"Extracted {len(entries)} entries from content")
        
        return entries
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into major sections based on headers."""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            matches = list(pattern.finditer(content))
            
            for i, match in enumerate(matches):
                start_pos = match.end()
                
                # Find the end of this section (next section or end of content)
                end_pos = len(content)
                for other_pattern in self.section_patterns.values():
                    if other_pattern == pattern:
                        continue
                    next_matches = list(other_pattern.finditer(content[start_pos:]))
                    if next_matches:
                        end_pos = min(end_pos, start_pos + next_matches[0].start())
                
                section_content = content[start_pos:end_pos].strip()
                if section_content:
                    section_key = f"{section_name}_{i}" if i > 0 else section_name
                    sections[section_key] = section_content
        
        return sections
    
    def _extract_entries_from_section(
        self, section_content: str, section_type: str, library_id: str, context: str
    ) -> List[DocumentationEntry]:
        """Extract individual entries from a section."""
        entries = []
        
        # Split section into individual entries using separator
        raw_entries = self.entry_separator.split(section_content)
        
        for i, raw_entry in enumerate(raw_entries):
            raw_entry = raw_entry.strip()
            if not raw_entry or len(raw_entry) < 20:  # Skip empty or tiny entries
                continue
            
            entry = self._parse_single_entry(raw_entry, section_type, library_id, context, i)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _extract_entries_fallback(
        self, content: str, library_id: str, context: str
    ) -> List[DocumentationEntry]:
        """Fallback parsing when no clear sections are found."""
        entries = []
        
        # Try to find individual entries by separator pattern
        raw_entries = self.entry_separator.split(content)
        
        if len(raw_entries) <= 1:
            # No separators found, treat entire content as single entry
            entry = self._parse_single_entry(content, "unknown", library_id, context, 0)
            if entry:
                entries.append(entry)
        else:
            # Process separated entries
            for i, raw_entry in enumerate(raw_entries):
                raw_entry = raw_entry.strip()
                if not raw_entry or len(raw_entry) < 20:
                    continue
                
                entry = self._parse_single_entry(raw_entry, "unknown", library_id, context, i)
                if entry:
                    entries.append(entry)
        
        return entries
    
    def _parse_single_entry(
        self, raw_entry: str, section_type: str, library_id: str, context: str, entry_index: int
    ) -> Optional[DocumentationEntry]:
        """Parse a single raw entry into structured data."""
        # Extract structured fields
        fields = {}
        for field_name, pattern in self.field_patterns.items():
            match = pattern.search(raw_entry)
            if match:
                if field_name == 'description':
                    # Clean up description - remove extra whitespace
                    fields[field_name] = re.sub(r'\s+', ' ', match.group(1).strip())
                else:
                    fields[field_name] = match.group(1).strip()
        
        # Must have at least a title or some substantial content
        if not fields.get('title') and len(raw_entry) < 50:
            return None
        
        # Determine entry type based on section and content
        entry_type = self._determine_entry_type(section_type, fields, raw_entry)
        
        # Generate entry ID
        entry_id = f"{library_id.replace('/', '_')}_{entry_type.value}_{entry_index}"
        
        # Create structured entry
        entry = DocumentationEntry(
            entry_id=entry_id,
            entry_type=entry_type,
            title=fields.get('title', self._extract_title_from_content(raw_entry)),
            description=fields.get('description', self._extract_description_from_content(raw_entry)),
            content=fields.get('code', raw_entry),
            source_url=fields.get('source'),
            language=fields.get('language'),
            metadata={
                'original_context': context,
                'section_type': section_type,
                'library_id': library_id,
                'entry_length': str(len(raw_entry)),
                'has_structured_fields': str(bool(fields))
            }
        )
        
        return entry
    
    def _determine_entry_type(self, section_type: str, fields: Dict[str, str], content: str) -> EntryType:
        """Determine the type of entry based on available information."""
        # Check section type first
        if 'code' in section_type.lower():
            return EntryType.CODE_SNIPPET
        elif 'qa' in section_type.lower():
            return EntryType.QA_SECTION
        elif 'example' in section_type.lower():
            return EntryType.EXAMPLE
        elif 'documentation' in section_type.lower():
            return EntryType.DOCUMENTATION
        
        # Check for code blocks
        if fields.get('code') or '```' in content:
            return EntryType.CODE_SNIPPET
        
        # Check for installation patterns
        if any(keyword in content.lower() for keyword in ['install', 'pip install', 'npm install', 'setup']):
            return EntryType.INSTALLATION
        
        # Check for configuration patterns
        if any(keyword in content.lower() for keyword in ['config', 'settings', 'environment']):
            return EntryType.CONFIGURATION
        
        # Check for tutorial patterns
        if any(keyword in content.lower() for keyword in ['tutorial', 'guide', 'step', 'how to']):
            return EntryType.TUTORIAL
        
        # Default to documentation
        return EntryType.DOCUMENTATION
    
    def _extract_title_from_content(self, content: str) -> str:
        """Extract a title from unstructured content."""
        lines = content.strip().split('\n')
        
        # Look for first substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 10 and not line.startswith(('=', '-', '#')):
                return line[:100] + ('...' if len(line) > 100 else '')
        
        return f"Entry ({len(content)} chars)"
    
    def _extract_description_from_content(self, content: str) -> str:
        """Extract a description from unstructured content."""
        lines = content.strip().split('\n')
        
        # Skip first line (likely title), get next few lines as description
        description_lines = []
        for line in lines[1:5]:  # Take up to 4 lines after title
            line = line.strip()
            if len(line) > 5 and not line.startswith(('=', '-', '#', '```')):
                description_lines.append(line)
        
        if description_lines:
            description = ' '.join(description_lines)
            return description[:300] + ('...' if len(description) > 300 else '')
        
        return "No description available"


class EntryDeduplicator:
    """Deduplicates documentation entries to ensure uniqueness."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize deduplicator with similarity threshold."""
        self.similarity_threshold = similarity_threshold
    
    def deduplicate_entries(self, entries: List[DocumentationEntry]) -> List[DocumentationEntry]:
        """
        Remove duplicate entries based on content similarity.
        
        Args:
            entries: List of documentation entries
            
        Returns:
            Deduplicated list of entries
        """
        if not entries:
            return entries
        
        logger.info(f"Deduplicating {len(entries)} entries")
        
        unique_entries = []
        seen_hashes = set()
        
        for entry in entries:
            # First check exact hash match
            if entry.content_hash in seen_hashes:
                logger.debug(f"Skipping duplicate entry: {entry.title[:50]}...")
                continue
            
            # Check similarity with existing entries
            is_duplicate = False
            for existing_entry in unique_entries:
                if self._are_similar(entry, existing_entry):
                    logger.debug(f"Skipping similar entry: {entry.title[:50]}...")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_entries.append(entry)
                seen_hashes.add(entry.content_hash)
        
        logger.info(f"Kept {len(unique_entries)} unique entries ({len(entries) - len(unique_entries)} duplicates removed)")
        
        return unique_entries
    
    def _are_similar(self, entry1: DocumentationEntry, entry2: DocumentationEntry) -> bool:
        """Check if two entries are similar enough to be considered duplicates."""
        # Quick hash check first
        if entry1.content_hash == entry2.content_hash:
            return True
        
        # Check title similarity
        title_similarity = self._calculate_text_similarity(entry1.title, entry2.title)
        if title_similarity > 0.9:  # Very similar titles
            return True
        
        # Check content similarity for code entries
        if entry1.entry_type == entry2.entry_type == EntryType.CODE_SNIPPET:
            content_similarity = self._calculate_text_similarity(entry1.content, entry2.content)
            return content_similarity > self.similarity_threshold
        
        # Check combined similarity for other types
        combined1 = f"{entry1.title} {entry1.description}"
        combined2 = f"{entry2.title} {entry2.description}"
        combined_similarity = self._calculate_text_similarity(combined1, combined2)
        
        return combined_similarity > self.similarity_threshold
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using simple metrics."""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


# Export main classes
__all__ = [
    'EntryType',
    'DocumentationEntry', 
    'ContentParser',
    'EntryDeduplicator'
]