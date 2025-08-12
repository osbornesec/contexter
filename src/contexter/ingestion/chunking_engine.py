"""
Intelligent Chunking Engine - Semantic-aware document chunking system.

Advanced chunking system that preserves semantic boundaries while optimizing
for embedding generation with programming language awareness and content
type-specific strategies.
"""

import re
import logging
import tiktoken
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
from datetime import datetime

from .json_parser import ParsedSection

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies for different content types."""
    NARRATIVE_TEXT = "narrative_text"
    CODE_AWARE = "code_aware"
    API_DOCUMENTATION = "api_documentation"
    STRUCTURED_CONTENT = "structured_content"
    MIXED_CONTENT = "mixed_content"


@dataclass
class DocumentChunk:
    """
    Document chunk with semantic boundaries and metadata.
    
    Represents a semantically coherent piece of content optimized
    for embedding generation and vector search.
    """
    chunk_id: str
    library_id: str
    version: str
    chunk_index: int
    total_chunks: int
    content: str
    content_hash: str
    token_count: int
    char_count: int
    chunk_type: str  # 'text', 'code', 'api', 'example'
    programming_language: Optional[str]
    semantic_boundary: bool  # True if chunk ends at semantic boundary
    embedding_ready: bool = True
    
    # Metadata from source section
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Chunking metadata
    chunking_strategy: str = ChunkingStrategy.NARRATIVE_TEXT.value
    has_overlap: bool = False
    overlap_start_tokens: int = 0
    overlap_end_tokens: int = 0
    
    # Quality indicators
    completeness_score: float = 1.0  # How complete this chunk is (0.0-1.0)
    context_preservation_score: float = 1.0  # How well context is preserved
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate content hash if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode('utf-8')
            ).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage/serialization."""
        return {
            'chunk_id': self.chunk_id,
            'library_id': self.library_id,
            'version': self.version,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'content': self.content,
            'content_hash': self.content_hash,
            'token_count': self.token_count,
            'char_count': self.char_count,
            'chunk_type': self.chunk_type,
            'programming_language': self.programming_language,
            'semantic_boundary': self.semantic_boundary,
            'embedding_ready': self.embedding_ready,
            'metadata': self.metadata,
            'chunking_strategy': self.chunking_strategy,
            'has_overlap': self.has_overlap,
            'overlap_start_tokens': self.overlap_start_tokens,
            'overlap_end_tokens': self.overlap_end_tokens,
            'completeness_score': self.completeness_score,
            'context_preservation_score': self.context_preservation_score,
            'created_at': self.created_at.isoformat()
        }


class IntelligentChunkingEngine:
    """
    Intelligent document chunking with semantic boundary preservation.
    
    Features:
    - Multiple chunking strategies for different content types
    - Semantic boundary detection and preservation
    - Programming language-aware chunking for code content
    - Token-accurate chunking with tiktoken integration
    - Context overlap management for continuity
    - Quality scoring for chunk assessment
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_chunks_per_doc: int = 100,
        tokenizer_model: str = "cl100k_base"
    ):
        """
        Initialize the chunking engine.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap size in tokens for context preservation
            max_chunks_per_doc: Maximum chunks per document
            tokenizer_model: Tiktoken model for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_doc = max_chunks_per_doc
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_model)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_model}: {e}")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Fallback
        
        # Semantic boundary patterns for different content types
        self.boundary_patterns = {
            'code': [
                r'\n\n(?=def\s+\w+)',           # Function definitions
                r'\n\n(?=class\s+\w+)',         # Class definitions
                r'\n\n(?=async\s+def\s+\w+)',   # Async function definitions
                r'\n\n(?=@\w+)',                # Decorators
                r'\n\n(?=if\s+__name__)',       # Main blocks
                r'\n\n(?=import\s+\w+)',        # Import blocks
                r'\n\n(?=from\s+\w+)',          # From imports
                r'\n```\n',                     # Code block boundaries
                r'\n\n(?=//\s*[A-Z])',          # Comment sections
                r'\n\n(?=#\s+[A-Z])',           # Header comments
            ],
            'api': [
                r'\n\n(?=GET\s+/)',             # API endpoints
                r'\n\n(?=POST\s+/)',
                r'\n\n(?=PUT\s+/)',
                r'\n\n(?=DELETE\s+/)',
                r'\n\n(?=PATCH\s+/)',
                r'\n\n(?=##\s+)',               # API section headers
                r'\n\n(?=###\s+)',              # API subsection headers
                r'\n\n(?=Parameters:)',         # Parameter sections
                r'\n\n(?=Response:)',           # Response sections
                r'\n\n(?=Example:)',            # Example sections
            ],
            'text': [
                r'\n\n(?=[A-Z][^.!?]*[.!?]\s*\n)', # Paragraph breaks
                r'\n\n(?=#\s+)',                   # Markdown headers
                r'\n\n(?=##\s+)',                  # Markdown subheaders
                r'\n\n(?=###\s+)',                 # Markdown sub-subheaders
                r'\n\n(?=\*\s+)',                  # List items
                r'\n\n(?=\d+\.\s+)',               # Numbered lists
                r'\n\n(?=>\s+)',                   # Blockquotes
                r'\n\n---\n',                      # Horizontal rules
            ],
            'example': [
                r'\n\n(?=>>>)',                 # Python REPL examples
                r'\n\n(?=\$\s+)',               # Shell commands
                r'\n\n(?=Example\s*\d*:)',      # Example headers
                r'\n\n(?=Demo:)',               # Demo sections
                r'\n\n(?=Usage:)',              # Usage sections
                r'\n```\n',                     # Code block boundaries
            ]
        }
        
        # Language-specific patterns
        self.language_boundaries = {
            'python': [
                r'\n(?=def\s+\w+)',
                r'\n(?=class\s+\w+)',
                r'\n(?=if\s+__name__)',
                r'\n(?=@\w+)',
                r'\n(?=import\s+)',
                r'\n(?=from\s+\w+\s+import)',
            ],
            'javascript': [
                r'\n(?=function\s+\w+)',
                r'\n(?=const\s+\w+\s*=)',
                r'\n(?=let\s+\w+\s*=)',
                r'\n(?=var\s+\w+\s*=)',
                r'\n(?=class\s+\w+)',
                r'\n(?=export\s+)',
                r'\n(?=import\s+)',
            ],
            'java': [
                r'\n(?=public\s+class)',
                r'\n(?=private\s+class)',
                r'\n(?=public\s+static)',
                r'\n(?=public\s+\w+)',
                r'\n(?=private\s+\w+)',
                r'\n(?=@\w+)',
                r'\n(?=import\s+)',
                r'\n(?=package\s+)',
            ],
            'go': [
                r'\n(?=func\s+\w+)',
                r'\n(?=func\s+\(\w+)',
                r'\n(?=type\s+\w+)',
                r'\n(?=var\s+\w+)',
                r'\n(?=const\s+\w+)',
                r'\n(?=package\s+)',
                r'\n(?=import\s+)',
            ]
        }
        
        # Content quality indicators
        self.quality_indicators = {
            'complete_sentences': r'[.!?]\s*$',
            'complete_code_blocks': r'```[\s\S]*?```',
            'complete_functions': r'def\s+\w+\([^)]*\):\s*[\s\S]*?(?=\n(?:def|class|\Z))',
            'complete_paragraphs': r'\n\n',
        }
        
        logger.info(f"Chunking engine initialized (size: {chunk_size}, overlap: {chunk_overlap})")
    
    async def chunk_document_sections(
        self,
        sections: List[ParsedSection]
    ) -> List[DocumentChunk]:
        """
        Chunk document sections with semantic boundary preservation.
        
        Args:
            sections: Parsed document sections to chunk
            
        Returns:
            List of document chunks optimized for embeddings
        """
        all_chunks = []
        
        for section in sections:
            try:
                # Determine chunking strategy
                strategy = self._select_chunking_strategy(section)
                
                # Chunk the section
                section_chunks = await self._chunk_section(section, strategy)
                
                all_chunks.extend(section_chunks)
                
                # Check maximum chunks limit
                if len(all_chunks) >= self.max_chunks_per_doc:
                    all_chunks = all_chunks[:self.max_chunks_per_doc]
                    logger.warning(
                        f"Reached maximum chunks limit ({self.max_chunks_per_doc}), "
                        f"truncating document"
                    )
                    break
                    
            except Exception as e:
                logger.error(f"Failed to chunk section {section.section_id}: {e}")
                continue
        
        # Update total chunks count and finalize
        await self._finalize_chunks(all_chunks)
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    def _select_chunking_strategy(self, section: ParsedSection) -> ChunkingStrategy:
        """Select optimal chunking strategy based on section characteristics."""
        content_type = section.section_type
        content = section.content.lower()
        
        # Check for mixed content
        has_code = '```' in content or 'def ' in content or 'function ' in content
        has_api = any(method in content for method in ['get /', 'post /', 'put /', 'delete /'])
        has_text = len(re.findall(r'[.!?]\s+[A-Z]', content)) > 3
        
        content_types = sum([has_code, has_api, has_text])
        
        if content_types > 1:
            return ChunkingStrategy.MIXED_CONTENT
        elif content_type == 'code' or has_code:
            return ChunkingStrategy.CODE_AWARE
        elif content_type == 'api' or has_api:
            return ChunkingStrategy.API_DOCUMENTATION
        elif 'example' in section.metadata.get('section_name', '').lower():
            return ChunkingStrategy.CODE_AWARE  # Examples often contain code
        else:
            return ChunkingStrategy.NARRATIVE_TEXT
    
    async def _chunk_section(
        self,
        section: ParsedSection,
        strategy: ChunkingStrategy
    ) -> List[DocumentChunk]:
        """Chunk a single section using the specified strategy."""
        
        if strategy == ChunkingStrategy.CODE_AWARE:
            return await self._chunk_code_aware_content(section)
        elif strategy == ChunkingStrategy.API_DOCUMENTATION:
            return await self._chunk_api_documentation(section)
        elif strategy == ChunkingStrategy.MIXED_CONTENT:
            return await self._chunk_mixed_content(section)
        elif strategy == ChunkingStrategy.STRUCTURED_CONTENT:
            return await self._chunk_structured_content(section)
        else:
            return await self._chunk_narrative_content(section)
    
    async def _chunk_code_aware_content(self, section: ParsedSection) -> List[DocumentChunk]:
        """Chunk content with code-aware boundary detection."""
        content = section.content
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= self.chunk_size:
            # Content fits in single chunk
            return [await self._create_chunk(
                content, 0, 1, section, ChunkingStrategy.CODE_AWARE
            )]
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(tokens) and len(chunks) < self.max_chunks_per_doc:
            chunk_end = min(current_pos + self.chunk_size, len(tokens))
            
            # Find semantic boundary
            if chunk_end < len(tokens):
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                # Try to find code-aware boundary
                boundary_pos = self._find_code_boundary(chunk_text, section)
                
                if boundary_pos and boundary_pos > len(chunk_text) * 0.5:
                    # Use semantic boundary if it's not too early
                    chunk_text = chunk_text[:boundary_pos].rstrip()
                    chunk_tokens = self.tokenizer.encode(chunk_text)
                    chunk_end = current_pos + len(chunk_tokens)
                    semantic_boundary = True
                else:
                    # No good boundary found, use token limit
                    semantic_boundary = False
            else:
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                semantic_boundary = True  # Last chunk
            
            # Add overlap from previous chunk
            if chunk_index > 0 and self.chunk_overlap > 0:
                overlap_start = max(0, current_pos - self.chunk_overlap)
                overlap_tokens = tokens[overlap_start:current_pos]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                
                final_chunk_text = overlap_text + chunk_text
                has_overlap = True
                overlap_start_tokens = len(overlap_tokens)
            else:
                final_chunk_text = chunk_text
                has_overlap = False
                overlap_start_tokens = 0
            
            # Create chunk
            chunk = await self._create_chunk(
                final_chunk_text, chunk_index, -1, section, 
                ChunkingStrategy.CODE_AWARE, has_overlap, 
                overlap_start_tokens, semantic_boundary
            )
            
            chunks.append(chunk)
            
            # Move to next position (accounting for overlap)
            current_pos = chunk_end - (self.chunk_overlap // 2 if chunk_index > 0 else 0)
            chunk_index += 1
        
        return chunks
    
    async def _chunk_api_documentation(self, section: ParsedSection) -> List[DocumentChunk]:
        """Chunk API documentation preserving endpoint boundaries."""
        content = section.content
        
        # Split by API endpoints first
        api_pattern = r'\n(?=(?:GET|POST|PUT|DELETE|PATCH)\s+/)'
        api_sections = re.split(api_pattern, content, flags=re.MULTILINE)
        
        chunks = []
        chunk_index = 0
        
        for api_section in api_sections:
            if not api_section.strip():
                continue
            
            section_tokens = self.tokenizer.encode(api_section)
            
            if len(section_tokens) <= self.chunk_size:
                # API section fits in one chunk
                chunk = await self._create_chunk(
                    api_section, chunk_index, -1, section,
                    ChunkingStrategy.API_DOCUMENTATION
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Need to split large API section
                sub_chunks = await self._split_large_content(
                    api_section, section, ChunkingStrategy.API_DOCUMENTATION,
                    chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            
            if len(chunks) >= self.max_chunks_per_doc:
                break
        
        return chunks
    
    async def _chunk_mixed_content(self, section: ParsedSection) -> List[DocumentChunk]:
        """Chunk mixed content with adaptive boundary detection."""
        content = section.content
        
        # First, try to separate code blocks from text
        code_block_pattern = r'(```[\s\S]*?```)'
        parts = re.split(code_block_pattern, content)
        
        chunks = []
        chunk_index = 0
        current_chunk_content = ""
        current_chunk_tokens = 0
        
        for part in parts:
            if not part.strip():
                continue
            
            part_tokens = len(self.tokenizer.encode(part))
            
            # If this part would make chunk too large, finalize current chunk
            if (current_chunk_tokens + part_tokens > self.chunk_size and 
                current_chunk_content.strip()):
                
                chunk = await self._create_chunk(
                    current_chunk_content, chunk_index, -1, section,
                    ChunkingStrategy.MIXED_CONTENT
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk_content = part
                current_chunk_tokens = part_tokens
            else:
                current_chunk_content += part
                current_chunk_tokens += part_tokens
        
        # Add final chunk if there's remaining content
        if current_chunk_content.strip():
            chunk = await self._create_chunk(
                current_chunk_content, chunk_index, -1, section,
                ChunkingStrategy.MIXED_CONTENT
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_structured_content(self, section: ParsedSection) -> List[DocumentChunk]:
        """Chunk structured content like lists, tables, etc."""
        content = section.content
        
        # Split by structural elements
        structure_patterns = [
            r'\n(?=\d+\.\s+)',    # Numbered lists
            r'\n(?=\*\s+)',       # Bullet lists
            r'\n(?=##\s+)',       # Headers
            r'\n(?=\|\s*)',       # Table rows
            r'\n\n(?=[A-Z])',     # Paragraph breaks
        ]
        
        # Find all structural boundaries
        boundaries = []
        for pattern in structure_patterns:
            for match in re.finditer(pattern, content):
                boundaries.append(match.start())
        
        # Sort boundaries and create chunks
        boundaries = sorted(set(boundaries))
        if not boundaries:
            # No structure found, use narrative chunking
            return await self._chunk_narrative_content(section)
        
        chunks = []
        chunk_index = 0
        start_pos = 0
        
        for boundary in boundaries + [len(content)]:
            chunk_content = content[start_pos:boundary]
            
            if chunk_content.strip():
                chunk_tokens = len(self.tokenizer.encode(chunk_content))
                
                if chunk_tokens <= self.chunk_size:
                    chunk = await self._create_chunk(
                        chunk_content, chunk_index, -1, section,
                        ChunkingStrategy.STRUCTURED_CONTENT
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                else:
                    # Large structured section, split further
                    sub_chunks = await self._split_large_content(
                        chunk_content, section, 
                        ChunkingStrategy.STRUCTURED_CONTENT, chunk_index
                    )
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
            
            start_pos = boundary
            
            if len(chunks) >= self.max_chunks_per_doc:
                break
        
        return chunks
    
    async def _chunk_narrative_content(self, section: ParsedSection) -> List[DocumentChunk]:
        """Chunk narrative text content with paragraph preservation."""
        content = section.content
        tokens = self.tokenizer.encode(content)
        
        if len(tokens) <= self.chunk_size:
            return [await self._create_chunk(
                content, 0, 1, section, ChunkingStrategy.NARRATIVE_TEXT
            )]
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(tokens) and len(chunks) < self.max_chunks_per_doc:
            chunk_end = min(current_pos + self.chunk_size, len(tokens))
            
            # Find text boundary
            if chunk_end < len(tokens):
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                boundary_pos = self._find_text_boundary(chunk_text)
                
                if boundary_pos and boundary_pos > len(chunk_text) * 0.6:
                    chunk_text = chunk_text[:boundary_pos].rstrip()
                    chunk_tokens = self.tokenizer.encode(chunk_text)
                    chunk_end = current_pos + len(chunk_tokens)
                    semantic_boundary = True
                else:
                    semantic_boundary = False
            else:
                chunk_tokens = tokens[current_pos:chunk_end]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                semantic_boundary = True
            
            # Add overlap
            if chunk_index > 0 and self.chunk_overlap > 0:
                overlap_start = max(0, current_pos - self.chunk_overlap)
                overlap_tokens = tokens[overlap_start:current_pos]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                
                final_chunk_text = overlap_text + chunk_text
                has_overlap = True
                overlap_start_tokens = len(overlap_tokens)
            else:
                final_chunk_text = chunk_text
                has_overlap = False
                overlap_start_tokens = 0
            
            chunk = await self._create_chunk(
                final_chunk_text, chunk_index, -1, section,
                ChunkingStrategy.NARRATIVE_TEXT, has_overlap,
                overlap_start_tokens, semantic_boundary
            )
            chunks.append(chunk)
            
            current_pos = chunk_end - (self.chunk_overlap // 3 if chunk_index > 0 else 0)
            chunk_index += 1
        
        return chunks
    
    async def _split_large_content(
        self,
        content: str,
        section: ParsedSection,
        strategy: ChunkingStrategy,
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """Split large content that doesn't fit in a single chunk."""
        tokens = self.tokenizer.encode(content)
        chunks = []
        current_pos = 0
        chunk_index = start_chunk_index
        
        while current_pos < len(tokens):
            chunk_end = min(current_pos + self.chunk_size, len(tokens))
            chunk_tokens = tokens[current_pos:chunk_end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunk = await self._create_chunk(
                chunk_text, chunk_index, -1, section, strategy
            )
            chunks.append(chunk)
            
            current_pos = chunk_end
            chunk_index += 1
        
        return chunks
    
    def _find_code_boundary(
        self, 
        text: str, 
        section: ParsedSection
    ) -> Optional[int]:
        """Find optimal code boundary position."""
        language = section.metadata.get('detected_language', 'python')
        
        # Get language-specific patterns
        patterns = self.language_boundaries.get(language, [])
        patterns.extend(self.boundary_patterns.get('code', []))
        
        best_boundary = None
        best_score = 0
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            for match in matches:
                pos = match.start()
                # Prefer boundaries that are not too early or too late
                position_score = 1.0 - abs((pos / len(text)) - 0.75)
                if position_score > best_score:
                    best_score = position_score
                    best_boundary = pos
        
        return best_boundary
    
    def _find_text_boundary(self, text: str) -> Optional[int]:
        """Find optimal text boundary position."""
        patterns = self.boundary_patterns.get('text', [])
        
        # Look for sentence boundaries first
        sentence_ends = []
        for match in re.finditer(r'[.!?]\s+', text):
            sentence_ends.append(match.end())
        
        if sentence_ends:
            # Find sentence end closest to 75% of text
            target_pos = len(text) * 0.75
            closest_boundary = min(sentence_ends, key=lambda x: abs(x - target_pos))
            return closest_boundary
        
        # Fall back to other patterns
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            if matches:
                # Use last match that's not too close to the end
                for match in reversed(matches):
                    if match.start() < len(text) * 0.9:
                        return match.start()
        
        return None
    
    async def _create_chunk(
        self,
        content: str,
        chunk_index: int,
        total_chunks: int,
        section: ParsedSection,
        strategy: ChunkingStrategy,
        has_overlap: bool = False,
        overlap_start_tokens: int = 0,
        semantic_boundary: bool = True
    ) -> DocumentChunk:
        """Create a document chunk with full metadata."""
        
        # Calculate tokens and character count
        token_count = len(self.tokenizer.encode(content))
        char_count = len(content)
        
        # Determine content quality scores
        completeness_score = self._calculate_completeness_score(content, strategy)
        context_preservation_score = self._calculate_context_score(
            content, has_overlap, semantic_boundary
        )
        
        # Create chunk metadata
        chunk_metadata = {
            **section.metadata,
            'source_section_id': section.section_id,
            'source_section_type': section.section_type,
            'chunking_timestamp': datetime.now().isoformat(),
            'chunk_boundaries_detected': semantic_boundary,
            'content_analysis': {
                'has_code_blocks': '```' in content,
                'has_api_references': bool(re.search(r'(?:GET|POST|PUT|DELETE)\s+/', content)),
                'has_examples': bool(re.search(r'(?:example|demo|>>>|\$ )', content, re.IGNORECASE)),
                'sentence_count': len(re.findall(r'[.!?]+', content)),
                'line_count': len(content.splitlines())
            }
        }
        
        chunk = DocumentChunk(
            chunk_id=f"{section.metadata.get('library_id', 'unknown')}_{section.metadata.get('version', '1.0')}_{section.section_id}_{chunk_index}",
            library_id=section.metadata.get('library_id', 'unknown'),
            version=section.metadata.get('version', ''),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            content=content,
            content_hash="",  # Will be generated in __post_init__
            token_count=token_count,
            char_count=char_count,
            chunk_type=section.section_type,
            programming_language=section.metadata.get('detected_language'),
            semantic_boundary=semantic_boundary,
            metadata=chunk_metadata,
            chunking_strategy=strategy.value,
            has_overlap=has_overlap,
            overlap_start_tokens=overlap_start_tokens,
            completeness_score=completeness_score,
            context_preservation_score=context_preservation_score
        )
        
        return chunk
    
    def _calculate_completeness_score(
        self, 
        content: str, 
        strategy: ChunkingStrategy
    ) -> float:
        """Calculate how complete/coherent the chunk content is."""
        score = 0.0
        
        # Base score for having content
        if content.strip():
            score += 0.3
        
        # Strategy-specific completeness checks
        if strategy == ChunkingStrategy.CODE_AWARE:
            # Check for complete code structures
            if re.search(r'def\s+\w+\([^)]*\):\s*[\s\S]*?(?=\n(?:def|class|\Z))', content):
                score += 0.4  # Complete function
            elif re.search(r'class\s+\w+[\s\S]*?(?=\n(?:class|def|\Z))', content):
                score += 0.4  # Complete class
            elif content.count('```') % 2 == 0 and '```' in content:
                score += 0.3  # Complete code blocks
            else:
                score += 0.1  # Partial code
                
        elif strategy == ChunkingStrategy.API_DOCUMENTATION:
            # Check for complete API documentation
            if re.search(r'(?:GET|POST|PUT|DELETE|PATCH)\s+/', content):
                score += 0.2  # Has endpoint
                if 'response' in content.lower():
                    score += 0.2  # Has response info
                if 'parameter' in content.lower():
                    score += 0.2  # Has parameter info
                if 'example' in content.lower():
                    score += 0.2  # Has examples
        
        elif strategy == ChunkingStrategy.NARRATIVE_TEXT:
            # Check for complete sentences and paragraphs
            sentences = re.findall(r'[.!?]+', content)
            if sentences:
                score += 0.4
            
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                score += 0.2
        
        # General completeness indicators
        if content.endswith(('.', '!', '?', '```', '}', ')')):
            score += 0.1  # Ends naturally
        
        return min(1.0, score)
    
    def _calculate_context_score(
        self,
        content: str,
        has_overlap: bool,
        semantic_boundary: bool
    ) -> float:
        """Calculate how well context is preserved in the chunk."""
        score = 0.5  # Base score
        
        # Overlap preservation
        if has_overlap:
            score += 0.2
        
        # Semantic boundary preservation
        if semantic_boundary:
            score += 0.3
        
        # Content length (longer chunks preserve more context)
        token_count = len(self.tokenizer.encode(content))
        if token_count > self.chunk_size * 0.8:
            score += 0.1
        elif token_count < self.chunk_size * 0.3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _finalize_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Finalize chunks by updating total counts and relationships."""
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            chunk.total_chunks = total_chunks
            chunk.chunk_index = i
            
            # Update chunk ID to ensure uniqueness
            chunk.chunk_id = f"{chunk.library_id}_{chunk.version}_{chunk.metadata.get('source_section_id', 'unknown')}_{i:03d}"
    
    def get_chunking_statistics(self) -> Dict[str, Any]:
        """Get chunking engine statistics and configuration."""
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_chunks_per_doc': self.max_chunks_per_doc,
            'tokenizer_model': self.tokenizer.name,
            'supported_strategies': [strategy.value for strategy in ChunkingStrategy],
            'supported_languages': list(self.language_boundaries.keys()),
            'boundary_patterns': {
                content_type: len(patterns) 
                for content_type, patterns in self.boundary_patterns.items()
            }
        }