# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Contexter is an AI agent orchestration system with 42 specialized agents designed for autonomous development workflows. The system transforms todo items into comprehensive Product Requirement Prompts (PRPs) that can be executed with high success rates.

**Target Implementation**: Context7 Documentation Downloader (C7DocDownloader) - A high-performance Python CLI application for comprehensive documentation retrieval using intelligent proxy rotation and advanced deduplication.

## Core Commands

### Agent Invocation
```bash
# Primary entry point - transforms todos into PRPs
@agent-prp-todo-to-prp-orchestrator

# Execute generated PRPs with validation
@agent-prp-execution-orchestrator run ./ai_docs/prps/[prp-name].md

# Invoke any of the 42 specialized agents
@agent-[agent-name]
```

### Development Operations
Based on the ai_docs deliverables, key development patterns include:

```bash
# Code quality and performance testing
pytest tests/ --cov=src/contexter --cov-report=xml
mypy src/
ruff check src/
black src/

# Performance benchmarking
pytest tests/performance/ -v

# Integration testing (requires credentials)
pytest tests/integration/ -v --integration
```

## Architecture

### System Architecture (From ai_docs/deliverables/system-architect/)
The C7DocDownloader implements a sophisticated async-first architecture:

- **Async-First Design**: Python asyncio-based for optimal I/O-bound performance
- **Proxy Abstraction Layer**: BrightData residential proxy integration with circuit breaker pattern
- **Multi-Query Strategy**: Parallel request processing with intelligent context generation
- **Post-Processing Deduplication**: Advanced content merging with semantic analysis
- **Modular Storage**: JSON-based local storage with compression and version management

### Key Components
```
src/c7doc/
├── cli/                    # CLI interface layer (Click/Rich)
├── core/                   # Core business logic
│   ├── download_engine.py  # Async request processing
│   ├── deduplication.py    # Content merging algorithms
│   └── storage.py         # Compressed local storage
├── integration/            # External integrations
│   ├── proxy_manager.py   # BrightData proxy rotation
│   ├── context7_client.py # Context7 API integration
│   └── config_manager.py  # YAML configuration
├── models/                # Data models and types
└── utils/                 # Utility functions
```

### Agent Communication Protocol
From the deliverables analysis, agents use standardized file-based communication:
- **Deliverable Structure**: Organized outputs in `ai_docs/deliverables/[agent-name]/`
- **Self-Critique**: Mandatory quality evaluation for every agent execution
- **Handoff Protocol**: Agents use `rg` to check for inter-agent messages
- **Artifact Management**: Timestamped deliverables with comprehensive metadata

## Implementation Patterns (From Code Patterns Deliverable)

### Async HTTP Operations
```python
# Proxy-Aware HTTP Client Pattern
class ProxyAwareHTTPClient:
    async def request_with_retry(self, method: str, url: str, 
                                max_retries: int = 3, **kwargs) -> httpx.Response:
        # Implements exponential backoff with jitter
        # Automatic proxy rotation on 429 responses
        # Circuit breaker pattern for failed proxies
```

### Error Handling
```python
# Comprehensive Error Classification
class ErrorCategory(Enum):
    NETWORK = "network"
    PROXY = "proxy" 
    API = "api"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    DEDUPLICATION = "deduplication"

# Each error provides actionable resolution steps
```

### Performance Optimization
- **Concurrent Processing**: Semaphore-based rate limiting (max 10 concurrent)
- **Memory Management**: Streaming processing for large documents
- **Request Optimization**: Intelligent jitter to prevent thundering herd
- **Compression**: gzip compression achieving >60% size reduction

## Development Workflow

### PRP Creation Pipeline (14 agents)
1. **prp-todo-to-prp-orchestrator** - Entry point orchestrator
2. **prp-user-story-architect** - User-centered design and acceptance criteria
3. **prp-context-engineer** - Documentation curation and context management
4. **prp-blueprint-architect** - Implementation task breakdown with dependencies
5. **prp-specification-writer** - Technical specifications and API contracts
6. **prp-research-engineer** - Technology evaluation and recommendations
7. **prp-integration-planner** - System integration strategies
8. **prp-validation-designer** - Multi-level testing strategies
9. **prp-success-metrics-designer** - KPIs and success criteria
10. **prp-gotcha-curator** - Pitfall documentation and mitigation
11. **prp-quality-assurance-specialist** - Final quality validation
12. **prp-template-architect** - Reusable template management
13. **prp-proof-of-concept-planner** - POC and risk validation
14. **prp-execution-orchestrator** - PRP execution management

### Implementation Sprint Structure (From Task Breakdown)
**Sprint 1 (Weeks 1-2): Core Infrastructure**
- PROXY-001: Proxy Manager Implementation (12h)
- CONFIG-001: Configuration Manager (6h) 
- DOWNLOAD-001: Download Engine Foundation (16h)
- STORAGE-001: Storage Manager Implementation (10h)

**Sprint 2 (Weeks 3-4): Feature Completion**
- CONTEXT7-001: Context7 API Client completion
- DEDUPE-001: Basic Deduplication Engine (12h)
- CLI-001: Command Line Interface (10h)
- INTEGRATION-001: End-to-End Integration (12h)

## Quality Assurance Framework

### Requirements Traceability (From Traceability Matrix)
All requirements traced from business objectives through test cases:
- **FR-003**: Multi-Query Documentation Fetching → 95%+ coverage
- **FR-004**: Deduplication and Merging → 99%+ accuracy
- **FR-005**: BrightData Proxy Integration → <5s failover
- **NFR-001**: Performance → 90% downloads complete within 30s

### Testing Strategy
- **Unit Tests**: >90% code coverage target
- **Integration Tests**: All major workflows with Context7 API
- **Performance Tests**: Load testing with timing measurements
- **Security Tests**: Credential handling and proxy TLS validation

## Claude Code Best Practices Integration

Based on Claude Code documentation research:

### File Operations
- Use atomic file operations with temp file + move pattern
- Implement proper error handling with actionable resolution steps
- Structure project with clear separation of concerns

### Development Workflow
- Use `/init` to generate project context documentation
- Leverage CLAUDE.md imports with `@path/to/file.md` syntax
- Implement comprehensive error categorization with user-friendly messages

### Agent Communication
- Provide clear, detailed tool descriptions for agent selection
- Use structured deliverables with consistent naming patterns
- Implement self-critique processes for quality assurance

## Agent Execution Notes

### Prerequisites
- ContextS MCP server must be available for documentation services
- BrightData credentials configured via environment variables
- Python 3.8+ with asyncio support

### Environment Configuration
```bash
# Required credentials
export BRIGHTDATA_CUSTOMER_ID="your_customer_id"
export BRIGHTDATA_PASSWORD="your_password"

# Optional configuration
export CONTEXTER_CONFIG_PATH="~/.contexter/config.yaml"
export CONTEXTER_STORAGE_PATH="~/.contexter/downloads"
export CONTEXTER_LOG_LEVEL="INFO"
```

### Performance Targets
- **Download Speed**: 90% complete within 30 seconds
- **Concurrent Connections**: 10 without degradation
- **Memory Usage**: <512MB during operation
- **Deduplication Speed**: 100 chunks in <5 seconds
- **Compression Ratio**: >60% size reduction
- **Success Rate**: >98% for valid libraries

## Implementation Priorities

**Must Complete for MVP**:
1. ✅ Agent ecosystem definition (42 agents)
2. ✅ Communication protocols and workflows  
3. ⏳ Proxy rotation with health monitoring
4. ⏳ Multi-context download orchestration
5. ⏳ Basic deduplication (hash-based minimum)
6. ⏳ Compressed local storage with integrity checking
7. ⏳ CLI interface with progress reporting

**Risk Mitigation**:
- Context7 API changes → Adaptive rate limiting strategy
- BrightData service disruption → Multi-provider proxy support
- Deduplication accuracy → Manual review capabilities  
- High proxy costs → Cost monitoring and usage limits