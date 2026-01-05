# CodeDreamer Documentation Overview

CodeDreamer is an autonomous code improvement tool that uses Local LLMs to "dream" about your codebase. Point it at a project, let it run, and collect actionable improvement suggestions.

## Documentation Index

| Document | Description |
|----------|-------------|
| [ELI5](eli5.md) | Simple explanation of how CodeDreamer works |
| [Graph Edges](graph_edges.md) | Knowledge graph edge creation and relationships |
| [Proactive Memory](proactive_memory.md) | Anticipatory context fetching system |
| [Applied Dreams](APPLIED_DREAMS.md) | Log of dreams that have been implemented |
| [Changelog](changelog/changelog.md) | Version history and changes |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CodeDreamer                                 │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Indexer    │───►│   Dreamer    │───►│     Validator        │  │
│  │ (ChromaDB)   │    │              │    │ (Novelty Checking)   │  │
│  └──────────────┘    └──────┬───────┘    └──────────────────────┘  │
│                             │                                       │
│           ┌─────────────────┼─────────────────┐                    │
│           ▼                 ▼                 ▼                    │
│  ┌──────────────┐    ┌──────────────┐   ┌─────────────┐           │
│  │ 32B Thinker  │    │  7B Coder    │   │   Critic    │           │
│  │  (Reasoning) │    │  (Generate)  │   │   (Refine)  │           │
│  └──────────────┘    └──────────────┘   └─────────────┘           │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Memory Systems                            │   │
│  │  ┌───────────────┐ ┌─────────┐ ┌───────────────────────┐   │   │
│  │  │Knowledge Graph│ │   TRM   │ │   Proactive Memory    │   │   │
│  │  │(nodes, edges) │ │(stream) │ │   (anticipation)      │   │   │
│  │  └───────────────┘ └─────────┘ └───────────────────────────┘   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Core Modules

- **`dreamer.py`** - Dream generation engine with 4-level drilling
- **`conductor.py`** - Central orchestration layer
- **`indexer.py`** - Codebase parsing and ChromaDB indexing
- **`validator.py`** - Novelty checking and deduplication
- **`graph.py`** - Knowledge graph with temporal decay

### LLM Backends

- **`models.py`** - llama-cpp-python backend (requires GGUF files)
- **`models_ollama.py`** - Ollama API backend (recommended for Mac/Docker)
- **`llm_ollama.py`** - Low-level Ollama client

### Memory Systems

- **`trm.py`** - Temporal Recursive Memory
- **`proactive.py`** - Proactive Memory for anticipated context
- **`scratch.py`** - Working memory buffer

### API & Interface

- **`server.py`** - FastAPI server with WebSocket support
- **`cli.py`** - Command-line interface
- **`static/index.html`** - Live dashboard

## Deployment Options

### 1. Docker with Ollama (Recommended for Mac)

```bash
cd ~/Docker/containers/codedreamer
docker compose up -d
open http://localhost:8088
```

### 2. Local with Pixi

```bash
pixi install
pixi run setup-cuda  # For GPU support
DREAMER_DAEMON_MODE=true pixi run serve
```

### 3. Docker with GGUF Models

```bash
docker compose up -d  # Uses original Dockerfile
```

## Configuration

All settings use the `DREAMER_` prefix for environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DREAMER_DREAM_INTERVAL_SEC` | 300 | Seconds between dream cycles |
| `DREAMER_NOVELTY_THRESHOLD` | 0.4 | Minimum score to save a dream |
| `DREAMER_REASONING_MAX_TOKENS` | 4000 | Max tokens for analysis |
| `DREAMER_CODER_MAX_TOKENS` | 3000 | Max tokens for code generation |

For Ollama backend:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | localhost:11434 | Ollama API endpoint |
| `OLLAMA_REASONING_MODEL` | qwen2.5-coder:32b | Model for reasoning |
| `OLLAMA_CODER_MODEL` | qwen2.5-coder:7b | Model for code generation |
| `OLLAMA_EMBED_MODEL` | nomic-embed-text | Model for embeddings |

## Dream Cycle

1. **Select seed code** - Random chunk from indexed codebase
2. **Inject context** - TRM insights + proactive memory + avoidance themes
3. **Stage 1 (Think)** - 32B model does deep analysis
4. **Stage 2 (Code)** - 7B model generates implementation
5. **Stage 3 (Critic)** - Self-critique and refinement
6. **Validate** - Check novelty, deduplicate
7. **Drill or Reset** - Continue drilling or start fresh discovery
8. **Store** - Save to disk, graph, and leaderboard

## Commands Reference

```bash
codedreamer index <path>           # Index a codebase
codedreamer dream --once           # Single dream cycle
codedreamer serve --port 8080      # Start HTTP server
codedreamer daemon --codebase <p>  # Continuous dreaming
codedreamer briefing --hours 8     # Morning summary
codedreamer graph stats            # Graph statistics
codedreamer dreams actionable      # High-priority dreams
```

