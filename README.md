# CodeDreamer

Autonomous code improvement suggestions via LLM dreaming.

Point it at a codebase. Let it dream while you sleep. Wake up to actionable improvement suggestions.


## To do:
- CRUD the code
- Fix for STR formats (src as code)
- Remove unused functions
- Re-run CodeDreamer over codebase. 

## Quick Start (Docker)

```bash
# Copy and configure .env
cp .env.example .env
# Edit .env with your model and codebase paths

# Start dreaming
docker compose up -d

# View live dashboard
open http://localhost:8080

# Check dreams after a few hours
ls -la ./dreams/

# View morning briefing
docker exec codedreamer codedreamer briefing --hours 8
```

## Features

- **Multi-Model Architecture**: Separate reasoning, coder, and embedding models
- **3-Stage Pipeline**: Reasoning → Coding → Critic Loop for high-quality output
- **Deep Dive Drilling**: 4-level refinement (Discovery → Framework → Implementation → Code)
- **Deep Reasoning**: Configurable token limits for thorough analysis (default 4000+ tokens)
- **Live Dashboard**: Real-time visualization of thinking, knowledge graph, and dream leaderboard
- **Knowledge Graph**: Nodes with edges, temporal decay, and entanglement scoring
- **TRM (Temporal Recursive Memory)**: Carries insights between dream cycles
- **Proactive Memory**: Anticipates context before it's needed
- **Loop Detection**: Prevents repetitive content with active truncation
- **Avoidance Prompts**: Tells model what NOT to generate based on theme history
- **Graph Jump**: Escapes local minima by jumping to random graph nodes
- **Priority System**: CRITICAL/HIGH/MEDIUM/LOW ranking with periodic re-ranking
- **Auto-Indexing**: Automatically indexes codebase on startup when empty
- **Self-Improvement**: Dreams that pass validation can be integrated back into the codebase

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                       DREAM CYCLE                            │
│                                                              │
│  1. Check Drill    →  Continue drilling or start fresh?     │
│  2. Graph Jump?    →  If stuck, jump to random graph node   │
│  3. Select Code    →  Random chunk from indexed codebase    │
│  4. Avoidance      →  Inject "don't repeat these themes"    │
│  5. Proactive Ctx  →  Anticipate imports, related files     │
│  6. TRM Context    →  Pull insights from previous cycles    │
│  7. Stage 1: Think →  Reasoning model deep analysis         │
│  8. Stage 2: Code  →  Coder model generates implementation  │
│  9. Stage 3: Critic→  Self-critique and refine              │
│ 10. Loop Detect    →  Truncate if repetitive content        │
│ 11. Validate       →  Novelty check, deduplication          │
│ 12. Drill Deeper?  →  If good, drill to next level (0→3)   │
│ 13. Store          →  Save to disk, graph, leaderboard      │
│ 14. Connect        →  Auto-create graph edges               │
│                                                              │
│  DRILL LEVELS:                                               │
│  L0: Discovery   → Generate ONE useful insight              │
│  L1: Framework   → Design concrete architecture             │
│  L2: Implementation → Detailed plan with signatures         │
│  L3: Code        → Write production-ready code              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Installation (Local)

NOTE: Docker is prefered. 

```bash
pixi install
pixi run setup-cuda  # For GPU support

# Run with dashboard
DREAMER_MODEL_PATH=/path/to/model.gguf \
DREAMER_CODEBASE_PATH=/path/to/project \
DREAMER_DAEMON_MODE=true \
pixi run codedreamer serve --port 8080
```

## Configuration

All settings via environment variables with `DREAMER_` prefix:

```bash
# === Model Paths ===
DREAMER_MODEL_PATH=/models/your-reasoning-model.gguf
DREAMER_CODER_MODEL_PATH=/models/your-coder-model.gguf  # Optional
DREAMER_EMBED_MODEL_PATH=/models/your-embed-model.gguf  # Optional

# === Deep Reasoning Token Limits ===
DREAMER_REASONING_MAX_TOKENS=4000  # Analysis depth (higher = deeper)
DREAMER_CODER_MAX_TOKENS=3000      # Code generation length

# === GPU/CPU ===
DREAMER_N_GPU_LAYERS=99
DREAMER_N_CTX=8192

# === Dream Cycle ===
DREAMER_DREAM_INTERVAL_SEC=120      # 2 minutes between cycles
DREAMER_DECAY_INTERVAL_SEC=3600     # Graph decay every hour
DREAMER_DREAM_TEMPERATURE=0.8       # Creativity level
DREAMER_NOVELTY_THRESHOLD=0.4       # Min score to save

# === Paths ===
DREAMER_CODEBASE_PATH=/path/to/project
DREAMER_DREAMS_DIR=./dreams

# === Indexing ===
DREAMER_EXCLUDE_PATTERNS=tests,test_*,*_test.py,conftest.py  # Skip test files
DREAMER_CLEAR_INDEX_ON_START=false  # Set true to wipe index on each startup

# === Extended Context Mode (for 32GB+ VRAM) ===
DREAMER_EXTENDED_CONTEXT=true       # Enables: 3x context, import snippets, codebase overview
DREAMER_EXTENDED_CONTEXT_LIMIT=12000  # Context limit when extended mode is on (default 12000)
```

## Commands

```bash
# HTTP API with live dashboard (recommended)
codedreamer serve --port 8080

# Autonomous daemon (headless)
codedreamer daemon --codebase /path/to/project

# Manual operations
codedreamer index /path/to/project     # Index codebase
codedreamer dream --once               # Single dream cycle
codedreamer briefing --hours 12        # Morning briefing

# Knowledge graph
codedreamer graph stats                # Graph statistics (nodes, edges, entanglement)
codedreamer graph hot --limit 10       # Hottest nodes by momentum
codedreamer graph entangled --limit 10 # Most connected nodes
codedreamer graph backfill             # Create edges for existing nodes
codedreamer graph decay                # Apply temporal decay
codedreamer graph prune                # Remove cold nodes

# Dream management
codedreamer dreams list                # List all dreams with priorities
codedreamer dreams actionable          # Show HIGH/CRITICAL dreams
codedreamer dreams mark <id> applied   # Mark dream as applied
codedreamer dreams mark <id> rejected  # Mark dream as rejected
codedreamer dreams reflect             # Trigger priority re-ranking
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DREAM DAEMON                            │
│                                                              │
│  ┌─────────┐   ┌───────────┐   ┌───────────┐   ┌─────────┐ │
│  │ Indexer │──►│ Conductor │──►│ Validator │──►│ Storage │ │
│  └─────────┘   └─────┬─────┘   └─────┬─────┘   └─────────┘ │
│                      │               │                       │
│       ┌──────────────┼───────────────┤                      │
│       ▼              ▼               ▼                      │
│  ┌─────────┐   ┌───────────┐   ┌─────────┐                 │
│  │Thinker  │   │  Coder    │   │ Critic  │                 │
│  │ Model   │   │  Model    │   │  Loop   │                 │
│  └─────────┘   └───────────┘   └─────────┘                 │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Memory Systems                        │ │
│  │  ┌──────────────┐  ┌─────────┐  ┌───────────────────┐ │ │
│  │  │Knowledge Graph│  │   TRM   │  │ Proactive Memory │ │ │
│  │  │(edges, decay) │  │(stream) │  │  (anticipation)  │ │ │
│  │  └──────────────┘  └─────────┘  └───────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────┐   ┌─────────────────┐                  │
│  │   Leaderboard   │   │  Dream Storage  │                  │
│  │  (top dreams)   │   │  (./dreams/*.md)│                  │
│  └─────────────────┘   └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Dashboard

Access at `http://localhost:8080`:

- **Live Thought Stream**: Watch the model think in real-time
- **Knowledge Graph**: Interactive visualization with edges showing connections
- **Dream Timeline**: Recent dreams with click-to-expand
- **Leaderboard**: Top-ranked dreams by novelty score
- **Neural Activity**: Token generation histogram over time
- **Graph Stats**: Nodes, edges, and average entanglement

## Inspired and dreamed features

CodeDreamer implements concepts

| Feature | Description |
|---------|-------------|
| **Deep Dive Drilling** | 4-level refinement: Discovery → Framework → Implementation → Code |
| **Avoidance Prompts** | Inject overused themes into prompt so model knows what NOT to generate |
| **Graph Jump** | After 3 consecutive failures, jump to random graph node to escape local minima |
| **Loop Detection** | Active detection of repetitive content during generation |
| **Repeat Penalty** | 1.15 penalty to prevent degenerate loops |
| **TRM** | Temporal Recursive Memory - insights decay over time, reinforced when accessed |
| **Critic Loop** | Self-critique before accepting output |
| **Entanglement** | Measure how connected ideas are in the knowledge graph |
| **Proactive Memory** | Anticipate needed context before model asks |
| **Priority Ranking** | Reflection cycle assigns CRITICAL/HIGH/MEDIUM/LOW priorities |

## Documentation

See `docs/` for detailed documentation:

- `eli5.md` - Simple explanation of how it works
- `graph_edges.md` - Knowledge graph edge creation
- `proactive_memory.md` - Anticipatory context fetching

## Development

```bash
pixi run test      # Run tests
pixi run lint      # Linting
pixi run typecheck # Type checking
```

## License

MIT
# CodeDreamer
