# CodeDreamer: ELI5

## What is CodeDreamer?

CodeDreamer is an AI system that **continuously analyzes your codebase and suggests improvements** while you're not actively working on it. Think of it as a tireless code reviewer that works in the background.

## The Core Idea

Traditional code review happens when you ask for it. CodeDreamer flips this:

```
Traditional:  You write code → You ask for review → AI responds
CodeDreamer:  AI watches code → AI thinks continuously → AI offers ideas when ready
```

## How It Works (Simple Version)

1. **Index** - CodeDreamer reads all your code and understands what each function/class does
2. **Anticipate** - Before dreaming, it gathers context it thinks will be useful
3. **Dream** - Every 2 minutes, it picks a random piece of code and thinks: "How could this be better?"
4. **Critique** - It questions its own suggestion: "Is this actually good?"
5. **Validate** - It checks if the idea is novel and not something it already suggested
6. **Save** - Good ideas become "dreams" saved as markdown files
7. **Connect** - New ideas are linked to related previous ideas in the knowledge graph

## The 3-Stage Pipeline

Each dream goes through three stages:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: THINKER                                            │
│  "Deep analysis - what's wrong and how to fix it"           │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: CODER                                              │
│  "Generate actual implementation code"                       │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: CRITIC                                             │
│  "Wait, is this actually good? Let me check..."             │
└─────────────────────────────────────────────────────────────┘
```

## The Components

### The Dreamer (dreamer.py)
The creative engine. It takes code snippets and generates improvement ideas using an LLM.

### The Validator (validator.py)
The quality filter. It asks:
- Is this idea novel? (not too similar to recent ideas)
- Is it actually useful? (passes a "lucid check" - asking the LLM to self-critique)
- Is it actionable? (can someone implement this?)

### The Knowledge Graph (graph.py)
The memory system. Stores facts, concepts, and dreams with **temporal decay**:
- Hot nodes: Recently accessed, high relevance
- Warm nodes: Still useful, medium relevance  
- Cold nodes: Fading, candidates for removal

**NEW:** Nodes are now connected with edges:
- `same_file` edges: Link ideas about the same file
- `same_cycle` edges: Link ideas discovered together

**Entanglement:** Measures how connected a node is. Highly connected = more central/valuable.

### TRM - Temporal Recursive Memory (trm.py)
A stream of recent thoughts that carries forward between dream cycles:
- Each insight has a "salience" that decays over time
- Frequently accessed insights get reinforced
- Creates compound insights by building on previous observations

### Proactive Memory (proactive.py)
Anticipates what context will be useful before dreaming:
- Analyzes imports to understand dependencies
- Finds related files from graph edges
- Pulls relevant previous insights
- Pre-builds context so the model doesn't start "cold"

### The Indexer (indexer.py)
The code reader. Uses **AST parsing** to extract:
- Functions (complete semantic units, including decorators)
- Classes (with all their methods)
- Docstrings (for better semantic search)

This is better than splitting code arbitrarily because each chunk represents a complete concept.

**Auto-indexing**: On startup, if the index is empty, CodeDreamer automatically indexes the configured codebase. No manual `codedreamer index` needed.

### The Conductor (conductor.py)
The orchestrator. Routes requests to the right handler:
- `DREAM` → Generate improvement ideas
- `CODE` → Generate actual code
- `PLAN` → Create multi-step plans
- `REFLECT` → Analyze and critique

### The Models (models.py)
Manages three LLMs with lazy loading:
- **Thinker**: For deep analysis and idea generation
- **Coder**: For actual code generation (optional, falls back to Thinker)
- **Embed**: For semantic search (optional)

## Key Mechanisms

### Critic Loop
Before accepting any dream, the system asks itself:
- "Is this specific enough?"
- "Are the claims accurate?"
- "Is this actually implementable?"
- "Did I miss anything important?"

If issues are found, it refines the suggestion before saving.

### Deep Dive
When a good idea is found, CodeDreamer "drills down" through levels:
1. **Discovery**: High-level idea ("this could be faster")
2. **Framework**: Approach outline ("use caching here")
3. **Implementation**: Detailed design ("add LRU cache with TTL")
4. **Code**: Actual code snippet

### Graph Jump
When stuck in repetitive ideas (3+ consecutive discards), CodeDreamer jumps to a completely different part of the codebase. Prevents getting trapped in local optima.

### Avoidance Prompts
Before generating a dream, CodeDreamer tells the model what NOT to suggest:
- "AVOID these overused topics: caching, error handling, logging"
- This prevents the model from repeating the same suggestions

### Loop Detection
During generation, CodeDreamer watches for:
- Repeated conclusion phrases ("Therefore", "In summary")
- Repeated 50-character blocks
- Excessive conclusions (2+ summaries = stop)

If detected, output is truncated before it degenerates.

### Priority System
Dreams are automatically prioritized:
- **CRITICAL**: Security vulnerabilities, data loss risks
- **HIGH**: Real bugs, performance issues
- **MEDIUM**: Code quality, maintainability
- **LOW**: Style, cosmetic improvements

A reflection cycle periodically re-ranks all dreams as context grows.

### Entanglement Scoring
Ideas that connect to many other ideas are more valuable:
```
codedreamer graph entangled --limit 10
```
Shows the most connected nodes in the knowledge graph.

## The Dashboard

A web UI at `http://localhost:8080` showing:
- **Live thought stream**: What the AI is currently generating
- **Model indicator**: Which LLM is active (Thinker/Coder/Critic)
- **Dream timeline**: History of saved dreams (click to expand)
- **Knowledge graph**: Visualization of connected concepts with edges
- **Leaderboard**: Top dreams ranked by novelty
- **Countdown timer**: Seconds until next dream cycle
- **Graph stats**: Nodes, edges, and average entanglement

## Running It

```bash
# Start the server
pixi run codedreamer serve --port 8080

# With daemon mode (automatic dreaming)
DREAMER_DAEMON_MODE=true pixi run codedreamer serve

# Dreams are saved to ./dreams/*.md
```

## Why "Dreams"?

The metaphor is intentional:
- Dreams happen in the background while you're not actively engaged
- Dreams connect unrelated concepts in novel ways
- Some dreams are useful insights; others are discarded
- Dreams decay if not reinforced (temporal decay in the knowledge graph)
- Dreams build on each other (TRM carries context forward)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Dashboard                            │
│              (WebSocket for real-time updates)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
│                    (server.py, daemon)                       │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ Dreamer  │    │Conductor │    │ Indexer  │
       │  (idea   │    │ (route   │    │  (parse  │
       │generator)│    │requests) │    │  code)   │
       └──────────┘    └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
       ┌──────────────────────────────────────────┐
       │              Model Orchestra              │
       │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
       │  │Thinker  │ │  Coder  │ │ Embed   │    │
       │  │ Model   │ │  Model  │ │ Model   │    │
       │  └────┬────┘ └────┬────┘ └─────────┘    │
       │       │           │                      │
       │       └─────┬─────┘                      │
       │             ▼                            │
       │       ┌──────────┐                       │
       │       │  Critic  │                       │
       │       │   Loop   │                       │
       │       └──────────┘                       │
       └──────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │Knowledge │    │   TRM    │    │Proactive │
       │  Graph   │    │ (memory  │    │ Memory   │
       │(w/edges) │    │  stream) │    │(context) │
       └──────────┘    └──────────┘    └──────────┘
```

## Self-Contained Design

CodeDreamer is designed to run **entirely locally** with no external services:
- Models: Local GGUF files via llama-cpp-python
- Storage: Local ChromaDB + JSON files
- GPU: CUDA acceleration via Pixi environment

No API keys, no cloud dependencies, no data leaving your machine.

## Heritage

CodeDreamer implements concepts from cognitive architecture:

| Concept | What It Does |
|---------|--------------|
| **Deep Dive Drilling** | 4 levels: Discovery → Framework → Implementation → Code |
| **Avoidance Prompts** | Tell model what NOT to generate (overused themes) |
| **Graph Jump** | Escape local optima by jumping to random graph node |
| **Loop Detection** | Active truncation of repetitive content |
| **Repeat Penalty** | 1.15 penalty prevents degenerate loops |
| **TRM** | Memory that decays over time, reinforced by use |
| **Critic Loop** | Self-critique before accepting output |
| **Entanglement** | Value ideas by how connected they are |
| **Proactive Memory** | Anticipate context before it's needed |
| **Priority Ranking** | Reflection assigns CRITICAL/HIGH/MEDIUM/LOW |
