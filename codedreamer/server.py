"""
FastAPI server for CodeDreamer.

Provides HTTP API for:
- Processing requests through the conductor
- Querying the knowledge graph
- Managing dreams and briefings
- Health and status checks
- WebSocket for real-time thought streaming

When DREAMER_DAEMON_MODE=true, also runs the dream scheduler in-process.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .conductor import TaskType, get_conductor
from .config import settings
from .graph import KnowledgeGraph, NodeType, get_graph
from .leaderboard import get_leaderboard
from .models import get_orchestra
from .scratch import get_scratch
from .trm import get_trm

# Global scheduler for daemon mode
_scheduler: BackgroundScheduler | None = None

# Lock to prevent concurrent model access (dream cycle vs reflection)
import threading
_model_lock = threading.Lock()

logger = logging.getLogger(__name__)


# ============================================
# Real-time state tracking for dashboard
# ============================================
@dataclass
class ThinkingState:
    """Tracks current model thinking state for dashboard visualization."""

    active_model: str | None = None
    current_prompt: str = ""
    current_output: str = ""
    token_count: int = 0
    is_generating: bool = False
    start_time: float = 0.0

    # Recent token rates for brainwave chart
    token_rates: list[float] = field(default_factory=lambda: [0.0] * 100)


# Global state singleton
_thinking_state = ThinkingState()


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        data = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager
_ws_manager = ConnectionManager()


def get_thinking_state() -> ThinkingState:
    """Get the global thinking state."""
    return _thinking_state


def get_ws_manager() -> ConnectionManager:
    """Get the global WebSocket manager."""
    return _ws_manager


# Validation service (Dream #3: dream_20251228_112357_code_idea.md)
class TaskTypeValidator:
    """
    Validates and converts task_type strings to TaskType enum.

    Used with FastAPI's dependency injection for clean separation of concerns.
    """

    VALID_TYPES = [t.name for t in TaskType]

    def validate(self, task_type: str | None) -> TaskType | None:
        """
        Validate and convert a task type string.

        Args:
            task_type: String task type or None.

        Returns:
            TaskType enum or None if not provided.

        Raises:
            HTTPException: If task_type is invalid.
        """
        if task_type is None:
            return None

        task_type_upper = task_type.upper()
        if task_type_upper not in self.VALID_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type '{task_type}'. Must be one of: {self.VALID_TYPES}",
            )

        return TaskType[task_type_upper]


def get_task_validator() -> TaskTypeValidator:
    """Dependency provider for TaskTypeValidator."""
    return TaskTypeValidator()


# Request/Response models
class ProcessRequest(BaseModel):
    """Request to process through the conductor."""

    prompt: str = Field(..., min_length=1, max_length=10000)
    task_type: str | None = Field(None, description="Optional: QUERY, CODE, DREAM, PLAN, REFLECT")


class ProcessResponse(BaseModel):
    """Response from conductor processing."""

    success: bool
    output: str
    task_type: str
    steps_taken: int
    duration_ms: int
    error: str | None = None


class DreamRequest(BaseModel):
    """Request to generate a dream."""

    seed_text: str | None = Field(None, description="Optional seed context for dreaming")
    temperature: float = Field(0.9, ge=0.0, le=2.0)


class DreamResponse(BaseModel):
    """Response from dream generation."""

    content: str
    category: str
    novelty_score: float
    timestamp: str


class GraphQueryRequest(BaseModel):
    """Request to query the knowledge graph."""

    node_type: str | None = Field(None, description="Filter by type: FACT, CONCEPT, CODE, DREAM")
    min_momentum: float = Field(0.0, ge=0.0, le=1.0)
    limit: int = Field(10, ge=1, le=100)


class GraphNode(BaseModel):
    """A node from the knowledge graph."""

    id: str
    content: str
    node_type: str
    momentum: float
    tier: str
    age_hours: float


class GraphStatsResponse(BaseModel):
    """Statistics about the knowledge graph."""

    total_nodes: int
    total_edges: int
    avg_entanglement: float = 0.0
    tiers: dict[str, int]
    types: dict[str, int]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: dict[str, bool]
    graph_nodes: int
    active_tasks: int
    thinking: dict[str, Any] | None = None
    next_dream_seconds: int | None = None  # Seconds until next dream cycle
    trm_stats: dict[str, Any] | None = None  # TRM stream statistics


class StatusResponse(BaseModel):
    """Detailed status response."""

    config: dict[str, Any]
    models: dict[str, bool]
    graph: dict[str, Any]
    scratch: str


class LeaderboardEntryResponse(BaseModel):
    """A single leaderboard entry."""

    rank: int
    content: str
    category: str
    novelty_score: float
    source_file: str
    timestamp: str
    dream_id: str


class LeaderboardResponse(BaseModel):
    """Leaderboard of top dreams."""

    entries: list[LeaderboardEntryResponse]
    total_dreams: int


def _extract_code_entities(code: str, source_file: str, graph: KnowledgeGraph) -> int:
    """
    Extract code entities (functions, classes) from code and add as ENTITY nodes.
    
    Returns:
        Number of entities added.
    """
    import re
    
    entities_added = 0
    
    # Extract Python function definitions
    func_pattern = r"def\s+(\w+)\s*\([^)]*\):"
    for match in re.finditer(func_pattern, code):
        func_name = match.group(1)
        if not func_name.startswith("_"):  # Skip private functions
            graph.add_node(
                content=f"function: {func_name} (in {source_file})",
                node_type=NodeType.ENTITY,
                metadata={"entity_type": "function", "file": source_file, "name": func_name},
            )
            entities_added += 1
    
    # Extract class definitions
    class_pattern = r"class\s+(\w+)\s*[:\(]"
    for match in re.finditer(class_pattern, code):
        class_name = match.group(1)
        graph.add_node(
            content=f"class: {class_name} (in {source_file})",
            node_type=NodeType.ENTITY,
            metadata={"entity_type": "class", "file": source_file, "name": class_name},
        )
        entities_added += 1
    
    # Extract facts about imports (external dependencies)
    import_pattern = r"^(?:from\s+(\S+)|import\s+(\S+))"
    imports_found = set()
    for match in re.finditer(import_pattern, code, re.MULTILINE):
        module = match.group(1) or match.group(2)
        if module and module not in imports_found:
            imports_found.add(module.split(".")[0])  # Top-level module only
    
    # Add unique imports as FACT nodes (just top 3)
    for module in list(imports_found)[:3]:
        if module not in {"os", "sys", "typing", "re"}:  # Skip stdlib
            graph.add_node(
                content=f"depends on: {module}",
                node_type=NodeType.FACT,
                metadata={"fact_type": "dependency", "module": module, "file": source_file},
            )
            entities_added += 1
    
    return entities_added


# Dream cycle for daemon mode
def _run_dream_cycle() -> None:
    """Run a single dream cycle (called by scheduler)."""
    # Acquire lock to prevent concurrent model access with reflection
    if not _model_lock.acquire(blocking=False):
        logger.info("Skipping dream cycle - model in use by another cycle")
        return
    
    try:
        _run_dream_cycle_impl()
    finally:
        _model_lock.release()


def _run_dream_cycle_impl() -> None:
    """Actual dream cycle implementation."""
    from .indexer import CodebaseIndexer
    from .validator import get_validator

    codebase = settings.codebase_path
    if not codebase or not codebase.exists():
        logger.warning("No codebase configured for dreaming")
        return

    conductor = get_conductor()
    graph = get_graph()
    validator = get_validator()  # Use shared instance for deduplication
    
    # Initialize indexer first (needed for project terms)
    indexer = CodebaseIndexer()
    
    # Initialize project-specific domain terms for novelty scoring
    validator.initialize_project_terms(indexer)

    # Get drill state for deep dive drilling
    from .drill_state import get_drill_state, get_random_graph_context
    drill = get_drill_state()
    
    # Check for graph jump (after too many consecutive failures)
    using_graph_jump = False
    graph_context = ""
    
    if drill.should_graph_jump():
        logger.info("Plateau detected - jumping to random graph node")
        graph_context, graph_source = get_random_graph_context(graph)
        if graph_context:
            using_graph_jump = True
            drill.record_graph_jump()
    
    # If drilling, try to stay on same file
    chunk = None
    if drill.depth > 0 and drill.source_file:
        # Continue drilling on same file
        chunks = indexer.query(drill.source_file, n_results=3)
        if chunks:
            import random
            chunk = random.choice(chunks)
            logger.info(f"Continuing drill (depth {drill.depth}) on {drill.source_file}")
    
    if not chunk:
        # Try to get a random chunk, fallback to query
        chunk = indexer.get_random_chunk()
        if not chunk:
            # Fallback to query
            chunks = indexer.query("code patterns functions classes", n_results=5)
            if not chunks:
                logger.warning("No chunks found for dreaming")
                return
            import random
            chunk = random.choice(chunks)

    chunk_source = chunk.file_path or "unknown"
    mode_str = f"[DRILL L{drill.depth}]" if drill.depth > 0 else ("[GRAPH JUMP]" if using_graph_jump else "")
    logger.info(f"Dreaming {mode_str} about: {chunk_source[:50]}...")

    # Update thinking state for dashboard
    thinking = get_thinking_state()
    thinking.active_model = "14B"
    thinking.is_generating = True
    thinking.current_prompt = f"Analyzing: {chunk_source}"
    thinking.token_count = 0

    # Generate improvement suggestion using multi-model pipeline
    chunk_content = chunk.content or ""
    orchestra = get_orchestra()
    trm = get_trm()

    # Get TRM context (insights from previous dreams)
    trm_context = trm.get_context(max_fragments=3)
    trm.tick()  # Apply temporal decay

    # Get Proactive Memory context (anticipated relevant info)
    from .proactive import get_proactive_memory
    proactive = get_proactive_memory()
    proactive_ctx = proactive.get_context(chunk_source, chunk_content)
    proactive_section = proactive_ctx.to_prompt_section()
    
    # Log if proactive context is being used
    if proactive_section:
        logger.info(f"Proactive context included ({len(proactive_section)} chars)")
    else:
        logger.warning("No proactive context available - cold start?")

    # === STAGE 1: REASONING MODEL (14B) - Deep Analysis ===
    thinking.active_model = "14B"

    # Build prompt with TRM and Proactive context if available
    context_section = ""
    if trm_context or proactive_section:
        context_parts = []
        if trm_context:
            context_parts.append(f"""## Previous Insights (TRM)
{trm_context}
""")
        if proactive_section:
            context_parts.append(proactive_section)

        context_section = "\n".join(context_parts) + """
Consider how this context might relate to the code below.
Look for patterns, connections, or opportunities to build on prior observations.

---

"""
    # Legacy support for trm_section variable
    trm_section = context_section

    # Get avoidance prompt (themes to NOT repeat)
    avoidance_prompt = validator.get_avoidance_prompt()
    
    # Get drill context (if continuing a deep dive)
    drill_context = drill.get_context_for_drilling()
    
    # Get the appropriate task prompt for current drill level
    drill_task = drill.get_drill_prompt()

    # Build the analysis prompt
    if using_graph_jump and graph_context:
        # Graph jump mode: explore new territory
        analysis_prompt = f"""You are a senior software architect exploring unfamiliar code.
{trm_section}
{graph_context}
{avoidance_prompt}

## Fresh Code to Explore
```
{chunk_content[:settings.prompt_context_limit]}
```

## Your Task
{drill_task}

Be SPECIFIC. Be CONCRETE. Be NOVEL."""

    elif drill.depth > 0 and drill_context:
        # Deep dive mode: build on previous insight
        analysis_prompt = f"""You are in a DEEP DIVE session, refining an insight into actionable code.

{drill_context}
{avoidance_prompt}

## Related Code from {chunk_source}:
```
{chunk_content[:settings.prompt_context_limit]}
```

## Your Task (Level {drill.depth})
{drill_task}

Build DIRECTLY on your previous insight. Make it MORE CONCRETE and ACTIONABLE."""

    else:
        # Discovery mode: normal analysis
        analysis_prompt = f"""You are a senior software architect performing a deep code review.
{trm_section}
{avoidance_prompt}

## Task
{drill_task}

## Code from {chunk_source} (may be a fragment):
```
{chunk_content[:settings.prompt_context_limit]}
```

## Your Analysis Should Include:

1. **Current State Assessment** (2-3 sentences)
   - What is this code doing?
   - What patterns/paradigms is it using?

2. **Identified Issues** (be specific)
   - Name exact functions, classes, or lines
   - Explain WHY each is problematic
   - Consider: maintainability, performance, readability, testability

3. **Proposed Improvement** (detailed)
   - What specific change would you make?
   - Why is this better than the current approach?
   - What are the trade-offs?

4. **Implementation Strategy**
   - Step-by-step approach to implement
   - What tests would validate the change?

Think deeply. Take your time. Quality over brevity."""

    try:
        from .loop_detector import detect_and_truncate, get_discard_tracker
        discard_tracker = get_discard_tracker()
        
        logger.info("Stage 1: Deep analysis with reasoning model...")
        analysis = orchestra.reason(analysis_prompt, temperature=0.8, max_tokens=settings.reasoning_max_tokens)
        
        # Loop detection: truncate if repetition detected
        analysis, loop_result = detect_and_truncate(analysis)
        if loop_result:
            logger.warning(f"Truncated analysis due to {loop_result.loop_type}: {loop_result.details}")
        
        thinking.current_output = analysis[:300] if analysis else ""
        thinking.token_count = len(analysis.split()) if analysis else 0

        # === STAGE 2: CODER MODEL (7B) - Generate implementation ===
        # Only if we have a coder model and the analysis looks actionable
        code_output = None
        if orchestra._coder is not None and analysis and len(analysis) > 200:
            thinking.active_model = "7B"
            logger.info("Stage 2: Generating implementation with coder model...")

            code_prompt = f"""You are an expert Python developer implementing a code improvement.

## Analysis & Improvement Plan:
{analysis[:1200]}

## Original Code Context (may be a fragment of a larger file):
```python
{chunk_content[:settings.prompt_context_limit]}
```

## Your Task:
Write the complete improved implementation based on the analysis above.

Requirements:
- Include proper type hints
- Add docstrings where appropriate
- Handle edge cases
- Follow Python best practices
- Make the code production-ready

Output ONLY the improved code, no explanations:
```python"""

            code_output = orchestra.code(code_prompt, temperature=0.4, max_tokens=settings.coder_max_tokens)
            
            # Loop detection on code output
            code_output, code_loop = detect_and_truncate(code_output)
            if code_loop:
                logger.warning(f"Truncated code output due to {code_loop.loop_type}: {code_loop.details}")
            
            thinking.current_output = code_output[:300] if code_output else ""
            thinking.token_count += len(code_output.split()) if code_output else 0

        # Combine analysis and code into final output
        if code_output and len(code_output.strip()) > 50:
            final_output = f"{analysis}\n\n## Suggested Implementation\n\n```python\n{code_output}\n```"
        else:
            final_output = analysis

        # === STAGE 3: CRITIC LOOP - Self-critique and refine ===
        thinking.active_model = "Critic"
        logger.info("Stage 3: Running critic loop for self-refinement...")

        refined_output, critique_result = validator.critique_and_refine(
            final_output, max_refinements=1
        )

        if critique_result and critique_result.refined_content:
            logger.info(
                f"Critic refined dream: severity={critique_result.severity}, "
                f"len {len(final_output)} → {len(refined_output)}"
            )
            final_output = refined_output
            thinking.current_output = f"[REFINED] {refined_output[:250]}"
        elif critique_result:
            logger.debug(f"Critic verdict: {critique_result.severity} - no refinement needed")

        thinking.token_count += len(final_output.split())

        # Create a result-like object
        class DreamResult:
            def __init__(self, output: str, was_refined: bool = False) -> None:
                self.success = True
                self.output = output
                self.was_refined = was_refined

        result = DreamResult(
            final_output,
            was_refined=critique_result.refined_content is not None if critique_result else False
        )

    except Exception as e:
        logger.error(f"Dream generation failed: {e}")

        class FailedResult:
            success = False
            output = None

        result = FailedResult()
    finally:
        # Clear thinking state when done
        thinking.is_generating = False
        thinking.active_model = None

    if result.success and result.output:
        # Validate and save (with source file for cooldown tracking)
        validated = validator.validate(result.output, source_file=chunk_source)
        if validated.novelty_score >= 0.3:
            # Save to disk
            category = validated.category.replace(" ", "_").lower()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dream_dir = settings.dreams_dir
            dream_dir.mkdir(exist_ok=True)

            filename = f"dream_{ts}_{category}.md"
            filepath = dream_dir / filename

            # Build theme warnings section if any
            warnings_section = ""
            if validated.theme_warnings:
                warnings_list = "\n".join(f"- {w}" for w in validated.theme_warnings)
                warnings_section = f"""
> **Theme Repetition Warning**
> {warnings_list}

"""

            content = f"""# {validated.category.title()}

**Generated**: {datetime.now().isoformat()}
**Novelty Score**: {validated.novelty_score:.2f}
**Source File**: {chunk_source}

---
{warnings_section}
{result.output}

---

## Seed Context

```
{chunk_content[:500]}
```
"""
            filepath.write_text(content)
            logger.info(f"Saved: {filename}")

            # Add insight to TRM stream for future dreams
            trm.add_insight(
                content=result.output[:500],
                source_file=chunk_source,
                category=category,
                salience=validated.novelty_score,
            )
            logger.debug(f"TRM: Added insight from {chunk_source}")

            # Submit to leaderboard
            leaderboard = get_leaderboard()
            new_rank = leaderboard.submit(
                content=result.output,
                category=category,
                novelty_score=validated.novelty_score,
                source_file=chunk_source,
                dream_id=filename,
            )
            if new_rank:
                logger.info(f"Leaderboard update: #{new_rank}")

            # Add dream to graph
            graph.add_node(
                content=result.output[:500],
                node_type=NodeType.DREAM,
                metadata={
                    "category": category,
                    "source": chunk_source,
                    "novelty": validated.novelty_score,
                },
            )
            
            # Extract and add code entities (functions, classes) as ENTITY nodes
            _extract_code_entities(chunk_content, chunk_source, graph)
            
            # Extract concepts/patterns mentioned in the dream as CONCEPT nodes
            if "pattern" in result.output.lower() or "architecture" in result.output.lower():
                concept_lines = [
                    line.strip() for line in result.output.split("\n") 
                    if any(kw in line.lower() for kw in ["pattern", "principle", "approach", "design"])
                    and len(line.strip()) > 30
                ]
                for concept in concept_lines[:2]:  # Max 2 concepts per dream
                    graph.add_node(
                        content=concept[:300],
                        node_type=NodeType.CONCEPT,
                        metadata={"source": "dream_extraction", "file": chunk_source},
                    )
            
            graph.save()
            
            # Record success - reset discard counter
            discard_tracker.record_success()
            
            # Deep Dive Drilling: check if we should drill deeper
            if drill.should_drill_deeper(validated.category, validated.novelty_score):
                drill.drill_down(result.output, chunk_source)
                logger.info(f"Drilling deeper to level {drill.depth} on {chunk_source[:40]}")
            else:
                # End of drill, reset for next discovery
                if drill.depth > 0:
                    logger.info(f"Drill complete: reached depth {drill.depth} on {drill.source_file}")
                drill.reset()
                drill.record_success()  # Reset consecutive discard counter
        else:
            logger.info(f"Dream discarded (novelty {validated.novelty_score:.2f} < 0.3)")
            
            # Track discards for drilling state
            drill.record_discard()
            
            # Track discards - reset state if too many consecutive
            if discard_tracker.record_discard(f"novelty={validated.novelty_score:.2f}"):
                logger.warning("Too many consecutive discards - resetting dreamer state")
                # Reset proactive memory focus to break out of stuck patterns
                from .proactive import get_proactive_memory
                proactive = get_proactive_memory()
                proactive._last_focus = None
                discard_tracker.reset()
                drill.reset()


def _run_decay_cycle() -> None:
    """Apply decay to graph nodes."""
    graph = get_graph()
    decayed = graph.decay_all()
    if decayed > 0:
        logger.info(f"Decayed {decayed} nodes below threshold")
        graph.save()


def _run_reflection_cycle() -> None:
    """Run the reflection cycle to prioritize unranked dreams."""
    # Acquire lock to prevent concurrent model access with dream cycle
    if not _model_lock.acquire(blocking=False):
        logger.info("Skipping reflection cycle - model in use by dream cycle")
        return
    
    try:
        from .reflection import get_reflection_cycle
        
        orchestra = get_orchestra()
        reflection = get_reflection_cycle(orchestra)
        prioritized = reflection.run()
        
        if prioritized > 0:
            logger.info(f"Reflection cycle prioritized {prioritized} dreams")
    finally:
        _model_lock.release()


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Handle startup and shutdown."""
    global _scheduler
    logger.info("CodeDreamer server starting...")

    # Initialize singletons
    get_orchestra()
    get_graph()
    get_scratch()
    get_conductor()

    # Start daemon scheduler if enabled
    daemon_mode = os.environ.get("DREAMER_DAEMON_MODE", "").lower() == "true"
    if daemon_mode:
        interval = settings.dream_interval_sec
        logger.info(f"Daemon mode enabled. Dream interval: {interval}s")

        # Auto-index codebase on startup if configured
        codebase = settings.codebase_path
        if codebase and codebase.exists():
            from .indexer import CodebaseIndexer
            indexer = CodebaseIndexer()
            
            # Clear index on start if configured (useful when changing exclude patterns)
            if settings.clear_index_on_start:
                logger.info("Clearing index on startup (DREAMER_CLEAR_INDEX_ON_START=true)")
                indexer.clear()
                # Also clear the graph to start completely fresh
                graph = get_graph()
                graph.clear()
                logger.info("Index and graph cleared")
            
            # Check if index is empty and needs rebuilding
            try:
                chunk = indexer.get_random_chunk()
                if not chunk:
                    logger.info(f"Auto-indexing codebase: {codebase}")
                    indexer.index_directory(codebase)
                    logger.info("Auto-indexing complete")
            except Exception as e:
                logger.warning(f"Auto-index check failed: {e}, indexing anyway")
                indexer.index_directory(codebase)

        from datetime import datetime

        _scheduler = BackgroundScheduler()
        # Run dream cycle immediately on startup, then at intervals
        _scheduler.add_job(_run_dream_cycle, "interval", seconds=interval, max_instances=1, next_run_time=datetime.now())
        _scheduler.add_job(_run_decay_cycle, "interval", seconds=3600)
        # Reflection cycle runs every 10 dream cycles to re-prioritize
        reflection_interval = interval * 10
        _scheduler.add_job(_run_reflection_cycle, "interval", seconds=reflection_interval, max_instances=1)
        _scheduler.start()
        logger.info(f"Dream scheduler started (reflection every {reflection_interval}s)")

    logger.info("Server ready")
    yield

    # Cleanup
    logger.info("Server shutting down...")

    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    orchestra = get_orchestra()
    orchestra.unload_all()

    graph = get_graph()
    graph.save()

    logger.info("Shutdown complete")


# Create app
app = FastAPI(
    title="CodeDreamer",
    description="Autonomous code improvement suggestions via LLM dreaming",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Root - serve dashboard
@app.get("/")
async def root() -> FileResponse:
    """Serve the dashboard."""
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============================================
# WebSocket endpoint for real-time updates
# ============================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time thought streaming."""
    manager = get_ws_manager()
    await manager.connect(websocket)

    try:
        while True:
            # Keep connection alive and listen for client messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                # Handle ping/pong
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_thinking_update(
    model: str,
    prompt: str,
    output: str,
    token_count: int,
    is_generating: bool,
) -> None:
    """Broadcast a thinking update to all connected clients."""
    state = get_thinking_state()
    manager = get_ws_manager()

    state.active_model = model if is_generating else None
    state.current_prompt = prompt
    state.current_output = output
    state.token_count = token_count
    state.is_generating = is_generating

    await manager.broadcast({
        "type": "thinking",
        "model": model,
        "prompt": prompt[:200] if prompt else "",
        "output": output[-500:] if output else "",
        "token_count": token_count,
        "is_generating": is_generating,
    })


async def broadcast_dream_event(
    event_type: str,
    category: str | None = None,
    novelty: float | None = None,
    content: str | None = None,
) -> None:
    """Broadcast a dream event to all connected clients."""
    manager = get_ws_manager()

    await manager.broadcast({
        "type": "dream_event",
        "event": event_type,
        "category": category,
        "novelty": novelty,
        "content": content[:200] if content else None,
        "timestamp": datetime.now().isoformat(),
    })


# ============================================
# REST Endpoints
# ============================================
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with thinking state."""
    orchestra = get_orchestra()
    graph = get_graph()
    scratch = get_scratch()
    state = get_thinking_state()

    # Calculate seconds until next dream cycle
    next_dream_seconds = None
    if _scheduler:
        jobs = _scheduler.get_jobs()
        for job in jobs:
            if job.name == "_run_dream_cycle" and job.next_run_time:
                from datetime import datetime, timezone

                now = datetime.now(timezone.utc)
                delta = job.next_run_time - now
                next_dream_seconds = max(0, int(delta.total_seconds()))
                break

    # Get TRM stats
    trm = get_trm()
    trm_stats = trm.get_stats()

    return HealthResponse(
        status="healthy",
        models_loaded=orchestra.status(),
        graph_nodes=len(graph._nodes),
        active_tasks=len(scratch._tasks),
        thinking={
            "active_model": state.active_model,
            "is_generating": state.is_generating,
            "token_count": state.token_count,
            "current_prompt": state.current_prompt[:100] if state.current_prompt else "",
            "current_output": state.current_output[:300] if state.current_output else "",
        },
        next_dream_seconds=next_dream_seconds,
        trm_stats=trm_stats,
    )


@app.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard_endpoint() -> LeaderboardResponse:
    """Get the dream leaderboard - top ranked dreams by novelty."""
    leaderboard = get_leaderboard()
    entries = leaderboard.get_top(10)

    # Count total dreams on disk
    total_dreams = len(list(settings.dreams_dir.glob("dream_*.md"))) if settings.dreams_dir.exists() else 0

    return LeaderboardResponse(
        entries=[
            LeaderboardEntryResponse(
                rank=e.rank,
                content=e.content,
                category=e.category,
                novelty_score=e.novelty_score,
                source_file=e.source_file,
                timestamp=e.timestamp,
                dream_id=e.dream_id,
            )
            for e in entries
        ],
        total_dreams=total_dreams,
    )


@app.get("/dreams")
async def list_dreams(
    priority: str | None = None,
    status: str | None = None,
    min_score: float = 0.0,
    limit: int = 50,
) -> dict:
    """
    List dreams with optional filters.
    
    Query params:
        priority: Filter by priority (critical, high, medium, low, unranked)
        status: Filter by status (pending, applied, rejected, deferred)
        min_score: Minimum novelty score
        limit: Max results (default 50)
    """
    from .leaderboard import get_leaderboard
    
    leaderboard = get_leaderboard()
    entries = leaderboard.entries
    
    # Apply filters
    if priority:
        entries = [e for e in entries if e.priority == priority.lower()]
    if status:
        entries = [e for e in entries if e.status == status.lower()]
    if min_score > 0:
        entries = [e for e in entries if e.novelty_score >= min_score]
    
    entries = entries[:limit]
    
    return {
        "dreams": [e.to_dict() for e in entries],
        "total": len(entries),
        "filters": {"priority": priority, "status": status, "min_score": min_score},
    }


@app.get("/dreams/actionable")
async def get_actionable_dreams(limit: int = 20) -> dict:
    """Get pending dreams sorted by priority (CRITICAL first)."""
    from .leaderboard import get_leaderboard
    
    leaderboard = get_leaderboard()
    entries = leaderboard.get_actionable()[:limit]
    
    return {
        "dreams": [e.to_dict() for e in entries],
        "total": len(entries),
    }


@app.post("/dreams/{dream_id}/status")
async def update_dream_status(
    dream_id: str,
    status: str,
    reason: str = "",
) -> dict:
    """
    Update a dream's review status.
    
    Args:
        dream_id: The dream ID (e.g., dream_20251229_191401_code_idea)
        status: New status (pending, applied, rejected, deferred)
        reason: Optional reason for rejection/deferral
    """
    from .leaderboard import ReviewStatus, get_leaderboard
    
    try:
        review_status = ReviewStatus(status.lower())
    except ValueError:
        return {"error": f"Invalid status: {status}", "valid": ["pending", "applied", "rejected", "deferred"]}
    
    leaderboard = get_leaderboard()
    if leaderboard.update_status(dream_id, review_status, reason):
        return {"success": True, "dream_id": dream_id, "new_status": status}
    else:
        return {"error": f"Dream not found: {dream_id}"}


@app.post("/dreams/reflect")
async def trigger_reflection(limit: int = 10) -> dict:
    """
    Trigger a reflection cycle to prioritize unranked dreams.
    
    Uses the reasoning model to assign priority levels to pending dreams.
    Will return immediately if model is busy with dream cycle.
    """
    # Use lock to prevent concurrent model access
    if not _model_lock.acquire(blocking=False):
        return {
            "prioritized": 0,
            "message": "Model busy - try again later",
        }
    
    try:
        from .reflection import get_reflection_cycle
        
        orchestra = get_orchestra()
        reflection = get_reflection_cycle(orchestra)
        prioritized = reflection.run(batch_size=limit)
        
        return {
            "prioritized": prioritized,
            "message": f"Prioritized {prioritized} dreams",
        }
    finally:
        _model_lock.release()


@app.get("/dreams/stats")
async def get_dream_stats() -> dict:
    """Get dream statistics by priority and status."""
    from collections import Counter
    from .leaderboard import get_leaderboard
    
    leaderboard = get_leaderboard()
    
    priorities = Counter(e.priority for e in leaderboard.entries)
    statuses = Counter(e.status for e in leaderboard.entries)
    
    return {
        "total": len(leaderboard.entries),
        "by_priority": dict(priorities),
        "by_status": dict(statuses),
    }


@app.get("/dreams/{dream_id}/download")
async def download_dream(dream_id: str):
    """
    Download the full dream markdown file.
    
    Returns the raw .md file content for saving locally.
    """
    from fastapi.responses import FileResponse, PlainTextResponse
    
    # Find the dream file
    dream_file = settings.dreams_dir / f"{dream_id}.md"
    
    if not dream_file.exists():
        return PlainTextResponse(f"Dream not found: {dream_id}", status_code=404)
    
    return FileResponse(
        path=dream_file,
        filename=f"{dream_id}.md",
        media_type="text/markdown",
    )


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get detailed system status."""
    orchestra = get_orchestra()
    graph = get_graph()
    scratch = get_scratch()

    return StatusResponse(
        config={
            "model_path": str(settings.model_path),
            "n_gpu_layers": settings.n_gpu_layers,
            "dream_temperature": settings.dream_temperature,
            "log_level": settings.log_level,
        },
        models=orchestra.status(),
        graph=graph.stats(),
        scratch=scratch.summarize(),
    )


@app.post("/process", response_model=ProcessResponse)
async def process_request(
    request: ProcessRequest,
    validator: TaskTypeValidator = Depends(get_task_validator),  # noqa: B008
) -> ProcessResponse:
    """
    Process a request through the conductor.

    Uses dependency injection for validation (Dream #3).
    """
    conductor = get_conductor()

    # Validate and convert task type using injected validator
    task_type = validator.validate(request.task_type)

    result = conductor.process(request.prompt, task_type)

    return ProcessResponse(
        success=result.success,
        output=result.output,
        task_type=result.task_type.name,
        steps_taken=result.steps_taken,
        duration_ms=result.duration_ms,
        error=result.error,
    )


@app.post("/dream", response_model=DreamResponse)
async def generate_dream(request: DreamRequest) -> DreamResponse:
    """Generate a creative dream."""
    conductor = get_conductor()

    seed = request.seed_text or "Explore potential improvements to the codebase"
    result = conductor.process(seed, TaskType.DREAM)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Dream generation failed")

    return DreamResponse(
        content=result.output,
        category="dream",
        novelty_score=0.8,  # Would be calculated by validator in full implementation
        timestamp=datetime.now().isoformat(),
    )


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats() -> GraphStatsResponse:
    """Get knowledge graph statistics."""
    graph = get_graph()
    stats = graph.stats()

    return GraphStatsResponse(
        total_nodes=stats["total_nodes"],
        total_edges=stats["total_edges"],
        avg_entanglement=stats.get("avg_entanglement", 0.0),
        tiers=stats["tiers"],
        types=stats["types"],
    )


@app.post("/graph/query", response_model=list[GraphNode])
async def query_graph(request: GraphQueryRequest) -> list[GraphNode]:
    """Query nodes from the knowledge graph."""
    graph = get_graph()

    # Parse node type if provided
    node_type = None
    if request.node_type:
        try:
            node_type = NodeType[request.node_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid node_type. Must be one of: {[t.name for t in NodeType]}",
            ) from None

    if node_type:
        nodes = graph.query_by_type(
            node_type=node_type,
            min_momentum=request.min_momentum,
            limit=request.limit,
        )
    else:
        nodes = graph.query_hot(limit=request.limit)

    return [
        GraphNode(
            id=n.id,
            content=n.content[:500],
            node_type=n.node_type.name,
            momentum=n.momentum,
            tier=n.tier.name,
            age_hours=n.age_hours,
        )
        for n in nodes
    ]


@app.post("/graph/add")
async def add_to_graph(
    content: str,
    node_type: str = "FACT",
) -> dict[str, str]:
    """Add a node to the knowledge graph."""
    graph = get_graph()

    try:
        ntype = NodeType[node_type.upper()]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_type. Must be one of: {[t.name for t in NodeType]}",
        ) from None

    node = graph.add_node(content=content, node_type=ntype)
    graph.save()

    return {"id": node.id, "status": "created"}


@app.post("/graph/decay")
async def decay_graph() -> dict[str, int]:
    """Apply temporal decay to all nodes."""
    graph = get_graph()
    archived = graph.decay_all()
    graph.save()

    return {"nodes_decayed": len(graph._nodes), "below_threshold": archived}


@app.delete("/graph/prune")
async def prune_graph(threshold: float = 0.01) -> dict[str, int]:
    """Remove cold nodes below momentum threshold."""
    graph = get_graph()
    removed = graph.prune_cold(threshold)
    graph.save()

    return {"nodes_removed": removed}


@app.get("/events")
async def event_stream():  # type: ignore[no-untyped-def]
    """Server-sent events for real-time updates."""
    import asyncio
    import json

    from starlette.responses import StreamingResponse

    async def generate():  # type: ignore[no-untyped-def]
        last_count = 0
        while True:
            try:
                graph = get_graph()
                current_count = len(graph._nodes)

                if current_count != last_count:
                    stats = graph.stats()
                    event_data = {
                        "type": "update",
                        "nodes": current_count,
                        "tiers": stats["tiers"],
                        "types": stats["types"],
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                    last_count = current_count

                await asyncio.sleep(2)
            except Exception:
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "codedreamer.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
        access_log=False,  # Disable verbose HTTP request logs
    )

