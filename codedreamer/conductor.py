"""
Conductor - Central orchestration layer.

The Conductor is the "conscious executive" that:
- Analyzes incoming requests
- Decomposes into sub-tasks (HRM)
- Delegates to appropriate models
- Synthesizes results
- Updates knowledge graph
"""

import logging
import textwrap
import time
from dataclasses import dataclass
from enum import Enum, auto

from .graph import KnowledgeGraph, NodeType, get_graph
from .models import ModelOrchestra, get_orchestra
from .scratch import ScratchBuffer, Task, get_scratch

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks the conductor can handle."""

    QUERY = auto()  # Answer a question
    CODE = auto()  # Generate or modify code
    DREAM = auto()  # Creative exploration
    PLAN = auto()  # Decompose complex goal
    REFLECT = auto()  # Self-analysis
    INDEX = auto()  # Process and store information


@dataclass
class ConductorResult:
    """Result from conductor processing."""

    success: bool
    output: str
    task_type: TaskType
    steps_taken: int
    duration_ms: int
    nodes_created: int = 0
    error: str | None = None


class Conductor:
    """
    Central orchestration layer for cognitive processing.

    Implements the "Gemini Conductor" pattern:
    1. Perception: Gather context from graph, scratch, input
    2. Planning: Decompose complex tasks (HRM)
    3. Execution: Delegate to appropriate models
    4. Synthesis: Combine results, update memory
    """

    # Prompts for different cognitive operations
    CLASSIFY_PROMPT = """Classify this request into one of these categories:
- QUERY: Asking for information or explanation
- CODE: Requesting code generation or modification
- DREAM: Creative exploration or brainstorming
- PLAN: Complex multi-step task requiring decomposition
- REFLECT: Self-analysis or improvement suggestion
- INDEX: Processing information for storage

Request: {request}

Category (one word):"""

    PLAN_PROMPT = """Break down this complex task into clear, sequential steps.
Each step should be a single, actionable item.

Task: {task}

Relevant Context:
{context}

Steps (numbered list):"""

    SYNTHESIZE_PROMPT = """Synthesize these results into a coherent response.

Original Goal: {goal}

Step Results:
{results}

Synthesized Response:"""

    def __init__(
        self,
        orchestra: ModelOrchestra | None = None,
        graph: KnowledgeGraph | None = None,
        scratch: ScratchBuffer | None = None,
    ) -> None:
        """
        Initialize the conductor.

        Args:
            orchestra: Model orchestra. Uses singleton if None.
            graph: Knowledge graph. Uses singleton if None.
            scratch: Scratch buffer. Uses singleton if None.
        """
        self.orchestra = orchestra or get_orchestra()
        self.graph = graph or get_graph()
        self.scratch = scratch or get_scratch()

    def process(self, request: str, task_type: TaskType | None = None) -> ConductorResult:
        """
        Process a request through the cognitive pipeline.

        Args:
            request: The user request or goal.
            task_type: Optional explicit task type. Auto-classified if None.

        Returns:
            ConductorResult with output and metadata.

        Raises:
            ValueError: If request is empty or task_type is invalid.
        """
        # Input validation (Dream #1 suggestion)
        if not request or not request.strip():
            raise ValueError("Request cannot be empty")

        if task_type is not None and not isinstance(task_type, TaskType):
            raise ValueError(f"task_type must be TaskType or None, got {type(task_type)}")

        start_time = time.time()

        # Create task in scratch buffer
        task = self.scratch.create_task(goal=request)
        self.scratch.start_task(task.id)

        try:
            # 1. Classify if not provided
            if task_type is None:
                task_type = self._classify(request, task)

            # 2. Gather context
            context = self._gather_context(request)
            self.scratch.set_context("relevant_context", context)

            # 3. Route to appropriate handler
            if task_type == TaskType.PLAN:
                output = self._handle_plan(request, context, task)
            elif task_type == TaskType.CODE:
                output = self._handle_code(request, context, task)
            elif task_type == TaskType.DREAM:
                output = self._handle_dream(request, context, task)
            elif task_type == TaskType.REFLECT:
                output = self._handle_reflect(request, context, task)
            elif task_type == TaskType.INDEX:
                output = self._handle_index(request, task)
            else:  # QUERY
                output = self._handle_query(request, context, task)

            # 4. Complete task
            self.scratch.complete_task(task.id, output)

            duration_ms = int((time.time() - start_time) * 1000)

            return ConductorResult(
                success=True,
                output=output,
                task_type=task_type,
                steps_taken=task.current_step,
                duration_ms=duration_ms,
            )

        except Exception as e:
            # Enhanced error handling with context (Dream #1 suggestion)
            error_msg = (
                f"Task {task.id} ({task_type.name if task_type else 'UNKNOWN'}): {e}"
            )
            self.scratch.fail_task(task.id, error_msg)
            logger.exception(
                f"Conductor error - Task ID: {task.id}, "
                f"Type: {task_type.name if task_type else 'UNKNOWN'}, "
                f"Step: {task.current_step}, Error: {e}"
            )

            duration_ms = int((time.time() - start_time) * 1000)

            return ConductorResult(
                success=False,
                output="",
                task_type=task_type or TaskType.QUERY,
                steps_taken=task.current_step,
                duration_ms=duration_ms,
                error=error_msg,
            )

    def _classify(self, request: str, task: Task) -> TaskType:
        """Classify the request into a task type."""
        # Use consistent truncation for both prompt and logging
        truncated_request = request[:500]
        prompt = self.CLASSIFY_PROMPT.format(request=truncated_request)

        start = time.time()
        response = self.orchestra.reason(prompt, temperature=0.1, max_tokens=20)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="classify",
            input_text=truncated_request[:200],  # Log more context, still bounded
            output_text=response.strip(),
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        # Parse response
        response_upper = response.strip().upper()
        for task_type in TaskType:
            if task_type.name in response_upper:
                return task_type

        return TaskType.QUERY  # Default

    def _gather_context(self, request: str, limit: int = 5) -> str:
        """
        Gather relevant context from knowledge graph.

        Args:
            request: The request to gather context for (unused currently, for future semantic search).
            limit: Maximum number of hot nodes to retrieve.

        Returns:
            Formatted context string or error/empty message.
        """
        try:
            hot_nodes = self.graph.query_hot(limit=limit)

            if not hot_nodes:
                return "(no prior context)"

            context_parts = []
            for node in hot_nodes:
                # Use textwrap.shorten for smarter truncation (avoids mid-word cuts)
                truncated = textwrap.shorten(node.content, width=200, placeholder="...")
                context_parts.append(f"[{node.node_type.name}] {truncated}")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Error gathering context from graph: {e}")
            return "(error fetching context)"

    def _handle_query(self, request: str, context: str, task: Task) -> str:
        """Handle a query request."""
        prompt = f"""Answer this question using the available context.

Context:
{context}

Question: {request}

Answer:"""

        start = time.time()
        response = self.orchestra.reason(prompt, temperature=0.7)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="think",
            input_text=request,
            output_text=response,
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        return response

    def _handle_code(self, request: str, context: str, task: Task) -> str:
        """Handle a code generation request."""
        prompt = f"""Generate code for this request.

Context:
{context}

Request: {request}

Code:"""

        start = time.time()
        response = self.orchestra.code(prompt, temperature=0.3)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="code",
            input_text=request,
            output_text=response,
            model_used="coder",
            duration_ms=duration,
            task_id=task.id,
        )

        # Store in graph
        self.graph.add_node(
            content=response[:500],
            node_type=NodeType.CODE,
            metadata={"request": request[:100]},
        )

        return response

    def _handle_dream(self, request: str, context: str, task: Task) -> str:
        """Handle a creative exploration request."""
        prompt = f"""Think creatively about this topic. Explore possibilities,
consider unconventional approaches, and generate novel ideas.

Context:
{context}

Topic: {request}

Creative Exploration:"""

        start = time.time()
        response = self.orchestra.dream(prompt, temperature=0.9)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="dream",
            input_text=request,
            output_text=response,
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        # Note: Dreams are added to graph by the dream cycle after validation
        # to avoid duplicates and ensure quality control

        return response

    def _handle_plan(self, request: str, context: str, task: Task) -> str:
        """Handle a complex task requiring decomposition."""
        # Step 1: Generate plan
        plan_prompt = self.PLAN_PROMPT.format(task=request, context=context)

        start = time.time()
        plan = self.orchestra.reason(plan_prompt, temperature=0.5)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="plan",
            input_text=request,
            output_text=plan,
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        # Step 2: Execute each step (simplified - just concatenate for now)
        # In full implementation, would recursively call process() for each step

        # Step 3: Synthesize
        synth_prompt = self.SYNTHESIZE_PROMPT.format(
            goal=request,
            results=plan,
        )

        start = time.time()
        synthesis = self.orchestra.reason(synth_prompt, temperature=0.7)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="synthesize",
            input_text="Combine plan steps",
            output_text=synthesis,
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        return synthesis

    def _handle_reflect(self, request: str, context: str, task: Task) -> str:
        """Handle a self-reflection request."""
        # Include scratch buffer state in reflection
        scratch_state = self.scratch.summarize()

        prompt = f"""Reflect on the current state and provide insights.

Current Working Memory:
{scratch_state}

Knowledge Context:
{context}

Reflection Request: {request}

Insights:"""

        start = time.time()
        response = self.orchestra.reason(prompt, temperature=0.7)
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="reflect",
            input_text=request,
            output_text=response,
            model_used="reasoning",
            duration_ms=duration,
            task_id=task.id,
        )

        # Store insight in graph
        self.graph.add_node(
            content=response[:500],
            node_type=NodeType.CONCEPT,
            metadata={"type": "reflection"},
        )

        return response

    def _handle_index(self, request: str, task: Task) -> str:
        """Handle an indexing/storage request."""
        # Extract and store facts from the request
        prompt = f"""Extract key facts from this text.
List each fact on a separate line.

Text: {request}

Facts:"""

        start = time.time()
        facts = self.orchestra.code(prompt, temperature=0.2)  # Low temp for extraction
        duration = int((time.time() - start) * 1000)

        self.scratch.add_step(
            action="extract",
            input_text=request[:100],
            output_text=facts,
            model_used="coder",
            duration_ms=duration,
            task_id=task.id,
        )

        # Store each fact as a node
        fact_lines = [f.strip() for f in facts.split("\n") if f.strip()]
        nodes_created = 0

        for fact in fact_lines[:10]:  # Limit to 10 facts
            if len(fact) > 20:  # Skip very short lines
                self.graph.add_node(content=fact, node_type=NodeType.FACT)
                nodes_created += 1

        self.graph.save()

        return f"Indexed {nodes_created} facts from the input."


# Singleton
_conductor: Conductor | None = None


def get_conductor() -> Conductor:
    """Get or create the conductor singleton."""
    global _conductor
    if _conductor is None:
        _conductor = Conductor()
    return _conductor

