"""
ScratchBuffer - Working memory for multi-step reasoning.

Provides a structured workspace for the Conductor to:
- Track current task state
- Accumulate intermediate results
- Maintain reasoning chains
- Store temporary context
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a reasoning task."""

    PENDING = auto()
    IN_PROGRESS = auto()
    WAITING = auto()  # Waiting for sub-task
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    step_num: int
    action: str  # "think", "code", "search", "decide", "synthesize"
    input_text: str
    output_text: str | None = None
    model_used: str | None = None
    duration_ms: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return self.output_text is not None


@dataclass
class Task:
    """A task being processed in working memory."""

    id: str
    goal: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    steps: list[ReasoningStep] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def current_step(self) -> int:
        return len(self.steps)

    @property
    def duration_ms(self) -> int:
        if not self.steps:
            return 0
        return sum(s.duration_ms for s in self.steps)

    def add_step(
        self,
        action: str,
        input_text: str,
        output_text: str | None = None,
        model_used: str | None = None,
        duration_ms: int = 0,
    ) -> ReasoningStep:
        """Add a reasoning step to this task."""
        step = ReasoningStep(
            step_num=self.current_step + 1,
            action=action,
            input_text=input_text,
            output_text=output_text,
            model_used=model_used,
            duration_ms=duration_ms,
        )
        self.steps.append(step)
        return step

    def complete(self, result: str) -> None:
        """Mark task as completed with result."""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.status = TaskStatus.FAILED
        self.error = error
        self.completed_at = time.time()


class ScratchBuffer:
    """
    Working memory for cognitive processing.

    The ScratchBuffer maintains:
    - Active tasks being processed
    - Reasoning chains and intermediate results
    - Temporary context for multi-step operations
    - Recent outputs for reflection
    """

    MAX_TASKS = 10
    MAX_HISTORY = 50

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._active_task_id: str | None = None
        self._history: list[Task] = []
        self._context: dict[str, Any] = {}
        self._task_counter = 0

    @property
    def active_task(self) -> Task | None:
        """Get the currently active task."""
        if self._active_task_id:
            return self._tasks.get(self._active_task_id)
        return None

    def create_task(self, goal: str, priority: int = 0) -> Task:
        """
        Create a new task in working memory.

        Args:
            goal: Description of what the task should accomplish.
            priority: Higher priority tasks are processed first.

        Returns:
            The created task.
        """
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        task = Task(
            id=task_id,
            goal=goal,
            priority=priority,
        )

        self._tasks[task_id] = task

        # Evict old tasks if at capacity
        if len(self._tasks) > self.MAX_TASKS:
            self._evict_lowest_priority()

        logger.debug(f"Created task: {task_id} - {goal[:50]}")
        return task

    def start_task(self, task_id: str) -> Task | None:
        """
        Set a task as the active task.

        Args:
            task_id: ID of task to activate.

        Returns:
            The activated task, or None if not found.
        """
        task = self._tasks.get(task_id)
        if task:
            task.status = TaskStatus.IN_PROGRESS
            self._active_task_id = task_id
            logger.debug(f"Started task: {task_id}")
        return task

    def complete_task(self, task_id: str, result: str) -> None:
        """
        Mark a task as completed and move to history.

        Args:
            task_id: ID of task to complete.
            result: Final result of the task.
        """
        task = self._tasks.get(task_id)
        if task:
            task.complete(result)
            self._move_to_history(task_id)

            if self._active_task_id == task_id:
                self._active_task_id = None

            logger.debug(f"Completed task: {task_id}")

    def fail_task(self, task_id: str, error: str) -> None:
        """
        Mark a task as failed and move to history.

        Args:
            task_id: ID of task that failed.
            error: Error description.
        """
        task = self._tasks.get(task_id)
        if task:
            task.fail(error)
            self._move_to_history(task_id)

            if self._active_task_id == task_id:
                self._active_task_id = None

            logger.warning(f"Failed task: {task_id} - {error}")

    def add_step(
        self,
        action: str,
        input_text: str,
        output_text: str | None = None,
        model_used: str | None = None,
        duration_ms: int = 0,
        task_id: str | None = None,
    ) -> ReasoningStep | None:
        """
        Add a reasoning step to a task.

        Args:
            action: Type of action (think, code, search, etc.)
            input_text: Input to the action.
            output_text: Output from the action.
            model_used: Which model performed the action.
            duration_ms: How long the action took.
            task_id: Task to add step to. Uses active task if None.

        Returns:
            The created step, or None if no task found.
        """
        tid = task_id or self._active_task_id
        if not tid:
            return None

        task = self._tasks.get(tid)
        if not task:
            return None

        return task.add_step(
            action=action,
            input_text=input_text,
            output_text=output_text,
            model_used=model_used,
            duration_ms=duration_ms,
        )

    def set_context(self, key: str, value: Any) -> None:
        """Store a value in temporary context."""
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from temporary context."""
        return self._context.get(key, default)

    def clear_context(self) -> None:
        """Clear all temporary context."""
        self._context.clear()

    def get_pending_tasks(self) -> list[Task]:
        """Get all pending tasks sorted by priority."""
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: t.priority, reverse=True)

    def get_recent_completions(self, limit: int = 5) -> list[Task]:
        """Get recently completed tasks."""
        completed = [t for t in self._history if t.status == TaskStatus.COMPLETED]
        return completed[-limit:]

    def summarize(self) -> str:
        """Generate a summary of current scratch buffer state."""
        lines = []

        if self.active_task:
            t = self.active_task
            lines.append(f"ACTIVE: {t.goal[:60]}")
            lines.append(f"  Steps: {t.current_step}, Status: {t.status.name}")
            if t.steps:
                last_step = t.steps[-1]
                lines.append(f"  Last: [{last_step.action}] {last_step.input_text[:40]}...")

        pending = self.get_pending_tasks()
        if pending:
            lines.append(f"\nPENDING: {len(pending)} tasks")
            for t in pending[:3]:
                lines.append(f"  - {t.goal[:50]}")

        if self._context:
            lines.append(f"\nCONTEXT: {len(self._context)} items")
            for key in list(self._context.keys())[:5]:
                lines.append(f"  - {key}")

        return "\n".join(lines) if lines else "(empty)"

    def _move_to_history(self, task_id: str) -> None:
        """Move a task from active to history."""
        task = self._tasks.pop(task_id, None)
        if task:
            self._history.append(task)
            # Trim history if needed
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY :]

    def _evict_lowest_priority(self) -> None:
        """Remove the lowest priority pending task.
        
        Uses generator expression to avoid creating intermediate list,
        improving memory efficiency for large task queues.
        """
        # Generator expression - no intermediate list allocation
        pending_gen = (t for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        lowest = min(pending_gen, key=lambda t: t.priority, default=None)
        if lowest is not None:
            self._move_to_history(lowest.id)


# Singleton
_scratch: ScratchBuffer | None = None


def get_scratch() -> ScratchBuffer:
    """Get or create the scratch buffer singleton."""
    global _scratch
    if _scratch is None:
        _scratch = ScratchBuffer()
    return _scratch


