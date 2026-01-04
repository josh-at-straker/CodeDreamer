"""Tests for scratch buffer (working memory)."""

from codedreamer.scratch import ScratchBuffer, TaskStatus


def test_create_task() -> None:
    """Test task creation."""
    scratch = ScratchBuffer()

    task = scratch.create_task(goal="Test task")

    assert task.id.startswith("task_")
    assert task.goal == "Test task"
    assert task.status == TaskStatus.PENDING


def test_start_task() -> None:
    """Test starting a task."""
    scratch = ScratchBuffer()

    task = scratch.create_task(goal="Test task")
    scratch.start_task(task.id)

    assert task.status == TaskStatus.IN_PROGRESS
    assert scratch.active_task == task


def test_complete_task() -> None:
    """Test completing a task."""
    scratch = ScratchBuffer()

    task = scratch.create_task(goal="Test task")
    scratch.start_task(task.id)
    scratch.complete_task(task.id, result="Task completed successfully")

    assert task.status == TaskStatus.COMPLETED
    assert task.result == "Task completed successfully"
    assert scratch.active_task is None


def test_add_reasoning_step() -> None:
    """Test adding reasoning steps to a task."""
    scratch = ScratchBuffer()

    task = scratch.create_task(goal="Multi-step task")
    scratch.start_task(task.id)

    step1 = scratch.add_step(
        action="think",
        input_text="What should we do?",
        output_text="We should analyze first.",
        model_used="reasoning",
    )

    step2 = scratch.add_step(
        action="code",
        input_text="Generate solution",
        output_text="def solution(): pass",
        model_used="coder",
    )

    assert task.current_step == 2
    assert step1.step_num == 1
    assert step2.step_num == 2
    assert step1.action == "think"
    assert step2.action == "code"


def test_context_storage() -> None:
    """Test temporary context storage."""
    scratch = ScratchBuffer()

    scratch.set_context("key1", "value1")
    scratch.set_context("key2", {"nested": "data"})

    assert scratch.get_context("key1") == "value1"
    assert scratch.get_context("key2") == {"nested": "data"}
    assert scratch.get_context("missing", "default") == "default"


def test_pending_tasks_priority() -> None:
    """Test pending tasks are sorted by priority."""
    scratch = ScratchBuffer()

    task_low = scratch.create_task(goal="Low priority", priority=1)
    task_high = scratch.create_task(goal="High priority", priority=10)
    task_mid = scratch.create_task(goal="Mid priority", priority=5)

    pending = scratch.get_pending_tasks()

    assert pending[0] == task_high
    assert pending[1] == task_mid
    assert pending[2] == task_low



