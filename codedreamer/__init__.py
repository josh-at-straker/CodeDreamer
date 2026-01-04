"""
CodeDreamer - Autonomous code improvement suggestions via LLM dreaming.

A cognitive architecture for code analysis with:
- Multi-model orchestration (reasoning, coding, embedding)
- Knowledge graph with temporal decay
- Working memory (ScratchBuffer)
- Conductor for task orchestration
"""

from .conductor import Conductor, ConductorResult, TaskType, get_conductor
from .config import Settings, settings
from .dreamer import Dream, DreamCycleStats, Dreamer
from .graph import KnowledgeGraph, KnowledgeNode, NodeType, get_graph
from .indexer import CodebaseIndexer, CodeChunk, IndexStats
from .llm import GenerationResult, LLMClient, get_llm_client
from .models import GenerationParams, ManagedModel, ModelOrchestra, ModelRole, get_orchestra
from .scratch import ReasoningStep, ScratchBuffer, Task, TaskStatus, get_scratch
from .trm import TRMStream, ThoughtFragment, get_trm
from .proactive import ProactiveContext, ProactiveMemory, get_proactive_memory
from .validator import DreamValidator, ValidationResult
from .loop_detector import (
    ConsecutiveDiscardTracker,
    LoopDetectionResult,
    detect_and_truncate,
    get_discard_tracker,
)
from .drill_state import (
    DrillState,
    get_drill_state,
    get_random_graph_context,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "Settings",
    "settings",
    # Models
    "ModelOrchestra",
    "ManagedModel",
    "ModelRole",
    "GenerationParams",
    "get_orchestra",
    # Legacy LLM (kept for compatibility)
    "LLMClient",
    "GenerationResult",
    "get_llm_client",
    # Graph
    "KnowledgeGraph",
    "KnowledgeNode",
    "NodeType",
    "get_graph",
    # Scratch
    "ScratchBuffer",
    "Task",
    "TaskStatus",
    "ReasoningStep",
    "get_scratch",
    # Conductor
    "Conductor",
    "ConductorResult",
    "TaskType",
    "get_conductor",
    # Indexer
    "CodebaseIndexer",
    "CodeChunk",
    "IndexStats",
    # Validator
    "DreamValidator",
    "ValidationResult",
    # Dreamer
    "Dreamer",
    "Dream",
    "DreamCycleStats",
    # TRM
    "TRMStream",
    "ThoughtFragment",
    "get_trm",
    # Proactive Memory
    "ProactiveContext",
    "ProactiveMemory",
    "get_proactive_memory",
    # Loop Detection
    "LoopDetectionResult",
    "ConsecutiveDiscardTracker",
    "detect_and_truncate",
    "get_discard_tracker",
    # Deep Dive Drilling
    "DrillState",
    "get_drill_state",
    "get_random_graph_context",
]
