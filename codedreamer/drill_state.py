"""
Deep Dive Drilling State Management.

Implements 4-level drilling to transform vague insights
into concrete, actionable code.

Levels:
  0 - Discovery: Generate ONE useful insight
  1 - Framework: Design concrete architecture/framework
  2 - Implementation: Create detailed implementation plan
  3 - Code: Write actual production code

When a dream passes validation at level N, we drill to level N+1
on the SAME theme until we reach code or hit a plateau.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph import KnowledgeGraph

logger = logging.getLogger(__name__)


# Drilling prompts - each level gets more concrete
DRILL_PROMPTS = {
    0: """Generate ONE useful insight or improvement idea.
Be specific, concrete, and NOVEL. Focus on something you haven't suggested before.""",

    1: """Take this concept and design a concrete FRAMEWORK or ARCHITECTURE.
Include:
- Specific components and their responsibilities
- Data flows between components
- Key interfaces and contracts
- How this integrates with existing code""",

    2: """Create a detailed IMPLEMENTATION PLAN.
Include:
- Specific functions/methods with signatures
- Data structures and types
- Error handling strategy
- Step-by-step implementation order""",

    3: """Write PRODUCTION-READY CODE.
Include:
- Complete function/class definitions
- Type hints and docstrings
- Error handling
- Follow the existing codebase style
Output ONLY the code, minimal explanation.""",
}


@dataclass
class DrillState:
    """Tracks the current drilling state across dream cycles."""
    
    # Current drill depth (0-3)
    depth: int = 0
    
    # The theme/insight we're drilling on
    current_theme: str = ""
    
    # Last saved dream content (to build upon)
    last_dream: str = ""
    
    # Source file we're focusing on
    source_file: str = ""
    
    # Consecutive discards at this depth
    discards_at_depth: int = 0
    
    # Total consecutive discards (for graph jump)
    total_consecutive_discards: int = 0
    
    # Max depth reached in current drill
    max_depth_reached: int = 0
    
    def should_drill_deeper(self, category: str, novelty: float) -> bool:
        """
        Determine if we should drill deeper on this dream.
        
        Drillable categories: insight, code_idea, code_fix
        Or any dream with high novelty (>= 0.5)
        """
        if self.depth >= 3:
            return False  # Already at code level
        
        drillable_categories = {"insight", "code_idea", "code_fix", "refactor"}
        is_drillable = category in drillable_categories or novelty >= 0.5
        
        return is_drillable
    
    def drill_down(self, dream_content: str, source_file: str) -> None:
        """Advance to next drill level with the given dream as context."""
        self.depth += 1
        self.last_dream = dream_content
        self.source_file = source_file
        self.discards_at_depth = 0
        self.max_depth_reached = max(self.max_depth_reached, self.depth)
        
        logger.info(f"Drilling to depth {self.depth} on {source_file}")
    
    def record_discard(self) -> None:
        """Record a discarded dream."""
        self.discards_at_depth += 1
        self.total_consecutive_discards += 1
        
        # If we've hit 2 discards at current depth during drilling, reset
        if self.depth > 0 and self.discards_at_depth >= 2:
            logger.info(f"Deep dive exhausted at depth {self.depth} - resetting")
            self.reset()
    
    def record_success(self) -> None:
        """Record a successful dream (not drilling deeper)."""
        self.total_consecutive_discards = 0
        self.discards_at_depth = 0
    
    def reset(self) -> None:
        """Reset to discovery mode."""
        self.depth = 0
        self.current_theme = ""
        self.last_dream = ""
        self.source_file = ""
        self.discards_at_depth = 0
        # Don't reset total_consecutive_discards - that's for graph jump
    
    def should_graph_jump(self, threshold: int = 3) -> bool:
        """Check if we should jump to a random graph node."""
        return self.total_consecutive_discards >= threshold
    
    def record_graph_jump(self) -> None:
        """Record that we performed a graph jump."""
        self.total_consecutive_discards = 0
        self.reset()
        logger.info("Performed graph jump - exploring new territory")
    
    def get_drill_prompt(self) -> str:
        """Get the appropriate prompt for current drill level."""
        return DRILL_PROMPTS.get(self.depth, DRILL_PROMPTS[0])
    
    def get_context_for_drilling(self) -> str:
        """Get context from previous drill level to build upon."""
        if self.depth == 0 or not self.last_dream:
            return ""
        
        return f"""## Previous Insight (Level {self.depth - 1})
You generated this insight in the previous step:

{self.last_dream[:800]}

---

Now take this further. Build DIRECTLY on this insight.
Make it MORE CONCRETE and MORE ACTIONABLE.
"""


def get_random_graph_context(graph: "KnowledgeGraph", skip_types: list | None = None) -> tuple[str, str]:
    """
    Get context from a random graph node for exploration.
    
    Returns:
        Tuple of (context_string, source_file or "")
    """
    from .graph import NodeType
    
    from .graph import StorageTier
    
    skip_types = skip_types or []
    
    # Get all active nodes - those not in COLD tier (access private _nodes dict)
    active_nodes = [n for n in graph._nodes.values() if n.tier != StorageTier.COLD]
    
    if not active_nodes:
        return "", ""
    
    # Filter out unwanted types (e.g., DREAM nodes - we want CODE, FACT, CONCEPT)
    preferred_types = {NodeType.CODE, NodeType.FACT, NodeType.CONCEPT, NodeType.ENTITY}
    candidate_nodes = [n for n in active_nodes if n.node_type in preferred_types]
    
    if not candidate_nodes:
        candidate_nodes = active_nodes
    
    # Pick a random node
    seed_node = random.choice(candidate_nodes)
    
    # Build context
    context = f"""## GRAPH JUMP: Exploring a different part of the codebase

### Seed Node: [{seed_node.node_type.name}]
{seed_node.content[:500]}

"""
    
    # Find connected nodes via networkx graph edges
    connected = []
    try:
        # Get neighbors from the internal networkx graph
        neighbors = list(graph._graph.neighbors(seed_node.id))
        for neighbor_id in neighbors:
            if neighbor_id in graph._nodes:
                neighbor_node = graph._nodes[neighbor_id]
                if neighbor_node.tier != StorageTier.COLD:
                    connected.append(neighbor_node)
    except Exception:
        pass  # Graph may not have this node
    
    if connected:
        context += "### Related Nodes:\n"
        for n in connected[:3]:  # Max 3 connected nodes
            context += f"- [{n.node_type.name}] {n.content[:100]}...\n"
    
    # If no edges, pick random diverse nodes
    if not connected and len(candidate_nodes) > 1:
        others = random.sample(candidate_nodes, min(3, len(candidate_nodes) - 1))
        context += "### Other Codebase Areas:\n"
        for n in others:
            if n.id != seed_node.id:
                context += f"- [{n.node_type.name}] {n.content[:100]}...\n"
    
    context += """
---

Think about this from a FRESH perspective. What improvements or patterns do you see?
"""
    
    # Extract source file if available
    source_file = seed_node.metadata.get("file", seed_node.metadata.get("source", ""))
    
    logger.info(f"Graph jump to {seed_node.node_type.name} node: {seed_node.content[:50]}...")
    
    return context, source_file


# Global drill state
_drill_state: DrillState | None = None


def get_drill_state() -> DrillState:
    """Get the global drill state."""
    global _drill_state
    if _drill_state is None:
        _drill_state = DrillState()
    return _drill_state


