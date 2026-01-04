"""
Reflection Cycle - Periodic re-ranking of dreams by the model.

As the model accumulates knowledge about the codebase, it periodically
reflects on pending dreams and assigns priority levels:
- CRITICAL: Security, data loss, crashes
- HIGH: Performance, significant bugs
- MEDIUM: Quality, maintainability
- LOW: Style, minor improvements

The model also consolidates similar dreams and dismisses irrelevant ones.
"""

import logging
from datetime import datetime

from .config import settings
from .graph import KnowledgeGraph, get_graph
from .leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    Priority,
    ReviewStatus,
    get_leaderboard,
)
from .models import ModelOrchestra

logger = logging.getLogger(__name__)

# How many dreams to process per reflection cycle
REFLECTION_BATCH_SIZE = 10

REFLECTION_PROMPT = """You are a pragmatic senior engineer reviewing code improvement suggestions.
Be skeptical - most suggestions are LOW or MEDIUM priority. HIGH/CRITICAL are rare.

## Current Codebase Knowledge
{graph_context}

## Dreams to Prioritize
{dreams_to_review}

## Priority Criteria (BE STRICT)

CRITICAL (very rare - maybe 1 in 50):
- Active security vulnerabilities (SQL injection, auth bypass, data exposure)
- Data loss/corruption bugs
- Crashes in production paths

HIGH (rare - maybe 1 in 20):
- Bugs that cause incorrect behavior in normal usage
- Memory leaks or significant performance issues
- Missing error handling that would crash the app
- Race conditions or concurrency bugs

MEDIUM (common):
- Defensive error handling for edge cases
- Code that works but could be cleaner
- Missing input validation for unlikely inputs
- Refactoring that improves testability

LOW (very common):
- Style preferences (logging, constants, naming)
- "Nice to have" improvements
- Adding error handling where Python already raises clear errors
- Suggestions that add complexity without clear benefit

## Rules
1. If the suggestion adds error handling for something Python already handles clearly → LOW
2. If the suggestion is "add logging" to a small utility function → LOW  
3. If unsure between two levels, pick the LOWER one
4. Refactoring alone is not HIGH - it needs to fix an actual problem

## Output Format
DREAM_ID|PRIORITY|REASON (one line each, nothing else)

Example:
dream_20251230_065356|MEDIUM|Adds validation but not critical path
dream_20251230_064734|LOW|Style improvement only"""


class ReflectionCycle:
    """
    Periodically re-ranks dreams using accumulated knowledge.
    
    The model reviews pending dreams and assigns priority levels
    based on its understanding of the codebase.
    """

    def __init__(
        self,
        orchestra: ModelOrchestra,
        graph: KnowledgeGraph | None = None,
        leaderboard: Leaderboard | None = None,
    ):
        self.orchestra = orchestra
        self.graph = graph or get_graph()
        self.leaderboard = leaderboard or get_leaderboard()

    def run(self, batch_size: int = REFLECTION_BATCH_SIZE) -> int:
        """
        Run a reflection cycle to prioritize unranked dreams.
        
        Returns:
            Number of dreams prioritized.
        """
        unranked = self.leaderboard.get_unranked()
        if not unranked:
            logger.info("No unranked dreams to reflect on")
            return 0

        # Take a batch
        batch = unranked[:batch_size]
        logger.info(f"Reflecting on {len(batch)} dreams...")

        # Build context
        graph_context = self._build_graph_context()
        dreams_context = self._build_dreams_context(batch)

        # Ask the model to prioritize
        prompt = REFLECTION_PROMPT.format(
            graph_context=graph_context,
            dreams_to_review=dreams_context,
        )

        try:
            response = self.orchestra.reason(
                prompt,
                temperature=0.3,  # Low temp for consistent ranking
                max_tokens=500,
            )
        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            return 0

        # Parse response and update priorities
        prioritized = self._parse_rankings(response, batch)
        
        logger.info(f"Prioritized {prioritized} dreams")
        return prioritized

    def _build_graph_context(self) -> str:
        """Build context from knowledge graph."""
        stats = self.graph.stats()
        
        # Get most entangled nodes (most connected = most important)
        # Returns list of (node, score) tuples
        top_nodes = self.graph.get_most_entangled(limit=10)
        
        context_parts = [
            f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges",
            f"Node types: {stats.get('type_counts', {})}",
            "",
            "Most Connected Concepts:",
        ]
        
        for node, score in top_nodes:
            context_parts.append(f"- [{node.node_type.name}] {node.content[:100]}")
        
        return "\n".join(context_parts)

    def _build_dreams_context(self, dreams: list[LeaderboardEntry]) -> str:
        """Build context from dreams to review."""
        parts = []
        for d in dreams:
            # Extract first ~300 chars of actual content (skip markdown header)
            content = d.content
            if "---" in content:
                content = content.split("---", 1)[-1]
            content = content.strip()[:300].replace("\n", " ")
            
            parts.append(f"""
### {d.dream_id}
Source: {d.source_file} | Category: {d.category} | Novelty: {d.novelty_score:.2f}
Summary: {content}...
""")
        
        return "\n".join(parts)

    def _parse_rankings(
        self,
        response: str,
        batch: list[LeaderboardEntry],
    ) -> int:
        """Parse model response and update priorities."""
        prioritized = 0
        dream_ids = {d.dream_id for d in batch}
        
        # Flexible priority mapping (handles truncated outputs like "MED", "HI", etc.)
        priority_map = {
            "critical": Priority.CRITICAL,
            "crit": Priority.CRITICAL,
            "high": Priority.HIGH,
            "hi": Priority.HIGH,
            "medium": Priority.MEDIUM,
            "med": Priority.MEDIUM,
            "low": Priority.LOW,
            "lo": Priority.LOW,
        }

        logger.debug(f"Model response:\n{response[:500]}")

        for line in response.strip().split("\n"):
            line = line.strip()
            if "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 2:
                continue
            
            dream_id_partial = parts[0].strip()
            priority_str = parts[1].strip().lower()
            
            # Find matching dream ID (model may omit _code_idea/_code_fix suffix)
            matched_id = None
            for full_id in dream_ids:
                if full_id.startswith(dream_id_partial) or dream_id_partial in full_id:
                    matched_id = full_id
                    break
            
            if matched_id is None:
                logger.debug(f"No match for: {dream_id_partial}")
                continue
            
            priority = priority_map.get(priority_str)
            if priority is None:
                logger.debug(f"Unknown priority: {priority_str}")
                continue
            
            if self.leaderboard.update_priority(matched_id, priority):
                prioritized += 1
                logger.debug(f"Set {matched_id} to {priority.value}")

        return prioritized


# Singleton
_reflection: ReflectionCycle | None = None


def get_reflection_cycle(orchestra: ModelOrchestra) -> ReflectionCycle:
    """Get the global reflection cycle instance."""
    global _reflection
    if _reflection is None:
        _reflection = ReflectionCycle(orchestra)
    return _reflection

