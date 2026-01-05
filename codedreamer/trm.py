"""
Temporal Recursive Memory (TRM) Stream.

Maintains a stream of thought that carries forward between dream cycles,
enabling compound insights rather than isolated observations.

Features:
- Applies temporal decay: Z(t) = Z₀ · e^(-λt)
- Maintains continuity between cognitive cycles
- Enables "building on previous insights"
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class ThoughtFragment:
    """A single insight in the thought stream."""

    content: str
    source_file: str | None
    category: str
    timestamp: float = field(default_factory=time.time)
    initial_salience: float = 1.0
    access_count: int = 0

    def get_salience(self, decay_lambda: float = 0.1) -> float:
        """
        Calculate current salience with temporal decay.

        Uses exponential decay: Z(t) = Z₀ · e^(-λt)
        where t is hours since creation.
        """
        hours_elapsed = (time.time() - self.timestamp) / 3600
        decayed = self.initial_salience * math.exp(-decay_lambda * hours_elapsed)

        # Boost for access (reinforcement)
        access_boost = min(0.3, self.access_count * 0.05)

        return min(1.0, decayed + access_boost)

    def activate(self) -> None:
        """Mark this fragment as accessed (reinforces memory)."""
        self.access_count += 1


class TRMStream:
    """
    Temporal Recursive Memory Stream.

    Maintains a rolling buffer of recent insights that inform future dreams.
    Implements temporal decay so old insights fade naturally.
    """

    MAX_FRAGMENTS = 20
    DECAY_LAMBDA = 0.15  # Decay rate (higher = faster forgetting)
    SALIENCE_THRESHOLD = 0.2  # Below this, fragments are pruned

    def __init__(self) -> None:
        self._fragments: list[ThoughtFragment] = []
        self._cycle_count = 0

    def add_insight(
        self,
        content: str,
        source_file: str | None = None,
        category: str = "insight",
        salience: float = 1.0,
    ) -> None:
        """
        Add a new insight to the thought stream.

        Args:
            content: The insight content (will be truncated).
            source_file: Source file that inspired this insight.
            category: Category of the insight.
            salience: Initial importance (0.0 to 1.0).
        """
        # Truncate for efficiency
        truncated = content[:500] if len(content) > 500 else content

        fragment = ThoughtFragment(
            content=truncated,
            source_file=source_file,
            category=category,
            initial_salience=salience,
        )

        self._fragments.append(fragment)
        logger.debug(f"TRM: Added insight ({category}) - salience={salience:.2f}")

        # Prune if over capacity
        if len(self._fragments) > self.MAX_FRAGMENTS:
            self._prune()

    def get_context(self, max_fragments: int = 5) -> str:
        """
        Get recent insights as context for the next dream.

        Returns formatted string of top insights by salience.
        """
        if not self._fragments:
            return ""

        # Sort by current salience
        sorted_fragments = sorted(
            self._fragments,
            key=lambda f: f.get_salience(self.DECAY_LAMBDA),
            reverse=True,
        )

        # Take top N
        top = sorted_fragments[:max_fragments]

        # Mark as accessed
        for f in top:
            f.activate()

        # Format as context
        lines = ["## Recent Insights (TRM Stream)"]
        for i, f in enumerate(top, 1):
            salience = f.get_salience(self.DECAY_LAMBDA)
            source = f.source_file.split("/")[-1] if f.source_file else "general"
            lines.append(f"\n### Insight {i} (salience: {salience:.2f}, from: {source})")
            lines.append(f"{f.content}")

        return "\n".join(lines)

    def get_themes(self) -> dict[str, int]:
        """Get counts of themes/categories in the stream."""
        themes: dict[str, int] = {}
        for f in self._fragments:
            themes[f.category] = themes.get(f.category, 0) + 1
        return themes

    def get_relevant(
        self, source_file: str, max_fragments: int = 3
    ) -> list["ThoughtFragment"]:
        """
        Get fragments relevant to a specific source file.

        Used by ProactiveMemory to anticipate needed context.

        Args:
            source_file: File path to find relevant fragments for.
            max_fragments: Maximum number of fragments to return.

        Returns:
            List of relevant ThoughtFragment objects.
        """
        from pathlib import Path

        source_name = Path(source_file).name

        # Find fragments about this file or related
        relevant = []
        for f in self._fragments:
            if f.source_file and source_name in str(f.source_file):
                relevant.append((f, f.get_salience(self.DECAY_LAMBDA) * 1.5))  # Boost same-file
            elif f.get_salience(self.DECAY_LAMBDA) > 0.5:
                relevant.append((f, f.get_salience(self.DECAY_LAMBDA)))

        # Sort by boosted salience
        relevant.sort(key=lambda x: x[1], reverse=True)

        # Activate retrieved fragments (reinforcement learning)
        result = []
        for f, _ in relevant[:max_fragments]:
            f.activate()
            result.append(f)

        return result

    def tick(self) -> None:
        """
        Called each dream cycle - prunes low-salience fragments.
        """
        self._cycle_count += 1

        # Prune every cycle
        before = len(self._fragments)
        self._prune()
        after = len(self._fragments)

        if before != after:
            logger.debug(f"TRM: Pruned {before - after} fragments (cycle {self._cycle_count})")

    def _prune(self) -> None:
        """Remove fragments below salience threshold."""
        self._fragments = [
            f for f in self._fragments
            if f.get_salience(self.DECAY_LAMBDA) >= self.SALIENCE_THRESHOLD
        ]

        # Also enforce max capacity
        if len(self._fragments) > self.MAX_FRAGMENTS:
            # Sort by salience and keep top
            self._fragments = sorted(
                self._fragments,
                key=lambda f: f.get_salience(self.DECAY_LAMBDA),
                reverse=True,
            )[:self.MAX_FRAGMENTS]

    def clear(self) -> None:
        """Clear all fragments."""
        self._fragments.clear()
        self._cycle_count = 0

    def __len__(self) -> int:
        return len(self._fragments)

    def __iter__(self) -> Iterator[ThoughtFragment]:
        return iter(self._fragments)

    def get_stats(self) -> dict:
        """Get TRM statistics for monitoring."""
        if not self._fragments:
            return {
                "fragment_count": 0,
                "avg_salience": 0.0,
                "cycle_count": self._cycle_count,
                "themes": {},
            }

        saliences = [f.get_salience(self.DECAY_LAMBDA) for f in self._fragments]
        return {
            "fragment_count": len(self._fragments),
            "avg_salience": sum(saliences) / len(saliences),
            "max_salience": max(saliences),
            "min_salience": min(saliences),
            "cycle_count": self._cycle_count,
            "themes": self.get_themes(),
        }


# Singleton
_trm: TRMStream | None = None


def get_trm() -> TRMStream:
    """Get or create the TRM stream singleton."""
    global _trm
    if _trm is None:
        _trm = TRMStream()
    return _trm

