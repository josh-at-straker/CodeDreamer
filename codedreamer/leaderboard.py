"""
Dream Leaderboard - Tracks the top performing dreams.

Maintains a ranked list of the best dreams based on novelty score,
persisted to disk for continuity across restarts.

Features:
- Priority tiers (CRITICAL, HIGH, MEDIUM, LOW)
- Review status (PENDING, APPLIED, REJECTED, DEFERRED)
- Periodic re-ranking with model reflection
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .config import settings

logger = logging.getLogger(__name__)

# Maximum entries in the leaderboard
MAX_LEADERBOARD_SIZE = 100


class Priority(str, Enum):
    """Dream priority levels."""
    CRITICAL = "critical"  # Security, data loss, crashes
    HIGH = "high"          # Performance, significant bugs
    MEDIUM = "medium"      # Quality, maintainability
    LOW = "low"            # Style, minor improvements
    UNRANKED = "unranked"  # Not yet prioritized


class ReviewStatus(str, Enum):
    """Dream review status."""
    PENDING = "pending"    # Not yet reviewed
    APPLIED = "applied"    # Implemented
    REJECTED = "rejected"  # Decided not to implement
    DEFERRED = "deferred"  # Maybe later


@dataclass
class LeaderboardEntry:
    """A ranked dream entry."""

    rank: int
    content: str  # Full dream content
    category: str
    novelty_score: float
    source_file: str
    timestamp: str
    dream_id: str  # Unique identifier (filename)
    priority: str = Priority.UNRANKED.value  # Priority tier
    status: str = ReviewStatus.PENDING.value  # Review status
    rejection_reason: str = ""  # Why rejected/deferred
    last_ranked: str = ""  # When priority was last assessed

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "rank": self.rank,
            "content": self.content,
            "category": self.category,
            "novelty_score": self.novelty_score,
            "source_file": self.source_file,
            "timestamp": self.timestamp,
            "dream_id": self.dream_id,
            "priority": self.priority,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "last_ranked": self.last_ranked,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LeaderboardEntry":
        """Deserialize from dictionary."""
        return cls(
            rank=data["rank"],
            content=data["content"],
            category=data["category"],
            novelty_score=data["novelty_score"],
            source_file=data["source_file"],
            timestamp=data["timestamp"],
            dream_id=data["dream_id"],
            priority=data.get("priority", Priority.UNRANKED.value),
            status=data.get("status", ReviewStatus.PENDING.value),
            rejection_reason=data.get("rejection_reason", ""),
            last_ranked=data.get("last_ranked", ""),
        )


@dataclass
class Leaderboard:
    """
    Maintains a ranked list of top dreams.
    
    Dreams are ranked by novelty score. The leaderboard persists
    across restarts via JSON storage.
    """

    entries: list[LeaderboardEntry] = field(default_factory=list)
    _path: Path = field(default_factory=lambda: settings.dreams_dir / "leaderboard.json")

    def __post_init__(self) -> None:
        """Load existing leaderboard from disk."""
        self.load()

    def submit(
        self,
        content: str,
        category: str,
        novelty_score: float,
        source_file: str,
        dream_id: str,
    ) -> int | None:
        """
        Submit a dream to the leaderboard.
        
        Args:
            content: The dream content (will be truncated)
            category: Dream category (code_idea, refactor, etc.)
            novelty_score: The novelty score (0.0 - 1.0)
            source_file: Source file that inspired the dream
            dream_id: Unique dream identifier (typically filename)
            
        Returns:
            New rank if dream made the leaderboard, None otherwise.
        """
        # Check if this dream is already on the leaderboard
        existing_ids = {e.dream_id for e in self.entries}
        if dream_id in existing_ids:
            return None

        # Check if it qualifies
        min_score = self.entries[-1].novelty_score if len(self.entries) >= MAX_LEADERBOARD_SIZE else 0.0
        if novelty_score <= min_score and len(self.entries) >= MAX_LEADERBOARD_SIZE:
            return None

        # Create entry - store full content for modal display
        entry = LeaderboardEntry(
            rank=0,  # Will be set during re-ranking
            content=content.strip(),  # Full content, no truncation
            category=category,
            novelty_score=novelty_score,
            source_file=Path(source_file).name if source_file else "unknown",
            timestamp=datetime.now().isoformat(),
            dream_id=dream_id,
        )

        # Add and re-rank
        self.entries.append(entry)
        self._rerank()
        self.save()

        # Find new rank
        for e in self.entries:
            if e.dream_id == dream_id:
                logger.info(f"Dream entered leaderboard at rank #{e.rank}: {dream_id}")
                return e.rank

        return None

    def get_top(self, n: int = 5) -> list[LeaderboardEntry]:
        """Get top N entries."""
        return self.entries[:n]

    def get_by_priority(self, priority: Priority) -> list[LeaderboardEntry]:
        """Get entries by priority level."""
        return [e for e in self.entries if e.priority == priority.value]

    def get_by_status(self, status: ReviewStatus) -> list[LeaderboardEntry]:
        """Get entries by review status."""
        return [e for e in self.entries if e.status == status.value]

    def get_actionable(self) -> list[LeaderboardEntry]:
        """Get pending entries sorted by priority (CRITICAL first)."""
        pending = [e for e in self.entries if e.status == ReviewStatus.PENDING.value]
        priority_order = {
            Priority.CRITICAL.value: 0,
            Priority.HIGH.value: 1,
            Priority.MEDIUM.value: 2,
            Priority.LOW.value: 3,
            Priority.UNRANKED.value: 4,
        }
        return sorted(pending, key=lambda e: (priority_order.get(e.priority, 5), -e.novelty_score))

    def update_status(
        self,
        dream_id: str,
        status: ReviewStatus,
        reason: str = "",
    ) -> bool:
        """Update the review status of a dream."""
        for entry in self.entries:
            if entry.dream_id == dream_id:
                entry.status = status.value
                entry.rejection_reason = reason
                self.save()
                logger.info(f"Updated {dream_id} status to {status.value}")
                return True
        return False

    def update_priority(
        self,
        dream_id: str,
        priority: Priority,
    ) -> bool:
        """Update the priority of a dream."""
        for entry in self.entries:
            if entry.dream_id == dream_id:
                entry.priority = priority.value
                entry.last_ranked = datetime.now().isoformat()
                self.save()
                logger.info(f"Updated {dream_id} priority to {priority.value}")
                return True
        return False

    def get_unranked(self) -> list[LeaderboardEntry]:
        """Get entries that haven't been priority-ranked yet."""
        return [e for e in self.entries if e.priority == Priority.UNRANKED.value]

    def _rerank(self) -> None:
        """Re-sort and assign ranks."""
        # Sort by novelty score (descending)
        self.entries.sort(key=lambda e: e.novelty_score, reverse=True)
        
        # Trim to max size
        self.entries = self.entries[:MAX_LEADERBOARD_SIZE]
        
        # Assign ranks
        for i, entry in enumerate(self.entries):
            entry.rank = i + 1

    def save(self) -> None:
        """Persist leaderboard to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": [e.to_dict() for e in self.entries]}
        self._path.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved leaderboard: {len(self.entries)} entries")

    def load(self) -> None:
        """Load leaderboard from disk."""
        if not self._path.exists():
            self.entries = []
            return

        try:
            data = json.loads(self._path.read_text())
            self.entries = [LeaderboardEntry.from_dict(e) for e in data.get("entries", [])]
            self._rerank()  # Ensure proper ordering
            logger.info(f"Loaded leaderboard: {len(self.entries)} entries")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load leaderboard: {e}")
            self.entries = []

    def clear(self) -> None:
        """Clear the leaderboard."""
        self.entries = []
        if self._path.exists():
            self._path.unlink()
        logger.info("Leaderboard cleared")


# Singleton instance
_leaderboard: Leaderboard | None = None


def get_leaderboard() -> Leaderboard:
    """Get the global leaderboard instance."""
    global _leaderboard
    if _leaderboard is None:
        _leaderboard = Leaderboard()
    return _leaderboard

