"""
Dream validation and novelty scoring.

Implements the "lucid check" - self-validation of dream quality,
and novelty tracking to avoid repetitive suggestions.

Enhanced with semantic deduplication to reduce noise in dream generation.
"""

import hashlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import ClassVar

from .config import settings
from .llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dream validation."""

    is_valid: bool
    novelty_score: float
    category: str
    rejection_reason: str | None = None
    was_refined: bool = False  # True if critic loop refined the dream


@dataclass
class CritiqueResult:
    """Result of critic loop analysis."""

    has_issues: bool
    critique: str
    severity: str  # "minor", "moderate", "major"
    refined_content: str | None = None


@dataclass
class ThemeEntry:
    """Tracked theme with occurrence count and recency."""

    theme: str
    count: int = 1
    last_seen: datetime = field(default_factory=datetime.now)


class DreamValidator:
    """Validates dream quality and tracks novelty."""

    # Domain terms to track for novelty scoring
    DOMAIN_TERMS: ClassVar[list[str]] = [
        "refactor",
        "optimize",
        "cache",
        "parallel",
        "async",
        "error",
        "exception",
        "logging",
        "test",
        "validate",
        "security",
        "performance",
        "memory",
        "database",
        "api",
        "endpoint",
        "config",
        "dependency",
        "import",
        "export",
        "function",
        "class",
        "method",
        "variable",
        "type",
        "interface",
    ]

    # Category keywords for classification
    CATEGORY_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "code_fix": ["bug", "fix", "error", "issue", "problem", "broken", "incorrect"],
        "code_idea": ["add", "implement", "create", "new", "feature", "enhance"],
        "insight": ["pattern", "notice", "observe", "consider", "architecture"],
        "refactor": ["refactor", "clean", "simplify", "restructure", "reorganize"],
    }

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """
        Initialize the validator.

        Args:
            llm_client: LLM client for lucid checks. If None, lucid check is skipped.
        """
        self.llm_client = llm_client
        self.theme_history: dict[str, ThemeEntry] = {}
        self.content_hashes: set[str] = set()
        self.theme_decay_hours = 24
        self.max_theme_history = 100
        self.repetition_threshold = 3

        # Semantic deduplication: track recent dreams and analyzed files
        # Use deque for O(1) append/trim instead of O(n) list slicing (per dream_20251229_184523)
        self.max_recent_dreams = 20
        self.recent_dreams: deque[str] = deque(maxlen=self.max_recent_dreams)
        self.similarity_threshold = 0.7  # Reject if >70% similar

        # File cooldown: avoid analyzing same file repeatedly
        self.file_cooldowns: dict[str, datetime] = {}
        self.file_cooldown_minutes = 30

    def get_avoidance_prompt(self) -> str:
        """
        Generate a prompt section telling the model what topics to AVOID.
        
        ZetaZero-style: inject overused themes into the prompt so the model
        knows what NOT to generate, rather than rejecting after the fact.
        """
        self._decay_themes()
        
        # Find overused themes
        overused = []
        for theme, entry in self.theme_history.items():
            if entry.count >= self.repetition_threshold:
                overused.append((theme, entry.count))
        
        if not overused:
            return ""
        
        # Sort by count descending
        overused.sort(key=lambda x: x[1], reverse=True)
        
        # Build avoidance prompt (top 5 themes)
        theme_list = ", ".join(t[0] for t in overused[:5])
        
        return f"""
## Topics to AVOID (already covered extensively)
{theme_list}

Focus on UNEXPLORED aspects of the codebase. Think of something NEW and DIFFERENT.
"""

    def validate(
        self, dream_content: str, source_file: str | None = None
    ) -> ValidationResult:
        """
        Validate a dream for quality and novelty.

        Args:
            dream_content: The generated dream text.
            source_file: Optional source file path for cooldown tracking.

        Returns:
            ValidationResult with validity, score, and category.
        """
        # Check for exact duplicates
        content_hash = self._hash_content(dream_content)
        if content_hash in self.content_hashes:
            return ValidationResult(
                is_valid=False,
                novelty_score=0.0,
                category="unknown",
                rejection_reason="Exact duplicate",
            )

        # Check minimum length
        if len(dream_content.strip()) < 100:
            return ValidationResult(
                is_valid=False,
                novelty_score=0.0,
                category="unknown",
                rejection_reason="Too short",
            )

        # Check file cooldown
        if source_file and self._is_file_on_cooldown(source_file):
            return ValidationResult(
                is_valid=False,
                novelty_score=0.0,
                category=self._categorize(dream_content),
                rejection_reason=f"File on cooldown: {source_file}",
            )

        # Semantic similarity check against recent dreams
        similarity_score = self._check_semantic_similarity(dream_content)
        if similarity_score > self.similarity_threshold:
            return ValidationResult(
                is_valid=False,
                novelty_score=1.0 - similarity_score,
                category=self._categorize(dream_content),
                rejection_reason=f"Too similar to recent dream ({similarity_score:.0%})",
            )

        # Lucid check if enabled and LLM available
        if (
            settings.lucid_check_enabled
            and self.llm_client
            and not self._lucid_check(dream_content)
        ):
            return ValidationResult(
                is_valid=False,
                novelty_score=0.0,
                category=self._categorize(dream_content),
                rejection_reason="Failed lucid check",
            )

        # Calculate novelty score (theme-based)
        novelty_score = self._score_novelty(dream_content)

        # Combine with semantic novelty
        semantic_novelty = 1.0 - similarity_score
        combined_novelty = (novelty_score + semantic_novelty) / 2

        logger.debug(
            f"Novelty calculation: theme={novelty_score:.2f}, "
            f"semantic={semantic_novelty:.2f} (sim={similarity_score:.2f}), "
            f"combined={combined_novelty:.2f}, "
            f"recent_dreams_count={len(self.recent_dreams)}"
        )

        if combined_novelty < settings.novelty_threshold:
            return ValidationResult(
                is_valid=False,
                novelty_score=combined_novelty,
                category=self._categorize(dream_content),
                rejection_reason=f"Too repetitive (novelty={combined_novelty:.2f})",
            )

        # Dream is valid - record it
        self.content_hashes.add(content_hash)
        self._record_themes(dream_content)
        self._record_dream(dream_content)
        if source_file:
            self._record_file_access(source_file)

        return ValidationResult(
            is_valid=True,
            novelty_score=combined_novelty,
            category=self._categorize(dream_content),
        )

    def _lucid_check(self, content: str) -> bool:
        """
        Self-validation: ask the model if the dream is useful.

        Uses low temperature for consistent yes/no response.
        Returns False on error to avoid false positives (per dream_20251229_171627).
        """
        # Input validation (per dream_20251229_171627)
        if not isinstance(content, str) or not content.strip():
            return False

        if not self.llm_client:
            return True

        prompt = f"""Review this code improvement suggestion and determine if it is:
1. Specific (mentions concrete code elements)
2. Actionable (describes a clear change to make)
3. Useful (would improve the codebase)

Suggestion:
{content[:800]}

Answer only YES if all three criteria are met, otherwise NO.
Answer:"""

        try:
            result = self.llm_client.generate(
                prompt,
                max_tokens=10,
                temperature=0.1,  # Low temp for deterministic response
            )
            response = result.text.strip().upper()
            return "YES" in response
        except Exception as e:
            logger.warning(f"Lucid check failed: {e}")
            return False  # Reject on error to avoid false positives

    def critique_and_refine(
        self, content: str, max_refinements: int = 1
    ) -> tuple[str, CritiqueResult | None]:
        """
        Critic Loop: Self-critique the dream and optionally refine it.

        Implements the ZetaZero Critic pattern - asking "what's wrong with this?"
        before accepting output, and refining if issues are found.

        Args:
            content: The dream content to critique.
            max_refinements: Maximum number of refinement attempts (default 1).

        Returns:
            Tuple of (final_content, critique_result).
            final_content may be refined version if critique found issues.
        """
        if not self.llm_client:
            return content, None

        # === PHASE 1: Critique ===
        critique_prompt = f"""You are a critical code reviewer. Analyze this code improvement suggestion and identify any weaknesses.

## Suggestion to Review:
{content[:1500]}

## Critique Checklist:
1. **Specificity**: Does it name exact functions/classes/lines? Or is it vague?
2. **Accuracy**: Are the claims about the code correct? Any misconceptions?
3. **Feasibility**: Is the proposed change actually implementable?
4. **Completeness**: Does it miss important edge cases or considerations?
5. **Originality**: Is this a genuine insight or generic advice?

## Your Response Format:
SEVERITY: [NONE|MINOR|MODERATE|MAJOR]
ISSUES:
- [List specific issues, or "None found" if the suggestion is solid]

Be honest and critical. A good suggestion should pass scrutiny."""

        try:
            critique_result = self.llm_client.generate(
                critique_prompt,
                max_tokens=400,
                temperature=0.3,
            )
            critique_text = critique_result.text.strip()

            # Parse severity
            severity = "minor"
            if "MAJOR" in critique_text.upper():
                severity = "major"
            elif "MODERATE" in critique_text.upper():
                severity = "moderate"
            elif "NONE" in critique_text.upper()[:50]:
                severity = "none"

            has_issues = severity in ("moderate", "major")

            logger.debug(f"Critic verdict: {severity} - {critique_text[:100]}...")

            # If no significant issues, return original
            if not has_issues or max_refinements <= 0:
                return content, CritiqueResult(
                    has_issues=has_issues,
                    critique=critique_text,
                    severity=severity,
                    refined_content=None,
                )

            # === PHASE 2: Refine based on critique ===
            logger.info(f"Critic found {severity} issues - requesting refinement...")

            refine_prompt = f"""You are improving a code suggestion based on critical feedback.

## Original Suggestion:
{content[:1200]}

## Critique (issues identified):
{critique_text[:600]}

## Your Task:
Rewrite the suggestion to address the critique. Make it:
- More specific (name exact code elements)
- More accurate (fix any misconceptions)
- More complete (address edge cases)
- More actionable (clearer implementation steps)

Provide ONLY the improved suggestion, no meta-commentary:"""

            refined_result = self.llm_client.generate(
                refine_prompt,
                max_tokens=1500,
                temperature=0.5,
            )
            refined_content = refined_result.text.strip()

            # Only use refined version if it's substantial
            if len(refined_content) > 200:
                logger.info(f"Refinement successful: {len(content)} → {len(refined_content)} chars")
                return refined_content, CritiqueResult(
                    has_issues=True,
                    critique=critique_text,
                    severity=severity,
                    refined_content=refined_content,
                )
            else:
                logger.warning("Refinement too short, using original")
                return content, CritiqueResult(
                    has_issues=True,
                    critique=critique_text,
                    severity=severity,
                    refined_content=None,
                )

        except Exception as e:
            logger.warning(f"Critic loop failed: {e}")
            return content, None

    def _score_novelty(self, content: str) -> float:
        """
        Score content novelty based on theme repetition.

        Returns:
            Float between 0 (completely repetitive) and 1 (completely novel).
        """
        self._decay_themes()

        themes = self._extract_themes(content)
        if not themes:
            return 0.5  # Neutral if no themes detected

        penalties = []
        for theme in themes:
            if theme in self.theme_history:
                entry = self.theme_history[theme]
                # Penalty increases with repetition count
                penalty = min(1.0, entry.count * 0.3)
                penalties.append(penalty)
            else:
                penalties.append(0.0)

        avg_penalty = sum(penalties) / len(penalties) if penalties else 0.0
        return max(0.0, 1.0 - avg_penalty)

    def _extract_themes(self, content: str) -> list[str]:
        """Extract domain-relevant themes from content."""
        content_lower = content.lower()
        return [term for term in self.DOMAIN_TERMS if term in content_lower]

    def _record_themes(self, content: str) -> None:
        """Record themes from validated dream."""
        themes = self._extract_themes(content)
        now = datetime.now()

        for theme in themes:
            if theme in self.theme_history:
                self.theme_history[theme].count += 1
                self.theme_history[theme].last_seen = now
            else:
                self.theme_history[theme] = ThemeEntry(theme=theme, last_seen=now)

        # Prune old themes if over limit
        if len(self.theme_history) > self.max_theme_history:
            sorted_themes = sorted(
                self.theme_history.items(),
                key=lambda x: x[1].last_seen,
            )
            for theme, _ in sorted_themes[: len(sorted_themes) - self.max_theme_history]:
                del self.theme_history[theme]

    def _decay_themes(self) -> None:
        """Remove themes older than decay threshold."""
        cutoff = datetime.now() - timedelta(hours=self.theme_decay_hours)
        expired = [
            theme for theme, entry in self.theme_history.items() if entry.last_seen < cutoff
        ]
        for theme in expired:
            del self.theme_history[theme]

    def _categorize(self, content: str) -> str:
        """Categorize dream based on content keywords."""
        content_lower = content.lower()

        category_scores: dict[str, int] = defaultdict(int)
        for category, keywords in self.CATEGORY_PATTERNS.items():
            for keyword in keywords:
                if keyword in content_lower:
                    category_scores[category] += 1

        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "insight"  # Default category

    def _hash_content(self, content: str) -> str:
        """Generate hash for duplicate detection.
        
        Uses SHA256 for better collision resistance (per dream_20251229_164501).
        """
        # Normalize whitespace before hashing
        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    # --- Semantic Deduplication Methods ---

    def _check_semantic_similarity(self, content: str) -> float:
        """
        Check semantic similarity against recent dreams.

        Uses SequenceMatcher for fast approximate matching.
        Returns highest similarity score (0.0 to 1.0).
        """
        if not self.recent_dreams:
            return 0.0

        # Normalize content for comparison
        normalized = " ".join(content.lower().split())[:1000]

        max_similarity = 0.0
        for recent in self.recent_dreams:
            recent_normalized = " ".join(recent.lower().split())[:1000]
            ratio = SequenceMatcher(None, normalized, recent_normalized).ratio()
            max_similarity = max(max_similarity, ratio)

            # Early exit if we find a very similar dream
            if max_similarity > self.similarity_threshold:
                break

        return max_similarity

    def _record_dream(self, content: str) -> None:
        """Record a dream for future similarity checks."""
        # deque with maxlen automatically discards oldest when full (O(1))
        self.recent_dreams.append(content)
        logger.debug(f"Recorded dream. Total recent dreams: {len(self.recent_dreams)}")

    def _is_file_on_cooldown(self, file_path: str) -> bool:
        """Check if a file is on cooldown (recently analyzed)."""
        if file_path not in self.file_cooldowns:
            return False

        last_access = self.file_cooldowns[file_path]
        cooldown_end = last_access + timedelta(minutes=self.file_cooldown_minutes)
        return datetime.now() < cooldown_end

    def _record_file_access(self, file_path: str) -> None:
        """Record that a file was just analyzed."""
        self.file_cooldowns[file_path] = datetime.now()

        # Clean up old entries
        cutoff = datetime.now() - timedelta(hours=2)
        expired = [f for f, t in self.file_cooldowns.items() if t < cutoff]
        for f in expired:
            del self.file_cooldowns[f]

    def get_stats(self) -> dict:
        """Get validator statistics for monitoring."""
        return {
            "recent_dreams_count": len(self.recent_dreams),
            "content_hashes_count": len(self.content_hashes),
            "theme_history_count": len(self.theme_history),
            "files_on_cooldown": len(
                [f for f in self.file_cooldowns if self._is_file_on_cooldown(f)]
            ),
            "similarity_threshold": self.similarity_threshold,
            "file_cooldown_minutes": self.file_cooldown_minutes,
        }


# Singleton instance
_validator: DreamValidator | None = None


def get_validator() -> DreamValidator:
    """Get the shared DreamValidator instance."""
    global _validator
    if _validator is None:
        _validator = DreamValidator()
    return _validator

