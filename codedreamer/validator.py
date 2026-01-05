"""
Dream validation and novelty scoring.

Implements the "lucid check" - self-validation of dream quality,
and novelty tracking to avoid repetitive suggestions.

Enhanced with semantic deduplication to reduce noise in dream generation.

Enhanced repetition handling:
- Time-based theme decay (24h default)
- Exponential penalty for over-threshold themes
- Project-aware domain term extraction
"""

import hashlib
import logging
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, ClassVar

from .config import settings
from .llm import LLMClient

if TYPE_CHECKING:
    from .indexer import CodebaseIndexer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of dream validation."""

    is_valid: bool
    novelty_score: float
    category: str
    rejection_reason: str | None = None
    was_refined: bool = False  # True if critic loop refined the dream
    theme_warnings: list[str] | None = None  # Warnings about approaching theme limits


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
    """
    Validates dream quality and tracks novelty.
    
    Features:
    - Exponential penalty for themes exceeding repetition threshold
    - Time-based theme decay (themes expire after 24 hours)
    - Project-aware domain terms extracted from indexed codebase
    """

    # Fallback domain terms - used if no project terms available
    # These should be SPECIFIC patterns, not common programming words
    # Avoid: handler, manager, client, error, config - too common!
    FALLBACK_DOMAIN_TERMS: ClassVar[list[str]] = [
        # Specific architectural patterns
        "singleton", "factory", "observer", "decorator", "middleware", "facade",
        # Specific performance patterns  
        "memoize", "memoization", "throttle", "debounce", "pooling",
        # Specific concurrency patterns
        "mutex", "semaphore", "deadlock", "race condition",
        # Specific error patterns
        "circuit breaker", "backoff", "retry strategy",
        # Specific data patterns
        "serialization", "deserialization", "marshalling",
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
        
        # Repetition handling configuration
        self.theme_decay_hours = 24          # Themes expire after 24 hours
        self.max_theme_history = 100         # Max themes to track
        self.repetition_threshold = 3        # After 3 occurrences, apply exponential penalty
        self.novelty_weight = 0.4            # Weight of theme novelty in final score

        # Project-specific domain terms (populated from indexed codebase)
        self._project_terms: set[str] = set()
        self._project_terms_initialized = False

        # Semantic deduplication: track recent dreams and analyzed files
        # Use deque for O(1) append/trim instead of O(n) list slicing (per dream_20251229_184523)
        self.max_recent_dreams = 20
        self.recent_dreams: deque[str] = deque(maxlen=self.max_recent_dreams)
        self.similarity_threshold = 0.7  # Reject if >70% similar

        # File cooldown: avoid analyzing same file repeatedly
        self.file_cooldowns: dict[str, datetime] = {}
        self.file_cooldown_minutes = 30

    def initialize_project_terms(self, indexer: "CodebaseIndexer") -> None:
        """
        Extract project-specific domain terms from the indexed codebase.
        
        Instead of generic terms like
        "function" or "class", we track terms specific to THIS project like
        "conductor", "dreamer", "validator", etc.
        
        Args:
            indexer: The codebase indexer to extract terms from.
        """
        if self._project_terms_initialized:
            return
            
        try:
            # Get all chunk metadata from the collection
            results = indexer.collection.get(include=["metadatas"])
            
            term_counts: dict[str, int] = defaultdict(int)
            
            for metadata in results.get("metadatas", []):
                if not metadata:
                    continue
                    
                # Extract from file paths (e.g., "conductor.py" -> "conductor")
                file_path = metadata.get("file_path", "")
                if file_path:
                    # Get the filename without extension
                    filename = file_path.split("/")[-1].rsplit(".", 1)[0]
                    if len(filename) >= 4 and filename not in {"__init__", "main", "test", "utils"}:
                        term_counts[filename.lower()] += 1
                
                # Extract from chunk names (function/class names)
                # Only use FULL names, not parts - avoids common words like "handler", "manager"
                name = metadata.get("name", "")
                if name and len(name) >= 6:  # Minimum 6 chars to filter common words
                    # Common words to exclude (appear in too many dreams)
                    common_words = {
                        "handler", "manager", "client", "server", "config", "settings",
                        "error", "message", "response", "request", "session", "socket",
                        "service", "controller", "model", "schema", "utils", "helper",
                        "async", "await", "class", "function", "method", "return",
                    }
                    name_lower = name.lower()
                    if name_lower not in common_words:
                        term_counts[name_lower] += 1
            
            # Take top 50 most common terms (these are project-specific)
            sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            self._project_terms = {term for term, count in sorted_terms[:50] if count >= 2}
            
            logger.info(
                f"Initialized {len(self._project_terms)} project-specific domain terms: "
                f"{list(self._project_terms)[:10]}..."
            )
            self._project_terms_initialized = True
            
        except Exception as e:
            logger.warning(f"Failed to extract project terms: {e}")
            self._project_terms_initialized = True  # Don't retry on failure

    def get_avoidance_prompt(self) -> str:
        """
        Generate a prompt section telling the model what topics to AVOID.
        
        Inject overused themes into the prompt so the model
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

        # Get theme warnings BEFORE recording (so we see current state)
        theme_warnings = self._get_theme_warnings(dream_content)
        
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
            theme_warnings=theme_warnings,
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

        Implements the Critic pattern - asking "what's wrong with this?"
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
        
        Exponential penalty:
        - Below threshold: linear penalty (0.1 * count)
        - At/above threshold: exponential penalty (0.3 * (count - threshold + 1))

        Returns:
            Float between 0 (completely repetitive) and 1 (completely novel).
        """
        self._decay_themes()

        themes = self._extract_themes(content)
        if not themes:
            return 1.0  # Novel by default if no themes detected (changed from 0.5)

        total_penalty = 0.0
        theme_count = 0

        for theme in themes:
            if theme in self.theme_history:
                entry = self.theme_history[theme]
                occurrences = entry.count
                
                # Exponential penalty for repetition
                if occurrences >= self.repetition_threshold:
                    # Exponential penalty once threshold exceeded
                    penalty = 0.3 * (occurrences - self.repetition_threshold + 1)
                else:
                    # Linear penalty below threshold
                    penalty = 0.1 * occurrences
                
                total_penalty += penalty
                theme_count += 1

        if theme_count == 0:
            return 1.0

        avg_penalty = total_penalty / theme_count
        novelty = max(0.0, 1.0 - avg_penalty)
        
        logger.debug(
            f"[NOVELTY] Score: {novelty:.2f} (themes: {theme_count}, "
            f"avg_penalty: {avg_penalty:.2f}, threshold: {self.repetition_threshold})"
        )

        return novelty

    def _extract_themes(self, content: str) -> list[str]:
        """
        Extract domain-relevant themes from content.
        
        Uses project-specific terms if available, falls back to
        high-signal generic terms.
        """
        content_lower = content.lower()
        
        # Prefer project-specific terms
        if self._project_terms:
            themes = [term for term in self._project_terms if term in content_lower]
            # Also check fallback terms for cross-project patterns
            themes.extend(
                term for term in self.FALLBACK_DOMAIN_TERMS 
                if term in content_lower and term not in themes
            )
            return themes
        
        # Fallback to generic high-signal terms
        return [term for term in self.FALLBACK_DOMAIN_TERMS if term in content_lower]

    def _get_theme_warnings(self, content: str) -> list[str]:
        """Get warnings for themes approaching the repetition limit."""
        themes = self._extract_themes(content)
        warnings = []
        
        for theme in themes:
            if theme in self.theme_history:
                count = self.theme_history[theme].count
                # Warn if this would be 2nd or 3rd occurrence
                if count >= 1:
                    next_count = count + 1
                    warnings.append(
                        f"Theme '{theme}': occurrence {next_count}/{self.repetition_threshold}"
                    )
        
        return warnings if warnings else None

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
        # Find overused themes (at or above threshold)
        overused_themes = [
            (theme, entry.count) 
            for theme, entry in self.theme_history.items() 
            if entry.count >= self.repetition_threshold
        ]
        
        return {
            "recent_dreams_count": len(self.recent_dreams),
            "content_hashes_count": len(self.content_hashes),
            "theme_history_count": len(self.theme_history),
            "project_terms_count": len(self._project_terms),
            "project_terms_initialized": self._project_terms_initialized,
            "overused_themes": overused_themes,
            "files_on_cooldown": len(
                [f for f in self.file_cooldowns if self._is_file_on_cooldown(f)]
            ),
            "similarity_threshold": self.similarity_threshold,
            "file_cooldown_minutes": self.file_cooldown_minutes,
            "theme_decay_hours": self.theme_decay_hours,
            "repetition_threshold": self.repetition_threshold,
        }


# Singleton instance
_validator: DreamValidator | None = None


def get_validator() -> DreamValidator:
    """Get the shared DreamValidator instance."""
    global _validator
    if _validator is None:
        _validator = DreamValidator()
    return _validator

