"""
Dream generation engine.

Generates code improvement suggestions by prompting the LLM with
codebase context at high temperature for creative exploration.

Implements two key mechanisms:
- Graph Jump: When plateau detected (3+ rejections), jump to different code area
- Deep Dive: When insight found, drill down through 4 levels to concrete code
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from .config import settings
from .indexer import CodebaseIndexer, CodeChunk
from .llm import LLMClient
from .validator import DreamValidator

logger = logging.getLogger(__name__)


class DrillLevel(IntEnum):
    """Deep Dive drilling levels - from abstract to concrete."""

    DISCOVERY = 0  # Initial observation: "this could be improved"
    FRAMEWORK = 1  # Design: "here's the architecture"
    IMPLEMENTATION = 2  # Plan: "these functions and structures"
    CODE = 3  # Actual: "here's the implementation"


@dataclass
class Dream:
    """A validated dream ready for storage."""

    content: str
    category: str
    novelty_score: float
    timestamp: datetime
    seed_context: str
    seed_file: str | None = None

    @property
    def filename(self) -> str:
        """Generate filename for this dream."""
        ts = self.timestamp.strftime("%Y%m%d_%H%M%S")
        return f"dream_{ts}_{self.category}.md"


@dataclass
class DreamCycleStats:
    """Statistics from a dream cycle."""

    dreams_generated: int = 0
    dreams_saved: int = 0
    dreams_discarded: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)


class Dreamer:
    """Generates and validates code improvement dreams."""

    # Prompt template for discovery (Level 0)
    DISCOVERY_PROMPT = """You are an expert software engineer reviewing code from a project.

Based on the following code context, think creatively about potential improvements.
Consider: bugs, performance issues, code clarity, missing error handling,
testing gaps, security concerns, or architectural improvements.

Be specific and actionable. Reference actual code elements where possible.

Code Context:
```{language}
{code}
```
File: {file_path}

What improvements do you suggest?"""

    # Deep Dive prompts for levels 1-3
    DRILL_PROMPTS = {
        DrillLevel.FRAMEWORK: """You are in DEEP DIVE mode - building on a previous insight.

PREVIOUS INSIGHT:
{previous_insight}

YOUR TASK: Design a concrete FRAMEWORK or ARCHITECTURE for this improvement.
Include:
- Specific components needed
- Data structures and their relationships
- Interface definitions
- How it integrates with existing code

Be technical and specific. This should be a blueprint someone could follow.""",
        DrillLevel.IMPLEMENTATION: """You are in DEEP DIVE mode - creating an implementation plan.

PREVIOUS FRAMEWORK:
{previous_insight}

YOUR TASK: Create a detailed IMPLEMENTATION PLAN.
Include:
- Specific function signatures with types
- Class definitions with methods
- Algorithm pseudocode
- Error handling approach
- Test cases to consider

Be concrete enough that a developer could implement this directly.""",
        DrillLevel.CODE: """You are in DEEP DIVE mode - writing actual code.

PREVIOUS PLAN:
{previous_insight}

YOUR TASK: Write the actual Python CODE.
Include:
- Complete function implementations
- Type hints
- Docstrings
- Error handling
- Example usage

Output working, production-quality Python code.""",
    }

    # Categories that trigger deep dive (actionable improvements)
    DRILLABLE_CATEGORIES = {"code_fix", "code_idea", "refactor"}

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        indexer: CodebaseIndexer | None = None,
        validator: DreamValidator | None = None,
    ) -> None:
        """
        Initialize the dreamer.

        Args:
            llm_client: LLM client for generation. Created if not provided.
            indexer: Codebase indexer. Created if not provided.
            validator: Dream validator. Created if not provided.
        """
        self.llm = llm_client or LLMClient()
        self.indexer = indexer or CodebaseIndexer()
        self.validator = validator or DreamValidator(self.llm)

        # Deep Dive state
        self._drill_level = DrillLevel.DISCOVERY
        self._last_saved_dream: str | None = None
        self._drill_failures = 0  # Consecutive failures during drilling
        self._current_seed_file: str | None = None  # Track file being drilled

    def _reset_drill_state(self) -> None:
        """Reset deep dive state for fresh discovery."""
        self._drill_level = DrillLevel.DISCOVERY
        self._last_saved_dream = None
        self._drill_failures = 0
        self._current_seed_file = None
        logger.debug("Deep dive state reset - ready for new discovery")

    def run_cycle(
        self,
        max_iterations: int | None = None,
        save_to: Path | None = None,
    ) -> tuple[list[Dream], DreamCycleStats]:
        """
        Run a single dream cycle with Deep Dive and Graph Jump.

        Deep Dive: When a novel insight is found, drill down through 4 levels
        (discovery → framework → implementation → code) before moving on.

        Graph Jump: After 3 consecutive failures, jump to a completely different
        part of the codebase to break out of repetitive patterns.

        Args:
            max_iterations: Maximum dream attempts. Defaults to settings.max_dreams_per_cycle.
            save_to: Directory to save dreams. Defaults to settings.dreams_dir.

        Returns:
            Tuple of (list of validated dreams, cycle statistics).
        """
        max_iterations = max_iterations or settings.max_dreams_per_cycle
        save_to = save_to or settings.dreams_dir
        save_to.mkdir(parents=True, exist_ok=True)

        stats = DreamCycleStats()
        dreams: list[Dream] = []
        consecutive_failures = 0
        graph_jump_used = False

        # Reset drill state at cycle start
        self._reset_drill_state()

        logger.info(f"Starting dream cycle (max {max_iterations} iterations)")

        for iteration in range(max_iterations):
            # Determine mode for logging
            mode = ""
            if self._drill_level > DrillLevel.DISCOVERY:
                mode = f"[DEEP DIVE L{self._drill_level}]"
            elif graph_jump_used:
                mode = "[GRAPH JUMP]"

            logger.info(f"Iteration {iteration + 1}/{max_iterations} {mode}")

            # === GRAPH JUMP: After 3 failures, jump to different code ===
            if consecutive_failures >= 3:
                logger.info("Plateau detected - performing GRAPH JUMP to new code area")
                self._reset_drill_state()  # Reset drilling on jump
                graph_jump_used = True
                consecutive_failures = 0

            # === SELECT SEED ===
            # If drilling, we don't need a new seed (use previous insight)
            # If discovery mode, get a seed (random if graph jumped, normal otherwise)
            seed = None
            if self._drill_level == DrillLevel.DISCOVERY:
                if graph_jump_used:
                    seed = self._select_random_seed_from_different_file()
                    graph_jump_used = False  # Reset flag
                else:
                    seed = self._select_seed()

                if seed is None:
                    logger.warning("No codebase indexed. Run 'codedreamer index' first.")
                    break

                self._current_seed_file = seed.file_path

            # === GENERATE DREAM ===
            if self._drill_level == DrillLevel.DISCOVERY:
                raw_dream = self._generate_dream(seed)  # type: ignore[arg-type]
            else:
                # Deep dive: build on previous insight
                raw_dream = self._generate_drill_dream()

            stats.dreams_generated += 1

            # === VALIDATE ===
            # Skip file cooldown during drilling (we're deliberately focusing)
            is_drilling = self._drill_level > DrillLevel.DISCOVERY
            source_file = None if is_drilling else self._current_seed_file
            validation = self.validator.validate(raw_dream, source_file=source_file)

            if validation.is_valid:
                dream = Dream(
                    content=raw_dream,
                    category=self._get_drill_category(validation.category),
                    novelty_score=validation.novelty_score,
                    timestamp=datetime.now(),
                    seed_context=self._last_saved_dream[:500] if self._last_saved_dream else "",
                    seed_file=self._current_seed_file,
                )
                dreams.append(dream)
                self._save_dream(dream, save_to)
                stats.dreams_saved += 1
                consecutive_failures = 0
                self._drill_failures = 0

                # === DEEP DIVE: Decide whether to drill deeper ===
                is_drillable = validation.category in self.DRILLABLE_CATEGORIES
                high_novelty = validation.novelty_score >= 0.5

                if self._drill_level == DrillLevel.CODE:
                    # Reached code level - success! Reset for new discovery
                    logger.info(
                        f"Dream saved: [{dream.category}] L{self._drill_level} "
                        f"novelty={validation.novelty_score:.2f} → CODE COMPLETE!"
                    )
                    self._reset_drill_state()
                elif (is_drillable or high_novelty) and self._drill_level < DrillLevel.CODE:
                    # Save insight and drill deeper
                    self._last_saved_dream = raw_dream
                    self._drill_level = DrillLevel(self._drill_level + 1)
                    logger.info(
                        f"Dream saved: [{dream.category}] → DRILLING to L{self._drill_level}"
                    )
                else:
                    # Not drillable (story, low novelty insight) - reset
                    logger.info(
                        f"Dream saved: [{dream.category}] novelty={validation.novelty_score:.2f}"
                    )
                    self._reset_drill_state()
            else:
                stats.dreams_discarded += 1
                reason = validation.rejection_reason or "Unknown"
                stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1
                consecutive_failures += 1
                self._drill_failures += 1

                logger.debug(f"Dream discarded: {reason}")

                # If drilling fails twice, give up on this drill and reset
                if self._drill_level > DrillLevel.DISCOVERY and self._drill_failures >= 2:
                    logger.info(
                        f"Deep dive exhausted at L{self._drill_level} - resetting"
                    )
                    self._reset_drill_state()

        logger.info(
            f"Dream cycle complete: {stats.dreams_saved}/{stats.dreams_generated} saved"
        )
        return dreams, stats

    def _generate_drill_dream(self) -> str:
        """Generate a deep dive dream building on previous insight."""
        if not self._last_saved_dream:
            logger.warning("No previous insight for drilling - using empty context")
            return ""

        prompt_template = self.DRILL_PROMPTS.get(self._drill_level)
        if not prompt_template:
            logger.error(f"No prompt for drill level {self._drill_level}")
            return ""

        prompt = prompt_template.format(
            previous_insight=self._last_saved_dream[:1500]
        )

        # Slightly lower temperature for more focused drilling
        temp = max(0.6, settings.dream_temperature - 0.1 * self._drill_level)

        # More tokens for deeper levels (code needs more space)
        extra_tokens = 100 * self._drill_level
        result = self.llm.generate(
            prompt,
            max_tokens=settings.dream_max_tokens + extra_tokens,
            temperature=temp,
            repeat_penalty=1.1,
        )

        return result.text

    def _get_drill_category(self, base_category: str) -> str:
        """Add drill level suffix to category for tracking."""
        if self._drill_level > DrillLevel.DISCOVERY:
            return f"{base_category}_L{self._drill_level}"
        return base_category

    def _select_random_seed_from_different_file(self) -> CodeChunk | None:
        """Select a random chunk from a DIFFERENT file (for graph jump)."""
        max_attempts = 10
        for _ in range(max_attempts):
            chunk = self.indexer.get_random_chunk()
            if chunk is None:
                return None
            # Ensure it's from a different file
            if chunk.file_path != self._current_seed_file:
                logger.debug(f"Graph jump: {self._current_seed_file} → {chunk.file_path}")
                return chunk
        # Fallback: just return any chunk
        return self.indexer.get_random_chunk()

    def dream_once(self, seed: CodeChunk | None = None) -> Dream | None:
        """
        Generate a single dream.

        Args:
            seed: Optional specific code chunk to dream about.
                  If None, selects randomly from indexed code.

        Returns:
            Validated Dream or None if generation/validation failed.
        """
        if seed is None:
            seed = self._select_seed()
            if seed is None:
                logger.warning("No codebase indexed")
                return None

        raw_dream = self._generate_dream(seed)
        validation = self.validator.validate(raw_dream, source_file=seed.file_path)

        if not validation.is_valid:
            logger.debug(f"Dream rejected: {validation.rejection_reason}")
            return None

        return Dream(
            content=raw_dream,
            category=validation.category,
            novelty_score=validation.novelty_score,
            timestamp=datetime.now(),
            seed_context=seed.content[:500],
            seed_file=seed.file_path,
        )

    def _select_seed(self) -> CodeChunk | None:
        """Select a code chunk to use as dream seed."""
        return self.indexer.get_random_chunk()

    def _generate_dream(self, seed: CodeChunk) -> str:
        """Generate discovery-level dream from a seed chunk."""
        prompt = self.DISCOVERY_PROMPT.format(
            language=seed.language,
            code=seed.content[:2000],  # Limit context size
            file_path=seed.file_path,
        )

        result = self.llm.generate(
            prompt,
            max_tokens=settings.dream_max_tokens,
            temperature=settings.dream_temperature,
            repeat_penalty=1.15,  # Prevent repetition loops while still creative
        )

        return result.text

    def _save_dream(self, dream: Dream, directory: Path) -> Path:
        """Save a dream to disk as markdown."""
        filepath = directory / dream.filename

        content = f"""# {dream.category.replace('_', ' ').title()}

**Generated**: {dream.timestamp.isoformat()}
**Novelty Score**: {dream.novelty_score:.2f}
**Source File**: {dream.seed_file or 'Unknown'}

---

{dream.content}

---

## Seed Context

```
{dream.seed_context}
```
"""
        filepath.write_text(content)
        return filepath

