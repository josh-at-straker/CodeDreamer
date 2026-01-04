"""
Autonomous dream daemon.

Runs the complete dream loop:
1. Index codebase on startup (if not already indexed)
2. Continuous dream generation via Conductor
3. Automatic graph decay on schedule
4. Persist dreams to disk
"""

import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from .conductor import TaskType, get_conductor
from .config import settings
from .graph import NodeType, get_graph
from .indexer import CodebaseIndexer
from .models import get_orchestra
from .validator import DreamValidator

logger = logging.getLogger(__name__)


class DreamDaemon:
    """
    Autonomous dreaming daemon.

    Indexes codebase, then continuously generates and validates
    code improvement suggestions while you sleep.
    """

    def __init__(
        self,
        codebase_path: Path | None = None,
        dream_interval: int | None = None,
        decay_interval: int | None = None,
    ) -> None:
        """
        Initialize the daemon.

        Args:
            codebase_path: Path to codebase to analyze. Required.
            dream_interval: Seconds between dream cycles. Defaults to settings.
            decay_interval: Seconds between graph decay cycles. Defaults to settings.
        """
        self.codebase_path = codebase_path
        self.dream_interval = dream_interval or settings.dream_interval_sec
        # Use configurable decay interval (per dream_20251229_165715)
        self.decay_interval = decay_interval or settings.decay_interval_sec

        self.conductor = get_conductor()
        self.graph = get_graph()
        self.indexer = CodebaseIndexer()
        self.validator = DreamValidator()
        self.orchestra = get_orchestra()

        self.scheduler = BackgroundScheduler()
        self._running = False
        self._dreams_generated = 0
        self._dreams_saved = 0

        # Ensure dreams directory exists
        settings.dreams_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the daemon."""
        logger.info("Starting CodeDreamer daemon...")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Index codebase if provided
        if self.codebase_path:
            self._index_codebase()

        # Check we have something to dream about
        if self.indexer.collection.count() == 0:
            logger.error("No codebase indexed. Provide DREAMER_CODEBASE_PATH.")
            sys.exit(1)

        # Schedule jobs
        self.scheduler.add_job(
            self._dream_cycle,
            "interval",
            seconds=self.dream_interval,
            id="dream_cycle",
            next_run_time=datetime.now(),  # Run immediately
        )

        self.scheduler.add_job(
            self._decay_cycle,
            "interval",
            seconds=self.decay_interval,
            id="decay_cycle",
        )

        self.scheduler.start()
        self._running = True

        logger.info(
            f"Daemon started. Dream interval: {self.dream_interval}s, "
            f"Decay interval: {self.decay_interval}s"
        )

        # Keep main thread alive
        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._shutdown(None, None)

    def _index_codebase(self) -> None:
        """Index the codebase if not already indexed or if changed."""
        if not self.codebase_path or not self.codebase_path.exists():
            logger.warning(f"Codebase path not found: {self.codebase_path}")
            return

        current_count = self.indexer.collection.count()

        if current_count > 0:
            logger.info(f"Codebase already indexed ({current_count} chunks). Skipping.")
            return

        logger.info(f"Indexing codebase: {self.codebase_path}")
        stats = self.indexer.index_directory(self.codebase_path)
        logger.info(
            f"Indexed {stats.files_processed} files, "
            f"{stats.chunks_created} chunks created"
        )

    def _dream_cycle(self) -> None:
        """Run a single dream cycle."""
        logger.info("Starting dream cycle...")

        for _ in range(settings.max_dreams_per_cycle):
            try:
                dream = self._generate_dream()
                if dream:
                    self._dreams_saved += 1
                    logger.info(f"Dream saved ({self._dreams_saved} total)")
            except Exception as e:
                logger.exception(f"Dream generation error: {e}")

        logger.info(
            f"Dream cycle complete. "
            f"Total: {self._dreams_saved}/{self._dreams_generated} saved"
        )

    def _generate_dream(self) -> dict | None:
        """Generate a single dream using the full pipeline."""
        # Get a random code chunk as seed
        seed = self.indexer.get_random_chunk()
        if not seed:
            logger.warning("No code chunks available for dreaming")
            return None

        self._dreams_generated += 1

        # Build prompt with context
        prompt = f"""Analyze this code and suggest a specific improvement.
Be concrete: name the file, function, or pattern you're addressing.
Explain why the change would help and how to implement it.

Code from {seed.file_path}:
```{seed.language}
{seed.content[:2000]}
```

Improvement suggestion:"""

        # Process through conductor
        result = self.conductor.process(prompt, TaskType.DREAM)

        if not result.success:
            logger.warning(f"Dream generation failed: {result.error}")
            return None

        # Validate (with source file for cooldown/dedup)
        validation = self.validator.validate(result.output, source_file=seed.file_path)

        if not validation.is_valid:
            logger.debug(f"Dream rejected: {validation.rejection_reason}")
            return None

        # Save to disk
        dream_data = {
            "content": result.output,
            "category": validation.category,
            "novelty_score": validation.novelty_score,
            "source_file": seed.file_path,
            "timestamp": datetime.now().isoformat(),
            "steps": result.steps_taken,
            "duration_ms": result.duration_ms,
        }

        self._save_dream(dream_data)

        # Add to knowledge graph
        self.graph.add_node(
            content=result.output[:500],
            node_type=NodeType.DREAM,
            metadata={
                "category": validation.category,
                "source": seed.file_path,
                "novelty": validation.novelty_score,
            },
        )
        self.graph.save()

        return dream_data

    def _save_dream(self, dream: dict) -> Path:
        """Save dream to disk as markdown."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dream_{ts}_{dream['category']}.md"
        filepath = settings.dreams_dir / filename

        content = f"""# {dream['category'].replace('_', ' ').title()}

**Generated**: {dream['timestamp']}
**Novelty Score**: {dream['novelty_score']:.2f}
**Source File**: {dream['source_file']}
**Processing**: {dream['steps']} steps, {dream['duration_ms']}ms

---

{dream['content']}
"""
        filepath.write_text(content)
        logger.info(f"Saved: {filepath.name}")
        return filepath

    def _decay_cycle(self) -> None:
        """Apply decay to knowledge graph."""
        archived = self.graph.decay_all()
        pruned = self.graph.prune_cold(threshold=0.05)
        self.graph.save()

        if archived or pruned:
            logger.info(f"Decay cycle: {archived} decayed, {pruned} pruned")

    def _shutdown(self, signum: int | None, frame: object) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down daemon...")
        self._running = False

        self.scheduler.shutdown(wait=False)
        self.graph.save()
        self.orchestra.unload_all()

        logger.info(
            f"Shutdown complete. Generated {self._dreams_generated} dreams, "
            f"saved {self._dreams_saved}"
        )
        sys.exit(0)


def run_daemon(
    codebase_path: Path | None = None,
    dream_interval: int | None = None,
) -> None:
    """Entry point for the daemon."""
    daemon = DreamDaemon(
        codebase_path=codebase_path,
        dream_interval=dream_interval,
    )
    daemon.start()

