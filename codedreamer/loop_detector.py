"""
Loop detection for generated content.

Implements active loop detection to catch and truncate
repetitive content before it ruins dream quality.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conclusion/summary phrases that indicate potential looping
CONCLUSION_MARKERS = [
    "Therefore",
    "In summary",
    "Thus,",
    "Hence,",
    "In conclusion",
    "To summarize",
    "To sum up",
    "As a result",
    "Consequently",
    "The answer is",
    "Final answer",
    "In short,",
    "Overall,",
    "To conclude",
    "### Summary",
    "## Summary",
    "### Conclusion",
    "## Conclusion",
    "### Final",
    "## Final",
]

# Minimum block size for repetition detection
MIN_BLOCK_SIZE = 50
# Maximum distance to look back for repeated blocks
MAX_LOOKBACK = 500


@dataclass
class LoopDetectionResult:
    """Result of loop detection analysis."""
    
    has_loop: bool
    loop_type: str | None  # "block", "conclusion", "code_block"
    truncate_at: int | None  # Character position to truncate at
    details: str | None


def detect_repeated_blocks(text: str, block_size: int = MIN_BLOCK_SIZE) -> LoopDetectionResult:
    """
    Detect repeated text blocks.
    
    Checks if the last N characters appear earlier in the text,
    indicating a generation loop.
    """
    if len(text) < block_size * 2:
        return LoopDetectionResult(False, None, None, None)
    
    # Get the last block
    last_block = text[-block_size:]
    
    # Look for it earlier in the text (not overlapping with current position)
    search_area = text[:-block_size]
    prev_pos = search_area.rfind(last_block)
    
    if prev_pos != -1:
        # Found a repeat - check how recent
        distance = len(text) - block_size - prev_pos
        if distance < MAX_LOOKBACK:
            logger.warning(f"Detected repeated {block_size}-char block at distance {distance}")
            return LoopDetectionResult(
                has_loop=True,
                loop_type="block",
                truncate_at=prev_pos + block_size,  # Keep first occurrence
                details=f"Repeated {block_size}-char block, distance={distance}"
            )
    
    return LoopDetectionResult(False, None, None, None)


def detect_conclusion_spam(text: str, threshold: int = 2) -> LoopDetectionResult:
    """
    Detect repeated conclusion/summary phrases.
    
    When a model is stuck, it often outputs multiple "In conclusion" or
    "Therefore" phrases in quick succession.
    """
    # Check the last portion of text
    tail_size = min(500, len(text))
    tail = text[-tail_size:] if len(text) > tail_size else text
    
    # Count conclusion markers
    marker_count = 0
    first_marker_pos = None
    
    for marker in CONCLUSION_MARKERS:
        # Case-insensitive search
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        matches = list(pattern.finditer(tail))
        
        if matches:
            marker_count += len(matches)
            if first_marker_pos is None or matches[0].start() < first_marker_pos:
                # Track position in original text
                first_marker_pos = len(text) - tail_size + matches[0].start()
    
    if marker_count >= threshold:
        logger.warning(f"Detected {marker_count} conclusion markers (threshold={threshold})")
        return LoopDetectionResult(
            has_loop=True,
            loop_type="conclusion",
            truncate_at=first_marker_pos if first_marker_pos else len(text) - tail_size,
            details=f"Found {marker_count} conclusion markers"
        )
    
    return LoopDetectionResult(False, None, None, None)


def detect_code_block_spam(text: str, threshold: int = 3) -> LoopDetectionResult:
    """
    Detect repeated identical code blocks.
    
    A common failure mode where the model outputs the same code block
    multiple times.
    """
    # Find all code blocks
    code_pattern = re.compile(r'```[\w]*\n(.*?)```', re.DOTALL)
    blocks = code_pattern.findall(text)
    
    if len(blocks) < threshold:
        return LoopDetectionResult(False, None, None, None)
    
    # Check for duplicates
    seen = {}
    for i, block in enumerate(blocks):
        # Normalize whitespace for comparison
        normalized = ' '.join(block.split())
        if len(normalized) < 20:  # Skip tiny blocks
            continue
            
        if normalized in seen:
            # Found duplicate
            logger.warning(f"Detected repeated code block (occurrence {seen[normalized]+1} and {i+1})")
            
            # Find position of second occurrence
            first_match = list(code_pattern.finditer(text))[seen[normalized]]
            
            return LoopDetectionResult(
                has_loop=True,
                loop_type="code_block",
                truncate_at=first_match.end(),  # Keep first code block
                details=f"Code block repeated at positions {seen[normalized]} and {i}"
            )
        seen[normalized] = i
    
    return LoopDetectionResult(False, None, None, None)


def detect_and_truncate(text: str) -> tuple[str, LoopDetectionResult | None]:
    """
    Run all loop detection checks and truncate if needed.
    
    Returns:
        Tuple of (cleaned_text, detection_result or None)
    """
    if not text or len(text) < 100:
        return text, None
    
    # Run detection checks in order of severity
    checks = [
        detect_repeated_blocks,
        detect_code_block_spam,
        detect_conclusion_spam,
    ]
    
    for check in checks:
        result = check(text)
        if result.has_loop and result.truncate_at:
            truncated = text[:result.truncate_at].rstrip()
            logger.info(f"Truncated content at position {result.truncate_at} due to {result.loop_type}")
            return truncated, result
    
    return text, None


class ConsecutiveDiscardTracker:
    """
    Track consecutive discards to detect when generation is stuck.
    
    Resets drill state after too many discards.
    """
    
    def __init__(self, max_discards: int = 3):
        self.max_discards = max_discards
        self.consecutive_discards = 0
        self.total_discards = 0
    
    def record_discard(self, reason: str) -> bool:
        """
        Record a discard. Returns True if max reached (should reset state).
        """
        self.consecutive_discards += 1
        self.total_discards += 1
        logger.debug(f"Consecutive discards: {self.consecutive_discards} (reason: {reason})")
        
        if self.consecutive_discards >= self.max_discards:
            logger.warning(f"Max consecutive discards ({self.max_discards}) reached - recommend state reset")
            return True
        return False
    
    def record_success(self) -> None:
        """Record a successful generation, resetting consecutive count."""
        self.consecutive_discards = 0
    
    def should_reset(self) -> bool:
        """Check if we should reset generation state."""
        return self.consecutive_discards >= self.max_discards
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.consecutive_discards = 0


# Global tracker instance
_discard_tracker: ConsecutiveDiscardTracker | None = None


def get_discard_tracker() -> ConsecutiveDiscardTracker:
    """Get the global discard tracker."""
    global _discard_tracker
    if _discard_tracker is None:
        _discard_tracker = ConsecutiveDiscardTracker()
    return _discard_tracker


