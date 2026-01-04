"""Tests for dream validation."""

from codedreamer.validator import DreamValidator, ValidationResult


def test_validator_rejects_short_content() -> None:
    """Short content should be rejected."""
    validator = DreamValidator()

    result = validator.validate("Too short.")

    assert not result.is_valid
    assert result.rejection_reason == "Too short"


def test_validator_rejects_duplicates() -> None:
    """Exact duplicates should be rejected."""
    validator = DreamValidator()

    content = """
    This is a detailed code improvement suggestion.
    Consider refactoring the authentication module to use
    dependency injection for better testability. The current
    implementation tightly couples the session store.
    """

    # First time should pass
    result1 = validator.validate(content)
    assert result1.is_valid

    # Second time should fail as duplicate
    result2 = validator.validate(content)
    assert not result2.is_valid
    assert result2.rejection_reason == "Exact duplicate"


def test_validator_categorizes_content() -> None:
    """Content should be categorized by keywords."""
    validator = DreamValidator()

    bug_content = "Found a bug in the error handling. The fix should check null values."
    result = validator.validate(bug_content * 3)  # Make it long enough
    assert result.category == "code_fix"

    idea_content = "Consider implementing a new caching feature for better performance."
    result = validator.validate(idea_content * 3)
    assert result.category == "code_idea"


def test_novelty_decreases_with_repetition() -> None:
    """Novelty score should decrease when themes repeat."""
    validator = DreamValidator()

    # First dream about refactoring
    content1 = """
    The refactor of this module would improve code clarity.
    Consider extracting the validation logic into a separate function.
    This would make testing easier and reduce complexity.
    """
    result1 = validator.validate(content1)
    assert result1.is_valid
    score1 = result1.novelty_score

    # Second dream also about refactoring
    content2 = """
    Another refactor opportunity: the config parser could be simplified.
    Extract the validation into its own module for better organization.
    This change would improve maintainability significantly.
    """
    result2 = validator.validate(content2)
    score2 = result2.novelty_score

    # Novelty should be lower on repeated themes
    assert score2 <= score1



