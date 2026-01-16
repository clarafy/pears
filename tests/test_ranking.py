"""Unit tests for ranking strategies."""

import pytest

from pears.ranking import rank_conservative, rank_liberal


class TestRankingInputValidation:
    """Test input validation for ranking functions."""

    def test_input_validation_mismatched_keys(self):
        """Test that mismatched keys raise ValueError."""
        scores = {"A": 0.5, "B": 0.3}
        cis = {"A": (0.4, 0.6), "B": (0.2, 0.4), "C": (0.1, 0.3)}

        with pytest.raises(ValueError, match="must have the same item keys"):
            rank_conservative(scores, cis)

        with pytest.raises(ValueError, match="must have the same item keys"):
            rank_liberal(scores, cis)

    def test_empty_input(self):
        """Test that empty dicts return empty list."""
        assert rank_conservative({}, {}) == []
        assert rank_liberal({}, {}) == []


class TestConservativeRanking:
    """Test conservative ranking strategy."""

    def test_conservative_ranking_simple(self):
        """Test conservative ranking with non-overlapping CIs."""
        scores = {"A": 0.5, "B": 0.3, "C": 0.2}
        cis = {
            "A": (0.4, 0.6),  # No overlap
            "B": (0.2, 0.4),  # No overlap
            "C": (0.1, 0.3),  # No overlap
        }
        result = rank_conservative(scores, cis)
        assert result == ["A", "B", "C"]

    def test_conservative_ranking_overlapping_cis(self):
        """Test conservative ranking with overlapping CIs."""
        scores = {"A": 0.5, "B": 0.3, "C": 0.2}
        cis = {
            "A": (0.3, 0.6),  # A's lower bound not > others' upper bounds
            "B": (0.25, 0.35),  # Overlaps with A
            "C": (0.15, 0.25),  # Overlaps with B
        }
        result = rank_conservative(scores, cis)
        # When CIs overlap, conservative ranking falls back to score ordering
        # via tie-breaking
        assert result[0] == "A"  # A has best score


class TestLiberalRanking:
    """Test liberal ranking strategy."""

    def test_liberal_ranking_simple(self):
        """Test liberal ranking with non-overlapping CIs."""
        scores = {"A": 0.5, "B": 0.3, "C": 0.2}
        cis = {
            "A": (0.4, 0.6),  # No overlap
            "B": (0.2, 0.4),  # No overlap
            "C": (0.1, 0.3),  # No overlap
        }
        result = rank_liberal(scores, cis)
        assert result == ["A", "B", "C"]

    def test_liberal_ranking_overlapping_cis(self):
        """Test liberal ranking with overlapping CIs."""
        scores = {"A": 0.5, "B": 0.3, "C": 0.2}
        cis = {
            "A": (0.3, 0.6),
            "B": (0.25, 0.35),  # Overlaps with A
            "C": (0.15, 0.25),  # Overlaps with B
        }
        result = rank_liberal(scores, cis)
        # All items likely have higher ranks in liberal ranking due to overlaps
        # But A should still be first due to score tie-breaking
        assert result[0] == "A"
