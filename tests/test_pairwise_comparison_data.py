"""Unit tests for PairwiseComparisonData."""

import pytest

from pears.data import PairwiseComparisonData


class TestPairwiseComparisonData:
    """Test suite for PairwiseComparisonData."""

    def test_valid_initialization_and_encoding(self):
        """Test happy path: valid initialization and encoding."""
        items = ["A", "B", "C"]
        observations = [("A", "B"), ("A", "C"), ("B", "C")]

        data = PairwiseComparisonData(observations, items)

        # Verify encoded_observations returns list[tuple[int, int]]
        encoded = data.encoded_observations
        assert len(encoded) == 3
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in encoded)
        assert all(isinstance(i, int) and isinstance(j, int) for i, j in encoded)

        # Verify each encoding is consistent (same observation encodes the same way)
        assert data.encoded_observations == [
            (data.encoder.encode(pair[0]), data.encoder.encode(pair[1])) for pair in observations
        ]

        # Verify len() returns correct count
        assert len(data) == 3

        # Test empty observations list works
        empty_data = PairwiseComparisonData([], items)
        assert len(empty_data) == 0
        assert empty_data.encoded_observations == []

    def test_validation_structure(self):
        """Test structural validation of observations."""
        items = ["A", "B", "C"]

        # Observations not a list
        with pytest.raises(TypeError, match="observations must be a list"):
            PairwiseComparisonData("not a list", items)

        # Observation not a list or tuple (string instead)
        with pytest.raises(TypeError, match="observation at index 0 must be a list or tuple"):
            PairwiseComparisonData(["A", "B"], items)

        # Observation with wrong number of items
        with pytest.raises(ValueError, match="exactly 2 items"):
            PairwiseComparisonData([("A", "B", "C")], items)

        with pytest.raises(ValueError, match="exactly 2 items"):
            PairwiseComparisonData([("A",)], items)

    def test_validation_content(self):
        """Test content validation of observations."""
        items = ["A", "B", "C"]

        # Items not strings (integers)
        with pytest.raises(TypeError, match="observation at index 0 must contain strings"):
            PairwiseComparisonData([(1, 2)], items)

        # Identical items (winner == loser)
        with pytest.raises(ValueError, match="identical items"):
            PairwiseComparisonData([("A", "A")], items)

        # Invalid item not in encoder
        with pytest.raises(ValueError, match="not found in items list"):
            PairwiseComparisonData([("A", "D")], items)

        with pytest.raises(ValueError, match="not found in items list"):
            PairwiseComparisonData([("D", "A")], items)

    def test_helper_methods(self):
        """Test __len__, __repr__, and num_items."""
        items = ["A", "B", "C"]
        observations = [("A", "B"), ("C", "A")]
        data = PairwiseComparisonData(observations, items)

        # Test __len__
        assert len(data) == 2

        # Test __repr__
        repr_str = repr(data)
        assert "num_observations=2" in repr_str
        assert "num_items=3" in repr_str

        # Test num_items property
        assert data.num_items == 3

    def test_encoded_win_count_matrix_with_missing_observations(self):
        """Test encoded_win_count_matrix with items that have missing observations."""
        # Items A, B, C, D but D never appears in any observation
        items = ["A", "B", "C", "D"]
        observations = [("A", "B"), ("A", "C"), ("B", "C"), ("A", "B")]

        data = PairwiseComparisonData(observations, items)
        win_matrix = data.encoded_win_count_matrix()

        # Verify shape
        assert win_matrix.shape == (4, 4)

        # Verify encoding: A=0, B=1, C=2, D=3
        # A beat B twice and C once: W[0, 1] = 2, W[0, 2] = 1
        assert win_matrix[0, 1] == 2
        assert win_matrix[0, 2] == 1
        # B beat C once: W[1, 2] = 1
        assert win_matrix[1, 2] == 1

        # Verify all other entries are 0
        assert win_matrix[0, 0] == 0
        assert win_matrix[0, 3] == 0
        assert win_matrix[1, 0] == 0
        assert win_matrix[1, 1] == 0
        assert win_matrix[1, 3] == 0
        assert win_matrix[2, :].sum() == 0  # C never beat anyone
        assert win_matrix[3, :].sum() == 0  # D never appears

    def test_encoded_comparison_matrix_with_missing_observations(self):
        """Test encoded_comparison_matrix with items that have missing observations."""
        # Items A, B, C, D but D never appears in any observation
        items = ["A", "B", "C", "D"]
        observations = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "A")]

        data = PairwiseComparisonData(observations, items)
        comparison_matrix = data.encoded_comparison_matrix()

        # Verify shape
        assert comparison_matrix.shape == (4, 4)

        # Verify encoding: A=0, B=1, C=2, D=3
        # A vs B: A beat B once, B beat A once → N[0, 1] = 2
        assert comparison_matrix[0, 1] == 2
        assert comparison_matrix[1, 0] == 2  # Symmetric
        # A vs C: A beat C once → N[0, 2] = 1
        assert comparison_matrix[0, 2] == 1
        assert comparison_matrix[2, 0] == 1  # Symmetric
        # B vs C: B beat C once → N[1, 2] = 1
        assert comparison_matrix[1, 2] == 1
        assert comparison_matrix[2, 1] == 1  # Symmetric

        # D has no observations, so all D rows/columns are 0
        assert comparison_matrix[3, :].sum() == 0
        assert comparison_matrix[:, 3].sum() == 0
