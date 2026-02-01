"""Unit tests for PairwiseComparisonData."""

from collections import Counter

import numpy as np
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
        comparison_matrix = data.encoded_comparison_count_matrix()

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

    def test_encoded_win_count_matrix_with_padding(self):
        """Test encoded_win_count_matrix with padding applied to all entries."""
        # Test data: 3 items (A, B, C), 3 observations: A beats B (x2), B beats C (x1)
        items = ["A", "B", "C"]
        observations = [("A", "B"), ("A", "B"), ("B", "C")]

        data = PairwiseComparisonData(observations, items)
        win_matrix = data.encoded_win_count_matrix(padding=1.5)

        # Verify encoding: A=0, B=1, C=2
        # A beat B twice: W[0, 1] = 2 + 1.5 = 3.5
        assert win_matrix[0, 1] == 3.5
        # B beat C once: W[1, 2] = 1 + 1.5 = 2.5
        assert win_matrix[1, 2] == 2.5
        # All other entries should be just padding (1.5)
        assert win_matrix[0, 0] == 1.5
        assert win_matrix[0, 2] == 1.5
        assert win_matrix[1, 0] == 1.5
        assert win_matrix[1, 1] == 1.5
        assert win_matrix[2, 0] == 1.5
        assert win_matrix[2, 1] == 1.5
        assert win_matrix[2, 2] == 1.5

    def test_encoded_comparison_count_matrix_with_padding(self):
        """Test encoded_comparison_count_matrix with padding."""
        # Test data: 3 items (A, B, C), 3 observations: A beats B (x2), B beats C (x1)
        items = ["A", "B", "C"]
        observations = [("A", "B"), ("A", "B"), ("B", "C")]

        data = PairwiseComparisonData(observations, items)
        comparison_matrix = data.encoded_comparison_count_matrix(padding=2.0)

        # Verify encoding: A=0, B=1, C=2
        # A vs B: A beat B twice, B beat A zero times
        # W[0, 1] = 2 + 2.0 = 4.0, W[1, 0] = 0 + 2.0 = 2.0
        # N[0, 1] = W[0, 1] + W[1, 0] = 4.0 + 2.0 = 6.0
        assert comparison_matrix[0, 1] == 6.0
        assert comparison_matrix[1, 0] == 6.0

        # B vs C: B beat C once, C beat B zero times
        # W[1, 2] = 1 + 2.0 = 3.0, W[2, 1] = 0 + 2.0 = 2.0
        # N[1, 2] = W[1, 2] + W[2, 1] = 3.0 + 2.0 = 5.0
        assert comparison_matrix[1, 2] == 5.0
        assert comparison_matrix[2, 1] == 5.0

        # A vs C: A beat C zero times, C beat A zero times
        # W[0, 2] = 0 + 2.0 = 2.0, W[2, 0] = 0 + 2.0 = 2.0
        # N[0, 2] = W[0, 2] + W[2, 0] = 2.0 + 2.0 = 4.0
        assert comparison_matrix[0, 2] == 4.0
        assert comparison_matrix[2, 0] == 4.0

        # Diagonal: W[i, i] = padding, so N[i, i] = padding + padding = 2 * padding
        assert comparison_matrix[0, 0] == 4.0
        assert comparison_matrix[1, 1] == 4.0
        assert comparison_matrix[2, 2] == 4.0

        # Verify symmetry
        assert np.array_equal(comparison_matrix, comparison_matrix.T)

    def test_sample_respects_sample_size(self):
        """Test that sample() returns correct sample size and type."""
        items = ["A", "B", "C", "D"]
        observations = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C"), ("B", "D")]
        data = PairwiseComparisonData(observations, items)

        # Test without replacement, sample_size=3
        sample1 = data.sample(sample_size=3, with_replacement=False, seed=42)
        assert len(sample1) == 3
        assert isinstance(sample1, PairwiseComparisonData)
        assert sample1.num_items == data.num_items

        # Test without replacement, sample_size=5
        sample2 = data.sample(sample_size=5, with_replacement=False, seed=42)
        assert len(sample2) == 5
        assert isinstance(sample2, PairwiseComparisonData)
        assert sample2.num_items == data.num_items

        # Test with replacement, sample_size=10
        sample3 = data.sample(sample_size=10, with_replacement=True, seed=42)
        assert len(sample3) == 10
        assert isinstance(sample3, PairwiseComparisonData)
        assert sample3.num_items == data.num_items

    def test_sample_reproducibility_with_seed(self):
        """Test that sample() is reproducible with seed."""
        items = ["A", "B", "C", "D"]
        observations = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]
        data = PairwiseComparisonData(observations, items)

        # Sample twice with same seed
        sample1 = data.sample(sample_size=3, with_replacement=True, seed=42)
        sample2 = data.sample(sample_size=3, with_replacement=True, seed=42)

        # Verify same results
        assert sample1.observations == sample2.observations

        # Sample with different seed
        sample3 = data.sample(sample_size=3, with_replacement=True, seed=99)
        assert len(sample3) == 3
        # With high probability, different seed produces different results
        # (though not guaranteed, it's virtually certain with different seeds)

    def test_sample_reproducibility_with_rng(self):
        """Test that sample() with rng maintains state correctly."""
        items = ["A", "B", "C", "D"]
        observations = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]
        data = PairwiseComparisonData(observations, items)

        # Create two separate RNGs with same seed
        rng1 = np.random.Generator(np.random.PCG64(42))
        rng2 = np.random.Generator(np.random.PCG64(42))

        # Sample sequentially with each
        sample1a = data.sample(sample_size=3, with_replacement=True, rng=rng1)
        sample1b = data.sample(sample_size=3, with_replacement=True, rng=rng1)

        sample2a = data.sample(sample_size=3, with_replacement=True, rng=rng2)
        sample2b = data.sample(sample_size=3, with_replacement=True, rng=rng2)

        # Verify: same initial state produces same first sample
        assert sample1a.observations == sample2a.observations
        # Verify: same state progression produces same second sample
        assert sample1b.observations == sample2b.observations
        # Make sure a/b samples are actually different
        assert sample1a.observations != sample1b.observations

    def test_sample_with_replacement_allows_duplicates(self):
        """Test that with_replacement=True allows duplicate observations."""
        items = ["A", "B", "C"]
        observations = [("A", "B"), ("B", "C"), ("C", "A")]
        data = PairwiseComparisonData(observations, items)

        # Sample with replacement to likely get duplicates
        sample = data.sample(sample_size=50, with_replacement=True, seed=42)
        assert len(sample) == 50

        # Count occurrences of each observation
        obs_counts = Counter(sample.observations)

        # Verify that at least one observation appears more than once
        assert max(obs_counts.values()) > 1

        # Verify all sampled observations are from original set
        original_obs_set = set(data.observations)
        assert all(obs in original_obs_set for obs in sample.observations)
