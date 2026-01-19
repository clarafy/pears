"""Unit tests for BradleyTerryModel."""

import numpy as np
import pytest

from pears.data import PairwiseComparisonData
from pears.models.bradley_terry import (
    BradleyTerryModel,
    compute_gradient_outer_product,
    compute_hessian,
)


@pytest.fixture
def fitted_model():
    """Fixture to create a fitted Bradley-Terry model for testing."""
    items = ["A", "B", "C", "D", "E"]
    observations = [
        ["A", "B"],
        ["A", "C"],
        ["A", "D"],
        ["B", "C"],
        ["B", "D"],
        ["C", "D"],
        ["D", "E"],
        ["A", "E"],
        ["B", "E"],
        ["C", "E"],
    ]
    data = PairwiseComparisonData(observations, items)
    model = BradleyTerryModel()
    model.fit(data)
    return model


class TestBradleyTerryModel:
    """Test suite for BradleyTerryModel."""

    def test_require_fit_decorator(self):
        """Test that scores() raises ValueError when called before fit()."""
        model = BradleyTerryModel()

        # Verify the model is not fitted initially
        assert model.fitted_ is False

        # Attempt to call scores() before fitting
        with pytest.raises(ValueError, match="This model instance is not fitted yet"):
            model.scores()

    def test_fit_and_scores_correctness(self, fitted_model):
        """Test fit() and scores() with exact expected values on dummy dataset."""
        # Get scores
        scores = fitted_model.scores()

        # Expected values (generated from iterative_scaling_bt)
        expected = {
            "A": 0.8521485125714814,
            "B": 0.12905368216123939,
            "C": 0.01755555815920866,
            "D": 0.001242247108070438,
            "E": 0.0,
        }

        # Verify all items present in output
        assert set(scores.keys()) == {"A", "B", "C", "D", "E"}

        # Verify scores match expected (using approximate equality for floats)
        for item in expected:
            assert scores[item] == pytest.approx(expected[item], abs=1e-6)

        # Verify ranking order (sanity check)
        assert scores["A"] > scores["B"] > scores["C"] > scores["D"] > scores["E"]


class TestBradleyTerryConfidenceIntervals:
    """Test suite for confidence_intervals() method."""

    def test_require_fit_decorator(self):
        """Test that confidence_intervals() requires fitted model."""
        model = BradleyTerryModel()
        with pytest.raises(ValueError, match="This model instance is not fitted yet"):
            model.confidence_intervals()

    def test_method_validation(self, fitted_model):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            fitted_model.confidence_intervals(method="invalid")

    def test_alpha_validation(self, fitted_model):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError):
            fitted_model.confidence_intervals(alpha=1.5)
        with pytest.raises(ValueError):
            fitted_model.confidence_intervals(alpha=-0.1)

    def test_sandwich_output_format(self, fitted_model):
        """Test sandwich method returns correct format."""
        cis = fitted_model.confidence_intervals(method="sandwich", alpha=0.05)

        assert isinstance(cis, dict)
        assert set(cis.keys()) == {"A", "B", "C", "D", "E"}

        for _label, (lower, upper) in cis.items():
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower <= upper
            assert 0 <= lower <= 1
            assert 0 <= upper <= 1

    def test_bootstrap_output_format(self, fitted_model):
        """Test bootstrap method returns correct format."""
        cis = fitted_model.confidence_intervals(
            method="bootstrap", alpha=0.05, n_bootstrap=100, seed=42
        )

        assert isinstance(cis, dict)
        assert set(cis.keys()) == {"A", "B", "C", "D", "E"}

        for _label, (lower, upper) in cis.items():
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower <= upper

    def test_bootstrap_reproducibility(self, fitted_model):
        """Test that bootstrap with seed produces reproducible results."""
        cis1 = fitted_model.confidence_intervals(
            method="bootstrap", alpha=0.05, n_bootstrap=100, seed=42
        )
        cis2 = fitted_model.confidence_intervals(
            method="bootstrap", alpha=0.05, n_bootstrap=100, seed=42
        )

        # Results should be identical with same seed
        for label in cis1:
            assert cis1[label][0] == pytest.approx(cis2[label][0])
            assert cis1[label][1] == pytest.approx(cis2[label][1])

    def test_contains_point_estimate(self, fitted_model):
        """Test that point estimates fall within CIs for both methods."""
        scores = fitted_model.scores()

        # Test sandwich method
        cis_sandwich = fitted_model.confidence_intervals(method="sandwich", alpha=0.05)
        for label in scores:
            lower, upper = cis_sandwich[label]
            assert lower - 1e-6 <= scores[label] <= upper + 1e-6

        # Test bootstrap method
        cis_bootstrap = fitted_model.confidence_intervals(
            method="bootstrap", alpha=0.05, n_bootstrap=100, seed=42
        )
        for label in scores:
            lower, upper = cis_bootstrap[label]
            assert lower - 1e-6 <= scores[label] <= upper + 1e-6

    def test_different_alpha_levels(self, fitted_model):
        """Test that smaller alpha gives wider intervals."""
        ci_95 = fitted_model.confidence_intervals(method="sandwich", alpha=0.05)
        ci_90 = fitted_model.confidence_intervals(method="sandwich", alpha=0.10)

        for label in ci_95:
            width_95 = ci_95[label][1] - ci_95[label][0]
            width_90 = ci_90[label][1] - ci_90[label][0]
            assert width_95 >= width_90 - 1e-6


class TestHessianAndGradientFunctions:
    """Test suite for standalone Hessian and gradient functions."""

    def test_compute_hessian_small_case(self):
        """Test Hessian computation with manually verified values.

        Test case: 3 items with indices 0, 1, 2 (item 2 is reference).
        pi = [0.5, 0.3, 0.2]
        comparison_matrix[i,j] = total comparisons between i and j
        """
        pi = np.array([0.5, 0.3, 0.2])
        comparison_matrix = np.array(
            [
                [0, 2, 1],  # Item 0: 2 comparisons with item 1, 1 with item 2
                [2, 0, 1],  # Item 1: 2 comparisons with item 0, 1 with item 2
                [1, 1, 0],  # Item 2: 1 comparison each with items 0 and 1
            ]
        )

        result = compute_hessian(pi, comparison_matrix)

        # Expected A (2x2) - computed for items 0 and 1 only
        # A[0,0] = sum_{k≠0} N[0,k] * (pi[0]*pi[k]) / (pi[0]+pi[k])²
        #        = N[0,1] * (0.5*0.3)/(0.8)² + N[0,2] * (0.5*0.2)/(0.7)²
        #        = 2 * 0.15/0.64 + 1 * 0.1/0.49
        #        = 0.46875 + 0.204081632653...
        #        = 0.672831632653...
        # A[1,1] = sum_{k≠1} N[1,k] * (pi[1]*pi[k]) / (pi[1]+pi[k])²
        #        = N[1,0] * (0.3*0.5)/(0.8)² + N[1,2] * (0.3*0.2)/(0.5)²
        #        = 2 * 0.15/0.64 + 1 * 0.06/0.25
        #        = 0.46875 + 0.24
        #        = 0.70875
        # A[0,1] = A[1,0] = -N[0,1] * (pi[0]*pi[1]) / (pi[0]+pi[1])²
        #                  = -2 * 0.15/0.64
        #                  = -0.46875
        expected = np.array(
            [
                [0.672831632653, -0.46875],
                [-0.46875, 0.70875],
            ]
        )

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_compute_gradient_outer_product_small_case(self):
        """Test gradient outer product with manually verified values.

        Test case: 3 items with indices 0, 1, 2 (item 2 is reference).
        pi = [0.5, 0.3, 0.2]
        win_count_matrix[i,j] = number of times i beat j
        """
        pi = np.array([0.5, 0.3, 0.2])
        win_count_matrix = np.array(
            [
                [0, 2, 1],  # Item 0 beat item 1 twice, item 2 once
                [0, 0, 1],  # Item 1 beat item 2 once
                [0, 0, 0],  # Item 2 never won (worst item)
            ]
        )

        result = compute_gradient_outer_product(pi, win_count_matrix)

        # Expected B (2x2) - computed for items 0 and 1 only
        # Process pairs (i,j) where i < n-1:

        # Pair (0,1): n_total = W[0,1] + W[1,0] = 2 + 0 = 2
        #   B[0,0] += 2 * (pi[1]²) / (pi[0]+pi[1])² = 2 * 0.09/0.64 = 0.28125
        #   B[0,1] = B[1,0] = -2 * (pi[0]*pi[1]) / 0.64 = -0.46875
        #   B[1,1] += 2 * (pi[0]²) / 0.64 = 2 * 0.25/0.64 = 0.78125

        # Pair (0,2): n_total = W[0,2] + W[2,0] = 1 + 0 = 1
        #   B[0,0] += 1 * (pi[2]²) / (pi[0]+pi[2])² = 1 * 0.04/0.49 = 0.081632653061...
        #   (No off-diagonal since j=2 is reference)

        # Pair (1,2): n_total = W[1,2] + W[2,1] = 1 + 0 = 1
        #   B[1,1] += 1 * (pi[2]²) / (pi[1]+pi[2])² = 1 * 0.04/0.25 = 0.16
        #   (No off-diagonal since j=2 is reference)

        # Final B matrix:
        # B[0,0] = 0.28125 + 0.081632653061... = 0.362882653061...
        # B[1,1] = 0.78125 + 0.16 = 0.94125
        # B[0,1] = B[1,0] = -0.46875
        expected = np.array(
            [
                [0.362882653061, -0.46875],
                [-0.46875, 0.94125],
            ]
        )

        assert result.shape == (2, 2)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
