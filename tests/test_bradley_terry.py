"""Unit tests for BradleyTerryModel."""

import pytest

from pears.data import PairwiseComparisonData
from pears.models.bradley_terry import BradleyTerryModel


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

    def test_fit_and_scores_correctness(self):
        """Test fit() and scores() with exact expected values on dummy dataset."""
        # Define test dataset
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

        # Create data and model
        data = PairwiseComparisonData(observations, items)
        model = BradleyTerryModel()

        # Fit the model
        model.fit(data)

        # Verify model is now fitted
        assert model.fitted_ is True

        # Get scores
        scores = model.scores()

        # Expected values (generated from iterative_scaling_bt)
        expected = {
            "A": 0.8521485125714814,
            "B": 0.12905368216123939,
            "C": 0.01755555815920866,
            "D": 0.001242247108070438,
            "E": 0.0,
        }

        # Verify all items present in output
        assert set(scores.keys()) == set(items)

        # Verify scores match expected (using approximate equality for floats)
        for item in items:
            assert scores[item] == pytest.approx(expected[item], abs=1e-6)

        # Verify ranking order (sanity check)
        assert scores["A"] > scores["B"] > scores["C"] > scores["D"] > scores["E"]
