from abc import ABC, abstractmethod

import numpy as np


class BaseRankingModel(ABC):
    """Abstract base class for ranking models.

    Subclasses must implement fit() and log_likelihood() methods.
    After fitting, the model should have estimates_ and covariance_ attributes.
    """

    estimates_: np.ndarray
    covariance_: np.ndarray

    @abstractmethod
    def fit(self, comparisons: list) -> None:
        """Fit the model to comparison data.

        Parameters
        ----------
        comparisons : list
            Comparison data (format depends on the concrete model implementation).

        Returns
        -------
        None
            Updates the model in-place. Sets estimates_ and covariance_ attributes.
        """
        pass

    @abstractmethod
    def log_likelihood(self, comparison: tuple[int, int]) -> float:
        """Compute log likelihood of a single comparison under the fitted model.

        Parameters
        ----------
        comparison : tuple[int, int]
            A tuple (winner_id, loser_id) representing a comparison.

        Returns
        -------
        float
            Log likelihood of the comparison under the fitted model.
        """
        pass
