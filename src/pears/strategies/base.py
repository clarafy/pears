from abc import ABC, abstractmethod

import numpy as np

from ..models.base import BaseRankingModel


class RankingStrategy(ABC):
    """Abstract base class for ranking strategies.

    A ranking strategy converts fitted model parameters and covariance estimates
    into a ranking of items.
    """

    @abstractmethod
    def rank(self, model: BaseRankingModel) -> np.ndarray:
        """Produce a ranking from the fitted model.

        Parameters
        ----------
        model : BaseRankingModel
            A fitted ranking model with estimates_ and covariance_ attributes.

        Returns
        -------
        np.ndarray
            Array of item indices sorted by rank (highest ranking first).
        """
        pass
