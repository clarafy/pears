import numpy as np
from scipy.stats import norm

from ..base import RankingStrategy
from ...models.base import BaseRankingModel


class ConfidenceIntervalRankingStrategy(RankingStrategy):
    """Rank items by lower confidence bounds on their estimates.

    Items are ranked in descending order of their lower confidence bounds,
    providing a conservative ranking that accounts for uncertainty.
    """

    def __init__(self, alpha: float = 0.05):
        """Initialize the strategy.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for confidence intervals. Confidence level is 1 - alpha.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha

    def rank(self, model: BaseRankingModel) -> np.ndarray:
        """Produce a ranking from lower confidence bounds.

        Parameters
        ----------
        model : BaseRankingModel
            A fitted ranking model with estimates_ and covariance_ attributes.

        Returns
        -------
        np.ndarray
            Array of item indices sorted by lower confidence bound (highest bound first).
        """
        se = np.sqrt(np.diag(model.covariance_))
        z = norm.ppf(1 - self.alpha / 2)
        lower_bounds = model.estimates_ - z * se
        return np.argsort(-lower_bounds)
