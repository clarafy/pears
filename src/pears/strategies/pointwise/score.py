import numpy as np

from ...models.base import BaseRankingModel
from ..base import RankingStrategy


class ScoreRankingStrategy(RankingStrategy):
    """Rank items by point estimates only.

    Items are ranked in descending order of their estimated scores.
    """

    def rank(self, model: BaseRankingModel) -> np.ndarray:
        """Produce a ranking from point estimates.

        Parameters
        ----------
        model : BaseRankingModel
            A fitted ranking model with estimates_ attribute.

        Returns
        -------
        np.ndarray
            Array of item indices sorted by score (highest score first).
        """
        return np.argsort(-model.estimates_)
