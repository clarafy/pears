from collections import defaultdict

import numpy as np

from ..base import BaseRankingModel


def iterative_scaling_bt(
    match_results: list[tuple[int, int]],
    initial_theta: dict[int, float] | None = None,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> dict[int, float]:
    """Iterative Scaling Algorithm for Bradley-Terry model MLE.

    Based on the update rule in Equation (4) from:
    M. E. J. Newman. Efficient Computation of Rankings from Pairwise Comparisons.
    JMLR, 24(238):1-25, 2023.
    https://jmlr.org/papers/v24/22-1086.html

    Parameters
    ----------
    match_results : list[tuple[int, int]]
        List of tuples (i, j) where i won over j.
    initial_theta : dict[int, float], optional
        Initial skill parameters for each item. If None, initializes all items to 1.0.
    iterations : int, default=20
        Maximum number of iterations.
    tolerance : float, default=1e-6
        Convergence tolerance based on maximum parameter change.

    Returns
    -------
    dict[int, float]
        Mapping of item ID to estimated skill parameter.
    """
    if initial_theta is None:
        unique_items = set()
        for winner, loser in match_results:
            unique_items.add(winner)
            unique_items.add(loser)
        initial_theta = dict.fromkeys(unique_items, 1.0)

    item_ids = sorted(initial_theta.keys())

    # Calculate W_i (total wins for item i)
    W: defaultdict[int, int] = defaultdict(int)
    for winner, _ in match_results:
        W[winner] += 1

    # Calculate n_ij (total comparisons between i and j)
    N: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
    for winner, loser in match_results:
        N[winner][loser] += 1
        N[loser][winner] += 1

    pi = np.array([initial_theta[i] for i in item_ids], dtype=float)

    for _ in range(iterations):
        pi_prev = pi.copy()

        for i_idx, i in enumerate(item_ids):
            W_i = W[i]

            denominator_sum = 0.0
            for j_idx, j in enumerate(item_ids):
                if i == j:
                    continue

                n_ij = N[i][j]
                if n_ij > 0:
                    denominator_sum += n_ij / (pi_prev[i_idx] + pi_prev[j_idx])

            if W_i > 0 and denominator_sum > 0:
                pi[i_idx] = W_i / denominator_sum
            elif W_i == 0:
                pi[i_idx] = 0.0

        sum_pi = np.sum(pi)
        if sum_pi > 0:
            pi /= sum_pi

        max_change = np.max(np.abs(pi - pi_prev))
        if max_change < tolerance:
            break

    final_pi = {item_ids[i]: float(pi[i]) for i in range(len(item_ids))}
    return final_pi


class BradleyTerryModel(BaseRankingModel):
    """Bradley-Terry model for ranking from pairwise comparisons.

    The Bradley-Terry model estimates skill parameters for each item based on
    pairwise comparison outcomes. Parameters are estimated via maximum likelihood
    using iterative scaling.
    """

    def __init__(self) -> None:
        """Initialize the Bradley-Terry model."""
        self.match_results_: list[tuple[int, int]] | None = None
        self.item_ids_: list[int] | None = None

    def fit(self, comparisons: list[tuple[int, int]]) -> None:
        """Fit the Bradley-Terry model to comparison data.

        Parameters
        ----------
        comparisons : list[tuple[int, int]]
            List of tuples (winner_id, loser_id) indicating pairwise comparisons.

        Returns
        -------
        None
            Updates estimates_ and covariance_ attributes in-place.
        """
        self.match_results_ = comparisons

        # Fit model using iterative scaling
        pi_dict = iterative_scaling_bt(comparisons)
        self.item_ids_ = sorted(pi_dict.keys())
        self.estimates_ = np.array([pi_dict[item_id] for item_id in self.item_ids_])

        # TODO: Implement covariance estimation (sandwich estimator or bootstrap)
        self.covariance_ = np.eye(len(self.item_ids_))

    def log_likelihood(self, comparison: tuple[int, int]) -> float:
        """Compute log likelihood of a comparison under the fitted model.

        The likelihood of item i beating item j is proportional to pi_i / (pi_i + pi_j).

        Parameters
        ----------
        comparison : tuple[int, int]
            Tuple (winner_id, loser_id).

        Returns
        -------
        float
            Log likelihood of the comparison.
        """
        if self.estimates_ is None:
            raise ValueError("Model must be fit before computing log likelihood")

        winner_id, loser_id = comparison

        assert self.item_ids_ is not None, "Model must be fit before computing log likelihood"
        try:
            winner_idx = self.item_ids_.index(winner_id)
            loser_idx = self.item_ids_.index(loser_id)
        except ValueError as e:
            raise ValueError(f"Unknown item ID in comparison: {comparison}") from e

        pi_winner = self.estimates_[winner_idx]
        pi_loser = self.estimates_[loser_idx]

        # Log likelihood: log(pi_i / (pi_i + pi_j))
        return float(np.log(pi_winner) - np.log(pi_winner + pi_loser))
