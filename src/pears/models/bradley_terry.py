import warnings
from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy import stats

from pears.data import PairwiseComparisonData
from pears.encoders import SequentialEncoder
from pears.models.base import require_fit


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


class BradleyTerryModel:
    """Bradley-Terry model for ranking from pairwise comparisons.

    The Bradley-Terry model estimates skill parameters for each item based on
    pairwise comparison outcomes. Parameters are estimated via maximum likelihood
    using iterative scaling.
    """

    def __init__(self) -> None:
        """Initialize the Bradley-Terry model."""
        self.fitted_: bool = False
        self.params_: dict[int, float] | None = None
        self.encoder_: SequentialEncoder | None = None

    def fit(self, data: PairwiseComparisonData) -> None:
        """Fit the Bradley-Terry model to comparison data.

        Parameters
        ----------
        comparisons : PairwiseComparisonData
            Pairwise comparison data containing win/loss observations.

        Returns
        -------
        None
            Updates params_ and encoder_ in place
        """
        self.match_results_ = data.encoded_observations

        # Fit model using iterative scaling
        self.params_ = iterative_scaling_bt(data.encoded_observations)
        self.encoder_ = deepcopy(data.encoder)
        self.fitted_ = True

    @require_fit
    def scores(self) -> dict[str, float]:
        assert self.params_ is not None
        assert self.encoder_ is not None
        return {self.encoder_.decode(item_idx): score for item_idx, score in self.params_.items()}

    @require_fit
    def confidence_intervals(
        self,
        method: str = "sandwich",
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for Bradley-Terry parameters.

        Parameters
        ----------
        method : str, default="sandwich"
            Method for computing confidence intervals:
            - "sandwich": Sandwich robust standard errors (Huber et al. 1967)
            - "bootstrap": Percentile bootstrap
        alpha : float, default=0.05
            Significance level (1-alpha = confidence level). E.g., 0.05 for 95% CIs.
        n_bootstrap : int, default=1000
            Number of bootstrap samples (only used when method="bootstrap")
        seed : int | None, default=None
            Random seed for reproducible bootstrap (only used when method="bootstrap")

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping from item label to (lower_bound, upper_bound) confidence interval.

        References
        ----------
        - Huber et al. (1967): Sandwich robust standard errors
        - DiCiccio & Efron (1996): Pivot bootstrap confidence intervals
        - Newman (2023): Bradley-Terry iterative scaling algorithm (JMLR)
        """
        # Validate parameters
        if method not in ["sandwich", "bootstrap"]:
            raise ValueError(f"method must be 'sandwich' or 'bootstrap', got '{method}'")
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if method == "bootstrap" and n_bootstrap <= 0:
            raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap}")

        # Dispatch to appropriate method
        if method == "sandwich":
            return self._sandwich_confidence_intervals(alpha)
        return self._bootstrap_confidence_intervals(alpha, n_bootstrap, seed)

    def _compute_fisher_information(self) -> np.ndarray:
        """Compute Fisher Information Matrix for first n-1 parameters.

        Returns
        -------
        np.ndarray
            (n-1) x (n-1) Fisher Information Matrix, where n is the number of items.
            The last item is treated as a reference (constrained) parameter.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None

        # Extract params as numpy array
        item_ids = sorted(self.params_.keys())
        n = len(item_ids)
        pi = np.array([self.params_[i] for i in item_ids], dtype=float)

        # Build n_ij matrix (comparison counts)
        N: defaultdict[int, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        for winner, loser in self.match_results_:
            N[winner][loser] += 1
            N[loser][winner] += 1

        # Compute (n-1) x (n-1) Fisher Information
        # FIM[i,j] = sum_k n_ik / (pi_i + pi_k)^2  if i==j
        # FIM[i,j] = n_ij / (pi_i + pi_j)^2        if i!=j
        FIM = np.zeros((n - 1, n - 1), dtype=float)
        for i in range(n - 1):
            for j in range(n - 1):
                if i == j:
                    # Diagonal: sum over all opponents
                    for k in range(n):
                        if k != i:
                            n_ik = N[item_ids[i]][item_ids[k]]
                            if n_ik > 0:
                                FIM[i, i] += n_ik / (pi[i] + pi[k]) ** 2
                else:
                    # Off-diagonal
                    n_ij = N[item_ids[i]][item_ids[j]]
                    if n_ij > 0:
                        FIM[i, j] = n_ij / (pi[i] + pi[j]) ** 2

        return FIM

    def _compute_gradient_outer_product(self) -> np.ndarray:
        """Compute B = sum of outer products of score gradients.

        Returns
        -------
        np.ndarray
            (n-1) x (n-1) matrix B = sum of outer products, where n is the number
            of items. Used in sandwich covariance computation.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None

        item_ids = sorted(self.params_.keys())
        n = len(item_ids)
        pi = np.array([self.params_[i] for i in item_ids], dtype=float)

        B = np.zeros((n - 1, n - 1), dtype=float)

        for winner, loser in self.match_results_:
            # Compute score vector for this observation
            score = np.zeros(n - 1, dtype=float)

            # Find indices (working with n-1 free parameters)
            winner_idx = item_ids.index(winner)
            loser_idx = item_ids.index(loser)

            # Gradient contributions
            if winner_idx < n - 1:
                score[winner_idx] = 1 / pi[winner_idx] - 1 / (pi[winner_idx] + pi[loser_idx])
            if loser_idx < n - 1:
                score[loser_idx] = -1 / (pi[winner_idx] + pi[loser_idx])

            # Outer product
            B += np.outer(score, score)

        return B

    def _sandwich_confidence_intervals(self, alpha: float) -> dict[str, tuple[float, float]]:
        """Compute sandwich robust confidence intervals.

        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05 for 95% CIs)

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping from item label to (lower_bound, upper_bound) confidence interval.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None

        # Get Fisher Information and gradient outer product
        FIM = self._compute_fisher_information()
        B = self._compute_gradient_outer_product()

        # Compute sandwich covariance: V = FIM^-1 * B * FIM^-1
        try:
            FIM_inv = np.linalg.inv(FIM)
        except np.linalg.LinAlgError as e:
            raise ValueError(
                "Fisher Information Matrix is singular. Model may be degenerate."
            ) from e

        V = FIM_inv @ B @ FIM_inv

        # Extract standard errors
        item_ids = sorted(self.params_.keys())
        n = len(item_ids)

        # Standard errors for first n-1 items
        se = np.sqrt(np.diag(V))

        # Compute SE for last item using delta method:
        # Since π_n = 1 - sum(π_i), Var(π_n) = sum_i sum_j V[i,j]
        se_last = np.sqrt(np.sum(V))

        # Get z-score for confidence level
        z = stats.norm.ppf(1 - alpha / 2)

        # Build confidence intervals
        cis: dict[str, tuple[float, float]] = {}
        for idx, item_id in enumerate(item_ids):
            pi_i = self.params_[item_id]
            se_i = se[idx] if idx < n - 1 else se_last

            # Handle edge case: π=0 (never won)
            if pi_i == 0.0:
                warnings.warn(
                    f"Item {item_id} has π=0 (never won). CI is unreliable.",
                    stacklevel=2,
                )
                ci_lower, ci_upper = 0.0, min(0.01, z * se_i)
            else:
                ci_lower = max(0.0, pi_i - z * se_i)
                ci_upper = min(1.0, pi_i + z * se_i)

            # Decode to string label
            label = self.encoder_.decode(item_id)
            cis[label] = (ci_lower, ci_upper)

        # Warn if small sample
        n_obs = len(self.match_results_)
        if n_obs < 30 or n_obs < 10 * n:
            warnings.warn(
                f"Small sample size (n={n_obs}, items={n}). "
                "Asymptotic approximation may be unreliable. Consider using bootstrap.",
                stacklevel=2,
            )

        return cis

    def _bootstrap_confidence_intervals(
        self, alpha: float, n_bootstrap: int, seed: int | None = None
    ) -> dict[str, tuple[float, float]]:
        """Compute percentile bootstrap confidence intervals.

        Parameters
        ----------
        alpha : float
            Significance level (e.g., 0.05 for 95% CIs)
        n_bootstrap : int
            Number of bootstrap samples
        seed : int | None, default=None
            Random seed for reproducibility

        Returns
        -------
        dict[str, tuple[float, float]]
            Mapping from item label to (lower_bound, upper_bound) confidence interval.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None

        # Set random seed for reproducibility if provided
        if seed is not None:
            rng = np.random.Generator(np.random.PCG64(seed))
        else:
            rng = np.random.default_rng()

        item_ids = sorted(self.params_.keys())
        n_obs = len(self.match_results_)

        # Store bootstrap estimates
        bootstrap_estimates: dict[int, list[float]] = {item_id: [] for item_id in item_ids}
        failed_count = 0

        for _b in range(n_bootstrap):
            # Resample observations with replacement
            bootstrap_indices = rng.choice(n_obs, size=n_obs, replace=True)
            bootstrap_matches = [self.match_results_[i] for i in bootstrap_indices]

            # Fit model to bootstrap sample
            try:
                bootstrap_params = iterative_scaling_bt(
                    bootstrap_matches, tolerance=1e-6, iterations=20
                )

                for item_id in item_ids:
                    bootstrap_estimates[item_id].append(bootstrap_params.get(item_id, 0.0))
            except Exception:
                failed_count += 1
                continue

        # Warn if many failures
        if failed_count > 0.05 * n_bootstrap:
            warnings.warn(
                f"{failed_count}/{n_bootstrap} bootstrap samples failed to converge",
                stacklevel=2,
            )

        # Compute percentile confidence intervals
        cis: dict[str, tuple[float, float]] = {}
        for item_id in item_ids:
            estimates = bootstrap_estimates[item_id]

            if len(estimates) < 0.5 * n_bootstrap:
                raise ValueError(
                    f"Too many bootstrap failures ({len(estimates)}/{n_bootstrap} succeeded)"
                )

            ci_lower_val = float(np.percentile(estimates, 100 * alpha / 2))
            ci_upper_val = float(np.percentile(estimates, 100 * (1 - alpha / 2)))

            # Clip to [0, 1]
            ci_lower = max(0.0, ci_lower_val)
            ci_upper = min(1.0, ci_upper_val)

            # Decode to string label
            label = self.encoder_.decode(item_id)
            cis[label] = (ci_lower, ci_upper)

        return cis
