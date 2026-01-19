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


def compute_hessian(pi: np.ndarray, comparison_matrix: np.ndarray) -> np.ndarray:
    """Compute Hessian matrix for Bradley-Terry model.

    Parameters
    ----------
    pi : np.ndarray
        Array of skill parameters (must sum to 1), shape (n,)
    comparison_matrix : np.ndarray
        Symmetric matrix where N[i,j] = total comparisons between i and j, shape (n, n)

    Returns
    -------
    np.ndarray
        Hessian matrix A of shape (n-1, n-1)
    """
    n = len(pi)
    N = comparison_matrix

    A = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(i, n - 1):
            if i == j:
                # Diagonal: sum variance contributions over all possible opponents k
                for k in range(n):
                    if k != i:
                        n_ik = N[i, k]
                        if n_ik > 0:
                            A[i, i] += n_ik * (pi[i] * pi[k]) / (pi[i] + pi[k]) ** 2
            else:
                # Off-diagonal: interaction term between i and j
                n_ij = N[i, j]
                if n_ij > 0:
                    val = -n_ij * (pi[i] * pi[j]) / (pi[i] + pi[j]) ** 2
                    A[i, j] = val
                    A[j, i] = val
    return A


def compute_gradient_outer_product(pi: np.ndarray, win_count_matrix: np.ndarray) -> np.ndarray:
    """Compute gradient outer product matrix for Bradley-Terry model.

    Parameters
    ----------
    pi : np.ndarray
        Array of skill parameters (must sum to 1), shape (n,)
    win_count_matrix : np.ndarray
        Matrix where W[i,j] = number of times i beat j, shape (n, n)

    Returns
    -------
    np.ndarray
        Gradient outer product matrix B of shape (n-1, n-1)
    """
    n = len(pi)
    n_win = win_count_matrix

    B = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(i, n):
            # Total battles between this specific pair
            n_ij_total = n_win[i, j] + n_win[j, i]
            if n_ij_total == 0:
                continue

            denom_sq = (pi[i] + pi[j]) ** 2

            # Diagonal contribution for model i
            B[i, i] += n_ij_total * (pi[j] ** 2 / denom_sq)

            # Off-diagonal and diagonal contribution for model j
            if j < n - 1:
                off_diag_val = n_ij_total * (-pi[i] * pi[j] / denom_sq)
                B[i, j] += off_diag_val
                B[j, i] += off_diag_val
                B[j, j] += n_ij_total * (pi[i] ** 2 / denom_sq)

    return B


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
        self.confidence_intervals_: dict[int, tuple[float, float]] | None = None

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
        data: PairwiseComparisonData,
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
            return self._sandwich_confidence_intervals(data, alpha)
        return self._bootstrap_confidence_intervals(data, alpha, n_bootstrap, seed)

    def _compute_hessian_log_space(self, data: PairwiseComparisonData) -> np.ndarray:
        """Computes the Hessian matrix (Bread) for M-1 parameters in log-space."""
        assert self.params_ is not None
        item_ids = sorted(self.params_.keys())
        pi = np.array([self.params_[i] for i in item_ids], dtype=float)
        N = data.encoded_comparison_count_matrix()
        return compute_hessian(pi, N)

    def _compute_gradient_outer_product_log_space(self, data: PairwiseComparisonData) -> np.ndarray:
        """Computes the sum of gradient outer products (Meat) for M-1 parameters."""
        assert self.params_ is not None
        item_ids = sorted(self.params_.keys())
        pi = np.array([self.params_[i] for i in item_ids], dtype=float)
        W = data.encoded_win_count_matrix()
        return compute_gradient_outer_product(pi, W)

    def _sandwich_confidence_intervals(
        self, data: PairwiseComparisonData, alpha: float
    ) -> dict[str, tuple[float, float]]:
        """
        Generates multiplicity-corrected CIs using the sandwich estimator.

        The final interval for model i is derived as: xi_i +/- width_factor * sqrt(V_ii).
        - xi_i: The BT coefficient in log-space (ln(pi_i)).
        - width_factor: sqrt(chi2_critical_value / T), where T is total votes.
        - V_ii: The diagonal element of the sandwich covariance matrix V = A^-1 B A^-1.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None
        item_ids = sorted(self.params_.keys())
        n = len(item_ids)
        n_obs = len(self.match_results_)

        # Calculate sandwich components
        A = self._compute_hessian_log_space(data)
        B = self._compute_gradient_outer_product_log_space(data)

        # Robust Covariance calculation
        A_inv = np.linalg.inv(A)
        V = A_inv @ B @ A_inv

        # Multiplicity correction using chi-squared for uniform validity
        chi2_val = stats.chi2.ppf(1 - alpha, n - 1)
        width_factor = np.sqrt(chi2_val / n_obs)

        cis: dict[str, tuple[float, float]] = {}
        for i in range(n):
            label = self.encoder_.decode(item_ids[i])
            # Point estimate transformed to log-space
            xi_i = np.log(self.params_[item_ids[i]])

            # Reference model (last item) is fixed at xi=0
            se_i = np.sqrt(V[i, i]) if i < n - 1 else 0.0

            cis[label] = (xi_i - width_factor * se_i, xi_i + width_factor * se_i)

        return cis

    def _bootstrap_confidence_intervals(
        self,
        data: PairwiseComparisonData,
        alpha: float,
        n_bootstrap: int,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute percentile bootstrap confidence intervals.

        Parameters
        ----------
        data : PairwiseComparisonData
            Pairwise comparison data (for compatibility, not used in bootstrap)
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
