import warnings
from copy import deepcopy
from typing import cast

import numpy as np
from scipy import stats

from pears.data import PairwiseComparisonData
from pears.encoders import SequentialEncoder
from pears.models.base import require_fit


def iterative_scaling_bt(
    win_count_matrix: np.ndarray,
    comparison_count_matrix: np.ndarray,
    initial_theta: np.ndarray | None = None,
    iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Iterative Scaling Algorithm for Bradley-Terry model MLE.

    Based on the update rule in Equation (4) from:
    M. E. J. Newman. Efficient Computation of Rankings from Pairwise Comparisons.
    JMLR, 24(238):1-25, 2023.
    https://jmlr.org/papers/v24/22-1086.html

    Scales parameters so that the weakest item has a score of 1.

    Parameters
    ----------
    win_count_matrix : np.ndarray
        Matrix W where W[i,j] is the number of times i beat j.
    initial_theta : np.ndarray, optional
        Initial skill parameters for each item. If None, initializes all items to 1/n.
    iterations : int, default=20
        Maximum number of iterations.
    tolerance : float, default=1e-6
        Convergence tolerance based on maximum parameter change.

    Returns
    -------
    np.ndarray
        Estimated skill parameters (pi).
    """
    n = win_count_matrix.shape[0]
    W_i = win_count_matrix.sum(axis=1)  # Total wins per item
    N = comparison_count_matrix

    pi = np.ones(n) / n if initial_theta is None else initial_theta.copy()

    for _ in range(iterations):
        pi_prev = pi.copy()
        for i in range(n):
            # Sum_{j != i} n_ij / (pi_i + pi_j)
            # Sufficient statistics (N) allow matrix-row operations for speed
            denom_sum = np.sum(N[i, :] / (pi_prev[i] + pi_prev))

            if W_i[i] > 0 and denom_sum > 0:
                pi[i] = W_i[i] / denom_sum
            elif W_i[i] == 0:
                pi[i] = 0.0

        pi /= np.sum(pi)
        if np.max(np.abs(pi - pi_prev)) < tolerance:
            break
    return cast(np.ndarray, (pi / pi[-1]).astype(np.float64))


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
                # Diagonal A_ii = sum_{k != i} n_ik * [pi_i * pi_k / (pi_i + pi_k)^2]
                terms = N[i, :] * (pi[i] * pi) / (pi[i] + pi) ** 2
                A[i, i] = np.nansum(terms)
            else:
                # Off-diagonal A_ij = -n_ij * [pi_i * pi_j / (pi_i + pi_j)^2]
                n_ij = N[i, j]
                if n_ij > 0:
                    val = n_ij * (pi[i] * pi[j]) / (pi[i] + pi[j]) ** 2
                    A[i, j] = A[j, i] = val
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
    N = win_count_matrix + win_count_matrix.T

    B = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(i, n):
            n_ij_total = N[i, j]
            if n_ij_total == 0:
                continue

            denom_sq = (pi[i] + pi[j]) ** 2

            # Diagonal contribution (variance per item)
            B[i, i] += n_ij_total * (pi[j] ** 2 / denom_sq)

            if j < n - 1:
                # Off-diagonal interaction contribution
                off_diag_val = n_ij_total * (-pi[i] * pi[j] / denom_sq)
                B[i, j] = B[j, i] = off_diag_val
                if i != j:
                    B[j, j] += n_ij_total * (pi[i] ** 2 / denom_sq)

    return B


def sandwich_confidence_intervals(
    pi: np.ndarray,
    win_count_matrix: np.ndarray,
    comparison_count_matrix: np.ndarray,
    n_obs: int,
    alpha: float,
) -> list[tuple[float, float]]:
    n = win_count_matrix.shape[0]

    # Calculate sandwich components
    A = compute_hessian(pi, comparison_count_matrix)
    B = compute_gradient_outer_product(pi, win_count_matrix)

    # Robust Covariance calculation
    A_inv = np.linalg.inv(A)
    V = A_inv @ B @ A_inv

    # Multiplicity correction using chi-squared for uniform validity
    chi2_val = stats.chi2.ppf(1 - alpha, n - 1)
    width_factor = np.sqrt(chi2_val / n_obs)

    cis: list[tuple[float, float]] = []
    for i in range(n):
        # Point estimate transformed to log-space
        xi_i = np.log(pi[i])
        if i < n - 1:
            # Reference model (last item) is fixed at xi=0
            se_i = np.sqrt(V[i, i])

            cis.append((xi_i - width_factor * se_i, xi_i + width_factor * se_i))
        else:
            # Reference model's score is hardcoded, so there's no uncertainty
            cis.append((xi_i, xi_i))

    return cis


class BradleyTerryModel:
    """Bradley-Terry model for ranking from pairwise comparisons.

    The Bradley-Terry model estimates skill parameters for each item based on
    pairwise comparison outcomes. Parameters are estimated via maximum likelihood
    using iterative scaling.
    """

    def __init__(self) -> None:
        """Initialize the Bradley-Terry model."""
        self.fitted_: bool = False
        self.params_: np.ndarray | None = None
        self.encoder_: SequentialEncoder | None = None
        # Sufficient statistics cached for CI efficiency
        self._W: np.ndarray | None = None
        self._N: np.ndarray | None = None
        self.confidence_intervals_: list[tuple[float, float]] | None = None

    def fit(
        self,
        data: PairwiseComparisonData,
        ci_method: str = "sandwich",
        alpha: float = 0.05,
        n_bootstrap: int = 200,
        seed: int | None = None,
    ) -> None:
        """Fit the Bradley-Terry model and cache sufficient statistics.

        Parameters
        ----------
        data : PairwiseComparisonData
            Pairwise comparison data containing win/loss observations.

        Returns
        -------
        None
            Updates params_, encoder_, and cached matrices in place.
        """
        # Cache sufficient statistics during fit for efficient ranking updates
        W = data.encoded_win_count_matrix(padding=0.1)
        self.encoder_ = deepcopy(data.encoder)

        # Estimate parameters (pi) using aggregated win matrix
        self.params_ = iterative_scaling_bt(W, W + W.T)
        self.confidence_intervals_ = self._calculate_confidence_intervals(
            data, method=ci_method, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed
        )
        self.fitted_ = True

    @require_fit
    def scores(self) -> dict[str, float]:
        assert self.params_ is not None
        assert self.encoder_ is not None
        return {
            self.encoder_.decode(item_idx): np.log(score)
            for item_idx, score in enumerate(self.params_)
        }

    @require_fit
    def confidence_intervals(self) -> dict[str, tuple[float, float]]:
        assert self.encoder_ is not None
        assert self.confidence_intervals_ is not None
        return {
            self.encoder_.decode(item_idx): ci
            for item_idx, ci in enumerate(self.confidence_intervals_)
        }

    def _calculate_confidence_intervals(
        self,
        data: PairwiseComparisonData,
        method: str = "sandwich",
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> list[tuple[float, float]]:
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
            W = data.encoded_win_count_matrix(padding=0.1)
            cis = self._sandwich_confidence_intervals(len(data), W, alpha)
        else:
            cis = self._bootstrap_confidence_intervals(data, alpha, n_bootstrap, seed)

        return cis

    def _sandwich_confidence_intervals(
        self, n_obs: int, encoded_win_count_matrix: np.ndarray, alpha: float
    ) -> list[tuple[float, float]]:
        """
        Generates multiplicity-corrected CIs using the sandwich estimator.

        The final interval for model i is derived as: xi_i +/- width_factor * sqrt(V_ii).
        - xi_i: The BT coefficient in log-space (ln(pi_i)).
        - width_factor: sqrt(chi2_critical_value / T), where T is total votes.
        - V_ii: The diagonal element of the sandwich covariance matrix V = A^-1 B A^-1.
        """
        assert self.params_ is not None
        return sandwich_confidence_intervals(
            self.params_,
            encoded_win_count_matrix,
            encoded_win_count_matrix + encoded_win_count_matrix.T,
            n_obs,
            alpha,
        )

    def _bootstrap_confidence_intervals(
        self,
        data: PairwiseComparisonData,
        alpha: float,
        n_bootstrap: int,
        seed: int | None = None,
    ) -> list[tuple[float, float]]:
        """Compute percentile bootstrap confidence intervals.

        Parameters
        ----------
        data : PairwiseComparisonData
            Pairwise comparison data
        alpha : float
            Significance level (e.g., 0.05 for 95% CIs)
        n_bootstrap : int
            Number of bootstrap samples
        seed : int | None, default=None
            Random seed for reproducibility

        Returns
        -------
        list[tuple[float, float]]
            List of (lower_bound, upper_bound) confidence intervals for each item.
        """
        assert self.params_ is not None
        assert self.encoder_ is not None

        n_items = len(self.params_)

        # Create a single RNG for the entire bootstrap process
        if seed is not None:
            rng = np.random.Generator(np.random.PCG64(seed))
        else:
            rng = np.random.default_rng()

        # Store bootstrap estimates
        bootstrap_estimates: list[list[float]] = [[] for _ in range(n_items)]
        failed_count = 0

        for _b in range(n_bootstrap):
            # Resample observations with replacement using the same RNG
            bootstrap_data = data.sample(len(data), with_replacement=True, rng=rng)

            # Fit model to bootstrap sample
            try:
                W = bootstrap_data.encoded_win_count_matrix(padding=0.1)
                bootstrap_params = np.log(iterative_scaling_bt(W, W + W.T))

                for i in range(n_items):
                    bootstrap_estimates[i].append(bootstrap_params[i])
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
        cis: list[tuple[float, float]] = []
        for item_id in range(len(self.params_)):
            estimates = bootstrap_estimates[item_id]

            if len(estimates) < 0.5 * n_bootstrap:
                raise ValueError(
                    f"Too many bootstrap failures ({len(estimates)}/{n_bootstrap} succeeded)"
                )
            ci_lower_val = float(np.percentile(estimates, 100 * alpha / 2))
            ci_upper_val = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
            cis.append((ci_lower_val, ci_upper_val))

        return cis
