"""Type definitions and data structures for pairwise comparisons."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pears.encoders import SequentialEncoder


class PairwiseComparisonData:
    """
    A specialized dataset for binary 'win/loss' observations.

    Enforces the contract that all data points are simple comparisons between exactly two items.

    Parameters
    ----------
    observations : list[list[str]]
        List of pairwise comparisons where each comparison is [winner, loser].
        Each inner list must contain exactly 2 different strings.
    items : Sequence[str]
        Sequence of all possible items that can appear in comparisons.

    Raises
    ------
    TypeError
        If observations is not a list, any observation is not a list, or items
        are not strings.
    ValueError
        If any observation doesn't have exactly 2 items, contains identical items,
        or contains items not in the items list.

    Examples
    --------
    >>> items = ["A", "B", "C"]
    >>> observations = [["A", "B"], ["A", "C"], ["B", "C"]]
    >>> data = PairwiseComparisonData(observations, items)
    >>> data.encoded_observations
    [(0, 1), (0, 2), (1, 2)]
    >>> len(data)
    3
    """

    encoder: SequentialEncoder
    observations: list[tuple[str, str]]

    def __init__(self, observations: list[tuple[str, str]], items: Sequence[str]) -> None:
        """Initialize PairwiseComparisonData with validation.

        Parameters
        ----------
        observations : list[tuple[str, str]]
            List of pairwise comparisons where each comparison is [winner, loser].
        items : Sequence[str]
            Sequence of all possible items that can appear in comparisons.
        """
        self.encoder = SequentialEncoder(items)

        # Validate observations is a list
        if not isinstance(observations, list):
            raise TypeError(f"observations must be a list, got {type(observations).__name__}")

        # Validate each observation
        for i, obs in enumerate(observations):
            # Check it's a list or tuple
            if not isinstance(obs, (list, tuple)):
                raise TypeError(
                    f"observation at index {i} must be a list or tuple, got {type(obs).__name__}"
                )

            # Check it has exactly 2 items
            if len(obs) != 2:
                raise ValueError(
                    f"observation at index {i} must contain exactly 2 items, got {len(obs)}"
                )

            # Check both items are strings
            if not isinstance(obs[0], str) or not isinstance(obs[1], str):
                raise TypeError(
                    f"observation at index {i} must contain strings, "
                    f"got {type(obs[0]).__name__} and {type(obs[1]).__name__}"
                )

            # Check items are different
            if obs[0] == obs[1]:
                raise ValueError(
                    f"observation at index {i} contains identical items: '{obs[0]}' == '{obs[1]}'"
                )

            # Check both items are valid labels
            if not self.encoder.is_valid_label(obs[0]):
                raise ValueError(
                    f"observation at index {i} contains invalid item: '{obs[0]}' "
                    f"not found in items list"
                )
            if not self.encoder.is_valid_label(obs[1]):
                raise ValueError(
                    f"observation at index {i} contains invalid item: '{obs[1]}' "
                    f"not found in items list"
                )

        self.observations = observations

    @property
    def encoded_observations(self) -> list[tuple[int, int]]:
        """Convert observations to integer-encoded (winner_id, loser_id) tuples.

        Returns
        -------
        list[tuple[int, int]]
            List of (winner_id, loser_id) tuples where IDs are integer encodings.
        """
        return [
            (self.encoder.encode(obs[0]), self.encoder.encode(obs[1])) for obs in self.observations
        ]

    def encoded_win_count_matrix(self, padding: float = 0) -> NDArray[Any]:
        """
        Computes the win matrix W where entry [i, j] is the count of times i beat j.

        This matrix represents the empirical win results gathered from live user
        interaction
        """
        # Initialize a dense matrix of zeros
        win_matrix = np.zeros((self.num_items, self.num_items), dtype=int)

        # Single pass O(T) over observations to populate the matrix
        for winner_idx, loser_idx in self.encoded_observations:
            # winner_idx is model i, loser_idx is model j
            win_matrix[winner_idx, loser_idx] += 1

        return win_matrix + padding

    def encoded_comparison_count_matrix(self, padding: float = 0) -> NDArray[Any]:
        """
        Computes the total comparison matrix N where N = W + W.T.

        Entry [i, j] represents the total number of battles (samples) between
        models i and j, regardless of the outcome.
        """
        # Calculate the win matrix first
        W = self.encoded_win_count_matrix(padding=padding)

        # The sum of W and its transpose yields the total n_ij
        return cast(NDArray[Any], W + W.T)

    def __len__(self) -> int:
        """Return the number of pairwise comparisons.

        Returns
        -------
        int
            Number of observations.
        """
        return len(self.observations)

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            Informative string representation.
        """
        return (
            f"PairwiseComparisonData(num_observations={len(self)}, num_items={len(self.encoder)})"
        )

    @property
    def num_items(self) -> int:
        """Return the number of unique items.

        Returns
        -------
        int
            Number of unique items in the encoder.
        """
        return len(self.encoder)

    def sample(
        self,
        sample_size: int,
        with_replacement: bool,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "PairwiseComparisonData":
        """Resample observations with replacement.

        Parameters
        ----------
        sample_size : int
            Number of observations to sample.
        with_replacement : bool
            Whether to sample with replacement.
        seed : int | None, default=None
            Random seed for reproducibility. Used only if rng is not provided.
        rng : np.random.Generator | None, default=None
            Random number generator object. If provided, seed is ignored.

        Returns
        -------
        PairwiseComparisonData
            New dataset with resampled observations.
        """
        # If rng is provided, use it (for cases like bootstrap where state is managed externally)
        # Otherwise create one from seed
        if rng is None:
            if seed is not None:
                rng = np.random.Generator(np.random.PCG64(seed))
            else:
                rng = np.random.default_rng()

        bootstrap_indices = rng.choice(len(self), size=sample_size, replace=with_replacement)
        bootstrap_observations = [self.observations[i] for i in bootstrap_indices]
        return PairwiseComparisonData(bootstrap_observations, self.encoder.items())
