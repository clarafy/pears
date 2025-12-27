"""Type definitions and data structures for pairwise comparisons."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairedComparison:
    """Immutable representation of a pairwise comparison.

    Attributes
    ----------
    winner_id : Any
        ID of the item that won the comparison.
    loser_id : Any
        ID of the item that lost the comparison.
    """

    winner_id: Any
    loser_id: Any
