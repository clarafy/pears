"""pears: Statistical ranking models with uncertainty quantification."""

from .models.pointwise.bradley_terry import BradleyTerryModel
from .strategies.pointwise.confidence_interval import ConfidenceIntervalRankingStrategy
from .strategies.pointwise.score import ScoreRankingStrategy

__all__ = [
    "BradleyTerryModel",
    "ConfidenceIntervalRankingStrategy",
    "ScoreRankingStrategy",
]

__version__ = "0.1.0"
