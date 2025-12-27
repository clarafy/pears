"""pears: Statistical ranking models with uncertainty quantification."""

from .models.pointwise.bradley_terry import BradleyTerryModel
from .strategies.pointwise.score import ScoreRankingStrategy
from .strategies.pointwise.confidence_interval import ConfidenceIntervalRankingStrategy

__all__ = [
    "BradleyTerryModel",
    "ScoreRankingStrategy",
    "ConfidenceIntervalRankingStrategy",
]

__version__ = "0.1.0"
