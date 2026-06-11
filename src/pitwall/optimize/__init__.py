"""Strategy optimisation: deterministic enumeration + robust MC selection."""

from .deterministic import Candidate, StintCostTable, enumerate_candidates
from .robust import (
    OBJECTIVES,
    OptimizeResult,
    ScoredCandidate,
    optimize,
    score_strategies,
)

__all__ = [
    "Candidate",
    "StintCostTable",
    "enumerate_candidates",
    "OBJECTIVES",
    "OptimizeResult",
    "ScoredCandidate",
    "optimize",
    "score_strategies",
]
