"""pitwall — an F1 race-strategy engine.

Physics-calibrated tyre/pace models, a Monte-Carlo race simulator, strategy
optimisation under uncertainty, a learned PyTorch surrogate for real-time
what-ifs, and a live Bayesian in-race tyre updater.
"""

from .models import RaceModel
from .types import Compound, RaceConfig, Stint, Strategy

__version__ = "0.1.0"
__all__ = ["RaceModel", "Compound", "RaceConfig", "Stint", "Strategy", "__version__"]
