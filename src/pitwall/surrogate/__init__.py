"""PyTorch surrogate of the Monte-Carlo simulator for real-time what-ifs."""

from .dataset import LABELS, generate_dataset, random_strategy
from .features import encode, encode_context, encode_strategy, feature_dim
from .model import SurrogateNet
from .train import evaluate_surrogate, load, save, train

__all__ = [
    "LABELS", "generate_dataset", "random_strategy",
    "encode", "encode_context", "encode_strategy", "feature_dim",
    "SurrogateNet", "evaluate_surrogate", "load", "save", "train",
]
