"""Surrogate feature-encoding and network tests (no training needed)."""

import pytest
pytest.importorskip("torch", reason="optional ML suite requires torch")

import numpy as np

from pitwall.models import RaceModel
from pitwall.surrogate import SurrogateNet, encode, feature_dim, random_strategy
from pitwall.types import Compound, Stint, Strategy


def test_feature_dim_matches_encoding():
    model = RaceModel.for_circuit("bahrain")
    strat = Strategy([Stint(Compound.MEDIUM, 28), Stint(Compound.HARD, 29)])
    x = encode(model, strat, focal_grid=3, focal_delta=0.2, pace_spread=1.5)
    assert x.shape == (feature_dim(),)


def test_random_strategy_is_legal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = random_strategy(57, rng)
        assert s.total_laps == 57
        assert s.uses_two_compounds()
        assert all(st.length >= 3 for st in s.stints)


def test_surrogate_forward_and_predict_shapes():
    net = SurrogateNet()
    x = np.random.randn(feature_dim()).astype(np.float32)
    out = net.predict(x)
    for k in ("mean_pos", "pos_uncertainty", "p_podium", "p_points"):
        assert k in out
    batch = np.random.randn(8, feature_dim()).astype(np.float32)
    bout = net.predict(batch)
    assert bout["mean_pos"].shape == (8,)
    assert np.all((bout["p_podium"] >= 0) & (bout["p_podium"] <= 1))
