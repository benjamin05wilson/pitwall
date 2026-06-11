"""Network-dependent data tests — OFF by default for a deterministic suite.

Enable with:  PITWALL_NETWORK_TESTS=1 pytest tests/test_data_network.py
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PITWALL_NETWORK_TESTS") != "1",
    reason="set PITWALL_NETWORK_TESTS=1 to run network/data tests",
)


def test_fastf1_loader_schema():
    from pitwall.data.fastf1_loader import fastf1_available, load_clean_laps
    if not fastf1_available():
        pytest.skip("fastf1 not installed")
    df = load_clean_laps(2023, "Bahrain", "R")
    assert not df.empty
    assert set(["driver", "lap", "compound", "tyre_age", "lap_time"]).issubset(df.columns)
    assert df["lap_time"].between(60, 130).mean() > 0.9  # plausible F1 lap times


def test_calibration_recovers_realistic_bahrain_deg():
    from pitwall.data.calibrate import calibrate_races
    res = calibrate_races([(2023, "Bahrain"), (2024, "Bahrain")], "bahrain")
    assert res is not None
    # Bahrain degradation should land in a realistic 0.05-0.25 s/lap band.
    for cp in res.compounds.values():
        assert 0.03 <= cp.k1 <= 0.30
    assert res.rmse < 1.0


def test_backtest_reproduces_real_order():
    from pitwall.data.backtest import reconstruct_race
    res = reconstruct_race(2023, "Bahrain", "bahrain", n_scenarios=150,
                           pool_years=[2023, 2024, 2025])
    assert res is not None
    assert res.spearman > 0.5  # sim order correlates with reality
