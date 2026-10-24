import numpy as np
import pandas as pd

from research.evaluate import deflated_sharpe, regimes


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.03, 120))   # decent monthly excess
    p1 = deflated_sharpe(r, 1)
    p100 = deflated_sharpe(r, 100)
    assert p1 > 0.95
    assert p100 < p1


def test_deflated_sharpe_noise_is_not_significant():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.03, 120))
    assert deflated_sharpe(r, 20) < 0.95


def test_regimes_use_only_past():
    idx = pd.date_range("2020-01-31", periods=30, freq="ME")
    b = pd.Series(0.0, idx)
    b.iloc[20] = 0.5   # a huge month
    lab = regimes(b)
    assert lab.iloc[20] == "flat"   # the month itself does not see its own return
    assert lab.iloc[21] == "up"
