"""Signals. Each signal is a function (past data up to s, signal date s, universe) -> score, higher = more
attractive. Only rows dated <= s are visible (see backtest.Data.upto).

REGISTRY maps a name to a factory taking numeric parameters, so a pre-registered hypothesis can name a
signal and its parameters, and the robustness check can move each parameter +-20%.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

MONTH, YEAR = 21, 252


def _close_ago(past, uni, n):
    c = past.close[uni]
    return c.iloc[-1 - n] if len(c) > n else pd.Series(np.nan, uni)


# ---------- price signals ----------
def momentum(lookback: int = YEAR, skip: int = MONTH):
    """Return from `lookback` days ago to `skip` days ago."""
    def f(past, s, uni):
        return _close_ago(past, uni, skip) / _close_ago(past, uni, lookback) - 1
    return f


def reversal(window: int = MONTH):
    """Recent losers score high."""
    def f(past, s, uni):
        return -(_close_ago(past, uni, 0) / _close_ago(past, uni, window) - 1)
    return f


def low_vol(window: int = YEAR):
    """Lower daily volatility scores high."""
    def f(past, s, uni):
        return -past.ret[uni].tail(window).std()
    return f


def small_size():
    """Smaller market cap scores high."""
    def f(past, s, uni):
        return -np.log(past.mcap.loc[s, uni])
    return f


SIGNALS = {"momentum_12_1": momentum(), "reversal_1m": reversal(),
           "low_vol": low_vol(), "small_size": small_size()}


# ---------- fundamental signals ----------
@lru_cache(maxsize=1)
def _fund():
    from stock_agent.data.fundamentals import asof
    from stock_agent.data.prices import load_panel
    from stock_agent.config import CLEAN
    fund = pd.read_parquet(CLEAN / "fundamentals.parquet")
    days = pd.DatetimeIndex(sorted(load_panel()["date"].unique()))
    return asof(fund, days)


def _latest(s, uni, fund=None):
    from stock_agent.data.fundamentals import latest_as_of
    return latest_as_of(_fund() if fund is None else fund, s, uni).reindex(uni)


def earnings_yield(fund=None):
    def f(past, s, uni):
        x = _latest(s, uni, fund)
        return x["net_income_ttm"] / past.mcap.loc[s, uni]
    return f


def book_to_market(fund=None):
    def f(past, s, uni):
        x = _latest(s, uni, fund)
        return (x["equity"] / past.mcap.loc[s, uni]).where(x["equity"] > 0)
    return f


def op_yield(fund=None):
    def f(past, s, uni):
        x = _latest(s, uni, fund)
        return x["op_income_ttm"] / past.mcap.loc[s, uni]
    return f


def roe(fund=None):
    def f(past, s, uni):
        x = _latest(s, uni, fund)
        return (x["net_income_ttm"] / x["equity"]).where(x["equity"] > 0)
    return f


def op_growth_yoy(fund=None):
    """Latest quarter's operating income minus the same quarter a year earlier, scaled by market cap
    (a simple earnings-surprise proxy)."""
    def f(past, s, uni):
        x = _latest(s, uni, fund)
        return (x["op_income_q"] - x["op_income_q_yoy"]) / past.mcap.loc[s, uni]
    return f


def fundamental_signals(fund_avail: pd.DataFrame) -> dict:
    return {"earnings_yield": earnings_yield(fund_avail), "book_to_market": book_to_market(fund_avail),
            "op_yield": op_yield(fund_avail), "roe": roe(fund_avail), "op_growth_yoy": op_growth_yoy(fund_avail)}


def composite(components=("op_growth_yoy", "roe", "low_vol")):
    """Equal-weight average of each component's cross-sectional percentile rank. A name missing a
    component is ranked on the components it has."""
    fs = [REGISTRY[c]() for c in components]

    def f(past, s, uni):
        ranks = pd.concat([fn(past, s, uni).rank(pct=True) for fn in fs], axis=1)
        return ranks.mean(axis=1, skipna=True).where(ranks.notna().sum(axis=1) >= 2)
    return f


REGISTRY = {"momentum": momentum, "reversal": reversal, "low_vol": low_vol, "small_size": small_size,
            "earnings_yield": earnings_yield, "book_to_market": book_to_market, "op_yield": op_yield,
            "roe": roe, "op_growth_yoy": op_growth_yoy, "composite": composite}
