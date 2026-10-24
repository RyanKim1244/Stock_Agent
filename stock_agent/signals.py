"""Price-based signals. Each takes (past data up to s, signal date s, universe) and returns a score;
higher score = more attractive. Only rows dated <= s are visible (see backtest.Data.upto)."""
import numpy as np
import pandas as pd

MONTH, YEAR = 21, 252


def _close_ago(past, uni, n):
    c = past.close[uni]
    return c.iloc[-1 - n] if len(c) > n else pd.Series(np.nan, uni)


def momentum_12_1(past, s, uni):
    """Return from 12 months ago to 1 month ago (skips the latest month)."""
    return _close_ago(past, uni, MONTH) / _close_ago(past, uni, YEAR) - 1


def reversal_1m(past, s, uni):
    """Last month's losers score high."""
    return -(_close_ago(past, uni, 0) / _close_ago(past, uni, MONTH) - 1)


def low_vol(past, s, uni):
    """Lower daily volatility over the past year scores high."""
    return -past.ret[uni].tail(YEAR).std()


def small_size(past, s, uni):
    """Smaller market cap scores high."""
    return -np.log(past.mcap.loc[s, uni])


SIGNALS = {"momentum_12_1": momentum_12_1, "reversal_1m": reversal_1m,
           "low_vol": low_vol, "small_size": small_size}


def fundamental_signals(fund_avail: pd.DataFrame) -> dict:
    """Signals from point-in-time fundamentals (output of fundamentals.asof). Each looks up, for every
    stock, the latest report whose `avail` date is on or before the signal date."""
    from stock_agent.data.fundamentals import latest_as_of

    def _f(past, s, uni):
        return latest_as_of(fund_avail, s, uni).reindex(uni)

    def earnings_yield(past, s, uni):
        f = _f(past, s, uni)
        return f["net_income_ttm"] / past.mcap.loc[s, uni]

    def book_to_market(past, s, uni):
        f = _f(past, s, uni)
        bm = f["equity"] / past.mcap.loc[s, uni]
        return bm.where(f["equity"] > 0)

    def op_yield(past, s, uni):
        f = _f(past, s, uni)
        return f["op_income_ttm"] / past.mcap.loc[s, uni]

    def roe(past, s, uni):
        f = _f(past, s, uni)
        return (f["net_income_ttm"] / f["equity"]).where(f["equity"] > 0)

    def op_growth_yoy(past, s, uni):
        """Latest quarter's operating income vs the same quarter a year earlier, scaled by market cap
        (a simple earnings-surprise proxy)."""
        f = _f(past, s, uni)
        return (f["op_income_q"] - f["op_income_q_yoy"]) / past.mcap.loc[s, uni]

    return {"earnings_yield": earnings_yield, "book_to_market": book_to_market, "op_yield": op_yield,
            "roe": roe, "op_growth_yoy": op_growth_yoy}
