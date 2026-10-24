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
