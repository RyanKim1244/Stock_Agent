"""Monthly cross-sectional backtester.

Timeline for each rebalance k:
  signal date s_k   = last trading day of a month; the signal may use data up to and including s_k's close
  execution date e_k = next trading day; trades fill at e_k's open (adjusted)
  holding period    = e_k open -> e_{k+1} open

Leakage guards:
  - the universe and the signal see only rows dated <= s_k (Data.upto)
  - a name that cannot trade at e_k (halted, delisted) is not bought; its weight sits in cash
  - a name that disappears before e_{k+1} is valued at its last close (delisting), never dropped

Costs: one-way cost per unit of turnover (commission + tax + slippage), charged at each rebalance.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Data:
    close: pd.DataFrame      # adjusted close
    open: pd.DataFrame       # adjusted open
    ret: pd.DataFrame        # adjusted close-to-close return
    value: pd.DataFrame      # traded value (KRW)
    mcap: pd.DataFrame       # market cap (KRW)
    tradable: pd.DataFrame   # bool
    common: pd.Series        # bool per ticker

    @classmethod
    def from_panel(cls, panel: pd.DataFrame) -> "Data":
        w = lambda c: panel.pivot(index="date", columns="ticker", values=c).sort_index()
        common = panel.groupby("ticker")["is_common"].first()
        trad = w("tradable")
        return cls(w("adj_close"), w("adj_open"), w("ret"), w("value"), w("mcap"),
                   trad.astype("boolean").fillna(False).astype(bool), common)

    def upto(self, d) -> "Data":
        sl = slice(None, d)
        return Data(self.close.loc[sl], self.open.loc[sl], self.ret.loc[sl], self.value.loc[sl],
                    self.mcap.loc[sl], self.tradable.loc[sl], self.common)


def rebalance_dates(dates: pd.DatetimeIndex, start=None, end=None):
    dates = pd.DatetimeIndex(dates).sort_values()
    s = pd.Series(dates, index=dates)
    month_end = s.groupby([dates.year, dates.month]).max().values
    sig, exe = [], []
    for d in month_end:
        i = dates.searchsorted(d)
        if i + 1 < len(dates):
            sig.append(dates[i]); exe.append(dates[i + 1])
    sig, exe = pd.DatetimeIndex(sig), pd.DatetimeIndex(exe)
    keep = np.ones(len(sig), bool)
    if start is not None: keep &= sig >= pd.Timestamp(start)
    if end is not None: keep &= sig <= pd.Timestamp(end)
    return sig[keep], exe[keep]


def universe(d: Data, s, min_value: float = 1e9, min_days: int = 260) -> pd.Index:
    """Eligible names at signal date s, using only data up to s."""
    v = d.value.loc[:s].tail(20)
    trad = d.tradable.loc[s]
    hist = d.close.loc[:s].notna().sum()
    ok = (v.mean() >= min_value) & trad & d.common.reindex(v.columns).fillna(False) & (hist >= min_days)
    return ok[ok].index


def period_returns(d: Data, names, e0, e1) -> pd.Series:
    """Open(e0) -> open(e1) adjusted return; a name with no open at e1 is valued at its last close before e1."""
    o0 = d.open.loc[e0, names]
    o1 = d.open.loc[e1, names]
    last_close = d.close.loc[:e1, names].iloc[:-1].ffill().iloc[-1]
    end_val = o1.where(o1.notna() & d.tradable.loc[e1, names], last_close)
    return end_val / o0 - 1.0


@dataclass
class Result:
    returns: pd.Series                 # net portfolio return per holding period, indexed by e_k
    gross: pd.Series
    benchmark: pd.Series               # equal-weight universe, gross
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)
    quantiles: pd.DataFrame | None = None  # gross EW return per signal quintile (Q1 low ... Q5 high)


def run(d: Data, signal: Callable[[Data, pd.Timestamp, pd.Index], pd.Series],
        start=None, end=None, top: float = 0.2, cost: float = 0.0025,
        min_value: float = 1e9, n_quantiles: int = 5) -> Result:
    sig_dates, exe_dates = rebalance_dates(d.close.index, start, end)
    out, gross, bench, turn, qrows, idx = [], [], [], [], [], []
    holdings = {}
    w_prev = pd.Series(dtype=float)
    for k in range(len(sig_dates) - 1):
        s, e0, e1 = sig_dates[k], exe_dates[k], exe_dates[k + 1]
        past = d.upto(s)
        uni = universe(past, s, min_value)
        if len(uni) < 50:
            continue
        score = signal(past, s, uni).reindex(uni).dropna()
        if len(score) < 50:
            continue
        r_all = period_returns(d, uni, e0, e1)
        can_buy = d.tradable.loc[e0, uni] & d.open.loc[e0, uni].notna()
        r_all = r_all.where(can_buy)
        n = max(1, int(len(score) * top))
        picks = score.sort_values(ascending=False).index[:n]
        w = pd.Series(1.0 / n, index=picks)
        w = w[can_buy.reindex(picks).fillna(False)]          # unbuyable -> cash
        r_p = float((w * r_all.reindex(w.index).fillna(0.0)).sum())
        # turnover against drifted previous weights
        tw = w_prev.reindex(w.index.union(w_prev.index)).fillna(0.0)
        nw = w.reindex(tw.index).fillna(0.0)
        to = float((nw - tw).abs().sum())
        net = r_p - to * cost
        # drift for next period
        r_held = r_all.reindex(w.index).fillna(0.0)
        grown = w * (1 + r_held)
        w_prev = grown / (1 + r_p) if (1 + r_p) > 0 else grown * 0
        b = float(r_all.dropna().mean())
        if n_quantiles:
            sc = score[can_buy.reindex(score.index).fillna(False)]
            q = pd.qcut(sc.rank(method="first"), n_quantiles, labels=False)
            qrows.append(r_all.reindex(sc.index).groupby(q).mean().rename(lambda i: f"Q{i + 1}"))
        out.append(net); gross.append(r_p); bench.append(b); turn.append(to); idx.append(e0)
        holdings[e0] = list(w.index)
    ix = pd.DatetimeIndex(idx)
    qdf = pd.DataFrame(qrows, index=ix) if qrows else None
    return Result(pd.Series(out, ix), pd.Series(gross, ix), pd.Series(bench, ix), pd.Series(turn, ix), holdings, qdf)


def stats(r: pd.Series, bench: pd.Series | None = None, periods: int = 12) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / periods
    out = {
        "CAGR": eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "vol": r.std() * np.sqrt(periods),
        "sharpe": r.mean() / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan,
        "MDD": (eq / eq.cummax() - 1).min(),
        "months": len(r),
    }
    if bench is not None:
        ex = (r - bench.reindex(r.index)).dropna()
        out["excess_ann"] = ex.mean() * periods
        out["IR"] = ex.mean() / ex.std() * np.sqrt(periods) if ex.std() > 0 else np.nan
        out["hit_rate"] = (ex > 0).mean()
    return out
