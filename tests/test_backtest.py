"""Synthetic checks for the price adjustment and the backtester's timing, costs and delisting rules."""
import numpy as np
import pandas as pd
import pytest

from stock_agent import backtest as bt
from stock_agent.data.prices import build_panel


def make_raw(n_days=400, n_stocks=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for j in range(n_stocks):
        t = f"{j:05d}0"
        price = 10000.0
        for d in dates:
            chg = rng.normal(0.0005, 0.02)
            prev = price
            price = prev * (1 + chg)
            o = prev * (1 + rng.normal(0, 0.005))
            rows.append(dict(ticker=t, date=d, open=o, high=max(o, price), low=min(o, price), close=price,
                             volume=1000, value=5e9, chg_pct=chg * 100, mcap=price * 1e6, market="KOSPI"))
    return pd.DataFrame(rows)


def test_split_is_adjusted():
    raw = make_raw(30, 1)
    # 2-for-1 split on day 10: raw prices halve, KRX change measured from the adjusted base stays small
    m = raw.index >= 10
    raw.loc[m, ["open", "high", "low", "close"]] /= 2
    p = build_panel(raw, save=False)
    raw_jump = raw["close"].iloc[10] / raw["close"].iloc[9] - 1
    adj_jump = p["adj_close"].iloc[10] / p["adj_close"].iloc[9] - 1
    assert raw_jump < -0.4
    assert abs(adj_jump - raw["chg_pct"].iloc[10] / 100) < 1e-12
    # adjusted open is on the same basis as adjusted close
    assert abs(p["adj_open"].iloc[10] / p["adj_close"].iloc[9] - 1) < 0.05


def test_no_lookahead_signal_sees_only_past():
    p = build_panel(make_raw(), save=False)
    d = bt.Data.from_panel(p)
    seen = []

    def sig(past, s, uni):
        seen.append((s, past.close.index.max()))
        return past.ret.loc[s, uni]

    bt.run(d, sig, top=0.2)
    assert seen and all(mx == s for s, mx in seen)


def test_future_peeking_signal_would_be_huge():
    """Sanity: a signal that cheats (uses the holding period's return) must look spectacular,
    and the honest pipeline must not."""
    p = build_panel(make_raw(seed=1), save=False)
    d = bt.Data.from_panel(p)
    sig_dates, exe = bt.rebalance_dates(d.close.index)
    nxt = dict(zip(exe[:-1], exe[1:]))
    s2e = dict(zip(sig_dates, exe))

    def cheat(past, s, uni):
        e0 = s2e[s]
        return bt.period_returns(d, uni, e0, nxt[e0]) if e0 in nxt else pd.Series(0.0, uni)

    honest = bt.run(d, lambda past, s, uni: past.ret.loc[s, uni], top=0.2, cost=0)
    cheated = bt.run(d, cheat, top=0.2, cost=0)
    assert (cheated.gross - cheated.benchmark).mean() > 0.03
    assert abs((honest.gross - honest.benchmark).mean()) < 0.01


def test_costs_and_turnover():
    p = build_panel(make_raw(seed=2), save=False)
    d = bt.Data.from_panel(p)
    r = bt.run(d, lambda past, s, uni: past.ret.loc[s, uni], top=0.2, cost=0.01)
    assert np.allclose(r.gross - r.returns, r.turnover * 0.01)
    assert r.turnover.iloc[0] == pytest.approx(1.0)  # first period buys from cash


def test_delisted_name_valued_at_last_close():
    raw = make_raw(300, 60, seed=3)
    dates = sorted(raw["date"].unique())
    victim = "000000"
    cut = dates[200]
    raw = raw[~((raw["ticker"] == victim) & (raw["date"] > cut))]
    p = build_panel(raw, save=False)
    d = bt.Data.from_panel(p)
    sig, exe = bt.rebalance_dates(d.close.index)
    k = next(i for i in range(len(exe) - 1) if exe[i] < cut < exe[i + 1])
    r = bt.period_returns(d, pd.Index([victim]), exe[k], exe[k + 1])
    expect = d.close.loc[cut, victim] / d.open.loc[exe[k], victim] - 1
    assert r[victim] == pytest.approx(expect)
