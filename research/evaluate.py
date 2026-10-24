"""Pre-registered hypothesis evaluation and the hypothesis ledger.

A hypothesis is a JSON file in research/hypotheses/ written BEFORE it is tested:
    {"id": "H001", "statement": "...", "rationale": "...", "signal": "low_vol",
     "params": {"window": 252}, "top": 0.2, "registered": "2026-10-24"}

evaluate(h) runs it on the development period only and applies the plan's acceptance criteria:
  1. profitability  cost-adjusted excess return vs the equal-weight universe > 0
  2. efficiency     information ratio >= 0.4
  3. stability      positive excess in at least 2 of 3 market regimes (up / flat / down)
  4. robustness     every numeric parameter moved +-20% keeps the excess positive
  5. multiple tests deflated Sharpe probability >= 0.95, counting every trial in the ledger

The holdout period (last two years) is opened only with holdout=True, once per hypothesis, and that
access is written to the ledger.

Usage:  python -m research.evaluate H001 [--holdout]
"""
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent import backtest as bt
from stock_agent.data.prices import load_panel
from stock_agent.signals import REGISTRY

HERE = Path(__file__).parent
HYP = HERE / "hypotheses"
LEDGER = HERE / "ledger.csv"
DEV = ("2016-01-01", "2024-10-31")
HOLDOUT = ("2024-11-01", None)
COST = 0.0025


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var: float | None = None) -> float:
    """Probability that the true Sharpe exceeds the best Sharpe expected from n_trials of pure noise
    (Bailey & Lopez de Prado, 2014). r = per-period excess returns."""
    r = r.dropna()
    T = len(r)
    if T < 24 or r.std() == 0:
        return float("nan")
    sr = r.mean() / r.std()
    skew = ((r - r.mean()) ** 3).mean() / r.std() ** 3
    kurt = ((r - r.mean()) ** 4).mean() / r.std() ** 4
    n = max(1, n_trials)
    v = sr_var if sr_var is not None else 1.0 / T          # variance of SR estimates under the null
    g = 0.5772156649
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    sr0 = math.sqrt(v) * ((1 - g) * z(1 - 1 / n) + g * z(1 - 1 / (n * math.e))) if n > 1 else 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return _norm_cdf((sr - sr0) * math.sqrt(T - 1) / denom)


def regimes(bench: pd.Series, window: int = 12) -> pd.Series:
    """Label each month by the trailing 12-month return of the equal-weight universe."""
    trail = (1 + bench).rolling(window, min_periods=6).apply(np.prod, raw=True) - 1
    trail = trail.shift(1)  # known at the start of the period
    lab = pd.Series("flat", index=bench.index)
    lab[trail > 0.10] = "up"
    lab[trail < -0.05] = "down"
    return lab


def ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()


def _append(row: dict):
    df = pd.concat([ledger(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LEDGER, index=False)


def run_signal(d, h, params, start, end):
    sig = REGISTRY[h["signal"]](**params)
    return bt.run(d, sig, start=start, end=end, top=h.get("top", 0.2), cost=COST)


def evaluate(hid: str, holdout: bool = False, data=None) -> dict:
    h = json.loads((HYP / f"{hid}.json").read_text())
    d = data or bt.Data.from_panel(load_panel())
    params = h.get("params", {})
    start, end = HOLDOUT if holdout else DEV
    if holdout:
        lg = ledger()
        if not lg.empty and ((lg["id"] == hid) & (lg["period"] == "holdout")).any():
            raise RuntimeError(f"{hid} already used its one look at the holdout")
    res = run_signal(d, h, params, start, end)
    ex = res.returns - res.benchmark
    st = bt.stats(res.returns, res.benchmark)
    out = {"id": hid, "period": "holdout" if holdout else "dev", "date": str(date.today()),
           "signal": h["signal"], "params": json.dumps(params),
           "excess_ann": st["excess_ann"], "IR": st["IR"], "CAGR": st["CAGR"], "MDD": st["MDD"],
           "turnover": float(res.turnover.mean()), "months": st["months"]}
    if not holdout:
        reg = regimes(res.benchmark)
        by = ex.groupby(reg).mean() * 12
        out["regime_excess"] = json.dumps({k: round(float(v), 4) for k, v in by.items()})
        out["regimes_positive"] = int((by > 0).sum())
        robust = []
        for k, v in params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                for m in (0.8, 1.2):
                    p2 = dict(params, **{k: type(v)(round(v * m)) if isinstance(v, int) else v * m})
                    r2 = run_signal(d, h, p2, start, end)
                    robust.append(float((r2.returns - r2.benchmark).mean() * 12))
        out["robust_min_excess"] = min(robust) if robust else float("nan")
        n_trials = len(ledger()) + 1
        out["n_trials"] = n_trials
        out["deflated_sharpe_p"] = deflated_sharpe(ex, n_trials)
        checks = {
            "profitability": out["excess_ann"] > 0,
            "efficiency": out["IR"] >= 0.4,
            "stability": out["regimes_positive"] >= 2,
            "robustness": None if not robust else out["robust_min_excess"] > 0,  # None = no parameter to move
            "multiple_tests": out["deflated_sharpe_p"] >= 0.95,
        }
        checks = {k: (None if v is None else bool(v)) for k, v in checks.items()}
        out["checks"] = json.dumps(checks)
        out["verdict"] = "candidate" if all(v for v in checks.values() if v is not None) else "rejected"
    else:
        out["verdict"] = "holdout_pass" if out["excess_ann"] > 0 else "holdout_fail"
    _append(out)
    return out


if __name__ == "__main__":
    r = evaluate(sys.argv[1], holdout="--holdout" in sys.argv)
    for k, v in r.items():
        print(f"{k:20s} {v}")
