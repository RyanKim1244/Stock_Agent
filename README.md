# Stock Agent

A personal research system for Korean equities: collect point-in-time data, let Claude propose and test
hypotheses, and keep only signals that survive honest backtests. Medium-term (weeks to months), monthly rebalancing.
Not investment advice.

## Layout

| Path | What it is |
|---|---|
| `stock_agent/data/krx.py` | Daily whole-market snapshots from KRX (delisted names included, so no survivorship bias). |
| `stock_agent/data/dart.py` | OpenDART company codes and headline financials, with filing dates for point-in-time use. |
| `stock_agent/data/prices.py` | Split/rights-adjusted price panel built from KRX's own daily change. |
| `stock_agent/backtest.py` | Monthly backtester: signal at month-end close, trade at next open, costs, delisting handling. |
| `stock_agent/signals.py` | Price-based signals (momentum, reversal, low volatility, size). |
| `research/gate1.py` | Gate 1: reproduce known factor behaviour before testing new ideas. |
| `tests/` | Synthetic tests for adjustment, timing (no look-ahead), costs and delisting. |

## Setup

```sh
pip install -r requirements.txt
cp .env.example .env    # then fill in your keys; .env is git-ignored
python -m stock_agent.data.krx 2015-01-01 2026-10-23   # ~1 hour, resumable
python -m stock_agent.data.dart codes
python -m stock_agent.data.dart accounts 2015 2026
python -m stock_agent.data.prices
python -m research.gate1
pytest -q
```

Data lands in `data/` (git-ignored).

## Rules the code enforces

- A signal sees only rows dated on or before its signal date; trades fill at the next trading day's open.
- A financial figure is usable from the trading day after its filing date (`rcept_no` prefix).
- Names that stop trading are valued at their last close, never silently dropped.
- Every backtest charges costs on turnover (default 0.25% one-way).
