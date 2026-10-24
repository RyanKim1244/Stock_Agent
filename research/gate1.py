"""Gate 1: does the pipeline reproduce well-known facts before we trust it with new hypotheses?

Checks
  1. Data: the equal-weight universe tracks the KOSPI / KOSDAQ indices in direction and rough size.
  2. Adjustment: raw-vs-adjusted jumps look like real share-basis events.
  3. Known factors: quintile spreads for momentum, 1-month reversal, low volatility and size,
     to compare with what published studies report for Korea.

Usage:  python -m research.gate1  (writes research/out/gate1.md)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent import backtest as bt
from stock_agent.data.prices import build_panel, check_adjustment
from stock_agent.signals import SIGNALS

OUT = Path(__file__).parent / "out"


def index_monthly(code: str, exe: pd.DatetimeIndex) -> pd.Series:
    import FinanceDataReader as fdr
    s = fdr.DataReader(code, str(exe.min().date()), str(exe.max().date()))["Open"]
    s = s.reindex(exe, method="ffill")
    return (s.shift(-1) / s - 1).dropna()


def fmt(x):
    return f"{x:.1%}" if isinstance(x, float) and abs(x) < 10 else (f"{x:.2f}" if isinstance(x, float) else str(x))


def main(start="2016-01-01", cost=0.0025):
    OUT.mkdir(exist_ok=True)
    panel = build_panel()
    d = bt.Data.from_panel(panel)
    lines = ["# Gate 1 report", "",
             f"Data: {panel['date'].min().date()} to {panel['date'].max().date()}, "
             f"{panel['ticker'].nunique()} tickers (delisted included).", ""]

    # 2. adjustment events
    ev = check_adjustment(panel)
    lines += ["## Share-basis events picked up", "",
              f"{len(ev)} days where the raw close jumps but KRX's adjusted change does not (splits, mergers, rights).", ""]

    # 3. factors
    rows, qtabs, bench = [], {}, None
    for name, f in SIGNALS.items():
        res = bt.run(d, f, start=start, top=0.2, cost=cost)
        bench = res.benchmark
        st = bt.stats(res.returns, res.benchmark)
        q = res.quantiles
        spread = q["Q5"] - q["Q1"]
        rows.append(dict(signal=name, CAGR=st["CAGR"], excess_ann=st["excess_ann"], IR=st["IR"],
                         MDD=st["MDD"], turnover=res.turnover.mean(),
                         Q5_minus_Q1_ann=spread.mean() * 12,
                         t_stat=spread.mean() / spread.std() * np.sqrt(len(spread))))
        qtabs[name] = (q.mean() * 12)

    # 1. benchmark vs indices
    ix = bench.index
    bl = ["## Universe vs indices (annualised, gross)", "",
          "| series | CAGR | vol | MDD |", "|---|---|---|---|"]
    series = {"EW universe": bench}
    for code, label in [("KS11", "KOSPI"), ("KQ11", "KOSDAQ")]:
        try:
            series[label] = index_monthly(code, ix)
        except Exception as e:  # noqa: BLE001
            bl.append(f"| {label} | unavailable: {type(e).__name__} | | |")
    for k, s in series.items():
        st = bt.stats(s)
        bl.append(f"| {k} | {fmt(st['CAGR'])} | {fmt(st['vol'])} | {fmt(st['MDD'])} |")
    if "KOSPI" in series:
        both = pd.concat([series["EW universe"], series["KOSPI"]], axis=1).dropna()
        bl.append(f"\nMonthly correlation EW universe vs KOSPI: {both.corr().iloc[0, 1]:.2f}")
    lines += bl + [""]

    lines += ["## Factor results (top 20% equal weight, monthly, "
              f"cost {cost:.2%} one-way, from {start})", "",
              "| signal | CAGR | excess vs EW | IR | MDD | turnover/mo | Q5-Q1 ann | t |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['signal']} | {fmt(r['CAGR'])} | {fmt(r['excess_ann'])} | {r['IR']:.2f} | "
                     f"{fmt(r['MDD'])} | {fmt(r['turnover'])} | {fmt(r['Q5_minus_Q1_ann'])} | {r['t_stat']:.2f} |")
    lines += ["", "## Quintile returns (annualised, gross; Q1 = lowest score)", "",
              "| signal | " + " | ".join(f"Q{i}" for i in range(1, 6)) + " |", "|---" * 6 + "|"]
    for name, q in qtabs.items():
        lines.append(f"| {name} | " + " | ".join(fmt(q[f'Q{i}']) for i in range(1, 6)) + " |")
    text = "\n".join(lines) + "\n"
    (OUT / "gate1.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
