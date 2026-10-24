"""Turn raw daily KRX snapshots into an adjusted price panel.

KRX's daily change `chg_pct` is measured from an adjusted base price, so chaining it gives a
split- and rights-adjusted close index without needing a separate adjustment factor table:

    base_t      = close_t / (1 + chg_t)          (yesterday's close, adjusted to today's share basis)
    adj_close_t = adj_close_{t-1} * (1 + chg_t)
    adj_open_t  = adj_close_{t-1} * open_t / base_t

Days on which a stock did not trade (volume 0, halts) carry chg_t = 0 and are flagged untradable.
"""
import glob

import pandas as pd

from stock_agent.config import CLEAN, RAW


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(str(RAW / "krx_daily" / "*" / "*.parquet")))
    if not files:
        raise FileNotFoundError("no KRX snapshots; run python -m stock_agent.data.krx first")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_panel(df: pd.DataFrame | None = None, save: bool = True) -> pd.DataFrame:
    df = load_raw() if df is None else df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    chg = (df["chg_pct"].fillna(0.0) / 100.0).clip(lower=-0.99)
    # A stock's first row has no previous close: start its index at its own close.
    first = df["ticker"].ne(df["ticker"].shift())
    chg = chg.where(~first, 0.0)
    growth = (1.0 + chg).groupby(df["ticker"]).cumprod()
    first_close = df["close"].where(first).groupby(df["ticker"]).transform("first")
    df["adj_close"] = first_close * growth
    prev_adj = df.groupby("ticker")["adj_close"].shift()
    base = df["close"] / (1.0 + chg)
    ratio = (df["open"] / base).where((df["open"] > 0) & (base > 0))
    df["adj_open"] = (prev_adj * ratio).where(~first, df["open"].astype(float))
    df["ret"] = chg  # adjusted close-to-close return
    df["tradable"] = (df["volume"] > 0) & (df["open"] > 0)
    df["is_common"] = df["ticker"].str[-1] == "0"  # preferred shares end in 5, 7, 9, K, L ...
    if save:
        CLEAN.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CLEAN / "panel.parquet", index=False)
    return df


def load_panel() -> pd.DataFrame:
    return pd.read_parquet(CLEAN / "panel.parquet")


def wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="ticker", values=col).sort_index()


def check_adjustment(panel: pd.DataFrame, tol: float = 0.35) -> pd.DataFrame:
    """Rows where the raw close jumps a lot but KRX's adjusted change does not: share-basis events.
    Useful to eyeball that adjustments are being picked up (should be splits, mergers, rights issues)."""
    p = panel.sort_values(["ticker", "date"])
    raw_ret = p.groupby("ticker")["close"].pct_change()
    gap = (raw_ret - p["ret"]).abs()
    return p.assign(raw_ret=raw_ret, gap=gap)[gap > tol][["date", "ticker", "close", "raw_ret", "ret"]]


if __name__ == "__main__":
    pnl = build_panel()
    print(pnl.shape, pnl["date"].min().date(), pnl["date"].max().date(), pnl["ticker"].nunique(), "tickers")
    ev = check_adjustment(pnl)
    print(len(ev), "share-basis events, e.g.\n", ev.head(10).to_string())
    print("days with |ret| > 30%:", int((pnl["ret"].abs() > 0.30).sum()), "(KRX daily limit is 30%)")
