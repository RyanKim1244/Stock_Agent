"""Daily whole-market snapshots from KRX (via pykrx, needs KRX_ID / KRX_PW).

One snapshot per trading day and market holds every stock listed that day, delisted ones included,
so a universe built from these files has no survivorship bias.

Raw prices are not adjusted for splits or rights issues. Returns come from KRX's own daily change
(`등락률`), which is measured against an adjusted base price; see build_panel in prices.py.

Usage:  python -m stock_agent.data.krx 2015-01-01 2026-10-24
"""
import sys
import time
import warnings
from datetime import date, timedelta

import pandas as pd

from stock_agent.config import RAW

warnings.filterwarnings("ignore")
OUT = RAW / "krx_daily"
MARKETS = ("KOSPI", "KOSDAQ")
COLS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
        "거래대금": "value", "등락률": "chg_pct", "시가총액": "mcap"}


def day_path(d: date):
    return OUT / str(d.year) / f"{d:%Y%m%d}.parquet"


def fetch_day(d: date) -> pd.DataFrame:
    from pykrx import stock
    frames = []
    for m in MARKETS:
        df = stock.get_market_ohlcv(f"{d:%Y%m%d}", market=m)
        if df is None or df.empty:
            continue
        df = df.rename(columns=COLS)[list(COLS.values())]
        df.index.name = "ticker"
        df = df.reset_index()
        df["market"] = m
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.Timestamp(d)
    return out


def is_trading_snapshot(df: pd.DataFrame) -> bool:
    # On holidays KRX returns the last session's prices with zero volume everywhere.
    return not df.empty and df["volume"].sum() > 0


def collect(start: date, end: date, pause: float = 0.2) -> None:
    d = start
    done = skipped = 0
    t0 = time.time()
    while d <= end:
        p = day_path(d)
        if d.weekday() < 5 and not p.exists() and not p.with_suffix(".holiday").exists():
            for attempt in range(3):
                try:
                    df = fetch_day(d)
                    break
                except Exception as e:  # network hiccup or expired session: wait and retry
                    print(f"{d} attempt {attempt + 1} failed: {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                d += timedelta(days=1)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            if is_trading_snapshot(df):
                df.to_parquet(p, index=False)
                done += 1
            else:
                p.with_suffix(".holiday").touch()
                skipped += 1
            if (done + skipped) % 50 == 0:
                print(f"{d} saved={done} holidays={skipped} elapsed={time.time() - t0:.0f}s", flush=True)
            time.sleep(pause)
        d += timedelta(days=1)
    print(f"finished: saved={done} holidays={skipped} elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    a, b = (date.fromisoformat(x) for x in sys.argv[1:3])
    collect(a, b)
