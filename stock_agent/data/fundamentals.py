"""Point-in-time fundamentals from the DART headline accounts.

One row per (stock, report) with the date the market could first use it:
    avail = the trading day after the filing date (first 8 digits of rcept_no).

Income-statement items are turned into quarterly and trailing-twelve-month (TTM) values:
    Q1 = Q1 YTD, Q2 = H1 YTD - Q1 YTD, Q3 = Q3 YTD - H1 YTD, Q4 = FY - Q3 YTD.
Balance-sheet items are as of the report's period end.

Caveat: DART returns the latest version of a report. When a report was later amended, rcept_no is the
amendment's, so the figure becomes "available" later than the original filing (conservative on timing,
but the values are the revised ones).
"""
import glob
import json

import numpy as np
import pandas as pd

from stock_agent.config import CLEAN, RAW

ACCOUNTS = {"매출액": "revenue", "영업이익": "op_income", "당기순이익(손실)": "net_income",
            "자산총계": "assets", "부채총계": "liabilities", "자본총계": "equity"}
IS_ITEMS = ["revenue", "op_income", "net_income"]
ORDER = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}


def _num(s):
    if s is None:
        return np.nan
    s = str(s).replace(",", "").strip()
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def load_raw_accounts() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RAW / "dart" / "main_accounts" / "*.json")):
        rows += json.loads(open(f).read())
    df = pd.DataFrame(rows)
    df = df[df["account_nm"].isin(ACCOUNTS)].copy()
    df["item"] = df["account_nm"].map(ACCOUNTS)
    # income statement: the cumulative year-to-date amount when given, else this period's amount
    ytd = df.get("thstrm_add_amount")
    df["value"] = df["thstrm_amount"].map(_num)
    if ytd is not None:
        y = ytd.map(_num)
        is_is = df["sj_div"].eq("IS") & df["reprt_code"].ne("11011")
        df.loc[is_is & y.notna(), "value"] = y[is_is & y.notna()]
    return df


def build(save: bool = True) -> pd.DataFrame:
    df = load_raw_accounts()
    # consolidated statements where available, otherwise separate
    df["cfs"] = df["fs_div"].eq("CFS")
    key = ["stock_code", "bsns_year", "reprt_code"]
    has_cfs = df.groupby(key)["cfs"].transform("any")
    df = df[df["cfs"] == has_cfs]
    wide = df.pivot_table(index=key + ["rcept_no"], columns="item", values="value", aggfunc="first").reset_index()
    wide["q"] = wide["reprt_code"].map(ORDER)
    wide["year"] = wide["bsns_year"].astype(int)
    wide = wide.sort_values(["stock_code", "year", "q", "rcept_no"]).drop_duplicates(["stock_code", "year", "q"], keep="last")

    # quarterly values from year-to-date figures
    for it in IS_ITEMS:
        if it not in wide:
            wide[it] = np.nan
        prev_ytd = wide.groupby(["stock_code", "year"])[it].shift()
        wide[it + "_q"] = np.where(wide["q"] == 1, wide[it], wide[it] - prev_ytd)
        # needs consecutive quarters; a gap makes the quarter unknown
        gap = wide.groupby(["stock_code", "year"])["q"].diff().fillna(1).ne(1) & wide["q"].ne(1)
        wide.loc[gap, it + "_q"] = np.nan
    # TTM = sum of the last 4 quarters, only when they are consecutive
    wide["qidx"] = wide["year"] * 4 + wide["q"]
    consec = wide.groupby("stock_code")["qidx"].diff(3).eq(3)
    for it in IS_ITEMS:
        roll = wide.groupby("stock_code")[it + "_q"].transform(lambda s: s.rolling(4, min_periods=4).sum())
        wide[it + "_ttm"] = roll.where(consec)
        # same quarter a year earlier, for growth
        wide[it + "_q_yoy"] = wide.groupby("stock_code")[it + "_q"].shift(4).where(
            wide.groupby("stock_code")["qidx"].diff(4).eq(4))
    wide["filed"] = pd.to_datetime(wide["rcept_no"].str[:8], format="%Y%m%d")
    out = wide.drop(columns=["bsns_year", "reprt_code"])
    if save:
        CLEAN.mkdir(parents=True, exist_ok=True)
        out.to_parquet(CLEAN / "fundamentals.parquet", index=False)
    return out


def asof(fund: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Add `avail`: the first trading day strictly after the filing date."""
    td = pd.DatetimeIndex(trading_days).sort_values()
    pos = td.searchsorted(fund["filed"].values, side="right")
    ok = pos < len(td)
    f = fund[ok].copy()
    f["avail"] = td[pos[ok]]
    return f


def latest_as_of(f: pd.DataFrame, s, tickers) -> pd.DataFrame:
    """Most recent report per stock usable on date s (avail <= s)."""
    g = f[(f["avail"] <= s) & f["stock_code"].isin(tickers)]
    return g.sort_values(["avail", "qidx"]).groupby("stock_code").tail(1).set_index("stock_code")


if __name__ == "__main__":
    fd = build()
    print(fd.shape, fd["stock_code"].nunique(), "stocks", fd["filed"].min().date(), fd["filed"].max().date())
    print(fd[fd["stock_code"] == "005930"].tail(4)[["year", "q", "filed", "revenue_ttm", "op_income_ttm", "equity"]])
