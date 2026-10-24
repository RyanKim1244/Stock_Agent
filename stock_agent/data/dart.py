"""OpenDART collector (needs DART_API_KEY).

- corp_codes(): every company DART knows, with its 6-digit stock code when listed.
- collect_main_accounts(): headline accounts (revenue, operating income, net income, assets, liabilities,
  equity) for listed companies, 100 companies per call, every annual and quarterly report from 2015.

Point in time: each row keeps rcept_no, whose first 8 digits are the date the report was filed.
A figure may be used only from the trading day after that date.

Usage:  python -m stock_agent.data.dart codes
        python -m stock_agent.data.dart accounts 2015 2026
"""
import io
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from stock_agent.config import RAW

OUT = RAW / "dart"
API = "https://opendart.fss.or.kr/api"
REPORTS = {"11013": "Q1", "11012": "H1", "11014": "Q3", "11011": "FY"}


class DartLimit(Exception):
    """Daily call quota used up (status 020)."""


def _key() -> str:
    k = os.environ.get("DART_API_KEY")
    if not k:
        raise RuntimeError("DART_API_KEY is not set (.env)")
    return k


def _get(path: str, timeout: int = 600, **params) -> requests.Response:
    params["crtfc_key"] = _key()
    for attempt in range(4):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"{path} attempt {attempt + 1}: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{path} failed after retries")


def corp_codes(refresh: bool = False) -> pd.DataFrame:
    p = OUT / "corp_codes.parquet"
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    r = _get("corpCode.xml")
    xml = zipfile.ZipFile(io.BytesIO(r.content)).read("CORPCODE.xml")
    rows = [{c.tag: (c.text or "").strip() for c in el} for el in ET.fromstring(xml)]
    df = pd.DataFrame(rows)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def listed_corps() -> pd.DataFrame:
    df = corp_codes()
    return df[df["stock_code"].str.len() == 6].reset_index(drop=True)


def fetch_main_accounts(corps: list[str], year: int, reprt_code: str) -> list[dict]:
    r = _get("fnlttMultiAcnt.json", timeout=120, corp_code=",".join(corps),
             bsns_year=str(year), reprt_code=reprt_code)
    js = r.json()
    status = js.get("status")
    if status == "020":
        raise DartLimit(js.get("message"))
    if status == "013":  # no data
        return []
    if status != "000":
        raise RuntimeError(f"DART status {status}: {js.get('message')}")
    return js["list"]


def collect_main_accounts(y0: int, y1: int, batch: int = 100) -> None:
    corps = listed_corps()["corp_code"].tolist()
    chunks = [corps[i:i + batch] for i in range(0, len(corps), batch)]
    d = OUT / "main_accounts"
    d.mkdir(parents=True, exist_ok=True)
    calls = 0
    for year in range(y0, y1 + 1):
        for code, label in REPORTS.items():
            for i, chunk in enumerate(chunks):
                p = d / f"{year}_{label}_{i:03d}.json"
                if p.exists():
                    continue
                try:
                    rows = fetch_main_accounts(chunk, year, code)
                except DartLimit as e:
                    print(f"daily limit reached after {calls} calls: {e}", flush=True)
                    return
                p.write_text(json.dumps(rows, ensure_ascii=False))
                calls += 1
                if calls % 25 == 0:
                    print(f"{year} {label} chunk {i}/{len(chunks)} calls={calls}", flush=True)
                time.sleep(0.3)
    print(f"finished, calls={calls}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "codes":
        df = corp_codes(refresh=True)
        print(len(df), "companies,", (df["stock_code"].str.len() == 6).sum(), "with a stock code")
    elif cmd == "accounts":
        collect_main_accounts(int(sys.argv[2]), int(sys.argv[3]))
