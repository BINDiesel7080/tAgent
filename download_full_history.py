#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_full_history.py (v2) — дневные свечи через /candles (как легаси),
включая выходные сессии. OHLC + volume -> candles_full.csv
Запуск: python download_full_history.py
"""
import time, sys
import pandas as pd
import requests

TICKERS = ["SBER", "MGNT", "PLZL", "NVTK", "LKOH", "TATN"]
BOARD = "TQBR"
FROM, TILL = "2014-01-01", "2026-10-23"
BASE = (f"https://iss.moex.com/iss/engines/stock/markets/shares/"
        f"boards/{BOARD}/securities")
UA = "Mozilla/5.0 (personal research)"
PAUSE = 0.35


def fetch_candles(sec):
    rows, offset = [], 0
    while True:
        url = (f"{BASE}/{sec}/candles.json?interval=24&from={FROM}&till={TILL}"
               f"&start={offset}&iss.meta=off")
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        block = r.json().get("candles") or {}
        data = block.get("data") or []
        if not data:
            break
        ix = {c: i for i, c in enumerate(block["columns"])}
        for row in data:
            rows.append({
                "date": row[ix["begin"]][:10],
                "open": row[ix["open"]], "high": row[ix["high"]],
                "low": row[ix["low"]], "close": row[ix["close"]],
                "volume": row[ix["volume"]], "value": row[ix["value"]],
            })
        offset += len(data)
        if len(data) < 500:
            break
        time.sleep(PAUSE)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").drop_duplicates("date", keep="last").set_index("date")


def main():
    frames = []
    for t in TICKERS:
        print(f"[{t}] качаю {FROM} .. {TILL} ...", flush=True)
        try:
            df = fetch_candles(t)
        except Exception as e:
            print(f"[{t}] ОШИБКА: {e}")
            continue
        if df.empty:
            print(f"[{t}] пусто")
            continue
        df["SECID"] = t
        frames.append(df)
        print(f"[{t}] {df.index.min().date()} .. {df.index.max().date()}, баров: {len(df)}")
        time.sleep(PAUSE)
    if not frames:
        sys.exit("Ничего не скачалось")
    allc = pd.concat(frames)
    allc["weekend"] = allc.index.dayofweek >= 5
    per_year = allc.groupby([allc.index.year, allc["weekend"]]).size().unstack(fill_value=0)
    print("\n=== Строк по годам: False=будни, True=выходные сессии ===")
    print(per_year.to_string())
    allc = allc.reset_index()   # дата возвращается из индекса в колонку
    allc.to_csv("candles_full.csv", index=False)
    print("\nСохранено: candles_full.csv (rl_prices_full.pkl больше не нужен)")


if __name__ == "__main__":
    main()
