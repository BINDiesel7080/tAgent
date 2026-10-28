#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_per_ticker_features.py — сборка rl_features_v3: базовый датасет + 5 пер-тикерных фичей.

Кандидаты (скрининг 2026-10-23, вердикты без DUP):
  ret5, ret60      - доходности за 5 и 60 баров
  pct_above_sma50  - доля последних 20 дней выше SMA50 (персистентность тренда)
  mfi14            - Money Flow Index (14), нормирован в 0..1
  dd_from_high252  - просадка от максимума за ~252 бара (замена SMA200)

Конвенции (как в build_market_features.py):
  - колонки {TICKER}_{фича}, dtype float32;
  - нейтральные заполнения там, где история короче окна:
    ret/dd -> 0.0, доля/MFI -> 0.5;
  - число строк = числу строк базового датасета: фолды walk-forward не сдвигаются.

Источники данных (по цепочке):
  close/high/low - rl_prices.pkl (обязательно);
  volume         - _volume в rl_prices.pkl -> кэш candles_cache/<T>.csv -> докачка ISS (urllib).
  MFI без объёма -> нейтральный 0.5 с явным предупреждением.

В конце: dup-check новых колонок против ВСЕХ существующих (порог |rho|>0.85).

Запуск:
  python build_per_ticker_features.py --features rl_features_v2.pkl --out rl_features_v3.pkl
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

CACHE_DIR = 'candles_cache'
FRESH_DAYS = 7
DUP_RHO = 0.85
PAGE = 500
UA = 'Mozilla/5.0 (build_per_ticker_features; personal research)'
W_SMA, W_PCT, W_MFI = 50, 20, 14
BUFFER_DAYS = 400          # запас календарных дней на прогрев при докачке


def log(msg=''):
    print(msg, flush=True)


# ── Данные ──────────────────────────────────────────────────────
def fetch_candles(sec, from_date):
    """Свечи ISS с кэшем. from_date - str 'YYYY-MM-DD'. None при сбое."""
    path = os.path.join(CACHE_DIR, f'{sec}.csv')
    if os.path.exists(path):
        fresh = (time.time() - os.path.getmtime(path)) < FRESH_DAYS * 86400
        if fresh:
            try:
                df = pd.read_csv(path, parse_dates=['date'])
                if len(df) and df['date'].min() <= pd.Timestamp(from_date):
                    return df
                log(f'  {sec}: кэш не покрывает {from_date}, докачка')
            except Exception as e:
                log(f'  {sec}: кэш нечитаем ({type(e).__name__}), докачка')
    base = ('https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/'
            f'securities/{sec}/candles.json?interval=24')
    rows, offset = [], 0
    try:
        while True:
            url = f'{base}&from={from_date}&start={offset}'
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            block = data.get('candles') or {}
            cols = block.get('columns') or []
            page = block.get('data') or []
            if not cols or not page:
                break
            ix = {c: i for i, c in enumerate(cols)}
            for r in page:
                rows.append({
                    'date': r[ix['begin']][:10],
                    'open': r[ix['open']], 'high': r[ix['high']],
                    'low': r[ix['low']], 'close': r[ix['close']],
                    'value': r[ix['value']], 'volume': r[ix['volume']],
                })
            offset += PAGE
        if not rows:
            log(f'  {sec}: ISS вернул 0 свечей')
            return None
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', kind='stable').drop_duplicates('date', keep='last')
        df.to_csv(path, index=False)
        log(f'  {sec}: скачано {len(df)} баров с {df["date"].min().date()}')
        return df
    except Exception as e:
        log(f'  {sec}: сеть недоступна - {type(e).__name__}: {e}')
        return None


def series_from(df, dates, col):
    """Серия col по датам df, выровненная по dates; NaN там, где нет бара."""
    if df is None or col not in df.columns:
        return None
    s = pd.Series(df[col].to_numpy(), index=pd.DatetimeIndex(df['date']))
    return s.reindex(dates)


# ── Расчёт фичей ────────────────────────────────────────────────
def per_ticker_features(t, dates, prices, cache_dfs):
    """Пять пер-тикерных фичей на индексе dates. Возвращает dict серий."""
    c = prices[f'{t}_close'].astype(float)
    assert list(c.index) == list(dates)

    # high/low: из prices -> из кэша -> degenerate = close
    h = prices[f'{t}_high'].astype(float) if f'{t}_high' in prices.columns else None
    l = prices[f'{t}_low'].astype(float) if f'{t}_low' in prices.columns else None
    src = 'prices'
    if h is None:
        h = series_from(cache_dfs.get(t), dates, 'high')
        src = 'cache'
    if l is None:
        l = series_from(cache_dfs.get(t), dates, 'low')
    if h is None:
        h, l, src = c, c, 'degenerate (close)'
    h, l = h.fillna(c), l.fillna(c)

    # volume: из prices -> из кэша -> None (MFI нейтральный)
    v = prices[f'{t}_volume'].astype(float) if f'{t}_volume' in prices.columns else None
    vsrc = 'prices'
    if v is None:
        v = series_from(cache_dfs.get(t), dates, 'volume')
        vsrc = 'cache' if v is not None else 'none'
    if v is not None:
        v = v.fillna(0.0)

    out = {}

    # 1) доходности на 5 и 60 баров
    out[f'{t}_ret5'] = c.pct_change(5).fillna(0.0)
    out[f'{t}_ret60'] = c.pct_change(60).fillna(0.0)

    # 2) доля последних 20 дней выше SMA50 (строгая: пока SMA50 не определена - не считаем)
    sma50 = c.rolling(W_SMA, min_periods=W_SMA).mean()
    above = (c > sma50).astype(float)
    above[sma50.isna()] = np.nan
    out[f'{t}_pct_above_sma50'] = above.rolling(W_PCT, min_periods=W_PCT).mean().fillna(0.5)

    # 3) MFI14 в 0..1
    if v is None:
        out[f'{t}_mfi14'] = pd.Series(0.5, index=dates)
    else:
        tp = (h + l + c) / 3.0
        flow = tp * v
        d = tp.diff()
        pos = flow.where(d > 0, 0.0).rolling(W_MFI, min_periods=W_MFI).sum()
        neg = flow.where(d < 0, 0.0).rolling(W_MFI, min_periods=W_MFI).sum()
        mfi = pos / (pos + neg)
        mfi = mfi.where((pos + neg) > 0, 0.5)      # нулевой оборот - нейтраль
        out[f'{t}_mfi14'] = mfi.fillna(0.5)

    # 4) просадка от максимума за 252 бара (expanding до 252 - без доп. прогрева)
    hh = c.rolling(252, min_periods=1).max()
    out[f'{t}_dd_from_high252'] = c / hh - 1.0

    for k, s in out.items():
        out[k] = s.fillna(0.5 if k.endswith('_mfi14') or k.endswith('pct_above_sma50') else 0.0)

    info = {'src_hl': src, 'src_v': vsrc, 'n_nan_src': int(series_from(cache_dfs.get(t), dates, 'close').isna().sum() if cache_dfs.get(t) is not None else 0)}
    return out, info


# ── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='rl_features.pkl',
                    help='базовый датасет (rl_features.pkl или rl_features_v2.pkl)')
    ap.add_argument('--prices', default='rl_prices.pkl')
    ap.add_argument('--out', default='rl_features_v3.pkl')
    args = ap.parse_args()

    features = pd.read_pickle(args.features).astype(np.float32)
    prices = pd.read_pickle(args.prices).astype(np.float32)

    tickers = sorted({c.rsplit('_', 1)[0] for c in features.columns
                      if c.endswith('_returns')})
    missing = [t for t in tickers if f'{t}_close' not in prices.columns]
    assert not missing, f'В prices нет close для: {missing}'
    assert list(prices.index) == list(features.index), \
        'Индексы features и prices не совпадают'

    dates = pd.DatetimeIndex(features.index)
    t0 = dates.min()
    from_date = (t0 - dt.timedelta(days=BUFFER_DAYS)).isoformat()[:10]
    log(f"База: {args.features}: {len(features)} строк, {features.shape[1]} колонок")
    log(f"Период: {t0.date()} .. {dates.max().date()} | тикеры: {', '.join(tickers)}")
    log(f"Докачка свечей с {from_date} (кэш: {CACHE_DIR}/)")

    # кэш: грузим всё, что есть, по тикерам
    cache_dfs = {}
    for t in tickers:
        p = os.path.join(CACHE_DIR, f'{t}.csv')
        if os.path.exists(p):
            try:
                cache_dfs[t] = pd.read_csv(p, parse_dates=['date'])
            except Exception as e:
                log(f'  {t}: кэш нечитаем ({type(e).__name__})')

    out = features.copy()
    n_added = 0
    for t in tickers:
        df = cache_dfs.get(t)
        # докачка нужна, если в prices нет объёма/high/low,
        # а кэш не покрывает начало датасета
        covers = df is not None and df['date'].min() <= pd.Timestamp(from_date)
        need_fetch = (not covers) or f'{t}_volume' not in prices.columns \
                                   or f'{t}_high' not in prices.columns
        if need_fetch:
            got = fetch_candles(t, from_date)
            if got is not None:
                cache_dfs[t] = got

        feats, info = per_ticker_features(t, dates, prices, cache_dfs)
        for name, s in feats.items():
            out[name] = s.to_numpy().astype(np.float32)
            n_added += 1
        vnote = {'prices': 'volume из prices', 'cache': 'volume из кэша',
                 'none': 'объёма нет - MFI нейтральный 0.5'}[info['src_v']]
        log(f'  {t}: 5 фичей | h/l: {info["src_hl"]} | {vnote}')
        if info['n_nan_src']:
            log(f'      !! в кэше {t} не хватает {info["n_nan_src"]} дат датасета -'
                f' заполнено нейтрально (ret/dd 0.0, доли/MFI 0.5)')

    # ── Dup-check новых против всех существующих ──
    old_cols = list(features.columns)
    new_cols = [c for c in out.columns if c not in old_cols]
    log(f"\n=== DUP-CHECK: {len(new_cols)} новых против {len(old_cols)} существующих ===")
    worst, dups = 0.0, []
    for nc in new_cols:
        with np.errstate(all='ignore'):
            rhos = out[old_cols].corrwith(out[nc]).abs()
        m, best = float(rhos.max()), rhos.idxmax()
        worst = max(worst, m)
        flag = 'DUP!' if m > DUP_RHO else 'ok '
        if m > DUP_RHO:
            dups.append((nc, best, m))
        log(f'  {flag} {nc:<28} max |rho| = {m:.3f} против {best}')
    if dups:
        log('\n!!! DUP-колонки (убрать и пересобрать):')
        for nc, best, m in dups:
            log(f'      {nc} ~ {best} (rho={m:.3f})')

    nan_n = int(out[new_cols].isna().sum().sum())
    out = out.astype(np.float32)
    log(f"\nКолонок было: {features.shape[1]}, стало: {out.shape[1]} (+{n_added})")
    log(f"Строк: {len(out)} (совпало с базой: {len(out) == len(features)})")
    log(f"NaN в новых колонках: {nan_n}")
    log(f"Худшая корреляция новых с существующими: {worst:.3f}")

    out.to_pickle(args.out)
    log(f"Сохранено: {args.out}")


if __name__ == '__main__':
    main()
