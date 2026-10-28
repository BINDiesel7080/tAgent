
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEATURE SCREENING v2: пер-тикерные кандидаты для rl_features_v3.

Кандидаты:
  ret5, ret20, ret60      - доходности на 3 горизонтах (неделя/месяц/квартал)
  sma_dev20, sma_dev50    - отклонение цены от SMA
  pct_above_sma50         - доля последних 20 дней выше SMA50
  vol_rub_z               - z-score оборота в рублях (поле value ISS), окно 20
  mfi14                   - Money Flow Index (14), нормирован в 0..1
  dd_from_high252         - просадка от максимума за ~252 торговых дня
                            (замена SMA200 из решения о 200-дневной)

Зависимости: только pandas + numpy. Сеть - штатный urllib из stdlib,
никаких сторонних HTTP-библиотек. При любом сбое печатается точная причина.

Кэш: candles_cache/<TICKER>.csv, не перекачивается 7 дней.

Запуск:
  python feature_screening.py            # обычный прогон
  python feature_screening.py --days 0   # только из кэша, свежесть не проверять
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

UNIVERSE = ['SBER', 'LKOH', 'GMKN', 'MOEX', 'CHMF', 'PLZL']
CACHE_DIR = 'candles_cache'
FRESH_DAYS = 7
DUP_RHO = 0.85
PAGE = 500
UA = 'Mozilla/5.0 (feature_screening; personal research)'

CANDIDATES = ['ret5', 'ret20', 'ret60', 'sma_dev20', 'sma_dev50',
              'pct_above_sma50', 'vol_rub_z', 'mfi14', 'dd_from_high252']


def log(msg=''):
    print(msg, flush=True)


# ── Данные ──────────────────────────────────────────────────────
def fetch_candles(sec, days, only_cache=False):
    """Свечи ISS c кэшем. При сетевом сбое печатает точную причину."""
    path = os.path.join(CACHE_DIR, f'{sec}.csv')
    if os.path.exists(path):
        fresh = (time.time() - os.path.getmtime(path)) < FRESH_DAYS * 86400
        if fresh or only_cache:
            try:
                df = pd.read_csv(path, parse_dates=['date'])
                if len(df):
                    return df
            except Exception:
                pass
    if only_cache:
        log(f'  {sec}: в кэше нет валидных данных, сеть отключена (--days 0)')
        return None

    base = (f'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/'
            f'securities/{sec}/candles.json?interval=24')
    frm = (dt.date.today() - dt.timedelta(days=days + 30)).isoformat()
    rows, offset = [], 0
    try:
        while True:
            url = f'{base}&from={frm}&start={offset}'
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
            if len(page) < PAGE:
                break
            offset += PAGE
    except Exception as e:                     # точная причина, без заглушек
        log(f'  {sec}: сеть недоступна - {type(e).__name__}: {e}')
        return None

    if not rows:
        log(f'  {sec}: ISS вернул 0 свечей')
        return None
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = (df.drop_duplicates('date').sort_values('date')
            .reset_index(drop=True))
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ── Кандидаты ───────────────────────────────────────────────────
def compute_candidates(df):
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    out = pd.DataFrame(index=df['date'])
    for k in (5, 20, 60):
        out[f'ret{k}'] = c.pct_change(k).values
    for w in (20, 50):
        out[f'sma_dev{w}'] = (c / c.rolling(w).mean() - 1).values
    above = (c > c.rolling(50).mean()).astype(float)
    out['pct_above_sma50'] = above.rolling(20).mean().values
    mean_v, std_v = v.rolling(20).mean(), v.rolling(20).std()
    out['vol_rub_z'] = ((v - mean_v) / std_v.replace(0, np.nan)).values
    tp = (h + l + c) / 3
    mf = tp * v
    diff = tp.diff()
    pos = mf.where(diff > 0, 0.0).rolling(14).sum()
    neg = mf.where(diff < 0, 0.0).rolling(14).sum()
    denom = (pos + neg).replace(0, np.nan)
    out['mfi14'] = (pos / denom).values
    out['dd_from_high252'] = (c / c.rolling(252, min_periods=60).max() - 1).values
    return out


# ── Диагностика: автокорреляция доходностей, лаги 1..60 ─────────
def acf_summary(df, ticker, max_lag=60):
    ret = df['close'].pct_change().dropna()
    n = len(ret)
    thr = 2.0 / math.sqrt(n) if n else np.nan
    rows = []
    for lag in range(1, max_lag + 1):
        a = ret.autocorr(lag)
        if pd.isna(a):
            continue
        rows.append({'ticker': ticker, 'lag': lag, 'acf': round(a, 4),
                     'significant': bool(abs(a) > thr)})
    return rows, thr


# ── Сверка с существующими фичами ───────────────────────────────
def aligned_corr(series, col):
    """Корреляция с выравниванием: по датам, если есть, иначе по хвосту."""
    s = series.dropna()
    if isinstance(s.index, pd.DatetimeIndex) and isinstance(col.index, pd.DatetimeIndex):
        m = pd.concat([s.rename('a'), col.rename('b')], axis=1, join='inner').dropna()
        if len(m) < 50:
            return None
        return float(m['a'].corr(m['b']))
    n = min(len(s), len(col))
    if n < 50:
        return None
    a = s.iloc[-n:].reset_index(drop=True)
    b = col.iloc[-n:].reset_index(drop=True)
    mask = a.notna() & b.notna()
    if mask.sum() < 50:
        return None
    return float(a[mask].corr(b[mask]))


def load_existing(path):
    if not os.path.exists(path):
        return None, 'файл не найден'
    try:
        feat = pd.read_pickle(path)
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'
    if not isinstance(feat, pd.DataFrame):
        return None, f'ожидался DataFrame, пришёл {type(feat).__name__}'
    return feat, None


def main():
    ap = argparse.ArgumentParser(description='Скрининг кандидатов фичей v3')
    ap.add_argument('--days', type=int, default=900,
                    help='глубина истории; 0 = только кэш')
    ap.add_argument('--universe', nargs='+', default=UNIVERSE)
    ap.add_argument('--features', default='rl_features.pkl')
    ap.add_argument('--dup-rho', type=float, default=DUP_RHO)
    args = ap.parse_args()
    only_cache = args.days == 0

    log('=' * 64)
    log('  FEATURE SCREENING v2: пер-тикерные кандидаты для rl_features_v3')
    log(f'  Вселенная: {args.universe} | история: {args.days} дней')
    log('  Сеть: urllib из stdlib (requests больше не нужен)')
    log('=' * 64)

    data = {}
    for sec in args.universe:
        df = fetch_candles(sec, args.days, only_cache)
        if df is None or len(df) < 120:
            n = 0 if df is None else len(df)
            log(f'  {sec}: данных мало ({n} баров) - пропускаю')
            continue
        log(f'  {sec}: {len(df)} баров, '
            f'{df["date"].min():%Y-%m-%d}..{df["date"].max():%Y-%m-%d}')
        data[sec] = df
    if not data:
        log('\nДанных нет: кэш пуст, сеть недоступна или вернула ошибку. Выход.')
        sys.exit(1)

    cand = {sec: compute_candidates(df) for sec, df in data.items()}

    # ── 1. ACF-диагностика горизонтов ──
    log('\n=== АВТОКОРРЕЛЯЦИЯ ДОХОДНОСТЕЙ (лаги 1-60) ===')
    acf_all, sig_all = [], []
    for sec, df in data.items():
        rows, thr = acf_summary(df, sec)
        acf_all.extend(rows)
        sig = [r for r in rows if r['significant']]
        sig_all.append(sig)
        if sig:
            top = sorted(sig, key=lambda r: -abs(r['acf']))[:5]
            txt = ', '.join(f"lag {r['lag']}: {r['acf']:+.2f}" for r in top)
            log(f'  {sec}: {len(sig)} значимых лагов (порог |ACF|>{thr:.2f}); топ: {txt}')
        else:
            log(f'  {sec}: значимых лагов НЕТ (порог |ACF|>{thr:.2f}) - '
                f'медвежьи снимки выжимают почти всё')
    pd.DataFrame(acf_all).to_csv('acf_diag.csv', index=False)
    log('  сохранено: acf_diag.csv')

    # ── 2. Кросс-корреляция кандидатов между собой ──
    log('\n=== КРОСС-КОРРЕЛЯЦИЯ КАНДИДАТОВ (средняя |rho| по тикерам) ===')
    cross_mean = {c: 0.0 for c in CANDIDATES}
    dup_pairs = []
    for sec, cdf in cand.items():
        corr = cdf.corr().abs()
        for c in CANDIDATES:
            others = [d for d in CANDIDATES if d != c]
            if c in corr.columns and all(d in corr.columns for d in others):
                cross_mean[c] += float(corr.loc[c, others].mean()) / len(cand)
        for i, c1 in enumerate(CANDIDATES):
            for c2 in CANDIDATES[i + 1:]:
                if c1 in corr.columns and c2 in corr.columns:
                    r = corr.loc[c1, c2]
                    if r > 0.9:
                        dup_pairs.append((sec, c1, c2, r))
    for c in CANDIDATES:
        log(f'  {c:18s} средняя |rho| с остальными: {cross_mean[c]:.2f}')
    if dup_pairs:
        log('  ВНИМАНИЕ, пары-дубли внутри кандидатов (|rho|>0.9):')
        for sec, c1, c2, r in dup_pairs:
            log(f'    {sec}: {c1} - {c2}: {r:.2f}')

    # ── 3. Сверка с rl_features.pkl (полный скан, DUP по макс. |rho|) ──
    log('\n=== СВЕРКА С СУЩЕСТВУЮЩИМИ ФИЧАМИ ===')
    feat, err = load_existing(args.features)
    result_rows = []
    if feat is None:
        log(f'  {args.features}: пропускаю сверку ({err})')
    else:
        cols = [c for c in feat.columns
                if pd.api.types.is_numeric_dtype(feat[c])]
        by_ticker = {sec: [c for c in cols if sec.upper() in c.upper()]
                     for sec in cand}
        matched_any = any(by_ticker.values())
        if not matched_any:
            log('  Пер-тикерных колонок не нашёл. Первые 40 колонок файла:')
            log('  ' + ' | '.join(map(str, feat.columns[:40])))
            log('  Скажите Алисе формат шапки - добавлю парсер.')
        else:
            for cname in CANDIDATES:
                best_name, best_rho, best_t = None, None, None
                for sec, cdf in cand.items():
                    if cname not in cdf.columns:
                        continue
                    for col in by_ticker.get(sec, []):
                        r = aligned_corr(cdf[cname], feat[col])
                        if r is not None and (best_rho is None
                                              or abs(r) > abs(best_rho)):
                            best_name, best_rho, best_t = col, r, sec
                verdict = 'DUP' if best_rho is not None and abs(best_rho) > args.dup_rho else 'OK'
                match = ('-' if best_rho is None
                         else f'{best_t}:{best_name} (rho={best_rho:+.3f})')
                result_rows.append({'candidate': cname, 'verdict': verdict,
                                    'best_match': match,
                                    'max_abs_rho': ('' if best_rho is None
                                                    else f'{abs(best_rho):.3f}'),
                                    'cross_corr_mean': f'{cross_mean[cname]:.2f}'})
                log(f'  {cname:18s} {verdict:3s} лучший матч: {match}')

    if result_rows:
        pd.DataFrame(result_rows).to_csv('candidates_screening.csv', index=False)
        log('\n  сохранено: candidates_screening.csv')
    log('\nГотово!')


if __name__ == '__main__':
    main()
