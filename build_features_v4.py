#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_v4.py — легаси-пайплайн на полной истории (2014-2026).

31 колонка: is_weekend + {T}_{returns,atr_pct,sma_dev,vol_z,range_pct}.
Выход:
  rl_features_v4.pkl + rl_prices_v4.pkl — данные до 2026-10-02 включительно
    (цены выровнены по фичам, на «мосте» остановки торгов 02-03.2022 сплайс
    без порчи последующих доходностей);
  rl_features_v4_ext.pkl — строки после 2026-10-02 (пригодятся позже).

Запуск: python build_features_v4.py
"""
import numpy as np
import pandas as pd

CANDLES_FILE = 'candles_full.csv'
OUT_FEATS, OUT_PRICES = 'rl_features_v4.pkl', 'rl_prices_v4.pkl'
OUT_EXT = 'rl_features_v4_ext.pkl'
V2_FILE = 'rl_features_v2.pkl'
DATA_END = '2026-10-02'          # прежний конец данных (holdout не двигаем)
ATR_PERIOD, SMA_PERIOD, VOL_NORM_WINDOW = 14, 50, 50
GAP_DAYS = 10                    # обычные праздники до 10 дней не трогаем
TICKERS = ['SBER', 'MGNT', 'PLZL', 'NVTK', 'LKOH', 'TATN']
FEATS = ['returns', 'atr_pct', 'sma_dev', 'vol_z', 'range_pct']


def compute_features(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    prev_close = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - prev_close).abs(),
                    (df['low'] - prev_close).abs()], axis=1).max(axis=1)
    df['atr_pct'] = tr.rolling(ATR_PERIOD).mean() / df['close'] * 100
    sma = df['close'].rolling(SMA_PERIOD).mean()
    df['sma_dev'] = (df['close'] - sma) / sma * 100
    vol_mean = df['volume'].rolling(VOL_NORM_WINDOW).mean()
    vol_std = df['volume'].rolling(VOL_NORM_WINDOW).std()
    df['vol_z'] = (df['volume'] - vol_mean) / vol_std
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    return df[FEATS]


def main():
    raw = pd.read_csv(CANDLES_FILE, parse_dates=['date'])
    feats, closes = {}, {}
    for t in TICKERS:
        df = raw[raw['SECID'] == t].set_index('date').sort_index()
        feats[t] = compute_features(df)
        closes[t] = df['close']
        print(f"{t}: {len(df)} баров, {df.index.min().date()} .. {df.index.max().date()}")
    common = None
    for t in TICKERS:
        d = set(feats[t].index)
        common = d if common is None else common & d
    common = pd.DatetimeIndex(sorted(common))
    print(f"Общих сессий: {len(common)}, {common[0].date()} .. {common[-1].date()}")

    features = pd.DataFrame(index=common, dtype=np.float32)
    features['is_weekend'] = (common.dayofweek >= 5).astype(np.float32)
    for t in TICKERS:
        for f in FEATS:
            features[f'{t}_{f}'] = feats[t].reindex(common)[f]
    features = features.dropna()
    prices = pd.DataFrame(index=features.index, dtype=np.float64)
    for t in TICKERS:
        prices[f'{t}_close'] = closes[t].reindex(features.index)
    assert prices.notna().all().all(), "NaN в ценах после выравнивания"
    print(f"После dropna: {len(features)} строк (старт {features.index.min().date()})")

    # мост через остановку торгов: доходность = 0, цены сплайсятся
    ddays = features.index.to_series().diff().dt.days
    bridge = (ddays > GAP_DAYS).fillna(False).values
    ret_cols = [f'{t}_returns' for t in TICKERS]
    features.loc[bridge, ret_cols] = 0.0
    for t in TICKERS:
        col = f'{t}_close'
        for i in np.where(bridge)[0]:
            factor = prices[col].iloc[i - 1] / prices[col].iloc[i]
            prices.loc[prices.index[i:], col] *= factor
    for d in features.index[bridge]:
        print(f"мост: {d.date()} (разрыв {int(ddays.loc[d])} дн.)")

    # обрезка по прежней границе данных + свежие строки отдельно
    main_mask = features.index <= pd.Timestamp(DATA_END)
    ext = features.loc[~main_mask]
    features = features.loc[main_mask]
    prices = prices.loc[main_mask]
    if len(ext):
        ext.to_pickle(OUT_EXT)
        print(f"Свежие строки после {DATA_END}: {len(ext)} -> {OUT_EXT} (в базу v4 не входят)")

    features.to_pickle(OUT_FEATS)
    prices.to_pickle(OUT_PRICES)
    print(f"\n{OUT_FEATS}: {features.shape}, {OUT_PRICES}: {prices.shape}")

    # сверка с v2: индекс на индекс, хвост v4 длиной v2
    try:
        v2 = pd.read_pickle(V2_FILE)
        v2i = pd.DatetimeIndex(v2.index)
        if len(features) < len(v2i):
            print(f"Сверка: в v4 меньше строк ({len(features)}) чем в v2 ({len(v2i)}) — не сверяю")
        else:
            seg = features.index[-len(v2i):]
            mism = np.where(seg.values != v2i.values)[0]
            if len(mism) == 0:
                print(f"КАЛЕНДАРЬ СОВПАДАЕТ: хвост {len(v2i)} дат v4 "
                      f"({seg[0].date()} .. {seg[-1].date()}) = v2 "
                      f"({v2i[0].date()} .. {v2i[-1].date()})")
            else:
                print(f"РАСХОЖДЕНИЕ: {len(mism)} дат из {len(v2i)}")
                for k in mism[:5]:
                    print(f"  строка {k}: v4 {seg[k].date()} vs v2 {v2i[k].date()}")
        print(f"holdout v4 (последние 101): {features.index[-101].date()} .. "
              f"{features.index[-1].date()}")
    except FileNotFoundError:
        print("v2 не найдена — сверка пропущена")


if __name__ == '__main__':
    main()
