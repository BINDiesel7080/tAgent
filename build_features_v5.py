#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features_v5.py — фичи для итерации v1.2 (ковариационные признаки).

Version: v5.0
Date:    2026-10-28
Новое относительно v4:
  - добавлены 18 ковариационных колонок второго порядка:
      {T}_rv20      = std({T}_returns, окно 20) * 100
      {T}_rv60      = std({T}_returns, окно 60) * 100
      {T}_correq20  = corr({T}_returns, mean_returns, окно 20)
    где mean_returns — среднее шести {T}_returns в каждой строке.
  - окна считаются на МАКСИМАЛЬНО доступной истории candles_full.csv
    (включая строки до 2014-08-19) и только потом нарезаются на индекс v4.

Что НЕ меняется:
  - 31 легаси-колонка v4: формулы импортируются из build_features_v4.py
    (compute_features, TICKERS, FEATS, GAP_DAYS, DATA_END), а не переписываются;
  - rl_features_v4.pkl / rl_prices_v4.pkl только читаются (сверка), не пишутся;
  - разметка фолдов, награда и среда не трогаются.

Доходностная серия для окон (согласовано с пользователем, вариант A):
  1) база — сырые pct_change() по полной истории тикера, выровненные на общий
     календарь (пересечение сессий всех 6 тикеров);
  2) строки, входящие в индекс rl_features_v4.pkl, ПЕРЕЗАПИСЫВАЮТСЯ значениями
     из колонок {T}_returns файла v4 байт-в-байт (там доходность «моста»
     2022-03-24 уже обнулена).
  Итог: на всех 3161 строках агент видит в obs и в окнах ОДНИ И ТЕ ЖЕ доходности.

Выход: rl_features_v5.pkl — 3161 строка x 49 колонок
       (31 легаси + rv20 x6 + rv60 x6 + correq20 x6).
       dtype: 31 легаси-колонка сохраняет dtype файла v4 (is_weekend float32,
       остальные 30 — float64), 18 новых — float32. Это единственный способ
       выполнить требование «байт-в-байт как в v4»: приведение легаси к float32
       даёт расхождение ~1e-6 и ломает np.array_equal.
       PortfolioEnv при загрузке сам делает .astype(np.float32).

Гарантии:
  - assert np.array_equal с rl_features_v4.pkl по 31 легаси-колонке; при
    расхождении файл НЕ пишется (завершение с ненулевым кодом);
  - NaN в финальном файле НЕ допускаются: rolling(min_periods=20), недобор
    наблюдений и вырожденный std=0 заполняются нейтрально 0.0.

Запуск: python build_features_v5.py
"""
import datetime
import os
import sys

import numpy as np
import pandas as pd

from build_features_v4 import (compute_features, TICKERS, FEATS, GAP_DAYS,
                               DATA_END)

CANDLES_FILE = 'candles_full.csv'
V4_FILE = 'rl_features_v4.pkl'
OUT_FILE = 'rl_features_v5.pkl'

EXPECTED_ROWS = 3161
RV_FAST, RV_SLOW, CORR_WINDOW = 20, 60, 20
MIN_PERIODS = 20


class _Tee:
    """Дублирует вывод в консоль и в файл лога (правило AGENTS.md: логи в logs/)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False


def setup_log(prefix):
    os.makedirs('logs', exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join('logs', f'{prefix}_{stamp}.log')
    fh = open(path, 'w', encoding='utf-8', buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f'[LOG] Лог пишется в {path}')
    return path


def rebuild_legacy(raw):
    """Пересобрать 31 легаси-колонку ровно тем же путём, что build_features_v4.py."""
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
    print(f"Общих сессий (полная история): {len(common)}, "
          f"{common[0].date()} .. {common[-1].date()}")

    features = pd.DataFrame(index=common, dtype=np.float32)
    features['is_weekend'] = (common.dayofweek >= 5).astype(np.float32)
    for t in TICKERS:
        for f in FEATS:
            features[f'{t}_{f}'] = feats[t].reindex(common)[f]
    features = features.dropna()
    print(f"После dropna: {len(features)} строк (старт {features.index.min().date()})")

    # мост через остановку торгов: доходность = 0 (как в v4)
    ddays = features.index.to_series().diff().dt.days
    bridge = (ddays > GAP_DAYS).fillna(False).values
    ret_cols = [f'{t}_returns' for t in TICKERS]
    features.loc[bridge, ret_cols] = 0.0
    for d in features.index[bridge]:
        print(f"мост: {d.date()} (разрыв {int(ddays.loc[d])} дн.) "
              f"-> доходность обнулена (как в v4)")

    # обрезка по прежней границе данных (holdout не двигаем)
    features = features.loc[features.index <= pd.Timestamp(DATA_END)]

    # маска моста пересчитывается уже на обрезанном индексе (длина = строки v4)
    ddays = features.index.to_series().diff().dt.days
    bridge = (ddays > GAP_DAYS).fillna(False).values
    return features, common, bridge, ddays


def build_return_series(raw, common, features, bridge):
    """
    Объединённая доходностная серия: сырая история + перезапись значениями v4
    на индексе v4 (вариант A). Возвращает (combined, bridge_raw_diag).
    """
    ret_cols = [f'{t}_returns' for t in TICKERS]
    raw_common = pd.DataFrame(
        {f'{t}_returns': raw[raw['SECID'] == t].set_index('date').sort_index()
                             ['close'].pct_change().reindex(common)
         for t in TICKERS},
        index=common)[ret_cols]

    # ── диагностика: насколько крупным был выброс на мосту (для протокола) ──
    bridge_raw_diag = []
    for d in features.index[bridge]:
        vals = raw_common.loc[d, ret_cols].astype(float)
        print(f"\nДиагностика моста {d.date()} — СЫРАЯ доходность из candles_full.csv:")
        for c in ret_cols:
            print(f"  {c:<18}: {vals[c] * 100:+9.2f}%")
        print(f"  max|r| = {np.nanmax(np.abs(vals.values)) * 100:.2f}%  "
              f"(в v4 и в окнах используется 0.0)")
        bridge_raw_diag.append((d, vals.copy()))

    # ── перезапись строками v4 (байт-в-байт) ──
    combined = raw_common.copy()
    combined.loc[features.index, ret_cols] = features[ret_cols].values
    for c in ret_cols:
        if not np.array_equal(combined.loc[features.index, c].values,
                             features[c].values):
            raise SystemExit(f'ОШИБКА: доходностная серия {c} не совпала с v4 на индексе v4')
    print(f"\nДоходностная серия: {len(common)} строк полной истории, "
          f"на {len(features)} строках v4 совпадает с rl_features_v4.pkl байт-в-байт")
    return combined, bridge_raw_diag


def build_cov_features(combined, v4_index):
    """18 новых колонок: rv20 x6, rv60 x6, correq20 x6 (окна на полной истории)."""
    ret_cols = [f'{t}_returns' for t in TICKERS]
    mean_ret = combined[ret_cols].mean(axis=1)

    rv20 = pd.DataFrame(
        {f'{t}_rv20': combined[f'{t}_returns']
                     .rolling(RV_FAST, min_periods=MIN_PERIODS).std() * 100.0
         for t in TICKERS}, index=combined.index)
    rv60 = pd.DataFrame(
        {f'{t}_rv60': combined[f'{t}_returns']
                     .rolling(RV_SLOW, min_periods=MIN_PERIODS).std() * 100.0
         for t in TICKERS}, index=combined.index)
    corr = pd.DataFrame(
        {f'{t}_correq20': combined[f'{t}_returns']
                         .rolling(CORR_WINDOW, min_periods=MIN_PERIODS)
                         .corr(mean_ret)
         for t in TICKERS}, index=combined.index)

    # нарезка на индекс v4 ПОСЛЕ счёта окон
    rv20 = rv20.reindex(v4_index)
    rv60 = rv60.reindex(v4_index)
    corr = corr.reindex(v4_index)

    # мягкая граница [-1, 1] на случай float32-переполнения корреляции
    n_clipped = int(((corr > 1.0) | (corr < -1.0)).sum().sum())
    corr = corr.clip(-1.0, 1.0)
    if n_clipped:
        print(f"Корреляция: {n_clipped} значений подрезано в [-1, 1] (float32-погрешность)")

    fills = {
        'rv20': int(rv20.isna().sum().sum()),
        'rv60': int(rv60.isna().sum().sum()),
        'correq20': int(corr.isna().sum().sum()),
    }
    rv20 = rv20.fillna(0.0)
    rv60 = rv60.fillna(0.0)
    corr = corr.fillna(0.0)
    print(f"Нейтральные заполнения 0.0 (недобор наблюдений / std=0): {fills}")
    return rv20, rv60, corr, fills


def main():
    setup_log('build_features_v5')

    raw = pd.read_csv(CANDLES_FILE, parse_dates=['date'])

    # ── 1. 31 легаси-колонка ──
    features, common, bridge, ddays = rebuild_legacy(raw)
    ret_cols = [f'{t}_returns' for t in TICKERS]

    # ── 2. гейт: совпадение с rl_features_v4.pkl байт-в-байт ──
    v4 = pd.read_pickle(V4_FILE)
    if not features.index.equals(pd.DatetimeIndex(v4.index)):
        raise SystemExit('ОШИБКА: индекс пересобранных фичей не совпал с rl_features_v4.pkl')
    if list(features.columns) != list(v4.columns):
        raise SystemExit(f'ОШИБКА: порядок колонок не совпал с v4\n'
                         f'  получено: {list(features.columns)}\n'
                         f'  в v4:     {list(v4.columns)}')
    if not np.array_equal(features.values, v4.values):
        bad = [c for c in features.columns
               if not np.array_equal(features[c].values, v4[c].values)]
        raise SystemExit(f'ОШИБКА: легаси-колонки не совпали с v4: {bad}\n'
                         f'rl_features_v5.pkl НЕ записан')
    print(f"\nГЕЙТ ПРОЙДЕН: 31 легаси-колонка совпала с {V4_FILE} байт-в-байт "
          f"({features.shape[0]} строк x {features.shape[1]} колонок)")
    if len(features) != EXPECTED_ROWS:
        raise SystemExit(f'ОШИБКА: строк {len(features)}, ожидалось {EXPECTED_ROWS}')

    # ── 3. доходностная серия и 18 новых колонок ──
    combined, _ = build_return_series(raw, common, features, bridge)
    rv20, rv60, corr, fills = build_cov_features(combined, features.index)

    # ── 4. сборка финального фрейма ──
    # 31 легаси-колонка НЕ кастится: сохраняется dtype v4 ради байт-в-байт.
    out = pd.concat([features, rv20, rv60, corr], axis=1)
    exp_cols = (list(features.columns)
                + [f'{t}_rv20' for t in TICKERS]
                + [f'{t}_rv60' for t in TICKERS]
                + [f'{t}_correq20' for t in TICKERS])
    new_cols = exp_cols[31:]
    out[new_cols] = out[new_cols].astype(np.float32)   # только новые -> float32
    assert list(out.columns) == exp_cols, 'порядок колонок нарушен'
    assert out.shape == (EXPECTED_ROWS, 49), f'форма {out.shape} != (3161, 49)'
    assert not out.isna().any().any(), 'NaN в финальном фрейме'
    assert (out[[f'{t}_rv20' for t in TICKERS]].values >= 0).all(), 'rv20 < 0'
    assert (out[[f'{t}_rv60' for t in TICKERS]].values >= 0).all(), 'rv60 < 0'
    cc = out[[f'{t}_correq20' for t in TICKERS]].values
    assert cc.min() >= -1.0 and cc.max() <= 1.0, 'correq20 вне [-1, 1]'
    assert (out[new_cols].dtypes == np.float32).all(), 'dtype новых колонок не float32'
    # легаси-колонки обязаны сохранить dtype v4 (иначе не будет байт-в-байт)
    assert list(out.dtypes[:31]) == list(v4.dtypes), 'dtype легаси не совпал с v4'

    out.to_pickle(OUT_FILE)

    dtypes_desc = (f"легаси: {len(set(map(str, out.dtypes[:31])))} типа(ов) как в v4, "
                   f"новые: float32")
    print(f"\n{OUT_FILE}: {out.shape} (31 легаси + 18 новых); {dtypes_desc}")
    print(f"  индекс: {out.index[0].date()} .. {out.index[-1].date()}")
    print(f"  rv20    : min {out[[f'{t}_rv20' for t in TICKERS]].values.min():.4f}  "
          f"max {out[[f'{t}_rv20' for t in TICKERS]].values.max():.4f}")
    print(f"  rv60    : min {out[[f'{t}_rv60' for t in TICKERS]].values.min():.4f}  "
          f"max {out[[f'{t}_rv60' for t in TICKERS]].values.max():.4f}")
    print(f"  correq20: min {cc.min():.4f}  max {cc.max():.4f}")
    print(f"  заполнений 0.0: {fills}")
    print("\nГотово.")


if __name__ == '__main__':
    main()
