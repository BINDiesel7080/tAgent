#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_v5.py — самопроверка датасета v1.2 (rl_features_v5.pkl).

Version: v5.0
Date:    2026-10-28

Проверки (любая ошибка -> ненулевой код возврата):
  1. Размер и индекс: 3161 строка, DatetimeIndex совпадает с v4 до элемента.
  2. Первые 31 колонка: np.array_equal с v4 (байт-в-байт).
  3. Новые 18 колонок: нет NaN; correq20 в [-1, 1]; rv20/rv60 >= 0.
  4. Границы фолдов: PortfolioEnv.build_folds даёт те же даты, что для v4;
     fold 5 test = 2026-06-16..2026-10-02.
  5. PortfolioEnv(features_path='rl_features_v5.pkl') создаётся, obs.shape == (56,).
  6. Прогон 3 шагов env с obs из v5 — не падает, info корректен.

ВАЖНО: fold 5 (holdout) здесь только ЧИТАЕТСЯ как границы индексов.
Среда создаётся с fold=0, на holdout ничего не запускается.

Запуск: python check_v5.py
"""
import sys

import numpy as np
import pandas as pd

from build_features_v5 import TICKERS, setup_log
from portfolio_env import PortfolioEnv

V4_FILE = 'rl_features_v4.pkl'
V5_FILE = 'rl_features_v5.pkl'
EXPECTED_ROWS = 3161
EXPECTED_COLS = 49
EXPECTED_OBS_DIM = 56          # 31 легаси + 18 новых + 7 (веса + кэш)
HOLDOUT_TEST_START = '2026-06-16'
HOLDOUT_TEST_END = '2026-10-02'
N_FOLDS = 6

_failures = []


def check(name, ok, detail=''):
    """Зафиксировать результат проверки и напечатать строку отчёта."""
    status = 'OK  ' if ok else 'FAIL'
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ''))
    if not ok:
        _failures.append(name)
    return ok


def main():
    setup_log('check_v5')
    print('=' * 70)
    print('  САМОПРОВЕРКА v1.2 — rl_features_v5.pkl')
    print('=' * 70)

    v4 = pd.read_pickle(V4_FILE)
    v5 = pd.read_pickle(V5_FILE)
    print(f"v4: {v4.shape}, v5: {v5.shape}\n")

    # ── 1. Размер и индекс ────────────────────────────────────────────
    print('--- 1. Размер и индекс ---')
    check('строк 3161', len(v5) == EXPECTED_ROWS, f'{len(v5)} строк')
    check('колонок 49', v5.shape[1] == EXPECTED_COLS, f'{v5.shape[1]} колонок')
    check('DatetimeIndex', isinstance(v5.index, pd.DatetimeIndex), type(v5.index).__name__)
    check('индекс совпадает с v4 до элемента',
          v5.index.equals(pd.DatetimeIndex(v4.index)),
          f'{v5.index[0].date()} .. {v5.index[-1].date()}')

    # ── 2. Первые 31 колонка байт-в-байт ──────────────────────────────
    print('\n--- 2. Легаси 31 колонка ---')
    check('порядок легаси-колонок = v4',
          list(v5.columns[:31]) == list(v4.columns),
          'совпадает' if list(v5.columns[:31]) == list(v4.columns) else 'РАСХОЖДЕНИЕ')
    check('dtype легаси-колонок = v4',
          list(v5.dtypes[:31]) == list(v4.dtypes),
          f"v4: {sorted({str(d) for d in v4.dtypes})}, "
          f"v5: {sorted({str(d) for d in v5.dtypes[:31]})}")
    bad = [c for c in v4.columns
           if not np.array_equal(v5[c].values, v4[c].values)]
    check('np.array_equal с v4 по 31 колонке', not bad,
          'байт-в-байт' if not bad else f'расхождения: {bad[:5]}')
    if bad:
        worst = max(float(np.nanmax(np.abs(v5[c].values.astype(np.float64)
                                          - v4[c].values.astype(np.float64))))
                    for c in bad)
        check('расхождение на уровне округления float', worst < 1e-5,
              f'max|diff| = {worst:.3e}')

    # ── 3. Новые 18 колонок ───────────────────────────────────────────
    print('\n--- 3. Новые 18 колонок ---')
    new_cols = list(v5.columns[31:])
    exp_new = ([f'{t}_rv20' for t in TICKERS]
               + [f'{t}_rv60' for t in TICKERS]
               + [f'{t}_correq20' for t in TICKERS])
    check('состав и порядок 18 новых колонок', new_cols == exp_new,
          f'{len(new_cols)} колонок')
    check('dtype новых колонок float32',
          bool((v5[new_cols].dtypes == np.float32).all()),
          str(sorted({str(d) for d in v5[new_cols].dtypes})))
    check('нет NaN', not v5[new_cols].isna().any().any(),
          f'{int(v5[new_cols].isna().sum().sum())} NaN')
    check('нет inf', bool(np.isfinite(v5[new_cols].values).all()),
          f'{int((~np.isfinite(v5[new_cols].values)).sum())} не-конечных')

    rv20 = v5[[f'{t}_rv20' for t in TICKERS]].values
    rv60 = v5[[f'{t}_rv60' for t in TICKERS]].values
    cc = v5[[f'{t}_correq20' for t in TICKERS]].values
    check('rv20 >= 0', bool((rv20 >= 0).all()), f'min {rv20.min():.4f} max {rv20.max():.4f}')
    check('rv60 >= 0', bool((rv60 >= 0).all()), f'min {rv60.min():.4f} max {rv60.max():.4f}')
    check('correq20 в [-1, 1]', bool((cc >= -1).all() and (cc <= 1).all()),
          f'min {cc.min():.4f} max {cc.max():.4f}')

    # ── 4. Границы фолдов ─────────────────────────────────────────────
    print('\n--- 4. Границы фолдов (только чтение индексов, env не запускается) ---')
    folds_v4 = PortfolioEnv.build_folds(pd.DatetimeIndex(v4.index))
    folds_v5 = PortfolioEnv.build_folds(pd.DatetimeIndex(v5.index))
    check(f'фолдов {N_FOLDS}', len(folds_v5) == N_FOLDS, f'{len(folds_v5)} фолдов')
    check('границы фолдов v5 == v4', folds_v5 == folds_v4,
          'идентичны' if folds_v5 == folds_v4 else 'РАСХОЖДЕНИЕ')

    idx = v5.index
    print(f"\n{'fold':>5} | {'train':^25} | {'test':^25} | holdout")
    print('-' * 88)
    for i, (ts, te, vs, ve) in enumerate(folds_v5):
        mark = 'HOLDOUT' if i == N_FOLDS - 1 else ''
        print(f"{i:>5} | {idx[ts].date()} .. {idx[te - 1].date()} | "
              f"{idx[vs].date()} .. {idx[ve - 1].date()} | {mark}")

    ts_h, te_h, vs_h, ve_h = folds_v5[-1]
    check('fold 5 test = 2026-06-16..2026-10-02',
          idx[vs_h].date().isoformat() == HOLDOUT_TEST_START
          and idx[ve_h - 1].date().isoformat() == HOLDOUT_TEST_END,
          f'{idx[vs_h].date()} .. {idx[ve_h - 1].date()}')
    check('fold 5 test = последние 101 строка',
          vs_h == len(idx) - PortfolioEnv.HOLDOUT_ROWS and ve_h == len(idx),
          f'[{vs_h}:{ve_h}] из {len(idx)}')

    # ── 5. Среда на v5, obs_dim ───────────────────────────────────────
    print('\n--- 5. PortfolioEnv на v5 ---')
    env = PortfolioEnv(features_path=V5_FILE, fold=0, seed=42)
    check('env создаётся', True, f'fold 0, n_days={env.n_days}')
    check('observation_space == (56,)',
          env.observation_space.shape == (EXPECTED_OBS_DIM,),
          str(env.observation_space.shape))
    check('число фич в датасете = 49', env.features.shape[1] == EXPECTED_COLS,
          f'{env.features.shape[1]}')

    obs, info = env.reset(seed=42)
    check('obs.shape == (56,)', obs.shape == (EXPECTED_OBS_DIM,), str(obs.shape))
    check('obs dtype float32', obs.dtype == np.float32, str(obs.dtype))
    check('obs без NaN/inf', bool(np.isfinite(obs).all()),
          f'{int((~np.isfinite(obs)).sum())} не-конечных')

    # ── 6. Прогон 3 шагов ─────────────────────────────────────────────
    print('\n--- 6. Прогон 3 шагов env ---')
    info_keys = {'step', 'date', 'weights', 'portfolio_ret', 'portfolio_value',
                 'turnover', 'commission_cost', 'equal_value', 'hhi_risk'}
    ok_steps = True
    for k in range(1, 4):
        action = np.zeros(env.n_actions, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        w = info['weights']
        ok = (obs.shape == (EXPECTED_OBS_DIM,)
              and np.isfinite(obs).all()
              and np.isfinite(reward)
              and info_keys <= set(info)
              and abs(float(w.sum()) - 1.0) < 1e-4
              and np.isfinite(info['portfolio_value'])
              and 0.0 <= info['hhi_risk'] <= 1.0)
        ok_steps = ok_steps and ok
        print(f"  шаг {k}: date={info['date']} reward={reward:+.6f} "
              f"PV={info['portfolio_value']:.6f} EQ={info['equal_value']:.6f} "
              f"HHI={info['hhi_risk']:.4f} turnover={info['turnover']:.4f} "
              f"sum_w={float(w.sum()):.6f} obs{obs.shape} -> {'OK' if ok else 'FAIL'}")
    check('3 шага без ошибок, info корректен', ok_steps)

    # ── Итог ──────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    if _failures:
        print(f'  ПРОВАЛЕНО ПРОВЕРОК: {len(_failures)}')
        for f in _failures:
            print(f'    - {f}')
        print('=' * 70)
        return 1
    print('  ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ')
    print('=' * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
