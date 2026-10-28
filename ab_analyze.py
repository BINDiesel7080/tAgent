#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ab_analyze.py — разбор результатов A/B-эксперимента фичей.

Читает ab_results.csv, который пишет ab_features.py, и считает:
  1) парные дельты (market - baseline) по каждой паре (фолд, сид);
  2) агрегаты: mean/median/std по фолдам, по opt-фолдам (0-2) и holdout (3);
  3) sign-test: в скольких парах дельта > 0, точный биномиальный p-value
     (без scipy, через math.comb);
  4) та же статистика для excess над Equal (PPO_ret - Eq_ret) — важно:
     победа над рынком и победа над baseline — разные вопросы;
  5) готовый вердикт по заранее фиксированным критериям.

Критерии решения (фиксируем ДО просмотра результатов, чтобы не подгонять):
  - verdict WIN   : средняя дельта holdout > +1 п.п. И sign-test p < 0.20;
  - verdict TIE   : средняя дельта holdout в [-1, +1] п.п.;
  - verdict LOSE  : средняя дельта holdout < -1 п.п.
  Дополнительно печатается решающее правило по opt-фолдам (там обучались,
  там же ловим переобучение: если opt >> holdout — сигнал тревоги).

Запуск:
  python ab_analyze.py                     # читает ab_results.csv
  python ab_analyze.py --csv logs/ab_results.csv
  python ab_analyze.py --csv ab_results.csv --a market --b baseline
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

HOLDOUT_FOLD = 3
OPT_FOLDS = (0, 1, 2)


def sign_test_p(deltas):
    """Двусторонний точный биномиальный p: вероятность столь же неравномерного
    расклада wins/losses при p=0.5 (нули учитываются пополам — консервативно)."""
    d = np.asarray(deltas, dtype=float)
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    n = wins + losses
    if n == 0:
        return 1.0, wins, losses
    k = max(wins, losses)
    # точная биномиальная двусторонняя: сумма вероятностей хвостов >= наблюдаемой
    def binom(c, n):
        return math.comb(n, c) * 0.5 ** n
    tail = sum(binom(c, n) for c in range(k, n + 1))
    p = min(1.0, 2 * tail)
    return p, wins, losses


def block(name, deltas, excess=None):
    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        print(f"  {name}: нет данных")
        return
    p, wins, losses = sign_test_p(d)
    print(f"  {name}: n={len(d)}  mean={d.mean():+.2f} п.п.  median={np.median(d):+.2f}  "
          f"std={d.std(ddof=1) if len(d) > 1 else float('nan'):.2f}")
    print(f"      пар лучше/хуже: {wins}/{losses}  sign-test p={p:.3f}")
    if excess is not None:
        e = np.asarray(excess, dtype=float)
        pe, w2, l2 = sign_test_p(e)
        print(f"      excess над Equal (PPO-Eq): mean={e.mean():+.2f} п.п., "
              f"лучше рынка у {w2}/{w2+l2} пар (p={pe:.3f})")


def verdict(mean_hold, p_hold):
    if mean_hold > 1.0 and p_hold < 0.20:
        return "WIN"
    if mean_hold < -1.0:
        return "LOSE"
    return "TIE"


def main():
    ap = argparse.ArgumentParser(description='Разбор A/B: парные дельты и вердикт')
    ap.add_argument('--csv', default='ab_results.csv')
    ap.add_argument('--a', default='market', help='вариант A (новые фичи)')
    ap.add_argument('--b', default='baseline', help='вариант B (база)')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"Файл не найден: {args.csv}")
        print("Укажите путь: python ab_analyze.py --csv logs/ab_results.csv")
        return

    df = pd.read_csv(args.csv)
    need = {'variant', 'fold', 'seed', 'ppo_ret'}
    missing = need - set(df.columns)
    if missing:
        print(f"В CSV нет колонок: {missing}. Это точно ab_results.csv от ab_features.py?")
        return

    variants = sorted(df['variant'].unique())
    print("=" * 64)
    print(f"  A/B РАЗБОР: {args.csv}")
    print(f"  Варианты: {variants} | строк: {len(df)}")
    print("=" * 64)

    if args.a not in variants or args.b not in variants:
        print(f"В CSV нет варианта {'A' if args.a not in variants else 'B'} "
              f"({args.a if args.a not in variants else args.b}). "
              f"Доступны: {variants}. Задайте вручную: --a/--b")
        return

    piv = df.pivot_table(index=['fold', 'seed'], columns='variant',
                         values='ppo_ret', aggfunc='first')
    piv_eq = df.pivot_table(index=['fold', 'seed'], columns='variant',
                            values='eq_ret', aggfunc='first')
    pairs = piv.dropna()
    if len(pairs) < len(piv):
        print(f"\nВНИМАНИЕ: {len(piv) - len(pairs)} пар неполные — сравнение только "
              f"по полным ({len(pairs)}). Прогон завершился не до конца?")
    if len(pairs) == 0:
        print("Нет полных пар — разбирать нечего.")
        return

    delta = pairs[args.a] - pairs[args.b]
    delta_eq = piv_eq.loc[delta.index, args.a] - piv_eq.loc[delta.index, args.b]

    folds = sorted({f for f, _ in delta.index})
    print(f"\n--- Парные дельты {args.a} - {args.b} (п.п. доходности) ---")
    for f in folds:
        mask = [f == ff for ff, _ in delta.index]
        tag = " [HOLD]" if f == HOLDOUT_FOLD else ""
        block(f"fold {f}{tag}", delta[mask], delta_eq[mask])

    opt_mask = [ff in OPT_FOLDS for ff, _ in delta.index]
    hold_mask = [ff == HOLDOUT_FOLD for ff, _ in delta.index]
    print("\n--- Агрегаты ---")
    block("OPT-фолды (0-2, где обучались)", delta[opt_mask], delta_eq[opt_mask])
    block("HOLDOUT (fold 3)", delta[hold_mask], delta_eq[hold_mask])
    block("Все фолды", delta, delta_eq)

    # расхождение opt vs holdout — маркер переобучения на фичах
    if any(opt_mask) and any(hold_mask):
        m_opt, m_hold = delta[opt_mask].mean(), delta[hold_mask].mean()
        gap = m_opt - m_hold
        print(f"\n  Разрыв opt-holdout: {gap:+.2f} п.п. "
              + ("(opt заметно лучше holdout — возможное переобучение под фолды)"
                 if gap > 1.5 else "(расхождения нет — дельта устойчива)"))

    m_hold = delta[hold_mask].mean() if any(hold_mask) else float('nan')
    p_hold, _, _ = sign_test_p(delta[hold_mask]) if any(hold_mask) else (float('nan'), 0, 0)
    v = verdict(m_hold, p_hold) if any(hold_mask) else "?"
    print("\n" + "=" * 64)
    print(f"  ВЕРДИКТ: {v}")
    print(f"  (критерий: holdout mean {m_hold:+.2f} п.п., sign-test p={p_hold:.3f}; "
          f"WIN = >+1 п.п. и p<0.20, LOSE = <-1 п.п., иначе TIE)")
    print("=" * 64)
    if v == "WIN":
        print(f"  {args.a}: переходим на неё в train_ppo и ищем деньги дальше.")
        print("  Не забыть: в evaluate() train_ppo путь зашит 'rl_features.pkl' — править.")
    elif v == "TIE":
        print("  Ветка фичей закрыта: скаляры-снимки уже выжимают почти всё.")
        print("  Следующий шаг — траектория/reward, не новые колонки.")
    elif v == "LOSE":
        print(f"  {args.a} хуже базы — новые фичи добавляют шум. Откат к v2 и")
        print("  разбор: какая из 30 колонок тянет вниз (LOPO, если захотим добить).")


if __name__ == '__main__':
    main()
