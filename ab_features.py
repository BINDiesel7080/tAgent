"""
A/B-эксперимент: базовые фичи vs расширенные рыночными (breadth20, xsec_ret5, corr20).

Гиперпараметры зафиксированы = trial 30 прогона v1.0:
  max_weight=0.35, lambda_hhi=0.0682, total_timesteps=150000,
  dd_alpha=0.17, gamma=0.995, n_epochs=10.

Обучение — в train-режиме среды v1.0 (случайная длина эпизода 30-200,
случайный стартовый портфель с вероятностью 0.5); оценка — детерминированная,
на test-окне каждого фолда. Сравнение парное: одни и те же фолды и сиды.

Запуск:
  python ab_features.py                      # полный прогон (~4 часа)
  python ab_features.py --folds 3            # только holdout (~1 час)
  python ab_features.py --timesteps 20000    # быстрая проверка пайплайна
"""

import argparse
import csv
import inspect
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from portfolio_env import PortfolioEnv

# ── Фиксированные параметры = trial 30 ──────────────────────────
FIXED = {
    'max_weight': 0.35,
    'lambda_hhi': 0.0682,
    'total_timesteps': 150_000,
    'dd_alpha': 0.17,
    'gamma': 0.995,
    'n_epochs': 10,
}
SEEDS = [42, 1042, 2042, 3042, 4042, 5042]


def make_env_kwargs(features_path, fold, train_mode, seed):
    """Аргументы среды; совместимо и с v1.0, и со старыми версиями."""
    kw = dict(
        features_path=features_path,
        prices_path='rl_prices.pkl',
        fold=fold,
        seed=seed,
        max_weight=FIXED['max_weight'],
        lambda_hhi=FIXED['lambda_hhi'],
        dd_alpha=FIXED['dd_alpha'],
    )
    sig = inspect.signature(PortfolioEnv.__init__)
    if train_mode:
        kw['eval_mode'] = False
        if 'episode_length_range' in sig.parameters:      # v1.0: случайная длина
            kw['episode_length_range'] = (30, 200)
        else:                                             # fallback: фикс. длина
            kw['episode_length'] = 200
        if 'random_start_prob' in sig.parameters:         # v1.0: случайный старт
            kw['random_start_prob'] = 0.5
    else:
        kw['eval_mode'] = True
        kw['eval_split'] = 'test'
    return kw


def run_one(features_path, fold, seed, timesteps, verbose=0):
    """Обучить и оценить: один вариант x фолд x сид."""
    train_env = Monitor(PortfolioEnv(**make_env_kwargs(features_path, fold, True, seed)))
    model = PPO(
        "MlpPolicy", train_env, verbose=verbose,
        learning_rate=1e-4, n_steps=2048, batch_size=256, clip_range=0.1,
        n_epochs=FIXED['n_epochs'], gamma=FIXED['gamma'],
        ent_coef=0.01, seed=seed, device="auto",
    )
    model.learn(total_timesteps=timesteps)

    eval_env = PortfolioEnv(**make_env_kwargs(features_path, fold, False, seed))
    obs, _ = eval_env.reset(seed=seed)
    values, eq_values, weights = [1.0], [1.0], []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = eval_env.step(action)
        values.append(info['portfolio_value'])
        eq_values.append(info['equal_value'])
        weights.append(info['weights'])
        if terminated or truncated:
            break

    w = np.array(weights)
    return {
        'ppo_ret': (values[-1] - 1) * 100,
        'eq_ret': (eq_values[-1] - 1) * 100,
        'cash': float(w[:, -1].mean()),
        'turnover': float(np.abs(np.diff(w, axis=0)).mean(axis=1).mean()),
        'values': np.array(values),
        'eq_values': np.array(eq_values),
    }


def main():
    ap = argparse.ArgumentParser(description='A/B: рыночные фичи vs базовые')
    ap.add_argument('--baseline', default='rl_features.pkl')
    ap.add_argument('--market', default='rl_features_v2.pkl')
    ap.add_argument('--variants', nargs='+', default=['baseline', 'market'],
                    choices=['baseline', 'market'])
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--timesteps', type=int, default=None)
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--out-csv', default='ab_results.csv')
    args = ap.parse_args()

    timesteps = args.timesteps if args.timesteps else FIXED['total_timesteps']
    variants = {v: (args.baseline if v == 'baseline' else args.market)
                for v in args.variants}

    print('=' * 60)
    print('  A/B: рыночные фичи (breadth20, xsec_ret5, corr20)')
    print(f"  Параметры: mw={FIXED['max_weight']} lhhi={FIXED['lambda_hhi']} "
          f"ts={timesteps:,} a_dd={FIXED['dd_alpha']} gamma={FIXED['gamma']} "
          f"ep={FIXED['n_epochs']}")
    print(f"  Фолды: {args.folds} | Сиды: {args.seeds} | Варианты: {args.variants}")
    print('=' * 60)

    # проверка версии среды
    sig = inspect.signature(PortfolioEnv.__init__)
    has_range = 'episode_length_range' in sig.parameters
    has_start = 'random_start_prob' in sig.parameters
    print(f"  Среда: случайная длина эпизода {'да' if has_range else 'НЕТ (fallback: фикс. 200)'}"
          f", случайный старт {'да' if has_start else 'НЕТ'}")
    if not (has_range and has_start):
        print('  ВНИМАНИЕ: portfolio_env.py без изменений v1.0 — сравнение '
              'с прогоном v1.0 будет нечестным!')

    n_runs = len(variants) * len(args.folds) * len(args.seeds)
    print(f"  Прогонов обучения: {n_runs} "
          f"(ориентир ~{n_runs * 10} мин при ts=150k)\n")

    results = {v: {} for v in variants}
    t0 = time.time()
    done = 0
    for vname, fpath in variants.items():
        for fold in args.folds:
            for seed in args.seeds:
                r = run_one(fpath, fold, seed, timesteps)
                results[vname][(fold, seed)] = r
                done += 1
                print(f"[{done:2d}/{n_runs}] {vname:8s} fold {fold} seed {seed}: "
                      f"PPO {r['ppo_ret']:+7.2f}% | Equal {r['eq_ret']:+7.2f}% | "
                      f"cash {r['cash']:.2f} | прошло {(time.time()-t0)/60:.0f} мин")

    # ── Детальные таблицы по вариантам ──
    for vname in variants:
        print(f"\n=== ДЕТАЛЬНО: {vname} ({variants[vname]}) ===")
        print(f"{'Fold':>4} | {'seed':>5} | {'PPO':>9} | {'Equal':>9} | {'Cash':>5}")
        for fold in args.folds:
            for seed in args.seeds:
                r = results[vname][(fold, seed)]
                hold = ' [HOLD]' if fold == 3 else ''
                print(f"{fold:>4} | {seed:>5} | {r['ppo_ret']:>+8.2f}% | "
                      f"{r['eq_ret']:>+8.2f}% | {r['cash']:>5.2f}{hold}")

    # ── Сводка по вариантам ──
    def agg(vname, folds):
        v = [results[vname][(f, s)]['ppo_ret'] for f in folds for s in args.seeds]
        e = [results[vname][(f, s)]['eq_ret'] for f in folds for s in args.seeds]
        return (np.mean(v), np.mean(e)) if v else (float('nan'), float('nan'))

    opt_folds = [f for f in args.folds if f in (0, 1, 2)]
    print('\n=== СВОДКА ПО ВАРИАНТАМ ===')
    print(f"{'Вариант':>9} | {'AVG(0-2) PPO':>13} | {'AVG(0-2) Eq':>12} | "
          f"{'HOLDOUT PPO':>12} | {'HOLDOUT Eq':>11}")
    for vname in variants:
        a_p, a_e = agg(vname, opt_folds) if opt_folds else (float('nan'), float('nan'))
        h_p, h_e = agg(vname, [3]) if 3 in args.folds else (float('nan'), float('nan'))
        print(f"{vname:>9} | {a_p:>+12.2f}% | {a_e:>+11.2f}% | "
              f"{h_p:>+11.2f}% | {h_e:>+10.2f}%")

    # ── Парные дельты ──
    if 'market' in results and 'baseline' in results:
        print('\n=== ПАРНЫЕ ДЕЛЬТЫ: market - baseline (одни сиды, одни фолды) ===')
        print('Fold  | ' + ' | '.join(f'seed {s}' for s in args.seeds) + ' | среднее')
        hold_deltas = None
        for fold in args.folds:
            deltas = [results['market'][(fold, s)]['ppo_ret'] -
                      results['baseline'][(fold, s)]['ppo_ret']
                      for s in args.seeds]
            label = f"{fold:>4}" + (' [HOLD]' if fold == 3 else '')
            print(f"{label} | " + ' | '.join(f'{d:+7.2f}%' for d in deltas) +
                  f" | {np.mean(deltas):+7.2f}%")
            if fold == 3:
                hold_deltas = deltas
        if hold_deltas is not None:
            wins = sum(d > 0 for d in hold_deltas)
            print(f"\n  HOLDOUT: средняя дельта {np.mean(hold_deltas):+.2f} п.п.; "
                  f"market лучше у {wins} из {len(hold_deltas)} сидов")

    # ── CSV ──
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['variant', 'fold', 'seed', 'ppo_ret', 'eq_ret', 'cash', 'turnover'])
        for vname in variants:
            for (fold, seed), r in results[vname].items():
                wr.writerow([vname, fold, seed, f"{r['ppo_ret']:.3f}",
                             f"{r['eq_ret']:.3f}", f"{r['cash']:.3f}",
                             f"{r['turnover']:.4f}"])
    print(f"\nРезультаты сохранены: {args.out_csv}")

    # ── График holdout ──
    if not args.no_plot and 3 in args.folds \
            and 'market' in results and 'baseline' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        eq_mean = np.mean([results['baseline'][(3, s)]['eq_values']
                           for s in args.seeds], axis=0)
        for vname, color in [('baseline', 'tab:blue'), ('market', 'tab:red')]:
            curves = np.array([results[vname][(3, s)]['values'] for s in args.seeds])
            for c in curves:
                ax.plot(c, color=color, alpha=0.25, linewidth=0.8)
            ax.plot(curves.mean(axis=0), color=color, linewidth=2.2,
                    label=f'{vname} (среднее по сидам)')
        ax.plot(eq_mean, color='gray', linestyle='--', linewidth=1.5,
                label='Equal Weight')
        ax.set_title('HOLDOUT (fold 3): базовые фичи vs рыночные фичи')
        ax.set_xlabel('День test-окна')
        ax.set_ylabel('Капитал')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('ab_holdout.png', dpi=150)
        print('График сохранён: ab_holdout.png')

    print('\nГотово!')


if __name__ == '__main__':
    main()
