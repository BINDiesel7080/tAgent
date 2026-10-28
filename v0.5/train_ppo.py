"""
Обучение PPO-агента для портфеля Мосбирже — Walk-forward (4 фолда).

Version: v0.4
Date:   2026-10-02
Changes from v0.3:
  - Walk-forward: 4 фолда, каждый — отдельное обучение с нуля
  - Комиссии в среде (0.04%)
  - Оценка на train и test каждого фолда
  - Сводная таблица по всем фолдам
  - График: 4 фолда × (train/test)

Запуск:
  python train_ppo.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from portfolio_env import PortfolioEnv

# ═══════════════════════════════════════════════
# КОНФИГ
# ═══════════════════════════════════════════════
VERSION = "v0.5"
TOTAL_TIMESTEPS = 500_000
EPISODE_LENGTH = 30
SEED = 42
N_FOLDS = 4

# ═══════════════════════════════════════════════
# ФУНКЦИИ
# ═══════════════════════════════════════════════

def evaluate(model, fold, eval_split, seed):
    """Прогон модели по указанному сплиту фолда."""
    env = PortfolioEnv(
        features_path='rl_features.pkl',
        prices_path='rl_prices.pkl',
        eval_mode=True,
        eval_split=eval_split,
        fold=fold,
        seed=seed,
    )
    obs, info = env.reset(seed=seed)
    values = [1.0]
    weights_history = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(info['portfolio_value'])
        weights_history.append(info['weights'].copy())
        if terminated or truncated:
            break

    return values, np.array(weights_history), env


def benchmarks_on_split(features, tickers, start, end):
    """Бенчмарки для участка [start, end)."""
    rets = features[[f"{t}_returns" for t in tickers]].values
    segment = rets[start + 1 : end + 1]

    eq_daily = segment.mean(axis=1)
    eq_values = np.concatenate([[1.0], np.cumprod(1 + eq_daily)])

    ticker_values = {}
    for i, t in enumerate(tickers):
        t_daily = segment[:, i]
        ticker_values[t] = np.concatenate([[1.0], np.cumprod(1 + t_daily)])

    cash_values = np.ones(len(eq_values))
    return eq_values, ticker_values, cash_values


# ═══════════════════════════════════════════════
# ЦИКЛ ПО ФОЛДАМ
# ═══════════════════════════════════════════════

all_results = []

for fold in range(N_FOLDS):
    ts, te, vs, ve = PortfolioEnv.FOLDS[fold]

    print(f"\n{'='*60}")
    print(f"  FOLD {fold}: train [{ts}:{te}] ({te-ts} дн.)  "
          f"test [{vs}:{ve}] ({ve-vs} дн.)")
    print(f"{'='*60}")

    # ── Среда обучения ──
    train_env = Monitor(PortfolioEnv(
        features_path='rl_features.pkl',
        prices_path='rl_prices.pkl',
        episode_length=EPISODE_LENGTH,
        eval_mode=False,
        fold=fold,
        seed=SEED,
    ))

    print(f"  Timesteps:  {TOTAL_TIMESTEPS:,}")
    print(f"  Ep. длина:  {EPISODE_LENGTH} дней")
    print()

    # ── PPO ──
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        clip_range=0.1,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        seed=SEED,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = f"ppo_portfolio_{VERSION}_fold{fold}"
    model.save(model_path)
    print(f"\nМодель сохранена: {model_path}.zip")


    # ── Оценка ──
    features = train_env.env.features
    tickers = train_env.env.tickers

    # Train
    ppo_train, w_train, env_train = evaluate(model, fold, 'train', SEED)
    eq_train, tick_train, cash_train = benchmarks_on_split(
        features, tickers, ts, te - 1)

    # Test
    ppo_test, w_test, env_test = evaluate(model, fold, 'test', SEED)
    eq_test, tick_test, cash_test = benchmarks_on_split(
        features, tickers, vs, ve - 1)

    # Метрики
    ppo_train_ret = (ppo_train[-1] - 1) * 100
    ppo_test_ret = (ppo_test[-1] - 1) * 100
    eq_test_ret = (eq_test[-1] - 1) * 100
    turnover_test = np.abs(np.diff(w_test, axis=0)).mean(axis=1).mean()
    cash_share_test = w_test[:, -1].mean()

    print(f"\n--- Fold {fold} результаты ---")
    print(f"  TRAIN PPO:    {ppo_train_ret:+.2f}%")
    print(f"  TEST  PPO:    {ppo_test_ret:+.2f}%")
    print(f"  TEST  Equal:  {eq_test_ret:+.2f}%")
    for t in tickers:
        print(f"  TEST  {t:5s}: {(tick_test[t][-1] - 1) * 100:+.2f}%")
    print(f"  TEST  Cash:   {(cash_test[-1] - 1) * 100:+.2f}%")
    print(f"  Оборот/шаг:   {turnover_test:.4f}")
    print(f"  Доля кэша:    {cash_share_test:.2f}")
    print(f"  Финальные веса:")
    for t, w in zip(tickers + ["CASH"], w_test[-1]):
        print(f"    {t:5s}: {w:.3f}")

    all_results.append({
        'fold': fold,
        'train_start': ts, 'train_end': te,
        'test_start': vs, 'test_end': ve,
        'ppo_train': ppo_train, 'ppo_test': ppo_test,
        'w_train': w_train, 'w_test': w_test,
        'eq_train': eq_train, 'eq_test': eq_test,
        'tick_train': tick_train, 'tick_test': tick_test,
        'cash_train': cash_train, 'cash_test': cash_test,
        'ppo_train_ret': ppo_train_ret,
        'ppo_test_ret': ppo_test_ret,
        'eq_test_ret': eq_test_ret,
        'turnover_test': turnover_test,
        'cash_share_test': cash_share_test,
        'final_weights': w_test[-1],
    })

# ═══════════════════════════════════════════════
# СВОДНАЯ ТАБЛИЦА
# ═══════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  СВОДКА ПО ВСЕМ ФОЛДАМ")
print(f"{'='*60}")
print(f"{'Fold':>4} | {'Train PPO':>10} | {'Test PPO':>10} | "
      f"{'Test Equal':>10} | {'Оборот':>8} | {'Кэш':>5}")
print(f"{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*5}")

for r in all_results:
    print(f"{r['fold']:>4} | {r['ppo_train_ret']:>+9.2f}% | "
          f"{r['ppo_test_ret']:>+9.2f}% | {r['eq_test_ret']:>+9.2f}% | "
          f"{r['turnover_test']:>8.4f} | {r['cash_share_test']:>5.2f}")

avg_train = np.mean([r['ppo_train_ret'] for r in all_results])
avg_test = np.mean([r['ppo_test_ret'] for r in all_results])
avg_eq = np.mean([r['eq_test_ret'] for r in all_results])
print(f"{'AVG':>4} | {avg_train:>+9.2f}% | {avg_test:>+9.2f}% | "
      f"{avg_eq:>+9.2f}% |")

# ═══════════════════════════════════════════════
# ГРАФИК
# ═══════════════════════════════════════════════

fig, axes = plt.subplots(8, 2, figsize=(16, 32),
                         gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1, 3, 1]})

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#999999']
tickers_all = tickers + ["CASH"]

for fold_idx, r in enumerate(all_results):
    # ── Value: Train ──
    ax_val_t = axes[fold_idx * 2, 0]
    ax_val_t.plot(r['ppo_train'], label='PPO', linewidth=2, color='royalblue')
    ax_val_t.plot(r['eq_train'], label='Equal', alpha=0.7, color='orange')
    ax_val_t.plot(r['cash_train'], label='Cash', alpha=0.5,
                  color='gray', linestyle='--')
    for t in tickers:
        ax_val_t.plot(r['tick_train'][t], label=t, alpha=0.5, linewidth=1)
    ax_val_t.set_title(f"Fold {r['fold']} TRAIN [{r['train_start']}:{r['train_end']}]")
    ax_val_t.legend(loc='upper left', fontsize=7)
    ax_val_t.grid(True, alpha=0.3)

    # ── Value: Test ──
    ax_val_v = axes[fold_idx * 2, 1]
    ax_val_v.plot(r['ppo_test'], label='PPO', linewidth=2, color='royalblue')
    ax_val_v.plot(r['eq_test'], label='Equal', alpha=0.7, color='orange')
    ax_val_v.plot(r['cash_test'], label='Cash', alpha=0.5,
                  color='gray', linestyle='--')
    for t in tickers:
        ax_val_v.plot(r['tick_test'][t], label=t, alpha=0.5, linewidth=1)
    ax_val_v.set_title(f"Fold {r['fold']} TEST [{r['test_start']}:{r['test_end']}] "
                       f"— PPO: {r['ppo_test_ret']:+.1f}%")
    ax_val_v.legend(loc='upper left', fontsize=7)
    ax_val_v.grid(True, alpha=0.3)

    # ── Weights: Train ──
    ax_w_t = axes[fold_idx * 2 + 1, 0]
    for i, name in enumerate(tickers_all):
        ax_w_t.plot(r['w_train'][:, i], label=name, alpha=0.7,
                    color=colors[i], linewidth=1)
    ax_w_t.set_title(f'Fold {r["fold"]} TRAIN — Weights')
    ax_w_t.legend(loc='upper right', fontsize=6, ncol=4)
    ax_w_t.grid(True, alpha=0.3)

    # ── Weights: Test ──
    ax_w_v = axes[fold_idx * 2 + 1, 1]
    for i, name in enumerate(tickers_all):
        ax_w_v.plot(r['w_test'][:, i], label=name, alpha=0.7,
                    color=colors[i], linewidth=1)
    ax_w_v.set_title(f'Fold {r["fold"]} TEST — Weights')
    ax_w_v.legend(loc='upper right', fontsize=6, ncol=4)
    ax_w_v.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'ppo_{VERSION}_results.png', dpi=150)
print(f"\nГрафик сохранён: ppo_{VERSION}_results.png")
print(f"\nГотово! Модели: ppo_portfolio_{VERSION}_fold0..{N_FOLDS-1}.zip")
