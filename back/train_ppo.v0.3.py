"""
Обучение PPO-агента для портфеля Мосбирже — с train/test split.

Version: v0.2
Date:   2026-10-02
Changes from v0.1:
  - Train/test split (600 / 228 дней)
  - Оценка на train и test отдельно
  - Нормализация только по train (через env)
  - OMP fix
  - График: train и test на отдельных панелях

Запуск:
  python train_ppo.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')  # не открывать окно, сразу в файл
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from portfolio_env import PortfolioEnv

# ═══════════════════════════════════════════════
# КОНФИГ
# ═══════════════════════════════════════════════
VERSION = "v0.2"
TOTAL_TIMESTEPS = 500_000
EPISODE_LENGTH = 25
SPLIT_IDX = 600       # ~73% train, ~27% test
SEED = 42
MODEL_PATH = f"ppo_portfolio_{VERSION}"

# ═══════════════════════════════════════════════
# 1. СРЕДА (обучение — только train)
# ═══════════════════════════════════════════════
train_env = Monitor(PortfolioEnv(
    features_path='rl_features.pkl',
    prices_path='rl_prices.pkl',
    episode_length=EPISODE_LENGTH,
    eval_mode=False,
    split_idx=SPLIT_IDX,
    seed=SEED,
))

print(f"=== Обучение PPO {VERSION} ===")
print(f"  Timesteps:    {TOTAL_TIMESTEPS:,}")
print(f"  Split:        дни 0..{SPLIT_IDX-1} (train) / "
      f"{SPLIT_IDX}..{train_env.env.n_days-1} (test)")
print(f"  Ep. длина:    {EPISODE_LENGTH} дней")
print(f"  State:        {train_env.observation_space.shape}")
print(f"  Action:       {train_env.action_space.shape}")
print()

# ═══════════════════════════════════════════════
# 2. PPO
# ═══════════════════════════════════════════════
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
model.save(MODEL_PATH)
print(f"\nМодель сохранена: {MODEL_PATH}.zip")

# ═══════════════════════════════════════════════
# 3. ОЦЕНКА
# ═══════════════════════════════════════════════

def evaluate(model, eval_split, split_idx, seed):
    """Прогон модели по указанному сплиту. Возвращает values, weights, env."""
    env = PortfolioEnv(
        features_path='rl_features.pkl',
        prices_path='rl_prices.pkl',
        eval_mode=True,
        eval_split=eval_split,
        split_idx=split_idx,
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
    """Бенчмарки для участка [start, end).
    Возвращает eq_values, dict тикер→values, cash_values.
    """
    rets = features[[f"{t}_returns" for t in tickers]].values
    segment = rets[start + 1 : end + 1]   # +1: доходность next_step

    # Equal weight (ребалансировка каждый день)
    eq_daily = segment.mean(axis=1)
    eq_values = np.concatenate([[1.0], np.cumprod(1 + eq_daily)])

    # Каждый тикер buy & hold
    ticker_values = {}
    for i, t in enumerate(tickers):
        t_daily = segment[:, i]
        ticker_values[t] = np.concatenate([[1.0], np.cumprod(1 + t_daily)])

    # Cash
    cash_values = np.ones(len(eq_values))

    return eq_values, ticker_values, cash_values


features = train_env.env.features
tickers = train_env.env.tickers

# ── Train eval ──
print("\n=== Оценка на TRAIN ===")
ppo_train, w_train, env_train = evaluate(model, 'train', SPLIT_IDX, SEED)
print(f"  PPO:   {(ppo_train[-1] - 1) * 100:+.2f}%")

eq_train, tick_train, cash_train = benchmarks_on_split(
    features, tickers, 0, SPLIT_IDX - 1)
print(f"  Equal: {(eq_train[-1] - 1) * 100:+.2f}%")
for t in tickers:
    print(f"  {t:5s}: {(tick_train[t][-1] - 1) * 100:+.2f}%")
print(f"  Cash:  {(cash_train[-1] - 1) * 100:+.2f}%")

turnover_train = np.abs(np.diff(w_train, axis=0)).mean(axis=1).mean()
print(f"  Оборот/шаг: {turnover_train:.4f}")
print(f"  Доля кэша:  {w_train[:, -1].mean():.2f}")

# ── Test eval ──
print("\n=== Оценка на TEST ===")
ppo_test, w_test, env_test = evaluate(model, 'test', SPLIT_IDX, SEED)
print(f"  PPO:   {(ppo_test[-1] - 1) * 100:+.2f}%")

eq_test, tick_test, cash_test = benchmarks_on_split(
    features, tickers, SPLIT_IDX, env_test.n_days - 1)
print(f"  Equal: {(eq_test[-1] - 1) * 100:+.2f}%")
for t in tickers:
    print(f"  {t:5s}: {(tick_test[t][-1] - 1) * 100:+.2f}%")
print(f"  Cash:  {(cash_test[-1] - 1) * 100:+.2f}%")

turnover_test = np.abs(np.diff(w_test, axis=0)).mean(axis=1).mean()
print(f"  Оборот/шаг: {turnover_test:.4f}")
print(f"  Доля кэша:  {w_test[:, -1].mean():.2f}")

# Финальные веса (test)
print(f"\n  Финальные веса (test):")
for t, w in zip(tickers + ["CASH"], w_test[-1]):
    print(f"    {t:5s}: {w:.3f}")

# ═══════════════════════════════════════════════
# 4. ГРАФИК
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                         gridspec_kw={'height_ratios': [3, 1]})

# ── Train: стоимость портфеля ──
ax1 = axes[0, 0]
ax1.plot(ppo_train, label='PPO', linewidth=2, color='royalblue')
ax1.plot(eq_train, label='Equal', alpha=0.7, color='orange')
ax1.plot(cash_train, label='Cash', alpha=0.5, color='gray', linestyle='--')
for t in tickers:
    ax1.plot(tick_train[t], label=t, alpha=0.5, linewidth=1)
ax1.set_title(f'TRAIN — PPO {VERSION}')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Test: стоимость портфеля ──
ax2 = axes[0, 1]
ax2.plot(ppo_test, label='PPO', linewidth=2, color='royalblue')
ax2.plot(eq_test, label='Equal', alpha=0.7, color='orange')
ax2.plot(cash_test, label='Cash', alpha=0.5, color='gray', linestyle='--')
for t in tickers:
    ax2.plot(tick_test[t], label=t, alpha=0.5, linewidth=1)
ax2.set_title(f'TEST — PPO {VERSION}')
ax2.set_ylabel('Value')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Train: веса ──
ax3 = axes[1, 0]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#999999']
for i, name in enumerate(tickers + ["CASH"]):
    ax3.plot(w_train[:, i], label=name, alpha=0.7,
             color=colors[i], linewidth=1)
ax3.set_title('TRAIN — Weights')
ax3.set_ylabel('Weight')
ax3.legend(loc='upper right', fontsize=7, ncol=4)
ax3.grid(True, alpha=0.3)

# ── Test: веса ──
ax4 = axes[1, 1]
for i, name in enumerate(tickers + ["CASH"]):
    ax4.plot(w_test[:, i], label=name, alpha=0.7,
             color=colors[i], linewidth=1)
ax4.set_title('TEST — Weights')
ax4.set_ylabel('Weight')
ax4.set_xlabel('Step')
ax4.legend(loc='upper right', fontsize=7, ncol=4)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'ppo_{VERSION}_results.png', dpi=150)
print(f"\nГрафик сохранён: ppo_{VERSION}_results.png")
