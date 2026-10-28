"""
Обучение PPO-агента для портфеля Мосбирже.

Version: v0.1
Date:   2026-10-02
Env:    PortfolioEnv v0.1
Algo:   PPO (Stable Baselines3)

Запуск:
  pip install stable-baselines3 gymnasium matplotlib
  python train_ppo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from portfolio_env import PortfolioEnv

# ═══════════════════════════════════════════════
# КОНФИГ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════
VERSION = "v0.1"
TOTAL_TIMESTEPS = 200_000
EPISODE_LENGTH = 60
SEED = 42
LOG_DIR = "./ppo_logs/"
MODEL_PATH = f"ppo_portfolio_{VERSION}"

# ═══════════════════════════════════════════════
# 1. СРЕДА
# ═══════════════════════════════════════════════
train_env = Monitor(PortfolioEnv(
    features_path='rl_features.pkl',
    prices_path='rl_prices.pkl',
    episode_length=EPISODE_LENGTH,
    eval_mode=False,
    seed=SEED,
))

# ═══════════════════════════════════════════════
# 2. PPO
# ═══════════════════════════════════════════════
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,           # поощряем разведку
    seed=SEED,
    tensorboard_log=LOG_DIR,
)

print(f"=== Обучение PPO {VERSION} ===")
print(f"  Timesteps:    {TOTAL_TIMESTEPS:,}")
print(f"  Ep. длина:    {EPISODE_LENGTH} дней")
print(f"  Эпизодов:     ~{TOTAL_TIMESTEPS // EPISODE_LENGTH:,}")
print(f"  State:        {train_env.observation_space.shape}")
print(f"  Action:       {train_env.action_space.shape}")
print()

model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(MODEL_PATH)
print(f"\nМодель сохранена: {MODEL_PATH}.zip")

# ═══════════════════════════════════════════════
# 3. ОЦЕНКА — полный прогон по датасету
# ═══════════════════════════════════════════════
eval_env = PortfolioEnv(
    features_path='rl_features.pkl',
    prices_path='rl_prices.pkl',
    eval_mode=True,
    seed=SEED,
)

obs, info = eval_env.reset(seed=SEED)
ppo_values = [1.0]
ppo_weights = []

step_count = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    ppo_values.append(info['portfolio_value'])
    ppo_weights.append(info['weights'].copy())
    step_count += 1
    if terminated or truncated:
        break

print(f"\n=== Оценка ({step_count} шагов) ===")
print(f"  PPO:   {(ppo_values[-1] - 1) * 100:+.2f}%")

# ═══════════════════════════════════════════════
# 4. БЕНЧМАРКИ
# ═══════════════════════════════════════════════
tickers = eval_env.tickers
returns_all = eval_env.features[
    [f"{t}_returns" for t in tickers]
].values  # shape (828, 6)

# Равные веса (1/6 на тикер, ребалансировка каждый день)
eq_daily = returns_all[1:].mean(axis=1)           # доходность с дня 1
eq_values = np.concatenate([[1.0], np.cumprod(1 + eq_daily)])

# 100% SBER
sber_daily = returns_all[1:, 0]
sber_values = np.concatenate([[1.0], np.cumprod(1 + sber_daily)])

# 100% кэш
cash_values = np.ones(len(eq_values))

print(f"  Equal: {(eq_values[-1] - 1) * 100:+.2f}%")
print(f"  SBER:  {(sber_values[-1] - 1) * 100:+.2f}%")
print(f"  Cash:  {(cash_values[-1] - 1) * 100:+.2f}%")

# Средний оборот (насколько агент меняет веса за шаг)
weights_arr = np.array(ppo_weights)
turnover = np.abs(np.diff(weights_arr, axis=0)).mean(axis=1).mean()
print(f"\n  Средний оборот/шаг: {turnover:.4f}")
print(f"  Средняя доля кэша:  {weights_arr[:, -1].mean():.2f}")

# Финальные веса
print(f"\n  Финальные веса:")
for t, w in zip(tickers + ["CASH"], ppo_weights[-1]):
    print(f"    {t:5s}: {w:.3f}")

# ═══════════════════════════════════════════════
# 5. ГРАФИК
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={'height_ratios': [3, 1]})

# ── Стоимость портфеля ──
ax1 = axes[0]
ax1.plot(ppo_values, label='PPO Agent', linewidth=2, color='royalblue')
ax1.plot(eq_values, label='Equal Weight', alpha=0.7, color='orange')
ax1.plot(sber_values, label='100% SBER', alpha=0.7, color='green')
ax1.plot(cash_values, label='100% Cash', alpha=0.5,
         color='gray', linestyle='--')
ax1.set_title(f'Portfolio Value — PPO {VERSION} '
              f'({TOTAL_TIMESTEPS:,} timesteps)')
ax1.set_ylabel('Value')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Веса во времени ──
ax2 = axes[1]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
          '#ff7f00', '#a65628', '#999999']
for i, name in enumerate(tickers + ["CASH"]):
    ax2.plot(weights_arr[:, i], label=name, alpha=0.7,
             color=colors[i], linewidth=1)
ax2.set_title('Portfolio Weights Over Time')
ax2.set_ylabel('Weight')
ax2.set_xlabel('Step')
ax2.legend(loc='upper right', fontsize=8, ncol=4)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'ppo_{VERSION}_results.png', dpi=150)
print(f"\nГрафик сохранён: ppo_{VERSION}_results.png")
plt.show()
