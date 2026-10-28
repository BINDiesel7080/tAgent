#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_v1_2.py — одноразовый смоук-тест итерации v1.2 (ковариационные фичи).

Version: v1.2
Date:    2026-10-28

Что делает:
  - обучает PPO на fold 0 с ДЕФОЛТНЫМИ параметрами среды (max_weight,
    lambda_hhi, dd_alpha, gamma, n_epochs — как в PortfolioEnv/PPO по умолчанию);
  - total_timesteps = 2000, seed = 42, наблюдения из rl_features_v5.pkl;
  - проверяет: обучение стартует и не падает, PPO принимает obs (56,),
    эпизод проходит 100 шагов;
  - печатает obs_dim, первые 10 значений obs, финальные веса.

Проверки:
  - obs_dim == 56 (49 фич + 7 весов);
  - model.observation_space.shape == (56,);
  - 100 шагов среды подряд без исключений (с учётом truncation);
  - info корректен на каждом шаге.

Чего НЕ делает:
  - не создаёт и не трогает базу Optuna (импорт optuna отсутствует);
  - не обучается на fold 5 / holdout (только fold 0);
  - не сохраняет модели и не пишет в артефакты v1.1.

Запуск: python smoke_v1_2.py
"""
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from build_features_v5 import setup_log
from portfolio_env import PortfolioEnv

FEATURES_FILE = 'rl_features_v5.pkl'
PRICES_FILE = 'rl_prices_v4.pkl'
FOLD = 0                 # только opt-фолд, holdout (fold 5) не трогаем
SEED = 42
TOTAL_TIMESTEPS = 2000
N_CHECK_STEPS = 100
EXPECTED_OBS_DIM = 56


def main():
    setup_log('smoke_v1_2')
    ok = True

    print('=' * 70)
    print('  СМОУК-ТЕСТ v1.2 — PPO на fold 0, obs из rl_features_v5.pkl')
    print('=' * 70)

    # ── 1. Среда обучения с дефолтными параметрами ──
    train_env = Monitor(PortfolioEnv(
        features_path=FEATURES_FILE,
        prices_path=PRICES_FILE,
        fold=FOLD,
        seed=SEED,
        eval_mode=False,
    ))
    obs_dim = int(train_env.observation_space.shape[0])
    print(f'\nobs_dim (observation_space) = {obs_dim}')
    print(f'n_assets = {train_env.unwrapped.n_assets}, '
          f'n_actions = {train_env.unwrapped.n_actions}')
    print(f'n_days = {train_env.unwrapped.n_days}, фич в датасете = '
          f'{train_env.unwrapped.features.shape[1]}')
    if obs_dim != EXPECTED_OBS_DIM:
        print(f'ПРОВАЛ: obs_dim {obs_dim} != {EXPECTED_OBS_DIM}')
        ok = False

    # ── 2. PPO с дефолтными гиперпараметрами ──
    model = PPO("MlpPolicy", train_env, seed=SEED, device="auto", verbose=0)
    print(f'\nPPO observation_space = {model.observation_space.shape}')
    if tuple(model.observation_space.shape) != (EXPECTED_OBS_DIM,):
        print('ПРОВАЛ: PPO не принял obs (56,)')
        ok = False

    # ── 3. Обучение ──
    print(f'\nОбучение: total_timesteps={TOTAL_TIMESTEPS} (n_steps=2048 -> '
          f'один полный роллаут), seed={SEED}')
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        print('Обучение завершилось без ошибок')
    except Exception as e:                        # noqa: BLE001
        print(f'ПРОВАЛ: обучение упало: {type(e).__name__}: {e}')
        return 1

    # ── 4. 100 шагов среды под управлением обученной модели ──
    print(f'\nПроверка {N_CHECK_STEPS} шагов среды:')
    env = PortfolioEnv(
        features_path=FEATURES_FILE,
        prices_path=PRICES_FILE,
        fold=FOLD,
        seed=SEED,
        eval_mode=False,
    )
    obs, reset_info = env.reset(seed=SEED)
    print(f'  reset: start_date={reset_info["start_date"]}, '
          f'ep_length={reset_info["ep_length"]}, obs{obs.shape}')

    steps, resets = 0, 0
    final_info, final_obs = None, obs
    try:
        while steps < N_CHECK_STEPS:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            final_info, final_obs = info, obs
            if not np.isfinite(obs).all() or not np.isfinite(reward):
                print(f'  ПРОВАЛ на шаге {steps}: не-конечные значения')
                ok = False
                break
            if terminated or truncated:
                resets += 1
                obs, _ = env.reset(seed=SEED + steps)
    except Exception as e:                        # noqa: BLE001
        print(f'  ПРОВАЛ: шаг упал: {type(e).__name__}: {e}')
        return 1

    print(f'  пройдено шагов: {steps}/{N_CHECK_STEPS}, '
          f'сбросов эпизода: {resets}')
    print(f'  последний info: date={final_info["date"]} '
          f'PV={final_info["portfolio_value"]:.6f} '
          f'EQ={final_info["equal_value"]:.6f} '
          f'HHI={final_info["hhi_risk"]:.4f} '
          f'turnover={final_info["turnover"]:.4f}')
    if steps != N_CHECK_STEPS:
        print(f'  ПРОВАЛ: пройдено {steps}, ожидалось {N_CHECK_STEPS}')
        ok = False

    # ── 5. Диагностический вывод ──
    print(f'\n{"=" * 70}\n  РЕЗУЛЬТАТЫ СМОУКА\n{"=" * 70}')
    print(f'obs_dim = {obs_dim}')
    print(f'первые 10 значений obs: {np.array2string(final_obs[:10], precision=4)}')
    print(f'\nфинальные веса:')
    uenv = env
    for t, w in zip(uenv.tickers + ['CASH'], final_info['weights']):
        print(f'  {t:5s}: {w:.4f}')
    print(f'  сумма: {float(np.sum(final_info["weights"])):.6f}')

    print(f'\n{"=" * 70}')
    print('  СМОУК ПРОЙДЕН' if ok else '  СМОУК ПРОВАЛЕН')
    print('=' * 70)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
