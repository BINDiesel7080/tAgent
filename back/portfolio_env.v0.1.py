"""
PortfolioEnv — Gymnasium-среда для RL-агента на Мосбирже.

Version: v0.1
Date:   2026-10-02
Data:   rl_features.pkl (828 × 31), rl_prices.pkl (828 × 6)

Action:    Box(-1, 1, shape=(7,)) — сырые веса
           → softmax(scale=5) → портфельные веса (6 активов + кэш, сумма=1)
State:     31 рыночных фичей (нормализованных) + 7 текущих весов = 38
Reward:    дневная доходность портфеля (без штрафов, без шортов)
Episode:   60 торговых дней (train) / весь датасет (eval)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """RL-среда для управления портфелем на Мосбирже."""

    metadata = {"render_modes": ["human"]}

    # ── Конфиг ──
    EPISODE_LENGTH = 60       # дней в тренировочном эпизоде
    SOFTMAX_SCALE = 5.0       # масштаб перед softmax (больше = концентрированнее)

    def __init__(self, features_path='rl_features.pkl',
                 prices_path='rl_prices.pkl',
                 episode_length=None,
                 eval_mode=False,
                 seed=None):
        """
        Args:
            features_path:  путь к rl_features.pkl (state)
            prices_path:     путь к rl_prices.pkl (для бэктеста)
            episode_length:  дней в эпизоде (None = EPISODE_LENGTH)
            eval_mode:       True → старт с дня 0, полный прогон
            seed:            зерно для воспроизводимости
        """
        super().__init__()

        self.version = "v0.1"
        self.eval_mode = eval_mode

        # ── Данные ──
        self.features = pd.read_pickle(features_path)
        self.prices = pd.read_pickle(prices_path)
        self.tickers = [c.replace('_close', '')
                        for c in self.prices.columns]
        self.n_assets = len(self.tickers)    # 6
        self.n_actions = self.n_assets + 1  # +кэш = 7
        self.n_days = len(self.features)

        # ── Эпизоды ──
        self.episode_length = episode_length or self.EPISODE_LENGTH

        # ── Нормализация фичей (фиксированная, по всему датасету) ──
        raw = self.features.values.astype(np.float32)
        self.feat_mean = raw.mean(axis=0)
        self.feat_std = raw.std(axis=0) + 1e-8

        # ── Пространства ──
        obs_dim = self.features.shape[1] + self.n_actions  # 31 + 7 = 38
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_actions,), dtype=np.float32
        )

        # ── Внутреннее состояние ──
        self.current_step = 0
        self.start_idx = 0
        self.ep_length = self.episode_length
        self.weights = np.zeros(self.n_actions, dtype=np.float32)
        self.weights[-1] = 1.0  # старт: 100% кэш
        self.portfolio_value = 1.0

        # ── RNG ──
        self.rng = np.random.default_rng(seed)

    # ═══════════════════════════════════════════
    # ПРИВАТНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════

    def _get_obs(self):
        """Собрать наблюдение: нормализованные фичи + текущие веса."""
        feat = self.features.iloc[self.current_step].values.astype(np.float32)
        feat_norm = (feat - self.feat_mean) / self.feat_std
        obs = np.concatenate([feat_norm, self.weights])
        return obs.astype(np.float32)

    def _softmax(self, x):
        """Softmax с масштабированием (численно стабильный)."""
        x = x * self.SOFTMAX_SCALE
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def _build_info(self, reward):
        return {
            "step": self.current_step,
            "date": str(self.features.index[self.current_step].date()),
            "weights": self.weights.copy(),
            "portfolio_ret": reward,
            "portfolio_value": self.portfolio_value,
        }

    # ═══════════════════════════════════════════
    # GYMNASIUM API
    # ═══════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.eval_mode:
            # Полный прогон с первого дня
            self.start_idx = 0
            self.ep_length = self.n_days - 1
        else:
            # Случайное окно
            max_start = self.n_days - self.episode_length - 1
            self.start_idx = int(self.rng.integers(0, max_start))
            self.ep_length = self.episode_length

        self.current_step = self.start_idx
        self.weights = np.zeros(self.n_actions, dtype=np.float32)
        self.weights[-1] = 1.0
        self.portfolio_value = 1.0

        obs = self._get_obs()
        info = {
            "start_date": str(self.features.index[self.current_step].date()),
            "start_idx": self.start_idx,
        }
        return obs, info

    def step(self, action):
        """
        Один шаг среды.

        Логика:
          1. Агент видит features[t] и выдаёт action
          2. action → softmax → веса портфеля
          3. Доходность = dot(веса, returns[t+1])
             (агент принимает решение на день t,
              доходность реализуется от t к t+1)
          4. Переход к t+1
        """
        # ── 1. Softmax → веса ──
        action = np.asarray(action, dtype=np.float32)
        self.weights = self._softmax(action)

        # ── 2. Следующий шаг ──
        next_step = self.current_step + 1

        # ── 3. Конец данных ──
        if next_step >= self.n_days:
            reward = 0.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = self._build_info(reward)
            return obs, reward, terminated, truncated, info

        # ── 4. Доходности следующего дня ──
        next_returns = np.array([
            self.features.iloc[next_step][f"{t}_returns"]
            for t in self.tickers
        ], dtype=np.float32)

        # ── 5. Доходность портфеля ──
        portfolio_ret = float(
            np.dot(self.weights[:self.n_assets], next_returns)
        )
        reward = portfolio_ret

        # ── 6. Обновление стоимости портфеля ──
        self.portfolio_value *= (1 + reward)

        # ── 7. Переход ──
        self.current_step = next_step
        steps_elapsed = self.current_step - self.start_idx
        truncated = steps_elapsed >= self.ep_length
        terminated = False

        obs = self._get_obs()
        info = self._build_info(reward)

        return obs, reward, terminated, truncated, info

    def render(self):
        w_str = ", ".join(
            f"{t}:{w:.2f}"
            for t, w in zip(self.tickers + ["CASH"], self.weights)
        )
        print(f"  [{self.current_step:4d}] {w_str} | "
              f"PV={self.portfolio_value:.4f}")
