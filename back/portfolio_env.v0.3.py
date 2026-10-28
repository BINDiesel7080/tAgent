"""
PortfolioEnv — Gymnasium-среда для RL-агента на Мосбирже.

Version: v0.3
Date:   2026-10-02
Changes from v0.2:
  - Комиссия за сделки (COMMISSION_RATE = 0.0004, т.е. 0.04%)
  - Штраф за оборот включён в награду: reward -= turnover * commission_rate
  - prev_weights хранит веса прошлого шага для расчёта оборота

Data:   rl_features.pkl (828 × 31), rl_prices.pkl (828 × 6)

Action:    Box(-1, 1, shape=(7,)) — сырые веса
           → softmax(scale=5) → портфельные веса (6 активов + кэш, сумма=1)
State:     31 рыночных фичей (нормализованных) + 7 текущих весов = 38
Reward:    дневная доходность портфеля — штраф за просадку — комиссия за оборот
Episode:   25 торговых дней (train) / весь сплит (eval)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """RL-среда для управления портфелем на Мосбирже."""

    metadata = {"render_modes": ["human"]}

    # ── Конфиг ──
    EPISODE_LENGTH = 25       # дней в тренировочном эпизоде
    SOFTMAX_SCALE = 5.0       # масштаб перед softmax
    DD_ALPHA = 1.0            # коэффициент штрафа за просадку
    COMMISSION_RATE = 0.0004  # 0.04% за оборот

    def __init__(self, features_path='rl_features.pkl',
                 prices_path='rl_prices.pkl',
                 episode_length=None,
                 eval_mode=False,
                 eval_split='test',
                 split_idx=600,
                 seed=None):
        """
        Args:
            features_path:  путь к rl_features.pkl (state)
            prices_path:    путь к rl_prices.pkl (для бэктеста)
            episode_length: дней в эпизоде (None = EPISODE_LENGTH)
            eval_mode:      True → полный прогон по сплиту
            eval_split:     'train' или 'test' (только при eval_mode=True)
            split_idx:      индекс разреза (дни 0..split_idx-1 = train)
            seed:           зерно для воспроизводимости
        """
        super().__init__()

        self.version = "v0.3"
        self.eval_mode = eval_mode
        self.eval_split = eval_split
        self.split_idx = split_idx

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

        # ── Нормализация ТОЛЬКО по train-данным ──
        train_raw = self.features.iloc[:self.split_idx].values.astype(np.float32)
        self.feat_mean = train_raw.mean(axis=0)
        self.feat_std = train_raw.std(axis=0) + 1e-8

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
        self.prev_weights = self.weights.copy()  # для расчёта оборота
        self.portfolio_value = 1.0
        self.max_peak = 1.0     # пик портфеля для расчёта просадки

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

    def _build_info(self, reward, turnover=0.0, commission_cost=0.0):
        return {
            "step": self.current_step,
            "date": str(self.features.index[self.current_step].date()),
            "weights": self.weights.copy(),
            "portfolio_ret": reward,
            "portfolio_value": self.portfolio_value,
            "turnover": float(turnover),
            "commission_cost": float(commission_cost),
        }

    # ═══════════════════════════════════════════
    # GYMNASIUM API
    # ═══════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.eval_mode:
            if self.eval_split == 'train':
                # Полный прогон по train-части
                self.start_idx = 0
                self.ep_length = self.split_idx - 1
            else:
                # Полный прогон по test-части
                self.start_idx = self.split_idx
                self.ep_length = self.n_days - self.split_idx - 1
        else:
            # Случайное окно ТОЛЬКО в train-части
            max_start = self.split_idx - self.episode_length - 1
            self.start_idx = int(self.rng.integers(0, max_start))
            self.ep_length = self.episode_length

        self.current_step = self.start_idx
        self.weights = np.zeros(self.n_actions, dtype=np.float32)
        self.weights[-1] = 1.0
        self.prev_weights = self.weights.copy()  # сброс оборота
        self.portfolio_value = 1.0
        self.max_peak = 1.0     # сброс пика в начале эпизода

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
          4. Комиссия = turnover * COMMISSION_RATE
          5. Награда = доходность — штраф за просадку — комиссия
          6. Переход к t+1
        """
        # ── 1. Softmax → веса ──
        action = np.asarray(action, dtype=np.float32)
        self.weights = self._softmax(action)

        # ── 2. Расчёт оборота и комиссии ──
        turnover = float(np.sum(np.abs(self.weights - self.prev_weights)))
        commission_cost = turnover * self.COMMISSION_RATE

        # ── 3. Следующий шаг ──
        next_step = self.current_step + 1

        # ── 4. Конец данных ──
        if next_step >= self.n_days:
            reward = 0.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = self._build_info(reward, turnover, commission_cost)
            return obs, reward, terminated, truncated, info

        # ── 5. Доходности следующего дня ──
        next_returns = np.array([
            self.features.iloc[next_step][f"{t}_returns"]
            for t in self.tickers
        ], dtype=np.float32)

        # ── 6. Доходность портфеля ──
        portfolio_ret = float(
            np.dot(self.weights[:self.n_assets], next_returns)
        )

        # ── 7. Обновление стоимости портфеля (с учётом комиссии) ──
        self.portfolio_value *= (1 + portfolio_ret - commission_cost)

        # ── 8. Штраф за просадку ──
        if self.portfolio_value > self.max_peak:
            self.max_peak = self.portfolio_value
        current_dd = (self.max_peak - self.portfolio_value) / self.max_peak
        dd_penalty = self.DD_ALPHA * (current_dd ** 2)

        # ── 9. Финальная награда ──
        reward = portfolio_ret - dd_penalty - commission_cost

        # ── 10. Запоминаем веса для следующего шага ──
        self.prev_weights = self.weights.copy()

        # ── 11. Переход ──
        self.current_step = next_step
        steps_elapsed = self.current_step - self.start_idx
        truncated = steps_elapsed >= self.ep_length
        terminated = False

        obs = self._get_obs()
        info = self._build_info(reward, turnover, commission_cost)

        return obs, reward, terminated, truncated, info

    def render(self):
        w_str = ", ".join(
            f"{t}:{w:.2f}"
            for t, w in zip(self.tickers + ["CASH"], self.weights)
        )
        print(f"  [{self.current_step:4d}] {w_str} | "
              f"PV={self.portfolio_value:.4f}")
