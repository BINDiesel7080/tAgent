"""
PortfolioEnv — Gymnasium-среда для RL-агента на Мосбирже.

Version: v0.6
Date:   2026-10-07
Changes from v0.5:
  - Hard cap на макс. долю одного актива (max_weight), кэш не ограничен
  - Штраф за концентрацию через HHI (без кэша): -lambda_hhi × sum(w_i^2)
  - Избыток веса при cap уходит в кэш (cash как «сток»)

Data:   rl_features.pkl (828 × 31), rl_prices.pkl (828 × 6)

Action:    Box(-1, 1, shape=(7,)) — сырые веса
           → softmax(scale=5) → cap(max_weight) → портфельные веса (6 активов + кэш, сумма=1)
State:     31 рыночных фичей (нормализованных) + 7 текущих весов = 38
Reward:    ежедневно: net_ret - dd_penalty - lambda_hhi × HHI_risk
           терминально (конец эпизода): + TERMINAL_REWARD_SCALE × (portfolio_value - 1.0)
           HHI_risk = sum(w_i^2) только по акциям, без кэша
Episode:   30 торговых дней (train) / весь test-сплит фолда (eval)

Folds (sliding window, train=400, test=100, step=109):
  Fold 0: train [  0:400]  test [400:500]
  Fold 1: train [109:509]  test [509:609]
  Fold 2: train [218:618]  test [618:718]
  Fold 3: train [327:727]  test [727:828]
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """RL-среда для управления портфелем на Мосбирже."""

    metadata = {"render_modes": ["human"]}

    # ── Конфиг ──
    EPISODE_LENGTH = 30          # дней в тренировочном эпизоде
    SOFTMAX_SCALE = 5.0          # масштаб перед softmax
    DD_ALPHA = 1.0               # коэффициент штрафа за просадку
    COMMISSION_RATE = 0.0004     # 0.04% за оборот
    TERMINAL_REWARD_SCALE = 3.0  # масштаб терминальной награды
    DEFAULT_MAX_WEIGHT = 0.5     # макс. доля одного актива (кэш не ограничен)
    DEFAULT_LAMBDA_HHI = 0.5     # штраф за концентрацию (HHI без кэша)

    # ── Walk-forward фолды ──
    TRAIN_WINDOW = 400
    TEST_WINDOW = 100
    STEP = 109
    FOLDS = []
    for i in range(4):
        train_start = i * STEP
        train_end = train_start + TRAIN_WINDOW
        test_start = train_end
        test_end = min(test_start + TEST_WINDOW, 828)
        FOLDS.append((train_start, train_end, test_start, test_end))

    def __init__(self, features_path='rl_features.pkl',
                 prices_path='rl_prices.pkl',
                 episode_length=None,
                 eval_mode=False,
                 eval_split='test',
                 fold=0,
                 seed=None,
                 max_weight=None,
                 lambda_hhi=None):
        """
        Args:
            features_path:  путь к rl_features.pkl (state)
            prices_path:    путь к rl_prices.pkl (для бэктеста)
            episode_length: дней в эпизоде (None = EPISODE_LENGTH)
            eval_mode:      True → полный прогон по сплиту фолда
            eval_split:     'train' или 'test' (только при eval_mode=True)
            fold:           номер фолда walk-forward (0..3)
            seed:           зерно для воспроизводимости
            max_weight:     макс. доля одного актива, кэш не ограничен (None = DEFAULT)
            lambda_hhi:     штраф за концентрацию HHI без кэша (None = DEFAULT)
        """
        super().__init__()

        self.version = "v0.6"
        self.eval_mode = eval_mode
        self.eval_split = eval_split
        self.fold = fold
        self.max_weight = max_weight if max_weight is not None else self.DEFAULT_MAX_WEIGHT
        self.lambda_hhi = lambda_hhi if lambda_hhi is not None else self.DEFAULT_LAMBDA_HHI

        # ── Границы фолда ──
        self.train_start, self.train_end, self.test_start, self.test_end = \
            self.FOLDS[fold]

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

        # ── Нормализация ТОЛЬКО по train-данным текущего фолда ──
        train_raw = self.features.iloc[
            self.train_start:self.train_end
        ].values.astype(np.float32)
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
        self.prev_weights = self.weights.copy()
        self.portfolio_value = 1.0
        self.max_peak = 1.0
        self.equal_value = 1.0      # Equal Weight benchmark
        self.equal_peak = 1.0       # пик Equal Weight

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

    def _apply_cap(self, weights):
        """
        Жёсткое ограничение: ни один актив (кроме кэша) не больше max_weight.
        Избыток веса перераспределяется в кэш.
        Сумма весов сохраняется = 1.
        """
        w = weights.copy()
        # Обрезаем только рисковые активы (не кэш)
        excess = 0.0
        for i in range(self.n_assets):
            if w[i] > self.max_weight:
                excess += w[i] - self.max_weight
                w[i] = self.max_weight
        # Избыток уходит в кэш
        w[-1] += excess
        return w

    def _compute_hhi_risk(self):
        """HHI только по рисковым активам, без кэша."""
        return float(np.sum(self.weights[:self.n_assets] ** 2))

    def _build_info(self, reward, turnover=0.0, commission_cost=0.0, hhi=0.0):
        return {
            "step": self.current_step,
            "date": str(self.features.index[self.current_step].date()),
            "weights": self.weights.copy(),
            "portfolio_ret": reward,
            "portfolio_value": self.portfolio_value,
            "turnover": float(turnover),
            "commission_cost": float(commission_cost),
            "equal_value": self.equal_value,
            "hhi_risk": float(hhi),
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
                self.start_idx = self.train_start
                self.ep_length = self.train_end - self.train_start - 1
            else:
                self.start_idx = self.test_start
                self.ep_length = self.test_end - self.test_start - 1
        else:
            # Случайное окно в train-части текущего фолда
            max_start = self.train_end - self.episode_length - 1
            self.start_idx = int(self.rng.integers(self.train_start, max_start))
            self.ep_length = self.episode_length

        self.current_step = self.start_idx
        self.weights = np.zeros(self.n_actions, dtype=np.float32)
        self.weights[-1] = 1.0
        self.prev_weights = self.weights.copy()
        self.portfolio_value = 1.0
        self.max_peak = 1.0
        self.equal_value = 1.0
        self.equal_peak = 1.0

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
          1.  Агент видит features[t] и выдаёт action
          2.  action → softmax → веса
          3.  Жёсткий cap: max_weight на каждый актив (кэш свободен), избыток → в кэш
          4.  Оборот = сумма |веса - prev_веса| (только по акциям)
          5.  Комиссия = оборот × COMMISSION_RATE
          6.  Доходность = dot(веса, returns[t+1]) — комиссия
          7.  Equal Weight доходность = mean(returns[t+1])
          8.  Штраф за просадку от max(PPO peak, Equal peak)
          9.  HHI_risk = sum(w_i^2) по акциям (без кэша)
          10. Награда = доходность — штраф за просадку — lambda_hhi × HHI_risk
          11. В конце эпизода: += TERMINAL_REWARD_SCALE × (portfolio_value - 1.0)
          12. Переход к t+1
        """
        # ── 1. Softmax → веса ──
        action = np.asarray(action, dtype=np.float32)
        self.weights = self._softmax(action)

        # ── 2. Жёсткий cap на активы (кэш свободен) ──
        self.weights = self._apply_cap(self.weights)

        # ── 3. Следующий шаг ──
        next_step = self.current_step + 1

        # ── 4. Конец данных ──
        if next_step >= self.n_days:
            reward = 0.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = self._build_info(reward)
            return obs, reward, terminated, truncated, info

        # ── 5. Оборот (только по акциям, без кэша) ──
        turnover = float(np.sum(np.abs(
            self.weights[:self.n_assets] - self.prev_weights[:self.n_assets]
        )))
        commission_cost = turnover * self.COMMISSION_RATE

        # ── 6. Доходности следующего дня ──
        next_returns = np.array([
            self.features.iloc[next_step][f"{t}_returns"]
            for t in self.tickers
        ], dtype=np.float32)

        # ── 7. Доходность портфеля (брутто) ──
        portfolio_ret = float(
            np.dot(self.weights[:self.n_assets], next_returns)
        )

        # ── 8. Доходность нетто (с комиссией) ──
        net_ret = portfolio_ret - commission_cost

        # ── 9. Обновление стоимости портфеля ──
        self.portfolio_value *= (1 + net_ret)

        # ── 10. Equal Weight benchmark ──
        equal_ret = float(np.mean(next_returns))
        self.equal_value *= (1 + equal_ret)
        if self.equal_value > self.equal_peak:
            self.equal_peak = self.equal_value

        # ── 11. Пики ──
        if self.portfolio_value > self.max_peak:
            self.max_peak = self.portfolio_value

        # ── 12. Просадка от max(PPO peak, Equal peak) ──
        ref_peak = max(self.max_peak, self.equal_peak)
        current_dd = max(0.0, (ref_peak - self.portfolio_value) / ref_peak)
        dd_penalty = self.DD_ALPHA * (current_dd ** 2)

        # ── 13. HHI без кэша (штраф за концентрацию в активах) ──
        hhi_risk = self._compute_hhi_risk()
        hhi_penalty = self.lambda_hhi * hhi_risk

        # ── 14. Базовая награда (ежедневный сигнал) ──
        reward = net_ret - dd_penalty - hhi_penalty

        # ── 15. Сохраняем веса для следующего шага ──
        self.prev_weights = self.weights.copy()

        # ── 16. Переход ──
        self.current_step = next_step
        steps_elapsed = self.current_step - self.start_idx
        truncated = steps_elapsed >= self.ep_length
        terminated = False

        # ── 17. Терминальная награда (конец эпизода) ──
        if truncated:
            episode_return = self.portfolio_value - 1.0
            reward += self.TERMINAL_REWARD_SCALE * episode_return

        obs = self._get_obs()
        info = self._build_info(reward, turnover, commission_cost, hhi_risk)

        return obs, reward, terminated, truncated, info

    def render(self):
        w_str = ", ".join(
            f"{t}:{w:.2f}"
            for t, w in zip(self.tickers + ["CASH"], self.weights)
        )
        print(f"  [{self.current_step:4d}] {w_str} | "
              f"PV={self.portfolio_value:.4f} | EQ={self.equal_value:.4f} | "
              f"HHI={self._compute_hhi_risk():.3f}")
