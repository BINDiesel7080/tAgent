"""
PortfolioEnv — Gymnasium-среда для RL-агента на Мосбирже.

Version: v1.1
Date:   2026-10-21
Changes from v0.9:
  - Случайная длина тренировочного эпизода: episode_length_range (по умолчанию 30-200),
    сэмплируется на каждом reset — против заучивания одних и тех же окон.
  - Случайный стартовый портфель в train-эпизодах (random_start_prob):
    агент учится не только разворачиваться из 100% кэша, но и управлять
    уже существующим портфелем. В eval-режиме старт всегда 100% кэш.
  - Excess-HHI: штраф начисляется только за концентрацию ВЫШЕ равных весов:
    penalty = lambda_hhi * max(0, HHI - 1/n_assets). Ровный портфель штрафа не платит.
  - Просадка считается от собственного пика портфеля, а не от max(PPO peak, Equal peak).
  - dd_alpha: корректная проверка None вместо `or` (0.0 теперь передаётся честно).
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """RL-среда для управления портфелем на Мосбирже."""

    metadata = {"render_modes": ["human"]}

    # ────────────────────────── Конфиг среды ───────────────────────────
    EPISODE_LENGTH = 30           # дефолтная длина эпизода (если диапазон не задан)
    SOFTMAX_SCALE = 5.0           # масштаб перед softmax
    DEFAULT_DD_ALPHA = 0.7        # коэффициент штрафа за просадку (*дефолт*)
    COMMISSION_RATE = 0.0004      # 0.04% за оборот
    DEFAULT_MAX_WEIGHT = 0.5      # макс. доля одного актива (кэш не ограничен)
    DEFAULT_LAMBDA_HHI = 0.4      # штраф за excess-концентрацию

    # Диапазон случайной длины тренировочного эпизода (п.1: против заучивания окон)
    DEFAULT_EP_LENGTH_RANGE = (30, 200)
    # Вероятность случайного стартового портфеля в train-эпизодах (п.3)
    DEFAULT_RANDOM_START_PROB = 0.5

    # ──────── Walk-forward v2: расширенная история ────────
    TRAIN_WINDOW = 1200    # ~3.3 года строк с выходными
    TEST_WINDOW  = 100
    EMBARGO      = 5
    HOLDOUT_ROWS = 101     # прежний holdout = последние 101 строка
    N_OPT_FOLDS  = 5
    EPISODES_START = '2014-09-01'   # PLZL: июнь-2014 + 50 баров прогрева

    @classmethod
    def build_folds(cls, index):
        idx = pd.DatetimeIndex(index)
        ep_start = int(idx.searchsorted(pd.Timestamp(cls.EPISODES_START)))
        h_start = len(idx) - cls.HOLDOUT_ROWS
        usable_end = h_start - cls.EMBARGO
        span = usable_end - ep_start - cls.TRAIN_WINDOW - cls.EMBARGO - cls.TEST_WINDOW
        step = max(1, span // (cls.N_OPT_FOLDS - 1))
        folds = []
        for i in range(cls.N_OPT_FOLDS):
            ts = ep_start + i * step
            te, vs = ts + cls.TRAIN_WINDOW, ts + cls.TRAIN_WINDOW + cls.EMBARGO
            ve = vs + cls.TEST_WINDOW
            if te > usable_end or ve > usable_end:
                break
            folds.append((ts, te, vs, ve))
        folds.append((h_start - cls.EMBARGO - cls.TRAIN_WINDOW,
                      h_start - cls.EMBARGO, h_start, len(idx)))
        return folds


    def __init__(self,
                 features_path='rl_features_v4.pkl',
                 prices_path='rl_prices_v4.pkl',
                 episode_length=None,
                 episode_length_range=None,   # <--- NEW: (min, max) длины эпизода
                 random_start_prob=None,      # <--- NEW: вероятность случайного старта
                 eval_mode=False,
                 eval_split='test',  # Только 'train' или 'test'
                 fold=0,
                 seed=None,
                 max_weight=None,
                 lambda_hhi=None,
                 dd_alpha=None):
        """
        Args:
            episode_length: фиксированная длина эпизода (если episode_length_range=None).
            episode_length_range: (min, max) — на каждом reset длина эпизода
                сэмплируется из этого диапазона. Применяется ТОЛЬКО в train-режиме;
                в eval-режиме длина всегда равна размеру сплита.
            random_start_prob: вероятность стартовать эпизод со случайным портфелем
                (Дирихле по активам + примесь кэша) вместо 100% кэша.
                Применяется ТОЛЬКО в train-режиме; в eval-режиме старт всегда из кэша.
            dd_alpha: Коэффициент штрафа за просадку внутри эпизода.
                Если None, используется дефолтный параметр класса.
        """
        super().__init__()

        self.version = "v1.1"
        self.eval_mode = eval_mode
        self.eval_split = eval_split
        self.fold = fold
        self.max_weight = max_weight if max_weight is not None else self.DEFAULT_MAX_WEIGHT
        self.lambda_hhi = lambda_hhi if lambda_hhi is not None else self.DEFAULT_LAMBDA_HHI

        # Параметры награды (dd_alpha теперь с корректной None-проверкой)
        self.dd_alpha = self.DEFAULT_DD_ALPHA if dd_alpha is None else dd_alpha

        # Регуляризация эпизодов
        self.episode_length = self.EPISODE_LENGTH if episode_length is None else episode_length
        self.episode_length_range = (self.DEFAULT_EP_LENGTH_RANGE
                                     if episode_length_range is None else episode_length_range)
        self.random_start_prob = (self.DEFAULT_RANDOM_START_PROB
                                  if random_start_prob is None else random_start_prob)

        # Данные
        self.features = pd.read_pickle(features_path).astype(np.float32)
        self.prices = pd.read_pickle(prices_path).astype(np.float32)
        self.tickers = [c.replace('_close', '') for c in self.prices.columns]
        self.n_assets = len(self.tickers)
        self.n_actions = self.n_assets + 1  # + кэш
        self.n_days = len(self.features)

        # Границы фолда (после загрузки данных: фолды зависят от длины датасета)
        self.folds = self.build_folds(self.features.index)
        ts, te, vs, ve = self.folds[fold]
        self.train_start, self.train_end, self.test_start, self.test_end = ts, te, vs, ve

        # Минимально возможный HHI при равных весах по активам (= 1/n_assets)
        self.hhi_min = 1.0 / self.n_assets

        # Нормализация ТОЛЬКО по train-данным текущего фолда
        train_raw = self.features.iloc[self.train_start:self.train_end].values.astype(np.float32)
        self.feat_mean = train_raw.mean(axis=0)
        self.feat_std = train_raw.std(axis=0) + 1e-8

        # Пространства
        obs_dim = self.features.shape[1] + self.n_actions  # 31 + 7 = 38
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_actions,), dtype=np.float32)

        # Внутреннее состояние
        self.current_step = 0
        self.start_idx = 0
        self.ep_length = self.episode_length
        self.weights = np.zeros(self.n_actions, dtype=np.float32)
        self.weights[-1] = 1.0  # старт: 100% кэш
        self.prev_weights = self.weights.copy()
        self.portfolio_value = 1.0
        self.max_peak = 1.0
        self.equal_value = 1.0  # Equal Weight benchmark
        self.equal_peak = 1.0   # трекается для диагностики, в награде больше не используется

        # RNG
        self.rng = np.random.default_rng(seed)

        # карта разрывов календаря: для каждого старта — расстояние до следующего гэпа
        ddays = self.features.index.to_series().diff().dt.days.values
        gap_rows = set((np.where(ddays > 10)[0] + 1).tolist())
        self._next_gap = np.full(self.n_days, self.n_days, dtype=np.int64)
        nxt = self.n_days
        for i in range(self.n_days - 1, -1, -1):
            if i in gap_rows:
                nxt = i
            self._next_gap[i] = nxt


# ────────────────────────── Приватные методы ───────────────────────────

    def _get_obs(self):
        """Собрать наблюдение: нормализованные фичи + текущие веса."""
        feat = self.features.iloc[self.current_step].values
        feat_norm = (feat - self.feat_mean) / self.feat_std
        return np.concatenate([feat_norm, self.weights]).astype(np.float32)

    def _softmax(self, x):
        """Softmax с масштабированием (численно стабильный)."""
        x = x * self.SOFTMAX_SCALE
        x = x - x.max()  # Для численной стабильности
        e = np.exp(x)
        return e / e.sum()

    def _apply_cap(self, weights):
        """
        Жёсткое ограничение: ни один актив (кроме кэша) не больше max_weight.
        Избыток веса перераспределяется в кэш.
        Сумма весов сохраняется = 1.
        """
        w = weights.copy()
        excess = 0.0
        for i in range(self.n_assets):
            if w[i] > self.max_weight:
                excess += w[i] - self.max_weight
                w[i] = self.max_weight
        w[-1] += excess  # Избыток уходит в кэш
        return w

    def _sample_start_weights(self):
        """
        Стартовые веса эпизода (только train-режим).
        С вероятностью random_start_prob — случайный портфель:
          долю invested в [0.3, 1.0] раскидываем по активам по Дирихле,
          остальное — в кэш; активы ограничены max_weight.
        Иначе — 100% кэш (как раньше).
        """
        if self.random_start_prob > 0 and self.rng.random() < self.random_start_prob:
            w_assets = self.rng.dirichlet(np.full(self.n_assets, 1.0)).astype(np.float32)
            invested = self.rng.uniform(0.3, 1.0)
            w = np.zeros(self.n_actions, dtype=np.float32)
            w[:self.n_assets] = w_assets * invested
            w = self._apply_cap(w)
            w[-1] = 1.0 - w[:self.n_assets].sum()  # остаток — кэш
            return w
        w = np.zeros(self.n_actions, dtype=np.float32)
        w[-1] = 1.0
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

# ────────────────────────── GYMNASIUM API ───────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.eval_mode:
            start_map = {'train': self.train_start, 'test': self.test_start}
            self.start_idx = start_map[self.eval_split]
            self.ep_length = getattr(self, f"{self.eval_split}_end") - self.start_idx - 1
        else:
            # Случайная длина эпизода из диапазона (п.1: против заучивания окон)
            lo, hi = self.episode_length_range
            self.ep_length = int(self.rng.integers(lo, hi + 1))
            max_start = self.train_end - self.ep_length - 1
            # эпизод не должен пересекать разрыв календаря (остановка торгов 02-03.2022)
            for _ in range(64):
                s = int(self.rng.integers(self.train_start, max(2, max_start)))
                if self._next_gap[s] - s >= self.ep_length + 1:
                    self.start_idx = s
                    break
            else:
                self.start_idx = int(self.rng.integers(self.train_start, max(2, max_start)))

        self.current_step = self.start_idx

        # Стартовые веса (п.3): в eval всегда 100% кэш, в train — случайно с вер. random_start_prob
        if self.eval_mode:
            self.weights = np.zeros(self.n_actions, dtype=np.float32)
            self.weights[-1] = 1.0
        else:
            self.weights = self._sample_start_weights()
        self.prev_weights = self.weights.copy()

        self.portfolio_value = 1.0
        self.max_peak = 1.0
        self.equal_value = 1.0
        self.equal_peak = 1.0

        obs = self._get_obs()
        info = {
            "start_date": str(self.features.index[self.current_step].date()),
            "start_idx": self.start_idx,
            "ep_length": self.ep_length,
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
          8.  Штраф за просадку от СОБСТВЕННОГО пика портфеля
          9.  HHI_risk = sum(w_i^2) по акциям (без кэша)
          10. Награда = доходность — штраф за просадку — lambda_hhi × max(0, HHI - 1/n)
          11. Переход к t+1
        """
        # 1. Softmax → веса
        action = np.asarray(action, dtype=np.float32)
        self.weights = self._softmax(action)

        # 2. Жёсткий cap на активы (кэш свободен)
        self.weights = self._apply_cap(self.weights)

        # 3. Следующий шаг
        next_step = self.current_step + 1

        # 4. Конец данных
        if next_step >= self.n_days:
            reward = 0.0
            terminated = True
            truncated = False
            obs = self._get_obs()
            info = self._build_info(reward)
            return obs, reward, terminated, truncated, info

        # 5. Оборот (только по акциям, без кэша)
        turnover = float(np.sum(np.abs(
            self.weights[:self.n_assets] - self.prev_weights[:self.n_assets])))
        commission_cost = turnover * self.COMMISSION_RATE

        # 6. Доходности следующего дня
        next_returns = np.array([
            self.features.iloc[next_step][f"{t}_returns"]
            for t in self.tickers
        ], dtype=np.float32)

        # 7. Доходность портфеля (брутто)
        portfolio_ret = float(
            np.dot(self.weights[:self.n_assets], next_returns))  # Без учёта кэша

        # 8. Доходность нетто (с комиссией)
        net_ret = portfolio_ret - commission_cost

        # 9. Обновление стоимости портфеля
        self.portfolio_value *= (1 + net_ret)

        # 10. Equal Weight benchmark (для диагностики, в награде не участвует)
        equal_ret = float(np.mean(next_returns))
        self.equal_value *= (1 + equal_ret)
        if self.equal_value > self.equal_peak:
            self.equal_peak = self.equal_value

        # 11. Пик портфеля
        if self.portfolio_value > self.max_peak:
            self.max_peak = self.portfolio_value

        # 12. Просадка от СОБСТВЕННОГО пика (п.13: без привязки к пику Equal)
        current_dd = max(0.0, (self.max_peak - self.portfolio_value) / self.max_peak)
        dd_penalty = self.dd_alpha * current_dd

        # 13. Excess-HHI (п.12): штраф только за концентрацию выше равных весов
        hhi_risk = self._compute_hhi_risk()
        hhi_excess = max(0.0, hhi_risk - self.hhi_min)
        hhi_penalty = self.lambda_hhi * hhi_excess

        # 14. Базовая награда (ежедневный сигнал)
        reward = net_ret - dd_penalty - hhi_penalty

        # 15. Сохраняем веса для следующего шага
        self.prev_weights = self.weights.copy()

        # 16. Переход
        self.current_step = next_step
        steps_elapsed = self.current_step - self.start_idx
        truncated = steps_elapsed >= self.ep_length
        terminated = False

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
