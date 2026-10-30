"""
stage2_only_v1_2.py — ЭТАП 2 итерации v1.2: ЕДИНСТВЕННЫЙ заход на holdout (fold 5).

Version: v1.2 (rev 2)
Date:    2026-10-29

Назначение:
    Взять лучший триал из уже накопленной базы Optuna v1.2 (строго READ-ONLY),
    обучить PPO на train-части holdout-фолда (fold 5, последний фолд) на 7 сидах
    (как в v1.1), прогнать каждый сид на test-окне holdout
    (2026-06-16 .. 2026-10-02, 101 строка) и сравнить с Equal-Weight бенчмарком
    по расширенному набору метрик (доходность + максимальная просадка + кэш +
    оборот). Поиск Optuna НЕ запускается, база НЕ модифицируется.

Отличия от stage2_only.py (v1.1):
  1. FEATURES_FILE = rl_features_v5.pkl (49 колонок) -> obs_dim 56 (в v1.1 было 38).
     features_path передаётся ЯВНО во ВСЕХ созданиях PortfolioEnv; дефолт
     portfolio_env.py (rl_features_v4.pkl) не используется и не меняется.
  2. VERSION = "v1.2", DB = optuna_ppo_v1_2.db, STUDY = ppo_v1.2; модели
     ppo_v1.2_fold5_s{seed}.zip; лог logs/stage2_v1_2_<timestamp>.log.
  3. База Optuna читается только read-only (sqlite URI ?mode=ro&uri=true,
     рецепт plot_progress.py); если read-only не открывается из-за WAL
     активного прогона — читается КОПИЯ базы во временной папке.
     Ни одной операции записи: study не создаётся, поиск не запускается,
     триалы не добавляются, user-attrs не пишутся.
  4. Сверка фактического лидера с ожидаемым по протоколу AGENTS.md
     (trial 14, mw=0.55, λhhi=0.0023, ts=250k, dd_alpha=8.38, γ=0.998, ep=4,
     robust ≈ +0.0080): построчный маркер OK / MISMATCH. При MISMATCH —
     предупреждение в лог, но обучение ВСЁ РАВНО идёт по фактическому
     лидеру (не падаем).
  5. Защитный гейт obs_dim: если features.shape[1] + 7 != 56 — ОСТАНОВКА
     (sys.exit(1)) ДО обучения. Ловит случайно подсунутый rl_features_v4.pkl.
  6. Предохранитель повторного захода: если модели ppo_v1.2_fold5_s*.zip уже
     существуют — ОСТАНОВКА (sys.exit(1)) ДО обучения. Существующие файлы
     не перезаписываются и не удаляются автоматически.
  7. Расширенный отчёт по каждому сиду: ret_PPO, ret_Equal, diff п.п.,
     maxDD_PPO, maxDD_Equal (обе просадки — на том же test-окне), mean_cash,
     mean_turnover. Агрегат по 7 сидам: mean/std, min/max ret_PPO, число сидов
     выше Equal; диагностика «повадки победителя» v1.1 (кэш 0.10-0.30).
  8. Итоговый вердикт по ДВУМ критериям (доходность не хуже Equal И просадка
     меньше Equal) по СРЕДНЕМУ 7 сидов; код возврата 0 (PASS) / 2 (FAIL).
  9. Весь вывод дублируется в logs/stage2_v1_2_<timestamp>.log
     (tee с методом isatty(), как в train_ppo.py).

Changes rev 2 (2026-10-29) — по итогам ревизии перед единственным запуском:
  F1. Лидер берётся через штатный study.best_trial (учитывает направление),
      направление печатается в лог; если оно не MAXIMIZE — ОСТАНОВКА (sys.exit(1)).
      Топ-5 сортируется с учётом направления. Печатается число триалов по
      состояниям (COMPLETE / PRUNED / FAIL / RUNNING) для сверки с протоколом.
  F2. Гиперпараметры читаются через params.get(key, default) с дефолтами,
      совпадающими с PortfolioEnv.DEFAULT_* / train_ppo; при отсутствии ключа
      печатается [WARN] с указанием источника дефолта.
  F3. _Tee.write переписан: построчный буфер, префикс [HH:MM:SS] ровно один раз
      на строку, без двойных переводов строки; остаток строки флашится в
      flush()/close(); '\r' (прогресс-бар SB3) нормализуется в '\n'.
  F4. Убрана глобальная подмена встроенной print-функции — подменяются только
      sys.stdout/sys.stderr (как в train_ppo._Tee), и ничего больше.
  F5. Read-only доступ к базе: абсолютный путь с прямыми слэшами + рабочий
      рецепт plot_progress.py (два кандидата URI, затем fallback на копию);
      в логе явно печатается, какой источник реально использован.
  F6. Допуск сверки lambda_hhi = 1e-4 (в протоколе значение округлено), для
      остальных параметров max(1e-6, 1e-3*|expected|); использованный tol
      печатается в таблице, текст-«оправдание» из предупреждения убран.
  F7. Сверено с фактическим кодом portfolio_env.py/train_ppo.py:
      имена аргументов конструктора, features/tickers, obs_dim = shape[1] + 7,
      info['weights'] (7 элементов, последний — кэш), info['portfolio_value']
      (эквити от 1.0), PortfolioEnv.TRAIN_WINDOW/TEST_WINDOW/EMBARGO/HOLDOUT_ROWS
      как class-атрибуты; устаревший атрибут со списком фолдов не используется
      вообще (фолды берутся через build_folds + env.folds).
  F8. evaluate() не создаёт Monitor и ничего не пишет на диск; неиспользуемый
      третий элемент помечен # noqa: F841.

Что НЕ изменено относительно v1.1 (для сопоставимости чисел):
    - train_one_fold / evaluate / benchmarks_on_split / get_folds скопированы
      из stage2_only.py 1:1;
    - сиды [42, 1042, 2042, 3042, 4042, 5042, 6042];
    - EP_LEN_RANGE=(30,200), RANDOM_START_PROB=0.5, гиперпараметры PPO
      (lr=1e-4, n_steps=2048, batch_size=256, clip_range=0.1, ent_coef=0.01);
    - разметка фолдов, TRAIN_WINDOW/TEST_WINDOW/EMBARGO/HOLDOUT_ROWS не трогаются;
    - модель сохраняется в ppo_v1.2_fold5_s{seed}.zip и обучается с seed=seed,
      env.reset(seed=seed) — как в v1.1.

Запуск (ровно ОДИН раз, вручную, в окружении trading-agent):
    python stage2_only_v1_2.py
"""

import datetime
import glob
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
import optuna
from optuna.study import StudyDirection
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from portfolio_env import PortfolioEnv

# ═══════════════════════════════════════════════
# КОНФИГ РАЗМЕТКИ / ФАЙЛОВ
# ═══════════════════════════════════════════════

VERSION       = "v1.2"
FEATURES_FILE = "rl_features_v5.pkl"     # v1.2: 49 колонок (31 легаси + 18 новых)
PRICES_FILE   = "rl_prices_v4.pkl"       # файл цен не менялся
DB_FILE       = "optuna_ppo_v1_2.db"
STUDY_NAME    = "ppo_v1.2"

HOLDOUT_FOLD_EXPECTED = 5                # индекс последнего фолда по протоколу
SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042]

EXPECTED_OBS_DIM = 56                    # v1.2 = 49 фичей + 7 (6 весов + кэш)
N_WEIGHT_SLOTS   = 7                     # доли (6 тикеров + кэш)

EP_LEN_RANGE      = (30, 200)
RANDOM_START_PROB = 0.5

LOGS_DIR = "logs"

# Диагностический диапазон кэша «повадки победителя» v1.1 (0.12-0.14 в среднем)
CASH_DIAG_LO = 0.10
CASH_DIAG_HI = 0.30

# Допуск сверки lambda_hhi: в протоколе значение округлено до 4 знаков (0.0023),
# поэтому 1e-3*|expected| = 2.3e-6 давал бы ложный MISMATCH.
TOL_LAMBDA_HHI = 1e-4

# Ожидаемый лидер по протоколу AGENTS.md (2026-10-29 20:40)
EXPECTED_LEADER = {
    "trial":           14,
    "max_weight":      0.55,
    "lambda_hhi":      0.0023,
    "total_timesteps": 250_000,
    "dd_alpha":        8.38,
    "gamma":           0.998,
    "n_epochs":        4,
    "value":           0.0080,   # робастная метрика, в протоколе округлена до 4 знаков
}

# Ожидаемое распределение триалов по состояниям (для сверки глазами с протоколом):
# 21 закрытый триал = 16 COMPLETE + 6 PRUNED + прерванный хвост (RUNNING/FAIL).
EXPECTED_TRIAL_STATES = {"COMPLETE": 16, "PRUNED": 6}

# Запасные значения гиперпараметров. Источники (сверено по коду):
#   max_weight / lambda_hhi / dd_alpha — class-атрибуты PortfolioEnv.DEFAULT_*
#   total_timesteps — train_ppo.TOTAL_TIMESTEPS_DEFAULT
#   gamma / n_epochs — дефолты train_ppo.run_final_training
_PARAM_DEFAULTS = {
    "max_weight":      (PortfolioEnv.DEFAULT_MAX_WEIGHT,  "PortfolioEnv.DEFAULT_MAX_WEIGHT"),
    "lambda_hhi":      (PortfolioEnv.DEFAULT_LAMBDA_HHI,  "PortfolioEnv.DEFAULT_LAMBDA_HHI"),
    "total_timesteps": (300_000,                          "train_ppo.TOTAL_TIMESTEPS_DEFAULT"),
    "dd_alpha":        (PortfolioEnv.DEFAULT_DD_ALPHA,    "PortfolioEnv.DEFAULT_DD_ALPHA"),
    "gamma":           (0.99,                             "train_ppo.run_final_training"),
    "n_epochs":        (10,                               "train_ppo.run_final_training"),
}

_FOLDS_CACHE = None
_LOG_PATH = None
_TEE_OUT = None
_TEE_ERR = None


# ═══════════════════════════════════════════════
# ЛОГИРОВАНИЕ (tee в logs/ + консоль)
# ═══════════════════════════════════════════════

class _Tee:
    """Дублирует печать в консоль и в файл; каждая полная строка получает
    префикс "[HH:MM:SS] ".

    Устроено как ПОСТРОЧНЫЙ буфер (self._buf): write() может прийти кусками
    ("текст", затем "\\n") — строка считается завершённой только когда пришёл
    перевод строки, поэтому лишних пустых строк не возникает. '\r'
    (прогресс-бар tqdm/stable-baselines3) нормализуется в '\n', чтобы прогресс
    обучения попадал в лог построчно, а не висел в буфере.
    Пустая строка пишется без префикса — это просто разделитель.
    flush() закрывает неполную строку (дописывает её с переводом строки).
    Метод isatty() обязателен: optuna/colorlog и stable-baselines3 вызывают
    stream.isatty() у sys.stdout/sys.stderr и падают на потоках без него.
    """

    _PARTIAL_FLUSH = 4096    # предохранитель от бесконечного роста буфера

    def __init__(self, *streams):
        self.streams = streams
        self._buf = ""

    @staticmethod
    def _stamp():
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _fmt(self, line):
        """Одна завершённая строка -> текст для записи (ровно один '\\n')."""
        if not line:
            return "\n"
        return f"[{self._stamp()}] {line}\n"

    def _emit(self, payload):
        for st in self.streams:
            try:
                if st is None or getattr(st, "closed", False):
                    continue
                st.write(payload)
            except (ValueError, OSError):
                pass      # файл закрыли — пишем в остальные потоки

    def write(self, s):
        if s is None:
            return 0
        if isinstance(s, (bytes, bytearray)):
            s = bytes(s).decode("utf-8", "replace")
        elif not isinstance(s, str):
            s = str(s)
        if not s:
            return 0

        text = self._buf + s.replace("\r\n", "\n").replace("\r", "\n")
        parts = []
        while True:
            idx = text.find("\n")
            if idx < 0:
                break
            parts.append(self._fmt(text[:idx]))
            text = text[idx + 1:]
        if len(text) >= self._PARTIAL_FLUSH:        # гигантский кусок без '\n'
            parts.append(self._fmt(text))
            text = ""
        self._buf = text
        if parts:
            self._emit("".join(parts))
        return len(s)

    def flush(self):
        if self._buf:
            payload = self._fmt(self._buf)
            self._buf = ""
            self._emit(payload)
        for st in self.streams:
            try:
                if st is not None and not getattr(st, "closed", False):
                    st.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        # Не закрываем консольные потоки и общий файловый хэндл: это сделает
        # интерпретатор при выходе. Нам достаточно дослать остаток строки.
        self.flush()

    def isatty(self):
        for st in self.streams:
            try:
                if hasattr(st, "isatty"):
                    return bool(st.isatty())
            except (ValueError, OSError):
                pass
        return False

    @property
    def encoding(self):
        for st in self.streams:
            enc = getattr(st, "encoding", None)
            if enc:
                return enc
        return "utf-8"

    @property
    def buffer(self):
        # Страховка: если сторонний код пишет в sys.stdout.buffer.
        for st in self.streams:
            buf = getattr(st, "buffer", None)
            if buf is not None:
                return buf
        raise AttributeError("buffer")


def setup_logging():
    """Открыть logs/stage2_v1_2_<timestamp>.log и подменить stdout/stderr.

    Подменяются ТОЛЬКО sys.stdout / sys.stderr (как в train_ppo._Tee).
    Глобальная подмена встроенной print-функции НЕ делается: это лишний риск
    (SB3/optuna/gymnasium печатают неожиданно, откат не предусмотрен).
    """
    global _LOG_PATH, _TEE_OUT, _TEE_ERR

    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"stage2_v1_2_{stamp}.log")
    if os.path.exists(path):                      # два запуска в одну секунду
        path = os.path.join(LOGS_DIR,
                            f"stage2_v1_2_{stamp}_{int(time.time() % 1e6)}.log")
    _LOG_PATH = path

    fh = open(path, "a", encoding="utf-8", buffering=1)
    _TEE_OUT = _Tee(sys.__stdout__, fh)
    _TEE_ERR = _Tee(sys.__stderr__, fh)
    sys.stdout = _TEE_OUT
    sys.stderr = _TEE_ERR

    def _excepthook(t, v, tb):
        for line in traceback.format_exception(t, v, tb):
            _TEE_ERR.write(line)
        _TEE_ERR.flush()

    sys.excepthook = _excepthook

    print(f"[LOG] Лог пишется в {path}")
    return path


def flush_all():
    """Дослать остатки строк и сбросить буферы обоих потоков."""
    for st in (sys.stdout, sys.stderr):
        try:
            st.flush()
        except (ValueError, OSError):
            pass


# ═══════════════════════════════════════════════
# ФОЛДЫ / БЕНЧМАРКИ / ОБУЧЕНИЕ / ОЦЕНКА
#        (логика 1:1 из stage2_only.py)
# ═══════════════════════════════════════════════

def get_folds():
    """Границы фолдов из PortfolioEnv.build_folds по датасету v5 (кэш)."""
    global _FOLDS_CACHE
    if _FOLDS_CACHE is None:
        idx = pd.read_pickle(FEATURES_FILE).index
        _FOLDS_CACHE = PortfolioEnv.build_folds(idx)
    return _FOLDS_CACHE


def benchmarks_on_split(features, tickers, start, end):
    """Бенчмарки для участка [start, end) — 1:1 как в v1.1.

    eq_values[0] = 1.0, далее cumprod(1 + mean(returns)) по акциям.
    """
    rets = features[[f"{t}_returns" for t in tickers]].values
    segment = rets[start + 1: end + 1]
    eq_daily = segment.mean(axis=1)
    eq_values = np.concatenate([[1.0], np.cumprod(1 + eq_daily)])
    ticker_values = {}
    for i, t in enumerate(tickers):
        t_daily = segment[:, i]
        ticker_values[t] = np.concatenate([[1.0], np.cumprod(1 + t_daily)])
    cash_values = np.ones(len(eq_values))
    return eq_values, ticker_values, cash_values


def train_one_fold(fold, total_timesteps, max_weight, lambda_hhi, dd_alpha,
                   gamma, n_epochs, seed, verbose=0):
    """Обучить PPO на train-части фолда и вернуть модель (1:1 с v1.1)."""
    train_env = Monitor(PortfolioEnv(
        features_path=FEATURES_FILE,          # ЯВНО: v5, не дефолт v4
        prices_path=PRICES_FILE,
        episode_length_range=EP_LEN_RANGE,
        random_start_prob=RANDOM_START_PROB,
        eval_mode=False,
        fold=fold,
        seed=seed,
        max_weight=max_weight,
        lambda_hhi=lambda_hhi,
        dd_alpha=dd_alpha,
    ))
    model = PPO(
        "MlpPolicy", train_env,
        verbose=verbose,
        learning_rate=1e-4, n_steps=2048, batch_size=256, clip_range=0.1,
        n_epochs=n_epochs, gamma=gamma, ent_coef=0.01,
        seed=seed, device="auto",
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate(model, fold, eval_split, seed, max_weight=0.5, lambda_hhi=0.5):
    """Детерминированный прогон модели по сплиту фолда (1:1 с v1.1).

    Только предсказания (model.predict, deterministic=True): модель не обучается.
    Monitor здесь НЕ создаётся (лишняя обёртка в eval не нужна), на диск ничего
    не пишется — функция возвращает (эквити, история весов, env).
    """
    env = PortfolioEnv(
        features_path=FEATURES_FILE,          # ЯВНО: v5, не дефолт v4
        prices_path=PRICES_FILE,
        eval_mode=True,
        eval_split=eval_split,
        fold=fold,
        seed=seed,
        max_weight=max_weight,
        lambda_hhi=lambda_hhi,
    )
    obs, info = env.reset(seed=seed)
    values = [1.0]
    weights_history = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        values.append(info['portfolio_value'])
        weights_history.append(info['weights'].copy())
        if terminated or truncated:
            break
    return values, np.array(weights_history), env


# ═══════════════════════════════════════════════
# МЕТРИКИ
# ═══════════════════════════════════════════════

def max_drawdown_pct(values):
    """Максимальная просадка в % от running-пика эквити.

    Формула: max_t (peak_t - eq_t) / peak_t * 100, где peak — running max
    по массиву эквити, а первый элемент массива — стартовое эквити 1.0.
    Одинаковая формула для PPO и Equal.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    peak = np.maximum.accumulate(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peak - v) / peak * 100.0
    dd = dd[np.isfinite(dd)]
    return float(np.max(dd)) if dd.size else 0.0


def equal_weight_turnover(features, tickers, start, end):
    """Средний оборот ежедневного ребаланса Equal-Weight (справочно).

    Ежедневно веса дрейфуют вместе с доходностями, затем модель «возвращает»
    их к 1/N. Оборот шага = sum_i |1/N - w_дрейф_i|; усредняем по шагам
    test-окна. Кэша у Equal нет (cash = 0.0).
    """
    rets = features[[f"{t}_returns" for t in tickers]].values
    segment = rets[start + 1: end + 1]
    n = len(tickers)
    target = np.full(n, 1.0 / n, dtype=float)
    w = target.copy()
    turnovers = []
    for r in segment:
        turnovers.append(float(np.sum(np.abs(target - w))))
        port_ret = float(np.dot(target, r))
        denom = 1.0 + port_ret
        if abs(denom) > 1e-12:
            w = target * (1.0 + r) / denom
        else:
            w = target.copy()
    return float(np.mean(turnovers)) if turnovers else 0.0


def mean_turnover_ppo(weights_history):
    """Средний оборот агента за шаг: mean_t sum_i |w_i(t) - w_i(t-1)| (1:1 с v1.1)."""
    if len(weights_history) < 2:
        return 0.0
    return float(np.abs(np.diff(weights_history, axis=0)).mean(axis=1).mean())


# ═══════════════════════════════════════════════
# ГЕЙТЫ БЕЗОПАСНОСТИ (до обучения)
# ═══════════════════════════════════════════════

def guard_existing_models(fold_index):
    """Остановиться, если артефакты stage2 v1.2 уже существуют (holdout израсходован)."""
    prefix = f"ppo_{VERSION}_fold{fold_index}"
    found = []
    for seed in SEEDS:
        for cand in (f"{prefix}_s{seed}.zip", f"{prefix}_s{seed}"):
            if os.path.exists(cand):
                found.append(cand)
    found.extend(sorted(glob.glob(f"{prefix}_s*")))
    uniq = []
    for item in found:
        if item not in uniq:
            uniq.append(item)
    if uniq:
        print("ОШИБКА: артефакты stage2 v1.2 уже присутствуют "
              f"({', '.join(uniq)}). Holdout израсходован — повторный запуск "
              "запрещён без решения пользователя.")
        print("Файлы НЕ перезаписываются и НЕ удаляются автоматически.")
        flush_all()
        sys.exit(1)
    print(f"[GUARD] Моделей {prefix}_s*.zip нет — holdout-заход чистый.")


def check_obs_dim(tmp_env):
    """Проверить, что файл фичей действительно v5 (obs_dim = 56), иначе ОСТАНОВКА."""
    n_features = int(tmp_env.features.shape[1])
    calc_obs = n_features + N_WEIGHT_SLOTS
    env_obs = int(tmp_env.observation_space.shape[0])

    print(f"\n[OBS] features.shape[1] = {n_features}  ({FEATURES_FILE})")
    print(f"[OBS] расчётный obs_dim = {n_features} + {N_WEIGHT_SLOTS} = {calc_obs}")
    print(f"[OBS] env.observation_space.shape[0] = {env_obs}")

    if calc_obs != EXPECTED_OBS_DIM:
        print(f"ОШИБКА: obs_dim={calc_obs} , ожидалось {EXPECTED_OBS_DIM} "
              f"(v1.2 = 49 фичей + 7). Похоже, подсунули rl_features_v4.pkl — "
              f"ОСТАНОВКА.")
        flush_all()
        sys.exit(1)
    if env_obs != calc_obs:
        print(f"ОШИБКА: env.observation_space.shape[0]={env_obs} не совпадает "
              f"с расчётным obs_dim={calc_obs} — ОСТАНОВКА.")
        flush_all()
        sys.exit(1)

    print(f"[OBS] OK: obs_dim = {EXPECTED_OBS_DIM} (49 фичей + 7 весов)")


# ═══════════════════════════════════════════════
# ЧТЕНИЕ БАЗЫ OPTUNA (ТОЛЬКО READ-ONLY)
# ═══════════════════════════════════════════════

def _ro_uri_candidates(db_file):
    """URI-кандидаты для read-only открытия sqlite (рецепт plot_progress.py).

    Порядок: сначала абсолютный путь с прямыми слэшами (устойчив к смене cwd),
    затем ровно та форма, которая уже проверена в plot_progress.py.
    """
    db_abs = os.path.abspath(db_file).replace(os.sep, "/")
    db_rel = db_file.replace(os.sep, "/")
    cands = []
    cands.append(("абсолютный путь, read-only URI",
                  f"sqlite:///file:{db_abs}?mode=ro&uri=true"))
    if db_rel != db_abs:
        cands.append(("относительный путь (рецепт plot_progress.py), read-only URI",
                      f"sqlite:///file:{db_rel}?mode=ro&uri=true"))
    return cands


def load_study_readonly(db_file, study_name, attempts=3, retry_sleep=10):
    """Загрузить study строго read-only. Возвращает (study|None, источник|None).

    1) read-only URI (sqlite ?mode=ro&uri=true) к оригиналу — 2 варианта пути;
    2) если не вышло (WAL активного прогона / lock) — читаем КОПИЮ базы
       во временной папке (оригинал не модифицируется).
    Записи нет вообще: study только загружается (load_study), схема в
    оригинале не мигрируется, ни один триал не добавляется.
    """
    last_err = None
    for label, url in _ro_uri_candidates(db_file):
        for attempt in range(1, attempts + 1):
            try:
                storage = optuna.storages.RDBStorage(
                    url=url,
                    engine_kwargs={"connect_args": {"uri": True, "timeout": 30}},
                )
                study = optuna.load_study(study_name=study_name, storage=storage)
                print(f"[DB] открыто read-only ({label}): {url}")
                return study, f"{label}: {url}"
            except Exception as e:              # noqa: BLE001 — нужен широкий перехват
                last_err = e
                text = f"{type(e).__name__}: {e}"
                print(f"[WARN] read-only ({label}) попытка {attempt}/{attempts} "
                      f"не удалась: {text}")
                if "locked" in text.lower() and attempt < attempts:
                    print(f"[INFO] database is locked -> sleep {retry_sleep}s и повтор")
                    time.sleep(retry_sleep)
                    continue
                break

    print("[INFO] fallback: копирую базу во временную папку (оригинал не трогаем)")
    try:
        tmpdir = tempfile.mkdtemp(prefix="optuna_ro_")
        target = os.path.join(tmpdir, os.path.basename(db_file))
        shutil.copy2(db_file, target)      # shutil только ЧИТАЕТ оригинал
        for suf in ("-wal", "-shm"):
            src = db_file + suf
            if os.path.exists(src):
                try:
                    shutil.copy2(src, target + suf)
                except (OSError, shutil.Error) as e:
                    print(f"[WARN] не удалось скопировать {src}: {e}")

        target_url = f"sqlite:///file:{target.replace(os.sep, '/')}?mode=ro&uri=true"
        try:
            storage = optuna.storages.RDBStorage(
                url=target_url,
                engine_kwargs={"connect_args": {"uri": True, "timeout": 30}},
            )
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"[DB] открыта КОПИЯ read-only: {target}")
            return study, f"копия в tempdir (read-only): {target} (оригинал: {db_file})"
        except Exception as e:             # noqa: BLE001
            print(f"[WARN] копия не открылась read-only ({type(e).__name__}: {e}); "
                  f"открываю копию обычным образом — это ВРЕМЕННАЯ копия, "
                  f"оригинал {db_file} не затрагивается")
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{target.replace(os.sep, '/')}",
                engine_kwargs={"connect_args": {"timeout": 30}},
            )
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"[DB] открыта КОПИЯ (read-write по необходимости): {target}")
            return study, (f"копия в tempdir (оригинал не изменялся): {target} "
                           f"(оригинал: {db_file})")
    except Exception as e:                 # noqa: BLE001
        print(f"[ERROR] fallback-копия тоже не удалась: {type(e).__name__}: {e}")
        print(f"[ERROR] исходная ошибка read-only: {last_err}")
        return None, None


# ═══════════════════════════════════════════════
# СВЕРКА ЛИДЕРА С ПРОТОКОЛОМ
# ═══════════════════════════════════════════════

def _tol_for(key, expected):
    """Допуск сверки float-параметра.

    lambda_hhi в протоколе округлён до 4 знаков (0.0023) -> жёсткий 1e-4;
    остальные — max(1e-6, 1e-3*|expected|).
    """
    if key == "lambda_hhi":
        return TOL_LAMBDA_HHI
    return max(1e-6, 1e-3 * abs(float(expected)))


def _marker(actual, expected, tol):
    """Вернуть (marker, abs_diff) для сверки float-параметра."""
    if actual is None:
        return "MISMATCH", float("nan")
    diff = abs(float(actual) - float(expected))
    return ("OK" if diff <= tol else "MISMATCH"), diff


def report_and_verify_leader(study, source):
    """Напечатать направление/статистику/лидера, сверить с ожидаемым.

    Лидер берётся ТОЛЬКО через штатный study.best_trial (учитывает направление
    оптимизации). Если direction != MAXIMIZE — ОСТАНОВКА: самодельная сортировка
    по value в этом случае выбрала бы ХУДШИЙ триал.
    """
    direction = study.direction

    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    n_all = len(study.trials)
    n_complete = states.get("COMPLETE", 0)

    print(f"\n{'=' * 78}")
    print(f"  БАЗА OPTUNA (read-only): {DB_FILE} / study={STUDY_NAME}")
    print(f"  Источник: {source}")
    print(f"  Направление study: {direction.name}")
    print(f"  Триалов всего: {n_all}")
    print("  По состояниям: " +
          ", ".join(f"{k}={v}" for k, v in sorted(states.items())))
    print("  Ожидание по протоколу: " +
          ", ".join(f"{k}={v}" for k, v in sorted(EXPECTED_TRIAL_STATES.items())) +
          " + прерванный хвост (trial 21 закрыт после fold-прунинга)")
    print(f"{'=' * 78}")

    # ── F1: гейт направления ──
    if direction != StudyDirection.MAXIMIZE:
        print(f"ОШИБКА: study.direction = {direction.name} , ожидалось MAXIMIZE — "
              f"есть риск выбрать худший триал. ОСТАНОВКА.")
        flush_all()
        sys.exit(1)

    if n_complete == 0:
        print("ОШИБКА: в базе нет ни одного завершённого триала — брать лидера "
              "не из чего. ОСТАНОВКА.")
        flush_all()
        sys.exit(1)

    # Топ-5 — только для информации; сортировка с учётом направления.
    reverse = (direction == StudyDirection.MAXIMIZE)
    ranked = sorted([t for t in study.trials
                     if t.state.name == "COMPLETE" and t.value is not None],
                    key=lambda t: t.value, reverse=reverse)
    print("\nТоп-5 завершённых триалов по value:")
    for t in ranked[:5]:
        print(f"  trial {t.number:>2}: value={t.value:+.6f}  {t.params}")

    # ── F1: лидер через штатный API ──
    best_trial = study.best_trial
    print(f"\nФАКТИЧЕСКИЙ ЛИДЕР (study.best_trial): trial #{best_trial.number}, "
          f"robust value = {best_trial.value:+.6f} "
          f"(study.best_value = {study.best_value:+.6f})")
    if ranked and ranked[0].number != best_trial.number:
        print(f"[WARN] best_trial (#{best_trial.number}) не совпал с вершиной "
              f"ручной сортировки (#{ranked[0].number}) — проверь направление "
              f"и значения триалов.")

    print("Гиперпараметры лидера (полностью):")
    for k in sorted(best_trial.params.keys()):
        print(f"  {k:>16} = {best_trial.params[k]!r}")
    print(f"  params (dict)      = {best_trial.params}")
    if best_trial.user_attrs:
        print("user_attrs:")
        for k, v in best_trial.user_attrs.items():
            print(f"  {k:>16} = {v!r}")
    else:
        print("user_attrs: (пусто)")

    # ── Сверка с протоколом ──
    print(f"\n{'-' * 78}")
    print("  ОЖИДАЕМЫЕ ЗНАЧЕНИЯ ЛИДЕРА (по протоколу AGENTS.md)")
    print("  trial 14, max_weight=0.55, lambda_hhi=0.0023, total_timesteps=250000,")
    print("  dd_alpha=8.38, gamma=0.998, n_epochs=4, robust value ~ +0.0080")
    print(f"{'-' * 78}")
    print(f"  {'параметр':<17} {'факт':>14} {'ожидание':>12}   {'|diff|':>10}  "
          f"{'tol':>9}  маркер")
    print(f"  {'-' * 17} {'-' * 14} {'-' * 12}   {'-' * 10}  {'-' * 9}  {'-' * 8}")

    mismatches = []

    # trial number — точное совпадение
    m_trial = "OK" if int(best_trial.number) == int(EXPECTED_LEADER["trial"]) else "MISMATCH"
    print(f"  {'trial':<17} {best_trial.number:>14} {EXPECTED_LEADER['trial']:>12}   "
          f"{'—':>10}  {'0':>9}  {m_trial}")
    if m_trial != "OK":
        mismatches.append(f"trial: факт {best_trial.number}, "
                          f"ожидание {EXPECTED_LEADER['trial']}")

    # float-параметры — tol из _tol_for (lambda_hhi отдельно)
    for key in ("max_weight", "lambda_hhi", "total_timesteps", "dd_alpha",
                "gamma", "n_epochs"):
        actual = best_trial.params.get(key)
        exp = EXPECTED_LEADER[key]
        tol = _tol_for(key, exp)
        m, diff = _marker(actual, exp, tol)
        a_txt = f"{actual!r}" if actual is not None else "None"
        print(f"  {key:<17} {a_txt:>14} {exp:>12}   {diff:>10.6g}  "
              f"{tol:>9.3g}  {m}")
        if m != "OK":
            mismatches.append(f"{key}: факт {actual!r}, ожидание {exp} "
                              f"(|diff|={diff:.3g} > tol={tol:.3g})")

    # robust value — информативно (в протоколе округлено до 4 знаков)
    tol_val = 5e-4
    m_val, diff_val = _marker(best_trial.value, EXPECTED_LEADER["value"], tol_val)
    print(f"  {'robust value':<17} {best_trial.value:>14.6f} "
          f"{EXPECTED_LEADER['value']:>12.4f}   {diff_val:>10.6g}  "
          f"{tol_val:>9.3g}  {m_val}   (информативно)")

    if mismatches:
        print("")
        print("*" * 78)
        print("**  ВНИМАНИЕ: ЛИДЕР НЕ СОВПАДАЕТ С ОЖИДАЕМЫМ ИЗ ПРОТОКОЛА          **")
        print("**  Сверь с протоколом AGENTS.md ПЕРЕД запуском: возможно, база     **")
        print("**  изменилась, прогон продолжался или номера триалов сдвинулись.   **")
        print("**  Обучение всё равно пойдёт по ФАКТИЧЕСКОМУ лидеру (см. выше).    **")
        for item in mismatches:
            print(f"**    - {item}")
        print("*" * 78)
        print("")
    else:
        print("\n  [OK] Все параметры лидера совпадают с ожидаемыми по протоколу.")

    return best_trial


def resolve_hyperparams(params):
    """F2: достать гиперпараметры лидера с запасными значениями (без KeyError)."""
    hp = {}
    for key, (default, src) in _PARAM_DEFAULTS.items():
        value = params.get(key)
        if value is None:
            print(f"[WARN] параметр {key} не найден в триале, взят дефолт "
                  f"{default!r} (источник: {src})")
            value = default
        hp[key] = value
    hp["total_timesteps"] = int(hp["total_timesteps"])
    hp["n_epochs"] = int(hp["n_epochs"])
    hp["max_weight"] = float(hp["max_weight"])
    hp["lambda_hhi"] = float(hp["lambda_hhi"])
    hp["dd_alpha"] = float(hp["dd_alpha"])
    hp["gamma"] = float(hp["gamma"])
    return hp


# ═══════════════════════════════════════════════
# ПЕЧАТЬ ТАБЛИЦ
# ═══════════════════════════════════════════════

_HEADER = (f"{'| seed':>6} | {'ret_PPO %':>10} | {'ret_Equal %':>11} | "
           f"{'diff п.п.':>9} | {'maxDD_PPO %':>11} | {'maxDD_Equal %':>12} | "
           f"{'mean_cash':>9} | {'mean_turn':>9} |")
_SEP = ("-" * 6 + "-+-" + "-" * 10 + "-+-" + "-" * 11 + "-+-" + "-" * 9 + "-+-" +
        "-" * 11 + "-+-" + "-" * 12 + "-+-" + "-" * 9 + "-+-" + "-" * 9 + "-+")


def print_table_header():
    print(_HEADER)
    print(_SEP)


def print_table_row(label, r):
    """r — dict с ключами ret_ppo, ret_eq, dd_ppo, dd_eq, cash, turn."""
    print(f"{label:>6} | "
          f"{r['ret_ppo']:>10.2f} | "
          f"{r['ret_eq']:>11.2f} | "
          f"{r['ret_ppo'] - r['ret_eq']:>+9.2f} | "
          f"{r['dd_ppo']:>11.2f} | "
          f"{r['dd_eq']:>12.2f} | "
          f"{r['cash']:>9.3f} | "
          f"{r['turn']:>9.4f} |")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    setup_logging()

    t_start_all = time.time()
    print("=" * 78)
    print("  STAGE 2 — HOLDOUT v1.2 (ЕДИНСТВЕННЫЙ ЗАХОД)")
    print("=" * 78)
    print(f"  VERSION       = {VERSION}")
    print(f"  FEATURES_FILE = {FEATURES_FILE}")
    print(f"  PRICES_FILE   = {PRICES_FILE}")
    print(f"  DB            = {DB_FILE} (read-only)")
    print(f"  STUDY         = {STUDY_NAME}")
    print(f"  SEEDS         = {SEEDS}")
    print(f"  Старт: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Лог: {_LOG_PATH}")

    # ── 1. Предохранитель повторного захода (ДО обучения) ──
    guard_existing_models(HOLDOUT_FOLD_EXPECTED)

    # ── 2. Разметка фолдов + гейт obs_dim (ДО обучения и ДО чтения базы) ──
    folds = get_folds()
    holdout_fold = len(folds) - 1
    if holdout_fold != HOLDOUT_FOLD_EXPECTED:
        print(f"[WARN] Индекс последнего фолда = {holdout_fold}, а по протоколу "
              f"ожидался {HOLDOUT_FOLD_EXPECTED}. Разметка фолдов изменилась? "
              f"Числа будут несопоставимы с v1.1.")
    ts, te, vs, ve = folds[holdout_fold]

    # Временная среда: features/tickers + гейт obs_dim
    tmp_env = PortfolioEnv(
        features_path=FEATURES_FILE,          # ЯВНО: v5, не дефолт v4
        prices_path=PRICES_FILE,
        fold=holdout_fold, eval_mode=True, eval_split='test',
    )
    features, tickers = tmp_env.features, tmp_env.tickers
    check_obs_dim(tmp_env)

    d_start = features.index[vs].date()
    d_end = features.index[ve - 1].date()
    n_test_rows = ve - vs
    print(f"\nHOLDOUT fold {holdout_fold} (последний):")
    print(f"  train [{features.index[ts].date()} .. {features.index[te - 1].date()}] "
          f"({te - ts} строк)")
    print(f"  test  [{d_start} .. {d_end}] ({n_test_rows} строк)")
    print(f"  Эмбарго/окна НЕ менялись (TRAIN_WINDOW={PortfolioEnv.TRAIN_WINDOW}, "
          f"TEST_WINDOW={PortfolioEnv.TEST_WINDOW}, EMBARGO={PortfolioEnv.EMBARGO}, "
          f"HOLDOUT_ROWS={PortfolioEnv.HOLDOUT_ROWS})")

    # ── 3. Лидер из базы Optuna (read-only) ──
    if not os.path.exists(DB_FILE):
        print(f"ОШИБКА: база Optuna не найдена: {os.path.abspath(DB_FILE)} — "
              f"ОСТАНОВКА.")
        flush_all()
        sys.exit(1)

    study, source = load_study_readonly(DB_FILE, STUDY_NAME)
    if study is None:
        print("ОШИБКА: не удалось прочитать базу Optuna ни read-only, ни из копии.")
        flush_all()
        sys.exit(1)

    best_trial = report_and_verify_leader(study, source)
    hp = resolve_hyperparams(best_trial.params)

    max_weight      = hp["max_weight"]
    lambda_hhi      = hp["lambda_hhi"]
    total_timesteps = hp["total_timesteps"]
    dd_alpha        = hp["dd_alpha"]
    gamma           = hp["gamma"]
    n_epochs        = hp["n_epochs"]

    print(f"\nГиперпараметры запуска: mw={max_weight:.2f}, "
          f"λhhi={lambda_hhi:.6f}, ts={total_timesteps:,}, "
          f"dd_alpha={dd_alpha:.2f}, γ={gamma}, ep={n_epochs}")

    # ── 4. Бенчмарк Equal на test-окне ──
    eq_values, tick_values, _cash_values = benchmarks_on_split(
        features, tickers, vs, ve - 1)
    eq_ret = (eq_values[-1] - 1) * 100
    eq_dd = max_drawdown_pct(eq_values)
    eq_turn = equal_weight_turnover(features, tickers, vs, ve - 1)

    print(f"\nEqual-Weight на test-окне: {eq_ret:+.2f}%  (эталон v1.1: -10.91%)")
    print(f"  maxDD_Equal = {eq_dd:.2f}%   справочный оборот ребаланса = {eq_turn:.4f}")
    print("  По тикерам:")
    for t in tickers:
        print(f"    {t:5s}: {(tick_values[t][-1] - 1) * 100:+.2f}%")

    print("\n  Формулы метрик (PPO и Equal — одинаковые):")
    print("    ret %      = (eq_last - 1) * 100, эквити стартует с 1.0")
    print("    maxDD %    = max_t (peak_t - eq_t) / peak_t * 100,")
    print("                 peak = running max по эквити, старт peak = eq_0 = 1.0")
    print("    mean_cash  = средний вес кэша агента за test-окно (доля 0..1)")
    print("    mean_turn  = средний за шаг sum_i |w_i(t) - w_i(t-1)| (оборот)")
    print("    Equal: cash = 0.0 (кэша нет), оборот — ребаланс к 1/N (справочно)")

    # ── 5. Обучение и оценка по 7 сидам ──
    print(f"\n{'=' * 78}")
    print(f"  ОБУЧЕНИЕ НА TRAIN-ЧАСТИ FOLD {holdout_fold}, ОЦЕНКА НА TEST-ОКНЕ")
    print(f"  {len(SEEDS)} сидов, {total_timesteps:,} шагов каждый")
    print(f"{'=' * 78}")

    rows = []
    print("\n" + _HEADER)
    print(_SEP)

    for seed in SEEDS:
        t_seed = time.time()
        t_seed_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n--- Сид {seed}: старт {t_seed_str} ---")

        model = train_one_fold(holdout_fold, total_timesteps, max_weight,
                               lambda_hhi, dd_alpha, gamma, n_epochs, seed,
                               verbose=1)

        model_path = f"ppo_{VERSION}_fold{holdout_fold}_s{seed}"
        model.save(model_path)
        print(f"  Модель сохранена: {model_path}.zip")

        values, w, _env = evaluate(  # noqa: F841 — env не нужен, keep 1:1 с v1.1
            model, holdout_fold, 'test', seed, max_weight, lambda_hhi)

        ret_ppo = (values[-1] - 1) * 100
        dd_ppo = max_drawdown_pct(values)
        cash = float(w[:, -1].mean())
        turn = mean_turnover_ppo(w)

        row = {
            'seed': seed,
            'ret_ppo': ret_ppo,
            'ret_eq': eq_ret,
            'dd_ppo': dd_ppo,
            'dd_eq': eq_dd,
            'cash': cash,
            'turn': turn,
        }
        rows.append(row)
        print_table_row(str(seed), row)

        t_elapsed = time.time() - t_seed
        print(f"  Сид {seed}: закончен в "
              f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} "
              f"(заняло {t_elapsed / 60:.1f} мин)")
        print(f"  Сид {seed}: PPO {ret_ppo:+.2f}% | Equal {eq_ret:+.2f}% | "
              f"diff {ret_ppo - eq_ret:+.2f} п.п. | "
              f"maxDD {dd_ppo:.2f}% vs {eq_dd:.2f}% | "
              f"кэш {cash:.3f} | оборот/шаг {turn:.4f}")
        print("  Финальные веса:")
        for t, wv in zip(tickers + ["CASH"], w[-1]):
            print(f"    {t:5s}: {wv:.3f}")
        flush_all()

    # ── 6. Агрегат по сидам ──
    arr_ret_ppo = np.array([r['ret_ppo'] for r in rows], dtype=float)
    arr_dd_ppo = np.array([r['dd_ppo'] for r in rows], dtype=float)
    arr_cash = np.array([r['cash'] for r in rows], dtype=float)
    arr_turn = np.array([r['turn'] for r in rows], dtype=float)

    mean_ret_ppo = float(arr_ret_ppo.mean())
    std_ret_ppo = float(arr_ret_ppo.std())
    mean_dd_ppo = float(arr_dd_ppo.mean())
    std_dd_ppo = float(arr_dd_ppo.std())
    mean_cash = float(arr_cash.mean())
    std_cash = float(arr_cash.std())
    mean_turn = float(arr_turn.mean())
    std_turn = float(arr_turn.std())
    min_ret_ppo = float(arr_ret_ppo.min())
    max_ret_ppo = float(arr_ret_ppo.max())
    n_above_eq = int(np.sum(arr_ret_ppo >= eq_ret))
    n_both_pass = int(np.sum((arr_ret_ppo >= eq_ret) & (arr_dd_ppo < eq_dd)))

    agg = {
        'ret_ppo': mean_ret_ppo, 'ret_eq': eq_ret,
        'dd_ppo': mean_dd_ppo, 'dd_eq': eq_dd,
        'cash': mean_cash, 'turn': mean_turn,
    }

    print(f"\n{'=' * 78}")
    print("  АГРЕГАТ ПО 7 СИДАМ (среднее и разброс)")
    print(f"{'=' * 78}")
    print_table_header()
    print_table_row("СРЕДН", agg)
    print(f"\n  ret_PPO:     mean = {mean_ret_ppo:+.2f}%  std = {std_ret_ppo:.2f} п.п.  "
          f"min = {min_ret_ppo:+.2f}%  max = {max_ret_ppo:+.2f}%")
    print(f"  ret_Equal:   {eq_ret:+.2f}% (фиксирован для всех сидов)")
    print(f"  maxDD_PPO:   mean = {mean_dd_ppo:.2f}%  std = {std_dd_ppo:.2f} п.п.")
    print(f"  maxDD_Equal: {eq_dd:.2f}% (фиксирован для всех сидов)")
    print(f"  mean_cash:   mean = {mean_cash:.3f}  std = {std_cash:.3f}  "
          f"(диапазон по сидам {arr_cash.min():.3f}..{arr_cash.max():.3f})")
    print(f"  mean_turn:   mean = {mean_turn:.4f}  std = {std_turn:.4f}")
    print(f"  Сидов выше Equal по доходности: {n_above_eq} / {len(SEEDS)}")
    print(f"  Сидов PASS по ОБОИМ критериям индивидуально: {n_both_pass} / {len(SEEDS)}")

    cash_marker = "OK" if (CASH_DIAG_LO <= mean_cash <= CASH_DIAG_HI) else "ВНЕ ДИАПАЗОНА"
    print(f"\n  Диагностика кэша (не критерий вердикта): mean_cash = {mean_cash:.3f} "
          f"при ориентире v1.1 {CASH_DIAG_LO:.2f}-{CASH_DIAG_HI:.2f}  -> {cash_marker}")
    if cash_marker != "OK":
        print("    ПРИМЕЧАНИЕ: кэш вне «повадки победителя» v1.1 (0.12-0.14). "
              "Это диагностика, вердикт по ней НЕ выносится.")

    # ── 7. Итоговый вердикт ──
    crit1_ok = mean_ret_ppo >= eq_ret
    crit2_ok = mean_dd_ppo < eq_dd
    verdict_pass = crit1_ok and crit2_ok
    exit_code = 0 if verdict_pass else 2

    print(f"\n{'=' * 78}")
    print("  === ВЕРДИКТ v1.2 HOLDOUT ===")
    print(f"  Критерий 1 (доходность): mean_ret_PPO = {mean_ret_ppo:.2f}% vs "
          f"mean_ret_Equal = {eq_ret:.2f}%  -> {'OK' if crit1_ok else 'FAIL'}")
    print(f"  Критерий 2 (просадка):   mean_maxDD_PPO = {mean_dd_ppo:.2f}% < "
          f"mean_maxDD_Equal = {eq_dd:.2f}% -> {'OK' if crit2_ok else 'FAIL'}")
    print("  ИТОГ: PASS, если mean_PPO >= mean_Equal и maxDD_PPO < maxDD_Equal, "
          "иначе FAIL")
    print(f"  => {'PASS' if verdict_pass else 'FAIL'}")
    print(f"  Сидов PASS по обоим критериям индивидуально: {n_both_pass} / {len(SEEDS)}")
    print(f"  Код возврата: {exit_code} (0 = PASS, 2 = FAIL)")
    print(f"{'=' * 78}")

    t_total = time.time() - t_start_all
    print(f"\n  Окончание: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"(всего {t_total / 60:.1f} мин)")
    print(f"  Лог: {_LOG_PATH}")
    print(f"  Модели: ppo_{VERSION}_fold{holdout_fold}_s*.zip "
          f"({len(SEEDS)} файлов, сиды {SEEDS})")
    flush_all()

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
