"""
Обучение PPO-агента для Мосбирже v1.0.
Changes from v0.9:
  - Optuna: расширенные диапазоны (lambda_hhi 1e-4..0.1, dd_alpha 0.1..20)
  - Optuna: добавлены gamma (0.99/0.995/0.998) и n_epochs (3-10)
  - Optuna: 40 триалов, мульти-сид внутри триала (2 сида на фолд)
  - Робастная метрика: mean - LAMBDA_VAR * std по фолдам
  - Файловое хранилище study (sqlite) — старые триалы не теряются
  - В логе фолдов: доходность Equal и доля кэша
  - Сводка: fold 5 помечен как HOLDOUT
  - График истории оптимизации (best value vs trial)
  - Мульти-сид проверка лучших параметров (3 сида в финальном обучении)
"""

from portfolio_env import PortfolioEnv
import optuna
import warnings
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner

import pandas as pd

FEATURES_FILE = 'rl_features_v4.pkl'
PRICES_FILE = 'rl_prices_v4.pkl'
_FOLDS_CACHE = None

def get_folds():
    """Границы фолдов из build_folds по фактическому датасету v4 (кэш)."""
    global _FOLDS_CACHE
    if _FOLDS_CACHE is None:
        idx = pd.read_pickle(FEATURES_FILE).index
        _FOLDS_CACHE = PortfolioEnv.build_folds(idx)
    return _FOLDS_CACHE

VERSION = "v1.1"
SEED = 42
N_FOLDS = 6
N_TRIALS = 40              # число триалов Optuna
OPT_FOLDS = 5              # Optuna смотрит folds 0-4; fold 5 — holdout
N_SEEDS_TRIAL = 2          # мульти-сид внутри триала (п.7)
SEEDS_TRIAL = [SEED, SEED + 1000]
N_SEEDS_FINAL = 3          # мульти-сид проверка финальных параметров (п.11)
FINAL_SEEDS = [SEED, SEED + 1000, SEED + 2000]

# Диапазон случайной длины тренировочного эпизода (п.1) — в среду, НЕ в Optuna
EP_LEN_RANGE = (30, 200)
# Вероятность случайного стартового портфеля (п.3)
RANDOM_START_PROB = 0.5

# Вес штрафа за разброс фолдов в робастной метрике (п.8): M = mean - LAMBDA_VAR * std
LAMBDA_VAR = 0.5

TOTAL_TIMESTEPS_DEFAULT = 300_000

USE_SORTINO_RATIO = False  # Теперь ищем максимальную чистую доходность PPO
DOWNSIDE_DEV_THRESHOLD = -0.001

# ═══════════════════════════════════════════════
# ФУНКЦИИ
# ═══════════════════════════════════════════════

def evaluate(model, fold, eval_split, seed, max_weight=0.5, lambda_hhi=0.5):
    """Прогон модели по указанному сплиту фолда."""
    env = PortfolioEnv(
        features_path=FEATURES_FILE,
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


def train_one_fold(fold, total_timesteps, max_weight, lambda_hhi, dd_alpha,
                   gamma, n_epochs, seed, verbose=0):
    """Обучить PPO на одном фолде и вернуть модель."""
    train_env = Monitor(PortfolioEnv(
        features_path=FEATURES_FILE,
        prices_path=PRICES_FILE,
        episode_length_range=EP_LEN_RANGE,     # п.1: случайная длина эпизода
        random_start_prob=RANDOM_START_PROB,   # п.3: случайный стартовый портфель
        eval_mode=False,
        fold=fold,
        seed=seed,
        max_weight=max_weight,
        lambda_hhi=lambda_hhi,
        dd_alpha=dd_alpha,
    ))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        clip_range=0.1,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=0.01,
        seed=seed,
        device="auto",
    )

    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_fold(model, fold, max_weight, lambda_hhi):
    """Оценка модели на train и test фолда. Возвращает метрики."""
    # Нужны features и tickers — создаём временную среду (folds считаются в ней)
    tmp_env = PortfolioEnv(fold=fold, eval_mode=True, eval_split='test')
    ts, te, vs, ve = tmp_env.folds[fold]
    features = tmp_env.features
    tickers = tmp_env.tickers

    # Train
    ppo_train, w_train, env_train = evaluate(
        model, fold, 'train', SEED, max_weight, lambda_hhi)
    eq_train, tick_train, cash_train = benchmarks_on_split(
        features, tickers, ts, te - 1)

    # Test
    ppo_test, w_test, env_test = evaluate(
        model, fold, 'test', SEED, max_weight, lambda_hhi)
    eq_test, tick_test, cash_test = benchmarks_on_split(
        features, tickers, vs, ve - 1)

    return {
        'fold': fold,
        'train_start': ts, 'train_end': te,
        'test_start': vs, 'test_end': ve,
        'ppo_train': ppo_train, 'ppo_test': ppo_test,
        'w_train': w_train, 'w_test': w_test,
        'eq_train': eq_train, 'eq_test': eq_test,
        'tick_train': tick_train, 'tick_test': tick_test,
        'cash_train': cash_train, 'cash_test': cash_test,
        'ppo_train_ret': (ppo_train[-1] - 1) * 100,
        'ppo_test_ret': (ppo_test[-1] - 1) * 100,
        'eq_test_ret': (eq_test[-1] - 1) * 100,
        'turnover_test': np.abs(np.diff(w_test, axis=0)).mean(axis=1).mean(),
        'cash_share_test': w_test[:, -1].mean(),
        'final_weights': w_test[-1],
    }


# ──────────────────────────
# OPTUNA OBJECTIVE
# ──────────────────────────

def calculate_sortino_ratio(rets, threshold=-0.001):
    """
    Вычисляет Sortino Ratio по массиву дневных доходностей.
    Returns:
        float: Значение Sortino Ratio или -inf, если потерь нет.
    """
    daily_rets = np.diff(rets) / rets[:-1]

    avg_return = daily_rets.mean()

    down_rets = daily_rets[daily_rets < threshold]

    if len(down_rets) == 0:
        return float('-inf')

    down_deviation = down_rets.std(ddof=0)
    if down_deviation == 0:
        return float('-inf')

    sortino = avg_return / down_deviation
    return sortino


def robust_metric(fold_metrics):
    """
    Робастная метрика по фолдам (п.8):
        M = mean(folds) - LAMBDA_VAR * std(folds)
    Штрафует триалы, которые всплеснули на одном фолде, а на других провалились.
    """
    return float(np.mean(fold_metrics) - LAMBDA_VAR * np.std(fold_metrics))


def objective(trial):
    # Внешние параметры среды
    max_weight = trial.suggest_float("max_weight", 0.3, 0.6, step=0.05)
    lambda_hhi = trial.suggest_float("lambda_hhi", 0.0001, 0.1, log=True)  # расширен диапазон
    total_timesteps = trial.suggest_categorical(
        "total_timesteps", [150_000, 200_000, 250_000])

    # Внутренняя структура награды (расширен диапазон)
    dd_alpha = trial.suggest_float("dd_alpha", 0.1, 20, log=True)

    # Параметры обучения PPO (п.2 и п.10: gamma и n_epochs в поиск)
    gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.998])
    n_epochs = trial.suggest_int("n_epochs", 3, 10)

    # Оптимизируемся только на folds 0-4, fold 5 — holdout
    fold_metrics = []
    for fold in range(OPT_FOLDS):
        seed_metrics = []
        for seed in SEEDS_TRIAL:  # мульти-сид (п.7): метрика фолда = среднее по сидам
            model = train_one_fold(fold, total_timesteps, max_weight, lambda_hhi,
                                   dd_alpha, gamma, n_epochs, seed)

            result = evaluate_fold(model, fold, max_weight, lambda_hhi)

            metric = (
                calculate_sortino_ratio(result['ppo_test'], DOWNSIDE_DEV_THRESHOLD)
                if USE_SORTINO_RATIO else
                result['ppo_test_ret'] / 100
            )
            seed_metrics.append(metric)

            print(f"    fold {fold} seed {seed}: {metric:+.3f} | "
                  f"eq={result['eq_test_ret']/100:+.3f} | "
                  f"cash={result['cash_share_test']:.2f}")

        fold_metrics.append(np.mean(seed_metrics))

        # ── Pruner: репортим робастный промежуточный результат ──
        r_so_far = robust_metric(fold_metrics)
        trial.report(r_so_far, step=fold)

        print(f"    fold {fold}: {fold_metrics[-1]:+.3f} | "
              f"robust={r_so_far:+.3f}")

        if trial.should_prune():
            print(f"    → Trial {trial.number} pruned после fold {fold}")
            raise TrialPruned()

    avg_return = robust_metric(fold_metrics)
    print(f"  Trial {trial.number}: "
          f"mw={max_weight:.2f} λhhi={lambda_hhi:.4f} ts={total_timesteps:,}"
          f" | α_dd={dd_alpha:.2f} | γ={gamma} | ep={n_epochs}"
          f" → robust_test={avg_return:+.3f}")

    return avg_return


# ──────────────────────────
# ФИНАЛЬНОЕ ОБУЧЕНИЕ (с лучшими параметрами)
# ──────────────────────────

def run_final_training(best_params, seed=SEED, save_models=True):
    """Обучить финальные модели на всех 6 фолдах (5 opt + holdout)."""
    max_weight = best_params['max_weight']
    lambda_hhi = best_params['lambda_hhi']
    total_timesteps = best_params['total_timesteps']
    dd_alpha = best_params.get('dd_alpha', PortfolioEnv.DEFAULT_DD_ALPHA)
    gamma = best_params.get('gamma', 0.99)
    n_epochs = best_params.get('n_epochs', 10)

    print(f"\n{'=' * 60}")
    print(f"  ФИНАЛЬНОЕ ОБУЧЕНИЕ (seed={seed})")
    print(f"  max_weight      = {max_weight:.2f}")
    print(f"  lambda_hhi      = {lambda_hhi:.4f}")
    print(f"  timesteps       = {total_timesteps:,}")
    print(f"  DD_ALPHA        = {dd_alpha:.2f} (штраф за просадку)")
    print(f"  gamma           = {gamma}")
    print(f"  n_epochs        = {n_epochs}")
    print(f"{'=' * 60}\n")

    all_results = []

    for fold in range(N_FOLDS):
        ts, te, vs, ve = get_folds()[fold]

        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold}: train [{ts}:{te}] ({te-ts} дн.)  "
              f"test [{vs}:{ve}] ({ve-vs} дн.)")
        print(f"{'=' * 60}\n")

        model = train_one_fold(fold, total_timesteps, max_weight, lambda_hhi,
                               dd_alpha, gamma, n_epochs, seed, verbose=1)

        if save_models:
            model_path = f"ppo_portfolio_{VERSION}_fold{fold}_s{seed}"
            model.save(model_path)
            print(f"\nМодель сохранена: {model_path}.zip\n")

        result = evaluate_fold(model, fold, max_weight, lambda_hhi)

        # Печать результатов
        print(f"--- Fold {fold} результаты (seed={seed}) ---")
        print(f"  TRAIN PPO:    {result['ppo_train_ret']:+.2f}%")
        print(f"  TEST  PPO:    {result['ppo_test_ret']:+.2f}%")
        print(f"  TEST  Equal:  {result['eq_test_ret']:+.2f}%")
        tickers = PortfolioEnv(fold=fold, eval_mode=True).tickers
        for t in tickers:
            t_ret = (result["tick_test"][t][-1] - 1) * 100
            print(f"  TEST  {t:5s}: {t_ret:+.2f}%")
        print(f"  TEST  Cash:   {(result['cash_test'][-1] - 1) * 100:+.2f}%")
        print(f"  Оборот/шаг:   {result['turnover_test']:.4f}")
        print(f"  Доля кэша:    {result['cash_share_test']:.2f}")
        print(f"  Финальные веса:")
        for t, w in zip(tickers + ["CASH"], result["final_weights"]):
            print(f"    {t:5s}: {w:.3f}")

        all_results.append(result)

    return all_results


def print_summary(all_results, seed=None):
    """Сводная таблица по всем фолдам. Fold 5 помечен как HOLDOUT (п.15)."""
    title = "  СВОДКА ПО ВСЕМ ФОЛДАМ" + (f" (seed={seed})" if seed is not None else "")
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"{'Fold':>10} | {'Train PPO':>10} | {'Test PPO':>10} | "
          f"{'Test Equal':>10} | {'Оборот':>8} | {'Кэш':>5}")
    print(f"{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 5}")

    for r in all_results:
        fold_label = f"{r['fold']} (HOLD)" if r['fold'] == N_FOLDS - 1 else f"{r['fold']}"
        print(f"{fold_label:>10} | {r['ppo_train_ret']:>+9.2f}% | "
              f"{r['ppo_test_ret']:>+9.2f}% | {r['eq_test_ret']:>+9.2f}% | "
              f"{r['turnover_test']:>8.4f} | {r['cash_share_test']:>5.2f}")

    # Средние отдельно по подборным фолдам и по holdout (п.15: не сравнивать на равных)
    opt_res = [r for r in all_results if r['fold'] < OPT_FOLDS]
    hold_res = [r for r in all_results if r['fold'] == N_FOLDS - 1]
    avg_train = np.mean([r['ppo_train_ret'] for r in all_results])
    avg_test_opt = np.mean([r['ppo_test_ret'] for r in opt_res])
    avg_eq_opt = np.mean([r['eq_test_ret'] for r in opt_res])
    print(f"{'AVG (0-2)':>10} | {'':>10} | {avg_test_opt:>+9.2f}% | "
          f"{avg_eq_opt:>+9.2f}% |")
    if hold_res:
        r = hold_res[0]
        print(f"{'HOLDOUT':>10} | {'':>10} | {r['ppo_test_ret']:>+9.2f}% | "
              f"{r['eq_test_ret']:>+9.2f}% |")
    print(f"{'AVG (all)':>10} | {avg_train:>+9.2f}% |\n")


def plot_optimization_history(study):
    """График истории оптимизации: best value vs номер триала (п.16)."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("Нет завершённых триалов — график истории не строим")
        return

    completed.sort(key=lambda t: t.number)
    xs, best_so_far = [], []
    cur_best = -np.inf
    for t in completed:
        cur_best = max(cur_best, t.value)
        xs.append(t.number)
        best_so_far.append(cur_best)

    plt.figure(figsize=(10, 5))
    plt.plot(xs, best_so_far, marker='o', color='royalblue', linewidth=2)
    plt.xlabel("Номер триала")
    plt.ylabel("Best value (running)")
    plt.title(f"Optuna: история оптимизации ({VERSION})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'ppo_{VERSION}_optuna_history.png', dpi=150)
    print(f"График истории оптимизации: ppo_{VERSION}_optuna_history.png")


def multi_seed_check(best_params):
    """
    Мульти-сид проверка лучших параметров (п.11).
    Прогоняет финальное обучение на N_SEEDS_FINAL сидaх и печатает сводную
    таблицу: если разброс между сидами сравним с самим результатом —
    метрика пока шум.
    """
    print(f"\n{'='*60}")
    print(f"  МУЛЬТИ-СИД ПРОВЕРКА ЛУЧШИХ ПАРАМЕТРОВ ({N_SEEDS_FINAL} сидa)")
    print(f"{'='*60}")

    rows = []
    for seed in FINAL_SEEDS:
        results = run_final_training(best_params, seed=seed, save_models=(seed == SEED))
        print_summary(results, seed=seed)
        opt_res = [r for r in results if r['fold'] < OPT_FOLDS]
        hold_res = [r for r in results if r['fold'] == N_FOLDS - 1]
        rows.append({
            'seed': seed,
            'avg_test_opt': np.mean([r['ppo_test_ret'] for r in opt_res]),
            'avg_eq_opt': np.mean([r['eq_test_ret'] for r in opt_res]),
            'holdout_ppo': hold_res[0]['ppo_test_ret'] if hold_res else np.nan,
            'holdout_eq': hold_res[0]['eq_test_ret'] if hold_res else np.nan,
        })
        # Графики строим только для базового сида
        if seed == SEED:
            save_plots(results)

    print(f"\n{'='*60}")
    print(f"  СВОДКА МУЛЬТИ-СИД ПРОВЕРКИ")
    print(f"{'='*60}")
    print(f"{'Seed':>8} | {'AVG test PPO (0-2)':>18} | {'AVG Equal (opt)':>15} | "
          f"{'HOLDOUT PPO':>12} | {'HOLDOUT Equal':>14}")
    for r in rows:
        print(f"{r['seed']:>8} | {r['avg_test_opt']:>+17.2f}% | "
              f"{r['avg_eq_opt']:>+14.2f}% | {r['holdout_ppo']:>+11.2f}% | "
              f"{r['holdout_eq']:>+13.2f}%")


def save_plots(all_results):
    """Сохранить графики (8×2 как раньше)."""
    tickers = PortfolioEnv(fold=0, eval_mode=True).tickers
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
              '#ff7f00', '#a65628', '#999999']
    tickers_all = tickers + ["CASH"]

    fig, axes = plt.subplots(8, 2, figsize=(16, 32),
                             gridspec_kw={'height_ratios': [3, 1, 3, 1, 3, 1, 3, 1]})

    for fold_idx, r in enumerate(all_results):
        # Value: Train
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

        # Value: Test
        ax_val_v = axes[fold_idx * 2, 1]
        ax_val_v.plot(r['ppo_test'], label='PPO', linewidth=2, color='royalblue')
        ax_val_v.plot(r['eq_test'], label='Equal', alpha=0.7, color='orange')
        ax_val_v.plot(r['cash_test'], label='Cash', alpha=0.5,
                      color='gray', linestyle='--')
        for t in tickers:
            ax_val_v.plot(r['tick_test'][t], label=t, alpha=0.5, linewidth=1)
        hold_tag = " [HOLDOUT]" if r['fold'] == N_FOLDS - 1 else ""
        ax_val_v.set_title(f"Fold {r['fold']} TEST [{r['test_start']}:{r['test_end']}]{hold_tag}"
                           f" — PPO: {r['ppo_test_ret']:+.1f}%")
        ax_val_v.legend(loc='upper left', fontsize=7)
        ax_val_v.grid(True, alpha=0.3)

        # Weights: Train
        ax_w_t = axes[fold_idx * 2 + 1, 0]
        for i, name in enumerate(tickers_all):
            ax_w_t.plot(r['w_train'][:, i], label=name, alpha=0.7,
                        color=colors[i % len(colors)], linewidth=1)
        ax_w_t.set_title(f'Fold {r["fold"]} TRAIN — Weights')
        ax_w_t.legend(loc='upper right', fontsize=6, ncol=4)
        ax_w_t.grid(True, alpha=0.3)

        # Weights: Test
        ax_w_v = axes[fold_idx * 2 + 1, 1]
        for i, name in enumerate(tickers_all):
            ax_w_v.plot(r['w_test'][:, i], label=name, alpha=0.7,
                        color=colors[i % len(colors)], linewidth=1)
        ax_w_v.set_title(f'Fold {r["fold"]} TEST — Weights')
        ax_w_v.legend(loc='upper right', fontsize=6, ncol=4)
        ax_w_v.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'ppo_{VERSION}_results.png', dpi=150)
    print(f"\nГрафик сохранён: ppo_{VERSION}_results.png")


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

if __name__ == "__main__":

    # ── Этап 0: логирование в файл (самодостаточно, без logger.py) ──
    import builtins, datetime, os, sys, time as _time, traceback
    _LOG_DIR = "logs"
    _STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_LOG_DIR, exist_ok=True)
    _logpath = os.path.join(_LOG_DIR, f"train_ppo_{_STAMP}.log")
    if os.path.exists(_logpath):                       # два запуска в одну секунду
        _logpath = os.path.join(_LOG_DIR, f"train_ppo_{_STAMP}_{int(_time.time()%1e6)}.log")
    class _Tee:
        """Дублирует печать в консоль и в файл с таймстампами.
        Состояние 'начало строки' переживает несколько вызовов write,
        потому print может писать текст и \\n отдельными вызовами."""
        def __init__(self, *streams):
            self.streams = streams
            self.at_line_start = True
        def write(self, s):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            for st in self.streams:
                buf = []
                for chunk in s.split("\n"):
                    if chunk or not self.at_line_start:
                        buf.append(f"[{ts}] " + chunk if self.at_line_start else chunk)
                    self.at_line_start = False
                out = "\n".join(buf)
                if s.endswith("\n"):
                    out += "\n"
                    self.at_line_start = True
                else:
                    self.at_line_start = False
                try:
                    if not st.closed:
                        st.write(out)
                except (ValueError, OSError):
                    pass  # файл закрыли - пишем только в остальные потоки
            return len(s)
        def flush(self):
            for st in self.streams:
                try:
                    if not st.closed:
                        st.flush()
                except (ValueError, OSError):
                    pass
        def isatty(self):
            # optuna/colorlog вызывает stream.isatty() у sys.stderr:
            # сообщаем TTY-статус консольного потока, на файл он не влияет.
            for st in self.streams:
                try:
                    if hasattr(st, "isatty"):
                        return bool(st.isatty())
                except (ValueError, OSError):
                    pass
            return False
    _f = open(_logpath, "a", encoding="utf-8", buffering=1)
    _out, _err = _Tee(sys.__stdout__, _f), _Tee(sys.__stderr__, _f)
    sys.stdout = _out
    sys.stderr = _err
    # print - это функция, а не объект: делаем обёртку с поддержкой file=/flush=
    _tee_out, _tee_err = sys.stdout, sys.stderr        # это уже _Tee
    def _print(*args, sep=' ', end='\n', file=None, flush=False):
        out = sep.join(str(a) for a in args) + end
        target = file if file is not None else sys.stdout
        try:
            target.write(out)
            if flush:
                target.flush()
        except (ValueError, OSError):
            pass
    builtins.print = _print
    def _excepthook(t, v, tb):
        for line in traceback.format_exception(t, v, tb):
            _tee_err.write(line)
    sys.excepthook = _excepthook
    print(f"[LOG] Лог пишется в {_logpath}")

    # ── Этап 1: Optuna ──

    # ── Этап 1: Optuna ──
    print(f"\n{'='*60}")
    print(f"  ЭТАП 1: OPTUNA — подбор гиперпараметров")
    print(f"  Trials: {N_TRIALS}, Folds for opt: {OPT_FOLDS}, "
          f"Seeds per fold: {N_SEEDS_TRIAL}")
    print(f"  Метрика: mean - {LAMBDA_VAR}*std по фолдам (робастная)")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=f"ppo_{VERSION}",
        storage=f"sqlite:///optuna_ppo_{VERSION.replace('.', '_')}.db",
        load_if_exists=True,   # старые триалы не теряются (п.9)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\n{'='*60}")
    print(f"  ЛУЧШИЕ ПАРАМЕТРЫ (trial #{study.best_trial.number})")
    print(f"  max_weight      = {best['max_weight']:.2f}")
    print(f"  lambda_hhi      = {best['lambda_hhi']:.4f}")
    print(f"  total_timesteps = {best['total_timesteps']:,}")
    print(f"  dd_alpha        = {best['dd_alpha']:.2f}")
    print(f"  gamma           = {best.get('gamma', 0.99)}")
    print(f"  n_epochs        = {best.get('n_epochs', 10)}")
    print(f"  best robust value = {study.best_value:+.4f}")
    print(f"{'='*60}\n")

    # ── График истории оптимизации (п.16) ──
    plot_optimization_history(study)

    # ── Этап 2: Финальное обучение + мульти-сид проверка (п.11) ──
    multi_seed_check(best)

    print(f"\nГотово! Модели: ppo_portfolio_{VERSION}_fold0..{N_FOLDS-1}.zip "
          f"(базовый seed={SEED})")
