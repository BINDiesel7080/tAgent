"""
Обучение PPO-агента для портфеля Мосбирже — Walk-forward (4 фолда) + Optuna.

Version: v0.7
Date:   2026-10-09
Changes from v0.6:
  - Введён порог отсечки для расчёта downside deviation (для Sortino)
  - Метрика Optuna изменена на Sortino Ratio вместо чистой доходности
  - portfolio_env v0.7: снижение terminal reward + усиление dd_penalty
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from portfolio_env import PortfolioEnv # Предполагается, что это ваш файл из предыдущего сообщения

import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

# ═══════════════════════════════════════════════
# КОНФИГ
# ═══════════════════════════════════════════════
VERSION = "v0.7"
EPISODE_LENGTH = 30
SEED = 42
N_FOLDS = 4
N_TRIALS = 20               # число испытаний Optuna
OPT_FOLDS = N_FOLDS               # сколько фолдов использовать для оптимизации (меньше = быстрее, больше = точнее)
TOTAL_TIMESTEPS_DEFAULT = 300_000

# Новые параметры для расчёта Sortino Ratio
DOWNSIDE_DEV_THRESHOLD = -0.001  # Отсекаем все доходности > -0.1%
USE_SORTINO_RATIO = True         # Использовать Sortino вместо чистой доходности

# ═══════════════════════════════════════════════
# ФУНКЦИИ
# ═══════════════════════════════════════════════

def evaluate(model, fold, eval_split, seed, max_weight=0.5, lambda_hhi=0.5):
    """Прогон модели по указанному сплиту фолда."""
    env = PortfolioEnv(
        features_path='rl_features.pkl',
        prices_path='rl_prices.pkl',
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


def train_one_fold(fold, total_timesteps, max_weight, lambda_hhi, verbose=0):
    """Обучить PPO на одном фолде и вернуть модель + среду."""
    train_env = Monitor(PortfolioEnv(
        features_path='rl_features.pkl',
        prices_path='rl_prices.pkl',
        episode_length=EPISODE_LENGTH,
        eval_mode=False,
        fold=fold,
        seed=SEED,
        max_weight=max_weight,
        lambda_hhi=lambda_hhi,
    ))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        clip_range=0.1,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        seed=SEED,
    )

    model.learn(total_timesteps=total_timesteps)
    return model, train_env


def evaluate_fold(model, fold, max_weight, lambda_hhi):
    """Оценка модели на train и test фолда. Возвращает метрики."""
    ts, te, vs, ve = PortfolioEnv.FOLDS[fold]

    # Нужны features и tickers — создаём временную среду
    tmp_env = PortfolioEnv(fold=fold, eval_mode=True, eval_split='test')
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
    Args:
        ret: Массив значений капитала (например, result['ppo_test']).
        threshold: Порог отсечения. Доходности выше этого порога считаются «хорошими».
                  По умолчанию -0.001 (-0.1%), т.е. учитываются только потери глубже этой отметки.
    Returns:
        float: Значение Sortino Ratio или -inf, если потерь нет.
    """
    daily_rets = np.diff(rets) / rets[:-1]  # Преобразуем значения капитала в дневные доходности

    avg_return = daily_rets.mean()
    
    # Берём только отрицательные отклонения от порога
    down_rets = daily_rets[daily_rets < threshold]
    
    # Если потерь ниже порога вообще не было, считаем результат плохим
    if len(down_rets) == 0:
        return float('-inf')
    
    down_deviation = down_rets.std(ddof=0)  # Без коррекции Бесселя (ddof=0)
    if down_deviation == 0:
        return float('-inf')

    sortino = avg_return / down_deviation
    return sortino


def objective(trial):
    """Целевая функция Optuna: максимизировать средний Test Sortino по фолдам."""
    max_weight = trial.suggest_float("max_weight", 0.3, 0.6, step=0.05)
    lambda_hhi = trial.suggest_float("lambda_hhi", 0.01, 0.2, log=True)
    total_timesteps = trial.suggest_categorical(
        "total_timesteps", [150_000, 200_000, 250_000])

    scores = []  # Собираем метрику для каждого фолда

    for fold in range(OPT_FOLDS):
        model, _ = train_one_fold(
            fold, total_timesteps, max_weight, lambda_hhi, verbose=0)
        
        # Оцениваем тестовый период
        result = evaluate_fold(model, fold, max_weight, lambda_hhi)

        # Выбираем нужную метрику
        metric = (
            calculate_sortino_ratio(result['ppo_test'], DOWNSIDE_DEV_THRESHOLD)
            if USE_SORTINO_RATIO else
            result['ppo_test_ret']
        )

        scores.append(metric)

        # Pruning: если уже после первого фолда результат ужасный — прерываем
        if fold == 0 and metric <= 0:
            raise optuna.TrialPruned()

    avg_score = np.mean(scores)
    print(f"  Trial {trial.number}: "
          f"max_w={max_weight:.2f} lambda={lambda_hhi:.3f} ts={total_timesteps:,}"
          f" → avg_{'sortino' if USE_SORTINO_RATIO else 'test'}={avg_score:+.3f}")

    return avg_score


# ──────────────────────────
# ФИНАЛЬНОЕ ОБУЧЕНИЕ (с лучшими параметрами)
# ──────────────────────────

def run_final_training(best_params):
    """Обучить финальные модели на всех 4 фолдах с лучшими параметрами."""
    max_weight = best_params['max_weight']
    lambda_hhi = best_params['lambda_hhi']
    total_timesteps = best_params['total_timesteps']

    print(f"\n{'='*60}")
    print(f"  ФИНАЛЬНОЕ ОБУЧЕНИЕ")
    print(f"  max_weight  = {max_weight:.2f}")
    print(f"  lambda_hhi  = {lambda_hhi:.4f}")
    print(f"  timesteps   = {total_timesteps:,}")
    print(f"{'='*60}\n")

    all_results = []

    for fold in range(N_FOLDS):
        ts, te, vs, ve = PortfolioEnv.FOLDS[fold]

        print(f"\n{'='*60}")
        print(f"  FOLD {fold}: train [{ts}:{te}] ({te-ts} дн.)  "
              f"test [{vs}:{ve}] ({ve-vs} дн.)")
        print(f"{'='*60}\n")

        model, train_env = train_one_fold(
            fold, total_timesteps, max_weight, lambda_hhi, verbose=1)

        model_path = f"ppo_portfolio_{VERSION}_fold{fold}"
        model.save(model_path)
        print(f"\nМодель сохранена: {model_path}.zip\n")

        result = evaluate_fold(model, fold, max_weight, lambda_hhi)

        # Печать результатов
        print(f"--- Fold {fold} результаты ---")
        print(f"  TRAIN PPO:    {result['ppo_train_ret']:+.2f}%")
        print(f"  TEST  PPO:    {result['ppo_test_ret']:+.2f}%")
        print(f"  TEST  Equal:  {result['eq_test_ret']:+.2f}%")
        tickers = PortfolioEnv(fold=fold, eval_mode=True).tickers
        # Бенчмарки по тикерам
        for t in tickers:
            t_ret = (result['tick_test'][t][-1] - 1) * 100
            print(f"  TEST  {t:5s}: {t_ret:+.2f}%")
        print(f"  TEST  Cash:   {(result['cash_test'][-1] - 1) * 100:+.2f}%")
        print(f"  Оборот/шаг:   {result['turnover_test']:.4f}")
        print(f"  Доля кэша:    {result['cash_share_test']:.2f}")
        print(f"  Финальные веса:")
        for t, w in zip(tickers + ["CASH"], result['final_weights']):
            print(f"    {t:5s}: {w:.3f}")

        all_results.append(result)

    return all_results


def print_summary(all_results):
    """Сводная таблица по всем фолдам."""
    print(f"\n{'='*60}")
    print(f"  СВОДКА ПО ВСЕМ ФОЛДАМ")
    print(f"{'='*60}")
    print(f"{'Fold':>4} | {'Train PPO':>10} | {'Test PPO':>10} | "
          f"{'Test Equal':>10} | {'Оборот':>8} | {'Кэш':>5}")
    print(f"{'-' * 4}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 5}")

    for r in all_results:
        print(f"{r['fold']:>4} | {r['ppo_train_ret']:>+9.2f}% | "
              f"{r['ppo_test_ret']:>+9.2f}% | {r['eq_test_ret']:>+9.2f}% | "
              f"{r['turnover_test']:>8.4f} | {r['cash_share_test']:>5.2f}")

    avg_train = np.mean([r['ppo_train_ret'] for r in all_results])
    avg_test = np.mean([r['ppo_test_ret'] for r in all_results])
    avg_eq = np.mean([r['eq_test_ret'] for r in all_results])
    print(f"{'AVG':>4} | {avg_train:>+9.2f}% | {avg_test:>+9.2f}% | "
          f"{avg_eq:>+9.2f}% |\n")


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
        ax_val_v.set_title(f"Fold {r['fold']} TEST [{r['test_start']}:{r['test_end']}] "
                           f"— PPO: {r['ppo_test_ret']:+.1f}%")
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

    # ── Этап 1: Optuna ──
    print(f"\n{'='*60}")
    print(f"  ЭТАП 1: OPTUNA — подбор гиперпараметров")
    print(f"  Trials: {N_TRIALS}, Folds for opt: {OPT_FOLDS}")
    print(f"{'='*60}")

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"ppo_{VERSION}",
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=4)

    best = study.best_params
    print(f"\n{'='*60}")
    print(f"  ЛУЧШИЕ ПАРАМЕТРЫ (trial #{study.best_trial.number})")
    print(f"  max_weight    = {best['max_weight']:.2f}")
    print(f"  lambda_hhi    = {best['lambda_hhi']:.4f}")
    print(f"  total_timesteps = {best['total_timesteps']:,}")
    print(f"  best avg_{'sortino' if USE_SORTINO_RATIO else 'test'} = {study.best_value:+.2f}")
    print(f"{'='*60}\n")

    # ── Этап 2: Финальное обучение ──
    all_results = run_final_training(best)

    # ── Этап 3: Сводка и графики ──
    print_summary(all_results)
    save_plots(all_results)

    print(f"\nГотово! Модели: ppo_portfolio_{VERSION}_fold0..{N_FOLDS-1}.zip")