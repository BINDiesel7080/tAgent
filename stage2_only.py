"""
stage2_only.py — только ЭТАП 2: обучение на holdout (fold 5) с лучшими
параметрами из уже накопленной базы Optuna. Поиск не запускает.
"""
from portfolio_env import PortfolioEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import optuna
import numpy as np
import pandas as pd

FEATURES_FILE = 'rl_features_v4.pkl'
PRICES_FILE   = 'rl_prices_v4.pkl'
STUDY_NAME    = 'ppo_v1.1'
STORAGE       = 'sqlite:///optuna_ppo_v1_1.db'
VERSION       = 'v1.1'
HOLDOUT_FOLD  = 5
FINAL_SEEDS = [42, 1042, 2042, 3042, 4042, 5042, 6042]

EP_LEN_RANGE      = (30, 200)
RANDOM_START_PROB = 0.5

_FOLDS_CACHE = None

def get_folds():
    global _FOLDS_CACHE
    if _FOLDS_CACHE is None:
        idx = pd.read_pickle(FEATURES_FILE).index
        _FOLDS_CACHE = PortfolioEnv.build_folds(idx)
    return _FOLDS_CACHE


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
    """Обучить PPO на train-части фолда и вернуть модель."""
    train_env = Monitor(PortfolioEnv(
        features_path=FEATURES_FILE,
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
    """Детерминированный прогон модели по сплиту фолда."""
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


def main():
    # ── 1. Топ триалов из базы ──
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    complete = sorted([t for t in study.trials if t.state.name == 'COMPLETE'],
                      key=lambda t: t.value, reverse=True)
    print(f"\nТоп-{min(5, len(complete))} триалов из базы ({len(complete)} завершённых):")
    for t in complete[:5]:
        print(f"  trial {t.number:>2}: value={t.value:+.4f}  {t.params}")
    best = complete[0].params
    print(f"\nБерём trial #{complete[0].number} (robust={complete[0].value:+.4f})")
    print(f"  max_weight      = {best['max_weight']:.2f}")
    print(f"  lambda_hhi      = {best['lambda_hhi']:.4f}")
    print(f"  total_timesteps = {best['total_timesteps']:,}")
    print(f"  dd_alpha        = {best['dd_alpha']:.2f}")
    print(f"  gamma           = {best.get('gamma', 0.99)}")
    print(f"  n_epochs        = {best.get('n_epochs', 10)}")

    # ── 2. Holdout-фолд и бенчмарки ──
    folds = get_folds()
    ts, te, vs, ve = folds[HOLDOUT_FOLD]
    tmp = PortfolioEnv(fold=HOLDOUT_FOLD, eval_mode=True, eval_split='test')
    features, tickers = tmp.features, tmp.tickers
    d_start = features.index[vs].date()
    d_end = features.index[ve - 1].date()
    print(f"\nHOLDOUT fold {HOLDOUT_FOLD}: "
          f"train [{features.index[ts].date()} .. {features.index[te-1].date()}], "
          f"test [{d_start} .. {d_end}] ({ve - vs} строк)")

    eq_values, tick_values, cash_values = benchmarks_on_split(
        features, tickers, vs, ve - 1)
    eq_ret = (eq_values[-1] - 1) * 100
    print(f"Equal на holdout: {eq_ret:+.2f}%  (baseline прежнего прогона: -11.07%)")
    for t in tickers:
        print(f"  {t:5s}: {(tick_values[t][-1] - 1) * 100:+.2f}%")

    # ── 3. Обучение на 3 сидах + оценка ──
    dd_alpha  = best['dd_alpha']
    gamma     = best.get('gamma', 0.99)
    n_epochs  = best.get('n_epochs', 10)
    ppo_rets = []
    for seed in FINAL_SEEDS:
        print(f"\n{'=' * 60}\n  Сид {seed}: обучение 200k... {'=' * 60}\n")
        model = train_one_fold(HOLDOUT_FOLD, best['total_timesteps'],
                               best['max_weight'], best['lambda_hhi'],
                               dd_alpha, gamma, n_epochs, seed, verbose=1)
        model_path = f"ppo_portfolio_{VERSION}_fold{HOLDOUT_FOLD}_s{seed}"
        model.save(model_path)
        print(f"Модель сохранена: {model_path}.zip")

        values, w, _ = evaluate(model, HOLDOUT_FOLD, 'test', seed,
                                best['max_weight'], best['lambda_hhi'])
        ppo_ret = (values[-1] - 1) * 100
        ppo_rets.append(ppo_ret)
        cash_share = w[:, -1].mean()
        turnover = np.abs(np.diff(w, axis=0)).mean(axis=1).mean()
        print(f"\n--- Сид {seed}: PPO {ppo_ret:+.2f}% | Equal {eq_ret:+.2f}% | "
              f"кэш {cash_share:.2f} | оборот/шаг {turnover:.4f}")
        print("  Финальные веса:")
        for t, wv in zip(tickers + ["CASH"], w[-1]):
            print(f"    {t:5s}: {wv:.3f}")

    # ── 4. Вердикт ──
    mean_ppo = float(np.mean(ppo_rets))
    print(f"\n{'=' * 60}")
    print(f"  ИТОГ ПО HOLDOUT (среднее по {len(FINAL_SEEDS)} сидам)")
    print(f"  PPO:   {mean_ppo:+.2f}%")
    print(f"  Equal: {eq_ret:+.2f}%")
    verdict = "ПОБЕДА" if mean_ppo >= eq_ret else "проигрыш"
    print(f"  Агент {verdict} бенчмарк (разница {mean_ppo - eq_ret:+.2f} п.п.)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
