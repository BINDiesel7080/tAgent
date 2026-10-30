"""
probe_folds.py — одноразовая read-only сверка границ фолдов (шаг 0 parking_exp v0.3).

Что делает:
  - читает ТОЛЬКО rl_features_v5.pkl (индекс дат);
  - вызывает PortfolioEnv.build_folds(idx) — арифметика разметки;
  - печатает train/test-окна фолдов 0-5 фактическими датами и число строк
    test-окна, попавших внутрь train-окна fold5 (in-sample статус).

Чего НЕ делает:
  - не создаёт PortfolioEnv (env не инстанцируется, веса не считаются);
  - не читает и не пишет базу Optuna;
  - не качает данные;
  - ничего не пишет на диск (ни pkl, ни csv, ни лога).

Границы индексов: te и ve — ЭКСКЛЮЗИВНЫЕ, поэтому в печать идёт te-1 и ve-1
(помечены «(вкл)»).

Запуск (обязательна активация conda — прямой python.exe ломает LAPACK):
  call F:\\anaconda3\\condabin\\conda.bat activate trading-agent && python probe_folds.py

Version: v1.0
Date:    2026-10-30
"""

import pandas as pd

from portfolio_env import PortfolioEnv

FEATURES_FILE = "rl_features_v5.pkl"


def main():
    idx = pd.DatetimeIndex(pd.read_pickle(FEATURES_FILE).index)
    print(f"датасет: {FEATURES_FILE}  rows={len(idx)}  "
          f"{idx[0].date()} .. {idx[-1].date()}")

    folds = PortfolioEnv.build_folds(idx)
    print(f"фолдов построено: {len(folds)} "
          f"(TRAIN_WINDOW={PortfolioEnv.TRAIN_WINDOW}, "
          f"TEST_WINDOW={PortfolioEnv.TEST_WINDOW}, "
          f"EMBARGO={PortfolioEnv.EMBARGO}, "
          f"HOLDOUT_ROWS={PortfolioEnv.HOLDOUT_ROWS})")

    tr5, te5 = folds[-1][0], folds[-1][1]
    print(f"train fold5 (норм-статистики моделей v1.2): idx [{tr5},{te5})  "
          f"{idx[tr5].date()} .. {idx[te5 - 1].date()} (вкл)  "
          f"строк={te5 - tr5}")

    print("\n=== Границы фолдов и in-sample статус ===")
    header = (f"{'fold':<5} {'train idx':<15} {'train даты':<25} "
              f"{'test idx':<15} {'test даты':<25} "
              f"{'строк':<7} {'строк в train5':<15} статус")
    print(header)
    print("-" * len(header))

    for k, (ts, te, vs, ve) in enumerate(folds):
        lo, hi = max(vs, tr5), min(ve, te5)
        ov = max(0, hi - lo)
        total = ve - vs
        if ov == 0:
            status = "OOS"
        elif ov == total:
            status = "IN-SAMPLE"
        else:
            status = f"PARTIAL {ov / total * 100:.0f}%"
        tr_dates = f"{idx[ts].date()} .. {idx[te - 1].date()} (вкл)"
        te_dates = f"{idx[vs].date()} .. {idx[ve - 1].date()} (вкл)"
        print(f"{k:<5} [{ts},{te}){'':<2} {tr_dates:<25} "
              f"[{vs},{ve}){'':<2} {te_dates:<25} "
              f"{total:<7} {ov:<15} {status}")

    print("\nПояснение: 'строк в train5' — сколько строк test-окна фолда "
          "попадает внутрь [tr5,te5). 0 = окно вне-sample, "
          "total = полностью in-sample.")


if __name__ == "__main__":
    main()
