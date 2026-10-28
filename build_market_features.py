"""
Сборка расширенного набора фичей (v2): базовые 31 + 3 рыночные.

Новые фичи (все считаются только из ПРОШЛЫХ данных, без заглядывания вперёд):
  market_breadth20 — доля тикеров выше своей 20-дневной SMA;
  market_xsec_ret5 — средняя 5-дневная доходность по всем тикерам;
  market_corr20    — средняя попарная корреляция доходностей за 20 дней.

Первые дни, где история короче окна, заполняются нейтрально
(breadth=0.5, остальные 0.0) — индексы и фолды не сдвигаются.

Запуск:
  python build_market_features.py
  python build_market_features.py --features rl_features.pkl --prices rl_prices.pkl --out rl_features_v2.pkl
"""

import argparse

import numpy as np
import pandas as pd


def compute_market_features(features, prices):
    """Добавить к фичам 3 рыночные. Ничего не меняет в исходных колонках."""
    tickers = [c.replace('_close', '') for c in prices.columns]
    ret_cols = [f"{t}_returns" for t in tickers]
    missing = [c for c in ret_cols if c not in features.columns]
    assert not missing, f"В features нет колонок: {missing}"

    assert list(prices.index) == list(features.index), \
        "Индексы features и prices не совпадают"

    n = len(features)
    closes = prices[[f"{t}_close" for t in tickers]].to_numpy(dtype=np.float64)
    rets = features[ret_cols].to_numpy(dtype=np.float64)

    # 1) breadth20: доля тикеров выше своей 20-дневной SMA
    sma20 = pd.DataFrame(closes).rolling(20, min_periods=20).mean().to_numpy()
    breadth = (closes > sma20).mean(axis=1)
    breadth[:19] = 0.5  # нейтрально, пока нет полной истории

    # 2) xsec_ret5: средняя 5-дневная доходность по тикерам
    close_df = pd.DataFrame(closes)
    ret5 = (close_df / close_df.shift(5) - 1.0).to_numpy()
    xsec5 = np.nanmean(ret5, axis=1)
    xsec5[:5] = 0.0

    # 3) corr20: средняя попарная корреляция доходностей за скользящие 20 дней
    corr20 = np.zeros(n, dtype=np.float64)
    n_t = len(tickers)
    for t in range(n):
        lo = max(0, t - 19)
        window = rets[lo:t + 1]
        if window.shape[0] < 20:
            continue  # остаётся 0.0
        stds = window.std(axis=0)
        if stds.min() < 1e-12:
            continue  # вырожденное окно (например, каникулы) — нейтральный 0
        C = np.corrcoef(window, rowvar=False)
        off = C[~np.eye(n_t, dtype=bool)]
        corr20[t] = off.mean()

    out = features.copy()
    out['market_breadth20'] = breadth.astype(np.float32)
    out['market_xsec_ret5'] = xsec5.astype(np.float32)
    out['market_corr20'] = corr20.astype(np.float32)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='rl_features.pkl')
    ap.add_argument('--prices', default='rl_prices.pkl')
    ap.add_argument('--out', default='rl_features_v2.pkl')
    args = ap.parse_args()

    features = pd.read_pickle(args.features).astype(np.float32)
    prices = pd.read_pickle(args.prices).astype(np.float32)

    out = compute_market_features(features, prices)
    out.to_pickle(args.out)

    print(f"Сохранено: {args.out}")
    print(f"  Колонок было: {features.shape[1]}, стало: {out.shape[1]}")
    for col in ['market_breadth20', 'market_xsec_ret5', 'market_corr20']:
        s = out[col]
        print(f"  {col:20s}: mean={s.mean():+.4f} std={s.std():.4f} "
              f"min={s.min():+.4f} max={s.max():+.4f}")

    # перекрёстные корреляции новых фичей — проверка на дублирование
    new_cols = out[['market_breadth20', 'market_xsec_ret5', 'market_corr20']]
    print("\n  Корреляции между новыми фичами:")
    print(new_cols.corr().round(3).to_string())


if __name__ == '__main__':
    main()
