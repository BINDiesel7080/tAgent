# -*- coding: utf-8 -*-
"""
weekend_diag.py — диагностика режима выходных сессий (ДСВД) против будней.

Зачем: с марта 2025 Мосбиржа проводит сессии по выходным (коридор +/-3%,
тонкий стакан). Проверяем, статистически ли это другой режим рынка:
  1) сравнение ТРЁХ групп (против календарной примеси):
       - будни ДО ДСВД    (start_date .. 2025-02-28)
       - будни ПОСЛЕ      (2025-03-01 .. end)
       - выходные ПОСЛЕ   (2025-03-01 .. end)
     ключевое сравнение: будни-после против выходных (один календарный период).
  2) метрики по группам: средняя доходность (bp), std, средний |ret|,
     средний размах high-low, средний объём.
  3) статтестаты разницы |ret|: Welch t и Манн-Уитни (вручную, без scipy).
  4) ACF доходностей lag 1..10 в двух вариантах: все бары vs будни-only
     (тест гипотезы: инерция PLZL lag1 ~ +0.12 — артефакт выходных или нет).

Запуск: python weekend_diag.py   (читает candles_cache/*.csv, ничего не качает)
Выход:  консоль + weekend_diag.csv + acf_weekend.csv
"""

import glob
import os
import math
import numpy as np
import pandas as pd

CACHE_DIR = "candles_cache"
DATE_COL = "date"
TICKER_COL = "ticker"
VALUE_COL = "value"      # оборот в рублях из ISS (если колонка есть)
DSWD_START = pd.Timestamp("2025-03-01")   # старт ДСВД


def load_csv(path):
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    # мягкий парсер колонок
    date_c = cols.get(DATE_COL) or cols.get("trade_date") or cols.get("datetime")
    if date_c is None:
        raise ValueError(f"{path}: не нашёл колонку даты; колонки: {list(df.columns)}")
    df["date"] = pd.to_datetime(df[date_c], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", kind="stable")
    df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)
    ren = {}
    for std in ["open", "high", "low", "close", "volume", "value"]:
        if std in cols:
            ren[cols[std]] = std
    return df.rename(columns=ren)


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return np.nan, np.nan
    t = (a.mean() - b.mean()) / se
    dfree = (va + vb) ** 2 / (va ** 2 / max(len(a) - 1, 1) + vb ** 2 / max(len(b) - 1, 1))
    # двусторонний p-value через точную формулу t-распределения невозможна без scipy:
    # используем нормальную аппроксимацию при df>30, иначе погрешность оговорим в выводе
    z = abs(t)
    p = math.erfc(z / math.sqrt(2))
    return t, p


def mann_whitney_p(a, b):
    """Манн-Уитни вручную с поправкой на связки. Возвращает (U, p двусторонний)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    r1 = ranks[:n1].sum()
    mu = n1 * n2 / 2.0   # E[U] для статистики U (не суммы рангов!)
    # поправка на связки в дисперсии
    _, counts = np.unique(allv, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n1 + n2 + 1) - tie_term / ((n1 + n2) * (n1 + n2 - 1))))
    if sigma == 0:
        return np.nan, np.nan
    U = r1 - n1 * (n1 + 1) / 2.0
    z = (U - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return U, p


def acf_lags(x, max_lag):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    out = {}
    for lag in range(1, max_lag + 1):
        if len(x) <= lag + 2:
            out[lag] = np.nan
            continue
        a, b = x[lag:], x[:-lag]
        if a.std() == 0 or b.std() == 0:
            out[lag] = np.nan
            continue
        out[lag] = float(np.corrcoef(a, b)[0, 1])
    return out


def main():
    files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.csv")))
    if not files:
        print("Кэш пуст:", CACHE_DIR, "— нет CSV. Выход.")
        return

    print("=" * 64)
    print("  ДИАГНОСТИКА РЕЖИМА ДСВД: будни до / будни после / выходные после")
    print("  Граница ДСВД:", DSWD_START.date())
    print("=" * 64)

    rows = []          # для weekend_diag.csv
    acf_rows = []      # для acf_weekend.csv

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_csv(path)
        except Exception as e:
            print(f"  {name}: ошибка чтения ({e}) — пропускаю")
            continue

        r = df["close"].pct_change()
        weekend_mask = df["date"].dt.dayofweek >= 5
        after_mask = df["date"] >= DSWD_START
        before_mask = df["date"] < DSWD_START

        groups = {
            "wd_before": before_mask & ~weekend_mask,
            "wd_after":  after_mask & ~weekend_mask,
            "we_after":  after_mask & weekend_mask,
        }
        stats = {}
        for gname, m in groups.items():
            rr = r[m].dropna()
            if len(rr) == 0:
                stats[gname] = None
                continue
            entry = {
                "n": len(rr),
                "mean_ret_bp": rr.mean() * 1e4,
                "std_bp": rr.std(ddof=1) * 1e4,
                "mean_abs_bp": rr.abs().mean() * 1e4,
            }
            if {"high", "low"} <= set(df.columns):
                rng = (df.loc[m, "high"] - df.loc[m, "low"]) / df.loc[m, "close"]
                entry["mean_range_pct"] = float(rng[m].mean()) * 100 if m.any() else np.nan
            else:
                entry["mean_range_pct"] = np.nan
            if "volume" in df.columns:
                entry["mean_volume"] = float(df.loc[m, "volume"].mean()) if m.any() else np.nan
            else:
                entry["mean_volume"] = np.nan
            stats[gname] = entry

        if stats.get("wd_after") and stats.get("we_after"):
            a = r[groups["we_after"]].dropna()
            b = r[groups["wd_after"]].dropna()
            ratio = a.abs().mean() / b.abs().mean() if b.abs().mean() else np.nan
            t, p_t = welch_t(a.abs(), b.abs())
            U, p_mw = mann_whitney_p(a.abs(), b.abs())
        else:
            ratio = t = p_t = U = p_mw = np.nan

        print(f"\n--- {name} ---")
        for gname, st in stats.items():
            if st is None:
                print(f"  {gname:10s}: нет данных")
            else:
                vol_str = f"{st['mean_volume']:,.0f}" if not np.isnan(st["mean_volume"]) else "n/a"
                print(f"  {gname:10s}: n={st['n']:4d}  ret={st['mean_ret_bp']:+6.1f}bp  "
                      f"std={st['std_bp']:7.1f}bp  |ret|={st['mean_abs_bp']:6.1f}bp  "
                      f"range={st['mean_range_pct']:.2f}%  vol={vol_str}")
        print(f"  ratio |ret| (вых/будни-после): {ratio if not np.isnan(ratio) else float('nan'):.3f}  "
              f"Welch t={t:.2f} p={p_t:.3f}   MW U={U:.0f} p={p_mw:.3f}")
        if p_t < 0.05:
            print("  -> режимы различимы по |ret| (p<0.05)")

        # ACF: все бары vs будни-only
        r_all = r.dropna()
        r_wd = r[~weekend_mask].dropna()
        acf_all = acf_lags(r_all, 10)
        acf_wd = acf_lags(r_wd, 10)
        print("  ACF lag 1..10 (все бары | будни-only):")
        for lag in range(1, 11):
            a1, a2 = acf_all[lag], acf_wd[lag]
            print(f"    lag {lag:2d}: {a1:+.3f} | {a2:+.3f}")
            acf_rows.append({"ticker": name, "lag": lag,
                             "acf_all": a1, "acf_wd_only": a2,
                             "n_all": len(r_all), "n_wd": len(r_wd)})

        for gname, st in stats.items():
            if st:
                rows.append({"ticker": name, "group": gname, **st,
                             "ratio_abs_vs_wd_after": ratio,
                             "welch_t": t, "p_welch": p_t,
                             "mw_U": U, "p_mw": p_mw})

    if rows:
        pd.DataFrame(rows).to_csv("weekend_diag.csv", index=False)
    if acf_rows:
        pd.DataFrame(acf_rows).to_csv("acf_weekend.csv", index=False)
    print("\nГотово: weekend_diag.csv, acf_weekend.csv")


if __name__ == "__main__":
    main()
