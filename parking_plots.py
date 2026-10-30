#!/usr/bin/env python3
"""
parking_plots.py — ШАГ 6 mini-эксперимента парковки кэша (parking_exp v0.3).

НАЗНАЧЕНИЕ
    Визуализация результата шага 3. НИЧЕГО не пересчитывает по-новому:
    все числа берутся из готовых артефактов (results/parking_grid.csv,
    results/parking_traj_*.csv, data/r_fund_scenarios.csv) и переиспользуют
    функции политики из parking_grid.py — чтобы картинки не расходились
    с вердиктом.

ГРАФИКИ
    G1  parking_heatmap.png   — 3 панели (по одному floor), цвет = mean d_ret,
                                невалидные пары закрашены серым, значения в ячейках.
    G2  parking_equity.png    — эквити агента vs эквити с парковкой (eq_adj),
                                лучшая конфигурация, худший и лучший фолд.
    G3  parking_cash.png      — доля кэша агента и позиция в фонде по шагам:
                                видно, где политика паркует, а где идёт выкуп.
    G4  parking_commission.png— mean d_ret против множителя комиссии
                                (x0.0 / x1.0 / x2.0) для всех конфигураций.

ЧЕГО НЕ ДЕЛАЕТ
    - не пишет в базу Optuna, не инстанцирует среду, не грузит модели;
    - не трогает fold 5, portfolio_env.py, файлы шага 1;
    - не меняет сетку и не пересматривает вердикт.

Version: v1.0
Date:    2026-10-30
"""

import glob
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import parking_config as C
import parking_grid as G

try:
    if C.FORCE_UTF8:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ═══════════════════════════════════════════════════════════════════
# G1: ТЕПЛОВАЯ КАРТА (theta x floor, панели по floor)
# ═══════════════════════════════════════════════════════════════════

def plot_heatmap(df, out_path):
    """Три панели по floor: цвет = mean d_ret по всем фолдам (сценарий x1.0)."""
    sub = df[df["scenario"] == C.BASELINE_SCENARIO]
    thetas = sorted(sub["theta"].unique())
    floors = sorted(sub["floor"].unique())

    fig, axes = plt.subplots(1, len(floors), figsize=(5.0 * len(floors), 4.4),
                             squeeze=False)
    axes = axes[0]

    all_vals = []
    for f in floors:
        for t in thetas:
            v = sub[(sub["theta"] == t) & (sub["floor"] == f)]["d_ret_mean"]
            if len(v):
                all_vals.append(float(v.mean()) * 100)
    if all_vals:
        vmax = max(abs(min(all_vals)), abs(max(all_vals))) or 1.0
        vmin = min(all_vals)
    else:
        vmax, vmin = 1.0, 0.0

    for ax, f in zip(axes, floors):
        grid = np.full((1, len(thetas)), np.nan)
        for j, t in enumerate(thetas):
            v = sub[(sub["theta"] == t) & (sub["floor"] == f)]["d_ret_mean"]
            if not len(v):
                continue
            if C.is_valid_pair(t, f):
                grid[0, j] = float(v.mean()) * 100
        masked = np.ma.masked_invalid(grid)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#9e9e9e")
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels([f"{t:.2f}" for t in thetas])
        ax.set_yticks([0])
        ax.set_yticklabels([f"floor={f:.2f}"])
        ax.set_xlabel("theta (порог входа)")
        ax.set_title(f"floor = {f:.2f}")
        for j, t in enumerate(thetas):
            if np.isnan(grid[0, j]):
                ax.text(j, 0, "невалидно\nθ−floor≤1%", ha="center", va="center",
                        fontsize=8, color="#333333")
            else:
                ax.text(j, 0, f"{grid[0, j]:+.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="mean d_ret, п.п.")

    fig.suptitle("G1. Окупаемость парковки: mean d_ret по (theta, floor), "
                 f"сценарий {C.BASELINE_SCENARIO}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# ВЫБОР ЛУЧШЕЙ КОНФИГУРАЦИИ (правило N5)
# ═══════════════════════════════════════════════════════════════════

def pick_best(df):
    """N5: argmax mean d_ret при x1.0; tie-break — меньше переключений, затем theta."""
    sub = df[df["scenario"] == C.BASELINE_SCENARIO].copy()
    if sub.empty:
        raise RuntimeError("в parking_grid.csv нет строк сценария x1.0")
    agg = (sub.groupby(["theta", "floor"])
              .agg(d_ret=("d_ret_mean", "mean"),
                   switches=("switches_mean", "sum"))
              .reset_index()
              .sort_values(["d_ret", "switches", "theta"],
                           ascending=[False, True, True]))
    top = agg.iloc[0]
    return float(top["theta"]), float(top["floor"])


def load_inputs():
    grid_path = os.path.join(C.RESULTS_DIR, C.GRID_CSV)
    if not os.path.exists(grid_path):
        raise RuntimeError(f"нет {grid_path} — сначала выполните шаг 3")
    df = pd.read_csv(grid_path)
    print(f"\n[СЕТКА] {grid_path}: {len(df)} строк")

    files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "parking_traj_*.csv")))
    if not files:
        raise RuntimeError(f"нет траекторий в {C.RESULTS_DIR}/")
    trajs = {}
    for p in files:
        d = pd.read_csv(p)
        trajs[(int(d["fold"].iloc[0]), int(d["seed"].iloc[0]))] = d
    print(f"[ТРАЕКТОРИИ] {len(trajs)} файлов")

    scen = pd.read_csv(os.path.join(C.DATA_DIR, C.R_FUND_CSV), parse_dates=["date"])
    cal = set(pd.read_csv(os.path.join(C.DATA_DIR, C.RUONIA_CSV),
                          parse_dates=["date"])["date"].dt.strftime("%Y-%m-%d"))
    scen_idx = scen.set_index(scen["date"].dt.strftime("%Y-%m-%d"))
    return df, trajs, scen_idx["r_fund_base"], cal


# ═══════════════════════════════════════════════════════════════════
# G2 / G3: ЭКВИТИ И КЭШ ДЛЯ ЛУЧШЕЙ КОНФИГУРАЦИИ
# ═══════════════════════════════════════════════════════════════════

def run_best(trajs, r_base_all, calendar, theta, floor, fold, seed, mult):
    traj = trajs[(fold, seed)]
    dates = traj["decision_date"].to_numpy()
    r = np.array([r_base_all.get(d, np.nan) for d in dates], dtype=np.float64)
    res = G.run_policy(traj, r, calendar, theta, floor, max(C.DELTA_GRID),
                       mult, C.COMMISSION_RATE)
    return traj, res


def plot_equity(trajs, r_base_all, calendar, theta, floor, d_fold, out_path):
    """G2: эквити без парковки vs с парковкой, два фолда (худший и лучший)."""
    mult = C.COMMISSION_SCENARIOS[C.BASELINE_SCENARIO]
    folds = sorted(d_fold, key=lambda f: d_fold[f])

    fig, axes = plt.subplots(1, len(folds), figsize=(6.4 * len(folds), 4.4),
                             squeeze=False)
    axes = axes[0]
    for ax, fold in zip(axes, folds):
        traj, res = run_best(trajs, r_base_all, calendar, theta, floor,
                             fold, C.SEEDS[0], mult)
        x = np.arange(len(res["eq_series"]))
        ax.plot(x, res["base_series"], label="агент (без парковки)",
                color="#1f77b4", lw=1.8)
        ax.plot(x, res["eq_series"], label="агент + парковка (eq_adj)",
                color="#2ca02c", lw=1.8)
        ax.set_title(f"fold {fold}: d_ret {res['d_ret'] * 100:+.4f} п.п., "
                     f"операций {res['switches']}")
        ax.set_xlabel("шаг (решение)")
        ax.set_ylabel("эквити, старт = 1.0")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"G2. Эквити: theta={theta:.2f}, floor={floor:.2f}, "
                 f"сид {C.SEEDS[0]}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [OK] {out_path}")


def plot_cash(trajs, r_base_all, calendar, theta, floor, fold, out_path):
    """G3: доля кэша агента на фоне позиции в фонде — видно вход и вынужденные выкупы."""
    mult = C.COMMISSION_SCENARIOS[C.BASELINE_SCENARIO]
    traj, res = run_best(trajs, r_base_all, calendar, theta, floor,
                         fold, C.SEEDS[0], mult)
    x = np.arange(len(traj))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(x, traj["CASH"].to_numpy(float), label="кэш агента",
            color="#ff7f0e", lw=1.6)
    ax.fill_between(x, 0, res["fund_pos"], label="позиция в фонде",
                    color="#2ca02c", alpha=0.45)
    ax.axhline(theta, color="#666666", ls="--", lw=1.0,
               label=f"theta={theta:.2f}")
    ax.axhline(floor, color="#999999", ls=":", lw=1.0,
               label=f"floor={floor:.2f}")
    ax.set_title(f"G3. Кэш и парковка, fold {fold}, сид {C.SEEDS[0]} "
                 f"(вынужденных выкупов: {res['forced_buybacks']})")
    ax.set_xlabel("шаг (решение)")
    ax.set_ylabel("доля портфеля")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# G4: d_ret ПРОТИВ МНОЖИТЕЛЯ КОМИССИИ
# ═══════════════════════════════════════════════════════════════════

def plot_commission(df, out_path):
    """G4: для каждой конфигурации — mean d_ret при x0.0 / x1.0 / x2.0."""
    scenarios = list(C.COMMISSION_SCENARIOS)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    xs = np.arange(len(scenarios))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, 9))

    pairs = sorted({(r.theta, r.floor) for r in df.itertuples()})
    for k, (t, f) in enumerate(pairs):
        vals = []
        for sc in scenarios:
            v = df[(df["theta"] == t) & (df["floor"] == f)
                   & (df["scenario"] == sc)]["d_ret_mean"]
            vals.append(float(v.mean()) * 100 if len(v) else np.nan)
        ax.plot(xs, vals, marker="o", lw=1.6, ms=5,
                color=colors[k % len(colors)],
                label=f"theta={t:.2f}/floor={f:.2f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{sc}\n(×{C.COMMISSION_SCENARIOS[sc]:.1f})"
                        for sc in scenarios])
    ax.axhline(0.0, color="#d62728", lw=1.0, ls="--")
    ax.set_xlabel("сценарий комиссии")
    ax.set_ylabel("mean d_ret, п.п.")
    ax.set_title(f"G4. Устойчивость к комиссии (ставка "
                 f"{C.COMMISSION_RATE * 100:.2f}% на операцию, только процент)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [OK] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 74)
    print("parking_exp ШАГ 6: графики G1-G4")
    print("=" * 74)
    print("  режим: ТОЛЬКО чтение артефактов, запись PNG в results/")
    print("  среда, модели, Optuna, fold 5: НЕ используются")

    df, trajs, r_base_all, calendar = load_inputs()
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    theta, floor = pick_best(df)
    print(f"\n[ЛУЧШАЯ КОНФИГУРАЦИЯ по N5] theta={theta:.2f}, floor={floor:.2f}")

    print("\n[G1] тепловая карта")
    plot_heatmap(df, os.path.join(C.RESULTS_DIR, "parking_heatmap.png"))

    # d_ret по фолдам для лучшей конфигурации (для выбора худшего/лучшего)
    sub = df[(df["scenario"] == C.BASELINE_SCENARIO)
             & (df["theta"] == theta) & (df["floor"] == floor)]
    d_fold = {int(r.fold): float(r.d_ret_mean) for r in sub.itertuples()}
    print("\n  d_ret по фолдам (лучшая конфигурация):")
    for f in sorted(d_fold):
        print(f"    fold {f}: {d_fold[f] * 100:+.4f} п.п.")

    print("\n[G2] эквити без парковки vs с парковкой")
    plot_equity(trajs, r_base_all, calendar, theta, floor, d_fold,
                os.path.join(C.RESULTS_DIR, "parking_equity.png"))

    worst = min(d_fold, key=lambda f: d_fold[f])
    print(f"\n[G3] кэш и парковка (худший фолд {worst})")
    plot_cash(trajs, r_base_all, calendar, theta, floor, worst,
              os.path.join(C.RESULTS_DIR, "parking_cash.png"))

    print("\n[G4] чувствительность к комиссии")
    plot_commission(df, os.path.join(C.RESULTS_DIR, "parking_commission.png"))

    print("\n" + "=" * 74)
    print("ШАГ 6 ЗАВЕРШЁН: 4 графика записаны в results/")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except KeyboardInterrupt:
        print("\n[ABORT] прервано пользователем")
        rc = 130
    except Exception:                             # noqa: BLE001
        traceback.print_exc()
        rc = 1
    sys.exit(rc)
