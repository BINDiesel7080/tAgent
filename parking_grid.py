#!/usr/bin/env python3
"""
parking_grid.py — ШАГ 3 mini-эксперимента парковки кэша (parking_exp v0.3).

НАЗНАЧЕНИЕ
    Чистая арифметика политики: поверх реальных траекторий кэша агента
    (получены на шаге 1) посчитать, окупается ли парковка свободного кэша
    в фонд денежного рынка. Среда НЕ инстанцируется, обучение НЕ идёт.

ЧТО ДЕЛАЕТ
    1) читает results/parking_traj_{fold}_{seed}.csv (35 файлов шага 1);
    2) читает data/r_fund_scenarios.csv (base = RUSFAR-TER, upper = пай SBMM);
    3) для каждой конфигурации политики (theta, floor) из сдвинутой сетки:
       - ведёт гистерезис по свободному кэшу агента;
       - исполняет перестановки только в торговые дни (дата есть в ряду
         ставки) — иначе переносит их (status="pending");
       - считает ОДИН журнал переключений (switch journal), а затраты
         оценивает в трёх комиссионных сценариях (x0.0 / x1.0 / x2.0);
       - накапливает eq_adj рекуррентно по N4 (парковочный вклад АДДИТИВЕН
         к доходности агента: кэш в базовой эквити дохода не приносит);
    4) агрегирует d_ret = eq_adj/base - 1 (относительный прирост эквити
       от парковки) по фолдам и сидам;
    5) выносит вердикт по заранее зафиксированным правилам (N8);
    6) пишет results/parking_grid.csv и лог в logs/.

ЧЕГО НЕ ДЕЛАЕТ
    - не инстанцирует PortfolioEnv, не грузит модели, не открывает Optuna;
    - не трогает fold 5 (holdout), portfolio_env.py, файлы шага 1;
    - не меняет сетку по результатам (сетка выбрана ДО прогона правилом N7).

Инвариант E5 (КРИТИЧНО)
    fund_pos(t) <= cash(t) на КАЖДОМ шаге. Нарушение — стоп как ошибка
    реализации: мёртвая зона иначе открыла бы скрытое плечо.

Version: v1.0
Date:    2026-10-30
"""

import csv
import datetime as dt
import glob
import os
import sys
import traceback

import numpy as np
import pandas as pd

import parking_config as C

try:
    if C.FORCE_UTF8:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════

class _Tee:
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, s):
        try:
            self._stream.write(s)
        except (UnicodeEncodeError, OSError):
            try:
                self._stream.write(s.encode("ascii", "replace").decode("ascii"))
            except Exception:                     # noqa: BLE001
                pass
        try:
            self._fh.write(s)
            self._fh.flush()
        except Exception:                         # noqa: BLE001
            pass
        return len(s)

    def flush(self):
        for x in (self._stream, self._fh):
            try:
                x.flush()
            except Exception:                     # noqa: BLE001
                pass

    def isatty(self):
        try:
            return bool(self._stream.isatty())
        except Exception:                         # noqa: BLE001
            return False


_LOG_HANDLE = None


def setup_logging(tag="step3"):
    global _LOG_HANDLE
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.LOGS_DIR, f"parking_{tag}_{stamp}.log")
    _LOG_HANDLE = open(path, "w", encoding=C.LOG_ENCODING, buffering=1)
    sys.stdout = _Tee(sys.stdout, _LOG_HANDLE)
    sys.stderr = _Tee(sys.stderr, _LOG_HANDLE)
    sys.excepthook = lambda t, v, tb: traceback.print_exception(t, v, tb)
    print(f"[LOG] {path}")
    return path


def flush_all():
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:                         # noqa: BLE001
            pass


def _auto_flag():
    raw = os.environ.get(C.AUTO_ENV_VAR)
    return None if raw is None else raw.strip().lower()


def confirm(prompt):
    flag = _auto_flag()
    if flag is not None and flag not in C.AUTO_FLAG_TRUTHY_ANY_EXCEPT:
        print(f"{prompt} -> авто-подтверждение ({C.AUTO_ENV_VAR}={flag!r})")
        return True
    try:
        ans = input(f"{prompt} [y/Enter]: ").strip().lower()
    except EOFError:
        ans = ""
    return ans in ("y", "yes", "д", "да")


# ═══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ВХОДНЫХ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════

def load_trajectories():
    """Читает все parking_traj_{fold}_{seed}.csv шага 1."""
    files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "parking_traj_*.csv")))
    if not files:
        raise RuntimeError(
            f"нет траекторий в {C.RESULTS_DIR}/ — сначала выполните шаг 1")
    print(f"\n[ТРАЕКТОРИИ] найдено файлов: {len(files)}")
    out = {}
    for p in files:
        df = pd.read_csv(p)
        fold = int(df["fold"].iloc[0])
        seed = int(df["seed"].iloc[0])
        out[(fold, seed)] = df
        print(f"  fold{fold} s{seed:<5d} строк={len(df):3d} "
              f"{df['decision_date'].iloc[0]} .. {df['decision_date'].iloc[-1]}")
    return out


def load_scenarios():
    """Читает data/r_fund_scenarios.csv: base и upper."""
    p = os.path.join(C.DATA_DIR, C.R_FUND_CSV)
    if not os.path.exists(p):
        raise RuntimeError(f"нет {p} — сначала выполните шаг 2")
    df = pd.read_csv(p, parse_dates=["date"])
    print(f"\n[СТАВКА] {p}: {len(df)} строк, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    return df


def load_rate_calendar():
    """Торговый календарь = даты, присутствующие в ряду ставки (N3)."""
    p = os.path.join(C.DATA_DIR, C.RUONIA_CSV)
    if not os.path.exists(p):
        raise RuntimeError(f"нет {p} — сначала выполните шаг 2")
    df = pd.read_csv(p, parse_dates=["date"])
    cal = set(df["date"].dt.strftime("%Y-%m-%d"))
    print(f"[КАЛЕНДАРЬ] торговых дней в ряду ставки: {len(cal)}")
    return cal


# ═══════════════════════════════════════════════════════════════════
# ПОЛИТИКА ПАРКОВКИ
# ═══════════════════════════════════════════════════════════════════

def compute_targets(cash, theta, floor, delta):
    """Гистерезис. Возвращает массив целевых позиций в фонде.

    Логика (зафиксирована в ТЗ):
        free(t)  = cash(t) + fund_pos(t-1)     — весь незанятый акциями капитал
        free > theta            -> входим/увеличиваем:
                                   target = min(max(0, cash - floor), cash)
        free < theta - delta    -> выходим полностью: target = 0
        иначе (мёртвая зона)    -> держим: target = min(fund_pos(t-1), cash)

    Инвариант E5: target <= cash ВСЕГДА (см. min(..., cash) во всех ветвях).
    """
    n = len(cash)
    target = np.zeros(n, dtype=np.float64)
    fund_prev = 0.0
    for i in range(n):
        c = float(cash[i])
        free = c + fund_prev
        if free > theta:
            t = min(max(0.0, c - floor), c)
        elif free < theta - delta:
            t = 0.0
        else:
            t = min(fund_prev, c)
        # предохранитель инварианта (страховка от float-краёв)
        if t > c + 1e-12:
            t = c
        if t < 0.0:
            t = 0.0
        target[i] = t
        fund_prev = t
    return target


def run_policy(traj, r_fund, calendar, theta, floor, delta,
               comm_mult, commission_rate):
    """Прогон одной траектории одной конфигурацией в одном комиссионном сценарии.

    Возвращает dict с метриками и журналом. Инвариант E5 проверяется построчно.
    """
    cash = traj["CASH"].to_numpy(dtype=np.float64)
    dec_dates = traj["decision_date"].to_numpy()
    base_ret = traj["net_ret"].to_numpy(dtype=np.float64)

    target = compute_targets(cash, theta, floor, delta)

    n = len(cash)
    fund_pos = np.zeros(n, dtype=np.float64)
    costs = np.zeros(n, dtype=np.float64)
    held = np.zeros(n, dtype=np.float64)
    pending = np.zeros(n, dtype=bool)
    switches = 0
    pending_carried = 0.0          # неисполненная заявка, ждущая торгового дня

    eq_adj = 1.0
    eq_series = np.ones(n + 1, dtype=np.float64)
    base_series = np.ones(n + 1, dtype=np.float64)

    forced = 0                     # вынужденные выкупы (кэш упал ниже парковки)
    for i in range(n):
        date_str = str(dec_dates[i])
        is_trading = date_str in calendar

        prev = held[i - 1] if i > 0 else 0.0
        # Неисполненная заявка прошлого неторгового дня имеет приоритет над
        # свежим решением: иначе отложенная перестановка терялась бы молча.
        if i > 0 and pending[i - 1]:
            want = pending_carried
        else:
            want = target[i]

        if is_trading:
            new_pos_raw = want
            pending[i] = False
            pending_carried = 0.0
        else:
            new_pos_raw = prev                  # не исполняем
            pending[i] = True
            pending_carried = want

        # E5 (жёсткий инвариант): парковка НЕ МОЖЕТ превышать свободный кэш.
        # Агент не знает о парковке: вложив кэш в акции, он «съедает» его
        # под ногами политики. Физически это выкуп из фонда, поэтому объём
        # ограничивается текущим кэшем, а не обрывает прогон ошибкой.
        new_pos = min(new_pos_raw, cash[i])
        if new_pos_raw > cash[i] + 1e-12:
            forced += 1
        if new_pos > cash[i] + 1e-12:           # страховка: обязано быть 0
            raise RuntimeError(
                f"НАРУШЕН ИНВАРИАНТ E5 на шаге {i} ({date_str}): "
                f"fund_pos={new_pos:.10f} > cash={cash[i]:.10f}")
        if new_pos < -1e-12:
            raise RuntimeError(f"ОТРИЦАТЕЛЬНАЯ позиция фонда на шаге {i}: {new_pos}")

        delta_pos = abs(new_pos - prev)
        if delta_pos > 1e-12:
            switches += 1
        cost = delta_pos * commission_rate * comm_mult

        fund_pos[i] = new_pos
        held[i] = new_pos
        costs[i] = cost

        # N4: парковочный вклад АДДИТИВЕН к доходности агента (кэш в базовой
        # эквити не приносит дохода). Поэтому eq_adj накапливает
        #   1 + net_ret(t) + held(t)*r_fund(t) - costs(t),
        # а не только парковочный множитель: иначе d_ret оказался бы
        # отношением парковочного множителя к доходности акций.
        rf = 0.0 if np.isnan(r_fund[i]) else float(r_fund[i])
        base_r = 0.0 if np.isnan(base_ret[i]) else float(base_ret[i])

        eq_adj *= (1.0 + base_r + new_pos * rf - cost)
        eq_series[i + 1] = eq_adj
        base_series[i + 1] = base_series[i] * (1.0 + base_r)

    base_final = float(base_series[-1])
    d_ret = eq_adj / base_final - 1.0 if base_final > 0 else 0.0

    return {
        "eq_adj": float(eq_adj),
        "base": base_final,
        "d_ret": float(d_ret),
        "switches": int(switches),
        "pending_steps": int(pending.sum()),
        "forced_buybacks": int(forced),
        "total_cost": float(costs.sum()),
        "fund_mean": float(fund_pos.mean()),
        "fund_max": float(fund_pos.max()),
        "days_in_fund": int((fund_pos > 1e-9).sum()),
        "eq_series": eq_series,
        "base_series": base_series,
        "fund_pos": fund_pos,
        "costs": costs,
    }


def max_drawdown(values):
    """Максимальная просадка в %."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return 0.0
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / peak
    return float(np.max(dd) * 100.0)


# ═══════════════════════════════════════════════════════════════════
# ВЕРДИКТ
# ═══════════════════════════════════════════════════════════════════

def evaluate_verdict(fold_means, conf_name):
    """Вердикт по заранее зафиксированным правилам N8."""
    pos_folds = [f for f, v in fold_means.items() if v > 0]
    n_pos = len(pos_folds)
    worst = min(fold_means.values())
    best = max(fold_means.values())

    confirm = n_pos >= C.VERDICT_CONFIRM_FOLDS_MIN
    tail_ok = worst >= C.VERDICT_TAIL_BARRIER_PP / 100.0

    if confirm and tail_ok:
        status = "ПОДТВЕРЖДЕНО"
    elif confirm:
        status = "ХВОСТ ПРОБИТ (mean>0, но барьер нарушен)"
    elif n_pos > 0:
        status = "СЛАБО"
    else:
        status = "ОТКЛОНЕНО"

    strict = None
    for f in C.STRICT_INTERPRET_FOLDS:
        if f in fold_means:
            strict = fold_means[f]
    neutral = None
    for f in C.NEUTRAL_INTERPRET_FOLDS:
        if f in fold_means:
            neutral = fold_means[f]

    return {
        "config": conf_name,
        "n_pos": n_pos,
        "status": status,
        "worst_pp": worst * 100,
        "best_pp": best * 100,
        "mean_pp": float(np.mean(list(fold_means.values()))) * 100,
        "strict_fold": strict,
        "neutral_fold": neutral,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    setup_logging("step3")
    t0 = dt.datetime.now()
    print("═" * 74)
    print("parking_exp ШАГ 3: сетка политики парковки, eq_adj и вердикт")
    print(f"старт: {t0:%Y-%m-%d %H:%M:%S}")
    print("═" * 74)
    for line in C.describe():
        print(f"  {line}")

    trajs = load_trajectories()
    scen = load_scenarios()
    calendar = load_rate_calendar()

    # индексация ставки по дате РЕШЕНИЯ (интервал (date_t; date_t+1] уже заложен)
    scen_idx = scen.set_index(scen["date"].dt.strftime("%Y-%m-%d"))
    r_base_all = scen_idx["r_fund_base"]
    r_upper_all = scen_idx["r_fund_upper"]

    configs = C.valid_configs(C.THETA_GRID_SHIFTED)
    scenarios_comm = C.COMMISSION_SCENARIOS
    n_cells = C.cell_count(len(configs))
    print(f"\n[СЕТКА] конфигураций {len(configs)}: {configs}")
    print(f"  комиссионных сценариев {len(scenarios_comm)}: {list(scenarios_comm)}")
    print(f"  сидов {len(C.SEEDS)}, фолдов {len(C.EVAL_FOLDS)} -> ячеек {n_cells}")

    print("\n" + "─" * 74)
    print("СВОДКА ПЕРЕД ПРОГОНОМ")
    print("─" * 74)
    print("  операции записи: results/parking_grid.csv (перезаписываемый), лог")
    print("  среда, модели, Optuna, fold 5: НЕ используются")
    print("  инвариант E5 (fund_pos <= cash) проверяется НА КАЖДОМ шаге")
    flush_all()
    if not confirm("Запустить шаг 3?"):
        print("[CANCEL] отменено (безопасный вариант)")
        return 2

    # ── Прогон ──
    print("\n" + "═" * 74)
    print("ПРОГОН СЕТКИ")
    print("═" * 74)

    rows = []
    cube = {}     # (theta, floor, comm, fold) -> [d_ret по сидам]
    for theta, floor in configs:
        delta = max(C.DELTA_GRID)          # ширина мёртвой зоны: берём максимум
        # ДЕЛЬТА: в ТЗ v0.3 мёртвая зона — параметр политики, но сетка по ней
        # не строится (иначе ячейки множатся). Фиксируем максимальную ширину:
        # она только подавляет лишние переключения, не искажая логику входа.
        for cname, mult in scenarios_comm.items():
            for fold in C.EVAL_FOLDS:
                d_list, sw_list, cost_list, pend_list = [], [], [], []
                forced_list = []
                fundmean_list, ddincl = [], []
                for seed in C.SEEDS:
                    key = (fold, seed)
                    if key not in trajs:
                        print(f"  [WARN] нет траектории fold{fold} s{seed}")
                        continue
                    traj = trajs[key]
                    dates = traj["decision_date"].to_numpy()
                    r_base = np.array(
                        [r_base_all.get(d, np.nan) for d in dates], dtype=np.float64)

                    res = run_policy(traj, r_base, calendar, theta, floor, delta,
                                     mult, C.COMMISSION_RATE)
                    d_list.append(res["d_ret"])
                    sw_list.append(res["switches"])
                    cost_list.append(res["total_cost"])
                    pend_list.append(res["pending_steps"])
                    forced_list.append(res["forced_buybacks"])
                    fundmean_list.append(res["fund_mean"])
                    ddincl.append(max_drawdown(res["eq_series"])
                                  - max_drawdown(res["base_series"]))
                if not d_list:
                    continue
                cube.setdefault((theta, floor, cname, fold), []).extend(d_list)
                rows.append({
                    "theta": theta, "floor": floor, "delta": delta,
                    "scenario": cname, "fold": fold, "n_seeds": len(d_list),
                    "d_ret_mean": float(np.mean(d_list)),
                    "d_ret_std": float(np.std(d_list)),
                    "d_ret_min": float(np.min(d_list)),
                    "d_ret_max": float(np.max(d_list)),
                    "switches_mean": float(np.mean(sw_list)),
                    "cost_mean": float(np.mean(cost_list)),
                    "pending_mean": float(np.mean(pend_list)),
                    "forced_mean": float(np.mean(forced_list)),
                    "fund_mean": float(np.mean(fundmean_list)),
                    "dd_diff_pp": float(np.mean(ddincl)),
                })
        # краткий прогресс по конфигурации
        sub = [r for r in rows if r["theta"] == theta and r["floor"] == floor
               and r["scenario"] == "x1.0"]
        if sub:
            m = np.mean([r["d_ret_mean"] for r in sub]) * 100
            print(f"  theta={theta:.2f} floor={floor:.2f}: "
                  f"база x1.0 -> mean d_ret {m:+.4f} п.п.")

    # ── Запись ──
    out = os.path.join(C.RESULTS_DIR, C.GRID_CSV)
    pd.DataFrame(rows).to_csv(out, index=False, encoding=C.LOG_ENCODING,
                              float_format="%.10g")
    print(f"\n[OK] сохранено: {out} ({len(rows)} строк)")

    # ── Агрегация и вердикт ──
    print("\n" + "═" * 74)
    print("РЕЗУЛЬТАТЫ ПО ФОЛДАМ (сценарий 'x1.0' — базовая комиссия)")
    print("═" * 74)

    verdicts = []
    for theta, floor in configs:
        fold_means = {}
        for fold in C.EVAL_FOLDS:
            v = cube.get((theta, floor, "x1.0", fold))
            if v:
                fold_means[fold] = float(np.mean(v))
        if not fold_means:
            continue
        name = f"theta={theta:.2f}/floor={floor:.2f}"
        print(f"\n  {name}")
        print(f"    {'fold':>4} | {'mean d_ret':>12} | {'std':>9} | {'ставка':<22}")
        for fold in sorted(fold_means):
            v = cube[(theta, floor, "x1.0", fold)]
            print(f"    {fold:>4} | {fold_means[fold] * 100:+11.4f} п.п. | "
                  f"{np.std(v) * 100:8.4f} | "
                  f"{C.FOLD_RATE_CONTEXT.get(fold, 'вне окна пая')}")
        vd = evaluate_verdict(fold_means, name)
        verdicts.append(vd)
        print(f"    -> {vd['status']}: mean {vd['mean_pp']:+.4f} п.п., "
              f"фолдов с d_ret>0: {vd['n_pos']}/{len(fold_means)}, "
              f"худший {vd['worst_pp']:+.4f} п.п.")
        if vd["strict_fold"] is not None:
            tag = "OK" if vd["strict_fold"] > 0 else "ВНИМАНИЕ"
            print(f"    строгий тест fold 0 (~7.5% годовых): "
                  f"{vd['strict_fold'] * 100:+.4f} п.п. [{tag}]")
        if vd["neutral_fold"] is not None:
            print(f"    нейтральная зона fold 1 (4.25-6.0%): "
                  f"{vd['neutral_fold'] * 100:+.4f} п.п. (знак не критичен)")

    # ── Справочно: upper и стресс ──
    print("\n" + "═" * 74)
    print("СПРАВОЧНО: верхний сценарий ставки и комиссионный стресс")
    print("═" * 74)
    best = max(verdicts, key=lambda d: d["mean_pp"]) if verdicts else None
    if best:
        parts = best["config"].replace("theta=", "").replace("floor=", "").split("/")
        bt, bf = float(parts[0]), float(parts[1])
        print(f"  лучшая конфигурация (N5, максимум mean d_ret): {best['config']}")
        for sc in ("x0.0", "x1.0", "x2.0"):
            vals = []
            for fold in C.EVAL_FOLDS:
                v = cube.get((bt, bf, sc, fold))
                if v:
                    vals.append(np.mean(v))
            if vals:
                print(f"    {sc}: mean d_ret {np.mean(vals) * 100:+.4f} п.п., "
                      f"фолдов>0: {sum(1 for x in vals if x > 0)}/{len(vals)}")
        # Верхний сценарий ставки считается ОТДЕЛЬНО: цикл выше идёт только по
        # комиссионным сценариям, поэтому ветки "upper" в нём нет.
        up_vals = []
        for fold in C.EVAL_FOLDS:
            fold_d = []
            for seed in C.SEEDS:
                key = (fold, seed)
                if key not in trajs:
                    continue
                traj = trajs[key]
                dates = traj["decision_date"].to_numpy()
                r_base = np.array(
                    [r_base_all.get(d, np.nan) for d in dates], dtype=np.float64)
                r_up = np.array(
                    [r_upper_all.get(d, np.nan) for d in dates], dtype=np.float64)
                # где покрытия пая нет -> используем базовый (UPPER_FALLBACK_TO_BASE)
                r_use = np.where(np.isnan(r_up), r_base, r_up)
                res = run_policy(traj, r_use, calendar, bt, bf, max(C.DELTA_GRID),
                                 scenarios_comm[C.BASELINE_SCENARIO], C.COMMISSION_RATE)
                fold_d.append(res["d_ret"])
            if fold_d:
                up_vals.append(float(np.mean(fold_d)))
        if up_vals:
            print(f"    upper (пай по факту): mean d_ret "
                  f"{np.mean(up_vals) * 100:+.4f} п.п., фолдов>0: "
                  f"{sum(1 for x in up_vals if x > 0)}/{len(up_vals)}")
            print("    — справочно, НЕ для вердикта (base всегда <= upper по построению)")

    # ── Итог ──
    print("\n" + "═" * 74)
    print("ИТОГ")
    print("═" * 74)
    confirmed = [v for v in verdicts if v["status"] == "ПОДТВЕРЖДЕНО"]
    print(f"  конфигураций проверено: {len(verdicts)}")
    print(f"  ПОДТВЕРЖДЕНО (mean d_ret>0 в >= {C.VERDICT_CONFIRM_FOLDS_MIN}/5 "
          f"и барьер {C.VERDICT_TAIL_BARRIER_PP} п.п. не пробит): {len(confirmed)}")
    if confirmed:
        print("  -> ПАРКОВКА ОКУПАЕТСЯ на исторических траекториях кэша (базовый сценарий)")
    else:
        print("  -> ПАРКОВКА НЕ ОКУПАЕТСЯ по зафиксированным критериям")
    print(f"\n  ОГОВОРКА (N8): хвостовой барьер {C.VERDICT_TAIL_BARRIER_PP} п.п. лежит")
    print("  ВНУТРИ ФАКТИЧЕСКОГО шума прокси. Уточнение N8 (факт, 2026-10-30):")
    print("  сверка RUSFAR против пая SBMM дала −0.9529 п.п. годовых против")
    print("  оценки ±0.5 п.п./год при формулировке барьера, то есть шум прокси")
    print("  до ±0.27 п.п. на 100-дневное окно (ранее оценивали ±0.14 п.п.).")
    print("  Решающее правило — 'mean d_ret > 0'; барьер защищает только от")
    print("  катастрофических хвостов. Эксперимент проверяет АРИФМЕТИКУ ПОЛИТИКИ")
    print("  на исторических траекториях, а не качество агента; настоящая")
    print("  проверка — теневой контур (слои L4/L5).")

    dur = (dt.datetime.now() - t0).total_seconds()
    print(f"\n  время: {dur:.1f} с")
    flush_all()
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
    flush_all()
    sys.exit(rc)
