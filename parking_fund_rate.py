#!/usr/bin/env python3
"""
parking_fund_rate.py — ШАГ 2 mini-эксперимента парковки кэша (parking_exp v0.3).

НАЗНАЧЕНИЕ
    Получить ставку фонда, проверить её достоверность и построить ряд r_fund
    на строках датасета, чтобы шаг 3 мог считать арифметику парковки поверх
    готовых траекторий шага 1.

ЧТО ДЕЛАЕТ
    1) качает RUSFAR (MOEX ISS, board MMIX) с ПАГИНАЦИЕЙ и сверкой
       cursor.TOTAL (N11: ISS отдаёт максимум 100 строк на страницу);
    2) прогоняет гейты ставки: коридор 0..40% годовых, дубликаты дат,
       монотонность, пустые значения;
    3) подтверждает покрытие fold 0 (N14): первая дата ставки должна быть
       <= начала test-окна fold 0 (2019-06-14);
    4) качает пай SBMM (board TQTF, БПИФ -> не TQBR);
    5) сверяет ставку-минус-TER с ФАКТИЧЕСКИМ паем на общих датах
       (порог 0.5 п.п. годовых, R3.3);
    6) гэп-гейт пая по N12 (гибрид): гэп > 2 п.п. допустим, если дата
       примыкает к разрыву календаря > 10 дней ИЛИ входит в явный fallback
       {2022-02-24, 2022-02-25}; иначе — СТОП;
    7) строит r_fund по N2 (аддитивная сумма по календарным дням);
    8) пишет data/ruonia_daily.csv, data/sbmm_close.csv, лог в logs/.

ЧЕГО НЕ ДЕЛАЕТ
    - не открывает базу Optuna, не грузит модели, не читает траектории;
    - не обучает, не меняет portfolio_env.py, не трогает fold 5;
    - не строит политику парковки и не считает вердикт (это шаг 3).

Замечание по имени файла ставки
    data/ruonia_daily.csv назван так ради совместимости с ТЗ v0.2;
    фактическое содержимое — RUSFAR (F1 обновлён 2026-10-30).

Version: v1.0
Date:    2026-10-30
"""

import csv
import datetime as dt
import os
import sys
import traceback

import numpy as np
import pandas as pd
import requests

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
    """Пишет в консоль и в файл лога с явной кодировкой UTF-8."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

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


def setup_logging(tag="step2"):
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
# HTTP + ПАГИНАЦИЯ (N11)
# ═══════════════════════════════════════════════════════════════════

_HEADERS = {"User-Agent": C.USER_AGENT}


def _get_json(url, params_extra=None):
    url = url if params_extra is None else url.format(**params_extra)
    r = requests.get(url, headers=_HEADERS, timeout=C.HTTP_TIMEOUT)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "json" not in ctype.lower():
        raise RuntimeError(f"ожидался JSON, получено '{ctype}' ({len(r.content)} байт)")
    return r.json()


def fetch_rusfar(frm, till):
    """Скачать RUSFAR с пагинацией. Возвращает DataFrame [date, rate_pct].

    КРИТИЧНО (N11): /iss/history/... отдаёт максимум ISS_PAGE_SIZE строк.
    Без листания ряд МОЛЧА обрывается и результат неверен.
    После сборки сверяем длину с cursor.TOTAL.
    """
    rows = []
    total_expected = None
    start = 0
    for page in range(1, C.ISS_MAX_PAGES + 1):
        data = _get_json(C.RUSFAR_URL, {"frm": frm, "till": till, "start": start})
        hist = data.get("history", {})
        cols = hist.get("columns", [])
        recs = hist.get("data", [])
        cursor = data.get("history.cursor", {}).get("data", [])
        if cursor:
            total_expected = int(cursor[0][1])
        if not recs:
            break
        idx = {c: i for i, c in enumerate(cols)}
        for rec in recs:
            try:
                rows.append((rec[idx["TRADEDATE"]],
                             rec[idx["SECID"]],
                             rec[idx["BOARDID"]],
                             rec[idx["CLOSE"]]))
            except (KeyError, IndexError, TypeError):
                continue
        print(f"  стр.{page}: получено {len(recs):3d}, всего {len(rows):4d}"
              f"{'' if total_expected is None else f' / cursor.TOTAL={total_expected}'}")
        if total_expected is not None and len(rows) >= total_expected:
            break
        if len(recs) < C.ISS_PAGE_SIZE:
            break
        start += C.ISS_PAGE_SIZE
        import time
        time.sleep(C.HTTP_PAUSE)

    if not rows:
        raise RuntimeError("RUSFAR: пустой ответ")

    if C.ISS_VERIFY_CURSOR_TOTAL and total_expected is not None:
        if len(rows) != total_expected:
            raise RuntimeError(
                f"пагинация неполна: собрано {len(rows)}, cursor.TOTAL={total_expected}")
        print(f"  [OK] пагинация сверена: {len(rows)} == cursor.TOTAL {total_expected}")

    df = pd.DataFrame(rows, columns=["date", "secid", "board", "rate_pct"])
    # допускаем только строки нужной бумаги и площадки
    bad_sec = df[df["secid"] != C.RUSFAR_SECID]
    if len(bad_sec):
        print(f"  [WARN] отброшено строк с чужим SECID: {len(bad_sec)}")
    bad_board = df[df["board"] != C.RUSFAR_BOARD]
    if len(bad_board):
        print(f"  [WARN] отброшено строк с чужой BOARDID: {len(bad_board)}")
    df = df[(df["secid"] == C.RUSFAR_SECID) & (df["board"] == C.RUSFAR_BOARD)]
    df = df.drop(columns=["secid", "board"])
    df["date"] = pd.to_datetime(df["date"])
    df["rate_pct"] = pd.to_numeric(df["rate_pct"], errors="coerce")
    return df


def _fetch_sbmm_chunk(frm, till):
    """Один запрос candles на окно <= ~400 календарных дней (< 500 свечей)."""
    url = C.SBMM_URL.format(frm=frm, till=till)
    r = requests.get(url, headers=_HEADERS, timeout=C.HTTP_TIMEOUT)
    r.raise_for_status()
    blk = r.json().get("candles", {})
    cols, recs = blk.get("columns", []), blk.get("data", [])
    if not recs:
        return []
    idx = {c: i for i, c in enumerate(cols)}
    return [(rec[idx["begin"]][:10], rec[idx["close"]]) for rec in recs]


def fetch_sbmm(frm, till):
    """Скачать цену пая SBMM (board TQTF). Возвращает DataFrame [date, close].

    ПОЧЕМУ ЧАНКАМИ, А НЕ ЧЕРЕЗ `start` (найдено 2026-10-30):
    `candles.json` при interval=24 отдаёт до 500 свечей, но параметр `start`
    в этом эндпоинте НЕ сдвигает выдачу — запрос возвращает те же 500 записей.
    Наивный цикл `start += len(recs)` прокрутился 200 раз по 500 ДУБЛЕЙ и
    после drop_duplicates дал всего 500 строк (обрыв на 2023-10-25) — сверка
    с паем считалась на урезанном окне. Поэтому режем по ДАТАМ: каждая порция
    заведомо короче лимита, дубликаты невозможны, прогресс гарантирован.
    """
    rows = []
    cur = dt.date.fromisoformat(frm)
    end = dt.date.fromisoformat(till)
    step = dt.timedelta(days=300)          # ~205 торговых дней < 500 свечей
    chunk_no = 0
    while cur <= end:
        chunk_end = min(cur + step, end)
        part = _fetch_sbmm_chunk(str(cur), str(chunk_end))
        chunk_no += 1
        rows.extend(part)
        print(f"  чанк {chunk_no:2d}: {cur} .. {chunk_end} -> {len(part):3d} свечей,"
              f" всего {len(rows):4d}")
        cur = chunk_end + dt.timedelta(days=1)
        if chunk_no > C.ISS_MAX_PAGES:
            raise RuntimeError("SBMM: превышен предохранитель числа чанков")
        import time
        time.sleep(C.HTTP_PAUSE)
    if not rows:
        raise RuntimeError("SBMM: пустой ответ")

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    print(f"  итог: {len(df)} уникальных свечей, "
          f"{df['date'].min().date()} .. {df['date'].max().date()}")
    return df


# ═══════════════════════════════════════════════════════════════════
# ГЕЙТЫ
# ═══════════════════════════════════════════════════════════════════

def gate_rate(df):
    """Гейты ряда ставки. Возвращает (df_clean, ok)."""
    ok = True
    print(f"\n[ГЕЙТЫ СТАВКИ] строк до очистки: {len(df)}")

    n_null = int(df["rate_pct"].isna().sum())
    if n_null:
        print(f"  [WARN] пустых значений: {n_null} -> отброшены")
        df = df[df["rate_pct"].notna()]

    lo, hi = C.RATE_RANGE_PCT
    out = df[(df["rate_pct"] < lo) | (df["rate_pct"] > hi)]
    if len(out):
        print(f"  [FAIL] вне коридора {lo}..{hi}% годовых: {len(out)} строк")
        print(out.head(10).to_string(index=False))
        ok = False
    else:
        print(f"  [OK] все значения в коридоре {lo}..{hi}% годовых")

    df = df.sort_values("date").reset_index(drop=True)
    n_dup = int(df["date"].duplicated().sum())
    if n_dup:
        print(f"  [FAIL] дубликатов дат: {n_dup}")
        ok = False
    else:
        print("  [OK] дубликатов дат нет")

    if not df["date"].is_monotonic_increasing:
        print("  [FAIL] даты не монотонны после сортировки")
        ok = False
    else:
        print("  [OK] даты монотонны")

    print(f"  диапазон: {df['date'].min().date()} .. {df['date'].max().date()}"
          f", строк {len(df)}, средняя {df['rate_pct'].mean():.4f}% годовых")
    return df, ok


def gate_fold0_coverage(df):
    """N14: первая дата ставки должна покрывать старт test-окна fold 0."""
    first = df["date"].min().date()
    need = dt.date.fromisoformat(C.FOLD0_TEST_START_EXPECTED)
    print("\n[ГЕЙТ ПОКРЫТИЯ FOLD 0]")
    print(f"  первая дата ставки: {first} (ожидалась <= {need})")
    if first > need:
        print("  [FAIL] ставка не покрывает fold 0 -> прогон невозможен")
        return False
    print(f"  [OK] покрытие есть, запас {(need - first).days} календарных дней")
    return True


def compare_with_fund(rate_df, fund_df):
    """R3.3: сверка ставки-минус-TER с фактическим паем на общих датах."""
    r = rate_df.set_index("date")["rate_pct"]
    f = fund_df.set_index("date")["close"]
    common = r.index.intersection(f.index)
    print("\n[СВЕРКА СТАВКИ С ПАЕМ SBMM]")
    print(f"  общих дат: {len(common)}")
    if len(common) < 2:
        print("  [FAIL] слишком мало общих дат")
        return False, None
    r_c, f_c = r.loc[common].sort_index(), f.loc[common].sort_index()
    span = (f_c.index[-1] - f_c.index[0]).days
    fund_ann = (float(f_c.iloc[-1]) / float(f_c.iloc[0])) ** (C.DAYS_IN_YEAR / span) - 1.0
    rate_ann = float(r_c.mean()) / 100.0 - C.TER
    diff_pp = (rate_ann - fund_ann) * 100.0
    print(f"  окно: {f_c.index[0].date()} .. {f_c.index[-1].date()} ({span} дн.)")
    print(f"  ставка RUSFAR средняя: {float(r_c.mean()):.4f}% годовых")
    print(f"  ставка минус TER:      {rate_ann * 100:.4f}% годовых")
    print(f"  пай по факту:          {fund_ann * 100:.4f}% годовых")
    print(f"  РАЗНИЦА: {diff_pp:+.4f} п.п. годовых (порог ±{C.RATE_VS_FUND_MAX_DIFF_PP})")
    ok = abs(diff_pp) <= C.RATE_VS_FUND_MAX_DIFF_PP
    if ok:
        print("  [OK] в пределах порога")
    else:
        level = getattr(C, "RATE_VS_FUND_LEVEL", "warn")
        tag = "FAIL" if level == "stop" else "WARN"
        print(f"  [{tag}] расхождение сверх порога {C.RATE_VS_FUND_MAX_DIFF_PP} п.п."
              f" (уровень: {level})")
        print(f"  причина (зафиксирована в конфиге): {C.RATE_VS_FUND_REASON}")
        print("  следствие: сценарий 'base' = нижняя граница выгоды парковки,")
        print("             сценарий 'upper' (пай по факту) считается отдельно.")
    return ok, diff_pp


def gate_fund_gaps(fund_df):
    """N12 (ГИБРИД): гэпы пая > 2 п.п. допустимы по правилу ИЛИ по fallback."""
    print("\n[ГЭП-ГЕЙТ ПАЯ (N12, гибрид)]")
    f = fund_df.sort_values("date").reset_index(drop=True)
    ret_pct = (f["close"] / f["close"].shift(1) - 1.0) * 100.0
    gaps = f["date"].diff().dt.days

    small_neg = 0
    big = []
    for i in range(1, len(f)):
        x = ret_pct.iloc[i]
        if pd.isna(x):
            continue
        if x < 0 and abs(x) < C.FUND_NEG_SMALL_PCT:
            small_neg += 1
            continue
        if abs(x) > C.FUND_GAP_STOP_PP:
            d = f["date"].iloc[i].date()
            prev_gap = gaps.iloc[i]
            next_gap = gaps.iloc[i + 1] if i + 1 < len(gaps) else np.nan
            near_calendar_gap = ((pd.notna(prev_gap) and prev_gap > C.DATASET_GAP_DAYS)
                                 or (pd.notna(next_gap) and next_gap > C.DATASET_GAP_DAYS))
            in_fallback = str(d) in C.FUND_GAP_FALLBACK_DATES
            big.append((d, x, prev_gap, next_gap, near_calendar_gap, in_fallback))

    print(f"  мелких отрицательных дней (|r| < {C.FUND_NEG_SMALL_PCT}%): {small_neg}"
          f" (пометка, не стоп)")
    if not big:
        print("  [OK] гэпов сверх порога нет")
        return True

    print(f"  гэпов сверх {C.FUND_GAP_STOP_PP} п.п.: {len(big)}")
    ok = True
    for d, x, pg, ng, near, fb in big:
        by_rule = bool(near)
        covered = by_rule or fb
        src = "правило: разрыв календаря" if by_rule else ""
        if fb:
            src = (src + " + " if src else "") + "явный fallback"
        print(f"    {d}  {x:+.4f}%  разрыв до={pg} после={ng}  "
              f"-> {'ДОПУСТИМО' if covered else 'НЕ ПОКРЫТО'}  ({src or 'нет основания'})")
        if not covered:
            ok = False
    if not ok:
        print("  [FAIL] есть непокрытые гэпы -> СТОП")
    else:
        print("  [OK] все гэпы покрыты основанием (выше — построчно)")
    return ok


# ═══════════════════════════════════════════════════════════════════
# r_fund (N2)
# ═══════════════════════════════════════════════════════════════════

def build_daily_net_rate(rate_df):
    """Дневной ряд чистой ставки: rate/365 - TER/365, forward-fill на все дни.

    Нужен, чтобы r_fund начислялся за КАЛЕНДАРНЫЕ дни между строками
    датасета (в фичах есть выходные сессии, реальное РЕПО O/N начисляет
    проценты и за них).
    """
    s = rate_df.set_index("date")["rate_pct"].sort_index() / 100.0
    full = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full).ffill()
    return (s - C.TER) / C.DAYS_IN_YEAR


def build_r_fund(dataset_dates, daily_net_rate):
    """N2: r_fund(t) = сумма по дням d из (date_t; date_t+1] дневной чистой ставки."""
    idx = pd.DatetimeIndex(dataset_dates)
    vals = np.zeros(len(idx), dtype=np.float64)
    for i in range(len(idx) - 1):
        window = daily_net_rate.loc[
            (daily_net_rate.index > idx[i]) & (daily_net_rate.index <= idx[i + 1])]
        vals[i] = float(window.sum())
    vals[-1] = np.nan          # последняя строка не имеет следующей даты
    return vals


def annualized_check(dataset_dates, r_fund):
    """Санити: сумма r_fund за окно должна давать ставку-минус-TER за год."""
    idx = pd.DatetimeIndex(dataset_dates)
    r = np.nan_to_num(r_fund, nan=0.0)
    span = (idx[-1] - idx[0]).days
    ann = float(r.sum()) * C.DAYS_IN_YEAR / span
    return ann


def build_r_fund_upper(dataset_dates, fund_df):
    """Верхний сценарий: доходность пая SBMM на календарных днях шага.

    Аналог N2, но по ФАКТИЧЕСКОЙ доходности пая:
        r_fund_upper(t) = prod(1 + f_ret(d)) - 1  по дням (date_t; date_t+1]
    где f_ret(d) — дневная доходность пая (close/close_prev - 1).

    Даты моста (UPPER_EXCLUDE_GAP_DATES) дают доходность 0: скачок цены
    2022-02-24/25 — артефакт остановки торгов, а не доход фонда.

    Вне окна котировок пая -> NaN (см. UPPER_FALLBACK_TO_BASE: шаг 3 обязан
    явно перейти на базовый сценарий с пометкой в журнале).
    """
    f = fund_df.sort_values("date").set_index("date")["close"]
    daily = f.pct_change()
    if getattr(C, "UPPER_EXCLUDE_GAP_DATES", True):
        mask = pd.Index(daily.index.strftime("%Y-%m-%d")).isin(C.FUND_GAP_FALLBACK_DATES)
        if mask.any():
            daily = daily.copy()
            daily[mask] = 0.0
            print(f"  доходность пая: обнулено {int(mask.sum())} дней моста "
                  f"{sorted(C.FUND_GAP_FALLBACK_DATES)}")
    daily = daily.dropna()

    idx = pd.DatetimeIndex(dataset_dates)
    vals = np.full(len(idx), np.nan, dtype=np.float64)
    for i in range(len(idx) - 1):
        window = daily[(daily.index > idx[i]) & (daily.index <= idx[i + 1])]
        if len(window) == 0:
            continue
        vals[i] = float(np.prod(1.0 + window.values) - 1.0)
    return vals


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    setup_logging("step2")
    t0 = dt.datetime.now()
    print("═" * 74)
    print("parking_exp ШАГ 2: ставка фонда, сверка с паем, ряд r_fund")
    print(f"старт: {t0:%Y-%m-%d %H:%M:%S}")
    print("═" * 74)
    for line in C.describe():
        print(f"  {line}")

    if C.CREATE_DIRS_ON_RUN:
        os.makedirs(C.DATA_DIR, exist_ok=True)

    # диапазон: от старта RUSFAR до конца датасета
    feats = pd.read_pickle(C.FEATURES_FILE)
    ds_first, ds_last = feats.index[0].date(), feats.index[-1].date()
    print(f"\n  датасет: {len(feats)} строк, {ds_first} .. {ds_last}")

    print("\n" + "─" * 74)
    print("СВОДКА ПЕРЕД ЗАГРУЗКОЙ")
    print("─" * 74)
    print(f"  RUSFAR: {C.RUSFAR_SECID} / {C.RUSFAR_BOARD}, окно "
          f"{C.RUSFAR_HISTORY_START_EXPECTED} .. {ds_last}")
    print(f"  SBMM:   {C.SBMM_SECID} / {C.SBMM_BOARD}, окно "
          f"{C.SBMM_FIRST_TRADE} .. {ds_last}")
    print(f"  запись: {C.DATA_DIR}/{C.RUONIA_CSV}, {C.DATA_DIR}/{C.SBMM_CSV}")
    print("  база Optuna, модели, fold 5: НЕ используются")
    flush_all()
    if not confirm("Начать загрузку шага 2?"):
        print("[CANCEL] отменено (безопасный вариант)")
        return 2

    # ── RUSFAR ──
    print("\n" + "═" * 74)
    print("ЗАГРУЗКА СТАВКИ RUSFAR (с пагинацией)")
    print("═" * 74)
    try:
        rate_df = fetch_rusfar(C.RUSFAR_HISTORY_START_EXPECTED, str(ds_last))
    except Exception as e:                        # noqa: BLE001
        print(f"[ERROR] загрузка RUSFAR не удалась: {type(e).__name__}: {e}")
        return 1
    rate_df, ok_rate = gate_rate(rate_df)
    if not ok_rate:
        print("[ERROR] гейты ставки не пройдены -> стоп")
        return 1
    if not gate_fold0_coverage(rate_df):
        return 1
    print(f"\n  первые 3: {rate_df.head(3).to_dict('records')}")
    print(f"  последние 3: {rate_df.tail(3).to_dict('records')}")

    out_rate = os.path.join(C.DATA_DIR, C.RUONIA_CSV)
    rate_df.assign(date=rate_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        out_rate, index=False, encoding=C.LOG_ENCODING)
    print(f"  [OK] сохранено: {out_rate} ({len(rate_df)} строк)")

    # ── SBMM ──
    print("\n" + "═" * 74)
    print("ЗАГРУЗКА ПАЯ SBMM")
    print("═" * 74)
    try:
        fund_df = fetch_sbmm(C.SBMM_FIRST_TRADE, str(ds_last))
    except Exception as e:                        # noqa: BLE001
        print(f"[ERROR] загрузка SBMM не удалась: {type(e).__name__}: {e}")
        return 1
    print(f"  строк: {len(fund_df)}, {fund_df['date'].min().date()} .. "
          f"{fund_df['date'].max().date()}")
    out_fund = os.path.join(C.DATA_DIR, C.SBMM_CSV)
    fund_df.assign(date=fund_df["date"].dt.strftime("%Y-%m-%d")).to_csv(
        out_fund, index=False, encoding=C.LOG_ENCODING)
    print(f"  [OK] сохранено: {out_fund}")

    # ── Сверка и гэп-гейт ──
    ok_cmp, diff_pp = compare_with_fund(rate_df, fund_df)
    ok_gaps = gate_fund_gaps(fund_df)
    if not ok_cmp and getattr(C, "RATE_VS_FUND_LEVEL", "warn") == "stop":
        print("\n[ERROR] уровень сверки = stop и порог превышен -> стоп")
        return 1
    if not ok_cmp:
        print("  [INFO] прогон продолжается: считаются ДВА сценария ставки")
        print("         (решение пользователя 2026-10-30, см. RATE_VS_FUND_REASON)")
    if not ok_gaps:
        print("\n[ERROR] гэп-гейт пая не пройден -> стоп")
        return 1

    # ── r_fund ──
    print("\n" + "═" * 74)
    print("РЯД r_fund НА СТРОКАХ ДАТАСЕТА (N2)")
    print("═" * 74)
    daily = build_daily_net_rate(rate_df)
    print(f"  дневной ряд чистой ставки: {len(daily)} дней, "
          f"{daily.index.min().date()} .. {daily.index.max().date()}")
    r_fund = build_r_fund(feats.index, daily)
    ann = annualized_check(feats.index, r_fund)
    print(f"  строк r_fund: {len(r_fund)} (последняя = NaN: нет следующей даты)")
    print(f"  сумма r_fund -> годовая: {ann * 100:.4f}% (должна быть ≈ ставка - TER)")
    n_nan = int(np.isnan(r_fund).sum())
    print(f"  NaN в ряду: {n_nan} (ожидается 1)")
    if n_nan != 1:
        print("  [WARN] неожиданное число NaN")

    # ── Верхний сценарий: фактическая доходность пая ──
    print("\n" + "─" * 74)
    print("СЦЕНАРИЙ 'upper': доходность пая SBMM (справочно, не для вердикта)")
    print("─" * 74)
    r_upper = build_r_fund_upper(feats.index, fund_df)
    valid = np.where(~np.isnan(r_upper))[0]
    if len(valid):
        i0, i1 = int(valid[0]), int(valid[-1])
        print(f"  покрыто строк: {len(valid)} из {len(r_upper)} "
              f"({feats.index[i0].date()} .. {feats.index[i1].date()})")
        print(f"  вне покрытия (NaN): {len(r_upper) - len(valid)} строк "
              f"-> шаг 3 возьмёт 'base' (UPPER_FALLBACK_TO_BASE="
              f"{getattr(C, 'UPPER_FALLBACK_TO_BASE', True)})")
        u_ann = annualized_check(feats.index[valid], r_upper[valid])
        # base обязан считаться на ТОМ ЖЕ подмножестве строк, иначе сравнение
        # некорректно: вне окна пая base=0 (ставки до 2019 нет), и разница
        # раздувается до +6.7 п.п. Ниже — сравнение «яблоки с яблоками».
        b_ann = annualized_check(feats.index[valid], r_fund[valid])
        print(f"  'upper' годовая (в окне пая): {u_ann * 100:.4f}%")
        print(f"  'base'  годовая (то же окно): {b_ann * 100:.4f}%")
        print(f"  'upper' минус 'base': {(u_ann - b_ann) * 100:+.4f} п.п. годовых")
        print("  (положительная разница = фактический пай доходнее RUSFAR-TER,")
        print("   т.е. базовый сценарий консервативен)")
    else:
        print("  [WARN] ни одной строки с доходностью пая")

    out_r = os.path.join(C.DATA_DIR, C.R_FUND_CSV)
    pd.DataFrame({
        "date": feats.index.strftime("%Y-%m-%d"),
        "r_fund_base": r_fund,
        "r_fund_upper": r_upper,
    }).to_csv(out_r, index=False, encoding=C.LOG_ENCODING, float_format="%.12g")
    print(f"\n  [OK] сохранено: {out_r} ({len(r_fund)} строк, оба сценария)")

    print("\n  ВЫБОРОЧНЫЕ ЗНАЧЕНИЯ (base / upper)")
    for i in (0, 1, len(r_fund) // 2, len(r_fund) - 2):
        d = feats.index[i].date()
        up = "n/a" if np.isnan(r_upper[i]) else f"{r_upper[i] * 100:+.5f}%"
        print(f"    {d}: base={r_fund[i] * 100:+.5f}%  upper={up}")

    dur = (dt.datetime.now() - t0).total_seconds()
    print(f"\n  время: {dur:.1f} с")
    print("═" * 74)
    print("ШАГ 2 ЗАВЕРШЁН УСПЕШНО")
    print("═" * 74)
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
