
# -*- coding: utf-8 -*-
"""
dedup_cache.py — диагностика и чистка кэша свечей от дубликатов дат.

Что делает:
1) по каждому CSV из папки кэша печатает: полные дубли строк,
   дубли дат, дубли дат с РАЗНЫМИ данными;
2) чистит: сортировка по дате, на каждую дату остаётся одна строка.
   Если за дату было несколько строк с разными значениями —
   оставляем последнюю (вечерняя сессия / более поздняя выгрузка);
3) перезаписывает файл, оригинал сохраняет один раз в *.bak
   (существующий .bak не трогает).

Запуск:
    python dedup_cache.py                 # кэш по умолчанию ./candles_cache
    python dedup_cache.py --cache DIR     # другая папка
"""

import os
import sys
import glob
import shutil
import argparse

import pandas as pd

DATE_CANDIDATES = ["date", "trade_date", "tradedate", "tradedate_", "begin"]
BACKUP_EXT = ".bak"


def find_date_col(df):
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in DATE_CANDIDATES:
        if cand in cols:
            return cols[cand]
    return None


def dedup_file(path):
    """Возвращает (was_dirty, n_full_dups, n_conflicts, n_final) или None при сбое."""
    df = pd.read_csv(path)
    if df.empty:
        print(f"  {os.path.basename(path)}: файл пуст - пропускаю")
        return None

    date_col = find_date_col(df)
    if date_col is None:
        print(f"  {os.path.basename(path)}: не нашёл колонку с датой "
              f"(колонки: {', '.join(map(str, df.columns[:8]))}) - пропускаю")
        return None

    dates = pd.to_datetime(df[date_col], errors="coerce")
    bad = int(dates.isna().sum())
    if bad == len(df):
        print(f"  {os.path.basename(path)}: даты не распознаются - пропускаю")
        return None
    if bad:
        print(f"  {os.path.basename(path)}: {bad} строк с нечитаемой датой удалю")
    df = df.loc[dates.notna()].copy()
    df[date_col] = dates.loc[df.index]

    n_total = len(df)
    n_full_dups = int(df.duplicated().sum())          # полные копии строк
    n_date_dups = int(df[date_col].duplicated().sum())  # дубли дат вообще
    # конфликты = даты, у которых строки-кандидаты различаются между собой
    data_cols = [c for c in df.columns if c != date_col]
    n_conflicts = int(
        df.groupby(date_col)[data_cols]
          .apply(lambda g: 1 if g.drop_duplicates().shape[0] > 1 else 0)
          .sum()
    )

    # stable-сортировка обязательна: только она гарантирует, что "последняя" строка
    # за дату - это именно более поздняя строка файла (вечерняя сессия), а не случайная
    df = df.sort_values(date_col, kind="stable").drop_duplicates(subset=date_col, keep="last")
    df = df.sort_values(date_col).reset_index(drop=True)

    was_dirty = (n_date_dups > 0) or (bad > 0)
    if was_dirty:
        bak = path + BACKUP_EXT
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        df.to_csv(path, index=False)

    return was_dirty, n_full_dups, n_conflicts, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="candles_cache",
                    help="папка кэша (по умолчанию candles_cache)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache, "*.csv")))
    if not files:
        print(f"В {args.cache} нет ни одного CSV - выход.")
        return

    print("=" * 64)
    print(f"  ДЕДУПЛИКАЦИЯ КЭША: {args.cache} | файлов: {len(files)}")
    print("=" * 64)

    total_conflicts = 0
    for path in files:
        res = dedup_file(path)
        if res is None:
            continue
        was_dirty, n_full, n_conf, n_final = res
        total_conflicts += n_conf
        status = "почищен -> *.bak" if was_dirty else "чист"
        print(f"  {os.path.basename(path):<14} {status:<18} "
              f"полных дублей: {n_full}, конфликтов дат: {n_conf}, "
              f"итог баров: {n_final}")

    print("-" * 64)
    if total_conflicts:
        print(f"  Внимание: {total_conflicts} дат имели РАЗНЫЕ значения - "
              f"оставлена последняя строка (вечерняя сессия). "
              f"Оригиналы лежат рядом в *.bak - можно проверить руками.")
    print("Готово. Теперь можно перезапускать: python feature_screening.py")


if __name__ == "__main__":
    main()
