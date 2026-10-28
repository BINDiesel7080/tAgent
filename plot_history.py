# -*- coding: utf-8 -*-
"""
plot_history.py — история цен + кризисы + горизонт агента (старт фичей).
Примеры:
  python plot_history.py --features-file rl_features_v2.pkl --out history.png
  python plot_history.py --file prices.pkl --features-file rl_features_v2.pkl --out history.png
"""
import argparse, sys, traceback
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

CRISIS_BANDS = [
    ("Санкции/рубль 2014", "2014-06-01", "2015-01-31"),
    ("Q4 2018",            "2018-10-01", "2018-12-31"),
    ("COVID-обвал",        "2020-02-20", "2020-03-31"),
    ("2022 год",           "2022-01-01", "2022-12-31"),
]
HALT_START, HALT_END = "2022-02-25", "2022-03-24"  # остановка торгов MOEX


def load_pkl(path):
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for key in ("prices", "data", "df", "close"):
            if key in obj and isinstance(obj[key], pd.DataFrame):
                return obj[key]
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    raise ValueError(f"Не понял структуру {path!r}: тип {type(obj)}. Пришлите вывод scriptа целиком.")


def pick_price_columns(df):
    cols = [c for c in df.columns if str(c).lower().endswith("_close")]
    if cols:
        return cols, "суффикс _close"
    cols = [c for c in df.columns if "close" in str(c).lower()]
    if cols:
        return cols, "подстрока close"
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return num, "числовые колонки (fallback)"


def to_datetime_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for col in ("date", "Date", "datetime", "time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    try:
        idx = pd.to_datetime(df.index)
        df.index = idx
        return df.sort_index()
    except Exception:
        raise ValueError("Не нашёл дату: ни DatetimeIndex, ни колонки date. Пришлите список колонок.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=False, default=None, help="pickle с ценами (OHLC)")
    ap.add_argument("--features-file", default="rl_features_v2.pkl")
    ap.add_argument("--features-start", default=None, help="переопределить старт фичей, YYYY-MM-DD")
    ap.add_argument("--out", default="history.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(14, 7))

    # --- цены ---
    if args.file:
        try:
            df = to_datetime_index(load_pkl(args.file))
            cols, how = pick_price_columns(df)
            print(f"[цены] файл: {args.file}")
            print(f"[цены] диапазон: {df.index.min()} .. {df.index.max()}")
            print(f"[цены] колонки ({how}): {cols}")
            first_valid = df[cols].apply(lambda s: s.first_valid_index())
            for c in cols:
                print(f"[цены] {c}: данных с {first_valid[c]}")
            for c in cols:
                s = df[c].dropna()
                if len(s) == 0:
                    continue
                ax.plot(s.index, s / s.iloc[0], lw=1.2, alpha=0.9, label=c.replace("_close", ""))
        except Exception:
            print("!!! Не удалось разобрать файл цен. Трейсбек ниже — пришлите его мне.")
            traceback.print_exc()
            print("Колонки файла:", list(load_pkl(args.file).columns)[:40] if args.file else None)
    else:
        print("[цены] файл цен не задан — рисую только горизонт агента. Добавьте --file для кривых.")

    # --- кризисы ---
    for name, start, end in CRISIS_BANDS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(s, e, color="grey", alpha=0.15, zorder=0)
    ax.axvspan(pd.Timestamp(HALT_START), pd.Timestamp(HALT_END),
               facecolor="none", hatch="///", edgecolor="red", lw=0, zorder=1)

    # --- горизонт агента ---
    feat_start = None
    if args.features_start:
        feat_start = pd.Timestamp(args.features_start)
    elif args.features_file:
        try:
            fdf = load_pkl(args.features_file)
            fdf = to_datetime_index(fdf)
            feat_start = fdf.index.min()
            print(f"[фичи] файл: {args.features_file}")
            print(f"[фичи] диапазон: {fdf.index.min()} .. {fdf.index.max()}  (строк: {len(fdf)})")
        except Exception:
            print("!!! Не удалось разобрать файл фичей. Трейсбек ниже — пришлите его мне.")
            traceback.print_exc()
    if feat_start is not None:
        ax.axvline(feat_start, color="tab:blue", ls="--", lw=1.5)

    # --- подписи ---
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color="grey", alpha=0.15, label="кризисные периоды"))
    handles.append(mpatches.Patch(facecolor="none", hatch="///", edgecolor="red", label="остановка торгов 02–03.2022"))
    if feat_start is not None:
        handles.append(Line2D([0], [0], color="tab:blue", ls="--", label=f"старт фичей ({feat_start.date()})"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.set_title("История цен: кризисы и горизонт агента")
    ax.set_ylabel("Нормированная цена (старт = 1.0)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"График сохранён: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
