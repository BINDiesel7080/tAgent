"""
plot_progress.py — прогресс Optuna-поиска v1.2 в виде графика (READ-ONLY к базе).

Version: 1.0
Date:    2026-10-28

Что делает:
  1. Открывает базу Optuna (по умолчанию optuna_ppo_v1_2.db, study ppo_v1.2)
     строго read-only (sqlite URI ?mode=ro&uri=true). Если read-only не
     открывается (WAL запущенного прогона) — копирует базу (+ -wal/-shm)
     во временную папку и читает копию. Оригинал не модифицируется.
  2. Рисует: значения всех триалов (тонкая линия), running best по
     завершённым (жирная линия), пруннутые триалы — крестиками,
     горизонтальный эталон v1.1 trial 14 robust +0.0052.
  3. Печатает топ-5 завершённых триалов по value.
  4. Сохраняет plots/optuna_progress_YYYYMMDD_HHMMSS.png (dpi=150).
  5. Дублирует консольный вывод в logs/plot_progress_YYYYMMDD_HHMMSS.log.

Запуск (только вручную, НЕ встраивается в train_ppo):
    python plot_progress.py
    python plot_progress.py --db optuna_ppo_v1_1.db --study ppo_v1.1

Отклонение от ТЗ по факту: файла optuna_ppo_v1_2.db в workspace пока нет
(прогон v1.2 не запускался), поэтому механика проверена на optuna_ppo_v1_1.db
через явные --db/--study. Дефолт остаётся v1.2 и подхватится, как только
прогон создаст базу.
"""

import argparse
import datetime
import os
import shutil
import sqlite3
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")          # ОБЯЗАТЕЛЬНО до pyplot: без окон при savefig
import matplotlib.pyplot as plt
import numpy as np
import optuna

DEFAULT_DB = "optuna_ppo_v1_2.db"
DEFAULT_STUDY = "ppo_v1.2"
BENCHMARK_VALUE = 0.0052       # trial 14 из v1.1 (робастная метрика)
BENCHMARK_LABEL = "v1.1 trial 14 robust +0.0052"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

_LOG_FH = None


def log(*args, sep=" ", end="\n"):
    """Печать в консоль + дублирование в лог-файл (если открыт)."""
    msg = sep.join(str(a) for a in args)
    print(msg, end=end)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(msg + end)
            _LOG_FH.flush()
        except (ValueError, OSError):
            pass


def open_log():
    """Новый лог-файл в logs/ (существующие логи не трогаем)."""
    global _LOG_FH
    os.makedirs(LOGS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOGS_DIR, f"plot_progress_{stamp}.log")
    _LOG_FH = open(path, "a", encoding="utf-8", buffering=1)
    return path


def load_study_readonly(db_file, study_name, attempts=3, retry_sleep=10):
    """Загрузить study. Возвращает (study|None, описание источника|None).

    Сначала пробуем read-only URI к оригиналу. При невозможности — копия во
    временную папку (оригинал не модифицируется). При 'database is locked'
    спим и повторяем, максимум `attempts` попыток.
    """
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///file:{db_file}?mode=ro&uri=true",
                engine_kwargs={"connect_args": {"uri": True, "timeout": 30}},
            )
            study = optuna.load_study(study_name=study_name, storage=storage)
            return study, f"read-only URI: sqlite:///file:{db_file}?mode=ro&uri=true"
        except Exception as e:              # noqa: BLE001 - нужен широкий перехват
            last_err = e
            text = f"{type(e).__name__}: {e}"
            log(f"[WARN] read-only попытка {attempt}/{attempts} не удалась: {text}")
            if "locked" in text.lower() and attempt < attempts:
                log(f"[INFO] database is locked -> sleep {retry_sleep}s и повтор")
                time.sleep(retry_sleep)
                continue
            break

    # Fallback: копия базы во временную папку (оригинал только читается)
    log("[INFO] fallback: копирую базу во временную папку (оригинал не трогаем)")
    try:
        tmpdir = tempfile.mkdtemp(prefix="optuna_ro_")
        base = os.path.basename(db_file)
        target = os.path.join(tmpdir, base)
        shutil.copy2(db_file, target)
        for suf in ("-wal", "-shm"):
            src = db_file + suf
            if os.path.exists(src):
                try:
                    shutil.copy2(src, target + suf)
                except (OSError, shutil.Error) as e:
                    log(f"[WARN] не удалось скопировать {src}: {e}")
        rw_storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{target.replace(os.sep, '/')}",
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.load_study(study_name=study_name, storage=rw_storage)
        return study, f"копия в tempdir: {target} (оригинал: {db_file})"
    except Exception as e:                  # noqa: BLE001
        log(f"[ERROR] fallback-копия тоже не удалась: {type(e).__name__}: {e}")
        log(f"[ERROR] исходная ошибка read-only: {last_err}")
        return None, None


def plot_progress(study, study_name, out_path):
    """Нарисовать прогресс и сохранить в out_path (dpi=150)."""
    trials = [t for t in study.trials if t.state != optuna.trial.TrialState.RUNNING]
    trials.sort(key=lambda t: t.number)

    complete = [t for t in trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    if trials:
        xs = [t.number for t in trials if t.value is not None]
        ys = [t.value for t in trials if t.value is not None]
        ax.plot(xs, ys, marker='.', markersize=4, linewidth=0.8, alpha=0.6,
                color='steelblue', label=f"trial value (n={len(xs)})")

    best_x, best_y, cur = [], [], -np.inf
    for t in complete:
        cur = max(cur, t.value)
        best_x.append(t.number)
        best_y.append(cur)
    if best_x:
        ax.plot(best_x, best_y, linewidth=2.2, color='royalblue',
                label=f"running best (n={len(complete)})")
        ax.annotate(f"best {cur:+.4f} (trial #{best_x[int(np.argmax(best_y))]})",
                    xy=(best_x[-1], best_y[-1]),
                    xytext=(6, -14), textcoords='offset points',
                    fontsize=8, color='navy')

    if pruned:
        px = [t.number for t in pruned]
        py = [t.value if t.value is not None else np.nan for t in pruned]
        ax.plot(px, py, linestyle='none', marker='x', color='tab:red',
                markersize=7, label=f"pruned (n={len(pruned)})")

    ax.axhline(BENCHMARK_VALUE, linestyle='--', color='green', linewidth=1.2,
               label=BENCHMARK_LABEL)

    if not complete:
        ax.text(0.5, 0.5, "no completed trials yet", transform=ax.transAxes,
                ha='center', va='center', fontsize=16, color='gray', alpha=0.7)

    ax.set_xlabel("Номер триала")
    ax.set_ylabel("Robust value (mean - 0.5*std)")
    ax.set_title(f"Optuna {study_name}: {len(complete)} complete / "
                 f"{len(pruned)} pruned / {len(failed)} fail")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return len(complete), len(pruned), len(failed)


def print_top(complete, top_n=5):
    """Топ-N завершённых триалов по value."""
    if not complete:
        log("\n[WARN] Нет завершённых триалов — топ печатать нечего.")
        return
    top = sorted(complete, key=lambda t: t.value, reverse=True)[:top_n]
    log(f"\nТОП-{len(top)} завершённых триалов по value:")
    log("-" * 72)
    for rank, t in enumerate(top, 1):
        log(f"#{rank}  trial {t.number:>4}  value={t.value:+.6f}")
        for k, v in t.params.items():
            log(f"      {k} = {v}")
        log("-" * 72)


def main():
    parser = argparse.ArgumentParser(description="Progress plot для Optuna (read-only)")
    parser.add_argument("--db", default=DEFAULT_DB,
                        help=f"файл базы (default {DEFAULT_DB})")
    parser.add_argument("--study", default=DEFAULT_STUDY,
                        help=f"study_name (default {DEFAULT_STUDY})")
    args = parser.parse_args()

    log_path = open_log()
    log(f"[LOG] Лог: {log_path}")
    log(f"[INFO] База: {args.db}")
    log(f"[INFO] study_name: {args.study}")

    if not os.path.exists(args.db):
        log(f"[ERROR] Файл базы не найден: {args.db} — выход.")
        return 1

    study, storage_desc = load_study_readonly(args.db, args.study)
    if study is None:
        log("[ERROR] Не удалось открыть study — выход.")
        return 1
    log(f"[INFO] Использован источник: {storage_desc}")

    all_trials = study.trials
    complete = [t for t in all_trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    log(f"[INFO] Всего триалов в study: {len(all_trials)}, завершённых: {len(complete)}")
    if not complete:
        log("[WARN] Нет завершённых триалов — график будет с пометкой "
            "'no completed trials yet'.")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(PLOTS_DIR, f"optuna_progress_{stamp}.png")
    n_comp, n_pruned, n_fail = plot_progress(study, args.study, out_path)

    log(f"[INFO] complete/pruned/fail: {n_comp}/{n_pruned}/{n_fail}")
    print_top(complete, top_n=5)
    log(f"\n[OK] График сохранён: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except sqlite3.OperationalError as e:
        log(f"[ERROR] sqlite: {e}")
        sys.exit(1)
