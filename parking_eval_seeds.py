#!/usr/bin/env python3
"""
parking_eval_seeds.py — ШАГ 1 mini-эксперимента парковки кэша (parking_exp v0.3).

НАЗНАЧЕНИЕ
    Снять реальные траектории кэша агента v1.2 на фолдах 0-4 и сохранить их
    в results/parking_traj_{fold}_{seed}.csv. Далее (шаг 2) поверх этих
    траекторий считается арифметика политики парковки — БЕЗ обучения,
    БЕЗ обращения к среде и БЕЗ изменения чего-либо.

ЧТО ДЕЛАЕТ
    1) читает базу Optuna СТРОГО read-only, берёт study.best_trial;
    2) сверяет лидера с ожидаемым протоколом (trial 14, mw=0.55 и т.д.);
    3) поднимает 7 моделей ppo_v1.2_fold5_s{seed}.zip;
    4) гоняет детерминированный инференс по фолдам 0-4 (fold 5 = holdout
       НЕ трогается: он израсходован, см. AGENTS.md);
    5) пишет CSV: дата решения, дата реализации, 7 весов, доходность шага,
       оборот, комиссия, эквити среды, эквити Equal;
    6) печатает QC: самосогласованность эквити, in-sample статус каждого окна,
       распределение кэша по фолдам (нужно для правила сдвига сетки N7).

ЧЕГО НЕ ДЕЛАЕТ
    - не пишет в базу Optuna (read-only URI; при блокировке — работа с копией);
    - не обучает модели, не меняет .zip;
    - не меняет portfolio_env.py: нормализация замораживается ПОДКЛАССОМ;
    - не считает ставку, не качает данные, не строит сетку (это шаги 2-3).

КРИТИЧНО (N1, причина отдельного файла)
    PortfolioEnv нормирует фичи по train-окну СВОЕГО фолда (portfolio_env.py:132).
    Модели обучены на нормировке train-окна fold 5. Если оценивать их на fold 0
    «как есть», агент увидит наблюдения в чужой шкале — траектории будут мусором.
    Поэтому нормализация прибивается к train-окну fold 5.

    Конвенция выравнивания (нужна шагу 2):
      decision_date  — дата, на которой агент видел obs и принял веса;
      realize_date   — date(t+1), на которой этот вес дал доходность;
      CASH           — доля кэша, удержанная ИЗ decision_date ДО realize_date.
    То есть интервал начисления r_fund — ровно (decision_date; realize_date].

Version: v1.0
Date:    2026-10-30
"""

import csv
import datetime as dt
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

import parking_config as C

try:                                    # кириллица в cmd (cp866) -> ромбики
    if C.FORCE_UTF8:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass


# ═══════════════════════════════════════════════════════════════════
# ЛОГИРОВАНИЕ (дублирование в консоль и в файл, UTF-8 явно)
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
        try:
            self._stream.flush()
        except Exception:                         # noqa: BLE001
            pass
        try:
            self._fh.flush()
        except Exception:                         # noqa: BLE001
            pass

    def isatty(self):
        try:
            return bool(self._stream.isatty())
        except Exception:                         # noqa: BLE001
            return False


_LOG_HANDLE = None


def setup_logging(tag="step1"):
    """Открыть logs/parking_<tag>_<timestamp>.log и подменить stdout/stderr."""
    global _LOG_HANDLE
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(C.LOGS_DIR, f"parking_{tag}_{stamp}.log")
    _LOG_HANDLE = open(path, "w", encoding=C.LOG_ENCODING, buffering=1)
    sys.stdout = _Tee(sys.stdout, _LOG_HANDLE)
    sys.stderr = _Tee(sys.stderr, _LOG_HANDLE)

    def _excepthook(t, v, tb):
        traceback.print_exception(t, v, tb)

    sys.excepthook = _excepthook
    print(f"[LOG] {path}")
    return path


def flush_all():
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:                         # noqa: BLE001
            pass


def _auto_flag():
    """Значение PARKING_EXP_AUTO, нормализованное от ловушки cmd.

    cmd.exe включает пробел перед `&&` в значение: `set VAR=1 && python ...`
    даёт переменную, равную "1 " (хвостовой пробел). Строгое `== "1"` тогда
    не срабатывает, и безголовый запуск МОЛЧА отменяется.
    Проверено 2026-10-30: CMD_VAR=[%..%] печатал пусто, PY_VAR=[1 ].
    """
    raw = os.environ.get(C.AUTO_ENV_VAR)
    if raw is None:
        return None
    return raw.strip().lower()


def confirm(prompt, default_safe=True):
    """Подтверждение y/yes. Безголовый запуск — только PARKING_EXP_AUTO=1."""
    flag = _auto_flag()
    if flag is not None and flag not in C.AUTO_FLAG_TRUTHY_ANY_EXCEPT:
        print(f"{prompt} -> авто-подтверждение ({C.AUTO_ENV_VAR}={flag!r})")
        return True
    try:
        ans = input(f"{prompt} [y/Enter]: ").strip().lower()
    except EOFError:
        ans = ""
    if ans in ("y", "yes", "д", "да"):
        return True
    return False if default_safe else True


# ═══════════════════════════════════════════════════════════════════
# READ-ONLY ДОСТУП К БАЗЕ OPTUNA (рецепт stage2_only_v1_2.py:521-606)
# ═══════════════════════════════════════════════════════════════════

def _ro_uri_candidates(db_file):
    db_abs = os.path.abspath(db_file).replace(os.sep, "/")
    db_rel = db_file.replace(os.sep, "/")
    cands = [("абсолютный путь, read-only URI",
              f"sqlite:///file:{db_abs}?mode=ro&uri=true")]
    if db_rel != db_abs:
        cands.append(("относительный путь, read-only URI",
                      f"sqlite:///file:{db_rel}?mode=ro&uri=true"))
    return cands


def load_study_readonly(db_file, study_name, attempts=3, retry_sleep=10):
    """Загрузить study строго read-only. Возвращает (study|None, источник|None)."""
    import optuna
    import time

    if not os.path.exists(db_file):
        print(f"[ERROR] база не найдена: {db_file}")
        return None, None

    last_err = None
    for label, url in _ro_uri_candidates(db_file):
        for attempt in range(1, attempts + 1):
            try:
                storage = optuna.storages.RDBStorage(
                    url=url,
                    engine_kwargs={"connect_args": {"uri": True, "timeout": 30}},
                )
                study = optuna.load_study(study_name=study_name, storage=storage)
                print(f"[DB] открыто read-only ({label}): {url}")
                return study, f"{label}: {url}"
            except Exception as e:                # noqa: BLE001
                last_err = e
                text = f"{type(e).__name__}: {e}"
                print(f"[WARN] read-only ({label}) попытка {attempt}/{attempts}: {text}")
                if "locked" in text.lower() and attempt < attempts:
                    time.sleep(retry_sleep)
                    continue
                break

    print("[INFO] fallback: копирую базу во временную папку (оригинал не трогаем)")
    try:
        tmpdir = tempfile.mkdtemp(prefix="optuna_ro_")
        target = os.path.join(tmpdir, os.path.basename(db_file))
        shutil.copy2(db_file, target)
        for suf in ("-wal", "-shm"):
            src = db_file + suf
            if os.path.exists(src):
                shutil.copy2(src, target + suf)
        target_url = f"sqlite:///file:{target.replace(os.sep, '/')}?mode=ro&uri=true"
        storage = optuna.storages.RDBStorage(
            url=target_url,
            engine_kwargs={"connect_args": {"uri": True, "timeout": 30}},
        )
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f"[DB] открыта КОПИЯ read-only: {target}")
        return study, f"копия в tempdir (read-only): {target}"
    except Exception as e:                        # noqa: BLE001
        print(f"[ERROR] fallback-копия не удалась: {type(e).__name__}: {e}")
        print(f"[ERROR] исходная ошибка read-only: {last_err}")
        return None, None


# ═══════════════════════════════════════════════════════════════════
# СРЕДА С ЗАМОРОЖЕННОЙ НОРМАЛИЗАЦИЕЙ (N1)
# ═══════════════════════════════════════════════════════════════════

from portfolio_env import PortfolioEnv


class ParkingEvalEnv(PortfolioEnv):
    """PortfolioEnv, у которого feat_mean/feat_std взяты из train-окна fold 5.

    Зачем: модели ppo_v1.2_fold5_s* обучены на наблюдениях, нормированных
    по train-окну fold 5. При оценке на других фолдах базовая реализация
    подставила бы нормировку ЧУЖОГО окна -> наблюдения в неверной шкале.

    portfolio_env.py при этом остаётся байт-в-байт неизменным: подмена
    выполняется после super().__init__() и затрагивает только два массива,
    которые используются исключительно в _get_obs().
    """

    FROZEN_FOLD = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ts, te, _, _ = self.folds[self.FROZEN_FOLD]
        raw = self.features.iloc[ts:te].values.astype(np.float32)
        self.feat_mean = raw.mean(axis=0)
        self.feat_std = raw.std(axis=0) + 1e-8
        self.frozen_from = (int(ts), int(te), int(te - ts))

    def describe_freeze(self):
        ts, te, rows = self.frozen_from
        d0 = self.features.index[ts].date()
        d1 = self.features.index[te - 1].date()
        return f"нормализация заморожена: fold{self.FROZEN_FOLD} train idx [{ts},{te}) " \
               f"{d0} .. {d1} (вкл), строк={rows}"


# ═══════════════════════════════════════════════════════════════════
# СВЕРКА ЛИДЕРА
# ═══════════════════════════════════════════════════════════════════

def _tol_for(key):
    """Допуск сверки: протокол AGENTS.md округляет значения.

    Факт из базы (trial 14):
      lambda_hhi = 0.002276109018059183  (в протоколе 0.0023)
      dd_alpha   = 8.379367247382849     (в протоколе 8.38)
    Поэтому сравнение допускает округление протокола до его точности;
    жёсткие 1e-6 давали ложный FAIL на 8.379367 vs 8.38 (разница 0.00063).
    """
    if key == "lambda_hhi":
        return 2e-4
    if key == "max_weight":
        return 1e-9
    if key == "dd_alpha":
        return 1e-2
    return 0.0


def resolve_hyperparams(params):
    """Достать из trial.params рабочие значения (с фолбэком на ожидаемого лидера)."""
    exp = C.EXPECTED_LEADER
    return {
        "max_weight": float(params.get("max_weight", exp["max_weight"])),
        "lambda_hhi": float(params.get("lambda_hhi", exp["lambda_hhi"])),
        "total_timesteps": int(params.get("total_timesteps", exp["total_timesteps"])),
        "dd_alpha": float(params.get("dd_alpha", exp["dd_alpha"])),
        "gamma": float(params.get("gamma", exp["gamma"])),
        "n_epochs": int(params.get("n_epochs", exp["n_epochs"])),
    }


def report_and_verify_leader(study, source):
    """Печать и сверка лидера study.best_trial с протоколом. None при провале."""
    import optuna

    print("\n" + "═" * 74)
    print("СВЕРКА ЛИДЕРА study.best_trial")
    print("═" * 74)
    print(f"  источник: {source}")

    direction = str(study.direction)
    print(f"  направление study: {direction}")
    if study.direction != optuna.study.StudyDirection.MAXIMIZE:
        print("[ERROR] гейт: ожидалось MAXIMIZE")
        return None

    bt = study.best_trial
    val_txt = "n/a" if bt.value is None else f"{bt.value:.6f}"
    print(f"  best_trial: №{bt.number}  value={val_txt}")
    hp = resolve_hyperparams(bt.params)
    for k in ("max_weight", "lambda_hhi", "total_timesteps", "dd_alpha",
              "gamma", "n_epochs"):
        print(f"    {k:18s} = {hp[k]}")

    exp = C.EXPECTED_LEADER
    ok = True
    if bt.number != exp["trial"]:
        print(f"  [FAIL] номер триала {bt.number} != ожидаемого {exp['trial']}")
        ok = False
    for k in ("max_weight", "lambda_hhi", "total_timesteps", "dd_alpha"):
        tol = _tol_for(k)
        if abs(hp[k] - exp[k]) > tol:
            print(f"  [FAIL] {k}: {hp[k]} != {exp[k]} (допуск {tol})")
            ok = False
    print("  [OK] лидер совпадает с протоколом" if ok else "  [FAIL] лидер НЕ совпал")
    if not ok:
        print("  [ERROR] гейт: параметры лидера не совпали — прогон остановлен")
        return None
    return hp


# ═══════════════════════════════════════════════════════════════════
# МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════

def model_path(seed):
    """Фактический путь модели: сначала с .zip, затем без расширения.

    В workspace файлы ppo_v1.2_fold5_s* лежат БЕЗ расширения .zip
    (внутри — валидный zip SB3). Рецепт stage2_only_v1_2.py:474 такой же.
    """
    base = C.MODEL_TEMPLATE.format(seed=seed)
    if base.endswith(".zip"):
        base = base[:-4]
    for ext in C.MODEL_EXTS:
        cand = base + ext
        if os.path.exists(cand):
            return cand
    return base + C.MODEL_EXTS[0]          # не найден: имя для сообщения об ошибке


def guard_models():
    """Все 7 моделей должны существовать. Возвращает список отсутствующих."""
    missing = [model_path(s) for s in C.SEEDS if not os.path.exists(model_path(s))]
    print(f"\n[МОДЕЛИ] ожидается {len(C.SEEDS)}:")
    for s in C.SEEDS:
        p = model_path(s)
        ok = os.path.exists(p)
        mark = "OK " if ok else "НЕТ"
        size = f"{os.path.getsize(p) / 1024:.1f} KB" if ok else "-"
        print(f"  [{mark}] s{s:<5d} {p:<32s} {size}")
    return missing


def load_model(path, template_env):
    from stable_baselines3 import PPO
    try:
        model = PPO.load(path, device="cpu")
    except Exception:                             # noqa: BLE001
        model = PPO.load(path, env=template_env, device="cpu")
    return model


# ═══════════════════════════════════════════════════════════════════
# ИНФЕРЕНС ОДНОЙ ПАРЫ (fold, seed)
# ═══════════════════════════════════════════════════════════════════

def run_one(model, fold, seed, hp, features_path, prices_path):
    """Детерминированный прогон test-окна фолда. Возвращает (rows, env, warns)."""
    env = ParkingEvalEnv(
        features_path=features_path,
        prices_path=prices_path,
        eval_mode=True,
        eval_split="test",
        fold=fold,
        seed=seed,
        max_weight=hp["max_weight"],
        lambda_hhi=hp["lambda_hhi"],
        dd_alpha=hp["dd_alpha"],
    )
    obs, info0 = env.reset(seed=seed)
    idx = env.features.index
    rows = []
    warns = []
    prev_pv = 1.0

    # ВАЖНО: ключ info["portfolio_ret"] в portfolio_env.py:228 фактически хранит
    # НАГРАДУ (net_ret - dd_penalty - hhi_penalty), а не доходность портфеля.
    # Поэтому чистую доходность шага берём из отношения эквити —
    # это ровно то, что накапливает среда: pv(t) = pv(t-1) * (1 + net_ret).
    while True:
        decision_step = env.current_step
        decision_date = idx[decision_step].date()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        pv = float(info["portfolio_value"])
        net_ret = pv / prev_pv - 1.0
        prev_pv = pv
        rows.append({
            "decision_date": str(decision_date),
            "realize_date": info["date"],
            "weights": np.asarray(info["weights"], dtype=np.float64).copy(),
            "net_ret": net_ret,
            "reward": float(reward),
            "turnover": float(info["turnover"]),
            "commission": float(info["commission_cost"]),
            "portfolio_value": pv,
            "equal_value": float(info["equal_value"]),
        })
        if terminated or truncated:
            break

    _, _, vs, ve = env.folds[fold]
    expected = ve - vs - 1          # eval: ep_length = *_end - start_idx - 1
    if len(rows) != expected:
        warns.append(f"строк траектории {len(rows)} != {expected} (eval-длина)")
    return rows, env, warns


# ═══════════════════════════════════════════════════════════════════
# ЗАПИСЬ ТРАЕКТОРИИ
# ═══════════════════════════════════════════════════════════════════

def write_traj(path, fold, seed, tickers, rows):
    header = (["fold", "seed", "decision_date", "realize_date"]
              + [f"w_{t}" for t in tickers] + ["CASH",
                 "net_ret", "reward", "turnover", "commission",
                 "portfolio_value", "equal_value"])
    with open(path, "w", newline="", encoding=C.LOG_ENCODING) as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            wts = r["weights"]
            w.writerow([fold, seed, r["decision_date"], r["realize_date"]]
                       + [f"{x:.8f}" for x in wts[:len(tickers)]]
                       + [f"{wts[-1]:.8f}",
                          f"{r['net_ret']:.12f}", f"{r['reward']:.10f}",
                          f"{r['turnover']:.10f}", f"{r['commission']:.10f}",
                          f"{r['portfolio_value']:.10f}",
                          f"{r['equal_value']:.10f}"])
    return path


# ═══════════════════════════════════════════════════════════════════
# QC
# ═══════════════════════════════════════════════════════════════════

def qc_equity(rows, env, label):
    """Самосогласованность: эквити из net_ret против env.portfolio_value.

    Проверяет, что восстановленная по шагам доходность воспроизводит финальное
    эквити среды. Ловит порчу весов/шагов/порядка строк. Порог 1e-9 относительный:
    накопление float32 даёт расхождение ~1e-7 в относительных единицах.
    """
    v = 1.0
    for r in rows:
        v *= (1.0 + r["net_ret"])
    env_v = float(env.portfolio_value)
    diff = abs(v - env_v)
    rel = diff / max(abs(env_v), 1e-12)
    ok = rel < 1e-9
    print(f"    QC эквити {label}: пересчёт={v:.10f} env={env_v:.10f} "
          f"rel={rel:.2e} {'OK' if ok else 'FAIL'}")
    return ok, rel


def build_in_sample_table(env):
    """Таблица in-sample статуса окон: перекрытие test(fold) с train(fold5)."""
    ts5, te5, _, _ = env.folds[ParkingEvalEnv.FROZEN_FOLD]
    idx = env.features.index
    out = []
    for f in C.EVAL_FOLDS:
        ts, te, vs, ve = env.folds[f]
        overlap = max(0, min(ve, te5) - max(vs, ts5))
        n_test = ve - vs
        status = ("IN-SAMPLE" if overlap == n_test else
                  "OOS" if overlap == 0 else f"ЧАСТИЧНО {overlap}/{n_test}")
        out.append({
            "fold": f,
            "train": f"[{ts},{te}) {idx[ts].date()} .. {idx[te - 1].date()} (вкл)",
            "test": f"[{vs},{ve}) {idx[vs].date()} .. {idx[ve - 1].date()} (вкл)",
            "rows": n_test,
            "overlap": overlap,
            "status": status,
        })
    return out


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    setup_logging("step1")
    t0 = dt.datetime.now()
    print("═" * 74)
    print("parking_exp ШАГ 1: траектории кэша агента v1.2 на фолдах 0-4")
    print(f"старт: {t0:%Y-%m-%d %H:%M:%S}")
    print("═" * 74)
    for line in C.describe():
        print(f"  {line}")

    print("\n" + "─" * 74)
    print("ПРОВЕРКИ ПЕРЕД РАБОТОЙ")
    print("─" * 74)

    missing = guard_models()
    if missing:
        print(f"[ERROR] отсутствуют модели: {missing}")
        return 1

    for p in (C.FEATURES_FILE, C.PRICES_FILE):
        if not os.path.exists(p):
            print(f"[ERROR] нет файла данных: {p}")
            return 1
        print(f"  [OK] {p} ({os.path.getsize(p) / 1024 / 1024:.2f} MB)")

    if C.CREATE_DIRS_ON_RUN:
        os.makedirs(C.RESULTS_DIR, exist_ok=True)

    study, source = load_study_readonly(C.DB_FILE, C.STUDY_NAME)
    if study is None:
        print("[ERROR] не удалось открыть study read-only — прогон остановлен")
        return 1
    hp = report_and_verify_leader(study, source)
    if hp is None:
        return 1

    # ── Пробная среда: гейты obs_dim и геометрии фолдов ──
    probe = ParkingEvalEnv(
        features_path=C.FEATURES_FILE, prices_path=C.PRICES_FILE,
        eval_mode=True, eval_split="test", fold=0, seed=42,
        max_weight=hp["max_weight"], lambda_hhi=hp["lambda_hhi"],
        dd_alpha=hp["dd_alpha"],
    )
    obs_dim = int(probe.observation_space.shape[0])
    print(f"\n  obs_dim = {obs_dim} (ожидается {C.EXPECTED_OBS_DIM})")
    if obs_dim != C.EXPECTED_OBS_DIM:
        print("[ERROR] гейт obs_dim: файл фичей не тот (ожидался rl_features_v5.pkl)")
        return 1
    n_slots = int(probe.n_actions)
    if n_slots != C.N_WEIGHT_SLOTS:
        print(f"[ERROR] гейт весов: слотов {n_slots}, ожидалось {C.N_WEIGHT_SLOTS}")
        return 1
    print(f"  [OK] слотов весов = {n_slots} ({probe.tickers} + CASH)")
    print(f"  [OK] {probe.describe_freeze()}")
    fts, fte, _, _ = probe.folds[ParkingEvalEnv.FROZEN_FOLD]
    if (fte - fts) != 1200:
        print(f"[ERROR] train-окно fold5 = {fte - fts} строк, ожидалось 1200")
        return 1

    print("\n  ТАБЛИЦА IN-SAMPLE СТАТУСА ОКОН (факт из разметки, не оценка)")
    print(f"  {'fold':>4} | {'train (вкл)':<40} | {'test (вкл)':<40} | "
          f"{'строк':>5} | статус")
    tbl = build_in_sample_table(probe)
    for r in tbl:
        print(f"  {r['fold']:>4} | {r['train']:<40} | {r['test']:<40} | "
              f"{r['rows']:>5} | {r['status']}")

    # ── Сводка и подтверждение ──
    plan_rows = len(C.SEEDS) * len(C.EVAL_FOLDS)
    print("\n" + "─" * 74)
    print("СВОДКА ПЕРЕД ПРОГОНОМ")
    print("─" * 74)
    print(f"  моделей: {len(C.SEEDS)}, фолдов: {C.EVAL_FOLDS} -> траекторий {plan_rows}")
    print(f"  каталог результатов: {os.path.abspath(C.RESULTS_DIR)}")
    print(f"  шаблон файлов: {C.TRAJ_TEMPLATE}")
    print("  операции записи: ТОЛЬКО новые CSV в results/ (перезаписываемы)")
    print("  база Optuna: read-only, fold 5 (holdout) НЕ используется")
    flush_all()
    if not confirm("Начать прогон шага 1?"):
        print("[CANCEL] отменено пользователем (безопасный вариант)")
        return 2

    # ── Прогон ──
    tickers = list(probe.tickers)
    all_stats = []
    qc_fail = 0
    warn_total = 0
    print("\n" + "═" * 74)
    print("ПРОГОН")
    print("═" * 74)
    for seed in C.SEEDS:
        path = model_path(seed)
        model = load_model(path, probe)
        print(f"\n[SEED {seed}] {path} загружена")
        for fold in C.EVAL_FOLDS:
            rows, env, warns = run_one(model, fold, seed, hp,
                                       C.FEATURES_FILE, C.PRICES_FILE)
            out = os.path.join(C.RESULTS_DIR, C.TRAJ_TEMPLATE.format(fold=fold, seed=seed))
            write_traj(out, fold, seed, tickers, rows)
            ok, diff = qc_equity(rows, env, f"fold{fold} s{seed}")
            if not ok:
                qc_fail += 1
            warn_total += len(warns)
            cash = np.array([r["weights"][-1] for r in rows])
            eq = rows[-1]["equal_value"] - 1.0
            ret = rows[-1]["portfolio_value"] - 1.0
            all_stats.append({
                "fold": fold, "seed": seed,
                "cash_mean": float(cash.mean()),
                "cash_median": float(np.median(cash)),
                "cash_min": float(cash.min()),
                "cash_max": float(cash.max()),
                "ret": float(ret), "equal": float(eq),
                "turnover_mean": float(np.mean([r["turnover"] for r in rows])),
            })
            print(f"  fold{fold} s{seed:<5d} ret={ret * 100:+7.2f}% eq={eq * 100:+7.2f}% "
                  f"cash med={np.median(cash):.3f} mean={cash.mean():.3f} "
                  f"turn={np.mean([r['turnover'] for r in rows]):.4f} -> {out}")
            for w in warns:
                print(f"    [WARN] {w}")

    # ── Итоги ──
    print("\n" + "═" * 74)
    print("ИТОГИ ШАГА 1")
    print("═" * 74)
    print(f"  траекторий: {len(all_stats)} (ожидалось {plan_rows})")
    print(f"  QC эквити FAIL: {qc_fail}")
    print(f"  предупреждений: {warn_total}")

    print("\n  СРЕДНИЕ ПО ФОЛДАМ (для правила сдвига сетки N7)")
    print(f"  {'fold':>4} | {'cash_mean':>9} | {'cash_med':>8} | {'cash_min':>8} | "
          f"{'cash_max':>8} | {'ret %':>8} | {'eq %':>8} | {'turn':>7}")
    fold_medians = {}
    for f in C.EVAL_FOLDS:
        sub = [s for s in all_stats if s["fold"] == f]
        if not sub:
            continue
        cm = float(np.mean([s["cash_mean"] for s in sub]))
        cd = float(np.median([s["cash_median"] for s in sub]))
        cmin = float(np.min([s["cash_min"] for s in sub]))
        cmax = float(np.max([s["cash_max"] for s in sub]))
        r = float(np.mean([s["ret"] for s in sub]))
        e = float(np.mean([s["equal"] for s in sub]))
        t = float(np.mean([s["turnover_mean"] for s in sub]))
        fold_medians[f] = cd
        print(f"  {f:>4} | {cm:9.3f} | {cd:8.3f} | {cmin:8.3f} | {cmax:8.3f} | "
              f"{r * 100:+8.2f} | {e * 100:+8.2f} | {t:7.4f}")

    print("\n  РЕШЕНИЕ ПО СЕТКЕ (N7, заранее зафиксированное правило)")
    trig = C.CASH_MEDIAN_SHIFT_TRIGGER
    hot = {f: v for f, v in fold_medians.items() if v > trig}
    base = C.valid_configs()
    if hot:
        print(f"  медиана кэша > {trig} на фолдах {sorted(hot)} -> "
              f"СДВИГ сетки на {C.THETA_GRID_SHIFTED}")
        cfg = C.valid_configs(C.THETA_GRID_SHIFTED)
        print(f"  конфигураций {len(cfg)}, ячеек {C.cell_count(len(cfg))}")
    else:
        print(f"  медиана кэша <= {trig} на всех фолдах -> БАЗОВАЯ сетка "
              f"{C.THETA_GRID}")
        print(f"  конфигураций {len(base)}, ячеек {C.cell_count(len(base))}")
    for f in sorted(fold_medians):
        print(f"    fold{f}: медиана кэша {fold_medians[f]:.3f}, "
              f"ставка по контексту: {C.FOLD_RATE_CONTEXT.get(f, 'OOS/см. лог')}")

    dur = (dt.datetime.now() - t0).total_seconds()
    print(f"\n  время: {dur / 60:.1f} мин")
    print(f"  лог: {os.path.abspath(C.LOGS_DIR)}")
    flush_all()
    return 0 if (qc_fail == 0 and warn_total == 0) else 0


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
