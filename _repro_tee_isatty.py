"""Проверка фикса: реальный класс _Tee из train_ppo.py + colorlog/optuna-путь.

Класс _Tee извлекается из train_ppo.py через ast, а не копируется,
чтобы тест проверял именно тот код, который запускается.
"""
import ast
import datetime
import io
import logging
import sys
import tempfile

import colorlog

SRC = "train_ppo.py"


def load_tee_class(path=SRC):
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "_Tee"
    )
    mod = ast.Module(body=[node], type_ignores=[])
    ns = {"datetime": datetime}
    exec(compile(mod, path, "exec"), ns)  # noqa: S102
    return ns["_Tee"]


def make_record():
    return logging.LogRecord(
        "optuna", logging.INFO, __file__, 1, "Trial 0 finished with value: -0.0045", None, None
    )


if __name__ == "__main__":
    Tee = load_tee_class()
    real_err = sys.__stderr__
    tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False, encoding="utf-8")

    tee = Tee(real_err, tmp)
    ok = True

    # 1) isatty присутствует и не падает
    try:
        val = tee.isatty()
        print(f"[1] Tee.isatty() -> {val!r}  OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[1] Tee.isatty() FAIL -> {type(e).__name__}: {e}")

    # 2) воспроизводим падение из трейсбека: colorlog-форматтер с stream=tee
    fmt = colorlog.TTYColoredFormatter("%(message)s", stream=tee)
    try:
        out = fmt.format(make_record())
        print(f"[2] TTYColoredFormatter.format -> {out!r}  OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[2] TTYColoredFormatter.format FAIL -> {type(e).__name__}: {e}")

    # 3) logging.StreamHandler(tee).emit — как _TqdmLoggingHandler/flush
    h = logging.StreamHandler(tee)
    h.setFormatter(colorlog.TTYColoredFormatter("%(message)s", stream=tee))
    try:
        h.emit(make_record())
        tee.flush()
        print("[3] StreamHandler.emit + flush OK")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[3] StreamHandler.emit FAIL -> {type(e).__name__}: {e}")

    # 4) запись в файл реально происходит
    tee.write("line for file\n")
    tee.flush()
    body = open(tmp.name, encoding="utf-8").read()
    has = "line for file" in body
    ok = ok and has
    print(f"[4] file содержимое содержит запись -> {has}  {'OK' if has else 'FAIL'}")
    print(f"    python {sys.version.split()[0]}")
    print("RESULT:", "PASS" if ok else "FAIL")
