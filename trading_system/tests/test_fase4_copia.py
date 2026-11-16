"""
Pruebas de la Fase 4 — grupo D: copia de seguridad y histórico propio en la
validación (no cambia qué señales salen).

  - `backup` / `restore`: gzip JSON con todas las tablas de estado (sin ohlcv); ida
    y vuelta exacta en SQLite; vista previa sin escrituras; solo inserta lo que
    falta (conflictos y claves únicas se informan y no se tocan); idempotente; ids
    conservados y autoincremento correcto; ficheros inválidos; escritura atómica.
  - Copia mensual por Telegram (tarea nocturna): día 1 (reintento hasta el día 7),
    documento con límite total de 10 s y 50 MB, estado en kv, línea en el resumen;
    nunca tumba nada; va antes del resumen y no impide el ciclo de las 03:01.
  - Histórico propio (data/db_history.py): velas source='gateio' de la tabla ohlcv
    ANTERIORES a la primera que sirve Gate.io, solo en validate/check-live/protocolo;
    sin velas antiguas en la BD devuelve los mismos objetos; el vivo no cambia.

    cd trading_system && python -m pytest tests/test_fase4_copia.py -q

Sin red: requests.get/post y Session.request fallan; Telegram es un doble y la BD
es SQLite temporal.
"""
import datetime as _dt
import gzip
import json
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures y utilidades
# ---------------------------------------------------------------------------

def _use_db(monkeypatch, path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    import data.database as db
    if db._engine is not None:
        db._engine.dispose()
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    return db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    db = _use_db(monkeypatch, tmp_path / "test.db")
    db.init_db()
    yield db
    if db._engine is not None:
        db._engine.dispose()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.fixture(autouse=True)
def _clean_schedule():
    import schedule
    schedule.clear()
    yield
    schedule.clear()


def _seed_state(db, n_sig=12, n_shadow=40):
    """Filas de todas las tablas de estado (con acentos y emojis en los textos)."""
    with db.get_engine().begin() as c:
        for i in range(n_sig):
            c.execute(text(
                "INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, stop_loss, "
                "take_profit, position_size, atr, resultado_real, pnl_r, setup_ts, mode, notified, alert_text, "
                "price_at_emit, portfolio_flag) VALUES (:s, :ts, 'breakout', :d, 0.61, :e, :sl, :tp, 250.5, 1.25, "
                ":res, :r, :sts, 'live', 1, :txt, :pe, 'ok')"),
                {"s": "BTCUSDT" if i % 2 else "XAUUSD", "ts": 1_760_000_000 + i * 3600,
                 "d": "long" if i % 3 else "short", "e": 100.0 + i / 3, "sl": 97.123456789, "tp": 104.5,
                 "res": None if i % 4 == 0 else "win", "r": None if i % 4 == 0 else 1.5 + i / 7,
                 "sts": 1_760_000_000 + i * 3600 - 14400,
                 "txt": f"🟢 <b>SEÑAL LONG</b> — oro ñandú «{i}»\nEntrada: 4,050.25", "pe": 100.0 + i / 3})
        for i in range(n_shadow):
            c.execute(text(
                "INSERT INTO shadow_setups (symbol, setup_ts, direction, setup_type, raw_score, approved, motivo, "
                "entry, sl, tp, atr, created_ts, features) VALUES (:s, :ts, 'long', 'breakout', 0.5, 1, "
                "'below_threshold', 100, 97, 104.5, 2, :c, :f)"),
                {"s": "ETHUSDT", "ts": 1_760_000_000 + i * 14400, "c": 1_760_000_100 + i * 14400,
                 "f": json.dumps({"sma200_with": True, "tsmom": 0.1 * i})})
        c.execute(text("INSERT INTO data_quality (symbol, ts, check_name, detalle, grave, created_ts) "
                       "VALUES ('BTCUSDT', 1760000000, 'gap', 'hueco de 2 velas', 1, 1760000100)"))
        c.execute(text("INSERT INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume, source) "
                       "VALUES ('BTCUSDT', '4h', 1652169600, 1, 2, 0.5, 1.5, 10, 'gateio')"))
    db.save_fill(2, "fill", price=101.5, note="entré tarde", update_id=555)
    db.save_fill(3, "skip", note="sin margen", update_id=556)
    db.kv_set("telegram:offset", "778899")
    db.set_last_train("BTCUSDT", "2026-11-01T03:00:00+00:00")


def _dump(db, exclude=("ohlcv",)):
    from sqlalchemy import inspect
    eng = db.get_engine()
    out = {}
    with eng.connect() as c:
        for t in inspect(eng).get_table_names():
            if t in exclude or t.startswith("sqlite_"):
                continue
            out[t] = sorted(tuple(r) for r in c.execute(text(f'SELECT * FROM "{t}"')).fetchall())
    return out


# ---------------------------------------------------------------------------
# 1. backup / restore
# ---------------------------------------------------------------------------

def test_backup_formato_tablas_y_sin_ohlcv(temp_db, tmp_path, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    with temp_db.get_engine().begin() as c:     # tabla extra de estado (como la de la Fase 0)
        c.execute(text("CREATE TABLE signals_backup_fase0 AS SELECT * FROM signals"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRETO-123")
    info = B.export_backup(tmp_path / "copia.json.gz")
    raw = gzip.open(info["path"], "rt", encoding="utf-8").read()
    data = json.loads(raw)
    assert data["format"] == B.BACKUP_FORMAT and data["version"] == B.BACKUP_VERSION
    assert data["dialect"] == "sqlite" and data["excluded"] == ["ohlcv"]
    assert "ohlcv" not in data["tables"]
    for t in ("signals", "fills", "kv", "shadow_setups", "data_quality", "retrain_log", "signals_backup_fase0"):
        assert t in data["tables"], t
    assert info["counts"]["signals"] == 12 and info["counts"]["shadow_setups"] == 40
    assert info["counts"]["fills"] == 2 and info["counts"]["signals_backup_fase0"] == 12
    assert data["counts"] == info["counts"]
    # Sin datos de conexión ni secretos: ni la URL de la BD ni tokens.
    assert "sqlite:///" not in raw and "SECRETO" not in raw and str(tmp_path) not in raw
    assert "ñandú" in raw and "🟢" in raw
    assert not list(tmp_path.glob("*.tmp"))


def test_restore_vista_previa_no_escribe_y_apply_ida_y_vuelta_exacta(temp_db, tmp_path, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    before = _dump(temp_db)
    f = B.export_backup(tmp_path / "c.json.gz")["path"]

    db2 = _use_db(monkeypatch, tmp_path / "nueva.db")
    db2.init_db()
    empty = _dump(db2)
    rep = B.restore_backup(f)                          # vista previa
    assert rep["applied"] is False
    assert _dump(db2) == empty                         # nada escrito
    plans = {p["table"]: p for p in rep["plans"]}
    assert plans["signals"]["new"] == 12 and plans["shadow_setups"]["new"] == 40
    assert "VISTA PREVIA" in B.format_restore(rep)

    rep = B.restore_backup(f, apply=True)
    assert rep["applied"] and rep["inserted"]["signals"] == 12
    after = _dump(db2)
    for t in ("signals", "fills", "kv", "shadow_setups", "data_quality", "retrain_log"):
        assert after[t] == before[t], t
    # Idempotente: una segunda restauración no inserta nada.
    rep2 = B.restore_backup(f, apply=True)
    assert sum(rep2["inserted"].values()) == 0
    assert sum(p["same"] for p in rep2["plans"]) == sum(p["rows"] for p in rep2["plans"] if not p["skipped"])
    # Los id se conservan y el autoincremento sigue después del mayor.
    sig = {"symbol": "BTCUSDT", "ts": "2026-11-16T12:00:00Z", "setup_type": "breakout", "direction": "long",
           "ml_score": 0.6, "entry_price": 1.0, "stop_loss": 0.9, "take_profit": 1.2, "position_size": 1.0,
           "atr": 0.1}
    assert db2.save_signal(sig) == 13
    assert db2.save_fill(13, "fill", price=1.0, update_id=999) is True
    assert db2.save_fill(2, "fill", price=1.0, update_id=555) is False     # update_id restaurado
    assert db2.kv_get("telegram:offset") == "778899"


def test_restore_crea_tablas_en_bd_sin_esquema(temp_db, tmp_path, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    f = B.export_backup(tmp_path / "c.json.gz")["path"]
    db2 = _use_db(monkeypatch, tmp_path / "vacia.db")   # sin init_db: sin tablas
    rep = B.restore_backup(f)
    assert all(p["creates_table"] for p in rep["plans"] if not p["skipped"])
    rep = B.restore_backup(f, apply=True)
    assert rep["inserted"]["signals"] == 12
    assert db2.load_signal_by_id(5)["entry_price"] == pytest.approx(100.0 + 4 / 3)


def test_restore_conflictos_y_claves_unicas_no_se_tocan(temp_db, tmp_path, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    f = B.export_backup(tmp_path / "c.json.gz")["path"]
    db2 = _use_db(monkeypatch, tmp_path / "otra.db")
    db2.init_db()
    # BD recreada: el worker ya guardó la señal 1 (otra), un setup sombra que también
    # está en la copia (con otro id) y su propio offset de Telegram.
    sig = {"symbol": "ETHUSDT", "ts": "2026-11-20T00:00:00Z", "setup_type": "pullback", "direction": "short",
           "ml_score": 0.7, "entry_price": 3000.0, "stop_loss": 3100.0, "take_profit": 2800.0,
           "position_size": 10.0, "atr": 60.0}
    assert db2.save_signal(sig) == 1
    with db2.get_engine().begin() as c:
        c.execute(text("INSERT INTO shadow_setups (id, symbol, setup_ts, direction, motivo, created_ts) "
                       "VALUES (900, 'ETHUSDT', 1760000000, 'long', 'not_fresh', 1)"))
    db2.kv_set("telegram:offset", "999999")
    rep = B.restore_backup(f, apply=True)
    plans = {p["table"]: p for p in rep["plans"]}
    assert plans["signals"]["conflict"] == 1 and plans["signals"]["conflict_ids"] == [1]
    assert rep["inserted"]["signals"] == 11
    assert plans["shadow_setups"]["unique"] == 1 and rep["inserted"]["shadow_setups"] == 39
    assert plans["kv"]["conflict"] == 1
    txt = B.format_restore(rep)
    assert "EN CONFLICTO" in txt and "clave única" in txt
    # La BD manda: nada se sobrescribe.
    assert db2.load_signal_by_id(1)["symbol"] == "ETHUSDT"
    assert db2.kv_get("telegram:offset") == "999999"
    with db2.get_engine().connect() as c:
        assert c.execute(text("SELECT motivo FROM shadow_setups WHERE id = 900")).scalar() == "not_fresh"


def test_restore_copia_antigua_o_con_columnas_desconocidas(temp_db, tmp_path):
    from data import backup as B
    _seed_state(temp_db, n_sig=3, n_shadow=0)
    info = B.export_backup(tmp_path / "c.json.gz")
    data = json.loads(gzip.open(info["path"], "rt", encoding="utf-8").read())
    sig = data["tables"]["signals"]
    i = sig["columns"].index("price_at_emit")
    sig["columns"].pop(i)                       # copia de antes de la columna price_at_emit
    for r in sig["rows"]:
        r.pop(i)
    sig["columns"].append("columna_futura")     # y una columna que esta versión no conoce
    for r in sig["rows"]:
        r.append("x")
    data["tables"]["tabla_rara"] = {"columns": ["a"], "rows": [[1]]}
    with temp_db.get_engine().begin() as c:
        c.execute(text("DELETE FROM signals"))
    p = tmp_path / "vieja.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    rep = B.restore_backup(p, apply=True)
    plans = {q["table"]: q for q in rep["plans"]}
    assert rep["inserted"]["signals"] == 3
    assert plans["signals"]["dropped_cols"] == ["columna_futura"]
    assert plans["tabla_rara"]["skipped"]
    assert temp_db.load_signal_by_id(1)["price_at_emit"] is None
    assert "columna_futura" in B.format_restore(rep) and "tabla_rara" in B.format_restore(rep)


def test_restore_postgresql_ajusta_secuencias(temp_db, tmp_path):
    """Sin servidor PostgreSQL aquí: se comprueba con un motor doble que, con dialecto
    postgresql, tras insertar con id explícitos se ajusta la secuencia de cada tabla
    SERIAL (y no en kv/retrain_log), todo en la misma transacción."""
    from data import backup as B
    _seed_state(temp_db, n_sig=3, n_shadow=2)
    plans = B.plan_restore(B.read_backup(B.export_backup(tmp_path / "c.json.gz")["path"]))
    stmts = []

    class _Res:
        rowcount = 1

    class _Conn:
        def execute(self, sql, params=None):
            stmts.append((str(sql), params))
            return _Res()

    class _Begin:
        def __enter__(self):
            stmts.append(("BEGIN", None))
            return _Conn()

        def __exit__(self, *a):
            stmts.append(("COMMIT", None))

    class _PG:
        dialect = types.SimpleNamespace(name="postgresql")

        def begin(self):
            return _Begin()
    # La BD temporal ya tiene las filas: se fuerza el plan como "todo nuevo".
    for p in plans:
        if p["table"] == "signals":
            p["to_insert"] = [dict(zip(p["insert_cols"], [None] * len(p["insert_cols"])), id=i) for i in (1, 2, 3)]
        if p["table"] == "kv":
            p["to_insert"] = [{"name": "a", "value": "1", "updated_ts": 1}]
    B.apply_restore(plans, engine=_PG())
    sqls = [s for s, _ in stmts]
    assert sqls[0] == "BEGIN" and sqls[-1] == "COMMIT" and sqls.count("BEGIN") == 1
    setvals = [s for s in sqls if "setval" in s]
    assert any("pg_get_serial_sequence('signals', 'id')" in s and "MAX(id) FROM \"signals\"" in s for s in setvals)
    assert not any("'kv'" in s or "'retrain_log'" in s for s in setvals)
    ins = next(s for s in sqls if s.startswith('INSERT INTO "signals"'))
    assert ins.endswith("ON CONFLICT DO NOTHING")
    assert sqls.index(ins) < sqls.index(next(s for s in setvals if "'signals'" in s))


@pytest.mark.parametrize("content, msg", [
    (b"no soy gzip", "no es un JSON gzip"),
    (gzip.compress(b'{"format": "otra_cosa", "tables": {}}'), "no es una copia"),
    (gzip.compress(b'{"format": "trading_system_backup", "version": 99, "tables": {}}'), "versi"),
    (gzip.compress(b'{"format": "trading_system_backup", "version": 1, "tables": '
                   b'{"kv": {"columns": ["name", "value"], "rows": [["a"]]}}}'), "columnas"),
])
def test_ficheros_invalidos(temp_db, tmp_path, content, msg):
    from data import backup as B
    p = tmp_path / "mala.json.gz"
    p.write_bytes(content)
    with pytest.raises(B.BackupError, match=msg):
        B.restore_backup(p, apply=True)
    with pytest.raises(B.BackupError, match="no existe"):
        B.read_backup(tmp_path / "no_existe.json.gz")


def test_escritura_atomica_si_falla_no_queda_fichero(temp_db, tmp_path, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    calls = {"n": 0}
    orig = B._jsonable

    def _falla(v):
        calls["n"] += 1
        if calls["n"] > 50:
            raise RuntimeError("disco lleno")
        return orig(v)
    monkeypatch.setattr(B, "_jsonable", _falla)
    with pytest.raises(RuntimeError):
        B.export_backup(tmp_path / "c.json.gz")
    assert list(tmp_path.glob("c.json.gz*")) == []


def test_exporta_por_bloques_con_memoria_acotada(temp_db, tmp_path):
    import tracemalloc
    from data import backup as B
    rows = [{"s": "BTCUSDT", "ts": i, "c": i, "f": json.dumps({"tsmom": i / 7, "atr_rank": 0.5, "dvol": 55.5})}
            for i in range(20_000)]
    with temp_db.get_engine().begin() as c:
        c.execute(text("INSERT INTO shadow_setups (symbol, setup_ts, direction, motivo, created_ts, features) "
                       "VALUES (:s, :ts, 'long', 'below_threshold', :c, :f)"), rows)
    del rows
    tracemalloc.start()
    info = B.export_backup(tmp_path / "grande.json.gz")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert info["counts"]["shadow_setups"] == 20_000
    # Por bloques ~1,6 MB; cargando la tabla entera serían ~13 MB (medido).
    assert peak < 6e6, peak


def test_cli_backup_y_restore(temp_db, tmp_path, monkeypatch, capsys):
    import main
    _seed_state(temp_db, n_sig=4, n_shadow=2)
    out = tmp_path / "cli.json.gz"
    monkeypatch.setattr(sys, "argv", ["main.py", "backup", "--out", str(out)])
    main.main()
    txt = capsys.readouterr().out
    assert out.exists() and "Copia guardada" in txt and "signals 4" in txt and "ohlcv no se copia" in txt
    _use_db(monkeypatch, tmp_path / "cli2.db").init_db()
    monkeypatch.setattr(sys, "argv", ["main.py", "restore", "--file", str(out)])
    main.main()
    assert "VISTA PREVIA" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["main.py", "restore", "--file", str(out), "--apply"])
    main.main()
    assert "Restaurado: " in capsys.readouterr().out
    import data.database as db
    assert len(db.load_signals(limit=None)) == 4
    bad = tmp_path / "mala.json.gz"
    bad.write_bytes(b"xx")
    monkeypatch.setattr(sys, "argv", ["main.py", "restore", "--file", str(bad)])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 1 and "ERROR" in capsys.readouterr().out


def test_cli_backup_send_usa_send_document(temp_db, tmp_path, monkeypatch, capsys):
    import main
    import notifications.telegram as tg
    sent = []
    monkeypatch.setattr(tg, "send_document", lambda p, caption="", filename=None, **k:
                        sent.append((Path(p).exists(), caption, filename)) or (True, "ok"))
    out = tmp_path / "manual.json.gz"
    monkeypatch.setattr(sys, "argv", ["main.py", "backup", "--out", str(out), "--send"])
    main.main()
    assert sent and sent[0][0] and sent[0][2] == "manual.json.gz"
    assert "mensual" not in sent[0][1] and "restore --file manual.json.gz" in sent[0][1]
    assert "Telegram: enviada" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. Copia mensual por Telegram
# ---------------------------------------------------------------------------

def _utc(s):
    return pd.Timestamp(s, tz="UTC").to_pydatetime()


def test_cuando_toca_la_copia_mensual():
    from data.backup import monthly_backup_due as due
    assert due(_utc("2026-12-01 03:00"), state={})
    assert not due(_utc("2026-12-01 03:00"), state={"ok": True, "month": "2026-12"})
    assert due(_utc("2026-12-01 03:00"), state={"ok": True, "month": "2026-11"})
    assert due(_utc("2026-12-02 03:00"), state={"ok": False, "month": "2026-12"})      # reintento
    assert due(_utc("2026-12-07 03:00"), state={})
    assert not due(_utc("2026-12-08 03:00"), state={})
    assert not due(_utc("2026-12-15 03:00"), state={"ok": False, "month": "2026-12"})
    assert not due(_utc("2026-12-02 03:00"), state={"ok": True, "month": "2026-12"})


def test_copia_mensual_envia_documento_y_guarda_estado(temp_db, monkeypatch):
    from data import backup as B
    _seed_state(temp_db)
    calls = []

    def _send(path, caption, filename=None):
        p = Path(path)
        data = json.loads(gzip.open(p, "rt", encoding="utf-8").read())
        calls.append({"exists": p.exists(), "size": p.stat().st_size, "caption": caption,
                      "filename": filename, "signals": len(data["tables"]["signals"]["rows"]), "path": p})
        return True, "ok"
    st = B.run_monthly_backup(now=_utc("2026-12-01 03:00:05"), send_document=_send)
    assert st["ok"] is True and len(calls) == 1
    c = calls[0]
    assert c["exists"] and c["signals"] == 12 and c["filename"] == "trading_backup_2026-12-01.json.gz"
    assert "12 señales" in c["caption"] and "restore --file trading_backup_2026-12-01.json.gz" in c["caption"]
    assert len(c["caption"]) <= 1024
    assert not c["path"].exists() and not c["path"].parent.exists()      # temporal borrado
    saved = json.loads(temp_db.kv_get(B.KV_MONTHLY))
    assert saved["ok"] and saved["month"] == "2026-12" and saved["bytes"] == c["size"]
    # Ya enviada este mes: ni el día 1 (otra vez) ni el 2 se repite.
    assert B.run_monthly_backup(now=_utc("2026-12-01 03:00:30"), send_document=_send) is None
    assert B.run_monthly_backup(now=_utc("2026-12-02 03:00"), send_document=_send) is None
    assert len(calls) == 1
    line = B.summary_line(int(_utc("2026-12-01 03:01").timestamp()))
    assert line.startswith("💾 Copia de seguridad mensual enviada")
    assert B.summary_line(int(_utc("2026-12-02 03:30").timestamp())) is None      # > 24 h


def test_copia_mensual_fallos_nunca_lanzan_y_se_reintenta(temp_db, monkeypatch):
    from data import backup as B
    _seed_state(temp_db, n_sig=2, n_shadow=0)
    st = B.run_monthly_backup(now=_utc("2026-12-01 03:00"), send_document=lambda *a, **k: (False, "Telegram respondió 413"))
    assert st["ok"] is False and "413" in st["error"]
    line = B.summary_line(int(_utc("2026-12-01 03:01").timestamp()))
    assert "NO enviada" in line and "413" in line and "backup --send" in line

    def _boom(*a, **k):
        raise ConnectionError("red caída")
    st = B.run_monthly_backup(now=_utc("2026-12-02 03:00"), send_document=_boom)      # reintento del día 2
    assert st["ok"] is False and "ConnectionError" in st["error"]
    # Export que falla (BD rota) y kv que falla: tampoco lanza.
    monkeypatch.setattr(B, "export_backup", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BD caída")))
    import data.database as db
    monkeypatch.setattr(db, "kv_set", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kv caído")))
    st = B.run_monthly_backup(now=_utc("2026-12-03 03:00"), send_document=lambda *a, **k: (True, "ok"))
    assert st["ok"] is False and "BD caída" in st["error"]
    # Que no toque tampoco lanza aunque la BD falle al leer el estado.
    monkeypatch.setattr(db, "kv_get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert B.run_monthly_backup(now=_utc("2026-12-20 03:00")) is None
    assert B.summary_line() is None


def test_resumen_nocturno_lleva_la_linea_arriba(temp_db, monkeypatch):
    from data import backup as B
    from monitoring.daily_summary import build_daily_summary
    B.run_monthly_backup(now=_utc("2026-12-01 03:00"), send_document=lambda *a, **k: (True, "ok"))
    txt = build_daily_summary(now=pd.Timestamp("2026-12-01 03:00:20", tz="UTC"), symbols=[])
    lines = txt.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("💾"))
    assert i <= 4, lines[:6]
    txt2 = build_daily_summary(now=pd.Timestamp("2026-12-05 03:00", tz="UTC"), symbols=[])
    assert "💾" not in txt2


class _Resp:
    def __init__(self, code=200, body='{"ok":true}'):
        self.status_code, self.text = code, body


def test_send_document_limites_y_errores(tmp_path, monkeypatch):
    import requests
    import notifications.telegram as tg
    f = tmp_path / "c.json.gz"
    f.write_bytes(gzip.compress(b"{}"))
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "")
    assert tg.send_document(f, "x")[0] is False                      # sin Telegram
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TOK-SECRETO")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    big = tmp_path / "big.gz"
    with open(big, "wb") as fh:                                       # fichero disperso de 50 MB + 1
        fh.truncate(tg.TELEGRAM_DOC_MAX_BYTES + 1)
    ok, why = tg.send_document(big, "x")
    assert ok is False and "50 MB" in why
    got = {}

    def _post(url, data=None, files=None, timeout=None):
        got.update(url=url, data=data, files=files, timeout=timeout)
        return _Resp()
    monkeypatch.setattr(requests, "post", _post)
    ok, _ = tg.send_document(f, "c" * 2000, filename="copia.json.gz")
    assert ok and got["url"].endswith("/sendDocument") and got["data"]["chat_id"] == "4242"
    assert len(got["data"]["caption"]) == 1024 and got["files"]["document"][0] == "copia.json.gz"
    assert got["timeout"][1] <= 10
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(400, "Bad Request: file too big"))
    ok, why = tg.send_document(f, "x")
    assert not ok and "400" in why

    def _err(*a, **k):
        raise requests.ConnectionError("https://api.telegram.org/botTOK-SECRETO/sendDocument falló")
    monkeypatch.setattr(requests, "post", _err)
    ok, why = tg.send_document(f, "x")
    assert not ok and "TOK-SECRETO" not in why                         # el token no sale en el motivo


def test_send_document_limite_total_de_tiempo(tmp_path, monkeypatch):
    import requests
    import notifications.telegram as tg
    f = tmp_path / "c.json.gz"
    f.write_bytes(b"x" * 100)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "T")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    release = threading.Event()

    def _hang(*a, **k):
        release.wait(5)
        return _Resp()
    monkeypatch.setattr(requests, "post", _hang)
    t0 = time.monotonic()
    ok, why = tg.send_document(f, "x", deadline_s=0.3)
    assert not ok and "no respondió" in why and time.monotonic() - t0 < 1.5
    release.set()
    assert tg.DOCUMENT_DEADLINE_S <= 10


# --- Tarea nocturna real (cmd_schedule) con reloj falso ---------------------

class _StopLoop(Exception):
    pass


def _run_daemon(monkeypatch, start, end, send_ok=True):
    """main.cmd_schedule de verdad con reloj falso (como test_fase4_ciclo), ciclo y
    resumen simulados, copia mensual REAL con sendDocument simulado."""
    import schedule
    import main
    import config.settings as cs
    import data.outcome_tracker as ot
    import data.shadow as sh
    import monitoring.heartbeat as hb
    import monitoring.data_monitor as dm
    import monitoring.daily_summary as ds
    import notifications.telegram as tg
    import notifications.telegram_commands as tc
    from data import backup as B

    clock = {"now": _utc(start)}
    end_dt = _utc(end)

    class FakeDT(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return clock["now"].astimezone().replace(tzinfo=None)
            return clock["now"].astimezone(tz)
    monkeypatch.setattr(schedule, "datetime", types.SimpleNamespace(
        datetime=FakeDT, time=_dt.time, timedelta=_dt.timedelta, date=_dt.date, timezone=_dt.timezone))
    monkeypatch.setattr(main, "datetime", FakeDT)
    ev = []

    def _adv(s):
        clock["now"] = clock["now"] + _dt.timedelta(seconds=s)

    def _sleep(s):
        _adv(s)
        if clock["now"] >= end_dt:
            raise _StopLoop()

    def _send_doc(path, caption="", filename=None, **k):
        ev.append(("doc", pd.Timestamp(clock["now"]), Path(path).exists()))
        _adv(2)
        return (True, "ok") if send_ok else (False, "Telegram no respondió en 10 s")

    def _report(*a, **k):
        ev.append(("report", pd.Timestamp(clock["now"]), B.summary_line(int(clock["now"].timestamp()))))
        return True

    monkeypatch.setattr(main, "cmd_signal", lambda symbol, notify=False, ohlcv_cache=None: main._cycle_fresh.add(symbol))
    monkeypatch.setattr(main, "active_assets", lambda: ["BTCUSDT"])
    monkeypatch.setattr(main, "_flush_ohlcv_copies", lambda: 0)
    monkeypatch.setattr(ot, "resolve_open_signals",
                        lambda ohlcv_cache=None: ev.append(("cycle", pd.Timestamp(clock["now"]), None)) or 0)
    monkeypatch.setattr(sh, "resolve_shadow_setups", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(hb, "ping_heartbeat", lambda *a, **k: True)
    monkeypatch.setattr(dm, "run_nightly_checks", lambda notify=False: None)
    monkeypatch.setattr(ds, "send_daily_report", _report)
    monkeypatch.setattr(tg, "retry_failed_signal_alerts", lambda *a, **k: {})
    monkeypatch.setattr(tg, "test_connection", lambda *a, **k: True)
    monkeypatch.setattr(tg, "send_document", _send_doc)
    monkeypatch.setattr(tc, "poll_commands", lambda *a, **k: {})
    monkeypatch.setattr(cs, "TELEGRAM_TOKEN", "T")
    monkeypatch.setattr(cs, "TELEGRAM_CHAT_ID", "4242")
    monkeypatch.setattr(time, "sleep", _sleep)
    with pytest.raises(_StopLoop):
        main.cmd_schedule(60, "03:00")
    return ev


def test_tarea_nocturna_dia_1_envia_antes_del_resumen_y_el_ciclo_sigue(temp_db, monkeypatch):
    _seed_state(temp_db, n_sig=3, n_shadow=1)
    ev = _run_daemon(monkeypatch, "2026-12-01 02:58:30", "2026-12-01 03:10:00")
    docs = [e for e in ev if e[0] == "doc"]
    reps = [e for e in ev if e[0] == "report"]
    assert len(docs) == 1 and docs[0][2] is True and len(reps) == 1
    assert docs[0][1] < reps[0][1] and reps[0][2].startswith("💾 Copia de seguridad mensual enviada")
    c0301 = [t for k, t, _ in ev if k == "cycle" and t.strftime("%H:%M") >= "03:00"]
    assert len(c0301) == 1 and c0301[0] > reps[0][1]


def test_tarea_nocturna_fallo_de_envio_y_dia_sin_copia(temp_db, monkeypatch):
    ev = _run_daemon(monkeypatch, "2026-12-01 02:58:30", "2026-12-01 03:10:00", send_ok=False)
    reps = [e for e in ev if e[0] == "report"]
    assert len(reps) == 1 and "NO enviada" in reps[0][2]
    assert any(k == "cycle" and t.strftime("%H:%M") == "03:01" for k, t, _ in ev if t is not None)
    import schedule
    schedule.clear()
    ev = _run_daemon(monkeypatch, "2026-12-14 02:58:30", "2026-12-14 03:10:00")
    assert not [e for e in ev if e[0] == "doc"]
    assert [e for e in ev if e[0] == "report"][0][2] is None


# ---------------------------------------------------------------------------
# 3. Histórico propio en la validación
# ---------------------------------------------------------------------------

def _gate_frames(n4=900, start="2022-05-10 08:00", seed=3):
    """Velas con el formato de Gate.io (índice ms UTC llamado 'ts', float64)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n4, freq="4h", tz="UTC").as_unit("ms")
    c = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n4)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, n4))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, n4))
    d4 = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c,
                       "volume": rng.uniform(100, 900, n4)}, index=pd.Index(idx, name="ts"))
    g = d4.resample("1D", label="left", closed="left")
    d1 = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    d1 = d1[g["close"].count() == 6]
    d1.index = pd.Index(d1.index.as_unit("ms"), name="ts")
    return d4, d1


def _served(d4, d1, days):
    """Lo que serviría Gate.io `days` días más tarde (ventana desplazada)."""
    s4 = d4[d4.index >= d4.index[0] + pd.Timedelta(days=days)].copy()
    s1 = d1[d1.index >= d1.index[0] + pd.Timedelta(days=days)].copy()
    s4.attrs["source"] = "gateio"
    s1.attrs["source"] = "gateio"
    return s4, s1


def _seed_ohlcv(db, sym, d4, d1, source="gateio"):
    db.save_ohlcv(d4, sym, "4h", source=source)
    db.save_ohlcv(d1, sym, "1d", source=source)


NOW_S = int(pd.Timestamp("2026-11-16", tz="UTC").timestamp())


def test_sin_velas_antiguas_devuelve_los_mismos_objetos(temp_db):
    from data.db_history import extend_with_db_history
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 0)
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)              # la BD tiene lo mismo, nada más antiguo
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert r4 is s4 and r1 is s1 and info["used"] is False
    assert info["reason"] == "la BD no tiene velas 4H anteriores a las de Gate.io"
    # BD vacía o sin tabla ohlcv: igual.
    with temp_db.get_engine().begin() as c:
        c.execute(text("DROP TABLE ohlcv"))
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert r4 is s4 and r1 is s1 and not info["used"]


def test_anade_delante_las_velas_antiguas_identico_a_la_serie_completa(temp_db):
    from data.db_history import extend_with_db_history, describe
    d4, d1 = _gate_frames()
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)
    s4, s1 = _served(d4, d1, 40)
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert info["used"] and info["bars_4h"] == 240 and info["bars_1d"] == 40 and info["gaps"] == 0
    pd.testing.assert_frame_equal(r4, d4, check_freq=False)
    pd.testing.assert_frame_equal(r1, d1, check_freq=False)
    assert r4.index.dtype == s4.index.dtype and r4.index.name == "ts"
    assert r4.attrs["source"] == "gateio" and r4.attrs["db_history"]["bars_4h"] == 240
    assert "+240 velas 4H" in describe(info) and "2022-05-10 08:00" in describe(info)


def test_solo_gateio_cerradas_y_validas(temp_db):
    from data.db_history import extend_with_db_history
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 40)
    old4 = d4[d4.index < s4.index[0]]
    # OKX y sin fuente: no se usan nunca.
    temp_db.save_ohlcv(old4.iloc[:100], "BTCUSDT", "4h", source="okx")
    temp_db.save_ohlcv(old4.iloc[100:120], "BTCUSDT", "4h", source=None)
    r4, _, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert r4 is s4 and not info["used"]
    # Gate.io con una vela incoherente (low > close) y la BD de otro activo: se descartan.
    bad = old4.iloc[120:].copy()
    bad.iloc[5, bad.columns.get_loc("low")] = bad.iloc[5]["close"] * 1.5
    temp_db.save_ohlcv(bad, "BTCUSDT", "4h", source="gateio")
    temp_db.save_ohlcv(d4.iloc[:5], "ETHUSDT", "4h", source="gateio")
    r4, _, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert info["used"] and info["bars_4h"] == len(bad) - 1 and info["dropped"] >= 1
    assert r4.index[0] == bad.index[0] and bad.index[5] not in r4.index
    assert info["gaps"] == 1 and info["missing"] == 1
    # Una vela "abierta" según el reloj se descarta.
    r4b, _, info_b = extend_with_db_history("BTCUSDT", s4, s1, now_s=int(bad.index[3].timestamp()) + 100)
    assert info_b["bars_4h"] <= 3


def test_empalme_distinto_no_se_usa(temp_db):
    from data.db_history import extend_with_db_history
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 40)
    db4 = d4.copy()
    db4.loc[s4.index[2], "close"] *= 1.0005           # la copia no coincide con Gate.io
    _seed_ohlcv(temp_db, "BTCUSDT", db4, d1)
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert r4 is s4 and r1 is s1 and not info["used"] and "empalme" in info["reason"]


def test_hueco_se_usa_y_se_avisa(temp_db):
    from data.db_history import extend_with_db_history
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 40)
    _seed_ohlcv(temp_db, "BTCUSDT", d4.drop(d4.index[50:80]), d1)
    r4, _, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert info["used"] and info["gaps"] == 1 and info["missing"] == 30 and len(r4) == len(d4) - 30


def test_diario_se_amplia_solo_lo_mismo_que_el_4h(temp_db):
    from data.db_history import extend_with_db_history
    d4, d1 = _gate_frames(n4=1500)
    # Gate.io sirve el diario 20 días antes que el 4H (como hoy: 1.825 frente a 1.650 días).
    s4 = d4[d4.index >= d4.index[0] + pd.Timedelta(days=80)]
    s1 = d1[d1.index >= d1.index[0] + pd.Timedelta(days=60)]
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)
    # Y la BD tiene además 30 diarios aún más antiguos que el inicio del 4H.
    older = d1.iloc[:30].copy()
    older.index = d1.index[0] - pd.to_timedelta(range(30, 0, -1), unit="D")
    temp_db.save_ohlcv(older, "BTCUSDT", "1d", source="gateio")
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    # 4H: 80 días (480 velas) hacia atrás → diario: los mismos 80 días antes de su inicio
    # (60 diarios hasta d1[0] + 20 de los 30 más antiguos), no los 30 enteros.
    assert info["bars_4h"] == 480 and info["bars_1d"] == 80
    assert r1.index[0] == d1.index[0] - pd.Timedelta(days=20)
    assert (s1.index[0] - r1.index[0]) == (s4.index[0] - r4.index[0])
    # Si solo hay diarios antiguos (ningún 4H), el diario NO se amplía.
    with temp_db.get_engine().begin() as c:
        c.execute(text("DELETE FROM ohlcv WHERE timeframe = '4h' AND ts < :t"),
                  {"t": int(s4.index[0].timestamp())})
    r4, r1, info = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    assert r4 is s4 and r1 is s1 and not info["used"]


def test_activos_no_cripto_y_errores_no_cambian_nada(temp_db, monkeypatch):
    from data import db_history as H
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 40)
    _seed_ohlcv(temp_db, "XAUUSD", d4, d1, source="gateio")
    r4, r1, info = H.extend_with_db_history("XAUUSD", s4, s1, now_s=NOW_S)
    assert r4 is s4 and not info["used"]
    s4o = s4.copy()
    s4o.attrs["source"] = "yfinance"
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)
    assert H.extend_with_db_history("BTCUSDT", s4o, s1, now_s=NOW_S)[0] is s4o
    assert H.extend_with_db_history("BTCUSDT", None, None)[0] is None

    class _Broken:
        dialect = types.SimpleNamespace(name="sqlite")

        def connect(self):
            raise RuntimeError("BD caída")
    import backfill_history as bh
    monkeypatch.setattr(bh, "_ohlcv_columns", lambda e: {"source"})
    r4, r1, info = H.extend_with_db_history("BTCUSDT", s4, s1, engine=_Broken(), now_s=NOW_S)
    assert r4 is s4 and r1 is s1 and "BD caída" in info["reason"]


def test_validacion_usa_la_copia_y_el_vivo_no(temp_db, monkeypatch, capsys):
    import main
    import data.fetchers.binance_fetcher as bf
    from backtest import protocol as P
    monkeypatch.setattr(main, "_ohlcv_pending", {})     # la copia encolada no sale de esta prueba
    d4, d1 = _gate_frames(n4=1200)
    s4, s1 = _served(d4, d1, 60)
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)
    calls = []

    def _fetch(symbol, timeframe="4h", days=730):
        calls.append((symbol, timeframe, days))
        return (s4 if timeframe == "4h" else s1).copy()
    monkeypatch.setattr(bf, "fetch_ohlcv", _fetch)
    v4, v1, _ = main._load_validation_data("BTCUSDT")
    assert len(v4) == len(d4) and v4.index[0] == d4.index[0] and len(v1) == len(d1)
    # El vivo (cmd_signal → _load_data(history="live")) solo ve lo que sirve Gate.io.
    l4, l1, _ = main._load_data("BTCUSDT", history="live")
    assert l4.index[0] == s4.index[0] and len(l4) == len(s4)
    # La puerta usa el mismo cargador.
    p4, _ = P._default_load_data("BTCUSDT")
    assert p4.index[0] == d4.index[0]
    # check-live (CLI) con la copia: imprime la línea y el resultado es el de la serie completa.
    from backtest.live_emulation import compare_with_validator
    ref = compare_with_validator("BTCUSDT", d4, d1)
    capsys.readouterr()
    ok = main.cmd_check_live("BTCUSDT")
    out = capsys.readouterr().out
    assert "Histórico propio (tabla ohlcv, source='gateio'): +360 velas 4H" in out
    assert f"setups aprobados (validador): {ref['approved']}" in out
    assert ok == ref["match"]
    # Sin velas antiguas en la BD, check-live no imprime nada nuevo.
    with temp_db.get_engine().begin() as c:
        c.execute(text("DELETE FROM ohlcv"))
    main.cmd_check_live("BTCUSDT")
    assert "Histórico propio" not in capsys.readouterr().out


def test_protocolo_avisa_del_historico_propio_en_el_informe(temp_db, tmp_path, monkeypatch):
    from backtest import protocol as P
    d4, d1 = _gate_frames()
    s4, s1 = _served(d4, d1, 40)
    _seed_ohlcv(temp_db, "BTCUSDT", d4, d1)
    from data.db_history import extend_with_db_history
    e4, e1, _ = extend_with_db_history("BTCUSDT", s4, s1, now_s=NOW_S)
    echoed = []
    monkeypatch.setattr(P, "analyze_symbol", lambda *a, **k: (_ for _ in ()).throw(P.ProtocolError("parada")))
    code, rep = P.run_protocol("BTCUSDT", registry=tmp_path / "r.jsonl", load_data=lambda s: (e4, e1),
                               live_check=lambda s, a, b: {"match": True}, echo=echoed.append)
    assert code == 1
    assert any("Histórico propio" in e and "+240 velas 4H" in e for e in echoed)


# ---------------------------------------------------------------------------
# 4. Documentación
# ---------------------------------------------------------------------------

def test_reglas_documentan_copia_y_backfill():
    txt = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")
    sec = txt[txt.index("## Copia de seguridad"):]
    for frag in ("python trading_system/main.py backup", "restore --file", "--apply", "día 1",
                 "backfill_history.py --apply", "8 días", "source='gateio'", "ohlcv"):
        assert frag in sec, frag
