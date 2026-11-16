"""
Copia de seguridad de la base de datos (Fase 4, grupo D).

Qué se copia: todas las tablas con estado que no se puede reconstruir —señales,
fills (/fill y /skip), kv (offset de Telegram, estado de datos, avisos...),
registro sombra, incidencias de calidad de datos, retrain_log— y cualquier otra
tabla que haya en la BD (p. ej. signals_backup_fase0). NO se copia `ohlcv`: se
reconstruye con `python backfill_history.py --apply`.

Formato: un JSON comprimido con gzip (.json.gz), legible con cualquier
herramienta:
    {"format": "trading_system_backup", "version": 1, "created_utc": "...",
     "dialect": "postgresql", "excluded": ["ohlcv"],
     "tables": {"signals": {"columns": [...], "rows": [[...], ...]}, ...},
     "counts": {"signals": 312, ...}}
Sin datos de conexión (ni URL de la BD, ni tokens): solo el contenido de las tablas.

Uso:
    python main.py backup [--out FICHERO] [--send]   # crea la copia (y la envía por Telegram)
    python main.py restore --file FICHERO            # VISTA PREVIA: qué se insertaría
    python main.py restore --file FICHERO --apply    # inserta solo las filas que no existan

Restauración (probada en SQLite; PostgreSQL usa la misma vía más el ajuste de
secuencias):
  - Solo INSERTA filas que no existan; nunca modifica ni borra nada.
  - Una fila "existe" si en la BD hay otra con la misma clave primaria (id; en kv
    el nombre; en retrain_log el activo). Si es idéntica, se cuenta como "ya
    estaba"; si tiene otro contenido es un CONFLICTO y no se toca (la BD manda; el
    informe lo dice para que decidas).
  - Tampoco se inserta una fila que choque con una clave única (p. ej. el mismo
    setup del registro sombra ya guardado con otro id).
  - Conserva los id originales (los /fill usan el id de la señal). En PostgreSQL
    se ajusta después la secuencia de cada id para que las filas nuevas sigan
    numerándose tras la mayor restaurada.
  - Todo en UNA transacción: o entra todo lo nuevo o nada.

Copia mensual automática: run_monthly_backup() (tarea nocturna del worker, día 1
de cada mes; si falla se reintenta las noches siguientes hasta el día
MONTHLY_RETRY_UNTIL_DAY). Nunca lanza.
"""
from __future__ import annotations

import datetime as _dt
import decimal
import gzip
import json
import math
import os
import tempfile
import time
from pathlib import Path

from loguru import logger

BACKUP_FORMAT = "trading_system_backup"
BACKUP_VERSION = 1
# Tablas que no se copian: ohlcv se reconstruye con backfill_history.py.
EXCLUDED_TABLES = ("ohlcv",)
# Tablas conocidas (las crea data/database.init_db), en orden de restauración:
# signals antes que trades/fills, que apuntan a su id.
KNOWN_TABLES = ("signals", "trades", "fills", "shadow_setups", "kv", "data_quality",
                "retrain_log", "setups", "backtest_results")
PRIMARY_KEYS = {"kv": ("name",), "retrain_log": ("symbol",)}     # el resto: id
UNIQUE_KEYS = {"shadow_setups": (("symbol", "setup_ts", "direction"),),
               "fills": (("update_id",),),
               "data_quality": (("symbol", "check_name", "ts"),)}
SERIAL_TABLES = ("signals", "trades", "fills", "shadow_setups", "data_quality", "setups",
                 "backtest_results")
FETCH_CHUNK = 1000
INSERT_CHUNK = 500

# Copia mensual por Telegram.
KV_MONTHLY = "copia:mensual"          # kv: JSON con el último intento
MONTHLY_DAY = 1
MONTHLY_RETRY_UNTIL_DAY = 7           # si el día 1 falla, se reintenta hasta el día 7
SUMMARY_WINDOW_S = 24 * 3600          # el resumen muestra el intento de las últimas 24 h


class BackupError(RuntimeError):
    """Fichero de copia ilegible o con un formato que no es el de este sistema."""


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _jsonable(v):
    """Valor de la BD → tipo JSON. Fechas como 'AAAA-MM-DD HH:MM:SS' (el mismo
    texto que guarda SQLite con CURRENT_TIMESTAMP)."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return str(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return str(v)


def _same_value(a, b) -> bool:
    a, b = _jsonable(a), _jsonable(b)
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return False
        if math.isnan(fa) and math.isnan(fb):
            return True
        return fa == fb or abs(fa - fb) <= 1e-12 * max(abs(fa), abs(fb))
    return a == b


def _n(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB".replace(".", ",")
    return f"{n / 1024 ** 2:.2f} MB".replace(".", ",")


def _engine(engine=None):
    from data.database import get_engine
    return engine or get_engine()


def _table_names(engine) -> list[str]:
    from sqlalchemy import inspect
    return list(inspect(engine).get_table_names())


def _columns(engine, table: str) -> list[str]:
    from sqlalchemy import inspect
    return [c["name"] for c in inspect(engine).get_columns(table)]


def backup_tables(engine=None) -> list[str]:
    """Tablas que entran en la copia: las conocidas que existan (en orden) y
    después cualquier otra salvo ohlcv y las internas de SQLite."""
    engine = _engine(engine)
    names = _table_names(engine)
    known = [t for t in KNOWN_TABLES if t in names]
    other = sorted(t for t in names
                   if t not in KNOWN_TABLES and t not in EXCLUDED_TABLES and not t.startswith("sqlite_"))
    return known + other


def _pk(table: str) -> tuple:
    return PRIMARY_KEYS.get(table, ("id",))


def default_backup_path(now: _dt.datetime | None = None) -> Path:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return Path("backups") / f"trading_backup_{now:%Y%m%d_%H%M%S}.json.gz"


# ---------------------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------------------

def export_backup(path: str | Path | None = None, engine=None, now: _dt.datetime | None = None) -> dict:
    """
    Escribe la copia en `path` (gzip JSON) leyendo cada tabla por bloques (memoria
    acotada: nunca carga una tabla entera). Escribe primero en FICHERO.tmp y lo
    renombra al terminar (una copia a medias nunca queda con el nombre final).
    Devuelve {"path", "bytes", "counts", "created_utc", "dialect"}. Lanza si falla.
    """
    from sqlalchemy import text
    if engine is None:
        from data.database import init_db
        init_db()
    engine = _engine(engine)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    path = Path(path) if path else default_backup_path(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    created = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    counts: dict = {}
    tables = backup_tables(engine)
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
            head = {"format": BACKUP_FORMAT, "version": BACKUP_VERSION, "created_utc": created,
                    "dialect": engine.dialect.name, "excluded": list(EXCLUDED_TABLES)}
            f.write(json.dumps(head, ensure_ascii=False)[:-1] + ', "tables": {')
            for ti, table in enumerate(tables):
                cols = _columns(engine, table)
                order = [c for c in _pk(table) if c in cols]
                sql = f'SELECT * FROM "{table}"' + (f" ORDER BY {', '.join(order)}" if order else "")
                n = 0
                with engine.connect() as conn:
                    res = conn.execution_options(stream_results=True).execute(text(sql))
                    cols = list(res.keys())
                    f.write(("," if ti else "") + json.dumps(table) + ': {"columns": '
                            + json.dumps(cols) + ', "rows": [')
                    while True:
                        chunk = res.fetchmany(FETCH_CHUNK)
                        if not chunk:
                            break
                        for row in chunk:
                            f.write(("," if n else "") + json.dumps([_jsonable(v) for v in row],
                                                                   ensure_ascii=False))
                            n += 1
                    f.write("]}")
                counts[table] = n
            f.write('}, "counts": ' + json.dumps(counts) + "}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return {"path": str(path), "bytes": path.stat().st_size, "counts": counts,
            "created_utc": created, "dialect": engine.dialect.name}


def format_export(info: dict) -> str:
    parts = ", ".join(f"{t} {_n(n)}" for t, n in info["counts"].items())
    return (f"Copia guardada en {info['path']} ({_fmt_bytes(info['bytes'])} comprimida; BD "
            f"{info['dialect']}).\nFilas por tabla: {parts or 'ninguna tabla'}.\n"
            "La tabla ohlcv no se copia: se reconstruye con `python backfill_history.py --apply`.\n"
            "Para restaurar: python main.py restore --file FICHERO (vista previa) y después --apply.")


# ---------------------------------------------------------------------------
# Leer y restaurar
# ---------------------------------------------------------------------------

def read_backup(path: str | Path) -> dict:
    """Carga y valida una copia. Lanza BackupError si no es una copia válida."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise BackupError(f"no existe el fichero {path}")
    except (OSError, EOFError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BackupError(f"{path} no es un JSON gzip legible ({exc})")
    if not isinstance(data, dict) or data.get("format") != BACKUP_FORMAT:
        raise BackupError(f"{path} no es una copia de este sistema (format != {BACKUP_FORMAT})")
    if int(data.get("version", 0)) > BACKUP_VERSION:
        raise BackupError(f"la copia es de una versión más nueva ({data.get('version')}) que este código "
                          f"({BACKUP_VERSION}): actualiza el código antes de restaurar")
    tables = data.get("tables")
    if not isinstance(tables, dict):
        raise BackupError("la copia no tiene la sección 'tables'")
    for name, t in tables.items():
        if not isinstance(t, dict) or not isinstance(t.get("columns"), list) or not isinstance(t.get("rows"), list):
            raise BackupError(f"tabla {name}: formato inválido")
        width = len(t["columns"])
        if any(not isinstance(r, list) or len(r) != width for r in t["rows"]):
            raise BackupError(f"tabla {name}: hay filas con un número de columnas distinto de {width}")
    return data


def _key(row: dict, cols) -> tuple | None:
    vals = tuple(_jsonable(row.get(c)) for c in cols)
    return None if any(v is None for v in vals) else vals


def _plan_table(conn, table: str, spec: dict, target_cols: list[str] | None) -> dict:
    """Clasifica las filas de la copia de una tabla frente a la BD."""
    from sqlalchemy import text
    cols = list(spec["columns"])
    rows = [dict(zip(cols, r)) for r in spec["rows"]]
    plan = {"table": table, "rows": len(rows), "new": 0, "same": 0, "conflict": 0, "unique": 0,
            "nokey": 0, "dropped_cols": [], "insert_cols": [], "to_insert": [], "conflict_ids": [],
            "skipped": None, "creates_table": target_cols is None}
    if table not in KNOWN_TABLES:
        plan["skipped"] = "tabla sin esquema conocido: no se restaura automáticamente (sus filas siguen en el fichero)"
        return plan
    if target_cols is None:              # la tabla aún no existe: init_db la crea al aplicar
        target_cols = cols
    common = [c for c in cols if c in target_cols]
    plan["dropped_cols"] = [c for c in cols if c not in target_cols]
    plan["insert_cols"] = common
    pk = [c for c in _pk(table) if c in common]
    if len(pk) != len(_pk(table)):
        plan["skipped"] = f"la copia no trae la clave {', '.join(_pk(table))}"
        return plan
    uniq = [u for u in UNIQUE_KEYS.get(table, ()) if all(c in common for c in u)]

    existing: dict = {}
    seen_unique = [set() for _ in uniq]
    if not plan["creates_table"]:
        sel = ", ".join(f'"{c}"' for c in common)
        res = conn.execute(text(f'SELECT {sel} FROM "{table}"'))
        while True:
            chunk = res.fetchmany(FETCH_CHUNK)
            if not chunk:
                break
            for r in chunk:
                d = dict(zip(common, r))
                existing[_key(d, pk)] = d
                for i, u in enumerate(uniq):
                    k = _key(d, u)
                    if k is not None:
                        seen_unique[i].add(k)

    for row in rows:
        k = _key(row, pk)
        if k is None:
            plan["nokey"] += 1
            continue
        old = existing.get(k)
        if old is not None:
            if all(_same_value(row.get(c), old.get(c)) for c in common):
                plan["same"] += 1
            else:
                plan["conflict"] += 1
                if len(plan["conflict_ids"]) < 10:
                    plan["conflict_ids"].append(k[0] if len(k) == 1 else k)
            continue
        ukeys = [_key(row, u) for u in uniq]
        if any(uk is not None and uk in seen_unique[i] for i, uk in enumerate(ukeys)):
            plan["unique"] += 1
            continue
        for i, uk in enumerate(ukeys):
            if uk is not None:
                seen_unique[i].add(uk)
        existing[k] = row
        plan["new"] += 1
        plan["to_insert"].append({c: row.get(c) for c in common})
    return plan


def plan_restore(backup: dict, engine=None) -> list[dict]:
    """Plan por tabla (sin escribir nada): nuevas, ya iguales, conflictos..."""
    engine = _engine(engine)
    names = set(_table_names(engine))
    order = [t for t in KNOWN_TABLES if t in backup["tables"]] + \
            [t for t in backup["tables"] if t not in KNOWN_TABLES]
    plans = []
    with engine.connect() as conn:
        for table in order:
            if table in EXCLUDED_TABLES:
                continue
            target = _columns(engine, table) if table in names else None
            plans.append(_plan_table(conn, table, backup["tables"][table], target))
    return plans


def _insert_sql(table: str, cols: list[str]):
    from sqlalchemy import text
    names = ", ".join(f'"{c}"' for c in cols)
    params = ", ".join(f":p{i}" for i in range(len(cols)))
    # ON CONFLICT DO NOTHING (sin columna): red de seguridad ante cualquier clave
    # única; SQLite ≥ 3.24 y PostgreSQL ≥ 9.5.
    return text(f'INSERT INTO "{table}" ({names}) VALUES ({params}) ON CONFLICT DO NOTHING')


def apply_restore(plans: list[dict], engine=None) -> dict:
    """Inserta las filas nuevas de los planes en UNA transacción. Devuelve
    {tabla: filas insertadas}. Lanza si falla (y no queda nada a medias)."""
    from sqlalchemy import text
    engine = _engine(engine)
    dialect = engine.dialect.name
    inserted: dict = {}
    with engine.begin() as conn:
        for p in plans:
            if p["skipped"] or not p["to_insert"]:
                inserted[p["table"]] = 0
                continue
            cols = p["insert_cols"]
            sql = _insert_sql(p["table"], cols)
            n = 0
            batch = []
            for row in p["to_insert"]:
                batch.append({f"p{i}": row.get(c) for i, c in enumerate(cols)})
                if len(batch) >= INSERT_CHUNK:
                    n += max(0, conn.execute(sql, batch).rowcount or 0)
                    batch = []
            if batch:
                n += max(0, conn.execute(sql, batch).rowcount or 0)
            inserted[p["table"]] = n
            if dialect != "sqlite" and p["table"] in SERIAL_TABLES and "id" in cols:
                # PostgreSQL: con id explícitos la secuencia SERIAL no avanza; sin
                # esto la próxima señal del worker chocaría con un id restaurado.
                conn.execute(text(
                    f"SELECT setval(pg_get_serial_sequence('{p['table']}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM \"{p['table']}\"), 0) + 1, false)"))
            # SQLite (AUTOINCREMENT) actualiza sqlite_sequence solo al insertar ids mayores.
    return inserted


def restore_backup(path: str | Path, apply: bool = False, engine=None) -> dict:
    """Vista previa (por defecto) o restauración. Devuelve el informe."""
    backup = read_backup(path)
    if apply and engine is None:
        # Antes de planificar: crea tablas/columnas que falten (idempotente), así
        # el plan ya refleja el esquema real en el que se va a insertar.
        from data.database import init_db
        init_db()
    plans = plan_restore(backup, engine)
    report = {"path": str(path), "created_utc": backup.get("created_utc"),
              "dialect": backup.get("dialect"), "plans": plans, "applied": False, "inserted": {}}
    if apply:
        report["inserted"] = apply_restore(plans, engine)
        report["applied"] = True
    return report


def format_restore(report: dict) -> str:
    lines = [f"Copia {report['path']} (creada {report.get('created_utc') or '?'}, BD de origen "
             f"{report.get('dialect') or '?'}):"]
    any_conflict = False
    for p in report["plans"]:
        head = f"  {p['table']:<16} {_n(p['rows']):>7} filas en la copia"
        if p["skipped"]:
            lines.append(f"{head} · {p['skipped']}")
            continue
        verb = "insertadas" if report["applied"] else "se insertarían"
        n_new = report["inserted"].get(p["table"], 0) if report["applied"] else p["new"]
        txt = f"{head} · nuevas {verb}: {_n(n_new)} · ya estaban iguales: {_n(p['same'])}"
        if p["conflict"]:
            any_conflict = True
            ids = ", ".join(str(i) for i in p["conflict_ids"])
            txt += f" · EN CONFLICTO (misma clave, otro contenido; no se tocan): {_n(p['conflict'])} (p. ej. {ids})"
        if p["unique"]:
            txt += f" · ya existían con otra clave (clave única): {_n(p['unique'])}"
        if p["nokey"]:
            txt += f" · sin clave: {_n(p['nokey'])}"
        if p["creates_table"]:
            txt += " · la tabla no existía (se crea)"
        if p["dropped_cols"]:
            txt += f" · columnas que esta versión no tiene (se ignoran): {', '.join(p['dropped_cols'])}"
        lines.append(txt)
    if any_conflict:
        lines.append("Conflicto = en la BD hay una fila con la misma clave (id) y otro contenido, p. ej. si la "
                     "BD se recreó y el worker ya guardó señales nuevas con esos id. No se sobrescribe nada; "
                     "esas filas siguen en el fichero de la copia.")
    if not report["applied"]:
        lines.append(f"VISTA PREVIA: no se ha escrito nada. Para restaurar: python main.py restore --file "
                     f"{report['path']} --apply")
    else:
        lines.append(f"Restaurado: {_n(sum(report['inserted'].values()))} filas insertadas en una sola "
                     "transacción.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Copia mensual por Telegram (tarea nocturna)
# ---------------------------------------------------------------------------

def _now(now) -> _dt.datetime:
    if now is None:
        return _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc)


def _load_state() -> dict:
    try:
        from data.database import kv_get
        raw = kv_get(KV_MONTHLY)
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def monthly_backup_due(now=None, state: dict | None = None) -> bool:
    """True el día 1 de cada mes (UTC) y, si ese intento falló o no se hizo, las
    noches siguientes hasta el día MONTHLY_RETRY_UNTIL_DAY."""
    now = _now(now)
    if now.day < MONTHLY_DAY or now.day > MONTHLY_RETRY_UNTIL_DAY:
        return False
    state = _load_state() if state is None else state
    month = now.strftime("%Y-%m")
    if state.get("ok") and state.get("month") == month:
        return False                       # ya se envió este mes
    return True


def backup_caption(info: dict, filename: str, monthly: bool = True) -> str:
    c = info["counts"]
    kind = "Copia de seguridad mensual" if monthly else "Copia de seguridad"
    return (f"💾 {kind} de la base de datos ({info['created_utc'][:10]}): "
            f"{_n(c.get('signals', 0))} señales, {_n(c.get('fills', 0))} /fill y /skip, "
            f"{_n(c.get('shadow_setups', 0))} filas del registro sombra; {_fmt_bytes(info['bytes'])}.\n"
            f"Guárdala fuera de Render. Para restaurar: súbela al servidor y ejecuta\n"
            f"python trading_system/main.py restore --file {filename}\n"
            "(vista previa; añade --apply para insertar las filas que falten).")


def run_monthly_backup(now=None, send_document=None, force: bool = False) -> dict | None:
    """
    Si toca (monthly_backup_due) exporta la copia a un fichero temporal, la envía
    por Telegram como documento (sendDocument, límite total de 10 s) y la borra.
    Guarda el resultado en kv (KV_MONTHLY) para el resumen nocturno. Devuelve el
    estado guardado o None si no tocaba. NUNCA lanza.
    """
    try:
        now = _now(now)
        if not force and not monthly_backup_due(now):
            return None
    except Exception as exc:
        logger.warning(f"Copia mensual: no se pudo decidir si toca ({exc}).")
        return None
    t0 = time.monotonic()
    state = {"ts": int(now.timestamp()), "month": now.strftime("%Y-%m"), "ok": False}
    tmpdir = None
    try:
        if send_document is None:
            from notifications.telegram import send_document
        tmpdir = tempfile.mkdtemp(prefix="tsbackup_")
        filename = f"trading_backup_{now:%Y-%m-%d}.json.gz"
        info = export_backup(Path(tmpdir) / filename, now=now)
        state.update(bytes=info["bytes"], counts=info["counts"])
        ok, detail = send_document(info["path"], backup_caption(info, filename), filename=filename)
        state["ok"] = bool(ok)
        if not ok:
            state["error"] = str(detail)[:200]
    except Exception as exc:   # noqa: BLE001 — la copia nunca tumba la tarea nocturna
        state["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        if tmpdir:
            try:
                for p in Path(tmpdir).iterdir():
                    p.unlink()
                os.rmdir(tmpdir)
            except OSError:
                pass
    state["seconds"] = round(time.monotonic() - t0, 2)
    if state["ok"]:
        logger.info(f"Copia mensual enviada por Telegram ({_fmt_bytes(state.get('bytes', 0))}, "
                    f"{state['seconds']} s).")
    else:
        logger.warning(f"Copia mensual NO enviada: {state.get('error')}")
    try:
        from data.database import kv_set
        kv_set(KV_MONTHLY, json.dumps(state, ensure_ascii=False))
    except Exception as exc:
        logger.warning(f"Copia mensual: no se pudo guardar el estado en kv ({exc}).")
    return state


def summary_line(now_unix: int | None = None) -> str | None:
    """Línea para el resumen nocturno si hubo un intento de copia mensual en las
    últimas 24 h; None si no. Nunca lanza."""
    try:
        state = _load_state()
        if not state or "ts" not in state:
            return None
        now_unix = int(now_unix if now_unix is not None else time.time())
        if now_unix - int(state["ts"]) > SUMMARY_WINDOW_S or int(state["ts"]) > now_unix + 60:
            return None
        if state.get("ok"):
            return (f"💾 Copia de seguridad mensual enviada como documento "
                    f"({_fmt_bytes(int(state.get('bytes', 0)))}); guárdala fuera de Render.")
        import html
        return (f"💾 ⚠️ Copia de seguridad mensual NO enviada: {html.escape(str(state.get('error') or '?')[:160])}. "
                f"Se reintenta mañana (hasta el día {MONTHLY_RETRY_UNTIL_DAY}); a mano: "
                "python trading_system/main.py backup --send")
    except Exception:
        return None
