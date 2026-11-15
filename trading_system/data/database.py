"""
Capa de persistencia.

En desarrollo: SQLite (DB_PATH del settings).
En producción:  PostgreSQL vía variable de entorno DATABASE_URL.

SQLAlchemy abstrae ambos backends — el resto del código no cambia.
"""
import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.settings import DB_PATH

# ---------------------------------------------------------------------------
# Engine: PostgreSQL en producción, SQLite en desarrollo
# ---------------------------------------------------------------------------

def _build_engine() -> Engine:
    raw_url = os.getenv("DATABASE_URL", "")

    if raw_url:
        # Render expone "postgres://..." pero SQLAlchemy 2.x requiere "postgresql://"
        url = raw_url.replace("postgres://", "postgresql://", 1)
        logger.info("DB: PostgreSQL (DATABASE_URL)")
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)

    logger.info(f"DB: SQLite → {DB_PATH}")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})


_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


# Alias para compatibilidad con código existente que hace:
#   with get_connection() as conn:
#       conn.execute(...)
def get_connection():
    return get_engine().begin()


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ohlcv (
        id        SERIAL PRIMARY KEY,
        symbol    TEXT  NOT NULL,
        timeframe TEXT  NOT NULL,
        ts        BIGINT NOT NULL,
        open      DOUBLE PRECISION NOT NULL,
        high      DOUBLE PRECISION NOT NULL,
        low       DOUBLE PRECISION NOT NULL,
        close     DOUBLE PRECISION NOT NULL,
        volume    DOUBLE PRECISION NOT NULL,
        UNIQUE(symbol, timeframe, ts)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_ts ON ohlcv(symbol, timeframe, ts)",

    """
    CREATE TABLE IF NOT EXISTS setups (
        id         SERIAL PRIMARY KEY,
        symbol     TEXT  NOT NULL,
        timeframe  TEXT  NOT NULL,
        ts         BIGINT NOT NULL,
        setup_type TEXT  NOT NULL,
        direction  TEXT  NOT NULL,
        score      DOUBLE PRECISION,
        params     TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_setups_sym_ts ON setups(symbol, ts)",

    """
    CREATE TABLE IF NOT EXISTS signals (
        id              SERIAL PRIMARY KEY,
        symbol          TEXT   NOT NULL,
        ts              BIGINT NOT NULL,
        setup_type      TEXT   NOT NULL,
        direction       TEXT   NOT NULL,
        ml_score        DOUBLE PRECISION NOT NULL,
        sentiment_score DOUBLE PRECISION,
        entry_price     DOUBLE PRECISION NOT NULL,
        stop_loss       DOUBLE PRECISION NOT NULL,
        take_profit     DOUBLE PRECISION NOT NULL,
        position_size   DOUBLE PRECISION NOT NULL,
        atr             DOUBLE PRECISION NOT NULL,
        resultado_real  TEXT,
        pnl_usd         DOUBLE PRECISION,
        pnl_pct         DOUBLE PRECISION,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS trades (
        id          SERIAL PRIMARY KEY,
        signal_id   INTEGER REFERENCES signals(id),
        symbol      TEXT NOT NULL,
        direction   TEXT NOT NULL,
        entry_price DOUBLE PRECISION NOT NULL,
        exit_price  DOUBLE PRECISION,
        stop_loss   DOUBLE PRECISION NOT NULL,
        take_profit DOUBLE PRECISION NOT NULL,
        size_usd    DOUBLE PRECISION NOT NULL,
        pnl_usd     DOUBLE PRECISION,
        pnl_pct     DOUBLE PRECISION,
        outcome     TEXT,
        entry_ts    BIGINT,
        exit_ts     BIGINT,
        notes       TEXT
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS backtest_results (
        id             SERIAL PRIMARY KEY,
        run_id         TEXT NOT NULL,
        symbol         TEXT NOT NULL,
        start_date     TEXT NOT NULL,
        end_date       TEXT NOT NULL,
        setup_type     TEXT,
        total_trades   INTEGER,
        win_rate       DOUBLE PRECISION,
        avg_rr         DOUBLE PRECISION,
        sharpe         DOUBLE PRECISION,
        max_drawdown   DOUBLE PRECISION,
        total_return   DOUBLE PRECISION,
        annualized_ret DOUBLE PRECISION,
        params         TEXT,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # Registro del último reentrenamiento por activo. Vive en la DB (no en un
    # JSON en disco) para que un reinicio del Worker —p. ej. por OOM en Render—
    # no borre las fechas y dispare el reentrenamiento simultáneo de todos los
    # activos a la vez (lo que reproduciría el OOM en un bucle).
    """
    CREATE TABLE IF NOT EXISTS retrain_log (
        symbol      TEXT PRIMARY KEY,
        last_train  TEXT NOT NULL
    )
    """,

    # Fase 1 (punto 1.10) — REGISTRO SOMBRA. Cada setup detectado en las 2
    # últimas velas cerradas de cada ciclo (aprobado o no), con el motivo por el
    # que se emitió o no:
    #   emitted          registrado como señal live (notificada)
    #   shadow_asset     habría salido, pero el activo está en modo sombra (SPX)
    #   below_threshold  raw_score por debajo del umbral del activo
    #   not_fresh        aprobado, pero la entrada ya está a > max_dist × ATR
    #   event_blackout   aprobado, bloqueado por un evento macro inminente
    #   risk_invalid     aprobado, build_trade_setup lo rechazó
    #   duplicate        aprobado, bloqueado por el antiduplicado (precio/legado)
    #   not_best         aprobado y ejecutable, pero en ese ciclo salió otro de
    #                    mayor score (transitorio: suele pasar a 'emitted' en el
    #                    ciclo siguiente)
    # El tracker lo resuelve con la MISMA convención que las señales
    # (check_outcome_detail) para medir también lo que no se emitió. SL/TP = los
    # del setup. Tiempos en segundos Unix (como signals).
    """
    CREATE TABLE IF NOT EXISTS shadow_setups (
        id          SERIAL PRIMARY KEY,
        symbol      TEXT   NOT NULL,
        setup_ts    BIGINT NOT NULL,
        direction   TEXT   NOT NULL,
        setup_type  TEXT,
        raw_score   DOUBLE PRECISION,
        approved    INTEGER,
        motivo      TEXT   NOT NULL,
        entry       DOUBLE PRECISION,
        sl          DOUBLE PRECISION,
        tp          DOUBLE PRECISION,
        atr         DOUBLE PRECISION,
        created_ts  BIGINT,
        resultado   TEXT,
        pnl_r       DOUBLE PRECISION,
        exit_price  DOUBLE PRECISION,
        resolved_ts BIGINT,
        UNIQUE(symbol, setup_ts, direction)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_shadow_open ON shadow_setups(resultado)",
    "CREATE INDEX IF NOT EXISTS idx_shadow_created ON shadow_setups(created_ts)",

    # Fase 1 (punto 1.10) — ejecuciones reales que el usuario comunica por
    # Telegram: /fill <id> <precio> (kind='fill') y /skip <id> <motivo>
    # (kind='skip'). update_id = id de la actualización de Telegram: UNIQUE para
    # que procesar dos veces el mismo mensaje no duplique la fila.
    """
    CREATE TABLE IF NOT EXISTS fills (
        id         SERIAL PRIMARY KEY,
        signal_id  INTEGER NOT NULL,
        kind       TEXT    NOT NULL,
        price      DOUBLE PRECISION,
        note       TEXT,
        ts         BIGINT  NOT NULL,
        update_id  BIGINT  UNIQUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fills_signal ON fills(signal_id)",

    # Pares clave-valor persistentes (p. ej. el offset de getUpdates de
    # Telegram, para no procesar dos veces los comandos tras un reinicio).
    """
    CREATE TABLE IF NOT EXISTS kv (
        name       TEXT PRIMARY KEY,
        value      TEXT,
        updated_ts BIGINT
    )
    """,
]

# ---------------------------------------------------------------------------
# Migraciones de columnas (idempotentes). Se aplican tras crear las tablas.
# Fase 0 (2026-11): trazabilidad de cada señal y resultado en R.
#   setup_ts     hora (Unix) de la vela del setup → antiduplicado PERMANENTE por
#                (symbol, direction, setup_ts) en vez de por precio durante 48 h.
#   data_source  fuente de precios usada (gateio, GC=F, ^GSPC...).
#   last_bar_ts  hora (Unix) de la última vela cerrada al emitir.
#   mode         'live' (notificada) o 'shadow' (registrada sin notificar).
#   exit_price   precio de salida usado para el resultado.
#   pnl_r        resultado en R (múltiplos del riesgo entrada→stop).
#   resolved_ts  hora (Unix) de la vela en que se resolvió.
#   quality_flag marca de calidad para excluir de estadísticas (p. ej. re-emisiones
#                antiguas detectadas por la limpieza de la Fase 0).
# Fase 1 (2026-11): notificación fiable.
#   notified     1 = enviada a Telegram; 0 = el envío falló (se reintenta en los
#                ciclos siguientes mientras tenga < 2 h); NULL = no aplica (modo
#                sombra, registrada desde el dashboard o filas anteriores).
#                INTEGER (no BOOLEAN) para que funcione igual en SQLite y PostgreSQL.
#   alert_text   texto HTML exacto del mensaje, para reenviarlo idéntico si falla.
# ---------------------------------------------------------------------------
_SIGNAL_COLUMNS = [
    ("setup_ts",     "BIGINT"),
    ("data_source",  "TEXT"),
    ("last_bar_ts",  "BIGINT"),
    ("mode",         "TEXT"),
    ("exit_price",   "DOUBLE PRECISION"),
    ("pnl_r",        "DOUBLE PRECISION"),
    ("resolved_ts",  "BIGINT"),
    ("quality_flag", "TEXT"),
    ("notified",     "INTEGER"),
    ("alert_text",   "TEXT"),
    # Quién cerró la señal (Fase 1, 1.9): 'auto' = el tracker por velas;
    # 'manual' = el formulario del dashboard. NULL = filas anteriores.
    ("resolved_by",  "TEXT"),
]
_SIGNAL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_signals_setup ON signals(symbol, direction, setup_ts)",
    "CREATE INDEX IF NOT EXISTS idx_signals_open ON signals(resultado_real)",
]

# Fase 1 (punto 1.11) — copia propia del histórico en la tabla ohlcv.
#   source  de dónde salió cada vela: 'gateio' (la fuente del sistema), 'okx'
#           (perpetuos USDT de OKX, solo para fechas que Gate.io ya no sirve),
#           o el ticker de yfinance (GC=F, ^GSPC...). NULL = filas guardadas
#           antes de existir la columna (procedencia sin verificar).
_OHLCV_COLUMNS = [
    ("source", "TEXT"),
]


def _existing_columns(conn, table: str, dialect: str) -> set:
    if dialect == "sqlite":
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return {r[1] for r in rows}
    rows = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :t AND table_schema = current_schema()"
    ), {"t": table}).fetchall()
    return {r[0] for r in rows}


def _migrate(conn, dialect: str) -> None:
    # El worker y el dashboard pueden arrancar a la vez con el mismo despliegue:
    # en PostgreSQL se usa ADD COLUMN IF NOT EXISTS (y un candado de transacción)
    # y en SQLite se tolera "duplicate column" para que ninguno de los dos falle.
    if dialect != "sqlite":
        conn.execute(text("SELECT pg_advisory_xact_lock(724501)"))
    for table, columns in (("signals", _SIGNAL_COLUMNS), ("ohlcv", _OHLCV_COLUMNS)):
        have = _existing_columns(conn, table, dialect)
        for col, sqltype in columns:
            if col in have:
                continue
            if dialect == "sqlite":
                try:
                    conn.execute(text(_adapt_ddl(f"ALTER TABLE {table} ADD COLUMN {col} {sqltype}", dialect)))
                except Exception as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise
            else:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {sqltype}"))
            logger.info(f"Migración: columna {table}.{col} añadida.")
    for stmt in _SIGNAL_INDEXES:
        conn.execute(text(stmt))


# SQLite doesn't support SERIAL — patch DDL when needed
def _adapt_ddl(sql: str, dialect: str) -> str:
    if dialect == "sqlite":
        return (sql
                .replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
                .replace("DOUBLE PRECISION", "REAL")
                .replace("BIGINT", "INTEGER")
                .replace("TIMESTAMP", "TEXT"))
    return sql


_db_ready = False


def init_db(force: bool = False) -> None:
    """Crea todas las tablas si no existen y aplica las migraciones de columnas.
    Se ejecuta una sola vez por proceso (antes corría en cada carga de datos)."""
    global _db_ready
    if _db_ready and not force:
        return
    engine = get_engine()
    dialect = engine.dialect.name
    with engine.begin() as conn:
        for stmt in _DDL:
            conn.execute(text(_adapt_ddl(stmt, dialect)))
        _migrate(conn, dialect)
    _db_ready = True
    logger.info("Base de datos inicializada.")


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str, source: str | None = None) -> int:
    """Guarda velas en la copia OHLCV (sin sobrescribir las que ya existan).
    source: fuente de las velas; por defecto df.attrs["source"] (gateio, GC=F...)."""
    if df.empty:
        return 0

    init_db()
    if source is None:
        source = (getattr(df, "attrs", None) or {}).get("source")
    engine  = get_engine()
    dialect = engine.dialect.name
    records = []
    for ts, row in df.iterrows():
        records.append({
            "symbol":    symbol,
            "timeframe": timeframe,
            "ts":        int(pd.Timestamp(ts).timestamp()),
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row["volume"]),
            "source":    source,
        })

    if dialect == "sqlite":
        sql = text("""
            INSERT OR IGNORE INTO ohlcv
                (symbol, timeframe, ts, open, high, low, close, volume, source)
            VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :source)
        """)
    else:
        sql = text("""
            INSERT INTO ohlcv
                (symbol, timeframe, ts, open, high, low, close, volume, source)
            VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :source)
            ON CONFLICT (symbol, timeframe, ts) DO NOTHING
        """)

    with engine.begin() as conn:
        result = conn.execute(sql, records)
        return result.rowcount


def load_ohlcv(symbol: str, timeframe: str,
               start: str = None, end: str = None) -> pd.DataFrame:
    query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=:sym AND timeframe=:tf"
    params: dict = {"sym": symbol, "tf": timeframe}
    if start:
        params["start"] = int(pd.Timestamp(start, tz="UTC").timestamp())
        query += " AND ts >= :start"
    if end:
        params["end"] = int(pd.Timestamp(end, tz="UTC").timestamp())
        query += " AND ts <= :end"
    query += " ORDER BY ts ASC"

    with get_engine().connect() as conn:
        df = pd.read_sql_query(text(query), conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.set_index("ts", inplace=True)
    return df


def ohlcv_coverage() -> list[dict]:
    """Cobertura de la copia OHLCV por símbolo, timeframe y fuente:
    [{symbol, timeframe, source, n, first_ts, last_ts}] (ts en segundos Unix;
    source None = filas sin fuente registrada). Solo agrega en la BD."""
    init_db()
    sql = text("""
        SELECT symbol, timeframe, source, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts
        FROM ohlcv GROUP BY symbol, timeframe, source
        ORDER BY symbol, timeframe, MIN(ts)
    """)
    with get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(sql).mappings().fetchall()]


def ohlcv_gaps(symbol: str, timeframe: str, step_seconds: int) -> dict:
    """Huecos de una serie de paso fijo (cripto 24/7): número de saltos mayores
    que step_seconds entre velas consecutivas y velas que faltan en total.
    Se calcula en la BD (LAG, SQLite ≥ 3.25 y PostgreSQL) sin cargar la serie."""
    init_db()
    sql = text("""
        SELECT COUNT(*) AS gaps, COALESCE(SUM(d / :step - 1), 0) AS missing, MAX(d) AS max_gap
        FROM (SELECT ts - LAG(ts) OVER (ORDER BY ts) AS d
              FROM ohlcv WHERE symbol = :sym AND timeframe = :tf) t
        WHERE d > :step
    """)
    with get_engine().connect() as conn:
        row = conn.execute(sql, {"sym": symbol, "tf": timeframe, "step": int(step_seconds)}).mappings().fetchone()
    return {"gaps": int(row["gaps"] or 0), "missing": int(row["missing"] or 0),
            "max_gap_s": int(row["max_gap"] or 0)}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def save_signal(signal: dict) -> int:
    engine  = get_engine()
    dialect = engine.dialect.name

    init_db()
    cols = ("symbol, ts, setup_type, direction, ml_score, sentiment_score, "
            "entry_price, stop_loss, take_profit, position_size, atr, "
            "setup_ts, data_source, last_bar_ts, mode, notified, alert_text")
    vals = (":symbol, :ts, :setup_type, :direction, :ml_score, :sentiment_score, "
            ":entry_price, :stop_loss, :take_profit, :position_size, :atr, "
            ":setup_ts, :data_source, :last_bar_ts, :mode, :notified, :alert_text")
    if dialect == "sqlite":
        sql = text(f"INSERT INTO signals ({cols}) VALUES ({vals})")
        with engine.begin() as conn:
            conn.execute(sql, _signal_params(signal))
            row = conn.execute(text("SELECT last_insert_rowid()")).fetchone()
            return row[0] if row else -1
    else:
        sql = text(f"INSERT INTO signals ({cols}) VALUES ({vals}) RETURNING id")
        with engine.begin() as conn:
            row = conn.execute(sql, _signal_params(signal)).fetchone()
            return row[0] if row else -1


def _to_unix(value):
    """Timestamp/datetime/epoch → entero Unix (segundos), o None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    import numpy as np
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)   # segundos Unix (pd.Timestamp los tomaría como ns)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp())


def _signal_params(signal: dict) -> dict:
    notified = signal.get("notified")
    return {
        "notified":        None if notified is None else int(bool(notified)),
        "alert_text":      signal.get("alert_text"),
        "setup_ts":        _to_unix(signal.get("setup_ts")),
        "data_source":     signal.get("data_source"),
        "last_bar_ts":     _to_unix(signal.get("last_bar_ts")),
        "mode":            signal.get("mode") or "live",
        "symbol":          signal["symbol"],
        "ts":              int(pd.Timestamp(signal["ts"]).timestamp()),
        "setup_type":      signal["setup_type"],
        "direction":       signal["direction"],
        "ml_score":        signal["ml_score"],
        "sentiment_score": signal.get("sentiment_score"),
        "entry_price":     signal["entry_price"],
        "stop_loss":       signal["stop_loss"],
        "take_profit":     signal["take_profit"],
        "position_size":   signal["position_size"],
        "atr":             signal["atr"],
    }


def mark_signal_notified(signal_id: int, notified: bool = True) -> bool:
    """Marca una señal como enviada (1) o no enviada (0) a Telegram.
    Devuelve False si la escritura falla (se registra en el log, no lanza)."""
    try:
        with get_engine().begin() as conn:
            conn.execute(text("UPDATE signals SET notified=:n WHERE id=:id"),
                         {"n": int(bool(notified)), "id": int(signal_id)})
        return True
    except Exception as exc:
        logger.error(f"No se pudo marcar notified={int(bool(notified))} en la señal #{signal_id}: {exc}")
        return False


def load_unnotified_signals() -> list[dict]:
    """Señales live cuyo envío a Telegram falló (notified = 0), de la más antigua
    a la más reciente. Las filas con notified NULL (sombra, dashboard, filas
    anteriores a la Fase 1) no se tocan."""
    init_db()
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, symbol, ts, direction, setup_type, entry_price, stop_loss, "
            "take_profit, ml_score, alert_text FROM signals "
            "WHERE notified = 0 AND (mode IS NULL OR mode = 'live') ORDER BY ts ASC"
        )).mappings().fetchall()
    return [dict(r) for r in rows]


def load_signals(symbol: str = None, limit: int | None = 100) -> pd.DataFrame:
    """Señales de la BD, de la más reciente a la más antigua.
    limit=None → todas (las estadísticas del dashboard no deben recortarse)."""
    init_db()
    query  = "SELECT * FROM signals"
    params: dict = {}
    if symbol:
        query += " WHERE symbol = :symbol"
        params["symbol"] = symbol
    query += " ORDER BY ts DESC"
    if limit is not None:
        query += " LIMIT :limit"
        params["limit"] = int(limit)

    with get_engine().connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------

def save_backtest_result(result: dict) -> None:
    sql = text("""
        INSERT INTO backtest_results
            (run_id, symbol, start_date, end_date, setup_type,
             total_trades, win_rate, avg_rr, sharpe, max_drawdown,
             total_return, annualized_ret, params)
        VALUES (:run_id, :symbol, :start_date, :end_date, :setup_type,
                :total_trades, :win_rate, :avg_rr, :sharpe, :max_drawdown,
                :total_return, :annualized_ret, :params)
    """)
    params = {
        "run_id":        result["run_id"],
        "symbol":        result["symbol"],
        "start_date":    result["start_date"],
        "end_date":      result["end_date"],
        "setup_type":    result.get("setup_type"),
        "total_trades":  result.get("total_trades"),
        "win_rate":      result.get("win_rate"),
        "avg_rr":        result.get("avg_rr"),
        "sharpe":        result.get("sharpe"),
        "max_drawdown":  result.get("max_drawdown"),
        "total_return":  result.get("total_return"),
        "annualized_ret":result.get("annualized_ret"),
        "params":        json.dumps(result.get("params", {})),
    }
    with get_engine().begin() as conn:
        conn.execute(sql, params)


# ---------------------------------------------------------------------------
# Retrain log (persistente en DB — sobrevive a reinicios del Worker)
# ---------------------------------------------------------------------------

def get_last_train(symbol: str) -> str | None:
    """Devuelve el ISO timestamp del último reentrenamiento, o None si nunca."""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(
                text("SELECT last_train FROM retrain_log WHERE symbol = :sym"),
                {"sym": symbol},
            ).scalar()
            return row
    except Exception as exc:
        logger.warning(f"get_last_train falló para {symbol}: {exc}")
        return None


def set_last_train(symbol: str, iso_ts: str) -> None:
    """Registra (upsert) la fecha del último reentrenamiento del símbolo."""
    engine  = get_engine()
    dialect = engine.dialect.name
    if dialect == "sqlite":
        sql = text(
            "INSERT INTO retrain_log (symbol, last_train) VALUES (:sym, :ts) "
            "ON CONFLICT(symbol) DO UPDATE SET last_train = excluded.last_train"
        )
    else:
        sql = text(
            "INSERT INTO retrain_log (symbol, last_train) VALUES (:sym, :ts) "
            "ON CONFLICT (symbol) DO UPDATE SET last_train = EXCLUDED.last_train"
        )
    try:
        with engine.begin() as conn:
            conn.execute(sql, {"sym": symbol, "ts": iso_ts})
    except Exception as exc:
        logger.warning(f"set_last_train falló para {symbol}: {exc}")


# ---------------------------------------------------------------------------
# Clave-valor persistente (Fase 1): p. ej. offset de getUpdates de Telegram
# ---------------------------------------------------------------------------

def kv_get(name: str, default: str | None = None) -> str | None:
    """Valor guardado bajo `name`, o `default` si no existe. Lanza si la BD
    falla (quien llama decide: el polling de Telegram NO procesa comandos si no
    sabe por dónde iba)."""
    init_db()
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT value FROM kv WHERE name = :n"), {"n": name}).fetchone()
    return row[0] if row is not None else default


def kv_set(name: str, value) -> None:
    """Guarda (upsert) `value` como texto bajo `name`. Lanza si la BD falla."""
    import time
    init_db()
    sql = text(
        "INSERT INTO kv (name, value, updated_ts) VALUES (:n, :v, :t) "
        "ON CONFLICT (name) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts"
    )
    with get_engine().begin() as conn:
        conn.execute(sql, {"n": name, "v": None if value is None else str(value),
                           "t": int(time.time())})


# ---------------------------------------------------------------------------
# Fills (Fase 1): ejecuciones reales comunicadas por Telegram (/fill, /skip)
# ---------------------------------------------------------------------------

def save_fill(signal_id: int, kind: str, price: float | None = None,
              note: str | None = None, ts=None, update_id: int | None = None) -> bool:
    """Inserta un fill/skip. Devuelve False si ese update_id de Telegram ya se
    había guardado (mensaje procesado dos veces: no se duplica)."""
    import time
    if kind not in ("fill", "skip"):
        raise ValueError(f"kind debe ser 'fill' o 'skip', no {kind!r}")
    init_db()
    params = {"sid": int(signal_id), "kind": kind,
              "price": None if price is None else float(price),
              "note": note, "ts": _to_unix(ts) if ts is not None else int(time.time()),
              "uid": None if update_id is None else int(update_id)}
    sql = text(
        "INSERT INTO fills (signal_id, kind, price, note, ts, update_id) "
        "VALUES (:sid, :kind, :price, :note, :ts, :uid) ON CONFLICT (update_id) DO NOTHING"
    )
    with get_engine().begin() as conn:
        res = conn.execute(sql, params)
        return (res.rowcount or 0) > 0


def load_fills(signal_id: int | None = None) -> pd.DataFrame:
    """Fills/skips registrados (todos o de una señal), del más antiguo al más reciente."""
    init_db()
    query, params = "SELECT * FROM fills", {}
    if signal_id is not None:
        query += " WHERE signal_id = :sid"
        params["sid"] = int(signal_id)
    query += " ORDER BY ts ASC, id ASC"
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)


def load_signal_by_id(signal_id: int) -> dict | None:
    """Fila de signals con ese id (como dict) o None."""
    init_db()
    with get_engine().connect() as conn:
        row = conn.execute(text("SELECT * FROM signals WHERE id = :i"),
                           {"i": int(signal_id)}).mappings().fetchone()
    return dict(row) if row is not None else None


def set_signal_alert_text(signal_id: int, alert_text: str) -> bool:
    """Actualiza el texto guardado de la alerta (p. ej. tras añadir el ID de la
    señal). Devuelve False si falla (se registra, no lanza)."""
    try:
        with get_engine().begin() as conn:
            conn.execute(text("UPDATE signals SET alert_text=:t WHERE id=:id"),
                         {"t": alert_text, "id": int(signal_id)})
        return True
    except Exception as exc:
        logger.error(f"No se pudo actualizar alert_text de la señal #{signal_id}: {exc}")
        return False
