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
]

# SQLite doesn't support SERIAL — patch DDL when needed
def _adapt_ddl(sql: str, dialect: str) -> str:
    if dialect == "sqlite":
        return (sql
                .replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
                .replace("DOUBLE PRECISION", "REAL")
                .replace("BIGINT", "INTEGER")
                .replace("TIMESTAMP", "TEXT"))
    return sql


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    engine = get_engine()
    dialect = engine.dialect.name
    with engine.begin() as conn:
        for stmt in _DDL:
            conn.execute(text(_adapt_ddl(stmt, dialect)))
    logger.info("Base de datos inicializada.")


# ---------------------------------------------------------------------------
# OHLCV
# ---------------------------------------------------------------------------

def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    if df.empty:
        return 0

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
        })

    if dialect == "sqlite":
        sql = text("""
            INSERT OR IGNORE INTO ohlcv
                (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume)
        """)
    else:
        sql = text("""
            INSERT INTO ohlcv
                (symbol, timeframe, ts, open, high, low, close, volume)
            VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume)
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


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def save_signal(signal: dict) -> int:
    engine  = get_engine()
    dialect = engine.dialect.name

    if dialect == "sqlite":
        sql = text("""
            INSERT INTO signals
                (symbol, ts, setup_type, direction, ml_score, sentiment_score,
                 entry_price, stop_loss, take_profit, position_size, atr)
            VALUES (:symbol, :ts, :setup_type, :direction, :ml_score, :sentiment_score,
                    :entry_price, :stop_loss, :take_profit, :position_size, :atr)
        """)
        with engine.begin() as conn:
            conn.execute(sql, _signal_params(signal))
            row = conn.execute(text("SELECT last_insert_rowid()")).fetchone()
            return row[0] if row else -1
    else:
        sql = text("""
            INSERT INTO signals
                (symbol, ts, setup_type, direction, ml_score, sentiment_score,
                 entry_price, stop_loss, take_profit, position_size, atr)
            VALUES (:symbol, :ts, :setup_type, :direction, :ml_score, :sentiment_score,
                    :entry_price, :stop_loss, :take_profit, :position_size, :atr)
            RETURNING id
        """)
        with engine.begin() as conn:
            row = conn.execute(sql, _signal_params(signal)).fetchone()
            return row[0] if row else -1


def _signal_params(signal: dict) -> dict:
    return {
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


def load_signals(symbol: str = None, limit: int = 100) -> pd.DataFrame:
    query  = "SELECT * FROM signals"
    params: dict = {}
    if symbol:
        query += " WHERE symbol = :symbol"
        params["symbol"] = symbol
    query += " ORDER BY ts DESC LIMIT :limit"
    params["limit"] = limit

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
