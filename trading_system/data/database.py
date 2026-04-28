"""
Capa de persistencia SQLite.
Guarda OHLCV, señales generadas, operaciones y métricas de backtest.
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from loguru import logger

from config.settings import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    with get_connection() as conn:
        conn.executescript("""
        -- Barras OHLCV procesadas
        CREATE TABLE IF NOT EXISTS ohlcv (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            ts          INTEGER NOT NULL,    -- Unix timestamp UTC
            open        REAL    NOT NULL,
            high        REAL    NOT NULL,
            low         REAL    NOT NULL,
            close       REAL    NOT NULL,
            volume      REAL    NOT NULL,
            UNIQUE(symbol, timeframe, ts)
        );
        CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_ts
            ON ohlcv(symbol, timeframe, ts);

        -- Setups técnicos detectados
        CREATE TABLE IF NOT EXISTS setups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            timeframe   TEXT    NOT NULL,
            ts          INTEGER NOT NULL,
            setup_type  TEXT    NOT NULL,   -- breakout | pullback | reversal
            direction   TEXT    NOT NULL,   -- long | short
            score       REAL,              -- ML score (0-1), NULL si aún no evaluado
            params      TEXT,              -- JSON con parámetros del setup
            created_at  TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_setups_sym_ts
            ON setups(symbol, ts);

        -- Señales emitidas (setup + ML aprobado)
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            ts              INTEGER NOT NULL,
            setup_type      TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            ml_score        REAL    NOT NULL,
            sentiment_score REAL,
            entry_price     REAL    NOT NULL,
            stop_loss       REAL    NOT NULL,
            take_profit     REAL    NOT NULL,
            position_size   REAL    NOT NULL,   -- en USD
            atr             REAL    NOT NULL,
            resultado_real  TEXT,               -- win | loss | be | pendiente
            pnl_usd         REAL,
            pnl_pct         REAL,
            created_at      TEXT    DEFAULT (datetime('now'))
        );

        -- Historial de operaciones (para tracking real)
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id       INTEGER REFERENCES signals(id),
            symbol          TEXT    NOT NULL,
            direction       TEXT    NOT NULL,
            entry_price     REAL    NOT NULL,
            exit_price      REAL,
            stop_loss       REAL    NOT NULL,
            take_profit     REAL    NOT NULL,
            size_usd        REAL    NOT NULL,
            pnl_usd         REAL,
            pnl_pct         REAL,
            outcome         TEXT,               -- win | loss | be
            entry_ts        INTEGER,
            exit_ts         INTEGER,
            notes           TEXT
        );

        -- Resultados de backtest
        CREATE TABLE IF NOT EXISTS backtest_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL,
            symbol          TEXT    NOT NULL,
            start_date      TEXT    NOT NULL,
            end_date        TEXT    NOT NULL,
            setup_type      TEXT,
            total_trades    INTEGER,
            win_rate        REAL,
            avg_rr          REAL,
            sharpe          REAL,
            max_drawdown    REAL,
            total_return    REAL,
            annualized_ret  REAL,
            params          TEXT,   -- JSON
            created_at      TEXT    DEFAULT (datetime('now'))
        );
        """)
    logger.info(f"Base de datos inicializada en {DB_PATH}")


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str) -> int:
    """
    Inserta barras OHLCV. Ignora duplicados.
    Retorna número de filas insertadas.
    """
    if df.empty:
        return 0
    records = []
    for ts, row in df.iterrows():
        ts_int = int(pd.Timestamp(ts).timestamp())
        records.append((
            symbol, timeframe, ts_int,
            float(row["open"]), float(row["high"]),
            float(row["low"]),  float(row["close"]),
            float(row["volume"]),
        ))
    with get_connection() as conn:
        cursor = conn.executemany(
            """INSERT OR IGNORE INTO ohlcv
               (symbol, timeframe, ts, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?,?)""",
            records,
        )
        return cursor.rowcount


def load_ohlcv(symbol: str, timeframe: str,
               start: str = None, end: str = None) -> pd.DataFrame:
    """Carga barras desde la DB, retorna DataFrame indexado por datetime UTC."""
    query = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=?"
    params = [symbol, timeframe]
    if start:
        params.append(int(pd.Timestamp(start, tz="UTC").timestamp()))
        query += " AND ts >= ?"
    if end:
        params.append(int(pd.Timestamp(end, tz="UTC").timestamp()))
        query += " AND ts <= ?"
    query += " ORDER BY ts ASC"
    with get_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.set_index("ts", inplace=True)
    return df


def save_signal(signal: dict) -> int:
    """Persiste una señal. Retorna el ID asignado."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO signals
               (symbol, ts, setup_type, direction, ml_score, sentiment_score,
                entry_price, stop_loss, take_profit, position_size, atr)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal["symbol"],
                int(pd.Timestamp(signal["ts"]).timestamp()),
                signal["setup_type"],
                signal["direction"],
                signal["ml_score"],
                signal.get("sentiment_score"),
                signal["entry_price"],
                signal["stop_loss"],
                signal["take_profit"],
                signal["position_size"],
                signal["atr"],
            ),
        )
        return cursor.lastrowid


def load_signals(symbol: str = None, limit: int = 100) -> pd.DataFrame:
    query = "SELECT * FROM signals"
    params = []
    if symbol:
        query += " WHERE symbol = ?"
        params.append(symbol)
    query += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def save_backtest_result(result: dict) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO backtest_results
               (run_id, symbol, start_date, end_date, setup_type,
                total_trades, win_rate, avg_rr, sharpe, max_drawdown,
                total_return, annualized_ret, params)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                result["run_id"],
                result["symbol"],
                result["start_date"],
                result["end_date"],
                result.get("setup_type"),
                result.get("total_trades"),
                result.get("win_rate"),
                result.get("avg_rr"),
                result.get("sharpe"),
                result.get("max_drawdown"),
                result.get("total_return"),
                result.get("annualized_ret"),
                json.dumps(result.get("params", {})),
            ),
        )
