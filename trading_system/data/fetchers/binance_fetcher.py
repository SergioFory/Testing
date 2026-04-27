"""
Descarga datos de Binance: OHLCV (spot + futures), funding rates, open interest.
Usa ccxt para mayor compatibilidad y manejo de errores.
"""
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from loguru import logger

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    logger.warning("ccxt no instalado. Instala con: pip install ccxt")


def _ccxt_exchange(testnet: bool = False):
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
    ex = ccxt.binance({
        "apiKey":  BINANCE_API_KEY,
        "secret":  BINANCE_API_SECRET,
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    return ex


def fetch_ohlcv(symbol: str, timeframe: str = "4h",
                days: int = 730) -> pd.DataFrame:
    """
    Descarga barras OHLCV desde Binance Futures.

    Args:
        symbol:    e.g. "BTCUSDT"
        timeframe: "1h", "4h", "1d"
        days:      Días de historia hacia atrás

    Returns:
        DataFrame con columnas open/high/low/close/volume, index datetime UTC.
    """
    if not HAS_CCXT:
        return pd.DataFrame()

    ex = _ccxt_exchange()
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    logger.info(f"Descargando {symbol} {timeframe} ({days} días)...")
    while True:
        try:
            candles = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {e}")
            time.sleep(2)
            break

        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if len(candles) < limit:
            break
        since_ms = last_ts + 1
        time.sleep(ex.rateLimit / 1000)

    if not all_candles:
        logger.warning(f"Sin datos para {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles,
                      columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    logger.success(f"{symbol} {timeframe}: {len(df)} barras | "
                   f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def fetch_funding_rates(symbol: str, days: int = 365) -> pd.Series:
    """
    Retorna la tasa de financiamiento diaria promedio.
    Index: DatetimeIndex UTC. Values: float (tasa como decimal, e.g. 0.0001).
    """
    if not HAS_CCXT:
        return pd.Series(dtype=float)
    try:
        ex = _ccxt_exchange()
        since_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
        )
        all_rates = []
        while True:
            chunk = ex.fetch_funding_rate_history(
                symbol, since=since_ms, limit=1000
            )
            if not chunk:
                break
            all_rates.extend(chunk)
            since_ms = chunk[-1]["timestamp"] + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.5)

        if not all_rates:
            return pd.Series(dtype=float)

        df = pd.DataFrame(all_rates)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["rate"] = df["fundingRate"].astype(float)
        df.set_index("ts", inplace=True)
        daily = df["rate"].resample("1D").mean()
        logger.success(f"Funding rates {symbol}: {len(daily)} días")
        return daily
    except Exception as e:
        logger.warning(f"Funding rates no disponibles para {symbol}: {e}")
        return pd.Series(dtype=float)


def fetch_open_interest(symbol: str, days: int = 29) -> pd.DataFrame:
    """
    Open interest histórico diario (Binance limita a ~30 días).
    Retorna DataFrame con columna 'oi', index datetime UTC.
    """
    if not HAS_CCXT:
        return pd.DataFrame()
    try:
        from binance.client import Client
        from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        start_ms = int(
            (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
        )
        data = client.futures_open_interest_hist(
            symbol=symbol, period="1d", limit=30, startTime=start_ms
        )
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi"] = df["sumOpenInterest"].astype(float)
        df.set_index("ts", inplace=True)
        return df[["oi"]]
    except Exception as e:
        logger.warning(f"Open interest no disponible para {symbol}: {e}")
        return pd.DataFrame()


def fetch_multi_timeframe(symbol: str,
                          timeframes: list = None,
                          days: int = 730) -> dict:
    """
    Descarga múltiples timeframes para un símbolo.
    Retorna dict {timeframe: DataFrame}.
    """
    if timeframes is None:
        timeframes = ["4h", "1d"]
    result = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, days=days)
        if not df.empty:
            result[tf] = df
    return result
