"""
Descarga datos de mercado: OHLCV, funding rates.

Fuente primaria:  Binance Futures (ccxt) — puede fallar desde IPs de EE.UU.
Fuente de respaldo: Bybit (ccxt) — sin restricciones geográficas, mismos datos.
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

# Mapa de símbolos Binance → Bybit (futuros perpetuos lineales)
_BYBIT_SYMBOL = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
}


def _binance_exchange():
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
    config = {"options": {"defaultType": "future"}, "enableRateLimit": True}
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        config["apiKey"] = BINANCE_API_KEY
        config["secret"] = BINANCE_API_SECRET
    return ccxt.binance(config)


def _bybit_exchange():
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    return ccxt.bybit({"options": {"defaultType": "linear"}, "enableRateLimit": True})


def _fetch_ohlcv_from(exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Descarga OHLCV de cualquier exchange ccxt."""
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as e:
            logger.error(f"Error descargando {symbol} de {exchange.id}: {e}")
            time.sleep(2)
            return pd.DataFrame()

        if not candles:
            break
        all_candles.extend(candles)
        last_ts = candles[-1][0]
        if len(candles) < limit:
            break
        since_ms = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)
    return df


def fetch_ohlcv(symbol: str, timeframe: str = "4h", days: int = 730) -> pd.DataFrame:
    """
    Descarga OHLCV. Intenta Binance primero; si falla por bloqueo de IP (error 451),
    usa Bybit automáticamente.
    """
    if not HAS_CCXT:
        return pd.DataFrame()

    logger.info(f"Descargando {symbol} {timeframe} ({days} días)...")

    # Intento 1: Binance
    try:
        df = _fetch_ohlcv_from(_binance_exchange(), symbol, timeframe, days)
        if not df.empty:
            logger.success(f"{symbol} {timeframe} vía Binance: {len(df)} barras | "
                           f"{df.index[0].date()} → {df.index[-1].date()}")
            return df
    except Exception as e:
        logger.warning(f"Binance no disponible para {symbol}: {e}")

    # Intento 2: Bybit (sin restricciones geográficas)
    bybit_sym = _BYBIT_SYMBOL.get(symbol)
    if not bybit_sym:
        logger.warning(f"Sin datos para {symbol} {timeframe}")
        return pd.DataFrame()

    logger.info(f"Usando Bybit como fuente alternativa para {symbol}...")
    try:
        df = _fetch_ohlcv_from(_bybit_exchange(), bybit_sym, timeframe, days)
        if not df.empty:
            logger.success(f"{symbol} {timeframe} vía Bybit: {len(df)} barras | "
                           f"{df.index[0].date()} → {df.index[-1].date()}")
            return df
    except Exception as e:
        logger.error(f"Bybit también falló para {symbol}: {e}")

    logger.warning(f"Sin datos para {symbol} {timeframe}")
    return pd.DataFrame()


def fetch_funding_rates(symbol: str, days: int = 365) -> pd.Series:
    """
    Tasa de financiamiento diaria promedio.
    Intenta Binance (con API key) → Bybit (público).
    """
    if not HAS_CCXT:
        return pd.Series(dtype=float)

    # Intento Binance
    try:
        ex = _binance_exchange()
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        all_rates = []
        while True:
            chunk = ex.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
            if not chunk:
                break
            all_rates.extend(chunk)
            since_ms = chunk[-1]["timestamp"] + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.5)
        if all_rates:
            df = pd.DataFrame(all_rates)
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["rate"] = df["fundingRate"].astype(float)
            df.set_index("ts", inplace=True)
            return df["rate"].resample("1D").mean()
    except Exception as e:
        logger.warning(f"Funding rates Binance no disponibles para {symbol}: {e}")

    # Fallback Bybit
    bybit_sym = _BYBIT_SYMBOL.get(symbol)
    if not bybit_sym:
        return pd.Series(dtype=float)
    try:
        ex = _bybit_exchange()
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        all_rates = []
        while True:
            chunk = ex.fetch_funding_rate_history(bybit_sym, since=since_ms, limit=1000)
            if not chunk:
                break
            all_rates.extend(chunk)
            since_ms = chunk[-1]["timestamp"] + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.5)
        if all_rates:
            df = pd.DataFrame(all_rates)
            df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["rate"] = df["fundingRate"].astype(float)
            df.set_index("ts", inplace=True)
            daily = df["rate"].resample("1D").mean()
            logger.success(f"Funding rates {symbol} vía Bybit: {len(daily)} días")
            return daily
    except Exception as e:
        logger.warning(f"Funding rates Bybit no disponibles para {symbol}: {e}")

    return pd.Series(dtype=float)


def fetch_open_interest(symbol: str, days: int = 29) -> pd.DataFrame:
    """Open interest histórico — requiere API keys de Binance."""
    from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        return pd.DataFrame()
    try:
        from binance.client import Client
        client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        data = client.futures_open_interest_hist(symbol=symbol, period="1d", limit=30, startTime=start_ms)
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


def fetch_multi_timeframe(symbol: str, timeframes: list = None, days: int = 730) -> dict:
    if timeframes is None:
        timeframes = ["4h", "1d"]
    result = {}
    for tf in timeframes:
        df = fetch_ohlcv(symbol, tf, days=days)
        if not df.empty:
            result[tf] = df
    return result
