"""
Descarga datos de mercado: OHLCV, funding rates.

Jerarquía de fuentes (se prueban en orden hasta obtener datos):
  1. Binance Futures (ccxt)  — bloquea IPs de EE.UU. (HTTP 451)
  2. Bybit (ccxt)            — CloudFront bloquea IPs AWS US (HTTP 403)
  3. Gate.io (ccxt)          — futuros perpetuos, sin restricción geográfica conocida
  4. yfinance                — precios spot (BTC-USD/ETH-USD), último recurso
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

_BYBIT_SYMBOL = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
}
_GATEIO_SYMBOL = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
}
_YFINANCE_SYMBOL = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}


def _binance_exchange(private: bool = False):
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    config = {"options": {"defaultType": "future"}, "enableRateLimit": True}
    if private:
        from config.settings import BINANCE_API_KEY, BINANCE_API_SECRET
        if BINANCE_API_KEY and BINANCE_API_SECRET:
            config["apiKey"] = BINANCE_API_KEY
            config["secret"] = BINANCE_API_SECRET
    return ccxt.binance(config)


def _bybit_exchange():
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    return ccxt.bybit({"options": {"defaultType": "linear"}, "enableRateLimit": True})


def _gateio_exchange():
    if not HAS_CCXT:
        raise ImportError("pip install ccxt")
    return ccxt.gate({"options": {"defaultType": "swap"}, "enableRateLimit": True})


def _fetch_ohlcv_from(exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Descarga OHLCV de cualquier exchange ccxt."""
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_candles = []
    limit = 1000
    prev_last_ts = None

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        except Exception as e:
            logger.error(f"Error descargando {symbol} de {exchange.id}: {e}")
            time.sleep(2)
            return pd.DataFrame()

        if not candles:
            break
        last_ts = candles[-1][0]
        # Guard: si el exchange no avanza en el tiempo, salir
        if last_ts == prev_last_ts:
            break
        prev_last_ts = last_ts
        all_candles.extend(candles)
        # Terminar cuando se alcanza el tiempo actual (en lugar de contar candles)
        if last_ts >= now_ms:
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


def _fetch_ohlcv_yfinance(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fallback final: precios spot via yfinance (BTC-USD / ETH-USD)."""
    try:
        import yfinance as yf
    except ImportError:
        return pd.DataFrame()

    ticker = _YFINANCE_SYMBOL.get(symbol)
    if not ticker:
        return pd.DataFrame()

    # yfinance 1h solo va 730 días atrás; 1d va a max
    yf_interval = "1h" if timeframe in ("1h", "4h") else "1d"
    period = f"{min(days, 729)}d" if yf_interval == "1h" else "max"

    try:
        raw = yf.download(ticker, period=period, interval=yf_interval,
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return pd.DataFrame()

        df = raw.copy()
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                      for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        if timeframe == "4h" and yf_interval == "1h":
            df = df.resample("4h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()

        return df
    except Exception as e:
        logger.error(f"yfinance falló para {symbol}: {e}")
        return pd.DataFrame()


def fetch_ohlcv(symbol: str, timeframe: str = "4h", days: int = 730) -> pd.DataFrame:
    """
    Descarga OHLCV con fallback automático:
      Binance → Bybit → Gate.io → yfinance
    """
    if not HAS_CCXT:
        return pd.DataFrame()

    logger.info(f"Descargando {symbol} {timeframe} ({days} días)...")

    # 1. Binance (sin API keys — OHLCV es público; keys activan endpoints privados que dan 451)
    try:
        df = _fetch_ohlcv_from(_binance_exchange(private=False), symbol, timeframe, days)
        if not df.empty:
            logger.success(f"{symbol} {timeframe} vía Binance: {len(df)} barras | "
                           f"{df.index[0].date()} → {df.index[-1].date()}")
            return df
    except Exception as e:
        logger.warning(f"Binance no disponible para {symbol}: {e}")

    # 2. Bybit
    bybit_sym = _BYBIT_SYMBOL.get(symbol)
    if bybit_sym:
        logger.info(f"Intentando Bybit para {symbol}...")
        try:
            df = _fetch_ohlcv_from(_bybit_exchange(), bybit_sym, timeframe, days)
            if not df.empty:
                logger.success(f"{symbol} {timeframe} vía Bybit: {len(df)} barras | "
                               f"{df.index[0].date()} → {df.index[-1].date()}")
                return df
        except Exception as e:
            logger.warning(f"Bybit no disponible para {symbol}: {e}")

    # 3. Gate.io (máx 10000 puntos; limitar days según timeframe)
    gateio_sym = _GATEIO_SYMBOL.get(symbol)
    if gateio_sym:
        _gateio_max = {"1h": 400, "4h": 1650, "1d": 9000}
        gateio_days = min(days, _gateio_max.get(timeframe, 1000))
        logger.info(f"Intentando Gate.io para {symbol}...")
        try:
            df = _fetch_ohlcv_from(_gateio_exchange(), gateio_sym, timeframe, gateio_days)
            if not df.empty:
                logger.success(f"{symbol} {timeframe} vía Gate.io: {len(df)} barras | "
                               f"{df.index[0].date()} → {df.index[-1].date()}")
                return df
        except Exception as e:
            logger.warning(f"Gate.io no disponible para {symbol}: {e}")

    # 4. yfinance (precios spot, último recurso)
    logger.info(f"Intentando yfinance (spot) para {symbol}...")
    df = _fetch_ohlcv_yfinance(symbol, timeframe, days)
    if not df.empty:
        logger.success(f"{symbol} {timeframe} vía yfinance: {len(df)} barras | "
                       f"{df.index[0].date()} → {df.index[-1].date()}")
        return df

    logger.warning(f"Sin datos para {symbol} {timeframe} (todas las fuentes fallaron)")
    return pd.DataFrame()


def fetch_funding_rates(symbol: str, days: int = 365) -> pd.Series:
    """
    Tasa de financiamiento diaria promedio.
    Intenta Binance → Bybit → Gate.io.
    """
    if not HAS_CCXT:
        return pd.Series(dtype=float)

    def _fetch_rates(ex, sym):
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        all_rates = []
        while True:
            chunk = ex.fetch_funding_rate_history(sym, since=since_ms, limit=1000)
            if not chunk:
                break
            all_rates.extend(chunk)
            since_ms = chunk[-1]["timestamp"] + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.5)
        return all_rates

    def _rates_to_series(all_rates):
        df = pd.DataFrame(all_rates)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["rate"] = df["fundingRate"].astype(float)
        df.set_index("ts", inplace=True)
        return df["rate"].resample("1D").mean()

    # Binance
    try:
        rates = _fetch_rates(_binance_exchange(private=True), symbol)
        if rates:
            return _rates_to_series(rates)
    except Exception as e:
        logger.warning(f"Funding rates Binance no disponibles para {symbol}: {e}")

    # Bybit
    bybit_sym = _BYBIT_SYMBOL.get(symbol)
    if bybit_sym:
        try:
            rates = _fetch_rates(_bybit_exchange(), bybit_sym)
            if rates:
                series = _rates_to_series(rates)
                logger.success(f"Funding rates {symbol} vía Bybit: {len(series)} días")
                return series
        except Exception as e:
            logger.warning(f"Funding rates Bybit no disponibles para {symbol}: {e}")

    # Gate.io
    gateio_sym = _GATEIO_SYMBOL.get(symbol)
    if gateio_sym:
        try:
            rates = _fetch_rates(_gateio_exchange(), gateio_sym)
            if rates:
                series = _rates_to_series(rates)
                logger.success(f"Funding rates {symbol} vía Gate.io: {len(series)} días")
                return series
        except Exception as e:
            logger.warning(f"Funding rates Gate.io no disponibles para {symbol}: {e}")

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
