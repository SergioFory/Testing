"""
Descarga datos de mercado: OHLCV, funding rates.

OHLCV (Fase 0, 2026-11): ÚNICA fuente = Gate.io perpetuos (ccxt).
  - Es la fuente con la que se validó el sistema y la única que responde desde
    Render (Binance y Bybit bloquean la región).
  - Antes había una cadena Binance → Bybit → Gate.io → yfinance spot. Si fallaba
    una sola página de Gate.io se descartaba la serie y se caía EN SILENCIO a
    yfinance spot (2 años, otro volumen), lo que añadía entre un 27 y un 63 % de
    setups no validados. Ahora cada página se reintenta con espera y, si Gate.io
    sigue fallando, se devuelve un DataFrame vacío: el ciclo se salta y main.py
    avisa por Telegram.
  - La fuente usada queda en df.attrs["source"].

Funding rates: se mantiene la cadena Binance → Bybit → Gate.io (solo se usa en
el entrenamiento ML, que está apagado).
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
    # Fase 1: cargar SOLO los mercados de perpetuos (swap). Por defecto ccxt carga
    # también spot, futuros y opciones (6.746 mercados) en cada instancia: medido
    # en el sandbox, 17,6 s y ~70 MB de Python por carga, dos veces por activo y
    # ciclo. Solo swap: 1.052 mercados, 3,4 s y ~30 MB. Las velas son las mismas.
    return ccxt.gate({"options": {"defaultType": "swap",
                                  "fetchMarkets": {"types": ["swap"]}},
                      "enableRateLimit": True})


def _fetch_ohlcv_from(exchange, symbol: str, timeframe: str, days: int,
                      page_retries: int = 3) -> pd.DataFrame:
    """
    Descarga OHLCV paginado de un exchange ccxt.

    Cada página se reintenta hasta `page_retries` veces con espera exponencial
    (2 s, 4 s, ...). Si una página sigue fallando se devuelve un DataFrame vacío:
    una serie con huecos es peor que saltarse el ciclo.
    """
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_candles = []
    limit = 1000
    prev_last_ts = None
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    while True:
        candles = None
        for attempt in range(1, page_retries + 1):
            try:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
                break
            except Exception as e:
                logger.warning(
                    f"Error descargando {symbol} {timeframe} de {exchange.id} "
                    f"(página desde {since_ms}, intento {attempt}/{page_retries}): {e}"
                )
                if attempt < page_retries:
                    time.sleep(2 ** attempt)
        if candles is None:
            logger.error(f"{exchange.id}: {symbol} {timeframe} sin datos tras {page_retries} intentos.")
            return pd.DataFrame()

        if not candles:
            break
        last_ts = candles[-1][0]
        # Última página (llega hasta la vela en curso) o el exchange no avanza en el
        # tiempo: se usa entera y se sale. (Gate.io devuelve 999 velas por página
        # aunque se pidan 1000: no se puede usar len(candles) < limit.)
        if last_ts == prev_last_ts or len(candles) == 1 or last_ts + tf_ms > now_ms:
            all_candles.extend(candles)
            break
        prev_last_ts = last_ts
        # La ÚLTIMA vela de una página intermedia NO se usa: se vuelve a pedir como
        # primera de la página siguiente. Medido en Gate.io (2026-11-15): la vela que
        # cae justo en el final de la ventana llega PLANA (O=H=L=C) y con volumen 0;
        # pedida dentro de otra página llega con sus valores reales. Antes quedaban
        # ~8 velas falsas por serie 4H de ~10.000 en validate/check-live, en
        # posiciones que dependían del "ahora" (por eso el nº de aprobados de un
        # mismo histórico podía variar ±1 de un día a otro). Igual que
        # backfill_history.iter_closed_pages.
        all_candles.extend(candles[:-1])
        since_ms = last_ts
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


# Gate.io sirve como máximo ~10.000 velas por serie: limitar los días pedidos.
_GATEIO_MAX_DAYS = {"1h": 400, "4h": 1650, "1d": 9000}


def fetch_ohlcv(symbol: str, timeframe: str = "4h", days: int = 730) -> pd.DataFrame:
    """
    Descarga OHLCV de Gate.io perpetuos (única fuente; ver docstring del módulo).

    Returns:
        DataFrame OHLCV (índice = hora de APERTURA de la vela, UTC) con
        df.attrs["source"] = "gateio", o DataFrame vacío si Gate.io falla.
        Incluye la vela en formación: quien detecte setups debe descartarla
        con data.bars.drop_unclosed_bars.
    """
    if not HAS_CCXT:
        return pd.DataFrame()

    gateio_sym = _GATEIO_SYMBOL.get(symbol)
    if not gateio_sym:
        logger.error(f"{symbol} no tiene símbolo de Gate.io configurado.")
        return pd.DataFrame()

    gateio_days = min(days, _GATEIO_MAX_DAYS.get(timeframe, 1000))
    logger.info(f"Descargando {symbol} {timeframe} de Gate.io ({gateio_days} días)...")
    try:
        df = _fetch_ohlcv_from(_gateio_exchange(), gateio_sym, timeframe, gateio_days)
    except Exception as e:
        logger.error(f"Gate.io no disponible para {symbol}: {e}")
        return pd.DataFrame()

    if df.empty:
        logger.error(f"Sin datos de Gate.io para {symbol} {timeframe}.")
        return pd.DataFrame()

    df.attrs["source"] = "gateio"
    logger.success(f"{symbol} {timeframe} vía Gate.io: {len(df)} barras | "
                   f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


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
