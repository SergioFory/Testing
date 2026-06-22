"""
Fetcher de datos OHLCV vía Twelve Data REST API.
Soporta granularidades 4h y 1day para pares forex.

Free tier: 800 créditos/día, 8 req/min.
Cada llamada a /time_series = 1 crédito.
"""
import os
import time
import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://api.twelvedata.com"

# Mapeo símbolos internos → formato Twelve Data
SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
}


def fetch_twelvedata_ohlcv(symbol: str, interval: str = "4h",
                           outputsize: int = 5000) -> pd.DataFrame:
    """
    Descarga velas históricas de Twelve Data para un par forex.

    Args:
        symbol:     Símbolo interno (EURUSD, GBPUSD, USDJPY)
        interval:   Granularidad (4h, 1day, 1h …)
        outputsize: Barras a descargar (max 5000 en free tier).
                    5000 barras × 4h ≈ 3.2 años de datos forex (sin fines de semana).

    Returns:
        DataFrame OHLCV con index datetime UTC, o DataFrame vacío si falla.
    """
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        logger.error("TWELVEDATA_API_KEY no configurada. Añádela en Render → Environment.")
        return pd.DataFrame()

    td_symbol = SYMBOL_MAP.get(symbol)
    if not td_symbol:
        logger.error(f"Símbolo {symbol} no mapeado para Twelve Data.")
        return pd.DataFrame()

    params = {
        "symbol":     td_symbol,
        "interval":   interval,
        "outputsize": min(outputsize, 5000),
        "order":      "ASC",
        "timezone":   "UTC",
        "apikey":     api_key,
    }

    for attempt in range(3):
        try:
            resp = requests.get(f"{BASE_URL}/time_series", params=params, timeout=30)
            data = resp.json()

            if data.get("status") == "error":
                code = data.get("code", 0)
                msg  = data.get("message", "")
                if code in (401, 403):
                    logger.error(f"Twelve Data: API key inválida — {msg}")
                    return pd.DataFrame()
                if code == 429:
                    wait = 15 * (attempt + 1)
                    logger.warning(f"Twelve Data: rate limit, reintentando en {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f"Twelve Data error {code}: {msg}")
                return pd.DataFrame()

            values = data.get("values", [])
            if not values:
                logger.warning(f"Twelve Data: sin datos para {td_symbol} {interval}")
                return pd.DataFrame()

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.set_index("datetime").sort_index()

            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)

            # Forex puede devolver volume=0 en Twelve Data free tier — se rellena
            # con 0 igual que yfinance; los detectores ya tienen vol_factor=0.0
            # para forex y no dependen del volumen.
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            else:
                df["volume"] = 0.0

            df = df[["open", "high", "low", "close", "volume"]]
            df = df[~df.index.duplicated(keep="last")]

            logger.success(
                f"{td_symbol} cargado desde Twelve Data ({interval}) | "
                f"${df['close'].iloc[-1]:.5f} | {len(df)} barras | "
                f"{df.index[0].date()} → {df.index[-1].date()}"
            )
            return df

        except Exception as exc:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Twelve Data {td_symbol}: fallo tras 3 intentos — {exc}")

    return pd.DataFrame()
