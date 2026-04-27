"""
Descarga datos del Oro y activos macro via yfinance.
GC=F (Gold Futures CME) como fuente principal.
"""
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


def fetch_gold(timeframe: str = "1d", days: int = 730) -> pd.DataFrame:
    """
    Descarga OHLCV del Oro. Prioriza GC=F (futuros CME).
    Valida que el precio sea > $1500/oz.

    Args:
        timeframe: "1h", "4h" (usa yf interval), "1d"
        days: Historia hacia atrás

    Returns:
        DataFrame OHLCV con index datetime UTC.
    """
    if not HAS_YF:
        logger.warning("yfinance no instalado. pip install yfinance")
        return pd.DataFrame()

    end = datetime.now()

    # yfinance limita datos intradiarios a ~730 días; para 4h usamos
    # 1h con ventana corta y resampleamos. Si falla, usamos daily como proxy.
    if timeframe == "4h":
        df = _fetch_gold_4h(end, days)
        return df

    # Para 1d: descarga directa
    start = end - timedelta(days=days)
    return _fetch_gold_raw("1d", start, end)


def _fetch_gold_raw(yf_interval: str, start, end) -> pd.DataFrame:
    """Descarga OHLCV de GC=F (o XAUUSD=X como fallback)."""
    candidatos = [("GC=F", "Gold Futures CME"), ("XAUUSD=X", "Spot Gold FX")]
    for ticker, desc in candidatos:
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                interval=yf_interval,
                progress=False, auto_adjust=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.dropna()

            if df.empty:
                continue
            last_close = float(df.iloc[-1]["close"])
            if last_close < 1500:
                logger.warning(f"{ticker} precio inesperado (${last_close:.0f}), descartado")
                continue

            logger.success(f"Oro cargado desde {ticker} | "
                           f"${last_close:,.0f}/oz | {len(df)} barras")
            return df
        except Exception as e:
            logger.warning(f"Error con {ticker}: {e}")
            continue
    logger.error("No se pudo obtener datos del Oro")
    return pd.DataFrame()


def _fetch_gold_4h(end: datetime, days: int) -> pd.DataFrame:
    """
    Intenta obtener datos 4H del Oro.
    yfinance no tiene intervalo 4h nativo — usa 1h (max ~60 días) y resamplea.
    Si el resultado tiene < 100 barras, hace fallback a datos diarios como proxy.
    """
    # Intentar 1h para los últimos 59 días (límite seguro de yfinance)
    window_days = min(days, 59)
    start_1h = end - timedelta(days=window_days)
    df_1h = _fetch_gold_raw("1h", start_1h, end)

    if not df_1h.empty:
        df_4h = df_1h.resample("4h").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()
        if len(df_4h) >= 100:
            logger.info(f"Gold 4H: {len(df_4h)} barras (desde 1H resampled, últimos {window_days}d)")
            return df_4h

    # Fallback: datos diarios como proxy de 4H
    # (1 barra diaria ≈ comportamiento de sesión completa, válido para Gold)
    logger.info("Gold 4H: usando datos diarios como proxy (yfinance no provee 4H histórico)")
    start_daily = end - timedelta(days=days)
    return _fetch_gold_raw("1d", start_daily, end)


def fetch_macro(days: int = 730) -> pd.DataFrame:
    """
    Retorna retornos diarios de activos macro.
    Columnas: sp500_ret, nasdaq_ret, gold_ret, dxy_ret (+ versiones 3d)
    Index: datetime UTC.
    """
    if not HAS_YF:
        return pd.DataFrame()

    tickers = {
        "SPY": "sp500",
        "QQQ": "nasdaq",
        "GLD": "gold_etf",
        "UUP": "dxy",
    }
    end, start = datetime.now(), datetime.now() - timedelta(days=days)
    frames = {}

    for ticker, name in tickers.items():
        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].squeeze()
            if len(close) < 5:
                continue
            frames[f"{name}_ret"]  = close.pct_change()
            frames[f"{name}_ret3"] = close.pct_change(3)
        except Exception as e:
            logger.warning(f"Macro {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")

    logger.success(f"Macro: {len(result.columns)} series, {len(result)} filas")
    return result
