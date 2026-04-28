"""
Descarga datos del Oro y activos macro via yfinance.
GC=F (Gold Futures CME) como fuente principal.

Nota sobre timeframes:
  yfinance no tiene intervalo 4H nativo y limita datos intradiarios.
  Para Gold, se usan datos DIARIOS como timeframe primario (apropiado
  porque el oro opera en sesiones, no 24/7 como crypto, y los setups
  técnicos funcionan mejor en diario para commodities).
"""
import time
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
    Descarga OHLCV del Oro desde GC=F (futuros CME).
    Para Gold, siempre usa datos diarios (más fiables y con más historia).
    El timeframe "4h" devuelve los mismos datos diarios (proxy).

    Returns:
        DataFrame OHLCV con index datetime UTC.
    """
    if not HAS_YF:
        logger.warning("yfinance no instalado. pip install yfinance")
        return pd.DataFrame()

    end   = datetime.now()
    start = end - timedelta(days=days)
    return _fetch_gold_daily(start, end)


def _fetch_gold_daily(start, end) -> pd.DataFrame:
    """Descarga datos diarios de GC=F con reintentos."""
    candidatos = [("GC=F", "Gold Futures CME"), ("GLD", "SPDR Gold ETF")]

    for ticker, desc in candidatos:
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    interval="1d",
                    progress=False, auto_adjust=True,
                )
                if raw.empty:
                    break
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                df = df.dropna()

                if df.empty:
                    break

                last_close = float(df.iloc[-1]["close"])
                # GLD cotiza ~1/10 del precio del oro → multiplicar
                if ticker == "GLD":
                    df[["open", "high", "low", "close"]] *= 10

                last_close_adj = float(df.iloc[-1]["close"])
                if last_close_adj < 500:
                    logger.warning(f"{ticker} precio inesperado (${last_close_adj:.0f}), descartado")
                    break

                logger.success(
                    f"Oro cargado desde {ticker} ({desc}) | "
                    f"${last_close_adj:,.0f}/oz | {len(df)} barras"
                )
                return df

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"{ticker}: {e}")

    logger.error("No se pudo obtener datos del Oro")
    return pd.DataFrame()


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
