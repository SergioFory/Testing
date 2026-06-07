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
    candidatos = [
        ("GC=F", 1.0,  "Oro cargado desde GC=F (Gold Futures CME)"),
        ("GLD",  10.0, "Oro cargado desde GLD (SPDR Gold ETF)"),
    ]
    return _download_yfinance_daily(candidatos, start, end, min_price=500)


def fetch_yfinance_asset(symbol: str, days: int = 730) -> pd.DataFrame:
    """
    Descarga OHLCV de cualquier activo configurado con source='yfinance' en ASSETS.
    Lee yf_ticker (primario), yf_fallback (ticker, multiplicador) y yf_min_price.
    """
    from config.settings import ASSETS

    if not HAS_YF:
        logger.warning("yfinance no instalado. pip install yfinance")
        return pd.DataFrame()

    cfg       = ASSETS.get(symbol, {})
    primary   = cfg.get("yf_ticker")
    fallback  = cfg.get("yf_fallback")     # tuple (ticker, multiplier) opcional
    label     = cfg.get("label", symbol)
    min_price = cfg.get("yf_min_price", 0)

    if not primary:
        logger.error(f"No yf_ticker configurado para {symbol}")
        return pd.DataFrame()

    candidatos = [(primary, 1.0, f"{label} cargado desde {primary}")]
    if fallback:
        fb_ticker, fb_mult = fallback
        candidatos.append(
            (fb_ticker, fb_mult, f"{label} cargado desde {fb_ticker} (fallback)")
        )

    end   = datetime.now()
    start = end - timedelta(days=days)
    return _download_yfinance_daily(candidatos, start, end, min_price)


def _download_yfinance_daily(candidatos, start, end, min_price: float = 0) -> pd.DataFrame:
    """
    Descarga datos diarios de yfinance con reintentos y fallback entre tickers.

    Args:
        candidatos: lista de tuplas (ticker, multiplier, descripcion_log)
        start, end: rango de fechas
        min_price:  precio mínimo válido para validar el ticker (0 = sin validación)
    """
    for ticker, multiplier, desc in candidatos:
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
                # Forex (tickers =X) no reporta volumen → llega 0/NaN. Se rellena
                # con 0 para no eliminar todas las barras en el dropna siguiente.
                df["volume"] = df["volume"].fillna(0)
                df = df.dropna(subset=["open", "high", "low", "close"])

                if df.empty:
                    break

                if multiplier != 1.0:
                    df[["open", "high", "low", "close"]] *= multiplier

                last_close = float(df.iloc[-1]["close"])
                if min_price and last_close < min_price:
                    logger.warning(f"{ticker} precio inesperado (${last_close:.2f}), descartado")
                    break

                logger.success(f"{desc} | ${last_close:,.2f} | {len(df)} barras")
                return df

            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.warning(f"{ticker}: {e}")

    logger.error("No se pudo obtener datos yfinance para los candidatos")
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
