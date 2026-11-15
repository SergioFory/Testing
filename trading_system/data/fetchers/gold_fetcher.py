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
        # Sin respaldo GLD×10 (Fase 0): cotiza ~8.6 % por debajo de GC=F.
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
        if fb_mult != 1.0:
            # Respaldos ESCALADOS (GLD×10, SPY×10...) prohibidos (Fase 0): cotizan
            # con desfase respecto al subyacente (GLD×10 ≈ −8.6 % frente a GC=F) y
            # sus niveles no son operables. Mejor saltarse el ciclo y avisar.
            logger.warning(
                f"{symbol}: se ignora el respaldo escalado {fb_ticker}×{fb_mult} "
                "(no se emiten ni resuelven señales con precios sustitutos)."
            )
        else:
            candidatos.append(
                (fb_ticker, fb_mult, f"{label} cargado desde {fb_ticker} (fallback)")
            )

    end   = datetime.now()
    start = end - timedelta(days=days)
    return _download_yfinance_daily(candidatos, start, end, min_price)


# Esperas entre reintentos (s). yfinance 1.7.0 NO lanza excepción ante un 429
# (límite de peticiones de Yahoo): devuelve una tabla VACÍA. Antes el código hacía
# `if raw.empty: break` y abandonaba el ticker al primer intento, así que los
# reintentos solo servían ante excepciones. Ahora una respuesta vacía también se
# reintenta con espera.
_YF_RETRY_WAITS = (5, 15, 30)


def _download_yfinance_daily(candidatos, start, end, min_price: float = 0) -> pd.DataFrame:
    """
    Descarga datos diarios de yfinance con reintentos y fallback entre tickers.

    Args:
        candidatos: lista de tuplas (ticker, multiplier, descripcion_log)
        start, end: rango de fechas
        min_price:  precio mínimo válido para validar el ticker (0 = sin validación)

    Returns:
        DataFrame OHLCV con df.attrs["source"] = ticker usado, o vacío si todos
        los candidatos fallan tras los reintentos.
    """
    n_attempts = len(_YF_RETRY_WAITS) + 1
    for ticker, multiplier, desc in candidatos:
        for attempt in range(n_attempts):
            motivo = None
            try:
                raw = yf.download(
                    ticker, start=start, end=end,
                    interval="1d",
                    progress=False, auto_adjust=True,
                )
                if raw is None or raw.empty:
                    motivo = "respuesta vacía (posible 429 de Yahoo)"
                else:
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
                        motivo = "sin velas válidas"
                    else:
                        if multiplier != 1.0:
                            df[["open", "high", "low", "close"]] *= multiplier

                        last_close = float(df.iloc[-1]["close"])
                        if min_price and last_close < min_price:
                            # Precio imposible: no es un fallo transitorio, no reintentar.
                            logger.warning(f"{ticker} precio inesperado (${last_close:.2f}), descartado")
                            break

                        df.attrs["source"] = ticker
                        logger.success(f"{desc} | ${last_close:,.2f} | {len(df)} barras")
                        return df
            except Exception as e:
                motivo = f"error: {e}"

            if attempt < n_attempts - 1:
                wait = _YF_RETRY_WAITS[attempt]
                logger.warning(f"{ticker}: {motivo}. Reintento {attempt + 1}/{n_attempts - 1} en {wait}s.")
                time.sleep(wait)
            else:
                logger.warning(f"{ticker}: {motivo}. Sin más reintentos.")

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
        # --- Cross-asset macro ampliado ---
        # Drivers fundamentales de forex y oro: rendimientos de bonos, petróleo
        # y volatilidad. Se integran al ML automáticamente (features.py los toma
        # como macro_<name>_ret / _trend30 / _accel).
        "TLT": "bonds",     # Bonos 20Y EE.UU.: proxy de rendimientos largos (inverso al precio)
        "USO": "oil",       # Petróleo WTI: inflación y riesgo geopolítico
        "^VIX": "vix",      # Índice de volatilidad: régimen risk-on / risk-off
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
            # Para el VIX el NIVEL (15 vs 35) es la señal de régimen, no su retorno.
            if name == "vix":
                frames["vix_level"] = close
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
