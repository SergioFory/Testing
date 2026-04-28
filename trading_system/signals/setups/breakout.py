"""
Setup #1 — Breakout de Canal Donchian con confirmación de volumen.

Lógica:
  LONG:  Cierre supera el máximo de los últimos N períodos + volumen > promedio × factor
  SHORT: Cierre rompe el mínimo de los últimos N períodos + volumen > promedio × factor

Condiciones adicionales:
  - Barra de breakout debe ser >= 1 ATR de tamaño (no "breakout falso" de 1 tick)
  - No operar si ATR está en percentil bajo (mercado dormido)
  - Señal se marca en la barra que CIERRA por encima/debajo del canal
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger

from config.settings import BREAKOUT_PARAMS


def detect_breakouts(df: pd.DataFrame,
                     params: dict = None) -> pd.DataFrame:
    """
    Detecta setups de breakout en el DataFrame OHLCV.

    Args:
        df:     OHLCV con columnas open/high/low/close/volume
        params: Override de BREAKOUT_PARAMS

    Returns:
        DataFrame con columnas adicionales:
        - breakout_long  (bool): setup alcista detectado en esta barra
        - breakout_short (bool): setup bajista detectado en esta barra
        - breakout_dir   (str):  "long" | "short" | None
        - breakout_score (float): fuerza del breakout (0-1)
    """
    p = {**BREAKOUT_PARAMS, **(params or {})}
    df = df.copy()
    period = p["donchian_period"]

    # --- Donchian Channel ---
    df["dc_upper"] = df["high"].rolling(period).max().shift(1)  # shift: no look-ahead
    df["dc_lower"] = df["low"].rolling(period).min().shift(1)

    # --- ATR ---
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"]

    # --- Volumen ---
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma"] + 1e-9)

    # --- Tamaño de la barra de breakout ---
    df["bar_size"] = (df["high"] - df["low"]) / (df["atr"] + 1e-9)

    # --- ATR percentil (filtro: no operar en mercado dormido) ---
    df["atr_pct_rank"] = df["atr_pct"].rolling(100).rank(pct=True)

    # --- ADX (opcional: filtro de tendencia para evitar fakeouts en ranging) ---
    adx_min = p.get("adx_min_trend", 0)
    if adx_min > 0:
        df["_adx"] = ta.trend.ADXIndicator(
            df["high"], df["low"], df["close"], window=14
        ).adx()
        adx_filter = df["_adx"] >= adx_min
    else:
        adx_filter = pd.Series(True, index=df.index)

    # --- Condiciones LONG ---
    cond_long = (
        (df["close"] > df["dc_upper"]) &                # Breakout al alza
        (df["vol_ratio"] >= p["vol_factor"]) &           # Volumen confirmado
        (df["bar_size"]  >= p["atr_min_move"]) &         # Barra grande
        (df["atr_pct_rank"] >= 0.30) &                   # ATR no en zona muerta
        adx_filter                                        # Mercado en tendencia
    )

    # --- Condiciones SHORT ---
    cond_short = (
        (df["close"] < df["dc_lower"]) &
        (df["vol_ratio"] >= p["vol_factor"]) &
        (df["bar_size"]  >= p["atr_min_move"]) &
        (df["atr_pct_rank"] >= 0.30) &
        adx_filter
    )

    # --- Score de fuerza (0-1) ---
    # Combina: exceso sobre el canal, ratio de volumen y tamaño de barra
    excess_long  = ((df["close"] - df["dc_upper"]) / (df["atr"] + 1e-9)).clip(0, 3)
    excess_short = ((df["dc_lower"] - df["close"]) / (df["atr"] + 1e-9)).clip(0, 3)
    vol_score    = (df["vol_ratio"] - 1).clip(0, 3) / 3
    bar_score    = (df["bar_size"]  - 1).clip(0, 3) / 3

    df["breakout_long"]  = cond_long
    df["breakout_short"] = cond_short

    # Score final: ponderado
    df["breakout_score"] = np.where(
        cond_long,
        (excess_long / 3 * 0.4 + vol_score * 0.3 + bar_score * 0.3).clip(0, 1),
        np.where(
            cond_short,
            (excess_short / 3 * 0.4 + vol_score * 0.3 + bar_score * 0.3).clip(0, 1),
            0.0,
        ),
    )
    df["breakout_dir"] = np.where(
        cond_long,  "long",
        np.where(cond_short, "short", None)
    )

    n_long  = int(cond_long.sum())
    n_short = int(cond_short.sum())
    logger.debug(f"Breakouts detectados — LONG: {n_long} | SHORT: {n_short}")

    return df


def get_setup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra solo las filas con setup activo."""
    return df[df["breakout_long"] | df["breakout_short"]].copy()
