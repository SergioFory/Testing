"""
Setup #2 — Pullback en tendencia (la estrategia más clásica y efectiva).

Lógica:
  LONG:  Tendencia alcista confirmada (EMA_fast > EMA_slow) + pullback hasta
         zona EMA_fast + RSI sobreventa relativa + barra de reversión

  SHORT: Tendencia bajista confirmada (EMA_fast < EMA_slow) + rebote hasta
         zona EMA_fast + RSI sobrecompra relativa + barra de reversión

La tendencia se filtra en timeframe DIARIO.
El setup se detecta en timeframe 4H.
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger

from config.settings import PULLBACK_PARAMS


def add_trend_filter(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el contexto de tendencia en timeframe diario.
    Retorna df_daily con columna 'trend' (+1 alcista, -1 bajista, 0 lateral).
    """
    df = df_daily.copy()
    ema_fast = df["close"].ewm(span=PULLBACK_PARAMS["ema_fast"]).mean()
    ema_slow = df["close"].ewm(span=PULLBACK_PARAMS["ema_slow"]).mean()
    df["trend_ema_fast"] = ema_fast
    df["trend_ema_slow"] = ema_slow

    # Mínimo N barras en la misma dirección para confirmar tendencia
    min_bars = PULLBACK_PARAMS["min_trend_bars"]
    bull = (ema_fast > ema_slow)
    bear = (ema_fast < ema_slow)

    bull_streak = bull.rolling(min_bars).sum() == min_bars
    bear_streak = bear.rolling(min_bars).sum() == min_bars

    df["trend"] = np.where(bull_streak, 1, np.where(bear_streak, -1, 0))
    return df


def detect_pullbacks(df_4h: pd.DataFrame,
                     df_daily: pd.DataFrame = None,
                     params: dict = None) -> pd.DataFrame:
    """
    Detecta pullbacks en 4H usando el filtro de tendencia del daily.

    Args:
        df_4h:   OHLCV 4H
        df_daily: OHLCV diario (para contexto de tendencia). Si None, usa df_4h.
        params:  Override de PULLBACK_PARAMS

    Returns:
        df_4h con columnas adicionales de setup.
    """
    p = {**PULLBACK_PARAMS, **(params or {})}
    df = df_4h.copy()

    # --- EMAs en 4H ---
    df["ema_fast"] = df["close"].ewm(span=p["ema_fast"]).mean()
    df["ema_slow"] = df["close"].ewm(span=p["ema_slow"]).mean()

    # --- ATR ---
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr"] = atr

    # --- RSI ---
    df["rsi"] = ta.momentum.RSIIndicator(
        df["close"], window=p["rsi_period"]
    ).rsi()

    # --- Distancia a EMA fast (zona de pullback) ---
    df["dist_to_ema_fast_pct"] = (df["close"] - df["ema_fast"]) / df["ema_fast"]

    # --- Contexto de tendencia desde daily ---
    if df_daily is not None and not df_daily.empty:
        df_d = add_trend_filter(df_daily)
        # Propagar tendencia diaria a 4H (forward fill)
        trend_daily = df_d["trend"].copy()
        # Reindex daily trend to 4H timestamps (use date matching)
        df_dates    = [d.date() for d in df.index]
        daily_dates = [d.date() for d in trend_daily.index]
        daily_map   = dict(zip(daily_dates, trend_daily.values))
        df["trend_context"] = [daily_map.get(d, 0) for d in df_dates]
    else:
        # Si no hay daily, calcular tendencia en el mismo timeframe
        ema_f = df["close"].ewm(span=p["ema_fast"]).mean()
        ema_s = df["close"].ewm(span=p["ema_slow"]).mean()
        min_bars = p["min_trend_bars"]
        bull = (ema_f > ema_s).rolling(min_bars).sum() == min_bars
        bear = (ema_f < ema_s).rolling(min_bars).sum() == min_bars
        df["trend_context"] = np.where(bull, 1, np.where(bear, -1, 0))

    # --- Detección de pullback LONG ---
    # Tendencia alcista + precio en zona de EMA fast + RSI no sobrecomprado
    pullback_zone_long = (
        df["dist_to_ema_fast_pct"].between(
            -p["max_pullback_pct"], 0.005
        )
    )
    cond_long = (
        (df["trend_context"] == 1) &
        pullback_zone_long &
        (df["rsi"] <= p["rsi_oversold"]) &
        (df["close"] > df["ema_slow"])    # No romper soporte EMA slow
    )

    # --- Detección de pullback SHORT ---
    pullback_zone_short = (
        df["dist_to_ema_fast_pct"].between(
            -0.005, p["max_pullback_pct"]
        )
    )
    cond_short = (
        (df["trend_context"] == -1) &
        pullback_zone_short &
        (df["rsi"] >= p["rsi_overbought"]) &
        (df["close"] < df["ema_slow"])
    )

    # --- Barra de reversión (vela confirmada) ---
    # LONG: vela alcista (close > open) después de pullback
    # SHORT: vela bajista (close < open) después de rebote
    reversal_bar_long  = df["close"] > df["open"]
    reversal_bar_short = df["close"] < df["open"]

    cond_long  = cond_long  & reversal_bar_long
    cond_short = cond_short & reversal_bar_short

    # --- Score ---
    rsi_score_long  = ((p["rsi_oversold"]   - df["rsi"]).clip(0, 20) / 20)
    rsi_score_short = ((df["rsi"] - p["rsi_overbought"]).clip(0, 20) / 20)
    dist_score_long  = (1 - abs(df["dist_to_ema_fast_pct"]) / p["max_pullback_pct"]).clip(0, 1)
    dist_score_short = dist_score_long.copy()

    df["pullback_long"]  = cond_long
    df["pullback_short"] = cond_short
    df["pullback_dir"]   = np.where(
        cond_long, "long", np.where(cond_short, "short", None)
    )
    df["pullback_score"] = np.where(
        cond_long,
        (rsi_score_long * 0.5 + dist_score_long * 0.5).clip(0, 1),
        np.where(
            cond_short,
            (rsi_score_short * 0.5 + dist_score_short * 0.5).clip(0, 1),
            0.0,
        ),
    )

    n_long  = int(cond_long.sum())
    n_short = int(cond_short.sum())
    logger.debug(f"Pullbacks detectados — LONG: {n_long} | SHORT: {n_short}")
    return df


def get_setup_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pullback_long"] | df["pullback_short"]].copy()
