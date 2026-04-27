"""
Setup #3 — Reversal en extremos de RSI con confirmación de Bollinger Bands.

Lógica:
  LONG:  RSI < 30 (sobreventa extrema) + precio en/bajo banda baja de BB
         + divergencia alcista en RSI (precio hace LL, RSI no lo hace)

  SHORT: RSI > 70 (sobrecompra extrema) + precio en/sobre banda alta de BB
         + divergencia bajista en RSI (precio hace HH, RSI no lo hace)

Nota: Los reversals son arriesgados en tendencias fuertes.
      El filtro de tendencia daily los inhibe en tendencias muy extendidas.
"""
import pandas as pd
import numpy as np
import ta
from loguru import logger

from config.settings import REVERSAL_PARAMS


def _has_bullish_divergence(price: pd.Series, rsi: pd.Series,
                             lookback: int = 10) -> pd.Series:
    """
    True en la barra actual si: precio hace Lower Low pero RSI hace Higher Low
    en los últimos `lookback` períodos.
    """
    result = pd.Series(False, index=price.index)
    for i in range(lookback, len(price)):
        window_p   = price.iloc[i - lookback: i + 1]
        window_rsi = rsi.iloc[i - lookback: i + 1]
        # Mínimo actual
        cur_p_min   = window_p.iloc[-1]
        prev_p_min  = window_p.iloc[:-1].min()
        cur_rsi_min = window_rsi.iloc[-1]
        prev_rsi_min= window_rsi.iloc[:-1].min()
        # Divergencia: precio LL pero RSI no
        if cur_p_min < prev_p_min and cur_rsi_min > prev_rsi_min:
            result.iloc[i] = True
    return result


def _has_bearish_divergence(price: pd.Series, rsi: pd.Series,
                              lookback: int = 10) -> pd.Series:
    """
    True si: precio hace Higher High pero RSI hace Lower High.
    """
    result = pd.Series(False, index=price.index)
    for i in range(lookback, len(price)):
        window_p   = price.iloc[i - lookback: i + 1]
        window_rsi = rsi.iloc[i - lookback: i + 1]
        cur_p_max    = window_p.iloc[-1]
        prev_p_max   = window_p.iloc[:-1].max()
        cur_rsi_max  = window_rsi.iloc[-1]
        prev_rsi_max = window_rsi.iloc[:-1].max()
        if cur_p_max > prev_p_max and cur_rsi_max < prev_rsi_max:
            result.iloc[i] = True
    return result


def detect_reversals(df: pd.DataFrame,
                     df_daily: pd.DataFrame = None,
                     params: dict = None,
                     use_divergence: bool = True) -> pd.DataFrame:
    """
    Detecta setups de reversal.

    Args:
        df:             OHLCV (4h o 1d)
        df_daily:       Para filtro de tendencia (si coincide con tendencia fuerte, inhibe)
        params:         Override de REVERSAL_PARAMS
        use_divergence: Si True, requiere divergencia (más selectivo, menos señales)

    Returns:
        df con columnas de setup.
    """
    p = {**REVERSAL_PARAMS, **(params or {})}
    df = df.copy()

    # --- RSI ---
    df["rsi"] = ta.momentum.RSIIndicator(
        df["close"], window=p["rsi_period"]
    ).rsi()

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(
        df["close"], window=p["bb_period"], window_dev=p["bb_std"]
    )
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_pband"]  = bb.bollinger_pband()  # 0=bajo banda, 1=sobre banda

    # --- ATR ---
    atr = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr"] = atr

    # --- Volumen ---
    df["vol_ma"]    = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma"] + 1e-9)

    # --- Divergencias (computacionalmente costosas, opcionales) ---
    if use_divergence and len(df) >= 15:
        df["bull_div"] = _has_bullish_divergence(df["close"], df["rsi"])
        df["bear_div"] = _has_bearish_divergence(df["close"], df["rsi"])
    else:
        df["bull_div"] = False
        df["bear_div"] = False

    # --- Contexto de tendencia (inhibe reversal en tendencia extrema) ---
    if df_daily is not None and not df_daily.empty:
        ema_f = df_daily["close"].ewm(span=21).mean()
        ema_s = df_daily["close"].ewm(span=55).mean()
        # Tendencia muy fuerte = no hacer reversal counter-trend
        strong_bull = (ema_f > ema_s) & (
            (df_daily["close"] - ema_s) / ema_s > 0.08  # >8% sobre EMA55
        )
        strong_bear = (ema_f < ema_s) & (
            (ema_s - df_daily["close"]) / ema_s > 0.08
        )
        daily_dates = [d.date() for d in df_daily.index]
        sb_map = dict(zip(daily_dates, strong_bull.values))
        sd_map = dict(zip(daily_dates, strong_bear.values))
        df_dates = [d.date() for d in df.index]
        df["strong_bull_daily"] = [sb_map.get(d, False) for d in df_dates]
        df["strong_bear_daily"] = [sd_map.get(d, False) for d in df_dates]
    else:
        df["strong_bull_daily"] = False
        df["strong_bear_daily"] = False

    # --- Condiciones LONG reversal ---
    cond_long = (
        (df["rsi"]     <= p["rsi_extreme_low"]) &
        (df["bb_pband"] <= 0.1) &                   # Precio cerca o bajo banda baja
        (~df["strong_bear_daily"])                   # No en tendencia bajista extrema
    )
    if use_divergence:
        cond_long = cond_long & df["bull_div"]

    # Confirmación de volumen (opcional)
    if p.get("volume_confirm", True):
        cond_long = cond_long & (df["vol_ratio"] >= 1.1)

    # --- Condiciones SHORT reversal ---
    cond_short = (
        (df["rsi"]     >= p["rsi_extreme_high"]) &
        (df["bb_pband"] >= 0.9) &
        (~df["strong_bull_daily"])
    )
    if use_divergence:
        cond_short = cond_short & df["bear_div"]
    if p.get("volume_confirm", True):
        cond_short = cond_short & (df["vol_ratio"] >= 1.1)

    # --- Score ---
    rsi_score_long  = ((p["rsi_extreme_low"]  - df["rsi"]).clip(0, 10) / 10)
    rsi_score_short = ((df["rsi"] - p["rsi_extreme_high"]).clip(0, 10) / 10)
    bb_score_long   = (0.1 - df["bb_pband"]).clip(0, 0.1) / 0.1
    bb_score_short  = (df["bb_pband"] - 0.9).clip(0, 0.1) / 0.1

    df["reversal_long"]  = cond_long
    df["reversal_short"] = cond_short
    df["reversal_dir"]   = np.where(
        cond_long, "long", np.where(cond_short, "short", None)
    )
    df["reversal_score"] = np.where(
        cond_long,
        (rsi_score_long * 0.5 + bb_score_long * 0.5).clip(0, 1),
        np.where(
            cond_short,
            (rsi_score_short * 0.5 + bb_score_short * 0.5).clip(0, 1),
            0.0,
        ),
    )

    n_long  = int(cond_long.sum())
    n_short = int(cond_short.sum())
    logger.debug(f"Reversals detectados — LONG: {n_long} | SHORT: {n_short}")
    return df


def get_setup_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reversal_long"] | df["reversal_short"]].copy()
