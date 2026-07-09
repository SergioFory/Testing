"""
Orquestador de setups.
Corre los tres detectores, combina resultados en una tabla unificada
y retorna un DataFrame con todos los setups activos.
"""
import pandas as pd
import numpy as np
from loguru import logger

from signals.setups.breakout      import detect_breakouts
from signals.setups.trend_pullback import detect_pullbacks
from signals.setups.reversal       import detect_reversals
from config.settings               import ASSET_PARAMS_OVERRIDE


SETUP_WEIGHTS = {
    "breakout":  0.40,  # Alta probabilidad en rompimientos de consolidación
    "pullback":  0.40,  # Alta probabilidad en tendencias establecidas
    "reversal":  0.20,  # Más arriesgado, usar con cautela
}


def detect_all_setups(
    df_4h:   pd.DataFrame,
    df_daily: pd.DataFrame = None,
    symbol:  str = "UNKNOWN",
) -> pd.DataFrame:
    """
    Ejecuta los tres detectores sobre el DataFrame 4H.

    Args:
        df_4h:    OHLCV 4H limpio
        df_daily: OHLCV 1D para filtro de tendencia
        symbol:   Símbolo para logging

    Returns:
        DataFrame de SETUPS con columnas:
        ts, symbol, setup_type, direction, raw_score, atr, close,
        stop_loss, take_profit
    """
    if df_4h.empty:
        return pd.DataFrame()

    # Parámetros con overrides por símbolo
    overrides = ASSET_PARAMS_OVERRIDE.get(symbol.upper(), {})
    p_breakout = overrides.get("BREAKOUT", {})
    p_pullback = overrides.get("PULLBACK", {})
    p_reversal = overrides.get("REVERSAL", {})

    results = []

    # --- 1. Breakouts ---
    try:
        df_b = detect_breakouts(df_4h.copy(), params=p_breakout)
        for ts, row in df_b[df_b["breakout_long"] | df_b["breakout_short"]].iterrows():
            results.append({
                "ts":         ts,
                "symbol":     symbol,
                "setup_type": "breakout",
                "direction":  row["breakout_dir"],
                "raw_score":  float(row["breakout_score"]),
                "atr":        float(row.get("atr", 0)),
                "close":      float(row["close"]),
                "open":       float(row["open"]),
                "high":       float(row["high"]),
                "low":        float(row["low"]),
                "volume":     float(row["volume"]),
            })
    except Exception as e:
        logger.error(f"Error en breakout detector: {e}")

    # --- 2. Pullbacks ---
    try:
        df_p = detect_pullbacks(df_4h.copy(), df_daily, params=p_pullback)
        for ts, row in df_p[df_p["pullback_long"] | df_p["pullback_short"]].iterrows():
            results.append({
                "ts":         ts,
                "symbol":     symbol,
                "setup_type": "pullback",
                "direction":  row["pullback_dir"],
                "raw_score":  float(row["pullback_score"]),
                "atr":        float(row.get("atr", 0)),
                "close":      float(row["close"]),
                "open":       float(row["open"]),
                "high":       float(row["high"]),
                "low":        float(row["low"]),
                "volume":     float(row["volume"]),
            })
    except Exception as e:
        logger.error(f"Error en pullback detector: {e}")

    # --- 3. Reversals ---
    try:
        df_r = detect_reversals(df_4h.copy(), df_daily, params=p_reversal)
        for ts, row in df_r[df_r["reversal_long"] | df_r["reversal_short"]].iterrows():
            results.append({
                "ts":         ts,
                "symbol":     symbol,
                "setup_type": "reversal",
                "direction":  row["reversal_dir"],
                "raw_score":  float(row["reversal_score"]),
                "atr":        float(row.get("atr", 0)),
                "close":      float(row["close"]),
                "open":       float(row["open"]),
                "high":       float(row["high"]),
                "low":        float(row["low"]),
                "volume":     float(row["volume"]),
            })
    except Exception as e:
        logger.error(f"Error en reversal detector: {e}")

    if not results:
        return pd.DataFrame()

    df_setups = pd.DataFrame(results)
    df_setups = df_setups.sort_values("ts").reset_index(drop=True)

    # --- Filtros por activo: tipo de setup, dirección y tendencia ---
    # Se aplican aquí para afectar por igual a señales, entrenamiento y validación.
    #   "setups":          ["breakout", ...]  restringe los tipos de setup
    #   "directions":      ["long"]           restringe la dirección (p.ej. SP500 long-only)
    #   "require_uptrend": True               solo longs con close>MA200 (y shorts con close<MA200)
    from config.settings import ASSETS as _ASSETS
    _cfg = _ASSETS.get(symbol.upper(), {})
    _allowed = _cfg.get("setups")
    if _allowed:
        df_setups = df_setups[df_setups["setup_type"].isin(_allowed)]
    _dirs = _cfg.get("directions")
    if _dirs:
        df_setups = df_setups[df_setups["direction"].isin(_dirs)]
    if _cfg.get("require_uptrend") and not df_setups.empty:
        _ma200 = df_4h["close"].rolling(200).mean()
        _ma_at = df_setups["ts"].map(_ma200)
        _up = (df_setups["close"] > _ma_at).values
        _keep = np.where(df_setups["direction"].values == "long", _up, ~_up)
        df_setups = df_setups[_keep & _ma_at.notna().values]
    _sessions = _cfg.get("session_hours")   # franjas 4H (UTC) permitidas, p.ej. [12]
    if _sessions and not df_setups.empty:
        _slot = (df_setups["ts"].dt.hour // 4) * 4
        df_setups = df_setups[_slot.isin(_sessions)]
    df_setups = df_setups.reset_index(drop=True)
    if df_setups.empty:
        return pd.DataFrame()

    # --- Stop Loss y Take Profit basados en ATR ---
    # Permite override por activo (e.g. Gold usa R:R 1.33 en vez de 2.0 genérico).
    from config.settings import RISK, ASSETS
    asset_cfg  = ASSETS.get(symbol.upper(), {})
    atr_stop   = asset_cfg.get("atr_stop_mult",   RISK["atr_stop_mult"])
    atr_target = asset_cfg.get("atr_target_mult",  RISK["atr_target_mult"])

    df_setups["stop_loss"] = np.where(
        df_setups["direction"] == "long",
        df_setups["close"] - atr_stop   * df_setups["atr"],
        df_setups["close"] + atr_stop   * df_setups["atr"],
    )
    df_setups["take_profit"] = np.where(
        df_setups["direction"] == "long",
        df_setups["close"] + atr_target * df_setups["atr"],
        df_setups["close"] - atr_target * df_setups["atr"],
    )
    df_setups["rr_ratio"] = atr_target / atr_stop

    # --- Deduplicar: si dos setups en la misma barra y dirección, quedarse con mejor score ---
    df_setups = (
        df_setups.sort_values("raw_score", ascending=False)
        .drop_duplicates(subset=["ts", "direction"], keep="first")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    logger.info(
        f"{symbol} | Setups totales: {len(df_setups)} "
        f"(B:{(df_setups.setup_type=='breakout').sum()} "
        f"P:{(df_setups.setup_type=='pullback').sum()} "
        f"R:{(df_setups.setup_type=='reversal').sum()})"
    )
    return df_setups


def get_current_setup(df_setups: pd.DataFrame,
                      n_last_bars: int = 3) -> pd.DataFrame:
    """
    Retorna setups de las últimas N barras (señales recientes).
    """
    if df_setups.empty:
        return pd.DataFrame()
    cutoff = df_setups["ts"].max() - pd.Timedelta(hours=4 * n_last_bars)
    return df_setups[df_setups["ts"] >= cutoff].copy()
