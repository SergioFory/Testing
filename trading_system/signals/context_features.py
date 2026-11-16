"""
Variables de contexto de cada setup (Fase 3, punto 3.3; F14, F31, F32, E3, E6).

Dos usos:
  1. REGISTRO EN SOMBRA: en cada ciclo, cada setup de la ventana que se guarda en
     shadow_setups lleva una columna `features` (JSON compacto) con:
       sma200_with   el setup va a favor de la SMA200 diaria (long con el cierre
                     diario por encima de su SMA200, short por debajo): true/false
       tsmom         signo (+1/−1/0) de los retornos diarios a 21, 63 y 252 días,
                     en ese orden (sin alinear con la dirección)
       atr_rank      percentil del ATR(14)/precio en las últimas 100 velas (como
                     el filtro de breakout.py), en la vela del setup
       dist_atr_now  distancia del precio actual a la entrada, en ATR, con signo:
                     positiva = el precio ya se movió A FAVOR de la operación
       dvol, vrp     solo BTC/ETH: DVOL de Deribit en el ciclo y
                     vrp = (dvol/100) / volatilidad realizada anualizada a 30 días
                     (log-retornos 4H de las 180 velas hasta la del setup, ×√(6·365);
                     misma definición que E6). null si Deribit no responde.
     NO filtran nada: solo se registran para decidir más adelante con datos propios.
  2. FILTRO CANDIDATO "sma200_with" (solo para la puerta `validate --protocol
     --candidate`, clave de configuración "setup_filter"; ningún activo lo usa en
     el vivo): conserva solo los setups a favor de la SMA200 diaria.

Sin look-ahead: el contexto diario sale del ÚLTIMO DIARIO CERRADO al evaluar el
setup, con la regla de la Fase 1 (signals/setups/daily_context.map_daily_context):
en cripto 4H, el diario del día ANTERIOR al de la vela; en oro/SPX (la vela del
setup es el propio diario y se evalúa a su cierre), el mismo día. Si la SMA200 no
existe todavía (menos de 200 diarios), sma200_with = null y el filtro NO descarta
el setup (no hay información para filtrar).

Solo numpy/pandas/ta (ya usados por los detectores). Coste medido < 50 ms por
activo y ciclo (ver la bitácora de la Fase 3).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from signals.setups.daily_context import map_daily_context

TSMOM_DAYS = (21, 63, 252)
SMA_DAYS = 200
ATR_RANK_WINDOW = 100
RV_DAYS = 30
BARS_PER_DAY_4H = 6
SUPPORTED_FILTERS = ("sma200_with",)


def _utc_index(idx) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def _daily_frame(df_4h: pd.DataFrame, df_daily: pd.DataFrame | None) -> pd.DataFrame:
    """Diario para el contexto: df_daily; si falta, el propio df_4h (oro/SPX son diarios)."""
    return df_daily if df_daily is not None and not df_daily.empty else df_4h


def _bar_positions(df_4h: pd.DataFrame, ts) -> np.ndarray:
    """Posición de cada ts en el índice de df_4h (-1 si no está exactamente)."""
    idx = _utc_index(df_4h.index)
    t = _utc_index(pd.to_datetime(pd.Series(ts), utc=True))
    pos = idx.get_indexer(t)
    return np.asarray(pos, dtype=int)


def _daily_on_bars(df_4h: pd.DataFrame, daily_values: pd.Series) -> np.ndarray:
    """Valor diario (sin look-ahead) en CADA vela de df_4h. Se mapea sobre el índice
    COMPLETO de df_4h (no sobre los setups sueltos) para que la detección de
    "intradía vs diario" de map_daily_context sea la correcta."""
    vals = map_daily_context(df_4h.index, daily_values, np.nan)
    return np.asarray(vals, dtype=float)


def sma200_side_on_bars(df_4h: pd.DataFrame, df_daily: pd.DataFrame | None) -> np.ndarray:
    """+1 si el último diario cerrado está por encima de su SMA200, −1 si está por
    debajo, 0 si coincide, NaN si la SMA200 aún no existe; una por vela de df_4h."""
    d = _daily_frame(df_4h, df_daily)
    c = d["close"].astype(float)
    side = np.sign(c - c.rolling(SMA_DAYS).mean())      # NaN donde no hay SMA200
    return _daily_on_bars(df_4h, side)


def sma200_with(setups: pd.DataFrame, df_4h: pd.DataFrame,
                df_daily: pd.DataFrame | None, side_on_bars: np.ndarray | None = None) -> list:
    """True/False/None por setup: a favor de la SMA200 diaria (None = sin SMA200)."""
    if setups is None or len(setups) == 0:
        return []
    side = sma200_side_on_bars(df_4h, df_daily) if side_on_bars is None else side_on_bars
    pos = _bar_positions(df_4h, setups["ts"])
    out = []
    for p, d in zip(pos, setups["direction"].astype(str)):
        s = side[p] if p >= 0 else np.nan
        if not np.isfinite(s):
            out.append(None)
        else:
            out.append(bool(s > 0) if d == "long" else bool(s < 0))
    return out


def apply_setup_filter(setups: pd.DataFrame, name: str, df_4h: pd.DataFrame,
                       df_daily: pd.DataFrame | None) -> pd.DataFrame:
    """
    Aplica un filtro candidato (clave "setup_filter" de ASSETS; hoy solo lo usa la
    puerta). "sma200_with": quita los setups EN CONTRA de la SMA200 diaria; los que
    no tienen SMA200 todavía se conservan. Filtro desconocido → ValueError (un error
    de configuración no puede pasar en silencio).
    """
    if name not in SUPPORTED_FILTERS:
        raise ValueError(f"setup_filter desconocido: {name!r} (válidos: {', '.join(SUPPORTED_FILTERS)})")
    if setups is None or setups.empty:
        return setups
    w = sma200_with(setups, df_4h, df_daily)
    keep = np.array([x is not False for x in w], dtype=bool)
    return setups[keep]


# ---------------------------------------------------------------------------
# Registro en sombra
# ---------------------------------------------------------------------------

def _round(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return (round(x, nd) + 0.0) if np.isfinite(x) else None   # + 0.0: sin "-0.0" en el JSON


def _key(ts, direction) -> tuple:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (t, str(direction))


def _tsmom_on_bars(df_4h, df_daily) -> list[np.ndarray]:
    d = _daily_frame(df_4h, df_daily)
    c = d["close"].astype(float)
    return [_daily_on_bars(df_4h, np.sign(c / c.shift(n) - 1.0)) for n in TSMOM_DAYS]


def _atr_rank_on_bars(df_4h) -> np.ndarray:
    import ta
    atr = ta.volatility.AverageTrueRange(df_4h["high"], df_4h["low"], df_4h["close"],
                                         window=14).average_true_range()
    return (atr / df_4h["close"]).rolling(ATR_RANK_WINDOW).rank(pct=True).to_numpy(dtype=float)


def _rv30_on_bars(df_4h) -> np.ndarray:
    lr = np.log(df_4h["close"].astype(float)).diff()
    n = RV_DAYS * BARS_PER_DAY_4H
    return (lr.rolling(n).std() * np.sqrt(BARS_PER_DAY_4H * 365)).to_numpy(dtype=float)


def compute_setup_features(window: pd.DataFrame, df_4h: pd.DataFrame,
                           df_daily: pd.DataFrame | None, current_price: float | None,
                           dvol: float | None = None, with_vol: bool = False) -> dict:
    """
    JSON compacto de variables de contexto por setup de la ventana del ciclo.

    Args:
        window:        setups de la ventana (ts, direction, close, atr).
        df_4h/df_daily: velas cerradas del ciclo (las mismas de la detección).
        current_price: cierre de la última vela cerrada.
        dvol:          DVOL de Deribit del ciclo (None si no hay o falló).
        with_vol:      True para BTC/ETH: añade dvol, rv30 y vrp (null si falta algo).

    Returns:
        {(ts, direction): '{"sma200_with":true,...}'}. Cada variable se calcula en su
        propio try: si una falla queda null y las demás se guardan igual.
    """
    if window is None or window.empty or df_4h is None or df_4h.empty:
        return {}

    def _safe(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001 — una variable rota no tumba las demás
            return None

    side = _safe(lambda: sma200_side_on_bars(df_4h, df_daily))
    tsm = _safe(lambda: _tsmom_on_bars(df_4h, df_daily))
    arank = _safe(lambda: _atr_rank_on_bars(df_4h))
    rv = _safe(lambda: _rv30_on_bars(df_4h)) if with_vol else None
    pos = _bar_positions(df_4h, window["ts"])
    out = {}
    for p, (_, s) in zip(pos, window.iterrows()):
        try:
            d = str(s["direction"])
            f = {}
            sv = side[p] if (side is not None and p >= 0) else np.nan
            f["sma200_with"] = None if not np.isfinite(sv) else (bool(sv > 0) if d == "long" else bool(sv < 0))
            f["tsmom"] = None if tsm is None or p < 0 else \
                [None if not np.isfinite(a[p]) else int(a[p]) for a in tsm]
            f["atr_rank"] = None if arank is None or p < 0 else _round(arank[p])
            entry, atr = s.get("close"), s.get("atr")
            dist = None
            if current_price is not None and entry is not None and atr is not None \
                    and pd.notna(entry) and pd.notna(atr) and float(atr) > 0:
                dist = (float(current_price) - float(entry)) / float(atr)
                dist = dist if d == "long" else -dist
            f["dist_atr_now"] = _round(dist)
            if with_vol:
                rvp = rv[p] if (rv is not None and p >= 0) else np.nan
                f["dvol"] = _round(dvol, 2)
                f["rv30"] = _round(rvp)
                f["vrp"] = _round((float(dvol) / 100.0) / rvp) \
                    if (dvol is not None and np.isfinite(rvp) and rvp > 0) else None
            out[_key(s["ts"], d)] = json.dumps(f, separators=(",", ":"))
        except Exception:  # noqa: BLE001
            continue
    return out
