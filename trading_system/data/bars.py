"""
Utilidades de velas: descartar la vela EN FORMACIÓN y comprobar frescura.

Por qué existe (Fase 0, 2026-11): los fetchers devuelven también la vela que
todavía no ha cerrado (la 4H en curso de Gate.io, la diaria intradía de
yfinance). El sistema la evaluaba como si estuviera cerrada, así que un mismo
setup generaba varias alertas seguidas con entradas distintas (el 28-sep, 5
alertas de oro en 4 horas) y aparecían setups "fantasma" que desaparecían al
cerrar la vela. El validador solo ve velas cerradas: el vivo debe ver lo mismo.

Convenciones de índice:
  - ccxt (cripto): el índice es la HORA DE APERTURA de la vela (UTC). Una vela
    4H está cerrada cuando apertura + 4h <= ahora.
  - yfinance diario (GC=F, ^GSPC): el índice es la FECHA de la sesión a las
    00:00 UTC. La sesión D está cerrada a partir de D + DAILY_BAR_CLOSE_UTC.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

_TF_LEN = {
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def _now_utc(now=None) -> pd.Timestamp:
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def bar_close_times(df: pd.DataFrame, timeframe: str, session: bool = False) -> pd.DatetimeIndex:
    """Hora (UTC) a la que cierra cada vela del DataFrame."""
    idx = _utc_index(df)
    if session:
        from config.settings import DAILY_BAR_CLOSE_UTC
        h, m = DAILY_BAR_CLOSE_UTC
        return idx.normalize() + pd.Timedelta(hours=h, minutes=m)
    if timeframe not in _TF_LEN:
        raise ValueError(f"Timeframe no soportado: {timeframe}")
    return idx + _TF_LEN[timeframe]


def drop_unclosed_bars(df: pd.DataFrame | None, timeframe: str,
                       session: bool = False, now=None) -> pd.DataFrame | None:
    """
    Devuelve el DataFrame sin las velas que aún no han cerrado.

    Args:
        timeframe: "1h" | "4h" | "1d" (ignorado si session=True).
        session:   True para velas diarias de sesión (yfinance: GC=F, ^GSPC).
        now:       reloj inyectable (tests / emulación). Por defecto, ahora UTC.
    """
    if df is None or df.empty:
        return df
    closes = bar_close_times(df, timeframe, session=session)
    mask = closes <= _now_utc(now)
    out = df.loc[mask]
    if hasattr(df, "attrs"):
        out.attrs = dict(df.attrs)
    return out


def last_bar_age_hours(df: pd.DataFrame, timeframe: str,
                       session: bool = False, now=None) -> float:
    """Horas transcurridas desde el CIERRE de la última vela del DataFrame."""
    if df is None or df.empty:
        return float("inf")
    close = bar_close_times(df.iloc[[-1]], timeframe, session=session)[0]
    return (_now_utc(now) - close).total_seconds() / 3600.0
