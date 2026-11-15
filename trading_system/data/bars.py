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

import numpy as np
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


#: Hora de Nueva York a la que reabre la sesión de futuros (CME Globex, GC=F).
_NY_TZ = "America/New_York"
_EVENING_REOPEN_NY_HOUR = 18


def _session_rows_trustworthy(df: pd.DataFrame, now_utc: pd.Timestamp) -> "np.ndarray":
    """
    Filas diarias de yfinance (GC=F, ^GSPC) cuyo contenido es la sesión que dice su
    fecha. Corrección del 2026-11-15 (encontrada al ver en la BD una vela de oro con
    fecha DOMINGO):

    Yahoo devuelve la sesión EN CURSO de los futuros como una fila más, fechada con
    el día de Nueva York en que se negocia. yfinance 1.7.0
    (utils.fix_Yahoo_returning_live_separate) junta las dos últimas filas si tienen
    la misma fecha y se queda con la EN CURSO. Resultado, para el oro:
      - domingo desde la reapertura (18:00 ET) hasta medianoche ET: fila fechada
        domingo con unas horas de la sesión del lunes;
      - lunes a viernes desde la reapertura (18:00 ET) hasta medianoche ET: la vela
        COMPLETA del día D desaparece y en su lugar llega una fila "D" con las
        primeras horas de la sesión siguiente.
    En ambos casos la fila ya pasó las 22:15 UTC de su fecha y se daba por cerrada:
    el ciclo evaluaba setups sobre una vela que no existe (alertas fantasma de oro
    por la noche, como las del 28-sep).

    Regla: (1) nunca se aceptan filas con fecha de sábado o domingo (no hay sesiones
    en fin de semana); (2) la fila con fecha de HOY en Nueva York no se acepta desde
    las 18:00 ET (puede ser la sesión siguiente). La vela verdadera del día D vuelve
    a medianoche ET (la sesión en curso pasa a fecharse D+1) y se evalúa en el primer
    ciclo después: en invierno también en el hueco 22:15–23:00 UTC; en verano la
    reapertura es a las 22:00 UTC, antes de las 22:15, así que el oro se evalúa a
    partir de las 04:00 UTC. Para ^GSPC (sin sesión nocturna) solo retrasa unas horas
    la evaluación (además está en modo sombra).
    """
    idx = _utc_index(df)
    dates = idx.normalize()
    weekday_ok = dates.dayofweek < 5
    now_ny = now_utc.tz_convert(_NY_TZ)
    today_ny = pd.Timestamp(now_ny.date(), tz="UTC")
    evening = now_ny.hour >= _EVENING_REOPEN_NY_HOUR
    live_row = (dates == today_ny) & evening
    return np.asarray(weekday_ok & ~live_row, dtype=bool)


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
    now_utc = _now_utc(now)
    closes = bar_close_times(df, timeframe, session=session)
    mask = np.asarray(closes <= now_utc, dtype=bool)
    if session:
        mask = mask & _session_rows_trustworthy(df, now_utc)
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
