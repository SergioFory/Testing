"""
Contexto DIARIO para setups detectados en otro timeframe, sin look-ahead
(Fase 1, punto 1.8).

Problema que corrige: pullback y reversal propagaban a cada vela 4H el valor
diario de SU MISMO día (emparejando por fecha). La vela diaria del día D cierra
a las 00:00 del día siguiente, así que la vela 4H de las 04:00 del día D usaba
una tendencia calculada con el cierre de las 24:00: información del futuro en
el validador. En vivo, además, ese diario todavía no existe (solo se usan velas
cerradas) y el contexto salía 0: vivo y validador no veían lo mismo.

Regla:
  - Base INTRADÍA (cripto 4H): cada vela usa el último diario CERRADO antes de
    su día, es decir, el del día ANTERIOR (o el último disponible hasta
    MAX_STALE_DAYS días antes, para fines de semana en mercados que cierran).
  - Base DIARIA (oro, SPX: el DataFrame "4H" es el propio diario): el setup del
    día D se evalúa al cierre de D, cuando el diario D ya está cerrado → se
    empareja con el mismo día, sin desplazar (comportamiento anterior).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Antigüedad máxima (en días naturales) del diario que se usa como contexto de
# una vela intradía. 4 cubre un fin de semana largo; con más hueco → sin contexto.
MAX_STALE_DAYS = 4


def _day_index(idx) -> np.ndarray:
    """Fechas (datetime64[D], en UTC si el índice tiene zona) de un índice."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.values.astype("datetime64[D]")


def is_intraday_index(idx) -> bool:
    """True si el índice es de velas intradía (separación típica < 20 h)."""
    idx = pd.DatetimeIndex(idx)
    if len(idx) >= 2:
        step = pd.Series(idx).diff().dropna().median()
        return bool(step < pd.Timedelta(hours=20))
    if len(idx) == 1:
        t = idx[0]
        return not (t.hour == 0 and t.minute == 0)
    return False


def map_daily_context(base_index, daily_values: pd.Series, default) -> list:
    """
    Valor diario que corresponde a cada vela de base_index, sin look-ahead.

    Args:
        base_index:   índice del DataFrame en el que se detecta el setup.
        daily_values: serie indexada por la fecha de la vela diaria.
        default:      valor si no hay diario aplicable (0 / False).

    Returns:
        Lista con un valor por vela de base_index.
    """
    base_days = _day_index(base_index)
    d_days = _day_index(daily_values.index)
    vals = list(daily_values.values)
    if not is_intraday_index(base_index):
        # Base diaria: mismo día (el diario D está cerrado al evaluar el setup D).
        mapping = dict(zip(d_days.tolist(), vals))
        return [mapping.get(d, default) for d in base_days.tolist()]

    # Base intradía: último diario con fecha ESTRICTAMENTE anterior al día de la vela.
    order = np.argsort(d_days, kind="stable")
    d_sorted = d_days[order]
    v_sorted = [vals[k] for k in order]
    pos = np.searchsorted(d_sorted, base_days, side="left") - 1
    max_stale = np.timedelta64(MAX_STALE_DAYS, "D")
    out = []
    for day, p in zip(base_days, pos):
        if p < 0 or (day - d_sorted[p]) > max_stale:
            out.append(default)
        else:
            out.append(v_sorted[p])
    return out
