"""
Calendario de sesiones de los activos diarios (oro GC=F y SPX ^GSPC, yfinance).

Por qué existe (corrección de la revisión de la Fase 1): la hora de "Cerrar a
mercado si no toca SL/TP antes de" se calculaba sumando 20 días hábiles L-V
(pd.offsets.BDay), pero el tracker cuenta 20 VELAS REALES y yfinance no tiene
vela en los festivos de la bolsa de Nueva York. Con un festivo dentro del
horizonte (casi siempre: hay ~10 al año y el horizonte son 4 semanas) el mensaje
decía un día y el tracker cerraba otro posterior: en los datos desde 2024-11
solo coincidían el 36-38 % de las salidas por tiempo.

Aquí se definen los cierres de NYSE (reglas oficiales + cierres excepcionales
conocidos). Medido con los datos del arnés desde 2015 (G-correcciones/cal_diff.py):
  - ^GSPC: los únicos días L-V sin vela que no son festivos son los dos cierres
    excepcionales (2018-12-05 y 2025-01-09), incluidos abajo.
  - GC=F (COMEX) sigue los mismos festivos, pero el 2025-01-09 (luto por
    Carter, NYSE cerrada) SÍ tuvo sesión: su calendario ("comex") no lleva ese
    cierre excepcional. Quedan rarezas de yfinance imposibles de prever: 3 días
    sin vela sueltos en 2016-2018 y vela en 2023-11-23 y 2025-07-04; con ellas
    el plazo del mensaje puede diferir un día del cierre real del tracker.

Solo se usa para mostrar el plazo (mensaje y resumen); el tracker sigue contando
velas reales (misma convención que el validador, Fase 1 punto 1.7).
"""
from __future__ import annotations

from datetime import timedelta
from functools import lru_cache

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday,
)


def _new_year_observance(dt):
    """Año Nuevo en NYSE: si cae en domingo se cierra el lunes; si cae en sábado
    NO se cierra el viernes anterior (es el último día del año contable).
    Ej.: 2022-01-01 fue sábado y el 2021-12-31 hubo sesión normal."""
    if dt.weekday() == 6:
        return dt + timedelta(days=1)
    if dt.weekday() == 5:
        return None
    return dt


# Cierres excepcionales (no siguen ninguna regla), por calendario. Si en el
# futuro hay otro (luto nacional, huracán...), añadirlo aquí.
SPECIAL_CLOSURES = {
    "nyse": (
        "2007-01-02",   # luto nacional por Gerald Ford
        "2012-10-29",   # huracán Sandy
        "2012-10-30",   # huracán Sandy
        "2018-12-05",   # luto nacional por George H. W. Bush
        "2025-01-09",   # luto nacional por Jimmy Carter
    ),
    # GC=F: sin vela el 2018-12-05; CON vela el 2025-01-09 (COMEX abrió).
    "comex": (
        "2007-01-02",
        "2012-10-29",
        "2012-10-30",
        "2018-12-05",
    ),
}


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Año Nuevo", month=1, day=1, observance=_new_year_observance),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independencia", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Navidad", month=12, day=25, observance=nearest_workday),
    ]


def calendar_for(symbol: str) -> str:
    """Calendario de sesiones del activo: 'comex' para los metales (GC=F),
    'nyse' para el resto (^GSPC)."""
    return "comex" if str(symbol).upper().startswith(("XAU", "XAG")) else "nyse"


@lru_cache(maxsize=4)
def closures(calendar: str = "nyse") -> tuple:
    """Fechas (naive, 00:00) sin sesión entre 2000 y 2040 en ese calendario."""
    if calendar not in SPECIAL_CLOSURES:
        raise ValueError(f"Calendario desconocido: {calendar}")
    hol = NYSEHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31")
    special = pd.DatetimeIndex(pd.to_datetime(list(SPECIAL_CLOSURES[calendar])))
    return tuple(sorted(set(hol.append(special))))


@lru_cache(maxsize=4)
def session_offset(calendar: str = "nyse") -> pd.offsets.CustomBusinessDay:
    """Una sesión bursátil (L-V sin festivos ni cierres excepcionales)."""
    return pd.offsets.CustomBusinessDay(holidays=list(closures(calendar)))


def _naive_day(day) -> pd.Timestamp:
    t = pd.Timestamp(day)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def add_sessions(day, n: int, calendar: str = "nyse") -> pd.Timestamp:
    """Fecha (00:00 UTC) de la sesión número `n` posterior a `day`.

    Si `day` no es sesión (fin de semana o festivo con vela rara de yfinance), la
    primera sesión que cuenta es la siguiente, igual que con BDay."""
    target = _naive_day(day) + int(n) * session_offset(calendar)
    return pd.Timestamp(target).tz_localize("UTC")


def is_session(day, calendar: str = "nyse") -> bool:
    """True si ese día hay sesión en ese calendario."""
    t = _naive_day(day)
    return t.weekday() < 5 and t not in set(closures(calendar))
