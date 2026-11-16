"""
Histórico propio en la validación (Fase 4, grupo D).

Gate.io solo sirve las últimas ~10.000 velas 4H (~4,5 años; el sistema pide 1.650
días) y esa ventana se MUEVE: el primer tramo fijo de la puerta (may-2022) se va
perdiendo día a día. `backfill_history.py --apply` guarda una copia en la tabla
`ohlcv` y el worker le añade las últimas velas en cada ciclo.

Este módulo permite que `validate`, `check-live` y `validate --protocol` (y SOLO
ellos; nunca el ciclo vivo, que sigue con LIVE_HISTORY_DAYS de Gate.io) usen esa
copia para las velas ANTERIORES a la primera que sirve hoy Gate.io:

  - Solo BTC/ETH (cripto de Gate.io) y solo filas con source='gateio' (la misma
    fuente; nunca OKX ni filas sin fuente), cerradas y con precios coherentes.
  - Las velas que sirve Gate.io tienen prioridad: de la BD solo se toman las de
    ts < primera vela servida. Si la BD no tiene velas más antiguas, se devuelven
    los MISMOS objetos sin tocar (el arnés y los números no cambian).
  - Diario: se amplía hacia atrás lo MISMO que el 4H (no más), para mantener la
    misma distancia entre el inicio del diario y el del 4H (~175 días de margen
    para el filtro de tendencia diario). Si el 4H no se amplía, el diario tampoco:
    añadir solo diarios cambiaría las EMAs del filtro de tendencia sin aportar
    ninguna vela 4H nueva.
  - Comprobación de empalme: las primeras velas servidas por Gate.io se comparan
    con las de la BD (mismos precios OHLC). Si no coinciden, la copia no se usa
    (aviso en el log): mejor menos historia que una serie mezclada.
  - Huecos en la parte añadida (p. ej. una caída del worker de más de ~8 días que
    no se rellenó con backfill a tiempo): se usa igual y se avisa en el log.

Nunca lanza: ante cualquier error se devuelven los datos de Gate.io tal cual.
"""
from __future__ import annotations

import math
import time

import pandas as pd
from loguru import logger

# Activos cuya fuente es Gate.io y tienen copia propia (backfill_history.SYMBOLS).
DB_HISTORY_SYMBOLS = ("BTCUSDT", "ETHUSDT")
TF_SECONDS = {"4h": 4 * 3600, "1d": 86400}
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
# Velas servidas por Gate.io que se contrastan con la BD en el empalme.
OVERLAP_CHECK_BARS = 30
_PRICE_RTOL = 1e-9


def _valid(o, h, l, c, v) -> bool:
    """Misma regla que backfill_history._valid_candle (precios coherentes)."""
    try:
        o, h, l, c, v = (float(x) for x in (o, h, l, c, v))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (o, h, l, c, v)):
        return False
    return l > 0 and v >= 0 and l <= min(o, c) and h >= max(o, c)


def _same(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= _PRICE_RTOL * max(abs(float(a)), abs(float(b)), 1e-12)


def _first_ts(df: pd.DataFrame) -> int:
    return int(pd.Timestamp(df.index[0]).timestamp())


def _query(conn, symbol: str, tf: str, lo: int | None, hi: int) -> list[tuple]:
    """Filas source='gateio' de la copia con lo <= ts < hi (lo=None: sin límite)."""
    from sqlalchemy import text
    sql = ("SELECT ts, open, high, low, close, volume FROM ohlcv "
           "WHERE symbol = :s AND timeframe = :tf AND source = 'gateio' AND ts < :hi")
    params = {"s": symbol, "tf": tf, "hi": int(hi)}
    if lo is not None:
        sql += " AND ts >= :lo"
        params["lo"] = int(lo)
    sql += " ORDER BY ts"
    return [tuple(r) for r in conn.execute(text(sql), params).fetchall()]


def _overlap_mismatches(conn, symbol: str, tf: str, served: pd.DataFrame) -> tuple[int, int]:
    """(filas comparadas, filas distintas) entre las primeras velas servidas y la BD."""
    step = TF_SECONDS[tf]
    first = _first_ts(served)
    rows = _query(conn, symbol, tf, first, first + OVERLAP_CHECK_BARS * step)
    if not rows:
        return 0, 0
    idx = {int(pd.Timestamp(t).timestamp()): i for i, t in enumerate(served.index[:OVERLAP_CHECK_BARS + 1])}
    compared = bad = 0
    for ts, o, h, l, c, _v in rows:
        i = idx.get(int(ts))
        if i is None:
            continue
        compared += 1
        s = served.iloc[i]
        if not (_same(o, s["open"]) and _same(h, s["high"]) and _same(l, s["low"]) and _same(c, s["close"])):
            bad += 1
    return compared, bad


def _frame(rows: list[tuple], like: pd.DataFrame) -> pd.DataFrame:
    """DataFrame con el mismo formato que el de Gate.io (columnas, tipo e índice)."""
    df = pd.DataFrame(rows, columns=["ts"] + OHLCV_COLS)
    idx = pd.to_datetime(df.pop("ts").astype("int64"), unit="s", utc=True)
    idx = pd.DatetimeIndex(idx, name=like.index.name)
    unit = getattr(like.index, "unit", None)
    if unit:
        idx = idx.as_unit(unit)
    df.index = idx
    return df[list(like.columns)].astype(like.dtypes.to_dict())


def _clean(rows: list[tuple], step: int, now_s: int) -> tuple[list[tuple], int]:
    """Solo velas cerradas y coherentes. Devuelve (filas, descartadas)."""
    out, dropped = [], 0
    for r in rows:
        if int(r[0]) + step > now_s or not _valid(*r[1:6]):
            dropped += 1
            continue
        out.append(r)
    return out, dropped


def _gaps(ts_list: list[int], step: int) -> tuple[int, int]:
    """(saltos, velas que faltan) en una lista ordenada de ts."""
    gaps = missing = 0
    for a, b in zip(ts_list, ts_list[1:]):
        if b - a > step:
            gaps += 1
            missing += (b - a) // step - 1
    return gaps, missing


def extend_with_db_history(symbol: str, df_4h: pd.DataFrame | None, df_daily: pd.DataFrame | None,
                           *, engine=None, now_s: int | None = None):
    """
    Añade delante de las velas de Gate.io las de la copia propia (tabla ohlcv,
    source='gateio') que sean ANTERIORES a la primera vela servida.

    Devuelve (df_4h, df_daily, info). info = {"used": bool, "bars_4h", "bars_1d",
    "first_4h", "served_first_4h", "gaps", "missing", "dropped", "reason"}.
    Si no hay nada que añadir devuelve los mismos objetos que recibió. Nunca lanza.
    """
    info = {"used": False, "bars_4h": 0, "bars_1d": 0, "first_4h": None,
            "served_first_4h": None, "gaps": 0, "missing": 0, "dropped": 0, "reason": ""}
    try:
        if symbol not in DB_HISTORY_SYMBOLS:
            info["reason"] = "no es cripto de Gate.io"
            return df_4h, df_daily, info
        if df_4h is None or df_4h.empty:
            info["reason"] = "Gate.io no sirvió velas 4H"
            return df_4h, df_daily, info
        src = (getattr(df_4h, "attrs", None) or {}).get("source")
        if src not in (None, "gateio"):
            info["reason"] = f"fuente {src}, no gateio"
            return df_4h, df_daily, info
        if list(df_4h.columns) != OHLCV_COLS or (df_daily is not None and not df_daily.empty
                                                 and list(df_daily.columns) != OHLCV_COLS):
            info["reason"] = "columnas inesperadas"
            return df_4h, df_daily, info

        from backfill_history import _ohlcv_columns
        from data.database import get_engine
        engine = engine or get_engine()
        if "source" not in _ohlcv_columns(engine):
            info["reason"] = "la tabla ohlcv no existe o no tiene columna source"
            return df_4h, df_daily, info
        now_s = int(now_s if now_s is not None else time.time())
        served_4h = _first_ts(df_4h)
        info["served_first_4h"] = served_4h
        has_daily = df_daily is not None and not df_daily.empty

        with engine.connect() as conn:
            old_4h = _query(conn, symbol, "4h", None, served_4h)
            if not old_4h:
                info["reason"] = "la BD no tiene velas 4H anteriores a las de Gate.io"
                return df_4h, df_daily, info
            old_4h, dropped = _clean(old_4h, TF_SECONDS["4h"], now_s)
            info["dropped"] += dropped
            if not old_4h:
                info["reason"] = "las velas anteriores de la BD no son válidas"
                return df_4h, df_daily, info
            # Empalme: la copia debe coincidir con lo que sirve hoy Gate.io.
            checks = [("4h", df_4h)] + ([("1d", df_daily)] if has_daily else [])
            for tf, served in checks:
                compared, bad = _overlap_mismatches(conn, symbol, tf, served)
                if bad:
                    info["reason"] = (f"la copia de la BD no coincide con Gate.io en el empalme ({tf}: "
                                      f"{bad} de {compared} velas distintas)")
                    logger.warning(f"[{symbol}] Histórico propio NO usado: {info['reason']}. Revisa con "
                                   "`python backfill_history.py --apply --refresh` mientras Gate.io aún "
                                   "sirva esas fechas.")
                    return df_4h, df_daily, info
            old_1d = []
            if has_daily:
                # El diario se amplía lo mismo que el 4H (misma distancia entre inicios).
                shift = served_4h - int(old_4h[0][0])
                served_1d = _first_ts(df_daily)
                old_1d = _query(conn, symbol, "1d", served_1d - shift, served_1d)
                old_1d, dropped = _clean(old_1d, TF_SECONDS["1d"], now_s)
                info["dropped"] += dropped

        new_4h = pd.concat([_frame(old_4h, df_4h), df_4h])
        new_4h.attrs = dict(df_4h.attrs)
        ts4 = [int(r[0]) for r in old_4h] + [served_4h]
        info["gaps"], info["missing"] = _gaps(ts4, TF_SECONDS["4h"])
        new_1d = df_daily
        if old_1d:
            new_1d = pd.concat([_frame(old_1d, df_daily), df_daily])
            new_1d.attrs = dict(df_daily.attrs)
        if not (new_4h.index.is_monotonic_increasing and new_4h.index.is_unique
                and (new_1d is None or new_1d.empty
                     or (new_1d.index.is_monotonic_increasing and new_1d.index.is_unique))):
            info["reason"] = "orden de velas incoherente tras el empalme"
            logger.warning(f"[{symbol}] Histórico propio NO usado: {info['reason']}.")
            return df_4h, df_daily, info

        info.update(used=True, bars_4h=len(old_4h), bars_1d=len(old_1d), first_4h=int(old_4h[0][0]),
                    reason="ok")
        meta = {k: info[k] for k in ("bars_4h", "bars_1d", "first_4h", "served_first_4h",
                                      "gaps", "missing")}
        new_4h.attrs["db_history"] = meta
        if new_1d is not None and old_1d:
            new_1d.attrs["db_history"] = meta
        logger.info(f"[{symbol}] {describe(info)}")
        if info["gaps"]:
            logger.warning(f"[{symbol}] El histórico propio añadido tiene {info['gaps']} hueco(s) "
                           f"({info['missing']} velas 4H que faltan): se usa igual.")
        return new_4h, new_1d, info
    except Exception as exc:   # noqa: BLE001 — la copia es opcional: nunca rompe la validación
        info["used"] = False
        info["reason"] = f"error leyendo la copia: {exc}"
        logger.warning(f"[{symbol}] Histórico propio no disponible ({exc}); solo velas de Gate.io.")
        return df_4h, df_daily, info


def _fmt(ts: int | None) -> str:
    if ts is None:
        return "—"
    return pd.Timestamp(int(ts), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")


def describe(info: dict) -> str:
    """Una línea para el usuario con lo que se añadió."""
    txt = (f"Histórico propio (tabla ohlcv, source='gateio'): +{info['bars_4h']} velas 4H desde "
           f"{_fmt(info['first_4h'])} UTC y +{info['bars_1d']} diarias, delante de las que sirve hoy "
           f"Gate.io (desde {_fmt(info['served_first_4h'])} UTC)")
    if info.get("gaps"):
        txt += f"; {info['gaps']} hueco(s), faltan {info['missing']} velas 4H"
    return txt + "."


def describe_frame(df_4h: pd.DataFrame | None) -> str | None:
    """La misma línea a partir de df.attrs (None si no se añadió nada)."""
    meta = (getattr(df_4h, "attrs", None) or {}).get("db_history") if df_4h is not None else None
    return describe(meta) if meta else None
