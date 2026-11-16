"""
Monitor de datos y deriva (Fase 2, punto 2.3).

Para qué existe: esta semana aparecieron dos fallos de datos que nadie vio hasta
mirar la BD a mano:
  1. Gate.io devolvía PLANA (O=H=L=C, volumen 0) la vela que caía en el corte de
     cada página de la descarga (corregido en binance_fetcher._fetch_ohlcv_from).
  2. Yahoo devolvía la sesión EN CURSO de GC=F como una fila fechada en DOMINGO
     (corregido en data/bars._session_rows_trustworthy).
Este módulo los detectaría si volvieran (y otros parecidos), sin descargar nada
extra en el ciclo horario y SIN cambiar las señales: no bloquea nada. Si algún
día se quiere bloquear con datos sospechosos es una decisión del usuario
(propuesta en REGLAS_DE_DECISION.md), no de este código.

Tres partes:
  A) Por ciclo (cmd_signal, sobre los DataFrames ya cargados; últimas
     LOOKBACK_BARS velas cerradas): velas planas, índice duplicado o desordenado,
     filas en fin de semana (activos de sesión), huecos (cripto), precio
     congelado, saltos absurdos y cambio de fuente. Se guardan en la tabla
     data_quality (UNIQUE(symbol, check_name, ts): la misma incidencia vista en
     varios ciclos es UNA fila). Los graves avisan por Telegram con el mismo
     throttling de 6 h que los avisos de datos (main._DATA_ALERT_EVERY).
  B) Nocturno (03:00 UTC): contraste OKX vs Gate.io en BTC/ETH (una petición a
     OKX con timeout de 10 s; Gate.io sale de la tabla ohlcv) y deriva del
     detector (setups y aprobados de 30 días en shadow_setups frente a
     settings.DRIFT_BASELINE, calculado offline con el arnés).
  C) Bloque "Calidad de datos" del resumen nocturno (monitoring/daily_summary.py).

Oro y SPX: no hay fuente gratuita alternativa fiable para contrastar (Stooq y
Twelve Data gratis no dan GC=F/^GSPC diarios equivalentes con garantías; Yahoo es
ya la fuente). No se contrastan y el resumen lo dice; no se inventa una fuente.

Redactor opcional con LLM: hoy no hay clave de API (ANTHROPIC_API_KEY vacía), así
que todo el texto es fijo. Si algún día la hay, el sitio para enchufarlo es
format_quality_block(): recibe ya todos los datos estructurados (incidencias,
contraste y deriva) y devuelve las líneas del bloque; un redactor podría
reescribirlas en lenguaje natural, con este texto fijo como respaldo si falla.

Nada de este módulo puede tumbar ni colgar el ciclo: cada función pública
captura sus excepciones y las peticiones de red llevan timeout <= 10 s.
"""
from __future__ import annotations

import html
import json
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Parámetros (fijados en la especificación de la Fase 2, punto 2.3)
# ---------------------------------------------------------------------------
LOOKBACK_BARS = 60                     # velas cerradas revisadas en cada ciclo
FROZEN_MIN_RUN = 6                     # >= 6 cierres idénticos seguidos = congelado
JUMP_LIMIT = {"crypto": 0.25, "session": 0.10}   # |cierre/cierre previo − 1|
GRAVE_LAST_BARS = 2                    # plana/duplicada/fin de semana en las 2 últimas
ALERT_EVERY = timedelta(hours=6)       # = main._DATA_ALERT_EVERY (throttling de avisos)
NET_TIMEOUT_S = 10                     # timeout de cada petición de red (nocturno)
KEEP_DAYS = 90                         # data_quality: se purga lo más viejo en el nocturno

# Contraste entre fuentes (cripto 4H)
CROSS_BARS = 12
CROSS_MIN_COMMON = 6                   # con menos velas comunes: "sin datos suficientes"
CROSS_MEDIAN_LIMIT = 0.005             # mediana |Δcierre|/cierre > 0,5 % → alerta
CROSS_MAX_LIMIT = 0.02                 # cualquier vela > 2 % → alerta
CROSS_STALE_HOURS = 36                 # el resumen no muestra un contraste más viejo

# Deriva del detector
DRIFT_WINDOW_DAYS = 30

KV_CROSS_PREFIX = "calidad:contraste:"  # kv: último contraste por activo (JSON)

# Mercado de cada activo en OKX (perpetuo USDT) y en Gate.io (contrato USDT).
OKX_INST = {"BTCUSDT": "BTC-USDT-SWAP", "ETHUSDT": "ETH-USDT-SWAP"}
GATE_CONTRACT = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT"}

# Códigos de comprobación (columna check_name) y su texto en español.
CHECK_TEXT = {
    "flat":      "vela plana (apertura = máximo = mínimo = cierre y volumen 0)",
    "duplicate": "vela duplicada (misma hora dos veces)",
    "unsorted":  "velas desordenadas",
    "weekend":   "vela con fecha de fin de semana (no hay sesión)",
    "gap":       "faltan velas (hueco)",
    "frozen":    "precio congelado (cierres idénticos seguidos)",
    "jump":      "salto de precio anormal",
    "source":    "cambio de fuente de datos",
}
CHECK_SHORT = {
    "flat": "vela plana", "duplicate": "duplicada", "unsorted": "desordenadas",
    "weekend": "fin de semana", "gap": "hueco", "frozen": "precio congelado",
    "jump": "salto anormal", "source": "cambio de fuente",
}


def _e(v) -> str:
    return html.escape(str(v))


def _now(now=None) -> pd.Timestamp:
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _fmt_bar(ts: pd.Timestamp, daily: bool) -> str:
    return ts.strftime("%Y-%m-%d" if daily else "%Y-%m-%d %H:%M UTC")


def _pct(x: float) -> str:
    """Porcentaje en español; con 3 decimales si es < 0,1 % (lo normal entre OKX y
    Gate.io es ~0,005 %: con 2 decimales saldría "0,00 %")."""
    v = x * 100
    return (f"{v:.3f}" if abs(v) < 0.1 else f"{v:.2f}").replace(".", ",") + " %"


# ---------------------------------------------------------------------------
# A) Comprobaciones por ciclo (funciones puras: sin BD ni red)
# ---------------------------------------------------------------------------

def expected_source(symbol: str) -> str | None:
    """Fuente que DEBE traer cada activo (df.attrs['source'])."""
    from config.settings import ASSETS
    cfg = ASSETS.get(symbol, {})
    src = cfg.get("source", "binance")
    if src == "yfinance":
        return cfg.get("yf_ticker")
    if src == "twelvedata":
        return "twelvedata"
    return "gateio"


def check_frame(df: pd.DataFrame, timeframe: str = "4h", session: bool = False,
                expected: str | None = None, lookback: int = LOOKBACK_BARS,
                suffix: str = "") -> list[dict]:
    """
    Revisa las últimas `lookback` velas de un DataFrame OHLCV (ya sin la vela en
    formación). Devuelve una lista de incidencias
    {"check", "ts", "detalle", "grave"} (ts = vela afectada, pd.Timestamp UTC).

    timeframe: "4h" o "1d" (el paso esperado para buscar huecos en cripto).
    session:   True en activos de sesión (yfinance): se buscan filas en fin de
               semana y NO huecos (festivos y fines de semana son normales).
    suffix:    se añade al código (p. ej. ":1d" para el diario de cripto), para
               que la misma hora en 4H y en 1D no choque en el UNIQUE.
    """
    out: list[dict] = []
    if df is None or len(df) == 0:
        return out
    tail = df.iloc[-lookback:]
    idx = pd.DatetimeIndex(tail.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    n = len(tail)
    daily = session or timeframe == "1d"
    tf_label = "diaria" if daily else timeframe.upper()
    last_grave_pos = n - GRAVE_LAST_BARS

    def add(check, pos, detalle, grave=False):
        out.append({"check": check + suffix, "ts": idx[pos], "detalle": detalle,
                    "grave": bool(grave)})

    o = tail["open"].to_numpy(dtype=float)
    h = tail["high"].to_numpy(dtype=float)
    lo = tail["low"].to_numpy(dtype=float)
    c = tail["close"].to_numpy(dtype=float)
    v = tail["volume"].to_numpy(dtype=float) if "volume" in tail else np.full(n, np.nan)

    # 1) Velas planas con volumen 0 (el fallo del corte de página de Gate.io).
    flat = (o == h) & (h == lo) & (lo == c) & (v == 0)
    for p in np.flatnonzero(flat):
        add("flat", p, f"vela {tf_label} {_fmt_bar(idx[p], daily)}: O=H=L=C={c[p]:g}, volumen 0",
            grave=p >= last_grave_pos)

    # 2) Índice duplicado o desordenado.
    ns = idx.as_unit("ns").asi8
    dup = idx.duplicated(keep="first")
    for p in np.flatnonzero(dup):
        add("duplicate", p, f"la vela {_fmt_bar(idx[p], daily)} aparece más de una vez",
            grave=p >= last_grave_pos)
    if n > 1:
        back = np.flatnonzero(ns[1:] < ns[:-1]) + 1
        for p in back:
            add("unsorted", p, f"la vela {_fmt_bar(idx[p], daily)} llega después de "
                               f"{_fmt_bar(idx[p - 1], daily)}")

    # 3) Fin de semana en activos de sesión (el fallo de la vela de GC=F en domingo).
    if session:
        wk = idx.dayofweek >= 5
        for p in np.flatnonzero(wk):
            dia = "sábado" if idx[p].dayofweek == 5 else "domingo"
            add("weekend", p, f"fila con fecha {idx[p]:%Y-%m-%d} ({dia}): no hay sesión ese día",
                grave=p >= last_grave_pos)

    # 4) Huecos (solo cripto, que cotiza 24/7): velas que faltan entre dos seguidas.
    if not session and n > 1 and timeframe in ("4h", "1d"):
        step = pd.Timedelta(hours=4 if timeframe == "4h" else 24).value
        srt = np.sort(np.unique(ns))
        d = np.diff(srt)
        for k in np.flatnonzero(d > step):
            faltan = int(d[k] // step) - 1 if d[k] % step == 0 else int(d[k] // step)
            first_missing = pd.Timestamp(srt[k] + step, tz="UTC")
            out.append({"check": "gap" + suffix, "ts": first_missing, "grave": False,
                        "detalle": f"faltan {faltan} vela(s) {tf_label} entre "
                                   f"{_fmt_bar(pd.Timestamp(srt[k], tz='UTC'), daily)} y "
                                   f"{_fmt_bar(pd.Timestamp(srt[k + 1], tz='UTC'), daily)}"})

    # 5) Precio congelado: >= FROZEN_MIN_RUN cierres idénticos seguidos. La
    #    incidencia se fecha en la PRIMERA vela de la racha (estable entre ciclos).
    if n >= FROZEN_MIN_RUN:
        start = 0
        for p in range(1, n + 1):
            if p == n or c[p] != c[start]:
                run = p - start
                if run >= FROZEN_MIN_RUN:
                    add("frozen", start, f"{run} cierres seguidos idénticos ({c[start]:g}) desde "
                                         f"{_fmt_bar(idx[start], daily)}")
                start = p

    # 6) Saltos absurdos entre cierres consecutivos.
    if n > 1:
        lim = JUMP_LIMIT["session" if session else "crypto"]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c[1:] / c[:-1] - 1.0
        for k in np.flatnonzero(np.abs(r) > lim):
            p = k + 1
            add("jump", p, f"cierre {_fmt_bar(idx[p], daily)} {c[p]:g} frente a {c[p - 1]:g} "
                           f"({r[k] * 100:+.1f} %; límite ±{lim * 100:.0f} %)",
                grave=p == n - 1)

    # 7) Cambio de fuente (attrs['source'] distinto del esperado). Sin attrs no se
    #    juzga (los fetchers del sistema siempre lo ponen).
    src = (getattr(df, "attrs", None) or {}).get("source")
    if expected and src and src != expected:
        add("source", n - 1, f"los datos vienen de {src} (se esperaba {expected})")
    return out


def cycle_incidents(symbol: str, df_4h: pd.DataFrame, df_daily: pd.DataFrame | None = None) -> list[dict]:
    """Incidencias de un activo con los DataFrames del ciclo (sin BD ni red).
    Cripto: 4H (todas las comprobaciones) y diario (sufijo ':1d'). Activos de
    sesión (yfinance): df_4h ES el diario (no se revisa dos veces)."""
    from config.settings import ASSETS
    source = ASSETS.get(symbol, {}).get("source", "binance")
    session = source == "yfinance"
    exp = expected_source(symbol)
    if session:
        return check_frame(df_4h, "1d", session=True, expected=exp)
    out = check_frame(df_4h, "4h", expected=exp)
    if df_daily is not None and source == "binance":
        out += check_frame(df_daily, "1d", expected=exp, suffix=":1d")
    return out


# ---------------------------------------------------------------------------
# A) Registro en la BD y avisos graves
# ---------------------------------------------------------------------------

_alert_state: dict = {}        # symbol -> última hora de aviso (pd.Timestamp)
_alerted_keys: set = set()     # (symbol, check, ts) ya avisados en este proceso
_ALERTED_MAX = 2000
# Avisos graves PENDIENTES: su envío por Telegram falló. "Nueva en la BD" y "aviso
# enviado" son cosas distintas: si el envío falla, la incidencia ya está en
# data_quality y en el ciclo siguiente no sería nueva, así que sin esto el aviso
# se perdería. (symbol, check, ts) -> (incidencia, hora del primer fallo). Se
# reintenta en los ciclos siguientes (con el mismo throttling de 6 h) hasta
# PENDING_MAX_AGE; después solo queda en el resumen nocturno. Vive en la memoria
# del proceso (un reinicio del worker lo pierde: queda el resumen nocturno).
_pending: dict = {}
PENDING_MAX_AGE = timedelta(hours=24)
_PENDING_MAX = 200


def record_incidents(symbol: str, incidents: list[dict], now=None) -> set:
    """Inserta las incidencias en data_quality (ON CONFLICT DO NOTHING). Devuelve
    las claves (check, ts_unix) que eran NUEVAS. Lanza si la BD falla."""
    if not incidents:
        return set()
    from data.database import get_engine, init_db, _to_unix
    init_db()
    created = _to_unix(_now(now))
    sql = text("INSERT INTO data_quality (symbol, ts, check_name, detalle, grave, created_ts) "
               "VALUES (:s, :ts, :c, :d, :g, :t) ON CONFLICT (symbol, check_name, ts) DO NOTHING")
    new = set()
    with get_engine().begin() as conn:
        for inc in incidents:
            ts = _to_unix(inc["ts"])
            res = conn.execute(sql, {"s": symbol, "ts": ts, "c": inc["check"],
                                     "d": str(inc["detalle"])[:500], "g": int(bool(inc["grave"])),
                                     "t": created})
            if res.rowcount and res.rowcount > 0:
                new.add((inc["check"], ts))
    return new


def _maybe_alert(symbol: str, graves: list[dict], new_keys: set | None, notify: bool, now) -> bool:
    """Aviso inmediato por Telegram de las incidencias GRAVES nuevas, como máximo
    uno cada ALERT_EVERY por activo. new_keys=None = la BD falló (se decide solo
    con la memoria del proceso). Los graves cuyo envío falló (_pending) se
    reintentan aunque ya no sean nuevos en la BD. Devuelve True si se envió."""
    from data.database import _to_unix
    t = _now(now)
    fresh = []
    seen = set()
    for g in graves:
        key = (symbol, g["check"], _to_unix(g["ts"]))
        if key in _alerted_keys or key in seen:
            continue
        if new_keys is not None and (g["check"], key[2]) not in new_keys:
            continue      # ya estaba en la BD (si su envío falló, la recoge el bucle de _pending)
        seen.add(key)
        fresh.append((key, g))
    # Pendientes de este activo que ya no salen como graves en este ciclo (la vela
    # dejó de ser de las 2 últimas): se reintentan igual mientras no caduquen.
    for key, (g, since) in list(_pending.items()):
        if key[0] != symbol:
            continue
        if key in _alerted_keys or t - since > PENDING_MAX_AGE:
            _pending.pop(key, None)
            if key not in _alerted_keys:
                logger.warning(f"[{symbol}] Calidad de datos: el aviso de {key[1]}@{key[2]} no se pudo enviar "
                               f"en {PENDING_MAX_AGE}; queda solo en el resumen nocturno.")
            continue
        if key not in seen:
            seen.add(key)
            fresh.append((key, g))
    if not fresh or not notify:
        return False
    last = _alert_state.get(symbol)
    if last is not None and t - last < ALERT_EVERY:
        logger.info(f"[{symbol}] Calidad de datos: {len(fresh)} incidencia(s) grave(s) sin aviso "
                    f"(último aviso hace < {ALERT_EVERY}); irán al resumen nocturno.")
        return False
    from notifications.telegram import send_data_quality_alert
    try:
        ok = bool(send_data_quality_alert(symbol, [CHECK_TEXT.get(g["check"].split(":")[0], g["check"])
                                                   + (" [diario]" if g["check"].endswith(":1d") else "")
                                                   + ": " + g["detalle"] for _, g in fresh]))
    except Exception as exc:  # noqa: BLE001 — un fallo de envío es un aviso pendiente, no un error del ciclo
        logger.warning(f"[{symbol}] Calidad de datos: excepción al enviar el aviso: {exc}")
        ok = False
    if ok:
        _alert_state[symbol] = t
        if len(_alerted_keys) > _ALERTED_MAX:
            _alerted_keys.clear()
        _alerted_keys.update(k for k, _ in fresh)
        for k, _ in fresh:
            _pending.pop(k, None)
    else:
        logger.warning(f"[{symbol}] Calidad de datos: el aviso por Telegram falló; se reintentará en el "
                       "próximo ciclo.")
        if len(_pending) > _PENDING_MAX:
            _pending.clear()
        for k, g in fresh:
            _pending.setdefault(k, (g, t))
    return ok


def run_cycle_checks(symbol: str, df_4h: pd.DataFrame, df_daily: pd.DataFrame | None = None,
                     notify: bool = False, now=None) -> list[dict]:
    """Punto de entrada del ciclo (cmd_signal). Nunca lanza: devuelve las
    incidencias encontradas (lista vacía si algo falla)."""
    t0 = time.perf_counter()
    try:
        incidents = cycle_incidents(symbol, df_4h, df_daily)
    except Exception as exc:
        logger.warning(f"[{symbol}] Monitor de datos falló (no afecta a la señal): {exc}")
        return []
    ms = (time.perf_counter() - t0) * 1000
    if not incidents:
        logger.debug(f"[{symbol}] Calidad de datos: sin incidencias ({ms:.1f} ms).")
        if any(k[0] == symbol for k in _pending):
            try:                  # reintento de un aviso grave que no se pudo enviar
                _maybe_alert(symbol, [], set(), notify, now)
            except Exception as exc:
                logger.warning(f"[{symbol}] Calidad de datos: reintento del aviso falló: {exc}")
        return incidents
    resumen = "; ".join(f"{i['check']}@{i['ts']:%m-%d %H:%M}" for i in incidents[:6])
    logger.warning(f"[{symbol}] Calidad de datos: {len(incidents)} incidencia(s) ({ms:.1f} ms): {resumen}"
                   + (" …" if len(incidents) > 6 else ""))
    new_keys = None
    try:
        new_keys = record_incidents(symbol, incidents, now=now)
    except Exception as exc:
        logger.warning(f"[{symbol}] Calidad de datos: no se pudo guardar en data_quality: {exc}")
    graves = [i for i in incidents if i["grave"]]
    if graves or any(k[0] == symbol for k in _pending):
        try:
            _maybe_alert(symbol, graves, new_keys, notify, now)
        except Exception as exc:
            logger.warning(f"[{symbol}] Calidad de datos: aviso por Telegram falló: {exc}")
    return incidents


# ---------------------------------------------------------------------------
# B) Nocturno: contraste OKX vs Gate.io
# ---------------------------------------------------------------------------

def _okx_exchange():
    """Instancia ccxt de OKX SIN cargar mercados (se usa el endpoint público
    directamente: UNA petición). Separada para poder sustituirla en pruebas y
    en el arnés (proxy del sandbox)."""
    import ccxt
    return ccxt.okx({"timeout": NET_TIMEOUT_S * 1000, "enableRateLimit": False})


def _gate_exchange():
    import ccxt
    return ccxt.gate({"timeout": NET_TIMEOUT_S * 1000, "enableRateLimit": False})


def fetch_okx_closes(symbol: str, now=None, n: int = CROSS_BARS + 2) -> pd.Series:
    """Cierres de las últimas velas 4H CERRADAS del perpetuo USDT de OKX
    (índice = hora de apertura UTC). Una petición a /api/v5/market/candles."""
    inst = OKX_INST[symbol]
    raw = _okx_exchange().publicGetMarketCandles({"instId": inst, "bar": "4H", "limit": str(n)})
    rows = raw.get("data") or []
    t_now = _now(now)
    data = {}
    for r in rows:
        # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]; confirm "1" = cerrada
        ts = pd.Timestamp(int(r[0]), unit="ms", tz="UTC")
        closed = (str(r[8]) == "1") if len(r) > 8 else (ts + pd.Timedelta(hours=4) <= t_now)
        if closed and ts + pd.Timedelta(hours=4) <= t_now:
            data[ts] = float(r[4])
    return pd.Series(data, dtype=float).sort_index()


def gate_closes_from_db(symbol: str, now=None, bars: int = CROSS_BARS + 2) -> pd.Series:
    """Cierres 4H de Gate.io guardados por el ciclo en la tabla ohlcv (sin red)."""
    from data.database import get_engine, init_db
    init_db()
    t_now = _now(now)
    since = int((t_now - pd.Timedelta(hours=4 * (bars + 2))).timestamp())
    until = int((t_now - pd.Timedelta(hours=4)).timestamp())    # solo velas cerradas
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT ts, close FROM ohlcv WHERE symbol = :s AND timeframe = '4h' AND ts >= :a "
            "AND ts <= :b AND COALESCE(source, 'gateio') = 'gateio' ORDER BY ts"),
            {"s": symbol, "a": since, "b": until}).fetchall()
    return pd.Series({pd.Timestamp(int(t), unit="s", tz="UTC"): float(c) for t, c in rows},
                     dtype=float).sort_index()


def fetch_gate_closes(symbol: str, now=None, n: int = CROSS_BARS + 2) -> pd.Series:
    """Respaldo si la tabla no tiene las velas: UNA petición a Gate.io
    (/futures/usdt/candlesticks, sin cargar mercados)."""
    raw = _gate_exchange().publicFuturesGetSettleCandlesticks(
        {"settle": "usdt", "contract": GATE_CONTRACT[symbol], "interval": "4h", "limit": n + 1})
    t_now = _now(now)
    data = {}
    for r in raw or []:
        ts = pd.Timestamp(int(float(r["t"])), unit="s", tz="UTC")
        if ts + pd.Timedelta(hours=4) <= t_now:
            data[ts] = float(r["c"])
    return pd.Series(data, dtype=float).sort_index()


def compare_closes(a: pd.Series, b: pd.Series, bars: int = CROSS_BARS) -> dict:
    """Compara dos series de cierres en sus últimas `bars` velas comunes.
    Diferencia relativa = |a − b| / b (b = Gate.io, la fuente del sistema)."""
    common = a.index.intersection(b.index).sort_values()[-bars:]
    out = {"n": int(len(common)), "median": None, "max": None, "max_ts": None,
           "status": "insufficient"}
    if len(common) < CROSS_MIN_COMMON:
        return out
    d = (a.loc[common] - b.loc[common]).abs() / b.loc[common]
    out["median"] = float(d.median())
    out["max"] = float(d.max())
    out["max_ts"] = int(d.idxmax().timestamp())
    out["status"] = ("alert" if out["median"] > CROSS_MEDIAN_LIMIT or out["max"] > CROSS_MAX_LIMIT
                     else "ok")
    return out


def cross_source_check(symbol: str, now=None) -> dict:
    """Contraste OKX vs Gate.io de las últimas CROSS_BARS velas 4H cerradas.
    Nunca lanza: {"status": ok|alert|insufficient|error, ...}."""
    t_now = _now(now)
    res = {"symbol": symbol, "checked_ts": int(t_now.timestamp()), "gate_from": None}
    try:
        gate = gate_closes_from_db(symbol, now=t_now)
        res["gate_from"] = "tabla ohlcv"
        if len(gate) < CROSS_BARS:
            gate = fetch_gate_closes(symbol, now=t_now)
            res["gate_from"] = "petición a Gate.io"
        okx = fetch_okx_closes(symbol, now=t_now)
        res.update(compare_closes(okx, gate))
    except Exception as exc:
        res.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:150]}"})
    return res


# ---------------------------------------------------------------------------
# B) Nocturno: deriva del detector
# ---------------------------------------------------------------------------

def _level(n: int, p: dict) -> str:
    if n < p["p1"] or n > p["p99"]:
        return "alert"
    if n < p["p5"] or n > p["p95"]:
        return "note"
    return "ok"


def drift_status(symbol: str, now=None, conn=None, baseline: dict | None = None) -> dict:
    """Setups detectados y aprobados en los últimos DRIFT_WINDOW_DAYS días
    (shadow_setups, por fecha de la vela del setup) frente a la línea base.
    status: ok | note | alert | insufficient | no_baseline."""
    from config.settings import DRIFT_BASELINE
    base = (baseline if baseline is not None else DRIFT_BASELINE).get(symbol)
    t_now = _now(now)
    now_u = int(t_now.timestamp())
    since = now_u - DRIFT_WINDOW_DAYS * 86400
    out = {"symbol": symbol, "status": "no_baseline" if not base else "insufficient",
           "setups": None, "approved": None}
    if not base:
        return out

    def _q(c):
        first = c.execute(text("SELECT MIN(created_ts) FROM shadow_setups WHERE symbol = :s"),
                          {"s": symbol}).scalar()
        row = c.execute(text(
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN approved = 1 THEN 1 ELSE 0 END), 0) "
            "FROM shadow_setups WHERE symbol = :s AND setup_ts >= :a AND setup_ts < :b"),
            {"s": symbol, "a": since, "b": now_u}).fetchone()
        return first, row

    if conn is None:
        from data.database import get_engine, init_db
        init_db()
        with get_engine().connect() as c:
            first, row = _q(c)
    else:
        first, row = _q(conn)
    out["first_created_ts"] = None if first is None else int(first)
    if first is None or int(first) > since:
        return out
    n_set, n_app = int(row[0] or 0), int(row[1] or 0)
    lv_s, lv_a = _level(n_set, base["setups"]), _level(n_app, base["approved"])
    out.update({"setups": n_set, "approved": n_app, "level_setups": lv_s, "level_approved": lv_a,
                "baseline": base})
    order = {"ok": 0, "note": 1, "alert": 2}
    out["status"] = max((lv_s, lv_a), key=order.get)
    return out


# ---------------------------------------------------------------------------
# B) Tarea nocturna
# ---------------------------------------------------------------------------

def _drift_text(d: dict) -> str:
    b = d["baseline"]
    return (f"setups 30 d: {d['setups']} (normal {b['setups']['p5']:g}–{b['setups']['p95']:g}), "
            f"aprobados {d['approved']} (normal {b['approved']['p5']:g}–{b['approved']['p95']:g})")


def _cross_text(c: dict) -> str:
    return (f"OKX vs Gate.io: diferencia mediana {_pct(c['median'])}, máxima {_pct(c['max'])} "
            f"en {c['n']} velas 4H")


def purge_old(now=None, keep_days: int = KEEP_DAYS) -> int:
    from data.database import get_engine, init_db
    init_db()
    cutoff = int(_now(now).timestamp()) - keep_days * 86400
    with get_engine().begin() as conn:
        return int(conn.execute(text("DELETE FROM data_quality WHERE created_ts < :t"),
                                {"t": cutoff}).rowcount or 0)


def run_nightly_checks(notify: bool = False, now=None, symbols: list | None = None) -> dict:
    """Contraste de fuentes (cripto) y deriva (todos) antes del resumen nocturno.
    Guarda el contraste en kv para el resumen y avisa por Telegram de lo grave
    (contraste fuera de límites; deriva fuera de [p1, p99]). Nunca lanza."""
    from config.settings import active_assets, ASSETS, notify_enabled
    t_now = _now(now)
    symbols = active_assets() if symbols is None else symbols
    report = {"cross": {}, "drift": {}, "alerts": []}
    for sym in symbols:
        try:
            if ASSETS.get(sym, {}).get("source", "binance") == "binance" and sym in OKX_INST:
                c = cross_source_check(sym, now=t_now)
                report["cross"][sym] = c
                try:
                    from data.database import kv_set
                    kv_set(KV_CROSS_PREFIX + sym, json.dumps(c))
                except Exception as exc:
                    logger.warning(f"[{sym}] Contraste de fuentes no guardado en kv: {exc}")
                logger.info(f"[{sym}] Contraste OKX vs Gate.io: {c.get('status')} "
                            f"(mediana {c.get('median')}, máx {c.get('max')}, n={c.get('n')})")
                if c.get("status") == "alert":
                    report["alerts"].append((sym, "Las dos fuentes no coinciden: " + _cross_text(c)
                                             + " (límites: mediana 0,5 %, cualquier vela 2 %)."))
        except Exception as exc:
            logger.warning(f"[{sym}] Contraste de fuentes falló: {exc}")
        try:
            d = drift_status(sym, now=t_now)
            report["drift"][sym] = d
            if d["status"] == "alert":
                report["alerts"].append((sym, "El detector se comporta distinto de lo normal: "
                                         + _drift_text(d) + ". Fuera del rango del 98 % del histórico."))
        except Exception as exc:
            logger.warning(f"[{sym}] Deriva del detector falló: {exc}")
    try:
        purge_old(now=t_now)
    except Exception as exc:
        logger.debug(f"data_quality: purga fallida: {exc}")
    if notify:
        for sym, msg in report["alerts"]:
            if not notify_enabled(sym):
                continue
            try:
                from notifications.telegram import send_data_quality_alert
                send_data_quality_alert(sym, [msg])
            except Exception as exc:
                logger.warning(f"[{sym}] Aviso nocturno de calidad de datos falló: {exc}")
    return report


# ---------------------------------------------------------------------------
# C) Bloque "Calidad de datos" del resumen nocturno
# ---------------------------------------------------------------------------

def _incidents_24h(conn, symbol: str, since: int) -> list:
    return conn.execute(text(
        "SELECT check_name, ts, grave FROM data_quality WHERE symbol = :s AND created_ts >= :t "
        "ORDER BY ts DESC"), {"s": symbol, "t": since}).fetchall()


def _cross_from_kv(symbol: str, now_unix: int) -> dict | None:
    from data.database import kv_get
    raw = kv_get(KV_CROSS_PREFIX + symbol)
    if not raw:
        return None
    c = json.loads(raw)
    if now_unix - int(c.get("checked_ts") or 0) > CROSS_STALE_HOURS * 3600:
        return {"status": "stale"}
    return c


def quality_data(conn, symbols: list, now_unix: int) -> dict:
    """Datos estructurados del bloque (para format_quality_block)."""
    from config.settings import ASSETS
    since = now_unix - 24 * 3600
    out = {}
    for sym in symbols:
        item = {"session": ASSETS.get(sym, {}).get("source") == "yfinance"}
        try:
            item["incidents"] = _incidents_24h(conn, sym, since)
        except Exception as exc:
            item["incidents_error"] = type(exc).__name__
        if not item["session"] and sym in OKX_INST:
            try:
                item["cross"] = _cross_from_kv(sym, now_unix)
            except Exception as exc:
                item["cross"] = {"status": "error", "error": type(exc).__name__}
        try:
            item["drift"] = drift_status(sym, now=pd.Timestamp(now_unix, unit="s", tz="UTC"), conn=conn)
        except Exception as exc:
            item["drift"] = {"status": "error", "error": type(exc).__name__}
        out[sym] = item
    return out


def format_quality_block(data: dict) -> list[str]:
    """Líneas HTML del bloque "Calidad de datos" (texto fijo; ver la nota del
    redactor opcional en el docstring del módulo)."""
    lines = ["<b>Calidad de datos</b>"]
    any_session = False
    for sym, it in data.items():
        parts = []
        if "incidents_error" in it:
            parts.append(f"incidencias no disponibles ({_e(it['incidents_error'])})")
        else:
            inc = it.get("incidents") or []
            if not inc:
                parts.append("sin incidencias en 24 h")
            else:
                counts: dict = {}
                for chk, _ts, _g in inc:
                    base = chk.split(":")[0]
                    lab = CHECK_SHORT.get(base, base) + (" (diario)" if chk.endswith(":1d") else "")
                    counts[lab] = counts.get(lab, 0) + 1
                n_grave = sum(1 for r in inc if r[2])
                txt = ", ".join(f"{k} ×{v}" if v > 1 else k for k, v in counts.items())
                parts.append(f"⚠️ {len(inc)} incidencia(s) en 24 h: {_e(txt)}"
                             + (f" ({n_grave} grave(s))" if n_grave else ""))
        if it.get("session"):
            any_session = True
        elif "cross" in it:
            c = it["cross"]
            if c is None:
                parts.append("contraste OKX vs Gate.io: aún sin ejecutar")
            elif c.get("status") in ("ok", "alert"):
                mark = "✓" if c["status"] == "ok" else "⚠️ NO COINCIDEN"
                parts.append(f"OKX vs Gate.io {_pct(c['median'])} (máx {_pct(c['max'])}, "
                             f"{c['n']} velas) {mark}")
            elif c.get("status") == "insufficient":
                parts.append(f"contraste OKX vs Gate.io: sin datos suficientes ({c.get('n', 0)} velas comunes)")
            elif c.get("status") == "stale":
                parts.append("contraste OKX vs Gate.io: sin contraste reciente")
            else:
                parts.append(f"contraste OKX vs Gate.io: no disponible ({_e(c.get('error', '?'))})")
        d = it.get("drift") or {}
        st = d.get("status")
        if st in ("ok", "note", "alert"):
            mark = {"ok": "✓", "note": "ℹ️ algo fuera de lo habitual",
                    "alert": "⚠️ MUY fuera de lo normal"}[st]
            parts.append(_drift_text(d) + f" {mark}")
        elif st == "insufficient":
            parts.append("deriva: sin datos suficientes (&lt; 30 días de registro sombra)")
        elif st == "no_baseline":
            parts.append("deriva: sin línea base")
        else:
            parts.append(f"deriva: no disponible ({_e(d.get('error', '?'))})")
        lines.append(f"• {_e(sym)}: " + "; ".join(parts))
    if any_session:
        lines.append("<i>Oro y SPX: sin fuente gratuita alternativa fiable para contrastar precios.</i>")
    return lines


def quality_block(conn, symbols: list, now_unix: int) -> list[str]:
    return format_quality_block(quality_data(conn, symbols, now_unix))
