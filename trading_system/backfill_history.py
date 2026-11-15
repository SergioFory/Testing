"""
Copia propia del histórico de BTC y ETH (Fase 1, punto 1.11).

Por qué existe: Gate.io solo sirve las últimas ~10.000 velas de cada serie (en
4H, unos 4,5 años) y esa ventana se MUEVE: cada día desaparece historia con la
que se validó el sistema. Este script la guarda en la tabla `ohlcv` para no
depender de lo que el exchange quiera servir mañana.

Qué hace (manual; NO lo ejecuta el worker):
  1. Gate.io (perpetuos USDT, la fuente del sistema): todas las velas 4H y 1D
     CERRADAS que sirve hoy (máximo disponible).
  2. OKX (perpetuos USDT, vía ccxt) SOLO para fechas anteriores a la primera vela
     de Gate.io, desde --since (2018-01-01 por defecto; OKX no tiene perpetuos
     USDT antes de 2020: las ventanas vacías se saltan).
  Cada vela se guarda con su fuente en la columna ohlcv.source ('gateio'/'okx').

Reglas:
  - Por defecto es una VISTA PREVIA: descarga, compara con la BD e informa de lo
    que haría, pero NO escribe nada (ni siquiera crea la columna source).
  - --apply guarda las velas que faltan. Nunca sobrescribe filas existentes...
  - ...salvo --refresh: entonces las filas existentes cuyos valores difieren de
    lo que sirve hoy la fuente se reescriben. OKX nunca reescribe una fila de
    Gate.io (la fuente del sistema tiene prioridad).
  - Las filas existentes SIN fuente (guardadas por el worker antes de existir la
    columna) que coinciden exactamente con Gate.io se etiquetan 'gateio' con
    --apply (no cambia ningún precio, solo se registra de dónde salen).
  - Memoria: se procesa página a página (1.000 velas de Gate.io, 300 de OKX) y
    se escribe por bloques; nunca se carga la serie entera. Se usa un exchange
    a la vez (cada uno ocupa ~40 MB con sus mercados). Objetivo < 200 MB para
    convivir con el worker en los 512 MB de Render.

Volumen: cada exchange mide el volumen a su manera (contratos de distinto
tamaño), así que el volumen de OKX NO es comparable con el de Gate.io. Los
precios sí (mismos perpetuos USDT; ver LOG de la Fase 1 para la diferencia
medida en el solape). Quien use la serie larga debe mirar la columna source.

Uso (desde trading_system/, con DATABASE_URL apuntando a la BD):
  python backfill_history.py                      # vista previa BTC/ETH, 4h y 1d
  python backfill_history.py --apply              # guardar lo que falta
  python backfill_history.py --apply --refresh    # además corregir filas distintas
  python backfill_history.py --symbols BTCUSDT --timeframes 4h --since 2020-01-01
  python main.py history-stats                    # cobertura de la tabla ohlcv
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Símbolo del sistema → perpetuo USDT en ccxt (mismo en Gate.io y OKX).
SYMBOLS = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
}
TF_SECONDS = {"4h": 4 * 3600, "1d": 86400}
DEFAULT_SINCE = "2018-01-01"
SOURCES = ("gateio", "okx")
# Gate.io rechaza peticiones que empiecen más de 10.000 velas atrás
# ("Candlestick too long ago"): se deja un margen.
GATEIO_MAX_POINTS = 9_980
PAGE_LIMIT = {"gateio": 1000, "okx": 300}
PAGE_RETRIES = 3
# Tolerancias para decidir si una fila existente "coincide" con lo descargado.
_PRICE_RTOL = 1e-9
_VOLUME_RTOL = 1e-6


class PageError(RuntimeError):
    """Una página no se pudo descargar tras los reintentos."""


@dataclass
class SeriesReport:
    symbol: str
    timeframe: str
    source: str
    start_s: int
    end_s: int
    pages: int = 0
    downloaded: int = 0     # velas cerradas y válidas dentro del rango
    invalid: int = 0        # velas descartadas por precios incoherentes
    flat_zero: int = 0      # velas planas (O=H=L=C) con volumen 0: sospechosas
    new: int = 0            # no estaban en la BD
    identical: int = 0      # ya estaban con los mismos valores
    labelled: int = 0       # ... y sin fuente: se etiquetan con la fuente
    different: int = 0      # ya estaban con otros valores
    refreshed: int = 0      # ... reescritas (--refresh)
    protected: int = 0      # ... de otra fuente con prioridad: no se tocan
    first_ts: int | None = None
    last_ts: int | None = None
    gaps: int = 0           # saltos entre velas consecutivas descargadas
    missing: int = 0        # velas que faltan en esos saltos
    error: str | None = None
    skipped: str | None = None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _fmt_ts(ts: int | None, timeframe: str = "4h") -> str:
    if ts is None:
        return "—"
    d = datetime.fromtimestamp(int(ts), timezone.utc)
    return d.strftime("%Y-%m-%d") if timeframe == "1d" else d.strftime("%Y-%m-%d %H:%M")


def _n(x: int) -> str:
    """Número con separador de miles con punto (como el resto de textos)."""
    return f"{int(x):,}".replace(",", ".")


def _parse_since(value: str) -> int:
    d = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp())


def _align_up(ts: int, step: int) -> int:
    return int(math.ceil(ts / step) * step)


def _valid_candle(o, h, l, c, v) -> bool:
    vals = (o, h, l, c, v)
    if any(x is None for x in vals):
        return False
    try:
        o, h, l, c, v = (float(x) for x in vals)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(x) for x in (o, h, l, c, v)):
        return False
    return l > 0 and v >= 0 and l <= min(o, c) and h >= max(o, c)


def _same(a: float, b: float, rtol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= rtol * max(abs(float(a)), abs(float(b)), 1e-12)


def _same_candle(new: tuple, old) -> bool:
    """new = (ts, o, h, l, c, v); old = fila de la BD con open..volume."""
    return (_same(new[1], old["open"], _PRICE_RTOL) and _same(new[2], old["high"], _PRICE_RTOL)
            and _same(new[3], old["low"], _PRICE_RTOL) and _same(new[4], old["close"], _PRICE_RTOL)
            and _same(new[5], old["volume"], _VOLUME_RTOL))


def _may_overwrite(existing_source: str | None, source: str) -> bool:
    """--refresh: Gate.io (fuente del sistema) puede reescribir cualquier fila;
    OKX solo filas sin fuente o de OKX (nunca datos de Gate.io)."""
    return source == "gateio" or existing_source in (None, source)


def peak_memory_mb() -> float:
    """Memoria máxima (RSS) del proceso hasta ahora, en MB (Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Descarga paginada
# ---------------------------------------------------------------------------

def make_exchange(source: str):
    """Instancia ccxt cargando SOLO los mercados de perpetuos (swap): menos
    memoria y menos tiempo (Gate.io: 1.052 mercados en vez de 6.746)."""
    import ccxt
    config = {"options": {"defaultType": "swap", "fetchMarkets": {"types": ["swap"]}},
              "enableRateLimit": True}
    if source == "gateio":
        return ccxt.gate(config)
    if source == "okx":
        return ccxt.okx(config)
    raise ValueError(f"Fuente no soportada: {source}")


def iter_closed_pages(exchange, market: str, timeframe: str, start_s: int, end_s: int,
                      limit: int, now_s: int, retries: int = PAGE_RETRIES, sleep=time.sleep):
    """
    Genera páginas de velas CERRADAS y válidas como listas de tuplas
    (ts_segundos, open, high, low, close, volume), en orden, con
    start_s <= ts < end_s. Devuelve también cuántas se descartaron por inválidas:
    cada elemento es (filas, n_invalidas).

    Avanza por ventanas de `limit` velas desde start_s. Una ventana vacía (antes
    de que existiera el contrato, o un hueco largo del exchange) NO corta la
    descarga: se salta a la siguiente. Cada página se reintenta con espera; si
    sigue fallando se lanza PageError.

    La ÚLTIMA vela de cada página no se usa: se vuelve a pedir como primera de
    la página siguiente. Medido en Gate.io (2026-11-15): la vela que cae justo
    en el final de la ventana ('to') llega PLANA (O=H=L=C) y con volumen 0;
    pedida dentro de otra página llega con sus valores reales. Cuesta una vela
    repetida por página y vale para cualquier exchange.
    """
    step = TF_SECONDS[timeframe]
    since = _align_up(start_s, step)
    while since < end_s:
        candles = None
        for attempt in range(1, retries + 1):
            try:
                candles = exchange.fetch_ohlcv(market, timeframe, since=since * 1000, limit=limit)
                break
            except Exception as exc:  # noqa: BLE001 — se informa y se reintenta
                if attempt == retries:
                    raise PageError(f"{market} {timeframe} desde {_fmt_ts(since)}: {exc}") from exc
                sleep(2 ** attempt)
        candles = candles or []
        last = max((int(c[0] // 1000) for c in candles), default=None)
        rows, invalid = [], 0
        for c in candles:
            ts = int(c[0] // 1000)
            if last is not None and ts == last and last > since:
                continue            # última vela de la página: se vuelve a pedir
            if ts < since or ts >= end_s or ts + step > now_s:
                continue            # fuera de rango o vela aún abierta
            if not _valid_candle(*c[1:6]):
                invalid += 1
                continue
            rows.append((ts, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])))
        rows.sort()
        yield rows, invalid
        if last is None or last < since:
            since += limit * step   # ventana vacía: saltar a la siguiente
        elif last > since:
            since = last            # la última vela será la primera de la siguiente
        else:
            since = last + step     # página de una sola vela (la pedida): aceptada


# ---------------------------------------------------------------------------
# BD (por bloques)
# ---------------------------------------------------------------------------

def _ohlcv_columns(engine) -> set:
    """Columnas actuales de ohlcv sin crear nada (la vista previa no escribe)."""
    from sqlalchemy import inspect
    insp = inspect(engine)
    if not insp.has_table("ohlcv"):
        return set()
    return {c["name"] for c in insp.get_columns("ohlcv")}


def _existing_rows(conn, symbol: str, timeframe: str, lo: int, hi: int, db_cols: set) -> dict:
    """Filas de la BD en [lo, hi]. db_cols = columnas actuales de ohlcv (vacío si
    la tabla aún no existe; sin 'source' si aún no se migró)."""
    if not db_cols:
        return {}
    from sqlalchemy import text
    src = "source" if "source" in db_cols else "NULL AS source"
    rows = conn.execute(text(
        f"SELECT ts, open, high, low, close, volume, {src} FROM ohlcv "
        "WHERE symbol = :s AND timeframe = :tf AND ts >= :lo AND ts <= :hi"
    ), {"s": symbol, "tf": timeframe, "lo": int(lo), "hi": int(hi)}).mappings().fetchall()
    return {int(r["ts"]): r for r in rows}


def _oldest_gateio_ts(engine, symbol: str, timeframe: str, db_cols: set) -> int | None:
    if "source" not in db_cols:
        return None
    from sqlalchemy import text
    with engine.connect() as conn:
        v = conn.execute(text(
            "SELECT MIN(ts) FROM ohlcv WHERE symbol = :s AND timeframe = :tf AND source = 'gateio'"
        ), {"s": symbol, "tf": timeframe}).scalar()
    return int(v) if v is not None else None


_INSERT_SQL = """
    INSERT INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume, source)
    VALUES (:symbol, :timeframe, :ts, :open, :high, :low, :close, :volume, :source)
    ON CONFLICT (symbol, timeframe, ts) DO NOTHING
"""
_REFRESH_SQL = """
    UPDATE ohlcv SET open = :open, high = :high, low = :low, close = :close,
                     volume = :volume, source = :source
    WHERE symbol = :symbol AND timeframe = :timeframe AND ts = :ts
"""
_LABEL_SQL = """
    UPDATE ohlcv SET source = :source
    WHERE symbol = :symbol AND timeframe = :timeframe AND ts = :ts AND source IS NULL
"""


def _params(symbol, timeframe, source, row) -> dict:
    ts, o, h, l, c, v = row
    return {"symbol": symbol, "timeframe": timeframe, "ts": ts, "open": o, "high": h,
            "low": l, "close": c, "volume": v, "source": source}


def process_range(engine, exchange, source: str, symbol: str, timeframe: str,
                  start_s: int, end_s: int, *, apply: bool, refresh: bool, now_s: int,
                  db_cols: set, sleep=time.sleep) -> SeriesReport:
    """Descarga [start_s, end_s) de una fuente, página a página, la compara con la
    BD y (con apply) escribe cada página en su propia transacción."""
    from sqlalchemy import text
    rep = SeriesReport(symbol, timeframe, source, start_s, end_s)
    step = TF_SECONDS[timeframe]
    prev_ts = None
    try:
        pages = iter_closed_pages(exchange, SYMBOLS[symbol], timeframe, start_s, end_s,
                                  PAGE_LIMIT[source], now_s, sleep=sleep)
        for rows, invalid in pages:
            rep.pages += 1
            rep.invalid += invalid
            if not rows:
                continue
            rep.downloaded += len(rows)
            rep.first_ts = rows[0][0] if rep.first_ts is None else rep.first_ts
            rep.last_ts = rows[-1][0]
            for r in rows:
                if r[1] == r[2] == r[3] == r[4] and r[5] == 0:
                    rep.flat_zero += 1
                if prev_ts is not None and r[0] - prev_ts > step:
                    rep.gaps += 1
                    rep.missing += (r[0] - prev_ts) // step - 1
                prev_ts = r[0]

            with engine.connect() as conn:
                existing = _existing_rows(conn, symbol, timeframe, rows[0][0], rows[-1][0], db_cols)
            to_insert, to_refresh, to_label = [], [], []
            for r in rows:
                old = existing.get(r[0])
                if old is None:
                    to_insert.append(r)
                elif _same_candle(r, old):
                    rep.identical += 1
                    if old["source"] is None:
                        to_label.append(r)
                else:
                    rep.different += 1
                    if not _may_overwrite(old["source"], source):
                        rep.protected += 1
                    elif refresh:
                        to_refresh.append(r)
            rep.new += len(to_insert)
            rep.labelled += len(to_label)
            rep.refreshed += len(to_refresh)
            if apply and (to_insert or to_refresh or to_label):
                with engine.begin() as conn:
                    if to_insert:
                        conn.execute(text(_INSERT_SQL),
                                     [_params(symbol, timeframe, source, r) for r in to_insert])
                    if to_refresh:
                        conn.execute(text(_REFRESH_SQL),
                                     [_params(symbol, timeframe, source, r) for r in to_refresh])
                    if to_label:
                        conn.execute(text(_LABEL_SQL),
                                     [_params(symbol, timeframe, source, r) for r in to_label])
    except PageError as exc:
        rep.error = str(exc)
    return rep


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run_backfill(symbols=None, timeframes=None, since: str = DEFAULT_SINCE, *,
                 apply: bool = False, refresh: bool = False, sources=SOURCES,
                 exchange_factory=make_exchange, now_s: int | None = None,
                 sleep=time.sleep, log=print) -> list[SeriesReport]:
    """Rellena (o previsualiza) la copia OHLCV. Devuelve un informe por serie y fuente.

    Orden: primero Gate.io para todas las series (así se sabe dónde empieza su
    historia) y luego OKX solo para lo anterior. Un exchange a la vez."""
    from data.database import get_engine, init_db
    symbols = list(symbols or SYMBOLS)
    timeframes = list(timeframes or TF_SECONDS)
    for s in symbols:
        if s not in SYMBOLS:
            raise ValueError(f"Símbolo no soportado: {s} (admitidos: {', '.join(SYMBOLS)})")
    for tf in timeframes:
        if tf not in TF_SECONDS:
            raise ValueError(f"Timeframe no soportado: {tf} (admitidos: {', '.join(TF_SECONDS)})")
    if refresh and not apply:
        log("Aviso: --refresh sin --apply solo cambia lo que se informa (vista previa).")
    now_s = int(now_s if now_s is not None else time.time())
    since_s = _parse_since(since)

    if apply:
        init_db()                    # crea la tabla / la columna source si faltan
    engine = get_engine()
    db_cols = _ohlcv_columns(engine)

    reports: list[SeriesReport] = []
    gate_first: dict = {}

    if "gateio" in sources:
        ex = exchange_factory("gateio")
        for sym in symbols:
            for tf in timeframes:
                step = TF_SECONDS[tf]
                start = _align_up(max(since_s, now_s - GATEIO_MAX_POINTS * step), step)
                log(f"Gate.io {sym} {tf}: descargando desde {_fmt_ts(start, tf)}...")
                rep = process_range(engine, ex, "gateio", sym, tf, start, now_s, apply=apply,
                                    refresh=refresh, now_s=now_s, db_cols=db_cols, sleep=sleep)
                reports.append(rep)
                gate_first[(sym, tf)] = rep.first_ts if rep.error is None else None
        del ex
        gc.collect()

    if "okx" in sources:
        ex = None
        for sym in symbols:
            for tf in timeframes:
                if "gateio" in sources and gate_first.get((sym, tf)) is None:
                    rep = SeriesReport(sym, tf, "okx", since_s, since_s)
                    rep.skipped = ("no se sabe dónde empieza Gate.io (falló o vino vacío): "
                                   "no se rellena con OKX para no mezclar fuentes")
                    reports.append(rep)
                    continue
                bounds = [b for b in (gate_first.get((sym, tf)),
                                      _oldest_gateio_ts(engine, sym, tf, db_cols)) if b is not None]
                end = min(bounds) if bounds else now_s
                if end <= since_s:
                    rep = SeriesReport(sym, tf, "okx", since_s, end)
                    rep.skipped = f"Gate.io ya cubre desde {_fmt_ts(end, tf)} (≤ --since)"
                    reports.append(rep)
                    continue
                if ex is None:
                    ex = exchange_factory("okx")
                log(f"OKX {sym} {tf}: descargando {_fmt_ts(since_s, tf)} → {_fmt_ts(end, tf)}...")
                reports.append(process_range(engine, ex, "okx", sym, tf, since_s, end, apply=apply,
                                             refresh=refresh, now_s=now_s, db_cols=db_cols,
                                             sleep=sleep))
        del ex
        gc.collect()
    return reports


def format_report(reports: list[SeriesReport], *, apply: bool, refresh: bool) -> str:
    lines = []
    any_different = False
    for r in reports:
        head = f"{r.symbol} {r.timeframe} {r.source}:"
        if r.skipped:
            lines.append(f"{head} no se descarga — {r.skipped}.")
            continue
        if r.downloaded == 0 and not r.error:
            lines.append(f"{head} la fuente no tiene velas entre {_fmt_ts(r.start_s, r.timeframe)} y "
                         f"{_fmt_ts(r.end_s, r.timeframe)}.")
            continue
        verb_new = "guardadas" if apply else "se guardarían"
        txt = (f"{head} {_n(r.downloaded)} velas cerradas ({_fmt_ts(r.first_ts, r.timeframe)} → "
               f"{_fmt_ts(r.last_ts, r.timeframe)}) · nuevas {verb_new}: {_n(r.new)} · ya estaban iguales: "
               f"{_n(r.identical)}")
        if r.labelled:
            txt += f" ({_n(r.labelled)} sin fuente {'etiquetadas' if apply else 'se etiquetarían'} '{r.source}')"
        if r.different:
            any_different = True
            txt += f" · ya estaban con otros valores: {_n(r.different)}"
            if r.refreshed:
                txt += f" ({_n(r.refreshed)} {'reescritas' if apply else 'se reescribirían'})"
            if r.protected:
                txt += f" ({_n(r.protected)} de Gate.io, no se tocan)"
        txt += f" · huecos: {_n(r.gaps)}"
        if r.gaps:
            txt += f" (faltan {_n(r.missing)} velas)"
        if r.invalid:
            txt += f" · descartadas por precios incoherentes: {_n(r.invalid)}"
        if r.flat_zero:
            txt += f" · AVISO: {_n(r.flat_zero)} velas planas con volumen 0 (sospechosas, se guardan igual)"
        lines.append(txt)
        if r.error:
            lines.append(f"  ERROR: descarga interrumpida ({r.error}). Lo anterior es parcial; "
                         f"vuelve a ejecutar el script más tarde.")
    if any_different and not refresh:
        lines.append("Hay filas en la BD con valores distintos de los que sirve hoy la fuente (lo normal "
                     "son velas guardadas mientras aún se formaban, antes de la Fase 0, o velas planas con "
                     "volumen 0 que Gate.io devuelve en los cortes de página de la descarga completa). Se "
                     "dejan como están; para corregirlas: --apply --refresh.")
    if not apply:
        lines.append("VISTA PREVIA: no se ha escrito nada en la BD. Para guardar: "
                     "python backfill_history.py --apply")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cobertura (python main.py history-stats)
# ---------------------------------------------------------------------------

def format_history_stats(now_s: int | None = None) -> str:
    """Cobertura de la tabla ohlcv por símbolo / timeframe / fuente, con los
    huecos de las series cripto (24/7, paso fijo)."""
    from config.settings import ASSETS
    from data.database import ohlcv_coverage, ohlcv_gaps
    now_s = int(now_s if now_s is not None else time.time())
    rows = ohlcv_coverage()
    if not rows:
        return "La tabla ohlcv está vacía."
    lines = ["Copia del histórico (tabla ohlcv) — velas por símbolo, timeframe y fuente:"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["symbol"], r["timeframe"]), []).append(r)
    for (sym, tf), parts in groups.items():
        for r in parts:
            src = r["source"] or "(sin fuente)"
            lines.append(f"  {sym:<8} {tf:<3} {src:<13} {_n(r['n']):>7} velas  "
                         f"{_fmt_ts(r['first_ts'], tf)} → {_fmt_ts(r['last_ts'], tf)}")
        total = sum(int(r["n"]) for r in parts)
        first = min(int(r["first_ts"]) for r in parts)
        last = max(int(r["last_ts"]) for r in parts)
        txt = (f"  {sym:<8} {tf:<3} {'TOTAL':<13} {_n(total):>7} velas  "
               f"{_fmt_ts(first, tf)} → {_fmt_ts(last, tf)}")
        crypto = ASSETS.get(sym, {}).get("source", "binance") not in ("yfinance", "twelvedata")
        if crypto and tf in TF_SECONDS:
            g = ohlcv_gaps(sym, tf, TF_SECONDS[tf])
            txt += f"  · huecos: {_n(g['gaps'])}"
            if g["gaps"]:
                txt += f" (faltan {_n(g['missing'])} velas; el mayor de {g['max_gap_s'] / 86400:.1f} días)"
            age_h = (now_s - (last + TF_SECONDS[tf])) / 3600
            txt += (f" · última vela cerró hace {age_h:.0f} h" if age_h >= 0 else
                    " · la última fila es de una vela que aún no ha cerrado")
        lines.append(txt)
    lines.append("Nota: el volumen de OKX y el de Gate.io no son comparables (contratos distintos); "
                 "los precios sí.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copia propia del histórico 4H/1D de BTC y ETH (Gate.io + OKX para lo anterior). "
                    "Por defecto solo VISTA PREVIA.")
    p.add_argument("--apply", action="store_true", help="Escribir en la BD (sin esto no se escribe nada)")
    p.add_argument("--refresh", action="store_true",
                   help="Reescribir filas existentes con valores distintos (OKX nunca pisa Gate.io)")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=list(SYMBOLS))
    p.add_argument("--timeframes", nargs="+", default=list(TF_SECONDS), choices=list(TF_SECONDS))
    p.add_argument("--since", default=DEFAULT_SINCE, help="Fecha más antigua a pedir (AAAA-MM-DD)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    reports = run_backfill(args.symbols, args.timeframes, args.since,
                           apply=args.apply, refresh=args.refresh)
    print()
    print(format_report(reports, apply=args.apply, refresh=args.refresh))
    print(f"Tiempo: {time.time() - t0:.0f} s · memoria máxima del proceso: {peak_memory_mb():.0f} MB")
    if args.apply:
        print()
        print(format_history_stats())
    return 1 if any(r.error for r in reports) else 0


if __name__ == "__main__":
    # Logs de loguru (BD, etc.) solo a partir de WARNING para no tapar el informe.
    try:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass
    sys.exit(main())
