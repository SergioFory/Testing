"""
Calendario económico — eventos macro de alto impacto.

Fase 3 (punto 0, decisión del usuario del 16-nov-2026): el BLOQUEO de señales por
eventos se QUITÓ de la ruta de señales (main._emit_best_setup ya no llama a
is_event_blackout; la Fase 2 midió que nunca actuaba a las horas del vivo). Este
módulo se mantiene para: la línea de aviso del mensaje (events_between), los eventos
de 48 h del resumen nocturno (upcoming_events) y el aviso de fin de calendario.
is_event_blackout queda como utilidad (mediciones y pruebas), sin uso en el vivo.

Objetivo original (histórico): NO abrir operaciones justo antes/después de eventos que mueven el
mercado de forma aleatoria (NFP, CPI, decisiones de tasas FOMC/ECB/BoE/BoJ).
En esas ventanas el precio salta y revienta stops sin relación con el setup
técnico — es exactamente cuando el sistema pierde por azar.

Fuentes (Fase 2, punto 2.4):
  1. PRINCIPAL: calendario propio `data/econ_events.csv` (FOMC, CPI y NFP de EE. UU.
     desde 2015 hasta el último calendario oficial publicado; federalreserve.gov y
     bls.gov, horas en UTC). Se lee del disco: sin red, siempre disponible.
  2. COMPLEMENTO: feed JSON semanal de ForexFactory (mirror faireconomy), sin API
     key. Añade otros eventos de alto impacto de la semana (ventas minoristas,
     discursos del presidente de la Fed…).
Diseño FAIL-OPEN: si una fuente no se puede leer, se usa la otra; si no hay
ninguna, el filtro queda inactivo y nunca bloquea señales.

Ventanas y a quién aplica: SIN CAMBIOS en la Fase 2 (2 h antes, 1 h después; solo
activos con moneda en ASSET_CURRENCIES; cripto exento). Medido en 2.4 (ver
REGLAS_DE_DECISION.md): con las horas a las que el vivo evalúa el oro y el SPX
(tras el cierre diario: 22:15–23:00 UTC en invierno o desde las 00:00 ET), un
FOMC/CPI/NFP nunca cae en la ventana, así que en la práctica el filtro no bloquea
nada con estos eventos; el mensaje de la señal lista en cambio los eventos que
caen DENTRO del plazo de la operación (events_between).

Para actualizar el CSV cuando la Fed o el BLS publiquen el calendario de un año
nuevo: añadir filas (date_utc, time_utc, event, source) convirtiendo ET→UTC con
America/New_York (decisión FOMC 14:00 ET; CPI y empleo 08:30 ET).

No se usa como feature del ML. Su valor es 100% gestión de riesgo en tiempo real.
"""
import csv
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from loguru import logger

# Mirrors del feed JSON semanal de ForexFactory. Se prueban en orden; el
# primero que responda gana. Sin API key.
CAL_URLS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json",
]

# Caché en memoria: el calendario cambia poco; se refresca cada 6h.
_CACHE = {"ts": 0.0, "events": None}
_CACHE_TTL = 6 * 3600

# Monedas cuyos eventos de alto impacto afectan a cada activo.
# Crypto (set vacío) → nunca se filtra por calendario macro.
ASSET_CURRENCIES = {
    "EURUSD": {"USD", "EUR"},
    "GBPUSD": {"USD", "GBP"},
    "USDJPY": {"USD", "JPY"},
    "XAUUSD": {"USD"},          # oro reacciona a datos de EE.UU. (NFP, CPI, FOMC)
    "XAGUSD": {"USD"},
    "SPX500": {"USD"},
    "BTCUSDT": set(),
    "ETHUSDT": set(),
}

_IMPACT_RANK = {"Low": 1, "Medium": 2, "High": 3, "Holiday": 0}

# ---------------------------------------------------------------------------
# Calendario propio (fuente principal, sin red)
# ---------------------------------------------------------------------------
EVENTS_CSV = Path(__file__).resolve().parents[1] / "econ_events.csv"
CSV_EVENT_TITLES = {
    "FOMC": "FOMC (decisión de tipos de la Fed)",
    "CPI":  "CPI (inflación de EE. UU.)",
    "NFP":  "NFP (empleo de EE. UU.)",
}
_CSV_CACHE = {"key": None, "events": None}
_MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def load_events_csv(path=None) -> list:
    """
    Eventos del calendario propio (data/econ_events.csv) como dicts con el mismo
    formato que fetch_calendar: {currency "USD", impact "High", time (UTC), title,
    event ("FOMC"/"CPI"/"NFP"), source}. Ordenados por hora. Se cachea mientras el
    archivo no cambie. Fail-open: si no se puede leer, devuelve [] (y lo registra).
    """
    p = Path(path) if path else EVENTS_CSV
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if _CSV_CACHE["key"] == key and _CSV_CACHE["events"] is not None:
            return _CSV_CACHE["events"]
        events = []
        with open(p, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    ts = datetime.strptime(f"{row['date_utc']} {row['time_utc']}",
                                           "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                code = (row.get("event") or "").strip().upper()
                events.append({"currency": "USD", "impact": "High", "time": ts,
                               "title": CSV_EVENT_TITLES.get(code, code), "event": code,
                               "source": "csv"})
        events.sort(key=lambda e: e["time"])
        _CSV_CACHE["key"], _CSV_CACHE["events"] = key, events
        return events
    except Exception as exc:
        logger.warning(f"Calendario propio {p} no disponible ({exc}); se usa solo el feed (fail-open).")
        return []


def events_between(start, end, path=None) -> list:
    """Eventos del calendario propio con start <= hora <= end (UTC), en orden."""
    s, e = _as_utc(start), _as_utc(end)
    return [ev for ev in load_events_csv(path) if s <= ev["time"] <= e]


def upcoming_events(now=None, hours: float = 48.0, path=None) -> list:
    """Eventos del calendario propio en las próximas `hours` horas desde `now`."""
    now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    return events_between(now, now + timedelta(hours=hours), path)


def calendar_last_event(path=None):
    """Hora del último evento del calendario propio (None si no se puede leer):
    a partir de ahí el CSV ya no cubre nada y hay que actualizarlo."""
    evs = load_events_csv(path)
    return evs[-1]["time"] if evs else None


def format_event_short(ev: dict) -> str:
    """'CPI 18-nov 13:30 UTC' (texto plano; el que lo meta en HTML debe escaparlo)."""
    t = ev["time"]
    return f"{ev.get('event') or ev.get('title')} {t.day:02d}-{_MESES[t.month - 1]} {t:%H:%M} UTC"


def _as_utc(t) -> datetime:
    """datetime/pd.Timestamp/str → datetime aware en UTC."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    if hasattr(t, "to_pydatetime"):
        t = t.to_pydatetime()
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def fetch_calendar(force: bool = False) -> list:
    """
    Descarga (y cachea) los eventos de la semana actual.

    Returns:
        Lista de dicts {currency, impact, time (UTC), title}. Lista vacía si
        la fuente falla y no hay caché previa (comportamiento fail-open).
    """
    now = time.time()
    if (not force and _CACHE["events"] is not None
            and now - _CACHE["ts"] < _CACHE_TTL):
        return _CACHE["events"]

    raw = None
    last_exc = None
    for url in CAL_URLS:
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.json()
            break
        except Exception as exc:
            last_exc = exc
            continue

    try:
        if raw is None:
            raise last_exc or RuntimeError("sin respuesta de los mirrors")

        events = []
        for e in raw:
            dt = e.get("date")
            if not dt:
                continue
            try:
                ts = datetime.fromisoformat(
                    dt.replace("Z", "+00:00")
                ).astimezone(timezone.utc)
            except Exception:
                continue
            events.append({
                "currency": (e.get("country") or "").upper(),  # el feed usa el código de moneda
                "impact":   (e.get("impact")  or "").capitalize(),
                "time":     ts,
                "title":    e.get("title") or "",
            })

        _CACHE["events"] = events
        _CACHE["ts"]     = now
        n_high = sum(1 for ev in events if ev["impact"] == "High")
        logger.info(f"Calendario económico: {len(events)} eventos esta semana ({n_high} de alto impacto).")
        return events

    except Exception as exc:
        logger.warning(
            f"Calendario económico no disponible ({exc}); "
            "filtro de eventos inactivo (fail-open, no se bloquea ninguna señal)."
        )
        return _CACHE["events"] or []


def is_event_blackout(symbol: str,
                      now_utc: datetime = None,
                      hours_before: float = 2.0,
                      hours_after: float = 1.0,
                      min_impact: str = "High") -> tuple:
    """
    True si hay un evento macro relevante dentro de la ventana de blackout.
    Fase 3: ya NO se usa en la ruta de señales (filtro quitado); solo mediciones/pruebas.
    Eventos = calendario propio (CSV) + feed semanal (complemento).

    La ventana cubre [now - hours_after, now + hours_before]: se evita operar
    tanto antes del evento (volatilidad anticipada) como justo después (whipsaw).

    Returns:
        (blackout: bool, motivo: str). (False, "") si no aplica o falla la fuente.
    """
    currencies = ASSET_CURRENCIES.get(symbol.upper())
    if not currencies:                      # crypto o símbolo sin mapeo → sin filtro
        return False, ""

    now_utc = _as_utc(now_utc) if now_utc is not None else datetime.now(timezone.utc)
    # Fase 2 (2.4): calendario propio primero (sin red) y el feed como complemento.
    # Cada fuente falla por separado sin lanzar (fail-open).
    events = list(load_events_csv())
    try:
        events += fetch_calendar()
    except Exception as exc:                # fetch_calendar ya no lanza; por si acaso
        logger.warning(f"Feed del calendario no disponible ({exc}); se usa solo el CSV.")
    if not events:                          # fail-open
        return False, ""

    min_rank  = _IMPACT_RANK.get(min_impact.capitalize(), 3)
    win_start = now_utc - timedelta(hours=hours_after)
    win_end   = now_utc + timedelta(hours=hours_before)

    for e in events:
        if e["currency"] not in currencies:
            continue
        if _IMPACT_RANK.get(e["impact"], 0) < min_rank:
            continue
        if win_start <= e["time"] <= win_end:
            delta_min = (e["time"] - now_utc).total_seconds() / 60.0
            when = (f"en {int(delta_min)}min" if delta_min >= 0
                    else f"hace {int(-delta_min)}min")
            return True, f"{e['currency']} {e['title']} ({e['impact']}) {when}"

    return False, ""
