"""
Calendario económico — filtro de riesgo ante eventos macro de alto impacto.

Objetivo: NO abrir operaciones justo antes/después de eventos que mueven el
mercado de forma aleatoria (NFP, CPI, decisiones de tasas FOMC/ECB/BoE/BoJ).
En esas ventanas el precio salta y revienta stops sin relación con el setup
técnico — es exactamente cuando el sistema pierde por azar.

Fuente: feed JSON semanal de ForexFactory (mirror faireconomy), sin API key.
Diseño FAIL-OPEN: si el calendario no se puede leer, el filtro queda inactivo
y nunca bloquea señales. Por eso es seguro desplegarlo aunque la fuente falle.

No se usa como feature del ML: el feed solo cubre la semana actual (sin
histórico para alinear a barras de entrenamiento). Su valor es 100% gestión
de riesgo en tiempo real.
"""
import time
from datetime import datetime, timezone, timedelta

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

    La ventana cubre [now - hours_after, now + hours_before]: se evita operar
    tanto antes del evento (volatilidad anticipada) como justo después (whipsaw).

    Returns:
        (blackout: bool, motivo: str). (False, "") si no aplica o falla la fuente.
    """
    currencies = ASSET_CURRENCIES.get(symbol.upper())
    if not currencies:                      # crypto o símbolo sin mapeo → sin filtro
        return False, ""

    now_utc = now_utc or datetime.now(timezone.utc)
    events  = fetch_calendar()
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
