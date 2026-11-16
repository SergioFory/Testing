"""
DVOL de Deribit (índice de volatilidad implícita a 30 días de BTC y ETH).

Fase 3, punto 3.3 (E6): el VRP (prima de volatilidad = DVOL / volatilidad
realizada) se REGISTRA en sombra en la columna `features` de shadow_setups; no
filtra nada. Regla de activación en REGLAS_DE_DECISION.md ("Candidatos evaluados").

API pública, sin clave: public/get_volatility_index_data (velas de 1 h; se toma
el cierre de la última, que es el valor más reciente del índice).

Diseño (nada de esto puede tumbar ni colgar el ciclo):
  - UNA petición por moneda y ciclo, con timeout DERIBIT_DVOL["timeout_s"] (5 s).
  - Fail-silent: cualquier error → None (el registro guarda dvol/vrp = null) y
    una línea de aviso en el log.
  - Caché del ciclo: el valor (o el fallo) se reutiliza durante el ciclo; el
    worker la vacía al final de cada ciclo (clear_cycle_cache) y, por si no se
    llama, caduca sola a los CACHE_TTL_S segundos.
  - Si Deribit no responde (timeout o conexión), no se vuelve a intentar con la
    otra moneda en el mismo ciclo: como mucho se pierde un timeout por ciclo.
  - Límite de tiempo TOTAL por petición (revisión adversarial de la Fase 3): el
    timeout de requests vale para cada lectura del socket, no para la petición
    entera; una respuesta que llega byte a byte podía tardar minutos. La petición
    se hace en un hilo daemon y se espera como mucho timeout_s; si vence, None y
    la otra moneda no se pide en este ciclo (como con un timeout normal). El
    hilo abandonado termina solo (cada lectura sigue limitada por el timeout).
"""
from __future__ import annotations

import threading
import time

from loguru import logger

DERIBIT_DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
CACHE_TTL_S = 20 * 60          # menos que el ciclo horario: un valor por ciclo
LOOKBACK_MS = 3 * 3600 * 1000  # últimas 3 h de velas de 1 h (la última puede estar en curso)

_cache: dict[str, tuple[float, float | None]] = {}   # moneda → (instante monotónico, valor)
_down_until = 0.0                                     # Deribit caído: no reintentar hasta entonces


def _settings() -> dict:
    try:
        from config.settings import DERIBIT_DVOL
        return DERIBIT_DVOL
    except Exception:  # noqa: BLE001
        return {"enabled": True, "timeout_s": 5, "currencies": {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}}


def currency_for(symbol: str) -> str | None:
    """Moneda de Deribit del activo (BTCUSDT → 'BTC'); None si no tiene DVOL."""
    return (_settings().get("currencies") or {}).get(str(symbol).upper())


def clear_cycle_cache() -> None:
    """Vacía la caché del ciclo (la llama el worker al final de cada ciclo)."""
    global _down_until
    _cache.clear()
    _down_until = 0.0


def fetch_dvol(currency: str, http_get=None, now_ms: int | None = None) -> float | None:
    """
    Último valor del DVOL de `currency` ('BTC' o 'ETH') o None si no se puede
    obtener. Nunca lanza. `http_get` (por defecto requests.get) es inyectable
    para las pruebas, que no usan la red.
    """
    global _down_until
    cfg = _settings()
    if not cfg.get("enabled", True) or not currency:
        return None
    t = time.monotonic()
    hit = _cache.get(currency)
    if hit is not None and t - hit[0] < CACHE_TTL_S:
        return hit[1]
    if t < _down_until:
        return None
    value = None
    try:
        if http_get is None:
            import requests
            http_get = requests.get
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        timeout_s = float(cfg.get("timeout_s", 5))
        payload = _get_json_with_deadline(
            http_get, DERIBIT_DVOL_URL,
            params={"currency": currency, "start_timestamp": now_ms - LOOKBACK_MS,
                    "end_timestamp": now_ms, "resolution": "3600"},
            timeout_s=timeout_s)
        data = (payload.get("result") or {}).get("data") or []
        if data:
            v = float(data[-1][4])          # [ts, open, high, low, close]
            # Sanidad: el DVOL histórico de BTC/ETH ha estado entre ~30 y ~200.
            value = v if 1.0 < v < 500.0 else None
            if value is None:
                logger.warning(f"DVOL {currency}: valor fuera de rango ({v}); se registra null.")
        else:
            logger.warning(f"DVOL {currency}: Deribit no devolvió datos; se registra null.")
    except Exception as exc:  # noqa: BLE001 — fail-silent
        logger.warning(f"DVOL {currency}: Deribit no disponible ({type(exc).__name__}: {exc}); "
                       "dvol/vrp = null en este ciclo.")
        # Timeout o fallo de conexión (también ProxyError/SSLError, que heredan de
        # ConnectionError en requests): no se reintenta con la otra moneda en este ciclo.
        names = " ".join(c.__name__.lower() for c in type(exc).__mro__)
        if "timeout" in names or "connectionerror" in names:
            _down_until = t + CACHE_TTL_S
        value = None
    _cache[currency] = (t, value)
    return value


class DeribitDeadlineTimeout(TimeoutError):
    """La petición entera superó el límite total (p. ej. respuesta byte a byte)."""


def _get_json_with_deadline(http_get, url: str, params: dict, timeout_s: float) -> dict:
    """
    GET + raise_for_status + json() en un hilo daemon, esperando como mucho
    `timeout_s` segundos EN TOTAL. Si vence, lanza DeribitDeadlineTimeout (su
    nombre contiene "timeout": activa el corte de la otra moneda); si la petición
    falla, relanza su excepción. El hilo abandonado no bloquea el ciclo ni la
    salida del proceso (daemon).
    """
    box: dict = {}

    def _run():
        try:
            r = http_get(url, params=params, timeout=timeout_s)
            r.raise_for_status()
            box["json"] = r.json()
        except BaseException as exc:  # noqa: BLE001 — se relanza en el hilo principal
            box["exc"] = exc

    th = threading.Thread(target=_run, name="deribit-dvol", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise DeribitDeadlineTimeout(f"sin respuesta completa en {timeout_s:g} s")
    if "exc" in box:
        raise box["exc"]
    payload = box.get("json")
    if not isinstance(payload, dict):
        raise ValueError("respuesta JSON inesperada")
    return payload


def dvol_for(symbol: str, http_get=None) -> float | None:
    """DVOL del activo (solo BTCUSDT/ETHUSDT); None para los demás o si falla."""
    ccy = currency_for(symbol)
    return fetch_dvol(ccy, http_get=http_get) if ccy else None
