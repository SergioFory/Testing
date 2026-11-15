"""
Latido para un servicio externo de avisos (Fase 1, punto 1.4).

Si se define la variable de entorno HEARTBEAT_URL (p. ej. la URL de un check de
healthchecks.io, gratuito), al final de cada ciclo CORRECTO el worker hace un
GET a esa URL. Si los pings dejan de llegar (worker caído, colgado o sin datos),
el servicio te avisa por su cuenta (email/Telegram), aunque el propio worker ya
no pueda enviar nada.

Nunca lanza ni bloquea más de `timeout` segundos. La URL no se escribe en el log
(suele llevar un identificador secreto).
"""
from __future__ import annotations

import os

from loguru import logger


def heartbeat_url() -> str:
    """URL configurada (vacía = latido desactivado). Se lee en cada llamada para
    que cambiar la variable no requiera tocar código."""
    try:
        from config.settings import HEARTBEAT_URL
        url = HEARTBEAT_URL
    except Exception:
        url = ""
    return (url or os.getenv("HEARTBEAT_URL", "")).strip()


def ping_heartbeat(url: str | None = None, timeout: float = 5.0) -> bool:
    """GET a HEARTBEAT_URL. True si respondió 2xx; False si no hay URL o falló."""
    url = heartbeat_url() if url is None else url.strip()
    if not url:
        return False
    try:
        import requests
        resp = requests.get(url, timeout=timeout)
        if 200 <= resp.status_code < 300:
            logger.debug("Latido enviado.")
            return True
        logger.warning(f"Latido: el servicio respondió {resp.status_code}.")
    except Exception as exc:
        logger.warning(f"Latido: no se pudo enviar ({type(exc).__name__}).")
    return False
