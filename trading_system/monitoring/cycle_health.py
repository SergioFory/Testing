"""
Salud del ciclo horario para el latido (corrección de la revisión de la Fase 1).

Hasta ahora un ciclo era "correcto" si ningún cmd_signal lanzaba y algún activo
tenía datos frescos. Pero casi todos los accesos a la BD y los envíos a Telegram
se tragan su error (para no tumbar el ciclo), así que con la BD caída tras el
arranque o con Telegram rechazando todo (bot bloqueado, token revocado) el latido
seguía llegando cada hora y nadie avisaba al usuario de que no recibía señales.

Ahora los puntos críticos anotan aquí su fallo (`mark`) y el ciclo, antes de
hacer el ping, comprueba que no haya ninguno (y hace un `SELECT 1` a la BD). Si
hay alguno, no hay latido: el servicio externo (HEARTBEAT_URL) avisa al faltar.

Qué cuenta como fallo del ciclo:
  - "bd":       una lectura/escritura CRÍTICA falló (antiduplicado, setups ya
                emitidos, guardar la señal, resolver señales abiertas, reintentos,
                offset de los comandos) o el SELECT 1 final falla;
  - "telegram": el envío o el reenvío de una SEÑAL falló, o hay una señal que
                nunca llegó y aún no se ha podido avisar de ello.
No cuentan los fallos de partes no críticas (registro sombra, copia OHLCV,
resumen), que ya se registran en el log.

Es estado de proceso (un worker, un ciclo a la vez). Nunca lanza.
"""
from __future__ import annotations

from loguru import logger

_problems: list[tuple[str, str]] = []


def reset() -> None:
    """Al empezar cada ciclo."""
    _problems.clear()


def mark(kind: str, detail: str = "") -> None:
    """Anota un fallo crítico del ciclo en curso ('bd' o 'telegram')."""
    try:
        _problems.append((str(kind), str(detail)[:200]))
    except Exception:
        pass


def problems() -> list[tuple[str, str]]:
    return list(_problems)


def db_alive() -> bool:
    """SELECT 1 contra la BD (True si responde). Nunca lanza."""
    try:
        from sqlalchemy import text
        from data.database import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1")).scalar()
        return True
    except Exception as exc:
        logger.error(f"Latido: la base de datos no responde ({type(exc).__name__}: {str(exc)[:120]}).")
        return False


def summary() -> str:
    """Texto corto para el log: 'bd: … ; telegram: …'."""
    by_kind: dict[str, list[str]] = {}
    for kind, detail in _problems:
        by_kind.setdefault(kind, []).append(detail)
    return "; ".join(f"{k} ×{len(v)} ({v[0]})" for k, v in by_kind.items())
