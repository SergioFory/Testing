"""
Limpieza ÚNICA de señales duplicadas en la tabla `signals`.

Contexto: un bug de deduplicación (ya corregido) permitía que un setup
persistente se guardara una y otra vez con el mismo symbol+direction+entry_price,
inflando el Win Rate y el PnL del dashboard.

Este script conserva SOLO la primera ocurrencia (id más bajo) de cada grupo
de duplicados y elimina las repeticiones.

Uso:
    # 1) Previsualizar qué se eliminaría (no borra nada):
    python trading_system/cleanup_duplicate_signals.py

    # 2) Ejecutar la limpieza real:
    python trading_system/cleanup_duplicate_signals.py --apply

Funciona tanto en PostgreSQL (producción) como en SQLite (desarrollo).
Pensado para ejecutarse UNA sola vez.
"""
import sys
from loguru import logger
from sqlalchemy import text

from data.database import get_engine


# Agrupamos por el setup lógico. entry_price es DOUBLE PRECISION; se redondea
# para que el agrupamiento sea robusto frente a ruido de coma flotante.
GROUP_KEY = "symbol, direction, ROUND(CAST(entry_price AS NUMERIC), 2)"


def _group_key_sqlite() -> str:
    # SQLite no soporta CAST AS NUMERIC con ROUND de 2 dígitos igual que PG,
    # pero ROUND(entry_price, 2) sí funciona.
    return "symbol, direction, ROUND(entry_price, 2)"


def preview(engine) -> list:
    """Devuelve los grupos con duplicados y cuántas filas sobran."""
    dialect = engine.dialect.name
    group_key = _group_key_sqlite() if dialect == "sqlite" else GROUP_KEY

    sql = text(f"""
        SELECT symbol, direction,
               MIN(id)   AS keep_id,
               COUNT(*)  AS total,
               MIN(entry_price) AS entry_price
        FROM signals
        GROUP BY {group_key}
        HAVING COUNT(*) > 1
        ORDER BY symbol, direction
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    return rows


def count_signals(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM signals")).scalar() or 0


def cleanup(engine) -> int:
    """Elimina duplicados conservando MIN(id) por grupo. Retorna filas borradas."""
    dialect = engine.dialect.name
    group_key = _group_key_sqlite() if dialect == "sqlite" else GROUP_KEY

    delete_sql = text(f"""
        DELETE FROM signals
        WHERE id NOT IN (
            SELECT keep_id FROM (
                SELECT MIN(id) AS keep_id
                FROM signals
                GROUP BY {group_key}
            ) AS keepers
        )
    """)
    with engine.begin() as conn:
        result = conn.execute(delete_sql)
        return result.rowcount or 0


def main():
    apply = "--apply" in sys.argv
    engine = get_engine()

    total_before = count_signals(engine)
    groups = preview(engine)
    dup_rows = sum(g.total - 1 for g in groups)

    logger.info(f"Señales totales actualmente: {total_before}")
    if not groups:
        logger.success("No hay duplicados. Nada que limpiar.")
        return

    logger.info(f"Grupos con duplicados: {len(groups)} | filas sobrantes a eliminar: {dup_rows}")
    for g in groups:
        logger.info(
            f"  {g.symbol} {g.direction.upper()} @ {g.entry_price:.2f} → "
            f"{g.total} copias (se conserva id={g.keep_id}, se borran {g.total - 1})"
        )

    if not apply:
        logger.warning(
            "MODO PREVISUALIZACIÓN — no se borró nada. "
            "Ejecuta con --apply para aplicar la limpieza."
        )
        return

    deleted = cleanup(engine)
    total_after = count_signals(engine)
    logger.success(
        f"Limpieza completada: {deleted} filas eliminadas | "
        f"señales antes={total_before} → después={total_after}"
    )


if __name__ == "__main__":
    main()
