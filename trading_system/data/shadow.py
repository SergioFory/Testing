"""
Registro sombra (Fase 1, punto 1.10).

En cada ciclo de cmd_signal se guardan en la tabla shadow_setups TODOS los
setups detectados en las 2 últimas velas cerradas (la ventana que mira el vivo),
aprobados o no, con el motivo por el que se emitieron o no. El tracker los
resuelve después con la MISMA convención que las señales
(outcome_tracker.check_outcome_detail) y con la misma caché de OHLCV del ciclo.

Para qué sirve:
  - medir también lo que NO se emitió (¿el umbral, la frescura o el filtro de
    eventos están quitando operaciones buenas o malas?);
  - tener más N para la vigilancia secuencial (aprobados = emitidos + bloqueados);
  - el resumen nocturno: motivo más frecuente de no emisión por activo.

Nada de este módulo puede tumbar el ciclo: quien lo llama lo envuelve en
try/except y aquí cada fila se procesa por separado.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from sqlalchemy import text

from data.database import get_engine, init_db, _to_unix

# Motivos posibles (ver también el comentario del DDL en data/database.py).
# 'event_blackout' ya no se produce desde la Fase 3 (filtro de eventos quitado);
# se mantiene para leer filas antiguas.
MOTIVOS = ("emitted", "shadow_asset", "below_threshold", "not_fresh",
           "event_blackout", "risk_invalid", "duplicate", "not_best")
# Motivos definitivos: una vez registrado como señal, no se sobrescribe.
FINAL_MOTIVOS = ("emitted", "shadow_asset")


def _key(ts, direction) -> tuple:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return (t, str(direction))


def classify_window(window: pd.DataFrame, already, current_price: float,
                    max_dist_atr: float | None, chosen_key=None, chosen_motivo: str | None = None,
                    symbol_motivo: str | None = None, asset_live: bool = True) -> list[dict]:
    """
    Motivo de cada setup de la ventana del ciclo (función pura, sin BD).

    Args:
        window:        setups puntuados de la ventana (ts, direction, setup_type,
                       raw_score, ml_approved, close, stop_loss, take_profit, atr).
        already:       claves (ts, direction) ya registradas en signals.
        current_price: cierre de la última vela cerrada (filtro de frescura).
        max_dist_atr:  distancia máxima entrada↔precio en ATR (como get_best_setup).
        chosen_key:    (ts, direction) del setup que eligió get_best_setup, o None.
        chosen_motivo: qué le pasó al elegido: emitted / shadow_asset /
                       risk_invalid / event_blackout / duplicate (None = sin decidir:
                       esa fila no se registra en este ciclo).
        symbol_motivo: motivo que afecta a TODO el activo en este ciclo (antes solo
                       'event_blackout', que ya no se produce desde la Fase 3); se
                       aplica a los demás aprobados ejecutables.
        asset_live:    False si el activo está en modo sombra (para los ya registrados).

    Returns:
        Lista de dicts listos para record_shadow_setups. keep_existing=True indica
        que la fila solo se inserta si no existe (no se sobrescribe su motivo).
    """
    out: list[dict] = []
    if window is None or window.empty:
        return out
    already_keys = {_key(t, d) for t, d in (already or ())}
    chosen = _key(*chosen_key) if chosen_key is not None else None
    for _, s in window.iterrows():
        try:
            key = _key(s["ts"], s["direction"])
            entry = s.get("close")
            sl, tp = s.get("stop_loss"), s.get("take_profit")
            if entry is None or sl is None or tp is None or pd.isna(entry) or pd.isna(sl) or pd.isna(tp):
                continue   # sin niveles no se puede resolver: no se registra
            atr = s.get("atr")
            approved = bool(s.get("ml_approved", True))
            keep_existing = False
            if not approved:
                motivo = "below_threshold"
            elif key in already_keys:
                # Ya registrado en signals por un ciclo anterior: su fila sombra
                # ya dice emitted/shadow_asset. Solo se inserta si falta (p. ej.
                # señales emitidas antes de existir esta tabla).
                motivo = "emitted" if asset_live else "shadow_asset"
                keep_existing = True
            elif (current_price is not None and max_dist_atr is not None and atr is not None
                  and abs(float(entry) - float(current_price)) / max(float(atr), 1e-9) > max_dist_atr):
                motivo = "not_fresh"
            elif chosen is not None and key == chosen:
                if chosen_motivo is None:
                    continue
                motivo = chosen_motivo
            elif symbol_motivo:
                motivo = symbol_motivo
            else:
                motivo = "not_best"
            raw = s.get("raw_score")
            out.append({
                "setup_ts":   key[0],
                "direction":  key[1],
                "setup_type": s.get("setup_type"),
                "raw_score":  None if raw is None or pd.isna(raw) else float(raw),
                "approved":   int(approved),
                "motivo":     motivo,
                "entry":      float(entry),
                "sl":         float(sl),
                "tp":         float(tp),
                "atr":        None if atr is None or pd.isna(atr) else float(atr),
                "keep_existing": keep_existing,
            })
        except Exception as exc:
            logger.warning(f"Registro sombra: fila no clasificable ({exc}); se omite.")
    return out


def record_shadow_setups(symbol: str, rows: list[dict], now=None) -> int:
    """
    Guarda las filas de classify_window. UNIQUE(symbol, setup_ts, direction):
      - una fila nueva se inserta;
      - una existente actualiza su motivo (el del ciclo más reciente), salvo que
        ya sea definitivo (emitted / shadow_asset) o la fila venga con
        keep_existing. Los niveles (entry/SL/TP/ATR) no se tocan nunca.
    Devuelve el número de filas escritas (insertadas o actualizadas).
    """
    if not rows:
        return 0
    init_db()
    created = _to_unix(now if now is not None else pd.Timestamp.now(tz="UTC"))
    finals = ", ".join(f"'{m}'" for m in FINAL_MOTIVOS)
    base = ("INSERT INTO shadow_setups (symbol, setup_ts, direction, setup_type, raw_score, "
            "approved, motivo, entry, sl, tp, atr, created_ts) VALUES (:symbol, :setup_ts, "
            ":direction, :setup_type, :raw_score, :approved, :motivo, :entry, :sl, :tp, :atr, "
            ":created_ts) ON CONFLICT (symbol, setup_ts, direction) ")
    sql_keep = text(base + "DO NOTHING")
    sql_upd = text(base + "DO UPDATE SET motivo = excluded.motivo, approved = excluded.approved, "
                   f"raw_score = excluded.raw_score WHERE shadow_setups.motivo NOT IN ({finals})")
    n = 0
    with get_engine().begin() as conn:
        for r in rows:
            if r["motivo"] not in MOTIVOS:
                logger.warning(f"Registro sombra: motivo desconocido {r['motivo']!r}; se omite.")
                continue
            params = {k: v for k, v in r.items() if k != "keep_existing"}
            params.update(symbol=symbol, setup_ts=_to_unix(r["setup_ts"]), created_ts=created)
            res = conn.execute(sql_keep if r.get("keep_existing") else sql_upd, params)
            n += max(0, res.rowcount or 0)
    return n


def resolve_shadow_setups(ohlcv_cache: dict | None = None, now=None) -> int:
    """
    Resuelve las filas de shadow_setups sin resultado con check_outcome_detail
    (misma convención que las señales: vela del setup = setup_ts, SL/TP desde la
    vela de entrada, salida por tiempo al cierre de i+forward_bars). Reutiliza
    `ohlcv_cache` (símbolo → DataFrame de velas cerradas) del ciclo; solo
    descarga los símbolos que falten y los añade a la caché.

    Red de seguridad igual que en resolve_open_signals: si pasado el horizonte
    con margen siguen faltando velas → 'expired' al último cierre; si ni siquiera
    hay velas que cubran el setup (2× margen) → 'expired' sin R.
    Devuelve el número de filas resueltas. Cada fila va en su propio try.
    """
    from data.outcome_tracker import (check_outcome_detail, compute_result_fields,
                                      safety_net_days, _load_outcome_ohlcv, _bar_close)
    init_db()
    cache = ohlcv_cache if ohlcv_cache is not None else {}
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT id, symbol, setup_ts, direction, entry, sl, tp FROM shadow_setups "
            "WHERE resultado IS NULL ORDER BY setup_ts ASC"
        )).fetchall()
    resolved = 0
    for sid, symbol, setup_unix, direction, entry, sl, tp in rows:
        try:
            setup_ts = pd.Timestamp(int(setup_unix), unit="s", tz="UTC")
            df = cache.get(symbol)
            if df is None or df.empty:
                df = _load_outcome_ohlcv(symbol)
                cache[symbol] = df
            if df is None or df.empty:
                continue
            res = check_outcome_detail(symbol, setup_ts, direction, entry, sl, tp,
                                       df=df, setup_ts=setup_ts)
            age_days = (now - setup_ts).total_seconds() / 86400
            if res is None and age_days > safety_net_days(symbol):
                after = df[df.index > setup_ts]
                if not after.empty and df.index[0] <= setup_ts:
                    res = {"outcome": "expired", "exit_price": float(after["close"].iloc[-1]),
                           "exit_ts": _bar_close(symbol, after.index[-1])}
                elif age_days > 2 * safety_net_days(symbol):
                    _update(sid, "expired", None, None, None)
                    resolved += 1
                    continue
            if res is None:
                continue
            f = compute_result_fields(direction, entry, sl, res["exit_price"])
            _update(sid, res["outcome"], f["pnl_r"], res["exit_price"], res.get("exit_ts"))
            resolved += 1
        except Exception as exc:
            logger.warning(f"Registro sombra: error resolviendo la fila #{sid}: {exc}")
    if resolved:
        logger.info(f"Registro sombra: {resolved} setup(s) resuelto(s).")
    return resolved


def _update(row_id: int, outcome: str, pnl_r, exit_price, resolved_ts) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(
            "UPDATE shadow_setups SET resultado=:o, pnl_r=:r, exit_price=:p, resolved_ts=:t "
            "WHERE id=:id"
        ), {"o": outcome, "r": pnl_r, "p": exit_price,
            "t": _to_unix(resolved_ts) if resolved_ts is not None else None, "id": int(row_id)})


def load_shadow_setups(symbol: str | None = None, since=None) -> pd.DataFrame:
    """Filas de shadow_setups (opcionalmente de un símbolo y con setup_ts >= since)."""
    init_db()
    query, params, where = "SELECT * FROM shadow_setups", {}, []
    if symbol:
        where.append("symbol = :s")
        params["s"] = symbol
    if since is not None:
        where.append("setup_ts >= :since")
        params["since"] = _to_unix(since)
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY setup_ts ASC, id ASC"
    with get_engine().connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params)
