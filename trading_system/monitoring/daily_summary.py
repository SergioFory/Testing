"""
Resumen nocturno REAL desde la base de datos (Fase 1, punto 1.4).

Sustituye al de PortfolioRiskManager, que no veía la BD y siempre decía "Sin
operaciones cerradas hoy". Por activo operativo:
  - fuente y antigüedad de la última vela cerrada (la guarda cada ciclo de
    cmd_signal en la tabla kv con record_data_status);
  - setups vistos y señales emitidas en las últimas 24 h;
  - motivo más frecuente de no emisión (registro sombra, shadow_setups);
  - señales abiertas (con la hora límite de cierre por tiempo).
Y en conjunto: resultados de las últimas 24 h en R, estado de la vigilancia
SPRT/CUSUM (monitoring/sequential.py) y fills frente a lo teórico.
Fase 2 (2.3): bloque "Calidad de datos" por activo (monitoring/data_monitor.py).

Solo lee la BD (consultas pequeñas, sin descargas). Cada bloque va en su propio
try: si uno falla, el resto del resumen sale igual.
"""
from __future__ import annotations

import html
import json

import pandas as pd
from loguru import logger
from sqlalchemy import text

KV_DATA_PREFIX = "datos:"          # kv: estado de datos por activo (JSON)
KV_LAST_CYCLE_OK = "ultimo_ciclo_ok"  # kv: Unix del último ciclo correcto
TELEGRAM_MAX_CHARS = 4000

MOTIVO_TEXT = {
    "emitted":         "emitido",
    "shadow_asset":    "activo en modo sombra",
    "below_threshold": "score bajo el umbral",
    "not_fresh":       "precio ya lejos de la entrada",
    "event_blackout":  "evento macro cercano",
    "risk_invalid":    "no pasa el control de riesgo",
    "duplicate":       "duplicado",
    "not_best":        "salió otro de mayor score",
}
RESULT_TEXT = {"win": "ganada (TP)", "loss": "perdida (SL)", "expired": "cierre por tiempo",
               "be": "break-even"}


def _e(v) -> str:
    return html.escape(str(v))


def _unix_now(now) -> int:
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def _fmt_ts(unix) -> str:
    return pd.Timestamp(int(unix), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")


def _fmt_hours(h: float) -> str:
    if h < 1:
        return f"{h * 60:.0f} min"
    if h < 48:
        return f"{h:.1f} h"
    return f"{h / 24:.1f} días"


# ---------------------------------------------------------------------------
# Estado de datos por activo (lo escribe cmd_signal en cada ciclo)
# ---------------------------------------------------------------------------

def record_data_status(symbol: str, source: str | None, last_bar_ts=None,
                       last_bar_close=None, problem: str | None = None, now=None) -> None:
    """Guarda en kv la fuente y la última vela cerrada vista por el ciclo. Si hay
    un problema de datos y no llegó vela, conserva la última conocida. Nunca lanza."""
    try:
        from data.database import kv_get, kv_set, _to_unix
        key = KV_DATA_PREFIX + symbol
        prev = {}
        try:
            raw = kv_get(key)
            prev = json.loads(raw) if raw else {}
        except Exception:
            prev = {}
        st = {
            "source":         source or prev.get("source"),
            "last_bar_ts":    _to_unix(last_bar_ts) if last_bar_ts is not None else prev.get("last_bar_ts"),
            "last_bar_close": (_to_unix(last_bar_close) if last_bar_close is not None
                               else prev.get("last_bar_close")),
            "checked_ts":     _unix_now(now),
            "problem":        problem,
        }
        kv_set(key, json.dumps(st))
    except Exception as exc:
        logger.debug(f"[{symbol}] No se pudo guardar el estado de datos: {exc}")


def _data_line(symbol: str, now_unix: int) -> str:
    from data.database import kv_get
    raw = kv_get(KV_DATA_PREFIX + symbol)
    if not raw:
        return "• Datos: aún sin registro de ciclos"
    st = json.loads(raw)
    parts = [f"• Datos: {_e(st.get('source') or '?')}"]
    if st.get("last_bar_ts") is not None:
        txt = f"última vela {_fmt_ts(st['last_bar_ts'])}"
        if st.get("last_bar_close") is not None:
            txt += f" (cerró hace {_fmt_hours(max(0, now_unix - st['last_bar_close']) / 3600)})"
        parts.append(txt)
    if st.get("checked_ts") is not None:
        parts.append(f"revisado hace {_fmt_hours(max(0, now_unix - st['checked_ts']) / 3600)}")
    line = ", ".join(parts)
    if st.get("problem"):
        line += f"\n  ⚠️ {_e(st['problem'])}"
    return line


# ---------------------------------------------------------------------------
# Bloques
# ---------------------------------------------------------------------------

def _asset_block(conn, symbol: str, now_unix: int, since: int) -> list[str]:
    from config.settings import notify_enabled
    live = notify_enabled(symbol)
    lines = [f"<b>{_e(symbol)}</b>" + ("" if live else " <i>(modo sombra: no envía)</i>")]
    try:
        lines.append(_data_line(symbol, now_unix))
    except Exception as exc:
        lines.append(f"• Datos: no disponible ({_e(type(exc).__name__)})")

    seen = conn.execute(text(
        "SELECT motivo, COUNT(*) FROM shadow_setups WHERE symbol=:s AND created_ts >= :t "
        "GROUP BY motivo"), {"s": symbol, "t": since}).fetchall()
    by_motivo = {m: int(n) for m, n in seen}
    n_seen = sum(by_motivo.values())
    n_emit = conn.execute(text(
        "SELECT COUNT(*) FROM signals WHERE symbol=:s AND ts >= :t AND quality_flag IS NULL "
        "AND COALESCE(mode, 'live') = :m"), {"s": symbol, "t": since,
                                              "m": "live" if live else "shadow"}).scalar() or 0
    if live:
        emit_word = "señal emitida" if n_emit == 1 else "señales emitidas"
    else:
        emit_word = "señal registrada en sombra" if n_emit == 1 else "señales registradas en sombra"
    lines.append(f"• 24 h: {n_seen} setup{'s' if n_seen != 1 else ''} visto{'s' if n_seen != 1 else ''}, "
                 f"{n_emit} {emit_word}")
    not_emitted = {m: n for m, n in by_motivo.items() if m != "emitted"}
    if not_emitted:
        top = max(not_emitted.items(), key=lambda kv: (kv[1], kv[0]))
        lines.append(f"• Motivo más frecuente de no emisión: {_e(MOTIVO_TEXT.get(top[0], top[0]))} "
                     f"({top[1]} de {sum(not_emitted.values())})")

    open_rows = conn.execute(text(
        "SELECT id, direction, setup_ts, ts, entry_price, mode FROM signals WHERE symbol=:s "
        "AND resultado_real IS NULL AND quality_flag IS NULL ORDER BY ts ASC"), {"s": symbol}).fetchall()
    if open_rows:
        from data.outcome_tracker import time_exit_deadline
        items = []
        for sid, direction, setup_ts, ts, entry, mode in open_rows[:5]:
            txt = f"#{sid} {_e(direction)}"
            if setup_ts is not None:
                try:
                    dl = time_exit_deadline(symbol, setup_ts)
                    txt += f" (cierre por tiempo {dl:%m-%d %H:%M} UTC)"
                except Exception:
                    pass
            items.append(txt)
        more = f" y {len(open_rows) - 5} más" if len(open_rows) > 5 else ""
        lines.append(f"• Abiertas: {len(open_rows)} — " + "; ".join(items) + more)
    else:
        lines.append("• Abiertas: 0")
    return lines


def _results_block(conn, since: int) -> list[str]:
    from data.outcome_tracker import r_from_outcome
    rows = conn.execute(text(
        "SELECT id, symbol, direction, resultado_real, pnl_r, COALESCE(mode, 'live'), "
        "entry_price, stop_loss, take_profit, resolved_by FROM signals "
        "WHERE resultado_real IS NOT NULL AND quality_flag IS NULL "
        "AND COALESCE(resolved_ts, 0) >= :t ORDER BY resolved_ts ASC, id ASC"), {"t": since}).fetchall()
    lines = ["<b>Resultados de las últimas 24 h</b>"]
    live = [r for r in rows if r[5] == "live"]
    shadow = [r for r in rows if r[5] != "live"]
    if not live:
        lines.append("• Ninguna señal live resuelta.")
    else:
        total = 0.0
        for sid, sym, direction, res, pnl_r, _m, entry, sl, tp, rby in live:
            r = float(pnl_r) if pnl_r is not None else r_from_outcome(res, entry, sl, tp, resolved_by=rby)
            if r is not None:
                total += r
            rtxt = f"{r:+.2f}R" if r is not None else "sin R"
            lines.append(f"• #{sid} {_e(sym)} {_e(direction)}: {_e(RESULT_TEXT.get(res, res))} {rtxt}")
        lines.append(f"Total del día: <b>{total:+.2f}R</b> en {len(live)} señal"
                     f"{'es' if len(live) != 1 else ''} <i>(sin costes)</i>")
    if shadow:
        rs = [float(r[4]) for r in shadow if r[4] is not None]
        lines.append(f"• Activo en modo sombra: {len(shadow)} resuelta(s), {sum(rs):+.2f}R")
    sh = conn.execute(text(
        "SELECT COUNT(*), SUM(pnl_r) FROM shadow_setups WHERE approved = 1 AND resultado IS NOT NULL "
        "AND motivo NOT IN ('emitted', 'shadow_asset') AND COALESCE(resolved_ts, 0) >= :t"),
        {"t": since}).fetchone()
    if sh and sh[0]:
        lines.append(f"• Aprobados NO emitidos (registro sombra): {int(sh[0])} resuelto(s), "
                     f"{float(sh[1] or 0):+.2f}R")
    return lines


def _undelivered_block(conn, since: int) -> list[str]:
    """Señales live que NUNCA llegaron por Telegram (notified = 0): las de las
    últimas 24 h y las que siguen abiertas. Sin esto, una señal perdida por una
    caída larga de Telegram solo quedaba en el log (revisión de la Fase 1)."""
    rows = conn.execute(text(
        "SELECT id, symbol, direction, ts, entry_price, stop_loss, take_profit, resultado_real "
        "FROM signals WHERE notified = 0 AND COALESCE(mode, 'live') = 'live' "
        "AND (ts >= :t OR resultado_real IS NULL) ORDER BY ts ASC, id ASC"), {"t": since}).fetchall()
    if not rows:
        return []
    from config.settings import fmt_price
    lines = [f"⚠️ <b>Señales que NO te llegaron por Telegram: {len(rows)}</b>"]
    for sid, sym, direction, ts, entry, sl, tp, res in rows[:5]:
        estado = (f"cerrada: {_e(RESULT_TEXT.get(res, res))}" if res else "sigue abierta")
        lines.append(f"• #{sid} {_e(sym)} {_e(str(direction).upper())} del {_fmt_ts(_ts_unix(ts))}: "
                     f"entrada {_e(fmt_price(float(entry)))}, SL {_e(fmt_price(float(sl)))}, "
                     f"TP {_e(fmt_price(float(tp)))} ({estado})")
    if len(rows) > 5:
        lines.append(f"• … y {len(rows) - 5} más")
    return lines


def _ts_unix(ts) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    t = pd.Timestamp(ts)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t).timestamp())


def fills_vs_theory() -> dict:
    """Compara lo ejecutado (/fill, el más reciente por señal) con lo teórico en
    las señales ya resueltas. R ejecutado = misma salida, entrando a tu precio,
    en R de la señal: sign × (salida − tu precio) / |entrada − SL|. La diferencia
    con el R teórico es la de la entrada (no se registra tu precio de salida)."""
    from data.database import load_fills, get_engine
    fills = load_fills()
    out = {"n_fill": 0, "n_skip": 0, "n_resolved": 0, "mean_diff_r": None, "sum_diff_r": 0.0}
    if fills.empty:
        return out
    out["n_skip"] = int(fills.loc[fills["kind"] == "skip", "signal_id"].nunique())
    f = fills[fills["kind"] == "fill"].sort_values(["ts", "id"]).groupby("signal_id").tail(1)
    out["n_fill"] = int(len(f))
    if f.empty:
        return out
    ids = [int(x) for x in f["signal_id"]]
    params = {f"i{k}": v for k, v in enumerate(ids)}
    q = ("SELECT id, direction, entry_price, stop_loss, exit_price, pnl_r FROM signals "
         f"WHERE id IN ({', '.join(':' + k for k in params)}) AND resultado_real IS NOT NULL "
         "AND exit_price IS NOT NULL AND pnl_r IS NOT NULL")
    with get_engine().connect() as conn:
        sig = {int(r[0]): r for r in conn.execute(text(q), params).fetchall()}
    diffs = []
    for _, row in f.iterrows():
        s = sig.get(int(row["signal_id"]))
        if s is None or row["price"] is None or pd.isna(row["price"]):
            continue
        _id, direction, entry, sl, exit_price, pnl_r = s
        risk = abs(float(entry) - float(sl))
        if risk <= 0:
            continue
        sign = 1.0 if direction == "long" else -1.0
        r_exec = sign * (float(exit_price) - float(row["price"])) / risk
        diffs.append(r_exec - float(pnl_r))
    out["n_resolved"] = len(diffs)
    if diffs:
        out["mean_diff_r"] = sum(diffs) / len(diffs)
        out["sum_diff_r"] = sum(diffs)
    return out


def _fills_block() -> list[str]:
    fv = fills_vs_theory()
    lines = ["<b>Tu ejecución</b>"]
    if not fv["n_fill"] and not fv["n_skip"]:
        lines.append("• Sin fills registrados (responde /fill ID PRECIO o /skip ID motivo a cada señal).")
        return lines
    lines.append(f"• Señales ejecutadas (/fill): {fv['n_fill']}; no ejecutadas (/skip): {fv['n_skip']}")
    if fv["mean_diff_r"] is not None:
        lines.append(f"• R ejecutado − R teórico (entrada): {fv['mean_diff_r']:+.3f}R de media en "
                     f"{fv['n_resolved']} señal(es) resuelta(s)")
        if fv["n_resolved"] >= 20 and fv["mean_diff_r"] < -0.10:
            lines.append("  ⚠️ Regla 6: con ≥20 fills la diferencia es < −0,10R → ajustar los costes del modelo "
                         "(en el oro, descuenta antes la base futuro GC=F / contado de XM, ≈0,1R; ver reglas).")
    return lines


def build_daily_summary(now=None, symbols: list | None = None) -> str:
    """Texto HTML del resumen nocturno (sin enviarlo)."""
    from config.settings import active_assets
    from data.database import get_engine, init_db, kv_get
    init_db()
    now_unix = _unix_now(now)
    since = now_unix - 24 * 3600
    symbols = active_assets() if symbols is None else symbols
    out = [f"📊 <b>Resumen diario</b> — {_fmt_ts(now_unix)}"]
    try:
        last_ok = kv_get(KV_LAST_CYCLE_OK)
        if last_ok:
            out.append(f"Último ciclo correcto: hace {_fmt_hours(max(0, now_unix - int(float(last_ok))) / 3600)}")
        else:
            out.append("Último ciclo correcto: sin registro")
    except Exception:
        pass

    with get_engine().connect() as conn:
        try:
            undelivered = _undelivered_block(conn, since)
            if undelivered:
                out.append("")
                out.extend(undelivered)
        except Exception as exc:
            logger.warning(f"Resumen: bloque de señales no entregadas falló: {exc}")
        for sym in symbols:
            out.append("")
            try:
                out.extend(_asset_block(conn, sym, now_unix, since))
            except Exception as exc:
                logger.warning(f"Resumen: bloque de {sym} falló: {exc}")
                out.append(f"<b>{_e(sym)}</b>: no disponible ({_e(type(exc).__name__)})")
        # Calidad de datos (Fase 2, 2.3): incidencias de 24 h, contraste OKX vs
        # Gate.io (lo calcula la tarea nocturna justo antes) y deriva del detector.
        if symbols:
            out.append("")
            try:
                from monitoring.data_monitor import quality_block
                out.extend(quality_block(conn, symbols, now_unix))
            except Exception as exc:
                logger.warning(f"Resumen: bloque de calidad de datos falló: {exc}")
                out.append(f"<b>Calidad de datos</b>: no disponible ({_e(type(exc).__name__)})")
        out.append("")
        try:
            out.extend(_results_block(conn, since))
        except Exception as exc:
            logger.warning(f"Resumen: bloque de resultados falló: {exc}")
            out.append(f"<b>Resultados</b>: no disponible ({_e(type(exc).__name__)})")
    out.append("")
    try:
        from monitoring.sequential import monitoring_state, format_monitoring
        out.append(format_monitoring(monitoring_state()))
    except Exception as exc:
        logger.warning(f"Resumen: vigilancia SPRT/CUSUM falló: {exc}")
        out.append(f"🧭 Vigilancia: no disponible ({_e(type(exc).__name__)})")
    out.append("")
    try:
        out.extend(_fills_block())
    except Exception as exc:
        logger.warning(f"Resumen: bloque de fills falló: {exc}")
        out.append(f"<b>Tu ejecución</b>: no disponible ({_e(type(exc).__name__)})")
    return "\n".join(out)


def send_daily_report(now=None) -> bool:
    """Construye y envía el resumen por Telegram. Nunca lanza."""
    try:
        from notifications.telegram import send_text
        msg = build_daily_summary(now=now)
        if len(msg) > TELEGRAM_MAX_CHARS:   # límite de Telegram: 4096 caracteres
            tail = "\n… (recortado; ver: python main.py summary)"
            msg = msg[:TELEGRAM_MAX_CHARS - len(tail)].rsplit("\n", 1)[0] + tail
        return bool(send_text(msg))
    except Exception as exc:
        logger.error(f"Resumen diario: no se pudo construir o enviar: {exc}")
        return False
