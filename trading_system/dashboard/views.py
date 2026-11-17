"""
Funciones de SOLO LECTURA del dashboard (Fase 4, grupo B).

El dashboard ya no escribe en la BD ni envía nada por Telegram. Aquí está toda
la lógica que muestra, sin Streamlit, para poder probarla con pytest:

  - preview_signal(): VISTA de lo que ve el worker para un activo: la misma
    historia corta del vivo (main._load_data(history="live") → LIVE_HISTORY_DAYS
    en cripto), solo velas cerradas, el mismo detector, el mismo umbral
    (ml_threshold del activo), la misma ventana anclada a la última vela cerrada,
    el mismo filtro de frescura, los mismos antiduplicados y el mismo tamaño
    (TRADING_CAPITAL y el 0,25 % de RISK). Repite la decisión de
    main.cmd_signal/_emit_best_setup paso a paso pero NO guarda la señal, NO
    escribe el registro sombra ni la copia de velas y NO envía nada. Las señales
    oficiales llegan solo por Telegram (con su ID para /fill y /skip).
  - vigilance(): SPRT live y sombra, CUSUM y "configuración congelada N/150"
    (monitoring/sequential.py).
  - portfolio_view(): cartera según el sistema (risk/portfolio.py) y cuántas
    señales habría bloqueado cada regla (monitoring/daily_summary.portfolio_block).
  - executions(): /fill y /skip registrados por señal y R teórico frente a R
    ejecutado cuando la señal ya está resuelta.
  - shadow_summary(): registro sombra (conteos por motivo y resultados resueltos).
  - risk_line(): riesgo por operación (0,25 % de TRADING_CAPITAL).

Por qué se quitaron los botones que escribían (Fase 4): registrar un setup desde
el dashboard antes que el worker hacía que el worker lo descartara como
duplicado (no llegaba por Telegram ni tenía ID para /fill) y "Enviar a Telegram"
mandaba una señal sin ID. "Resolver señales pendientes" repetía lo que el worker
hace en cada ciclo (cada hora) y descargaba velas desde el proceso del
dashboard; el cierre manual sustituía el resultado del sistema en la vigilancia
(SPRT/CUSUM), que mide el sistema con la entrada del mensaje y no la ejecución
real (esa se registra con /fill y /skip). Ver REGLAS_DE_DECISION.md.

Memoria: sin ML ni macro (use_ml=False en todos los activos), sin descargas de
historia larga; de lo cargado solo se guardan las últimas velas para el gráfico.
"""
from __future__ import annotations

import re

import pandas as pd
from loguru import logger
from sqlalchemy import text

# Velas que se devuelven para el gráfico (el resto se descarta tras la vista).
CHART_BARS = 120

# Estados de la vista (preview_signal()["status"]) y su texto para el usuario.
STATUS_TEXT = {
    "no_data":      "La fuente de precios no devolvió datos: el worker se saltaría este activo en este ciclo.",
    "stale":        "La última vela cerrada es demasiado vieja (fuente congelada): el worker no emitiría nada.",
    "no_setups":    "No hay setups detectados en la historia que mira el worker.",
    "no_new":       "Ningún setup nuevo, aprobado y todavía ejecutable en la ventana del worker.",
    "risk_invalid": "Hay un setup elegido, pero no pasa las validaciones de riesgo: el worker no lo emitiría.",
    "duplicate":    "El setup elegido ya está registrado: el worker no lo repetiría.",
    "blocked":      "El control de cartera (modo enforce) bloquearía esta señal.",
    "would_emit":   "El worker emitiría esta señal en su próximo ciclo si nada cambia antes.",
    "would_shadow": "Activo en modo sombra: el worker la registraría sin enviarla por Telegram.",
}

OFFICIAL_NOTE = ("Esto es solo una vista. Las señales oficiales llegan solo por Telegram, "
                 "con su ID para responder /fill y /skip. Desde aquí no se guarda ni se envía nada.")


def _settings():
    from config import settings
    return settings


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t


def _discard_ohlcv_copies(symbol: str) -> None:
    """_load_data encola una copia de las últimas velas para guardarla al final
    del ciclo del worker; el dashboard no la guarda nunca: se descarta aquí."""
    try:
        import main
        for tf in ("4h", "1d"):
            main._ohlcv_pending.pop((symbol, tf), None)
    except Exception:  # noqa: BLE001
        pass


def preview_signal(symbol: str, loader=None) -> dict:
    """
    Vista de lo que haría el worker con el activo AHORA (sin escribir nada).

    loader: función (symbol) → (df_4h, df_daily, macro_df). Por defecto
    main._load_data(symbol, history="live"), la misma que usa el worker.

    Devuelve un dict con:
      status, message       estado (ver STATUS_TEXT) y texto llano;
      trade                 TradeSetup que emitiría (o None);
      setup_ts, last_bar_ts, data_source, current_price, threshold, history_bars,
      window                DataFrame con cada setup de la ventana y qué le pasaría
                            (mismos motivos que el registro sombra);
      chart                 últimas CHART_BARS velas (OHLCV) para el gráfico;
      chart_setups          setups detectados dentro de esas velas (ts, direction, close);
      portfolio_lines       reglas de cartera que incumpliría la señal (texto);
      mode                  'live' o 'shadow' (activo en modo sombra).
    Nunca escribe en la BD ni envía nada. Las excepciones de datos se devuelven
    como status 'no_data' con el motivo.
    """
    from config.settings import (ASSETS, MAX_ENTRY_DIST_ATR, MAX_SETUP_AGE_BARS,
                                 MAX_DATA_STALENESS_HOURS, ML_PARAMS, notify_enabled)
    from data.bars import last_bar_age_hours

    cfg = ASSETS.get(symbol, {})
    source = cfg.get("source", "binance")
    session_daily = source == "yfinance"
    mode = "live" if notify_enabled(symbol) else "shadow"
    out = {
        "symbol": symbol, "status": None, "message": "", "trade": None,
        "setup_ts": None, "last_bar_ts": None, "data_source": None,
        "current_price": None, "history_bars": 0, "mode": mode,
        "threshold": float(cfg.get("ml_threshold", ML_PARAMS["prob_threshold"])),
        "window": pd.DataFrame(), "chart": pd.DataFrame(), "chart_setups": pd.DataFrame(),
        "portfolio_lines": [], "timeframe": "1D" if session_daily else "4H",
    }

    def _done(status: str, extra: str = "") -> dict:
        out["status"] = status
        out["message"] = STATUS_TEXT[status] + (f" {extra}" if extra else "")
        return out

    # 1) Datos: la MISMA función y la misma historia que el worker.
    try:
        if loader is None:
            import main
            df_4h, df_daily, macro_df = main._load_data(symbol, history="live")
        else:
            df_4h, df_daily, macro_df = loader(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{symbol}] Vista del dashboard: datos no disponibles: {exc}")
        return _done("no_data", f"({exc})")
    finally:
        _discard_ohlcv_copies(symbol)
    if df_4h is None or df_4h.empty:
        return _done("no_data")

    out["history_bars"] = int(len(df_4h))
    out["data_source"] = df_4h.attrs.get("source", source)
    out["last_bar_ts"] = df_4h.index[-1]
    out["current_price"] = float(df_4h["close"].iloc[-1])
    out["chart"] = df_4h[["open", "high", "low", "close", "volume"]].iloc[-CHART_BARS:].copy()

    # 2) Fuente congelada (igual que cmd_signal).
    stale_key = "1d_session" if session_daily else "4h"
    age_h = last_bar_age_hours(df_4h, "4h", session=session_daily)
    if age_h > MAX_DATA_STALENESS_HOURS[stale_key]:
        return _done("stale", f"(última vela {df_4h.index[-1]}, hace {age_h:.0f} h)")

    # 3) Detector y puntuación (raw_score con el umbral del activo; ML apagado).
    from signals.detector import detect_all_setups
    from ml.predictor import score_setups, get_best_setup
    df_setups = detect_all_setups(df_4h, df_daily, symbol)
    if df_setups is None or df_setups.empty:
        return _done("no_setups")
    df_scored = score_setups(df_setups, df_4h, symbol, macro_df)
    try:
        ts_all = pd.to_datetime(df_scored["ts"], utc=True)
        first_chart = _utc(out["chart"].index[0])
        cs = df_scored[(ts_all >= first_chart).to_numpy()]
        out["chart_setups"] = cs[["ts", "direction", "close"]].copy()
    except Exception:  # noqa: BLE001
        pass

    # 4) Ventana anclada a la última vela cerrada y frescura (igual que cmd_signal).
    min_pos = max(0, len(df_4h) - 1 - MAX_SETUP_AGE_BARS)
    min_setup_ts = df_4h.index[min_pos]
    last_bar_ts = df_4h.index[-1]
    current_price = out["current_price"]
    max_dist_atr = cfg.get("max_entry_dist_atr", MAX_ENTRY_DIST_ATR)

    from data.outcome_tracker import (emitted_setup_keys, is_duplicate_setup,
                                      has_recent_legacy_alert, is_duplicate_signal)
    already = emitted_setup_keys(symbol, min_setup_ts)
    exclude = set(already or ())
    try:
        from risk.portfolio import control_mode, blocked_setup_keys
        if control_mode() == "enforce":
            exclude |= blocked_setup_keys(symbol, min_setup_ts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{symbol}] Vista: no se pudieron leer los bloqueados por cartera: {exc}")

    best = get_best_setup(df_scored, current_price=current_price, max_dist_atr=max_dist_atr,
                          min_setup_ts=min_setup_ts, max_setup_ts=last_bar_ts,
                          exclude_keys=exclude)

    chosen_key, chosen_motivo, status = None, None, None
    trade = None
    if best is None:
        status = "no_new"
    else:
        from risk.position_sizer import build_trade_setup
        setup_ts = pd.Timestamp(best["ts"])
        chosen_key = (setup_ts, best["direction"])
        out["setup_ts"] = setup_ts
        ml_score = float(best.get("ml_score", best.get("raw_score", 0)))
        # Mismo capital y riesgo que el worker (TRADING_CAPITAL; RISK: 0,25 %, 1×).
        trade = build_trade_setup(best, _settings().TRADING_CAPITAL, ml_score)
        if trade is None:
            status, chosen_motivo = "risk_invalid", "risk_invalid"
        elif (is_duplicate_setup(trade.symbol, trade.direction, setup_ts)
              or has_recent_legacy_alert(trade.symbol, trade.direction, setup_ts)
              or is_duplicate_signal(trade.symbol, trade.direction, trade.entry_price)):
            status, chosen_motivo = "duplicate", "duplicate"
        else:
            status = "would_emit" if mode == "live" else "would_shadow"
            chosen_motivo = "emitted" if mode == "live" else "shadow_asset"
            # Control de cartera: la misma comprobación (solo lee la BD).
            try:
                from risk.portfolio import check_new_signal
                chk = check_new_signal(trade, capital=_settings().TRADING_CAPITAL)
                if chk.broken:
                    estado = ("en pruebas: no bloquea" if chk.mode == "shadow" else "bloquea")
                    out["portfolio_lines"] = [f"{b['rule']}: {b['text']} ({estado})"
                                              for b in chk.broken]
                    if chk.blocks and mode == "live":
                        status, chosen_motivo = "blocked", chk.motivo
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[{symbol}] Vista: control de cartera no disponible: {exc}")
        if status in ("would_emit", "would_shadow", "blocked"):
            out["trade"] = trade
    # 5) Qué le pasaría a cada setup de la ventana (mismos motivos que el registro
    #    sombra; función pura, aquí no se guarda nada).
    try:
        from data.shadow import classify_window
        from monitoring.daily_summary import MOTIVO_TEXT
        ts = pd.to_datetime(df_scored["ts"], utc=True)
        lo, hi = _utc(min_setup_ts), _utc(last_bar_ts)
        window = df_scored[((ts >= lo) & (ts <= hi)).to_numpy()]
        rows = classify_window(window, already, current_price, max_dist_atr,
                               chosen_key=chosen_key, chosen_motivo=chosen_motivo,
                               asset_live=(mode == "live"))
        if rows:
            w = pd.DataFrame(rows)
            w["que_pasaria"] = w["motivo"].map(lambda m: MOTIVO_TEXT.get(m, m))
            out["window"] = w[["setup_ts", "direction", "setup_type", "raw_score", "approved",
                               "entry", "sl", "tp", "que_pasaria"]].rename(columns={
                                   "setup_ts": "vela del setup", "direction": "dirección",
                                   "setup_type": "tipo", "raw_score": "score",
                                   "approved": "aprobado", "entry": "entrada", "sl": "SL",
                                   "tp": "TP", "que_pasaria": "qué haría el worker"})
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{symbol}] Vista: no se pudo clasificar la ventana: {exc}")
    return _done(status)


# ---------------------------------------------------------------------------
# Texto de Telegram (HTML) → Markdown de Streamlit
# ---------------------------------------------------------------------------

def html_to_markdown(s: str) -> str:
    """Convierte el HTML sencillo de los textos de Telegram (<b>, <i>) a Markdown
    y deshace los escapes. Cada línea se separa con dos espacios + salto."""
    import html as _html
    s = re.sub(r"</?b>", "**", s or "")
    s = re.sub(r"</?i>", "_", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = _html.unescape(s)
    # Sin sangría: en Markdown 4 espacios al principio pueden volverse código.
    return "  \n".join(line.strip() for line in s.splitlines())


# ---------------------------------------------------------------------------
# Vigilancia (monitoring/sequential.py)
# ---------------------------------------------------------------------------

def vigilance() -> dict:
    """Estado de la vigilancia pre-registrada: {'state', 'rows' (DataFrame),
    'frozen' ('N/150'), 'text' (Markdown del mismo bloque del resumen nocturno)}."""
    from monitoring.sequential import monitoring_state, format_monitoring, DECISION_TEXT
    st = monitoring_state()
    rows = []
    for name, res in (("SPRT señales live", st["sprt_live"]),
                      ("SPRT aprobados del registro sombra (orientativo)", st["sprt_shadow"])):
        rows.append({"prueba": name, "señales": res.n,
                     "R medio": None if res.n == 0 else round(res.mean_r, 3),
                     "indicador": round(res.llr, 3),
                     "límites": f"{res.lower:+.2f} / {res.upper:+.2f}",
                     "decisión": DECISION_TEXT[res.decision]
                     + (f" (en la nº {res.decided_at})" if res.decided_at else "")})
    cu = st["cusum_live"]
    rows.append({"prueba": "CUSUM de deterioro (live)", "señales": cu.n, "R medio": None,
                 "indicador": round(cu.s, 3), "límites": f"alarma en {cu.h:.0f}R",
                 "decisión": (f"ALARMA (en la nº {cu.alarm_at})" if cu.alarm else "sin alarma")})
    return {"state": st, "rows": pd.DataFrame(rows),
            "frozen": f"{st['n_live']}/{st['freeze_max']}",
            "text": html_to_markdown(format_monitoring(st))}


# ---------------------------------------------------------------------------
# Cartera según el sistema (risk/portfolio.py)
# ---------------------------------------------------------------------------

def portfolio_view(now=None) -> dict:
    """{'positions' (DataFrame), 'risk_pct' (abierto, % del capital), 'day_loss_pct',
    'mode', 'text' (Markdown del bloque del resumen nocturno)}."""
    from risk.portfolio import load_open_positions, realized_loss_today, control_mode
    from monitoring.daily_summary import portfolio_block
    from data.database import get_engine, init_db
    from data.outcome_tracker import _unix
    init_db()
    now_ts = pd.Timestamp.now(tz="UTC") if now is None else _utc(now)
    opens = load_open_positions(now=now_ts)
    pos = pd.DataFrame(opens)
    if not pos.empty:
        pos["vela del setup"] = pd.to_datetime(pos["setup_ts"], unit="s", utc=True)
        pos["riesgo (% capital)"] = (pos["risk_pct"] * 100).round(3)
        pos = pos[["id", "symbol", "direction", "entry_price", "stop_loss", "position_size",
                   "vela del setup", "riesgo (% capital)"]]
    with get_engine().connect() as conn:
        lines = portfolio_block(conn, _unix(now_ts))
    return {"positions": pos,
            "risk_pct": float(sum(p["risk_pct"] for p in opens)) * 100,
            # Convención de la regla R4: pérdida POSITIVA, ganancia negativa (+0.0 quita
            # el "-0.0" de un día sin nada). Para enseñarlo, day_result_pct (signo natural).
            "day_loss_pct": float(realized_loss_today(now=now_ts)) * 100 + 0.0,
            "day_result_pct": -float(realized_loss_today(now=now_ts)) * 100 + 0.0,
            "mode": control_mode(),
            "text": html_to_markdown("\n".join(lines))}


# ---------------------------------------------------------------------------
# Ejecuciones (/fill y /skip)
# ---------------------------------------------------------------------------

def executions() -> dict:
    """
    /fill y /skip registrados, uno por señal (el ÚLTIMO registro manda, igual que
    la cartera), con el R teórico de la señal y, si tiene /fill y está resuelta,
    el R ejecutado = misma salida entrando a tu precio, en R de la señal:
    signo × (salida − tu precio) / |entrada − SL| (igual que el resumen nocturno).
    Devuelve {'table' (DataFrame), 'summary' (monitoring.daily_summary.fills_vs_theory)}.
    """
    from data.database import load_fills, get_engine
    from monitoring.daily_summary import fills_vs_theory, is_gold_symbol, GOLD_BASIS_NOTE
    from data.outcome_tracker import signal_r
    summary = fills_vs_theory()
    fills = load_fills()
    if fills.empty:
        return {"table": pd.DataFrame(), "summary": summary}
    last = fills.sort_values(["ts", "id"]).groupby("signal_id").tail(1)
    ids = [int(x) for x in last["signal_id"]]
    params = {f"i{k}": v for k, v in enumerate(ids)}
    q = ("SELECT id, symbol, direction, ts, entry_price, stop_loss, take_profit, "
         "resultado_real, exit_price, pnl_r FROM signals "
         f"WHERE id IN ({', '.join(':' + k for k in params)})")
    with get_engine().connect() as conn:
        sig = {int(r["id"]): dict(r) for r in conn.execute(text(q), params).mappings().fetchall()}
    rows = []
    for _, f in last.sort_values(["ts", "id"], ascending=False).iterrows():
        s = sig.get(int(f["signal_id"]), {})
        price = None if f["price"] is None or pd.isna(f["price"]) else float(f["price"])
        r_exec = None
        try:
            entry, sl = float(s["entry_price"]), float(s["stop_loss"])
            risk = abs(entry - sl)
            if (f["kind"] == "fill" and price is not None and risk > 0
                    and s.get("exit_price") is not None and s.get("resultado_real") is not None):
                sign = 1.0 if s.get("direction") == "long" else -1.0
                r_exec = sign * (float(s["exit_price"]) - price) / risk
        except (KeyError, TypeError, ValueError):
            pass
        # R teórico con la misma regla que las estadísticas (pnl_r o, si falta, el
        # que se deduce del resultado).
        r_theo = signal_r(s) if s.get("resultado_real") is not None else None
        rows.append({
            "señal": int(f["signal_id"]), "activo": s.get("symbol"), "dirección": s.get("direction"),
            "emitida": (pd.Timestamp(int(s["ts"]), unit="s", tz="UTC") if s.get("ts") is not None
                        else None),
            "tu registro": "/fill" if f["kind"] == "fill" else "/skip",
            "registrado": pd.Timestamp(int(f["ts"]), unit="s", tz="UTC"),
            "tu precio": price, "entrada del mensaje": s.get("entry_price"),
            "nota": f.get("note"), "resultado": s.get("resultado_real"),
            "R teórico": None if r_theo is None or pd.isna(r_theo) else round(float(r_theo), 3),
            "R ejecutado": None if r_exec is None else round(r_exec, 3),
            # Revisión de la Fase 4: en el oro el R ejecutado supone la misma salida
            # ABSOLUTA; si pusiste SL/TP a las distancias del mensaje, no es tu R real.
            "aviso": (GOLD_BASIS_NOTE.capitalize()
                      if r_exec is not None and is_gold_symbol(s.get("symbol")) else ""),
        })
    return {"table": pd.DataFrame(rows), "summary": summary}


# ---------------------------------------------------------------------------
# Registro sombra (shadow_setups)
# ---------------------------------------------------------------------------

def shadow_summary(since=None) -> dict:
    """
    Registro sombra: {'counts' (activo × motivo), 'results' (por motivo: resueltos,
    TP, SL, por tiempo, R medio), 'total', 'resolved'}. Solo lee las columnas
    necesarias (sin el JSON de variables de contexto). R medio con la misma regla
    que la vigilancia (monitoring.sequential._r_value).
    """
    from data.database import get_engine, init_db, _to_unix
    from monitoring.daily_summary import MOTIVO_TEXT
    from monitoring.sequential import _r_value
    init_db()
    q = ("SELECT symbol, motivo, approved, resultado, pnl_r, entry, sl, tp FROM shadow_setups")
    params = {}
    if since is not None:
        q += " WHERE setup_ts >= :since"
        params["since"] = _to_unix(since)
    with get_engine().connect() as conn:
        df = pd.DataFrame(conn.execute(text(q), params).fetchall(),
                          columns=["symbol", "motivo", "approved", "resultado", "pnl_r",
                                   "entry", "sl", "tp"])
    out = {"counts": pd.DataFrame(), "results": pd.DataFrame(), "total": int(len(df)),
           "resolved": int(df["resultado"].notna().sum()) if not df.empty else 0}
    if df.empty:
        return out
    df["motivo_txt"] = df["motivo"].map(lambda m: MOTIVO_TEXT.get(m, m))
    counts = pd.crosstab(df["motivo_txt"], df["symbol"], margins=True, margins_name="Total")
    counts.index.name, counts.columns.name = "motivo", None
    out["counts"] = counts
    res = df[df["resultado"].notna()].copy()
    if not res.empty:
        res["R"] = [_r_value(p, r, e, s, t) for p, r, e, s, t in
                    zip(res["pnl_r"], res["resultado"], res["entry"], res["sl"], res["tp"])]
        g = res.groupby("motivo_txt")
        tab = pd.DataFrame({
            "resueltos": g.size(),
            "TP": g["resultado"].apply(lambda s: int((s == "win").sum())),
            "SL": g["resultado"].apply(lambda s: int((s == "loss").sum())),
            "por tiempo": g["resultado"].apply(lambda s: int((s == "expired").sum())),
            "R medio": g["R"].apply(lambda s: None if s.dropna().empty
                                    else round(float(s.dropna().mean()), 3)),
        })
        tab.index.name = "motivo"
        out["results"] = tab.sort_values("resueltos", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Riesgo por operación
# ---------------------------------------------------------------------------

def signed_pct(x) -> str:
    """'+0.38 %' / '-1.25 %' / '+0.00 %' (nunca '-0.00 %'). Revisión de la Fase 4."""
    try:
        v = round(float(x), 2) + 0.0
    except (TypeError, ValueError):
        return "—"
    return f"{v:+.2f} %"


DAY_RESULT_HELP = ("Suma de las señales live cerradas hoy (UTC), en % del capital: + ganancia, − pérdida. "
                   "La regla R4 bloquea señales nuevas si la pérdida realizada del día llega al 3 %.")


def level_deltas(entry, stop_loss, take_profit) -> tuple[str, str]:
    """Distancia CON SIGNO del SL y del TP respecto a la entrada, en %: en un largo el
    SL sale '-' y el TP '+'; en un corto al revés. Revisión de la Fase 4 (antes el
    signo era fijo y en los cortos salía cambiado)."""
    e = float(entry)
    return (signed_pct((float(stop_loss) - e) / e * 100).replace(" ", ""),
            signed_pct((float(take_profit) - e) / e * 100).replace(" ", ""))


def risk_line() -> str:
    """'Riesgo por operación: 0.25 % del capital ($2.50 de $1,000); nominal máx. 1×
    el capital' (la misma línea que el resumen nocturno)."""
    from monitoring.daily_summary import risk_line as _rl
    return _rl()


def last_signals(symbol: str | None = None, limit: int = 200) -> pd.DataFrame:
    """Últimas señales guardadas por el worker (columnas útiles, horas legibles)."""
    from data.database import load_signals
    df = load_signals(symbol=symbol, limit=limit)
    if df.empty:
        return df
    df = df.copy()
    for c in ("ts", "setup_ts", "resolved_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(pd.to_numeric(df[c], errors="coerce"), unit="s", utc=True)
    keep = [c for c in ("id", "ts", "symbol", "direction", "setup_type", "mode", "notified",
                        "entry_price", "stop_loss", "take_profit", "setup_ts", "resultado_real",
                        "pnl_r", "resolved_ts", "resolved_by", "portfolio_flag", "quality_flag")
            if c in df.columns]
    return df[keep].sort_values("ts", ascending=False)
