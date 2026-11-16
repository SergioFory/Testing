"""
Sistema de Trading v5.0 — Punto de entrada principal.

Modos de uso:
  python main.py signal   --symbol BTCUSDT     # Señal actual
  python main.py backtest --symbol BTCUSDT     # Backtest completo
  python main.py train    --symbol BTCUSDT     # Entrenar modelo
  python main.py update                         # Actualizar todos los datos
  python main.py schedule                       # Modo daemon (señales cada hora en HH:01 UTC + 22:20)
  python main.py check-live                     # Regresión "vivo = validador"
  python main.py validate --protocol            # Puerta de validación (Fase 2, offline)
  python main.py history-stats                  # Cobertura de la copia OHLCV (backfill_history.py)
"""
import argparse
import sys
import os
from pathlib import Path

# Agregar el directorio raíz del sistema al path de Python
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from config.settings import ASSETS, BACKTEST, RISK, ML_PARAMS, DAYS_HISTORY, active_assets

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

# Log heredado en disco. Se mantiene solo como fallback de lectura para una
# transición suave; la fuente de verdad ahora es la tabla retrain_log en la DB,
# que sobrevive a reinicios del Worker (ver data/database.py).
_RETRAIN_LOG = Path(__file__).parent / "data" / "retrain_log.json"


def _get_last_train_ts(symbol: str) -> datetime | None:
    """Lee la fecha del último entrenamiento (DB primero, JSON como fallback)."""
    from data.database import get_last_train
    ts_str = get_last_train(symbol)
    if not ts_str:
        # Fallback al JSON heredado (deploy en transición que aún lo tenga en disco)
        try:
            data = json.loads(_RETRAIN_LOG.read_text()) if _RETRAIN_LOG.exists() else {}
            ts_str = data.get(symbol)
        except Exception:
            ts_str = None
    try:
        return datetime.fromisoformat(ts_str) if ts_str else None
    except Exception:
        return None


def _set_last_train_ts(symbol: str) -> None:
    """Guarda la fecha actual como último entrenamiento del símbolo (en la DB)."""
    from data.database import set_last_train
    set_last_train(symbol, datetime.now(timezone.utc).isoformat())


def _should_retrain(symbol: str) -> bool:
    """True si han pasado ML_PARAMS['retrain_every'] días desde el último entreno."""
    last = _get_last_train_ts(symbol)
    if last is None:
        return True
    days = ML_PARAMS.get("retrain_every", 7)
    return (datetime.now(timezone.utc) - last) >= timedelta(days=days)


# ---------------------------------------------------------------------------
# Helpers de datos
# ---------------------------------------------------------------------------

# Copia de las últimas velas cerradas pendiente de guardar en la tabla ohlcv
# (Fase 1): la tabla es solo una copia —nada la lee para decidir— así que se
# escribe al FINAL del ciclo, fuera de la ruta crítica, y solo ~50 velas por
# activo y timeframe (antes ~11.000 filas por activo y hora, también en
# validate y check-live).
OHLCV_COPY_BARS = 50
_ohlcv_pending: dict = {}    # (symbol, timeframe) -> DataFrame (últimas velas)


def _queue_ohlcv_copy(symbol: str, timeframe: str, df) -> None:
    if df is not None and not df.empty:
        _ohlcv_pending[(symbol, timeframe)] = df.tail(OHLCV_COPY_BARS).copy()


def _flush_ohlcv_copies() -> int:
    """Guarda en la BD las velas encoladas por _load_data. Nunca lanza: un fallo
    aquí no afecta a ninguna señal. Devuelve el número de filas nuevas."""
    from data.database import save_ohlcv
    total = 0
    while _ohlcv_pending:
        (symbol, tf), df = _ohlcv_pending.popitem()
        try:
            total += max(0, int(save_ohlcv(df, symbol, tf) or 0))
        except Exception as exc:
            logger.warning(f"[{symbol}] No se pudo guardar la copia OHLCV {tf} en la BD: {exc}")
    return total


def _load_data(symbol: str, closed_only: bool = True, history: str = "full"):
    """
    Descarga o carga datos OHLCV para el símbolo dado.

    Con closed_only=True (por defecto) se descarta la vela EN FORMACIÓN: el vivo
    debe ver lo mismo que el validador, que solo trabaja con velas cerradas
    (Fase 0, F03). La fuente usada queda en df_4h.attrs["source"].

    history (Fase 1):
      "full" → historia completa (validate, check-live, train, limpieza, update).
      "live" → ciclo vivo (cmd_signal): en cripto solo LIVE_HISTORY_DAYS días de
               Gate.io (300 de 4H y 900 de diario en vez de ~1.650 y 1.825). Se
               demostró con el arnés que da setups, raw_score, ATR y aprobaciones
               idénticos en los últimos 12 meses (ver settings). yfinance y
               Twelve Data no cambian (una sola petición).
    """
    from data.fetchers.binance_fetcher import fetch_multi_timeframe, fetch_ohlcv
    from data.fetchers.gold_fetcher    import fetch_yfinance_asset, fetch_macro
    from data.database                 import init_db
    from data.bars                     import drop_unclosed_bars
    from config.settings               import LIVE_HISTORY_DAYS

    if history not in ("full", "live"):
        raise ValueError(f"history debe ser 'full' o 'live', no {history!r}")
    init_db()

    cfg    = ASSETS.get(symbol, {})
    source = cfg.get("source", "binance")
    # La macro solo alimenta al ML; con use_ml=False no se descarga (21 series de
    # Yahoo por ciclo que no se usaban y aumentaban el riesgo de 429).
    need_macro = cfg.get("use_ml", True)

    logger.info(f"Cargando datos para {symbol}...")

    if source == "yfinance":
        # Activos no-crypto (Gold, Silver, S&P 500): datos diarios para ambos timeframes.
        yf_days  = cfg.get("days_history", 2500)
        df_daily = fetch_yfinance_asset(symbol, days=yf_days)
        if closed_only:
            df_daily = drop_unclosed_bars(df_daily, "1d", session=True)
        df_4h    = df_daily.copy() if df_daily is not None and not df_daily.empty else None
        if df_4h is not None:
            df_4h.attrs = dict(df_daily.attrs)
        macro_df = fetch_macro(days=yf_days) if need_macro else None
    elif source == "twelvedata":
        # Pares forex (EUR/USD, GBP/USD, USD/JPY): velas 4H reales vía Twelve Data.
        from data.fetchers.twelvedata_fetcher import fetch_twelvedata_ohlcv
        td_days = cfg.get("days_history", 1825)
        df_4h = fetch_twelvedata_ohlcv(symbol, interval="4h", outputsize=5000)
        if closed_only:
            df_4h = drop_unclosed_bars(df_4h, "4h")
        if df_4h is not None and not df_4h.empty:
            df_4h.attrs.setdefault("source", "twelvedata")
            df_daily = (df_4h
                        .resample("1D")
                        .agg({"open": "first", "high": "max",
                              "low": "min", "close": "last", "volume": "sum"})
                        .dropna())
        else:
            df_daily = None
        macro_df = fetch_macro(days=td_days) if need_macro else None
    else:
        crypto_days = cfg.get("days_history", DAYS_HISTORY)
        if history == "live":
            df_4h    = fetch_ohlcv(symbol, "4h", days=LIVE_HISTORY_DAYS["4h"])
            df_daily = fetch_ohlcv(symbol, "1d", days=LIVE_HISTORY_DAYS["1d"])
            df_4h    = None if df_4h is None or df_4h.empty else df_4h
            df_daily = None if df_daily is None or df_daily.empty else df_daily
        else:
            data     = fetch_multi_timeframe(symbol, ["4h", "1d"], days=crypto_days)
            df_4h    = data.get("4h")
            df_daily = data.get("1d")
        if closed_only:
            df_4h    = drop_unclosed_bars(df_4h, "4h")
            df_daily = drop_unclosed_bars(df_daily, "1d")
        use_macro = cfg.get("use_macro", True) and need_macro
        macro_df = fetch_macro(days=crypto_days) if use_macro else None

    if df_4h is None or df_4h.empty:
        logger.error(f"No se pudieron obtener datos 4H para {symbol}")
        return None, None, None

    # Copia en la BD: solo se encola aquí; se escribe al final del ciclo
    # (_flush_ohlcv_copies). En yfinance df_4h es el mismo diario: no se guarda
    # como "4h" (antes se duplicaba con una etiqueta falsa).
    if source != "yfinance":
        _queue_ohlcv_copy(symbol, "4h", df_4h)
    _queue_ohlcv_copy(symbol, "1d", df_daily)

    return df_4h, df_daily, macro_df


# ---------------------------------------------------------------------------
# Avisos de datos (Fase 0): si un activo se salta un ciclo por fallo de datos se
# avisa por Telegram, sin inundar: primer fallo, luego como máximo cada 6 h, y un
# aviso de recuperación cuando vuelve.
# ---------------------------------------------------------------------------
_DATA_ALERT_EVERY = timedelta(hours=6)
_data_alert_state: dict = {}   # symbol -> {"since": dt, "last_alert": dt}


def _report_data_problem(symbol: str, motivo: str, notify: bool) -> None:
    now = datetime.now(timezone.utc)
    st = _data_alert_state.get(symbol)
    if st is None:
        st = {"since": now, "last_alert": None}
        _data_alert_state[symbol] = st
    logger.error(f"[{symbol}] Ciclo saltado por datos: {motivo}")
    if notify and (st["last_alert"] is None or now - st["last_alert"] >= _DATA_ALERT_EVERY):
        from notifications.telegram import send_data_alert
        send_data_alert(symbol, motivo, since=st["since"])
        st["last_alert"] = now


def _report_data_ok(symbol: str, notify: bool) -> None:
    st = _data_alert_state.pop(symbol, None)
    if st and notify and st.get("last_alert") is not None:
        from notifications.telegram import send_data_recovered
        send_data_recovered(symbol, since=st["since"])


# Activos con datos frescos en el ciclo en curso (para el latido: un ciclo solo
# es "correcto" si al menos un activo tuvo datos y ninguno lanzó una excepción).
_cycle_fresh: set = set()


def _record_data_status(symbol: str, source, df_4h=None, session_daily: bool = False,
                        problem: str | None = None) -> None:
    """Fuente y última vela cerrada del ciclo → tabla kv, para el resumen nocturno
    (Fase 1, punto 1.4). Nunca lanza."""
    try:
        from monitoring.daily_summary import record_data_status
        last_ts = last_close = None
        if df_4h is not None and not df_4h.empty:
            from data.bars import bar_close_times
            last_ts = df_4h.index[-1]
            last_close = bar_close_times(df_4h.iloc[[-1]], "4h", session=session_daily)[0]
        record_data_status(symbol, source, last_ts, last_close, problem=problem)
    except Exception as exc:
        logger.debug(f"[{symbol}] Estado de datos no guardado: {exc}")


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_signal(symbol: str, notify: bool = False, ohlcv_cache: dict | None = None):
    """Genera la señal actual para el símbolo (solo con velas cerradas).

    ohlcv_cache: caché de velas del ciclo (Fase 1, 1.10). Si se pasa, se deja en
    ella el DataFrame de velas cerradas descargado aquí para que el registro
    sombra resuelva sus setups sin volver a descargar."""
    from signals.detector          import detect_all_setups
    from ml.predictor              import score_setups
    from data.bars                 import last_bar_age_hours
    from config.settings           import (MAX_ENTRY_DIST_ATR, MAX_SETUP_AGE_BARS,
                                           MAX_DATA_STALENESS_HOURS, notify_enabled)

    cfg    = ASSETS.get(symbol, {})
    source = cfg.get("source", "binance")
    session_daily = source == "yfinance"
    # Un activo en modo sombra no envía nada por Telegram, tampoco avisos de datos.
    notify_asset = notify and notify_enabled(symbol)

    df_4h, df_daily, macro_df = _load_data(symbol, history="live")
    if df_4h is None:
        motivo = "la fuente de precios no devolvió datos tras los reintentos."
        _record_data_status(symbol, source, problem=motivo)
        _report_data_problem(symbol, motivo, notify_asset)
        return

    # Fuente congelada: si la última vela cerrada es demasiado vieja, no emitir.
    stale_key = "1d_session" if session_daily else "4h"
    age_h = last_bar_age_hours(df_4h, "4h", session=session_daily)
    if age_h > MAX_DATA_STALENESS_HOURS[stale_key]:
        motivo = f"la última vela cerrada ({df_4h.index[-1]}) tiene {age_h:.0f} h; la fuente parece congelada."
        _record_data_status(symbol, df_4h.attrs.get("source", source), df_4h, session_daily, problem=motivo)
        _report_data_problem(symbol, motivo, notify_asset)
        return
    _report_data_ok(symbol, notify_asset)
    _record_data_status(symbol, df_4h.attrs.get("source", source), df_4h, session_daily)
    _cycle_fresh.add(symbol)
    if ohlcv_cache is not None:
        prev = ohlcv_cache.get(symbol)
        if prev is None or prev.empty:
            ohlcv_cache[symbol] = df_4h

    # Monitor de calidad de datos (Fase 2, 2.3): sobre las velas ya cargadas, sin
    # descargas extra (< 50 ms). Solo registra y avisa; NO cambia ni bloquea la
    # señal. Nunca lanza.
    try:
        from monitoring.data_monitor import run_cycle_checks
        run_cycle_checks(symbol, df_4h, df_daily, notify=notify_asset)
    except Exception as exc:
        logger.warning(f"[{symbol}] Monitor de datos no disponible: {exc}")

    data_source = df_4h.attrs.get("source", source)
    last_bar_ts = df_4h.index[-1]

    # Detectar setups
    df_setups = detect_all_setups(df_4h, df_daily, symbol)
    if df_setups.empty:
        logger.info(f"[{symbol}] Sin setups detectados en este momento.")
        return

    # Scoring ML
    df_scored = score_setups(df_setups, df_4h, symbol, macro_df)

    # Ventana ANCLADA A LA ÚLTIMA VELA CERRADA (Fase 0, F01): solo la vela del
    # setup puede ser la última cerrada o hasta MAX_SETUP_AGE_BARS anteriores.
    # Se cuenta por posición en el índice (en diarios salta fines de semana).
    min_pos      = max(0, len(df_4h) - 1 - MAX_SETUP_AGE_BARS)
    min_setup_ts = df_4h.index[min_pos]

    # Filtro de frescura: la entrada debe seguir cerca del precio actual (cierre
    # de la última vela CERRADA, no de la vela en formación).
    current_price = float(df_4h["close"].iloc[-1])
    max_dist_atr  = cfg.get("max_entry_dist_atr", MAX_ENTRY_DIST_ATR)

    from data.outcome_tracker import emitted_setup_keys
    already = emitted_setup_keys(symbol, min_setup_ts)

    # Registro sombra (Fase 1, punto 1.10): TODOS los setups de la ventana (las 2
    # últimas velas cerradas), aprobados o no, se guardan al final con su motivo.
    # `fate` recoge qué le pasó al setup elegido; el registro va en el finally
    # para que se guarde salga por donde salga el ciclo, y nunca lo tumba.
    fate = {"key": None, "motivo": None, "symbol_motivo": None, "portfolio_flag": None}
    try:
        _emit_best_setup(symbol, cfg, df_scored, current_price, max_dist_atr, min_setup_ts,
                         last_bar_ts, already, data_source, notify, fate)
    finally:
        _record_shadow_window(symbol, df_scored, min_setup_ts, last_bar_ts, already,
                              current_price, max_dist_atr, fate, df_4h=df_4h, df_daily=df_daily)


def _shadow_features(symbol, window, df_4h, df_daily, current_price) -> dict | None:
    """Variables de contexto de los setups de la ventana (Fase 3, 3.3): sma200_with,
    tsmom, atr_rank, dist_atr_now y, en BTC/ETH, dvol y vrp (Deribit, una petición
    por moneda y ciclo, timeout 5 s, fail-silent). Solo se registran: no cambian
    ninguna señal. Nunca lanza (si falla, la fila sombra se guarda sin features)."""
    if window is None or window.empty or df_4h is None:
        return None
    try:
        from signals.context_features import compute_setup_features
        from data.fetchers.deribit_fetcher import currency_for, dvol_for
        with_vol = currency_for(symbol) is not None
        dvol = dvol_for(symbol) if with_vol else None
        return compute_setup_features(window, df_4h, df_daily, current_price,
                                      dvol=dvol, with_vol=with_vol)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[{symbol}] Variables de contexto no disponibles (no afecta a nada): {exc}")
        return None


def _record_shadow_window(symbol, df_scored, min_setup_ts, last_bar_ts, already,
                          current_price, max_dist_atr, fate, df_4h=None, df_daily=None) -> None:
    """Guarda en shadow_setups los setups de la ventana del ciclo. Nunca lanza."""
    try:
        from data.shadow import classify_window, record_shadow_setups
        from config.settings import notify_enabled
        ts = pd.to_datetime(df_scored["ts"], utc=True)
        lo, hi = pd.Timestamp(min_setup_ts), pd.Timestamp(last_bar_ts)
        lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo
        hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi
        window = df_scored[((ts >= lo) & (ts <= hi)).to_numpy()]
        features = _shadow_features(symbol, window, df_4h, df_daily, current_price)
        rows = classify_window(window, already, current_price, max_dist_atr,
                               chosen_key=fate.get("key"), chosen_motivo=fate.get("motivo"),
                               symbol_motivo=fate.get("symbol_motivo"),
                               asset_live=notify_enabled(symbol),
                               chosen_portfolio_flag=fate.get("portfolio_flag"),
                               features=features)
        if rows:
            n = record_shadow_setups(symbol, rows)
            resumen = ", ".join(f"{r['direction']}@{r['setup_ts']:%m-%d %H:%M}={r['motivo']}" for r in rows)
            logger.info(f"[{symbol}] Registro sombra: {len(rows)} setup(s) en la ventana "
                        f"({n} escrito(s)): {resumen}")
    except Exception as exc:
        logger.warning(f"[{symbol}] Registro sombra falló (no afecta a la señal): {exc}")


def _emit_best_setup(symbol, cfg, df_scored, current_price, max_dist_atr, min_setup_ts,
                     last_bar_ts, already, data_source, notify, fate) -> None:
    """Parte final de cmd_signal: elige el mejor setup nuevo aprobado, lo valida
    (riesgo, antiduplicado), lo registra y lo notifica. Anota en `fate`
    el motivo del elegido para el registro sombra."""
    from ml.predictor              import get_best_setup
    from risk.position_sizer       import build_trade_setup
    from notifications.telegram    import safe_format_signal_alert, send_text
    from config.settings           import (MAX_SETUP_AGE_BARS, notify_enabled, fmt_price,
                                           TRADING_CAPITAL)
    from data.outcome_tracker      import (is_duplicate_signal, is_duplicate_setup,
                                           has_recent_legacy_alert, save_signal_from_trade)

    # Control de cartera en modo "enforce" (Fase 3, 3.1; hoy NO activado): un setup
    # que la cartera ya bloqueó no se reintenta en ciclos posteriores (se emitiría
    # tarde). En modo "shadow" no hay setups bloqueados y esto no hace nada.
    exclude = set(already or ())
    try:
        from risk.portfolio import control_mode, blocked_setup_keys
        if control_mode() == "enforce":
            exclude |= blocked_setup_keys(symbol, min_setup_ts)
    except Exception as exc:
        logger.warning(f"[{symbol}] Control de cartera: no se pudieron leer los bloqueados: {exc}")

    best = get_best_setup(df_scored, current_price=current_price, max_dist_atr=max_dist_atr,
                          min_setup_ts=min_setup_ts, max_setup_ts=last_bar_ts,
                          exclude_keys=exclude)
    if best is None:
        logger.info(
            f"[{symbol}] Sin setup nuevo aprobado en las últimas {MAX_SETUP_AGE_BARS + 1} velas "
            f"cerradas (desde {min_setup_ts}; última {last_bar_ts}; precio {current_price:.4f})."
        )
        return

    setup_ts = pd.Timestamp(best["ts"])
    fate["key"] = (setup_ts, best["direction"])
    ml_score = float(best.get("ml_score", best.get("raw_score", 0)))
    # Capital REAL de la cuenta (variable TRADING_CAPITAL, por defecto 10.000 USD;
    # Fase 1, 1.1). Solo cambia el tamaño en USD/unidades, no qué setup se emite.
    # Riesgo 0,25 % y apalancamiento máximo 1× (RISK; Fase 3, 3.2).
    trade = build_trade_setup(best, TRADING_CAPITAL, ml_score)

    if trade is None:
        fate["motivo"] = "risk_invalid"
        logger.info(f"[{symbol}] Setup detectado pero no pasa validaciones de riesgo.")
        return

    # Fase 3 (punto 0, decisión del usuario del 16-nov-2026): ya NO hay bloqueo por
    # eventos macro (nunca actuaba a las horas del vivo; REGLAS_DE_DECISION.md). El
    # mensaje sigue avisando de los eventos dentro del plazo (event_warning_line).

    # Antiduplicado PERMANENTE por setup (símbolo, dirección, vela del setup) y,
    # como segunda barrera, el heredado por precio.
    if is_duplicate_setup(trade.symbol, trade.direction, setup_ts) or \
       has_recent_legacy_alert(trade.symbol, trade.direction, setup_ts) or \
       is_duplicate_signal(trade.symbol, trade.direction, trade.entry_price):
        fate["motivo"] = "duplicate"
        logger.info(
            f"[{symbol}] Señal {trade.direction.upper()} @ {fmt_price(trade.entry_price)} "
            f"(setup {setup_ts}) ya registrada — se omite."
        )
        return

    mode = "live" if notify_enabled(symbol) else "shadow"

    # Control de cartera (Fase 3, 3.1; F11, F19): cartera SEGÚN EL SISTEMA (señales
    # live abiertas no saltadas con /skip) y reglas R1–R4 de PORTFOLIO_RULES. En
    # modo "shadow" (el actual) la señal sale igual: se guarda la marca y el mensaje
    # lo cuenta. En "enforce" (no activado) no se emite si incumple alguna regla
    # (solo activos que notifican). Si la comprobación falla, la señal sigue su
    # camino sin marca (nunca tumba el ciclo).
    portfolio = None
    try:
        from risk.portfolio import check_new_signal
        portfolio = check_new_signal(trade, capital=TRADING_CAPITAL)
    except Exception as exc:
        logger.warning(f"[{symbol}] Control de cartera no disponible (la señal sigue igual): {exc}")
    fate["portfolio_flag"] = portfolio.flag if portfolio is not None else None
    if portfolio is not None and portfolio.broken:
        logger.info(f"[{symbol}] Control de cartera ({portfolio.mode}): la señal incumpliría "
                    f"{portfolio.flag}: " + "; ".join(b["text"] for b in portfolio.broken))
        if portfolio.blocks and mode == "live":
            fate["motivo"] = portfolio.motivo
            logger.warning(f"[{symbol}] Señal {trade.direction.upper()} (setup {setup_ts}) NO emitida "
                           f"por el control de cartera ({portfolio.motivo}).")
            return

    # Sentimiento: APAGADO (Fase 1, SENTIMENT['enabled']=False; el veto contrarian
    # se descartó con datos). Solo se consulta si alguien lo reactiva.
    from config.settings import SENTIMENT
    sentiment = None
    if SENTIMENT.get("enabled", False):
        from sentiment.cryptopanic import get_sentiment_summary
        sentiment = get_sentiment_summary(symbol.replace("USDT", ""))

    _print_signal(trade, sentiment)

    # Notificación fiable (Fase 1): se guarda ANTES de enviar con notified=0 y el
    # texto exacto; se marca notified=1 solo cuando Telegram confirma. Si el envío
    # falla, el ciclo siguiente la reenvía (retry_failed_signal_alerts) mientras
    # tenga menos de 2 h.
    will_notify = bool(notify) and mode == "live"
    alert_text = (safe_format_signal_alert(trade, sentiment, setup_ts=setup_ts,
                                           current_price=current_price,
                                           threshold=cfg.get("ml_threshold"),
                                           capital=TRADING_CAPITAL, portfolio=portfolio)
                  if will_notify else None)
    signal_id = save_signal_from_trade(trade, sentiment, setup_ts=setup_ts,
                                       data_source=data_source, last_bar_ts=last_bar_ts,
                                       mode=mode,
                                       notified=False if will_notify else None,
                                       alert_text=alert_text,
                                       portfolio_flag=fate["portfolio_flag"],
                                       # Fase 4 (F14): precio actual usado al emitir.
                                       price_at_emit=current_price)
    # -1 = el guardado lo rechazó (duplicado o error de escritura, ya registrado en el log).
    fate["motivo"] = ("duplicate" if signal_id is None or signal_id <= 0
                      else "emitted" if mode == "live" else "shadow_asset")

    if mode == "shadow":
        logger.info(f"[{symbol}] Modo sombra: señal registrada (ID={signal_id}) sin notificar.")
        return

    # Solo notificar si es una señal nueva (no un duplicado ya registrado)
    if will_notify and signal_id > 0:
        from data.database import mark_signal_notified, set_signal_alert_text
        # El ID solo se conoce tras guardar: se añade al mensaje para que el
        # usuario pueda responder /fill ID PRECIO o /skip ID motivo (Fase 1, 1.10).
        try:
            from notifications.telegram_commands import signal_id_footer
            alert_text = alert_text + signal_id_footer(signal_id)
            set_signal_alert_text(signal_id, alert_text)
        except Exception as exc:
            logger.warning(f"[{symbol}] No se pudo añadir el ID al mensaje: {exc}")
        from monitoring.cycle_health import mark as _mark_cycle
        if send_text(alert_text):
            if not mark_signal_notified(signal_id, True):
                # Llegó pero la BD no lo apuntó: el ciclo siguiente la reenviará
                # (con la nota de que puede ser la misma) y este ciclo no tiene latido.
                _mark_cycle("bd", f"marcar notificada la señal #{signal_id}")
        else:
            _mark_cycle("telegram", f"envío de la señal #{signal_id} de {symbol}")
            logger.error(f"[{symbol}] Señal #{signal_id} guardada pero Telegram falló; "
                         "se reintentará en el próximo ciclo.")


def cmd_backtest(symbol: str, start: str = None, end: str = None,
                  capital: float = None):
    """Ejecuta backtest para el símbolo."""
    from backtest.engine  import run_backtest
    from backtest.metrics import format_metrics_table

    df_4h, df_daily, macro_df = _load_data(symbol)
    if df_4h is None:
        return

    logger.info(f"Iniciando backtest para {symbol}...")
    metrics = run_backtest(
        df_4h        = df_4h,
        df_daily     = df_daily,
        symbol       = symbol,
        macro_df     = macro_df,
        start_date   = start,
        end_date     = end,
        initial_capital = capital or BACKTEST["initial_capital"],
        verbose      = True,
    )

    if not metrics:
        logger.error("Backtest no produjo resultados.")
        return

    tbl = format_metrics_table(metrics)
    logger.info("\n" + tbl.to_string(index=False))

    # Guardar en base de datos
    from data.database import save_backtest_result
    save_backtest_result(metrics)
    logger.info("Resultados guardados en base de datos.")


def cmd_validate(symbol: str = None):
    """
    Validación rigurosa de expectativa: ¿tienen los setups edge tras costos?
    Sin look-ahead (entrada a la apertura de la barra siguiente), con comisión
    y slippage. Prueba la materia prima (los setups), no el ML.
    Si symbol es None, valida todos los activos.
    """
    from backtest.validate import (validate_symbol, print_validation_report,
                                    validate_variants, print_variants_report,
                                    validate_vol_stops, print_vol_stops_report)

    symbols = list(ASSETS.keys()) if not symbol else [symbol]
    results = []
    for sym in symbols:
        try:
            df_4h, df_daily, macro_df = _load_data(sym)
            if df_4h is None:
                logger.warning(f"[{sym}] Sin datos; se omite de la validación.")
                continue
            results.append(validate_symbol(sym, df_4h, df_daily))
            # Con un solo activo, además probar variantes de estrategia (dirección,
            # tipo, sesión, régimen) y estimadores de volatilidad para el stop.
            if symbol:
                variants = validate_variants(sym, df_4h, df_daily)
                if variants:
                    print_variants_report(sym, variants)
                vol_stops = validate_vol_stops(sym, df_4h, df_daily)
                if vol_stops:
                    print_vol_stops_report(sym, vol_stops)
        except Exception as exc:
            logger.error(f"[{sym}] Error en validación: {exc}")

    if results:
        print_validation_report(results)
    return results


def cmd_validate_protocol(symbol: str = None, candidate: str = None, final: bool = False,
                          okx: bool = False, out: str = None) -> int:
    """
    Puerta de validación de la Fase 2 (backtest/protocol.py). Comando OFFLINE:
    nunca lo llama el worker. Devuelve el código de salida (0 informe completo,
    1 detenido, 2 sin registro de pruebas → sin veredicto).
    """
    from backtest.protocol import run_protocol
    if okx and not final:
        logger.warning("--okx solo se usa con --final (el holdout OKX solo se mira en la evaluación final).")
    code, _ = run_protocol(symbol, candidate_path=candidate, final=final, okx=okx, out=out)
    sys.stdout.flush()
    return code


def cmd_check_live(symbol: str = None) -> bool:
    """Prueba de regresión 'vivo = validador' (ver backtest/live_emulation.py).
    Retorna True si todos los activos revisados coinciden."""
    from backtest.live_emulation import compare_with_validator, print_comparison

    import gc
    symbols = [symbol] if symbol else active_assets()
    all_ok = True
    for sym in symbols:
        df_4h, df_daily, _ = _load_data(sym)
        if df_4h is None:
            logger.warning(f"[{sym}] Sin datos; se omite.")
            all_ok = False
            continue
        res = compare_with_validator(sym, df_4h, df_daily)
        print_comparison(res)
        sys.stdout.flush()
        all_ok &= res["match"]
        del df_4h, df_daily
        gc.collect()
    print("\nResultado global:", "OK" if all_ok else "HAY DIFERENCIAS")
    return all_ok


def cmd_train(symbol: str):
    """Entrena el modelo ML para el símbolo."""
    from signals.detector import detect_all_setups
    from ml.features      import build_training_dataset, compute_base_features
    from ml.trainer       import train_model

    df_4h, df_daily, macro_df = _load_data(symbol)
    if df_4h is None:
        return

    logger.info(f"Detectando setups históricos para {symbol}...")
    df_setups = detect_all_setups(df_4h, df_daily, symbol)

    if len(df_setups) < 50:
        logger.warning(
            f"Solo {len(df_setups)} setups detectados. "
            "Se recomienda ≥ 200 para entrenamiento robusto."
        )

    forward_bars = ASSETS.get(symbol, {}).get("forward_bars", 12)

    # Funding rates para crypto (señal de sentimiento de mercado)
    funding_series = None
    if ASSETS.get(symbol, {}).get("source") == "binance":
        from data.fetchers.binance_fetcher import fetch_funding_rates
        days_hist = ASSETS.get(symbol, {}).get("days_history", 730)
        funding_series = fetch_funding_rates(symbol, days=days_hist)

    logger.info(f"Construyendo dataset de entrenamiento ({len(df_setups)} setups, forward_bars={forward_bars})...")
    dataset = build_training_dataset(
        df_setups      = df_setups,
        df_ohlcv       = df_4h,
        forward_bars   = forward_bars,
        macro_df       = macro_df,
        funding_series = funding_series,
        symbol         = symbol,
    )

    if dataset.empty:
        logger.error("Dataset vacío. No se puede entrenar.")
        return

    wins = (dataset["target"] == 1).sum()
    total = len(dataset)
    logger.info(
        f"Dataset: {total} setups etiquetados | "
        f"Win rate histórico: {wins/total*100:.1f}%"
    )

    model, feat_cols, metrics = train_model(dataset, symbol=symbol, save=True)
    if model is None:
        logger.error("Entrenamiento fallido.")
        return

    logger.info(
        f"Modelo guardado | "
        f"Walk-Forward AUC: {metrics.get('auc', 0):.3f} | "
        f"Precision@{ML_PARAMS['prob_threshold']}: {metrics.get('precision', 0):.3f} | "
        f"Folds: {metrics.get('n_folds', 0)}"
    )


def cmd_update():
    """Actualiza datos OHLCV para todos los activos configurados."""
    from data.database import init_db
    init_db()

    for symbol, cfg in ASSETS.items():
        try:
            _load_data(symbol)
            n = _flush_ohlcv_copies()
            logger.info(f"{symbol}: datos actualizados ({n} velas nuevas en la copia OHLCV).")
        except Exception as e:
            logger.error(f"{symbol}: error al actualizar — {e}")


def cmd_auto_retrain(symbol: str, notify: bool = False) -> bool:
    """
    Reentrena el modelo del símbolo si han pasado retrain_every días.
    Retorna True si se reentrenó efectivamente.
    """
    if not _should_retrain(symbol):
        last = _get_last_train_ts(symbol)
        days_ago = (datetime.now(timezone.utc) - last).days if last else "?"
        logger.info(
            f"[{symbol}] Reentrenamiento no necesario — "
            f"último hace {days_ago} día(s), umbral={ML_PARAMS['retrain_every']}d"
        )
        return False

    logger.info(f"[{symbol}] Iniciando reentrenamiento automático...")
    try:
        from signals.detector import detect_all_setups
        from ml.features      import build_training_dataset
        from ml.trainer       import train_model

        df_4h, df_daily, macro_df = _load_data(symbol)
        if df_4h is None:
            logger.error(f"[{symbol}] Sin datos para reentrenar.")
            return False

        df_setups    = detect_all_setups(df_4h, df_daily, symbol)
        forward_bars = ASSETS.get(symbol, {}).get("forward_bars", 12)

        funding_series = None
        if ASSETS.get(symbol, {}).get("source") == "binance":
            from data.fetchers.binance_fetcher import fetch_funding_rates
            days_hist = ASSETS.get(symbol, {}).get("days_history", 730)
            funding_series = fetch_funding_rates(symbol, days=days_hist)

        dataset = build_training_dataset(
            df_setups=df_setups, df_ohlcv=df_4h,
            forward_bars=forward_bars, macro_df=macro_df,
            funding_series=funding_series,
            symbol=symbol,
        )
        if dataset.empty or len(dataset) < ML_PARAMS.get("min_train_samples", 50):
            logger.warning(f"[{symbol}] Dataset insuficiente para reentrenar.")
            return False

        model, _, metrics = train_model(dataset, symbol=symbol, save=True)
        if model is None:
            logger.error(f"[{symbol}] Entrenamiento fallido.")
            return False

        metrics["n_samples"] = len(dataset)
        metrics["threshold"] = ML_PARAMS["prob_threshold"]
        _set_last_train_ts(symbol)

        logger.info(
            f"[{symbol}] Reentrenamiento OK | "
            f"AUC={metrics.get('auc',0):.3f} | "
            f"Precision={metrics.get('precision',0):.3f} | "
            f"Folds={metrics.get('n_folds',0)}"
        )

        from config.settings import notify_enabled
        if notify and notify_enabled(symbol):
            from notifications.telegram import send_retrain_alert
            send_retrain_alert(symbol, metrics)

        return True

    except Exception as exc:
        logger.error(f"[{symbol}] Error durante reentrenamiento: {exc}")
        from config.settings import notify_enabled
        if notify and notify_enabled(symbol):
            from notifications.telegram import send_error_alert
            send_error_alert(symbol, f"Reentrenamiento fallido: {exc}")
        return False


def _nightly_retrain_symbols() -> list:
    """Activos que la tarea nocturna reentrena: solo los operativos con
    use_ml=True. Hoy ninguno (el ML no decide nada en ningún activo)."""
    return [s for s in active_assets() if ASSETS.get(s, {}).get("use_ml", True)]


# ---------------------------------------------------------------------------
# Ciclo alineado a hora fija UTC (Fase 4). Ver CYCLE_SCHEDULE en settings.
# ---------------------------------------------------------------------------

def _fmt_hhmm(minute_of_day: int) -> str:
    m = int(minute_of_day) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def extra_cycle_times() -> list:
    """Ciclo(s) extra "HH:MM" UTC tras el cierre de la vela diaria de sesión
    (DAILY_BAR_CLOSE_UTC + after_daily_close_min = 22:20): en invierno el oro se
    evalúa esa misma noche (ver CYCLE_SCHEDULE)."""
    from config.settings import CYCLE_SCHEDULE, DAILY_BAR_CLOSE_UTC
    after = CYCLE_SCHEDULE.get("after_daily_close_min")
    if after is None or int(after) < 0:
        return []
    h, m = DAILY_BAR_CLOSE_UTC
    return [_fmt_hhmm(int(h) * 60 + int(m) + int(after))]


def cycle_slot_times(interval_minutes: int = 60, minute: int | None = None,
                     extra: list | None = None) -> list | None:
    """Horas "HH:MM" (UTC, ordenadas y sin repetir) a las que corre el ciclo de
    señales cada día: minuto `minute` (CYCLE_SCHEDULE, :01) + k·intervalo desde
    las 00:00 UTC, más las horas `extra` (por defecto extra_cycle_times()).

    None si el intervalo no divide el día (1.440 min): el llamador usa entonces
    el modo antiguo (cada N minutos desde el arranque) más las horas extra."""
    from config.settings import CYCLE_SCHEDULE
    if minute is None:
        minute = int(CYCLE_SCHEDULE.get("minute", 1))
    if extra is None:
        extra = extra_cycle_times()
    interval = int(interval_minutes)
    if interval <= 0 or 1440 % interval != 0:
        return None
    mins = {(int(minute) + k * interval) % 1440 for k in range(1440 // interval)}
    for hhmm in extra:
        hh, mm = str(hhmm).split(":")
        mins.add((int(hh) * 60 + int(mm)) % 1440)
    return [_fmt_hhmm(m) for m in sorted(mins)]


def _last_slot_start(hhmm: str, now_utc: datetime) -> datetime:
    """Última vez (≤ now_utc) que tocó la hora diaria `hhmm` UTC."""
    hh, mm = (int(x) for x in hhmm.split(":"))
    t = now_utc.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t > now_utc:
        t -= timedelta(days=1)
    return t


def _describe_slots(slots: list, interval_minutes: int, minute: int, extra: list) -> str:
    """Texto del log de arranque con las horas del ciclo."""
    if int(interval_minutes) == 60:
        txt = f"cada hora en :{int(minute) % 60:02d}"
        rest = [x for x in extra if not x.endswith(f":{int(minute) % 60:02d}")]
        if rest:
            txt += " y a las " + ", ".join(rest) + " (cierre de la vela diaria de oro/SPX)"
        return txt + " UTC"
    if len(slots) <= 12:
        return "a las " + ", ".join(slots) + " UTC"
    return f"{len(slots)} veces al día (" + ", ".join(slots[:4]) + ", ...) UTC"


def _at_utc(job, hhmm: str):
    """job.at(hhmm) en UTC (schedule usa pytz, dependencia ya instalada). Si no se
    pudiera, en hora local (el contenedor de Render está en UTC)."""
    try:
        return job.at(hhmm, "UTC")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Planificador: sin zona horaria UTC ({exc}); {hhmm} en hora local.")
        return job.at(hhmm)


def cmd_schedule(interval_minutes: int = 60, retrain_hour: str = "03:00"):
    """
    Modo daemon: genera señales a horas FIJAS en UTC (Fase 4, CYCLE_SCHEDULE).

    Trabajos (todos en el mismo hilo, uno tras otro; `schedule` nunca los solapa):
      - ciclo de señales: cada hora en HH:01 UTC (con --interval 60; otros
        intervalos: ver CYCLE_SCHEDULE) y un ciclo extra a las 22:20 UTC para la
        vela diaria del oro; además una vez nada más arrancar;
      - comandos de Telegram: cada TELEGRAM_POLL_MINUTES (2) min solo se leen
        /fill /skip /help (sin descargar velas);
      - tarea nocturna a `retrain_hour` UTC (03:00): va antes que el ciclo de las
        03:01 y, si se alarga, este corre en cuanto termina.
    Envía notificaciones por Telegram si está configurado.
    """
    import time
    import schedule as sched
    from notifications.telegram  import test_connection
    from config.settings         import (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, CYCLE_SCHEDULE,
                                         TELEGRAM_POLL_MINUTES)

    notify = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    if notify:
        test_connection()

    try:
        interval_minutes = int(interval_minutes)
    except (TypeError, ValueError):
        interval_minutes = 60
    if interval_minutes <= 0:
        logger.warning(f"--interval {interval_minutes} no válido: se usa 60.")
        interval_minutes = 60
    minute = int(CYCLE_SCHEDULE.get("minute", 1)) % 60
    extra = extra_cycle_times()
    slots = cycle_slot_times(interval_minutes, minute=minute, extra=extra)
    poll_minutes = int(TELEGRAM_POLL_MINUTES or 0) if notify else 0
    if slots is None:
        when = f"cada {interval_minutes} min desde el arranque (modo antiguo) y a las {', '.join(extra)} UTC"
    else:
        when = _describe_slots(slots, interval_minutes, minute, extra)

    logger.info(
        f"Iniciando modo daemon — señales {when} | "
        f"comandos de Telegram: {f'cada {poll_minutes} min' if poll_minutes > 0 else 'al inicio de cada ciclo'} | "
        f"Reentrenamiento diario a las {retrain_hour} UTC | "
        f"Telegram: {'ON' if notify else 'OFF'}. "
        "Ctrl+C para detener."
    )

    # Hora (UTC) a la que empezó el último ciclo de señales.
    state = {"last_cycle_start": None}

    def _run_all_signals():
        import gc
        from monitoring import cycle_health
        state["last_cycle_start"] = datetime.now(timezone.utc)
        _cycle_fresh.clear()
        cycle_health.reset()
        failed = []
        # Comandos de Telegram (/fill /skip /help, Fase 1 punto 1.10): polling sin
        # espera al inicio del ciclo. Nunca lanza ni bloquea.
        if notify:
            try:
                from notifications.telegram_commands import poll_commands
                poll_commands()
            except Exception as exc:
                logger.warning(f"Comandos Telegram: error ({exc}); el ciclo sigue.")

        # Caché de velas cerradas del ciclo (símbolo → DataFrame): la rellenan el
        # tracker y cmd_signal, y el registro sombra la reutiliza al final.
        ohlcv_cache: dict = {}

        # Resolver resultados pendientes antes de generar nuevas señales
        try:
            from data.outcome_tracker import resolve_open_signals
            resolved = resolve_open_signals(ohlcv_cache=ohlcv_cache)
            if resolved:
                logger.info(f"Outcome tracker: {resolved} señal(es) resuelta(s).")
        except Exception as exc:
            logger.warning(f"Outcome tracker error: {exc}")
            cycle_health.mark("bd", f"resolver señales abiertas: {exc}")

        # Reenviar señales cuyo envío a Telegram falló en un ciclo anterior
        # (Fase 1, columna notified). Nunca lanza.
        if notify:
            from notifications.telegram import retry_failed_signal_alerts
            retry_failed_signal_alerts()

        for symbol in active_assets():
            try:
                cmd_signal(symbol, notify=notify, ohlcv_cache=ohlcv_cache)
            except Exception as exc:
                failed.append(symbol)
                logger.error(f"Error generando señal para {symbol}: {exc}")
            finally:
                # Cada ciclo carga datos de los activos activos; liberar
                # entre cada uno evita que la RAM crezca de forma sostenida.
                gc.collect()

        # Registro sombra: resolver los setups pendientes con las velas ya
        # descargadas en este ciclo (fuera de la ruta crítica; nunca lanza).
        try:
            from data.shadow import resolve_shadow_setups
            resolve_shadow_setups(ohlcv_cache=ohlcv_cache)
        except Exception as exc:
            logger.warning(f"Registro sombra: error resolviendo ({exc}); el ciclo sigue.")
        ohlcv_cache.clear()
        # Caché del DVOL de Deribit (Fase 3, 3.3): un valor por moneda y ciclo.
        try:
            from data.fetchers.deribit_fetcher import clear_cycle_cache
            clear_cycle_cache()
        except Exception:  # noqa: BLE001
            pass

        # Copia de las últimas velas en la tabla ohlcv: al final, cuando todas
        # las señales ya se enviaron (fuera de la ruta crítica).
        _flush_ohlcv_copies()

        # Latido (Fase 1, punto 1.4): ciclo correcto = ningún activo lanzó una
        # excepción, al menos uno tuvo datos frescos y (revisión de la Fase 1)
        # ningún acceso crítico a la BD ni envío de señal a Telegram falló en el
        # ciclo, y la BD responde ahora (monitoring/cycle_health.py). Si no, no se
        # hace ping y el servicio externo (HEARTBEAT_URL) avisa cuando falte.
        # Nunca lanza.
        db_ok = cycle_health.db_alive()
        if not db_ok:
            cycle_health.mark("bd", "SELECT 1 al final del ciclo")
        problems = cycle_health.problems()
        if not failed and _cycle_fresh and not problems:
            try:
                from data.database import kv_set
                from monitoring.daily_summary import KV_LAST_CYCLE_OK
                kv_set(KV_LAST_CYCLE_OK, int(time.time()))
            except Exception as exc:
                logger.debug(f"No se pudo guardar la hora del último ciclo correcto: {exc}")
            from monitoring.heartbeat import ping_heartbeat
            ping_heartbeat()
        else:
            logger.warning(f"Ciclo incompleto (errores: {failed or 'ninguno'}; activos con datos: "
                           f"{sorted(_cycle_fresh) or 'ninguno'}; fallos críticos: "
                           f"{cycle_health.summary() or 'ninguno'}): sin latido.")

    def _run_nightly():
        """Reentrena si es necesario y envía el resumen (se ejecuta 1× al día).

        Fase 1: los activos con use_ml=False (hoy TODOS) no se reentrenan ni se
        descargan: el modelo no se usa para decidir y el reentreno solo gastaba
        RAM/tiempo y enviaba alertas de "modelo actualizado" engañosas."""
        import gc
        logger.info("Tarea nocturna: verificando reentrenamiento...")
        skipped = [s for s in active_assets() if s not in _nightly_retrain_symbols()]
        if skipped:
            logger.info(f"Sin reentreno nocturno (use_ml=False): {', '.join(skipped)}.")
        for symbol in _nightly_retrain_symbols():
            try:
                cmd_auto_retrain(symbol, notify=notify)
            except Exception as exc:
                logger.error(f"Error en tarea nocturna para {symbol}: {exc}")
            finally:
                # Liberar memoria entre activos: con 8 modelos en un solo Worker,
                # acumular el dataset + folds de cada uno provoca el OOM en Render.
                gc.collect()

        # Monitor de datos nocturno (Fase 2, 2.3): contraste OKX vs Gate.io en
        # cripto (una petición, timeout 10 s) y deriva del detector. Va antes del
        # resumen para que este muestre el contraste de hoy. Nunca lanza.
        try:
            from monitoring.data_monitor import run_nightly_checks
            run_nightly_checks(notify=notify)
        except Exception as exc:
            logger.warning(f"Monitor de datos nocturno falló: {exc}")

        # Resumen diario por Telegram, construido desde la BD (Fase 1, punto 1.4;
        # antes PortfolioRiskManager, que no veía la BD y siempre decía "Sin
        # operaciones"). Nunca lanza.
        if notify:
            from monitoring.daily_summary import send_daily_report
            if send_daily_report():
                logger.info("Resumen diario enviado a Telegram.")

    def _run_slot(hhmm: str):
        """Ciclo de la hora fija `hhmm`. Si ya empezó otro ciclo después de esa
        hora (p. ej. el proceso estuvo bloqueado y se juntaron varias horas
        pendientes), no se repite: ese ciclo ya vio las mismas velas."""
        last = state["last_cycle_start"]
        if last is not None and last >= _last_slot_start(hhmm, datetime.now(timezone.utc)):
            logger.info(f"Ciclo de las {hhmm} UTC omitido: ya corrió uno después ({last:%H:%M:%S} UTC).")
            return
        _run_all_signals()

    def _poll_commands_job():
        """Solo lee /fill /skip /help (Fase 4). Sin velas, sin latido. Nunca lanza."""
        try:
            from notifications.telegram_commands import poll_commands
            poll_commands(log_failures=False)
        except Exception as exc:
            logger.warning(f"Comandos Telegram (sondeo): error ({exc}).")

    # Ciclo de señales a horas fijas (UTC). Cada hora es un trabajo diario: si un
    # ciclo se alarga y pisa la hora siguiente, esa corre en cuanto termina (tarde,
    # nunca dos veces ni en paralelo).
    if slots is None:
        logger.warning(f"--interval {interval_minutes} no divide el día (1.440 min): ciclo cada "
                       f"{interval_minutes} min contado desde el arranque (no alineado al cierre de "
                       f"vela), más el ciclo extra de las {', '.join(extra)} UTC.")
        sched.every(interval_minutes).minutes.do(_run_all_signals)
        slot_jobs = extra
    else:
        slot_jobs = slots
    for hhmm in slot_jobs:
        _at_utc(sched.every().day, hhmm).do(_run_slot, hhmm)

    # Tarea nocturna a hora fija (UTC)
    _at_utc(sched.every().day, retrain_hour).do(_run_nightly)

    # Comandos de Telegram entre ciclos (el ciclo también los lee al empezar).
    if poll_minutes > 0:
        sched.every(poll_minutes).minutes.do(_poll_commands_job)

    _run_all_signals()   # Ejecución inmediata al arrancar

    sleep_s = max(1.0, float(CYCLE_SCHEDULE.get("loop_sleep_s", 5)))
    while True:
        sched.run_pending()
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Output formateado
# ---------------------------------------------------------------------------

def _print_signal(trade, sentiment: dict = None) -> None:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import box

    console = Console()

    direction_color = "green" if trade.direction == "long" else "red"
    direction_label = "LONG  ▲" if trade.direction == "long" else "SHORT ▼"

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Campo",  style="bold cyan", width=22)
    table.add_column("Valor",  style="white")

    table.add_row("Símbolo",      trade.symbol)
    table.add_row("Dirección",    f"[{direction_color}]{direction_label}[/{direction_color}]")
    table.add_row("Setup Type",   trade.setup_type.replace("_", " ").title())
    table.add_row("Score setup",  f"{trade.ml_score:.2f}")
    table.add_row("",             "")
    from config.settings import fmt_price
    table.add_row("Entrada",      f"${fmt_price(trade.entry_price)}")
    table.add_row("Stop Loss",    f"${fmt_price(trade.stop_loss)}")
    table.add_row("Take Profit",  f"${fmt_price(trade.take_profit)}")
    table.add_row("R:R Ratio",    f"{trade.rr_ratio:.2f}")
    table.add_row("",             "")
    table.add_row("Tamaño ($)",   f"${trade.position_size:,.0f}")
    table.add_row("Riesgo ($)",   f"${trade.risk_usd:,.2f}")
    table.add_row("Potencial ($)",f"${trade.reward_usd:,.0f}")

    if sentiment:   # None con SENTIMENT['enabled']=False (Fase 1)
        sent_color = sentiment.get("color", "gray")
        sent_label = sentiment.get("label", "NEUTRAL")
        sent_score = sentiment.get("score", 0)
        table.add_row("",             "")
        table.add_row("Sentimiento",  f"[{sent_color}]{sent_label} ({sent_score:+.2f})[/{sent_color}]")
        table.add_row("Noticias",     f"{sentiment.get('n_news', 0)} artículos recientes")

    console.print(Panel(
        table,
        title=f"[bold] SEÑAL — {trade.symbol} [/bold]",
        border_style=direction_color,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sistema de Trading v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py signal   --symbol BTCUSDT
  python main.py signal   --symbol BTCUSDT --notify
  python main.py backtest --symbol ETHUSDT --start 2023-01-01 --capital 10000
  python main.py train    --symbol BTCUSDT
  python main.py retrain  --symbol BTCUSDT          # fuerza reentrenamiento
  python main.py retrain  --all                     # reentrena todos los activos
  python main.py update
  python main.py schedule --interval 60
  python main.py schedule --interval 60 --retrain-hour 03:00
  python main.py telegram                           # test de conexión
  python main.py summary                            # resumen diario (vista previa)
  python main.py history-stats                      # cobertura de la tabla ohlcv
  python main.py validate --protocol                # puerta de validación (línea base)
  python main.py validate --protocol --candidate cand.json [--final [--okx]] [--out informe.json]
        """,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # signal
    ps = sub.add_parser("signal", help="Generar señal actual")
    ps.add_argument("--symbol", default="BTCUSDT",
                    choices=list(ASSETS.keys()),
                    help="Símbolo a analizar")
    ps.add_argument("--notify", action="store_true",
                    help="Enviar alerta a Telegram si hay señal")

    # backtest
    pb = sub.add_parser("backtest", help="Ejecutar backtest")
    pb.add_argument("--symbol",  default="BTCUSDT", choices=list(ASSETS.keys()))
    pb.add_argument("--start",   default=None,   help="Fecha inicio YYYY-MM-DD")
    pb.add_argument("--end",     default=None,   help="Fecha fin   YYYY-MM-DD")
    pb.add_argument("--capital", default=None,   type=float, help="Capital inicial USD")

    # validate — expectativa de los setups tras costos (sin look-ahead)
    pv = sub.add_parser("validate", help="Validar expectativa de los setups tras costos")
    pv.add_argument("--symbol", default=None, choices=list(ASSETS.keys()),
                    help="Un activo; omitir para validar todos")
    # Fase 2 (2.1): puerta de validación fija (backtest/protocol.py). Offline.
    pv.add_argument("--protocol", action="store_true",
                    help="Puerta de validación: costes por bróker, reserva intocable, registro de pruebas")
    pv.add_argument("--candidate", default=None, metavar="RUTA.json",
                    help="(--protocol) candidato a evaluar frente a la configuración actual")
    pv.add_argument("--final", action="store_true",
                    help="(--protocol) evaluar también la reserva y el holdout OKX (queda registrado)")
    pv.add_argument("--okx", action="store_true",
                    help="(--protocol --final) descargar OKX en memoria si la tabla ohlcv no lo tiene")
    pv.add_argument("--out", default=None, metavar="RUTA.json",
                    help="(--protocol) guardar además el informe en JSON")

    # check-live — regresión "vivo = validador"
    pcl = sub.add_parser("check-live", help="Comprobar que el vivo emite lo mismo que valida el validador")
    pcl.add_argument("--symbol", default=None, choices=list(ASSETS.keys()),
                     help="Un activo; omitir para todos los operativos")

    # train
    pt = sub.add_parser("train", help="Entrenar modelo ML")
    pt.add_argument("--symbol", default="BTCUSDT", choices=list(ASSETS.keys()))

    # update
    sub.add_parser("update", help="Actualizar todos los datos OHLCV")

    # retrain
    pr = sub.add_parser("retrain", help="Forzar reentrenamiento del modelo ML")
    pr_grp = pr.add_mutually_exclusive_group(required=True)
    pr_grp.add_argument("--symbol", choices=list(ASSETS.keys()),
                        help="Símbolo a reentrenar")
    pr_grp.add_argument("--all", action="store_true",
                        help="Reentrenar todos los activos")
    pr.add_argument("--notify", action="store_true",
                    help="Enviar resultado a Telegram")

    # schedule (daemon)
    psc = sub.add_parser("schedule", help="Modo daemon con señales periódicas")
    psc.add_argument("--interval", default=60, type=int,
                     help="Intervalo en minutos (default: 60 = cada hora en HH:01 UTC). Si divide "
                          "el día (15, 30, 120, 240...) se alinea a las 00:01 UTC; si no, cada N min "
                          "desde el arranque. Siempre se añade el ciclo de las 22:20 UTC (oro).")
    psc.add_argument("--retrain-hour", default="03:00", dest="retrain_hour",
                     help="Hora UTC para la tarea nocturna de reentrenamiento (HH:MM, default: 03:00)")

    # telegram (test de conexión)
    sub.add_parser("telegram", help="Verificar conexión a Telegram y enviar mensaje de prueba")

    # summary — resumen nocturno desde la BD (vista previa; --send lo envía)
    psm = sub.add_parser("summary", help="Resumen diario desde la BD con el estado SPRT/CUSUM (vista previa)")
    psm.add_argument("--send", action="store_true", help="Enviarlo además por Telegram")

    # history-stats — cobertura de la copia propia del histórico (Fase 1, 1.11)
    sub.add_parser("history-stats",
                   help="Cobertura de la tabla ohlcv por símbolo/timeframe/fuente (ver backfill_history.py)")

    return p


def main():
    # Crear directorios necesarios
    for d in ("logs", "models", "data/db"):
        Path(d).mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "signal":
        cmd_signal(args.symbol, notify=args.notify)
        _flush_ohlcv_copies()

    elif args.command == "backtest":
        cmd_backtest(args.symbol, args.start, args.end, args.capital)

    elif args.command == "validate":
        if args.protocol:
            sys.exit(cmd_validate_protocol(args.symbol, args.candidate, args.final,
                                           args.okx, args.out))
        if args.candidate or args.final or args.okx or args.out:
            parser.error("--candidate, --final, --okx y --out solo valen con --protocol")
        cmd_validate(args.symbol)

    elif args.command == "check-live":
        ok = cmd_check_live(args.symbol)
        sys.exit(0 if ok else 1)

    elif args.command == "train":
        cmd_train(args.symbol)

    elif args.command == "retrain":
        import gc
        # --all reentrena solo los activos operativos (enabled); los apagados
        # no gastan RAM ni tiempo. Un --symbol explícito se respeta siempre.
        symbols = active_assets() if args.all else [args.symbol]
        from data.database import set_last_train
        for sym in symbols:
            # Forzar reentrenamiento: fecha muy antigua → _should_retrain = True
            try:
                set_last_train(sym, "1970-01-01T00:00:00+00:00")
            except Exception:
                pass
            try:
                cmd_auto_retrain(sym, notify=args.notify)
            finally:
                # Liberar memoria entre activos. Desde la Fase 1 'retrain --all'
                # ya NO corre en el arranque del Worker (ver render.yaml): solo
                # a mano. Con la especialización son ~4 modelos, no 8.
                gc.collect()

    elif args.command == "update":
        cmd_update()

    elif args.command == "schedule":
        cmd_schedule(args.interval, args.retrain_hour)

    elif args.command == "telegram":
        from notifications.telegram import test_connection
        test_connection()

    elif args.command == "summary":
        import re
        from monitoring.daily_summary import build_daily_summary, send_daily_report
        import html as _html
        print(_html.unescape(re.sub(r"<[^>]+>", "", build_daily_summary())))
        if args.send:
            print("Enviado." if send_daily_report() else "No se pudo enviar (¿Telegram configurado?).")

    elif args.command == "history-stats":
        from backfill_history import format_history_stats
        print(format_history_stats())


if __name__ == "__main__":
    main()
