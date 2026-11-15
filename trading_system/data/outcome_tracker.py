"""
Outcome Tracker — registra señales y resuelve resultados automáticamente.

Flujo:
  1. Cuando se genera una señal → save_signal_from_trade() la persiste en DB
  2. Periódicamente (cada hora en el scheduler) → resolve_open_signals() revisa
     cada señal abierta contra los precios actuales y marca win/loss/expired
  3. Los resultados resueltos alimentan el reentrenamiento del modelo ML
"""
import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from data.database import get_engine, save_signal, load_signals
from sqlalchemy import text


def is_duplicate_signal(symbol: str, direction: str, entry_price: float,
                        hours: int = 48) -> bool:
    """(Guardia heredada) True si ya existe una señal con el mismo símbolo,
    dirección y entry_price que sigue pendiente o se creó dentro de `hours`.

    Desde la Fase 0 el antiduplicado principal es is_duplicate_setup (permanente
    por setup_ts). Esta comprobación se mantiene como segunda barrera.

    Nota: la columna `ts` se almacena como entero Unix (BIGINT).
    """
    from datetime import timedelta
    cutoff_unix = int((datetime.now(timezone.utc) - timedelta(hours=hours)).timestamp())
    # Tolerancia relativa: mismo nivel de entrada salvo ruido de redondeo float.
    tol = max(abs(entry_price) * 1e-4, 1e-6)
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM signals "
                "WHERE symbol=:sym AND direction=:dir "
                "AND ABS(entry_price - :entry) <= :tol "
                "AND (resultado_real IS NULL OR ts >= :cutoff)"
            ), {"sym": symbol, "dir": direction, "entry": entry_price,
                "tol": tol, "cutoff": cutoff_unix}).scalar()
            return (row or 0) > 0
    except Exception as exc:
        logger.warning(f"Dedup check falló ({exc}); no se bloquea la señal.")
        _mark_db(f"dedup por precio: {exc}")
        return False


# Alias retrocompatible (se mantenía con guion bajo internamente)
_is_duplicate_signal = is_duplicate_signal


def _mark_db(detail: str) -> None:
    """Anota un fallo CRÍTICO de BD en la salud del ciclo: ese ciclo no hará
    latido (revisión de la Fase 1). Nunca lanza."""
    try:
        from monitoring.cycle_health import mark
        mark("bd", detail)
    except Exception:
        pass


def _unix(ts) -> int:
    """Timestamp/datetime → segundos Unix. Los números se toman como segundos Unix
    (no como nanosegundos, que es lo que haría pd.Timestamp con un entero)."""
    import numpy as np
    if isinstance(ts, (int, float, np.integer, np.floating)):
        return int(ts)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def is_duplicate_setup(symbol: str, direction: str, setup_ts) -> bool:
    """True si ESTE setup (símbolo, dirección y vela del setup) ya se registró
    alguna vez, en modo live o shadow. Es permanente: un setup se emite una sola
    vez, igual que el validador lo opera una sola vez (Fase 0, F01).

    Si la consulta falla se BLOQUEA la señal (fail-closed): es preferible perder
    una alerta que repetir una."""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM signals "
                "WHERE symbol=:sym AND direction=:dir AND setup_ts=:sts"
            ), {"sym": symbol, "dir": direction, "sts": _unix(setup_ts)}).scalar()
            return (row or 0) > 0
    except Exception as exc:
        logger.error(f"Dedup por setup falló ({exc}); se bloquea la señal por prudencia.")
        _mark_db(f"dedup por setup: {exc}")
        return True


def has_recent_legacy_alert(symbol: str, direction: str, setup_ts) -> bool:
    """Transición a la Fase 0: True si existe una alerta ANTIGUA (sin setup_ts)
    del mismo activo y dirección emitida desde la apertura de la vela del setup.

    Las alertas antiguas salían con la vela aún abierta y con otro precio, así que
    ni el antiduplicado por setup_ts ni el de precio las reconocen: sin esta
    barrera, el primer ciclo tras desplegar podría reenviar ese mismo setup ya
    con la vela cerrada. Solo mira filas antiguas: no afecta a setups nuevos."""
    try:
        with get_engine().connect() as conn:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM signals WHERE symbol=:sym AND direction=:dir "
                "AND setup_ts IS NULL AND ts >= :sts"
            ), {"sym": symbol, "dir": direction, "sts": _unix(setup_ts)}).scalar()
            return (row or 0) > 0
    except Exception as exc:
        logger.error(f"Comprobación de alertas antiguas falló ({exc}); se bloquea la señal por prudencia.")
        _mark_db(f"alertas antiguas: {exc}")
        return True


def emitted_setup_keys(symbol: str, since_ts) -> set:
    """Setups ya registrados para el símbolo con vela de setup >= since_ts,
    como conjunto de (Timestamp UTC, direction). Se pasa a get_best_setup para
    saltarlos."""
    try:
        with get_engine().connect() as conn:
            rows = conn.execute(text(
                "SELECT setup_ts, direction FROM signals "
                "WHERE symbol=:sym AND setup_ts IS NOT NULL AND setup_ts >= :since"
            ), {"sym": symbol, "since": _unix(since_ts)}).fetchall()
        return {(pd.Timestamp(int(r[0]), unit="s", tz="UTC"), r[1]) for r in rows}
    except Exception as exc:
        logger.warning(f"No se pudieron leer los setups emitidos de {symbol}: {exc}")
        _mark_db(f"setups emitidos de {symbol}: {exc}")
        return set()


def save_signal_from_trade(trade, sentiment: dict = None, setup_ts=None,
                           data_source: str = None, last_bar_ts=None,
                           mode: str = None, notified: bool = None,
                           alert_text: str = None) -> int:
    """
    Persiste una señal generada (TradeSetup) en la tabla signals.
    Retorna el ID asignado, o -1 si falló o era duplicado.

    Args:
        setup_ts:    vela del setup (activa el antiduplicado permanente).
        data_source: fuente de precios usada (gateio, GC=F, ^GSPC...).
        last_bar_ts: última vela cerrada al emitir.
        mode:        'live' (se notifica) o 'shadow' (solo se registra). Si no
                     se indica, se deduce de la configuración del activo
                     (notify False → 'shadow'), también para el dashboard.
        notified:    False si se va a enviar a Telegram justo después (Fase 1: se
                     guarda ANTES de enviar y se marca True al confirmar el envío;
                     si falla, el ciclo siguiente la reintenta). None = no aplica.
        alert_text:  texto del mensaje, para poder reenviarlo idéntico.
    """
    if mode is None:
        from config.settings import notify_enabled
        mode = "live" if notify_enabled(trade.symbol) else "shadow"
    if setup_ts is not None and is_duplicate_setup(trade.symbol, trade.direction, setup_ts):
        logger.debug(f"Setup ya registrado: {trade.symbol} {trade.direction} @ {setup_ts}.")
        return -1
    if is_duplicate_signal(trade.symbol, trade.direction, trade.entry_price):
        logger.debug(f"Señal duplicada omitida: {trade.symbol} {trade.direction} ya registrada.")
        return -1
    try:
        signal_dict = {
            "symbol":          trade.symbol,
            "ts":              pd.Timestamp.now(tz="UTC"),
            "setup_type":      trade.setup_type,
            "direction":       trade.direction,
            "ml_score":        trade.ml_score,
            "sentiment_score": sentiment.get("score", 0.0) if sentiment else None,
            "entry_price":     trade.entry_price,
            "stop_loss":       trade.stop_loss,
            "take_profit":     trade.take_profit,
            "position_size":   trade.position_size,
            "atr":             trade.atr,
            "setup_ts":        setup_ts,
            "data_source":     data_source,
            "last_bar_ts":     last_bar_ts,
            "mode":            mode,
            "notified":        notified,
            "alert_text":      alert_text,
        }
        signal_id = save_signal(signal_dict)
        logger.info(f"Señal guardada en DB | {trade.symbol} {trade.direction.upper()} "
                    f"| modo={mode} | ID={signal_id}")
        return signal_id
    except Exception as exc:
        logger.error(f"Error guardando señal: {exc}")
        _mark_db(f"guardar señal: {exc}")
        return -1


# ---------------------------------------------------------------------------
# Resolución de resultados
# ---------------------------------------------------------------------------

def _is_session_daily(symbol: str) -> bool:
    from config.settings import ASSETS
    return ASSETS.get(symbol, {}).get("source") == "yfinance"


def horizon_days(symbol: str) -> float:
    """Horizonte del activo en días NATURALES (forward_bars velas).
    Diarios: 20 velas bursátiles ≈ 28 días naturales (5 sesiones por semana)."""
    from config.settings import ASSETS
    cfg = ASSETS.get(symbol, {})
    fwd = cfg.get("forward_bars", 24)
    if _is_session_daily(symbol):
        return fwd * 7 / 5
    if cfg.get("source") == "twelvedata":
        return fwd * 4 / 24 * 7 / 5   # forex 4H: 6 velas por sesión, 5 sesiones
    return fwd * 4 / 24


def safety_net_days(symbol: str) -> float:
    """Red de seguridad para casos anómalos (huecos de datos). La salida normal
    es POR NÚMERO DE VELAS dentro de _check_outcome; esta edad solo actúa si,
    pasado el horizonte con margen, aún no hay velas suficientes.
    Diarios: 28 + 5 = 33 días. Cripto: 4 + 1 = 5 días."""
    margin = 5 if _is_session_daily(symbol) else 1
    return horizon_days(symbol) + margin


def _signal_timestamp(ts_value) -> pd.Timestamp:
    # PostgreSQL devuelve datetime objects; SQLite/BIGINT devuelve enteros Unix
    if isinstance(ts_value, (int, float)):
        return pd.Timestamp(int(ts_value), unit="s", tz="UTC")
    t = pd.Timestamp(ts_value)
    return t.tz_localize("UTC") if t.tz is None else t


def compute_result_fields(direction: str, entry: float, stop_loss: float,
                          exit_price: float, position_size: float = None) -> dict:
    """pnl_pct, pnl_r y pnl_usd de una salida a `exit_price`.

    pnl_usd = position_size × pnl_pct (PnL de la POSICIÓN emitida). Antes se
    calculaba entry × pnl_pct, que es el PnL por UNA unidad del activo (1 BTC,
    1 onza) e inflaba el "PnL total" del dashboard hasta ~90× en BTC (F05)."""
    sign = 1.0 if direction == "long" else -1.0
    pnl_pct = sign * (exit_price - entry) / entry
    risk = abs(entry - stop_loss)
    pnl_r = sign * (exit_price - entry) / risk if risk > 0 else None
    pnl_usd = (round(float(position_size) * pnl_pct, 2)
               if position_size is not None and not pd.isna(position_size) else None)
    return {
        "pnl_pct": round(pnl_pct, 6),
        "pnl_r":   round(pnl_r, 4) if pnl_r is not None else None,
        "pnl_usd": pnl_usd,
    }


def resolve_open_signals(ohlcv_cache: dict | None = None) -> int:
    """
    Revisa todas las señales abiertas (resultado_real IS NULL) y las resuelve
    con velas CERRADAS posteriores a la señal:
      1. Primero SL/TP (vela a vela; si una vela toca ambos, cuenta el SL).
      2. Si pasan forward_bars velas sin tocar ninguno → salida por tiempo
         ('expired') al cierre de la última vela, CON su PnL y su R.
      3. Solo si, pasada la red de seguridad, siguen faltando velas (hueco de
         datos) → 'expired' al último cierre disponible.

    Antes la antigüedad se comprobaba ANTES que el SL/TP y con un plazo de 22
    días naturales para los diarios (su horizonte son 28–29): las señales de oro
    y SPX quedaban 'expired' sin PnL y el win rate en vivo del SP500 caía del
    47 % al 28 % (F04). En cripto el plazo era 7 días para un horizonte de 4.

    Las señales en modo sombra se resuelven igual (para medirlas).

    Args:
        ohlcv_cache: caché del ciclo (símbolo → velas cerradas). Si se pasa, se
                     reutiliza y se rellena con lo descargado aquí, para que el
                     registro sombra (data/shadow.py) no vuelva a descargar
                     (Fase 1, punto 1.10).

    Returns:
        Número de señales resueltas en este ciclo.
    """
    from data.database import init_db
    init_db()
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, symbol, ts, direction, entry_price, stop_loss, take_profit, "
            "position_size, setup_ts FROM signals WHERE resultado_real IS NULL ORDER BY ts ASC"
        )).fetchall()

    if not rows:
        return 0

    # Caché de OHLCV por símbolo: se descarga una sola vez por ciclo aunque
    # haya varias señales pendientes del mismo activo.
    if ohlcv_cache is None:
        ohlcv_cache = {}
    now = pd.Timestamp.now(tz="UTC")

    resolved_count = 0
    for row in rows:
        sig_id, symbol, ts_value, direction, entry, sl, tp, pos_size, setup_ts_value = row
        try:
            sig_ts = _signal_timestamp(ts_value)

            if symbol not in ohlcv_cache:
                ohlcv_cache[symbol] = _load_outcome_ohlcv(symbol)
            df = ohlcv_cache[symbol]
            if df is None or df.empty:
                logger.warning(f"Señal #{sig_id} {symbol}: sin datos para resolver; se reintenta en el próximo ciclo.")
                continue

            setup_ts = (pd.Timestamp(int(setup_ts_value), unit="s", tz="UTC")
                        if setup_ts_value is not None and not pd.isna(setup_ts_value) else None)
            res = check_outcome_detail(symbol, sig_ts, direction, entry, sl, tp, df=df,
                                       setup_ts=setup_ts)

            age_days = (now - sig_ts).total_seconds() / 86400
            if res is None and age_days > safety_net_days(symbol):
                after = df[df.index > sig_ts]
                if not after.empty and df.index[0] <= sig_ts:
                    res = {"outcome": "expired", "exit_price": float(after["close"].iloc[-1]),
                           "exit_ts": _bar_close(symbol, after.index[-1])}
                    logger.warning(
                        f"Señal #{sig_id} {symbol}: red de seguridad ({age_days:.1f}d > "
                        f"{safety_net_days(symbol):.0f}d) con solo {len(after)} velas; "
                        "se cierra al último cierre disponible."
                    )
                elif age_days > 2 * safety_net_days(symbol):
                    _update_signal_result(sig_id, "expired", None, None)
                    resolved_count += 1
                    logger.warning(f"Señal #{sig_id} {symbol}: sin velas que cubran la señal; expirada sin PnL.")
                    continue

            if res is None:
                continue

            fields = compute_result_fields(direction, entry, sl, res["exit_price"], pos_size)
            _update_signal_result(
                sig_id, res["outcome"], fields["pnl_usd"], fields["pnl_pct"],
                pnl_r=fields["pnl_r"], exit_price=res["exit_price"], resolved_ts=res.get("exit_ts"),
            )
            resolved_count += 1
            logger.info(
                f"Señal #{sig_id} {symbol} {direction.upper()} → {res['outcome'].upper()} | "
                f"{fields['pnl_r']:+.2f}R | PnL={fields['pnl_usd'] if fields['pnl_usd'] is not None else 'n/d'} USD"
            )
        except Exception as exc:
            logger.warning(f"Error resolviendo señal #{sig_id}: {exc}")
            from sqlalchemy.exc import SQLAlchemyError
            if isinstance(exc, SQLAlchemyError):
                _mark_db(f"resolver señal #{sig_id}: {exc}")

    if resolved_count:
        logger.info(f"Outcome tracker: {resolved_count} señales resueltas.")
    return resolved_count


def _load_outcome_ohlcv(symbol: str, days: int = None) -> pd.DataFrame:
    """
    Descarga el OHLCV reciente usado para resolver señales de un símbolo, SOLO
    con velas cerradas (la vela en formación podría tocar SL y TP más tarde).
    Usa la misma fuente que la generación de señales (sin precios sustitutos).
    """
    from config.settings import ASSETS
    from data.bars import drop_unclosed_bars

    source = ASSETS.get(symbol, {}).get("source", "binance")
    try:
        if source == "yfinance":
            from data.fetchers.gold_fetcher import fetch_yfinance_asset
            df = fetch_yfinance_asset(symbol, days=days or 75)
            return drop_unclosed_bars(df, "1d", session=True)
        if source == "twelvedata":
            from data.fetchers.twelvedata_fetcher import fetch_twelvedata_ohlcv
            df = fetch_twelvedata_ohlcv(symbol, interval="4h", outputsize=500)
            return drop_unclosed_bars(df, "4h")
        from data.fetchers.binance_fetcher import fetch_ohlcv
        df = fetch_ohlcv(symbol, timeframe="4h", days=days or 30)
        return drop_unclosed_bars(df, "4h")
    except Exception as exc:
        logger.warning(f"No se pudo cargar OHLCV de {symbol} para resolver: {exc}")
        return pd.DataFrame()


def check_outcome_detail(
    symbol: str,
    signal_ts: pd.Timestamp,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    df: pd.DataFrame = None,
    setup_ts: pd.Timestamp = None,
) -> dict | None:
    """
    Resultado de una señal con velas CERRADAS, con la MISMA convención que el
    validador (backtest/validate.py:_simulate_setup), Fase 1 punto 1.7:
      - vela del setup = i (la de setup_ts; en señales antiguas sin setup_ts, la
        vela que contiene la hora de la señal: entonces se emitía con la vela
        aún abierta);
      - la vela i+1 es la de ENTRADA (el mensaje llega justo tras el cierre de
        i) y SÍ se revisa: SL/TP se miran de i+1 a i+forward_bars, forward_bars
        velas en total (si una vela toca ambos, cuenta el SL);
      - salida por tiempo al cierre de la vela i+forward_bars (la hora que
        devuelve time_exit_deadline y que muestra el mensaje).
    Hasta la Fase 1 la vela de entrada no se revisaba (de i+2 a i+1+forward):
    un SL tocado en las primeras 4 horas no contaba, aunque en la cuenta real
    del usuario sí habría saltado.

    Returns:
        {"outcome": "win"|"loss"|"expired", "exit_price": float, "exit_ts": Timestamp}
        o None si todavía no se puede resolver (faltan velas o el DataFrame no
        cubre la fecha de la señal).
    """
    from config.settings import ASSETS

    forward_bars = ASSETS.get(symbol, {}).get("forward_bars", 24)

    if df is None:
        df = _load_outcome_ohlcv(symbol)
    if df is None or df.empty:
        return None

    # El histórico debe empezar ANTES de la señal: si empieza después, las
    # "primeras velas posteriores" serían otras y el resultado sería falso.
    anchor = setup_ts if setup_ts is not None else signal_ts
    if df.index[0] > anchor:
        logger.warning(
            f"{symbol}: el histórico cargado empieza el {df.index[0]} y la señal es del "
            f"{anchor}; no se puede resolver con estos datos."
        )
        return None

    if setup_ts is not None and setup_ts in df.index:
        setup_pos = int(df.index.get_loc(setup_ts))
    else:
        setup_pos = int(df.index.searchsorted(anchor, side="right")) - 1
    if setup_pos < 0:
        return None
    future = df.iloc[setup_pos + 1: setup_pos + 1 + forward_bars]
    if future.empty:
        return None  # Muy pronto para resolver

    for ts, bar in future.iterrows():
        if direction == "long":
            if bar["low"] <= stop_loss:
                return {"outcome": "loss", "exit_price": float(stop_loss), "exit_ts": _bar_close(symbol, ts)}
            if bar["high"] >= take_profit:
                return {"outcome": "win", "exit_price": float(take_profit), "exit_ts": _bar_close(symbol, ts)}
        else:  # short
            if bar["high"] >= stop_loss:
                return {"outcome": "loss", "exit_price": float(stop_loss), "exit_ts": _bar_close(symbol, ts)}
            if bar["low"] <= take_profit:
                return {"outcome": "win", "exit_price": float(take_profit), "exit_ts": _bar_close(symbol, ts)}

    # Se agotaron las velas del horizonte sin tocar SL ni TP → salida por tiempo
    if len(future) >= forward_bars:
        return {"outcome": "expired", "exit_price": float(future["close"].iloc[-1]),
                "exit_ts": _bar_close(symbol, future.index[-1])}

    return None  # Aún hay velas por venir


def time_exit_deadline(symbol: str, setup_ts) -> pd.Timestamp:
    """
    Hora UTC del cierre de la vela i+forward_bars (i = vela del setup): si a esa
    hora no se ha tocado ni el SL ni el TP, la operación se cierra a mercado.
    Es la misma vela en la que check_outcome_detail da la salida por tiempo.

      - Cripto 4H: setup_ts es la APERTURA de la vela i, así que la vela
        i+forward_bars cierra en setup_ts + (forward_bars + 1) × 4h.
      - Diarios de sesión (oro, SPX; yfinance): forward_bars sesiones bursátiles
        después de la del setup (calendario de NYSE, o COMEX para el oro: L-V sin
        festivos ni cierres excepcionales; data/market_calendar.py), a la hora de cierre de la vela
        diaria (DAILY_BAR_CLOSE_UTC, 22:15 UTC). yfinance no tiene vela en esos
        días, así que la vela número forward_bars que cuenta el tracker cae en la
        misma fecha. (Hasta la revisión de la Fase 1 se usaba BDay, que no conoce
        los festivos: el mensaje decía un día y el tracker cerraba uno a seis días
        después en ~62 % de los casos.) Solo un cierre imprevisto o una rareza de
        yfinance (p. ej. vela de GC=F en un festivo) puede desplazarlo un día.
      - Forex 4H (Twelve Data, apagado): aproximación continua como en cripto
        (no descuenta el fin de semana).

    Args:
        setup_ts: vela del setup (Timestamp, datetime o segundos Unix).
    """
    from config.settings import ASSETS, DAILY_BAR_CLOSE_UTC

    fwd = int(ASSETS.get(symbol, {}).get("forward_bars", 24))
    t = pd.Timestamp(_unix(setup_ts), unit="s", tz="UTC")
    if _is_session_daily(symbol):
        from data.market_calendar import add_sessions, calendar_for
        h, m = DAILY_BAR_CLOSE_UTC
        # Si la fecha del setup no fuera sesión (fin de semana, festivo con vela
        # rara), la primera sesión que cuenta es la siguiente.
        target = add_sessions(t.normalize(), fwd, calendar_for(symbol))
        return target + pd.Timedelta(hours=h, minutes=m)
    return t + pd.Timedelta(hours=4) * (fwd + 1)


def _bar_close(symbol: str, bar_open_ts) -> pd.Timestamp:
    """Hora de CIERRE de la vela que abre en bar_open_ts (para resolved_ts)."""
    from data.bars import bar_close_times
    one = pd.DataFrame(index=pd.DatetimeIndex([bar_open_ts]))
    if _is_session_daily(symbol):
        return bar_close_times(one, "1d", session=True)[0]
    return bar_close_times(one, "4h")[0]


def _check_outcome(symbol, signal_ts, direction, entry, stop_loss, take_profit, df=None) -> tuple:
    """Compatibilidad: (outcome, pnl_pct) o (None, None)."""
    res = check_outcome_detail(symbol, signal_ts, direction, entry, stop_loss, take_profit, df=df)
    if res is None:
        return None, None
    sign = 1.0 if direction == "long" else -1.0
    return res["outcome"], round(sign * (res["exit_price"] - entry) / entry, 6)


def _update_signal_result(signal_id: int, outcome: str, pnl_usd, pnl_pct,
                          pnl_r=None, exit_price=None, resolved_ts=None) -> None:
    engine = get_engine()
    rts = None
    if resolved_ts is not None:
        rts = _unix(resolved_ts)
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE signals SET resultado_real=:outcome, pnl_usd=:pnl_usd, pnl_pct=:pnl_pct, "
            "pnl_r=:pnl_r, exit_price=:exit_price, resolved_ts=:rts, resolved_by='auto' WHERE id=:id"
        ), {"outcome": outcome, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "pnl_r": pnl_r,
            "exit_price": exit_price, "rts": rts, "id": signal_id})


# ---------------------------------------------------------------------------
# Cierre manual desde el dashboard (Fase 1, 1.9)
# ---------------------------------------------------------------------------

MANUAL_OUTCOMES = ("win", "loss", "be", "expired")


def resolve_signal_manually(signal_id: int, outcome: str, exit_price: float = None,
                            now=None) -> dict:
    """
    Cierra a mano una señal ABIERTA (formulario del dashboard).

    Escribe resolved_by='manual', resolved_ts=ahora y, si se puede calcular,
    exit_price, pnl_pct, pnl_r y pnl_usd (con el tamaño emitido):
      - con precio de salida → se calculan con compute_result_fields;
      - 'be' sin precio → salida = entrada, R = 0;
      - win/loss/expired sin precio → sin R (cuenta en los recuentos, no en el E[R]).
    Solo actúa sobre señales sin resultado (si el tracker la cerró antes, no la
    pisa). Devuelve {"updated": bool, "pnl_r", "pnl_usd", "exit_price", "motivo"}.
    Lanza ValueError si el resultado o el precio no son válidos.
    """
    if outcome not in MANUAL_OUTCOMES:
        raise ValueError(f"Resultado no válido: {outcome!r} (usa {', '.join(MANUAL_OUTCOMES)})")
    if exit_price is not None:
        exit_price = float(exit_price)
        if not exit_price > 0:
            raise ValueError("El precio de salida debe ser mayor que 0.")
    from data.database import init_db
    init_db()
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT id, direction, entry_price, stop_loss, position_size, resultado_real "
            "FROM signals WHERE id=:id"), {"id": int(signal_id)}).mappings().fetchone()
    if row is None:
        return {"updated": False, "motivo": f"no existe la señal #{signal_id}"}
    if row["resultado_real"] is not None:
        return {"updated": False, "motivo": f"la señal #{signal_id} ya estaba cerrada "
                                            f"({row['resultado_real']})"}

    entry, sl = float(row["entry_price"]), float(row["stop_loss"])
    if exit_price is None and outcome == "be":
        exit_price = entry
    fields = {"pnl_pct": None, "pnl_r": None, "pnl_usd": None}
    if exit_price is not None:
        fields = compute_result_fields(row["direction"], entry, sl, exit_price,
                                       row["position_size"])
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    with engine.begin() as conn:
        res = conn.execute(text(
            "UPDATE signals SET resultado_real=:o, pnl_usd=:u, pnl_pct=:p, pnl_r=:r, "
            "exit_price=:x, resolved_ts=:rts, resolved_by='manual' "
            "WHERE id=:id AND resultado_real IS NULL"),
            {"o": outcome, "u": fields["pnl_usd"], "p": fields["pnl_pct"], "r": fields["pnl_r"],
             "x": exit_price, "rts": _unix(now), "id": int(signal_id)})
        updated = (res.rowcount or 0) > 0
    if updated:
        logger.info(f"Señal #{signal_id} cerrada A MANO: {outcome} | salida={exit_price} | "
                    f"R={fields['pnl_r']}")
    return {"updated": updated, "pnl_r": fields["pnl_r"], "pnl_usd": fields["pnl_usd"],
            "exit_price": exit_price,
            "motivo": None if updated else "otra escritura la cerró a la vez"}


# ---------------------------------------------------------------------------
# Estadísticas del dashboard en R (Fase 1, 1.3 / 1.9)
# ---------------------------------------------------------------------------

# Cuantiles t de Student al 95 % (una cola) → intervalo bilateral del 90 %.
_T95 = [6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812,
        1.796, 1.782, 1.771, 1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725,
        1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701, 1.699, 1.697]


def _t95(df: int) -> float:
    """Cuantil 0.95 de la t con `df` grados de libertad (tabla hasta 30; después,
    desarrollo de Cornish-Fisher, error < 0.001). Evita depender de scipy."""
    if df < 1:
        raise ValueError("df debe ser ≥ 1")
    if df <= 30:
        return _T95[df - 1]
    z = 1.6448536269514722
    return (z + (z**3 + z) / (4 * df)
            + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2))


def mean_ci90(values) -> tuple:
    """(media, límite inferior, límite superior) del intervalo de confianza del
    90 % de la media con la t de Student. Con menos de 2 valores el intervalo es
    (None, None). Se usa la t y no un bootstrap porque con pocas señales el
    bootstrap da intervalos demasiado estrechos y además es aleatorio."""
    import math
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None, None, None
    mean = sum(vals) / n
    if n < 2:
        return mean, None, None
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    half = _t95(n - 1) * math.sqrt(var / n)
    return mean, mean - half, mean + half


def signal_r(row) -> float | None:
    """R de una señal resuelta: pnl_r; si falta, se deduce de pnl_pct (filas
    antiguas: R = pnl_pct × entrada / |entrada − SL|); si tampoco hay pnl_pct,
    del resultado: 'loss' = −1R (salió por el SL), 'win' = |TP − entrada| /
    |entrada − SL| (salió por el TP), 'be' = 0. None solo si no se puede saber
    (expirada sin precio de salida, o faltan los niveles).

    (Revisión de la Fase 1: antes una 'loss' sin pnl guardado quedaba FUERA del
    E[R], que así salía sesgado hacia arriba: BTC +0.75R en vez de +0.60R.)
    La misma regla usa monitoring.sequential._r_value."""
    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(f) else f
    r = _num(row.get("pnl_r"))
    if r is not None:
        return r
    pct, entry, sl = _num(row.get("pnl_pct")), _num(row.get("entry_price")), _num(row.get("stop_loss"))
    if pct is not None and entry and sl is not None and abs(entry - sl) > 0:
        return pct * entry / abs(entry - sl)
    return r_from_outcome(row.get("resultado_real"), entry, sl, _num(row.get("take_profit")))


def r_from_outcome(resultado, entry=None, stop_loss=None, take_profit=None) -> float | None:
    """R implícito en el resultado cuando no se guardó el precio de salida:
    loss = −1, win = |TP − entrada| / |entrada − SL|, be = 0; si no, None."""
    if resultado == "be":
        return 0.0
    if resultado == "loss":
        return -1.0
    if resultado == "win":
        try:
            e, s, t = float(entry), float(stop_loss), float(take_profit)
        except (TypeError, ValueError):
            return None
        if any(pd.isna(v) for v in (e, s, t)) or abs(e - s) <= 0:
            return None
        return abs(t - e) / abs(e - s)
    return None


def get_outcome_stats(symbol: str = None) -> dict:
    """
    Estadísticas de señales para el dashboard, en R (Fase 1, 1.3 / 1.9).

    Universo: señales LIVE (las que recibe el usuario) sin quality_flag, de un
    activo o de todos (symbol=None), SIN límite de filas.
      - resueltas = cualquier resultado: win (TP), loss (SL), expired (salida por
        tiempo) y be (break-even, R = 0 si no hay precio);
      - win_rate = ganadas / resueltas;
      - E[R] medio con intervalo de confianza del 90 % (t de Student) y suma de R,
        sobre las resueltas cuyo R se conoce (n_r);
      - manual / auto: cuántas cerró el formulario y cuántas el tracker.
    Devuelve {} si no hay señales live.
    """
    df = load_signals(symbol=symbol, limit=None)
    if df.empty:
        return {}
    # Fase 0: las estadísticas solo cuentan señales notificadas (modo live) y
    # sin marca de calidad (las re-emisiones detectadas en la limpieza no cuentan).
    if "mode" in df.columns:
        df = df[df["mode"].fillna("live") == "live"]
    if "quality_flag" in df.columns:
        df = df[df["quality_flag"].isna()]
    if df.empty:
        return {}

    resolved = df[df["resultado_real"].notna()]
    out = {
        "total":    len(df),
        "resolved": len(resolved),
        "pending":  len(df) - len(resolved),
        "wins":     int((resolved["resultado_real"] == "win").sum()),
        "losses":   int((resolved["resultado_real"] == "loss").sum()),
        "expired":  int((resolved["resultado_real"] == "expired").sum()),
        "be":       int((resolved["resultado_real"] == "be").sum()),
        "win_rate": None, "n_r": 0, "mean_r": None, "ci90_low": None, "ci90_high": None,
        "sum_r": 0.0, "manual": 0, "auto": 0,
    }
    out["other"] = out["resolved"] - out["wins"] - out["losses"] - out["expired"] - out["be"]
    if resolved.empty:
        return out
    out["win_rate"] = round(out["wins"] / out["resolved"], 4)
    if "resolved_by" in resolved.columns:
        out["manual"] = int((resolved["resolved_by"] == "manual").sum())
        out["auto"] = int((resolved["resolved_by"] == "auto").sum())
    rs = [r for r in (signal_r(row) for row in resolved.to_dict("records")) if r is not None]
    mean, lo, hi = mean_ci90(rs)
    out.update(n_r=len(rs), sum_r=round(sum(rs), 4),
               mean_r=None if mean is None else round(mean, 4),
               ci90_low=None if lo is None else round(lo, 4),
               ci90_high=None if hi is None else round(hi, 4))
    return out
