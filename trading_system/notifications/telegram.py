"""
Notificaciones por Telegram usando la Bot API.

Requiere variables de entorno:
  TELEGRAM_BOT_TOKEN  — token del bot (BotFather)
  TELEGRAM_CHAT_ID    — ID del chat o canal destino

No añade dependencias nuevas: usa requests (ya en requirements.txt).
"""
import html
import time
import requests
from loguru import logger

from config.settings import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, fmt_price

_BASE_URL = "https://api.telegram.org/bot{token}/{method}"


def _send(text: str, parse_mode: str = "HTML", retries: int = 3) -> bool:
    """Envía un mensaje al chat configurado. Retorna True si tuvo éxito."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram no configurado (TOKEN o CHAT_ID ausentes). Notificación omitida.")
        return False

    url = _BASE_URL.format(token=TELEGRAM_TOKEN, method="sendMessage")
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    delay = 2
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.debug("Telegram: mensaje enviado correctamente.")
                return True
            logger.warning(
                f"Telegram API respondió {resp.status_code}: {resp.text[:200]}"
            )
        except requests.RequestException as exc:
            logger.warning(f"Error enviando Telegram (intento {attempt}/{retries}): {exc}")

        if attempt < retries:
            time.sleep(delay)
            delay *= 2

    logger.error("No se pudo enviar notificación a Telegram tras múltiples intentos.")
    return False


def _e(value) -> str:
    """Escapa HTML de cualquier texto dinámico (Telegram rechaza el mensaje
    entero si aparece un '<' o '&' sin escapar)."""
    return html.escape(str(value), quote=False)


def _fmt_duration(hours: float) -> str:
    """Duración legible: '3 d 4 h', '2 h 5 min', '25 min'."""
    mins = int(round(max(0.0, float(hours)) * 60))
    d, rem = divmod(mins, 24 * 60)
    h, m = divmod(rem, 60)
    if d:
        return f"{d} d {h} h" if h else f"{d} d"
    if h:
        return f"{h} h {m} min" if m else f"{h} h"
    return f"{m} min"


def _utc(ts) -> "pd.Timestamp":
    import pandas as pd
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _fmt_pct(x: float) -> str:
    """Porcentaje con 2 decimales y punto decimal, como los precios: 0.0023 → '0.23 %'."""
    return f"{x * 100:.2f} %"


def _fmt_qty(q: float) -> str:
    """Cantidad del activo con decimales según magnitud y sin ceros sobrantes
    (0.014509 BTC, 0.2399 oz, 12.5 unidades)."""
    q = float(q)
    dec = 6 if abs(q) < 1 else 4 if abs(q) < 100 else 2
    txt = f"{q:,.{dec}f}"
    return txt.rstrip("0").rstrip(".") if "." in txt else txt


def format_signal_alert(trade, sentiment: dict = None, *, setup_ts=None,
                        current_price: float = None, threshold: float = None,
                        capital: float = None, now=None) -> str:
    """
    Texto HTML de la alerta de señal (sin enviarlo). Se separa del envío para
    poder guardarlo en la BD (signals.alert_text) y reenviarlo idéntico si
    Telegram falla (Fase 1, notificación fiable).

    Formato Fase 1 (punto 1.1), en español claro:
      - "Score del setup: 0.62 (umbral 0.55)" (no es una probabilidad: el ML está
        apagado y el score es la puntuación técnica raw_score);
      - hora UTC de la vela del setup y cuánto hace que cerró;
      - entrada/SL/TP, precio actual (cierre de la última vela cerrada) y el R:R
        efectivo si se entra a ese precio;
      - riesgo REAL en % del capital (risk_usd / capital, con el tope de tamaño);
      - tamaño en unidades del activo y en LOTES de XM (XM_CONTRACTS; pendiente
        mientras no haya especificación del contrato);
      - "Cerrar a mercado si no toca SL/TP antes de: <fecha UTC>" = cierre de la
        vela i+forward_bars (outcome_tracker.time_exit_deadline, el mismo cálculo
        que usa el tracker para la salida por tiempo).
    Sin bloque de sentimiento (apagado y descartado). Todo texto dinámico va
    escapado. El pie "🆔 Señal #ID — /fill … /skip …" lo añade cmd_signal tras
    guardar la señal (notifications.telegram_commands.signal_id_footer).

    Args:
        trade:         TradeSetup con toda la info de la operación.
        sentiment:     sin uso (se acepta por compatibilidad).
        setup_ts:      vela del setup (hora de APERTURA, UTC). Sin ella se omiten
                       la hora de la vela y la fecha de cierre por tiempo.
        current_price: cierre de la última vela cerrada. Sin él se omite el R:R efectivo.
        threshold:     umbral de score del activo (por defecto, ASSETS[ml_threshold]).
        capital:       capital de la cuenta en USD (por defecto, TRADING_CAPITAL).
        now:           reloj inyectable (pruebas); por defecto, ahora UTC.
    """
    import pandas as pd
    from config.settings import ASSETS, RISK, TRADING_CAPITAL
    from risk.position_sizer import xm_lot_size

    cfg = ASSETS.get(trade.symbol, {})
    session_daily = cfg.get("source") == "yfinance"
    if threshold is None:
        threshold = cfg.get("ml_threshold")
    if capital is None:
        capital = TRADING_CAPITAL
    now = _utc(now if now is not None else pd.Timestamp.now(tz="UTC"))

    is_long = trade.direction == "long"
    direction_icon = "🟢" if is_long else "🔴"
    direction_label = "LONG ▲ (compra)" if is_long else "SHORT ▼ (venta)"
    setup_label = str(trade.setup_type).replace("_", " ").title()

    score_txt = f"{float(trade.ml_score):.2f}"
    if threshold is not None:
        score_txt += f" (umbral {float(threshold):.2f})"

    lines = [
        f"{direction_icon} <b>SEÑAL — {_e(trade.symbol)}</b>",
        "",
        f"<b>Dirección:</b> {_e(direction_label)}",
        f"<b>Setup:</b> {_e(setup_label)}",
        f"<b>Score del setup:</b> {_e(score_txt)}",
    ]

    # Vela del setup y antigüedad (desde su CIERRE).
    if setup_ts is not None:
        from data.bars import bar_close_times
        st = _utc(setup_ts)
        one = pd.DataFrame(index=pd.DatetimeIndex([st]))
        if session_daily:
            close = bar_close_times(one, "1d", session=True)[0]
            vela = f"sesión del {st:%Y-%m-%d} (vela diaria, cierra {close:%H:%M} UTC)"
        else:
            close = bar_close_times(one, "4h")[0]
            vela = f"{st:%Y-%m-%d %H:%M} UTC (vela 4H, cerró {close:%H:%M} UTC)"
        age_h = (now - close).total_seconds() / 3600
        lines.append(f"<b>Vela del setup:</b> {_e(vela)}")
        lines.append(f"<b>Antigüedad:</b> cerró hace {_e(_fmt_duration(age_h))}")

    lines += [
        "",
        f"<b>Entrada:</b> ${_e(fmt_price(trade.entry_price))} <i>(cierre de la vela del setup)</i>",
        f"<b>Stop Loss:</b> ${_e(fmt_price(trade.stop_loss))}",
        f"<b>Take Profit:</b> ${_e(fmt_price(trade.take_profit))}",
        f"<b>R:R a la entrada:</b> {float(trade.rr_ratio):.2f}",
    ]

    if current_price is not None:
        from notifications.telegram_commands import effective_rr
        cp = float(current_price)
        lines.append(f"<b>Precio actual:</b> ${_e(fmt_price(cp))} "
                     "<i>(cierre de la última vela cerrada)</i>")
        rr_eff = effective_rr(trade.direction, cp, float(trade.stop_loss), float(trade.take_profit))
        if rr_eff is None:
            lines.append("⚠️ <b>A ese precio la operación ya no tiene recorrido</b> "
                         "(está en o más allá del SL o del TP): no entrar.")
        else:
            lines.append(f"<b>R:R efectivo a ese precio:</b> {rr_eff:.2f}")

    # Riesgo real y tamaño.
    lines.append("")
    risk_usd = float(trade.risk_usd)
    if capital and capital > 0:
        risk_pct = risk_usd / float(capital)
        objetivo = float(RISK.get("risk_per_trade", 0.01))
        nota = ""
        if risk_pct < objetivo - 1e-6:
            nota = (f" <i>(objetivo {_e(_fmt_pct(objetivo))}; lo limita el tope de tamaño "
                    "del 10 % del capital)</i>")
        lines.append(f"<b>Capital:</b> ${float(capital):,.0f}")
        lines.append(f"<b>Riesgo real:</b> ${risk_usd:,.2f} = {_e(_fmt_pct(risk_pct))} del capital{nota}")
    else:
        lines.append(f"<b>Riesgo:</b> ${risk_usd:,.2f}")

    qty = getattr(trade, "position_qty", None)
    if qty is None:   # objetos antiguos / dobles de prueba sin la cantidad
        qty = float(trade.position_size) / float(trade.entry_price)
    lots = xm_lot_size(trade.symbol, qty, entry=trade.entry_price,
                       stop_loss=trade.stop_loss, capital=capital)
    unit = lots["unit"]
    lines.append(f"<b>Tamaño:</b> {_e(_fmt_qty(qty))} {_e(unit)} "
                 f"(${float(trade.position_size):,.0f} de nominal)")
    if lots["status"] == "pending":
        lines.append("<b>Lotes XM:</b> pendiente de la especificación del contrato "
                     f"<i>(opera {_e(_fmt_qty(qty))} {_e(unit)})</i>")
    elif lots["status"] == "below_min":
        txt = (f"el tamaño no llega al lote mínimo de XM ({lots['min_lot']:g}). "
               "No operar o asumir más riesgo")
        if lots["risk_pct"] is not None:
            txt += f": con {lots['min_lot']:g} lotes el riesgo sería {_fmt_pct(lots['risk_pct'])} del capital"
        lines.append(f"<b>Lotes XM:</b> {_e(txt)}")
    else:
        txt = f"{lots['lots']:g}"
        if lots["risk_pct"] is not None:
            txt += f" (riesgo con ese tamaño: ${lots['risk_usd']:,.2f} = {_fmt_pct(lots['risk_pct'])})"
        lines.append(f"<b>Lotes XM:</b> {_e(txt)}")

    if setup_ts is not None:
        from data.outcome_tracker import time_exit_deadline
        deadline = time_exit_deadline(trade.symbol, setup_ts)
        left_h = (deadline - now).total_seconds() / 3600
        falta = f" (dentro de {_fmt_duration(left_h)})" if left_h > 0 else ""
        lines += [
            "",
            f"⏱ <b>Cerrar a mercado si no toca SL/TP antes de:</b> "
            f"{_e(f'{deadline:%Y-%m-%d %H:%M} UTC')}{_e(falta)}",
        ]

    return "\n".join(lines)


def safe_format_signal_alert(trade, sentiment: dict = None, **kwargs) -> str:
    """format_signal_alert que nunca lanza: si algo falla al construir el mensaje
    completo (dato raro, fecha imposible…), se registra el error y se devuelve un
    mensaje mínimo con lo imprescindible (dirección, entrada, SL, TP) para que la
    señal se guarde y se envíe igualmente."""
    try:
        return format_signal_alert(trade, sentiment, **kwargs)
    except Exception as exc:
        logger.error(f"No se pudo construir el mensaje completo de {trade.symbol}: {exc}; "
                     "se envía el mínimo.")
        return _fallback_alert_text({
            "symbol": trade.symbol, "direction": trade.direction,
            "setup_type": trade.setup_type, "entry_price": trade.entry_price,
            "stop_loss": trade.stop_loss, "take_profit": trade.take_profit})


def send_signal_alert(trade, sentiment: dict = None, **kwargs) -> bool:
    """Envía la alerta de señal. True si Telegram confirmó el envío.
    kwargs: los de format_signal_alert (setup_ts, current_price, threshold, capital)."""
    return _send(safe_format_signal_alert(trade, sentiment, **kwargs))


RESEND_NOTE = "🔁 <i>Reenvío tras fallo de Telegram</i>"


def send_text(text: str) -> bool:
    """Envía un texto HTML ya formateado (p. ej. signals.alert_text)."""
    return _send(text)


def send_signal_resend(text: str) -> bool:
    """Reenvía una alerta cuyo primer envío falló, con una nota al principio
    para que el usuario sepa que es la misma señal y no una nueva."""
    return _send(f"{RESEND_NOTE}\n\n{text}")


# IDs de señales demasiado viejas para reenviar ya avisadas en el log (una vez
# por proceso, para no repetir la misma línea cada hora).
_stale_logged: set = set()


def _fallback_alert_text(row: dict) -> str:
    """Mensaje mínimo si la fila no guardó alert_text (no debería pasar)."""
    d = str(row.get("direction") or "")
    return (
        f"{'🟢' if d == 'long' else '🔴'} <b>SEÑAL — {html.escape(str(row.get('symbol')))}</b>\n\n"
        f"<b>Dirección:</b> {html.escape(d.upper())}\n"
        f"<b>Setup:</b> {html.escape(str(row.get('setup_type') or ''))}\n"
        f"<b>Entrada:</b> ${fmt_price(float(row['entry_price']))}\n"
        f"<b>Stop Loss:</b> ${fmt_price(float(row['stop_loss']))}\n"
        f"<b>Take Profit:</b> ${fmt_price(float(row['take_profit']))}"
    )


def retry_failed_signal_alerts(max_age_hours: float = None, now=None) -> dict:
    """
    Reintenta las señales live cuyo envío falló (signals.notified = 0).

      - Menos de `max_age_hours` (NOTIFY_RETRY_MAX_AGE_HOURS, 2 h) desde que se
        registró: se reenvía el MISMO texto con la nota "Reenvío tras fallo de
        Telegram" y, si Telegram confirma, se marca notified = 1.
      - Más antigua: se deja notified = 0 (queda constancia en la BD de que el
        usuario nunca la recibió) y se registra en el log una vez por proceso.

    Nunca lanza: cualquier error se registra y el ciclo sigue.
    Devuelve {"resent": n, "failed": n, "stale": n}.
    """
    import pandas as pd
    from config.settings import NOTIFY_RETRY_MAX_AGE_HOURS
    out = {"resent": 0, "failed": 0, "stale": 0}
    if max_age_hours is None:
        max_age_hours = NOTIFY_RETRY_MAX_AGE_HOURS
    try:
        from data.database import load_unnotified_signals, mark_signal_notified
        rows = load_unnotified_signals()
    except Exception as exc:
        logger.error(f"Reintento de notificaciones: no se pudieron leer las señales pendientes: {exc}")
        return out
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    for row in rows:
        try:
            ts = row["ts"]
            sig_ts = (pd.Timestamp(int(ts), unit="s", tz="UTC") if isinstance(ts, (int, float))
                      else pd.Timestamp(ts))
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.tz_localize("UTC")
            age_h = (now - sig_ts).total_seconds() / 3600
            if age_h >= max_age_hours:
                out["stale"] += 1
                if row["id"] not in _stale_logged:
                    _stale_logged.add(row["id"])
                    logger.error(
                        f"Señal #{row['id']} {row['symbol']} {str(row['direction']).upper()} NO se pudo "
                        f"notificar y ya tiene {age_h:.1f} h (> {max_age_hours:g} h): no se reenvía; "
                        "queda con notified=0 en la BD."
                    )
                continue
            text = row.get("alert_text") or _fallback_alert_text(row)
            if send_signal_resend(text):
                mark_signal_notified(row["id"], True)
                out["resent"] += 1
                logger.info(f"Señal #{row['id']} {row['symbol']}: reenviada tras fallo de Telegram.")
            else:
                out["failed"] += 1
                logger.warning(f"Señal #{row['id']} {row['symbol']}: el reenvío volvió a fallar; "
                               "se reintentará en el próximo ciclo.")
        except Exception as exc:
            out["failed"] += 1
            logger.error(f"Reintento de la señal #{row.get('id')}: {exc}")
    return out


def send_no_signal(symbol: str, reason: str = "") -> bool:
    """Notificación silenciosa cuando no hay señal (útil para modo daemon)."""
    msg = f"ℹ️ <b>{symbol}</b>: Sin señal de alta confianza en este ciclo."
    if reason:
        msg += f"\n<i>{reason}</i>"
    return _send(msg)


def send_error_alert(symbol: str, error: str) -> bool:
    """Alerta de error en el procesamiento de un símbolo."""
    # Escapar: un mensaje de excepción con "<" o "&" (p. ej. "<Response [429]>")
    # hacía que Telegram rechazara el HTML y la alerta de error se perdía.
    msg = (
        f"⚠️ <b>Error en {html.escape(str(symbol))}</b>\n"
        f"<code>{html.escape(str(error)[:400])}</code>"
    )
    return _send(msg)


def send_data_alert(symbol: str, motivo: str, since=None) -> bool:
    """Aviso de que un activo se saltó el ciclo por un problema de datos
    (Fase 0): mejor no emitir que emitir con precios equivocados o viejos."""
    desde = f"\n<i>Desde: {since:%Y-%m-%d %H:%M} UTC</i>" if since else ""
    msg = (
        f"⚠️ <b>{html.escape(symbol)}: ciclo saltado por datos</b>\n"
        f"{html.escape(motivo)}\n"
        "No se emiten señales de este activo hasta que la fuente responda bien."
        f"{desde}"
    )
    return _send(msg)


def send_data_recovered(symbol: str, since=None) -> bool:
    """Aviso de que la fuente de datos de un activo volvió a funcionar."""
    desde = f" (fallaba desde {since:%Y-%m-%d %H:%M} UTC)" if since else ""
    return _send(f"✅ <b>{html.escape(symbol)}</b>: datos de nuevo disponibles{desde}.")


def send_daily_summary(summary: dict) -> bool:
    """
    Resumen diario del estado del portfolio.

    Args:
        summary: dict con campos de PortfolioRiskManager.summary()
    """
    n = summary.get("n_trades", 0)
    if n == 0:
        return _send("📊 <b>Resumen diario</b>\nSin operaciones cerradas hoy.")

    wr = summary.get("win_rate", 0)
    pnl = summary.get("total_pnl_usd", 0)
    pnl_pct = summary.get("total_return_pct", 0)
    capital = summary.get("current_capital", 0)
    dd = summary.get("drawdown", 0)
    avg_rr = summary.get("avg_rr", 0)

    icon = "📈" if pnl >= 0 else "📉"
    sign = "+" if pnl >= 0 else ""

    msg = (
        f"{icon} <b>Resumen diario — Portfolio</b>\n\n"
        f"<b>Operaciones:</b>  {n}\n"
        f"<b>Win Rate:</b>     {wr:.0%}\n"
        f"<b>P&amp;L:</b>           {sign}${pnl:,.2f} ({sign}{pnl_pct:.2f}%)\n"
        f"<b>Capital:</b>      ${capital:,.2f}\n"
        f"<b>Drawdown:</b>     {dd:.2f}%\n"
        f"<b>R:R promedio:</b> {avg_rr:.2f}"
    )
    return _send(msg)


def send_retrain_alert(symbol: str, metrics: dict) -> bool:
    """Notificación de reentrenamiento automático con métricas del nuevo modelo."""
    auc       = metrics.get("auc", 0)
    precision = metrics.get("precision", 0)
    n_folds   = metrics.get("n_folds", 0)
    n_samples = metrics.get("n_samples", 0)
    threshold = metrics.get("adaptive_threshold", metrics.get("threshold", 0))

    icon = "🟢" if auc >= 0.55 else "🟡" if auc >= 0.50 else "🔴"

    msg = (
        f"🔁 <b>Reentrenamiento automático — {symbol}</b>\n\n"
        f"{icon} <b>AUC Walk-Forward:</b> {auc:.3f}\n"
        f"<b>Precision (p0.50):</b>  {precision:.3f}\n"
        f"<b>Umbral efectivo:</b>    {threshold:.4f}\n"
        f"<b>Folds:</b>              {n_folds}\n"
        f"<b>Muestras:</b>           {n_samples}\n\n"
        f"<i>Modelo actualizado y listo para operar.</i>"
    )
    return _send(msg)


def test_connection() -> bool:
    """Verifica que el bot y el chat_id son válidos enviando un mensaje de prueba."""
    msg = "✅ <b>Sistema de Trading</b>\nConexión a Telegram verificada correctamente."
    ok = _send(msg)
    if ok:
        logger.info("Telegram: conexión verificada.")
    else:
        logger.error("Telegram: fallo de verificación.")
    return ok
