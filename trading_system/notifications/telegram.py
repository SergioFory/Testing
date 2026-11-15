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


def format_signal_alert(trade, sentiment: dict = None) -> str:
    """
    Texto HTML de la alerta de señal (sin enviarlo). Se separa del envío para
    poder guardarlo en la BD (signals.alert_text) y reenviarlo idéntico si
    Telegram falla (Fase 1, notificación fiable).

    Args:
        trade:     TradeSetup con toda la info de la operación
        sentiment: dict con 'label', 'score', 'color', 'n_news' (opcional; con
                   SENTIMENT['enabled']=False el worker pasa None)
    """
    direction_icon = "🟢" if trade.direction == "long" else "🔴"
    direction_label = "LONG  ▲" if trade.direction == "long" else "SHORT ▼"
    setup_label = trade.setup_type.replace("_", " ").title()

    lines = [
        f"{direction_icon} <b>SEÑAL — {trade.symbol}</b>",
        "",
        f"<b>Dirección:</b>  {direction_label}",
        f"<b>Setup:</b>      {setup_label}",
        f"<b>Confianza ML:</b> {trade.ml_score:.1%}",
        "",
        f"<b>Entrada:</b>    ${fmt_price(trade.entry_price)}",
        f"<b>Stop Loss:</b>  ${fmt_price(trade.stop_loss)}",
        f"<b>Take Profit:</b> ${fmt_price(trade.take_profit)}",
        f"<b>R:R Ratio:</b>  {trade.rr_ratio:.2f}",
        "",
        f"<b>Tamaño ($):</b>  ${trade.position_size:,.0f}",
        f"<b>Riesgo ($):</b>  ${trade.risk_usd:,.0f}",
        f"<b>Potencial ($):</b> ${trade.reward_usd:,.0f}",
    ]

    if sentiment:
        sent_label = sentiment.get("label", "NEUTRAL")
        sent_score = sentiment.get("score", 0.0)
        n_news = sentiment.get("n_news", 0)
        lines += [
            "",
            f"<b>Sentimiento:</b> {sent_label} ({sent_score:+.2f})",
            f"<b>Noticias:</b>    {n_news} artículos recientes",
        ]

    return "\n".join(lines)


def send_signal_alert(trade, sentiment: dict = None) -> bool:
    """Envía la alerta de señal. True si Telegram confirmó el envío."""
    return _send(format_signal_alert(trade, sentiment))


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
