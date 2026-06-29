"""
Notificaciones por Telegram usando la Bot API.

Requiere variables de entorno:
  TELEGRAM_BOT_TOKEN  — token del bot (BotFather)
  TELEGRAM_CHAT_ID    — ID del chat o canal destino

No añade dependencias nuevas: usa requests (ya en requirements.txt).
"""
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


def send_signal_alert(trade, sentiment: dict = None) -> bool:
    """
    Envía alerta de señal de trading al canal de Telegram.

    Args:
        trade:     TradeSetup con toda la info de la operación
        sentiment: dict con 'label', 'score', 'color', 'n_news' (opcional)

    Returns:
        True si el mensaje fue enviado con éxito.
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

    return _send("\n".join(lines))


def send_no_signal(symbol: str, reason: str = "") -> bool:
    """Notificación silenciosa cuando no hay señal (útil para modo daemon)."""
    msg = f"ℹ️ <b>{symbol}</b>: Sin señal de alta confianza en este ciclo."
    if reason:
        msg += f"\n<i>{reason}</i>"
    return _send(msg)


def send_error_alert(symbol: str, error: str) -> bool:
    """Alerta de error en el procesamiento de un símbolo."""
    msg = (
        f"⚠️ <b>Error en {symbol}</b>\n"
        f"<code>{error[:400]}</code>"
    )
    return _send(msg)


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
