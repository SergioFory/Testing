"""
Comandos de Telegram (Fase 1, punto 1.10): el usuario cuenta al sistema qué hizo
con cada señal, para comparar más adelante el R EJECUTADO con el R teórico.

    /fill <id> <precio>   la ejecuté a ese precio  (p. ej. /fill 123 65432.5)
    /skip <id> <motivo>   no la ejecuté            (p. ej. /skip 123 llegué tarde)
    /help                 ayuda

Cómo funciona: no hay servidor web. Al INICIO de cada ciclo el worker pregunta a
Telegram por mensajes nuevos (getUpdates con timeout 0, no espera) a partir del
último offset guardado en la tabla kv, procesa los comandos y guarda el offset
nuevo. Solo se aceptan mensajes del chat configurado (TELEGRAM_CHAT_ID); los
demás se ignoran sin responder. Cada fill/skip va a la tabla fills con el
update_id de Telegram (UNIQUE): si un mensaje se procesa dos veces (p. ej. el
worker se reinicia antes de guardar el offset) no se duplica ni se responde dos
veces.

Nada de esto puede bloquear ni tumbar el ciclo: poll_commands nunca lanza.
"""
from __future__ import annotations

import html
import re

import requests
from loguru import logger

import notifications.telegram as tg
from config.settings import fmt_price

KV_OFFSET = "telegram_update_offset"

HELP_TEXT = (
    "🤖 <b>Comandos</b>\n\n"
    "<code>/fill ID PRECIO</code> — ejecutaste la señal #ID a ese precio.\n"
    "   Ej.: <code>/fill 123 65432.5</code>\n"
    "<code>/skip ID MOTIVO</code> — no la ejecutaste (y por qué).\n"
    "   Ej.: <code>/skip 123 llegué tarde</code>\n"
    "<code>/help</code> — esta ayuda.\n\n"
    "<i>El número de la señal (ID) aparece en el mensaje de la señal. Los comandos se leen "
    "al principio de cada ciclo (cada hora), así que la respuesta puede tardar hasta una hora.</i>"
)


# ---------------------------------------------------------------------------
# Utilidades puras (probadas sin red)
# ---------------------------------------------------------------------------

def parse_price(raw: str) -> float | None:
    """Precio escrito por una persona → float, o None si no es un número > 0.
    Acepta '65432.5', '65432,5', '65,432.5', '65.432,5', '2 650,5' y '$65432'."""
    s = str(raw).strip().replace("$", "").replace(" ", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        # El separador que aparece el ÚLTIMO es el decimal.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Solo comas: '65,432' (miles, grupos de 3) o '2650,5' (decimal).
        parts = s.split(",")
        if len(parts) > 2 or (len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3
                              and parts[0] not in ("0", "")):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 and v == v and v != float("inf") else None


def effective_rr(direction: str, price: float, stop_loss: float, take_profit: float) -> float | None:
    """R:R efectivo entrando a `price`: distancia al TP / distancia al SL.
    None si a ese precio la operación ya no tiene sentido (precio en o más allá
    del SL o del TP)."""
    if direction == "long":
        risk, reward = price - stop_loss, take_profit - price
    else:
        risk, reward = stop_loss - price, price - take_profit
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def entry_diff_r(direction: str, price: float, entry: float, stop_loss: float) -> float | None:
    """Diferencia entre tu precio y la entrada de la señal, en R de la señal.
    Positiva = a tu favor (compraste más barato / vendiste más caro)."""
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None
    sign = 1.0 if direction == "long" else -1.0
    return sign * (entry - price) / risk


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """'/fill@MiBot 12 100' → ('fill', ['12', '100']). None si no es un comando."""
    if not text or not text.strip().startswith("/"):
        return None
    parts = text.strip().split()
    cmd = parts[0][1:].split("@", 1)[0].lower()
    return cmd, parts[1:]


def _parse_id(raw: str) -> int | None:
    m = re.fullmatch(r"#?(\d{1,12})", raw.strip())
    return int(m.group(1)) if m else None


def _chat_matches(chat: dict) -> bool:
    """True si el mensaje viene del chat configurado (por id o por @usuario)."""
    want = str(tg.TELEGRAM_CHAT_ID or "").strip()
    if not want or not isinstance(chat, dict):
        return False
    if str(chat.get("id")) == want:
        return True
    user = chat.get("username")
    return bool(user) and want.lstrip("@").lower() == str(user).lower() and want.startswith("@")


def _fmt(v: float) -> str:
    return html.escape(fmt_price(float(v)))


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def handle_command(cmd: str, args: list[str], update_id: int | None = None, ts=None) -> str | None:
    """Ejecuta un comando y devuelve el texto HTML de la respuesta, o None si no
    hay que responder (mensaje ya procesado antes). No envía nada."""
    from data.database import load_signal_by_id, save_fill, load_fills

    if cmd in ("help", "start", "ayuda"):
        return HELP_TEXT
    if cmd not in ("fill", "skip"):
        return f"No conozco el comando /{html.escape(cmd)}.\n\n{HELP_TEXT}"

    usage = ("Uso: <code>/fill ID PRECIO</code> (p. ej. <code>/fill 123 65432.5</code>)"
             if cmd == "fill" else
             "Uso: <code>/skip ID MOTIVO</code> (p. ej. <code>/skip 123 llegué tarde</code>)")
    if not args:
        return usage
    sig_id = _parse_id(args[0])
    if sig_id is None:
        return f"«{html.escape(args[0])}» no es un número de señal.\n{usage}"
    sig = load_signal_by_id(sig_id)
    if sig is None:
        return f"No existe la señal #{sig_id}. Revisa el número en el mensaje de la señal."

    symbol = str(sig.get("symbol") or "")
    direction = str(sig.get("direction") or "")
    head = f"#{sig_id} {html.escape(symbol)} {html.escape(direction.upper())}"
    prev = load_fills(sig_id)
    nota_prev = (f"\n<i>Ya había {len(prev)} registro(s) de esta señal; se tomará el más reciente.</i>"
                 if not prev.empty else "")

    if cmd == "skip":
        note = " ".join(args[1:]).strip()[:500] or None
        if not save_fill(sig_id, "skip", None, note, ts=ts, update_id=update_id):
            return None
        return (f"📝 <b>Anotado: no ejecutaste la señal {head}</b>\n"
                f"Motivo: {html.escape(note) if note else '(sin motivo)'}{nota_prev}")

    # /fill
    if len(args) < 2:
        return usage
    price = parse_price(args[1])
    if price is None:
        return f"«{html.escape(args[1])}» no es un precio válido.\n{usage}"
    entry, sl, tp = float(sig["entry_price"]), float(sig["stop_loss"]), float(sig["take_profit"])
    note = " ".join(args[2:]).strip()[:500] or None
    if not save_fill(sig_id, "fill", price, note, ts=ts, update_id=update_id):
        return None

    rr = effective_rr(direction, price, sl, tp)
    rr_sig = effective_rr(direction, entry, sl, tp)
    diff = entry_diff_r(direction, price, entry, sl)
    lines = [
        f"✅ <b>Ejecución registrada — señal {head}</b>",
        "",
        f"Tu precio: {_fmt(price)} (entrada de la señal: {_fmt(entry)})",
        f"Stop Loss: {_fmt(sl)} · Take Profit: {_fmt(tp)}",
    ]
    if rr is None:
        lines.append("⚠️ <b>Ese precio está en o más allá del Stop Loss o del Take Profit</b>: "
                     "con él la operación no tiene recorrido. Revisa el precio o cierra la posición.")
    else:
        rr_sig_txt = f" (el de la señal: {rr_sig:.2f})" if rr_sig is not None else ""
        lines.append(f"<b>R:R efectivo a tu precio: {rr:.2f}</b>{rr_sig_txt}")
    if diff is not None:
        if abs(diff) < 0.005:
            lines.append("Entraste prácticamente al precio de la señal.")
        else:
            lado = "a tu favor" if diff > 0 else "en contra"
            lines.append(f"Diferencia de entrada: {diff:+.2f}R ({lado}).")
    if sig.get("resultado_real"):
        lines.append(f"<i>Nota: el sistema ya dio esta señal por cerrada "
                     f"({html.escape(str(sig['resultado_real']))}).</i>")
    if str(sig.get("mode") or "live") == "shadow":
        lines.append("<i>Nota: esta señal es de un activo en modo sombra (no se notificó).</i>")
    return "\n".join(lines) + nota_prev


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def _get_updates(offset: int | None) -> list[dict] | None:
    """getUpdates sin espera (timeout 0). None si falla."""
    url = tg._BASE_URL.format(token=tg.TELEGRAM_TOKEN, method="getUpdates")
    params = {"timeout": 0, "allowed_updates": '["message","channel_post"]'}
    if offset is not None:
        params["offset"] = int(offset)
    try:
        resp = requests.get(url, params=params, timeout=10)
    except requests.RequestException as exc:
        logger.warning(f"Comandos Telegram: getUpdates falló ({exc}).")
        return None
    if resp.status_code != 200:
        # 409 = hay un webhook configurado o se está usando el bot en otro sitio.
        logger.warning(f"Comandos Telegram: getUpdates respondió {resp.status_code}: {resp.text[:200]}")
        return None
    data = resp.json()
    if not data.get("ok"):
        logger.warning(f"Comandos Telegram: respuesta no OK: {str(data)[:200]}")
        return None
    return data.get("result") or []


def poll_commands() -> dict:
    """
    Lee los mensajes nuevos de Telegram, procesa /fill /skip /help del chat
    configurado, responde y guarda el offset. Nunca lanza.
    Devuelve {"updates": n, "commands": n, "ignored": n, "errors": n}.
    """
    out = {"updates": 0, "commands": 0, "ignored": 0, "errors": 0}
    try:
        if not tg.TELEGRAM_TOKEN or not tg.TELEGRAM_CHAT_ID:
            return out
        from data.database import kv_get, kv_set
        try:
            raw = kv_get(KV_OFFSET)
            offset = int(raw) if raw not in (None, "") else None
        except Exception as exc:
            # Sin saber por dónde íbamos no se procesa nada (se repetirían comandos).
            logger.error(f"Comandos Telegram: no se pudo leer el offset de la BD ({exc}); se omite.")
            out["errors"] += 1
            return out

        updates = _get_updates(offset)
        if not updates:
            return out
        out["updates"] = len(updates)
        new_offset = offset
        for upd in updates:
            uid = upd.get("update_id")
            try:
                if uid is not None:
                    new_offset = max(new_offset or 0, int(uid) + 1)
                msg = upd.get("message") or upd.get("channel_post") or {}
                if not _chat_matches(msg.get("chat") or {}):
                    out["ignored"] += 1
                    chat_id = (msg.get("chat") or {}).get("id")
                    logger.warning(f"Comandos Telegram: mensaje de un chat no autorizado ({chat_id}); ignorado.")
                    continue
                parsed = parse_command(msg.get("text") or "")
                if parsed is None:
                    continue          # texto normal: no se responde
                cmd, args = parsed
                reply = handle_command(cmd, args, update_id=uid, ts=msg.get("date"))
                out["commands"] += 1
                if reply:
                    tg._send(reply)
                logger.info(f"Comandos Telegram: /{cmd} {' '.join(args)[:80]} procesado.")
            except Exception as exc:
                out["errors"] += 1
                logger.error(f"Comandos Telegram: error procesando la actualización {uid}: {exc}")
        if new_offset is not None and new_offset != offset:
            try:
                kv_set(KV_OFFSET, new_offset)
            except Exception as exc:
                out["errors"] += 1
                logger.error(f"Comandos Telegram: no se pudo guardar el offset {new_offset}: {exc} "
                             "(los fills no se duplicarán: update_id es único).")
    except Exception as exc:
        out["errors"] += 1
        logger.error(f"Comandos Telegram: error inesperado: {exc}")
    return out


def signal_id_footer(signal_id: int) -> str:
    """Pie que se añade al mensaje de cada señal para poder responder con
    /fill o /skip (el ID solo se conoce después de guardar la señal)."""
    sid = int(signal_id)
    return (f"\n\n🆔 <b>Señal #{sid}</b> — si la ejecutas: <code>/fill {sid} PRECIO</code>; "
            f"si no: <code>/skip {sid} motivo</code>")
