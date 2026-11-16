"""
Comandos de Telegram (Fase 1, punto 1.10): el usuario cuenta al sistema qué hizo
con cada señal, para comparar más adelante el R EJECUTADO con el R teórico.

    /fill <id> <precio>   la ejecuté a ese precio  (p. ej. /fill 123 65432.5)
    /skip <id> <motivo>   no la ejecuté            (p. ej. /skip 123 llegué tarde)
    /help                 ayuda

Cómo funciona: no hay servidor web. Al INICIO de cada ciclo y, además, cada
TELEGRAM_POLL_MINUTES (2) minutos en un trabajo ligero del scheduler (Fase 4), el
worker pregunta a Telegram por mensajes nuevos (getUpdates con timeout 0, no
espera; límite de tiempo TOTAL de GETUPDATES_DEADLINE_S) a partir del último
offset guardado en la tabla kv, procesa los comandos y guarda el offset nuevo.
Los trabajos del scheduler van en serie en el mismo hilo: dos sondeos nunca se
solapan y cada uno lee el offset que guardó el anterior. Solo se aceptan mensajes del chat configurado (TELEGRAM_CHAT_ID); los
demás se ignoran sin responder. Cada fill/skip va a la tabla fills con el
update_id de Telegram (UNIQUE): si un mensaje se procesa dos veces (p. ej. el
worker se reinicia antes de guardar el offset) no se duplica ni se responde dos
veces.

Nada de esto puede bloquear ni tumbar el ciclo: poll_commands nunca lanza.
"""
from __future__ import annotations

import html
import re
import threading

import requests
from loguru import logger

import notifications.telegram as tg
from config.settings import fmt_price

KV_OFFSET = "telegram_update_offset"
# Límite de tiempo TOTAL de una consulta getUpdates (Fase 4: ahora se consulta
# cada 2 min; `requests` solo limita cada operación de red, no el total).
GETUPDATES_DEADLINE_S = 10.0
# Hilo de la última consulta: si sigue vivo (Telegram colgado), no se lanza otra
# (como mucho un hilo colgado a la vez; sin crecimiento de memoria).
_inflight: dict = {"thread": None}
# Intentos por comando antes de darlo por perdido y avisar (el contador vive en
# memoria del proceso: si la BD está caída tampoco se podría guardar en ella).
MAX_UPDATE_ATTEMPTS = 3
_update_failures: dict = {}

def _when_read_text() -> str:
    """Cuándo se leen los comandos (Fase 4: cada TELEGRAM_POLL_MINUTES minutos)."""
    try:
        from config.settings import TELEGRAM_POLL_MINUTES
        n = int(TELEGRAM_POLL_MINUTES or 0)
    except Exception:  # noqa: BLE001
        n = 0
    if n > 0:
        return (f"Los comandos se leen cada {n} minuto{'s' if n != 1 else ''} más o menos, así que "
                f"la respuesta suele tardar como mucho {n} minuto{'s' if n != 1 else ''} (algo más "
                "si en ese momento el sistema está revisando los mercados).")
    return ("Los comandos se leen al principio de cada ciclo (cada hora), así que la respuesta "
            "puede tardar hasta una hora.")


HELP_TEXT = (
    "🤖 <b>Comandos</b>\n\n"
    "<code>/fill ID PRECIO</code> — ejecutaste la señal #ID a ese precio.\n"
    "   Ej.: <code>/fill 123 65432.5</code>\n"
    "   Si te equivocas, envía otro <code>/fill</code>: cuenta el más reciente.\n"
    "<code>/skip ID MOTIVO</code> — no la ejecutaste (y por qué).\n"
    "   Ej.: <code>/skip 123 llegué tarde</code>\n"
    "   Además saca la señal de la <b>cartera del sistema</b> (las posiciones abiertas que "
    "cuenta el control de cartera: máx. 3 abiertas, no LONG y SHORT a la vez, riesgo BTC+ETH). "
    "Si luego la ejecutas, un <code>/fill</code> la vuelve a meter.\n"
    "<code>/help</code> — esta ayuda.\n\n"
    "<i>El número de la señal (ID) aparece en el mensaje de la señal. "
    + _when_read_text() + "</i>"
)


# ---------------------------------------------------------------------------
# Utilidades puras (probadas sin red)
# ---------------------------------------------------------------------------


def _short(t, n: int = 40) -> str:
    """Recorta texto del usuario que se repite en una respuesta (límite de 4096
    caracteres de Telegram: un mensaje más largo no se entrega)."""
    t = str(t)
    return t if len(t) <= n else t[:n] + "…"

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


# Palabras que, al final de un /fill, confirman un precio fuera de SL/TP.
CONFIRM_WORDS = ("confirmar", "confirmo", "ok", "si", "sí")
# Distancia máxima razonable entre tu precio y la entrada de la señal.
MAX_FILL_DEVIATION = 0.20
# Oro: entrada/SL/TP salen del futuro GC=F; XM opera oro al contado (revisión de
# la Fase 1: GC=F cotizó +0,3–0,5 % por encima del contado en 11 de 12 días
# medidos, ≈0,1R con el stop típico, del mismo tamaño que el umbral de la regla 6).
BASIS_NOTE_GOLD = ("<i>Oro: los precios de la señal son del futuro GC=F y XM cotiza el contado, "
                   "que suele estar ~0,3–0,5 % por debajo (≈0,1R). Parte de esta diferencia es esa "
                   "base, no coste de ejecución.</i>")


def plausible_price(raw: str, price: float, entry: float) -> tuple[float | None, bool]:
    """Comprueba que el precio leído esté a menos de MAX_FILL_DEVIATION (20 %)
    de la entrada. Si no lo está y el texto es ambiguo (un solo '.' o ',' seguido
    de exactamente 3 cifras: '84.837', '3,125'), prueba la otra lectura (miles o
    decimales). Devuelve (precio, releído) o (None, False) si ninguna cuadra."""
    def ok(p):
        return p is not None and entry > 0 and abs(p / entry - 1.0) <= MAX_FILL_DEVIATION
    if ok(price):
        return price, False
    s = str(raw).strip().replace("$", "").replace(" ", "").replace("\u00a0", "")
    m = re.fullmatch(r"(\d{1,3})([.,])(\d{3})", s)
    if m:
        for alt in (float(m.group(1) + m.group(3)), float(f"{m.group(1)}.{m.group(3)}")):
            if alt != price and ok(alt):
                return alt, True
    return None, False


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
        # Truncado: un "comando" de miles de caracteres hacía que la respuesta
        # superara el límite de 4.096 de Telegram y no llegara.
        nombre = cmd[:50] + ("…" if len(cmd) > 50 else "")
        return f"No conozco el comando /{html.escape(nombre)}.\n\n{HELP_TEXT}"

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
                f"Motivo: {html.escape(note) if note else '(sin motivo)'}\n"
                "<i>Sale de la cartera del sistema (ya no cuenta como posición abierta "
                "para el control de cartera).</i>" + nota_prev)

    # /fill
    if len(args) < 2:
        return usage
    raw_price = args[1]
    price = parse_price(raw_price)
    if price is None:
        return f"«{html.escape(_short(raw_price))}» no es un precio válido.\n{usage}"
    entry, sl, tp = float(sig["entry_price"]), float(sig["stop_loss"]), float(sig["take_profit"])
    rest = [a for a in args[2:]]
    confirmed = bool(rest) and rest[-1].lower() in CONFIRM_WORDS
    if confirmed:
        rest = rest[:-1]
    note = " ".join(rest).strip()[:500] or None

    # Revisión de la Fase 1: comprobar el precio ANTES de guardarlo. '84.837'
    # para un BTC de 84.837 USD se leía 84,837 y una sola lectura así arruinaba
    # la media de la regla 6 (umbral −0,10R) con −48R.
    price, reread = plausible_price(raw_price, price, entry)
    if price is None:
        return (f"⚠️ <b>No lo he guardado.</b> Leí «{html.escape(_short(raw_price))}» como "
                f"{_fmt(parse_price(raw_price))}, muy lejos de la entrada de la señal {head} "
                f"({_fmt(entry)}). Revisa el número y repite <code>/fill {sig_id} PRECIO</code> "
                "(p. ej. 65432.5 o 65,432.5).")
    rr = effective_rr(direction, price, sl, tp)
    if rr is None and not confirmed:
        return (f"⚠️ <b>No lo he guardado.</b> Tu precio {_fmt(price)} está en o más allá del "
                f"Stop Loss ({_fmt(sl)}) o del Take Profit ({_fmt(tp)}) de la señal {head}: con él "
                "la operación no tiene recorrido. Si es un error, repite el /fill con el precio "
                f"bueno; si de verdad entraste a ese precio, envía "
                f"<code>/fill {sig_id} {html.escape(_short(raw_price))} confirmar</code>.")
    if not save_fill(sig_id, "fill", price, note, ts=ts, update_id=update_id):
        return None

    rr_sig = effective_rr(direction, entry, sl, tp)
    diff = entry_diff_r(direction, price, entry, sl)
    lines = [
        f"✅ <b>Ejecución registrada — señal {head}</b>",
        "",
        f"Tu precio: {_fmt(price)} (entrada de la señal: {_fmt(entry)})",
        f"Stop Loss: {_fmt(sl)} · Take Profit: {_fmt(tp)}",
    ]
    if reread:
        lines.append(f"<i>Leí «{html.escape(_short(raw_price))}» como {_fmt(price)} (la otra lectura quedaba "
                     "absurda frente a la entrada).</i>")
    if rr is None:
        lines.append("⚠️ <b>Ese precio está en o más allá del Stop Loss o del Take Profit</b>: "
                     "con él la operación no tiene recorrido (guardado porque lo confirmaste).")
    else:
        rr_sig_txt = f" (el de la señal: {rr_sig:.2f})" if rr_sig is not None else ""
        lines.append(f"<b>R:R efectivo a tu precio: {rr:.2f}</b>{rr_sig_txt}")
    if diff is not None:
        if abs(diff) < 0.005:
            lines.append("Entraste prácticamente al precio de la señal.")
        else:
            lado = "a tu favor" if diff > 0 else "en contra"
            lines.append(f"Diferencia de entrada: {diff:+.2f}R ({lado}).")
    if symbol.upper().startswith("XAU"):
        lines.append(BASIS_NOTE_GOLD)
    if sig.get("resultado_real"):
        from monitoring.daily_summary import RESULT_TEXT
        res = str(sig["resultado_real"])
        lines.append(f"<i>Nota: el sistema ya dio esta señal por cerrada "
                     f"({html.escape(RESULT_TEXT.get(res, res))}).</i>")
    if str(sig.get("mode") or "live") == "shadow":
        lines.append("<i>Nota: esta señal es de un activo en modo sombra (no se notificó).</i>")
    lines.append("<i>¿Te equivocaste? Envía otro /fill de esta señal: cuenta el más reciente.</i>")
    return "\n".join(lines) + nota_prev


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

def _get_with_deadline(url: str, params: dict, deadline_s: float):
    """requests.get con límite de tiempo TOTAL (Fase 4).

    La petición va en un hilo daemon; si no termina en `deadline_s` se lanza
    TimeoutError y su resultado se descarta (getUpdates no cambia nada en
    Telegram: el offset solo avanza cuando se guarda en la BD). Si la consulta
    anterior sigue colgada no se lanza otra."""
    prev = _inflight.get("thread")
    if prev is not None and prev.is_alive():
        raise TimeoutError("la consulta anterior a Telegram sigue sin terminar")
    box: dict = {}

    def _run():
        try:
            box["resp"] = requests.get(url, params=params, timeout=10)
        except BaseException as exc:  # noqa: BLE001 — se relanza en el hilo principal
            box["exc"] = exc

    th = threading.Thread(target=_run, name="tg-getUpdates", daemon=True)
    _inflight["thread"] = th
    th.start()
    th.join(deadline_s)
    if th.is_alive():
        raise TimeoutError(f"getUpdates sin respuesta completa en {deadline_s:g} s")
    if "exc" in box:
        raise box["exc"]
    return box["resp"]


def _get_updates(offset: int | None, log_failures: bool = True) -> list[dict] | None:
    """getUpdates sin espera (timeout 0). None si falla.

    log_failures=False (sondeo de cada 2 min): los fallos de red van al log en
    DEBUG para no repetir el mismo aviso 30 veces por hora; el sondeo del inicio
    de cada ciclo los sigue registrando como WARNING."""
    warn = logger.warning if log_failures else logger.debug
    url = tg._BASE_URL.format(token=tg.TELEGRAM_TOKEN, method="getUpdates")
    params = {"timeout": 0, "allowed_updates": '["message","channel_post"]'}
    if offset is not None:
        params["offset"] = int(offset)
    try:
        resp = _get_with_deadline(url, params, GETUPDATES_DEADLINE_S)
    except (requests.RequestException, TimeoutError) as exc:
        warn(f"Comandos Telegram: getUpdates falló ({exc}).")
        return None
    if resp.status_code != 200:
        # 409 = hay un webhook configurado o se está usando el bot en otro sitio.
        warn(f"Comandos Telegram: getUpdates respondió {resp.status_code}: {resp.text[:200]}")
        if resp.status_code in (401, 404):
            # Token revocado o erróneo: tampoco saldrán las señales → sin latido.
            tg._mark_cycle("telegram", f"getUpdates {resp.status_code} (token no válido)")
        return None
    data = resp.json()
    if not data.get("ok"):
        warn(f"Comandos Telegram: respuesta no OK: {str(data)[:200]}")
        return None
    return data.get("result") or []


def poll_commands(log_failures: bool = True) -> dict:
    """
    Lee los mensajes nuevos de Telegram, procesa /fill /skip /help del chat
    configurado, responde y guarda el offset. Nunca lanza.
    Devuelve {"updates": n, "commands": n, "ignored": n, "errors": n}.
    log_failures=False: fallos de red de getUpdates solo en DEBUG (sondeo de
    cada 2 min del scheduler, Fase 4).
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
            tg._mark_cycle("bd", f"offset de comandos: {exc}")
            return out

        updates = _get_updates(offset, log_failures=log_failures)
        if not updates:
            return out
        out["updates"] = len(updates)
        new_offset = offset
        for upd in updates:
            uid = upd.get("update_id")
            advance = True            # False = repetir esta actualización el ciclo siguiente
            try:
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
                try:
                    reply = handle_command(cmd, args, update_id=uid, ts=msg.get("date"))
                except Exception as exc:
                    # Revisión de la Fase 1: antes el offset avanzaba igual y el
                    # /fill se perdía sin respuesta. Ahora NO se avanza más allá de
                    # esta actualización: se repite en el ciclo siguiente (update_id
                    # UNIQUE evita duplicar) hasta MAX_UPDATE_ATTEMPTS; después se
                    # responde que no se pudo registrar y se sigue.
                    out["errors"] += 1
                    n = _update_failures.get(uid, 0) + 1
                    _update_failures[uid] = n
                    logger.error(f"Comandos Telegram: error procesando /{cmd} (actualización {uid}, "
                                 f"intento {n}/{MAX_UPDATE_ATTEMPTS}): {exc}")
                    if uid is not None and n < MAX_UPDATE_ATTEMPTS:
                        advance = False
                        break          # el offset se queda en esta actualización
                    _update_failures.pop(uid, None)
                    texto = " ".join(["/" + cmd] + args)[:80]
                    tg._send(f"⚠️ No pude registrar «{html.escape(texto)}» (error de la base de datos "
                             f"tras {n} intentos). Reenvíalo más tarde.")
                    reply = None
                else:
                    _update_failures.pop(uid, None)
                    out["commands"] += 1
                    logger.info(f"Comandos Telegram: /{cmd} {' '.join(args)[:80]} procesado.")
                if reply:
                    tg._send(reply)
            except Exception as exc:
                out["errors"] += 1
                logger.error(f"Comandos Telegram: error procesando la actualización {uid}: {exc}")
            finally:
                if advance and uid is not None:
                    new_offset = max(new_offset or 0, int(uid) + 1)
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
