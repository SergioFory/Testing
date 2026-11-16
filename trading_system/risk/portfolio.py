"""
Control de cartera (Fase 3, punto 3.1; hallazgos F11 y F19 de la auditoría).

Hasta ahora cada señal se decidía sola: el sistema podía mandar a la vez cinco
operaciones, o un LONG y un SHORT del mismo activo, o BTC y ETH en la misma
dirección (se mueven juntos: riesgo duplicado). Este módulo reconstruye desde la
BD la CARTERA SEGÚN EL SISTEMA y comprueba cuatro reglas antes de cada señal:

  R1  máximo de posiciones abiertas (3);
  R2  nunca LONG y SHORT a la vez en el mismo activo;
  R3  grupo correlacionado {BTCUSDT, ETHUSDT}: riesgo abierto del grupo + el de
      la señal nueva ≤ 0,50 % del capital;
  R4  si la pérdida realizada del día (UTC) llega al 3 % del capital, no abrir.

Modo (settings.PORTFOLIO_CONTROL_MODE):
  "shadow"  (el activo HOY): la señal se emite igual; se guarda qué regla la
            habría bloqueado (columna portfolio_flag de signals y shadow_setups) y
            el mensaje lo dice ("control de cartera en pruebas: no bloquea").
  "enforce" (probado con tests, NO activado): la señal no se emite y el setup
            queda en shadow_setups con motivo 'portfolio_R1'…'portfolio_R4'.
            Solo afecta a activos que notifican (los de modo sombra, hoy SPX, se
            registran igual y nunca cuentan como posición abierta).

"Posición abierta según el sistema" = señal mode='live', sin quality_flag, con
setup_ts, sin resultado (resultado_real IS NULL) y cuyo ÚLTIMO registro en fills
no es un /skip (si el usuario dijo /skip, la señal sale de la cartera; un /fill
posterior la vuelve a meter). Riesgo de cada una en % del capital =
position_size × |entrada − SL| / entrada / TRADING_CAPITAL.

Las funciones de reglas son PURAS (sin BD) para poder usarlas igual en el vivo,
en las pruebas y en la medición histórica del arnés. Nada de aquí puede tumbar el
ciclo: quien llama envuelve en try/except y, si algo falla, la señal sigue su
camino normal sin marca de cartera.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger
from sqlalchemy import text

# Tolerancia al comparar riesgos (fracción del capital). position_size viene
# redondeado a céntimos: dos posiciones del 0,25 % pueden sumar 0,5000001 %.
RISK_EPS = 1e-6

RULE_IDS = ("R1", "R2", "R3", "R4")
MODES = ("shadow", "enforce")


def _settings():
    from config import settings
    return settings


def control_mode() -> str:
    """'shadow' o 'enforce' (cualquier otro valor se trata como 'shadow')."""
    mode = str(getattr(_settings(), "PORTFOLIO_CONTROL_MODE", "shadow") or "shadow").lower()
    return mode if mode in MODES else "shadow"


def rules_config() -> dict:
    return dict(getattr(_settings(), "PORTFOLIO_RULES", {}) or {})


def short_name(symbol: str) -> str:
    """'BTCUSDT' → 'BTC', 'XAUUSD' → 'XAU' (para los textos al usuario); 'SPX500' igual."""
    s = str(symbol or "")
    if s.endswith("USDT"):
        return s[:-4]
    if len(s) == 6 and s.endswith("USD"):
        return s[:-3]
    return s


def _pct(x: float) -> str:
    """0.0075 → '0,75 %' (texto en español)."""
    return f"{x * 100:.2f}".replace(".", ",") + " %"


def risk_pct_of(position_size, entry, stop_loss, capital) -> float | None:
    """Riesgo en fracción del capital = position_size × |entrada − SL| / entrada / capital."""
    try:
        ps, e, sl, cap = float(position_size), float(entry), float(stop_loss), float(capital)
    except (TypeError, ValueError):
        return None
    if any(pd.isna(v) for v in (ps, e, sl, cap)) or e <= 0 or cap <= 0:
        return None
    return ps * abs(e - sl) / e / cap


def trade_info(trade, capital: float | None = None) -> dict:
    """Datos de la señal nueva que necesitan las reglas (TradeSetup o dict)."""
    if capital is None:
        capital = _settings().TRADING_CAPITAL
    get = (lambda k: trade.get(k)) if isinstance(trade, dict) else (lambda k: getattr(trade, k, None))
    risk = get("risk_pct")
    if risk is None:
        risk = risk_pct_of(get("position_size"), get("entry_price"), get("stop_loss"), capital)
    return {"symbol": str(get("symbol")), "direction": str(get("direction")),
            "risk_pct": float(risk or 0.0)}


# ---------------------------------------------------------------------------
# Reglas (puras)
# ---------------------------------------------------------------------------

def evaluate_portfolio_rules(new_trade, open_positions: list[dict], day_loss_pct: float = 0.0,
                             rules: dict | None = None, capital: float | None = None) -> list[dict]:
    """
    Reglas de cartera que INCUMPLIRÍA la señal nueva, en orden R1…R4.

    Args:
        new_trade:      TradeSetup o dict con symbol, direction y risk_pct (o
                        position_size, entry_price y stop_loss para calcularlo).
        open_positions: lista de dicts {id, symbol, direction, risk_pct} (lo que
                        devuelve load_open_positions).
        day_loss_pct:   pérdida realizada del día UTC en fracción del capital
                        (positiva = pérdida; ver realized_loss_today).
        rules:          por defecto settings.PORTFOLIO_RULES.

    Returns:
        [{"rule": "R1", "text": "…"}, …]; lista vacía si no incumple ninguna.
    """
    rules = rules_config() if rules is None else rules
    new = trade_info(new_trade, capital)
    out: list[dict] = []
    opens = list(open_positions or [])

    # R1 — máximo de posiciones abiertas.
    max_open = rules.get("max_open_positions")
    if max_open is not None and len(opens) >= int(max_open):
        out.append({"rule": "R1", "text": (
            f"serían {len(opens) + 1} posiciones abiertas a la vez (máximo {int(max_open)})")})

    # R2 — nunca LONG y SHORT a la vez en el mismo activo.
    if rules.get("no_opposite_same_symbol", True):
        opp = [p for p in opens if p.get("symbol") == new["symbol"]
               and str(p.get("direction")) != new["direction"]]
        if opp:
            ids = ", ".join(f"#{p['id']}" for p in opp if p.get("id") is not None)
            out.append({"rule": "R2", "text": (
                f"ya hay una posición {str(opp[0].get('direction')).upper()} abierta en "
                f"{short_name(new['symbol'])}{' (' + ids + ')' if ids else ''} y esta señal es "
                f"{new['direction'].upper()} (nunca las dos a la vez)")})

    # R3 — riesgo abierto del grupo correlacionado (BTC + ETH).
    for grp in rules.get("correlated_groups") or ():
        syms = set(grp.get("symbols") or ())
        cap = grp.get("max_risk_pct")
        if new["symbol"] not in syms or cap is None:
            continue
        open_risk = sum(float(p.get("risk_pct") or 0.0) for p in opens if p.get("symbol") in syms)
        total = open_risk + new["risk_pct"]
        if total > float(cap) + RISK_EPS:
            name = grp.get("name") or "+".join(short_name(s) for s in sorted(syms))
            out.append({"rule": "R3", "text": (
                f"el riesgo abierto en {name} pasaría a {_pct(total)} del capital "
                f"(máximo {_pct(float(cap))})")})

    # R4 — pérdida realizada del día.
    max_loss = rules.get("max_daily_realized_loss_pct")
    if max_loss is not None and float(day_loss_pct or 0.0) >= float(max_loss) - RISK_EPS:
        out.append({"rule": "R4", "text": (
            f"hoy ya se ha perdido un {_pct(float(day_loss_pct))} del capital en operaciones "
            f"cerradas (límite {_pct(float(max_loss))}): no abrir más hoy")})
    return out


def flag_of(broken: list[dict]) -> str:
    """Valor de la columna portfolio_flag: 'R1,R3' o 'ok' si no incumple ninguna."""
    return ",".join(b["rule"] for b in broken) if broken else "ok"


# ---------------------------------------------------------------------------
# Cartera desde la BD
# ---------------------------------------------------------------------------

_OPEN_SQL = (
    "SELECT s.id, s.symbol, s.direction, s.entry_price, s.stop_loss, s.position_size, "
    "s.setup_ts, s.ts FROM signals s "
    "WHERE s.mode = 'live' AND s.quality_flag IS NULL AND s.setup_ts IS NOT NULL "
    "AND s.resultado_real IS NULL AND s.ts <= :now "
    # Último registro de fills de la señal (el de mayor ts y, a igualdad, mayor id).
    "AND COALESCE((SELECT f.kind FROM fills f WHERE f.signal_id = s.id "
    "              ORDER BY f.ts DESC, f.id DESC LIMIT 1), '') <> 'skip' "
    "ORDER BY s.ts ASC, s.id ASC"
)


def _unix(now) -> int:
    t = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(t.timestamp())


def load_open_positions(now=None, capital: float | None = None) -> list[dict]:
    """
    Posiciones abiertas según el sistema (ver docstring del módulo), de la más
    antigua a la más reciente: dicts {id, symbol, direction, entry_price,
    stop_loss, position_size, setup_ts, risk_pct}. `now` deja fuera las señales
    guardadas después (pruebas y mediciones). Lanza si la BD falla (quien llama
    lo captura).
    """
    from data.database import get_engine, init_db
    if capital is None:
        capital = _settings().TRADING_CAPITAL
    init_db()
    with get_engine().connect() as conn:
        rows = conn.execute(text(_OPEN_SQL), {"now": _unix(now)}).mappings().fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["risk_pct"] = risk_pct_of(d.get("position_size"), d.get("entry_price"),
                                    d.get("stop_loss"), capital) or 0.0
        out.append(d)
    return out


def realized_loss_today(now=None, capital: float | None = None) -> float:
    """
    Pérdida realizada del día UTC de `now`, en fracción del capital (positiva =
    pérdida; negativa = ganancia neta): −Σ pnl_r × riesgo de las señales live sin
    quality_flag, con setup_ts, resueltas hoy (resolved_ts entre las 00:00 UTC y
    `now`) y que el usuario no saltó (/skip como último registro). pnl_r: el
    guardado o, si falta, el que se deduce del resultado (outcome_tracker.signal_r).
    """
    from data.database import get_engine, init_db
    from data.outcome_tracker import signal_r
    if capital is None:
        capital = _settings().TRADING_CAPITAL
    now_u = _unix(now)
    day0 = int(pd.Timestamp(now_u, unit="s", tz="UTC").normalize().timestamp())
    init_db()
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT s.id, s.entry_price, s.stop_loss, s.take_profit, s.position_size, s.pnl_r, "
            "s.pnl_pct, s.resultado_real, s.resolved_by FROM signals s "
            "WHERE s.mode = 'live' AND s.quality_flag IS NULL AND s.setup_ts IS NOT NULL "
            "AND s.resultado_real IS NOT NULL AND s.resolved_ts >= :a AND s.resolved_ts <= :b "
            "AND COALESCE((SELECT f.kind FROM fills f WHERE f.signal_id = s.id "
            "              ORDER BY f.ts DESC, f.id DESC LIMIT 1), '') <> 'skip'"
        ), {"a": day0, "b": now_u}).mappings().fetchall()
    total = 0.0
    for r in rows:
        r = dict(r)
        rr = signal_r(r)
        risk = risk_pct_of(r.get("position_size"), r.get("entry_price"), r.get("stop_loss"), capital)
        if rr is None or risk is None:
            continue
        total += float(rr) * risk
    return -total


@dataclass
class PortfolioCheck:
    """Resultado de comprobar una señal nueva contra la cartera del sistema."""
    open_positions: list = field(default_factory=list)
    broken: list = field(default_factory=list)      # [{"rule", "text"}]
    day_loss_pct: float = 0.0
    mode: str = "shadow"

    @property
    def flag(self) -> str:
        return flag_of(self.broken)

    @property
    def blocks(self) -> bool:
        """True si en modo enforce esta señal NO se emitiría."""
        return self.mode == "enforce" and bool(self.broken)

    @property
    def motivo(self) -> str | None:
        """Motivo de shadow_setups si se bloquea (la primera regla incumplida)."""
        return f"portfolio_{self.broken[0]['rule']}" if self.broken else None


def check_new_signal(trade, now=None, capital: float | None = None) -> PortfolioCheck:
    """Carga la cartera y evalúa las reglas para la señal nueva. Lanza si la BD
    falla: el llamador (main._emit_best_setup) lo captura y sigue sin marca."""
    opens = load_open_positions(now=now, capital=capital)
    day_loss = realized_loss_today(now=now, capital=capital)
    broken = evaluate_portfolio_rules(trade, opens, day_loss_pct=day_loss, capital=capital)
    return PortfolioCheck(open_positions=opens, broken=broken, day_loss_pct=day_loss,
                          mode=control_mode())


# ---------------------------------------------------------------------------
# Textos al usuario
# ---------------------------------------------------------------------------

MAX_LISTED = 5


def positions_text(open_positions: list[dict]) -> str:
    """'2 abiertas (BTC long #12, ETH short #15)' o 'ninguna abierta'."""
    n = len(open_positions or [])
    if n == 0:
        return "ninguna posición abierta"
    items = [f"{short_name(p.get('symbol'))} {p.get('direction')} #{p.get('id')}"
             for p in open_positions[:MAX_LISTED]]
    more = f" y {n - MAX_LISTED} más" if n > MAX_LISTED else ""
    return f"{n} abierta{'s' if n != 1 else ''} (" + ", ".join(items) + more + ")"


def message_lines(check: PortfolioCheck | None) -> list[tuple[str, str]]:
    """
    Líneas del mensaje de la señal como pares (etiqueta, texto) SIN escapar (el
    mensaje las escapa):
      ("Cartera según el sistema", "2 abiertas (BTC long #12, ETH short #15)")
      ("⚠️ Con esta señal", "<regla>; <regla> (control de cartera en pruebas: no bloquea)")
    """
    if check is None:
        return []
    out = [("Cartera según el sistema", positions_text(check.open_positions))]
    if check.broken:
        txt = "; ".join(b["text"] for b in check.broken)
        if check.mode == "shadow":
            txt += " (control de cartera en pruebas: no bloquea)"
        out.append(("⚠️ Con esta señal", txt))
    return out


RULE_TEXT = {
    "R1": "máx. 3 posiciones abiertas",
    "R2": "LONG y SHORT a la vez en el mismo activo",
    "R3": "riesgo del grupo BTC+ETH > 0,50 %",
    "R4": "pérdida del día ≥ 3 %",
}


def blocked_setup_keys(symbol: str, since_ts) -> set:
    """
    (setup_ts, direction) de los setups del activo que el control de cartera ya
    bloqueó (motivo 'portfolio_*' en shadow_setups) con vela >= since_ts. En modo
    enforce se excluyen de get_best_setup: un setup bloqueado no se emite tarde
    en un ciclo posterior (entrar tarde cuesta; experimento E2). Nunca lanza.
    """
    try:
        from data.database import get_engine, _to_unix
        with get_engine().connect() as conn:
            rows = conn.execute(text(
                "SELECT setup_ts, direction FROM shadow_setups WHERE symbol = :s "
                "AND setup_ts >= :t AND motivo LIKE 'portfolio_%'"
            ), {"s": symbol, "t": _to_unix(since_ts)}).fetchall()
        return {(pd.Timestamp(int(r[0]), unit="s", tz="UTC"), r[1]) for r in rows}
    except Exception as exc:
        logger.warning(f"[{symbol}] No se pudieron leer los setups bloqueados por cartera: {exc}")
        return set()
