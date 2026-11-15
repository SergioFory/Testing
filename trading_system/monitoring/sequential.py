"""
Vigilancia secuencial escrita DE ANTEMANO (Fase 1, punto 1.12).

Las reglas están en trading_system/REGLAS_DE_DECISION.md. Aquí solo está la
aritmética, con los parámetros fijados ANTES de ver los resultados en vivo (no
se cambian mirando los datos: eso invalidaría las garantías de error).

1) SPRT (prueba secuencial de razón de verosimilitudes, Wald):
     H0: el R medio por señal es 0      (no hay ventaja)
     H1: el R medio por señal es +0.25R (hay ventaja)
   con R ~ Normal(μ, σ=1.3), α = 0.05 (aceptar una ventaja que no existe) y
   β = 0.20 (no ver una ventaja de +0.25R que sí existe). Por cada señal
   resuelta con resultado R se suma al acumulado:
       z = (μ1 − μ0)/σ² · (R − (μ0 + μ1)/2)
   y se decide en cuanto el acumulado cruza un límite:
       ≥ ln((1−β)/α) = +2.77  → "edge aceptado"
       ≤ ln(β/(1−α)) = −1.56  → "sin edge"
       en medio               → "seguir"
   La decisión es la del PRIMER cruce (no se deshace con señales posteriores).

2) CUSUM unilateral de deterioro (Page): detecta que el R medio ha bajado de
   lo esperado. S_0 = 0;  S_t = max(0, S_{t−1} + (k − R_t)),  k = 0.125R
   (punto medio entre +0.25R y 0). Alarma cuando S_t ≥ h = 20R. Se calcula
   SOLO sobre las señales live originales (lo que el usuario recibe).

Universo (señales live): signals con mode live, quality_flag NULL, setup_ts NOT
NULL (emitidas desde la Fase 0), resueltas, en orden de resolución. Versión con
más N: shadow_setups aprobados (approved=1), resueltos, sin los del activo en
modo sombra (motivo 'shadow_asset', hoy SPX: tiene su propia regla de vuelta).

Los R son los del tracker: SIN costes (comisión, spread, swap) y con la entrada
al precio del mensaje. Ver REGLAS_DE_DECISION.md para cómo leerlo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Parámetros PRE-REGISTRADOS (2026-11-15). No cambiar sin dejarlo escrito en
# REGLAS_DE_DECISION.md con fecha y motivo, y nunca a la vista de los resultados.
# ---------------------------------------------------------------------------
SPRT_PARAMS = {"mu0": 0.0, "mu1": 0.25, "sigma": 1.3, "alpha": 0.05, "beta": 0.20}
CUSUM_PARAMS = {"k": 0.125, "h": 20.0}
FREEZE_MAX_SIGNALS = 150     # la congelación acaba al decidir el SPRT o con 150 señales
# Congelación de la configuración (informe final): desde aquí, la prueba real.
CONFIG_FREEZE_UTC = "2026-09-18 17:21"

# Motivos de shadow_setups que NO entran en la versión "sombra" del SPRT.
SHADOW_EXCLUDE_MOTIVOS = ("shadow_asset",)

DECISION_TEXT = {
    "edge_aceptado": "edge aceptado",
    "sin_edge":      "sin edge",
    "seguir":        "seguir",
}


@dataclass
class SPRTResult:
    n: int                       # señales usadas
    llr: float                   # log-razón de verosimilitudes acumulada (todas)
    upper: float                 # límite para aceptar edge
    lower: float                 # límite para "sin edge"
    decision: str                # edge_aceptado / sin_edge / seguir
    decided_at: int | None       # nº de señal en el que se cruzó el límite
    llr_at_decision: float | None
    mean_r: float | None
    sum_r: float
    path: list = field(default_factory=list, repr=False)


@dataclass
class CUSUMResult:
    n: int
    s: float                     # valor actual del CUSUM (R)
    max_s: float                 # máximo alcanzado
    h: float
    alarm: bool                  # True si en algún momento S ≥ h
    alarm_at: int | None         # nº de señal del primer cruce
    path: list = field(default_factory=list, repr=False)


def sprt_bounds(alpha: float = None, beta: float = None) -> tuple[float, float]:
    """(límite inferior, límite superior) del log-LR de Wald."""
    a = SPRT_PARAMS["alpha"] if alpha is None else alpha
    b = SPRT_PARAMS["beta"] if beta is None else beta
    return math.log(b / (1 - a)), math.log((1 - b) / a)


def sprt_increment(r: float, mu0: float = None, mu1: float = None, sigma: float = None) -> float:
    """Aporte de una señal con resultado r al log-LR (modelo normal, σ conocida)."""
    m0 = SPRT_PARAMS["mu0"] if mu0 is None else mu0
    m1 = SPRT_PARAMS["mu1"] if mu1 is None else mu1
    s = SPRT_PARAMS["sigma"] if sigma is None else sigma
    return (m1 - m0) / (s * s) * (float(r) - (m0 + m1) / 2.0)


def sprt(rs, mu0: float = None, mu1: float = None, sigma: float = None,
         alpha: float = None, beta: float = None) -> SPRTResult:
    """SPRT sobre la secuencia de R (en orden de resolución)."""
    lower, upper = sprt_bounds(alpha, beta)
    llr, path = 0.0, []
    decision, decided_at, llr_dec = "seguir", None, None
    rs = [float(r) for r in rs]
    for i, r in enumerate(rs, start=1):
        llr += sprt_increment(r, mu0, mu1, sigma)
        path.append(llr)
        if decided_at is None:
            if llr >= upper:
                decision, decided_at, llr_dec = "edge_aceptado", i, llr
            elif llr <= lower:
                decision, decided_at, llr_dec = "sin_edge", i, llr
    n = len(rs)
    return SPRTResult(n=n, llr=llr, upper=upper, lower=lower, decision=decision,
                      decided_at=decided_at, llr_at_decision=llr_dec,
                      mean_r=(sum(rs) / n) if n else None, sum_r=sum(rs), path=path)


def cusum(rs, k: float = None, h: float = None) -> CUSUMResult:
    """CUSUM unilateral de deterioro: S_t = max(0, S_{t-1} + (k − R_t))."""
    k = CUSUM_PARAMS["k"] if k is None else k
    h = CUSUM_PARAMS["h"] if h is None else h
    s, max_s, alarm_at, path = 0.0, 0.0, None, []
    rs = [float(r) for r in rs]
    for i, r in enumerate(rs, start=1):
        s = max(0.0, s + (k - r))
        path.append(s)
        max_s = max(max_s, s)
        if alarm_at is None and s >= h:
            alarm_at = i
    return CUSUMResult(n=len(rs), s=s, max_s=max_s, h=h, alarm=alarm_at is not None,
                       alarm_at=alarm_at, path=path)


# ---------------------------------------------------------------------------
# Lectura de la BD
# ---------------------------------------------------------------------------

def _r_value(pnl_r, resultado, entry=None, stop_loss=None, take_profit=None, resolved_by=None):
    """R de una fila resuelta: pnl_r; si falta, el implícito en el resultado
    (loss = −1, win = |TP − entrada| / |entrada − SL|, be = 0; misma regla que
    outcome_tracker.signal_r). None solo si no se puede saber (expirada sin
    precio). Antes una 'loss' sin pnl_r se omitía y sesgaba el R medio al alza."""
    try:
        if pnl_r is not None and not (isinstance(pnl_r, float) and math.isnan(pnl_r)):
            return float(pnl_r)
    except (TypeError, ValueError):
        pass
    from data.outcome_tracker import r_from_outcome
    return r_from_outcome(resultado, entry, stop_loss, take_profit, resolved_by=resolved_by)


def _freeze_unix() -> int:
    import pandas as pd
    return int(pd.Timestamp(CONFIG_FREEZE_UTC, tz="UTC").timestamp())


def _live_symbols() -> list[str]:
    """Activos operativos que hoy se notifican (no apagados ni en modo sombra)."""
    from config.settings import active_assets, notify_enabled
    return [s for s in active_assets() if notify_enabled(s)]


def load_live_r(since_ts: int | None = None, phase0: bool = True) -> list[float]:
    """R de las señales live del universo de vigilancia, en orden de resolución.

    since_ts (Unix): solo señales resueltas desde esa hora (reinicio tras un
    cambio de configuración o una revisión; ver REGLAS_DE_DECISION.md).
    phase0=False: en vez del universo, las señales live ORIGINALES anteriores a
    la Fase 0 (sin setup_ts) emitidas DESDE LA CONGELACIÓN de la configuración
    (CONFIG_FREEZE_UTC) de los activos que hoy se notifican: "las 8 originales"
    del informe (§7, 1.12). Se reportan aparte y no cuentan. (Revisión de la
    Fase 1: antes entraban las 37 señales sin setup_ts de toda la historia,
    con configuraciones antiguas y activos ya apagados.)"""
    from sqlalchemy import text
    from data.database import get_engine, init_db
    init_db()
    q = ("SELECT pnl_r, resultado_real, entry_price, stop_loss, take_profit, "
         "COALESCE(resolved_ts, ts) AS rts, id, symbol, resolved_by FROM signals "
         "WHERE COALESCE(mode, 'live') = 'live' AND quality_flag IS NULL "
         f"AND setup_ts IS {'NOT ' if phase0 else ''}NULL AND resultado_real IS NOT NULL")
    params = {}
    if since_ts is not None:
        q += " AND COALESCE(resolved_ts, ts) >= :since"
        params["since"] = int(since_ts)
    if not phase0:
        q += " AND ts >= :freeze"
        params["freeze"] = _freeze_unix()
    q += " ORDER BY rts ASC, id ASC"
    with get_engine().connect() as conn:
        rows = conn.execute(text(q), params).fetchall()
    keep = None if phase0 else set(_live_symbols())
    out = []
    for pnl_r, resultado, entry, sl, tp, _rts, _id, symbol, resolved_by in rows:
        if keep is not None and symbol not in keep:
            continue
        r = _r_value(pnl_r, resultado, entry, sl, tp, resolved_by)
        if r is not None:
            out.append(r)
    return out


def load_shadow_approved_r(exclude_motivos=SHADOW_EXCLUDE_MOTIVOS,
                           since_ts: int | None = None) -> list[float]:
    """R de los setups APROBADOS del registro sombra (emitidos + bloqueados por
    frescura, eventos, riesgo, duplicado o por no ser el mejor), resueltos, en
    orden de resolución. Sin el activo en modo sombra por defecto."""
    from sqlalchemy import text
    from data.database import get_engine, init_db
    init_db()
    q = ("SELECT pnl_r, resultado, motivo, entry, sl, tp FROM shadow_setups "
         "WHERE approved = 1 AND resultado IS NOT NULL")
    params = {}
    if since_ts is not None:
        q += " AND COALESCE(resolved_ts, setup_ts) >= :since"
        params["since"] = int(since_ts)
    q += " ORDER BY COALESCE(resolved_ts, setup_ts) ASC, setup_ts ASC, id ASC"
    with get_engine().connect() as conn:
        rows = conn.execute(text(q), params).fetchall()
    excl = set(exclude_motivos or ())
    out = []
    for pnl_r, resultado, motivo, entry, sl, tp in rows:
        if motivo in excl:
            continue
        r = _r_value(pnl_r, resultado, entry, sl, tp)
        if r is not None:
            out.append(r)
    return out


# Reinicios (tabla kv, segundos Unix), escritos a mano y anotados en las reglas:
KV_SINCE = "vigilancia_desde"    # tras un cambio de configuración: SPRT y CUSUM desde ahí
KV_CUSUM_SINCE = "cusum_desde"   # tras la revisión de una alarma: solo el CUSUM desde ahí


def _kv_unix(name: str) -> int | None:
    try:
        from data.database import kv_get
        raw = kv_get(name)
        return int(float(raw)) if raw not in (None, "") else None
    except Exception:
        return None


def monitoring_state() -> dict:
    """Estado completo de la vigilancia: SPRT live, SPRT sombra, CUSUM live."""
    since = _kv_unix(KV_SINCE)
    cus_since = _kv_unix(KV_CUSUM_SINCE)
    if since is not None and cus_since is not None:
        cus_since = max(since, cus_since)
    elif cus_since is None:
        cus_since = since
    live = load_live_r(since_ts=since)
    shadow = load_shadow_approved_r(since_ts=since)
    cus_rs = load_live_r(since_ts=cus_since) if cus_since != since else live
    pre = load_live_r(phase0=False)
    return {
        "sprt_live":   sprt(live),
        "sprt_shadow": sprt(shadow),
        "cusum_live":  cusum(cus_rs),
        "since":       since,
        "cusum_since": cus_since,
        "n_live":      len(live),
        "freeze_max":  FREEZE_MAX_SIGNALS,
        "pre_phase0":  {"n": len(pre), "mean_r": (sum(pre) / len(pre)) if pre else None,
                        "sum_r": sum(pre)},
    }


# ---------------------------------------------------------------------------
# Texto para el usuario (HTML de Telegram; sin datos dinámicos de texto libre)
# ---------------------------------------------------------------------------

def _fmt_unix(ts: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(int(ts), _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_sprt(res: SPRTResult) -> str:
    if res.n == 0:
        return "aún sin señales resueltas"
    dec = DECISION_TEXT[res.decision]
    base = (f"{res.n} señales, R medio {res.mean_r:+.2f}R; "
            f"indicador {res.llr:+.2f} (decide en {res.lower:+.2f} / {res.upper:+.2f})")
    if res.decided_at is not None:
        return f"<b>{dec.upper()}</b> (decidido en la señal nº {res.decided_at}). {base}"
    return f"<b>{dec}</b>. {base}"


def format_monitoring(state: dict) -> str:
    """Bloque de texto (HTML) con el estado de la vigilancia, en español claro."""
    s_live, s_sh, cu = state["sprt_live"], state["sprt_shadow"], state["cusum_live"]
    lines = ["🧭 <b>Vigilancia (reglas escritas de antemano)</b>"]
    lines.append(f"• SPRT señales live: {_fmt_sprt(s_live)}")
    lines.append(f"• SPRT aprobados del registro sombra (más N, solo orientativo): {_fmt_sprt(s_sh)}")
    if cu.n == 0:
        lines.append("• CUSUM de deterioro: aún sin señales resueltas")
    elif cu.alarm:
        lines.append(f"• CUSUM de deterioro: ⚠️ <b>ALARMA</b> (saltó en la señal nº {cu.alarm_at}; "
                     f"ahora {cu.s:.1f}R de {cu.h:.0f}R) → riesgo a la mitad y revisión")
    else:
        lines.append(f"• CUSUM de deterioro: sin alarma ({cu.s:.1f}R de {cu.h:.0f}R)")
    n, fmax = state["n_live"], state["freeze_max"]
    if s_live.decision == "seguir" and n < fmax:
        lines.append(f"• Configuración congelada: {n}/{fmax} señales y el SPRT aún no ha decidido.")
    elif s_live.decision == "edge_aceptado":
        lines.append("• El SPRT aceptó el edge: se puede plantear subir el riesgo por encima del 0,25 %.")
    elif s_live.decision == "sin_edge":
        lines.append("• El SPRT dice SIN EDGE: no subir riesgo; revisar con calma (ver reglas).")
    else:
        lines.append(f"• {fmax} señales alcanzadas sin decisión del SPRT: fin de la congelación, revisar.")
    pre = state.get("pre_phase0") or {}
    if pre.get("n"):
        lines.append(f"• Aparte (originales desde la congelación de la configuración, "
                     f"{CONFIG_FREEZE_UTC[:10]}, anteriores a la Fase 0; no cuentan): "
                     f"{pre['n']} señal{'es' if pre['n'] != 1 else ''}, R medio {pre['mean_r']:+.2f}R")
    since, cus_since = state.get("since"), state.get("cusum_since")
    if since is not None:
        lines.append(f"• SPRT y CUSUM cuentan desde {_fmt_unix(since)}")
    if cus_since is not None and cus_since != since:
        lines.append(f"• CUSUM reiniciado el {_fmt_unix(cus_since)}")
    lines.append("<i>R sin costes del bróker. Ver trading_system/REGLAS_DE_DECISION.md.</i>")
    return "\n".join(lines)
