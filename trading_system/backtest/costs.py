"""
Modelo de costes por bróker (Fase 2, punto 2.1) para `validate --protocol`.

Convierte operaciones simuladas a PRECIOS MEDIOS (sin costes) en R netos bajo
varios escenarios de costes:

  - "xm_swap0", "xm_swap5", "xm_swap10": cuenta XM Standard del usuario con el
    spread de settings.BROKER_COSTS y un swap DESCONOCIDO del 0 %, 5 % o 10 %
    anual sobre el nominal (settings.SWAP_SCENARIOS_PCT_YEAR). El E[R] de
    decisión es "xm_swap5" (settings.COST_DECISION_SCENARIO).
  - "xm_real": solo si algún día se rellena swap_pct_year en BROKER_COSTS.
  - "binance": el escenario histórico del validador (comisión 0,04 % + slippage
    0,02 % por lado; settings.BACKTEST). Da EXACTAMENTE el R de
    backtest/validate._simulate_setup; se mantiene como referencia.

Convenciones (documentadas, no cambian el vivo):
  - Spread: 1 spread por operación, medio a la entrada y medio a la salida. El
    histórico son precios medios, así que equivale a restar 1 spread al PnL. No
    cambia qué vela toca el SL o el TP (igual que el slippage del validador).
    El spread se aplica en UNIDADES DE PRECIO fijas a toda la historia: en los
    años en que el precio era más bajo pesa más en R (conservador).
  - Comisión: fracción del precio por lado ((entrada + salida) × comisión).
  - Swap: siempre EN CONTRA (largos y cortos), nominal = precio medio de entrada
    × 1 unidad, coste por noche = nominal × swap_anual / 365, y por operación =
    coste por noche × noches ponderadas (ver rollover_nights). Con 7 noches
    ponderadas por semana en los dos calendarios, "swap anual" significa el coste
    de mantener la posición un año natural.
  - Rollover a las 00:00 UTC (aproximación: XM lo hace a las 00:00 hora del
    servidor, ≈ 21:00–22:00 UTC). Una operación paga las noches cuyas
    medianoches UTC caen entre el día de la vela de entrada y el día de la vela
    de salida: días d con fecha(entrada) <= d < fecha(salida), siendo
    "fecha(salida)" la de la mitad de la vela de salida.

Funciones puras (sin BD ni red): solo numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Peso de la noche que EMPIEZA al final del día d (d.weekday(): lunes=0).
# "session" (oro, SPX): 5 noches con triple el miércoles (cubre el fin de semana);
# viernes→lunes paga solo la del viernes. 7 noches ponderadas por semana.
SESSION_NIGHT_WEIGHTS = (1, 1, 3, 1, 1, 0, 0)
CRYPTO_NIGHT_WEIGHTS = (1, 1, 1, 1, 1, 1, 1)

BINANCE_SCENARIO = "binance"


def _settings():
    from config import settings
    return settings


def rollover_nights(d_in, d_out, schedule: str) -> np.ndarray:
    """
    Noches de swap ponderadas por operación.

    Args:
        d_in, d_out: fechas (array-like de Timestamps; se normalizan al día UTC)
            del día de entrada y del día de salida. Se cobran las noches que
            empiezan al final de cada día d con d_in <= d < d_out.
        schedule: "crypto" (7 noches/semana, peso 1) o "session" (lunes a
            viernes, triple el miércoles, sábado y domingo 0).

    Returns:
        np.ndarray de float con las noches ponderadas (0 si sale el mismo día).

    Ejemplos ("session"): lunes→jueves = 1+1+3 = 5; viernes→lunes = 1;
    una semana completa = 7. ("crypto"): viernes→lunes = 3.
    """
    if schedule not in ("crypto", "session"):
        raise ValueError(f"swap_nights debe ser 'crypto' o 'session', no {schedule!r}")
    a = pd.DatetimeIndex(pd.to_datetime(pd.Series(d_in))).normalize()
    b = pd.DatetimeIndex(pd.to_datetime(pd.Series(d_out))).normalize()
    days = np.asarray((b - a).days, dtype=int)
    if schedule == "crypto":
        return np.clip(days, 0, None).astype(float)
    w = np.array(SESSION_NIGHT_WEIGHTS, dtype=float)
    week = w.sum()
    out = np.zeros(len(days), dtype=float)
    start_wd = np.asarray(a.weekday, dtype=int)
    for k, (n, wd) in enumerate(zip(days, start_wd)):
        if n <= 0:
            continue
        full, rem = divmod(int(n), 7)
        out[k] = full * week + sum(w[(wd + j) % 7] for j in range(rem))
    return out


def cost_scenarios(symbol: str, broker: str = "xm") -> list[dict]:
    """
    Escenarios de costes para un activo, en el orden en que se muestran.

    Cada escenario es un dict con:
        name:          identificador estable ("xm_swap0", "xm_swap5", "xm_swap10",
                       "xm_real" si se conoce el swap, y "binance");
        label:         texto corto en español para el informe;
        kind:          "broker" o "binance";
        spread, commission, swap_pct_year, swap_nights (kind="broker");
        commission, slippage (kind="binance");
        decision:      True en el escenario de decisión (COST_DECISION_SCENARIO);
        origen:        de dónde salen las cifras del bróker.

    Si el activo no tiene costes de bróker definidos, solo se devuelve "binance".
    """
    s = _settings()
    out = []
    spec = s.BROKER_COSTS.get(broker, {}).get(symbol)
    if spec:
        base = {"kind": "broker", "spread": float(spec["spread"]),
                "commission": float(spec.get("commission", 0.0)),
                "swap_nights": spec["swap_nights"], "origen": spec.get("origen", "")}
        for pct in s.SWAP_SCENARIOS_PCT_YEAR:
            name = f"{broker}_swap{int(round(pct * 100))}"
            out.append(dict(base, name=name, swap_pct_year=float(pct),
                            label=f"{broker.upper()}, swap {pct * 100:.0f} %"))
        if spec.get("swap_pct_year") is not None:
            pct = float(spec["swap_pct_year"])
            out.append(dict(base, name=f"{broker}_real", swap_pct_year=pct,
                            label=f"{broker.upper()}, swap real {pct * 100:.1f} %"))
    out.append({"name": BINANCE_SCENARIO, "kind": "binance",
                "label": "Referencia validador (com. 0,04 % + slip. 0,02 %/lado)",
                "commission": float(s.BACKTEST["commission"]),
                "slippage": float(s.BACKTEST["slippage"])})
    for sc in out:
        sc["decision"] = sc["name"] == s.COST_DECISION_SCENARIO
    return out


def net_returns(trades: pd.DataFrame, scenario: dict) -> np.ndarray:
    """
    R netos de cada operación bajo un escenario de costes.

    Args:
        trades: DataFrame con una fila por operación y, al menos, las columnas
            direction ("long"/"short"), entry_mid (apertura de la vela de
            entrada), exit_mid (nivel de SL/TP o cierre de salida), risk
            (atr_stop × ATR, en precio), y para los escenarios de bróker
            d_in / d_out (días de entrada y de salida, ver rollover_nights).
        scenario: un elemento de cost_scenarios().

    Returns:
        np.ndarray de R netos (misma longitud y orden que trades).

    "binance" reproduce la aritmética de validate._simulate_setup (fills con
    slippage y comisión sobre los fills). Los escenarios de bróker restan
    spread + comisión + swap al PnL a precios medios.
    """
    if trades is None or len(trades) == 0:
        return np.zeros(0, dtype=float)
    long_ = (trades["direction"].to_numpy() == "long")
    entry = trades["entry_mid"].to_numpy(dtype=float)
    exit_ = trades["exit_mid"].to_numpy(dtype=float)
    risk = trades["risk"].to_numpy(dtype=float)

    if scenario["kind"] == "binance":
        slip = scenario["slippage"]
        comm = scenario["commission"]
        entry_fill = np.where(long_, entry * (1 + slip), entry * (1 - slip))
        exit_fill = np.where(long_, exit_ * (1 - slip), exit_ * (1 + slip))
        gross = np.where(long_, exit_fill - entry_fill, entry_fill - exit_fill)
        net = gross - (entry_fill + exit_fill) * comm
        return net / risk

    gross = np.where(long_, exit_ - entry, entry - exit_)
    cost = scenario["spread"] + (entry + exit_) * scenario["commission"]
    pct = scenario.get("swap_pct_year") or 0.0
    if pct:
        nights = rollover_nights(trades["d_in"], trades["d_out"], scenario["swap_nights"])
        cost = cost + entry * pct / 365.0 * nights
    return (gross - cost) / risk


def cost_breakdown_R(trades: pd.DataFrame, scenario: dict) -> dict:
    """
    Coste MEDIO en R por operación de cada componente (para el detalle técnico):
    {"spread": ..., "commission": ..., "swap": ..., "nights_mean": ...}.
    Solo escenarios de bróker; para "binance" devuelve {"total": coste medio}.
    """
    if trades is None or len(trades) == 0:
        return {}
    entry = trades["entry_mid"].to_numpy(dtype=float)
    exit_ = trades["exit_mid"].to_numpy(dtype=float)
    risk = trades["risk"].to_numpy(dtype=float)
    if scenario["kind"] == "binance":
        free = dict(scenario, slippage=0.0, commission=0.0)
        return {"total": float(np.mean(net_returns(trades, free) - net_returns(trades, scenario)))}
    nights = rollover_nights(trades["d_in"], trades["d_out"], scenario["swap_nights"])
    pct = scenario.get("swap_pct_year") or 0.0
    return {
        "spread": float(np.mean(scenario["spread"] / risk)),
        "commission": float(np.mean((entry + exit_) * scenario["commission"] / risk)),
        "swap": float(np.mean(entry * pct / 365.0 * nights / risk)),
        "nights_mean": float(np.mean(nights)),
    }
