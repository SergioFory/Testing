"""
Prueba de regresión "vivo = validador" (Fase 0, 2026-11).

Reproduce, vela a vela, la selección que hace el vivo (main.cmd_signal) con la
MISMA función de producción (ml.predictor.get_best_setup): ventana anclada a la
última vela cerrada, setups aprobados por umbral, filtro de frescura y
antiduplicado permanente por (vela del setup, dirección).

El validador (backtest/validate.py) opera cada setup aprobado UNA vez, entrando
en la vela siguiente. Si el vivo es fiel, el conjunto de setups emitidos debe
coincidir con el conjunto de setups aprobados, y todos con edad 0 (emitidos en
el primer ciclo tras el cierre de su vela).

Antes de la Fase 0 la emulación daba entre un 45 % y un 68 % de emisiones
repetidas de setups antiguos (re-emisiones). Uso:

    python main.py check-live --symbol BTCUSDT
    python main.py check-live            # todos los activos operativos
"""
from __future__ import annotations

import pandas as pd

from config.settings import MAX_ENTRY_DIST_ATR, MAX_SETUP_AGE_BARS, ASSETS


def emulate_live_signals(symbol: str, df_4h: pd.DataFrame,
                         df_daily: pd.DataFrame = None) -> pd.DataFrame:
    """
    Emula el ciclo vivo al cierre de cada vela. Devuelve un DataFrame con una
    fila por señal emitida: emit_ts (vela tras cuyo cierre se emite), setup_ts,
    direction, age_bars (velas entre el setup y la emisión), entry, raw_score.
    """
    from signals.detector import detect_all_setups
    from ml.predictor import score_setups, get_best_setup

    setups = detect_all_setups(df_4h, df_daily, symbol)
    if setups.empty:
        return pd.DataFrame(columns=["emit_ts", "setup_ts", "direction", "age_bars",
                                     "entry", "raw_score"])
    scored = score_setups(setups, df_4h, symbol, None).copy()
    idx = df_4h.index
    scored["pos"] = idx.searchsorted(scored["ts"])
    closes = df_4h["close"].to_numpy()
    max_dist = ASSETS.get(symbol, {}).get("max_entry_dist_atr", MAX_ENTRY_DIST_ATR)

    emitted_keys: set = set()
    rows = []
    for k in range(int(scored["pos"].min()), len(idx)):
        lo = max(0, k - MAX_SETUP_AGE_BARS)
        cand = scored[(scored["pos"] >= lo) & (scored["pos"] <= k)]
        if cand.empty:
            continue
        best = get_best_setup(cand, current_price=float(closes[k]), max_dist_atr=max_dist,
                              min_setup_ts=idx[lo], max_setup_ts=idx[k],
                              exclude_keys=emitted_keys)
        if best is None:
            continue
        key = (pd.Timestamp(best["ts"]), best["direction"])
        emitted_keys.add(key)
        rows.append({
            "emit_ts":   idx[k],
            "setup_ts":  key[0],
            "direction": key[1],
            "age_bars":  int(k - best["pos"]),
            "entry":     float(best["close"]),
            "raw_score": float(best.get("raw_score", best.get("ml_score", 0))),
        })
    return pd.DataFrame(rows)


def compare_with_validator(symbol: str, df_4h: pd.DataFrame,
                           df_daily: pd.DataFrame = None) -> dict:
    """
    Compara las señales emuladas con el conjunto de setups aprobados que opera
    el validador. Devuelve métricas y las diferencias.
    """
    from signals.detector import detect_all_setups
    from ml.predictor import score_setups

    setups = detect_all_setups(df_4h, df_daily, symbol)
    if setups.empty:
        return {"symbol": symbol, "approved": 0, "emitted": 0, "match": True,
                "missing": [], "extra": [], "repeated": 0, "pct_age_le1": 1.0}
    scored = score_setups(setups, df_4h, symbol, None)
    approved = {(pd.Timestamp(t), d) for t, d, ok in
                zip(scored["ts"], scored["direction"], scored["ml_approved"]) if ok}

    em = emulate_live_signals(symbol, df_4h, df_daily)
    emitted_list = list(zip(em["setup_ts"], em["direction"])) if not em.empty else []
    emitted = set(emitted_list)
    repeated = len(emitted_list) - len(emitted)
    missing = sorted(approved - emitted)
    extra = sorted(emitted - approved)
    pct_age_le1 = float((em["age_bars"] <= 1).mean()) if not em.empty else 1.0
    pct_age0 = float((em["age_bars"] == 0).mean()) if not em.empty else 1.0
    return {
        "symbol":      symbol,
        "approved":    len(approved),
        "emitted":     len(emitted_list),
        "repeated":    repeated,
        "missing":     missing,
        "extra":       extra,
        "pct_age0":    pct_age0,
        "pct_age_le1": pct_age_le1,
        # Aceptación Fase 0: 0 repeticiones, nada emitido fuera de lo aprobado,
        # todo lo aprobado emitido y >= 95 % con edad <= 1 vela.
        "match":       repeated == 0 and not extra and not missing and pct_age_le1 >= 0.95,
    }


def print_comparison(res: dict) -> None:
    ok = "OK" if res["match"] else "DIFERENCIAS"
    print(f"\n[{res['symbol']}] vivo = validador: {ok}")
    print(f"  setups aprobados (validador): {res['approved']}")
    print(f"  señales emitidas (vivo emulado): {res['emitted']}  |  repetidas: {res['repeated']}")
    print(f"  edad 0: {res.get('pct_age0', 1.0):.1%}  |  edad <= 1 vela: {res['pct_age_le1']:.1%}")
    if res["missing"]:
        print(f"  aprobados nunca emitidos ({len(res['missing'])}): "
              + ", ".join(f"{t:%Y-%m-%d %H:%M} {d}" for t, d in res["missing"][:10]))
    if res["extra"]:
        print(f"  emitidos sin estar aprobados ({len(res['extra'])}): "
              + ", ".join(f"{t:%Y-%m-%d %H:%M} {d}" for t, d in res["extra"][:10]))
