"""
Validación rigurosa de expectativa — ¿tienen los setups edge tras costos?

Responde la pregunta fundamental del sistema: ¿los setups técnicos
(breakout/pullback/reversal), ejecutados de forma realista y con costos, tienen
expectativa positiva? Si la respuesta es no, ningún scoring (ML o raw_score)
puede rescatarlos.

Rigor (sin look-ahead, realista):
  - Entrada a la APERTURA de la barra siguiente al setup (no al cierre del setup).
  - SL/TP desde ese entry con los multiplicadores ATR del activo.
  - Comisión (ida+vuelta) y slippage en entrada y salida.
  - Simulación hacia adelante hasta forward_bars; si toca SL y TP en la misma
    barra, se asume SL primero (conservador).
  - Expectativa medida en R (PnL neto / riesgo). E[R] > 0 = edge tras costos.

NO usa el ML: prueba la materia prima (los setups). Compara además el subconjunto
de mayor raw_score para ver si ese score tiene poder de selección.
"""
import numpy as np
import pandas as pd
from loguru import logger

from config.settings import ASSETS, BACKTEST, RISK
from signals.detector import detect_all_setups


def _simulate_setup(df: pd.DataFrame, i_setup: int, direction: str, atr: float,
                    atr_stop: float, atr_target: float, forward_bars: int,
                    comm: float, slip: float) -> float | None:
    """
    Simula una operación entrando a la apertura de la barra i_setup+1.
    Retorna el PnL en R (neto de costos) o None si no se puede simular.
    """
    entry_pos = i_setup + 1
    if entry_pos >= len(df) or atr <= 0:
        return None

    raw_entry = float(df.iloc[entry_pos]["open"])
    if raw_entry <= 0:
        return None

    # Niveles desde el precio de entrada previsto (mid); los fills incluyen slippage.
    if direction == "long":
        entry_fill = raw_entry * (1 + slip)
        sl = raw_entry - atr_stop   * atr
        tp = raw_entry + atr_target * atr
    else:
        entry_fill = raw_entry * (1 - slip)
        sl = raw_entry + atr_stop   * atr
        tp = raw_entry - atr_target * atr

    risk_per_unit = atr_stop * atr
    if risk_per_unit <= 0:
        return None

    # Escaneo hacia adelante desde la barra posterior a la entrada.
    exit_price = None
    last_pos = min(entry_pos + forward_bars, len(df) - 1)
    for j in range(entry_pos + 1, last_pos + 1):
        b = df.iloc[j]
        if direction == "long":
            hit_sl = b["low"]  <= sl
            hit_tp = b["high"] >= tp
        else:
            hit_sl = b["high"] >= sl
            hit_tp = b["low"]  <= tp
        if hit_sl and hit_tp:
            exit_price = sl            # conservador: se asume SL primero
            break
        if hit_sl:
            exit_price = sl
            break
        if hit_tp:
            exit_price = tp
            break
    if exit_price is None:
        exit_price = float(df.iloc[last_pos]["close"])   # expira → cierre a mercado

    exit_fill = exit_price * (1 - slip if direction == "long" else 1 + slip)

    if direction == "long":
        gross = exit_fill - entry_fill
    else:
        gross = entry_fill - exit_fill

    commission = (entry_fill + exit_fill) * comm          # comisión por lado
    net = gross - commission
    return net / risk_per_unit                            # PnL en R


def _stats(rs: list) -> dict:
    """Métricas de expectativa a partir de una lista de PnL en R."""
    if not rs:
        return {"n": 0}
    arr = np.array(rs, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())
    return {
        "n":            len(arr),
        "win_rate":     round(float((arr > 0).mean()), 3),
        "expectancy_R": round(float(arr.mean()), 3),   # E[R] por operación
        "avg_win_R":    round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_R":   round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor":round(float(gross_win / (gross_loss + 1e-9)), 2),
        "total_R":      round(float(arr.sum()), 1),
    }


def validate_symbol(symbol: str, df_4h: pd.DataFrame,
                    df_daily: pd.DataFrame = None) -> dict:
    """Corre la validación de expectativa para un activo. Retorna dict de métricas."""
    cfg        = ASSETS.get(symbol, {})
    atr_stop   = cfg.get("atr_stop_mult",   RISK["atr_stop_mult"])
    atr_target = cfg.get("atr_target_mult", RISK["atr_target_mult"])
    forward    = cfg.get("forward_bars", 24)
    comm       = BACKTEST["commission"]
    slip       = BACKTEST["slippage"]
    breakeven  = round(atr_stop / (atr_stop + atr_target), 3)   # win rate de equilibrio

    df_setups = detect_all_setups(df_4h, df_daily, symbol)
    if df_setups.empty:
        return {"symbol": symbol, "n": 0}

    idx = df_4h.index
    rows = []
    for _, s in df_setups.iterrows():
        pos = int(idx.searchsorted(s["ts"]))
        if pos >= len(df_4h):
            continue
        r = _simulate_setup(df_4h, pos, s["direction"], float(s.get("atr", 0)),
                            atr_stop, atr_target, forward, comm, slip)
        if r is None:
            continue
        rows.append({"R": r, "setup_type": s["setup_type"],
                     "direction": s["direction"], "raw_score": float(s.get("raw_score", 0))})

    if not rows:
        return {"symbol": symbol, "n": 0}

    dfr = pd.DataFrame(rows)
    overall = _stats(dfr["R"].tolist())
    overall.update({
        "symbol":     symbol,
        "rr":         round(atr_target / atr_stop, 2),
        "breakeven":  breakeven,
        # Subconjunto de mayor raw_score (top 25%): ¿tiene el score poder de selección?
        "top25_expectancy_R": _stats(
            dfr[dfr["raw_score"] >= dfr["raw_score"].quantile(0.75)]["R"].tolist()
        ).get("expectancy_R", 0.0),
        "by_setup": {
            st: _stats(dfr[dfr["setup_type"] == st]["R"].tolist())
            for st in dfr["setup_type"].unique()
        },
        "by_direction": {
            d: _stats(dfr[dfr["direction"] == d]["R"].tolist())
            for d in dfr["direction"].unique()
        },
    })
    return overall


def validate_variants(symbol: str, df_4h: pd.DataFrame,
                      df_daily: pd.DataFrame = None) -> list:
    """
    Prueba variantes candidatas de estrategia para un activo, en una sola corrida.
    Útil para rescatar activos con E[R] negativo: ¿alguna versión (long-only,
    con filtro de tendencia, por tipo de setup) tiene expectativa positiva?
    """
    cfg        = ASSETS.get(symbol, {})
    atr_stop   = cfg.get("atr_stop_mult",   RISK["atr_stop_mult"])
    atr_target = cfg.get("atr_target_mult", RISK["atr_target_mult"])
    forward    = cfg.get("forward_bars", 24)
    comm       = BACKTEST["commission"]
    slip       = BACKTEST["slippage"]

    df = df_4h.copy()
    df["_ma200"] = df["close"].rolling(200).mean()
    df_setups = detect_all_setups(df_4h, df_daily, symbol)
    if df_setups.empty:
        return []

    idx = df.index
    rows = []
    for _, s in df_setups.iterrows():
        pos = int(idx.searchsorted(s["ts"]))
        if pos >= len(df):
            continue
        r = _simulate_setup(df, pos, s["direction"], float(s.get("atr", 0)),
                            atr_stop, atr_target, forward, comm, slip)
        if r is None:
            continue
        ma200 = df.iloc[pos]["_ma200"]
        uptrend = bool(pd.notna(ma200) and df.iloc[pos]["close"] > ma200)
        rows.append({"R": r, "dir": s["direction"], "type": s["setup_type"],
                     "uptrend": uptrend})
    if not rows:
        return []

    d = pd.DataFrame(rows)
    long_up = (d["dir"] == "long") & d["uptrend"]
    variants = {
        "TODOS (baseline actual)":          d,
        "Solo LONG":                        d[d["dir"] == "long"],
        "Solo SHORT":                       d[d["dir"] == "short"],
        "LONG + tendencia (close>MA200)":   d[long_up],
        "LONG pullback":                    d[(d["dir"] == "long") & (d["type"] == "pullback")],
        "LONG pullback + tendencia":        d[long_up & (d["type"] == "pullback")],
        "LONG breakout + tendencia":        d[long_up & (d["type"] == "breakout")],
    }
    out = []
    for name, sub in variants.items():
        st = _stats(sub["R"].tolist())
        st["variant"] = name
        out.append(st)
    return out


def print_variants_report(symbol: str, variants: list) -> None:
    """Imprime la comparación de variantes de estrategia para un activo."""
    logger.info("=" * 78)
    logger.info(f"VARIANTES DE ESTRATEGIA — {symbol}  (¿alguna con E[R] > 0?)")
    logger.info("=" * 78)
    logger.info(f"{'Variante':38} {'N':>5} {'WinRate':>8} {'E[R]':>8} {'PF':>6} {'TotalR':>8}")
    logger.info("-" * 78)
    for v in variants:
        if v.get("n", 0) < 10:
            logger.info(f"{v['variant']:38} {v.get('n',0):>5}   (muestra insuficiente)")
            continue
        mark = "✓" if v["expectancy_R"] > 0 else "✗"
        logger.info(
            f"{v['variant']:38} {v['n']:>5} {_fmt_pct(v['win_rate']):>8} "
            f"{v['expectancy_R']:>8.3f} {v['profit_factor']:>6.2f} {v['total_R']:>8.1f}  {mark}"
        )
    logger.info("=" * 78)


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def print_validation_report(results: list) -> None:
    """Imprime la tabla resumen de la validación."""
    logger.info("=" * 78)
    logger.info("VALIDACIÓN DE EXPECTATIVA — ¿tienen edge los setups tras costos?")
    logger.info("(E[R] = ganancia media por operación en múltiplos de riesgo; >0 = edge)")
    logger.info("=" * 78)
    header = (f"{'Activo':8} {'N':>5} {'WinRate':>8} {'BE':>6} {'R:R':>5} "
              f"{'E[R]':>7} {'PF':>5} {'TotalR':>8} {'Top25 E[R]':>11}")
    logger.info(header)
    logger.info("-" * 78)
    for m in results:
        if m.get("n", 0) == 0:
            logger.info(f"{m['symbol']:8} {'sin datos':>30}")
            continue
        veredicto = "✓" if m["expectancy_R"] > 0 else "✗"
        logger.info(
            f"{m['symbol']:8} {m['n']:>5} {_fmt_pct(m['win_rate']):>8} "
            f"{_fmt_pct(m['breakeven']):>6} {m['rr']:>5.2f} "
            f"{m['expectancy_R']:>7.3f} {m['profit_factor']:>5.2f} "
            f"{m['total_R']:>8.1f} {m['top25_expectancy_R']:>11.3f}  {veredicto}"
        )
    logger.info("-" * 78)
    logger.info("Por tipo de setup (E[R] por operación):")
    for m in results:
        if m.get("n", 0) == 0:
            continue
        parts = []
        for st, s in m.get("by_setup", {}).items():
            if s.get("n", 0) >= 10:
                parts.append(f"{st}={s['expectancy_R']:+.3f}(n={s['n']})")
        if parts:
            logger.info(f"  {m['symbol']:8} " + " | ".join(parts))
    logger.info("=" * 78)
