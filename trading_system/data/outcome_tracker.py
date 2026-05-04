"""
Outcome Tracker — registra señales y resuelve resultados automáticamente.

Flujo:
  1. Cuando se genera una señal → save_signal_from_trade() la persiste en DB
  2. Periódicamente (cada hora en el scheduler) → resolve_open_signals() revisa
     cada señal abierta contra los precios actuales y marca win/loss/expired
  3. Los resultados resueltos alimentan el reentrenamiento del modelo ML
"""
import pandas as pd
from datetime import datetime, timezone
from loguru import logger

from data.database import get_engine, save_signal, load_signals
from sqlalchemy import text


def _is_duplicate_signal(symbol: str, direction: str, entry_price: float,
                          hours: int = 8) -> bool:
    """True si ya existe una señal similar (mismo símbolo+dirección) en las últimas N horas."""
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    engine = get_engine()
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT COUNT(*) FROM signals "
                "WHERE symbol=:sym AND direction=:dir AND ts >= :cutoff"
            ), {"sym": symbol, "dir": direction, "cutoff": cutoff}).scalar()
            return (row or 0) > 0
    except Exception:
        return False


def save_signal_from_trade(trade, sentiment: dict = None) -> int:
    """
    Persiste una señal generada (TradeSetup) en la tabla signals.
    Retorna el ID asignado, o -1 si falló o era duplicado.
    """
    if _is_duplicate_signal(trade.symbol, trade.direction, trade.entry_price):
        logger.debug(f"Señal duplicada omitida: {trade.symbol} {trade.direction} ya guardada recientemente.")
        return -1
    try:
        signal_dict = {
            "symbol":          trade.symbol,
            "ts":              pd.Timestamp.now(tz="UTC"),
            "setup_type":      trade.setup_type,
            "direction":       trade.direction,
            "ml_score":        trade.ml_score,
            "sentiment_score": sentiment.get("score", 0.0) if sentiment else None,
            "entry_price":     trade.entry_price,
            "stop_loss":       trade.stop_loss,
            "take_profit":     trade.take_profit,
            "position_size":   trade.position_size,
            "atr":             trade.atr,
        }
        signal_id = save_signal(signal_dict)
        logger.info(f"Señal guardada en DB | {trade.symbol} {trade.direction.upper()} | ID={signal_id}")
        return signal_id
    except Exception as exc:
        logger.error(f"Error guardando señal: {exc}")
        return -1


def resolve_open_signals() -> int:
    """
    Revisa todas las señales abiertas (resultado_real IS NULL) y las resuelve
    comparando entry/SL/TP contra los precios OHLCV reales desde la señal.

    Returns:
        Número de señales resueltas en este ciclo.
    """
    engine = get_engine()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, symbol, ts, direction, entry_price, stop_loss, take_profit "
            "FROM signals WHERE resultado_real IS NULL ORDER BY ts ASC"
        )).fetchall()

    if not rows:
        return 0

    resolved_count = 0
    for row in rows:
        sig_id, symbol, ts_unix, direction, entry, sl, tp = row

        try:
            # PostgreSQL devuelve datetime objects; SQLite devuelve enteros Unix
            if isinstance(ts_unix, (int, float)):
                sig_ts = pd.Timestamp(ts_unix, unit="s", tz="UTC")
            else:
                sig_ts = pd.Timestamp(ts_unix)
                if sig_ts.tz is None:
                    sig_ts = sig_ts.tz_localize("UTC")

            # Timeout absoluto: señales con más de 7 días se fuerzan a expiradas
            from datetime import timedelta
            age = datetime.now(timezone.utc) - sig_ts
            if age > timedelta(days=7):
                _update_signal_result(sig_id, "expired", None, None)
                resolved_count += 1
                logger.info(f"Señal #{sig_id} {symbol} expirada por antigüedad ({age.days}d > 7d)")
                continue

            outcome, pnl_pct = _check_outcome(symbol, sig_ts, direction, entry, sl, tp)

            if outcome is not None:
                pnl_usd = round(entry * pnl_pct, 2) if pnl_pct is not None else None
                _update_signal_result(sig_id, outcome, pnl_usd, pnl_pct)
                resolved_count += 1
                logger.info(
                    f"Señal #{sig_id} {symbol} {direction.upper()} → "
                    f"{outcome.upper()} | PnL={pnl_usd:+.2f} USD" if pnl_usd else
                    f"Señal #{sig_id} {symbol} → {outcome.upper()}"
                )
        except Exception as exc:
            logger.warning(f"Error resolviendo señal #{sig_id}: {exc}")

    if resolved_count:
        logger.info(f"Outcome tracker: {resolved_count} señales resueltas.")
    return resolved_count


def _check_outcome(
    symbol: str,
    signal_ts: pd.Timestamp,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> tuple:
    """
    Descarga OHLCV desde la señal y revisa si se tocó SL o TP.

    Returns:
        (outcome, pnl_pct) donde outcome ∈ {"win","loss","expired"} o (None, None)
    """
    from config.settings import ASSETS

    asset_cfg = ASSETS.get(symbol, {})
    forward_bars = asset_cfg.get("forward_bars", 24)
    is_gold = asset_cfg.get("source") == "yfinance"

    # Cargar barras desde la señal hasta ahora
    try:
        if is_gold:
            from data.fetchers.gold_fetcher import fetch_gold
            df = fetch_gold(timeframe="1d", days=60)
        else:
            from data.fetchers.binance_fetcher import fetch_ohlcv
            df = fetch_ohlcv(symbol, timeframe="4h", days=30)
    except Exception:
        return None, None

    if df is None or df.empty:
        return None, None

    # Filtrar barras posteriores a la señal
    future = df[df.index > signal_ts].head(forward_bars)
    if len(future) < 2:
        return None, None  # Muy pronto para resolver

    for _, bar in future.iterrows():
        if direction == "long":
            if bar["low"] <= stop_loss:
                pnl_pct = (stop_loss - entry) / entry
                return "loss", round(pnl_pct, 6)
            if bar["high"] >= take_profit:
                pnl_pct = (take_profit - entry) / entry
                return "win", round(pnl_pct, 6)
        else:  # short
            if bar["high"] >= stop_loss:
                pnl_pct = (entry - stop_loss) / entry
                return "loss", round(pnl_pct, 6)
            if bar["low"] <= take_profit:
                pnl_pct = (entry - take_profit) / entry
                return "win", round(pnl_pct, 6)

    # Se agotaron las barras sin resolución → expirado
    if len(future) >= forward_bars:
        last_close = future["close"].iloc[-1]
        if direction == "long":
            pnl_pct = (last_close - entry) / entry
        else:
            pnl_pct = (entry - last_close) / entry
        return "expired", round(pnl_pct, 6)

    return None, None  # Aún hay barras por venir


def _update_signal_result(signal_id: int, outcome: str, pnl_usd, pnl_pct) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE signals SET resultado_real=:outcome, pnl_usd=:pnl_usd, pnl_pct=:pnl_pct "
            "WHERE id=:id"
        ), {"outcome": outcome, "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "id": signal_id})


def get_outcome_stats(symbol: str = None) -> dict:
    """
    Estadísticas de señales resueltas para mostrar en el dashboard.
    """
    df = load_signals(symbol=symbol, limit=500)
    if df.empty:
        return {}

    resolved = df[df["resultado_real"].isin(["win", "loss", "expired"])]
    if resolved.empty:
        return {"total": len(df), "resolved": 0}

    wins = resolved[resolved["resultado_real"] == "win"]
    losses = resolved[resolved["resultado_real"] == "loss"]

    return {
        "total":       len(df),
        "resolved":    len(resolved),
        "pending":     len(df) - len(resolved),
        "wins":        len(wins),
        "losses":      len(losses),
        "win_rate":    round(len(wins) / len(resolved), 3) if len(resolved) > 0 else 0,
        "avg_pnl_pct": round(resolved["pnl_pct"].mean() * 100, 2) if "pnl_pct" in resolved else 0,
        "total_pnl":   round(resolved["pnl_usd"].sum(), 2) if "pnl_usd" in resolved else 0,
    }
