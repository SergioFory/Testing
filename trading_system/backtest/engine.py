"""
Motor de backtest basado en eventos (event-driven).
Simula el sistema completo: detección de setups → ML scoring → gestión de riesgo.

NO usa look-ahead bias:
  - Los setups se detectan con datos hasta la barra actual
  - El modelo ML no conoce datos futuros
  - Cada operación tiene stop loss y take profit definidos antes de entrar
"""
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from config.settings      import BACKTEST, RISK
from signals.detector     import detect_all_setups
from ml.features          import build_training_dataset, compute_base_features
from ml.trainer           import train_model
from ml.predictor         import score_setups
from risk.position_sizer  import build_trade_setup, PortfolioRiskManager
from backtest.metrics     import compute_metrics


def run_backtest(
    df_4h:      pd.DataFrame,
    df_daily:   pd.DataFrame,
    symbol:     str,
    macro_df:   pd.DataFrame = None,
    start_date: str = None,
    end_date:   str = None,
    initial_capital: float = None,
    verbose:    bool = True,
) -> dict:
    """
    Ejecuta el backtest completo del sistema.

    Flujo por barra:
    1. Detectar setups hasta barra actual
    2. Si hay setup nuevo, entrenamiento ML en datos pasados
    3. Puntuar setup con ML
    4. Verificar filtros de riesgo
    5. "Ejecutar" la operación (simular con las barras siguientes)

    Args:
        df_4h:    OHLCV 4H completo
        df_daily: OHLCV diario
        symbol:   Símbolo
        macro_df: Features macro
        start_date: Inicio del período de backtest
        end_date:   Fin del período (None=hasta el final)
        initial_capital: Capital inicial
        verbose:  Si True, logs detallados

    Returns:
        dict con métricas completas + DataFrame de operaciones
    """
    cap      = initial_capital or BACKTEST["initial_capital"]
    comm     = BACKTEST["commission"]
    slip     = BACKTEST["slippage"]
    run_id   = str(uuid.uuid4())[:8]

    start = start_date or BACKTEST["start_date"]
    end   = end_date   or str(datetime.utcnow().date())

    df = df_4h.copy()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    if len(df) < 200:
        logger.error(f"Datos insuficientes para backtest: {len(df)} barras")
        return {}

    logger.info(
        f"Backtest {symbol} | {start} → {end} | "
        f"{len(df)} barras 4H | Capital: ${cap:,.0f}"
    )

    portfolio    = PortfolioRiskManager(capital=cap)
    all_trades   = []
    open_trades  = {}   # symbol+ts -> TradeSetup
    equity_curve = [cap]
    equity_dates = [df.index[0]]

    # Ventana mínima de warm-up para indicadores
    WARMUP = 200

    for i in range(WARMUP, len(df)):
        bar       = df.iloc[i]
        bar_ts    = df.index[i]
        df_window = df.iloc[:i]   # Solo datos pasados (no look-ahead)

        # --- 1. Gestionar posiciones abiertas (stop/target check) ---
        to_close = []
        for trade_key, trade in open_trades.items():
            closed = _check_exit(trade, bar, comm, slip)
            if closed:
                to_close.append((trade_key, closed))

        for trade_key, closed_trade in to_close:
            pnl = closed_trade["pnl_usd"]
            portfolio.current_capital += pnl
            portfolio.daily_pnl       += pnl
            portfolio.peak_capital     = max(portfolio.peak_capital, portfolio.current_capital)
            all_trades.append(closed_trade)
            open_trades.pop(trade_key)
            if verbose:
                logger.debug(
                    f"{bar_ts.date()} | CERRADO {closed_trade['symbol']} "
                    f"{closed_trade['outcome'].upper()} | "
                    f"PnL: {closed_trade['pnl_usd']:+.2f}"
                )

        equity_curve.append(portfolio.current_capital)
        equity_dates.append(bar_ts)

        # --- 2. Detectar setups en ventana actual ---
        if i % 6 != 0:   # Solo revisar cada 6 barras para eficiencia
            continue

        df_daily_window = df_daily[df_daily.index <= bar_ts] if df_daily is not None else None
        df_setups = detect_all_setups(df_window, df_daily_window, symbol)

        if df_setups.empty:
            continue

        # Solo el setup más reciente (si es de las últimas 2 barras)
        recent_cutoff = bar_ts - pd.Timedelta(hours=8)
        recent_setups = df_setups[df_setups["ts"] >= recent_cutoff]
        if recent_setups.empty:
            continue

        # --- 3. ML Scoring (entrenamiento en datos pasados) ---
        # Para backtest eficiente, entrenamos cada N setups (no en cada barra)
        recent_setups = _apply_ml_scoring_backtest(
            recent_setups, df_window, macro_df, symbol, i
        )

        # --- 4. Evaluar cada setup ---
        for _, setup_row in recent_setups.iterrows():
            ml_score = float(setup_row.get("ml_score", setup_row["raw_score"]))
            if ml_score < 0.50:
                continue

            trade_setup = build_trade_setup(
                setup_row, portfolio.current_capital, ml_score
            )
            if trade_setup is None:
                continue

            can_open, reason = portfolio.can_open_trade(trade_setup)
            if not can_open:
                logger.debug(f"Setup rechazado: {reason}")
                continue

            # Abrir la operación
            trade_key = f"{symbol}_{bar_ts}"
            open_trades[trade_key] = {
                "symbol":       symbol,
                "direction":    trade_setup.direction,
                "setup_type":   trade_setup.setup_type,
                "entry_price":  trade_setup.entry_price * (1 + slip if trade_setup.direction == "long" else 1 - slip),
                "stop_loss":    trade_setup.stop_loss,
                "take_profit":  trade_setup.take_profit,
                "position_size":trade_setup.position_size,
                "atr":          trade_setup.atr,
                "ml_score":     ml_score,
                "entry_ts":     bar_ts,
                "rr_ratio":     trade_setup.rr_ratio,
            }
            if verbose:
                logger.debug(
                    f"{bar_ts.date()} | ABIERTO {symbol} "
                    f"{trade_setup.direction.upper()} | "
                    f"Entry: {trade_setup.entry_price:,.2f} | "
                    f"ML: {ml_score:.2f} | R:R {trade_setup.rr_ratio:.1f}"
                )
            break   # Un solo setup por período

    # Cerrar posiciones abiertas al precio del último bar
    last_close = float(df.iloc[-1]["close"])
    for trade_key, trade in list(open_trades.items()):
        closed = _force_close(trade, last_close, comm, slip)
        all_trades.append(closed)

    # --- Métricas finales ---
    equity_series = pd.Series(equity_curve, index=equity_dates)
    metrics = compute_metrics(all_trades, equity_series, cap)
    metrics.update({
        "run_id":    run_id,
        "symbol":    symbol,
        "start_date":start,
        "end_date":  end,
        "n_bars":    len(df),
    })
    _print_summary(metrics, verbose)
    return metrics


def _check_exit(trade: dict, bar: pd.Series,
                comm: float, slip: float) -> dict | None:
    """Verifica si la barra toca stop o target."""
    direction = trade["direction"]
    sl = trade["stop_loss"]
    tp = trade["take_profit"]
    entry = trade["entry_price"]
    size  = trade["position_size"]

    hit_sl = hit_tp = False

    if direction == "long":
        if bar["low"]  <= sl: hit_sl = True
        if bar["high"] >= tp: hit_tp = True
    else:
        if bar["high"] >= sl: hit_sl = True
        if bar["low"]  <= tp: hit_tp = True

    if not hit_sl and not hit_tp:
        return None

    # Si ambos se tocan en la misma barra, asumimos que se ejecuta el primero
    # (en práctica esto depende de la apertura de la barra)
    if hit_tp and hit_sl:
        # Usar apertura como discriminador
        if direction == "long":
            exit_price = tp if bar["open"] > sl else sl
        else:
            exit_price = tp if bar["open"] < sl else sl
    elif hit_tp:
        exit_price = tp
    else:
        exit_price = sl

    # Aplicar slippage y comisión
    exit_price_adj = exit_price * (1 - slip if direction == "long" else 1 + slip)
    if direction == "long":
        pnl_pct = (exit_price_adj - entry) / entry
    else:
        pnl_pct = (entry - exit_price_adj) / entry

    commission = size * (comm * 2)  # Entrada + salida
    pnl_usd = size * pnl_pct - commission
    outcome = "win" if pnl_usd > 0 else "loss"

    return {
        **trade,
        "exit_price": exit_price_adj,
        "pnl_pct":    round(pnl_pct * 100, 2),
        "pnl_usd":    round(pnl_usd, 2),
        "outcome":    outcome,
        "commission": round(commission, 4),
    }


def _force_close(trade: dict, exit_price: float,
                 comm: float, slip: float) -> dict:
    """Cierra posición al precio de mercado."""
    direction = trade["direction"]
    entry     = trade["entry_price"]
    size      = trade["position_size"]
    exit_adj  = exit_price * (1 - slip if direction == "long" else 1 + slip)
    pnl_pct   = (exit_adj - entry) / entry if direction == "long" else (entry - exit_adj) / entry
    commission= size * (comm * 2)
    pnl_usd   = size * pnl_pct - commission
    return {
        **trade,
        "exit_price": exit_adj,
        "pnl_pct":    round(pnl_pct * 100, 2),
        "pnl_usd":    round(pnl_usd, 2),
        "outcome":    "open_at_end",
        "commission": round(commission, 4),
    }


def _apply_ml_scoring_backtest(
    setups:  pd.DataFrame,
    df_window: pd.DataFrame,
    macro_df:  pd.DataFrame,
    symbol:  str,
    bar_idx: int,
) -> pd.DataFrame:
    """
    Scoring ML simplificado para backtest.
    Cada N barras, reentrenamos el modelo en el histórico disponible.
    """
    try:
        all_setups_hist = detect_all_setups(df_window, None, symbol)
        if len(all_setups_hist) < 50:
            setups["ml_score"] = setups["raw_score"]
            setups["ml_approved"] = True
            return setups

        dataset = build_training_dataset(
            all_setups_hist, df_window,
            forward_bars=12, macro_df=macro_df
        )
        if dataset.empty or len(dataset) < 30:
            setups["ml_score"] = setups["raw_score"]
            setups["ml_approved"] = True
            return setups

        model, feat_cols, _ = train_model(dataset, symbol=f"{symbol}_bt", save=False)
        if model is None:
            setups["ml_score"] = setups["raw_score"]
            setups["ml_approved"] = True
            return setups

        from ml.predictor import score_setups
        return score_setups(setups, df_window, f"{symbol}_bt", macro_df)

    except Exception as e:
        logger.debug(f"ML scoring en backtest falló, usando raw_score: {e}")
        setups["ml_score"] = setups["raw_score"]
        setups["ml_approved"] = True
        return setups


def _print_summary(metrics: dict, verbose: bool) -> None:
    if not verbose:
        return
    logger.info("=" * 55)
    logger.info(f"BACKTEST RESULTS — {metrics.get('symbol', '?')}")
    logger.info(f"  Período    : {metrics.get('start_date')} → {metrics.get('end_date')}")
    logger.info(f"  Trades     : {metrics.get('n_trades', 0)}")
    logger.info(f"  Win Rate   : {metrics.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Avg R:R    : {metrics.get('avg_rr', 0):.2f}")
    logger.info(f"  Return     : {metrics.get('total_return_pct', 0):+.1f}%")
    logger.info(f"  Ann. Return: {metrics.get('annualized_return_pct', 0):+.1f}%")
    logger.info(f"  Sharpe     : {metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Max DD     : {metrics.get('max_drawdown_pct', 0):.1f}%")
    logger.info("=" * 55)
