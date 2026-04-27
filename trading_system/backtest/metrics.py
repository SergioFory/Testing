"""
Cálculo de métricas de performance para backtests y operaciones reales.
"""
import pandas as pd
import numpy as np
from loguru import logger


def compute_metrics(
    trades:        list,
    equity_curve:  pd.Series,
    initial_capital: float,
) -> dict:
    """
    Calcula métricas completas de performance.

    Args:
        trades:          Lista de operaciones cerradas (dicts)
        equity_curve:    Serie temporal del capital
        initial_capital: Capital inicial

    Returns:
        dict con todas las métricas.
    """
    if not trades:
        return _empty_metrics()

    df_t = pd.DataFrame(trades)
    n    = len(df_t)

    # --- Win Rate y R:R ---
    wins      = df_t[df_t["outcome"] == "win"]
    losses    = df_t[df_t["outcome"] == "loss"]
    win_rate  = len(wins) / n if n > 0 else 0

    avg_win   = wins["pnl_usd"].mean()   if len(wins)   > 0 else 0
    avg_loss  = losses["pnl_usd"].mean() if len(losses) > 0 else 0
    avg_rr    = abs(avg_win / avg_loss)  if avg_loss != 0 else 0

    # Profit Factor
    gross_profit = wins["pnl_usd"].sum()   if len(wins)   > 0 else 0
    gross_loss   = abs(losses["pnl_usd"].sum()) if len(losses) > 0 else 1
    profit_factor = gross_profit / gross_loss

    # --- Retorno ---
    final_capital = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    total_return  = (final_capital - initial_capital) / initial_capital
    total_pnl_usd = final_capital - initial_capital

    # Retorno anualizado (CAGR)
    n_days = max(
        (equity_curve.index[-1] - equity_curve.index[0]).days, 1
    ) if len(equity_curve) > 1 else 365
    years  = n_days / 365.25
    if total_return > -1 and years > 0:
        ann_return = (1 + total_return) ** (1 / years) - 1
    else:
        ann_return = -1.0

    # --- Drawdown ---
    rolling_max    = equity_curve.cummax()
    drawdown_curve = (equity_curve - rolling_max) / rolling_max
    max_drawdown   = abs(drawdown_curve.min())
    avg_drawdown   = abs(drawdown_curve[drawdown_curve < 0].mean()) if (drawdown_curve < 0).any() else 0

    # --- Sharpe Ratio ---
    if len(equity_curve) > 1:
        daily_returns  = equity_curve.resample("1D").last().pct_change().dropna()
        if len(daily_returns) > 5 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # --- Calmar Ratio (retorno / max drawdown) ---
    calmar = ann_return / max_drawdown if max_drawdown > 0 else 0

    # --- Expectancy (esperanza matemática por trade) ---
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # --- Por setup type ---
    by_type = {}
    if "setup_type" in df_t.columns:
        for st in df_t["setup_type"].unique():
            sub = df_t[df_t["setup_type"] == st]
            sub_wins = sub[sub["outcome"] == "win"]
            by_type[st] = {
                "n":        len(sub),
                "win_rate": len(sub_wins) / len(sub) if len(sub) > 0 else 0,
                "pnl_usd":  round(sub["pnl_usd"].sum(), 2),
            }

    # --- Por dirección ---
    by_dir = {}
    if "direction" in df_t.columns:
        for d in ["long", "short"]:
            sub = df_t[df_t["direction"] == d]
            if len(sub) > 0:
                sub_wins = sub[sub["outcome"] == "win"]
                by_dir[d] = {
                    "n":        len(sub),
                    "win_rate": len(sub_wins) / len(sub),
                    "pnl_usd":  round(sub["pnl_usd"].sum(), 2),
                }

    return {
        "n_trades":           n,
        "n_wins":             len(wins),
        "n_losses":           len(losses),
        "win_rate":           round(win_rate, 4),
        "avg_win_usd":        round(avg_win, 2),
        "avg_loss_usd":       round(avg_loss, 2),
        "avg_rr":             round(avg_rr, 2),
        "profit_factor":      round(profit_factor, 2),
        "expectancy_usd":     round(expectancy, 2),
        "initial_capital":    initial_capital,
        "final_capital":      round(final_capital, 2),
        "total_pnl_usd":      round(total_pnl_usd, 2),
        "total_return_pct":   round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe":             round(sharpe, 2),
        "calmar":             round(calmar, 2),
        "max_drawdown_pct":   round(max_drawdown * 100, 2),
        "avg_drawdown_pct":   round(avg_drawdown * 100, 2),
        "by_setup_type":      by_type,
        "by_direction":       by_dir,
        "trades_df":          df_t,
        "equity_curve":       equity_curve,
    }


def _empty_metrics() -> dict:
    return {
        "n_trades": 0, "win_rate": 0, "sharpe": 0, "max_drawdown_pct": 0,
        "total_return_pct": 0, "annualized_return_pct": 0,
        "profit_factor": 0, "avg_rr": 0,
    }


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """Retorna un DataFrame formateado para mostrar en Streamlit."""
    rows = [
        ("Trades totales",         metrics.get("n_trades", 0),              ""),
        ("Win Rate",               f"{metrics.get('win_rate', 0)*100:.1f}%","≥ 45%"),
        ("R:R Promedio",           f"{metrics.get('avg_rr', 0):.2f}",        "≥ 1.5"),
        ("Profit Factor",          f"{metrics.get('profit_factor', 0):.2f}", "≥ 1.3"),
        ("Retorno Total",          f"{metrics.get('total_return_pct', 0):+.1f}%", ""),
        ("Retorno Anualizado",     f"{metrics.get('annualized_return_pct', 0):+.1f}%", "≥ 30%"),
        ("Sharpe Ratio",           f"{metrics.get('sharpe', 0):.2f}",        "≥ 1.0"),
        ("Calmar Ratio",           f"{metrics.get('calmar', 0):.2f}",        "≥ 0.5"),
        ("Max Drawdown",           f"{metrics.get('max_drawdown_pct', 0):.1f}%", "≤ 20%"),
        ("Expectativa / trade",    f"${metrics.get('expectancy_usd', 0):+.2f}", "> 0"),
    ]
    return pd.DataFrame(rows, columns=["Métrica", "Valor", "Target"])
