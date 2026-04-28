"""
Sistema de Trading v5.0 — Punto de entrada principal.

Modos de uso:
  python main.py signal   --symbol BTCUSDT     # Señal actual
  python main.py backtest --symbol BTCUSDT     # Backtest completo
  python main.py train    --symbol BTCUSDT     # Entrenar modelo
  python main.py update                         # Actualizar todos los datos
  python main.py schedule                       # Modo daemon (señales cada hora)
"""
import argparse
import sys
import os
from pathlib import Path

# Agregar el directorio raíz del sistema al path de Python
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configurar logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/trading_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from config.settings import ASSETS, BACKTEST, RISK, ML_PARAMS, DAYS_HISTORY


# ---------------------------------------------------------------------------
# Helpers de datos
# ---------------------------------------------------------------------------

def _load_data(symbol: str):
    """Descarga o carga datos OHLCV para el símbolo dado."""
    from data.fetchers.binance_fetcher import fetch_multi_timeframe
    from data.fetchers.gold_fetcher    import fetch_gold, fetch_macro
    from data.database                 import init_db, save_ohlcv, load_ohlcv

    init_db()

    is_gold = symbol.upper() in ("XAUUSD", "GC=F", "GOLD")

    logger.info(f"Cargando datos para {symbol}...")

    if is_gold:
        # Gold: datos diarios para ambos timeframes.
        # Usamos 2500 días (~6.8 años) para tener suficientes setups
        # para entrenamiento ML (GC=F tiene historia desde los años 80).
        gold_days = ASSETS.get(symbol, {}).get("days_history", 2500)
        df_daily  = fetch_gold(timeframe="1d", days=gold_days)
        df_4h     = df_daily.copy() if not df_daily.empty else None
        macro_df  = fetch_macro(days=min(gold_days, 730))
    else:
        crypto_days = ASSETS.get(symbol, {}).get("days_history", DAYS_HISTORY)
        data     = fetch_multi_timeframe(symbol, ["4h", "1d"], days=crypto_days)
        df_4h    = data.get("4h")
        df_daily = data.get("1d")
        macro_df = None

    if df_4h is None or df_4h.empty:
        logger.error(f"No se pudieron obtener datos 4H para {symbol}")
        return None, None, None

    save_ohlcv(df_4h, symbol, "4h")
    if df_daily is not None:
        save_ohlcv(df_daily, symbol, "1d")

    return df_4h, df_daily, macro_df


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_signal(symbol: str, notify: bool = False):
    """Genera la señal actual para el símbolo."""
    from signals.detector          import detect_all_setups, get_current_setup
    from ml.predictor              import score_setups, get_best_setup
    from risk.position_sizer       import build_trade_setup, PortfolioRiskManager
    from sentiment.cryptopanic     import get_sentiment_summary
    from notifications.telegram    import send_signal_alert, send_no_signal

    df_4h, df_daily, macro_df = _load_data(symbol)
    if df_4h is None:
        return

    # Detectar setups
    df_setups = detect_all_setups(df_4h, df_daily, symbol)
    if df_setups.empty:
        logger.info(f"[{symbol}] Sin setups detectados en este momento.")
        return

    # Scoring ML
    df_scored = score_setups(df_setups, df_4h, symbol, macro_df)

    # Mejor setup actual
    best = get_best_setup(df_scored, n_last_bars=3)
    if best is None:
        logger.info(f"[{symbol}] Sin setup de alta confianza en las últimas 3 barras.")
        return

    ml_score = float(best.get("ml_score", best.get("raw_score", 0)))
    portfolio = PortfolioRiskManager()
    trade = build_trade_setup(best, portfolio.current_capital, ml_score)

    if trade is None:
        logger.info(f"[{symbol}] Setup detectado pero no pasa validaciones de riesgo.")
        return

    # Sentimiento
    base_sym = symbol.replace("USDT", "")
    sentiment = get_sentiment_summary(base_sym)

    _print_signal(trade, sentiment)

    if notify:
        send_signal_alert(trade, sentiment)


def cmd_backtest(symbol: str, start: str = None, end: str = None,
                  capital: float = None):
    """Ejecuta backtest para el símbolo."""
    from backtest.engine  import run_backtest
    from backtest.metrics import format_metrics_table

    df_4h, df_daily, macro_df = _load_data(symbol)
    if df_4h is None:
        return

    logger.info(f"Iniciando backtest para {symbol}...")
    metrics = run_backtest(
        df_4h        = df_4h,
        df_daily     = df_daily,
        symbol       = symbol,
        macro_df     = macro_df,
        start_date   = start,
        end_date     = end,
        initial_capital = capital or BACKTEST["initial_capital"],
        verbose      = True,
    )

    if not metrics:
        logger.error("Backtest no produjo resultados.")
        return

    tbl = format_metrics_table(metrics)
    logger.info("\n" + tbl.to_string(index=False))

    # Guardar en base de datos
    from data.database import save_backtest_result
    save_backtest_result(metrics)
    logger.info("Resultados guardados en base de datos.")


def cmd_train(symbol: str):
    """Entrena el modelo ML para el símbolo."""
    from signals.detector import detect_all_setups
    from ml.features      import build_training_dataset, compute_base_features
    from ml.trainer       import train_model

    df_4h, df_daily, macro_df = _load_data(symbol)
    if df_4h is None:
        return

    logger.info(f"Detectando setups históricos para {symbol}...")
    df_setups = detect_all_setups(df_4h, df_daily, symbol)

    if len(df_setups) < 50:
        logger.warning(
            f"Solo {len(df_setups)} setups detectados. "
            "Se recomienda ≥ 200 para entrenamiento robusto."
        )

    forward_bars = ASSETS.get(symbol, {}).get("forward_bars", 12)

    # Funding rates para crypto (señal de sentimiento de mercado)
    funding_series = None
    if ASSETS.get(symbol, {}).get("source") == "binance":
        from data.fetchers.binance_fetcher import fetch_funding_rates
        days_hist = ASSETS.get(symbol, {}).get("days_history", 730)
        funding_series = fetch_funding_rates(symbol, days=days_hist)

    logger.info(f"Construyendo dataset de entrenamiento ({len(df_setups)} setups, forward_bars={forward_bars})...")
    dataset = build_training_dataset(
        df_setups      = df_setups,
        df_ohlcv       = df_4h,
        forward_bars   = forward_bars,
        macro_df       = macro_df,
        funding_series = funding_series,
    )

    if dataset.empty:
        logger.error("Dataset vacío. No se puede entrenar.")
        return

    wins = (dataset["target"] == 1).sum()
    total = len(dataset)
    logger.info(
        f"Dataset: {total} setups etiquetados | "
        f"Win rate histórico: {wins/total*100:.1f}%"
    )

    model, feat_cols, metrics = train_model(dataset, symbol=symbol, save=True)
    if model is None:
        logger.error("Entrenamiento fallido.")
        return

    logger.info(
        f"Modelo guardado | "
        f"Walk-Forward AUC: {metrics.get('auc', 0):.3f} | "
        f"Precision@{ML_PARAMS['prob_threshold']}: {metrics.get('precision', 0):.3f} | "
        f"Folds: {metrics.get('n_folds', 0)}"
    )


def cmd_update():
    """Actualiza datos OHLCV para todos los activos configurados."""
    from data.database import init_db
    init_db()

    for symbol, cfg in ASSETS.items():
        try:
            _load_data(symbol)
            logger.info(f"{symbol}: datos actualizados.")
        except Exception as e:
            logger.error(f"{symbol}: error al actualizar — {e}")


def cmd_schedule(interval_minutes: int = 60):
    """
    Modo daemon: genera señales en un intervalo fijo.
    Envía notificaciones por Telegram si está configurado.
    """
    import time
    import schedule as sched
    from notifications.telegram  import test_connection
    from config.settings         import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

    notify = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    if notify:
        test_connection()

    logger.info(
        f"Iniciando modo daemon — señales cada {interval_minutes} minutos. "
        f"Telegram: {'ON' if notify else 'OFF (configura TOKEN y CHAT_ID)'}. "
        "Ctrl+C para detener."
    )

    def _run_all_signals():
        for symbol in ASSETS:
            try:
                cmd_signal(symbol, notify=notify)
            except Exception as e:
                logger.error(f"Error generando señal para {symbol}: {e}")

    sched.every(interval_minutes).minutes.do(_run_all_signals)
    _run_all_signals()   # Ejecución inmediata al arrancar

    while True:
        sched.run_pending()
        time.sleep(30)


# ---------------------------------------------------------------------------
# Output formateado
# ---------------------------------------------------------------------------

def _print_signal(trade, sentiment: dict) -> None:
    from rich.console import Console
    from rich.table   import Table
    from rich.panel   import Panel
    from rich         import box

    console = Console()

    direction_color = "green" if trade.direction == "long" else "red"
    direction_label = "LONG  ▲" if trade.direction == "long" else "SHORT ▼"

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Campo",  style="bold cyan", width=22)
    table.add_column("Valor",  style="white")

    table.add_row("Símbolo",      trade.symbol)
    table.add_row("Dirección",    f"[{direction_color}]{direction_label}[/{direction_color}]")
    table.add_row("Setup Type",   trade.setup_type.replace("_", " ").title())
    table.add_row("Confianza ML", f"{trade.ml_score:.1%}")
    table.add_row("",             "")
    table.add_row("Entrada",      f"${trade.entry_price:,.2f}")
    table.add_row("Stop Loss",    f"${trade.stop_loss:,.2f}")
    table.add_row("Take Profit",  f"${trade.take_profit:,.2f}")
    table.add_row("R:R Ratio",    f"{trade.rr_ratio:.2f}")
    table.add_row("",             "")
    table.add_row("Tamaño ($)",   f"${trade.position_size:,.0f}")
    table.add_row("Riesgo ($)",   f"${trade.risk_usd:,.0f}")
    table.add_row("Potencial ($)",f"${trade.reward_usd:,.0f}")

    sent_color = sentiment.get("color", "gray")
    sent_label = sentiment.get("label", "NEUTRAL")
    sent_score = sentiment.get("score", 0)
    table.add_row("",             "")
    table.add_row("Sentimiento",  f"[{sent_color}]{sent_label} ({sent_score:+.2f})[/{sent_color}]")
    table.add_row("Noticias",     f"{sentiment.get('n_news', 0)} artículos recientes")

    console.print(Panel(
        table,
        title=f"[bold] SEÑAL — {trade.symbol} [/bold]",
        border_style=direction_color,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sistema de Trading v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py signal   --symbol BTCUSDT
  python main.py signal   --symbol BTCUSDT --notify
  python main.py backtest --symbol ETHUSDT --start 2023-01-01 --capital 10000
  python main.py train    --symbol BTCUSDT
  python main.py update
  python main.py schedule --interval 60
  python main.py telegram                           # test de conexión
        """,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # signal
    ps = sub.add_parser("signal", help="Generar señal actual")
    ps.add_argument("--symbol", default="BTCUSDT",
                    choices=list(ASSETS.keys()),
                    help="Símbolo a analizar")
    ps.add_argument("--notify", action="store_true",
                    help="Enviar alerta a Telegram si hay señal")

    # backtest
    pb = sub.add_parser("backtest", help="Ejecutar backtest")
    pb.add_argument("--symbol",  default="BTCUSDT", choices=list(ASSETS.keys()))
    pb.add_argument("--start",   default=None,   help="Fecha inicio YYYY-MM-DD")
    pb.add_argument("--end",     default=None,   help="Fecha fin   YYYY-MM-DD")
    pb.add_argument("--capital", default=None,   type=float, help="Capital inicial USD")

    # train
    pt = sub.add_parser("train", help="Entrenar modelo ML")
    pt.add_argument("--symbol", default="BTCUSDT", choices=list(ASSETS.keys()))

    # update
    sub.add_parser("update", help="Actualizar todos los datos OHLCV")

    # schedule (daemon)
    psc = sub.add_parser("schedule", help="Modo daemon con señales periódicas")
    psc.add_argument("--interval", default=60, type=int,
                     help="Intervalo en minutos (default: 60)")

    # telegram (test de conexión)
    sub.add_parser("telegram", help="Verificar conexión a Telegram y enviar mensaje de prueba")

    return p


def main():
    # Crear directorios necesarios
    for d in ("logs", "models", "data/db"):
        Path(d).mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "signal":
        cmd_signal(args.symbol, notify=args.notify)

    elif args.command == "backtest":
        cmd_backtest(args.symbol, args.start, args.end, args.capital)

    elif args.command == "train":
        cmd_train(args.symbol)

    elif args.command == "update":
        cmd_update()

    elif args.command == "schedule":
        cmd_schedule(args.interval)

    elif args.command == "telegram":
        from notifications.telegram import test_connection
        test_connection()


if __name__ == "__main__":
    main()
