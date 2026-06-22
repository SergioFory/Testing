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

import json
from datetime import datetime, timedelta, timezone

# Log heredado en disco. Se mantiene solo como fallback de lectura para una
# transición suave; la fuente de verdad ahora es la tabla retrain_log en la DB,
# que sobrevive a reinicios del Worker (ver data/database.py).
_RETRAIN_LOG = Path(__file__).parent / "data" / "retrain_log.json"


def _get_last_train_ts(symbol: str) -> datetime | None:
    """Lee la fecha del último entrenamiento (DB primero, JSON como fallback)."""
    from data.database import get_last_train
    ts_str = get_last_train(symbol)
    if not ts_str:
        # Fallback al JSON heredado (deploy en transición que aún lo tenga en disco)
        try:
            data = json.loads(_RETRAIN_LOG.read_text()) if _RETRAIN_LOG.exists() else {}
            ts_str = data.get(symbol)
        except Exception:
            ts_str = None
    try:
        return datetime.fromisoformat(ts_str) if ts_str else None
    except Exception:
        return None


def _set_last_train_ts(symbol: str) -> None:
    """Guarda la fecha actual como último entrenamiento del símbolo (en la DB)."""
    from data.database import set_last_train
    set_last_train(symbol, datetime.now(timezone.utc).isoformat())


def _should_retrain(symbol: str) -> bool:
    """True si han pasado ML_PARAMS['retrain_every'] días desde el último entreno."""
    last = _get_last_train_ts(symbol)
    if last is None:
        return True
    days = ML_PARAMS.get("retrain_every", 7)
    return (datetime.now(timezone.utc) - last) >= timedelta(days=days)


# ---------------------------------------------------------------------------
# Helpers de datos
# ---------------------------------------------------------------------------

def _load_data(symbol: str):
    """Descarga o carga datos OHLCV para el símbolo dado."""
    from data.fetchers.binance_fetcher import fetch_multi_timeframe
    from data.fetchers.gold_fetcher    import fetch_yfinance_asset, fetch_macro
    from data.database                 import init_db, save_ohlcv, load_ohlcv

    init_db()

    cfg    = ASSETS.get(symbol, {})
    source = cfg.get("source", "binance")

    logger.info(f"Cargando datos para {symbol}...")

    if source == "yfinance":
        # Activos no-crypto (Gold, Silver, S&P 500): datos diarios para ambos timeframes.
        # yfinance no tiene 4H confiable; en daily los setups técnicos funcionan mejor
        # para commodities e índices.
        yf_days  = cfg.get("days_history", 2500)
        df_daily = fetch_yfinance_asset(symbol, days=yf_days)
        df_4h    = df_daily.copy() if not df_daily.empty else None
        # Macro completo: cubre el mismo histórico que los precios para evitar
        # que las features macro queden 89-91% NaN y se descarten en el entrenamiento.
        macro_df = fetch_macro(days=yf_days)
    elif source == "twelvedata":
        # Pares forex (EUR/USD, GBP/USD, USD/JPY): velas 4H reales vía Twelve Data.
        # La barra diaria se deriva resampling del 4H (mitad de créditos API).
        # 5000 barras × 4H ≈ 3.2 años de historia forex (suficiente para ML).
        from data.fetchers.twelvedata_fetcher import fetch_twelvedata_ohlcv
        td_days = cfg.get("days_history", 1825)
        df_4h = fetch_twelvedata_ohlcv(symbol, interval="4h", outputsize=5000)
        if df_4h is not None and not df_4h.empty:
            df_daily = (df_4h
                        .resample("1D")
                        .agg({"open": "first", "high": "max",
                              "low": "min", "close": "last", "volume": "sum"})
                        .dropna())
        else:
            df_daily = None
        macro_df = fetch_macro(days=td_days)
    else:
        crypto_days = cfg.get("days_history", DAYS_HISTORY)
        data     = fetch_multi_timeframe(symbol, ["4h", "1d"], days=crypto_days)
        df_4h    = data.get("4h")
        df_daily = data.get("1d")
        # Macro para crypto: solo si el activo lo tiene habilitado.
        # BTC deshabilitado (use_macro=False): la macro añade ruido en 4H y bajó
        # la precisión de 0.193 a 0.128. ETH mantiene macro (mejoró 0.269→0.306).
        use_macro = cfg.get("use_macro", True)
        macro_df = fetch_macro(days=crypto_days) if use_macro else None

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
    # Para activos diarios (Gold, Silver, SP500) buscamos en un radio de 3 daily bars (72h).
    # Para crypto 4H buscamos en 6 × 4H bars = 24h.
    source = ASSETS.get(symbol, {}).get("source", "binance")
    n_bars = 18 if source != "binance" else 6  # 72h para diarios, 24h para 4H
    best = get_best_setup(df_scored, n_last_bars=n_bars)
    if best is None:
        window_desc = "3 días" if source != "binance" else "6 barras 4H (24h)"
        logger.info(f"[{symbol}] Sin setup de alta confianza en los últimos {window_desc}.")
        return

    ml_score = float(best.get("ml_score", best.get("raw_score", 0)))
    portfolio = PortfolioRiskManager()
    trade = build_trade_setup(best, portfolio.current_capital, ml_score)

    if trade is None:
        logger.info(f"[{symbol}] Setup detectado pero no pasa validaciones de riesgo.")
        return

    # Evitar re-emitir la misma señal en cada ciclo. Un setup persistente se
    # vuelve a detectar con el mismo entry_price; si ya existe una señal
    # equivalente (pendiente o reciente) se omite por completo: no se imprime,
    # no se guarda y no se notifica.
    from data.outcome_tracker import is_duplicate_signal, save_signal_from_trade
    if is_duplicate_signal(trade.symbol, trade.direction, trade.entry_price):
        logger.info(
            f"[{symbol}] Señal {trade.direction.upper()} @ {trade.entry_price:.2f} "
            "ya registrada — se omite para evitar duplicados."
        )
        return

    # Sentimiento
    base_sym = symbol.replace("USDT", "")
    sentiment = get_sentiment_summary(base_sym)

    _print_signal(trade, sentiment)

    # Persistir señal en la base de datos
    signal_id = save_signal_from_trade(trade, sentiment)

    # Solo notificar si es una señal nueva (no un duplicado ya registrado)
    if notify and signal_id > 0:
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
        symbol         = symbol,
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


def cmd_auto_retrain(symbol: str, notify: bool = False) -> bool:
    """
    Reentrena el modelo del símbolo si han pasado retrain_every días.
    Retorna True si se reentrenó efectivamente.
    """
    if not _should_retrain(symbol):
        last = _get_last_train_ts(symbol)
        days_ago = (datetime.now(timezone.utc) - last).days if last else "?"
        logger.info(
            f"[{symbol}] Reentrenamiento no necesario — "
            f"último hace {days_ago} día(s), umbral={ML_PARAMS['retrain_every']}d"
        )
        return False

    logger.info(f"[{symbol}] Iniciando reentrenamiento automático...")
    try:
        from signals.detector import detect_all_setups
        from ml.features      import build_training_dataset
        from ml.trainer       import train_model

        df_4h, df_daily, macro_df = _load_data(symbol)
        if df_4h is None:
            logger.error(f"[{symbol}] Sin datos para reentrenar.")
            return False

        df_setups    = detect_all_setups(df_4h, df_daily, symbol)
        forward_bars = ASSETS.get(symbol, {}).get("forward_bars", 12)

        funding_series = None
        if ASSETS.get(symbol, {}).get("source") == "binance":
            from data.fetchers.binance_fetcher import fetch_funding_rates
            days_hist = ASSETS.get(symbol, {}).get("days_history", 730)
            funding_series = fetch_funding_rates(symbol, days=days_hist)

        dataset = build_training_dataset(
            df_setups=df_setups, df_ohlcv=df_4h,
            forward_bars=forward_bars, macro_df=macro_df,
            funding_series=funding_series,
            symbol=symbol,
        )
        if dataset.empty or len(dataset) < ML_PARAMS.get("min_train_samples", 50):
            logger.warning(f"[{symbol}] Dataset insuficiente para reentrenar.")
            return False

        model, _, metrics = train_model(dataset, symbol=symbol, save=True)
        if model is None:
            logger.error(f"[{symbol}] Entrenamiento fallido.")
            return False

        metrics["n_samples"] = len(dataset)
        metrics["threshold"] = ML_PARAMS["prob_threshold"]
        _set_last_train_ts(symbol)

        logger.info(
            f"[{symbol}] Reentrenamiento OK | "
            f"AUC={metrics.get('auc',0):.3f} | "
            f"Precision={metrics.get('precision',0):.3f} | "
            f"Folds={metrics.get('n_folds',0)}"
        )

        if notify:
            from notifications.telegram import send_retrain_alert
            send_retrain_alert(symbol, metrics)

        return True

    except Exception as exc:
        logger.error(f"[{symbol}] Error durante reentrenamiento: {exc}")
        if notify:
            from notifications.telegram import send_error_alert
            send_error_alert(symbol, f"Reentrenamiento fallido: {exc}")
        return False


def cmd_schedule(interval_minutes: int = 60, retrain_hour: str = "03:00"):
    """
    Modo daemon: genera señales en un intervalo fijo.
    Envía notificaciones por Telegram si está configurado.
    Reentrena los modelos automáticamente según ML_PARAMS['retrain_every'].
    """
    import time
    import schedule as sched
    from notifications.telegram  import test_connection, send_daily_summary
    from config.settings         import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

    notify = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    if notify:
        test_connection()

    logger.info(
        f"Iniciando modo daemon — señales cada {interval_minutes} min | "
        f"Reentrenamiento diario a las {retrain_hour} UTC | "
        f"Telegram: {'ON' if notify else 'OFF'}. "
        "Ctrl+C para detener."
    )

    def _run_all_signals():
        import gc
        # Resolver resultados pendientes antes de generar nuevas señales
        try:
            from data.outcome_tracker import resolve_open_signals
            resolved = resolve_open_signals()
            if resolved:
                logger.info(f"Outcome tracker: {resolved} señal(es) resuelta(s).")
        except Exception as exc:
            logger.warning(f"Outcome tracker error: {exc}")

        for symbol in ASSETS:
            try:
                cmd_signal(symbol, notify=notify)
            except Exception as exc:
                logger.error(f"Error generando señal para {symbol}: {exc}")
            finally:
                # Cada ciclo horario carga datos de los 8 activos; liberar entre
                # cada uno evita que la RAM crezca de forma sostenida.
                gc.collect()

    def _run_nightly():
        """Actualiza datos y reentrena si es necesario (se ejecuta 1× al día)."""
        import gc
        logger.info("Tarea nocturna: actualizando datos y verificando reentrenamiento...")
        portfolio = None
        for symbol in ASSETS:
            try:
                _load_data(symbol)
                cmd_auto_retrain(symbol, notify=notify)
            except Exception as exc:
                logger.error(f"Error en tarea nocturna para {symbol}: {exc}")
            finally:
                # Liberar memoria entre activos: con 8 modelos en un solo Worker,
                # acumular el dataset + folds de cada uno provoca el OOM en Render.
                gc.collect()

        # Resumen diario por Telegram
        if notify:
            from risk.position_sizer import PortfolioRiskManager
            try:
                portfolio = PortfolioRiskManager()
                send_daily_summary(portfolio.summary())
                logger.info("Resumen diario enviado a Telegram.")
            except Exception as exc:
                logger.error(f"Error enviando resumen diario a Telegram: {exc}")

    # Señales recurrentes
    sched.every(interval_minutes).minutes.do(_run_all_signals)

    # Tarea nocturna a hora fija (UTC)
    sched.every().day.at(retrain_hour).do(_run_nightly)

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
  python main.py retrain  --symbol BTCUSDT          # fuerza reentrenamiento
  python main.py retrain  --all                     # reentrena todos los activos
  python main.py update
  python main.py schedule --interval 60
  python main.py schedule --interval 60 --retrain-hour 03:00
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

    # retrain
    pr = sub.add_parser("retrain", help="Forzar reentrenamiento del modelo ML")
    pr_grp = pr.add_mutually_exclusive_group(required=True)
    pr_grp.add_argument("--symbol", choices=list(ASSETS.keys()),
                        help="Símbolo a reentrenar")
    pr_grp.add_argument("--all", action="store_true",
                        help="Reentrenar todos los activos")
    pr.add_argument("--notify", action="store_true",
                    help="Enviar resultado a Telegram")

    # schedule (daemon)
    psc = sub.add_parser("schedule", help="Modo daemon con señales periódicas")
    psc.add_argument("--interval", default=60, type=int,
                     help="Intervalo en minutos (default: 60)")
    psc.add_argument("--retrain-hour", default="03:00", dest="retrain_hour",
                     help="Hora UTC para la tarea nocturna de reentrenamiento (HH:MM, default: 03:00)")

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

    elif args.command == "retrain":
        import gc
        symbols = list(ASSETS.keys()) if args.all else [args.symbol]
        from data.database import set_last_train
        for sym in symbols:
            # Forzar reentrenamiento: fecha muy antigua → _should_retrain = True
            try:
                set_last_train(sym, "1970-01-01T00:00:00+00:00")
            except Exception:
                pass
            try:
                cmd_auto_retrain(sym, notify=args.notify)
            finally:
                # Liberar memoria entre activos: este 'retrain --all' corre en
                # cada arranque del Worker (ver render.yaml) y reentrena los 8
                # modelos seguidos — es el pico de RAM que disparaba el OOM.
                gc.collect()

    elif args.command == "update":
        cmd_update()

    elif args.command == "schedule":
        cmd_schedule(args.interval, args.retrain_hour)

    elif args.command == "telegram":
        from notifications.telegram import test_connection
        test_connection()


if __name__ == "__main__":
    main()
