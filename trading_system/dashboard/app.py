"""
Dashboard v5.0 — Sistema de Trading ML
Flujo: Setup Detection → ML Scoring → Risk Management → Señal Accionable
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timezone
from loguru import logger

from config.settings import ASSETS, RISK, ML_PARAMS, SENTIMENT as SENT_CFG, DAYS_HISTORY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from data.database        import init_db, load_signals
from data.outcome_tracker import save_signal_from_trade, resolve_open_signals, get_outcome_stats
from notifications.telegram import send_signal_alert, test_connection as tg_test
from data.fetchers   import fetch_ohlcv, fetch_multi_timeframe, fetch_gold, fetch_macro, fetch_funding_rates
from signals.detector import detect_all_setups, get_current_setup
from ml.features      import build_training_dataset, compute_base_features
from ml.trainer       import train_model, load_model, get_model_importance
from ml.predictor     import score_setups
from risk.position_sizer import build_trade_setup, PortfolioRiskManager
from backtest.engine  import run_backtest
from backtest.metrics import format_metrics_table
from sentiment.cryptopanic import get_sentiment_summary, should_veto_signal

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Trading ML v5.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
.signal-long  { background:#d4edda; padding:16px; border-radius:8px; border-left:5px solid #28a745; }
.signal-short { background:#f8d7da; padding:16px; border-radius:8px; border-left:5px solid #dc3545; }
.signal-none  { background:#fff3cd; padding:16px; border-radius:8px; border-left:5px solid #ffc107; }
.metric-box   { background:#f8f9fa; padding:12px; border-radius:6px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# DB init
init_db()

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=40)
    st.title("Trading ML v5.0")
    st.markdown("---")

    asset_key = st.selectbox(
        "Activo",
        list(ASSETS.keys()),
        format_func=lambda k: f"{ASSETS[k]['label']} — {k}"
    )
    asset_cfg = ASSETS[asset_key]
    symbol    = asset_key

    st.markdown("---")
    st.subheader("Parámetros")

    capital = st.number_input(
        "Capital (USD)", min_value=100, max_value=1_000_000,
        value=int(RISK["capital_inicial"]), step=500
    )
    risk_pct = st.slider(
        "Riesgo por trade (%)", 0.5, 3.0,
        float(RISK["risk_per_trade"] * 100), 0.1
    ) / 100

    _default_threshold = float(asset_cfg.get("ml_threshold", ML_PARAMS["prob_threshold"]))
    ml_threshold = st.slider(
        "Umbral ML", 0.50, 0.80,
        _default_threshold, 0.01
    )

    st.markdown("---")
    st.subheader("Opciones")
    use_sentiment   = st.checkbox("Filtro de sentimiento", value=True)
    run_backtest_cb = st.checkbox("Ejecutar backtest", value=False)
    backtest_days   = st.slider("Días de backtest", 90, 730, 365) if run_backtest_cb else 365

    st.markdown("---")
    _tg_ready = bool(TELEGRAM_TOKEN and TELEGRAM_CHAT_ID)
    tg_notify = st.checkbox(
        "📲 Notificar por Telegram",
        value=_tg_ready,
        disabled=not _tg_ready,
        help="Envía la señal a tu Telegram si hay resultado. Requiere TOKEN y CHAT_ID en .env",
    )
    if not _tg_ready:
        st.caption("Telegram no configurado. Añade TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID al .env")
    else:
        if st.button("🔔 Test Telegram", help="Envía mensaje de prueba al bot"):
            ok = tg_test()
            if ok:
                st.success("Telegram: conexión OK")
            else:
                st.error("Telegram: fallo — revisa el .env")

    st.markdown("---")
    run_btn = st.button("🔄 Generar señal", width="stretch", type="primary")
    st.caption(f"v5.0 | Setup→ML→Risk | {datetime.now().strftime('%H:%M')}")

# =============================================================================
# TABS PRINCIPALES
# =============================================================================
tab_signal, tab_backtest, tab_model, tab_history = st.tabs([
    "📡 Señal Actual",
    "📊 Backtest",
    "🧠 Modelo ML",
    "📜 Historial",
])

# =============================================================================
# TAB 1: SEÑAL ACTUAL
# =============================================================================
with tab_signal:
    st.header(f"Señal en Tiempo Real — {asset_cfg['label']}")

    if not run_btn:
        st.info("Presiona **Generar señal** en el sidebar para ejecutar el análisis.")

        # Mostrar última señal guardada si existe
        hist_df = load_signals(symbol=symbol, limit=1)
        if not hist_df.empty:
            st.markdown("**Última señal guardada:**")
            last = hist_df.iloc[0]
            _dir = last.get("direction", "")
            if _dir == "long":
                st.success(f"LONG | ML: {last.get('ml_score',0):.2f} | "
                           f"Entry: ${last.get('entry_price',0):,.2f} | "
                           f"SL: ${last.get('stop_loss',0):,.2f} | "
                           f"TP: ${last.get('take_profit',0):,.2f}")
            elif _dir == "short":
                st.error(f"SHORT | ML: {last.get('ml_score',0):.2f} | "
                         f"Entry: ${last.get('entry_price',0):,.2f} | "
                         f"SL: ${last.get('stop_loss',0):,.2f} | "
                         f"TP: ${last.get('take_profit',0):,.2f}")

    else:
        # ----------------------------------------------------------------
        # Ejecutar análisis
        # ----------------------------------------------------------------
        with st.spinner("Descargando datos..."):
            gold_days = asset_cfg.get("days_history", 730)
            if asset_cfg["source"] == "binance":
                crypto_days = asset_cfg.get("days_history", DAYS_HISTORY)
                data = fetch_multi_timeframe(symbol, ["4h", "1d"], days=crypto_days)
                df_4h    = data.get("4h",  pd.DataFrame())
                df_daily = data.get("1d",  pd.DataFrame())
                funding  = fetch_funding_rates(symbol, days=90)
            else:
                df_daily = fetch_gold(timeframe="1d", days=gold_days)
                df_4h    = df_daily.copy() if not df_daily.empty else pd.DataFrame()
                funding  = pd.Series(dtype=float)

            macro_df = fetch_macro(days=min(gold_days, 730))

        if df_4h.empty:
            st.error("No se pudieron obtener datos. Revisa tu conexión o las credenciales.")
        else:
            # ----------------------------------------------------------------
            # Detectar setups
            # ----------------------------------------------------------------
            with st.spinner("Detectando setups técnicos..."):
                df_setups = detect_all_setups(df_4h, df_daily, symbol=symbol)

            # ----------------------------------------------------------------
            # Entrenamiento / carga de modelo ML
            # ----------------------------------------------------------------
            with st.spinner("Preparando modelo ML..."):
                model, feat_cols, medians = load_model(symbol)

                if model is None:
                    st.info("Entrenando modelo por primera vez (puede tardar 1-2 min)...")
                    _fwd_bars = asset_cfg.get("forward_bars", 12)
                    dataset = build_training_dataset(
                        df_setups, df_4h,
                        forward_bars=_fwd_bars,
                        macro_df=macro_df,
                    )
                    if not dataset.empty and len(dataset) >= ML_PARAMS["min_train_samples"]:
                        model, feat_cols, wf = train_model(dataset, symbol=symbol)
                        if wf:
                            st.success(
                                f"Modelo entrenado | AUC WF: {wf.get('auc',0):.3f} | "
                                f"Precision: {wf.get('precision',0):.3f}"
                            )
                    else:
                        st.warning(
                            f"Pocos setups históricos ({len(dataset) if not dataset.empty else 0}). "
                            "El modelo usará el score técnico como proxy."
                        )

            # ----------------------------------------------------------------
            # ML Scoring de setups
            # ----------------------------------------------------------------
            with st.spinner("Evaluando setups con ML..."):
                if not df_setups.empty:
                    df_setups = score_setups(df_setups, df_4h, symbol, macro_df, funding)

            # ----------------------------------------------------------------
            # Sentimiento
            # ----------------------------------------------------------------
            sentiment = {"score": 0.0, "label": "N/A", "color": "gray", "n_news": 0}
            if use_sentiment and asset_cfg["source"] == "binance":
                with st.spinner("Analizando sentimiento..."):
                    sentiment = get_sentiment_summary(asset_cfg["label"])

            # ----------------------------------------------------------------
            # Obtener mejor setup reciente
            # ----------------------------------------------------------------
            best = None
            # Para activos con velas diarias (Gold) usamos ventana de 3 días; crypto 4H usa 12h
            signal_lookback_h = 72 if asset_cfg["source"] != "binance" else 12
            if not df_setups.empty:
                recent = df_setups[
                    df_setups["ts"] >= df_setups["ts"].max() - pd.Timedelta(hours=signal_lookback_h)
                ]
                if "ml_score" in recent.columns:
                    approved = recent[recent["ml_score"] >= ml_threshold]
                else:
                    approved = recent
                if not approved.empty:
                    best = approved.loc[approved["ml_score"].idxmax()]

            # ----------------------------------------------------------------
            # Construir trade setup con gestión de riesgo
            # ----------------------------------------------------------------
            trade = None
            if best is not None:
                trade = build_trade_setup(
                    best, capital, float(best["ml_score"]),
                    sentiment_score=sentiment["score"]
                )
                # Veto por sentimiento
                if trade and use_sentiment and asset_cfg["source"] == "binance":
                    vetoed, veto_reason = should_veto_signal(trade.direction, sentiment["score"])
                    if vetoed:
                        st.warning(f"⚠️ Señal vetada por sentimiento: {veto_reason}")
                        trade = None

            # ================================================================
            # DISPLAY PRINCIPAL
            # ================================================================
            col_signal, col_risk = st.columns([3, 2])

            with col_signal:
                if trade is not None:
                    if trade.direction == "long":
                        st.markdown(f"""
                        <div class="signal-long">
                        <h2>🟢 LONG — {asset_cfg['label']}</h2>
                        <p>Setup: <b>{trade.setup_type.upper()}</b> |
                           Timeframe: 4H |
                           ML Score: <b>{trade.ml_score:.0%}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="signal-short">
                        <h2>🔴 SHORT — {asset_cfg['label']}</h2>
                        <p>Setup: <b>{trade.setup_type.upper()}</b> |
                           Timeframe: 4H |
                           ML Score: <b>{trade.ml_score:.0%}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    window_label = f"{signal_lookback_h} horas" if signal_lookback_h < 48 else f"{signal_lookback_h // 24} días"
                    st.markdown(f"""
                    <div class="signal-none">
                    <h2>⚪ SIN SEÑAL</h2>
                    <p>No hay setups de alta probabilidad en las últimas {window_label}.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Métricas de la señal
                if trade is not None:
                    st.markdown("---")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Entrada",      f"${trade.entry_price:,.2f}")
                    c2.metric("Stop Loss",    f"${trade.stop_loss:,.2f}",
                              delta=f"-{trade.stop_distance_pct*100:.1f}%",
                              delta_color="inverse")
                    c3.metric("Take Profit",  f"${trade.take_profit:,.2f}",
                              delta=f"+{trade.target_distance_pct*100:.1f}%")
                    c4.metric("R:R",          f"{trade.rr_ratio:.1f}:1")

                    c5, c6, c7, c8 = st.columns(4)
                    c5.metric("Tamaño USD",    f"${trade.position_size:,.0f}")
                    c6.metric("Riesgo USD",    f"${trade.risk_usd:,.0f}")
                    c7.metric("Reward USD",    f"${trade.reward_usd:,.0f}")
                    c8.metric("ATR (14)",      f"{trade.atr_pct*100:.2f}%")

                    col_btns = st.columns(2)
                    with col_btns[0]:
                        if st.button("💾 Registrar señal en DB", type="secondary"):
                            sig_id = save_signal_from_trade(trade, sentiment)
                            if sig_id > 0:
                                st.success(f"Señal guardada (ID #{sig_id}). El resultado se resolverá automáticamente.")
                            else:
                                st.error("Error al guardar la señal.")
                    with col_btns[1]:
                        if tg_notify:
                            if st.button("📲 Enviar a Telegram", type="secondary"):
                                ok = send_signal_alert(trade, sentiment)
                                if ok:
                                    st.success("Notificación enviada a Telegram.")
                                else:
                                    st.error("No se pudo enviar. Verifica TOKEN y CHAT_ID en .env")

            with col_risk:
                st.subheader("Gestión de Riesgo")
                st.info(
                    f"**Capital:** ${capital:,.0f}\n\n"
                    f"**Riesgo/trade:** {risk_pct*100:.1f}%\n\n"
                    f"**Max posiciones:** {RISK['max_positions']}\n\n"
                    f"**Stop:** {RISK['atr_stop_mult']}× ATR\n\n"
                    f"**Target:** {RISK['atr_target_mult']}× ATR"
                )

                if use_sentiment:
                    st.subheader("Sentimiento")
                    fg_val  = sentiment.get("fg_value", 0)
                    fg_cls  = sentiment.get("fg_classification", "N/A")
                    score   = sentiment["score"]
                    label   = sentiment["label"]
                    trend7  = sentiment.get("trend_7d", 0.0)

                    color_icon = {
                        "green":  "🟢", "red": "🔴",
                        "gray":   "⚪", "orange": "🟡",
                    }.get(sentiment["color"], "⚪")

                    if fg_val > 0:
                        # Barra de progreso del F&G Index
                        st.markdown(f"{color_icon} **{label}** — Fear & Greed: **{fg_val}/100**")
                        st.progress(fg_val / 100)
                        delta_str = f"{trend7:+.0f} pts vs hace 7 días" if trend7 != 0 else "sin cambio en 7 días"
                        st.caption(
                            f"Clasificación: *{fg_cls}* | "
                            f"Score contrarian: {score:+.2f} | {delta_str}"
                        )
                    else:
                        st.markdown(f"{color_icon} **{label}** (N/A para este activo)")
                        st.caption("El Fear & Greed Index es específico de crypto.")

            # ----------------------------------------------------------------
            # Gráfico de precio
            # ----------------------------------------------------------------
            st.markdown("---")
            if not df_4h.empty and len(df_4h) > 50:
                st.subheader(f"📈 {asset_cfg['label']} — Últimas 120 barras 4H")
                last_n = df_4h.iloc[-120:].copy()

                # EMAs
                last_n["ema_21"] = last_n["close"].ewm(span=21).mean()
                last_n["ema_55"] = last_n["close"].ewm(span=55).mean()

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.75, 0.25],
                )
                # Velas
                fig.add_trace(go.Candlestick(
                    x=last_n.index,
                    open=last_n["open"], high=last_n["high"],
                    low=last_n["low"],   close=last_n["close"],
                    name="OHLC",
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=last_n.index, y=last_n["ema_21"],
                    name="EMA 21", line=dict(color="orange", width=1.5)
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=last_n.index, y=last_n["ema_55"],
                    name="EMA 55", line=dict(color="royalblue", width=1.5, dash="dash")
                ), row=1, col=1)

                # Marcadores de setups
                if not df_setups.empty:
                    recent_setups = df_setups[df_setups["ts"] >= last_n.index[0]]
                    longs  = recent_setups[recent_setups["direction"] == "long"]
                    shorts = recent_setups[recent_setups["direction"] == "short"]
                    if not longs.empty:
                        fig.add_trace(go.Scatter(
                            x=longs["ts"], y=longs["close"] * 0.998,
                            mode="markers",
                            marker=dict(symbol="triangle-up", size=10, color="lime"),
                            name="Setup LONG",
                        ), row=1, col=1)
                    if not shorts.empty:
                        fig.add_trace(go.Scatter(
                            x=shorts["ts"], y=shorts["close"] * 1.002,
                            mode="markers",
                            marker=dict(symbol="triangle-down", size=10, color="red"),
                            name="Setup SHORT",
                        ), row=1, col=1)

                # Si hay trade activo, mostrar niveles
                if trade is not None:
                    fig.add_hline(
                        y=trade.entry_price, line_color="white",
                        line_dash="dash", line_width=1,
                        annotation_text="Entry", row=1, col=1
                    )
                    fig.add_hline(
                        y=trade.stop_loss, line_color="red",
                        line_dash="dot", line_width=1.5,
                        annotation_text="Stop Loss", row=1, col=1
                    )
                    fig.add_hline(
                        y=trade.take_profit, line_color="lime",
                        line_dash="dot", line_width=1.5,
                        annotation_text="Take Profit", row=1, col=1
                    )

                # Volumen
                colors = ["#26a69a" if last_n["close"].iloc[i] >= last_n["open"].iloc[i]
                          else "#ef5350" for i in range(len(last_n))]
                fig.add_trace(go.Bar(
                    x=last_n.index, y=last_n["volume"],
                    name="Volumen", marker_color=colors, opacity=0.6,
                ), row=2, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=550,
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, width="stretch")

            # ----------------------------------------------------------------
            # Todos los setups detectados
            # ----------------------------------------------------------------
            if not df_setups.empty:
                with st.expander(f"🔍 Todos los setups detectados ({len(df_setups)})"):
                    show = df_setups[["ts","setup_type","direction","raw_score",
                                       "ml_score","close","atr","rr_ratio"]].tail(30)
                    st.dataframe(show.sort_values("ts", ascending=False), use_container_width=True)


# =============================================================================
# TAB 2: BACKTEST
# =============================================================================
with tab_backtest:
    st.header("Backtest del Sistema")

    if not run_backtest_cb:
        st.info("Activa **Ejecutar backtest** en el sidebar para ver los resultados.")
    else:
        st.caption(
            "Simula el sistema completo sobre datos históricos. "
            "No hay look-ahead bias: el modelo se entrena solo con datos pasados."
        )

        if st.button("▶️ Ejecutar Backtest", type="primary"):
            with st.spinner("Ejecutando backtest (puede tardar 2-5 minutos)..."):
                bt_data = fetch_multi_timeframe(
                    symbol, ["4h", "1d"],
                    days=backtest_days + 30
                ) if asset_cfg["source"] == "binance" else None

                if bt_data:
                    df_4h_bt  = bt_data.get("4h", pd.DataFrame())
                    df_1d_bt  = bt_data.get("1d", pd.DataFrame())
                else:
                    df_4h_bt  = fetch_gold("4h", backtest_days + 30)
                    df_1d_bt  = fetch_gold("1d", backtest_days + 30)

                bt_macro = fetch_macro(days=backtest_days + 30)

                results = run_backtest(
                    df_4h=df_4h_bt,
                    df_daily=df_1d_bt,
                    symbol=symbol,
                    macro_df=bt_macro,
                    initial_capital=capital,
                )

            if not results:
                st.error("El backtest no produjo resultados.")
            else:
                st.success("Backtest completado.")
                # Métricas en tabla
                st.subheader("Métricas de Performance")
                metrics_table = format_metrics_table(results)
                st.dataframe(metrics_table, use_container_width=True, hide_index=True)

                # Equity curve
                eq = results.get("equity_curve")
                if eq is not None and len(eq) > 1:
                    st.subheader("Equity Curve")
                    fig_eq = px.line(
                        x=eq.index, y=eq.values,
                        labels={"x": "Fecha", "y": "Capital (USD)"},
                        template="plotly_dark",
                    )
                    fig_eq.add_hline(y=capital, line_dash="dash", line_color="gray",
                                      annotation_text="Capital inicial")
                    st.plotly_chart(fig_eq, width="stretch")

                # Trades
                trades_df = results.get("trades_df")
                if trades_df is not None and not trades_df.empty:
                    st.subheader(f"Operaciones ({len(trades_df)})")
                    cols = [c for c in ["entry_ts","direction","setup_type",
                                         "entry_price","exit_price","pnl_usd",
                                         "pnl_pct","outcome","rr_ratio","ml_score"]
                            if c in trades_df.columns]
                    st.dataframe(
                        trades_df[cols].sort_values("entry_ts", ascending=False)
                        .head(100),
                        use_container_width=True,
                    )

                # Por setup type
                by_type = results.get("by_setup_type", {})
                if by_type:
                    st.subheader("Rendimiento por tipo de setup")
                    bt_df = pd.DataFrame(by_type).T.reset_index()
                    bt_df.columns = ["Setup", "Trades", "Win Rate", "PnL (USD)"]
                    bt_df["Win Rate"] = bt_df["Win Rate"].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(bt_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 3: MODELO ML
# =============================================================================
with tab_model:
    st.header("Análisis del Modelo ML")

    model_loaded, feat_cols, medians = load_model(symbol)
    if model_loaded is None:
        st.info("Aún no hay modelo entrenado para este activo. Genera una señal primero.")
    else:
        # Feature importance
        imp = get_model_importance(symbol, top_n=20)
        if not imp.empty:
            st.subheader("Top 20 Features (Importancia)")
            fig_imp = px.bar(
                x=imp.values, y=imp.index,
                orientation="h",
                labels={"x": "Importancia", "y": "Feature"},
                template="plotly_dark",
                color=imp.values,
                color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig_imp, width="stretch")

        st.subheader("Parámetros del Modelo")
        params_data = {
            "Estimadores":    ML_PARAMS["n_estimators"],
            "Learning Rate":  ML_PARAMS["learning_rate"],
            "Max Leaves":     ML_PARAMS["num_leaves"],
            "Max Depth":      ML_PARAMS["max_depth"],
            "Min Samples":    ML_PARAMS["min_child_samples"],
            "Subsample":      ML_PARAMS["subsample"],
            "Umbral señal":   ML_PARAMS["prob_threshold"],
        }
        st.dataframe(
            pd.DataFrame(list(params_data.items()), columns=["Parámetro", "Valor"]),
            use_container_width=True, hide_index=True,
        )

        st.subheader("Guía de Interpretación")
        st.markdown("""
        | ML Score | Interpretación | Acción |
        |----------|---------------|--------|
        | > 0.70   | Setup de muy alta calidad | Operar con tamaño completo |
        | 0.58–0.70 | Setup sólido | Operar con tamaño normal |
        | 0.50–0.58 | Setup marginal | Reducir tamaño o esperar |
        | < 0.50   | Setup débil | No operar |
        """)


# =============================================================================
# TAB 4: HISTORIAL
# =============================================================================
with tab_history:
    st.header("Historial de Señales")

    col_refresh, col_resolve = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Resolver señales pendientes ahora"):
            with st.spinner("Consultando precios y resolviendo..."):
                n = resolve_open_signals()
            st.success(f"{n} señal(es) resuelta(s).") if n else st.info("No hay señales pendientes por resolver.")

    hist_df = load_signals(symbol=symbol, limit=200)
    if hist_df.empty:
        st.info("No hay señales guardadas para este activo aún. Genera una señal y pulsa 'Registrar señal en DB'.")
    else:
        stats = get_outcome_stats(symbol=symbol)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total señales",  stats.get("total", 0))
        c2.metric("Resueltas",      stats.get("resolved", 0))
        c3.metric("Pendientes",     stats.get("pending", 0))
        c4.metric("Win Rate real",
                  f"{stats['win_rate']*100:.1f}%" if stats.get("resolved", 0) > 0 else "—")
        c5.metric("PnL total",
                  f"${stats.get('total_pnl', 0):+,.2f}")

        st.dataframe(
            hist_df.sort_values("ts", ascending=False),
            use_container_width=True,
        )

        # Marcar resultados
        st.subheader("Actualizar resultado de señal")
        signal_ids = hist_df[hist_df["resultado_real"].isna()]["id"].tolist()
        if signal_ids:
            sel_id  = st.selectbox("ID de señal pendiente", signal_ids)
            outcome = st.radio("Resultado", ["win", "loss", "be"])
            pnl     = st.number_input("PnL (USD)", value=0.0, step=10.0)
            if st.button("Guardar resultado"):
                from data.database import get_connection
                with get_connection() as conn:
                    conn.execute(
                        "UPDATE signals SET resultado_real=?, pnl_usd=? WHERE id=?",
                        (outcome, pnl, sel_id)
                    )
                st.success("Resultado guardado.")
                st.rerun()
        else:
            st.caption("Todas las señales tienen resultado registrado.")
