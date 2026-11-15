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

from config.settings import ASSETS, RISK, ML_PARAMS, DAYS_HISTORY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, fmt_price
from data.database        import init_db, load_signals
from data.outcome_tracker import (save_signal_from_trade, resolve_open_signals, get_outcome_stats,
                                  resolve_signal_manually, MANUAL_OUTCOMES)
from notifications.telegram import send_signal_alert, test_connection as tg_test
from data.fetchers   import fetch_ohlcv, fetch_multi_timeframe, fetch_gold, fetch_yfinance_asset, fetch_macro, fetch_funding_rates
from signals.detector import detect_all_setups, get_current_setup
from ml.features      import build_training_dataset, compute_base_features
from ml.trainer       import train_model, load_model, get_model_importance
from ml.predictor     import score_setups, get_best_setup
from data.bars        import drop_unclosed_bars
from config.settings  import notify_enabled, MAX_SETUP_AGE_BARS, MAX_ENTRY_DIST_ATR
from risk.position_sizer import build_trade_setup, PortfolioRiskManager
# Fase 1: sin pestaña de backtest (backtest/engine.py da 0 operaciones en los
# diarios y no es el protocolo validado: usar `main.py validate`) y sin
# sentimiento (SENTIMENT['enabled']=False; el veto contrarian se descartó con
# datos: en BTC quitaría las mejores señales).

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

    # Capital por defecto = TRADING_CAPITAL (el mismo que usa el worker; revisión
    # de la Fase 1: antes arrancaba siempre en 10.000 y un envío desde aquí salía
    # con otro tamaño y otro % de riesgo que las señales del worker).
    from config.settings import TRADING_CAPITAL
    capital = st.number_input(
        "Capital (USD)", min_value=100, max_value=1_000_000,
        value=int(min(max(TRADING_CAPITAL, 100), 1_000_000)), step=500
    )
    risk_pct = st.slider(
        "Riesgo por trade (%)", 0.5, 3.0,
        float(RISK["risk_per_trade"] * 100), 0.1
    ) / 100

    _default_threshold = float(asset_cfg.get("ml_threshold", ML_PARAMS["prob_threshold"]))
    # Rango 0–1: los umbrales de oro (0.20) y SPX (0.12) quedan por debajo del
    # antiguo mínimo 0.50 (Streamlit movía el mínimo en silencio). Solo afecta a
    # la pestaña de señal del dashboard; el worker usa ml_threshold de settings.
    ml_threshold = st.slider(
        "Umbral del score", 0.0, 1.0,
        _default_threshold, 0.01
    )

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
tab_signal, tab_model, tab_history = st.tabs([
    "📡 Señal Actual",
    "🧠 Modelo ML",
    "📜 Historial",
])

# =============================================================================
# TAB 1: SEÑAL ACTUAL
# =============================================================================
with tab_signal:
    st.header(f"Señal en Tiempo Real — {asset_cfg['label']}")

    # Limpiar caché si el usuario cambia de activo en el selector
    if st.session_state.get("_sig_symbol") != symbol:
        for _k in ("_sig_trade", "_sig_sentiment", "_sig_df4h", "_sig_dfsetups", "_sig_lookback",
                   "_sig_meta", "_sig_price"):
            st.session_state.pop(_k, None)
        st.session_state["_sig_symbol"] = symbol

    # ---- Ejecutar análisis solo cuando se presiona "Generar señal" ----
    if run_btn:
        _trade     = None
        _df4h      = pd.DataFrame()
        _dfsetups  = pd.DataFrame()
        _sent      = None   # sentimiento apagado (Fase 1)
        _lookback  = 72 if asset_cfg["source"] != "binance" else 12

        with st.spinner("Descargando datos..."):
            yf_days = asset_cfg.get("days_history", 730)
            if asset_cfg["source"] == "binance":
                crypto_days = asset_cfg.get("days_history", DAYS_HISTORY)
                data = fetch_multi_timeframe(symbol, ["4h", "1d"], days=crypto_days)
                # Fase 0: solo velas CERRADAS, igual que el worker y el validador.
                _df4h    = drop_unclosed_bars(data.get("4h",  pd.DataFrame()), "4h")
                df_daily = drop_unclosed_bars(data.get("1d",  pd.DataFrame()), "1d")
                funding  = fetch_funding_rates(symbol, days=90)
            else:
                df_daily = drop_unclosed_bars(fetch_yfinance_asset(symbol, days=yf_days), "1d", session=True)
                _df4h    = df_daily.copy() if not df_daily.empty else pd.DataFrame()
                if not _df4h.empty:
                    _df4h.attrs = dict(df_daily.attrs)
                funding  = pd.Series(dtype=float)

            # Macro cubre el mismo período que los precios (sin cap)
            macro_df = fetch_macro(days=yf_days)

        if _df4h.empty:
            st.error("No se pudieron obtener datos. Revisa tu conexión o las credenciales.")
        else:
            with st.spinner("Detectando setups técnicos..."):
                _dfsetups = detect_all_setups(_df4h, df_daily, symbol=symbol)

            with st.spinner("Preparando modelo ML..."):
                model, feat_cols, medians, _ = load_model(symbol)
                if model is None:
                    st.info("Entrenando modelo por primera vez (puede tardar 1-2 min)...")
                    _fwd_bars = asset_cfg.get("forward_bars", 12)
                    dataset = build_training_dataset(
                        _dfsetups, _df4h,
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

            with st.spinner("Evaluando setups con ML..."):
                if not _dfsetups.empty:
                    _dfsetups = score_setups(_dfsetups, _df4h, symbol, macro_df, funding)

            # Fase 0: misma selección que el worker (main.cmd_signal): ventana
            # anclada a la última vela cerrada, frescura y sin setups ya emitidos.
            best = None
            _lookback = MAX_SETUP_AGE_BARS + 1
            _meta = {}
            if not _dfsetups.empty and "ml_score" in _dfsetups.columns:
                from data.outcome_tracker import emitted_setup_keys
                _dfsetups["ml_approved"] = _dfsetups["ml_score"] >= ml_threshold
                _min_ts = _df4h.index[max(0, len(_df4h) - 1 - MAX_SETUP_AGE_BARS)]
                best = get_best_setup(
                    _dfsetups,
                    current_price=float(_df4h["close"].iloc[-1]),
                    max_dist_atr=asset_cfg.get("max_entry_dist_atr", MAX_ENTRY_DIST_ATR),
                    min_setup_ts=_min_ts, max_setup_ts=_df4h.index[-1],
                    exclude_keys=emitted_setup_keys(symbol, _min_ts),
                )
                if best is not None:
                    _meta = {"setup_ts": pd.Timestamp(best["ts"]),
                             "data_source": _df4h.attrs.get("source", asset_cfg["source"]),
                             "last_bar_ts": _df4h.index[-1]}

            if best is not None:
                _trade = build_trade_setup(best, capital, float(best["ml_score"]))

        # Persistir en session_state para que los botones secundarios no borren la señal
        st.session_state["_sig_trade"]    = _trade
        st.session_state["_sig_sentiment"] = _sent
        st.session_state["_sig_df4h"]     = _df4h
        st.session_state["_sig_dfsetups"] = _dfsetups
        st.session_state["_sig_lookback"] = _lookback
        st.session_state["_sig_meta"]     = _meta if not _df4h.empty else {}
        st.session_state["_sig_price"]    = (float(_df4h["close"].iloc[-1])
                                             if not _df4h.empty else None)

    # =========================================================================
    # DISPLAY — siempre lee de session_state (persiste cuando se pulsan botones)
    # =========================================================================
    _trade    = st.session_state.get("_sig_trade")
    _sent     = st.session_state.get("_sig_sentiment")
    _df4h     = st.session_state.get("_sig_df4h",    pd.DataFrame())
    _dfsetups = st.session_state.get("_sig_dfsetups", pd.DataFrame())
    _lookback = st.session_state.get("_sig_lookback", 12)

    if "_sig_trade" not in st.session_state:
        # Primera visita — análisis nunca ejecutado
        st.info("Presiona **Generar señal** en el sidebar para ejecutar el análisis.")
        hist_df = load_signals(symbol=symbol, limit=1)
        if not hist_df.empty:
            st.markdown("**Última señal guardada:**")
            last = hist_df.iloc[0]
            _dir = last.get("direction", "")
            if _dir == "long":
                st.success(f"LONG | Score: {last.get('ml_score',0):.2f} | "
                           f"Entry: ${fmt_price(last.get('entry_price',0))} | "
                           f"SL: ${fmt_price(last.get('stop_loss',0))} | "
                           f"TP: ${fmt_price(last.get('take_profit',0))}")
            elif _dir == "short":
                st.error(f"SHORT | Score: {last.get('ml_score',0):.2f} | "
                         f"Entry: ${fmt_price(last.get('entry_price',0))} | "
                         f"SL: ${fmt_price(last.get('stop_loss',0))} | "
                         f"TP: ${fmt_price(last.get('take_profit',0))}")
    else:
        # Análisis ya ejecutado — mostrar resultado (y los botones de acción)
        col_signal, col_risk = st.columns([3, 2])

        with col_signal:
            if _trade is not None:
                if _trade.direction == "long":
                    st.markdown(f"""
                    <div class="signal-long">
                    <h2>🟢 LONG — {asset_cfg['label']}</h2>
                    <p>Setup: <b>{_trade.setup_type.upper()}</b> |
                       Timeframe: 4H |
                       Score del setup: <b>{_trade.ml_score:.2f}</b> (umbral {ml_threshold:.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-short">
                    <h2>🔴 SHORT — {asset_cfg['label']}</h2>
                    <p>Setup: <b>{_trade.setup_type.upper()}</b> |
                       Timeframe: 4H |
                       Score del setup: <b>{_trade.ml_score:.2f}</b> (umbral {ml_threshold:.2f})</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                window_label = f"{_lookback} horas" if _lookback < 48 else f"{_lookback // 24} días"
                st.markdown(f"""
                <div class="signal-none">
                <h2>⚪ SIN SEÑAL</h2>
                <p>No hay setups de alta probabilidad en las últimas {window_label}.</p>
                </div>
                """, unsafe_allow_html=True)

            if _trade is not None:
                st.markdown("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Entrada",      f"${fmt_price(_trade.entry_price)}")
                c2.metric("Stop Loss",    f"${fmt_price(_trade.stop_loss)}",
                          delta=f"-{_trade.stop_distance_pct*100:.1f}%",
                          delta_color="inverse")
                c3.metric("Take Profit",  f"${fmt_price(_trade.take_profit)}",
                          delta=f"+{_trade.target_distance_pct*100:.1f}%")
                c4.metric("R:R",          f"{_trade.rr_ratio:.1f}:1")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Tamaño USD",    f"${_trade.position_size:,.0f}")
                c6.metric("Riesgo USD",    f"${_trade.risk_usd:,.0f}")
                c7.metric("Reward USD",    f"${_trade.reward_usd:,.0f}")
                c8.metric("ATR (14)",      f"{_trade.atr_pct*100:.2f}%")

                col_btns = st.columns(2)
                with col_btns[0]:
                    if st.button("💾 Registrar señal en DB", type="secondary"):
                        sig_id = save_signal_from_trade(
                            _trade, _sent, **st.session_state.get("_sig_meta", {}))
                        if sig_id > 0:
                            st.success(f"Señal guardada (ID #{sig_id}). El resultado se resolverá automáticamente.")
                        else:
                            st.error("No se guardó: el setup ya estaba registrado o hubo un error.")
                with col_btns[1]:
                    if not notify_enabled(symbol):
                        st.caption("Activo en modo sombra: no se envía a Telegram.")
                    elif tg_notify:
                        if st.button("📲 Enviar a Telegram", type="secondary"):
                            _m = st.session_state.get("_sig_meta", {})
                            ok = send_signal_alert(
                                _trade, None, setup_ts=_m.get("setup_ts"),
                                current_price=st.session_state.get("_sig_price"),
                                threshold=ml_threshold, capital=capital)
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

        st.markdown("---")
        if not _df4h.empty and len(_df4h) > 50:
            st.subheader(f"📈 {asset_cfg['label']} — Últimas 120 barras 4H")
            last_n = _df4h.iloc[-120:].copy()
            last_n["ema_21"] = last_n["close"].ewm(span=21).mean()
            last_n["ema_55"] = last_n["close"].ewm(span=55).mean()

            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.75, 0.25],
            )
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

            if not _dfsetups.empty:
                recent_setups = _dfsetups[_dfsetups["ts"] >= last_n.index[0]]
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

            if _trade is not None:
                fig.add_hline(
                    y=_trade.entry_price, line_color="white",
                    line_dash="dash", line_width=1,
                    annotation_text="Entry", row=1, col=1
                )
                fig.add_hline(
                    y=_trade.stop_loss, line_color="red",
                    line_dash="dot", line_width=1.5,
                    annotation_text="Stop Loss", row=1, col=1
                )
                fig.add_hline(
                    y=_trade.take_profit, line_color="lime",
                    line_dash="dot", line_width=1.5,
                    annotation_text="Take Profit", row=1, col=1
                )

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

        if not _dfsetups.empty:
            with st.expander(f"🔍 Todos los setups detectados ({len(_dfsetups)})"):
                show = _dfsetups[["ts","setup_type","direction","raw_score",
                                   "ml_score","close","atr","rr_ratio"]].tail(30)
                st.dataframe(show.sort_values("ts", ascending=False), use_container_width=True)


# =============================================================================
# TAB 2: MODELO ML
# =============================================================================
with tab_model:
    st.header("Análisis del Modelo ML")

    model_loaded, feat_cols, medians, _ = load_model(symbol)
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
# TAB 3: HISTORIAL
# =============================================================================
with tab_history:
    st.header("Historial de Señales")

    col_refresh, col_resolve = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Resolver señales pendientes ahora"):
            with st.spinner("Consultando precios y resolviendo..."):
                n = resolve_open_signals()
            st.success(f"{n} señal(es) resuelta(s).") if n else st.info("No hay señales pendientes por resolver.")

    # Estadísticas en R (Fase 1, 1.3 / 1.9): solo señales LIVE (las que recibes
    # por Telegram) sin marca de calidad, sin límite de filas.
    _all_assets = st.checkbox("Estadísticas de todos los activos", value=False)
    stats = get_outcome_stats(symbol=None if _all_assets else symbol)
    _ambito = "todos los activos" if _all_assets else asset_cfg["label"]
    if not stats:
        st.info(f"Sin señales live registradas ({_ambito}).")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Señales live", stats["total"])
        c2.metric("Abiertas", stats["pending"])
        c3.metric("Ganadas (TP)", stats["wins"])
        c4.metric("Perdidas (SL)", stats["losses"])
        c5.metric("Cierre por tiempo", stats["expired"])
        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Break-even", stats["be"])
        c7.metric("% de aciertos",
                  f"{stats['win_rate']*100:.1f}%" if stats.get("win_rate") is not None else "—",
                  help="Ganadas (tocaron el TP) / resueltas (TP + SL + tiempo + break-even).")
        c8.metric("R medio por operación",
                  f"{stats['mean_r']:+.2f}R" if stats.get("mean_r") is not None else "—",
                  help="R = ganancia o pérdida medida en 'riesgos': −1R = tocó el SL; "
                       "+1,5R = ganó 1,5 veces lo arriesgado.")
        c9.metric("Intervalo 90 %",
                  (f"{stats['ci90_low']:+.2f} a {stats['ci90_high']:+.2f}R"
                   if stats.get("ci90_low") is not None else "—"),
                  help="Con estos resultados, el R medio real está probablemente (90 %) en "
                       "este rango. Si incluye el 0, aún no se puede saber si el sistema gana.")
        c10.metric("Suma de R", f"{stats['sum_r']:+.2f}R")
        _txt = (f"{_ambito}: {stats['resolved']} resueltas "
                f"({stats['auto']} por el sistema, {stats['manual']} a mano)")
        if stats["n_r"] < stats["resolved"]:
            _txt += f"; {stats['resolved'] - stats['n_r']} sin R conocido (no entran en el R medio)"
        if stats.get("other"):
            _txt += f"; {stats['other']} con otro resultado"
        st.caption(_txt + ". R sin costes del bróker y con la entrada al precio del mensaje.")
        if stats.get("ci90_low") is not None and stats["ci90_low"] <= 0 <= stats["ci90_high"]:
            st.caption("El intervalo incluye el 0: con tan pocas operaciones todavía no se puede "
                       "distinguir un sistema ganador de uno que no gana.")

    hist_df = load_signals(symbol=symbol, limit=200)
    if hist_df.empty:
        st.info("No hay señales guardadas para este activo aún. Genera una señal y pulsa 'Registrar señal en DB'.")
    else:
        st.dataframe(
            hist_df.sort_values("ts", ascending=False),
            use_container_width=True,
        )

        # Cerrar a mano una señal abierta (resolved_by='manual'; el tracker escribe 'auto').
        st.subheader("Cerrar una señal a mano")
        pend = hist_df[hist_df["resultado_real"].isna()]
        if not pend.empty:
            sel_id  = st.selectbox("ID de señal abierta", pend["id"].tolist())
            row     = pend[pend["id"] == sel_id].iloc[0]
            _labels = {"win": "Ganada (TP)", "loss": "Perdida (SL)",
                       "be": "Break-even", "expired": "Cerrada por tiempo"}
            outcome = st.radio("Resultado", list(MANUAL_OUTCOMES),
                               format_func=lambda k: _labels.get(k, k), horizontal=True)
            # Precio sugerido según el resultado (TP, SL o entrada); edítalo si
            # cerraste a otro precio. Con él se calcula el R.
            _default = {"win": row["take_profit"], "loss": row["stop_loss"]}.get(
                outcome, row["entry_price"])
            exit_px = st.number_input("Precio de salida", min_value=0.0,
                                      value=float(_default), format="%.5f",
                                      key=f"exit_{sel_id}_{outcome}")
            if st.button("Guardar resultado"):
                try:
                    res = resolve_signal_manually(int(sel_id), outcome,
                                                  exit_price=exit_px if exit_px > 0 else None)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    if res["updated"]:
                        _r = res.get("pnl_r")
                        st.success("Resultado guardado" +
                                   (f": {_r:+.2f}R." if _r is not None else " (sin R: falta el precio)."))
                        st.rerun()
                    else:
                        st.warning(f"No se guardó: {res.get('motivo')}.")
        else:
            st.caption("Todas las señales tienen resultado registrado.")
