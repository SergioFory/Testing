"""
Dashboard — vista de SOLO LECTURA del sistema de señales (Fase 4, grupo B).

Qué cambió en la Fase 4 (decisión aprobada por el usuario el 16-nov-2026):
  - Fuera los botones "Registrar señal en DB" y "Enviar a Telegram": registrar un
    setup desde aquí antes que el worker hacía que el worker lo descartara como
    duplicado (no llegaba por Telegram ni tenía ID para /fill) y el envío mandaba
    una señal sin ID.
  - Fuera la pestaña del modelo ML y cualquier entrenamiento o descarga de macro
    desde el dashboard (el ML está apagado y entrenar aquí podía agotar los 512 MB).
  - "Generar señal" pasa a ser una VISTA de lo que ve el worker (dashboard/views.py:
    misma historia corta del vivo, velas cerradas, mismo detector, umbral, ventana,
    frescura, antiduplicados y tamaño), sin escribir en la BD ni enviar nada.
  - Fuera también "Resolver señales pendientes" (el worker lo hace en cada ciclo,
    cada hora), el cierre manual (sustituía el resultado del sistema en la
    vigilancia SPRT/CUSUM; tu ejecución real va con /fill y /skip) y la prueba de
    Telegram (`python trading_system/main.py telegram`).
  - Nuevo: vigilancia (SPRT/CUSUM, configuración congelada N/150), cartera según
    el sistema, ejecuciones (/fill, /skip, R teórico frente a ejecutado), registro
    sombra y la línea de riesgo por operación.
Toda la lógica está en dashboard/views.py (sin Streamlit, probada con pytest).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timezone

from config.settings import ASSETS, fmt_price, active_assets, notify_enabled
from data.database import init_db
from data.outcome_tracker import get_outcome_stats
from dashboard import views

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
st.set_page_config(
    page_title="Señales de trading — vista",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.signal-long  { background:#d4edda; color:#111; padding:16px; border-radius:8px; border-left:5px solid #28a745; }
.signal-short { background:#f8d7da; color:#111; padding:16px; border-radius:8px; border-left:5px solid #dc3545; }
.signal-none  { background:#fff3cd; color:#111; padding:16px; border-radius:8px; border-left:5px solid #ffc107; }
</style>
""", unsafe_allow_html=True)

init_db()


def _safe(fn, *args, **kwargs):
    """Ejecuta una función de lectura; si falla, muestra el error sin tumbar la página."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo leer: {exc}")
        return None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.title("Señales de trading")
    st.caption("Vista de solo lectura")
    st.markdown("---")

    _assets = active_assets() or list(ASSETS.keys())
    symbol = st.selectbox(
        "Activo",
        _assets,
        format_func=lambda k: f"{ASSETS[k]['label']} — {k}"
                              + ("" if notify_enabled(k) else " (modo sombra)"),
    )
    asset_cfg = ASSETS[symbol]

    st.markdown("---")
    _rl = _safe(views.risk_line)
    if _rl:
        st.info(_rl)
    st.caption("Capital (TRADING_CAPITAL), riesgo y umbral son los del worker; "
               "aquí no se cambian.")

    st.markdown("---")
    run_btn = st.button("🔍 Ver lo que ve el worker", width="stretch", type="primary")
    st.caption(views.OFFICIAL_NOTE)
    st.caption(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

# =============================================================================
# PESTAÑAS
# =============================================================================
tab_view, tab_vig, tab_port, tab_shadow, tab_history = st.tabs([
    "📡 Vista del worker",
    "🧭 Vigilancia",
    "🧺 Cartera y ejecuciones",
    "🗂️ Registro sombra",
    "📜 Historial",
])

# =============================================================================
# 1) VISTA DEL WORKER (solo lectura)
# =============================================================================
with tab_view:
    st.header(f"Lo que ve el worker — {asset_cfg['label']}")
    st.caption(views.OFFICIAL_NOTE)

    # Solo se guarda la vista del activo elegido (pocas filas: gráfico y ventana).
    if run_btn:
        with st.spinner("Cargando las mismas velas que el worker..."):
            st.session_state["_view"] = views.preview_signal(symbol)
    view = st.session_state.get("_view")
    if view is not None and view.get("symbol") != symbol:
        view = None

    if view is None:
        st.info("Pulsa **Ver lo que ve el worker** en la barra lateral.")
    else:
        trade = view["trade"]
        if trade is not None:
            css = "signal-long" if trade.direction == "long" else "signal-short"
            icon = "🟢 LONG" if trade.direction == "long" else "🔴 SHORT"
            st.markdown(
                f'<div class="{css}"><h3>{icon} — {asset_cfg["label"]} (vista, no es una señal)</h3>'
                f'<p>Setup <b>{trade.setup_type}</b> de la vela {view["setup_ts"]} | '
                f'score {trade.ml_score:.2f} (umbral {view["threshold"]:.2f})</p></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="signal-none"><h3>⚪ Sin señal nueva</h3></div>',
                        unsafe_allow_html=True)
        st.write(view["message"])
        for line in view.get("portfolio_lines") or []:
            st.warning(f"Control de cartera: {line}")

        st.caption(
            f"Fuente {view['data_source'] or '—'} · {view['history_bars']} velas {view['timeframe']} "
            f"cerradas (historia del vivo) · última vela {view['last_bar_ts']} · "
            f"precio {fmt_price(view['current_price']) if view['current_price'] else '—'} · "
            f"{'modo sombra' if view['mode'] == 'shadow' else 'activo live'}")

        if trade is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entrada", f"${fmt_price(trade.entry_price)}")
            c2.metric("Stop Loss", f"${fmt_price(trade.stop_loss)}",
                      delta=f"-{trade.stop_distance_pct*100:.2f}%", delta_color="inverse")
            c3.metric("Take Profit", f"${fmt_price(trade.take_profit)}",
                      delta=f"+{trade.target_distance_pct*100:.2f}%")
            c4.metric("R:R", f"{trade.rr_ratio:.2f}:1")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Nominal USD", f"${trade.position_size:,.0f}")
            c6.metric("Riesgo USD", f"${trade.risk_usd:,.2f}")
            c7.metric("Beneficio USD en TP", f"${trade.reward_usd:,.2f}")
            c8.metric("ATR (14)", f"{trade.atr_pct*100:.2f}%")

        w = view.get("window")
        if w is not None and not w.empty:
            st.subheader("Setups de la ventana del worker")
            st.dataframe(w, width="stretch", hide_index=True)

        chart = view.get("chart")
        if chart is not None and len(chart) > 10:
            tf = view["timeframe"]
            st.subheader(f"Últimas {len(chart)} velas {tf} cerradas")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(
                x=chart.index, open=chart["open"], high=chart["high"], low=chart["low"],
                close=chart["close"], name="OHLC",
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
            fig.add_trace(go.Scatter(x=chart.index, y=chart["close"].ewm(span=21).mean(),
                                     name="EMA 21", line=dict(color="orange", width=1.5)),
                          row=1, col=1)
            cs = view.get("chart_setups")
            if cs is not None and not cs.empty:
                for d, sym_, col, k in (("long", "triangle-up", "lime", 0.998),
                                        ("short", "triangle-down", "red", 1.002)):
                    sub = cs[cs["direction"] == d]
                    if not sub.empty:
                        fig.add_trace(go.Scatter(x=sub["ts"], y=sub["close"] * k, mode="markers",
                                                 marker=dict(symbol=sym_, size=10, color=col),
                                                 name=f"Setup {d.upper()}"), row=1, col=1)
            if trade is not None:
                for y, colr, lab in ((trade.entry_price, "white", "Entrada"),
                                     (trade.stop_loss, "red", "SL"),
                                     (trade.take_profit, "lime", "TP")):
                    fig.add_hline(y=y, line_color=colr, line_dash="dot", line_width=1.2,
                                  annotation_text=lab, row=1, col=1)
            fig.add_trace(go.Bar(x=chart.index, y=chart["volume"], name="Volumen",
                                 marker_color="#888", opacity=0.6), row=2, col=1)
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=520,
                              margin=dict(l=0, r=0, t=10, b=0),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, width="stretch")

# =============================================================================
# 2) VIGILANCIA (SPRT / CUSUM)
# =============================================================================
with tab_vig:
    st.header("Vigilancia (reglas escritas de antemano)")
    vig = _safe(views.vigilance)
    if vig is not None:
        st.metric("Configuración congelada", vig["frozen"],
                  help="Señales live resueltas que cuentan para el SPRT; la congelación acaba "
                       "cuando el SPRT decide o al llegar a 150.")
        st.dataframe(vig["rows"], width="stretch", hide_index=True)
        st.markdown(vig["text"])

# =============================================================================
# 3) CARTERA Y EJECUCIONES
# =============================================================================
with tab_port:
    st.header("Cartera según el sistema")
    pv = _safe(views.portfolio_view)
    if pv is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Posiciones abiertas", len(pv["positions"]))
        c2.metric("Riesgo abierto", f"{pv['risk_pct']:.2f} % del capital")
        c3.metric("Pérdida realizada hoy (UTC)", f"{pv['day_loss_pct']:.2f} %")
        if not pv["positions"].empty:
            st.dataframe(pv["positions"], width="stretch", hide_index=True)
        st.markdown(pv["text"])

    st.header("Tus ejecuciones (/fill y /skip)")
    ex = _safe(views.executions)
    if ex is not None:
        s = ex["summary"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Ejecutadas (/fill)", s["n_fill"])
        c2.metric("No ejecutadas (/skip)", s["n_skip"])
        c3.metric("R ejecutado − R teórico (media)",
                  f"{s['mean_diff_r']:+.3f}R" if s.get("mean_diff_r") is not None else "—",
                  help=f"Sobre {s['n_resolved']} señal(es) con /fill ya resueltas. Diferencia "
                       "por la entrada: el sistema no conoce tu precio de salida.")
        if ex["table"].empty:
            st.caption("Sin registros. Responde a cada señal en Telegram con /fill ID PRECIO "
                       "o /skip ID motivo.")
        else:
            st.dataframe(ex["table"], width="stretch", hide_index=True)

# =============================================================================
# 4) REGISTRO SOMBRA
# =============================================================================
with tab_shadow:
    st.header("Registro sombra")
    st.caption("Todos los setups que vio el worker en su ventana, emitidos o no, con el motivo. "
               "Sirve para medir también lo que NO se emitió. R sin costes del bróker.")
    sh = _safe(views.shadow_summary)
    if sh is not None:
        if sh["total"] == 0:
            st.info("El registro sombra está vacío.")
        else:
            st.caption(f"{sh['total']} setups registrados; {sh['resolved']} resueltos.")
            st.subheader("Setups por motivo y activo")
            st.dataframe(sh["counts"], width="stretch")
            if not sh["results"].empty:
                st.subheader("Resultados de los resueltos, por motivo")
                st.dataframe(sh["results"], width="stretch")

# =============================================================================
# 5) HISTORIAL (solo lectura)
# =============================================================================
with tab_history:
    st.header("Historial de señales")
    st.caption("El worker resuelve las señales abiertas en cada ciclo (cada hora) con velas "
               "cerradas, con la misma regla que la validación. Tu ejecución real se registra "
               "con /fill y /skip en Telegram.")

    # Estadísticas en R (Fase 1, 1.3 / 1.9): solo señales LIVE sin marca de calidad.
    _all_assets = st.checkbox("Estadísticas de todos los activos", value=False)
    stats = _safe(get_outcome_stats, symbol=None if _all_assets else symbol)
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

    hist_df = _safe(views.last_signals, symbol=symbol, limit=200)
    if hist_df is None or hist_df.empty:
        st.info("No hay señales guardadas para este activo aún (las guarda el worker).")
    else:
        st.dataframe(hist_df, width="stretch", hide_index=True)
