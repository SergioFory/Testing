"""
Pruebas de la Fase 4 — grupo B: dashboard seguro y útil (no cambia señales).

  - dashboard/views.py (sin Streamlit):
      * preview_signal() da EXACTAMENTE lo que el worker emitiría en ese momento
        (mismo setup, dirección, entrada/SL/TP, tamaño y modo), con la historia
        corta del vivo, el umbral del activo y los mismos antiduplicados; y NO
        escribe nada en la BD (huella de todas las tablas) ni envía nada.
      * vigilancia, cartera, ejecuciones (/fill /skip, R teórico vs ejecutado),
        registro sombra y línea de riesgo, solo leyendo la BD.
  - dashboard/app.py: sin botones que escriban ni parte ML; con
    streamlit.testing.v1.AppTest si Streamlit está instalado (si no, se salta).

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post y Session.request fallan; Telegram es un doble y la BD
es SQLite temporal.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TABLES = ("signals", "shadow_setups", "fills", "kv", "data_quality", "ohlcv")


# ---------------------------------------------------------------------------
# Fixtures y utilidades
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    yield db
    if db._engine is not None:
        db._engine.dispose()


@pytest.fixture()
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.fixture()
def fake_telegram(monkeypatch, no_network):
    import notifications.telegram as tg
    state = {"sent": []}
    monkeypatch.setattr(tg, "_send",
                        lambda text, parse_mode="HTML", retries=3: state["sent"].append(text) or True)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    return state


@pytest.fixture()
def capital_1k(monkeypatch):
    import config.settings as st
    monkeypatch.setattr(st, "TRADING_CAPITAL", 1_000.0)
    return 1_000.0


def fingerprint(db) -> str:
    """Huella de TODAS las tablas con estado: si la vista escribe algo, cambia."""
    h = hashlib.sha256()
    with db.get_engine().connect() as conn:
        for t in TABLES:
            for r in conn.execute(text(f"SELECT * FROM {t} ORDER BY 1")).fetchall():
                h.update(repr(tuple(r)).encode())
            h.update(b"|")
    return h.hexdigest()


def _rows(db, table, where="1=1"):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table} WHERE {where} ORDER BY id"))
                .mappings().fetchall()]


IDX = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")


def _setup(ts, symbol="BTCUSDT", direction="long", raw=0.7, close=100.0, atr=2.0):
    sign = 1 if direction == "long" else -1
    return {"ts": ts, "symbol": symbol, "setup_type": "breakout", "direction": direction,
            "raw_score": raw, "atr": atr, "close": close,
            "stop_loss": close - sign * 1.5 * atr, "take_profit": close + sign * 2.25 * atr}


def _pipeline(monkeypatch, setups, calls=None, age_h=1.0):
    """Datos y detector simulados; TODO lo demás real: score_setups (raw_score con el
    umbral del activo), get_best_setup, build_trade_setup, antiduplicados y cartera."""
    import main
    import signals.detector as det
    import data.bars as bars
    import monitoring.data_monitor as dm

    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=IDX)
    df.attrs["source"] = "test"

    def _load(s, closed_only=True, history="full"):
        if calls is not None:
            calls.append((s, history))
        main._queue_ohlcv_copy(s, "4h", df)          # como el _load_data real
        main._queue_ohlcv_copy(s, "1d", df)
        return df, df, None
    monkeypatch.setattr(main, "_load_data", _load)
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: age_h)
    frame = pd.DataFrame(setups)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: frame.copy())
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    return main


def _expected(view):
    t = view["trade"]
    return {"symbol": t.symbol, "setup_ts": int(pd.Timestamp(view["setup_ts"]).timestamp()),
            "direction": t.direction, "entry_price": t.entry_price, "stop_loss": t.stop_loss,
            "take_profit": t.take_profit, "position_size": t.position_size, "mode": view["mode"]}


def _new_signals(db, after_id=0):
    keys = ("symbol", "setup_ts", "direction", "entry_price", "stop_loss", "take_profit",
            "position_size", "mode")
    return [{k: r[k] for k in keys} for r in _rows(db, "signals", f"id > {after_id}")]


# ---------------------------------------------------------------------------
# Vista = lo que emite el worker, sin escribir
# ---------------------------------------------------------------------------

def test_vista_igual_que_el_worker_y_no_escribe(temp_db, fake_telegram, capital_1k, monkeypatch):
    from dashboard import views
    calls = []
    t0, t1 = IDX[-2], IDX[-1]
    # BTC: umbral 0,55 (raw_score real). 0,54 no se aprueba; 0,90 sí y es el mejor.
    main = _pipeline(monkeypatch, [_setup(t0, raw=0.54), _setup(t1, raw=0.90, close=100.5)],
                     calls=calls)
    fp = fingerprint(temp_db)
    v = views.preview_signal("BTCUSDT")
    assert fingerprint(temp_db) == fp                  # la vista no escribe nada
    assert fake_telegram["sent"] == []                 # ni envía nada
    assert main._ohlcv_pending == {}                   # ni deja la copia de velas encolada
    assert calls == [("BTCUSDT", "live")]              # historia corta del vivo
    assert v["status"] == "would_emit" and v["mode"] == "live"
    assert v["threshold"] == pytest.approx(0.55)
    assert v["setup_ts"] == t1 and v["trade"].direction == "long"
    w = v["window"].set_index("vela del setup")["qué haría el worker"]
    assert w[t0] == "score bajo el umbral" and w[t1] == "emitido"

    main.cmd_signal("BTCUSDT", notify=True)            # el worker real
    assert _new_signals(temp_db) == [_expected(v)]
    assert len(fake_telegram["sent"]) == 1

    # Tras emitir, la vista ya no la ofrece (el worker tampoco la repetiría).
    fp = fingerprint(temp_db)
    v2 = views.preview_signal("BTCUSDT")
    assert fingerprint(temp_db) == fp
    assert v2["status"] == "no_new" and v2["trade"] is None
    assert v2["window"].set_index("vela del setup")["qué haría el worker"][t1] == "emitido"
    n = len(_rows(temp_db, "signals"))
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(_rows(temp_db, "signals")) == n


def test_vista_antes_que_el_worker_no_le_quita_la_senal(temp_db, fake_telegram, capital_1k,
                                                       monkeypatch):
    """El fallo que motivó el cambio: antes, registrar desde el dashboard hacía que el
    worker descartara el setup como duplicado. Ver la vista no cambia lo que emite."""
    from dashboard import views
    main = _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.8)])
    for _ in range(3):
        views.preview_signal("BTCUSDT")
    main.cmd_signal("BTCUSDT", notify=True)
    sig = _rows(temp_db, "signals")
    assert len(sig) == 1 and sig[0]["notified"] == 1
    assert f"/fill {sig[0]['id']}" in fake_telegram["sent"][0]


def test_vista_duplicado_por_precio_igual_que_el_worker(temp_db, fake_telegram, capital_1k,
                                                        monkeypatch):
    """Señal abierta con la misma entrada de otro setup: la vista dice 'duplicado' y el
    worker tampoco emite."""
    from dashboard import views
    from data.outcome_tracker import save_signal_from_trade
    from risk.position_sizer import build_trade_setup
    old = build_trade_setup(pd.Series(_setup(IDX[5], raw=0.9)), 1_000.0, 0.9)
    assert save_signal_from_trade(old, setup_ts=IDX[5], mode="live") > 0
    main = _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.9)])
    fp = fingerprint(temp_db)
    v = views.preview_signal("BTCUSDT")
    assert fingerprint(temp_db) == fp
    assert v["status"] == "duplicate" and v["trade"] is None
    assert v["window"]["qué haría el worker"].tolist() == ["duplicado"]
    n = len(_rows(temp_db, "signals"))
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(_rows(temp_db, "signals")) == n and fake_telegram["sent"] == []


def test_vista_activo_en_modo_sombra(temp_db, fake_telegram, capital_1k, monkeypatch):
    from dashboard import views
    from config.settings import notify_enabled
    assert notify_enabled("SPX500") is False
    t1 = IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t1, symbol="SPX500", raw=0.5)])
    v = views.preview_signal("SPX500")
    assert v["status"] == "would_shadow" and v["mode"] == "shadow"
    main.cmd_signal("SPX500", notify=True)
    assert _new_signals(temp_db) == [_expected(v)]
    assert fake_telegram["sent"] == []


def test_vista_fuente_congelada_y_sin_datos(temp_db, fake_telegram, monkeypatch):
    from dashboard import views
    _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.9)], age_h=100.0)
    fp = fingerprint(temp_db)
    v = views.preview_signal("BTCUSDT")
    assert v["status"] == "stale" and v["trade"] is None and "congelada" in v["message"]
    v = views.preview_signal("BTCUSDT", loader=lambda s: (None, None, None))
    assert v["status"] == "no_data"

    def _boom(s):
        raise RuntimeError("Gate.io caído")
    v = views.preview_signal("BTCUSDT", loader=_boom)
    assert v["status"] == "no_data" and "Gate.io caído" in v["message"]
    assert fingerprint(temp_db) == fp and fake_telegram["sent"] == []


def test_vista_ventana_y_frescura_igual_que_el_worker(temp_db, fake_telegram, capital_1k,
                                                     monkeypatch):
    """Un setup aprobado de hace 3 velas (fuera de la ventana de 2) y otro con la entrada
    a 5 ATR del precio actual: ni la vista ni el worker los emiten."""
    from dashboard import views
    main = _pipeline(monkeypatch, [_setup(IDX[-3], raw=0.95),
                                   _setup(IDX[-1], raw=0.9, close=110.0, atr=2.0)])
    v = views.preview_signal("BTCUSDT")
    assert v["status"] == "no_new" and v["trade"] is None
    w = v["window"].set_index("vela del setup")["qué haría el worker"]
    assert IDX[-3] not in w.index and w[IDX[-1]] == "precio ya lejos de la entrada"
    main.cmd_signal("BTCUSDT", notify=True)
    assert _rows(temp_db, "signals") == [] and fake_telegram["sent"] == []


def test_vista_no_aprobado_por_umbral(temp_db, fake_telegram, capital_1k, monkeypatch):
    from dashboard import views
    main = _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.549)])
    v = views.preview_signal("BTCUSDT")
    assert v["status"] == "no_new"
    main.cmd_signal("BTCUSDT", notify=True)
    assert _rows(temp_db, "signals") == []


def test_grafico_solo_ultimas_velas(temp_db, fake_telegram, capital_1k, monkeypatch):
    from dashboard import views
    monkeypatch.setattr(views, "CHART_BARS", 10)
    _pipeline(monkeypatch, [_setup(IDX[2], raw=0.9), _setup(IDX[-1], raw=0.9)])
    v = views.preview_signal("BTCUSDT")
    assert len(v["chart"]) == 10 and v["chart"].index[-1] == IDX[-1]
    assert v["chart_setups"]["ts"].tolist() == [IDX[-1]]     # el de IDX[2] queda fuera del gráfico


# ---------------------------------------------------------------------------
# Paneles de solo lectura
# ---------------------------------------------------------------------------

class _Trade:
    """Doble de TradeSetup: nominal 1.000 USD y stop al 3 % → riesgo 30 USD."""
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


def _signal(symbol="BTCUSDT", direction="long", setup_i=5, entry=100.0, mode="live"):
    from data.outcome_tracker import save_signal_from_trade
    sl = entry * (0.97 if direction == "long" else 1.03)
    tp = entry * (1.045 if direction == "long" else 0.955)
    sid = save_signal_from_trade(_Trade(symbol, direction, entry, sl, tp), setup_ts=IDX[setup_i],
                                 mode=mode)
    assert sid > 0
    return sid


def _resolve(db, sid, outcome, exit_price, pnl_r, rts):
    with db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET resultado_real=:o, exit_price=:x, pnl_r=:r, "
                          "resolved_ts=:t, resolved_by='auto' WHERE id=:i"),
                     {"o": outcome, "x": exit_price, "r": pnl_r, "t": rts, "i": sid})


def test_vigilancia(temp_db):
    from dashboard import views
    from monitoring.sequential import monitoring_state
    rts = int(IDX[-1].timestamp())
    for i, (o, x, r) in enumerate((("win", 104.5, 1.5), ("loss", 97.0, -1.0), ("win", 104.5, 1.5))):
        _resolve(temp_db, _signal(setup_i=i + 1, entry=100.0 + i), o, x, r, rts + i)
    fp = fingerprint(temp_db)
    v = views.vigilance()
    assert fingerprint(temp_db) == fp
    st = monitoring_state()
    assert v["frozen"] == "3/150"
    live = v["rows"].iloc[0]
    assert live["señales"] == 3 and live["R medio"] == pytest.approx(round(st["sprt_live"].mean_r, 3))
    assert live["decisión"] == "seguir"
    assert v["rows"].iloc[2]["decisión"] == "sin alarma"
    assert "**Vigilancia" in v["text"] and "<b>" not in v["text"]
    assert "Configuración congelada: 3/150" in v["text"]


def test_cartera_y_skip(temp_db, capital_1k):
    from dashboard import views
    from data.database import save_fill
    a = _signal("BTCUSDT", "long", setup_i=1, entry=200.0)
    b = _signal("ETHUSDT", "short", setup_i=2, entry=300.0)
    p = views.portfolio_view()
    assert sorted(p["positions"]["id"].tolist()) == [a, b]
    # Riesgo: 1.000 USD × 3 % = 30 USD de 1.000 → 3 % cada una.
    assert p["risk_pct"] == pytest.approx(6.0)
    assert p["mode"] == "shadow" and "Cartera según el sistema" in p["text"]
    save_fill(b, "skip", note="no llegué", update_id=1)
    fp = fingerprint(temp_db)
    p = views.portfolio_view()
    assert fingerprint(temp_db) == fp
    assert p["positions"]["id"].tolist() == [a]


def test_ejecuciones_r_teorico_y_ejecutado(temp_db):
    from dashboard import views
    from data.database import save_fill
    a = _signal("BTCUSDT", "long", setup_i=1, entry=100.0)      # SL 97, TP 104,5
    b = _signal("ETHUSDT", "short", setup_i=2, entry=200.0)
    c = _signal("XAUUSD", "long", setup_i=3, entry=300.0)
    save_fill(a, "fill", price=101.0, update_id=1)
    _resolve(temp_db, a, "win", 104.5, 1.5, int(IDX[-1].timestamp()))
    save_fill(b, "skip", note="tarde", update_id=2)
    d = _signal("ETHUSDT", "short", setup_i=4, entry=400.0)     # SL 412, TP 382
    save_fill(d, "fill", price=398.0, update_id=5)
    _resolve(temp_db, d, "loss", 412.0, -1.0, int(IDX[-1].timestamp()))
    save_fill(c, "fill", price=300.0, update_id=3)
    save_fill(c, "skip", note="me arrepentí", update_id=4)       # el último manda
    fp = fingerprint(temp_db)
    ex = views.executions()
    assert fingerprint(temp_db) == fp
    t = ex["table"].set_index("señal")
    assert t.loc[a, "tu registro"] == "/fill" and t.loc[a, "R teórico"] == pytest.approx(1.5)
    assert t.loc[a, "R ejecutado"] == pytest.approx(round((104.5 - 101.0) / 3.0, 3))
    assert t.loc[b, "tu registro"] == "/skip" and pd.isna(t.loc[b, "R ejecutado"])
    assert t.loc[c, "tu registro"] == "/skip" and t.loc[c, "nota"] == "me arrepentí"
    # Corto: entraste 2 USD por debajo de la entrada → pierdes más: −(412 − 398)/12.
    assert t.loc[d, "R ejecutado"] == pytest.approx(round(-(412.0 - 398.0) / 12.0, 3))
    s = ex["summary"]
    diffs = [(104.5 - 101.0) / 3.0 - 1.5, -(412.0 - 398.0) / 12.0 + 1.0]
    assert s["n_resolved"] == 2 and s["mean_diff_r"] == pytest.approx(sum(diffs) / 2)


def test_ejecuciones_vacio(temp_db):
    from dashboard import views
    ex = views.executions()
    assert ex["table"].empty and ex["summary"]["n_fill"] == 0


def test_registro_sombra(temp_db):
    from dashboard import views
    from data.shadow import record_shadow_setups
    rows = []
    for i, (motivo, appr) in enumerate((("emitted", 1), ("below_threshold", 0),
                                        ("below_threshold", 0), ("not_fresh", 1))):
        rows.append({"setup_ts": IDX[i], "direction": "long", "setup_type": "breakout",
                     "raw_score": 0.5, "approved": appr, "motivo": motivo, "entry": 100.0,
                     "sl": 97.0, "tp": 104.5, "atr": 2.0, "portfolio_flag": None,
                     "features": None, "keep_existing": False})
    assert record_shadow_setups("BTCUSDT", rows) == 4
    with temp_db.get_engine().begin() as conn:
        conn.execute(text("UPDATE shadow_setups SET resultado='loss', pnl_r=-1.0 "
                          "WHERE motivo='below_threshold'"))
        conn.execute(text("UPDATE shadow_setups SET resultado='win', pnl_r=NULL "
                          "WHERE motivo='not_fresh'"))   # sin pnl_r: R del TP = 1,5
    fp = fingerprint(temp_db)
    sh = views.shadow_summary()
    assert fingerprint(temp_db) == fp
    assert sh["total"] == 4 and sh["resolved"] == 3
    assert sh["counts"].loc["score bajo el umbral", "BTCUSDT"] == 2
    assert sh["counts"].loc["Total", "Total"] == 4
    r = sh["results"]
    assert r.loc["score bajo el umbral", "SL"] == 2 and r.loc["score bajo el umbral", "R medio"] == -1.0
    assert r.loc["precio ya lejos de la entrada", "R medio"] == pytest.approx(1.5)


def test_registro_sombra_vacio(temp_db):
    from dashboard import views
    sh = views.shadow_summary()
    assert sh["total"] == 0 and sh["counts"].empty


def test_linea_de_riesgo(capital_1k):
    from dashboard import views
    assert views.risk_line() == ("Riesgo por operación: 0.25 % del capital ($2.50 de $1,000); "
                                 "nominal máx. 1× el capital")


def test_html_a_markdown():
    from dashboard.views import html_to_markdown
    assert html_to_markdown("<b>A</b> &amp; <i>b</i>\n    • c") == "**A** & _b_  \n• c"


# ---------------------------------------------------------------------------
# Código del dashboard: sin escrituras ni ML
# ---------------------------------------------------------------------------

def test_dashboard_sin_botones_que_escriben_ni_ml():
    app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in app.splitlines() if not l.lstrip().startswith("#"))
    code = code.split('"""', 2)[-1]          # sin el docstring (explica lo que se quitó)
    for banned in ("Registrar señal en DB", "Enviar a Telegram", "save_signal_from_trade",
                   "send_signal_alert", "test_connection", "resolve_open_signals",
                   "resolve_signal_manually", "Guardar resultado", "train_model", "load_model",
                   "get_model_importance", "build_training_dataset", "fetch_macro",
                   "fetch_funding_rates", "Modelo ML", "lightgbm", "kv_set", "save_fill"):
        assert banned not in code, banned
    assert "views.preview_signal(symbol)" in code
    src = (ROOT / "dashboard" / "views.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src
    for banned in ("INSERT", "UPDATE ", "DELETE", "kv_set", "save_signal", "record_shadow",
                   "save_fill", "_flush_ohlcv_copies", "send_text", "_send("):
        assert banned not in src, banned
    # Mismo capital, riesgo y umbral que el worker (sin sliders).
    assert "build_trade_setup(best, _settings().TRADING_CAPITAL, ml_score)" in src
    assert 'history="live"' in src
    for slider in ('"Capital (USD)"', '"Riesgo por trade (%)"', '"Umbral del score"'):
        assert slider not in app


def test_dashboard_no_importa_lightgbm(no_network):
    """Importar las vistas no carga lightgbm/sklearn (memoria en 512 MB)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); import dashboard.views, main; "
            "from dashboard import views; "
            "print(any(m.split('.')[0] in ('lightgbm', 'sklearn', 'scipy') for m in sys.modules))"
            % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "False"


# ---------------------------------------------------------------------------
# AppTest (solo si Streamlit está instalado; en el arnés de pruebas no lo está)
# ---------------------------------------------------------------------------

def test_app_streamlit_vista_sin_escrituras(temp_db, fake_telegram, capital_1k, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.9)])
    _signal("ETHUSDT", "short", setup_i=2, entry=300.0)
    fp = fingerprint(temp_db)
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    labels = [b.label for b in at.button]
    assert labels == ["🔍 Ver lo que ve el worker"]
    assert [t.label for t in at.tabs] == ["📡 Vista del worker", "🧭 Vigilancia",
                                          "🧺 Cartera y ejecuciones", "🗂️ Registro sombra",
                                          "📜 Historial"]
    at.button[0].click().run()
    assert not at.exception, at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "LONG" in texts and "vista, no es una señal" in texts
    assert any("Las señales oficiales llegan solo por Telegram" in c.value for c in at.caption)
    assert fingerprint(temp_db) == fp and fake_telegram["sent"] == []
