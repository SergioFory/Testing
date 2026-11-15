"""
Pruebas de la Fase 1 — mensaje de Telegram (punto 1.1) y estadísticas del
dashboard en R (puntos 1.3 / 1.9):
  - "Score del setup: x (umbral y)", hora UTC de la vela del setup y antigüedad,
    precio actual y R:R efectivo, riesgo real en % del capital (TRADING_CAPITAL),
    lotes de XM (XM_CONTRACTS, pendiente mientras no haya especificación),
    "Cerrar a mercado si no toca SL/TP antes de" = time_exit_deadline, sin
    sentimiento y con todo el texto dinámico escapado;
  - get_outcome_stats: solo live sin quality_flag, ganadas / perdidas / por tiempo /
    be por separado, % de aciertos, E[R] con IC90 (t de Student), suma de R, sin
    límite de filas, manual/auto;
  - cierre manual (resolved_by='manual', pnl_r si se puede calcular) y el tracker
    escribe 'auto'; la limpieza de la Fase 0 respeta los cierres manuales.

    cd trading_system && python -m pytest tests -q

Sin red (requests.get/post fallan si alguien los usa); BD SQLite temporal.
"""
import sys
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures y dobles
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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)


def _trade(symbol="BTCUSDT", direction="short", entry=68923.2, sl=70527.8, tp=66516.29,
           capital=10_000.0, score=0.6161, setup_type="breakout"):
    """TradeSetup REAL construido con build_trade_setup (mismo cálculo del vivo)."""
    from risk.position_sizer import build_trade_setup
    row = pd.Series({"close": entry, "stop_loss": sl, "take_profit": tp,
                     "atr": abs(entry - sl) / 1.5, "direction": direction,
                     "symbol": symbol, "setup_type": setup_type})
    t = build_trade_setup(row, capital, score)
    assert t is not None
    return t


class _TagCheck(HTMLParser):
    """Comprueba que el HTML solo usa etiquetas que Telegram acepta y que están
    equilibradas (si no, Telegram rechaza el mensaje entero)."""
    ALLOWED = {"b", "i", "code"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.bad = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.ALLOWED:
            self.bad.append(tag)
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack.pop() != tag:
            self.bad.append("/" + tag)


def _assert_valid_html(msg):
    p = _TagCheck()
    p.feed(msg)
    p.close()
    assert p.bad == [] and p.stack == [], (p.bad, p.stack)


# ---------------------------------------------------------------------------
# Mensaje de Telegram (1.1)
# ---------------------------------------------------------------------------

def test_message_crypto_full_content():
    from notifications.telegram import format_signal_alert
    from data.outcome_tracker import time_exit_deadline
    t = _trade()
    st = pd.Timestamp("2026-11-13 00:00", tz="UTC")
    now = pd.Timestamp("2026-11-13 08:05", tz="UTC")
    msg = format_signal_alert(t, {"label": "FEAR", "score": -0.5, "n_news": 3},
                              setup_ts=st, current_price=68554.10, threshold=0.55,
                              capital=10_000, now=now)
    _assert_valid_html(msg)
    assert "Score del setup:</b> 0.62 (umbral 0.55)" in msg
    assert "Confianza" not in msg and "ML" not in msg
    assert "Sentimiento" not in msg and "Noticias" not in msg      # bloque quitado
    assert "2026-11-13 00:00 UTC (vela 4H, cerró 04:00 UTC)" in msg
    assert "cerró hace 4 h 5 min" in msg
    assert "Precio actual:</b> $68,554.10" in msg
    # R:R efectivo a 68554.10: (68554.10-66516.29)/(70527.80-68554.10) = 1.03
    assert "R:R efectivo a ese precio:</b> 1.03" in msg
    assert "R:R a la entrada:</b> 1.50" in msg
    # Riesgo real = risk_usd / capital (el tope del 10 % lo deja en ~0.23 %)
    assert f"${t.risk_usd:,.2f} = {t.risk_usd / 10_000 * 100:.2f} %" in msg
    assert "tope de tamaño del 10 % del capital" in msg
    assert "Lotes XM:</b> pendiente de la especificación del contrato" in msg
    assert f"{t.position_qty:.6f}".rstrip("0") + " BTC" in msg
    dl = time_exit_deadline("BTCUSDT", st)
    assert dl == pd.Timestamp("2026-11-17 04:00", tz="UTC")      # +25 × 4h
    assert "Cerrar a mercado si no toca SL/TP antes de:</b> 2026-11-17 04:00 UTC" in msg
    assert "dentro de 3 d 19 h" in msg


def test_message_daily_uses_tracker_deadline_and_session():
    from notifications.telegram import format_signal_alert
    from data.outcome_tracker import time_exit_deadline
    t = _trade(symbol="XAUUSD", entry=4168.40, sl=4323.68, tp=3961.36, score=0.2299)
    st = pd.Timestamp("2026-09-28", tz="UTC")          # lunes
    msg = format_signal_alert(t, setup_ts=st, current_price=4168.40,
                              now=pd.Timestamp("2026-09-28 22:20", tz="UTC"))
    _assert_valid_html(msg)
    assert "(umbral 0.20)" in msg                       # umbral del activo por defecto
    assert "sesión del 2026-09-28 (vela diaria, cierra 22:15 UTC)" in msg
    assert "cerró hace 5 min" in msg
    dl = time_exit_deadline("XAUUSD", st)
    assert dl == pd.Timestamp("2026-10-26 22:15", tz="UTC")       # 20 sesiones L-V
    assert "antes de:</b> 2026-10-26 22:15 UTC" in msg
    assert " oz" in msg
    assert "R:R efectivo a ese precio:</b> 1.33" in msg


def test_message_uses_trading_capital_by_default(monkeypatch):
    import config.settings as settings
    from notifications.telegram import format_signal_alert
    monkeypatch.setattr(settings, "TRADING_CAPITAL", 50_000.0)
    t = _trade(capital=50_000)
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz="UTC"),
                              current_price=t.entry_price)
    assert "Capital:</b> $50,000" in msg
    assert f"{t.risk_usd / 50_000 * 100:.2f} % del capital" in msg


def test_message_price_beyond_stop_warns():
    from notifications.telegram import format_signal_alert
    t = _trade()
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz="UTC"),
                              current_price=71000.0, now=pd.Timestamp("2026-11-13 05:00", tz="UTC"))
    assert "ya no tiene recorrido" in msg and "R:R efectivo" not in msg


def test_message_escapes_dynamic_text():
    from notifications.telegram import format_signal_alert
    t = _trade(symbol="X<&>Y", direction="long", entry=100.0, sl=97.0, tp=104.5,
               setup_type="a<b>&c")
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz="UTC"),
                              current_price=100.0, threshold=0.5, capital=10_000,
                              now=pd.Timestamp("2026-11-13 05:00", tz="UTC"))
    _assert_valid_html(msg)
    assert "X&lt;&amp;&gt;Y" in msg
    assert "A&lt;B&gt;&amp;C" in msg              # .title() del tipo de setup, escapado
    assert "<b>&c" not in msg


def test_message_without_context_still_works():
    """El dashboard o una llamada antigua sin vela/precio: se omiten esas líneas."""
    from notifications.telegram import format_signal_alert
    msg = format_signal_alert(_trade())
    _assert_valid_html(msg)
    assert "Score del setup" in msg and "Vela del setup" not in msg
    assert "Cerrar a mercado" not in msg and "Precio actual" not in msg


def test_safe_formatter_falls_back_on_error(monkeypatch):
    import notifications.telegram as tg

    def _boom(*a, **k):
        raise RuntimeError("fecha imposible")
    monkeypatch.setattr(tg, "format_signal_alert", _boom)
    msg = tg.safe_format_signal_alert(_trade())
    assert "SEÑAL — BTCUSDT" in msg and "Stop Loss" in msg


def test_xm_lot_size_rounding_and_minimum():
    from risk.position_sizer import xm_lot_size
    spec = {"BTCUSDT": {"unit": "BTC", "contract_size": 1.0, "min_lot": 0.01, "lot_step": 0.01}}
    r = xm_lot_size("BTCUSDT", 0.0347, entry=100.0, stop_loss=97.0, capital=10_000, contracts=spec)
    assert r["status"] == "ok" and r["lots"] == pytest.approx(0.03)      # hacia abajo
    assert r["risk_usd"] == pytest.approx(0.03 * 3.0)
    r = xm_lot_size("BTCUSDT", 0.30, contracts=spec)
    assert r["lots"] == pytest.approx(0.30)                              # sin perder un paso
    r = xm_lot_size("BTCUSDT", 0.004, entry=100.0, stop_loss=97.0, capital=10_000, contracts=spec)
    assert r["status"] == "below_min" and r["lots"] is None
    assert r["risk_pct"] == pytest.approx(0.01 * 3.0 / 10_000)
    assert xm_lot_size("BTCUSDT", 0.5)["status"] == "pending"            # settings: None
    assert xm_lot_size("NOPE", 1.0)["status"] == "pending"


def test_message_shows_lots_when_contract_known(monkeypatch):
    import config.settings as settings
    from notifications.telegram import format_signal_alert
    monkeypatch.setitem(settings.XM_CONTRACTS, "BTCUSDT",
                        {"unit": "BTC", "contract_size": 1.0, "min_lot": 0.01, "lot_step": 0.01})
    t = _trade()                                   # ~0.0145 BTC → 0.01 lotes
    msg = format_signal_alert(t, capital=10_000)
    assert "Lotes XM:</b> 0.01 (riesgo con ese tamaño:" in msg
    monkeypatch.setitem(settings.XM_CONTRACTS, "BTCUSDT",
                        {"unit": "BTC", "contract_size": 1.0, "min_lot": 0.1, "lot_step": 0.01})
    msg = format_signal_alert(t, capital=10_000)
    assert "no llega al lote mínimo de XM (0.1)" in msg


@pytest.mark.parametrize("raw,expected", [("", 10_000.0), ("25000", 25_000.0),
                                          ("25,000", 25_000.0), ("25_000.5", 25_000.5),
                                          ("abc", 10_000.0), ("-5", 10_000.0),
                                          ("changeme", 10_000.0)])
def test_trading_capital_env(monkeypatch, raw, expected):
    from config.settings import _env_capital
    monkeypatch.setenv("TRADING_CAPITAL", raw)
    assert _env_capital() == expected


def test_cmd_signal_sizes_with_trading_capital_and_sends_new_message(temp_db, monkeypatch):
    """cmd_signal real (datos y detección simulados): el tamaño se calcula con
    TRADING_CAPITAL y el texto enviado es el formato nuevo + el pie con el ID."""
    import main
    import config.settings as settings
    import signals.detector as det
    import ml.predictor as pred
    import data.bars as bars
    import notifications.telegram as tg

    idx = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=idx)
    df.attrs["source"] = "test"
    setup = {"ts": idx[-1], "symbol": "BTCUSDT", "setup_type": "breakout", "direction": "long",
             "raw_score": 0.7, "atr": 2.0, "close": 100.0, "stop_loss": 97.0, "take_profit": 104.5}
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: pd.DataFrame([setup]))
    monkeypatch.setattr(pred, "score_setups",
                        lambda d, *a, **k: d.assign(ml_score=d["raw_score"],
                                                    ml_approved=d["raw_score"] >= 0.55))
    monkeypatch.setattr(settings, "TRADING_CAPITAL", 40_000.0)
    sent = []
    monkeypatch.setattr(tg, "_send", lambda text, parse_mode="HTML", retries=3: sent.append(text) or True)

    main.cmd_signal("BTCUSDT", notify=True)
    assert len(sent) == 1
    msg = sent[0]
    _assert_valid_html(msg)
    with temp_db.get_engine().connect() as conn:
        row = conn.execute(text("SELECT * FROM signals")).mappings().fetchone()
    # 1 % de 40.000 = 400 USD de riesgo → nominal 13.333 > tope 10 % (4.000)
    assert row["position_size"] == pytest.approx(4_000.0)
    assert "Capital:</b> $40,000" in msg
    assert "Score del setup:</b> 0.70 (umbral 0.55)" in msg
    # vela del setup 2026-11-14 20:00 + 25 × 4h
    assert "Cerrar a mercado si no toca SL/TP antes de:</b> 2026-11-19 00:00 UTC" in msg
    assert f"Señal #{row['id']}" in msg and msg.count("/fill") == 1
    assert row["alert_text"] == msg and row["notified"] == 1


# ---------------------------------------------------------------------------
# Estadísticas del dashboard en R (1.3 / 1.9)
# ---------------------------------------------------------------------------

def _insert(db, **kw):
    base = {"symbol": "BTCUSDT", "ts": 1_763_000_000, "setup_type": "breakout",
            "direction": "long", "ml_score": 0.6, "entry_price": 100.0, "stop_loss": 97.0,
            "take_profit": 104.5, "position_size": 1000.0, "atr": 2.0, "mode": "live",
            "resultado_real": None, "pnl_r": None, "pnl_pct": None, "quality_flag": None,
            "resolved_by": None}
    base.update(kw)
    cols = ", ".join(base)
    vals = ", ".join(":" + k for k in base)
    with db.get_engine().begin() as conn:
        conn.execute(text(f"INSERT INTO signals ({cols}) VALUES ({vals})"), base)
        return conn.execute(text("SELECT MAX(id) FROM signals")).scalar()


def test_migration_adds_resolved_by(temp_db):
    temp_db.init_db(force=True)
    with temp_db.get_engine().connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(signals)"))}
    assert "resolved_by" in cols


def test_outcome_stats_in_r(temp_db):
    from data.outcome_tracker import get_outcome_stats, mean_ci90
    _insert(temp_db, resultado_real="win", pnl_r=1.5, resolved_by="auto")
    _insert(temp_db, resultado_real="win", pnl_r=1.5, resolved_by="auto")
    _insert(temp_db, resultado_real="loss", pnl_r=-1.0, resolved_by="auto")
    _insert(temp_db, resultado_real="expired", pnl_r=0.3, resolved_by="auto")
    _insert(temp_db, resultado_real="be", resolved_by="manual")              # R = 0
    # fila antigua sin pnl_r: R deducido de pnl_pct (−3 % con SL a 3 % = −1R)
    _insert(temp_db, resultado_real="loss", pnl_pct=-0.03)
    _insert(temp_db)                                                         # abierta
    # NO cuentan: sombra, marcada por calidad, otro activo
    _insert(temp_db, resultado_real="win", pnl_r=1.5, mode="shadow")
    _insert(temp_db, resultado_real="win", pnl_r=1.5, quality_flag="stale_reemit")
    _insert(temp_db, symbol="ETHUSDT", resultado_real="loss", pnl_r=-1.0)

    s = get_outcome_stats("BTCUSDT")
    assert (s["total"], s["resolved"], s["pending"]) == (7, 6, 1)
    assert (s["wins"], s["losses"], s["expired"], s["be"], s["other"]) == (2, 2, 1, 1, 0)
    assert s["win_rate"] == pytest.approx(2 / 6, abs=1e-4)
    rs = [1.5, 1.5, -1.0, 0.3, 0.0, -1.0]
    mean, lo, hi = mean_ci90(rs)
    assert s["n_r"] == 6
    assert s["mean_r"] == pytest.approx(mean, abs=1e-4)
    assert (s["ci90_low"], s["ci90_high"]) == (pytest.approx(lo, abs=1e-4), pytest.approx(hi, abs=1e-4))
    assert s["sum_r"] == pytest.approx(1.3)
    assert (s["manual"], s["auto"]) == (1, 4)

    s_all = get_outcome_stats(None)
    assert s_all["total"] == 8 and s_all["losses"] == 3


def test_outcome_stats_no_row_limit(temp_db):
    from data.outcome_tracker import get_outcome_stats
    with temp_db.get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, "
            "stop_loss, take_profit, position_size, atr, mode, resultado_real, pnl_r) VALUES "
            "('BTCUSDT', :ts, 'breakout', 'long', 0.6, 100, 97, 104.5, 1000, 2, 'live', 'loss', -1.0)"),
            [{"ts": 1_700_000_000 + i} for i in range(620)])
    s = get_outcome_stats("BTCUSDT")
    assert s["total"] == 620 and s["losses"] == 620 and s["sum_r"] == pytest.approx(-620)
    assert len(temp_db.load_signals("BTCUSDT", limit=None)) == 620
    assert len(temp_db.load_signals("BTCUSDT")) == 100                 # el resto, igual


def test_outcome_stats_edge_cases(temp_db):
    from data.outcome_tracker import get_outcome_stats
    assert get_outcome_stats("BTCUSDT") == {}
    _insert(temp_db)
    s = get_outcome_stats("BTCUSDT")
    assert s["resolved"] == 0 and s["win_rate"] is None and s["mean_r"] is None
    _insert(temp_db, resultado_real="win", pnl_r=1.5)
    s = get_outcome_stats("BTCUSDT")
    assert s["mean_r"] == 1.5 and s["ci90_low"] is None              # 1 resultado: sin IC


def test_mean_ci90_matches_t_distribution():
    from data.outcome_tracker import mean_ci90, _t95
    assert mean_ci90([]) == (None, None, None)
    assert mean_ci90([0.5]) == (0.5, None, None)
    m, lo, hi = mean_ci90([1.0, -1.0, 1.0, -1.0])        # media 0, s = 1.1547, t3 = 2.353
    assert m == 0 and hi == pytest.approx(2.353 * 1.1547005 / 2, abs=1e-4) and lo == -hi
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 10, 29, 30, 31, 45, 80, 200, 1000):
        assert _t95(df) == pytest.approx(scipy_stats.t.ppf(0.95, df), abs=1e-3)


def test_manual_resolution_writes_manual_and_r(temp_db):
    from data.outcome_tracker import resolve_signal_manually
    sid = _insert(temp_db)
    res = resolve_signal_manually(sid, "win", exit_price=103.0,
                                  now=pd.Timestamp("2026-11-15 10:00", tz="UTC"))
    assert res["updated"] and res["pnl_r"] == pytest.approx(1.0) and res["pnl_usd"] == pytest.approx(30.0)
    with temp_db.get_engine().connect() as conn:
        r = conn.execute(text("SELECT * FROM signals WHERE id=:i"), {"i": sid}).mappings().fetchone()
    assert r["resultado_real"] == "win" and r["resolved_by"] == "manual"
    assert r["pnl_r"] == pytest.approx(1.0) and r["exit_price"] == 103.0
    assert r["resolved_ts"] == int(pd.Timestamp("2026-11-15 10:00", tz="UTC").timestamp())
    # Ya cerrada: no se pisa
    again = resolve_signal_manually(sid, "loss", exit_price=97.0)
    assert not again["updated"] and "ya estaba cerrada" in again["motivo"]

    sid_be = _insert(temp_db)
    assert resolve_signal_manually(sid_be, "be")["pnl_r"] == 0.0          # sin precio: R = 0
    sid_nop = _insert(temp_db, direction="short", stop_loss=103.0, take_profit=95.5)
    res = resolve_signal_manually(sid_nop, "loss")                          # sin precio: sin R
    assert res["updated"] and res["pnl_r"] is None
    sid_short = _insert(temp_db, direction="short", stop_loss=103.0, take_profit=95.5)
    assert resolve_signal_manually(sid_short, "expired", exit_price=98.5)["pnl_r"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        resolve_signal_manually(sid_short, "ganada")
    with pytest.raises(ValueError):
        resolve_signal_manually(_insert(temp_db), "win", exit_price=0)
    assert not resolve_signal_manually(99_999, "win", exit_price=1.0)["updated"]


def test_tracker_writes_resolved_by_auto(temp_db):
    from data.outcome_tracker import _update_signal_result
    sid = _insert(temp_db)
    _update_signal_result(sid, "loss", -30.0, -0.03, pnl_r=-1.0, exit_price=97.0,
                          resolved_ts=pd.Timestamp("2026-11-15", tz="UTC"))
    with temp_db.get_engine().connect() as conn:
        assert conn.execute(text("SELECT resolved_by FROM signals WHERE id=:i"),
                            {"i": sid}).scalar() == "auto"


def test_cleanup_respects_manual_resolution():
    import fase0_db_cleanup as cl
    df = pd.DataFrame([{"id": 1, "symbol": "BTCUSDT", "resultado_real": "win", "pnl_pct": 0.03,
                        "resolved_by": "manual", "direction": "long", "entry_price": 100.0,
                        "stop_loss": 97.0, "take_profit": 104.5, "setup_ts": None,
                        "ts_dt": pd.Timestamp("2026-11-10", tz="UTC")}])
    out = cl.re_resolve(df, histories={})
    assert out.iloc[0]["new_outcome"] is None and bool(out.iloc[0]["manual"])
    assert "a mano" in out.iloc[0]["note"]
