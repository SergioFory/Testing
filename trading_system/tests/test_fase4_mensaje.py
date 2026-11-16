"""
Pruebas de la Fase 4 — grupo C: mensaje y registro (no cambia qué señales salen).

  - Mensaje de Telegram: distancias del SL y del TP desde la entrada (% y USD por
    unidad; puntos del índice en el SPX) en todos los activos y, en el oro, la nota
    del futuro GC=F frente al oro de XM. Peor caso < 4.096 caracteres y HTML cerrado.
  - Columna signals.price_at_emit: migración idempotente (también sobre una tabla
    antigua sin la columna) y el ciclo real guarda el precio actual que usó.
  - Resumen nocturno: bloque "Últimos 12 meses" (nº de señales live resueltas y R
    medio, por activo), con las mismas exclusiones que el resto del resumen.
  - Calendario de eventos: ningún FOMC de 2027 falta (verificado en federalreserve.gov).
  - Monitor de datos: los avisos graves pendientes y la hora del último aviso se
    guardan en kv y sobreviven a un "reinicio" (estado en memoria borrado).

    cd trading_system && python -m pytest tests/test_fase4_mensaje.py -q

Sin red: requests.get/post y Session.request fallan; Telegram es un doble y la BD
es SQLite temporal.
"""
import csv
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
UTC = "UTC"


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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setattr(requests.Session, "request", _boom)


@pytest.fixture()
def fake_telegram(monkeypatch):
    import notifications.telegram as tg
    state = {"sent": []}
    monkeypatch.setattr(tg, "_send",
                        lambda text, parse_mode="HTML", retries=3: state["sent"].append(text) or True)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    return state


class _TagCheck(HTMLParser):
    """Solo etiquetas que Telegram acepta en este sistema y bien equilibradas."""
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


def _trade(symbol="BTCUSDT", direction="long", entry=90_000.0, sl=88_650.0, tp=92_025.0,
           capital=1_000.0, score=0.61, setup_type="breakout"):
    """TradeSetup REAL (build_trade_setup, el mismo cálculo del vivo)."""
    from risk.position_sizer import build_trade_setup
    row = pd.Series({"close": entry, "stop_loss": sl, "take_profit": tp,
                     "atr": abs(entry - sl) / 1.5, "direction": direction,
                     "symbol": symbol, "setup_type": setup_type})
    t = build_trade_setup(row, capital, score)
    assert t is not None
    return t


def _utf16_len(s: str) -> int:
    """Telegram cuenta el límite de 4.096 en unidades UTF-16 (un emoji = 2)."""
    return len(s.encode("utf-16-le")) // 2


# ---------------------------------------------------------------------------
# 1. Distancias de SL/TP y nota del oro
# ---------------------------------------------------------------------------

def test_distancias_btc_long_exactas():
    from notifications.telegram import format_signal_alert
    t = _trade()   # SL 1.350 por debajo (1,50 %), TP 2.025 por encima (2,25 %)
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-16 00:00", tz=UTC),
                              current_price=90_100.0, capital=1_000,
                              now=pd.Timestamp("2026-11-16 04:01", tz=UTC))
    _assert_valid_html(msg)
    assert ("<b>Distancias desde la entrada:</b> SL a 1.50 % (1,350.00 USD por BTC); "
            "TP a 2.25 % (2,025.00 USD por BTC)") in msg
    # Va justo debajo de los niveles (tras el R:R a la entrada) y antes del precio actual.
    assert msg.index("R:R a la entrada") < msg.index("Distancias desde la entrada") < msg.index("Precio actual")
    assert "GC=F" not in msg                                   # la nota es solo del oro


def test_distancias_eth_short_y_spx_en_puntos():
    from notifications.telegram import distance_lines
    eth = _trade("ETHUSDT", "short", entry=3_000.0, sl=3_090.0, tp=2_820.0)
    (line,) = distance_lines(eth)
    assert line == ("<b>Distancias desde la entrada:</b> SL a 3.00 % (90.00 USD por ETH); "
                    "TP a 6.00 % (180.00 USD por ETH)")
    spx = _trade("SPX500", "long", entry=6_000.0, sl=5_880.0, tp=6_200.0)
    (line,) = distance_lines(spx)
    assert line == ("<b>Distancias desde la entrada:</b> SL a 2.00 % (120.00 puntos del índice); "
                    "TP a 3.33 % (200.00 puntos del índice)")


def test_oro_lleva_la_nota_del_futuro_y_las_distancias():
    from notifications.telegram import format_signal_alert, FUTURES_BASIS_NOTE
    t = _trade("XAUUSD", "short", entry=4_100.0, sl=4_190.0, tp=3_980.0)
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz=UTC),
                              current_price=4_095.0, capital=1_000,
                              now=pd.Timestamp("2026-11-13 22:20", tz=UTC))
    _assert_valid_html(msg)
    assert ("SL a 2.20 % (90.00 USD por oz); TP a 2.93 % (120.00 USD por oz)") in msg
    assert FUTURES_BASIS_NOTE["GC=F"] in msg
    assert "futuro GC=F" in msg and "por encima del oro de XM" in msg
    assert "~0.3–0.5 %" in msg and "~0.7–1 %" in msg
    assert "si tu precio en MT5 es distinto".lower() in msg.lower()
    assert "coloca el SL y el TP a esas distancias de tu precio de entrada" in msg
    # La nota va tras la línea de distancias.
    assert msg.index("Distancias desde la entrada") < msg.index("futuro GC=F")


def test_distancias_no_cambian_los_niveles_y_un_fallo_no_tumba_el_mensaje(monkeypatch):
    import notifications.telegram as tg
    t = _trade()
    before = (t.entry_price, t.stop_loss, t.take_profit, t.position_size, t.risk_usd)
    msg = tg.format_signal_alert(t, capital=1_000)
    assert (t.entry_price, t.stop_loss, t.take_profit, t.position_size, t.risk_usd) == before

    def _boom(*a, **k):
        raise RuntimeError("fallo raro")
    monkeypatch.setattr(tg, "distance_lines", _boom)
    msg2 = tg.safe_format_signal_alert(t, capital=1_000)
    assert "Distancias" not in msg2
    assert "Riesgo real" in msg2 and "Stop Loss" in msg2           # mensaje completo, no el mínimo
    # Es el mismo mensaje sin la línea de distancias.
    assert msg2 == "\n".join(l for l in msg.split("\n") if "Distancias desde la entrada" not in l)
    _assert_valid_html(msg2)


def test_distancias_entrada_invalida_no_salen():
    from notifications.telegram import distance_lines

    class _T:
        symbol, entry_price, stop_loss, take_profit = "BTCUSDT", 0.0, 1.0, 2.0
    assert distance_lines(_T()) == []
    _T.entry_price = float("nan")
    assert distance_lines(_T()) == []


def _worst_case_message(symbol, monkeypatch):
    """Mensaje más largo posible: todas las líneas opcionales a la vez, textos
    variables al máximo, pie con ID enorme y la nota de reenvío."""
    import notifications.telegram as tg
    import config.settings as st
    from risk.portfolio import PortfolioCheck, RULE_TEXT
    from notifications.telegram_commands import signal_id_footer

    # Lotes: "por debajo del mínimo" (la variante de texto más larga).
    contracts = dict(st.XM_CONTRACTS)
    contracts[symbol] = dict(contracts[symbol], contract_size=1.0, min_lot=1000.0, lot_step=0.01)
    monkeypatch.setattr(st, "XM_CONTRACTS", contracts)
    # Eventos: más de 3 dentro del plazo ("(y N más)").
    evs = [{"event": "FOMC", "title": "FOMC", "time": pd.Timestamp("2026-12-09 19:00", tz=UTC)
            + pd.Timedelta(days=i)} for i in range(12)]
    monkeypatch.setattr("data.fetchers.econ_calendar.events_between", lambda a, b: evs)

    entry = {"BTCUSDT": 123_456.78, "ETHUSDT": 12_345.67, "XAUUSD": 4_321.09, "SPX500": 7_654.32}[symbol]

    class _BigTrade:          # riesgo enorme → línea "Para respetar el 0,25 %"
        pass
    t = _BigTrade()
    t.symbol, t.direction, t.setup_type = symbol, "short", "trend_pullback"
    t.entry_price, t.stop_loss, t.take_profit = entry, entry * 1.3333, entry * 0.5555
    t.position_size, t.position_qty = 9_999_999.99, 81.0001
    t.risk_usd, t.rr_ratio, t.ml_score, t.atr = 999_999.99, 1.33, 0.9999, entry * 0.2
    t.leverage_cap = None
    opens = [{"symbol": s, "direction": "short", "id": 9_999_990 + i, "risk_pct": 0.0025}
             for i, s in enumerate(["BTCUSDT", "ETHUSDT", "XAUUSD", "SPX500"] * 3)]
    pc = PortfolioCheck(open_positions=opens,
                        broken=[{"rule": r, "text": RULE_TEXT[r] + " (" + "x" * 40 + ")"}
                                for r in RULE_TEXT], mode="shadow")
    msg = tg.format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-30 20:00", tz=UTC),
                                 current_price=entry * 1.5, threshold=0.5555, capital=1_234.56,
                                 now=pd.Timestamp("2026-12-01 23:59", tz=UTC), portfolio=pc)
    return f"{tg.RESEND_NOTE}\n\n{msg}{signal_id_footer(999_999_999)}"


@pytest.mark.parametrize("symbol", ["BTCUSDT", "ETHUSDT", "XAUUSD", "SPX500"])
def test_peor_caso_cabe_en_telegram_y_html_cerrado(symbol, monkeypatch):
    msg = _worst_case_message(symbol, monkeypatch)
    _assert_valid_html(msg)
    # Comprobación de que de verdad es el peor caso: salen todas las líneas opcionales.
    for frag in ("Distancias desde la entrada", "ya no tiene recorrido", "Para respetar el",
                 "lote mínimo", "Con esta señal", "y 7 más", "Evento de alto impacto",
                 "Reenvío tras fallo", "/fill 999999999"):
        assert frag in msg, frag
    if symbol == "XAUUSD":
        assert "futuro GC=F" in msg
    # Límite de Telegram: 4.096 (se mide el HTML entero, más largo que el texto visible).
    assert len(msg) < 4096 and _utf16_len(msg) < 4096, (len(msg), _utf16_len(msg))


# ---------------------------------------------------------------------------
# 2. Columna price_at_emit
# ---------------------------------------------------------------------------

def _cols(db, table="signals"):
    with db.get_engine().connect() as conn:
        return {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}


def test_migracion_price_at_emit_idempotente_sobre_tabla_antigua(tmp_path, monkeypatch):
    import data.database as db
    path = tmp_path / "antigua.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    # BD de antes de la Fase 4: signals sin price_at_emit, con una fila.
    from sqlalchemy import create_engine
    eng = create_engine(f"sqlite:///{path}")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, "
                          "ts INTEGER NOT NULL, setup_type TEXT NOT NULL, direction TEXT NOT NULL, "
                          "ml_score REAL NOT NULL, sentiment_score REAL, entry_price REAL NOT NULL, "
                          "stop_loss REAL NOT NULL, take_profit REAL NOT NULL, position_size REAL NOT NULL, "
                          "atr REAL NOT NULL, resultado_real TEXT, pnl_usd REAL, pnl_pct REAL, "
                          "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"))
        conn.execute(text("INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, "
                          "stop_loss, take_profit, position_size, atr) VALUES "
                          "('BTCUSDT', 1, 'breakout', 'long', 0.6, 100, 97, 104.5, 1000, 2)"))
    eng.dispose()
    try:
        db.init_db()
        assert "price_at_emit" in _cols(db)
        db.init_db(force=True)                         # segunda vez: no falla ni duplica
        db._migrate(db.get_engine().connect().__enter__(), "sqlite")   # y directa, tolera la columna
        assert "price_at_emit" in _cols(db)
        with db.get_engine().connect() as conn:
            row = conn.execute(text("SELECT entry_price, price_at_emit FROM signals")).one()
        assert row == (100.0, None)                    # la fila antigua queda vacía
    finally:
        if db._engine is not None:
            db._engine.dispose()


def test_save_signal_guarda_price_at_emit_y_tolera_valores_raros(temp_db):
    from data.outcome_tracker import save_signal_from_trade
    t = _trade()
    sid = save_signal_from_trade(t, setup_ts=pd.Timestamp("2026-11-16", tz=UTC), mode="live",
                                 price_at_emit=90_123.45)
    with temp_db.get_engine().connect() as conn:
        assert conn.execute(text("SELECT price_at_emit FROM signals WHERE id=:i"),
                            {"i": sid}).scalar() == pytest.approx(90_123.45)
    # Valores raros: la señal se guarda igual, con el precio vacío.
    for k, bad in enumerate([None, float("nan"), float("inf"), -5.0, "abc", np.float64(0.0)]):
        tk = _trade(entry=50_000.0 + 1_000 * k, sl=49_000.0 + 1_000 * k, tp=51_500.0 + 1_000 * k)
        sid = save_signal_from_trade(tk, setup_ts=pd.Timestamp("2026-11-17", tz=UTC) + pd.Timedelta(hours=4 * k),
                                     mode="live", price_at_emit=bad)
        assert sid > 0
        with temp_db.get_engine().connect() as conn:
            assert conn.execute(text("SELECT price_at_emit FROM signals WHERE id=:i"),
                                {"i": sid}).scalar() is None
    # np.float64 válido (lo que da df["close"].iloc[-1]) se guarda como número.
    tk = _trade(entry=60_000.0, sl=59_000.0, tp=61_500.0)
    sid = save_signal_from_trade(tk, setup_ts=pd.Timestamp("2026-11-20", tz=UTC), mode="live",
                                 price_at_emit=np.float64(60_010.5))
    with temp_db.get_engine().connect() as conn:
        assert conn.execute(text("SELECT price_at_emit FROM signals WHERE id=:i"),
                            {"i": sid}).scalar() == pytest.approx(60_010.5)


IDX = pd.date_range("2026-11-10", periods=30, freq="4h", tz=UTC)


def _pipeline(monkeypatch, setups, last_close):
    """Datos y detector simulados; todo lo demás real (como en test_fase4_dashboard)."""
    import main
    import signals.detector as det
    import data.bars as bars
    import monitoring.data_monitor as dm
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=IDX)
    df.loc[IDX[-1], "close"] = last_close
    df.attrs["source"] = "test"
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    frame = pd.DataFrame(setups)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: frame.copy())
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    return main


def _setup(ts, symbol="BTCUSDT", direction="long", raw=0.9, close=100.0, atr=2.0):
    sign = 1 if direction == "long" else -1
    return {"ts": ts, "symbol": symbol, "setup_type": "breakout", "direction": direction,
            "raw_score": raw, "atr": atr, "close": close,
            "stop_loss": close - sign * 1.5 * atr, "take_profit": close + sign * 2.25 * atr}


def test_ciclo_real_guarda_el_precio_actual_y_el_mensaje_lleva_distancias(temp_db, fake_telegram,
                                                                          monkeypatch):
    import config.settings as st
    monkeypatch.setattr(st, "TRADING_CAPITAL", 1_000.0)
    main = _pipeline(monkeypatch, [_setup(IDX[-1], close=100.5)], last_close=100.7)
    main.cmd_signal("BTCUSDT", notify=True)
    with temp_db.get_engine().connect() as conn:
        rows = conn.execute(text("SELECT entry_price, stop_loss, take_profit, price_at_emit, mode, alert_text "
                                 "FROM signals")).mappings().fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["mode"] == "live"
    assert r["entry_price"] == pytest.approx(100.5)        # la entrada NO cambia
    assert r["stop_loss"] == pytest.approx(97.5) and r["take_profit"] == pytest.approx(105.0)
    assert r["price_at_emit"] == pytest.approx(100.7)      # el precio actual del ciclo
    assert "Precio actual:</b> $100.700" in r["alert_text"]
    assert ("SL a 2.99 % (3.000 USD por BTC); TP a 4.48 % (4.500 USD por BTC)") in r["alert_text"]
    assert len(fake_telegram["sent"]) == 1 and "Distancias desde la entrada" in fake_telegram["sent"][0]
    _assert_valid_html(fake_telegram["sent"][0])


def test_ciclo_spx_en_sombra_tambien_guarda_price_at_emit(temp_db, fake_telegram, monkeypatch):
    main = _pipeline(monkeypatch, [_setup(IDX[-1], symbol="SPX500", raw=0.9, close=6_000.0, atr=40.0)],
                     last_close=6_010.0)
    main.cmd_signal("SPX500", notify=True)
    with temp_db.get_engine().connect() as conn:
        r = conn.execute(text("SELECT mode, price_at_emit, alert_text FROM signals")).mappings().one()
    assert r["mode"] == "shadow" and r["alert_text"] is None
    assert r["price_at_emit"] == pytest.approx(6_010.0)
    assert fake_telegram["sent"] == []


# ---------------------------------------------------------------------------
# 3. Resumen nocturno: últimos 12 meses
# ---------------------------------------------------------------------------

NOW = pd.Timestamp("2026-11-16 03:00", tz=UTC)
NOW_U = int(NOW.timestamp())
DAY = 86400


def _ins(db, symbol, res, pnl_r=None, resolved_days_ago=10, mode="live", quality_flag=None,
         entry=100.0, sl=98.0, tp=103.0, resolved_ts=True, resolved_by="auto"):
    rts = NOW_U - int(resolved_days_ago * DAY)
    with db.get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, stop_loss, "
            "take_profit, position_size, atr, resultado_real, pnl_r, resolved_ts, mode, quality_flag, "
            "resolved_by) VALUES (:s, :ts, 'breakout', 'long', 0.6, :e, :sl, :tp, 1000, 2, :res, :r, "
            ":rts, :m, :q, :rb)"),
            {"s": symbol, "ts": rts - 3 * DAY, "e": entry, "sl": sl, "tp": tp, "res": res, "r": pnl_r,
             "rts": rts if resolved_ts else None, "m": mode, "q": quality_flag, "rb": resolved_by})


def test_bloque_12_meses_cuenta_bien(temp_db):
    from monitoring.daily_summary import year_results, year_block
    db = temp_db
    _ins(db, "BTCUSDT", "win", 1.5, 20)
    _ins(db, "BTCUSDT", "loss", -1.0, 200)
    _ins(db, "ETHUSDT", "win", None, 100)             # sin pnl_r: R implícito = 3/2 = 1.5
    _ins(db, "XAUUSD", "expired", 0.4, 364)
    _ins(db, "XAUUSD", "loss", None, 30, resolved_ts=False)   # sin resolved_ts: cuenta por ts (33 días)
    _ins(db, "ETHUSDT", "expired", None, 50)           # sin R medible
    # Excluidas:
    _ins(db, "BTCUSDT", "win", 1.5, 366)               # hace más de 12 meses
    _ins(db, "BTCUSDT", "win", 1.5, 10, quality_flag="reemision")   # limpieza de la Fase 0
    _ins(db, "BTCUSDT", "win", 1.5, 10, mode="shadow")   # sombra
    _ins(db, "SPX500", "win", 1.67, 10, mode="shadow")   # activo en modo sombra
    _ins(db, "EURUSD", "loss", -1.0, 10)               # activo apagado
    with db.get_engine().connect() as conn:
        yr = year_results(conn, NOW_U)
        txt = "\n".join(year_block(conn, NOW_U))
    assert yr["n"] == 5 and yr["n_sin_r"] == 1
    assert yr["sum_r"] == pytest.approx(1.5 - 1.0 + 1.5 + 0.4 - 1.0)
    assert yr["mean_r"] == pytest.approx(1.4 / 5)
    assert yr["by_symbol"] == {"BTCUSDT": (2, pytest.approx(0.25)), "ETHUSDT": (1, pytest.approx(1.5)),
                               "XAUUSD": (2, pytest.approx(-0.3))}
    _assert_valid_html(txt)
    assert "Últimos 12 meses" in txt
    assert "5 señales resueltas; R medio <b>+0.28R</b> por señal (suma +1.40R; sin costes)" in txt
    assert "Por activo: BTC 2 (+0.25R) · ETH 1 (+1.50R) · XAU 2 (-0.30R)" in txt
    assert "Sin R medible" in txt and ": 1" in txt
    assert "meses con tendencia" in txt and "vigilancia" in txt


def test_bloque_12_meses_vacio_y_en_el_resumen(temp_db, monkeypatch):
    from monitoring import daily_summary as ds
    with temp_db.get_engine().connect() as conn:
        txt = "\n".join(ds.year_block(conn, NOW_U))
    assert "Aún ninguna señal live resuelta en los últimos 12 meses." in txt
    assert "Sin R medible" not in txt
    _ins(temp_db, "BTCUSDT", "win", 1.5, 5)
    full = ds.build_daily_summary(now=NOW)
    _assert_valid_html(full)
    assert "📆 <b>Últimos 12 meses</b>" in full
    assert "1 señal resuelta; R medio <b>+1.50R</b>" in full
    # Va después de "Resultados de las últimas 24 h" y antes de la vigilancia.
    assert full.index("Resultados de las últimas 24 h") < full.index("Últimos 12 meses") < full.index("Vigilancia")

    # Si el bloque falla, el resumen sale igual con una línea "no disponible".
    def _boom(*a, **k):
        raise RuntimeError("fallo")
    monkeypatch.setattr(ds, "year_block", _boom)
    full2 = ds.build_daily_summary(now=NOW)
    assert "📆 <b>Últimos 12 meses</b>: no disponible (RuntimeError)" in full2
    assert "Vigilancia" in full2


def test_resumen_realista_cabe_sin_recortar(temp_db):
    """Con 300 señales resueltas en el año, 6 abiertas y 400 setups sombra, el resumen
    no pasa del límite propio (4.000): no se recorta la vigilancia ni los fills.
    (Con muchas señales NO entregadas ya se pasaba antes de la Fase 4 y se recorta.)"""
    from monitoring import daily_summary as ds
    rng = np.random.default_rng(7)
    for k in range(300):
        sym = ["BTCUSDT", "ETHUSDT", "XAUUSD"][k % 3]
        res = "win" if rng.random() < 0.45 else "loss"
        _ins(temp_db, sym, res, 1.5 if res == "win" else -1.0, float(rng.uniform(0, 364)))
    with temp_db.get_engine().begin() as conn:
        for k in range(6):
            conn.execute(text(
                "INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, stop_loss, "
                "take_profit, position_size, atr, setup_ts, mode, notified) VALUES (:s, :ts, 'breakout', "
                "'long', 0.6, 100, 98, 103, 1000, 2, :ts, 'live', 1)"),
                {"s": ["BTCUSDT", "ETHUSDT", "XAUUSD"][k % 3], "ts": NOW_U - 3600 * (k + 1)})
        for k in range(400):
            conn.execute(text(
                "INSERT INTO shadow_setups (symbol, setup_ts, direction, setup_type, raw_score, approved, "
                "motivo, entry, sl, tp, atr, created_ts) VALUES (:s, :t, 'long', 'breakout', 0.3, 0, :m, "
                "100, 98, 103, 2, :c)"),
                {"s": ["BTCUSDT", "ETHUSDT", "XAUUSD", "SPX500"][k % 4], "t": NOW_U - 3600 * k - 7,
                 "m": ["below_threshold", "not_fresh", "duplicate"][k % 3], "c": NOW_U - 3600 * (k % 20)})
    full = ds.build_daily_summary(now=NOW)
    _assert_valid_html(full)
    assert "Últimos 12 meses" in full
    assert len(full) < ds.TELEGRAM_MAX_CHARS, len(full)


# ---------------------------------------------------------------------------
# 4. Calendario: FOMC de 2027 completos (federalreserve.gov, 16-nov-2026)
# ---------------------------------------------------------------------------

def test_fomc_2027_completo_segun_la_fed():
    rows = list(csv.DictReader(open(ROOT / "data" / "econ_events.csv", encoding="utf-8")))
    fomc27 = sorted(r["date_utc"] for r in rows if r["event"] == "FOMC" and r["date_utc"].startswith("2027"))
    # Página oficial "Meeting calendars" (actualizada el 12-nov-2026): segundo día de cada reunión.
    assert fomc27 == ["2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09", "2027-07-28",
                      "2027-09-15", "2027-10-27", "2027-11-17"]
    assert not any(r["date_utc"].startswith("2027-12") and r["event"] == "FOMC" for r in rows)
    # 14:00 ET → 18:00 UTC en horario de verano, 19:00 UTC en invierno (desde el 7-nov-2027).
    times = {r["date_utc"]: r["time_utc"] for r in rows if r["event"] == "FOMC"}
    assert times["2027-10-27"] == "18:00" and times["2027-11-17"] == "19:00"
    regl = (ROOT / "REGLAS_DE_DECISION.md").read_text(encoding="utf-8")
    assert "añadir el calendario de 2028 (Fed y BLS) antes de octubre de 2027" in regl


# ---------------------------------------------------------------------------
# 5. Monitor de datos: avisos pendientes persistentes en kv
# ---------------------------------------------------------------------------

@pytest.fixture()
def fresh_monitor(monkeypatch):
    import monitoring.data_monitor as dm
    monkeypatch.setattr(dm, "_alert_state", {})
    monkeypatch.setattr(dm, "_alerted_keys", set())
    monkeypatch.setattr(dm, "_pending", {})
    monkeypatch.setattr(dm, "_state_loaded", False)
    return dm


def _restart(dm, monkeypatch):
    """Simula un reinicio del worker: se borra todo lo que vivía en memoria."""
    monkeypatch.setattr(dm, "_alert_state", {})
    monkeypatch.setattr(dm, "_alerted_keys", set())
    monkeypatch.setattr(dm, "_pending", {})
    monkeypatch.setattr(dm, "_state_loaded", False)


def _flat_btc(end="2026-06-16 16:00", price=65835.3):
    idx = pd.date_range(end=pd.Timestamp(end, tz=UTC), periods=300, freq="4h")
    rng = np.random.default_rng(1)
    close = 60000 * np.exp(np.cumsum(rng.normal(0, 0.004, len(idx))))
    open_ = np.r_[60000, close[:-1]]
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                       "low": np.minimum(open_, close) * 0.998, "close": close,
                       "volume": 100.0}, index=idx)
    df.attrs["source"] = "gateio"
    df.loc[idx[-1], ["open", "high", "low", "close"]] = price
    df.loc[idx[-1], "volume"] = 0.0
    return df


def test_aviso_pendiente_sobrevive_al_reinicio(temp_db, fresh_monitor, monkeypatch):
    dm = fresh_monitor
    import notifications.telegram as tg
    calls, results = [], [False, True]
    monkeypatch.setattr(tg, "send_data_quality_alert",
                        lambda s, items: calls.append((s, list(items))) or results.pop(0))
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    df = _flat_btc()
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0)
    assert len(calls) == 1 and dm._pending                       # falló: pendiente
    st = json.loads(temp_db.kv_get(dm.KV_ALERTS))
    assert len(st["pending"]) == 1 and st["pending"][0]["symbol"] == "BTCUSDT"
    assert st["pending"][0]["since"] == int(t0.timestamp())

    _restart(dm, monkeypatch)                                    # reinicio del worker
    # Ciclo siguiente: la vela ya NO es nueva en la BD; sin kv el aviso se perdía.
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=1))
    assert len(calls) == 2 and "65835.3" in calls[1][1][0]       # se reenvía tras el reinicio
    assert not dm._pending
    st = json.loads(temp_db.kv_get(dm.KV_ALERTS))
    assert st["pending"] == [] and st["last_alert"]["BTCUSDT"] == int((t0 + pd.Timedelta(hours=1)).timestamp())

    _restart(dm, monkeypatch)
    # Enviado: otro reinicio no lo repite.
    dm.run_cycle_checks("BTCUSDT", df, None, notify=True, now=t0 + pd.Timedelta(hours=2))
    assert len(calls) == 2


def test_throttling_de_6h_sobrevive_al_reinicio(temp_db, fresh_monitor, monkeypatch):
    dm = fresh_monitor
    import notifications.telegram as tg
    calls = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: calls.append(s) or True)
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    dm.run_cycle_checks("BTCUSDT", _flat_btc(), None, notify=True, now=t0)
    assert len(calls) == 1
    _restart(dm, monkeypatch)
    # Nueva incidencia grave 1 h después (otra vela plana): dentro de las 6 h → sin aviso.
    df2 = _flat_btc(end="2026-06-16 20:00", price=65900.0)
    dm.run_cycle_checks("BTCUSDT", df2, None, notify=True, now=t0 + pd.Timedelta(hours=1))
    assert len(calls) == 1
    assert dm._alert_state["BTCUSDT"] == t0                      # recuperado de kv


def test_pendiente_caduca_tras_24h_tambien_tras_reinicio(temp_db, fresh_monitor, monkeypatch):
    dm = fresh_monitor
    import notifications.telegram as tg
    calls = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: calls.append(s) and False)
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    dm.run_cycle_checks("BTCUSDT", _flat_btc(), None, notify=True, now=t0)
    assert len(calls) == 1
    _restart(dm, monkeypatch)
    later = _flat_btc(end="2026-06-17 20:00")
    later.loc[later.index[-1], ["open", "high", "low", "close", "volume"]] = [66000, 66100, 65900, 66050, 10]
    dm.run_cycle_checks("BTCUSDT", later, None, notify=True, now=t0 + pd.Timedelta(hours=25))
    assert len(calls) == 1 and not dm._pending                   # caducado: no se reintenta
    assert json.loads(temp_db.kv_get(dm.KV_ALERTS))["pending"] == []


def test_sin_bd_el_monitor_sigue_en_memoria(fresh_monitor, monkeypatch):
    """Si kv falla (BD caída), no se lanza nada: el aviso queda en memoria."""
    dm = fresh_monitor
    import data.database as db
    import notifications.telegram as tg

    def _down(*a, **k):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(db, "kv_get", _down)
    monkeypatch.setattr(db, "kv_set", _down)
    monkeypatch.setattr(dm, "record_incidents", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("BD caída")))
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: False)
    t0 = pd.Timestamp("2026-06-16 20:05", tz=UTC)
    out = dm.run_cycle_checks("BTCUSDT", _flat_btc(), None, notify=True, now=t0)
    assert out and dm._pending and dm._state_loaded is False


def test_estado_de_kv_corrupto_no_rompe(temp_db, fresh_monitor, monkeypatch):
    dm = fresh_monitor
    temp_db.kv_set(dm.KV_ALERTS, "{no es json")
    import notifications.telegram as tg
    calls = []
    monkeypatch.setattr(tg, "send_data_quality_alert", lambda s, items: calls.append(s) or True)
    dm.run_cycle_checks("BTCUSDT", _flat_btc(), None, notify=True,
                        now=pd.Timestamp("2026-06-16 20:05", tz=UTC))
    assert len(calls) == 1                                       # el aviso sale igual
    assert json.loads(temp_db.kv_get(dm.KV_ALERTS))["pending"] == []   # y kv queda sano
