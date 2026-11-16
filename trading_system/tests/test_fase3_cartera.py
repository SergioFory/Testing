"""
Pruebas de la Fase 3 — grupo B (punto 3.1): control de cartera EN SOMBRA.

  - risk/portfolio.py: reglas R1–R4 (funciones puras), cartera según el sistema
    desde la BD (señales live abiertas cuyo último fill no es /skip) y pérdida
    realizada del día.
  - Migración idempotente de la columna portfolio_flag (signals y shadow_setups).
  - Ciclo vivo en modo "shadow" (el activo): la señal SALE igual, con la marca y
    las líneas "Cartera según el sistema" / "⚠️ Con esta señal" en el mensaje.
  - Modo "enforce" (NO activado; se prueba aquí): la señal no sale, el setup queda
    en shadow_setups con motivo 'portfolio_R*' y no se reintenta tarde.
  - Resumen nocturno, /help y respuesta de /skip.

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post y Session.request fallan si alguien intenta usarlos;
Telegram se sustituye por un doble y la BD es SQLite temporal.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
def capital_12k(monkeypatch):
    """Capital de 12.000 USD: los dobles de prueba (riesgo 30 USD) arriesgan el 0,25 %."""
    import config.settings as st
    monkeypatch.setattr(st, "TRADING_CAPITAL", 12_000.0)
    return 12_000.0


def _rows(db, table, where="1=1"):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table} WHERE {where} ORDER BY id"))
                .mappings().fetchall()]


IDX = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")


class _Trade:
    """Doble de TradeSetup: nominal 1.000 USD y stop al 3 % → riesgo 30 USD."""
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


def _open(symbol, direction="long", entry=200.0, setup_i=5, mode="live"):
    """Guarda una señal abierta (stop al 3 % de la entrada) y devuelve su id."""
    from data.outcome_tracker import save_signal_from_trade
    sl = entry * (0.97 if direction == "long" else 1.03)
    tp = entry * (1.045 if direction == "long" else 0.955)
    sid = save_signal_from_trade(_Trade(symbol, direction, entry, sl, tp), setup_ts=IDX[setup_i],
                                 mode=mode)
    assert sid > 0
    return sid


def _setup(ts, direction="long", raw=0.7, close=100.0, atr=2.0):
    sign = 1 if direction == "long" else -1
    return {"ts": ts, "symbol": "X", "setup_type": "breakout", "direction": direction,
            "raw_score": raw, "atr": atr, "close": close,
            "stop_loss": close - sign * 1.5 * atr, "take_profit": close + sign * 2.25 * atr}


def _pipeline(monkeypatch, setups, symbol="BTCUSDT"):
    """cmd_signal con datos y detección simulados (get_best_setup REAL)."""
    import main
    import signals.detector as det
    import ml.predictor as pred
    import risk.position_sizer as ps
    import data.bars as bars
    import monitoring.data_monitor as dm

    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=IDX)
    df.attrs["source"] = "test"
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    frame = pd.DataFrame(setups)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: frame.copy())
    monkeypatch.setattr(pred, "score_setups",
                        lambda d, *a, **k: d.assign(ml_score=d["raw_score"],
                                                    ml_approved=d["raw_score"] >= 0.55))
    monkeypatch.setattr(ps, "build_trade_setup",
                        lambda row, capital, score, **k: _Trade(symbol, row["direction"],
                                                                float(row["close"]),
                                                                float(row["stop_loss"]),
                                                                float(row["take_profit"])))
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    return main


def _shadow_by_key(db, symbol=None):
    out = {}
    for r in _rows(db, "shadow_setups"):
        if symbol is None or r["symbol"] == symbol:
            out[(pd.Timestamp(r["setup_ts"], unit="s", tz="UTC"), r["direction"])] = r
    return out


R = 0.0025   # riesgo por posición en las pruebas puras


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

def test_settings_modo_sombra_y_reglas():
    import config.settings as st
    from risk.portfolio import control_mode
    assert st.PORTFOLIO_CONTROL_MODE == "shadow"          # NO se activa en esta fase
    assert control_mode() == "shadow"
    rules = st.PORTFOLIO_RULES
    assert rules["max_open_positions"] == 3
    assert rules["no_opposite_same_symbol"] is True
    grp = rules["correlated_groups"][0]
    assert set(grp["symbols"]) == {"BTCUSDT", "ETHUSDT"} and grp["max_risk_pct"] == pytest.approx(0.005)
    assert rules["max_daily_realized_loss_pct"] == pytest.approx(0.03)


def test_modo_desconocido_se_trata_como_sombra(monkeypatch):
    import config.settings as st
    from risk.portfolio import control_mode
    monkeypatch.setattr(st, "PORTFOLIO_CONTROL_MODE", "ENFORCE")
    assert control_mode() == "enforce"
    monkeypatch.setattr(st, "PORTFOLIO_CONTROL_MODE", "bloquear")
    assert control_mode() == "shadow"


# ---------------------------------------------------------------------------
# Reglas puras
# ---------------------------------------------------------------------------

def _pos(i, symbol, direction="long", risk=R):
    return {"id": i, "symbol": symbol, "direction": direction, "risk_pct": risk}


def _new(symbol="BTCUSDT", direction="long", risk=R):
    return {"symbol": symbol, "direction": direction, "risk_pct": risk}


def test_sin_posiciones_no_incumple_nada():
    from risk.portfolio import evaluate_portfolio_rules, flag_of
    assert evaluate_portfolio_rules(_new(), []) == []
    assert flag_of([]) == "ok"


def test_r1_maximo_tres_abiertas():
    from risk.portfolio import evaluate_portfolio_rules
    two = [_pos(1, "XAUUSD"), _pos(2, "ETHUSDT")]
    assert [b["rule"] for b in evaluate_portfolio_rules(_new("SPX500"), two)] == []
    three = two + [_pos(3, "SPX500")]
    out = evaluate_portfolio_rules(_new("XAUUSD"), three)
    assert [b["rule"] for b in out] == ["R1"]
    assert "serían 4 posiciones abiertas" in out[0]["text"] and "máximo 3" in out[0]["text"]


def test_r2_long_y_short_mismo_activo():
    from risk.portfolio import evaluate_portfolio_rules
    out = evaluate_portfolio_rules(_new("XAUUSD", "short"), [_pos(12, "XAUUSD", "long")])
    assert [b["rule"] for b in out] == ["R2"]
    assert "LONG abierta en XAU (#12)" in out[0]["text"] and "SHORT" in out[0]["text"]
    # Misma dirección en el mismo activo: no es R2. Otro activo: tampoco.
    assert evaluate_portfolio_rules(_new("XAUUSD", "long"), [_pos(12, "XAUUSD", "long")]) == []
    assert evaluate_portfolio_rules(_new("SPX500", "short"), [_pos(12, "XAUUSD", "long")]) == []


def test_r3_grupo_btc_eth_con_tolerancia():
    from risk.portfolio import evaluate_portfolio_rules
    # Una cripto abierta + la nueva = 0,50 % → permitido (con ruido de redondeo).
    assert evaluate_portfolio_rules(_new("ETHUSDT"), [_pos(1, "BTCUSDT", risk=0.0025000004)]) == []
    # Dos cripto abiertas + la nueva = 0,75 % → R3.
    out = evaluate_portfolio_rules(_new("BTCUSDT", "short"),
                                   [_pos(1, "BTCUSDT", "short"), _pos(2, "ETHUSDT", "short")])
    assert [b["rule"] for b in out] == ["R3"]
    assert "BTC+ETH pasaría a 0,75 % del capital (máximo 0,50 %)" in out[0]["text"]
    # Posiciones antiguas con más riesgo (tamaño anterior a la Fase 3): una sola basta.
    assert [b["rule"] for b in evaluate_portfolio_rules(_new("ETHUSDT"), [_pos(1, "BTCUSDT", risk=0.004)])] == ["R3"]
    # Un activo fuera del grupo no suma ni se limita.
    assert evaluate_portfolio_rules(_new("XAUUSD"), [_pos(1, "BTCUSDT"), _pos(2, "ETHUSDT")]) == []


def test_r4_perdida_del_dia():
    from risk.portfolio import evaluate_portfolio_rules
    assert evaluate_portfolio_rules(_new(), [], day_loss_pct=0.0299) == []
    out = evaluate_portfolio_rules(_new(), [], day_loss_pct=0.031)
    assert [b["rule"] for b in out] == ["R4"]
    assert "3,10 % del capital" in out[0]["text"] and "límite 3,00 %" in out[0]["text"]
    assert evaluate_portfolio_rules(_new(), [], day_loss_pct=-0.05) == []   # ganancia


def test_varias_reglas_en_orden_y_flag():
    from risk.portfolio import evaluate_portfolio_rules, flag_of, PortfolioCheck
    opens = [_pos(1, "BTCUSDT", "long"), _pos(2, "ETHUSDT", "short"), _pos(3, "XAUUSD")]
    out = evaluate_portfolio_rules(_new("ETHUSDT", "long"), opens, day_loss_pct=0.05)
    assert [b["rule"] for b in out] == ["R1", "R2", "R3", "R4"]
    assert flag_of(out) == "R1,R2,R3,R4"
    chk = PortfolioCheck(open_positions=opens, broken=out, mode="shadow")
    assert chk.flag == "R1,R2,R3,R4" and chk.motivo == "portfolio_R1" and chk.blocks is False
    assert PortfolioCheck(broken=out, mode="enforce").blocks is True
    assert PortfolioCheck(broken=[], mode="enforce").blocks is False


def test_reglas_desactivables():
    from risk.portfolio import evaluate_portfolio_rules
    opens = [_pos(1, "BTCUSDT", "short"), _pos(2, "ETHUSDT"), _pos(3, "XAUUSD")]
    assert evaluate_portfolio_rules(_new("BTCUSDT"), opens, rules={"no_opposite_same_symbol": False}) == []


def test_riesgo_desde_el_tamano_y_textos():
    from risk.portfolio import risk_pct_of, trade_info, positions_text, message_lines, PortfolioCheck
    # 1.000 USD de nominal con stop al 3 % = 30 USD = 0,25 % de 12.000 USD.
    assert risk_pct_of(1000, 100, 97, 12_000) == pytest.approx(0.0025)
    assert risk_pct_of(1000, 0, 97, 12_000) is None and risk_pct_of(None, 1, 1, 1) is None
    assert trade_info(_Trade("ETHUSDT", "short", 100, 103, 95.5), capital=12_000)["risk_pct"] == pytest.approx(0.0025)
    assert positions_text([]) == "ninguna posición abierta"
    many = [_pos(i, "BTCUSDT" if i % 2 else "XAUUSD", "long") for i in range(1, 8)]
    txt = positions_text(many)
    assert txt.startswith("7 abiertas (BTC long #1, XAU") and txt.endswith("y 2 más)")
    chk = PortfolioCheck(open_positions=[_pos(12, "BTCUSDT")],
                         broken=[{"rule": "R3", "text": "x < y"}], mode="shadow")
    lines = message_lines(chk)
    assert lines[0] == ("Cartera según el sistema", "1 abierta (BTC long #12)")
    assert lines[1] == ("⚠️ Con esta señal", "x < y (control de cartera en pruebas: no bloquea)")
    assert message_lines(None) == []


# ---------------------------------------------------------------------------
# Cartera desde la BD
# ---------------------------------------------------------------------------

def test_migracion_portfolio_flag_idempotente(temp_db):
    with temp_db.get_engine().connect() as conn:
        for table in ("signals", "shadow_setups"):
            cols = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
            assert "portfolio_flag" in cols, table
    temp_db.init_db(force=True)
    temp_db.init_db(force=True)


def test_migracion_sobre_tabla_sombra_antigua(tmp_path, monkeypatch):
    """Una BD de la Fase 1 (shadow_setups sin la columna) se migra sin perder filas."""
    import sqlalchemy
    url = f"sqlite:///{tmp_path / 'old.db'}"
    eng = sqlalchemy.create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE shadow_setups (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                          "symbol TEXT NOT NULL, setup_ts INTEGER NOT NULL, direction TEXT NOT NULL, "
                          "setup_type TEXT, raw_score REAL, approved INTEGER, motivo TEXT NOT NULL, "
                          "entry REAL, sl REAL, tp REAL, atr REAL, created_ts INTEGER, resultado TEXT, "
                          "pnl_r REAL, exit_price REAL, resolved_ts INTEGER, "
                          "UNIQUE(symbol, setup_ts, direction))"))
        conn.execute(text("INSERT INTO shadow_setups (symbol, setup_ts, direction, motivo) "
                          "VALUES ('BTCUSDT', 1, 'long', 'emitted')"))
    eng.dispose()
    monkeypatch.setenv("DATABASE_URL", url)
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    rows = _rows(db, "shadow_setups")
    assert len(rows) == 1 and "portfolio_flag" in rows[0] and rows[0]["portfolio_flag"] is None
    db._engine.dispose()


def test_load_open_positions_universo(temp_db, capital_12k):
    from data.database import save_fill
    from risk.portfolio import load_open_positions
    a = _open("BTCUSDT", "long", entry=200.0, setup_i=1)            # abierta
    b = _open("ETHUSDT", "short", entry=300.0, setup_i=2)           # /skip → fuera
    c = _open("XAUUSD", "long", entry=400.0, setup_i=3)             # /skip y luego /fill → dentro
    d = _open("XAUUSD", "short", entry=410.0, setup_i=4)            # /fill y luego /skip → fuera
    e = _open("SPX500", "long", entry=500.0, setup_i=5, mode="shadow")   # activo en sombra → fuera
    f = _open("BTCUSDT", "short", entry=600.0, setup_i=6)           # resuelta → fuera
    g = _open("ETHUSDT", "long", entry=700.0, setup_i=7)            # quality_flag → fuera
    h = _open("ETHUSDT", "long", entry=800.0, setup_i=8)            # sin setup_ts → fuera
    save_fill(b, "skip", note="tarde", ts=1000, update_id=1)
    save_fill(c, "skip", ts=1000, update_id=2)
    save_fill(c, "fill", price=401.0, ts=2000, update_id=3)
    save_fill(d, "fill", price=409.0, ts=1000, update_id=4)
    save_fill(d, "skip", ts=2000, update_id=5)
    with temp_db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET resultado_real='loss', pnl_r=-1 WHERE id=:i"), {"i": f})
        conn.execute(text("UPDATE signals SET quality_flag='reemision' WHERE id=:i"), {"i": g})
        conn.execute(text("UPDATE signals SET setup_ts=NULL WHERE id=:i"), {"i": h})
    opens = load_open_positions()
    assert [p["id"] for p in opens] == [a, c]
    assert all(p["risk_pct"] == pytest.approx(0.0025) for p in opens)
    assert e not in [p["id"] for p in opens]
    # `now` anterior a todas: nada abierto todavía.
    assert load_open_positions(now=pd.Timestamp("2020-01-01", tz="UTC")) == []


def test_skip_por_telegram_saca_la_senal_de_la_cartera(temp_db, capital_12k, no_network):
    from notifications.telegram_commands import handle_command
    from risk.portfolio import load_open_positions
    sid = _open("BTCUSDT", "long")
    assert [p["id"] for p in load_open_positions()] == [sid]
    reply = handle_command("skip", [str(sid), "no", "llegué"], update_id=77)
    assert "Sale de la cartera del sistema" in reply
    assert load_open_positions() == []
    handle_command("fill", [str(sid), "200"], update_id=78)    # la ejecutó al final
    assert [p["id"] for p in load_open_positions()] == [sid]


def test_realized_loss_today(temp_db, capital_12k):
    from data.database import save_fill
    from risk.portfolio import realized_loss_today
    now = pd.Timestamp("2026-11-16 15:00", tz="UTC")
    today = int(pd.Timestamp("2026-11-16 09:00", tz="UTC").timestamp())
    yesterday = int(pd.Timestamp("2026-11-15 23:00", tz="UTC").timestamp())
    s1 = _open("BTCUSDT", "long", entry=200.0, setup_i=1)     # hoy, −1R → 0,25 %
    s2 = _open("ETHUSDT", "long", entry=300.0, setup_i=2)     # hoy, −2R → 0,50 %
    s3 = _open("XAUUSD", "long", entry=400.0, setup_i=3)      # ayer → no cuenta
    s4 = _open("BTCUSDT", "short", entry=500.0, setup_i=4)    # hoy pero /skip → no cuenta
    s5 = _open("ETHUSDT", "short", entry=600.0, setup_i=5)    # hoy, +1,5R → −0,375 %
    s6 = _open("XAUUSD", "short", entry=700.0, setup_i=6)     # cierre manual sin precio → sin R
    with temp_db.get_engine().begin() as conn:
        for sid, r, ts in ((s1, -1.0, today), (s2, -2.0, today), (s3, -5.0, yesterday),
                           (s4, -1.0, today), (s5, 1.5, today)):
            conn.execute(text("UPDATE signals SET resultado_real=:o, pnl_r=:r, resolved_ts=:t "
                              "WHERE id=:i"), {"o": "loss" if r < 0 else "win", "r": r, "t": ts, "i": sid})
        conn.execute(text("UPDATE signals SET resultado_real='loss', resolved_by='manual', "
                          "resolved_ts=:t WHERE id=:i"), {"t": today, "i": s6})
    save_fill(s4, "skip", ts=today, update_id=9)
    assert realized_loss_today(now=now) == pytest.approx(0.0025 + 0.005 - 0.00375)


def test_senales_que_no_llegaron_no_cuentan_en_la_cartera(temp_db, capital_12k):
    """Revisión adversarial Fase 3: una señal live con notified = 0 (Telegram falló)
    no es una posición del usuario: fuera de la cartera, de R3 y del texto; si el
    reintento la entrega (notified = 1) o el usuario manda /fill, vuelve a contar.
    notified NULL (filas antiguas) cuenta."""
    from data.database import mark_signal_notified, save_fill
    from risk.portfolio import check_new_signal, load_open_positions, positions_text
    btc = _open("BTCUSDT", "long", entry=200.0, setup_i=1)
    eth = _open("ETHUSDT", "long", entry=300.0, setup_i=2)
    old = _open("XAUUSD", "long", entry=400.0, setup_i=3)        # notified NULL → cuenta
    for sid in (btc, eth):
        assert mark_signal_notified(sid, False)
    assert [p["id"] for p in load_open_positions()] == [old]
    chk = check_new_signal(_Trade("BTCUSDT", "long"))
    assert chk.flag == "ok"                                     # sin R3 por posiciones fantasma
    assert positions_text(chk.open_positions) == f"1 abierta (XAU long #{old})"
    # El reintento entrega BTC → entra; ETH con /fill (la ejecutó igualmente) → entra.
    assert mark_signal_notified(btc, True)
    save_fill(eth, "fill", price=300.0, ts=1000, update_id=31)
    assert [p["id"] for p in load_open_positions()] == [btc, eth, old]
    assert check_new_signal(_Trade("BTCUSDT", "long")).flag == "R1,R3"
    # /skip después del /fill vuelve a sacarla.
    save_fill(eth, "skip", ts=2000, update_id=32)
    assert [p["id"] for p in load_open_positions()] == [btc, old]


def test_perdida_del_dia_ignora_senales_que_no_llegaron(temp_db, capital_12k):
    from data.database import mark_signal_notified, save_fill
    from risk.portfolio import realized_loss_today
    now = pd.Timestamp("2026-11-16 15:00", tz="UTC")
    today = int(pd.Timestamp("2026-11-16 09:00", tz="UTC").timestamp())
    s1 = _open("BTCUSDT", "long", entry=200.0, setup_i=1)     # entregada, −1R → 0,25 %
    s2 = _open("ETHUSDT", "long", entry=300.0, setup_i=2)     # no llegó, −2R → no cuenta
    s3 = _open("XAUUSD", "long", entry=400.0, setup_i=3)      # no llegó pero /fill, −1R → cuenta
    for sid in (s2, s3):
        assert mark_signal_notified(sid, False)
    assert mark_signal_notified(s1, True)
    save_fill(s3, "fill", price=400.0, ts=today - 60, update_id=41)
    with temp_db.get_engine().begin() as conn:
        for sid, r in ((s1, -1.0), (s2, -2.0), (s3, -1.0)):
            conn.execute(text("UPDATE signals SET resultado_real='loss', pnl_r=:r, resolved_ts=:t "
                              "WHERE id=:i"), {"r": r, "t": today, "i": sid})
    assert realized_loss_today(now=now) == pytest.approx(0.0025 + 0.0025)


def test_resumen_nocturno_cartera_sin_senales_que_no_llegaron(temp_db, capital_12k):
    """El resumen no puede decir a la vez 'no te llegaron 3' y 'abiertas: 3' con esas mismas."""
    import time
    from data.database import mark_signal_notified
    from monitoring.daily_summary import portfolio_block
    ids = [_open(s, "short", entry=e, setup_i=i)
           for i, (s, e) in enumerate((("BTCUSDT", 200.0), ("ETHUSDT", 300.0), ("XAUUSD", 400.0)), 1)]
    for sid in ids:
        assert mark_signal_notified(sid, False)
    with temp_db.get_engine().connect() as conn:
        txt = "\n".join(portfolio_block(conn, int(time.time())))
    assert "Abiertas: ninguna posición abierta" in txt
    assert "Cuentan las señales que te llegaron" in txt


# ---------------------------------------------------------------------------
# Ciclo vivo: modo sombra (el activo)
# ---------------------------------------------------------------------------

def test_sombra_emite_igual_con_marca_y_mensaje(temp_db, fake_telegram, capital_12k, monkeypatch):
    b = _open("BTCUSDT", "long", entry=200.0, setup_i=1)
    e = _open("ETHUSDT", "long", entry=300.0, setup_i=2)
    x = _open("XAUUSD", "short", entry=400.0, setup_i=3)
    t1 = IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)

    assert len(fake_telegram["sent"]) == 1                    # la señal SALE
    msg = fake_telegram["sent"][0]
    assert (f"<b>Cartera según el sistema:</b> 3 abiertas (BTC long #{b}, ETH long #{e}, "
            f"XAU short #{x})") in msg
    warn = [ln for ln in msg.splitlines() if "Con esta señal" in ln]
    assert len(warn) == 1
    assert "serían 4 posiciones abiertas a la vez (máximo 3)" in warn[0]
    assert "BTC+ETH pasaría a 0,75 % del capital (máximo 0,50 %)" in warn[0]
    assert "(control de cartera en pruebas: no bloquea)" in warn[0]
    # La línea va antes del pie con el ID y del cierre por tiempo.
    assert msg.index("Cartera según el sistema") < msg.index("Cerrar a mercado") < msg.index("/fill")

    sig = _rows(temp_db, "signals", "setup_ts = %d" % int(t1.timestamp()))[0]
    assert sig["portfolio_flag"] == "R1,R3" and sig["mode"] == "live" and sig["notified"] == 1
    sh = _shadow_by_key(temp_db)[(t1, "long")]
    assert sh["motivo"] == "emitted" and sh["portfolio_flag"] == "R1,R3"


def test_sombra_sin_posiciones_flag_ok(temp_db, fake_telegram, capital_12k, monkeypatch):
    t0, t1 = IDX[-2], IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t0, "short", raw=0.40), _setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)
    msg = fake_telegram["sent"][0]
    assert "<b>Cartera según el sistema:</b> ninguna posición abierta" in msg
    assert "Con esta señal" not in msg
    assert _rows(temp_db, "signals")[0]["portfolio_flag"] == "ok"
    sh = _shadow_by_key(temp_db)
    assert sh[(t1, "long")]["portfolio_flag"] == "ok"
    assert sh[(t0, "short")]["portfolio_flag"] is None        # no elegido: sin marca


def test_fallo_del_control_no_tumba_la_senal(temp_db, fake_telegram, capital_12k, monkeypatch):
    import risk.portfolio as pf

    def _boom(*a, **k):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(pf, "check_new_signal", _boom)
    t1 = IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == 1
    assert "Cartera según el sistema" not in fake_telegram["sent"][0]
    assert _rows(temp_db, "signals")[0]["portfolio_flag"] is None
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "emitted"


def test_mensaje_sale_aunque_falle_el_bloque_de_cartera(monkeypatch):
    import risk.portfolio as pf
    from notifications.telegram import format_signal_alert

    def _boom(*a, **k):
        raise ValueError("raro")
    monkeypatch.setattr(pf, "message_lines", _boom)
    msg = format_signal_alert(_Trade(), capital=12_000, portfolio=pf.PortfolioCheck())
    assert "SEÑAL — BTCUSDT" in msg and "Riesgo real" in msg and "Cartera" not in msg


def test_mensaje_escapa_textos_de_cartera():
    from notifications.telegram import format_signal_alert
    from risk.portfolio import PortfolioCheck
    chk = PortfolioCheck(open_positions=[], broken=[{"rule": "R1", "text": "<b>x</b> & y"}])
    msg = format_signal_alert(_Trade(), capital=12_000, portfolio=chk)
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in msg


# ---------------------------------------------------------------------------
# Modo enforce (probado, NO activado)
# ---------------------------------------------------------------------------

def test_enforce_bloquea_y_no_reintenta_tarde(temp_db, fake_telegram, capital_12k, monkeypatch):
    import config.settings as st
    monkeypatch.setattr(st, "PORTFOLIO_CONTROL_MODE", "enforce")
    _open("BTCUSDT", "short", entry=200.0, setup_i=1)
    _open("ETHUSDT", "short", entry=300.0, setup_i=2)
    t1 = IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)

    assert fake_telegram["sent"] == []                         # no sale
    assert len(_rows(temp_db, "signals")) == 2                 # no se guarda como señal
    sh = _shadow_by_key(temp_db)[(t1, "long")]
    assert sh["motivo"] == "portfolio_R2" and sh["portfolio_flag"] == "R2,R3" and sh["approved"] == 1

    # Se cierran las posiciones; el ciclo siguiente NO emite el setup bloqueado (llegaría tarde).
    with temp_db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET resultado_real='loss', pnl_r=-1"))
    main.cmd_signal("BTCUSDT", notify=True)
    assert fake_telegram["sent"] == []
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "portfolio_R2"


def test_enforce_deja_salir_otro_setup_y_las_que_cumplen(temp_db, fake_telegram, capital_12k, monkeypatch):
    import config.settings as st
    monkeypatch.setattr(st, "PORTFOLIO_CONTROL_MODE", "enforce")
    _open("BTCUSDT", "short", entry=200.0, setup_i=1)
    t0, t1 = IDX[-2], IDX[-1]
    # El mejor (long, contrario a la abierta) se bloquea por R2; en el ciclo siguiente
    # sale el otro setup aprobado de la ventana (short, mismo sentido: cumple).
    main = _pipeline(monkeypatch, [_setup(t0, "short", raw=0.6, close=100.5),
                                   _setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)
    assert fake_telegram["sent"] == []
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "portfolio_R2"
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == 1
    sh = _shadow_by_key(temp_db)
    assert sh[(t0, "short")]["motivo"] == "emitted" and sh[(t0, "short")]["portfolio_flag"] == "ok"
    assert sh[(t1, "long")]["motivo"] == "portfolio_R2"


def test_enforce_no_afecta_a_activos_en_sombra(temp_db, fake_telegram, capital_12k, monkeypatch):
    import config.settings as st
    monkeypatch.setattr(st, "PORTFOLIO_CONTROL_MODE", "enforce")
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT", "XAUUSD")):
        _open(sym, "long", entry=200.0 + 100 * i, setup_i=i + 1)
    t1 = IDX[-1]
    main = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)], symbol="SPX500")
    main.cmd_signal("SPX500", notify=True)
    sig = _rows(temp_db, "signals", "symbol = 'SPX500'")
    assert len(sig) == 1 and sig[0]["mode"] == "shadow" and sig[0]["portfolio_flag"] == "R1"
    assert _shadow_by_key(temp_db, "SPX500")[(t1, "long")]["motivo"] == "shadow_asset"
    assert fake_telegram["sent"] == []


def test_motivos_portfolio_registrables_y_definitivos(temp_db):
    from data.shadow import MOTIVOS, FINAL_MOTIVOS, record_shadow_setups
    for rid in ("R1", "R2", "R3", "R4"):
        assert f"portfolio_{rid}" in MOTIVOS and f"portfolio_{rid}" in FINAL_MOTIVOS
    row = {"setup_ts": IDX[-1], "direction": "long", "setup_type": "breakout", "raw_score": 0.9,
           "approved": 1, "motivo": "portfolio_R3", "entry": 100.0, "sl": 97.0, "tp": 104.5,
           "atr": 2.0, "portfolio_flag": "R3"}
    assert record_shadow_setups("BTCUSDT", [row]) == 1
    # Un ciclo posterior que lo vea como "not_best" no lo sobrescribe.
    record_shadow_setups("BTCUSDT", [dict(row, motivo="not_best", portfolio_flag=None)])
    r = _rows(temp_db, "shadow_setups")[0]
    assert r["motivo"] == "portfolio_R3" and r["portfolio_flag"] == "R3"


def test_blocked_setup_keys(temp_db):
    from data.shadow import record_shadow_setups
    from risk.portfolio import blocked_setup_keys
    base = {"direction": "long", "setup_type": "breakout", "raw_score": 0.9, "approved": 1,
            "entry": 100.0, "sl": 97.0, "tp": 104.5, "atr": 2.0}
    record_shadow_setups("BTCUSDT", [dict(base, setup_ts=IDX[-1], motivo="portfolio_R1"),
                                     dict(base, setup_ts=IDX[-2], motivo="not_best"),
                                     dict(base, setup_ts=IDX[-10], motivo="portfolio_R2")])
    assert blocked_setup_keys("BTCUSDT", IDX[-3]) == {(IDX[-1], "long")}
    assert blocked_setup_keys("ETHUSDT", IDX[0]) == set()


# ---------------------------------------------------------------------------
# Resumen nocturno y /help
# ---------------------------------------------------------------------------

def test_resumen_nocturno_bloque_cartera(temp_db, capital_12k, monkeypatch):
    from data.shadow import record_shadow_setups
    from monitoring.daily_summary import portfolio_block, build_daily_summary
    import time
    now_u = int(time.time())
    a = _open("BTCUSDT", "long", entry=200.0, setup_i=1)
    b = _open("ETHUSDT", "long", entry=300.0, setup_i=2)
    old = _open("XAUUSD", "long", entry=400.0, setup_i=3)
    with temp_db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET portfolio_flag='R3' WHERE id=:i"), {"i": b})
        conn.execute(text("UPDATE signals SET portfolio_flag='R1,R3', resultado_real='win', "
                          "pnl_r=1.5, ts=:t WHERE id=:i"), {"i": old, "t": now_u - 10 * 86400})
    record_shadow_setups("BTCUSDT", [{"setup_ts": IDX[-1], "direction": "short", "setup_type": "breakout",
                                      "raw_score": 0.9, "approved": 1, "motivo": "portfolio_R2",
                                      "entry": 100.0, "sl": 103.0, "tp": 95.5, "atr": 2.0,
                                      "portfolio_flag": "R2"}])
    with temp_db.get_engine().connect() as conn:
        lines = portfolio_block(conn, now_u)
    txt = "\n".join(lines)
    assert "Cartera según el sistema" in txt and "en pruebas: no bloquea" in txt
    assert f"2 abiertas (BTC long #{a}, ETH long #{b})" in txt
    assert "riesgo abierto 0,50 % del capital (BTC+ETH: 0,50 %)" in txt
    assert "evaluadas: 2 / 3" in txt
    assert "R1 (máx. 3 posiciones abiertas): 0 / 1" in txt
    assert "R2 (LONG y SHORT a la vez en el mismo activo): 1 / 1" in txt
    assert "R3 (riesgo del grupo BTC+ETH &gt; 0,50 %): 1 / 2" in txt
    assert "R4 (pérdida del día ≥ 3 %): 0 / 0" in txt
    full = build_daily_summary(symbols=[])
    assert "Cartera según el sistema" in full


def test_resumen_nocturno_sigue_si_falla_la_cartera(temp_db, monkeypatch):
    import risk.portfolio as pf
    from monitoring.daily_summary import build_daily_summary

    def _boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(pf, "load_open_positions", _boom)
    out = build_daily_summary(symbols=[])
    assert "Cartera según el sistema</b>: no disponible" in out and "Resumen diario" in out


def test_help_explica_que_skip_saca_de_la_cartera():
    from notifications.telegram_commands import HELP_TEXT
    assert "/skip" in HELP_TEXT and "cartera del sistema" in HELP_TEXT
    assert "vuelve a meter" in HELP_TEXT
