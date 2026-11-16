"""
Pruebas de la Fase 3 — grupo A (puntos 0 y 3.2):
  - Punto 0: el bloqueo por eventos macro se quitó de la ruta de señales (main ya no
    llama a is_event_blackout y EVENT_FILTER desapareció de settings). Se mantienen
    el calendario, la línea de aviso del mensaje y los eventos del resumen nocturno.
  - 3.2: riesgo explícito del 0,25 % por operación (RISK["risk_per_trade"]) y tope de
    apalancamiento RISK["max_leverage"] = 1× en lugar del tope del 10 % de nominal;
    el mensaje muestra 2,50 USD de riesgo con 1.000 USD y solo dice algo más si el
    apalancamiento recorta el tamaño o el riesgo supera el tope de las reglas.

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post fallan si alguien intenta usarlos; Telegram se sustituye
por un doble y la BD es SQLite temporal.
"""
import inspect
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Fixtures
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


def _row(symbol, direction, entry, sl, tp):
    return pd.Series({"close": entry, "stop_loss": sl, "take_profit": tp,
                      "atr": abs(entry - sl) / 1.5, "direction": direction,
                      "symbol": symbol, "setup_type": "breakout"})


def _lines(msg, key):
    return [ln for ln in msg.splitlines() if key in ln]


# ---------------------------------------------------------------------------
# Punto 0 — sin bloqueo por eventos
# ---------------------------------------------------------------------------

def test_sin_event_filter_en_settings_ni_llamada_en_la_ruta_de_senales():
    import config.settings as settings
    import main
    assert not hasattr(settings, "EVENT_FILTER")
    src = inspect.getsource(main._emit_best_setup) + inspect.getsource(main.cmd_signal)
    assert "is_event_blackout" not in src and "event_blackout" not in src
    # El calendario y la utilidad siguen existiendo (aviso del mensaje, resumen, mediciones).
    import data.fetchers.econ_calendar as cal
    assert callable(cal.is_event_blackout) and callable(cal.events_between)
    assert cal.EVENTS_CSV.exists()


def test_senal_de_oro_con_evento_ahora_se_emite_y_avisa(temp_db, fake_telegram, monkeypatch):
    """cmd_signal real (datos y detección simulados): aunque haya un CPI "ahora", la
    señal de oro SALE, nadie consulta is_event_blackout y el mensaje lleva el aviso."""
    import main
    import config.settings as settings
    import signals.detector as det
    import ml.predictor as pred
    import data.bars as bars
    import data.fetchers.econ_calendar as cal
    import monitoring.data_monitor as dm

    idx = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=idx)
    df.attrs["source"] = "test"
    setup = {"ts": idx[-1], "symbol": "XAUUSD", "setup_type": "breakout", "direction": "long",
             "raw_score": 0.9, "atr": 2.0, "close": 100.0, "stop_loss": 97.0, "take_profit": 104.0}
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: pd.DataFrame([setup]))
    monkeypatch.setattr(pred, "score_setups",
                        lambda d, *a, **k: d.assign(ml_score=d["raw_score"],
                                                    ml_approved=d["raw_score"] >= 0.20))
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    monkeypatch.setattr(settings, "TRADING_CAPITAL", 1_000.0)

    def _no_llamar(*a, **k):
        raise AssertionError("el filtro de eventos se quitó: no debe consultarse")
    monkeypatch.setattr(cal, "is_event_blackout", _no_llamar)
    monkeypatch.setattr(cal, "fetch_calendar", _no_llamar)
    ahora = datetime.now(timezone.utc)
    monkeypatch.setattr(cal, "events_between",
                        lambda s, e, path=None: [{"currency": "USD", "impact": "High", "time": ahora,
                                                  "title": "CPI", "event": "CPI", "source": "test"}])

    main.cmd_signal("XAUUSD", notify=True)
    assert len(fake_telegram["sent"]) == 1
    msg = fake_telegram["sent"][0]
    assert len(_lines(msg, "Evento de alto impacto dentro del plazo")) == 1
    assert "solo aviso: no es motivo para no entrar" in msg
    assert "Riesgo real:</b> $2.50 = 0.25 % del capital" in msg
    with temp_db.get_engine().connect() as conn:
        motivos = [r[0] for r in conn.execute(text("SELECT motivo FROM shadow_setups"))]
        n_sig = conn.execute(text("SELECT COUNT(*) FROM signals")).scalar()
    assert motivos == ["emitted"] and n_sig == 1


def test_resumen_nocturno_sigue_listando_eventos_de_48h():
    from monitoring.daily_summary import events_block
    # FOMC 09-dic-2026 19:00 UTC y CPI 10-dic 13:30 UTC (calendario propio).
    now = int(pd.Timestamp("2026-12-08 22:00", tz="UTC").timestamp())
    txt = "\n".join(events_block(now))
    assert "FOMC 09-dic 19:00 UTC" in txt and "CPI 10-dic 13:30 UTC" in txt


# ---------------------------------------------------------------------------
# 3.2 — riesgo 0,25 % y apalancamiento máximo
# ---------------------------------------------------------------------------

def test_configuracion_de_riesgo():
    from config.settings import RISK, RULES_MAX_RISK_PCT
    assert RISK["risk_per_trade"] == 0.0025 == RULES_MAX_RISK_PCT
    assert RISK["max_leverage"] == 1.0


def test_tamano_por_defecto_arriesga_el_025_sin_tope_de_nominal():
    from risk.position_sizer import calculate_position_size
    # Stop al 3 %: 2,50 USD de riesgo → 83,33 USD de nominal (8,3 % del capital).
    assert calculate_position_size(1_000, 100.0, 97.0) == (83.33, 0.833333, 2.5)
    # Stop al 1 %: 250 USD de nominal (25 % del capital). Con el tope antiguo del
    # 10 % de nominal se habría quedado en 100 USD y 1,00 USD de riesgo (0,10 %).
    size, qty, risk = calculate_position_size(1_000, 100.0, 99.0)
    assert (size, risk) == (250.0, 2.5)
    # El riesgo no depende del activo ni del stop mientras no actúe el apalancamiento.
    for entry, sl in [(95_000.0, 93_000.0), (3_100.0, 3_180.0), (4_100.0, 3_990.0), (6_700.0, 6_610.0)]:
        assert calculate_position_size(1_000, entry, sl)[2] == 2.5


def test_tope_de_apalancamiento_actua_y_el_mensaje_lo_dice():
    from risk.position_sizer import calculate_position_size, build_trade_setup
    from notifications.telegram import format_signal_alert
    # Stop al 0,1 %: el 0,25 % pediría 2.500 USD de nominal (2,5×) → se recorta a 1.000.
    size, qty, risk, capped = calculate_position_size(1_000, 100.0, 99.9, return_capped=True)
    assert capped and size == 1_000.0 and risk == pytest.approx(1.0)
    assert calculate_position_size(1_000, 100.0, 97.0, return_capped=True)[3] is False
    t = build_trade_setup(_row("BTCUSDT", "long", 100.0, 99.9, 100.15), 1_000.0, 0.7)
    assert t.leverage_cap == 1.0 and t.position_size == 1_000.0
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz="UTC"), capital=1_000,
                              now=pd.Timestamp("2026-11-13 04:05", tz="UTC"))
    assert "Riesgo real:</b> $1.00 = 0.10 % del capital" in msg
    assert len(_lines(msg, "apalancamiento máximo (1×")) == 1
    assert "Para respetar" not in msg
    # Sin recorte, ni rastro de la línea.
    t2 = build_trade_setup(_row("BTCUSDT", "long", 100.0, 97.0, 104.5), 1_000.0, 0.7)
    assert t2.leverage_cap is None
    assert "apalancamiento" not in format_signal_alert(t2, capital=1_000)


def test_configuracion_anterior_se_reproduce_con_parametros():
    """1 % con tope del 10 % de nominal = risk_per_trade 0,01 y apalancamiento 0,1×
    (lo usan las pruebas antiguas de la línea "Para respetar el 0,25 %")."""
    from risk.position_sizer import calculate_position_size
    for entry, sl in [(100.0, 97.0), (4168.40, 4323.68), (100.0, 80.0)]:
        stop = abs(entry - sl) / entry
        old_size = min(1_000 * 0.01 / stop, 1_000 * 0.10)
        size, _, risk = calculate_position_size(1_000, entry, sl, risk_per_trade=0.01, max_leverage=0.10)
        assert size == pytest.approx(old_size, abs=0.01)
        assert risk == pytest.approx(old_size * stop, abs=0.01)


@pytest.mark.parametrize("symbol,direction,entry,sl,tp", [
    ("BTCUSDT", "short", 95_000.0, 97_100.0, 91_850.0),
    ("XAUUSD", "long", 4_080.0, 3_960.0, 4_240.0),
    ("ETHUSDT", "long", 3_100.0, 3_010.0, 3_280.0),
    ("SPX500", "long", 6_700.0, 6_600.0, 6_866.7),
])
@pytest.mark.parametrize("capital", [1_000.0, 1_234.0, 777.0, 10_000.0])
def test_mensaje_riesgo_025_sin_linea_para_respetar(symbol, direction, entry, sl, tp, capital):
    from risk.position_sizer import build_trade_setup
    from notifications.telegram import format_signal_alert
    t = build_trade_setup(_row(symbol, direction, entry, sl, tp), capital, 0.6)
    assert t.risk_usd == pytest.approx(capital * 0.0025, abs=0.005)
    assert t.leverage_cap is None
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-11-13", tz="UTC"), capital=capital,
                              now=pd.Timestamp("2026-11-13 23:00", tz="UTC"))
    assert "Para respetar" not in msg and "más que el tope" not in msg
    assert "0.25 % del capital <i>(dentro del tope de las reglas: 0.25 %)</i>" in msg
    nominal_pct = t.position_size / capital * 100
    assert f"de nominal, {nominal_pct:.2f} % del capital)" in msg
    if capital == 1_000.0:
        assert "Riesgo real:</b> $2.50 = 0.25 % del capital" in msg


def test_resultado_en_usd_sigue_siendo_tamano_por_porcentaje():
    from data.outcome_tracker import compute_result_fields
    from risk.position_sizer import build_trade_setup
    t = build_trade_setup(_row("BTCUSDT", "long", 100.0, 97.0, 104.5), 1_000.0, 0.7)
    f = compute_result_fields("long", 100.0, 97.0, 104.5, position_size=t.position_size)
    # Gana el TP (+4,5 %): pnl_usd = 83,33 × 4,5 % = 3,75 USD = 1,5R × 2,50 USD.
    assert f["pnl_usd"] == pytest.approx(3.75, abs=0.01) and f["pnl_r"] == pytest.approx(1.5)


def test_dashboard_usa_el_riesgo_de_la_configuracion():
    src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    block = src[src.index('"Riesgo por trade (%)"') - 300: src.index('"Riesgo por trade (%)"') + 300]
    assert "RULES_MAX_RISK_PCT" in block and 'RISK["risk_per_trade"]' in block
    assert "0.5, 3.0" not in src                     # el rango antiguo no admitía el 0,25 %
    assert "risk_per_trade=risk_pct" in src           # el slider cambia el tamaño de verdad


def test_resumen_nocturno_muestra_el_riesgo_por_operacion(monkeypatch):
    import config.settings as settings
    from monitoring.daily_summary import risk_line
    monkeypatch.setattr(settings, "TRADING_CAPITAL", 1_000.0)
    assert risk_line() == ("Riesgo por operación: 0.25 % del capital ($2.50 de $1,000); "
                           "nominal máx. 1× el capital")
