"""
Pruebas de la Fase 1 — grupo de operación (2026-11):
  punto 5: arranque sin reentreno, historia "live"/"full" en _load_data y copia
           OHLCV reducida y fuera de la ruta crítica;
  punto 6: notificación fiable (columna notified, reintento < 2 h, html.escape);
  punto 3: sentimiento apagado;
  punto 4 (parte): sin pestaña de backtest en el dashboard.

    cd trading_system && python -m pytest tests -q

Sin red: Telegram se sustituye por un doble (nunca se envía nada de verdad) y los
datos son sintéticos con una BD SQLite temporal.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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
    """Cualquier intento de red real hace fallar la prueba."""
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)


@pytest.fixture()
def fake_telegram(monkeypatch, no_network):
    """Sustituye el envío de Telegram. sent = lista de textos; ok controla el
    resultado del envío."""
    import notifications.telegram as tg
    state = {"ok": True, "sent": []}

    def _fake_send(text, parse_mode="HTML", retries=3):
        state["sent"].append(text)
        return state["ok"]
    monkeypatch.setattr(tg, "_send", _fake_send)
    monkeypatch.setattr(tg, "_stale_logged", set())
    return state


class _Trade:
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


def _row(db, sig_id):
    from sqlalchemy import text
    with db.get_engine().connect() as conn:
        return conn.execute(text("SELECT * FROM signals WHERE id=:i"), {"i": sig_id}).mappings().one()


def _set_ts(db, sig_id, ts):
    from sqlalchemy import text
    with db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET ts=:t WHERE id=:i"),
                     {"t": int(pd.Timestamp(ts).timestamp()), "i": sig_id})


# ---------------------------------------------------------------------------
# Punto 6 — notificación fiable
# ---------------------------------------------------------------------------

def test_migration_adds_notified_columns(temp_db):
    from data.database import _existing_columns
    with temp_db.get_engine().connect() as conn:
        cols = _existing_columns(conn, "signals", "sqlite")
    assert {"notified", "alert_text"} <= cols


def test_saved_unnotified_then_marked(temp_db):
    from data.outcome_tracker import save_signal_from_trade
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="<b>hola</b>")
    assert sid > 0
    r = _row(temp_db, sid)
    assert r["notified"] == 0 and r["alert_text"] == "<b>hola</b>"
    assert [x["id"] for x in temp_db.load_unnotified_signals()] == [sid]
    assert temp_db.mark_signal_notified(sid, True)
    assert _row(temp_db, sid)["notified"] == 1
    assert temp_db.load_unnotified_signals() == []


def test_shadow_and_legacy_rows_are_never_retried(temp_db):
    from data.outcome_tracker import save_signal_from_trade
    s1 = save_signal_from_trade(_Trade(entry=100), setup_ts=pd.Timestamp("2026-11-15", tz="UTC"),
                                mode="shadow", notified=False)
    s2 = save_signal_from_trade(_Trade(entry=200, sl=194, tp=209),
                                setup_ts=pd.Timestamp("2026-11-14", tz="UTC"), mode="live")
    assert s1 > 0 and s2 > 0
    assert _row(temp_db, s2)["notified"] is None      # registro sin envío (p. ej. dashboard)
    assert temp_db.load_unnotified_signals() == []


def test_retry_resends_recent_with_note_and_marks(temp_db, fake_telegram):
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="TEXTO ORIGINAL")
    _set_ts(temp_db, sid, "2026-11-15 10:00")
    res = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-15 11:30", tz="UTC"))
    assert res == {"resent": 1, "failed": 0, "stale": 0}
    assert len(fake_telegram["sent"]) == 1
    msg = fake_telegram["sent"][0]
    assert msg.startswith(tg.RESEND_NOTE) and "TEXTO ORIGINAL" in msg
    assert "Reenvío tras fallo de Telegram" in msg
    assert _row(temp_db, sid)["notified"] == 1
    # Ya marcada: el siguiente ciclo no la reenvía.
    tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-15 12:00", tz="UTC"))
    assert len(fake_telegram["sent"]) == 1


def test_retry_failure_keeps_pending(temp_db, fake_telegram):
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    fake_telegram["ok"] = False
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="X")
    _set_ts(temp_db, sid, "2026-11-15 10:00")
    res = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-15 10:30", tz="UTC"))
    assert res["failed"] == 1 and res["resent"] == 0
    assert _row(temp_db, sid)["notified"] == 0


def test_retry_skips_signals_older_than_two_hours(temp_db, fake_telegram):
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="X")
    _set_ts(temp_db, sid, "2026-11-15 08:00")
    res = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-15 10:00", tz="UTC"))
    assert res == {"resent": 0, "failed": 0, "stale": 1}
    assert fake_telegram["sent"] == []
    assert _row(temp_db, sid)["notified"] == 0         # queda constancia de que no llegó


def test_retry_without_alert_text_uses_fallback(temp_db, fake_telegram):
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    sid = save_signal_from_trade(_Trade(symbol="ETHUSDT", entry=3000, sl=2900, tp=3200),
                                 setup_ts=pd.Timestamp("2026-11-15 00:00", tz="UTC"),
                                 mode="live", notified=False)
    _set_ts(temp_db, sid, "2026-11-15 10:00")
    tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-15 10:10", tz="UTC"))
    msg = fake_telegram["sent"][0]
    assert msg.startswith(tg.RESEND_NOTE)
    assert "ETHUSDT" in msg and "Entrada" in msg and "Stop Loss" in msg
    assert _row(temp_db, sid)["notified"] == 1


def test_error_alert_escapes_html(fake_telegram):
    import notifications.telegram as tg
    tg.send_error_alert("BTC<x>", "fallo <Response [429]> & otro")
    msg = fake_telegram["sent"][0]
    assert "&lt;Response [429]&gt; &amp; otro" in msg and "BTC&lt;x&gt;" in msg
    assert "<Response" not in msg


def _patch_signal_pipeline(monkeypatch, sentiment_must_not_run=True):
    """cmd_signal con datos y detección simulados: siempre sale un setup nuevo."""
    import main
    import signals.detector as det
    import ml.predictor as pred
    import risk.position_sizer as ps
    import data.bars as bars
    import data.fetchers.econ_calendar as cal

    idx = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=idx)
    df.attrs["source"] = "gateio"
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    setup = pd.DataFrame([{"ts": idx[-1], "symbol": "BTCUSDT", "setup_type": "breakout",
                           "direction": "long", "raw_score": 0.7, "atr": 2.0, "close": 100.0}])
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: setup)
    monkeypatch.setattr(pred, "score_setups", lambda d, *a, **k: d.assign(ml_score=d["raw_score"]))
    monkeypatch.setattr(pred, "get_best_setup", lambda d, **k: d.iloc[0])
    monkeypatch.setattr(ps, "build_trade_setup", lambda *a, **k: _Trade())
    monkeypatch.setattr(cal, "is_event_blackout", lambda *a, **k: (False, ""))
    if sentiment_must_not_run:
        import sentiment.cryptopanic as sc

        def _no(*a, **k):
            raise AssertionError("el sentimiento está apagado y no debe consultarse")
        monkeypatch.setattr(sc, "get_sentiment_summary", _no)
    return main


def test_cmd_signal_marks_notified_on_success(temp_db, fake_telegram, monkeypatch):
    main = _patch_signal_pipeline(monkeypatch)
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == 1
    r = _row(temp_db, 1)
    assert r["notified"] == 1 and r["alert_text"] == fake_telegram["sent"][0]
    assert r["sentiment_score"] is None
    assert "Sentimiento" not in fake_telegram["sent"][0]


def test_cmd_signal_failed_send_is_retried_next_cycle(temp_db, fake_telegram, monkeypatch):
    import notifications.telegram as tg
    main = _patch_signal_pipeline(monkeypatch)
    fake_telegram["ok"] = False
    main.cmd_signal("BTCUSDT", notify=True)
    assert _row(temp_db, 1)["notified"] == 0           # guardada, pero no llegó
    fake_telegram["ok"] = True
    res = tg.retry_failed_signal_alerts()
    assert res["resent"] == 1
    assert _row(temp_db, 1)["notified"] == 1
    assert fake_telegram["sent"][-1].startswith(tg.RESEND_NOTE)
    # El mismo setup no se vuelve a emitir como nuevo en el ciclo siguiente.
    n = len(fake_telegram["sent"])
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == n


def test_cmd_signal_without_notify_leaves_notified_null(temp_db, fake_telegram, monkeypatch):
    main = _patch_signal_pipeline(monkeypatch)
    main.cmd_signal("BTCUSDT", notify=False)
    assert fake_telegram["sent"] == []
    assert _row(temp_db, 1)["notified"] is None


# ---------------------------------------------------------------------------
# Punto 5 — arranque y descargas
# ---------------------------------------------------------------------------

def test_render_worker_starts_without_retrain():
    text = (ROOT.parent / "render.yaml").read_text()
    starts = [l.strip() for l in text.splitlines() if l.strip().startswith("startCommand:")]
    worker = [s for s in starts if "main.py" in s]
    assert worker == ["startCommand: python trading_system/main.py schedule --interval 60 --retrain-hour 03:00"]


def test_nightly_does_not_retrain_use_ml_false(monkeypatch):
    import main
    from config.settings import ASSETS, active_assets
    assert all(not ASSETS[s].get("use_ml", True) for s in active_assets())
    assert main._nightly_retrain_symbols() == []
    monkeypatch.setitem(ASSETS["BTCUSDT"], "use_ml", True)
    assert main._nightly_retrain_symbols() == ["BTCUSDT"]


def _crypto_bars(n, freq):
    idx = pd.date_range("2020-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)


def test_load_data_live_requests_reduced_crypto_history(temp_db, monkeypatch, no_network):
    import main
    import data.fetchers.binance_fetcher as bf
    from config.settings import LIVE_HISTORY_DAYS, ASSETS
    calls = []

    def _fake(symbol, timeframe="4h", days=730):
        calls.append((timeframe, days))
        return _crypto_bars(120, "4h" if timeframe == "4h" else "1D")
    monkeypatch.setattr(bf, "fetch_ohlcv", _fake)
    main._ohlcv_pending.clear()

    main._load_data("BTCUSDT", history="live")
    assert sorted(calls) == sorted([("4h", LIVE_HISTORY_DAYS["4h"]), ("1d", LIVE_HISTORY_DAYS["1d"])])
    calls.clear()
    main._load_data("BTCUSDT", history="full")
    full = ASSETS["BTCUSDT"]["days_history"]
    assert sorted(calls) == sorted([("4h", full), ("1d", full)])
    with pytest.raises(ValueError):
        main._load_data("BTCUSDT", history="otra")


def test_load_data_yfinance_same_request_in_both_modes(temp_db, monkeypatch, no_network):
    import main
    import data.fetchers.gold_fetcher as gf
    days_seen = []

    def _fake_yf(symbol, days=730):
        days_seen.append(days)
        return _crypto_bars(100, "1D")
    monkeypatch.setattr(gf, "fetch_yfinance_asset", _fake_yf)
    main._ohlcv_pending.clear()
    main._load_data("XAUUSD", history="live")
    main._load_data("XAUUSD", history="full")
    assert len(days_seen) == 2 and days_seen[0] == days_seen[1]
    # El diario de yfinance ya no se duplica como "4h" en la copia OHLCV.
    assert set(main._ohlcv_pending) == {("XAUUSD", "1d")}


def test_ohlcv_copy_is_small_and_deferred(temp_db, monkeypatch, no_network):
    import main
    import data.fetchers.binance_fetcher as bf
    from sqlalchemy import text
    monkeypatch.setattr(bf, "fetch_ohlcv",
                        lambda s, timeframe="4h", days=730: _crypto_bars(500, "4h" if timeframe == "4h" else "1D"))
    main._ohlcv_pending.clear()
    main._load_data("BTCUSDT", history="live")

    def _count():
        with temp_db.get_engine().connect() as conn:
            return conn.execute(text("SELECT timeframe, COUNT(*) FROM ohlcv GROUP BY timeframe")).fetchall()
    assert _count() == []                              # nada escrito durante la carga
    assert main._flush_ohlcv_copies() == 2 * main.OHLCV_COPY_BARS
    assert dict(_count()) == {"4h": main.OHLCV_COPY_BARS, "1d": main.OHLCV_COPY_BARS}
    assert main._ohlcv_pending == {}


def test_gateio_loads_only_swap_markets():
    import data.fetchers.binance_fetcher as bf
    if not bf.HAS_CCXT:
        pytest.skip("ccxt no instalado")
    ex = bf._gateio_exchange()          # el constructor no usa la red
    assert ex.options["defaultType"] == "swap"
    assert ex.options["fetchMarkets"]["types"] == ["swap"]


def test_ohlcv_copy_failure_is_not_fatal(temp_db, monkeypatch):
    import main
    import data.database as db

    def _boom(*a, **k):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(db, "save_ohlcv", _boom)
    main._ohlcv_pending.clear()
    main._queue_ohlcv_copy("BTCUSDT", "4h", _crypto_bars(80, "4h"))
    assert main._flush_ohlcv_copies() == 0
    assert main._ohlcv_pending == {}


# ---------------------------------------------------------------------------
# Punto 3 — sentimiento apagado y punto 4 (parte) — dashboard sin backtest
# ---------------------------------------------------------------------------

def test_sentiment_disabled():
    from config.settings import SENTIMENT
    assert SENTIMENT["enabled"] is False


def test_dashboard_has_no_backtest_or_sentiment_veto():
    src = (ROOT / "dashboard" / "app.py").read_text()
    for banned in ("tab_backtest", "run_backtest", "Ejecutar backtest", "should_veto_signal(",
                   "get_sentiment_summary(", "Score contrarian", "Filtro de sentimiento"):
        assert banned not in src, banned


def test_veto_not_wired_in_worker():
    src = (ROOT / "main.py").read_text()
    assert "should_veto_signal" not in src
