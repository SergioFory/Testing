"""
Pruebas de la Fase 0 (2026-11): emisión solo con velas cerradas, ventana anclada,
antiduplicado permanente, modo sombra y resolución por número de velas.

    cd trading_system && python -m pytest tests -q

No usan red: los datos son sintéticos y la BD es un SQLite temporal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """BD SQLite temporal y limpia para cada prueba."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    yield db
    if db._engine is not None:
        db._engine.dispose()


class _Trade:
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5,
                 size=1000.0, setup_type="breakout"):
        self.symbol, self.direction, self.setup_type = symbol, direction, setup_type
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = size, 2.0, 0.6


def _bars(start, n, freq="4h", price=100.0):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5,
                         "close": price, "volume": 1.0}, index=idx)


# ---------------------------------------------------------------------------
# Velas cerradas (F03)
# ---------------------------------------------------------------------------

def test_drop_unclosed_4h():
    from data.bars import drop_unclosed_bars
    df = _bars("2026-11-15 00:00", 4)               # 00, 04, 08, 12
    out = drop_unclosed_bars(df, "4h", now="2026-11-15 13:30")
    assert list(out.index.hour) == [0, 4, 8]        # la de las 12:00 sigue abierta
    out = drop_unclosed_bars(df, "4h", now="2026-11-15 16:00")
    assert len(out) == 4


def test_drop_unclosed_daily_session():
    from data.bars import drop_unclosed_bars
    df = _bars("2026-11-12", 3, freq="1D")          # 12, 13, 14
    # 14-nov a las 15:00 UTC: la sesión del 14 sigue abierta
    assert drop_unclosed_bars(df, "1d", session=True, now="2026-11-14 15:00").index[-1].day == 13
    # 14-nov a las 22:30 UTC: ya cerró
    assert drop_unclosed_bars(df, "1d", session=True, now="2026-11-14 22:30").index[-1].day == 14


# ---------------------------------------------------------------------------
# Ventana anclada a la última vela cerrada (F01)
# ---------------------------------------------------------------------------

def _setups(ts_list, direction="long", close=100.0, atr=2.0, score=0.9):
    return pd.DataFrame({
        "ts": pd.to_datetime(ts_list, utc=True), "direction": direction, "close": close,
        "atr": atr, "ml_score": score, "raw_score": score, "ml_approved": True,
        "setup_type": "breakout",
    })


def test_old_setup_not_reemitted_with_anchored_window():
    from ml.predictor import get_best_setup
    s = _setups(["2026-10-01 08:00"])               # único setup: hace semanas
    # Modo heredado: el setup viejo sigue "vigente" (el bug de las re-emisiones)
    assert get_best_setup(s, n_last_bars=12, current_price=100.5, max_dist_atr=1.0) is not None
    # Ventana anclada: fuera de las 2 últimas velas cerradas → no se emite
    best = get_best_setup(s, current_price=100.5, max_dist_atr=1.0,
                          min_setup_ts="2026-11-15 04:00", max_setup_ts="2026-11-15 08:00")
    assert best is None


def test_exclude_keys_skips_emitted_setup():
    from ml.predictor import get_best_setup
    s = pd.concat([_setups(["2026-11-15 08:00"], "long", score=0.9),
                   _setups(["2026-11-15 08:00"], "short", score=0.7)], ignore_index=True)
    kw = dict(current_price=100.0, max_dist_atr=1.0,
              min_setup_ts="2026-11-15 04:00", max_setup_ts="2026-11-15 08:00")
    assert get_best_setup(s, **kw)["direction"] == "long"
    done = {(pd.Timestamp("2026-11-15 08:00", tz="UTC"), "long")}
    assert get_best_setup(s, exclude_keys=done, **kw)["direction"] == "short"


def test_live_emulation_matches_validator_on_synthetic_trend():
    """En una serie sintética con rupturas, el vivo emulado emite exactamente
    los setups aprobados, una vez cada uno y con edad 0."""
    from backtest.live_emulation import compare_with_validator
    rng = np.random.default_rng(7)
    n = 900
    steps = rng.normal(0, 1, n) + np.where((np.arange(n) // 150) % 2 == 0, 0.25, -0.25)
    close = 1000 + np.cumsum(steps) * 5
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    high = close + rng.uniform(1, 12, n)
    low = close - rng.uniform(1, 12, n)
    vol = rng.uniform(50, 200, n) * np.where(rng.uniform(size=n) > 0.85, 3, 1)
    df = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    res = compare_with_validator("BTCUSDT", df, None)
    assert res["repeated"] == 0
    assert not res["extra"]
    assert not res["missing"]
    assert res["pct_age_le1"] == 1.0


# ---------------------------------------------------------------------------
# Resolución por número de velas, PnL de la posición y R (F04, F05)
# ---------------------------------------------------------------------------

def test_time_exit_after_forward_bars_with_pnl():
    from data.outcome_tracker import check_outcome_detail
    df = _bars("2026-11-01 00:00", 40, price=100.0)
    df.loc[df.index[-1], "close"] = 101.0
    sig = pd.Timestamp("2026-11-01 01:00", tz="UTC")
    res = check_outcome_detail("BTCUSDT", sig, "long", 100.0, 97.0, 104.5, df=df)
    assert res["outcome"] == "expired"
    # Señal antigua sin setup_ts: vela del setup = la que contiene la señal (pos 0).
    # Convención Fase 1 (igual que el validador): entrada en la 1, se revisan de la
    # 1 a la 24 y se sale al cierre de la 24 (resolved_ts = cierre de esa vela).
    assert res["exit_ts"] == df.index[24] + pd.Timedelta(hours=4)
    assert res["exit_price"] == pytest.approx(100.0)


def test_tracker_matches_validator_convention_daily():
    """Diario emitido tras el cierre (convención Fase 1): la vela de entrada (D+1)
    SÍ se revisa y la salida por tiempo cae en la vela i+forward, igual que
    _simulate_setup."""
    from data.outcome_tracker import check_outcome_detail
    df = _bars("2026-09-01", 40, freq="1D", price=4000.0)
    df["close"] = 4000.0 + np.arange(40)
    setup = df.index[5]
    sig = setup + pd.Timedelta(hours=23)
    res = check_outcome_detail("XAUUSD", sig, "short", 4005.0, 4160.0, 3800.0, df=df, setup_ts=setup)
    assert res["outcome"] == "expired"
    assert res["exit_price"] == pytest.approx(float(df["close"].iloc[5 + 20]))
    # La vela de entrada toca el SL: ahora cuenta (en el validador también)
    df.loc[df.index[6], "high"] = 4300.0
    res = check_outcome_detail("XAUUSD", sig, "short", 4005.0, 4160.0, 3800.0, df=df, setup_ts=setup)
    assert res["outcome"] == "loss" and res["exit_price"] == pytest.approx(4160.0)


def test_sl_checked_before_tp_and_before_age():
    from data.outcome_tracker import check_outcome_detail
    df = _bars("2026-11-01 00:00", 10, price=100.0)
    df.loc[df.index[3], ["low", "high"]] = [96.0, 105.0]      # toca ambos → cuenta SL
    res = check_outcome_detail("BTCUSDT", pd.Timestamp("2026-11-01 01:00", tz="UTC"),
                               "long", 100.0, 97.0, 104.5, df=df)
    assert res["outcome"] == "loss" and res["exit_price"] == 97.0


def test_history_must_cover_signal():
    from data.outcome_tracker import check_outcome_detail
    df = _bars("2026-11-10 00:00", 40)
    res = check_outcome_detail("BTCUSDT", pd.Timestamp("2026-11-01", tz="UTC"),
                               "long", 100.0, 97.0, 104.5, df=df)
    assert res is None


def test_result_fields_use_position_size():
    from data.outcome_tracker import compute_result_fields
    f = compute_result_fields("long", 100.0, 97.0, 104.5, position_size=1000.0)
    assert f["pnl_pct"] == pytest.approx(0.045)
    assert f["pnl_usd"] == pytest.approx(45.0)        # antes: entry × pct = 4.5 por unidad
    assert f["pnl_r"] == pytest.approx(1.5)
    f = compute_result_fields("short", 100.0, 103.0, 103.0, position_size=500.0)
    assert f["pnl_r"] == pytest.approx(-1.0) and f["pnl_usd"] == pytest.approx(-15.0)


def test_daily_safety_net_covers_horizon():
    from data.outcome_tracker import safety_net_days, horizon_days
    assert horizon_days("XAUUSD") == pytest.approx(28.0)
    assert safety_net_days("XAUUSD") == pytest.approx(33.0)
    assert safety_net_days("BTCUSDT") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# BD: migración, antiduplicado permanente, modo sombra
# ---------------------------------------------------------------------------

def test_migration_adds_columns_to_old_schema(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL,
        ts INTEGER NOT NULL, setup_type TEXT NOT NULL, direction TEXT NOT NULL, ml_score REAL NOT NULL,
        sentiment_score REAL, entry_price REAL NOT NULL, stop_loss REAL NOT NULL, take_profit REAL NOT NULL,
        position_size REAL NOT NULL, atr REAL NOT NULL, resultado_real TEXT, pnl_usd REAL, pnl_pct REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    con.commit(); con.close()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    import data.database as db
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(signals)")}
    assert {"setup_ts", "mode", "pnl_r", "exit_price", "quality_flag", "data_source"} <= cols
    db._engine.dispose()


def test_permanent_setup_dedup_and_shadow_mode(temp_db):
    from data.outcome_tracker import (save_signal_from_trade, is_duplicate_setup,
                                      emitted_setup_keys, get_outcome_stats)
    st = pd.Timestamp("2026-11-15 08:00", tz="UTC")
    sid = save_signal_from_trade(_Trade(), setup_ts=st, data_source="gateio",
                                 last_bar_ts=st, mode="live")
    assert sid > 0
    assert is_duplicate_setup("BTCUSDT", "long", st)
    # Mismo setup con otra entrada (p. ej. vela que antes estaba abierta): bloqueado
    assert save_signal_from_trade(_Trade(entry=101.0, sl=98.0, tp=105.5), setup_ts=st) == -1
    assert (st, "long") in emitted_setup_keys("BTCUSDT", st - pd.Timedelta(hours=4))
    # Señal en sombra de otro activo: se guarda, pero no cuenta en las estadísticas
    sid2 = save_signal_from_trade(_Trade(symbol="SPX500", entry=6000, sl=5900, tp=6150),
                                  setup_ts=st, mode="shadow")
    assert sid2 > 0
    df = temp_db.load_signals(limit=10)
    assert set(df["mode"]) == {"live", "shadow"}
    assert get_outcome_stats(symbol="SPX500") == {}


def test_resolve_open_signals_end_to_end(temp_db, monkeypatch):
    import data.outcome_tracker as ot
    st = pd.Timestamp("2026-11-01 00:00", tz="UTC")
    sid = ot.save_signal_from_trade(_Trade(size=2000.0), setup_ts=st)
    # La señal se registró "ahora"; forzar su hora al 1-nov 01:00
    with temp_db.get_engine().begin() as conn:
        from sqlalchemy import text
        conn.execute(text("UPDATE signals SET ts=:t WHERE id=:i"),
                     {"t": int(pd.Timestamp("2026-11-01 01:00", tz="UTC").timestamp()), "i": sid})
    df = _bars("2026-10-30 00:00", 60, price=100.0)
    df.loc[pd.Timestamp("2026-11-02 08:00", tz="UTC"), "high"] = 105.0     # toca TP
    monkeypatch.setattr(ot, "_load_outcome_ohlcv", lambda symbol, days=None: df)
    assert ot.resolve_open_signals() == 1
    row = temp_db.load_signals(limit=1).iloc[0]
    assert row["resultado_real"] == "win"
    assert row["pnl_r"] == pytest.approx(1.5)
    assert row["pnl_usd"] == pytest.approx(2000.0 * 0.045)
    assert row["exit_price"] == pytest.approx(104.5)


def test_legacy_alert_blocks_first_cycle_after_deploy(temp_db):
    """Una alerta antigua (sin setup_ts) emitida con la vela abierta bloquea que
    el primer ciclo tras desplegar reenvíe ese setup ya cerrado."""
    from sqlalchemy import text
    import data.outcome_tracker as ot
    with temp_db.get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO signals (symbol, ts, setup_type, direction, ml_score, entry_price, stop_loss, "
            "take_profit, position_size, atr) VALUES ('BTCUSDT', :ts, 'breakout', 'long', 0.6, 80584.3, "
            "79000, 83000, 1000, 1000)"), {"ts": int(pd.Timestamp("2026-11-04 15:26", tz="UTC").timestamp())})
    assert ot.has_recent_legacy_alert("BTCUSDT", "long", pd.Timestamp("2026-11-04 12:00", tz="UTC"))
    assert not ot.has_recent_legacy_alert("BTCUSDT", "long", pd.Timestamp("2026-11-04 16:00", tz="UTC"))
    assert not ot.has_recent_legacy_alert("BTCUSDT", "short", pd.Timestamp("2026-11-04 12:00", tz="UTC"))


def test_unix_helpers_take_numbers_as_seconds():
    import numpy as np
    from data.outcome_tracker import _unix
    from data.database import _to_unix
    assert _unix(np.int64(1758215000)) == 1758215000
    assert _to_unix(np.int64(1758215000)) == 1758215000


def test_cleanup_classification_respects_new_rows():
    """La limpieza marca solo filas antiguas; las de la Fase 0 (con setup_ts)
    conservan su quality_flag aunque sean de velas consecutivas."""
    from fase0_db_cleanup import classify_repeats
    t = lambda x: pd.Timestamp(x, tz="UTC")
    df = pd.DataFrame([
        # antiguas: original, re-alerta de vela abierta, re-emisión idéntica días después
        dict(id=1, symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=97.0, take_profit=104.5,
             ts_dt=t("2026-09-21 09:31"), setup_ts=None, quality_flag=None),
        dict(id=2, symbol="BTCUSDT", direction="long", entry_price=100.4, stop_loss=97.4, take_profit=104.9,
             ts_dt=t("2026-09-21 10:31"), setup_ts=None, quality_flag=None),
        dict(id=3, symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=97.0, take_profit=104.5,
             ts_dt=t("2026-09-25 10:00"), setup_ts=None, quality_flag=None),
        # otro setup con entrada casi igual pero SL/TP distintos → original
        dict(id=4, symbol="BTCUSDT", direction="long", entry_price=100.001, stop_loss=96.0, take_profit=106.0,
             ts_dt=t("2026-10-05 10:00"), setup_ts=None, quality_flag=None),
        # nuevas (Fase 0) en velas consecutivas → no se tocan
        dict(id=5, symbol="ETHUSDT", direction="long", entry_price=2093.6, stop_loss=2000, take_profit=2280,
             ts_dt=t("2026-11-10 12:05"), setup_ts=1762761600, quality_flag=None),
        dict(id=6, symbol="ETHUSDT", direction="long", entry_price=2022.4, stop_loss=1930, take_profit=2200,
             ts_dt=t("2026-11-10 16:05"), setup_ts=1762776000, quality_flag=None),
    ])
    flags = classify_repeats(df)
    assert list(flags) == [None, "intrabar_repeat", "stale_reemission", None, None, None]
