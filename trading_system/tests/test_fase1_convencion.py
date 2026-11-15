"""
Pruebas de la Fase 1 — grupo de convención (2026-11):
  punto 1 (1.7): convención de salida ÚNICA en validador y tracker: entrada en la
           vela i+1, SL/TP revisados de i+1 a i+forward_bars, salida por tiempo
           al cierre de i+forward_bars; helper time_exit_deadline;
  punto 7 (1.8): config honesta (sin pullback en cripto, atr_rank_min intacto) y
           contexto diario sin look-ahead en timeframes intradía.

    cd trading_system && python -m pytest tests -q

Sin red: datos sintéticos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _bars(start, n, freq="4h", price=100.0):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5,
                         "close": price, "volume": 1.0}, index=idx)


def _random_walk(n, freq, seed, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1.0}, index=idx)


# ---------------------------------------------------------------------------
# Punto 1 — convención de salida
# ---------------------------------------------------------------------------

def test_validator_checks_entry_bar():
    """Si la propia vela de entrada (i+1) toca el SL, la operación pierde 1R."""
    from backtest.validate import _simulate_setup
    df = _bars("2026-11-01", 40)
    df.loc[df.index[6], "low"] = 90.0            # vela de entrada del setup i=5
    r = _simulate_setup(df, 5, "long", 2.0, 1.5, 2.25, 24, 0.0, 0.0)
    assert r == pytest.approx(-1.0)


def test_validator_time_exit_at_close_of_i_plus_forward():
    from backtest.validate import _simulate_setup
    df = _bars("2026-11-01", 60)
    df["close"] = 100.0 + np.arange(60) * 0.01   # sin tocar SL (97) ni TP (104.5)
    i, fwd = 5, 24
    r = _simulate_setup(df, i, "long", 2.0, 1.5, 2.25, fwd, 0.0, 0.0)
    expected = (df["close"].iloc[i + fwd] - 100.0) / 3.0
    assert r == pytest.approx(expected)
    # La vela i+forward+1 ya no cuenta: un SL ahí no cambia nada
    df.loc[df.index[i + fwd + 1], "low"] = 50.0
    assert _simulate_setup(df, i, "long", 2.0, 1.5, 2.25, fwd, 0.0, 0.0) == pytest.approx(expected)
    # ... y la vela i+forward sí: un SL ahí cuenta
    df.loc[df.index[i + fwd], "low"] = 50.0
    assert _simulate_setup(df, i, "long", 2.0, 1.5, 2.25, fwd, 0.0, 0.0) == pytest.approx(-1.0)


@pytest.mark.parametrize("symbol,freq,seed", [("BTCUSDT", "4h", 1), ("ETHUSDT", "4h", 2),
                                              ("XAUUSD", "B", 3), ("SPX500", "B", 4)])
def test_tracker_equals_validator_on_random_series(symbol, freq, seed):
    """En cientos de setups aleatorios, el tracker (check_outcome_detail con
    setup_ts) da EXACTAMENTE el mismo R que el validador (sin costes) cuando el
    precio de entrada es la apertura de i+1."""
    from backtest.validate import _simulate_setup
    from config.settings import ASSETS, RISK
    from data.outcome_tracker import check_outcome_detail
    cfg = ASSETS[symbol]
    st = cfg.get("atr_stop_mult", RISK["atr_stop_mult"])
    tg = cfg.get("atr_target_mult", RISK["atr_target_mult"])
    fwd = cfg["forward_bars"]
    df = _random_walk(700, freq, seed)
    rng = np.random.default_rng(seed + 100)
    n_checked = 0
    for i in rng.choice(np.arange(5, len(df) - fwd - 2), size=150, replace=False):
        direction = "long" if rng.uniform() < 0.5 else "short"
        atr = float(rng.uniform(0.8, 2.5))
        r_val = _simulate_setup(df, int(i), direction, atr, st, tg, fwd, 0.0, 0.0)
        entry = float(df["open"].iloc[i + 1])
        sign = 1 if direction == "long" else -1
        sl, tp = entry - sign * st * atr, entry + sign * tg * atr
        res = check_outcome_detail(symbol, df.index[i] + pd.Timedelta(minutes=5), direction,
                                   entry, sl, tp, df=df, setup_ts=df.index[i])
        assert res is not None
        r_trk = sign * (res["exit_price"] - entry) / (st * atr)
        assert r_trk == pytest.approx(r_val, abs=1e-9), (i, direction)
        n_checked += 1
    assert n_checked == 150


def test_tracker_waits_for_last_bar_then_expires():
    from data.outcome_tracker import check_outcome_detail
    df = _bars("2026-11-01", 24)                  # setup = 0; faltan velas hasta la 24
    setup = df.index[0]
    assert check_outcome_detail("BTCUSDT", setup, "long", 100.0, 97.0, 104.5,
                                df=df, setup_ts=setup) is None
    df = _bars("2026-11-01", 25)
    res = check_outcome_detail("BTCUSDT", setup, "long", 100.0, 97.0, 104.5, df=df, setup_ts=setup)
    assert res["outcome"] == "expired"
    assert res["exit_ts"] == df.index[24] + pd.Timedelta(hours=4)


def test_time_exit_deadline_crypto_matches_tracker():
    from data.outcome_tracker import time_exit_deadline, check_outcome_detail
    setup = pd.Timestamp("2026-11-15 08:00", tz="UTC")
    # 24 velas tras la del setup: cierre de la vela i+24 = setup + 25 × 4h
    assert time_exit_deadline("BTCUSDT", setup) == pd.Timestamp("2026-11-19 12:00", tz="UTC")
    df = _bars("2026-11-14 00:00", 60)
    res = check_outcome_detail("BTCUSDT", setup, "long", 100.0, 97.0, 104.5, df=df, setup_ts=setup)
    assert res["outcome"] == "expired"
    assert res["exit_ts"] == time_exit_deadline("BTCUSDT", setup)


def test_time_exit_deadline_daily_business_days_matches_tracker():
    from data.outcome_tracker import time_exit_deadline, check_outcome_detail
    setup = pd.Timestamp("2026-11-13", tz="UTC")           # viernes
    # 20 sesiones después del viernes 13; el jueves 26-nov (Acción de Gracias) no
    # hay sesión ni vela en yfinance → lunes 14-dic, cierre 22:15 UTC.
    # (Revisión de la Fase 1: antes BDay daba el 11-dic y el tracker cerraba el 14.)
    assert time_exit_deadline("XAUUSD", setup) == pd.Timestamp("2026-12-14 22:15", tz="UTC")
    df = _bars("2026-11-02", 40, freq="B", price=4000.0)   # días hábiles…
    df = df[df.index.normalize() != pd.Timestamp("2026-11-26", tz="UTC")]   # …sin el festivo
    res = check_outcome_detail("XAUUSD", setup + pd.Timedelta(hours=23), "long",
                               4000.0, 3900.0, 4200.0, df=df, setup_ts=setup)
    assert res["outcome"] == "expired"
    assert res["exit_ts"] == time_exit_deadline("XAUUSD", setup)


def test_time_exit_deadline_accepts_unix_and_naive():
    from data.outcome_tracker import time_exit_deadline
    ts = pd.Timestamp("2026-11-15 08:00", tz="UTC")
    expected = time_exit_deadline("ETHUSDT", ts)
    assert time_exit_deadline("ETHUSDT", int(ts.timestamp())) == expected
    assert time_exit_deadline("ETHUSDT", pd.Timestamp("2026-11-15 08:00")) == expected
    assert time_exit_deadline("ETHUSDT", ts.to_pydatetime()) == expected


# ---------------------------------------------------------------------------
# Punto 7 — config honesta y contexto diario sin look-ahead
# ---------------------------------------------------------------------------

def test_crypto_config_without_pullback_and_frozen_values():
    from config.settings import ASSETS, ASSET_PARAMS_OVERRIDE
    for sym in ("BTCUSDT", "ETHUSDT"):
        assert ASSETS[sym]["setups"] == ["breakout"]
        assert ASSETS[sym]["ml_threshold"] == 0.55
        assert ASSET_PARAMS_OVERRIDE[sym]["BREAKOUT"]["atr_rank_min"] == 0.50
        assert ASSET_PARAMS_OVERRIDE[sym]["BREAKOUT"]["adx_min_trend"] == 20
    assert ASSETS["SPX500"]["setups"] == ["breakout", "reversal"]
    assert ASSETS["XAUUSD"]["setups"] == ["breakout"]


def test_map_daily_context_intraday_uses_previous_day():
    from signals.setups.daily_context import map_daily_context
    daily = pd.Series([1, 2, 3], index=pd.date_range("2026-11-01", periods=3, freq="D", tz="UTC"))
    base = pd.date_range("2026-11-01", periods=18, freq="4h", tz="UTC")   # 1, 2 y 3 de nov
    out = map_daily_context(base, daily, 0)
    assert out[:6] == [0] * 6            # 1-nov: no hay diario anterior
    assert out[6:12] == [1] * 6          # 2-nov: usa el diario del 1-nov (cerrado)
    assert out[12:18] == [2] * 6         # 3-nov: usa el del 2-nov, nunca el del 3


def test_map_daily_context_live_without_today_bar():
    """En vivo el diario de HOY aún no está cerrado (se descarta): las velas 4H
    de hoy deben recibir el mismo contexto que en el validador (antes: 0)."""
    from signals.setups.daily_context import map_daily_context
    daily = pd.Series([1, 2, 3], index=pd.date_range("2026-11-01", periods=3, freq="D", tz="UTC"))
    base = pd.date_range("2026-11-03", periods=3, freq="4h", tz="UTC")
    assert map_daily_context(base, daily, 0) == map_daily_context(base, daily.iloc[:2], 0) == [2, 2, 2]


def test_map_daily_context_daily_base_same_day_and_stale_gap():
    from signals.setups.daily_context import map_daily_context
    daily = pd.Series([1, 2, 3], index=pd.date_range("2026-11-02", periods=3, freq="D", tz="UTC"))
    # Base diaria (oro/SPX): mismo día, sin desplazar
    assert map_daily_context(daily.index, daily, 0) == [1, 2, 3]
    # Base intradía con un hueco de más de 4 días en el diario → sin contexto
    base = pd.date_range("2026-11-10", periods=3, freq="4h", tz="UTC")
    assert map_daily_context(base, daily, 0) == [0, 0, 0]


def _trend_daily_4h(n_days=160, seed=5):
    rng = np.random.default_rng(seed)
    n = n_days * 6
    steps = rng.normal(0, 1, n) + np.where((np.arange(n) // 240) % 2 == 0, 0.4, -0.4)
    close = 1000 + np.cumsum(steps) * 4
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    df4 = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "close": close,
                        "high": close + rng.uniform(1, 8, n), "low": close - rng.uniform(1, 8, n),
                        "volume": rng.uniform(50, 200, n)}, index=idx)
    df4["high"] = df4[["open", "close", "high"]].max(axis=1)
    df4["low"] = df4[["open", "close", "low"]].min(axis=1)
    dfd = df4.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"})
    return df4, dfd


def test_pullback_and_reversal_context_is_causal():
    """El contexto diario de cada vela 4H no cambia si se borra el futuro: con el
    diario recortado a lo que estaba cerrado al cierre de esa vela (como en vivo)
    se obtiene lo mismo que con la historia completa (como el validador)."""
    from signals.setups.trend_pullback import detect_pullbacks
    from signals.setups.reversal import detect_reversals
    df4, dfd = _trend_daily_4h()
    full_p = detect_pullbacks(df4, dfd)["trend_context"]
    full_r = detect_reversals(df4, dfd)[["strong_bull_daily", "strong_bear_daily"]]
    for k in range(400, len(df4), 37):
        t = df4.index[k]
        closed_daily = dfd[dfd.index + pd.Timedelta(days=1) <= t + pd.Timedelta(hours=4)]
        part = df4.iloc[: k + 1]
        assert detect_pullbacks(part, closed_daily)["trend_context"].iloc[-1] == full_p.iloc[k]
        rv = detect_reversals(part, closed_daily)
        assert bool(rv["strong_bull_daily"].iloc[-1]) == bool(full_r["strong_bull_daily"].iloc[k])
        assert bool(rv["strong_bear_daily"].iloc[-1]) == bool(full_r["strong_bear_daily"].iloc[k])


def test_daily_base_context_not_shifted():
    """Oro/SPX: el DataFrame base ES el diario; el setup del día D usa el diario D."""
    from signals.setups.trend_pullback import detect_pullbacks, add_trend_filter
    _, dfd = _trend_daily_4h(n_days=200)
    out = detect_pullbacks(dfd, dfd)["trend_context"]
    expected = add_trend_filter(dfd)["trend"]
    assert list(out.values) == list(expected.values)
