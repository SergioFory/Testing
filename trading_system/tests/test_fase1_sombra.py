"""
Pruebas de la Fase 1 — registro sombra y comandos de Telegram (punto 1.10):
  - tablas shadow_setups / fills / kv (migración idempotente);
  - cada ciclo de cmd_signal registra TODOS los setups de las 2 últimas velas
    cerradas con su motivo (emitted, shadow_asset, below_threshold, not_fresh,
    event_blackout, risk_invalid, duplicate, not_best);
  - el tracker los resuelve con check_outcome_detail reutilizando la caché;
  - /fill /skip /help por polling de getUpdates con offset persistido, solo del
    chat configurado, con respuesta que incluye el R:R efectivo.

    cd trading_system && python -m pytest tests -q

Sin red: requests.get/post fallan si alguien intenta usarlos; Telegram se sustituye
por dobles y la BD es SQLite temporal.
"""
import sys
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


@pytest.fixture()
def fake_telegram(monkeypatch, no_network):
    import notifications.telegram as tg
    state = {"ok": True, "sent": []}

    def _fake_send(text, parse_mode="HTML", retries=3):
        state["sent"].append(text)
        return state["ok"]
    monkeypatch.setattr(tg, "_send", _fake_send)
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    return state


def _rows(db, table, where="1=1"):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table} WHERE {where} ORDER BY id"))
                .mappings().fetchall()]


IDX = pd.date_range("2026-11-10", periods=30, freq="4h", tz="UTC")


def _setup(ts, direction="long", raw=0.7, close=100.0, atr=2.0, typ="breakout"):
    sign = 1 if direction == "long" else -1
    return {"ts": ts, "symbol": "X", "setup_type": typ, "direction": direction,
            "raw_score": raw, "atr": atr, "close": close,
            "stop_loss": close - sign * 1.5 * atr, "take_profit": close + sign * 2.25 * atr}


class _Trade:
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


def _pipeline(monkeypatch, setups, close_last=100.0, trade_factory=None, blackout=False,
              symbol="BTCUSDT"):
    """cmd_signal con datos y detección simulados. Usa el get_best_setup REAL
    (la clasificación sombra debe coincidir con la selección del vivo)."""
    import main
    import signals.detector as det
    import ml.predictor as pred
    import risk.position_sizer as ps
    import data.bars as bars
    import data.fetchers.econ_calendar as cal

    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                       "volume": 1.0}, index=IDX)
    df.iloc[-1, df.columns.get_loc("close")] = close_last
    df.attrs["source"] = "test"
    monkeypatch.setattr(main, "_load_data", lambda s, closed_only=True, history="full": (df, df, None))
    monkeypatch.setattr(bars, "last_bar_age_hours", lambda *a, **k: 1.0)
    frame = pd.DataFrame(setups)
    monkeypatch.setattr(det, "detect_all_setups", lambda *a, **k: frame.copy())
    monkeypatch.setattr(pred, "score_setups",
                        lambda d, *a, **k: d.assign(ml_score=d["raw_score"],
                                                    ml_approved=d["raw_score"] >= 0.55))

    def _build(row, capital, score):
        if trade_factory is not None:
            return trade_factory(row)
        return _Trade(symbol=symbol, direction=row["direction"],
                      entry=float(row["close"]), sl=float(row["stop_loss"]), tp=float(row["take_profit"]))
    monkeypatch.setattr(ps, "build_trade_setup", _build)
    monkeypatch.setattr(cal, "is_event_blackout", lambda *a, **k: (blackout, "NFP" if blackout else ""))
    # Fase 2: estas velas sintéticas (constantes, 4H con sábados también para el
    # oro, fuente "test") disparan con razón el monitor de datos; aquí se prueba
    # el registro sombra, así que se neutraliza (tiene sus pruebas en
    # test_fase2_monitor.py, incluida la de que no cambia la señal).
    import monitoring.data_monitor as dm
    monkeypatch.setattr(dm, "run_cycle_checks", lambda *a, **k: [])
    return main, df


def _shadow_by_key(db):
    out = {}
    for r in _rows(db, "shadow_setups"):
        out[(pd.Timestamp(r["setup_ts"], unit="s", tz="UTC"), r["direction"])] = r
    return out


# ---------------------------------------------------------------------------
# Migración
# ---------------------------------------------------------------------------

def test_migration_creates_tables_idempotent(temp_db):
    temp_db.init_db(force=True)
    temp_db.init_db(force=True)
    with temp_db.get_engine().connect() as conn:
        names = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(shadow_setups)"))}
    assert {"shadow_setups", "fills", "kv"} <= names
    assert {"symbol", "setup_ts", "direction", "setup_type", "raw_score", "approved", "motivo",
            "entry", "sl", "tp", "atr", "created_ts", "resultado", "pnl_r", "exit_price",
            "resolved_ts"} <= cols


def test_kv_roundtrip(temp_db):
    assert temp_db.kv_get("x") is None
    temp_db.kv_set("x", 5)
    temp_db.kv_set("x", 7)
    assert temp_db.kv_get("x") == "7"


# ---------------------------------------------------------------------------
# Clasificación y guardado
# ---------------------------------------------------------------------------

def test_classify_window_all_motivos():
    from data.shadow import classify_window
    t0, t1 = IDX[-2], IDX[-1]
    w = pd.DataFrame([
        _setup(t0, "long", raw=0.40),                 # below_threshold
        _setup(t0, "short", raw=0.80),                # ya registrado → emitted (keep)
        _setup(t1, "long", raw=0.70, close=110.0),    # 10 ATR del precio → not_fresh
        _setup(t1, "short", raw=0.90),                # elegido
    ]).assign(ml_approved=lambda d: d["raw_score"] >= 0.55)
    rows = classify_window(w, {(t0, "short")}, 100.0, 1.0, chosen_key=(t1, "short"),
                           chosen_motivo="risk_invalid")
    m = {(r["setup_ts"], r["direction"]): (r["motivo"], r["keep_existing"]) for r in rows}
    assert m[(t0, "long")] == ("below_threshold", False)
    assert m[(t0, "short")] == ("emitted", True)
    assert m[(t1, "long")] == ("not_fresh", False)
    assert m[(t1, "short")] == ("risk_invalid", False)

    # Otro aprobado ejecutable no elegido: not_best, o el motivo del activo.
    w2 = pd.DataFrame([_setup(t0, "long", raw=0.6), _setup(t1, "long", raw=0.9)]).assign(ml_approved=True)
    r2 = classify_window(w2, set(), 100.0, 1.0, chosen_key=(t1, "long"), chosen_motivo="emitted")
    assert [r["motivo"] for r in r2] == ["not_best", "emitted"]
    r3 = classify_window(w2, set(), 100.0, 1.0, chosen_key=(t1, "long"),
                         chosen_motivo="event_blackout", symbol_motivo="event_blackout")
    assert [r["motivo"] for r in r3] == ["event_blackout", "event_blackout"]
    # Elegido sin decidir (p. ej. excepción a mitad): no se registra aún.
    r4 = classify_window(w2, set(), 100.0, 1.0, chosen_key=(t1, "long"), chosen_motivo=None)
    assert [r["motivo"] for r in r4] == ["not_best"]
    # Activo en modo sombra: lo ya registrado es shadow_asset.
    r5 = classify_window(w.iloc[[1]], {(t0, "short")}, 100.0, 1.0, asset_live=False)
    assert r5[0]["motivo"] == "shadow_asset"


def test_record_upsert_final_motivos_are_sticky(temp_db):
    from data.shadow import record_shadow_setups
    t = IDX[-1]
    base = {"setup_ts": t, "direction": "long", "setup_type": "breakout", "raw_score": 0.7,
            "approved": 1, "entry": 100.0, "sl": 97.0, "tp": 104.5, "atr": 2.0, "keep_existing": False}
    record_shadow_setups("BTCUSDT", [dict(base, motivo="event_blackout")])
    record_shadow_setups("BTCUSDT", [dict(base, motivo="emitted", entry=999.0)])
    r = _rows(temp_db, "shadow_setups")
    assert len(r) == 1 and r[0]["motivo"] == "emitted"
    assert r[0]["entry"] == 100.0                      # los niveles no se tocan
    record_shadow_setups("BTCUSDT", [dict(base, motivo="not_fresh")])
    assert _rows(temp_db, "shadow_setups")[0]["motivo"] == "emitted"
    # keep_existing solo inserta si falta.
    record_shadow_setups("BTCUSDT", [dict(base, direction="short", motivo="not_fresh")])
    record_shadow_setups("BTCUSDT", [dict(base, direction="short", motivo="emitted", keep_existing=True)])
    assert _shadow_by_key(temp_db)[(t, "short")]["motivo"] == "not_fresh"
    # Otro símbolo con la misma vela y dirección es otra fila.
    record_shadow_setups("ETHUSDT", [dict(base, motivo="below_threshold")])
    assert len(_rows(temp_db, "shadow_setups")) == 3


# ---------------------------------------------------------------------------
# Integración con cmd_signal
# ---------------------------------------------------------------------------

def test_cmd_signal_records_whole_window_and_adds_id(temp_db, fake_telegram, monkeypatch):
    t_old, t0, t1 = IDX[-3], IDX[-2], IDX[-1]
    main, _ = _pipeline(monkeypatch, [
        _setup(t_old, "long", raw=0.9),               # fuera de la ventana: no se registra
        _setup(t0, "long", raw=0.40),                 # below_threshold
        _setup(t0, "short", raw=0.80, close=110.0),   # aprobado pero lejos → not_fresh
        _setup(t1, "long", raw=0.70),                 # elegido → emitted
    ])
    cache = {}
    main.cmd_signal("BTCUSDT", notify=True, ohlcv_cache=cache)
    sh = _shadow_by_key(temp_db)
    assert set(sh) == {(t0, "long"), (t0, "short"), (t1, "long")}
    assert sh[(t0, "long")]["motivo"] == "below_threshold" and sh[(t0, "long")]["approved"] == 0
    assert sh[(t0, "short")]["motivo"] == "not_fresh"
    assert sh[(t1, "long")]["motivo"] == "emitted"
    assert sh[(t1, "long")]["sl"] == pytest.approx(97.0) and sh[(t1, "long")]["tp"] == pytest.approx(104.5)
    # El mensaje lleva el ID para poder responder /fill; alert_text igual a lo enviado.
    assert len(fake_telegram["sent"]) == 1
    assert "/fill 1 PRECIO" in fake_telegram["sent"][0]
    assert _rows(temp_db, "signals")[0]["alert_text"] == fake_telegram["sent"][0]
    assert "BTCUSDT" in cache                          # caché del ciclo rellenada

    # Ciclo siguiente con los mismos datos: nada nuevo, el emitido sigue 'emitted'.
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == 1
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "emitted"
    assert len(_rows(temp_db, "shadow_setups")) == 3


def test_cmd_signal_not_best_then_emitted_next_cycle(temp_db, fake_telegram, monkeypatch):
    t0, t1 = IDX[-2], IDX[-1]
    main, _ = _pipeline(monkeypatch, [_setup(t0, "long", raw=0.60, close=100.5),
                                      _setup(t1, "short", raw=0.90)])
    main.cmd_signal("BTCUSDT", notify=True)
    sh = _shadow_by_key(temp_db)
    assert sh[(t1, "short")]["motivo"] == "emitted"
    assert sh[(t0, "long")]["motivo"] == "not_best"
    main.cmd_signal("BTCUSDT", notify=True)            # una hora después: sale el otro
    assert _shadow_by_key(temp_db)[(t0, "long")]["motivo"] == "emitted"
    assert len(fake_telegram["sent"]) == 2


def test_cmd_signal_risk_invalid_and_event_blackout(temp_db, fake_telegram, monkeypatch):
    t0, t1 = IDX[-2], IDX[-1]
    main, _ = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)], trade_factory=lambda r: None)
    main.cmd_signal("BTCUSDT", notify=True)
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "risk_invalid"
    assert fake_telegram["sent"] == []

    main, _ = _pipeline(monkeypatch, [_setup(t0, "short", raw=0.6, close=100.5),
                                      _setup(t1, "long", raw=0.9)], blackout=True, symbol="XAUUSD")
    main.cmd_signal("XAUUSD", notify=True)
    sh = _shadow_by_key(temp_db)
    xau = {k: v for k, v in sh.items() if v["symbol"] == "XAUUSD"}
    assert {v["motivo"] for v in xau.values()} == {"event_blackout"} and len(xau) == 2
    assert fake_telegram["sent"] == []


def test_cmd_signal_duplicate_by_price(temp_db, fake_telegram, monkeypatch):
    from data.outcome_tracker import save_signal_from_trade
    t1 = IDX[-1]
    # Señal antigua (otra vela) al mismo precio, aún abierta → antiduplicado por precio.
    save_signal_from_trade(_Trade(), setup_ts=IDX[-10], mode="live")
    main, _ = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)])
    main.cmd_signal("BTCUSDT", notify=True)
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "duplicate"
    assert fake_telegram["sent"] == []


def test_cmd_signal_shadow_asset(temp_db, fake_telegram, monkeypatch):
    import config.settings as st
    t1 = IDX[-1]
    assert st.notify_enabled("SPX500") is False        # SPX está en modo sombra
    main, _ = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)], symbol="SPX500")
    main.cmd_signal("SPX500", notify=True)
    assert _shadow_by_key(temp_db)[(t1, "long")]["motivo"] == "shadow_asset"
    assert fake_telegram["sent"] == []


def test_shadow_failure_never_breaks_signal(temp_db, fake_telegram, monkeypatch):
    import data.shadow as sh
    t1 = IDX[-1]
    main, _ = _pipeline(monkeypatch, [_setup(t1, "long", raw=0.9)])

    def _boom(*a, **k):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(sh, "record_shadow_setups", _boom)
    main.cmd_signal("BTCUSDT", notify=True)
    assert len(fake_telegram["sent"]) == 1             # la señal salió igual
    assert _rows(temp_db, "signals")[0]["notified"] == 1


# ---------------------------------------------------------------------------
# Resolución en el tracker
# ---------------------------------------------------------------------------

def _bars(n=40, start="2026-11-01"):
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0,
                         "volume": 1.0}, index=idx)


def test_resolve_shadow_same_convention_and_reuses_cache(temp_db, monkeypatch):
    import data.outcome_tracker as ot
    from data.shadow import record_shadow_setups, resolve_shadow_setups
    df = _bars()
    i = 5
    df.iloc[i + 1, df.columns.get_loc("low")] = 96.0          # SL en la vela de ENTRADA (i+1)
    df.iloc[i + 3, df.columns.get_loc("high")] = 105.0        # TP del 2.º setup (vela i+1)
    base = {"setup_type": "breakout", "raw_score": 0.7, "approved": 1, "atr": 2.0,
            "keep_existing": False, "motivo": "emitted"}
    record_shadow_setups("BTCUSDT", [
        dict(base, setup_ts=df.index[i], direction="long", entry=100.0, sl=97.0, tp=104.5),
        dict(base, setup_ts=df.index[i + 1], direction="long", entry=100.0, sl=97.0, tp=104.5,
             motivo="below_threshold"),
        dict(base, setup_ts=df.index[30], direction="long", entry=100.0, sl=90.0, tp=110.0,
             motivo="not_fresh"),                              # aún sin resolver
    ])

    def _no_download(*a, **k):
        raise AssertionError("debe reutilizar la caché del ciclo")
    monkeypatch.setattr(ot, "_load_outcome_ohlcv", _no_download)
    now = df.index[-1] + pd.Timedelta(hours=4)
    n = resolve_shadow_setups(ohlcv_cache={"BTCUSDT": df}, now=now)
    assert n == 2
    r = _rows(temp_db, "shadow_setups")
    # Idéntico a check_outcome_detail (la convención de las señales).
    for row in r[:2]:
        st = pd.Timestamp(row["setup_ts"], unit="s", tz="UTC")
        exp = ot.check_outcome_detail("BTCUSDT", st, "long", 100.0, 97.0, 104.5, df=df, setup_ts=st)
        assert row["resultado"] == exp["outcome"] and row["exit_price"] == exp["exit_price"]
    assert r[0]["resultado"] == "loss" and r[0]["pnl_r"] == pytest.approx(-1.0)
    assert r[1]["resultado"] == "win" and r[1]["pnl_r"] == pytest.approx(1.5)
    assert r[0]["resolved_ts"] == int((df.index[i + 1] + pd.Timedelta(hours=4)).timestamp())
    assert r[2]["resultado"] is None


def test_resolve_shadow_downloads_missing_symbol_once(temp_db, monkeypatch):
    import data.outcome_tracker as ot
    from data.shadow import record_shadow_setups, resolve_shadow_setups
    df = _bars()
    calls = []

    def _load(symbol, days=None):
        calls.append(symbol)
        return df
    monkeypatch.setattr(ot, "_load_outcome_ohlcv", _load)
    base = {"setup_type": "breakout", "raw_score": 0.7, "approved": 1, "atr": 2.0,
            "keep_existing": False, "motivo": "emitted", "direction": "long",
            "entry": 100.0, "sl": 97.0, "tp": 104.5}
    record_shadow_setups("ETHUSDT", [dict(base, setup_ts=df.index[2]), dict(base, setup_ts=df.index[3])])
    cache = {}
    n = resolve_shadow_setups(ohlcv_cache=cache, now=df.index[-1])
    assert calls == ["ETHUSDT"] and cache["ETHUSDT"] is df
    assert n == 2 and {r["resultado"] for r in _rows(temp_db, "shadow_setups")} == {"expired"}


def test_resolve_open_signals_fills_shared_cache(temp_db, monkeypatch):
    import data.outcome_tracker as ot
    df = _bars()
    monkeypatch.setattr(ot, "_load_outcome_ohlcv", lambda s, days=None: df)
    ot.save_signal_from_trade(_Trade(), setup_ts=df.index[5], mode="live")
    cache = {}
    ot.resolve_open_signals(ohlcv_cache=cache)
    assert cache["BTCUSDT"] is df


# ---------------------------------------------------------------------------
# Comandos de Telegram
# ---------------------------------------------------------------------------

def test_parse_price_and_rr():
    from notifications.telegram_commands import parse_price, effective_rr, entry_diff_r, parse_command
    assert parse_price("65432.5") == 65432.5
    assert parse_price("65432,5") == 65432.5
    assert parse_price("65,432.5") == 65432.5
    assert parse_price("65.432,5") == 65432.5
    assert parse_price("65,432") == 65432.0
    assert parse_price("2650,55") == 2650.55
    assert parse_price("0,5") == 0.5
    assert parse_price("$4,123.25") == 4123.25
    assert parse_price("abc") is None and parse_price("-3") is None and parse_price("0") is None
    assert effective_rr("long", 100.0, 97.0, 104.5) == pytest.approx(1.5)
    assert effective_rr("long", 101.0, 97.0, 104.5) == pytest.approx(3.5 / 4.0)
    assert effective_rr("short", 100.0, 103.0, 95.5) == pytest.approx(1.5)
    assert effective_rr("long", 96.0, 97.0, 104.5) is None
    assert entry_diff_r("long", 99.4, 100.0, 97.0) == pytest.approx(0.2)
    assert entry_diff_r("short", 99.4, 100.0, 103.0) == pytest.approx(-0.2)
    assert parse_command("/fill@MiBot 3 100") == ("fill", ["3", "100"])
    assert parse_command("hola") is None


def _fake_updates(monkeypatch, batches, seen_params):
    """requests.get de getUpdates: devuelve batches[k] en la llamada k."""
    import requests
    calls = {"n": 0}

    class _R:
        status_code = 200

        def __init__(self, data):
            self._d = data
            self.text = str(data)

        def json(self):
            return self._d

    def _get(url, params=None, timeout=None):
        assert "getUpdates" in url and params["timeout"] == 0
        seen_params.append(dict(params))
        k = min(calls["n"], len(batches) - 1)
        calls["n"] += 1
        return _R({"ok": True, "result": batches[k]})
    monkeypatch.setattr(requests, "get", _get)


def _upd(uid, text_, chat_id=4242):
    return {"update_id": uid, "message": {"date": 1763200000, "chat": {"id": chat_id}, "text": text_}}


def test_poll_fill_skip_help_and_offset(temp_db, fake_telegram, monkeypatch):
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import poll_commands, KV_OFFSET
    sid = save_signal_from_trade(_Trade(), setup_ts=IDX[-1], mode="live")
    params = []
    batch = [
        _upd(100, f"/fill {sid} 101"),
        _upd(101, f"/skip {sid} llegué tarde"),
        _upd(102, "/help"),
        _upd(103, f"/fill {sid} 100", chat_id=999),      # chat ajeno: ignorado
        _upd(104, "hola"),                                # texto normal: sin respuesta
        _upd(105, "/fill 777 100"),                       # señal inexistente
    ]
    _fake_updates(monkeypatch, [batch, []], params)
    out = poll_commands()
    assert out["ignored"] == 1 and out["commands"] == 4 and out["errors"] == 0
    fills = _rows(temp_db, "fills")
    assert [(f["kind"], f["price"], f["note"]) for f in fills] == [
        ("fill", 101.0, None), ("skip", None, "llegué tarde")]
    assert fills[0]["signal_id"] == sid and fills[0]["update_id"] == 100
    sent = fake_telegram["sent"]
    assert len(sent) == 4
    assert "R:R efectivo a tu precio: 0.88" in sent[0] and "(el de la señal: 1.50)" in sent[0]
    assert "-0.33R (en contra)" in sent[0]
    assert "llegué tarde" in sent[1]
    assert "/fill ID PRECIO" in sent[2]
    assert "No existe la señal #777" in sent[3]
    assert temp_db.kv_get(KV_OFFSET) == "106"
    # Siguiente ciclo: pide desde el offset guardado.
    poll_commands()
    assert "offset" not in params[0] and params[1]["offset"] == 106


def test_poll_does_not_duplicate_if_offset_not_saved(temp_db, fake_telegram, monkeypatch):
    import data.database as db
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import poll_commands
    sid = save_signal_from_trade(_Trade(), setup_ts=IDX[-1], mode="live")
    params = []
    _fake_updates(monkeypatch, [[_upd(7, f"/fill {sid} 99.5")]], params)

    def _fail(*a, **k):
        raise RuntimeError("BD caída al guardar el offset")
    monkeypatch.setattr(db, "kv_set", _fail)
    poll_commands()
    poll_commands()                                       # mismo mensaje otra vez
    assert len(_rows(temp_db, "fills")) == 1
    assert len(fake_telegram["sent"]) == 1


def test_poll_html_escaped_and_bad_inputs(temp_db, fake_telegram, monkeypatch):
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import poll_commands
    sid = save_signal_from_trade(_Trade(direction="short", entry=100.0, sl=103.0, tp=95.5),
                                 setup_ts=IDX[-1], mode="live")
    _fake_updates(monkeypatch, [[_upd(1, f"/skip {sid} <b>x</b> & y"), _upd(2, f"/fill {sid} abc"),
                                 _upd(3, "/fill"), _upd(4, f"/fill {sid} 104"), _upd(5, "/foo<x>")]], [])
    poll_commands()
    sent = fake_telegram["sent"]
    assert "&lt;b&gt;x&lt;/b&gt; &amp; y" in sent[0]
    assert "no es un precio válido" in sent[1] and "Uso:" in sent[2]
    # Revisión de la Fase 1: un precio más allá del SL/TP ya NO se guarda sin confirmar.
    assert "más allá del" in sent[3] and "No lo he guardado" in sent[3]
    assert f"/fill {sid} 104 confirmar" in sent[3]
    assert "/foo&lt;x&gt;" in sent[4]
    assert [f["kind"] for f in _rows(temp_db, "fills")] == ["skip"]


def test_poll_never_raises(temp_db, fake_telegram, monkeypatch):
    import requests
    import data.database as db
    from notifications.telegram_commands import poll_commands

    def _net_down(*a, **k):
        raise requests.ConnectionError("sin red")
    monkeypatch.setattr(requests, "get", _net_down)
    assert poll_commands()["updates"] == 0

    def _db_down(*a, **k):
        raise RuntimeError("BD caída")
    monkeypatch.setattr(db, "kv_get", _db_down)
    out = poll_commands()                                  # sin offset no procesa nada
    assert out["errors"] == 1 and fake_telegram["sent"] == []


def test_poll_without_telegram_config_does_nothing(temp_db, no_network, monkeypatch):
    import notifications.telegram as tg
    from notifications.telegram_commands import poll_commands
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "")
    assert poll_commands() == {"updates": 0, "commands": 0, "ignored": 0, "errors": 0}
