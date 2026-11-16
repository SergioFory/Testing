"""
Pruebas de las correcciones de la revisión adversarial de la Fase 1 (2026-11-15):
  1. "Cerrar a mercado antes de" en oro/SPX con el calendario de festivos (antes
     BDay: no coincidía con la salida por tiempo del tracker en ~62 % de los casos);
  2. init_db toma el candado de PostgreSQL ANTES de los CREATE TABLE;
  3. /fill o /skip que fallan al guardarse no se pierden (offset) y comando
     desconocido largo truncado;
  4. latido: sin ping si falla la BD o el envío de una señal durante el ciclo;
  5. reintento de notificación: dos reintentos reales y aviso de señal no entregada
     (también en el resumen nocturno);
  6. _send no reintenta un ReadTimeout (evita copias duplicadas);
  7. /fill comprueba el precio antes de guardarlo ('84.837' para un BTC de 84.837);
  8. mensaje: sin "objetivo 1.00 %", comparado con el tope de 0,25 % de las reglas;
  9. R de filas sin pnl: loss = −1R, win = R del TP (dashboard y SPRT/CUSUM);
 10. línea "anteriores a la Fase 0" = originales desde la congelación;
 11. capital del dashboard = TRADING_CAPITAL;
 12. nota de la base GC=F/contado en el /fill del oro;
 13. resultado del /fill en español.

    cd trading_system && python -m pytest tests -q

Sin red y con BD SQLite temporal (el candado de PostgreSQL se comprueba con una
conexión falsa que registra las sentencias).
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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests

    def _boom(*a, **k):
        raise AssertionError("intento de red en una prueba")
    monkeypatch.setattr(requests, "post", _boom)
    monkeypatch.setattr(requests, "get", _boom)


@pytest.fixture(autouse=True)
def clean_health():
    from monitoring import cycle_health
    cycle_health.reset()
    yield
    cycle_health.reset()


@pytest.fixture()
def fake_telegram(monkeypatch):
    import notifications.telegram as tg
    import notifications.telegram_commands as tc
    state = {"ok": True, "sent": []}

    def _fake_send(text_, parse_mode="HTML", retries=3):
        state["sent"].append(text_)
        return state["ok"]
    monkeypatch.setattr(tg, "_send", _fake_send)
    monkeypatch.setattr(tg, "_stale_logged", set())
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "TEST-TOKEN")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "4242")
    monkeypatch.setattr(tc, "_update_failures", {})
    return state


class _Trade:
    def __init__(self, symbol="BTCUSDT", direction="long", entry=100.0, sl=97.0, tp=104.5):
        self.symbol, self.direction, self.setup_type = symbol, direction, "breakout"
        self.entry_price, self.stop_loss, self.take_profit = entry, sl, tp
        self.position_size, self.atr, self.ml_score = 1000.0, 2.0, 0.6
        self.rr_ratio, self.risk_usd, self.reward_usd = 1.5, 30.0, 45.0


U = lambda s: int(pd.Timestamp(s, tz="UTC").timestamp())   # noqa: E731


def _signal(db, **kw):
    row = {"symbol": "BTCUSDT", "ts": U("2026-11-18 08:00"), "setup_type": "breakout",
           "direction": "long", "ml_score": 0.6, "entry_price": 100.0, "stop_loss": 98.0,
           "take_profit": 103.0, "position_size": 1000.0, "atr": 1.3,
           "setup_ts": U("2026-11-18 04:00"), "mode": "live", "resultado_real": None,
           "pnl_r": None, "pnl_pct": None, "exit_price": None, "resolved_ts": None,
           "quality_flag": None}
    row.update(kw)
    cols = ", ".join(row)
    with db.get_engine().begin() as conn:
        conn.execute(text(f"INSERT INTO signals ({cols}) VALUES ({', '.join(':' + c for c in row)})"), row)
        return conn.execute(text("SELECT MAX(id) FROM signals")).scalar()


def _rows(db, table):
    with db.get_engine().connect() as conn:
        return [dict(r) for r in conn.execute(text(f"SELECT * FROM {table} ORDER BY id")).mappings()]


def _session_bars(symbol, start, end, price=4000.0):
    """Velas diarias como las de yfinance: solo los días con sesión."""
    from data.market_calendar import is_session, calendar_for
    days = [d for d in pd.date_range(start, end, freq="D", tz="UTC")
            if is_session(d, calendar_for(symbol))]
    idx = pd.DatetimeIndex(days)
    return pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price, "volume": 1.0}, index=idx)


# ---------------------------------------------------------------------------
# 1. Festivos en el plazo de cierre por tiempo
# ---------------------------------------------------------------------------

def test_calendario_nyse_festivos_y_reglas():
    from data.market_calendar import is_session
    for d in ["2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
              "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
              "2026-07-03",            # 4 de julio en sábado → viernes
              "2025-01-09"]:           # cierre excepcional (luto por Carter)
        assert not is_session(d, "nyse"), d
    assert is_session("2021-12-31", "nyse")      # Año Nuevo en sábado: el viernes SÍ abre
    assert is_session("2021-06-18", "nyse")      # Juneteenth solo desde 2022
    assert is_session("2025-10-13", "nyse")      # Columbus no es festivo de NYSE
    # El oro (COMEX) sí tuvo sesión el 2025-01-09.
    assert is_session("2025-01-09", "comex") and not is_session("2025-01-01", "comex")


def test_plazo_oro_con_festivo_coincide_con_el_tracker():
    """El ejemplo de la revisión: setup del oro del 2026-02-06; el lunes 16-feb
    (Presidents Day) no hay vela → el mensaje decía 06-mar y el tracker salía el 09-mar."""
    from data.outcome_tracker import time_exit_deadline, check_outcome_detail
    st = pd.Timestamp("2026-02-06", tz="UTC")
    assert time_exit_deadline("XAUUSD", st) == pd.Timestamp("2026-03-09 22:15", tz="UTC")
    df = _session_bars("XAUUSD", "2026-01-20", "2026-04-30")
    res = check_outcome_detail("XAUUSD", st, "long", 4000.0, -1e9, 1e12, df=df, setup_ts=st)
    assert res["outcome"] == "expired" and res["exit_ts"] == time_exit_deadline("XAUUSD", st)


@pytest.mark.parametrize("symbol", ["XAUUSD", "SPX500"])
def test_plazo_diario_coincide_con_tracker_todo_un_anio(symbol):
    """Para cada sesión de 2025 (11 festivos y el cierre del 9 de enero en SPX):
    la salida por tiempo del tracker cae exactamente en el plazo del mensaje."""
    from data.outcome_tracker import time_exit_deadline, check_outcome_detail
    df = _session_bars(symbol, "2024-12-01", "2026-03-31")
    setups = df.index[(df.index >= "2025-01-01") & (df.index < "2026-01-01")]
    assert len(setups) > 240
    for st in setups:
        res = check_outcome_detail(symbol, st, "long", 4000.0, -1e9, 1e12, df=df, setup_ts=st)
        assert res["exit_ts"] == time_exit_deadline(symbol, st), st


def test_plazo_spx_salta_el_cierre_excepcional_y_el_oro_no():
    from data.outcome_tracker import time_exit_deadline
    st = pd.Timestamp("2024-12-31", tz="UTC")
    # 20 sesiones desde el 31-dic-2024: sin 1-ene y MLK (20-ene); SPX además sin el 9-ene.
    assert time_exit_deadline("XAUUSD", st) == pd.Timestamp("2025-01-30 22:15", tz="UTC")
    assert time_exit_deadline("SPX500", st) == pd.Timestamp("2025-01-31 22:15", tz="UTC")


def test_plazo_cripto_no_cambia():
    from data.outcome_tracker import time_exit_deadline
    st = pd.Timestamp("2026-11-26 08:00", tz="UTC")       # Acción de Gracias: cripto no cierra
    assert time_exit_deadline("BTCUSDT", st) == st + pd.Timedelta(hours=4) * 25


# ---------------------------------------------------------------------------
# 2. Candado de PostgreSQL antes de los CREATE TABLE
# ---------------------------------------------------------------------------

def test_init_db_candado_antes_de_crear_tablas(monkeypatch):
    import data.database as db
    executed = []

    class _Res:
        def fetchall(self):
            return []

        def scalar(self):
            return None

    class _Conn:
        def execute(self, stmt, params=None):
            executed.append(str(stmt))
            return _Res()

    class _Ctx:
        def __enter__(self):
            return _Conn()

        def __exit__(self, *a):
            return False

    class _Engine:
        class dialect:
            name = "postgresql"

        def begin(self):
            return _Ctx()

    monkeypatch.setattr(db, "get_engine", lambda: _Engine())
    monkeypatch.setattr(db, "_db_ready", False)
    db.init_db()
    monkeypatch.setattr(db, "_db_ready", False)
    first_create = next(i for i, s in enumerate(executed) if "CREATE TABLE" in s)
    assert "pg_advisory_xact_lock" in executed[0]
    assert first_create > 0


# ---------------------------------------------------------------------------
# 3. Comandos que fallan al guardarse no se pierden
# ---------------------------------------------------------------------------

def _fake_updates(monkeypatch, batches, seen):
    import requests
    calls = {"n": 0}

    class _R:
        status_code = 200

        def __init__(self, data):
            self._d, self.text = data, str(data)

        def json(self):
            return self._d

    def _get(url, params=None, timeout=None):
        seen.append(dict(params))
        off = params.get("offset")
        k = min(calls["n"], len(batches) - 1)
        calls["n"] += 1
        return _R({"ok": True, "result": [u for u in batches[k]
                                          if off is None or u["update_id"] >= int(off)]})
    monkeypatch.setattr(requests, "get", _get)


def _upd(uid, text_, chat_id=4242):
    return {"update_id": uid, "message": {"date": 1763200000, "chat": {"id": chat_id}, "text": text_}}


def test_fill_que_falla_al_guardar_se_reintenta_y_no_se_pierde(temp_db, fake_telegram, monkeypatch):
    import data.database as db
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import poll_commands, KV_OFFSET
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15", tz="UTC"), mode="live")
    batch = [_upd(200, "/help"), _upd(201, f"/fill {sid} 100.5"), _upd(202, "/help")]
    seen = []
    _fake_updates(monkeypatch, [batch], seen)
    real_save = db.save_fill

    def _down(*a, **k):
        raise RuntimeError("BD caída momentánea")
    monkeypatch.setattr(db, "save_fill", _down)
    out = poll_commands()
    assert out["errors"] == 1 and out["commands"] == 1
    assert temp_db.kv_get(KV_OFFSET) == "201"          # se queda EN el /fill, no lo salta
    assert len(fake_telegram["sent"]) == 1              # solo la ayuda de la 200
    # Ciclo siguiente: la BD vuelve → se guarda y se responde; sigue con la 202.
    monkeypatch.setattr(db, "save_fill", real_save)
    out = poll_commands()
    assert seen[-1]["offset"] == 201
    assert [f["price"] for f in _rows(temp_db, "fills")] == [100.5]
    assert "Ejecución registrada" in fake_telegram["sent"][1]
    assert temp_db.kv_get(KV_OFFSET) == "203"


def test_fill_que_falla_siempre_avisa_tras_tres_intentos(temp_db, fake_telegram, monkeypatch):
    import data.database as db
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import poll_commands, KV_OFFSET, MAX_UPDATE_ATTEMPTS
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15", tz="UTC"), mode="live")
    _fake_updates(monkeypatch, [[_upd(300, f"/fill {sid} 100.5")]], [])
    monkeypatch.setattr(db, "save_fill", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("caída")))
    for _ in range(MAX_UPDATE_ATTEMPTS - 1):
        poll_commands()
        assert temp_db.kv_get(KV_OFFSET) in (None, "300")
        assert fake_telegram["sent"] == []
    poll_commands()
    assert temp_db.kv_get(KV_OFFSET) == "301"
    assert len(fake_telegram["sent"]) == 1
    assert "No pude registrar" in fake_telegram["sent"][0] and "Reenvíalo" in fake_telegram["sent"][0]


def test_comando_desconocido_largo_se_trunca(temp_db):
    from notifications.telegram_commands import handle_command
    reply = handle_command("x" * 4000, [])
    assert len(reply) < 1500 and "x" * 50 + "…" in reply


# ---------------------------------------------------------------------------
# 4. Latido: sin ping si falla la BD o Telegram durante el ciclo
# ---------------------------------------------------------------------------

class _StopLoop(Exception):
    pass


def _run_cycle(monkeypatch, fake_cmd_signal, notify=False):
    import time
    import main
    import data.outcome_tracker as ot
    import data.shadow as sh
    import monitoring.heartbeat as hb
    import notifications.telegram_commands as tc
    import config.settings as cs
    pings = []
    monkeypatch.setattr(main, "cmd_signal", fake_cmd_signal)
    monkeypatch.setattr(main, "active_assets", lambda: ["BTCUSDT"])
    monkeypatch.setattr(ot, "resolve_open_signals", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(sh, "resolve_shadow_setups", lambda ohlcv_cache=None: 0)
    monkeypatch.setattr(tc, "poll_commands", lambda: {})
    monkeypatch.setattr(hb, "ping_heartbeat", lambda *a, **k: pings.append(1) or True)
    monkeypatch.setattr(time, "sleep", lambda _s: (_ for _ in ()).throw(_StopLoop()))
    monkeypatch.setattr(cs, "TELEGRAM_TOKEN", "T" if notify else "")
    monkeypatch.setattr(cs, "TELEGRAM_CHAT_ID", "4242" if notify else "")
    with pytest.raises(_StopLoop):
        main.cmd_schedule(60, "03:00")
    import schedule
    schedule.clear()
    return pings


def test_latido_ciclo_sano_hace_ping(temp_db, monkeypatch):
    import main

    def ok(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
    assert _run_cycle(monkeypatch, ok) == [1]


def test_latido_sin_ping_si_falla_la_bd_en_el_ciclo(temp_db, monkeypatch):
    """BD caída tras el arranque: emitted_setup_keys / el antiduplicado se tragan
    el error (cmd_signal no lanza), pero el ciclo ya no cuenta como correcto."""
    import main
    import data.outcome_tracker as ot

    def bd_caida(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
        with monkeypatch.context() as m:
            m.setattr(ot, "get_engine", lambda: (_ for _ in ()).throw(RuntimeError("BD caída")))
            assert ot.emitted_setup_keys(symbol, pd.Timestamp("2026-11-15", tz="UTC")) == set()
            assert ot.is_duplicate_setup(symbol, "long", pd.Timestamp("2026-11-15", tz="UTC")) is True
    assert _run_cycle(monkeypatch, bd_caida) == []


def test_latido_sin_ping_si_la_bd_no_responde_al_final(temp_db, monkeypatch):
    import main
    from monitoring import cycle_health

    def ok(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
    monkeypatch.setattr(cycle_health, "db_alive", lambda: False)
    assert _run_cycle(monkeypatch, ok) == []


def test_latido_sin_ping_si_telegram_rechaza_la_senal(temp_db, fake_telegram, monkeypatch):
    """Telegram rechaza todo (p. ej. 403 bot bloqueado): el reenvío de la señal
    pendiente falla → sin latido."""
    import main
    from data.outcome_tracker import save_signal_from_trade
    fake_telegram["ok"] = False
    save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp.now(tz="UTC").floor("4h"),
                           mode="live", notified=False, alert_text="SEÑAL")

    def ok(symbol, notify=False, ohlcv_cache=None):
        main._cycle_fresh.add(symbol)
    assert _run_cycle(monkeypatch, ok, notify=True) == []
    fake_telegram["ok"] = True
    assert _run_cycle(monkeypatch, ok, notify=True) == [1]     # vuelve: reenvía y hay latido


def test_envio_fallido_de_senal_marca_el_ciclo(temp_db, monkeypatch):
    """_emit_best_setup: si send_text falla, el ciclo queda marcado (telegram)."""
    from monitoring import cycle_health
    import notifications.telegram as tg
    monkeypatch.setattr(tg, "_send", lambda *a, **k: False)
    from test_fase1_sombra import _pipeline, _setup, IDX
    import main
    _pipeline(monkeypatch, [_setup(IDX[-1], raw=0.7)])
    main.cmd_signal("BTCUSDT", notify=True)
    assert [k for k, _ in cycle_health.problems()] == ["telegram"]


# ---------------------------------------------------------------------------
# 5. Reintento: dos reintentos de verdad y aviso de señal no entregada
# ---------------------------------------------------------------------------

def _set_ts(db, sig_id, ts):
    with db.get_engine().begin() as conn:
        conn.execute(text("UPDATE signals SET ts=:t WHERE id=:i"), {"t": U(ts), "i": sig_id})


def test_reintento_cabe_dos_veces_con_el_desfase_de_schedule(temp_db, fake_telegram):
    """Escenario de la revisión: Telegram caído de 04:00 a 06:30; señal a las
    04:05:06; ciclos a las 05:05:12 y 06:05:12 (schedule cuenta desde el FINAL)."""
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    from monitoring import cycle_health
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-13 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="SEÑAL #X")
    _set_ts(temp_db, sid, "2026-11-13 04:05:06")
    fake_telegram["ok"] = False
    r1 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 05:05:12", tz="UTC"))
    r2 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 06:05:12", tz="UTC"))
    assert r1["failed"] == 1 and r2["failed"] == 1           # 2.ª vez: antes ya era "vieja"
    assert [k for k, _ in cycle_health.problems()] == ["telegram", "telegram"]
    fake_telegram["ok"] = True
    r3 = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 06:20:00", tz="UTC"))
    assert r3["resent"] == 1 and temp_db.load_signal_by_id(sid)["notified"] == 1


def test_senal_no_entregada_se_avisa_una_vez(temp_db, fake_telegram):
    import notifications.telegram as tg
    from data.outcome_tracker import save_signal_from_trade
    from monitoring import cycle_health
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-13 00:00", tz="UTC"),
                                 mode="live", notified=False, alert_text="SEÑAL")
    _set_ts(temp_db, sid, "2026-11-13 04:05")
    fake_telegram["ok"] = False                     # sigue caído: el aviso tampoco sale
    r = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 07:05", tz="UTC"))
    assert r["stale"] == 1 and r["noticed"] == 0
    assert ("telegram" in [k for k, _ in cycle_health.problems()])   # sin latido
    fake_telegram["ok"] = True
    fake_telegram["sent"].clear()
    r = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 09:05", tz="UTC"))
    assert r["noticed"] == 1 and len(fake_telegram["sent"]) == 1
    msg = fake_telegram["sent"][0]
    assert f"No te llegó la señal #{sid}" in msg and "BTCUSDT" in msg and "97" in msg
    r = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 10:05", tz="UTC"))
    assert r["noticed"] == 0 and len(fake_telegram["sent"]) == 1     # no se repite
    assert temp_db.load_signal_by_id(sid)["notified"] == 0            # constancia en la BD
    # Muy vieja (> 48 h): ni se intenta.
    sid2 = save_signal_from_trade(_Trade(entry=200, sl=194, tp=209),
                                  setup_ts=pd.Timestamp("2026-11-01", tz="UTC"),
                                  mode="live", notified=False, alert_text="VIEJA")
    _set_ts(temp_db, sid2, "2026-11-01 04:00")
    r = tg.retry_failed_signal_alerts(now=pd.Timestamp("2026-11-13 10:05", tz="UTC"))
    assert r["stale"] == 2 and r["noticed"] == 0


def test_resumen_lista_senales_no_entregadas(temp_db):
    from monitoring.daily_summary import build_daily_summary
    now = pd.Timestamp("2026-11-19 03:00", tz="UTC")
    _signal(temp_db, notified=0, ts=U("2026-11-18 08:00"))                       # últimas 24 h
    _signal(temp_db, notified=0, ts=U("2026-11-10 08:00"), symbol="ETHUSDT",
            entry_price=3000.0, stop_loss=2900.0, take_profit=3200.0)            # abierta
    _signal(temp_db, notified=0, ts=U("2026-11-10 08:00"), resultado_real="loss")  # vieja y cerrada
    _signal(temp_db, notified=1, ts=U("2026-11-18 09:00"))                       # sí llegó
    txt = build_daily_summary(now=now, symbols=[])
    assert "NO te llegaron por Telegram: 2" in txt
    assert "ETHUSDT" in txt and "sigue abierta" in txt


# ---------------------------------------------------------------------------
# 6. _send: un ReadTimeout no se reintenta
# ---------------------------------------------------------------------------

def test_send_no_reintenta_read_timeout(monkeypatch):
    import requests
    import time
    import notifications.telegram as tg
    monkeypatch.setattr(tg, "TELEGRAM_TOKEN", "T")
    monkeypatch.setattr(tg, "TELEGRAM_CHAT_ID", "1")
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = []

    def _timeout(url, json=None, timeout=None):
        calls.append(1)
        raise requests.ReadTimeout("respuesta perdida")
    monkeypatch.setattr(requests, "post", _timeout)
    assert tg._send("hola") is False and len(calls) == 1

    calls.clear()

    def _conn(url, json=None, timeout=None):
        calls.append(1)
        raise requests.ConnectionError("sin red")
    monkeypatch.setattr(requests, "post", _conn)
    assert tg._send("hola") is False and len(calls) == 3       # esto sí se reintenta
    assert "MISMA señal" in tg.RESEND_NOTE


# ---------------------------------------------------------------------------
# 7, 12, 13. /fill: precio comprobado antes de guardar, nota del oro, español
# ---------------------------------------------------------------------------

def test_plausible_price_relee_miles():
    from notifications.telegram_commands import plausible_price, parse_price
    assert plausible_price("84.837", parse_price("84.837"), 84837.6) == (84837.0, True)
    assert plausible_price("3,125", parse_price("3,125"), 3.1) == (3.125, True)
    assert plausible_price("84837.5", 84837.5, 84837.6) == (84837.5, False)
    assert plausible_price("5", 5.0, 84837.6) == (None, False)


def test_fill_mal_leido_no_se_guarda_y_releido_si(temp_db, fake_telegram):
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import handle_command
    sid = save_signal_from_trade(_Trade(entry=84837.6, sl=86000.0, tp=83093.0, direction="short"),
                                 setup_ts=pd.Timestamp("2026-11-15", tz="UTC"), mode="live")
    r = handle_command("fill", [str(sid), "84.837"], update_id=1)
    assert "Ejecución registrada" in r and "Leí «84.837» como 84,837" in r
    r = handle_command("fill", [str(sid), "8483"], update_id=2)          # falta una cifra
    assert "No lo he guardado" in r
    assert [f["price"] for f in _rows(temp_db, "fills")] == [84837.0]
    assert "cuenta el más reciente" in r or "cuenta el más reciente" in handle_command("help", [])


def test_fill_mas_alla_del_sl_pide_confirmar(temp_db, fake_telegram):
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import handle_command
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15", tz="UTC"), mode="live")
    r = handle_command("fill", [str(sid), "96.5"], update_id=1)
    assert "No lo he guardado" in r and f"/fill {sid} 96.5 confirmar" in r
    assert _rows(temp_db, "fills") == []
    r = handle_command("fill", [str(sid), "96.5", "deslizamiento", "confirmar"], update_id=2)
    assert "Ejecución registrada" in r and "lo confirmaste" in r
    f = _rows(temp_db, "fills")
    assert [(x["price"], x["note"]) for x in f] == [(96.5, "deslizamiento")]


def test_fill_resultado_en_espanol_y_nota_del_oro(temp_db, fake_telegram):
    from data.outcome_tracker import save_signal_from_trade, _update_signal_result
    from notifications.telegram_commands import handle_command
    sid = save_signal_from_trade(_Trade(symbol="XAUUSD", entry=4000.0, sl=3900.0, tp=4133.0),
                                 setup_ts=pd.Timestamp("2026-11-13", tz="UTC"), mode="live")
    _update_signal_result(sid, "expired", 5.0, 0.005, pnl_r=0.2, exit_price=4020.0)
    r = handle_command("fill", [str(sid), "3995"], update_id=1)
    assert "(cierre por tiempo)" in r and "(expired)" not in r
    assert "GC=F" in r and "contado" in r
    sid2 = save_signal_from_trade(_Trade(entry=200.0, sl=194.0, tp=209.0),
                                  setup_ts=pd.Timestamp("2026-11-14", tz="UTC"), mode="live")
    r2 = handle_command("fill", [str(sid2), "200"], update_id=2)
    assert "GC=F" not in r2


# ---------------------------------------------------------------------------
# 8. Mensaje: riesgo frente al tope de 0,25 % de las reglas
# ---------------------------------------------------------------------------

def test_mensaje_riesgo_por_encima_del_tope_de_las_reglas():
    from risk.position_sizer import build_trade_setup
    from notifications.telegram import format_signal_alert
    # Oro real de la revisión: riesgo 0.37 % con 10.000 USD. Fase 3 (3.2): el sistema
    # ya arriesga el 0,25 %; se reproduce la configuración ANTERIOR (1 % con tope de
    # nominal del 10 % = apalancamiento 0,1×) para seguir probando la línea de aviso.
    row = pd.Series({"close": 4168.40, "stop_loss": 4323.68, "take_profit": 3961.36,
                     "atr": 103.52, "direction": "short", "symbol": "XAUUSD", "setup_type": "breakout"})
    t = build_trade_setup(row, 10_000.0, 0.23, risk_per_trade=0.01, max_leverage=0.10)
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-09-28", tz="UTC"), capital=10_000,
                              now=pd.Timestamp("2026-09-28 22:20", tz="UTC"))
    assert "objetivo" not in msg and "1.00 %" not in msg
    assert "más que el tope de las reglas: 0.25 %" in msg
    frac = 0.0025 / (t.risk_usd / 10_000)
    assert f"Para respetar el 0.25 %:</b> {t.position_qty * frac:.4f}".rstrip("0") in msg
    assert f"riesgo ${t.risk_usd * frac:,.2f}" in msg and "riesgo $25.00" in msg


# ---------------------------------------------------------------------------
# 9. R de filas resueltas sin pnl guardado
# ---------------------------------------------------------------------------

def test_signal_r_loss_y_win_sin_pnl():
    from data.outcome_tracker import signal_r
    base = {"entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0, "pnl_r": float("nan"),
            "pnl_pct": float("nan")}
    assert signal_r({**base, "resultado_real": "loss"}) == -1.0
    assert signal_r({**base, "resultado_real": "win"}) == pytest.approx(1.5)
    assert signal_r({**base, "resultado_real": "be"}) == 0.0
    assert signal_r({**base, "resultado_real": "expired"}) is None
    assert signal_r({**base, "resultado_real": "loss", "pnl_r": -0.8}) == -0.8   # manda pnl_r


def test_stats_y_sprt_cuentan_la_perdida_sin_pnl(temp_db):
    from data.outcome_tracker import get_outcome_stats
    from monitoring.sequential import load_live_r
    _signal(temp_db, resultado_real="win", pnl_r=1.5, resolved_ts=U("2026-11-19 10:00"))
    _signal(temp_db, resultado_real="loss", pnl_r=None, resolved_ts=U("2026-11-19 12:00"))
    _signal(temp_db, resultado_real="expired", pnl_r=None, resolved_ts=U("2026-11-19 13:00"))
    st = get_outcome_stats("BTCUSDT")
    assert st["n_r"] == 2 and st["mean_r"] == pytest.approx(0.25) and st["sum_r"] == pytest.approx(0.5)
    assert load_live_r() == [1.5, -1.0]


# ---------------------------------------------------------------------------
# 10. "Anteriores a la Fase 0" = originales desde la congelación
# ---------------------------------------------------------------------------

def test_pre_fase0_solo_originales_desde_la_congelacion(temp_db):
    from monitoring.sequential import load_live_r, monitoring_state, format_monitoring
    kw = {"setup_ts": None, "resolved_ts": U("2026-10-01")}
    _signal(temp_db, ts=U("2026-09-21 09:31"), resultado_real="win", pnl_r=1.5, **kw)      # cuenta
    _signal(temp_db, ts=U("2026-10-08 16:09"), resultado_real="loss", pnl_r=None,
            symbol="ETHUSDT", **kw)                                                      # cuenta (−1)
    _signal(temp_db, ts=U("2026-09-21 10:34"), resultado_real="win", pnl_r=1.5,
            quality_flag="intrabar_repeat", **kw)                                        # repetición
    _signal(temp_db, ts=U("2026-04-01"), resultado_real="win", pnl_r=2.0,
            symbol="EURUSD", **kw)                                                       # antes / apagado
    _signal(temp_db, ts=U("2026-09-01"), resultado_real="win", pnl_r=1.5, **kw)          # antes del freeze
    _signal(temp_db, ts=U("2026-10-10"), resultado_real="win", pnl_r=1.67,
            symbol="SPX500", **kw)                                                       # hoy en sombra
    assert load_live_r(phase0=False) == [1.5, -1.0]
    txt = format_monitoring(monitoring_state())
    assert "desde la congelación de la configuración, 2026-09-18" in txt
    assert "2 señales, R medio +0.25R" in txt


# ---------------------------------------------------------------------------
# 11. Capital del dashboard
# ---------------------------------------------------------------------------

def test_dashboard_capital_por_defecto_es_trading_capital():
    src = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    block = src[src.index('"Capital (USD)"') - 200: src.index('"Capital (USD)"') + 200]
    assert "TRADING_CAPITAL" in block and 'RISK["capital_inicial"]' not in block


# ---------------------------------------------------------------------------
# Segunda revisión (commit de correcciones)
# ---------------------------------------------------------------------------

def test_lotes_pendientes_indican_el_tamano_reducido_si_supera_el_tope():
    """Si el riesgo supera el 0,25 %, la línea de lotes no puede mandar operar el
    tamaño completo justo después de dar el reducido."""
    from risk.position_sizer import build_trade_setup
    from notifications.telegram import format_signal_alert
    row = pd.Series({"close": 4168.40, "stop_loss": 4323.68, "take_profit": 3961.36,
                     "atr": 103.52, "direction": "short", "symbol": "XAUUSD", "setup_type": "breakout"})
    # Configuración anterior a la Fase 3 (ver la prueba de arriba).
    t = build_trade_setup(row, 10_000.0, 0.23, risk_per_trade=0.01, max_leverage=0.10)
    msg = format_signal_alert(t, setup_ts=pd.Timestamp("2026-09-28", tz="UTC"), capital=10_000,
                              now=pd.Timestamp("2026-09-28 22:20", tz="UTC"))
    lot_line = [l for l in msg.splitlines() if "Lotes XM" in l][0]
    cap_line = [l for l in msg.splitlines() if "Para respetar" in l][0]
    reduced = cap_line.split("</b>")[1].strip().split()[0]
    assert f"opera {reduced} " in lot_line


def test_cierre_manual_sin_precio_queda_sin_r(temp_db):
    from data.outcome_tracker import signal_r, get_outcome_stats
    from monitoring.sequential import load_live_r
    base = {"entry_price": 100.0, "stop_loss": 98.0, "take_profit": 103.0, "pnl_r": None,
            "pnl_pct": None, "resolved_by": "manual"}
    assert signal_r({**base, "resultado_real": "win"}) is None
    assert signal_r({**base, "resultado_real": "loss"}) is None
    assert signal_r({**base, "resultado_real": "be"}) == 0.0
    assert signal_r({**base, "resultado_real": "loss", "resolved_by": "auto"}) == -1.0
    sid = _signal(temp_db, resultado_real="win", pnl_r=None, resolved_ts=U("2026-11-19 10:00"))
    with temp_db.get_engine().begin() as c:
        c.execute(text("UPDATE signals SET resolved_by='manual' WHERE id=:i"), {"i": sid})
    _signal(temp_db, resultado_real="loss", pnl_r=None, resolved_ts=U("2026-11-19 12:00"))
    assert load_live_r() == [-1.0]
    assert get_outcome_stats("BTCUSDT")["n_r"] == 1


def test_rechazo_de_fill_recorta_el_precio_largo(temp_db, fake_telegram):
    from data.outcome_tracker import save_signal_from_trade
    from notifications.telegram_commands import handle_command
    sid = save_signal_from_trade(_Trade(), setup_ts=pd.Timestamp("2026-11-15", tz="UTC"), mode="live")
    r = handle_command("fill", [str(sid), "9" * 5000], update_id=1)
    assert r is not None and len(r) < 1000


def test_fetcher_no_usa_la_vela_plana_del_final_de_pagina(monkeypatch):
    """Gate.io entrega PLANA (O=H=L=C, volumen 0) la vela que cae al final de cada
    página. El fetcher debe descartarla y volver a pedirla en la página siguiente."""
    import data.fetchers.binance_fetcher as bf
    monkeypatch.setattr(bf.time, "sleep", lambda s: None)
    step = 4 * 3600 * 1000
    now = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    start = now - 2500 * step

    class _Ex:
        id, rateLimit = "fake", 0
        calls = []

        def parse_timeframe(self, tf):
            return 4 * 3600

        def fetch_ohlcv(self, symbol, tf, since=None, limit=1000):
            self.calls.append(since)
            first = start + ((since - start + step - 1) // step) * step
            out = []
            for k in range(999):          # Gate.io: 999 velas aunque se pidan 1000
                ts = first + k * step
                if ts > now:
                    break
                out.append([ts, 1.0, 2.0, 0.5, 1.5, 10.0])
            if out and out[-1][0] + step <= now:   # vela final de página: plana
                ts = out[-1][0]
                out[-1] = [ts, 1.0, 1.0, 1.0, 1.0, 0.0]
            return out

    ex = _Ex()
    df = bf._fetch_ohlcv_from(ex, "BTC/USDT:USDT", "4h", days=int(2500 * 4 / 24))
    flat = (df["high"] == df["low"]) & (df["volume"] == 0)
    assert int(flat.sum()) == 0
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert len(ex.calls) >= 3
    gaps = df.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(hours=4)).all()


# ---------------------------------------------------------------------------
# Vela diaria "en curso" de los futuros de yfinance (GC=F)
# ---------------------------------------------------------------------------

def _daily(dates):
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("UTC")
    return pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)


@pytest.mark.parametrize("now, expected_last", [
    # Domingo 23:30 UTC (18:30 ET): fila fechada domingo = sesión del lunes en curso.
    ("2026-11-15 23:30", "2026-11-13"),
    # Lunes 22:30 UTC (17:30 ET, invierno): vela del lunes cerrada y aún sin reapertura.
    ("2026-11-16 22:30", "2026-11-16"),
    # Lunes 23:30 UTC (18:30 ET): la fila "lunes" puede ser la sesión del martes.
    ("2026-11-16 23:30", "2026-11-13"),
    # Martes 05:30 UTC (00:30 ET): la fila del lunes vuelve a ser la verdadera.
    ("2026-11-17 05:30", "2026-11-16"),
    # Verano: 22:30 UTC = 18:30 EDT, ya reabrió a las 22:00 UTC.
    ("2026-07-14 22:30", "2026-07-10"),
    ("2026-07-15 04:30", "2026-07-14"),
])
def test_vela_diaria_en_curso_de_futuros_no_se_da_por_cerrada(now, expected_last):
    from data.bars import drop_unclosed_bars
    if now.startswith("2026-07"):
        df = _daily(["2026-07-09", "2026-07-10", "2026-07-14", "2026-07-15"])
        if now.startswith("2026-07-14"):
            df = df.iloc[:3]
    else:
        df = _daily(["2026-11-12", "2026-11-13", "2026-11-15", "2026-11-16", "2026-11-17"])
        df = df[df.index <= pd.Timestamp(now, tz="UTC").normalize()]
    out = drop_unclosed_bars(df, "1d", session=True, now=pd.Timestamp(now, tz="UTC"))
    assert out.index[-1] == pd.Timestamp(expected_last, tz="UTC")
    assert (out.index.dayofweek < 5).all()


def test_filtro_de_sesion_no_toca_el_historico():
    from data.bars import drop_unclosed_bars
    df = _daily(pd.bdate_range("2025-01-01", "2026-11-13"))
    out = drop_unclosed_bars(df, "1d", session=True, now=pd.Timestamp("2026-11-15 23:30", tz="UTC"))
    assert len(out) == len(df)
